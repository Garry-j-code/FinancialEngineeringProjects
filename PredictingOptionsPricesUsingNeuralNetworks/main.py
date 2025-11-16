
import argparse
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

def _norm_cdf(x):
    """Standard normal CDF for array-like x.
    Tries scipy.special.erf; falls back to math.erf vectorized.
    """
    try:
        from scipy.special import erf as _erf
        return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
    except Exception:
        from math import erf as _m_erf
        return 0.5 * (1.0 + np.vectorize(_m_erf)(x / np.sqrt(2.0)))


# -----------------------------
# Config
# -----------------------------

REQUIRED_COLS = ["S", "K", "T", "r", "Sigma"]

# -----------------------------
# Robust parsing helpers
# -----------------------------

def _to_float_relaxed(x):
    import numpy as _np, re as _re
    if x is None:
        return _np.nan
    if isinstance(x, str):
        s = x.replace("\u00a0", " ").strip()  # NBSP -> space
        if s == "":
            return _np.nan
        s = s.replace("$", "").replace(",", "")
        s = _re.sub(r"\s+", "", s)            # remove inner spaces ("5 %")
        if s.endswith("%"):
            s = s[:-1]
            try:
                return float(s) / 100.0
            except ValueError:
                return _np.nan
        try:
            return float(s)
        except ValueError:
            return _np.nan
    try:
        return float(x)
    except Exception:
        return _np.nan

def _clean_col(c):
    import re as _re
    c = str(c).replace("\u00a0", " ").strip()
    return _re.sub(r"\s+", " ", c)

def load_excel(path: str, sheet_name=0, header=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, header=header)
    df.columns = [_clean_col(c) for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    df = df.dropna(how="all")
    for col in ["S", "K", "T"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["r"] = df["r"].apply(_to_float_relaxed)
    df["Sigma"] = df["Sigma"].apply(_to_float_relaxed)
    bad_mask = df[REQUIRED_COLS].isna()
    if bad_mask.any().any():
        print("\nRows with invalid/missing required inputs:")
        print(df.loc[bad_mask.any(axis=1), REQUIRED_COLS])
        counts = bad_mask.sum().to_dict()
        print("\nNaN count by column:", counts)
        raise ValueError("Some required inputs are missing or non-numeric. See rows above.")
    return df

# -----------------------------
# Analytics
# -----------------------------

def bs_call_price(S, K, T, r, sigma):
    S = np.maximum(np.asarray(S, dtype=float), 1e-12)
    K = np.maximum(np.asarray(K, dtype=float), 1e-12)
    T = np.maximum(np.asarray(T, dtype=float), 1e-12)
    r = np.asarray(r, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    N_d1 = _norm_cdf(d1)
    N_d2 = _norm_cdf(d2)
    discount = np.exp(-r * T)
    return S * N_d1 - K * discount * N_d2 # Option Value = (Prob. of being in the money × Stock’s future value) − (Prob. of exercise × Discounted strike price)

# -----------------------------
# Features & targets
# -----------------------------

def make_features(df: pd.DataFrame) -> np.ndarray:
    S = df["S"].values
    K = df["K"].values
    T = df["T"].values
    r = df["r"].values
    sigma = df["Sigma"].values
    # Engineered features
    log_m = np.log(S / K) # moneyness
    m = S / K
    sqrtT = np.sqrt(T)
    discK = K * np.exp(-r * T)
    X = np.column_stack([S, K, T, r, sigma, log_m, m, sqrtT, discK])
    return X

def get_targets(df: pd.DataFrame):
    if "P" in df.columns:
        y = pd.to_numeric(df["P"], errors="coerce").values
        if np.isnan(y).any():
            raise ValueError("Column 'P' contains non-numeric values.")
        return y, False
    else:
        y = bs_call_price(df["S"].values, df["K"].values, df["T"].values, df["r"].values, df["Sigma"].values)
        return y, True

# -----------------------------
# Model
# -----------------------------

def build_model(input_dim: int) -> tf.keras.Model:
    # Small MLP with L2 regularization (no dropout; tiny datasets)
    l2 = 1e-4
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                           tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model

# -----------------------------
# Training with K-Fold CV & target scaling
# -----------------------------

def train_kfold_predict(X, y, Kvals, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    kf = KFold(n_splits=min(n_splits, len(X)), shuffle=True, random_state=seed)

    oof_preds_scaled = np.zeros(len(X), dtype=float)
    models = []
    scalers = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        K_tr, K_va = Kvals[tr_idx], Kvals[va_idx]

        # Scale features
        xscaler = StandardScaler()
        X_tr_s = xscaler.fit_transform(X_tr)
        X_va_s = xscaler.transform(X_va)

        # Scale target by K (price per unit of strike)
        y_tr_scaled = (y[tr_idx] / K_tr).astype(float)
        y_va_scaled = (y[va_idx] / K_va).astype(float)

        model = build_model(input_dim=X_tr_s.shape[1])

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=25, min_lr=1e-6, verbose=0),
        ]

        model.fit(X_tr_s, y_tr_scaled,
                  validation_data=(X_va_s, y_va_scaled),
                  epochs=2000,
                  batch_size=max(4, len(X_tr)//4),
                  verbose=0,
                  callbacks=callbacks)

        # Store OOF predictions (scaled back by K)
        va_pred_scaled = model.predict(X_va_s, verbose=0).reshape(-1)
        oof_preds_scaled[va_idx] = va_pred_scaled * K_va

        models.append(model)
        scalers.append(xscaler)

    return models, scalers, oof_preds_scaled

def predict_ensemble(models, scalers, X, Kvals):
    preds_list = []
    for model, scaler in zip(models, scalers):
        Xs = scaler.transform(X)
        p_scaled = model.predict(Xs, verbose=0).reshape(-1) * Kvals
        preds_list.append(p_scaled)
    return np.mean(preds_list, axis=0)

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="NN option pricer with BS baseline and K-fold training.")
    parser.add_argument("excel_path", help="Path to Excel file with columns S, K, T, r, Sigma, (optional) P")
    parser.add_argument("--sheet", default=0, help="Sheet name or index (default=0)")
    parser.add_argument("--header-row", type=int, default=0, help="Row index to use as header (0-based, default=0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df = load_excel(args.excel_path, sheet_name=args.sheet, header=args.header_row)

    # Build features and targets
    X = make_features(df)
    y, synthesized = get_targets(df)
    Kvals = df["K"].values

    # Black–Scholes baseline
    p_bs = bs_call_price(df["S"].values, df["K"].values, df["T"].values, df["r"].values, df["Sigma"].values)

    # Train K-fold and produce predictions
    models, scalers, oof_preds = train_kfold_predict(X, y, Kvals, n_splits=5, seed=args.seed)
    # Use ensemble of folds to predict all rows (often slightly better than OOF)
    p_nn = predict_ensemble(models, scalers, X, Kvals)

    out = df.copy()
    out["P_bs"] = p_bs
    out["P_pred"] = p_nn

    if "P" in out.columns:
        out["abs_err_bs"] = (out["P_bs"] - out["P"]).abs()
        out["pct_err_bs"] = out["abs_err_bs"] / out["P"].replace(0, np.nan)
        out["abs_err_nn"] = (out["P_pred"] - out["P"]).abs()
        out["pct_err_nn"] = out["abs_err_nn"] / out["P"].replace(0, np.nan)

    base, ext = os.path.splitext(args.excel_path)
    out_path = f"{base}_with_predictions.xlsx"
    out.to_excel(out_path, index=False)

    # Print a quick summary
    if "P" in out.columns:
        mae_bs = float(np.nanmean(out["abs_err_bs"]))
        mae_nn = float(np.nanmean(out["abs_err_nn"]))
        print(f"MAE Black–Scholes: {mae_bs:.4f}")
        print(f"MAE Neural Net  : {mae_nn:.4f}")
    print(f"Saved predictions to: {out_path}")

if __name__ == "__main__":
    main()

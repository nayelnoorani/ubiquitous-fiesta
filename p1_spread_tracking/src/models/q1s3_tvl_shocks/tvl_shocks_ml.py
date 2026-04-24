"""
Q1S3 — TVL Shocks and Rate Divergence: ML Models

Target:  next-day spread change (spread_{t+1} - spread_t)
Split:   80/20 chronological
Models:
  1. OLS regression — interpretable baseline
  2. Lasso regression — automatic feature selection
  3. XGBoost regressor — captures asymmetric inflow/outflow effects

Metrics: MAE, RMSE, R², direction accuracy
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH = Path(__file__).parent / "results.md"

FEATURE_COLS = [
    "aave_tvl_change_pct_1d",
    "tvl_ratio",
    "aave_rate_change_1d",
    "compound_net_change_1d",
    "spread_vs_net_lag_1d",
    "spread_vs_net_rolling_std_7d",
    "spread_vs_net_rolling_mean_7d",
]


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def prepare_data(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = wide[FEATURE_COLS + ["spread_vs_net"]].copy().dropna()

    # Target: next-day spread change
    df["target"] = df["spread_vs_net"].shift(-1) - df["spread_vs_net"]
    df = df.dropna()

    split = int(len(df) * 0.8)
    return df, df.iloc[:split], df.iloc[split:]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)

    dir_true = np.sign(y_true)
    dir_pred = np.sign(y_pred)
    valid    = dir_pred != 0
    dir_acc  = float((dir_true[valid] == dir_pred[valid]).mean()) if valid.sum() > 0 else float("nan")

    return {
        "model":   name,
        "mae":     round(float(mae),  4),
        "rmse":    round(float(rmse), 4),
        "r2":      round(float(r2),   4),
        "dir_acc": round(float(dir_acc), 4) if not np.isnan(dir_acc) else None,
    }


# ----------------------------------------------------------------------
# Model 1 — OLS
# ----------------------------------------------------------------------

def run_ols(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = LinearRegression()
    model.fit(train[FEATURE_COLS].values, train["target"].values)
    y_pred = model.predict(test[FEATURE_COLS].values)

    result = evaluate("OLS regression", test["target"].values, y_pred)
    result["coefs"] = {col: round(float(c), 6)
                       for col, c in zip(FEATURE_COLS, model.coef_)}
    result["intercept"] = round(float(model.intercept_), 6)
    return result


# ----------------------------------------------------------------------
# Model 2 — Lasso (alpha chosen by cross-validation)
# ----------------------------------------------------------------------

def run_lasso(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(train[FEATURE_COLS].values)
    X_te   = scaler.transform(test[FEATURE_COLS].values)
    y_tr   = train["target"].values
    y_te   = test["target"].values

    # Choose alpha via 5-fold time-series-aware CV (no shuffle)
    cv_model = LassoCV(cv=5, max_iter=5000, random_state=42)
    cv_model.fit(X_tr, y_tr)

    y_pred = cv_model.predict(X_te)
    result = evaluate(f"Lasso (α={cv_model.alpha_:.4f})", y_te, y_pred)
    result["alpha"]        = round(float(cv_model.alpha_), 6)
    result["coefs"]        = {col: round(float(c), 6)
                               for col, c in zip(FEATURE_COLS, cv_model.coef_)}
    result["n_nonzero"]    = int(np.sum(cv_model.coef_ != 0))
    result["zeroed_feats"] = [col for col, c in zip(FEATURE_COLS, cv_model.coef_) if c == 0]
    return result


# ----------------------------------------------------------------------
# Model 3 — XGBoost regressor
# ----------------------------------------------------------------------

def run_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_tr = train[FEATURE_COLS].values
    y_tr = train["target"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["target"].values

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mae", verbosity=0, random_state=42,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    result = evaluate("XGBoost regressor", y_te, y_pred)
    result["feature_importance"] = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLS, model.feature_importances_)
    }
    return result


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(models: list[dict], train_len: int, test_len: int,
               train_start: str, train_end: str,
               test_start: str, test_end: str, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "\n---\n",
        "# Q1S3 — TVL Shocks and Rate Divergence: ML Models\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",
        "## Run Details\n",
        f"- Training: {train_start} → {train_end} ({train_len} days)",
        f"- Test:     {test_start} → {test_end} ({test_len} days)",
        "- Target:   next-day spread change (spread_{t+1} − spread_t)",
        f"- Features: `{'`, `'.join(FEATURE_COLS)}`\n",

        "## Model Performance\n",
        "| Model | MAE | RMSE | R² | Direction Acc |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        dacc = f"{m['dir_acc']:.1%}" if m["dir_acc"] is not None else "N/A"
        lines.append(f"| {m['model']} | {m['mae']} | {m['rmse']} | {m['r2']} | {dacc} |")

    best = min(models, key=lambda m: m["mae"])
    lines.append(f"\n**Best model by MAE:** {best['model']} (MAE={best['mae']})\n")

    # OLS coefficients
    ols = next((m for m in models if m["model"] == "OLS regression"), None)
    if ols:
        sorted_coefs = sorted(ols["coefs"].items(), key=lambda x: -abs(x[1]))
        lines += [
            "## OLS Coefficients\n",
            "| Feature | Coefficient |",
            "|---|---|",
        ]
        for feat, coef in sorted_coefs:
            lines.append(f"| `{feat}` | {coef} |")
        lines.append(f"| intercept | {ols['intercept']} |\n")

    # Lasso
    lasso = next((m for m in models if "alpha" in m), None)
    if lasso:
        active = {k: v for k, v in lasso["coefs"].items() if v != 0}
        zeroed = lasso["zeroed_feats"]
        sorted_active = sorted(active.items(), key=lambda x: -abs(x[1]))
        lines += [
            f"## Lasso — α={lasso['alpha']} | {lasso['n_nonzero']}/{len(FEATURE_COLS)} features retained\n",
            "| Feature | Coefficient |",
            "|---|---|",
        ]
        for feat, coef in sorted_active:
            lines.append(f"| `{feat}` | {coef} |")
        if zeroed:
            lines.append(f"\nZeroed out: {', '.join(f'`{f}`' for f in zeroed)}\n")

    # XGBoost feature importance
    xgb = next((m for m in models if "feature_importance" in m), None)
    if xgb:
        sorted_fi = sorted(xgb["feature_importance"].items(), key=lambda x: -x[1])
        lines += [
            "## XGBoost Feature Importance\n",
            "| Feature | Importance |",
            "|---|---|",
        ]
        for feat, imp in sorted_fi:
            lines.append(f"| `{feat}` | {imp} |")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    t0 = time.time()

    print("Loading data...")
    df   = load_all()
    wide = create_features(df)
    full, train, test = prepare_data(wide)
    print(f"  Train: {len(train)} days | Test: {len(test)} days\n")

    models = []

    print("1/3  OLS regression...")
    m = run_ols(train, test)
    models.append(m)
    print(f"     MAE={m['mae']}  R²={m['r2']}  DirAcc={m['dir_acc']}")

    print("2/3  Lasso (LassoCV)...")
    m = run_lasso(train, test)
    models.append(m)
    print(f"     α={m['alpha']}  non-zero={m['n_nonzero']}  MAE={m['mae']}  R²={m['r2']}")

    print("3/3  XGBoost regressor...")
    m = run_xgboost(train, test)
    models.append(m)
    print(f"     MAE={m['mae']}  R²={m['r2']}  DirAcc={m['dir_acc']}")

    elapsed = time.time() - t0
    note = build_note(
        models,
        train_len=len(train), test_len=len(test),
        train_start=str(train.index[0].date()), train_end=str(train.index[-1].date()),
        test_start=str(test.index[0].date()),   test_end=str(test.index[-1].date()),
        elapsed=elapsed,
    )

    with open(RESULTS_PATH, "a") as f:
        f.write(note)

    print(f"\nAppended to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

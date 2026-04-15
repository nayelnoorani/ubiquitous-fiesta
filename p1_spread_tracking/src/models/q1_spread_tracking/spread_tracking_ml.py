"""
Q1 — Spread Tracking
How closely do Aave V3 and Compound V3 USDC borrow rates track each other?

Models: naive persistence, ARIMA, linear regression, XGBoost
Metrics: MAE, RMSE, R², direction accuracy
Output: results written to results.md in this directory
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features, select_features

RESULTS_PATH = Path(__file__).parent / "results.md"

TARGET = "spread_vs_net"
LAG_COL = "spread_vs_net_lag_1d"

FEATURE_COLS = [
    "spread_vs_net_lag_1d",
    "spread_vs_net_rolling_mean_7d",
    "spread_vs_net_rolling_std_7d",
    "aave_rate_change_1d",
    "compound_net_change_1d",
    "tvl_ratio",
    "aave_tvl_change_pct_1d",
    "days_since_spike",
    "day_of_week",
]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_lag: np.ndarray) -> float:
    """Fraction of days where predicted direction of change matches actual."""
    actual_dir = np.sign(y_true - y_lag)
    pred_dir   = np.sign(y_pred - y_lag)
    valid      = pred_dir != 0          # exclude ties (naive baseline predicts no change)
    if valid.sum() == 0:
        return float("nan")
    return (actual_dir[valid] == pred_dir[valid]).mean()


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_lag: np.ndarray) -> dict:
    return {
        "MAE":      mean_absolute_error(y_true, y_pred),
        "RMSE":     np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":       r2_score(y_true, y_pred),
        "DirAcc":   direction_accuracy(y_true, y_pred, y_lag),
    }


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

def naive_persistence(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    y_pred = test[LAG_COL].values
    y_true = test[TARGET].values
    y_lag  = test[LAG_COL].values
    return metrics(y_true, y_pred, y_lag)


def run_arima(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    from statsmodels.tsa.arima.model import ARIMA

    series = train[TARGET].values   # plain array — avoids statsmodels DatetimeIndex warnings
    best_aic, best_order = np.inf, (1, 0, 1)

    for p in range(0, 3):
        for q in range(0, 3):
            try:
                aic = ARIMA(series, order=(p, 0, q)).fit().aic
                if aic < best_aic:
                    best_aic, best_order = aic, (p, 0, q)
            except Exception:
                continue

    fitted   = ARIMA(series, order=best_order).fit()
    y_pred   = fitted.forecast(steps=len(test))
    y_true   = test[TARGET].values
    y_lag    = test[LAG_COL].values

    result          = metrics(y_true, y_pred, y_lag)
    result["order"] = best_order
    result["aic"]   = round(best_aic, 2)
    return result


def run_linear_regression(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_true  = test[TARGET].values
    y_lag   = test[LAG_COL].values

    model   = LinearRegression().fit(X_train, y_train)
    y_pred  = model.predict(X_test)

    result              = metrics(y_true, y_pred, y_lag)
    result["coef"]      = dict(zip(FEATURE_COLS, model.coef_.round(4)))
    result["intercept"] = round(float(model.intercept_), 4)
    return result


def run_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_true  = test[TARGET].values
    y_lag   = test[LAG_COL].values

    # Walk-forward CV on training set to select n_estimators
    tscv      = TimeSeriesSplit(n_splits=5)
    best_n    = 100
    best_mae  = np.inf

    for n in [50, 100, 200, 300]:
        fold_maes = []
        for tr_idx, val_idx in tscv.split(X_train):
            xgb = XGBRegressor(n_estimators=n, learning_rate=0.05, max_depth=4,
                               random_state=42, verbosity=0)
            xgb.fit(X_train[tr_idx], y_train[tr_idx])
            fold_maes.append(mean_absolute_error(y_train[val_idx], xgb.predict(X_train[val_idx])))
        if np.mean(fold_maes) < best_mae:
            best_mae, best_n = np.mean(fold_maes), n

    model  = XGBRegressor(n_estimators=best_n, learning_rate=0.05, max_depth=4,
                          random_state=42, verbosity=0).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result                        = metrics(y_true, y_pred, y_lag)
    result["best_n"]              = best_n
    result["feature_importance"]  = dict(zip(FEATURE_COLS, model.feature_importances_.round(4)))
    return result


# ----------------------------------------------------------------------
# Markdown output
# ----------------------------------------------------------------------

def build_note(results: dict, train: pd.DataFrame, test: pd.DataFrame, elapsed: float) -> str:
    now      = datetime.now().strftime("%Y-%m-%d %H:%M")
    r        = results

    model_rows = [
        ("Naive persistence",                        r["naive"]),
        (f"ARIMA{r['arima']['order']}",              r["arima"]),
        ("Linear regression",                        r["lr"]),
        (f"XGBoost (n_estimators={r['xgb']['best_n']})", r["xgb"]),
    ]

    best_name, best_r = min(model_rows, key=lambda x: x[1]["MAE"])

    lines = [
        "# Q1 Spread Tracking — Results\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",
        "## Run Details\n",
        f"- Training: {train.index.min().date()} → {train.index.max().date()} ({len(train):,} days)",
        f"- Test:     {test.index.min().date()} → {test.index.max().date()} ({len(test):,} days)",
        f"- Target:   `spread_vs_net` (Aave apyBase − Compound net borrow cost)\n",
        "## Model Performance\n",
        "| Model | MAE | RMSE | R² | Direction Acc |",
        "|---|---|---|---|---|",
    ]

    for name, res in model_rows:
        da = f"{res['DirAcc']:.1%}" if not np.isnan(res["DirAcc"]) else "N/A"
        lines.append(f"| {name} | {res['MAE']:.4f} | {res['RMSE']:.4f} | {res['R2']:.4f} | {da} |")

    lines += [
        f"\n**Best model by MAE:** {best_name} (MAE={best_r['MAE']:.4f})\n",
        "## XGBoost Feature Importance\n",
        "| Feature | Importance |",
        "|---|---|",
    ]
    for feat, imp in sorted(r["xgb"]["feature_importance"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{feat}` | {imp:.4f} |")

    lines += [
        "\n## Linear Regression Coefficients\n",
        "| Feature | Coefficient |",
        "|---|---|",
    ]
    for feat, coef in sorted(r["lr"]["coef"].items(), key=lambda x: -abs(x[1])):
        lines.append(f"| `{feat}` | {coef:.4f} |")

    naive_mae    = r["naive"]["MAE"]
    improvement  = (naive_mae - best_r["MAE"]) / naive_mae * 100
    top_feature  = max(r["xgb"]["feature_importance"].items(), key=lambda x: x[1])[0]
    naive_da_str = "N/A (predicts no change)"
    best_da_str  = f"{best_r['DirAcc']:.1%}" if not np.isnan(best_r["DirAcc"]) else "N/A"

    lines += [
        "\n## Key Findings\n",
        f"- Best model ({best_name}) achieved MAE of {best_r['MAE']:.4f}% vs naive baseline "
        f"of {naive_mae:.4f}% — {improvement:.1f}% improvement.",
        f"- Naive persistence direction accuracy: {naive_da_str}.",
        f"- Best model direction accuracy: {best_da_str}.",
        f"- Top XGBoost feature: `{top_feature}`.",
        f"- Best ARIMA order: {r['arima']['order']} (AIC={r['arima']['aic']}).",
        "",
    ]

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    t0 = time.time()

    print("Loading and engineering features...")
    df   = load_all()
    wide = create_features(df)
    _, reduced = select_features(wide)

    # Ensure target is present
    if TARGET not in reduced.columns:
        reduced[TARGET] = wide[TARGET]

    all_cols = [TARGET] + FEATURE_COLS
    all_cols = list(dict.fromkeys(all_cols))   # deduplicate, preserve order
    data = reduced[all_cols].dropna()
    print(f"  {len(data):,} rows after dropping NaN warmup rows")

    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows\n")

    results = {}

    print("1/4  Naive persistence...")
    results["naive"] = naive_persistence(train, test)

    print("2/4  ARIMA (AIC grid search p,q ∈ [0,2])...")
    results["arima"] = run_arima(train, test)
    print(f"     Best order: {results['arima']['order']}  AIC={results['arima']['aic']}")

    print("3/4  Linear regression...")
    results["lr"] = run_linear_regression(train, test)

    print("4/4  XGBoost (walk-forward CV)...")
    results["xgb"] = run_xgboost(train, test)
    print(f"     Best n_estimators: {results['xgb']['best_n']}")

    elapsed = time.time() - t0

    # Console table
    print(f"\n{'Model':<38} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'DirAcc':>8}")
    print("-" * 72)
    rows = [
        ("Naive persistence",                            results["naive"]),
        (f"ARIMA{results['arima']['order']}",            results["arima"]),
        ("Linear regression",                            results["lr"]),
        (f"XGBoost (n={results['xgb']['best_n']})",      results["xgb"]),
    ]
    for name, res in rows:
        da = f"{res['DirAcc']:.1%}" if not np.isnan(res["DirAcc"]) else "  N/A"
        print(f"{name:<38} {res['MAE']:>7.4f} {res['RMSE']:>7.4f} {res['R2']:>7.4f} {da:>8}")

    print(f"\nElapsed: {elapsed:.1f}s")

    note = build_note(results, train, test, elapsed)
    RESULTS_PATH.write_text(note)
    print(f"Results written → {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

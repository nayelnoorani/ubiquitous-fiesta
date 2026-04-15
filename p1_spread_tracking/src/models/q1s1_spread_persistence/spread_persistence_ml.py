"""
Q1S1 — Spread Persistence: ML Models

Models:
  1. Naive persistence — predict tomorrow's spread = today's (baseline)
  2. AR(p) — autoregressive model, order chosen by AIC
  3. Linear regression — spread_t+1 ~ spread_lag_1d + rolling_mean_7d

Target:  next-day spread (spread_vs_net shifted by -1)
Split:   80/20 chronological
Metrics: MAE, RMSE, R², direction accuracy
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH = Path(__file__).parent / "results.md"

FEATURE_COLS = [
    "spread_vs_net_lag_1d",
    "spread_vs_net_rolling_mean_7d",
]
TARGET = "spread_vs_net"   # predict tomorrow → created as shift(-1) below


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                        y_prev: np.ndarray) -> float:
    """
    Fraction of days where predicted direction of change matches actual.
    Direction = sign(tomorrow - today). Ties (pred == 0) are excluded.
    """
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    valid = pred_dir != 0
    if valid.sum() == 0:
        return float("nan")
    return float((true_dir[valid] == pred_dir[valid]).mean())


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray,
             y_prev: np.ndarray) -> dict:
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = mean_squared_error(y_true, y_pred) ** 0.5
    r2    = r2_score(y_true, y_pred)
    dacc  = direction_accuracy(y_true, y_pred, y_prev)
    return {"model": name, "mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 4), "dir_acc": round(dacc, 4) if not np.isnan(dacc) else None}


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def prepare_data(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = wide[FEATURE_COLS + [TARGET]].copy()

    # Target: next-day spread
    df["target"] = df[TARGET].shift(-1)
    df = df.dropna()

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test  = df.iloc[split:]
    return df, train, test


# ----------------------------------------------------------------------
# Model 1 — Naive persistence
# ----------------------------------------------------------------------

def naive_persistence(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    # Predict tomorrow's spread = today's spread (lag_1d is today's spread)
    y_pred = test["spread_vs_net_lag_1d"].values
    y_true = test["target"].values
    y_prev = test["spread_vs_net_lag_1d"].values
    return evaluate("Naive persistence", y_true, y_pred, y_prev)


# ----------------------------------------------------------------------
# Model 2 — AR(p) via OLS, order chosen by AIC
# ----------------------------------------------------------------------

def run_ar(train: pd.DataFrame, test: pd.DataFrame,
           full: pd.DataFrame, max_p: int = 10) -> dict:
    """
    Fit AR(p) by OLS regression of target on p lags of spread.
    Select p by AIC on the training set.
    """
    spread_train = train[TARGET].values
    spread_test  = test[TARGET].values

    best_p, best_aic = 1, np.inf

    for p in range(1, max_p + 1):
        # Build lag matrix for training set
        n = len(spread_train)
        X = np.column_stack([spread_train[p - i - 1: n - i - 1] for i in range(p)])
        y = spread_train[p:]
        if len(y) < p + 2:
            break
        X_c = np.column_stack([np.ones(len(X)), X])
        coefs, resid, _, _ = np.linalg.lstsq(X_c, y, rcond=None)
        rss = float(np.sum((y - X_c @ coefs) ** 2))
        k   = p + 1  # intercept + p lags
        aic = len(y) * np.log(rss / len(y)) + 2 * k
        if aic < best_aic:
            best_aic, best_p = aic, p

    # Refit with best_p on full training set
    p = best_p
    n = len(spread_train)
    X_tr = np.column_stack([spread_train[p - i - 1: n - i - 1] for i in range(p)])
    y_tr = spread_train[p:]
    X_c  = np.column_stack([np.ones(len(X_tr)), X_tr])
    coefs, _, _, _ = np.linalg.lstsq(X_c, y_tr, rcond=None)

    # One-step-ahead forecast on test set (rolling, using actuals as lags)
    # Concatenate last p training values with test for lagged inputs
    history = list(spread_train[-p:])
    y_pred  = []
    for val in spread_test:
        x  = np.array([1.0] + history[-p:][::-1])
        y_pred.append(float(x @ coefs))
        history.append(val)

    y_true = spread_test
    y_prev = test["spread_vs_net_lag_1d"].values
    result = evaluate(f"AR({p}) — AIC={best_aic:.1f}", np.array(y_true),
                      np.array(y_pred), y_prev)
    result["ar_order"] = p
    result["aic"]      = round(best_aic, 2)
    return result


# ----------------------------------------------------------------------
# Model 3 — Linear regression
# ----------------------------------------------------------------------

def run_linear(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_tr = train[FEATURE_COLS].values
    y_tr = train["target"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["target"].values

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prev = test["spread_vs_net_lag_1d"].values

    result = evaluate("Linear regression", y_te, y_pred, y_prev)
    result["coefs"] = {col: round(float(c), 6)
                       for col, c in zip(FEATURE_COLS, model.coef_)}
    result["intercept"] = round(float(model.intercept_), 6)
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
        "# Q1S1 — Spread Persistence: ML Models\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",
        "## Run Details\n",
        f"- Training: {train_start} → {train_end} ({train_len} days)",
        f"- Test:     {test_start} → {test_end} ({test_len} days)",
        f"- Target:   next-day `spread_vs_net`",
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

    # AR order
    ar = next((m for m in models if "ar_order" in m), None)
    if ar:
        lines.append(f"## AR Model\n")
        lines.append(f"- Selected order: p={ar['ar_order']} (AIC={ar['aic']})\n")

    # Linear regression coefficients
    lr = next((m for m in models if "coefs" in m), None)
    if lr:
        lines += [
            "## Linear Regression Coefficients\n",
            "| Feature | Coefficient |",
            "|---|---|",
        ]
        for feat, coef in lr["coefs"].items():
            lines.append(f"| `{feat}` | {coef} |")
        lines.append(f"| intercept | {lr['intercept']} |\n")

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

    print("1/3  Naive persistence...")
    m = naive_persistence(train, test)
    models.append(m)
    print(f"     MAE={m['mae']}  R²={m['r2']}  DirAcc={m['dir_acc']}")

    print("2/3  AR(p) — order by AIC...")
    m = run_ar(train, test, full)
    models.append(m)
    print(f"     AR order={m['ar_order']}  AIC={m['aic']}  MAE={m['mae']}  R²={m['r2']}  DirAcc={m['dir_acc']}")

    print("3/3  Linear regression...")
    m = run_linear(train, test)
    models.append(m)
    print(f"     MAE={m['mae']}  R²={m['r2']}  DirAcc={m['dir_acc']}")

    elapsed = time.time() - t0

    note = build_note(
        models,
        train_len=len(train), test_len=len(test),
        train_start=str(train.index[0].date()), train_end=str(train.index[-1].date()),
        test_start=str(test.index[0].date()),  test_end=str(test.index[-1].date()),
        elapsed=elapsed,
    )

    with open(RESULTS_PATH, "a") as f:
        f.write(note)

    print(f"\nAppended to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

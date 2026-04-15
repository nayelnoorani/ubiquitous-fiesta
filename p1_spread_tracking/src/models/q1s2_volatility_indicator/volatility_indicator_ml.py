"""
Q1S2 — Volatility as a Leading Indicator: ML Models

Target:  binary — does |spread| decrease over the next 3 days? (1 = narrows, 0 = widens)
Split:   80/20 chronological
Models:
  1. Threshold rule — rolling_std > 75th pct → predict narrowing (baseline)
  2. Logistic regression
  3. XGBoost classifier
  4. Linear regression — predict magnitude of spread change (regression variant)

Metrics: AUC-ROC, precision, recall, F1 (classifiers); MAE, direction accuracy (regression)
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, mean_absolute_error)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH  = Path(__file__).parent / "results.md"
HORIZON       = 3      # days ahead
VOL_THRESHOLD = 0.75   # percentile for threshold rule

FEATURE_COLS = [
    "spread_vs_net_lag_1d",
    "spread_vs_net_rolling_std_7d",
    "spread_vs_net_rolling_mean_7d",
    "aave_rate_change_1d",
    "compound_net_change_1d",
]


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def prepare_data(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = wide[FEATURE_COLS + ["spread_vs_net"]].copy().dropna()

    # Binary target: |spread| narrows over the next HORIZON days
    future_abs  = df["spread_vs_net"].abs().shift(-HORIZON)
    current_abs = df["spread_vs_net"].abs()
    df["target_cls"] = (future_abs < current_abs).astype(int)

    # Regression target: change in |spread| over HORIZON days
    df["target_reg"] = future_abs - current_abs

    df = df.dropna()

    split = int(len(df) * 0.8)
    return df, df.iloc[:split], df.iloc[split:]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def clf_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray,
                threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model":     name,
        "auc":       round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "pos_rate":  round(float(y_pred.mean()), 4),
    }


# ----------------------------------------------------------------------
# Model 1 — Threshold rule
# ----------------------------------------------------------------------

def threshold_rule(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    cutoff   = train["spread_vs_net_rolling_std_7d"].quantile(VOL_THRESHOLD)
    y_prob   = (test["spread_vs_net_rolling_std_7d"] >= cutoff).astype(float).values
    y_true   = test["target_cls"].values
    result   = clf_metrics("Threshold rule (vol > 75th pct)", y_true, y_prob)
    result["vol_cutoff"] = round(float(cutoff), 4)
    return result


# ----------------------------------------------------------------------
# Model 2 — Logistic regression
# ----------------------------------------------------------------------

def run_logistic(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_tr = train[FEATURE_COLS].values
    y_tr = train["target_cls"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["target_cls"].values

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]

    result = clf_metrics("Logistic regression", y_te, y_prob)
    result["coefs"] = {col: round(float(c), 4)
                       for col, c in zip(FEATURE_COLS, model.coef_[0])}
    return result


# ----------------------------------------------------------------------
# Model 3 — XGBoost classifier
# ----------------------------------------------------------------------

def run_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_tr = train[FEATURE_COLS].values
    y_tr = train["target_cls"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["target_cls"].values

    scale_pos = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]

    result = clf_metrics("XGBoost classifier", y_te, y_prob)
    result["feature_importance"] = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLS, model.feature_importances_)
    }
    return result


# ----------------------------------------------------------------------
# Model 4 — Linear regression (magnitude of spread change)
# ----------------------------------------------------------------------

def run_linear_reg(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_tr = train[FEATURE_COLS].values
    y_tr = train["target_reg"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["target_reg"].values

    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    mae      = mean_absolute_error(y_te, y_pred)
    dir_true = np.sign(y_te)
    dir_pred = np.sign(y_pred)
    valid    = dir_pred != 0
    dir_acc  = float((dir_true[valid] == dir_pred[valid]).mean()) if valid.sum() > 0 else float("nan")

    return {
        "model":    "Linear regression (magnitude)",
        "mae":      round(float(mae), 4),
        "dir_acc":  round(dir_acc, 4),
        "coefs":    {col: round(float(c), 4)
                     for col, c in zip(FEATURE_COLS, model.coef_)},
        "intercept": round(float(model.intercept_), 4),
    }


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(clf_results: list[dict], reg_result: dict,
               class_balance: dict, train_len: int, test_len: int,
               train_start: str, train_end: str,
               test_start: str, test_end: str, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "\n---\n",
        "# Q1S2 — Volatility as a Leading Indicator: ML Models\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",
        "## Run Details\n",
        f"- Training: {train_start} → {train_end} ({train_len} days)",
        f"- Test:     {test_start} → {test_end} ({test_len} days)",
        f"- Target:   binary — does |spread| narrow over the next {HORIZON} days?",
        f"- Features: `{'`, `'.join(FEATURE_COLS)}`\n",
        "## Class Balance\n",
        "| | Train | Test |",
        "|---|---|---|",
        f"| Narrows (1) | {class_balance['train_pos']:.1%} | {class_balance['test_pos']:.1%} |",
        f"| Widens  (0) | {class_balance['train_neg']:.1%} | {class_balance['test_neg']:.1%} |",

        "\n## Classifier Performance\n",
        "| Model | AUC-ROC | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for m in clf_results:
        lines.append(f"| {m['model']} | {m['auc']} | {m['precision']} | {m['recall']} | {m['f1']} |")

    best = max(clf_results, key=lambda m: m["auc"])
    lines.append(f"\n**Best classifier by AUC:** {best['model']} (AUC={best['auc']})\n")

    # Logistic regression coefficients
    lr = next((m for m in clf_results if "coefs" in m and "auc" in m), None)
    if lr:
        lines += [
            "## Logistic Regression Coefficients\n",
            "| Feature | Coefficient |",
            "|---|---|",
        ]
        for feat, coef in lr["coefs"].items():
            lines.append(f"| `{feat}` | {coef} |")

    # XGBoost feature importance
    xgb = next((m for m in clf_results if "feature_importance" in m), None)
    if xgb:
        sorted_fi = sorted(xgb["feature_importance"].items(), key=lambda x: -x[1])
        lines += [
            "\n## XGBoost Feature Importance\n",
            "| Feature | Importance |",
            "|---|---|",
        ]
        for feat, imp in sorted_fi:
            lines.append(f"| `{feat}` | {imp} |")

    lines += [
        "\n## Regression Variant — Spread Change Magnitude\n",
        "| Metric | Value |",
        "|---|---|",
        f"| MAE | {reg_result['mae']}% |",
        f"| Direction accuracy | {reg_result['dir_acc']:.1%} |",
        "\n| Feature | Coefficient |",
        "|---|---|",
    ]
    for feat, coef in reg_result["coefs"].items():
        lines.append(f"| `{feat}` | {coef} |")
    lines.append(f"| intercept | {reg_result['intercept']} |")
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
    print(f"  Train: {len(train)} days | Test: {len(test)} days")

    cb = {
        "train_pos": float(train["target_cls"].mean()),
        "train_neg": float(1 - train["target_cls"].mean()),
        "test_pos":  float(test["target_cls"].mean()),
        "test_neg":  float(1 - test["target_cls"].mean()),
    }
    print(f"  Class balance — train: {cb['train_pos']:.1%} narrows | test: {cb['test_pos']:.1%} narrows\n")

    clf_results = []

    print("1/4  Threshold rule...")
    m = threshold_rule(train, test)
    clf_results.append(m)
    print(f"     AUC={m['auc']}  F1={m['f1']}  vol_cutoff={m['vol_cutoff']}%")

    print("2/4  Logistic regression...")
    m = run_logistic(train, test)
    clf_results.append(m)
    print(f"     AUC={m['auc']}  F1={m['f1']}")

    print("3/4  XGBoost classifier...")
    m = run_xgboost(train, test)
    clf_results.append(m)
    print(f"     AUC={m['auc']}  F1={m['f1']}")

    print("4/4  Linear regression (magnitude)...")
    reg_result = run_linear_reg(train, test)
    print(f"     MAE={reg_result['mae']}%  DirAcc={reg_result['dir_acc']:.1%}")

    elapsed = time.time() - t0
    note = build_note(
        clf_results, reg_result, cb,
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

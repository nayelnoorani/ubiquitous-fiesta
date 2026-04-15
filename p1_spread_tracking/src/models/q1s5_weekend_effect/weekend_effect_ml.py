"""
Q1S5 — Weekend Effect: ML Models

Target:  binary — does spread persist same sign to next day? (1 = persists, 0 = flips)
Split:   80/20 chronological
Models:
  1. Linear regression with day-of-week dummies — quantify per-day effect on spread change
  2. Logistic regression — predict persistence with day_of_week as a feature
  3. XGBoost classifier — allows interaction between day_of_week and spread features

Metrics: AUC-ROC, precision, recall, F1; feature importance of day_of_week
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, mean_absolute_error)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH = Path(__file__).parent / "results.md"

SPREAD_FEATURES = [
    "spread_vs_net_lag_1d",
    "spread_vs_net_rolling_mean_7d",
    "spread_vs_net_rolling_std_7d",
]
DOW_COL = "day_of_week"


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def prepare_data(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = wide[SPREAD_FEATURES + [DOW_COL, "spread_vs_net"]].copy().dropna()

    # Day-of-week dummies (drop Monday as reference)
    dummies = pd.get_dummies(df[DOW_COL], prefix="dow", drop_first=True).astype(float)
    df = pd.concat([df, dummies], axis=1)

    # Target: spread persists same sign to next day
    next_sign = np.sign(df["spread_vs_net"].shift(-1))
    curr_sign = np.sign(df["spread_vs_net"])
    df["target"] = ((curr_sign == next_sign) & (curr_sign != 0)).astype(int)

    # Spread change target for linear regression
    df["spread_change"] = df["spread_vs_net"].shift(-1) - df["spread_vs_net"]

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
    }


# ----------------------------------------------------------------------
# Model 1 — Linear regression with day-of-week dummies
# ----------------------------------------------------------------------

def run_linear_dummies(train: pd.DataFrame, test: pd.DataFrame,
                       dummy_cols: list[str]) -> dict:
    feature_cols = SPREAD_FEATURES + dummy_cols
    model = LinearRegression()
    model.fit(train[feature_cols].values, train["spread_change"].values)
    y_pred = model.predict(test[feature_cols].values)
    y_true = test["spread_change"].values

    mae = mean_absolute_error(y_true, y_pred)
    dir_true = np.sign(y_true)
    dir_pred = np.sign(y_pred)
    valid    = dir_pred != 0
    dir_acc  = float((dir_true[valid] == dir_pred[valid]).mean()) if valid.sum() > 0 else float("nan")

    coefs = {col: round(float(c), 6)
             for col, c in zip(feature_cols, model.coef_)}
    return {
        "model":     "Linear regression (DoW dummies)",
        "mae":       round(float(mae), 4),
        "dir_acc":   round(float(dir_acc), 4),
        "coefs":     coefs,
        "intercept": round(float(model.intercept_), 6),
    }


# ----------------------------------------------------------------------
# Model 2 — Logistic regression
# ----------------------------------------------------------------------

def run_logistic(train: pd.DataFrame, test: pd.DataFrame,
                 dummy_cols: list[str]) -> dict:
    feature_cols = SPREAD_FEATURES + [DOW_COL] + dummy_cols
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(train[feature_cols].values, train["target"].values)
    y_prob = model.predict_proba(test[feature_cols].values)[:, 1]
    y_true = test["target"].values

    result = clf_metrics("Logistic regression", y_true, y_prob)
    result["coefs"] = {col: round(float(c), 4)
                       for col, c in zip(feature_cols, model.coef_[0])}
    return result


# ----------------------------------------------------------------------
# Model 3 — XGBoost classifier
# ----------------------------------------------------------------------

def run_xgboost(train: pd.DataFrame, test: pd.DataFrame,
                dummy_cols: list[str]) -> dict:
    feature_cols = SPREAD_FEATURES + [DOW_COL] + dummy_cols
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    model.fit(train[feature_cols].values, train["target"].values)
    y_prob = model.predict_proba(test[feature_cols].values)[:, 1]
    y_true = test["target"].values

    result = clf_metrics("XGBoost classifier", y_true, y_prob)
    result["feature_importance"] = {
        col: round(float(imp), 4)
        for col, imp in zip(feature_cols, model.feature_importances_)
    }
    return result


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(linear: dict, clf_results: list[dict], dummy_cols: list[str],
               class_balance: dict, train_len: int, test_len: int,
               train_start: str, train_end: str,
               test_start: str, test_end: str, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "\n---\n",
        "# Q1S5 — Weekend Effect: ML Models\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",
        "## Run Details\n",
        f"- Training: {train_start} → {train_end} ({train_len} days)",
        f"- Test:     {test_start} → {test_end} ({test_len} days)",
        f"- Classifier target: spread persists same sign to next day",
        f"- Regression target: next-day spread change\n",
        "## Class Balance\n",
        "| | Train | Test |",
        "|---|---|---|",
        f"| Persists (1) | {class_balance['train_pos']:.1%} | {class_balance['test_pos']:.1%} |",
        f"| Flips    (0) | {class_balance['train_neg']:.1%} | {class_balance['test_neg']:.1%} |",

        "\n## Linear Regression — Day-of-Week Effect on Spread Change\n",
        f"MAE: {linear['mae']}%  |  Direction accuracy: {linear['dir_acc']:.1%}\n",
        "| Feature | Coefficient |",
        "|---|---|",
    ]
    dow_coefs = {k: v for k, v in linear["coefs"].items() if k.startswith("dow_")}
    other_coefs = {k: v for k, v in linear["coefs"].items() if not k.startswith("dow_")}
    for feat, coef in sorted(other_coefs.items(), key=lambda x: -abs(x[1])):
        lines.append(f"| `{feat}` | {coef} |")
    for feat, coef in sorted(dow_coefs.items()):
        lines.append(f"| `{feat}` (dummy) | {coef} |")
    lines.append(f"| intercept | {linear['intercept']} |\n")

    lines += [
        "## Classifier Performance\n",
        "| Model | AUC-ROC | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for m in clf_results:
        lines.append(f"| {m['model']} | {m['auc']} | {m['precision']} | {m['recall']} | {m['f1']} |")

    best = max(clf_results, key=lambda m: m["auc"])
    lines.append(f"\n**Best classifier by AUC:** {best['model']} (AUC={best['auc']})\n")

    # XGBoost feature importance
    xgb = next((m for m in clf_results if "feature_importance" in m), None)
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

    dummy_cols = [c for c in full.columns if c.startswith("dow_")]
    print(f"  Train: {len(train)} days | Test: {len(test)} days")

    cb = {
        "train_pos": float(train["target"].mean()),
        "train_neg": float(1 - train["target"].mean()),
        "test_pos":  float(test["target"].mean()),
        "test_neg":  float(1 - test["target"].mean()),
    }
    print(f"  Class balance — train: {cb['train_pos']:.1%} persists | test: {cb['test_pos']:.1%} persists\n")

    print("1/3  Linear regression with day-of-week dummies...")
    linear = run_linear_dummies(train, test, dummy_cols)
    print(f"     MAE={linear['mae']}  DirAcc={linear['dir_acc']:.1%}")

    clf_results = []

    print("2/3  Logistic regression...")
    m = run_logistic(train, test, dummy_cols)
    clf_results.append(m)
    print(f"     AUC={m['auc']}  F1={m['f1']}")

    print("3/3  XGBoost classifier...")
    m = run_xgboost(train, test, dummy_cols)
    clf_results.append(m)
    print(f"     AUC={m['auc']}  F1={m['f1']}")

    elapsed = time.time() - t0
    note = build_note(
        linear, clf_results, dummy_cols, cb,
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

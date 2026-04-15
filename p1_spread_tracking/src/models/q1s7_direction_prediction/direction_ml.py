"""
Q1S7 — Spread Direction Prediction: ML Models

Target:  binary — spread widens tomorrow (spread_{t+1} > spread_t) = 1
Split:   80/20 chronological; walk-forward CV (TimeSeriesSplit, 5 folds) on train set
Models:
  1. Logistic regression
  2. Random forest (walk-forward CV for n_estimators, max_depth)
  3. XGBoost classifier (walk-forward CV for learning_rate, max_depth)
  4. LSTM (sequence length 7)

Metrics: AUC-ROC, precision, recall, F1, accuracy, Brier score
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score, brier_score_loss)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH = Path(__file__).parent / "results.md"
N_SPLITS     = 5
LSTM_SEQ_LEN = 7

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
# Data prep
# ----------------------------------------------------------------------

def prepare(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = wide[FEATURE_COLS + ["spread_vs_net"]].copy().dropna()
    next_spread  = df["spread_vs_net"].shift(-1)
    df["target"] = (next_spread > df["spread_vs_net"]).astype(int)
    df = df.dropna()
    split = int(len(df) * 0.8)
    return df, df.iloc[:split], df.iloc[split:]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray,
             threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model":     name,
        "auc":       round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "brier":     round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def wf_cv_score(model_fn, X: np.ndarray, y: np.ndarray,
                n_splits: int = N_SPLITS) -> float:
    """Walk-forward cross-validation — returns mean AUC across folds."""
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr_idx, va_idx in tscv.split(X):
        m = model_fn()
        m.fit(X[tr_idx], y[tr_idx])
        prob = m.predict_proba(X[va_idx])[:, 1]
        try:
            scores.append(roc_auc_score(y[va_idx], prob))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else float("nan")


# ----------------------------------------------------------------------
# Model 1 — Logistic regression
# ----------------------------------------------------------------------

def run_logistic(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(train[FEATURE_COLS].values)
    X_te   = scaler.transform(test[FEATURE_COLS].values)
    y_tr   = train["target"].values
    y_te   = test["target"].values

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]

    result = evaluate("Logistic regression", y_te, y_prob)
    result["coefs"] = {col: round(float(c), 4)
                       for col, c in zip(FEATURE_COLS, model.coef_[0])}
    return result


# ----------------------------------------------------------------------
# Model 2 — Random forest with walk-forward CV
# ----------------------------------------------------------------------

def run_random_forest(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_tr = train[FEATURE_COLS].values
    y_tr = train["target"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["target"].values

    best_auc, best_params = -1, {}
    for n_est in [100, 300]:
        for max_d in [4, 6]:
            auc = wf_cv_score(
                lambda n=n_est, d=max_d: RandomForestClassifier(
                    n_estimators=n, max_depth=d, min_samples_leaf=10,
                    class_weight="balanced", random_state=42, n_jobs=-1),
                X_tr, y_tr)
            if auc > best_auc:
                best_auc, best_params = auc, {"n_estimators": n_est, "max_depth": max_d}

    model = RandomForestClassifier(
        **best_params, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]

    result = evaluate(
        f"Random forest (n={best_params['n_estimators']}, d={best_params['max_depth']})",
        y_te, y_prob)
    result["cv_auc"] = round(best_auc, 4)
    result["feature_importance"] = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLS, model.feature_importances_)
    }
    return result


# ----------------------------------------------------------------------
# Model 3 — XGBoost with walk-forward CV
# ----------------------------------------------------------------------

def run_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_tr = train[FEATURE_COLS].values
    y_tr = train["target"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["target"].values

    scale_pos = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)

    best_auc, best_params = -1, {}
    for lr in [0.05, 0.1]:
        for max_d in [3, 4]:
            auc = wf_cv_score(
                lambda l=lr, d=max_d: XGBClassifier(
                    n_estimators=300, learning_rate=l, max_depth=d,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos,
                    eval_metric="logloss", verbosity=0, random_state=42),
                X_tr, y_tr)
            if auc > best_auc:
                best_auc, best_params = auc, {"learning_rate": lr, "max_depth": max_d}

    model = XGBClassifier(
        n_estimators=300, **best_params,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss", verbosity=0, random_state=42)
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]

    result = evaluate(
        f"XGBoost (lr={best_params['learning_rate']}, d={best_params['max_depth']})",
        y_te, y_prob)
    result["cv_auc"] = round(best_auc, 4)
    result["feature_importance"] = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLS, model.feature_importances_)
    }
    return result


# ----------------------------------------------------------------------
# Model 4 — LSTM
# ----------------------------------------------------------------------

def run_lstm(train: pd.DataFrame, test: pd.DataFrame,
             seq_len: int = LSTM_SEQ_LEN) -> dict:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.get_logger().setLevel("ERROR")
    except ImportError:
        return {"model": "LSTM", "error": "tensorflow not installed — skipped"}

    scaler = StandardScaler()
    X_all  = scaler.fit_transform(
        pd.concat([train, test])[FEATURE_COLS].values)
    y_all  = pd.concat([train, test])["target"].values
    n_tr   = len(train)

    # Build sequences
    def make_sequences(X, y, seq):
        Xs, ys = [], []
        for i in range(seq, len(X)):
            Xs.append(X[i - seq: i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = make_sequences(X_all, y_all, seq_len)
    # Training sequences end before test start
    split_seq = n_tr - seq_len
    X_tr_s, y_tr_s = X_seq[:split_seq], y_seq[:split_seq]
    X_te_s, y_te_s = X_seq[split_seq:], y_seq[split_seq:]

    model = Sequential([
        LSTM(32, input_shape=(seq_len, len(FEATURE_COLS)), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(
        X_tr_s, y_tr_s,
        epochs=50, batch_size=32, verbose=0,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    )
    y_prob = model.predict(X_te_s, verbose=0).flatten()

    return evaluate(f"LSTM (seq={seq_len})", y_te_s, y_prob)


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(models: list[dict], class_balance: dict,
               train_len: int, test_len: int,
               train_start: str, train_end: str,
               test_start: str, test_end: str, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "\n---\n",
        "# Q1S7 — Spread Direction Prediction: ML Models\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",
        "## Run Details\n",
        f"- Training: {train_start} → {train_end} ({train_len} days)",
        f"- Test:     {test_start} → {test_end} ({test_len} days)",
        f"- Target:   spread widens tomorrow (spread_{{t+1}} > spread_t)",
        f"- Features: all {len(FEATURE_COLS)} selected features",
        f"- CV:       TimeSeriesSplit ({N_SPLITS} folds) on train set\n",
        "## Class Balance\n",
        "| | Train | Test |",
        "|---|---|---|",
        f"| Widens  (1) | {class_balance['train_pos']:.1%} | {class_balance['test_pos']:.1%} |",
        f"| Narrows (0) | {class_balance['train_neg']:.1%} | {class_balance['test_neg']:.1%} |",

        "\n## Model Performance\n",
        "| Model | AUC-ROC | Accuracy | Precision | Recall | F1 | Brier |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in models:
        if "error" in m:
            lines.append(f"| {m['model']} | — skipped: {m['error']} |")
            continue
        cv = f" (CV AUC={m['cv_auc']})" if "cv_auc" in m else ""
        lines.append(
            f"| {m['model']}{cv} | {m['auc']} | {m['accuracy']} | "
            f"{m['precision']} | {m['recall']} | {m['f1']} | {m['brier']} |"
        )

    valid = [m for m in models if "error" not in m]
    if valid:
        best = max(valid, key=lambda m: m["auc"])
        lines.append(f"\n**Best model by AUC:** {best['model']} (AUC={best['auc']})\n")

    # Logistic regression coefficients
    lr = next((m for m in models if m["model"] == "Logistic regression"), None)
    if lr and "coefs" in lr:
        sorted_coefs = sorted(lr["coefs"].items(), key=lambda x: -abs(x[1]))
        lines += [
            "## Logistic Regression Coefficients\n",
            "| Feature | Coefficient |",
            "|---|---|",
        ]
        for feat, coef in sorted_coefs:
            lines.append(f"| `{feat}` | {coef} |")

    # Feature importance (RF and XGB)
    for label, key in [("Random Forest", "Random forest"), ("XGBoost", "XGBoost")]:
        m = next((m for m in models if key in m["model"] and "feature_importance" in m), None)
        if m:
            sorted_fi = sorted(m["feature_importance"].items(), key=lambda x: -x[1])
            lines += [
                f"\n## {label} Feature Importance\n",
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
    full, train, test = prepare(wide)
    print(f"  Train: {len(train)} days | Test: {len(test)} days\n")

    cb = {
        "train_pos": float(train["target"].mean()),
        "train_neg": float(1 - train["target"].mean()),
        "test_pos":  float(test["target"].mean()),
        "test_neg":  float(1 - test["target"].mean()),
    }
    print(f"  Class balance — train: {cb['train_pos']:.1%} widens | test: {cb['test_pos']:.1%} widens\n")

    models = []

    print("1/4  Logistic regression...")
    m = run_logistic(train, test)
    models.append(m)
    print(f"     AUC={m['auc']}  Accuracy={m['accuracy']}  Brier={m['brier']}")

    print("2/4  Random forest (walk-forward CV)...")
    m = run_random_forest(train, test)
    models.append(m)
    print(f"     CV AUC={m['cv_auc']}  Test AUC={m['auc']}  Accuracy={m['accuracy']}  Brier={m['brier']}")

    print("3/4  XGBoost (walk-forward CV)...")
    m = run_xgboost(train, test)
    models.append(m)
    print(f"     CV AUC={m['cv_auc']}  Test AUC={m['auc']}  Accuracy={m['accuracy']}  Brier={m['brier']}")

    print("4/4  LSTM...")
    m = run_lstm(train, test)
    models.append(m)
    if "error" not in m:
        print(f"     AUC={m['auc']}  Accuracy={m['accuracy']}  Brier={m['brier']}")
    else:
        print(f"     {m['error']}")

    elapsed = time.time() - t0
    note = build_note(
        models, cb,
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

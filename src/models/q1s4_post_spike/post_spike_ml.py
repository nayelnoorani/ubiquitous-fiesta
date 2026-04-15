"""
Q1S4 — Post-Spike Spread Behaviour: ML Models

Target:  next-day spread change (spread_{t+1} - spread_t)
Split:   80/20 chronological
Models:
  1. Linear regression with interaction term — spread_change ~ lag + days_since_spike + lag×days_since_spike
  2. Random forest — captures non-linear regime effects without manual interaction terms

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH = Path(__file__).parent / "results.md"

BASE_FEATURES = [
    "spread_vs_net_lag_1d",
    "days_since_spike",
    "spread_vs_net_rolling_std_7d",
    "spread_vs_net_rolling_mean_7d",
]


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def prepare_data(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = wide[BASE_FEATURES + ["spread_vs_net"]].copy().dropna()

    # Target: next-day spread change
    df["target"] = df["spread_vs_net"].shift(-1) - df["spread_vs_net"]

    # Interaction term: lag × days_since_spike
    df["lag_x_dss"] = df["spread_vs_net_lag_1d"] * df["days_since_spike"]

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
# Model 1 — Linear regression with interaction term
# ----------------------------------------------------------------------

def run_linear_interaction(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    interaction_cols = BASE_FEATURES + ["lag_x_dss"]

    model = LinearRegression()
    model.fit(train[interaction_cols].values, train["target"].values)
    y_pred = model.predict(test[interaction_cols].values)

    result = evaluate("Linear regression (with lag×days_since_spike)", test["target"].values, y_pred)
    result["coefs"] = {col: round(float(c), 6)
                       for col, c in zip(interaction_cols, model.coef_)}
    result["intercept"] = round(float(model.intercept_), 6)
    return result


# ----------------------------------------------------------------------
# Model 2 — Random forest
# ----------------------------------------------------------------------

def run_random_forest(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    all_cols = BASE_FEATURES + ["lag_x_dss"]

    model = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )
    model.fit(train[all_cols].values, train["target"].values)
    y_pred = model.predict(test[all_cols].values)

    result = evaluate("Random forest", test["target"].values, y_pred)
    result["feature_importance"] = {
        col: round(float(imp), 4)
        for col, imp in zip(all_cols, model.feature_importances_)
    }
    return result


# ----------------------------------------------------------------------
# Interaction effect analysis
# ----------------------------------------------------------------------

def interaction_effect(train: pd.DataFrame) -> dict:
    """
    Quantify how the effect of spread_lag on spread_change varies by days_since_spike.
    Fit separate OLS slopes in spike window vs steady-state.
    """
    spike_mask  = train["days_since_spike"] <= 7
    steady_mask = ~spike_mask

    results = {}
    for label, mask in [("spike", spike_mask), ("steady", steady_mask)]:
        sub = train[mask]
        if len(sub) < 20:
            results[label] = float("nan")
            continue
        X = np.column_stack([np.ones(len(sub)), sub["spread_vs_net_lag_1d"].values])
        y = sub["target"].values
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        results[label] = round(float(coefs[1]), 4)  # slope on lag

    return results  # negative slope = mean reversion; more negative = faster


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(models: list[dict], interaction: dict,
               train_len: int, test_len: int,
               train_start: str, train_end: str,
               test_start: str, test_end: str, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "\n---\n",
        "# Q1S4 — Post-Spike Spread Behaviour: ML Models\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",
        "## Run Details\n",
        f"- Training: {train_start} → {train_end} ({train_len} days)",
        f"- Test:     {test_start} → {test_end} ({test_len} days)",
        f"- Target:   next-day spread change (spread_{{t+1}} − spread_t)\n",

        "## Model Performance\n",
        "| Model | MAE | RMSE | R² | Direction Acc |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        dacc = f"{m['dir_acc']:.1%}" if m["dir_acc"] is not None else "N/A"
        lines.append(f"| {m['model']} | {m['mae']} | {m['rmse']} | {m['r2']} | {dacc} |")

    best = min(models, key=lambda m: m["mae"])
    lines.append(f"\n**Best model by MAE:** {best['model']} (MAE={best['mae']})\n")

    lines += [
        "## Interaction Effect — Slope of Lag on Spread Change by Regime\n",
        "A more negative slope means stronger mean reversion in that regime.\n",
        "| Regime | Slope (spread_change ~ spread_lag) |",
        "|---|---|",
        f"| Spike window  | {interaction.get('spike', 'N/A')} |",
        f"| Steady-state  | {interaction.get('steady', 'N/A')} |",
    ]

    # Linear regression coefficients
    lr = next((m for m in models if "coefs" in m), None)
    if lr:
        lines += [
            "\n## Linear Regression Coefficients\n",
            "| Feature | Coefficient |",
            "|---|---|",
        ]
        for feat, coef in lr["coefs"].items():
            lines.append(f"| `{feat}` | {coef} |")
        lines.append(f"| intercept | {lr['intercept']} |")

    # RF feature importance
    rf = next((m for m in models if "feature_importance" in m), None)
    if rf:
        sorted_fi = sorted(rf["feature_importance"].items(), key=lambda x: -x[1])
        lines += [
            "\n## Random Forest Feature Importance\n",
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

    print("1/2  Linear regression with interaction term...")
    m = run_linear_interaction(train, test)
    models.append(m)
    print(f"     MAE={m['mae']}  R²={m['r2']}  DirAcc={m['dir_acc']}")

    print("2/2  Random forest...")
    m = run_random_forest(train, test)
    models.append(m)
    print(f"     MAE={m['mae']}  R²={m['r2']}  DirAcc={m['dir_acc']}")

    print("    Interaction effect analysis...")
    interaction = interaction_effect(train)
    print(f"     Spike slope: {interaction.get('spike')}  Steady slope: {interaction.get('steady')}")

    elapsed = time.time() - t0
    note = build_note(
        models, interaction,
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

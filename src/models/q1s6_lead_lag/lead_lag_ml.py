"""
Q1S6 — Lead/Lag Between Protocols: ML Models

Models:
  1. Linear regression — compound_net_change_t ~ aave_rate_change_{t-1} (and vice versa)
  2. VAR (Vector Autoregression) — jointly models both rate series

Split:   80/20 chronological
Metrics: MAE, RMSE, R²
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
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH = Path(__file__).parent / "results.md"
VAR_MAX_LAGS = 10


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def prepare_data(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = wide[["aave_rate_change_1d", "compound_net_change_1d"]].dropna().copy()
    split = int(len(df) * 0.8)
    return df, df.iloc[:split], df.iloc[split:]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    return {"model": name,
            "mae":  round(float(mae),  4),
            "rmse": round(float(rmse), 4),
            "r2":   round(float(r2),   4)}


# ----------------------------------------------------------------------
# Model 1 — Cross-lagged linear regressions
# ----------------------------------------------------------------------

def run_cross_lagged(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """
    Fit four regressions:
      a) compound_t ~ aave_{t-1}          (Aave leads Compound)
      b) aave_t     ~ compound_{t-1}      (Compound leads Aave)
      c) compound_t ~ aave_{t-1} + comp_{t-1}  (with autoregressive term)
      d) aave_t     ~ compound_{t-1} + aave_{t-1}
    """
    results = []

    specs = [
        ("Compound ~ Aave lag-1",
         ["aave_rate_change_1d"],          "compound_net_change_1d"),
        ("Aave ~ Compound lag-1",
         ["compound_net_change_1d"],       "aave_rate_change_1d"),
        ("Compound ~ Aave lag-1 + Comp lag-1",
         ["aave_rate_change_1d", "compound_net_change_1d"], "compound_net_change_1d"),
        ("Aave ~ Compound lag-1 + Aave lag-1",
         ["compound_net_change_1d", "aave_rate_change_1d"], "aave_rate_change_1d"),
    ]

    for label, x_cols, y_col in specs:
        # Build lag-1 features
        tr = train.copy()
        te = test.copy()
        X_tr = np.column_stack([tr[c].shift(1).fillna(0).values for c in x_cols])
        X_te = np.column_stack([te[c].shift(1).fillna(0).values for c in x_cols])
        y_tr = tr[y_col].values
        y_te = te[y_col].values

        model  = LinearRegression()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        res = evaluate(label, y_te, y_pred)
        res["coefs"] = {f"{c}_lag1": round(float(coef), 6)
                        for c, coef in zip(x_cols, model.coef_)}
        res["intercept"] = round(float(model.intercept_), 6)
        results.append(res)

    return results


# ----------------------------------------------------------------------
# Model 2 — VAR
# ----------------------------------------------------------------------

def run_var(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Fit VAR on training set, select lag order by AIC, then forecast
    one step ahead on the test set (rolling, using actuals as history).
    """
    model = VAR(train.values)
    selected = model.select_order(maxlags=VAR_MAX_LAGS)
    best_lag = selected.aic

    fitted = model.fit(best_lag)

    # One-step-ahead rolling forecast on test set
    history = list(train.values[-best_lag:])
    aave_preds, comp_preds = [], []

    for row in test.values:
        fc = fitted.forecast(np.array(history[-best_lag:]), steps=1)[0]
        aave_preds.append(fc[0])
        comp_preds.append(fc[1])
        history.append(row)

    aave_true = test["aave_rate_change_1d"].values
    comp_true = test["compound_net_change_1d"].values

    return {
        "lag_order":  best_lag,
        "aave":       evaluate(f"VAR({best_lag}) → Aave", aave_true, np.array(aave_preds)),
        "compound":   evaluate(f"VAR({best_lag}) → Compound", comp_true, np.array(comp_preds)),
        "coef_summary": {
            "aave_eq_aave_lag1":  round(float(fitted.coefs[0][0, 0]), 4),
            "aave_eq_comp_lag1":  round(float(fitted.coefs[0][0, 1]), 4),
            "comp_eq_aave_lag1":  round(float(fitted.coefs[0][1, 0]), 4),
            "comp_eq_comp_lag1":  round(float(fitted.coefs[0][1, 1]), 4),
        },
    }


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(lr_results: list[dict], var_res: dict,
               train_len: int, test_len: int,
               train_start: str, train_end: str,
               test_start: str, test_end: str, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "\n---\n",
        "# Q1S6 — Lead/Lag Between Protocols: ML Models\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",
        "## Run Details\n",
        f"- Training: {train_start} → {train_end} ({train_len} days)",
        f"- Test:     {test_start} → {test_end} ({test_len} days)\n",

        "## Cross-Lagged Linear Regressions\n",
        "| Model | MAE | RMSE | R² |",
        "|---|---|---|---|",
    ]
    for m in lr_results:
        lines.append(f"| {m['model']} | {m['mae']} | {m['rmse']} | {m['r2']} |")

    lines.append("\n### Coefficients\n")
    for m in lr_results:
        lines.append(f"**{m['model']}**")
        for feat, coef in m["coefs"].items():
            lines.append(f"- `{feat}`: {coef}")
        lines.append(f"- intercept: {m['intercept']}\n")

    lines += [
        f"## VAR({var_res['lag_order']}) Model\n",
        "| Series | MAE | RMSE | R² |",
        "|---|---|---|---|",
        f"| Aave    | {var_res['aave']['mae']} | {var_res['aave']['rmse']} | {var_res['aave']['r2']} |",
        f"| Compound | {var_res['compound']['mae']} | {var_res['compound']['rmse']} | {var_res['compound']['r2']} |",
        "\n### VAR Lag-1 Coefficients\n",
        "| Equation | aave_{t-1} | compound_{t-1} |",
        "|---|---|---|",
        f"| Aave_t    | {var_res['coef_summary']['aave_eq_aave_lag1']} | {var_res['coef_summary']['aave_eq_comp_lag1']} |",
        f"| Compound_t | {var_res['coef_summary']['comp_eq_aave_lag1']} | {var_res['coef_summary']['comp_eq_comp_lag1']} |",
        "",
    ]
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

    print("1/2  Cross-lagged linear regressions...")
    lr_results = run_cross_lagged(train, test)
    for m in lr_results:
        print(f"     {m['model']}: MAE={m['mae']}  R²={m['r2']}")

    print("2/2  VAR model...")
    var_res = run_var(train, test)
    print(f"     Lag order: {var_res['lag_order']}")
    print(f"     Aave:     MAE={var_res['aave']['mae']}  R²={var_res['aave']['r2']}")
    print(f"     Compound: MAE={var_res['compound']['mae']}  R²={var_res['compound']['r2']}")

    elapsed = time.time() - t0
    note = build_note(
        lr_results, var_res,
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

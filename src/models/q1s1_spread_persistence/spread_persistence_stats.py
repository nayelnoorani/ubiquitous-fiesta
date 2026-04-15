"""
Q1S1 — Spread Persistence: Statistical Analysis

Tests:
  1. ACF / PACF — characterise autocorrelation structure, identify AR lag order
  2. Ljung-Box — formal test of whether autocorrelation is significant
  3. Ornstein-Uhlenbeck fit — mean-reversion speed, half-life
     (mirrors Q1 stats; included here for Q1S1-specific framing)

Appends a section to results.md.
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, q_stat

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH = Path(__file__).parent / "results.md"
MAX_LAGS = 20


# ----------------------------------------------------------------------
# Test 1 — ACF / PACF
# ----------------------------------------------------------------------

def compute_acf_pacf(spread: pd.Series, nlags: int = MAX_LAGS) -> dict:
    acf_vals, acf_ci  = acf(spread.values,  nlags=nlags, alpha=0.05, fft=True)
    pacf_vals, pacf_ci = pacf(spread.values, nlags=nlags, alpha=0.05)

    # Confidence interval half-width at each lag (95%)
    acf_lower  = acf_ci[:, 0]  - acf_vals
    acf_upper  = acf_ci[:, 1]  - acf_vals
    pacf_lower = pacf_ci[:, 0] - pacf_vals
    pacf_upper = pacf_ci[:, 1] - pacf_vals

    # Significant lags: ACF value outside its 95% CI
    sig_acf_lags  = [i for i in range(1, nlags + 1)
                     if not (acf_ci[i, 0] <= 0 <= acf_ci[i, 1])]
    sig_pacf_lags = [i for i in range(1, nlags + 1)
                     if not (pacf_ci[i, 0] <= 0 <= pacf_ci[i, 1])]

    return {
        "acf":            [round(float(v), 4) for v in acf_vals],
        "pacf":           [round(float(v), 4) for v in pacf_vals],
        "acf_ci_lower":   [round(float(v), 4) for v in acf_lower],
        "acf_ci_upper":   [round(float(v), 4) for v in acf_upper],
        "sig_acf_lags":   sig_acf_lags,
        "sig_pacf_lags":  sig_pacf_lags,
        "suggested_ar_order": max(sig_pacf_lags) if sig_pacf_lags else 0,
    }


# ----------------------------------------------------------------------
# Test 2 — Ljung-Box
# ----------------------------------------------------------------------

def ljung_box(spread: pd.Series, lags: list[int] = None) -> dict:
    if lags is None:
        lags = [1, 5, 10, 20]

    acf_vals = acf(spread.values, nlags=max(lags), fft=True)
    results = {}
    for lag in lags:
        lb_stat, lb_p = q_stat(acf_vals[1:lag + 1], len(spread))
        results[f"lag_{lag}"] = {
            "stat":    round(float(lb_stat[-1]), 4),
            "p_value": round(float(lb_p[-1]),   4),
        }
    return results


# ----------------------------------------------------------------------
# Test 3 — Ornstein-Uhlenbeck fit
# ----------------------------------------------------------------------

def fit_ou(spread: pd.Series) -> dict:
    spread_lag  = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    idx         = spread_lag.index.intersection(spread_diff.index)

    X      = sm.add_constant(spread_lag.loc[idx])
    result = sm.OLS(spread_diff.loc[idx], X).fit()

    alpha = result.params.iloc[0]
    beta  = result.params.iloc[1]
    theta = -beta
    mu    = alpha / theta if theta != 0 else float("nan")
    sigma = result.resid.std()
    hl    = np.log(2) / theta if theta > 0 else float("nan")

    return {
        "theta":     round(float(theta), 6),
        "mu":        round(float(mu),    4),
        "sigma":     round(float(sigma), 4),
        "half_life": round(float(hl),    2),
        "r2":        round(float(result.rsquared), 4),
        "p_beta":    round(float(result.pvalues.iloc[1]), 4),
    }


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(results: dict, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    r   = results
    lb  = r["lb"]

    lines = [
        "# Q1S1 — Spread Persistence: Statistical Analysis\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",

        "## 1. ACF / PACF\n",
        f"Significant ACF lags (95%): {r['acf']['sig_acf_lags']}",
        f"Significant PACF lags (95%): {r['acf']['sig_pacf_lags']}",
        f"Suggested AR order (max significant PACF lag): **{r['acf']['suggested_ar_order']}**\n",
        "| Lag | ACF | PACF |",
        "|---|---|---|",
    ]
    for i in range(1, min(11, len(r["acf"]["acf"]))):
        lines.append(f"| {i} | {r['acf']['acf'][i]} | {r['acf']['pacf'][i]} |")

    lines += [
        "\n## 2. Ljung-Box Test\n",
        "| Lags | Statistic | p-value | Autocorrelation? |",
        "|---|---|---|---|",
    ]
    for lag_key, vals in lb.items():
        sig = "Yes" if vals["p_value"] < 0.05 else "No"
        lag_n = lag_key.replace("lag_", "")
        lines.append(f"| {lag_n} | {vals['stat']} | {vals['p_value']} | {sig} |")

    lines += [
        "\n## 3. Ornstein-Uhlenbeck Fit\n",
        "| Parameter | Value | Interpretation |",
        "|---|---|---|",
        f"| θ (mean-reversion speed) | {r['ou']['theta']} | higher = faster reversion |",
        f"| μ (long-run mean spread) | {r['ou']['mu']}% | spread reverts toward this level |",
        f"| σ (volatility) | {r['ou']['sigma']}% | daily noise in spread process |",
        f"| Half-life | {r['ou']['half_life']} days | time for shock to decay by 50% |",
        f"| R² | {r['ou']['r2']} | |",
        f"| p-value (β) | {r['ou']['p_beta']} | significance of mean-reversion term |",
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
    spread = wide["spread_vs_net"].dropna()
    print(f"  {len(spread):,} days | {spread.index.min().date()} → {spread.index.max().date()}\n")

    results = {}

    print("1/3  ACF / PACF...")
    results["acf"] = compute_acf_pacf(spread)
    print(f"     Significant ACF lags: {results['acf']['sig_acf_lags']}")
    print(f"     Significant PACF lags: {results['acf']['sig_pacf_lags']}")
    print(f"     Suggested AR order: {results['acf']['suggested_ar_order']}")

    print("2/3  Ljung-Box test...")
    results["lb"] = ljung_box(spread)
    for lag_key, vals in results["lb"].items():
        sig = "significant" if vals["p_value"] < 0.05 else "not significant"
        print(f"     {lag_key}: stat={vals['stat']}  p={vals['p_value']}  → {sig}")

    print("3/3  Ornstein-Uhlenbeck fit...")
    results["ou"] = fit_ou(spread)
    print(f"     θ={results['ou']['theta']} | half-life={results['ou']['half_life']} days | μ={results['ou']['mu']}%")

    elapsed = time.time() - t0
    note    = build_note(results, elapsed)

    with open(RESULTS_PATH, "w") as f:
        f.write(note)

    print(f"\nWritten to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

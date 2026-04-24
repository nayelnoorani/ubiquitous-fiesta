"""
Q1S7 — Spread Direction Prediction: Statistical Analysis

Pre-modelling analysis:
  1. Class distribution — base rate of widening vs narrowing
  2. Point-biserial correlations — linear association of each feature with direction
  3. Conditional widening rates by feature quartile — reveals non-linear relationships
  4. Autocorrelation of the direction signal — is tomorrow's direction predictable from today's?

Writes results.md (overwrites).
"""

import sys
import time
import warnings
import pandas as pd
from datetime import datetime
from pathlib import Path

from scipy.stats import pointbiserialr
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH = Path(__file__).parent / "results.md"

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

def prepare(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide[FEATURE_COLS + ["spread_vs_net"]].copy().dropna()
    next_spread = df["spread_vs_net"].shift(-1)
    df["target"] = (next_spread > df["spread_vs_net"]).astype(int)
    return df.dropna()


# ----------------------------------------------------------------------
# Test 1 — Class distribution
# ----------------------------------------------------------------------

def class_distribution(df: pd.DataFrame) -> dict:
    n_widen  = int(df["target"].sum())
    n_narrow = int((df["target"] == 0).sum())
    n_total  = len(df)
    return {
        "n_total":   n_total,
        "n_widen":   n_widen,
        "n_narrow":  n_narrow,
        "pct_widen": round(float(n_widen / n_total), 4),
        "pct_narrow": round(float(n_narrow / n_total), 4),
    }


# ----------------------------------------------------------------------
# Test 2 — Point-biserial correlations
# ----------------------------------------------------------------------

def feature_correlations(df: pd.DataFrame) -> list[dict]:
    results = []
    for col in FEATURE_COLS:
        r, p = pointbiserialr(df["target"].values, df[col].values)
        results.append({
            "feature":     col,
            "r":           round(float(r), 4),
            "p":           round(float(p), 4),
            "significant": p < 0.05,
        })
    return sorted(results, key=lambda x: -abs(x["r"]))


# ----------------------------------------------------------------------
# Test 3 — Conditional widening rates by quartile
# ----------------------------------------------------------------------

def conditional_rates(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in FEATURE_COLS:
        try:
            quartiles = pd.qcut(df[col], q=4, labels=["Q1", "Q2", "Q3", "Q4"],
                                duplicates="drop")
        except Exception:
            continue
        rates = df.groupby(quartiles, observed=True)["target"].mean()
        rows.append({
            "feature": col,
            "q1_rate": round(float(rates.get("Q1", float("nan"))), 4),
            "q2_rate": round(float(rates.get("Q2", float("nan"))), 4),
            "q3_rate": round(float(rates.get("Q3", float("nan"))), 4),
            "q4_rate": round(float(rates.get("Q4", float("nan"))), 4),
            "range":   round(float(rates.max() - rates.min()), 4),
        })
    return sorted(rows, key=lambda x: -x["range"])


# ----------------------------------------------------------------------
# Test 4 — Autocorrelation of direction signal
# ----------------------------------------------------------------------

def direction_acf(df: pd.DataFrame, nlags: int = 10) -> dict:
    acf_vals, acf_ci = acf(df["target"].values, nlags=nlags, alpha=0.05, fft=True)
    sig_lags = [i for i in range(1, nlags + 1)
                if not (acf_ci[i, 0] <= 0 <= acf_ci[i, 1])]
    return {
        "acf":      [round(float(v), 4) for v in acf_vals],
        "sig_lags": sig_lags,
    }


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(dist: dict, corr: list, cond: list,
               dacf: dict, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Q1S7 — Spread Direction Prediction: Statistical Analysis\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",
        "Target: spread widens tomorrow (spread_{{t+1}} > spread_t) = 1\n",

        "## 1. Class Distribution\n",
        "| Class | N | % |",
        "|---|---|---|",
        f"| Widens  (1) | {dist['n_widen']}  | {dist['pct_widen']:.1%} |",
        f"| Narrows (0) | {dist['n_narrow']} | {dist['pct_narrow']:.1%} |",
        f"| Total       | {dist['n_total']}  | 100% |",

        "\n## 2. Point-Biserial Correlations with Direction Label\n",
        "| Feature | r | p-value | Significant? |",
        "|---|---|---|---|",
    ]
    for c in corr:
        lines.append(f"| `{c['feature']}` | {c['r']} | {c['p']} | {'Yes' if c['significant'] else 'No'} |")

    lines += [
        "\n## 3. Conditional Widening Rate by Feature Quartile\n",
        "Range = Q4 rate − Q1 rate. Higher range = stronger non-linear association.\n",
        "| Feature | Q1 rate | Q2 rate | Q3 rate | Q4 rate | Range |",
        "|---|---|---|---|---|---|",
    ]
    for c in cond:
        lines.append(
            f"| `{c['feature']}` | {c['q1_rate']:.1%} | {c['q2_rate']:.1%} | "
            f"{c['q3_rate']:.1%} | {c['q4_rate']:.1%} | {c['range']:.3f} |"
        )

    lines += [
        "\n## 4. Autocorrelation of Direction Signal\n",
        f"Significant ACF lags (95%): {dacf['sig_lags']}\n",
        "| Lag | ACF |",
        "|---|---|",
    ]
    for i, v in enumerate(dacf["acf"]):
        lines.append(f"| {i} | {v} |")
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
    data = prepare(wide)
    print(f"  {len(data):,} days\n")

    print("1/4  Class distribution...")
    dist = class_distribution(data)
    print(f"     Widens: {dist['pct_widen']:.1%}  Narrows: {dist['pct_narrow']:.1%}")

    print("2/4  Point-biserial correlations...")
    corr = feature_correlations(data)
    for c in corr[:3]:
        print(f"     {c['feature']}: r={c['r']}  p={c['p']}")

    print("3/4  Conditional widening rates by quartile...")
    cond = conditional_rates(data)
    for c in cond[:3]:
        print(f"     {c['feature']}: range={c['range']:.3f}")

    print("4/4  Direction signal ACF...")
    dacf = direction_acf(data)
    print(f"     Significant lags: {dacf['sig_lags']}")

    elapsed = time.time() - t0
    note    = build_note(dist, corr, cond, dacf, elapsed)

    with open(RESULTS_PATH, "w") as f:
        f.write(note)

    print(f"\nWritten to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

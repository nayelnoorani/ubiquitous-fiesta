"""
Q1S2 — Volatility as a Leading Indicator: Statistical Analysis

Tests:
  1. Regime classification — high-vol (rolling_std > 75th pct) vs low-vol
  2. Regime characterisation — spread level, absolute spread, episode count by regime
  3. Mann-Whitney U test — are subsequent spread changes different in high vs low vol?
  4. Conditional OU fit — estimate mean-reversion half-life separately in each regime

Writes results.md (overwrites).
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import statsmodels.api as sm
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH  = Path(__file__).parent / "results.md"
VOL_THRESHOLD = 0.75   # percentile defining high-vol regime
HORIZON       = 3      # days ahead for "subsequent narrowing" test


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def fit_ou_regime(spread: pd.Series) -> dict:
    """OU fit on a sub-series."""
    if len(spread) < 30:
        return {"theta": float("nan"), "half_life": float("nan"), "mu": float("nan")}
    lag  = spread.shift(1).dropna()
    diff = spread.diff().dropna()
    idx  = lag.index.intersection(diff.index)
    X    = sm.add_constant(lag.loc[idx])
    res  = sm.OLS(diff.loc[idx], X).fit()
    beta  = res.params.iloc[1]
    alpha = res.params.iloc[0]
    theta = -beta
    mu    = alpha / theta if theta != 0 else float("nan")
    hl    = np.log(2) / theta if theta > 0 else float("nan")
    return {"theta": round(float(theta), 4),
            "mu":    round(float(mu),    4),
            "half_life": round(float(hl), 2)}


# ----------------------------------------------------------------------
# Test 1 + 2 — Regime classification and characterisation
# ----------------------------------------------------------------------

def regime_analysis(wide: pd.DataFrame) -> dict:
    spread  = wide["spread_vs_net"].dropna()
    vol     = wide["spread_vs_net_rolling_std_7d"].dropna()
    aligned = spread.align(vol, join="inner")
    spread, vol = aligned

    threshold = vol.quantile(VOL_THRESHOLD)
    high_vol  = vol >= threshold
    low_vol   = ~high_vol

    def describe_regime(mask: pd.Series, label: str) -> dict:
        s = spread[mask]
        return {
            "label":       label,
            "n_days":      int(mask.sum()),
            "pct_days":    round(float(mask.mean()), 4),
            "mean_spread": round(float(s.mean()), 4),
            "mean_abs":    round(float(s.abs().mean()), 4),
            "std_spread":  round(float(s.std()), 4),
            "median_abs":  round(float(s.abs().median()), 4),
        }

    return {
        "threshold":  round(float(threshold), 4),
        "high":       describe_regime(high_vol, "High-vol"),
        "low":        describe_regime(low_vol,  "Low-vol"),
        "spread":     spread,
        "vol":        vol,
        "high_mask":  high_vol,
    }


# ----------------------------------------------------------------------
# Test 3 — Mann-Whitney U: subsequent spread changes
# ----------------------------------------------------------------------

def mann_whitney_test(spread: pd.Series, high_mask: pd.Series,
                      horizon: int = HORIZON) -> dict:
    """
    Compare |spread_{t+horizon}| - |spread_t| in high-vol vs low-vol regimes.
    Negative values = spread narrowed.
    """
    abs_change = spread.abs().shift(-horizon) - spread.abs()
    abs_change = abs_change.dropna()

    aligned_mask = high_mask.reindex(abs_change.index).fillna(False)
    high_changes = abs_change[aligned_mask].values
    low_changes  = abs_change[~aligned_mask].values

    stat, p = mannwhitneyu(high_changes, low_changes, alternative="less")

    return {
        "horizon":          horizon,
        "high_vol_mean_change": round(float(high_changes.mean()), 4),
        "low_vol_mean_change":  round(float(low_changes.mean()),  4),
        "stat":             round(float(stat), 2),
        "p_value":          round(float(p),    4),
        "significant":      p < 0.05,
        "n_high":           len(high_changes),
        "n_low":            len(low_changes),
    }


# ----------------------------------------------------------------------
# Test 4 — OU fit by regime
# ----------------------------------------------------------------------

def ou_by_regime(spread: pd.Series, high_mask: pd.Series) -> dict:
    return {
        "high": fit_ou_regime(spread[high_mask]),
        "low":  fit_ou_regime(spread[~high_mask]),
        "full": fit_ou_regime(spread),
    }


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(reg: dict, mw: dict, ou: dict, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    h, l = reg["high"], reg["low"]

    lines = [
        "# Q1S2 — Volatility as a Leading Indicator: Statistical Analysis\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",

        "## 1. Volatility Regime Definition\n",
        f"High-vol threshold (75th percentile of `spread_vs_net_rolling_std_7d`): **{reg['threshold']}%**\n",
        "| Metric | High-vol | Low-vol |",
        "|---|---|---|",
        f"| Days in regime | {h['n_days']} | {l['n_days']} |",
        f"| % of sample | {h['pct_days']:.1%} | {l['pct_days']:.1%} |",
        f"| Mean spread | {h['mean_spread']}% | {l['mean_spread']}% |",
        f"| Mean |spread| | {h['mean_abs']}% | {l['mean_abs']}% |",
        f"| Std spread | {h['std_spread']}% | {l['std_spread']}% |",
        f"| Median |spread| | {h['median_abs']}% | {l['median_abs']}% |",

        "\n## 2. Mann-Whitney U — Subsequent Spread Change\n",
        f"Test: does high-vol regime produce *more narrowing* (lower |spread change|) "
        f"over the next {mw['horizon']} days?\n",
        "| Metric | Value |",
        "|---|---|",
        f"| High-vol mean change in |spread| | {mw['high_vol_mean_change']}% |",
        f"| Low-vol mean change in |spread|  | {mw['low_vol_mean_change']}% |",
        f"| Mann-Whitney U statistic | {mw['stat']} |",
        f"| p-value (one-sided: high < low) | {mw['p_value']} |",
        f"| Significant (p < 0.05) | {'Yes' if mw['significant'] else 'No'} |",
        f"| N (high-vol / low-vol) | {mw['n_high']} / {mw['n_low']} |",

        "\n## 3. Ornstein-Uhlenbeck Fit by Regime\n",
        "| Regime | θ | Half-life (days) | μ |",
        "|---|---|---|---|",
        f"| Full sample | {ou['full']['theta']} | {ou['full']['half_life']} | {ou['full']['mu']}% |",
        f"| High-vol    | {ou['high']['theta']} | {ou['high']['half_life']} | {ou['high']['mu']}% |",
        f"| Low-vol     | {ou['low']['theta']}  | {ou['low']['half_life']}  | {ou['low']['mu']}% |",
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
    print(f"  {len(wide):,} days\n")

    print("1/3  Regime classification and characterisation...")
    reg = regime_analysis(wide)
    print(f"     High-vol threshold: {reg['threshold']}%")
    print(f"     High-vol days: {reg['high']['n_days']} ({reg['high']['pct_days']:.1%})")
    print(f"     High-vol mean |spread|: {reg['high']['mean_abs']}%  Low-vol: {reg['low']['mean_abs']}%")

    print("2/3  Mann-Whitney U test...")
    mw = mann_whitney_test(reg["spread"], reg["high_mask"])
    print(f"     High-vol mean change: {mw['high_vol_mean_change']}%  Low-vol: {mw['low_vol_mean_change']}%")
    print(f"     p={mw['p_value']}  significant={mw['significant']}")

    print("3/3  OU fit by regime...")
    ou = ou_by_regime(reg["spread"], reg["high_mask"])
    print(f"     High-vol half-life: {ou['high']['half_life']} days")
    print(f"     Low-vol  half-life: {ou['low']['half_life']} days")

    elapsed = time.time() - t0
    note    = build_note(reg, mw, ou, elapsed)

    with open(RESULTS_PATH, "w") as f:
        f.write(note)

    print(f"\nWritten to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

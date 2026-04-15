"""
Q1S3 — TVL Shocks and Rate Divergence: Statistical Analysis

Tests:
  1. Granger causality — does aave_tvl_change_pct_1d precede spread changes? (both directions)
  2. Contemporaneous correlation — TVL changes vs same-day spread
  3. Conditional analysis — large TVL shock days vs normal: next-day spread change distribution

Writes results.md (overwrites).
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from scipy.stats import pearsonr, mannwhitneyu
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH   = Path(__file__).parent / "results.md"
GRANGER_LAGS   = [1, 2, 3, 5]
SHOCK_THRESHOLD = 0.75   # percentile defining a large TVL shock


# ----------------------------------------------------------------------
# Test 1 — Granger causality
# ----------------------------------------------------------------------

def granger_test(x: pd.Series, y: pd.Series, label: str, maxlag: int = 5) -> dict:
    """
    Test whether lags of x Granger-cause y.
    Returns min p-value across lags and per-lag F-test p-values.
    """
    df_aligned = pd.concat([y, x], axis=1).dropna()
    df_aligned.columns = ["y", "x"]

    results = grangercausalitytests(df_aligned[["y", "x"]], maxlag=maxlag, verbose=False)

    per_lag = {}
    for lag in range(1, maxlag + 1):
        f_stat = results[lag][0]["ssr_ftest"][0]
        p_val  = results[lag][0]["ssr_ftest"][1]
        per_lag[lag] = {"f_stat": round(float(f_stat), 4),
                        "p_value": round(float(p_val),  4)}

    min_p = min(v["p_value"] for v in per_lag.values())
    best_lag = min(per_lag, key=lambda k: per_lag[k]["p_value"])

    return {
        "label":    label,
        "min_p":    round(float(min_p), 4),
        "best_lag": best_lag,
        "per_lag":  per_lag,
        "significant": min_p < 0.05,
    }


# ----------------------------------------------------------------------
# Test 2 — Contemporaneous correlation
# ----------------------------------------------------------------------

def contemporaneous_corr(wide: pd.DataFrame) -> dict:
    pairs = [
        ("aave_tvl_change_pct_1d", "spread_vs_net",         "Aave TVL change vs spread (same day)"),
        ("aave_tvl_change_pct_1d", "aave_rate_change_1d",   "Aave TVL change vs Aave rate change"),
        ("tvl_ratio",              "spread_vs_net",         "TVL ratio vs spread"),
    ]
    results = []
    for col_x, col_y, label in pairs:
        aligned = wide[[col_x, col_y]].dropna()
        r, p = pearsonr(aligned[col_x].values, aligned[col_y].values)
        results.append({
            "label": label,
            "r":     round(float(r), 4),
            "p":     round(float(p), 4),
            "significant": p < 0.05,
        })
    return results


# ----------------------------------------------------------------------
# Test 3 — Conditional analysis: shock days vs normal
# ----------------------------------------------------------------------

def shock_conditional(wide: pd.DataFrame) -> dict:
    tvl   = wide["aave_tvl_change_pct_1d"].dropna()
    spread = wide["spread_vs_net"].dropna()
    aligned = pd.concat([tvl, spread], axis=1).dropna()
    aligned.columns = ["tvl_change", "spread"]

    # Next-day spread change
    aligned["next_spread_change"] = aligned["spread"].shift(-1) - aligned["spread"]
    aligned = aligned.dropna()

    threshold = aligned["tvl_change"].abs().quantile(SHOCK_THRESHOLD)
    shock_mask = aligned["tvl_change"].abs() >= threshold

    shock_changes  = aligned.loc[shock_mask,  "next_spread_change"].values
    normal_changes = aligned.loc[~shock_mask, "next_spread_change"].values

    stat, p = mannwhitneyu(np.abs(shock_changes), np.abs(normal_changes),
                           alternative="greater")

    # Split shocks into inflow vs outflow
    inflow_mask  = shock_mask & (aligned["tvl_change"] > 0)
    outflow_mask = shock_mask & (aligned["tvl_change"] < 0)

    return {
        "threshold":              round(float(threshold), 4),
        "n_shock":                int(shock_mask.sum()),
        "n_normal":               int((~shock_mask).sum()),
        "shock_mean_abs_change":  round(float(np.abs(shock_changes).mean()), 4),
        "normal_mean_abs_change": round(float(np.abs(normal_changes).mean()), 4),
        "shock_mean_change":      round(float(shock_changes.mean()), 4),
        "normal_mean_change":     round(float(normal_changes.mean()), 4),
        "mw_stat":                round(float(stat), 2),
        "mw_p":                   round(float(p), 4),
        "significant":            p < 0.05,
        "inflow_mean_change":     round(float(aligned.loc[inflow_mask,  "next_spread_change"].mean()), 4),
        "outflow_mean_change":    round(float(aligned.loc[outflow_mask, "next_spread_change"].mean()), 4),
        "n_inflow":               int(inflow_mask.sum()),
        "n_outflow":              int(outflow_mask.sum()),
    }


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(gc_fwd: dict, gc_rev: dict, gc_comp: dict,
               corr: list, shock: dict, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Q1S3 — TVL Shocks and Rate Divergence: Statistical Analysis\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",

        "## 1. Granger Causality\n",
        "### Aave TVL change → spread (does TVL precede spread?)\n",
        f"Minimum p-value across lags: **{gc_fwd['min_p']}** (lag {gc_fwd['best_lag']}) "
        f"— {'Significant ✓' if gc_fwd['significant'] else 'Not significant'}\n",
        "| Lag | F-stat | p-value |",
        "|---|---|---|",
    ]
    for lag, vals in gc_fwd["per_lag"].items():
        lines.append(f"| {lag} | {vals['f_stat']} | {vals['p_value']} |")

    lines += [
        "\n### Spread → Aave TVL change (reverse direction)\n",
        f"Minimum p-value: **{gc_rev['min_p']}** (lag {gc_rev['best_lag']}) "
        f"— {'Significant ✓' if gc_rev['significant'] else 'Not significant'}\n",
        "| Lag | F-stat | p-value |",
        "|---|---|---|",
    ]
    for lag, vals in gc_rev["per_lag"].items():
        lines.append(f"| {lag} | {vals['f_stat']} | {vals['p_value']} |")

    lines += [
        "\n### Compound net change → spread\n",
        f"Minimum p-value: **{gc_comp['min_p']}** (lag {gc_comp['best_lag']}) "
        f"— {'Significant ✓' if gc_comp['significant'] else 'Not significant'}\n",
        "| Lag | F-stat | p-value |",
        "|---|---|---|",
    ]
    for lag, vals in gc_comp["per_lag"].items():
        lines.append(f"| {lag} | {vals['f_stat']} | {vals['p_value']} |")

    lines += [
        "\n## 2. Contemporaneous Correlation\n",
        "| Pair | Pearson r | p-value | Significant? |",
        "|---|---|---|---|",
    ]
    for c in corr:
        lines.append(f"| {c['label']} | {c['r']} | {c['p']} | {'Yes' if c['significant'] else 'No'} |")

    lines += [
        "\n## 3. Conditional Analysis — Large TVL Shocks vs Normal Days\n",
        f"Shock threshold (75th pct of |aave_tvl_change_pct_1d|): **{shock['threshold']}%**\n",
        "| Metric | Shock days | Normal days |",
        "|---|---|---|",
        f"| N days | {shock['n_shock']} | {shock['n_normal']} |",
        f"| Mean next-day |spread change| | {shock['shock_mean_abs_change']}% | {shock['normal_mean_abs_change']}% |",
        f"| Mean next-day spread change | {shock['shock_mean_change']}% | {shock['normal_mean_change']}% |",
        f"\nMann-Whitney U (shock |change| > normal): stat={shock['mw_stat']}  p={shock['mw_p']}  "
        f"{'Significant ✓' if shock['significant'] else 'Not significant'}\n",
        "### Inflow vs Outflow Shocks\n",
        "| Direction | N | Mean next-day spread change |",
        "|---|---|---|",
        f"| Inflow  (TVL increase) | {shock['n_inflow']}  | {shock['inflow_mean_change']}% |",
        f"| Outflow (TVL decrease) | {shock['n_outflow']} | {shock['outflow_mean_change']}% |",
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
    spread  = wide["spread_vs_net"].dropna()
    tvl_chg = wide["aave_tvl_change_pct_1d"].dropna()
    comp_chg = wide["compound_net_change_1d"].dropna()
    print(f"  {len(spread):,} days\n")

    print("1/3  Granger causality tests...")
    gc_fwd  = granger_test(tvl_chg, spread,  "Aave TVL → spread")
    gc_rev  = granger_test(spread,  tvl_chg, "spread → Aave TVL")
    gc_comp = granger_test(comp_chg, spread, "Compound net change → spread")
    print(f"     Aave TVL → spread:  min_p={gc_fwd['min_p']}  sig={gc_fwd['significant']}")
    print(f"     spread → Aave TVL:  min_p={gc_rev['min_p']}  sig={gc_rev['significant']}")
    print(f"     Compound → spread:  min_p={gc_comp['min_p']}  sig={gc_comp['significant']}")

    print("2/3  Contemporaneous correlations...")
    corr = contemporaneous_corr(wide)
    for c in corr:
        print(f"     {c['label']}: r={c['r']}  p={c['p']}")

    print("3/3  Conditional analysis (shock vs normal days)...")
    shock = shock_conditional(wide)
    print(f"     Shock threshold: {shock['threshold']}%")
    print(f"     Shock mean |change|: {shock['shock_mean_abs_change']}%  Normal: {shock['normal_mean_abs_change']}%")
    print(f"     Mann-Whitney p={shock['mw_p']}  significant={shock['significant']}")

    elapsed = time.time() - t0
    note    = build_note(gc_fwd, gc_rev, gc_comp, corr, shock, elapsed)

    with open(RESULTS_PATH, "w") as f:
        f.write(note)

    print(f"\nWritten to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

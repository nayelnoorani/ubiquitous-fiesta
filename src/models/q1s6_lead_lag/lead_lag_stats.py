"""
Q1S6 — Lead/Lag Between Protocols: Statistical Analysis

Tests:
  1. Cross-correlation function (CCF) — peak lag between Aave and Compound rate changes
  2. Granger causality — both directions at lags 1–5
  3. Rolling Granger — does the lead/lag relationship change over time?

Note: CCF at lags 0–4 was computed in Q1 stats. This analysis extends to 10 lags
and uses rate *changes* (first differences) rather than levels.

Writes results.md (overwrites).
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from statsmodels.tsa.stattools import ccf, grangercausalitytests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH  = Path(__file__).parent / "results.md"
MAX_LAGS      = 10
ROLLING_WIN   = 180   # days for rolling Granger


# ----------------------------------------------------------------------
# Test 1 — CCF
# ----------------------------------------------------------------------

def compute_ccf(aave_chg: pd.Series, comp_chg: pd.Series, nlags: int = MAX_LAGS) -> dict:
    idx = aave_chg.index.intersection(comp_chg.index)
    a   = aave_chg.loc[idx].values
    c   = comp_chg.loc[idx].values

    # Aave leads Compound: positive lag = Aave moves first
    fwd = ccf(a, c, nlags=nlags, alpha=None)
    # Compound leads Aave
    rev = ccf(c, a, nlags=nlags, alpha=None)

    fwd_vals = {i: round(float(v), 4) for i, v in enumerate(fwd)}
    rev_vals = {i: round(float(v), 4) for i, v in enumerate(rev)}

    peak_fwd = int(np.argmax(np.abs(fwd[1:])) + 1)
    peak_rev = int(np.argmax(np.abs(rev[1:])) + 1)

    return {
        "aave_leads": fwd_vals,
        "comp_leads": rev_vals,
        "peak_aave_lag": peak_fwd,
        "peak_comp_lag": peak_rev,
        "contemp": round(float(fwd[0]), 4),
    }


# ----------------------------------------------------------------------
# Test 2 — Granger causality
# ----------------------------------------------------------------------

def granger_both_directions(aave_chg: pd.Series, comp_chg: pd.Series,
                             maxlag: int = 5) -> dict:
    idx = aave_chg.index.intersection(comp_chg.index)
    df  = pd.DataFrame({"aave": aave_chg.loc[idx].values,
                        "comp": comp_chg.loc[idx].values})

    def run_gc(df_pair, label):
        res     = grangercausalitytests(df_pair, maxlag=maxlag, verbose=False)
        per_lag = {}
        for lag in range(1, maxlag + 1):
            f_stat = res[lag][0]["ssr_ftest"][0]
            p_val  = res[lag][0]["ssr_ftest"][1]
            per_lag[lag] = {"f": round(float(f_stat), 4),
                            "p": round(float(p_val),  4)}
        min_p    = min(v["p"] for v in per_lag.values())
        best_lag = min(per_lag, key=lambda k: per_lag[k]["p"])
        return {"label": label, "min_p": round(float(min_p), 4),
                "best_lag": best_lag, "per_lag": per_lag,
                "significant": min_p < 0.05}

    # Aave → Compound: does aave_chg help predict comp_chg?
    gc_fwd = run_gc(df[["comp", "aave"]], "Aave → Compound")
    # Compound → Aave
    gc_rev = run_gc(df[["aave", "comp"]], "Compound → Aave")

    return {"fwd": gc_fwd, "rev": gc_rev}


# ----------------------------------------------------------------------
# Test 3 — Rolling Granger (lag 1 only, both directions)
# ----------------------------------------------------------------------

def rolling_granger(aave_chg: pd.Series, comp_chg: pd.Series,
                    window: int = ROLLING_WIN) -> dict:
    idx = aave_chg.index.intersection(comp_chg.index)
    a   = aave_chg.loc[idx]
    c   = comp_chg.loc[idx]

    fwd_p, rev_p, dates = [], [], []

    for end in range(window, len(idx)):
        sl_a = a.iloc[end - window: end].values
        sl_c = c.iloc[end - window: end].values
        df_s = pd.DataFrame({"y": sl_c, "x": sl_a})

        try:
            res_fwd = grangercausalitytests(df_s[["y", "x"]], maxlag=1, verbose=False)
            p_fwd   = res_fwd[1][0]["ssr_ftest"][1]
        except Exception:
            p_fwd = float("nan")

        try:
            res_rev = grangercausalitytests(df_s[["x", "y"]], maxlag=1, verbose=False)
            p_rev   = res_rev[1][0]["ssr_ftest"][1]
        except Exception:
            p_rev = float("nan")

        fwd_p.append(float(p_fwd))
        rev_p.append(float(p_rev))
        dates.append(idx[end])

    fwd_arr = np.array(fwd_p)
    rev_arr = np.array(rev_p)

    return {
        "window":            window,
        "n_windows":         len(dates),
        "fwd_pct_sig":       round(float(np.nanmean(fwd_arr < 0.05)), 4),
        "rev_pct_sig":       round(float(np.nanmean(rev_arr < 0.05)), 4),
        "fwd_median_p":      round(float(np.nanmedian(fwd_arr)), 4),
        "rev_median_p":      round(float(np.nanmedian(rev_arr)), 4),
    }


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(ccf_res: dict, gc: dict, roll: dict, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    gf, gr = gc["fwd"], gc["rev"]

    lines = [
        "# Q1S6 — Lead/Lag Between Protocols: Statistical Analysis\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",

        "## 1. Cross-Correlation Function (Rate Changes)\n",
        f"Contemporaneous correlation: {ccf_res['contemp']}\n",
        f"**Aave leads Compound** — peak at lag {ccf_res['peak_aave_lag']}:\n",
        "| Lag | CCF |",
        "|---|---|",
    ]
    for i in range(MAX_LAGS + 1):
        lines.append(f"| {i} | {ccf_res['aave_leads'].get(i, '')} |")

    lines += [
        f"\n**Compound leads Aave** — peak at lag {ccf_res['peak_comp_lag']}:\n",
        "| Lag | CCF |",
        "|---|---|",
    ]
    for i in range(MAX_LAGS + 1):
        lines.append(f"| {i} | {ccf_res['comp_leads'].get(i, '')} |")

    lines += [
        "\n## 2. Granger Causality\n",
        f"### Aave → Compound\n",
        f"Min p: **{gf['min_p']}** (lag {gf['best_lag']}) — "
        f"{'Significant ✓' if gf['significant'] else 'Not significant'}\n",
        "| Lag | F-stat | p-value |",
        "|---|---|---|",
    ]
    for lag, v in gf["per_lag"].items():
        lines.append(f"| {lag} | {v['f']} | {v['p']} |")

    lines += [
        f"\n### Compound → Aave\n",
        f"Min p: **{gr['min_p']}** (lag {gr['best_lag']}) — "
        f"{'Significant ✓' if gr['significant'] else 'Not significant'}\n",
        "| Lag | F-stat | p-value |",
        "|---|---|---|",
    ]
    for lag, v in gr["per_lag"].items():
        lines.append(f"| {lag} | {v['f']} | {v['p']} |")

    lines += [
        f"\n## 3. Rolling Granger Causality ({roll['window']}-day window, lag=1)\n",
        "| Metric | Aave → Compound | Compound → Aave |",
        "|---|---|---|",
        f"| % windows significant (p < 0.05) | {roll['fwd_pct_sig']:.1%} | {roll['rev_pct_sig']:.1%} |",
        f"| Median p-value | {roll['fwd_median_p']} | {roll['rev_median_p']} |",
        f"| N rolling windows | {roll['n_windows']} | {roll['n_windows']} |",
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
    aave_chg = wide["aave_rate_change_1d"].dropna()
    comp_chg = wide["compound_net_change_1d"].dropna()
    print(f"  {len(aave_chg):,} days of rate changes\n")

    print("1/3  Cross-correlation function...")
    ccf_res = compute_ccf(aave_chg, comp_chg)
    print(f"     Contemp corr: {ccf_res['contemp']}")
    print(f"     Aave leads peak lag: {ccf_res['peak_aave_lag']}  "
          f"Compound leads peak lag: {ccf_res['peak_comp_lag']}")

    print("2/3  Granger causality (both directions)...")
    gc = granger_both_directions(aave_chg, comp_chg)
    print(f"     Aave→Compound: min_p={gc['fwd']['min_p']}  sig={gc['fwd']['significant']}")
    print(f"     Compound→Aave: min_p={gc['rev']['min_p']}  sig={gc['rev']['significant']}")

    print(f"3/3  Rolling Granger ({ROLLING_WIN}-day windows)...")
    roll = rolling_granger(aave_chg, comp_chg)
    print(f"     Aave→Compound sig: {roll['fwd_pct_sig']:.1%} of windows")
    print(f"     Compound→Aave sig: {roll['rev_pct_sig']:.1%} of windows")

    elapsed = time.time() - t0
    note    = build_note(ccf_res, gc, roll, elapsed)

    with open(RESULTS_PATH, "w") as f:
        f.write(note)

    print(f"\nWritten to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

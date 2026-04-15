"""
Q1S4 — Post-Spike Spread Behaviour: Statistical Analysis

Tests:
  1. Regime segmentation — characterise spread in spike window (days_since_spike ≤ 7)
     vs steady-state (days_since_spike > 7)
  2. OU fit by regime — compare mean-reversion half-life in each regime
  3. Mann-Whitney U — is the absolute spread significantly larger in the spike window?
  4. Survival analysis — how quickly does |spread| fall below 1% after a spike?

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

RESULTS_PATH   = Path(__file__).parent / "results.md"
SPIKE_WINDOW   = 7     # days_since_spike threshold
EPISODE_THRESH = 1.0   # % — spread considered "elevated"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def fit_ou(spread: pd.Series) -> dict:
    if len(spread) < 30:
        return {"theta": float("nan"), "half_life": float("nan"), "mu": float("nan"), "r2": float("nan")}
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
    return {
        "theta":     round(float(theta), 4),
        "mu":        round(float(mu),    4),
        "half_life": round(float(hl),    2),
        "r2":        round(float(res.rsquared), 4),
    }


# ----------------------------------------------------------------------
# Test 1 — Regime segmentation
# ----------------------------------------------------------------------

def regime_segmentation(wide: pd.DataFrame) -> dict:
    spread = wide["spread_vs_net"].dropna()
    dss    = wide["days_since_spike"].dropna()
    df     = pd.concat([spread, dss], axis=1).dropna()
    df.columns = ["spread", "days_since_spike"]

    spike_mask  = df["days_since_spike"] <= SPIKE_WINDOW
    steady_mask = ~spike_mask

    def describe(mask, label):
        s = df.loc[mask, "spread"]
        return {
            "label":       label,
            "n_days":      int(mask.sum()),
            "pct_days":    round(float(mask.mean()), 4),
            "mean_spread": round(float(s.mean()), 4),
            "mean_abs":    round(float(s.abs().mean()), 4),
            "std_spread":  round(float(s.std()), 4),
            "median_abs":  round(float(s.abs().median()), 4),
            "pct_above_threshold": round(float((s.abs() > EPISODE_THRESH).mean()), 4),
        }

    return {
        "spike":       describe(spike_mask,  f"Spike window (≤{SPIKE_WINDOW}d)"),
        "steady":      describe(steady_mask, f"Steady-state (>{SPIKE_WINDOW}d)"),
        "spike_mask":  spike_mask,
        "steady_mask": steady_mask,
        "spread":      df["spread"],
        "dss":         df["days_since_spike"],
    }


# ----------------------------------------------------------------------
# Test 2 — OU fit by regime
# ----------------------------------------------------------------------

def ou_by_regime(spread: pd.Series, spike_mask: pd.Series) -> dict:
    return {
        "spike":  fit_ou(spread[spike_mask]),
        "steady": fit_ou(spread[~spike_mask]),
        "full":   fit_ou(spread),
    }


# ----------------------------------------------------------------------
# Test 3 — Mann-Whitney U
# ----------------------------------------------------------------------

def mann_whitney(spread: pd.Series, spike_mask: pd.Series) -> dict:
    spike_abs  = spread[spike_mask].abs().values
    steady_abs = spread[~spike_mask].abs().values

    stat, p = mannwhitneyu(spike_abs, steady_abs, alternative="greater")
    return {
        "spike_mean_abs":  round(float(spike_abs.mean()),  4),
        "steady_mean_abs": round(float(steady_abs.mean()), 4),
        "stat":            round(float(stat), 2),
        "p_value":         round(float(p),    6),
        "significant":     p < 0.05,
    }


# ----------------------------------------------------------------------
# Test 4 — Post-spike survival: how many days until |spread| < threshold?
# ----------------------------------------------------------------------

def spike_survival(spread: pd.Series, dss: pd.Series) -> dict:
    """
    For each day that is a spike onset (days_since_spike == 0 or 1),
    measure how many days until |spread| first falls below EPISODE_THRESH.
    """
    df = pd.concat([spread, dss], axis=1).dropna()
    df.columns = ["spread", "dss"]

    # Identify spike onset days (dss resets to 0 or 1)
    onset_idx = df.index[df["dss"] <= 1]

    durations = []
    for onset in onset_idx:
        pos = df.index.get_loc(onset)
        # Walk forward until |spread| < threshold or end of series
        for i in range(pos, len(df)):
            if abs(df["spread"].iloc[i]) < EPISODE_THRESH:
                durations.append(i - pos)
                break
        else:
            durations.append(len(df) - pos)  # censored — still elevated

    durations = np.array(durations)
    return {
        "n_episodes":    len(durations),
        "mean_days":     round(float(durations.mean()), 2),
        "median_days":   float(np.median(durations)),
        "pct_within_3d": round(float((durations <= 3).mean()), 4),
        "pct_within_7d": round(float((durations <= 7).mean()), 4),
        "max_days":      int(durations.max()),
    }


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(reg: dict, ou: dict, mw: dict, surv: dict, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sp, st = reg["spike"], reg["steady"]

    lines = [
        "# Q1S4 — Post-Spike Spread Behaviour: Statistical Analysis\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",

        "## 1. Regime Segmentation\n",
        f"Spike window: `days_since_spike` ≤ {SPIKE_WINDOW} | "
        f"Elevated threshold: |spread| > {EPISODE_THRESH}%\n",
        "| Metric | Spike window | Steady-state |",
        "|---|---|---|",
        f"| Days in regime | {sp['n_days']} | {st['n_days']} |",
        f"| % of sample | {sp['pct_days']:.1%} | {st['pct_days']:.1%} |",
        f"| Mean spread | {sp['mean_spread']}% | {st['mean_spread']}% |",
        f"| Mean |spread| | {sp['mean_abs']}% | {st['mean_abs']}% |",
        f"| Std spread | {sp['std_spread']}% | {st['std_spread']}% |",
        f"| Median |spread| | {sp['median_abs']}% | {st['median_abs']}% |",
        f"| % days |spread| > {EPISODE_THRESH}% | {sp['pct_above_threshold']:.1%} | {st['pct_above_threshold']:.1%} |",

        "\n## 2. Ornstein-Uhlenbeck Fit by Regime\n",
        "| Regime | θ | Half-life (days) | μ | R² |",
        "|---|---|---|---|---|",
        f"| Full sample  | {ou['full']['theta']}  | {ou['full']['half_life']}  | {ou['full']['mu']}%  | {ou['full']['r2']} |",
        f"| Spike window | {ou['spike']['theta']} | {ou['spike']['half_life']} | {ou['spike']['mu']}% | {ou['spike']['r2']} |",
        f"| Steady-state | {ou['steady']['theta']} | {ou['steady']['half_life']} | {ou['steady']['mu']}% | {ou['steady']['r2']} |",

        "\n## 3. Mann-Whitney U — Spike vs Steady-State |Spread|\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Spike window mean |spread|  | {mw['spike_mean_abs']}% |",
        f"| Steady-state mean |spread| | {mw['steady_mean_abs']}% |",
        f"| Mann-Whitney U statistic | {mw['stat']} |",
        f"| p-value (one-sided: spike > steady) | {mw['p_value']} |",
        f"| Significant (p < 0.05) | {'Yes ✓' if mw['significant'] else 'No'} |",

        "\n## 4. Post-Spike Survival — Days Until |Spread| < 1%\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Spike episodes detected | {surv['n_episodes']} |",
        f"| Mean days to close | {surv['mean_days']} |",
        f"| Median days to close | {surv['median_days']} |",
        f"| % closed within 3 days | {surv['pct_within_3d']:.1%} |",
        f"| % closed within 7 days | {surv['pct_within_7d']:.1%} |",
        f"| Max days to close | {surv['max_days']} |",
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

    print("1/4  Regime segmentation...")
    reg = regime_segmentation(wide)
    print(f"     Spike window days: {reg['spike']['n_days']} ({reg['spike']['pct_days']:.1%})")
    print(f"     Spike mean |spread|: {reg['spike']['mean_abs']}%  Steady: {reg['steady']['mean_abs']}%")

    print("2/4  OU fit by regime...")
    ou = ou_by_regime(reg["spread"], reg["spike_mask"])
    print(f"     Spike half-life: {ou['spike']['half_life']} days  Steady: {ou['steady']['half_life']} days")

    print("3/4  Mann-Whitney U test...")
    mw = mann_whitney(reg["spread"], reg["spike_mask"])
    print(f"     p={mw['p_value']}  significant={mw['significant']}")

    print("4/4  Post-spike survival analysis...")
    surv = spike_survival(reg["spread"], reg["dss"])
    print(f"     {surv['n_episodes']} episodes | mean={surv['mean_days']} days | "
          f"within 3d={surv['pct_within_3d']:.1%} | within 7d={surv['pct_within_7d']:.1%}")

    elapsed = time.time() - t0
    note    = build_note(reg, ou, mw, surv, elapsed)

    with open(RESULTS_PATH, "w") as f:
        f.write(note)

    print(f"\nWritten to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

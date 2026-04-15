"""
Q1S5 — Weekend Effect: Statistical Analysis

Tests:
  1. Day-of-week summary — mean |spread|, std, and persistence rate by day
  2. Mann-Whitney U — is weekend spread level significantly different from weekday?
  3. Cohen's d — effect size for weekend vs weekday |spread|
  4. Persistence analysis — does Friday spread persist into Monday more than other day pairs?

Writes results.md (overwrites).
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH = Path(__file__).parent / "results.md"
DAY_NAMES    = {0: "Monday", 1: "Tuesday", 2: "Wednesday",
                3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}


# ----------------------------------------------------------------------
# Test 1 — Day-of-week summary
# ----------------------------------------------------------------------

def dow_summary(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide[["spread_vs_net", "day_of_week"]].dropna().copy()
    df["abs_spread"]  = df["spread_vs_net"].abs()
    df["next_spread"] = df["spread_vs_net"].shift(-1)
    df["persists"]    = (np.sign(df["spread_vs_net"]) == np.sign(df["next_spread"])).astype(float)
    df = df.dropna()

    rows = []
    for dow in range(7):
        sub = df[df["day_of_week"] == dow]
        if len(sub) == 0:
            continue
        rows.append({
            "day":          DAY_NAMES[dow],
            "dow":          dow,
            "n":            len(sub),
            "mean_abs":     round(float(sub["abs_spread"].mean()), 4),
            "median_abs":   round(float(sub["abs_spread"].median()), 4),
            "std":          round(float(sub["spread_vs_net"].std()), 4),
            "persist_rate": round(float(sub["persists"].mean()), 4),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Test 2 — Mann-Whitney U: weekend vs weekday
# ----------------------------------------------------------------------

def mann_whitney_weekend(wide: pd.DataFrame) -> dict:
    df = wide[["spread_vs_net", "day_of_week"]].dropna()
    weekend_abs = df[df["day_of_week"].isin([5, 6])]["spread_vs_net"].abs().values
    weekday_abs = df[df["day_of_week"].isin([0, 1, 2, 3, 4])]["spread_vs_net"].abs().values

    stat, p = mannwhitneyu(weekend_abs, weekday_abs, alternative="two-sided")
    return {
        "weekend_mean_abs": round(float(weekend_abs.mean()), 4),
        "weekday_mean_abs": round(float(weekday_abs.mean()), 4),
        "stat":             round(float(stat), 2),
        "p_value":          round(float(p), 4),
        "significant":      p < 0.05,
        "n_weekend":        len(weekend_abs),
        "n_weekday":        len(weekday_abs),
    }


# ----------------------------------------------------------------------
# Test 3 — Cohen's d effect size
# ----------------------------------------------------------------------

def cohens_d(wide: pd.DataFrame) -> float:
    df = wide[["spread_vs_net", "day_of_week"]].dropna()
    weekend = df[df["day_of_week"].isin([5, 6])]["spread_vs_net"].abs().values
    weekday = df[df["day_of_week"].isin([0, 1, 2, 3, 4])]["spread_vs_net"].abs().values

    pooled_std = np.sqrt((weekend.std() ** 2 + weekday.std() ** 2) / 2)
    d = (weekend.mean() - weekday.mean()) / pooled_std if pooled_std > 0 else 0.0
    return round(float(d), 4)


# ----------------------------------------------------------------------
# Test 4 — Friday → Monday persistence
# ----------------------------------------------------------------------

def friday_monday_persistence(wide: pd.DataFrame) -> dict:
    """
    For each Friday observation, find the next Monday and compare whether
    the spread sign persists across the weekend gap.
    """
    df = wide[["spread_vs_net", "day_of_week"]].dropna().copy()

    fridays  = df[df["day_of_week"] == 4].index
    mondays  = df[df["day_of_week"] == 0].index

    persists = []
    for fri in fridays:
        # Find the first Monday after this Friday
        future_mondays = mondays[mondays > fri]
        if len(future_mondays) == 0:
            continue
        mon = future_mondays[0]
        # Only count if it's within 4 days (not a holiday-extended gap)
        if (mon - fri).days > 4:
            continue
        fri_sign = np.sign(df.loc[fri, "spread_vs_net"])
        mon_sign = np.sign(df.loc[mon, "spread_vs_net"])
        if fri_sign != 0:
            persists.append(int(fri_sign == mon_sign))

    # Compare with same-day persistence for adjacent weekday pairs
    # e.g. Tuesday → Wednesday
    mid_week_persists = []
    for i in range(len(df) - 1):
        if df["day_of_week"].iloc[i] == 2 and df["day_of_week"].iloc[i + 1] == 3:
            s1 = np.sign(df["spread_vs_net"].iloc[i])
            s2 = np.sign(df["spread_vs_net"].iloc[i + 1])
            if s1 != 0:
                mid_week_persists.append(int(s1 == s2))

    return {
        "n_friday_monday_pairs":    len(persists),
        "fri_mon_persist_rate":     round(float(np.mean(persists)), 4) if persists else float("nan"),
        "n_tue_wed_pairs":          len(mid_week_persists),
        "tue_wed_persist_rate":     round(float(np.mean(mid_week_persists)), 4) if mid_week_persists else float("nan"),
    }


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_note(summary: pd.DataFrame, mw: dict, d: float,
               fri_mon: dict, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Q1S5 — Weekend Effect: Statistical Analysis\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",

        "## 1. Day-of-Week Summary\n",
        "| Day | N | Mean |spread| | Median |spread| | Std | Persist rate |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['day']} | {row['n']} | {row['mean_abs']}% | "
            f"{row['median_abs']}% | {row['std']}% | {row['persist_rate']:.1%} |"
        )

    lines += [
        "\n## 2. Mann-Whitney U — Weekend vs Weekday |Spread|\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Weekend mean |spread| | {mw['weekend_mean_abs']}% |",
        f"| Weekday mean |spread| | {mw['weekday_mean_abs']}% |",
        f"| Mann-Whitney U | {mw['stat']} |",
        f"| p-value (two-sided) | {mw['p_value']} |",
        f"| Significant (p < 0.05) | {'Yes ✓' if mw['significant'] else 'No'} |",
        f"| N (weekend / weekday) | {mw['n_weekend']} / {mw['n_weekday']} |",

        "\n## 3. Cohen's d — Weekend vs Weekday Effect Size\n",
        f"Cohen's d = **{d}**\n",
        "Interpretation: |d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large\n",

        "\n## 4. Friday → Monday Persistence\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Friday → Monday pairs | {fri_mon['n_friday_monday_pairs']} |",
        f"| Fri→Mon persist rate | {fri_mon['fri_mon_persist_rate']:.1%} |",
        f"| Tue→Wed pairs | {fri_mon['n_tue_wed_pairs']} |",
        f"| Tue→Wed persist rate | {fri_mon['tue_wed_persist_rate']:.1%} |",
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

    print("1/4  Day-of-week summary...")
    summary = dow_summary(wide)
    for _, row in summary.iterrows():
        print(f"     {row['day']:<12} mean|spread|={row['mean_abs']}%  persist={row['persist_rate']:.1%}")

    print("2/4  Mann-Whitney U (weekend vs weekday)...")
    mw = mann_whitney_weekend(wide)
    print(f"     Weekend mean: {mw['weekend_mean_abs']}%  Weekday mean: {mw['weekday_mean_abs']}%")
    print(f"     p={mw['p_value']}  significant={mw['significant']}")

    print("3/4  Cohen's d...")
    d = cohens_d(wide)
    print(f"     d={d}")

    print("4/4  Friday → Monday persistence...")
    fri_mon = friday_monday_persistence(wide)
    print(f"     Fri→Mon: {fri_mon['fri_mon_persist_rate']:.1%} ({fri_mon['n_friday_monday_pairs']} pairs)")
    print(f"     Tue→Wed: {fri_mon['tue_wed_persist_rate']:.1%} ({fri_mon['n_tue_wed_pairs']} pairs)")

    elapsed = time.time() - t0
    note    = build_note(summary, mw, d, fri_mon, elapsed)

    with open(RESULTS_PATH, "w") as f:
        f.write(note)

    print(f"\nWritten to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

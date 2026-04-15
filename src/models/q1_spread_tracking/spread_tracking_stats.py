"""
Q1 — Spread Tracking: Statistical Analysis
Implements 6 statistical tests as outlined in q1_statistical_approach.md:
  1. Cointegration (Engle-Granger)
  2. Spread stationarity (ADF + KPSS)
  3. Ornstein-Uhlenbeck fit (mean-reversion speed + half-life)
  4. Rolling correlation (30-day window)
  5. Spread distribution + episode analysis
  6. Cross-correlation function (CCF)

Appends a new section to results.md.
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint, ccf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.features.engineering import create_features

RESULTS_PATH = Path(__file__).parent / "results.md"
SPREAD_THRESHOLD = 1.0   # % — defines a "spread episode"
ROLLING_WINDOW   = 30    # days for rolling correlation
CCF_LAGS         = 5     # lags each side for cross-correlation


# ----------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------

def load_series(wide: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (aave_rates, compound_net_rates, spread) aligned on date index."""
    aave     = wide["aave_apyBase"]
    compound = wide["compound_net"]
    spread   = wide["spread_vs_net"]
    return aave, compound, spread


# ----------------------------------------------------------------------
# Test 1 — Cointegration (Engle-Granger)
# ----------------------------------------------------------------------

def test_cointegration(aave: pd.Series, compound: pd.Series) -> dict:
    stat, p, crit = coint(aave.values, compound.values)
    return {
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(p), 4),
        "crit_1pct":   round(float(crit[0]), 4),
        "crit_5pct":   round(float(crit[1]), 4),
        "crit_10pct":  round(float(crit[2]), 4),
        "cointegrated": p < 0.05,
    }


# ----------------------------------------------------------------------
# Test 2 — Spread stationarity (ADF + KPSS)
# ----------------------------------------------------------------------

def test_stationarity(spread: pd.Series) -> dict:
    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(spread.values, autolag="AIC")
    kpss_stat, kpss_p, kpss_lags, kpss_crit   = kpss(spread.values, regression="c", nlags="auto")

    adf_stationary  = adf_p  < 0.05
    kpss_stationary = kpss_p > 0.05

    if adf_stationary and kpss_stationary:
        conclusion = "Stationary"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "Non-stationary"
    elif adf_stationary and not kpss_stationary:
        conclusion = "Trend-stationary (consider differencing)"
    else:
        conclusion = "Inconclusive"

    return {
        "adf_stat":       round(float(adf_stat), 4),
        "adf_p":          round(float(adf_p), 4),
        "adf_lags":       int(adf_lags),
        "adf_crit_5pct":  round(float(adf_crit["5%"]), 4),
        "kpss_stat":      round(float(kpss_stat), 4),
        "kpss_p":         round(float(kpss_p), 4),
        "kpss_lags":      int(kpss_lags),
        "kpss_crit_5pct": round(float(kpss_crit["5%"]), 4),
        "conclusion":     conclusion,
    }


# ----------------------------------------------------------------------
# Test 3 — Ornstein-Uhlenbeck fit
# ----------------------------------------------------------------------

def fit_ou(spread: pd.Series) -> dict:
    """
    Fit OU process by regressing daily spread change on lagged spread level:
      Δspread_t = α + β * spread_{t-1} + ε
    where θ = -β (mean-reversion speed), μ = -α/β (long-run mean)
    """
    spread_lag  = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    idx         = spread_lag.index.intersection(spread_diff.index)

    X      = sm.add_constant(spread_lag.loc[idx])
    result = sm.OLS(spread_diff.loc[idx], X).fit()

    alpha  = result.params.iloc[0]
    beta   = result.params.iloc[1]
    theta  = -beta
    mu     = alpha / theta if theta != 0 else float("nan")
    sigma  = result.resid.std()
    hl     = np.log(2) / theta if theta > 0 else float("nan")

    return {
        "theta":     round(float(theta), 6),
        "mu":        round(float(mu), 4),
        "sigma":     round(float(sigma), 4),
        "half_life": round(float(hl), 2),
        "r2":        round(float(result.rsquared), 4),
        "p_beta":    round(float(result.pvalues.iloc[1]), 4),
    }


# ----------------------------------------------------------------------
# Test 4 — Rolling correlation
# ----------------------------------------------------------------------

def rolling_correlation(aave: pd.Series, compound: pd.Series) -> dict:
    roll = aave.rolling(ROLLING_WINDOW).corr(compound).dropna()
    return {
        "mean":   round(float(roll.mean()), 4),
        "min":    round(float(roll.min()), 4),
        "max":    round(float(roll.max()), 4),
        "std":    round(float(roll.std()), 4),
        "pct_above_090": round(float((roll > 0.90).mean()), 4),
        "pct_below_050": round(float((roll < 0.50).mean()), 4),
        "series": roll,
    }


# ----------------------------------------------------------------------
# Test 5 — Spread distribution + episode analysis
# ----------------------------------------------------------------------

def spread_episodes(spread: pd.Series, threshold: float = SPREAD_THRESHOLD) -> dict:
    desc = spread.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    # Episode detection: consecutive days where |spread| > threshold
    above    = (spread.abs() > threshold).astype(int)
    episodes = []
    start    = None

    for date, val in above.items():
        if val == 1 and start is None:
            start = date
        elif val == 0 and start is not None:
            duration = (date - start).days
            peak     = spread.loc[start:date].abs().max()
            episodes.append({"start": start, "duration": duration, "peak": peak})
            start = None
    if start is not None:                              # episode still open at end
        duration = (spread.index[-1] - start).days
        peak     = spread.loc[start:].abs().max()
        episodes.append({"start": start, "duration": duration, "peak": peak})

    ep_df = pd.DataFrame(episodes) if episodes else pd.DataFrame(columns=["start", "duration", "peak"])

    return {
        "mean":          round(float(desc["mean"]), 4),
        "std":           round(float(desc["std"]), 4),
        "skew":          round(float(spread.skew()), 4),
        "kurtosis":      round(float(spread.kurtosis()), 4),
        "p5":            round(float(desc["5%"]), 4),
        "p25":           round(float(desc["25%"]), 4),
        "median":        round(float(desc["50%"]), 4),
        "p75":           round(float(desc["75%"]), 4),
        "p95":           round(float(desc["95%"]), 4),
        "n_episodes":    len(ep_df),
        "mean_duration": round(float(ep_df["duration"].mean()), 1) if not ep_df.empty else 0,
        "max_duration":  int(ep_df["duration"].max()) if not ep_df.empty else 0,
        "mean_peak":     round(float(ep_df["peak"].mean()), 4) if not ep_df.empty else 0,
        "max_peak":      round(float(ep_df["peak"].max()), 4) if not ep_df.empty else 0,
        "threshold":     threshold,
    }


# ----------------------------------------------------------------------
# Test 6 — Cross-correlation function (CCF)
# ----------------------------------------------------------------------

def cross_correlation(aave: pd.Series, compound: pd.Series, nlags: int = CCF_LAGS) -> dict:
    aave_diff = aave.diff().dropna()
    comp_diff = compound.diff().dropna()
    idx       = aave_diff.index.intersection(comp_diff.index)

    # CCF: aave leads compound (positive lag = aave moves first)
    fwd = ccf(aave_diff.loc[idx].values, comp_diff.loc[idx].values, nlags=nlags, alpha=None)
    # CCF: compound leads aave (positive lag = compound moves first)
    rev = ccf(comp_diff.loc[idx].values, aave_diff.loc[idx].values, nlags=nlags, alpha=None)

    fwd_vals = {f"lag_{i}": round(float(v), 4) for i, v in enumerate(fwd)}
    rev_vals = {f"lag_{i}": round(float(v), 4) for i, v in enumerate(rev)}

    peak_fwd_lag = int(np.argmax(np.abs(fwd[1:])) + 1)
    peak_rev_lag = int(np.argmax(np.abs(rev[1:])) + 1)

    return {
        "aave_leads_compound": fwd_vals,
        "compound_leads_aave": rev_vals,
        "peak_aave_lead_lag":  peak_fwd_lag,
        "peak_comp_lead_lag":  peak_rev_lag,
        "contemp_corr":        round(float(fwd[0]), 4),
    }


# ----------------------------------------------------------------------
# Markdown builder
# ----------------------------------------------------------------------

def build_stats_note(results: dict, elapsed: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    r   = results
    lines = [
        "\n---\n",
        "# Q1 Spread Tracking — Statistical Analysis\n",
        f"*Run: {now} | Elapsed: {elapsed:.1f}s*\n",

        "## 1. Cointegration (Engle-Granger)\n",
        f"| Statistic | p-value | Cointegrated? |",
        f"|---|---|---|",
        f"| {r['coint']['statistic']} | {r['coint']['p_value']} | {'Yes ✓' if r['coint']['cointegrated'] else 'No ✗'} |",
        f"\nCritical values: 1%={r['coint']['crit_1pct']}, 5%={r['coint']['crit_5pct']}, 10%={r['coint']['crit_10pct']}",

        "\n## 2. Spread Stationarity (ADF + KPSS)\n",
        f"| Test | Statistic | p-value | 5% Critical | Lags |",
        f"|---|---|---|---|---|",
        f"| ADF  | {r['stat']['adf_stat']} | {r['stat']['adf_p']} | {r['stat']['adf_crit_5pct']} | {r['stat']['adf_lags']} |",
        f"| KPSS | {r['stat']['kpss_stat']} | {r['stat']['kpss_p']} | {r['stat']['kpss_crit_5pct']} | {r['stat']['kpss_lags']} |",
        f"\n**Conclusion: {r['stat']['conclusion']}**",

        "\n## 3. Ornstein-Uhlenbeck Fit\n",
        f"| Parameter | Value | Interpretation |",
        f"|---|---|---|",
        f"| θ (mean-reversion speed) | {r['ou']['theta']} | higher = faster reversion |",
        f"| μ (long-run mean spread) | {r['ou']['mu']}% | spread reverts toward this level |",
        f"| σ (volatility) | {r['ou']['sigma']}% | daily noise in spread process |",
        f"| Half-life | {r['ou']['half_life']} days | time for shock to decay by 50% |",
        f"| R² | {r['ou']['r2']} | |",
        f"| p-value (β) | {r['ou']['p_beta']} | significance of mean-reversion term |",

        "\n## 4. Rolling Correlation (30-day window)\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mean correlation | {r['roll']['mean']} |",
        f"| Min correlation  | {r['roll']['min']} |",
        f"| Max correlation  | {r['roll']['max']} |",
        f"| Std              | {r['roll']['std']} |",
        f"| % days corr > 0.90 | {r['roll']['pct_above_090']:.1%} |",
        f"| % days corr < 0.50 | {r['roll']['pct_below_050']:.1%} |",

        "\n## 5. Spread Distribution & Episode Analysis\n",
        f"**Distribution (spread_vs_net):**\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mean   | {r['ep']['mean']}% |",
        f"| Std    | {r['ep']['std']}% |",
        f"| Skew   | {r['ep']['skew']} |",
        f"| Kurtosis | {r['ep']['kurtosis']} |",
        f"| p5 / p95 | {r['ep']['p5']}% / {r['ep']['p95']}% |",
        f"| p25 / p75 | {r['ep']['p25']}% / {r['ep']['p75']}% |",
        f"| Median | {r['ep']['median']}% |\n",
        f"**Episodes (|spread| > {r['ep']['threshold']}%):**\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Count           | {r['ep']['n_episodes']} |",
        f"| Mean duration   | {r['ep']['mean_duration']} days |",
        f"| Max duration    | {r['ep']['max_duration']} days |",
        f"| Mean peak       | {r['ep']['mean_peak']}% |",
        f"| Max peak        | {r['ep']['max_peak']}% |",

        "\n## 6. Cross-Correlation Function (CCF)\n",
        f"Contemporaneous correlation: {r['ccf']['contemp_corr']}\n",
        f"**Aave leads Compound** — peak CCF at lag {r['ccf']['peak_aave_lead_lag']} day(s):",
        "| Lag | CCF |",
        "|---|---|",
    ]

    for i in range(len(r['ccf']['aave_leads_compound'])):
        lines.append(f"| {i} | {r['ccf']['aave_leads_compound'][f'lag_{i}']} |")

    lines += [
        f"\n**Compound leads Aave** — peak CCF at lag {r['ccf']['peak_comp_lead_lag']} day(s):",
        "| Lag | CCF |",
        "|---|---|",
    ]
    for i in range(len(r['ccf']['aave_leads_compound'])):
        lines.append(f"| {i} | {r['ccf']['compound_leads_aave'][f'lag_{i}']} |")

    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    t0 = time.time()

    print("Loading data and engineering features...")
    df   = load_all()
    wide = create_features(df)
    aave, compound, spread = load_series(wide)
    print(f"  {len(spread):,} overlapping days | spread range: {spread.min():.2f}% to {spread.max():.2f}%\n")

    results = {}

    print("1/6  Cointegration (Engle-Granger)...")
    results["coint"] = test_cointegration(aave, compound)
    print(f"     p={results['coint']['p_value']} | Cointegrated: {results['coint']['cointegrated']}")

    print("2/6  Stationarity (ADF + KPSS)...")
    results["stat"] = test_stationarity(spread)
    print(f"     ADF p={results['stat']['adf_p']} | KPSS p={results['stat']['kpss_p']} | {results['stat']['conclusion']}")

    print("3/6  Ornstein-Uhlenbeck fit...")
    results["ou"] = fit_ou(spread)
    print(f"     θ={results['ou']['theta']} | half-life={results['ou']['half_life']} days | μ={results['ou']['mu']}%")

    print("4/6  Rolling correlation (30-day)...")
    results["roll"] = rolling_correlation(aave, compound)
    print(f"     Mean corr={results['roll']['mean']} | Min={results['roll']['min']} | Max={results['roll']['max']}")

    print("5/6  Spread distribution + episode analysis...")
    results["ep"] = spread_episodes(spread)
    print(f"     {results['ep']['n_episodes']} episodes > {SPREAD_THRESHOLD}% | mean duration={results['ep']['mean_duration']} days | max={results['ep']['max_duration']} days")

    print("6/6  Cross-correlation function...")
    results["ccf"] = cross_correlation(aave, compound)
    print(f"     Contemporaneous corr={results['ccf']['contemp_corr']} | Aave peak lead lag={results['ccf']['peak_aave_lead_lag']}d | Compound peak lead lag={results['ccf']['peak_comp_lead_lag']}d")

    elapsed = time.time() - t0
    note    = build_stats_note(results, elapsed)

    with open(RESULTS_PATH, "a") as f:
        f.write(note)

    print(f"\nAppended to {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

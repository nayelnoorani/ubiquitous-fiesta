import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format loader output to wide format.
    One row per date, columns prefixed by protocol name.
    Only rows present in both protocols are kept (inner join).
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    aave = (
        df[df["project"] == "aave-v3"]
        .set_index("date")[["apyBase", "apyReward", "tvlUsd"]]
        .rename(columns={"apyBase": "aave_apyBase", "apyReward": "aave_apyReward", "tvlUsd": "aave_tvlUsd"})
    )
    comp = (
        df[df["project"] == "compound-v3"]
        .set_index("date")[["apyBase", "apyReward", "tvlUsd"]]
        .rename(columns={"apyBase": "compound_apyBase", "apyReward": "compound_apyReward", "tvlUsd": "compound_tvlUsd"})
    )

    wide = aave.join(comp, how="inner")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    return wide


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all 22 features from the feature engineering spec.
    Accepts the long-format DataFrame returned by load_all().
    Returns a wide-format DataFrame (one row per date) with
    original protocol columns and all engineered features.
    """
    wide = _pivot_to_wide(df)

    # Compound net borrow cost: rewards are paid to borrowers, reducing their cost
    wide["compound_apyReward"] = wide["compound_apyReward"].fillna(0)
    wide["compound_net"] = wide["compound_apyBase"] - wide["compound_apyReward"]

    # ------------------------------------------------------------------
    # Spread features
    # ------------------------------------------------------------------
    wide["spread_vs_net"]  = wide["aave_apyBase"] - wide["compound_net"]
    wide["spread_vs_base"] = wide["aave_apyBase"] - wide["compound_apyBase"]

    for spread in ["spread_vs_net", "spread_vs_base"]:
        wide[f"{spread}_lag_1d"]          = wide[spread].shift(1)
        wide[f"{spread}_rolling_mean_7d"] = wide[spread].rolling(7,  min_periods=1).mean()
        wide[f"{spread}_rolling_std_7d"]  = wide[spread].rolling(7,  min_periods=2).std()
        roll30_mean = wide[spread].rolling(30, min_periods=10).mean()
        roll30_std  = wide[spread].rolling(30, min_periods=10).std()
        wide[f"{spread}_zscore_30d"] = (wide[spread] - roll30_mean) / roll30_std

    # ------------------------------------------------------------------
    # Rate momentum features
    # ------------------------------------------------------------------
    wide["aave_rate_change_1d"]        = wide["aave_apyBase"].diff(1)
    wide["compound_net_change_1d"]     = wide["compound_net"].diff(1)
    wide["compound_base_change_1d"]    = wide["compound_apyBase"].diff(1)

    wide["rate_divergence_direction_vs_net"]  = np.sign(
        wide["aave_rate_change_1d"] - wide["compound_net_change_1d"]
    )
    wide["rate_divergence_direction_vs_base"] = np.sign(
        wide["aave_rate_change_1d"] - wide["compound_base_change_1d"]
    )

    # ------------------------------------------------------------------
    # Reward component (standalone)
    # ------------------------------------------------------------------
    # compound_apyReward already present from above

    # ------------------------------------------------------------------
    # Liquidity features
    # ------------------------------------------------------------------
    wide["tvl_ratio"]                  = wide["aave_tvlUsd"] / wide["compound_tvlUsd"]
    wide["aave_tvl_change_pct_1d"]     = wide["aave_tvlUsd"].pct_change(1)
    wide["compound_tvl_change_pct_1d"] = wide["compound_tvlUsd"].pct_change(1)

    # ------------------------------------------------------------------
    # Regime features
    # ------------------------------------------------------------------
    wide["is_spike"] = (
        (wide["aave_apyBase"] > 10) | (wide["compound_apyBase"] > 10)
    ).astype(int)

    spike_dates        = wide.index.to_series().where(wide["is_spike"] == 1)
    last_spike         = spike_dates.ffill()
    wide["days_since_spike"] = (wide.index - last_spike).dt.days

    # ------------------------------------------------------------------
    # Calendar feature
    # ------------------------------------------------------------------
    wide["day_of_week"] = wide.index.dayofweek

    return wide


ENGINEERED_FEATURE_COLS = [
    "spread_vs_net", "spread_vs_base",
    "spread_vs_net_lag_1d", "spread_vs_base_lag_1d",
    "spread_vs_net_rolling_mean_7d", "spread_vs_base_rolling_mean_7d",
    "spread_vs_net_rolling_std_7d", "spread_vs_base_rolling_std_7d",
    "spread_vs_net_zscore_30d", "spread_vs_base_zscore_30d",
    "aave_rate_change_1d", "compound_net_change_1d", "compound_base_change_1d",
    "rate_divergence_direction_vs_net", "rate_divergence_direction_vs_base",
    "compound_apyReward",
    "tvl_ratio", "aave_tvl_change_pct_1d", "compound_tvl_change_pct_1d",
    "is_spike", "days_since_spike",
    "day_of_week",
]


def select_features(
    wide: pd.DataFrame,
    corr_threshold: float = 0.95,
    variance_threshold_pct: float = 0.01,
) -> tuple[list[str], pd.DataFrame]:
    """
    Filter engineered features by correlation and variance.

    Steps:
      1. Drop features with > corr_threshold absolute correlation to any
         earlier feature in the list (keep first, drop redundant).
      2. Drop features whose variance is below variance_threshold_pct of
         the mean variance across all remaining features.

    Parameters
    ----------
    wide : DataFrame returned by create_features()
    corr_threshold : drop feature if |corr| > this with any kept feature
    variance_threshold_pct : drop feature if variance < this fraction of
                             the mean variance across all candidates

    Returns
    -------
    selected_cols : list of surviving feature names
    reduced_df    : wide DataFrame containing only selected feature columns
    """
    candidates = [c for c in ENGINEERED_FEATURE_COLS if c in wide.columns]
    numeric    = wide[candidates].select_dtypes(include="number").dropna()

    dropped_corr     = {}   # col -> (correlated_with, corr_value)
    dropped_variance = {}   # col -> (variance, threshold)

    # ------------------------------------------------------------------
    # Step 1 — Correlation filter
    # ------------------------------------------------------------------
    corr_matrix = numeric.corr().abs()
    kept = []

    for col in numeric.columns:
        if kept:
            max_corr_with_kept = corr_matrix.loc[col, kept].max()
            most_correlated    = corr_matrix.loc[col, kept].idxmax()
            if max_corr_with_kept > corr_threshold:
                dropped_corr[col] = (most_correlated, round(max_corr_with_kept, 4))
                continue
        kept.append(col)

    # ------------------------------------------------------------------
    # Step 2 — Variance filter (applied to correlation-survivors only)
    # ------------------------------------------------------------------
    variances        = numeric[kept].var()
    mean_variance    = variances.mean()
    threshold        = variance_threshold_pct * mean_variance
    low_var_cols     = variances[variances < threshold].index.tolist()

    for col in low_var_cols:
        dropped_variance[col] = (round(variances[col], 6), round(threshold, 6))

    selected_cols = [c for c in kept if c not in low_var_cols]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    print("=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)
    print(f"  Candidates:          {len(candidates)}")
    print(f"  Dropped (corr):      {len(dropped_corr)}")
    print(f"  Dropped (variance):  {len(dropped_variance)}")
    print(f"  Selected:            {len(selected_cols)}")

    if dropped_corr:
        print(f"\nDropped — correlation > {corr_threshold}:")
        for col, (partner, val) in dropped_corr.items():
            print(f"  {col:<45}  |r|={val} with '{partner}'")

    if dropped_variance:
        print(f"\nDropped — variance < {variance_threshold_pct:.0%} of mean variance "
              f"(threshold={threshold:.6f}):")
        for col, (var, thr) in dropped_variance.items():
            print(f"  {col:<45}  var={var}")

    print(f"\nSelected features ({len(selected_cols)}):")
    for col in selected_cols:
        print(f"  {col}")
    print("=" * 60)

    return selected_cols, wide[selected_cols]


def print_feature_report(wide: pd.DataFrame):
    engineered_cols = [c for c in wide.columns if c not in [
        "aave_apyBase", "aave_apyReward", "aave_tvlUsd",
        "compound_apyBase", "compound_apyReward", "compound_tvlUsd", "compound_net",
    ]]

    print("=" * 60)
    print(f"FEATURE REPORT — {len(wide):,} rows × {wide.shape[1]} columns")
    print(f"Date range: {wide.index.min().date()} → {wide.index.max().date()}")
    print("=" * 60)

    print(f"\nOriginal columns: {wide.shape[1] - len(engineered_cols)}")
    print(f"Engineered features: {len(engineered_cols)}")

    print("\nEngineered feature null counts:")
    nulls = wide[engineered_cols].isnull().sum()
    for col, n in nulls.items():
        pct = n / len(wide) * 100
        flag = "  <-- expected (warmup)" if pct > 0 else ""
        print(f"  {col:<45} {n:>4} ({pct:4.1f}%){flag}")

    print("\nSample (first 3 non-null rows):")
    first_valid = wide[engineered_cols].dropna().head(3)
    print(first_valid.T.to_string())
    print("=" * 60)


if __name__ == "__main__":
    from src.data.loader import load_all

    df   = load_all()
    wide = create_features(df)
    print_feature_report(wide)
    print()
    selected_cols, reduced_df = select_features(wide)

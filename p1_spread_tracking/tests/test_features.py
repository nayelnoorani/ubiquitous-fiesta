import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineering import create_features, ENGINEERED_FEATURE_COLS

EXPECTED_FEATURE_COUNT = 22  # as per ENGINEERED_FEATURE_COLS


def _make_long_df(n_days: int = 400) -> pd.DataFrame:
    """Synthetic long-format df with both protocols over n_days dates."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")

    aave = pd.DataFrame({
        "timestamp":  dates,
        "apyBase":    rng.uniform(2.0, 8.0, n_days).astype("float64"),
        "apyReward":  np.zeros(n_days),
        "tvlUsd":     rng.integers(3_000_000, 6_000_000, n_days),
        "project":    "aave-v3",
        "symbol":     "USDC",
        "chain":      "Ethereum",
    })
    comp = pd.DataFrame({
        "timestamp":  dates,
        "apyBase":    rng.uniform(2.0, 8.0, n_days).astype("float64"),
        "apyReward":  rng.uniform(0.1, 0.8, n_days).astype("float64"),
        "tvlUsd":     rng.integers(1_000_000, 3_000_000, n_days),
        "project":    "compound-v3",
        "symbol":     "USDC",
        "chain":      "Ethereum",
    })
    return pd.concat([aave, comp], ignore_index=True)


@pytest.fixture(scope="module")
def wide():
    return create_features(_make_long_df())


class TestColumnCount:
    def test_all_engineered_features_present(self, wide):
        missing = [c for c in ENGINEERED_FEATURE_COLS if c not in wide.columns]
        assert missing == [], f"Missing engineered columns: {missing}"

    def test_engineered_feature_count(self, wide):
        present = [c for c in ENGINEERED_FEATURE_COLS if c in wide.columns]
        assert len(present) == EXPECTED_FEATURE_COUNT


class TestNullValues:
    # First 30 rows may have NaNs from lag/rolling warmup — skip them
    def test_no_nans_after_warmup(self, wide):
        warmup = 31
        # days_since_spike is legitimately NaN before the first spike in the dataset
        check_cols = [
            "spread_vs_net", "spread_vs_net_lag_1d",
            "spread_vs_net_rolling_mean_7d",
            "aave_rate_change_1d", "compound_net_change_1d",
            "tvl_ratio", "is_spike", "day_of_week",
        ]
        subset = wide.iloc[warmup:][check_cols]
        null_counts = subset.isnull().sum()
        assert null_counts.sum() == 0, f"Unexpected NaNs after warmup:\n{null_counts[null_counts > 0]}"


class TestFeatureRanges:
    def test_is_spike_binary(self, wide):
        assert wide["is_spike"].isin([0, 1]).all()

    def test_day_of_week_range(self, wide):
        assert wide["day_of_week"].between(0, 6).all()

    def test_tvl_ratio_positive(self, wide):
        assert (wide["tvl_ratio"] > 0).all()

    def test_rate_divergence_direction_values(self, wide):
        valid = {-1.0, 0.0, 1.0}
        assert set(wide["rate_divergence_direction_vs_net"].dropna().unique()).issubset(valid)

    def test_days_since_spike_non_negative(self, wide):
        non_null = wide["days_since_spike"].dropna()
        assert (non_null >= 0).all()

    def test_spread_vs_net_matches_formula(self, wide):
        expected = wide["aave_apyBase"] - wide["compound_net"]
        pd.testing.assert_series_equal(wide["spread_vs_net"], expected, check_names=False)

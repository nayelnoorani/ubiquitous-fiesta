import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.quality import check_data_quality


def _make_valid_df(n: int = 200) -> pd.DataFrame:
    """Minimal long-format df that should pass all quality checks."""
    rng = np.random.default_rng(42)
    half = n // 2
    dates = pd.date_range("2023-01-01", periods=half, freq="D", tz="UTC")

    aave = pd.DataFrame({
        "timestamp": dates,
        "apyBase":   rng.uniform(1.0, 10.0, half).astype("float64"),
        "tvlUsd":    rng.integers(1_000_000, 5_000_000, half),
        "project":   "aave-v3",
        "symbol":    "USDC",
        "chain":     "Ethereum",
    })
    comp = pd.DataFrame({
        "timestamp": dates,
        "apyBase":   rng.uniform(1.0, 10.0, half).astype("float64"),
        "tvlUsd":    rng.integers(500_000, 2_000_000, half),
        "project":   "compound-v3",
        "symbol":    "USDC",
        "chain":     "Ethereum",
    })
    return pd.concat([aave, comp], ignore_index=True)


class TestQualityGatePasses:
    def test_success_flag(self):
        result = check_data_quality(_make_valid_df())
        assert result["success"] is True

    def test_no_failures(self):
        result = check_data_quality(_make_valid_df())
        assert result["failures"] == []

    def test_statistics_populated(self):
        result = check_data_quality(_make_valid_df())
        assert result["statistics"]["total_rows"] == 200
        assert "group_distribution" in result["statistics"]


class TestQualityGateCatchesBadData:
    def test_catches_too_few_rows(self):
        df = _make_valid_df(n=20)  # only 10 rows per protocol — below MIN_ROWS_CRITICAL
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("row_count" in f for f in result["failures"])

    def test_catches_missing_column(self):
        df = _make_valid_df().drop(columns=["apyBase"])
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("schema" in f for f in result["failures"])

    def test_catches_wrong_dtype(self):
        df = _make_valid_df()
        # Break tvlUsd dtype: quality gate expects integer kind ('i'), float will fail
        df["tvlUsd"] = df["tvlUsd"].astype("float64")
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("schema" in f for f in result["failures"])

    def test_catches_out_of_range_values(self):
        df = _make_valid_df()
        df.loc[0, "apyBase"] = -5.0  # below minimum 0.0
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("range" in f for f in result["failures"])

    def test_catches_single_protocol(self):
        df = _make_valid_df()
        df = df[df["project"] == "aave-v3"].reset_index(drop=True)
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("distribution" in f for f in result["failures"])

    def test_catches_high_null_rate(self):
        df = _make_valid_df()
        # Set > 50% of apyBase to NaN → critical null failure
        df.loc[:110, "apyBase"] = np.nan
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("nulls" in f for f in result["failures"])

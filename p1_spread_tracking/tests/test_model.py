import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.q1s7_direction_prediction.direction_ml import (
    FEATURE_COLS,
    prepare,
    run_logistic,
)


def _make_wide_df(n: int = 400) -> pd.DataFrame:
    """
    Synthetic wide-format df with all columns required by prepare() and run_logistic().
    Uses a simple mean-reverting spread so the model has a weak signal to find.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.7 * spread[i - 1] + rng.normal(0, 0.3)

    aave_rate  = 5.0 + spread
    comp_rate  = 5.0 + rng.normal(0, 0.1, n)
    aave_tvl   = 4_000_000 + rng.normal(0, 100_000, n)
    comp_tvl   = 1_500_000 + rng.normal(0, 50_000,  n)

    df = pd.DataFrame(index=dates)
    df["aave_apyBase"]               = aave_rate
    df["compound_net"]               = comp_rate
    df["spread_vs_net"]              = spread
    df["spread_vs_net_lag_1d"]       = pd.Series(spread, index=dates).shift(1)
    df["spread_vs_net_rolling_mean_7d"] = pd.Series(spread, index=dates).rolling(7, min_periods=1).mean()
    df["spread_vs_net_rolling_std_7d"]  = pd.Series(spread, index=dates).rolling(7, min_periods=2).std().fillna(0)
    df["aave_rate_change_1d"]        = pd.Series(aave_rate, index=dates).diff().fillna(0)
    df["compound_net_change_1d"]     = pd.Series(comp_rate, index=dates).diff().fillna(0)
    df["tvl_ratio"]                  = aave_tvl / comp_tvl
    df["aave_tvl_change_pct_1d"]     = pd.Series(aave_tvl, index=dates).pct_change().fillna(0)
    df["days_since_spike"]           = np.arange(n, dtype=float)
    df["day_of_week"]                = pd.Series(dates).dt.dayofweek.values
    return df


@pytest.fixture(scope="module")
def trained_result():
    wide = _make_wide_df()
    _, train, test = prepare(wide)
    return run_logistic(train, test)


class TestModelLoadsAndPredicts:
    def test_result_has_expected_keys(self, trained_result):
        for key in ("model", "auc", "accuracy", "precision", "recall", "f1", "brier", "coefs"):
            assert key in trained_result, f"Missing key: {key}"

    def test_model_name(self, trained_result):
        assert trained_result["model"] == "Logistic regression"

    def test_coefs_cover_all_features(self, trained_result):
        assert set(trained_result["coefs"].keys()) == set(FEATURE_COLS)


class TestPredictionRange:
    def test_auc_in_valid_range(self, trained_result):
        assert 0.0 <= trained_result["auc"] <= 1.0

    def test_accuracy_in_valid_range(self, trained_result):
        assert 0.0 <= trained_result["accuracy"] <= 1.0

    def test_brier_in_valid_range(self, trained_result):
        # Brier score is in [0, 1]; a score of 1 would mean perfectly wrong
        assert 0.0 <= trained_result["brier"] <= 1.0

    def test_precision_recall_f1_in_valid_range(self, trained_result):
        for metric in ("precision", "recall", "f1"):
            val = trained_result[metric]
            assert 0.0 <= val <= 1.0, f"{metric} out of range: {val}"

    def test_predictions_above_chance(self, trained_result):
        # A trained logistic model on mean-reverting data should beat random (AUC > 0.45)
        assert trained_result["auc"] > 0.45

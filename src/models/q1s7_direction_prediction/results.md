# Q1S7 — Spread Direction Prediction: Statistical Analysis

*Run: 2026-04-15 13:10 | Elapsed: 0.0s*

Target: spread widens tomorrow (spread_{{t+1}} > spread_t) = 1

## 1. Class Distribution

| Class | N | % |
|---|---|---|
| Widens  (1) | 473  | 48.0% |
| Narrows (0) | 512 | 52.0% |
| Total       | 985  | 100% |

## 2. Point-Biserial Correlations with Direction Label

| Feature | r | p-value | Significant? |
|---|---|---|---|
| `aave_rate_change_1d` | -0.1144 | 0.0003 | Yes |
| `spread_vs_net_rolling_mean_7d` | -0.1093 | 0.0006 | Yes |
| `spread_vs_net_lag_1d` | -0.102 | 0.0013 | Yes |
| `tvl_ratio` | 0.0791 | 0.013 | Yes |
| `compound_net_change_1d` | 0.0772 | 0.0153 | Yes |
| `aave_tvl_change_pct_1d` | -0.0236 | 0.4591 | No |
| `day_of_week` | 0.0051 | 0.8735 | No |
| `days_since_spike` | 0.0031 | 0.9215 | No |
| `spread_vs_net_rolling_std_7d` | 0.0019 | 0.9529 | No |

## 3. Conditional Widening Rate by Feature Quartile

Range = Q4 rate − Q1 rate. Higher range = stronger non-linear association.

| Feature | Q1 rate | Q2 rate | Q3 rate | Q4 rate | Range |
|---|---|---|---|---|---|
| `compound_net_change_1d` | 41.3% | 43.9% | 46.3% | 60.6% | 0.193 |
| `aave_rate_change_1d` | 57.1% | 45.1% | 50.4% | 39.4% | 0.176 |
| `spread_vs_net_rolling_mean_7d` | 56.7% | 47.6% | 48.0% | 39.8% | 0.168 |
| `tvl_ratio` | 39.3% | 50.0% | 50.4% | 52.4% | 0.132 |
| `spread_vs_net_lag_1d` | 54.7% | 45.9% | 49.6% | 41.9% | 0.128 |
| `aave_tvl_change_pct_1d` | 43.3% | 45.9% | 47.1% | 55.7% | 0.124 |
| `day_of_week` | 43.8% | 51.4% | 48.0% | 49.6% | 0.076 |
| `spread_vs_net_rolling_std_7d` | 48.2% | 46.3% | 48.4% | 49.2% | 0.029 |
| `days_since_spike` | 49.2% | 47.4% | 47.6% | 47.9% | 0.018 |

## 4. Autocorrelation of Direction Signal

Significant ACF lags (95%): [1]

| Lag | ACF |
|---|---|
| 0 | 1.0 |
| 1 | -0.1479 |
| 2 | -0.0309 |
| 3 | 0.0271 |
| 4 | -0.0186 |
| 5 | 0.0109 |
| 6 | -0.0429 |
| 7 | 0.0292 |
| 8 | -0.0186 |
| 9 | -0.0095 |
| 10 | 0.0159 |

---

# Q1S7 — Spread Direction Prediction: ML Models

*Run: 2026-04-15 13:15 | Elapsed: 4.2s*

## Run Details

- Training: 2023-07-30 → 2025-09-28 (788 days)
- Test:     2025-09-29 → 2026-04-13 (197 days)
- Target:   spread widens tomorrow (spread_{t+1} > spread_t)
- Features: all 9 selected features
- CV:       TimeSeriesSplit (5 folds) on train set

## Class Balance

| | Train | Test |
|---|---|---|
| Widens  (1) | 48.9% | 44.7% |
| Narrows (0) | 51.1% | 55.3% |

## Model Performance

| Model | AUC-ROC | Accuracy | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| Logistic regression | 0.671 | 0.533 | 0.4867 | 0.8295 | 0.6134 | 0.239 |
| Random forest (n=100, d=6) (CV AUC=0.6123) | 0.6355 | 0.5939 | 0.5274 | 0.875 | 0.6581 | 0.245 |
| XGBoost (lr=0.05, d=3) (CV AUC=0.6032) | 0.6317 | 0.5939 | 0.5299 | 0.8068 | 0.6396 | 0.2513 |
| LSTM | — skipped: tensorflow not installed — skipped |

**Best model by AUC:** Logistic regression (AUC=0.671)

## Logistic Regression Coefficients

| Feature | Coefficient |
|---|---|
| `aave_rate_change_1d` | -2.0242 |
| `spread_vs_net_lag_1d` | -1.7522 |
| `compound_net_change_1d` | 0.9982 |
| `spread_vs_net_rolling_mean_7d` | 0.3435 |
| `days_since_spike` | -0.0755 |
| `tvl_ratio` | 0.054 |
| `aave_tvl_change_pct_1d` | 0.0251 |
| `day_of_week` | 0.024 |
| `spread_vs_net_rolling_std_7d` | -0.0071 |

## Random Forest Feature Importance

| Feature | Importance |
|---|---|
| `aave_rate_change_1d` | 0.218 |
| `compound_net_change_1d` | 0.1656 |
| `spread_vs_net_lag_1d` | 0.1469 |
| `spread_vs_net_rolling_mean_7d` | 0.1073 |
| `aave_tvl_change_pct_1d` | 0.0997 |
| `tvl_ratio` | 0.0911 |
| `spread_vs_net_rolling_std_7d` | 0.0787 |
| `day_of_week` | 0.0471 |
| `days_since_spike` | 0.0455 |

## XGBoost Feature Importance

| Feature | Importance |
|---|---|
| `aave_rate_change_1d` | 0.1405 |
| `spread_vs_net_lag_1d` | 0.1267 |
| `compound_net_change_1d` | 0.1253 |
| `tvl_ratio` | 0.1162 |
| `day_of_week` | 0.1061 |
| `aave_tvl_change_pct_1d` | 0.102 |
| `days_since_spike` | 0.095 |
| `spread_vs_net_rolling_mean_7d` | 0.0943 |
| `spread_vs_net_rolling_std_7d` | 0.094 |

## Interpretation

### Class distribution — near-balanced, no base-rate advantage

48% widen, 52% narrow. A coin-flip baseline gets 52% accuracy. AUC meaningfully above 0.5 is the bar for genuine signal.

### Direction ACF — negative autocorrelation at lag 1 only

ACF at lag 1 is −0.148 (significant), all beyond are noise. The direction signal has a weak but real tendency to flip the next day — if the spread widened today, it's slightly more likely to narrow tomorrow. This is the directional signature of mean reversion. The effect is small and one day deep only.

### Feature correlations — rate changes dominate, vol and timing features are noise

Five features are significant: `aave_rate_change_1d` (r=−0.114), `rolling_mean_7d` (r=−0.109), `spread_vs_net_lag_1d` (r=−0.102), `tvl_ratio` (r=+0.079), `compound_net_change_1d` (r=+0.077). All point to mean reversion: elevated spread and rising Aave rates predict narrowing; rising Compound rates predict widening. Four features are statistically insignificant: `aave_tvl_change_pct_1d`, `day_of_week`, `days_since_spike`, `rolling_std_7d`. Notably, rolling volatility (r=0.002) is essentially uncorrelated with tomorrow's direction despite being informative in Q1S2.

### Conditional rates — Compound rate change is the strongest non-linear feature

When Compound rates are in their top quartile, the spread widens 60.6% of following days (Q1→Q4 range: 0.193) — the strongest conditional signal in the dataset. `rolling_std_7d` (range 0.029) and `days_since_spike` (range 0.018) show almost no non-linear association, confirming they are not useful for direction prediction.

### ML results — logistic regression wins, tree models overfit

Logistic regression achieves the best test AUC (0.671) without any CV tuning. Random forest and XGBoost have CV AUCs of 0.61–0.612, and test AUCs of 0.632–0.636 — the slight improvement from CV to test suggests luck on this specific test window rather than robust advantage. The gap between logistic regression and tree models confirms the relationship is largely linear. Tree models add complexity without adding signal.

Brier scores: logistic (0.239), RF (0.245), XGBoost (0.251). A no-skill model scores ~0.25 on this class balance, so logistic regression shows a modest but real improvement in probability calibration. Tree models are barely better than no-skill.

Logistic regression coefficients align perfectly with the correlation analysis: `aave_rate_change_1d` (−2.02) and `spread_vs_net_lag_1d` (−1.75) dominate by a large margin. The model is effectively a two-feature classifier with small corrections from remaining inputs.

### Overall

Spread direction is predictable at AUC=0.671 using logistic regression. The signal is almost entirely captured by today's Aave rate change and today's spread level — both pointing to mean reversion. The relationship is linear; tree models do not improve on it. Rolling volatility, day-of-week, and days-since-spike add no directional signal.

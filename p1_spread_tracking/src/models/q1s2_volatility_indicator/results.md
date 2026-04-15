# Q1S2 — Volatility as a Leading Indicator: Statistical Analysis

*Run: 2026-04-15 12:02 | Elapsed: 0.0s*

## 1. Volatility Regime Definition

High-vol threshold (75th percentile of `spread_vs_net_rolling_std_7d`): **1.8419%**

| Metric | High-vol | Low-vol |
|---|---|---|
| Days in regime | 290 | 868 |
| % of sample | 25.0% | 75.0% |
| Mean spread | 1.0941% | 0.0609% |
| Mean |spread| | 3.0961% | 0.8802% |
| Std spread | 5.1351% | 1.2107% |
| Median |spread| | 2.0088% | 0.629% |

## 2. Mann-Whitney U — Subsequent Spread Change

Test: does high-vol regime produce *more narrowing* (lower |spread change|) over the next 3 days?

| Metric | Value |
|---|---|
| High-vol mean change in |spread| | -0.536% |
| Low-vol mean change in |spread|  | 0.1774% |
| Mann-Whitney U statistic | 111734.0 |
| p-value (one-sided: high < low) | 0.0027 |
| Significant (p < 0.05) | Yes |
| N (high-vol / low-vol) | 290 / 865 |

## 3. Ornstein-Uhlenbeck Fit by Regime

| Regime | θ | Half-life (days) | μ |
|---|---|---|---|
| Full sample | 0.7308 | 0.95 | 0.3203% |
| High-vol    | 0.8251 | 0.84 | 1.0896% |
| Low-vol     | 0.3859  | 1.8  | 0.0617% |

---

# Q1S2 — Volatility as a Leading Indicator: ML Models

*Run: 2026-04-15 12:02 | Elapsed: 0.2s*

## Run Details

- Training: 2023-02-07 → 2025-08-22 (924 days)
- Test:     2025-08-23 → 2026-04-10 (231 days)
- Target:   binary — does |spread| narrow over the next 3 days?
- Features: `spread_vs_net_lag_1d`, `spread_vs_net_rolling_std_7d`, `spread_vs_net_rolling_mean_7d`, `aave_rate_change_1d`, `compound_net_change_1d`

## Class Balance

| | Train | Test |
|---|---|---|
| Narrows (1) | 51.6% | 52.8% |
| Widens  (0) | 48.4% | 47.2% |

## Classifier Performance

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| Threshold rule (vol > 75th pct) | 0.5031 | 0.6 | 0.0246 | 0.0472 |
| Logistic regression | 0.5712 | 0.7143 | 0.1639 | 0.2667 |
| XGBoost classifier | 0.6354 | 0.6923 | 0.2951 | 0.4138 |

**Best classifier by AUC:** XGBoost classifier (AUC=0.6354)

## Logistic Regression Coefficients

| Feature | Coefficient |
|---|---|
| `spread_vs_net_lag_1d` | 0.0846 |
| `spread_vs_net_rolling_std_7d` | 0.1038 |
| `spread_vs_net_rolling_mean_7d` | -0.0488 |
| `aave_rate_change_1d` | 0.0883 |
| `compound_net_change_1d` | 0.0218 |

## XGBoost Feature Importance

| Feature | Importance |
|---|---|
| `spread_vs_net_lag_1d` | 0.2096 |
| `aave_rate_change_1d` | 0.2074 |
| `compound_net_change_1d` | 0.2028 |
| `spread_vs_net_rolling_std_7d` | 0.1968 |
| `spread_vs_net_rolling_mean_7d` | 0.1834 |

## Regression Variant — Spread Change Magnitude

| Metric | Value |
|---|---|
| MAE | 0.6795% |
| Direction accuracy | 50.6% |

| Feature | Coefficient |
|---|---|
| `spread_vs_net_lag_1d` | -0.3788 |
| `spread_vs_net_rolling_std_7d` | -0.3286 |
| `spread_vs_net_rolling_mean_7d` | 0.3194 |
| `aave_rate_change_1d` | -0.4621 |
| `compound_net_change_1d` | 0.0843 |
| intercept | 0.5703 |

## Interpretation

### Regime characterisation — high-vol periods are a different world

The 25% of days in the high-vol regime (rolling_std_7d > 1.84%) look nothing like normal days. Mean |spread| is 3.1% vs 0.88% in low-vol — 3.5× larger. Std spread is 5.1% vs 1.2% — 4× more dispersed. Mean spread is +1.09% vs +0.06% — Aave is substantially more expensive during high-vol periods. Rolling volatility is tracking real spread episodes, not noise.

### Mann-Whitney U — high-vol leads to more narrowing

Statistically significant (p=0.003). In high-vol periods the mean change in |spread| over the next 3 days is −0.54% (narrowing); in low-vol periods it is +0.18% (widening). When volatility is elevated, the spread is more likely to close over the following 3 days. This is economically intuitive: high volatility reflects a spread already far from equilibrium, and mean-reversion pulls it back. In low-vol periods the spread is near zero with no strong reversion force.

### OU by regime — mean reversion is faster when volatility is high

High-vol half-life is 0.84 days vs 1.8 days in low-vol. When the spread is large, arbitrage pressure is stronger and closes it faster. The full-sample 0.95-day half-life is a blend of these two regimes, skewed by the more common low-vol state.

### Class balance — not a minority class problem

The target narrows on 51.6% of training days and 52.8% of test days — nearly perfectly balanced. The concern about spike periods being the minority class does not apply here.

### ML results — weak but real predictability

The threshold rule completely fails (AUC=0.503). Simply flagging high-vol days as "will narrow" is not predictive because vol alone does not distinguish between "about to revert" and "still spiking." XGBoost at AUC=0.635 shows genuine signal beyond the threshold rule. The logistic regression coefficients are all small and similar in magnitude — no single feature dominates, explaining why the threshold rule fails.

XGBoost feature importances are nearly equal across all five features (0.18–0.21 each). The signal is non-linear and depends on feature interactions rather than any individual predictor.

The regression variant (magnitude of change) is useless — 50.6% direction accuracy is a coin flip. Predicting how much the spread will change is harder than predicting which direction it moves.

### Overall

Elevated volatility is a statistically significant leading indicator of spread narrowing, but translating this into predictions is difficult. The threshold rule fails. XGBoost captures some non-linear signal (AUC=0.635) but the effect is dispersed across all features with no single dominant predictor. Spread direction is somewhat predictable when volatility is elevated; magnitude is not.

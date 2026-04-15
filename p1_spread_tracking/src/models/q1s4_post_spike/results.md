# Q1S4 — Post-Spike Spread Behaviour: Statistical Analysis

*Run: 2026-04-15 12:19 | Elapsed: 0.0s*

## 1. Regime Segmentation

Spike window: `days_since_spike` ≤ 7 | Elevated threshold: |spread| > 1.0%

| Metric | Spike window | Steady-state |
|---|---|---|
| Days in regime | 264 | 721 |
| % of sample | 26.8% | 73.2% |
| Mean spread | 0.9863% | 0.3185% |
| Mean |spread| | 3.0702% | 0.9845% |
| Std spread | 5.2751% | 1.371% |
| Median |spread| | 1.9907% | 0.7129% |
| % days |spread| > 1.0% | 73.1% | 32.9% |

## 2. Ornstein-Uhlenbeck Fit by Regime

| Regime | θ | Half-life (days) | μ | R² |
|---|---|---|---|---|
| Full sample  | 0.7541  | 0.92  | 0.425%  | 0.4824 |
| Spike window | 0.7923 | 0.87 | 0.7306% | 0.5246 |
| Steady-state | 0.5542 | 1.25 | 0.316% | 0.2772 |

## 3. Mann-Whitney U — Spike vs Steady-State |Spread|

| Metric | Value |
|---|---|
| Spike window mean |spread|  | 3.0702% |
| Steady-state mean |spread| | 0.9845% |
| Mann-Whitney U statistic | 144823.0 |
| p-value (one-sided: spike > steady) | 0.0 |
| Significant (p < 0.05) | Yes ✓ |

## 4. Post-Spike Survival — Days Until |Spread| < 1%

| Metric | Value |
|---|---|
| Spike episodes detected | 146 |
| Mean days to close | 3.05 |
| Median days to close | 2.0 |
| % closed within 3 days | 68.5% |
| % closed within 7 days | 92.5% |
| Max days to close | 24 |

---

# Q1S4 — Post-Spike Spread Behaviour: ML Models

*Run: 2026-04-15 12:19 | Elapsed: 0.4s*

## Run Details

- Training: 2023-07-30 → 2025-09-27 (787 days)
- Test:     2025-09-28 → 2026-04-12 (197 days)
- Target:   next-day spread change (spread_{t+1} − spread_t)

## Model Performance

| Model | MAE | RMSE | R² | Direction Acc |
|---|---|---|---|---|
| Linear regression (with lag×days_since_spike) | 0.5624 | 0.982 | 0.0043 | 53.3% |
| Random forest | 0.4926 | 0.9537 | 0.0609 | 42.6% |

**Best model by MAE:** Random forest (MAE=0.4926)

## Interaction Effect — Slope of Lag on Spread Change by Regime

A more negative slope means stronger mean reversion in that regime.

| Regime | Slope (spread_change ~ spread_lag) |
|---|---|
| Spike window  | -0.1123 |
| Steady-state  | -0.0753 |

## Linear Regression Coefficients

| Feature | Coefficient |
|---|---|
| `spread_vs_net_lag_1d` | 0.024922 |
| `days_since_spike` | -0.004994 |
| `spread_vs_net_rolling_std_7d` | -0.150133 |
| `spread_vs_net_rolling_mean_7d` | -0.419894 |
| `lag_x_dss` | 0.000149 |
| intercept | 0.680884 |

## Random Forest Feature Importance

| Feature | Importance |
|---|---|
| `spread_vs_net_rolling_mean_7d` | 0.3616 |
| `spread_vs_net_rolling_std_7d` | 0.3151 |
| `days_since_spike` | 0.1472 |
| `spread_vs_net_lag_1d` | 0.1273 |
| `lag_x_dss` | 0.0487 |

## Interpretation

### Regime characterisation — spike window is a different distribution

The spike window (27% of days) has a mean |spread| of 3.07% vs 0.98% in steady-state — 3× larger. Spread exceeds 1% on 73% of spike-window days vs 33% of steady-state days. `days_since_spike` is cleanly tracking genuine spread episodes.

### OU fit by regime — faster reversion in spike window, not slower

Half-life is 0.87 days in the spike window vs 1.25 days in steady-state. This directly contradicts the hypothesis. Spreads close *faster* in the days following a spike, not more slowly. When the spread is large, arbitrage pressure is proportionally stronger — θ is higher when the spread is far from its mean. The R² difference reinforces this: 0.52 in the spike window vs 0.28 in steady-state, meaning mean reversion is a more reliable description of behaviour during spike periods than during calm ones.

### Survival analysis — most spikes close within 3 days

146 spike episodes detected. 68.5% close within 3 days and 92.5% within 7 days. Median closure is 2 days. The maximum of 24 days corresponds to the extreme tail events seen in the Q1 distribution analysis. The typical spike is short-lived; only a small minority persist beyond a week.

### ML results — interaction term is negligible

Linear regression R²=0.004 is essentially zero. Random forest reaches R²=0.061 — marginal improvement suggesting mild non-linear patterns that are not robust. The interaction term coefficient (`lag_x_dss` = 0.000149) is negligible: the linear model finds almost no evidence that the lag-to-change relationship is moderated by days since spike. The interaction effect exists statistically (spike slope −0.11 vs steady slope −0.08) but is too small to carry predictive weight. Random forest importance is dominated by `rolling_mean_7d` (0.36) and `rolling_std_7d` (0.32), with `days_since_spike` contributing 0.15.

### Overall

The hypothesis that spreads close more slowly post-spike is wrong — they close faster. Arbitrage force is proportional to spread size. 92.5% of spike episodes resolve within 7 days, and the typical episode closes in 2 days. `days_since_spike` adds modest information to models but is not a strong standalone predictor.

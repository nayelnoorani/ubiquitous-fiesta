# Q1S1 — Spread Persistence: Statistical Analysis

*Run: 2026-04-15 11:24 | Elapsed: 0.0s*

## 1. ACF / PACF

Significant ACF lags (95%): [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]
Significant PACF lags (95%): [1, 2, 3, 5, 7, 11]
Suggested AR order (max significant PACF lag): **11**

| Lag | ACF | PACF |
|---|---|---|
| 1 | 0.2692 | 0.2695 |
| 2 | 0.1585 | 0.0929 |
| 3 | 0.1764 | 0.1229 |
| 4 | 0.0796 | -0.0049 |
| 5 | 0.1034 | 0.0615 |
| 6 | 0.1124 | 0.0549 |
| 7 | 0.1336 | 0.0832 |
| 8 | 0.0378 | -0.0487 |
| 9 | 0.073 | 0.0395 |
| 10 | 0.0913 | 0.0402 |

## 2. Ljung-Box Test

| Lags | Statistic | p-value | Autocorrelation? |
|---|---|---|---|
| 1 | 84.2391 | 0.0 | Yes |
| 5 | 169.545 | 0.0 | Yes |
| 10 | 222.8262 | 0.0 | Yes |
| 20 | 342.26 | 0.0 | Yes |

## 3. Ornstein-Uhlenbeck Fit

| Parameter | Value | Interpretation |
|---|---|---|
| θ (mean-reversion speed) | 0.730745 | higher = faster reversion |
| μ (long-run mean spread) | 0.3197% | spread reverts toward this level |
| σ (volatility) | 2.7044% | daily noise in spread process |
| Half-life | 0.95 days | time for shock to decay by 50% |
| R² | 0.3654 | |
| p-value (β) | 0.0 | significance of mean-reversion term |

---

# Q1S1 — Spread Persistence: ML Models

*Run: 2026-04-15 11:24 | Elapsed: 0.0s*

## Run Details

- Training: 2023-02-07 → 2025-08-23 (925 days)
- Test:     2025-08-24 → 2026-04-12 (232 days)
- Target:   next-day `spread_vs_net`
- Features: `spread_vs_net_lag_1d`, `spread_vs_net_rolling_mean_7d`

## Model Performance

| Model | MAE | RMSE | R² | Direction Acc |
|---|---|---|---|---|
| Naive persistence | 0.6576 | 1.1911 | -0.6034 | N/A |
| AR(10) — AIC=1995.8 | 0.5747 | 0.9104 | 0.0641 | 58.2% |
| Linear regression | 0.6081 | 0.935 | 0.012 | 62.5% |

**Best model by MAE:** AR(10) — AIC=1995.8 (MAE=0.5747)

## AR Model

- Selected order: p=10 (AIC=1995.78)

## Linear Regression Coefficients

| Feature | Coefficient |
|---|---|
| `spread_vs_net_lag_1d` | 0.012888 |
| `spread_vs_net_rolling_mean_7d` | 0.487419 |
| intercept | 0.203485 |

## Interpretation

### ACF / PACF — autocorrelation exists, but it is small

The ACF is significant at 16 of the first 20 lags. The magnitudes are modest: lag 1 is only 0.27 and decays quickly. The PACF has meaningful values at lags 1, 2, 3, 5, 7, and 11 — scattered rather than consecutive, suggesting a weak multi-lag AR structure rather than a clean AR(1) process.

### Ljung-Box — autocorrelation is statistically real

p=0.0 at every lag tested. The spread is not white noise — there is genuine structure in its autocorrelation. Statistical significance and practical usefulness are different things, however, as the ML results show.

### Ornstein-Uhlenbeck — half-life under 1 day, but volatility swamps mean reversion

θ=0.73 gives a half-life of 0.95 days — spread shocks decay by 50% within a single trading day on average. But σ=2.7% means daily noise is roughly 3× the long-run mean level (0.32%). New shocks arrive faster than old ones close. This is why the ACF is weak: mean reversion is real but constantly overwhelmed by fresh volatility.

### ML results — yesterday barely predicts tomorrow

Naive persistence has R²=−0.60 — worse than predicting the mean every day. Carrying forward yesterday's value is actively misleading because the spread reverts so quickly. AR(10) is the best model but explains only 6.4% of next-day variance (R²=0.064). Linear regression explains 1.2%.

The linear regression coefficient on `spread_vs_net_lag_1d` is near-zero (0.013) — the lag carries almost no direct signal. The `rolling_mean_7d` coefficient (0.487) does the work: the model is predicting that tomorrow's spread will be close to the recent 7-day average, which is a smoothed mean-reversion signal consistent with the regime structure found in the trend-stationarity exploration.

Direction accuracy of 58–62% confirms there is weak directional signal, but it is not strong enough to rely on operationally.

### Overall

Today's spread level has almost no predictive power for tomorrow's level. The high daily volatility (σ=2.7%) means most of tomorrow's spread is determined by new shocks unknowable today. The 7-day rolling mean outperforms the 1-day lag because it proxies the current regime mean rather than the noisy prior day value.

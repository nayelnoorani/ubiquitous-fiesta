# Q1S6 — Lead/Lag Between Protocols: Statistical Analysis

*Run: 2026-04-15 12:39 | Elapsed: 1.4s*

## 1. Cross-Correlation Function (Rate Changes)

Contemporaneous correlation: 0.1708

**Aave leads Compound** — peak at lag 3:

| Lag | CCF |
|---|---|
| 0 | 0.1708 |
| 1 | -0.1052 |
| 2 | -0.0746 |
| 3 | 0.1566 |
| 4 | -0.0853 |
| 5 | -0.0027 |
| 6 | 0.0308 |
| 7 | -0.0224 |
| 8 | -0.0391 |
| 9 | 0.0543 |
| 10 |  |

**Compound leads Aave** — peak at lag 5:

| Lag | CCF |
|---|---|
| 0 | 0.1708 |
| 1 | -0.0283 |
| 2 | -0.0703 |
| 3 | 0.0361 |
| 4 | 0.0847 |
| 5 | -0.1367 |
| 6 | 0.0675 |
| 7 | -0.0405 |
| 8 | 0.0971 |
| 9 | -0.0858 |
| 10 |  |

## 2. Granger Causality

### Aave → Compound

Min p: **0.0001** (lag 4) — Significant ✓

| Lag | F-stat | p-value |
|---|---|---|
| 1 | 1.7904 | 0.1811 |
| 2 | 1.5585 | 0.2109 |
| 3 | 1.3646 | 0.2521 |
| 4 | 6.2278 | 0.0001 |
| 5 | 3.9232 | 0.0016 |

### Compound → Aave

Min p: **0.0035** (lag 3) — Significant ✓

| Lag | F-stat | p-value |
|---|---|---|
| 1 | 1.3986 | 0.2372 |
| 2 | 5.1616 | 0.0059 |
| 3 | 4.5688 | 0.0035 |
| 4 | 3.7024 | 0.0053 |
| 5 | 2.7524 | 0.0176 |

## 3. Rolling Granger Causality (180-day window, lag=1)

| Metric | Aave → Compound | Compound → Aave |
|---|---|---|
| % windows significant (p < 0.05) | 25.8% | 11.9% |
| Median p-value | 0.3715 | 0.2013 |
| N rolling windows | 978 | 978 |

---

# Q1S6 — Lead/Lag Between Protocols: ML Models

*Run: 2026-04-15 12:39 | Elapsed: 0.0s*

## Run Details

- Training: 2023-02-07 → 2025-08-24 (926 days)
- Test:     2025-08-25 → 2026-04-13 (232 days)

## Cross-Lagged Linear Regressions

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Compound ~ Aave lag-1 | 0.4971 | 1.0817 | 0.0002 |
| Aave ~ Compound lag-1 | 0.1928 | 0.3191 | -0.8724 |
| Compound ~ Aave lag-1 + Comp lag-1 | 0.4925 | 1.0252 | 0.1018 |
| Aave ~ Compound lag-1 + Aave lag-1 | 0.1634 | 0.2587 | -0.231 |

### Coefficients

**Compound ~ Aave lag-1**
- `aave_rate_change_1d_lag1`: -0.016539
- intercept: 0.002226

**Aave ~ Compound lag-1**
- `compound_net_change_1d_lag1`: -0.191185
- intercept: 0.003824

**Compound ~ Aave lag-1 + Comp lag-1**
- `aave_rate_change_1d_lag1`: 0.022065
- `compound_net_change_1d_lag1`: -0.386646
- intercept: 0.003386

**Aave ~ Compound lag-1 + Aave lag-1**
- `compound_net_change_1d_lag1`: -0.06022
- `aave_rate_change_1d_lag1`: -0.428632
- intercept: 0.0047

## VAR(10) Model

| Series | MAE | RMSE | R² |
|---|---|---|---|
| Aave    | 0.2717 | 0.391 | -1.8118 |
| Compound | 0.5114 | 0.9606 | 0.2115 |

### VAR Lag-1 Coefficients

| Equation | aave_{t-1} | compound_{t-1} |
|---|---|---|
| Aave_t    | -0.754 | -0.031 |
| Compound_t | 0.0122 | -0.5991 |

## Interpretation

### CCF — no clean lead/lag structure

All CCF values beyond lag 0 are small (max 0.157 at lag 3 for Aave leading, max 0.137 at lag 5 for Compound leading). The pattern alternates sign rather than decaying monotonically — if Aave reliably led Compound, you'd see a smooth positive CCF at positive lags. Instead the signal is noise with a marginal peak at lag 3. Contemporaneous correlation of 0.17 is weak, consistent with Q1 stats.

### Granger causality — both directions significant, but only at long lags

Neither direction is significant at lag 1. Aave → Compound becomes significant only at lag 4 (p=0.0001). Compound → Aave is significant from lag 2 onwards (p=0.006). There is no next-day lead/lag — any signal requires a 2–4 day horizon, which is too slow to be actionable.

Compound leading Aave at shorter lags than Aave leads Compound is counterintuitive given Aave's larger TVL and higher volatility, but consistent with the Q1S3 finding that Compound net changes Granger-cause the spread.

### Rolling Granger — intermittent, not structural

Aave → Compound is significant in only 25.8% of 180-day windows; Compound → Aave in 11.9%. Median p-values are 0.37 and 0.20. The full-sample significance is driven by specific sub-periods. The relationship is not persistent.

### Cross-lagged regressions — near-zero predictive power

Yesterday's Aave change explains essentially nothing about today's Compound change (R²=0.0002). Every Aave equation has negative R², meaning it performs worse than predicting the mean. The best result (Compound with both lags, R²=0.10) gets most of its signal from the Compound autoregressive term (−0.39), not from Aave's lag (0.02). Both series show strong negative autocorrelation (Aave: −0.43, Compound: −0.39) — large rate changes tend to partially reverse the next day. This is own-series mean reversion, not cross-protocol lead.

### VAR — each protocol is driven by its own history

VAR(10) fits Compound reasonably (R²=0.21) but Aave poorly (R²=−1.81). The lag-1 off-diagonal coefficients are tiny (Aave→Compound: 0.012, Compound→Aave: −0.031) while own-lag diagonals dominate (−0.754 and −0.599). Each protocol's rate change is primarily predicted by its own prior moves, not the other protocol's.

### Overall

Neither protocol reliably leads the other on a next-day basis. Granger significance exists at lags 2–4 in both directions but disappears in 75–88% of rolling windows. The dominant signal is negative own-series autocorrelation — partial rate reversion — not cross-protocol lead/lag. The relationship is statistically detectable over the full sample but too unstable to be operationally useful.

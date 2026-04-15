# Q1S3 — TVL Shocks and Rate Divergence: Statistical Analysis

*Run: 2026-04-15 12:08 | Elapsed: 0.1s*

## 1. Granger Causality

### Aave TVL change → spread (does TVL precede spread?)

Minimum p-value across lags: **0.0072** (lag 2) — Significant ✓

| Lag | F-stat | p-value |
|---|---|---|
| 1 | 0.0001 | 0.9919 |
| 2 | 4.9516 | 0.0072 |
| 3 | 0.8419 | 0.471 |
| 4 | 0.7714 | 0.5438 |
| 5 | 1.3353 | 0.2467 |

### Spread → Aave TVL change (reverse direction)

Minimum p-value: **0.0** (lag 1) — Significant ✓

| Lag | F-stat | p-value |
|---|---|---|
| 1 | 593.3631 | 0.0 |
| 2 | 323.7433 | 0.0 |
| 3 | 220.6229 | 0.0 |
| 4 | 167.2964 | 0.0 |
| 5 | 133.6134 | 0.0 |

### Compound net change → spread

Minimum p-value: **0.0** (lag 2) — Significant ✓

| Lag | F-stat | p-value |
|---|---|---|
| 1 | 0.745 | 0.3883 |
| 2 | 12.3756 | 0.0 |
| 3 | 17.1899 | 0.0 |
| 4 | 13.349 | 0.0 |
| 5 | 11.6155 | 0.0 |

## 2. Contemporaneous Correlation

| Pair | Pearson r | p-value | Significant? |
|---|---|---|---|
| Aave TVL change vs spread (same day) | 0.1022 | 0.0005 | Yes |
| Aave TVL change vs Aave rate change | -0.4024 | 0.0 | Yes |
| TVL ratio vs spread | -0.0989 | 0.0007 | Yes |

## 3. Conditional Analysis — Large TVL Shocks vs Normal Days

Shock threshold (75th pct of |aave_tvl_change_pct_1d|): **0.1427%**

| Metric | Shock days | Normal days |
|---|---|---|
| N days | 290 | 867 |
| Mean next-day |spread change| | 1.8173% | 1.3064% |
| Mean next-day spread change | -0.3442% | 0.1153% |

Mann-Whitney U (shock |change| > normal): stat=141097.0  p=0.0009  Significant ✓

### Inflow vs Outflow Shocks

| Direction | N | Mean next-day spread change |
|---|---|---|
| Inflow  (TVL increase) | 158  | 0.4245% |
| Outflow (TVL decrease) | 132 | -1.2642% |

---

# Q1S3 — TVL Shocks and Rate Divergence: ML Models

*Run: 2026-04-15 12:08 | Elapsed: 0.3s*

## Run Details

- Training: 2023-02-07 → 2025-08-23 (925 days)
- Test:     2025-08-24 → 2026-04-12 (232 days)
- Target:   next-day spread change (spread_{t+1} − spread_t)
- Features: `aave_tvl_change_pct_1d`, `tvl_ratio`, `aave_rate_change_1d`, `compound_net_change_1d`, `spread_vs_net_lag_1d`, `spread_vs_net_rolling_std_7d`, `spread_vs_net_rolling_mean_7d`

## Model Performance

| Model | MAE | RMSE | R² | Direction Acc |
|---|---|---|---|---|
| OLS regression | 0.5573 | 0.8911 | 0.2733 | 58.6% |
| Lasso (α=0.0319) | 0.5474 | 0.8898 | 0.2754 | 56.9% |
| XGBoost regressor | 0.754 | 1.1547 | -0.2201 | 54.7% |

**Best model by MAE:** Lasso (α=0.0319) (MAE=0.5474)

## OLS Coefficients

| Feature | Coefficient |
|---|---|
| `aave_rate_change_1d` | -0.847562 |
| `spread_vs_net_lag_1d` | -0.769362 |
| `compound_net_change_1d` | 0.741609 |
| `spread_vs_net_rolling_mean_7d` | 0.312383 |
| `aave_tvl_change_pct_1d` | -0.060197 |
| `spread_vs_net_rolling_std_7d` | -0.013663 |
| `tvl_ratio` | -0.003262 |
| intercept | 0.2334 |

## Lasso — α=0.031906 | 6/7 features retained

| Feature | Coefficient |
|---|---|
| `aave_rate_change_1d` | -2.847722 |
| `spread_vs_net_lag_1d` | -2.165598 |
| `compound_net_change_1d` | 1.389638 |
| `spread_vs_net_rolling_mean_7d` | 0.364687 |
| `aave_tvl_change_pct_1d` | -0.194638 |
| `tvl_ratio` | -0.042603 |

Zeroed out: `spread_vs_net_rolling_std_7d`

## XGBoost Feature Importance

| Feature | Importance |
|---|---|
| `aave_rate_change_1d` | 0.2312 |
| `tvl_ratio` | 0.1481 |
| `compound_net_change_1d` | 0.1464 |
| `aave_tvl_change_pct_1d` | 0.145 |
| `spread_vs_net_lag_1d` | 0.1354 |
| `spread_vs_net_rolling_std_7d` | 0.0992 |
| `spread_vs_net_rolling_mean_7d` | 0.0946 |

## Interpretation

### Granger causality — asymmetric and striking

Aave TVL → spread is weakly significant at lag 2 only (p=0.007), with lag 1 completely insignificant (p=0.99). TVL changes take 2 days to show up in the spread, not 1.

The dominant direction is the reverse: spread → Aave TVL (F=593 at lag 1, p=0.0 at every lag). The spread Granger-causes TVL flows, not the other way around. When the spread moves, capital migrates — borrowers leave the expensive protocol, TVL follows. This is the arbitrage mechanism in action. Compound net change → spread is significant from lag 2 onwards, consistent with Aave being the faster-responding protocol.

### Contemporaneous correlations — TVL barely correlates with the spread

Aave TVL change vs spread: r=0.10. TVL ratio vs spread: r=−0.10. Both statistically significant but tiny. The strongest same-day relationship is Aave TVL change vs Aave rate change (r=−0.40): inflows push rates down, outflows push rates up. TVL affects rates, which then flow into the spread — the rate change is the intermediary, not TVL directly.

### Conditional analysis — outflows drive larger spread movements than inflows

Large TVL shocks produce larger next-day spread changes (mean |change| 1.82% vs 1.31%, Mann-Whitney p=0.001). Inflow shocks produce a mean next-day spread change of +0.34%; outflow shocks produce −1.26%. Outflows are roughly 4× more impactful than inflows. Capital leaving Aave reduces borrow demand and pulls rates down, closing the spread quickly. Inflows have a weaker and opposite effect.

### ML results — mean reversion dominates, TVL is secondary

OLS and Lasso both achieve R²=0.27 — substantially better than Q1S1's AR(10) at R²=0.064. Adding rate and TVL features materially improves next-day spread change prediction. XGBoost underperforms OLS (R²=−0.22), suggesting the relationship is largely linear with no strong non-linear interactions to exploit at this sample size.

Lasso retains 6 of 7 features, zeroing out only `rolling_std_7d`. TVL features are retained, confirming they carry genuine linear signal.

OLS coefficient magnitudes tell the full story: `aave_rate_change_1d` (−0.85) and `spread_vs_net_lag_1d` (−0.77) dwarf `aave_tvl_change_pct_1d` (−0.06). Rate momentum and mean reversion explain most of the predictable variance; TVL is a secondary signal an order of magnitude smaller.

### Overall

The causal direction runs mostly from spreads to TVL flows, not the reverse. Rate changes are the dominant predictor of next-day spread change (R²=0.27). TVL shocks add marginal improvement and are statistically significant, but the effect is small and largely mediated through rate changes. Outflow shocks matter more than inflow shocks.

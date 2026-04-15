# Feature Engineering

## Context

Features are engineered for **Question 1**: how closely do Aave V3 and Compound V3 USDC borrow rates track each other, and how long do spreads persist before arbitrage closes them?

Source columns: `apyBase`, `apyReward`, `tvlUsd`, `timestamp` for each protocol (daily cadence).

---

## Note on dual spread definitions

All spread and rate-momentum features are computed in two variants:

- **`_vs_net`** — uses Compound's net borrow cost (`apyBase − apyReward`), reflecting what a borrower actually pays after COMP rewards.
- **`_vs_base`** — uses Compound's raw `apyBase`, ignoring rewards.

Both are retained because it is currently unknown whether Compound's `apyReward` is stable over time. If rewards are volatile, `spread_vs_net` and `spread_vs_base` will diverge meaningfully and carry different information — in which case both are useful. If rewards are roughly constant, the two spread series will be near-identical (offset by a fixed constant) and one can be dropped at modelling time based on feature importance. We keep both now and let the data decide.

---

## Features

### Spread features

| Feature | Definition | Rationale |
|---|---|---|
| `spread_vs_net` | Aave `apyBase` − Compound (`apyBase` − `apyReward`) | Primary measure of effective borrowing cost divergence |
| `spread_vs_base` | Aave `apyBase` − Compound `apyBase` | Spread ignoring reward incentives |
| `spread_vs_net_lag_1d` | `spread_vs_net` shifted 1 day | Tests autocorrelation — do spreads persist day-to-day? |
| `spread_vs_base_lag_1d` | `spread_vs_base` shifted 1 day | Same, on raw base spread |
| `spread_vs_net_rolling_mean_7d` | 7-day rolling mean of `spread_vs_net` | Medium-term trend in effective-cost divergence |
| `spread_vs_base_rolling_mean_7d` | 7-day rolling mean of `spread_vs_base` | Same, on raw base spread |
| `spread_vs_net_rolling_std_7d` | 7-day rolling std of `spread_vs_net` | Local volatility of the spread — spikes in this signal high uncertainty |
| `spread_vs_base_rolling_std_7d` | 7-day rolling std of `spread_vs_base` | Same, on raw base spread |
| `spread_vs_net_zscore_30d` | (`spread_vs_net` − 30d mean) / 30d std | How extreme the current effective-cost divergence is relative to recent history |
| `spread_vs_base_zscore_30d` | (`spread_vs_base` − 30d mean) / 30d std | Same, on raw base spread |

### Rate momentum features

| Feature | Definition | Rationale |
|---|---|---|
| `aave_rate_change_1d` | Aave `apyBase` − Aave `apyBase` lag 1d | Is Aave's rate moving, and in which direction? |
| `compound_net_change_1d` | Compound net cost − Compound net cost lag 1d | Daily momentum in Compound's effective rate |
| `compound_base_change_1d` | Compound `apyBase` − Compound `apyBase` lag 1d | Same, ignoring reward component |
| `rate_divergence_direction_vs_net` | sign(`aave_rate_change_1d` − `compound_net_change_1d`) | Is the effective-cost spread widening (+1) or narrowing (−1)? |
| `rate_divergence_direction_vs_base` | sign(`aave_rate_change_1d` − `compound_base_change_1d`) | Same, on raw base rates |

### Reward component

| Feature | Definition | Rationale |
|---|---|---|
| `compound_apyReward` | Compound `apyReward` (raw) | The reward itself as a standalone feature; captures how much the incentive programme is distorting the base rate on any given day. Directly answers whether reward stability affects the spread |

### Liquidity features

| Feature | Definition | Rationale |
|---|---|---|
| `tvl_ratio` | Aave `tvlUsd` / Compound `tvlUsd` | Relative liquidity imbalance; large ratios may precede rate divergence |
| `aave_tvl_change_pct_1d` | % change in Aave `tvlUsd` day-over-day | Proxy for demand shocks — sudden inflows/outflows drive rate spikes |
| `compound_tvl_change_pct_1d` | % change in Compound `tvlUsd` day-over-day | Same for Compound |

### Regime features

| Feature | Definition | Rationale |
|---|---|---|
| `is_spike` | 1 if either protocol's `apyBase` > 10%, else 0 | Separates spike from steady-state regimes; spread dynamics likely differ between them |
| `days_since_spike` | Days elapsed since last `is_spike == 1` | Tests whether spreads take longer to normalise after extreme events |

### Calendar feature

| Feature | Definition | Rationale |
|---|---|---|
| `day_of_week` | Integer 0–6 (Monday = 0) | DeFi operates 24/7 but human-driven arbitrage is not uniform — weekend spreads may persist longer before being closed |

---

## Summary

| Group | Count |
|---|---|
| Spread | 10 |
| Rate momentum | 5 |
| Reward component | 1 |
| Liquidity | 3 |
| Regime | 2 |
| Calendar | 1 |
| **Total** | **22** |

---

## Feature Selection Results

Selection was run with `corr_threshold=0.95` and `variance_threshold_pct=0.01`.

### Dropped — high correlation (> 0.95)

| Dropped feature | Correlated with | \|r\| |
|---|---|---|
| `spread_vs_base` | `spread_vs_net` | 0.990 |
| `spread_vs_base_lag_1d` | `spread_vs_net_lag_1d` | 0.990 |
| `spread_vs_base_rolling_mean_7d` | `spread_vs_net_rolling_mean_7d` | 0.962 |
| `spread_vs_base_rolling_std_7d` | `spread_vs_net_rolling_std_7d` | 1.000 |
| `spread_vs_base_zscore_30d` | `spread_vs_net_zscore_30d` | 0.997 |
| `compound_base_change_1d` | `compound_net_change_1d` | 1.000 |
| `rate_divergence_direction_vs_base` | `rate_divergence_direction_vs_net` | 0.968 |

**Interpretation:** The `_vs_base` and `_vs_net` feature families are highly correlated across the board, confirming that `compound_apyReward` is sufficiently stable over the sample period that both spread definitions carry nearly identical information. Keeping both was the right call ex-ante (reward stability was unknown); the data now justifies dropping the `_vs_base` variants.

### Dropped — low variance (< 1% of mean variance, threshold = 1.70)

| Dropped feature | Variance |
|---|---|
| `spread_vs_net_zscore_30d` | 1.209 |
| `rate_divergence_direction_vs_net` | 1.000 |
| `compound_apyReward` | 0.184 |
| `compound_tvl_change_pct_1d` | 0.229 |
| `is_spike` | 0.094 |

**Interpretation:** `compound_apyReward` having low variance is consistent with the correlation finding — the reward is stable, so it adds little signal beyond a constant offset. `is_spike` is binary and rare (~12% of days), giving it low variance by construction; regime information is still captured indirectly through `days_since_spike`.

### Selected features (10)

| Feature | Group |
|---|---|
| `spread_vs_net` | Spread |
| `spread_vs_net_lag_1d` | Spread |
| `spread_vs_net_rolling_mean_7d` | Spread |
| `spread_vs_net_rolling_std_7d` | Spread |
| `aave_rate_change_1d` | Rate momentum |
| `compound_net_change_1d` | Rate momentum |
| `tvl_ratio` | Liquidity |
| `aave_tvl_change_pct_1d` | Liquidity |
| `days_since_spike` | Regime |
| `day_of_week` | Calendar |

# Q1 Spread Tracking — ML Analysis Results

*Run: 2026-04-15 08:17 | Elapsed: 2.7s*

## Run Details

- Training: 2023-07-30 → 2025-09-28 (788 days)
- Test:     2025-09-29 → 2026-04-13 (197 days)
- Target:   `spread_vs_net` (Aave apyBase − Compound net borrow cost)

## Model Performance

| Model | MAE | RMSE | R² | Direction Acc |
|---|---|---|---|---|
| Naive persistence | 0.4796 | 0.9841 | -0.2533 | N/A |
| ARIMA(2, 0, 2) | 0.9629 | 1.2551 | -1.0387 | 50.8% |
| Linear regression | 0.0000 | 0.0000 | 1.0000 | 100.0% |
| XGBoost (n_estimators=300) | 0.1453 | 0.2530 | 0.9172 | 79.7% |

**Best model by MAE:** Linear regression (MAE=0.0000)

## XGBoost Feature Importance

| Feature | Importance |
|---|---|
| `aave_tvl_change_pct_1d` | 0.2851 |
| `aave_rate_change_1d` | 0.2607 |
| `spread_vs_net_rolling_mean_7d` | 0.0996 |
| `tvl_ratio` | 0.0945 |
| `compound_net_change_1d` | 0.0772 |
| `days_since_spike` | 0.0699 |
| `spread_vs_net_rolling_std_7d` | 0.0501 |
| `spread_vs_net_lag_1d` | 0.0366 |
| `day_of_week` | 0.0264 |

## Linear Regression Coefficients

| Feature | Coefficient |
|---|---|
| `spread_vs_net_lag_1d` | 1.0000 |
| `aave_rate_change_1d` | 1.0000 |
| `compound_net_change_1d` | -1.0000 |
| `spread_vs_net_rolling_mean_7d` | -0.0000 |
| `spread_vs_net_rolling_std_7d` | -0.0000 |
| `tvl_ratio` | 0.0000 |
| `aave_tvl_change_pct_1d` | 0.0000 |
| `days_since_spike` | -0.0000 |
| `day_of_week` | -0.0000 |

## Key Findings

- Best model (Linear regression) achieved MAE of 0.0000% vs naive baseline of 0.4796% — 100.0% improvement.
- Naive persistence direction accuracy: N/A (predicts no change).
- Best model direction accuracy: 100.0%.
- Top XGBoost feature: `aave_tvl_change_pct_1d`.
- Best ARIMA order: (2, 0, 2) (AIC=4022.64).

## Interpretation

### Linear regression — tautological result, not genuine predictive power

The perfect MAE (0.0000), R² (1.0000), and direction accuracy (100%) are not real. The feature set contains a tautological relationship: `spread_vs_net` today can be algebraically reconstructed from three features:

```
spread_vs_net_t = spread_vs_net_lag_1d + aave_rate_change_1d - compound_net_change_1d
```

`aave_rate_change_1d` and `compound_net_change_1d` are computed as today's value minus yesterday's, so together with the lag they reconstruct today's spread exactly. The linear regression found the exact coefficients (1.0, 1.0, −1.0) and solved the identity. XGBoost's inflated R² (0.92) is explained by the same relationship — it partially learned the same tautology.

This is not a modelling error to fix. The identity holds by construction and confirms that the spread is fully accounted for by yesterday's level plus today's rate movements on each protocol. The interesting modelling question — predicting tomorrow's spread from today's features — is a separate study (Q1S1).

### Naive persistence — spread is volatile, yesterday is a poor predictor

MAE of 0.48% and R² of −0.25 (worse than predicting the mean) confirm that the spread is volatile enough that simply carrying forward yesterday's value performs poorly. A mean-predicting baseline would beat it. This is consistent with our earlier finding that the daily spread has a standard deviation of 2.75% and a range of nearly 73 percentage points.

### ARIMA(2,0,2) — no exploitable autocorrelation in a single multi-step forecast

ARIMA performs worse than naive persistence (MAE 0.96 vs 0.48), which suggests the spread does not have autocorrelation structure that a single-fit ARIMA can exploit across a 197-day out-of-sample window. The model was fit once on training data and asked to forecast 197 steps ahead — forecast uncertainty compounds quickly. Rolling 1-step ahead forecasting would give a fairer picture; that is addressed in Q1S1 (spread persistence).

### XGBoost feature importance — rate changes dominate, lag is weak

Ignoring the tautological contamination, the feature importance ranking is informative:
- `aave_tvl_change_pct_1d` (0.285) and `aave_rate_change_1d` (0.261) are the top two — both reflect same-day Aave activity, consistent with Aave being the larger and more volatile protocol
- `spread_vs_net_lag_1d` ranks last (0.037), reinforcing the naive persistence finding: the prior day's spread level carries little predictive signal on its own
- `day_of_week` (0.026) is the weakest feature overall, suggesting no strong calendar effect on the spread level — though this is better tested directly in Q1S5

---

# Q1 Spread Tracking — Statistical Analysis

*Run: 2026-04-15 09:25 | Elapsed: 0.1s*

## 1. Cointegration (Engle-Granger)

| Statistic | p-value | Cointegrated? |
|---|---|---|
| -5.3245 | 0.0 | Yes ✓ |

Critical values: 1%=-3.9059, 5%=-3.3414, 10%=-3.0481

## 2. Spread Stationarity (ADF + KPSS)

| Test | Statistic | p-value | 5% Critical | Lags |
|---|---|---|---|---|
| ADF  | -5.985 | 0.0 | -2.8641 | 12 |
| KPSS | 0.6422 | 0.0188 | 0.463 | 13 |

**Conclusion: Trend-stationary (consider differencing)**

## 3. Ornstein-Uhlenbeck Fit

| Parameter | Value | Interpretation |
|---|---|---|
| θ (mean-reversion speed) | 0.730745 | higher = faster reversion |
| μ (long-run mean spread) | 0.3197% | spread reverts toward this level |
| σ (volatility) | 2.7044% | daily noise in spread process |
| Half-life | 0.95 days | time for shock to decay by 50% |
| R² | 0.3654 | |
| p-value (β) | 0.0 | significance of mean-reversion term |

## 4. Rolling Correlation (30-day window)

| Metric | Value |
|---|---|
| Mean correlation | 0.2693 |
| Min correlation  | -0.6996 |
| Max correlation  | 0.8761 |
| Std              | 0.321 |
| % days corr > 0.90 | 0.0% |
| % days corr < 0.50 | 74.3% |

## 5. Spread Distribution & Episode Analysis

**Distribution (spread_vs_net):**

| Metric | Value |
|---|---|
| Mean   | 0.3189% |
| Std    | 2.807% |
| Skew   | 7.0277 |
| Kurtosis | 126.7129 |
| p5 / p95 | -2.9309% / 3.7327% |
| p25 / p75 | -0.5873% / 0.9878% |
| Median | 0.1895% |

**Episodes (|spread| > 1.0%):**

| Metric | Value |
|---|---|
| Count           | 144 |
| Mean duration   | 3.3 days |
| Max duration    | 27 days |
| Mean peak       | 3.8857% |
| Max peak        | 54.1793% |

## 6. Cross-Correlation Function (CCF)

Contemporaneous correlation: 0.1708

**Aave leads Compound** — peak CCF at lag 3 day(s):
| Lag | CCF |
|---|---|
| 0 | 0.1708 |
| 1 | -0.1052 |
| 2 | -0.0746 |
| 3 | 0.1566 |
| 4 | -0.0853 |

**Compound leads Aave** — peak CCF at lag 4 day(s):
| Lag | CCF |
|---|---|
| 0 | 0.1708 |
| 1 | -0.0283 |
| 2 | -0.0703 |
| 3 | 0.0361 |
| 4 | 0.0847 |

## Interpretation

### 1. Cointegration — strongly confirmed

The test statistic (−5.32) clears the 1% critical value (−3.91) with room to spare. Aave and Compound are cointegrated — they share a long-run equilibrium and the spread cannot drift apart permanently. This is the most important result: arbitrage is functioning at the structural level.

### 2. Stationarity — conflicting result, trend-stationary

ADF strongly rejects a unit root (p=0.0 → stationary). KPSS also rejects its null (p=0.019 → non-stationary). When both reject, the standard interpretation is **trend-stationary**: the spread has a slowly drifting mean rather than a fixed one. In practice this means the long-run equilibrium level of the spread has shifted over the 2023–2026 sample — likely reflecting structural changes in each protocol's market share, risk parameters, or incentive levels. The cointegration result still holds; the drift is slow and bounded.

### 3. Ornstein-Uhlenbeck — half-life under 1 day

θ=0.73 gives a half-life of **0.95 days** — spread shocks decay by 50% within a single day on average. The long-run mean is +0.32% (Aave slightly more expensive than Compound net). This sounds like highly efficient arbitrage, but the σ=2.7% daily volatility means new shocks arrive faster than old ones close. The combination explains both the fast reversion and the extreme tail events: the spread is pulled back quickly but constantly hit by new shocks.

### 4. Rolling correlation — surprisingly weak (mean 0.27)

This is the most striking result. Despite being cointegrated over the full sample, on any given 30-day window the two protocols correlate at only 0.27 on average. On **74% of days** the rolling correlation is below 0.50, and it goes negative (down to −0.70) at times. This means the protocols do not track each other well day-to-day — they respond to different short-term demand signals and only converge over longer horizons. The cointegration is a long-run property, not a short-run one.

### 5. Distribution — near zero most of the time, extreme fat tails

Median spread is only 0.19% — for most days the two protocols are nearly identical. But the kurtosis of **126.7** (normal = 3) and skew of 7.0 reveal extreme fat tails driven by the spike events identified earlier. The 5th–95th percentile range is −2.9% to +3.7%, so 90% of days the spread is within a 6.6% band. The 144 episodes above 1% last an average of **3.3 days** (consistent with the OU half-life) but the worst case ran **27 days** with a peak of 54%. The picture: almost always calm, occasionally explosive.

### 6. Cross-correlation — no meaningful lead/lag

Contemporaneous correlation is only 0.17. CCF values across all lags are small (max 0.16) with no clear structure. Neither protocol reliably leads the other — when rates move, both tend to respond to the same external conditions independently rather than one following the other. Worth revisiting in Q1S6.

### Overall

The protocols are structurally linked (cointegrated) but behave independently day-to-day (mean rolling correlation 0.27). The spread is nearly always small, reverts within a day on average, but is hit by constant new shocks (σ=2.7%) and occasionally by extreme events that last weeks. There is no exploitable lead/lag relationship. The dominant dynamic is: common long-run equilibrium, independent short-run behaviour, with periodic demand shocks that temporarily overwhelm the mean-reversion force.

---

# Q1 Spread Tracking — Trend-Stationarity Exploration

*Run: 2026-04-15 | Notebook: trend_stationarity_exploration.ipynb*

## Background

The ADF + KPSS tests produced a conflicting result: ADF rejected a unit root (stationary) while KPSS rejected stationarity. This five-step exploration investigated whether the spread has a genuine deterministic trend or experienced structural breaks in its mean level.

## Results

### Step 1 — Visual Inspection

The 30-day and 90-day rolling means drift around the overall mean with no directional trend. The clipped (±10%) series shows no smooth upward or downward drift over the 2023–2026 sample. Visual inspection gives no support to a genuine linear trend.

### Step 2 — ADF + KPSS with Trend Term (`regression='ct'`)

Re-running both tests with a constant + linear trend specification still produced a conflicting result. KPSS continued to reject stationarity even when a linear trend was included. A linear trend does not resolve the conflict — the series is not trend-stationary in the classical sense.

### Step 3 — Zivot-Andrews Test (Single Structural Break)

| Metric | Value |
|---|---|
| Statistic | −7.6084 |
| p-value | 0.0010 |
| Critical value (1%) | −5.5756 |
| Critical value (5%) | −5.0733 |
| Identified break date | 2023-07-19 |
| Unit root rejected | Yes |

The statistic (−7.61) clears the 1% critical value (−5.58) with room to spare — a much stronger rejection than plain ADF alone. When the test allows for a single structural break, the unit root is decisively rejected. The identified break falls early in the sample (~14% in), consistent with a genuine level shift rather than a gradual drift.

### Step 4 — Bai-Perron Test (Multiple Structural Breaks)

| k | Break dates | RSS | BIC |
|---|---|---|---|
| 1 | 2023-07-19 | 0.96 | −8205.58 |
| 2 | 2023-07-30, 2024-02-06 | 0.90 | −8276.99 |
| 3 | 2023-07-30, 2024-01-16, 2024-09-23 | 0.89 | −8284.02 |

**Optimal by BIC: k=3**

BIC improvements are asymmetric: k=1→k=2 improves by 71 BIC units (substantial); k=2→k=3 improves by only 7 units (marginal). The early break (~2023-07-19/30) is consistent across all k values, confirming it is a stable feature of the data. The third break (2024-09-23) is statistically detectable but weak.

### Step 5 — Within-Regime Stationarity

| Segment | Date range | N | Mean | ADF p | KPSS p | Conclusion |
|---|---|---|---|---|---|---|
| 1 | 2023-02-06 → 2023-07-29 | 174 | −0.0072% | 0.0000 | 0.0508 | Stationary ✓ |
| 2 | 2023-07-30 → 2024-01-15 | 168 | +0.0227% | 0.0000 | 0.1000 | Stationary ✓ |
| 3 | 2024-01-16 → 2024-09-22 | 251 | +0.0074% | 0.0232 | 0.1000 | Stationary ✓ |
| 4 | 2024-09-23 → 2026-04-13 | 566 | −0.0010% | 0.0002 | 0.1000 | Stationary ✓ |

All four segments are individually stationary (ADF and KPSS both satisfied at 5%).

## Conclusion

The spread is stationary within regimes. The full-sample KPSS rejection was a statistical artifact of three small level shifts in the mean — not evidence of a genuine time trend or structural non-stationarity.

Notably, the segment means are economically negligible (all within 3 basis points of zero). The break detector identified statistically significant but practically immaterial mean shifts. There is no meaningful long-run drift in the equilibrium spread level.

**The cointegration finding and OU half-life (0.95 days) hold without qualification.** The spread has a stable near-zero long-run mean, reverts quickly after shocks, and does not trend. The earlier "trend-stationary" label from the ADF/KPSS conflict was a false signal caused by these regime shifts.

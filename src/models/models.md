# Model Plans

All questions use time-series data — no random shuffling. Splits are always chronological.

**Shared train/test convention:** 80% train (≈927 days, up to ~2025-09), 20% test (≈232 days, ~2025-09 → present). Walk-forward validation is used where noted instead of a single holdout, to better simulate live deployment.

**Available features (10 selected):**
`spread_vs_net`, `spread_vs_net_lag_1d`, `spread_vs_net_rolling_mean_7d`, `spread_vs_net_rolling_std_7d`, `aave_rate_change_1d`, `compound_net_change_1d`, `tvl_ratio`, `aave_tvl_change_pct_1d`, `days_since_spike`, `day_of_week`

---

## Q1 — Spread Tracking (main)
*How closely do the two protocols track each other? Are spreads persistent?*

**Data:** All 1,159 overlapping days. Target: `spread_vs_net`. Full feature set.

**Train/test:** 80/20 chronological split. Walk-forward validation (expanding window, 30-day step) on train set for hyperparameter tuning.

**Models:**
1. Naïve persistence — predict tomorrow's spread = today's (baseline)
2. ARIMA — classical time series, order selected by AIC
3. Linear regression — spread features + rate momentum as regressors
4. XGBoost regressor — captures non-linear interactions between features

**Metrics:** MAE, RMSE, R², direction accuracy (did the model get the sign of tomorrow's spread change correct?)

---

## Q1S1 — Spread Persistence
*Is today's spread predictive of tomorrow's? How long does a spread take to close?*

**Data:** `spread_vs_net`, `spread_vs_net_lag_1d`, `spread_vs_net_rolling_mean_7d`. Full sample for statistical characterisation; 80/20 split for regression model.

**Train/test:** Statistical tests (ACF, OU fit) use the full sample. Regression model uses 80/20 chronological split.

**Models:**
1. ACF/PACF analysis — characterise autocorrelation structure, determine AR lag order
2. AR(p) — autoregressive model, order chosen by AIC/BIC
3. Ornstein-Uhlenbeck fit — estimate mean-reversion speed θ, long-run mean μ, and volatility σ; derive spread half-life as ln(2)/θ
4. Linear regression — spread_t ~ spread_lag_1d + rolling_mean_7d

**Metrics:** Ljung-Box p-value (is autocorrelation significant?), half-life in days, R², AIC/BIC for model order selection

---

## Q1S2 — Volatility as a Leading Indicator
*When rolling spread volatility spikes, does the spread subsequently narrow?*

**Data:** `spread_vs_net_rolling_std_7d`, `spread_vs_net`, `spread_vs_net_lag_1d`. Target: binary label — does spread narrow within the next 3 days?

**Train/test:** 80/20 chronological split. Class balance check before training (spike periods are minority class).

**Models:**
1. Threshold rule — if rolling_std > 75th percentile, predict narrowing (interpretable baseline)
2. Logistic regression — probability of narrowing in next 3 days
3. XGBoost classifier — captures non-linear volatility thresholds
4. Linear regression — predict magnitude of spread change (regression variant)

**Metrics:** AUC-ROC, precision, recall, F1; for regression variant: MAE, direction accuracy

---

## Q1S3 — TVL Shocks and Rate Divergence
*Do sudden inflows or outflows on either protocol drive same-day or next-day spread widening?*

**Data:** `aave_tvl_change_pct_1d`, `tvl_ratio`, `aave_rate_change_1d`, `compound_net_change_1d`, `spread_vs_net`. Target: next-day spread change.

**Train/test:** 80/20 chronological split. Granger causality tests use full sample.

**Models:**
1. Granger causality test — establishes whether TVL changes statistically precede spread changes (both directions)
2. OLS regression — interpretable baseline; quantifies marginal effect of TVL shock on spread
3. Lasso regression — automatic feature selection across TVL and rate momentum features
4. XGBoost regressor — captures asymmetric effects (large inflows vs outflows may behave differently)

**Metrics:** Granger causality p-values, R², MAE, feature importance (XGBoost)

---

## Q1S4 — Post-Spike Spread Behaviour
*Do spreads close more slowly in the days following a rate spike?*

**Data:** `days_since_spike`, `spread_vs_net`, `spread_vs_net_rolling_std_7d`. Regime split: spike window (days_since_spike ≤ 7) vs steady-state (days_since_spike > 7).

**Train/test:** Regime segments compared statistically across full sample. Predictive model uses 80/20 chronological split.

**Models:**
1. Regime segmentation — compare spread half-life (from OU fit) inside vs outside spike window
2. Mann-Whitney U test — is spread closure time significantly longer post-spike?
3. Linear regression with interaction term — spread_change ~ lag + days_since_spike + lag×days_since_spike
4. Random forest — captures non-linear regime effects without manual interaction terms

**Metrics:** Half-life by regime (days), Mann-Whitney U p-value, R², MAE

---

## Q1S5 — Weekend Effect
*Are spreads opened on Friday more likely to persist into Monday?*

**Data:** `day_of_week`, `spread_vs_net`, `spread_vs_net_lag_1d`. Weekend = Saturday/Sunday (day_of_week ∈ {5, 6}).

**Train/test:** Statistical comparison uses full sample. Predictive model uses 80/20 chronological split.

**Models:**
1. Mann-Whitney U test — is weekend spread significantly different in level or persistence from weekday?
2. Linear regression with day-of-week dummies — quantify per-day effect on spread change
3. Logistic regression — predict whether spread persists (same sign) to next day, with day_of_week as a feature
4. XGBoost classifier — same target, allows interaction between day_of_week and spread features

**Metrics:** Effect size (Cohen's d), p-value, AUC-ROC, feature importance of `day_of_week`

---

## Q1S6 — Lead/Lag Between Protocols
*Does one protocol consistently change its rate before the other?*

**Data:** `aave_rate_change_1d`, `compound_net_change_1d`, raw `aave_apyBase` and `compound_net` series.

**Train/test:** Cross-correlation and Granger tests use full sample. VAR and regression model use 80/20 chronological split.

**Models:**
1. Cross-correlation function (CCF) — identify peak lag between Aave and Compound rate changes
2. Granger causality — test both directions (Aave→Compound and Compound→Aave) at lags 1–5
3. VAR (Vector Autoregression) — jointly models both rate series, captures bidirectional dynamics
4. Linear regression — compound_net_change_t ~ aave_rate_change_{t-1} (and vice versa)

**Metrics:** Granger causality p-values, CCF peak lag, VAR forecast MAE, R²

---

## Q1S7 — Spread Direction Prediction
*Can we predict whether the spread will widen or narrow tomorrow?*

**Data:** All 10 selected features. Target: binary label — spread widens (1) or narrows (0) the next day.

**Train/test:** 80/20 chronological split with walk-forward validation (30-day expanding window) on the train set. Predict probability, not just class, so calibration can be evaluated.

**Models:**
1. Logistic regression — interpretable baseline; coefficients directly indicate which features drive widening
2. Random forest — handles non-linear interactions, provides feature importance
3. XGBoost classifier — strong tabular baseline; tune with walk-forward CV
4. LSTM — sequence model; marginal given only 1,159 rows but worth benchmarking against tree models

**Metrics:** AUC-ROC (primary), precision, recall, F1, accuracy, Brier score (probability calibration)

# Project Steps

## Step 1 — Project scaffold
Created the full folder structure for a production ML project:
- `src/data/`, `src/features/`, `src/models/` — pipeline modules
- `app/` — FastAPI + Streamlit
- `tests/` — unit tests
- `notebooks/` — EDA
- `data/`, `models/` — local storage (gitignored)
- `__init__.py` in every Python package
- `requirements.txt` with all dependencies
- `.gitignore` excluding `data/`, `models/`, `.env`, `__pycache__`, `.ipynb_checkpoints`
- `setup.py` making `src/` importable as a package via `pip install -e .`

## Step 2 — Environment setup
Activated conda environment `build-the-death-star` and installed all packages from `requirements.txt`:
- Already present: `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `fastapi`, `uvicorn`, `pytest`, `plotly`
- Newly installed: `streamlit`, `mlflow`, `great-expectations`

## Step 3 — Project brief review
Read `projects-outline.md`. The project is a DeFi interest rate analytics study on Aave and Compound, structured as 10 research questions progressing from descriptive analysis to predictive modeling. Data sources are DefiLlama API, Dune Analytics, and The Graph subgraphs for Aave V3 and Compound V3.

**Planned execution order:**
1. Questions 1–3: Descriptive and visual (cross-protocol rate correlation, utilization curve validation, event-driven rate behavior)
2. Questions 4–6: Modeling (mean-reversion, volatility comparison, rate forecasting)
3. Questions 7–10: Original/advanced (lead-lag analysis, time patterns, Aave V3 impact, whale detection)

## Step 4 — Question 1: Data ingestion

**Question:** How closely do Aave and Compound track each other on the same asset? For a given asset like USDC, do the two protocols' borrow rates move in lockstep, or are there persistent spreads? If spreads exist, how long do they last before arbitrage closes them?

### Data source investigation
- Primary source: DefiLlama Yields API (no auth required, covers both protocols)
- Investigated all USDC/Ethereum pools across Aave and Compound
- Findings:
  - Aave V2: fully deprecated, not listed on DefiLlama — skipped
  - Compound V2: near-deprecated ($2.67M TVL), project name is `compound-v2` not `compound` — skipped
  - Aave V3: 3 pools found — main market (`aa70268e`, $799M TVL) + two isolated markets (`horizon-market`, `lido-market`) — use main market only
  - Compound V3: 2 pools found — main market (`7da72d09`, $109M TVL) + one niche pool — use main market only
- Decision: compare **Aave V3 vs Compound V3** main markets only (V2 markets are dead)
- Key insight: must use `apyBase` not `apy` for fair comparison — Compound V3 has COMP reward APY on top

### Scripts written
- `src/data/ingest.py` — fetches pool list, filters to main markets (`poolMeta == null`), pulls historical chart data per pool, saves raw JSON to `data/raw/`
- `src/data/loader.py` — loads raw JSON into a combined DataFrame, parses timestamps to UTC datetime, prints shape/dtypes/summary stats/missing values/date range

### Data pulled
- `data/raw/pools_metadata.json` — 2 matched pools with TVL and metadata
- `data/raw/chart_aave_v3_usdc_ethereum_aa70268e.json` — 1,163 daily records from 2023-02-06
- `data/raw/chart_compound_v3_usdc_ethereum_7da72d09.json` — 1,282 daily records

Status: data ingestion complete, ready for EDA / loader validation

### Fixes & improvements
- `.gitignore` bug: `data/` was matching `src/data/`, silently excluding all source files. Fixed by anchoring to root (`/data/`, `/models/`)
- `loader.py`: drops `il7d` and `apyBase7d` at load time (entirely null in both pools)
- `quality.py`: corrected `tvlUsd` expected dtype from float to int; removed `apyBase7d` from value bounds

### Scripts written
- `src/data/quality.py` — five-check data quality gate: schema validation, row count, null rates, value ranges, group distribution. Returns structured dict with `success`, `failures`, `warnings`, `statistics`

## Step 5 — Feature engineering

Created `src/features/engineering.py` with two functions:

- `create_features(df)` — pivots long-format data to wide, computes all 22 engineered features, returns wide DataFrame
- `select_features(wide)` — filters by correlation (threshold 0.95) and variance (threshold 1% of mean variance), logs dropped features and reasons, returns selected feature list and reduced DataFrame

**Selection outcome:** 22 candidates → 10 selected
- 7 dropped for high correlation: all `_vs_base` spread variants correlated > 0.95 with `_vs_net` equivalents; `compound_base_change_1d` correlated 1.0 with `compound_net_change_1d`
- 5 dropped for low variance: `compound_apyReward` (stable reward), `is_spike` (rare binary), `compound_tvl_change_pct_1d`, `rate_divergence_direction_vs_net`, `spread_vs_net_zscore_30d`

Findings documented in `feature_engineering.md`.

- `src/features/run_features.py` — pipeline script: loads raw data via `load_all()`, runs `create_features()` and `select_features()`, saves selected features to `data/features.csv`, prints before/after shape and timing

## Step 6 — EDA notebook

Created `notebooks/eda.ipynb` with 7 sections:
1. Overview — shape, dtypes, head
2. Target Analysis — record count per protocol, apyBase distribution per protocol, reward-adjusted rate comparison
3. Missing Values — null rate table + seaborn heatmap
4. Feature Distributions — 3×3 subplot grid of histograms (unused axes hidden)
5. Correlation Matrix — annotated heatmap, numeric features only
6. Features vs Target — box plots of top-3 correlated features grouped by protocol
7. Key Findings — data-grounded bullets (see below)

## Step 7 — Q1 main study: ML and statistical analysis

### ML analysis (`src/models/q1_spread_tracking/spread_tracking_ml.py`)
Four models predicting `spread_vs_net`: naive persistence, ARIMA(2,0,2), linear regression, XGBoost. Linear regression achieved perfect in-sample fit via an algebraic identity (`spread_t = spread_lag_1d + aave_rate_change_1d - compound_net_change_1d`) — not genuine predictive power. XGBoost R²=0.92 partially learned the same identity. ARIMA performed worse than naive persistence on the 197-day holdout. Results and interpretation written to `src/models/q1_spread_tracking/results.md`.

### Statistical analysis (`src/models/q1_spread_tracking/spread_tracking_stats.py`)
Six tests on the full 1,159-day sample:
1. Cointegration (Engle-Granger): confirmed at p=0.0, stat=−5.32 clearing the 1% critical value
2. Stationarity (ADF+KPSS): conflicting result — trend-stationary
3. Ornstein-Uhlenbeck fit: θ=0.73, half-life=0.95 days, σ=2.7%
4. Rolling correlation (30-day): mean=0.27, 74% of days below 0.50
5. Spread distribution: median 0.19%, kurtosis 126.7; 144 episodes >1%, mean 3.3 days
6. CCF: contemporaneous correlation 0.17, no meaningful lead/lag at any lag

### Trend-stationarity exploration (`src/models/q1_spread_tracking/trend_stationarity_exploration.ipynb`)
Investigated the ADF/KPSS conflict. Zivot-Andrews strongly rejected unit root (stat=−7.61 vs 1% critical=−5.58) with break at 2023-07-19. Bai-Perron found 3 optimal breaks (2023-07-30, 2024-01-16, 2024-09-23). All 4 regime segments individually stationary. Conclusion: KPSS rejection was a statistical artifact of small level shifts (all segment means within 3bp of zero), not a genuine trend.

## Step 8 — Q1 supplementary studies (Q1S1–Q1S7)

Each study has a statistical file (writes `results.md`) and an ML file (appends to `results.md`) in its subfolder under `src/models/`.

### Q1S1 — Spread Persistence
Stats: ACF significant at 16 of 20 lags (max 0.27 at lag 1), Ljung-Box p=0.0, OU half-life=0.95 days.
ML: AR(10) best model (R²=0.064), linear regression R²=0.012. Naive persistence R²=−0.60 — worse than predicting the mean. Rolling mean 7d outperforms the lag as a predictor.

### Q1S2 — Volatility as a Leading Indicator
Stats: High-vol mean |spread| 3.1% vs 0.88% low-vol. Mann-Whitney p=0.003 — high-vol leads to more narrowing. OU half-life 0.84 days (high-vol) vs 1.8 days (low-vol).
ML: Threshold rule AUC=0.503 (fails). XGBoost AUC=0.635. Feature importances nearly equal across all 5 features — signal is non-linear and diffuse.

### Q1S3 — TVL Shocks and Rate Divergence
Stats: Spread Granger-causes Aave TVL (F=593 at lag 1) — causal direction is mostly reversed from the hypothesis. Aave TVL → spread significant only at lag 2. Outflow shocks produce −1.26% next-day spread change vs +0.34% for inflows.
ML: OLS R²=0.27 — best result across all regression tasks. Lasso retains 6/7 features, zeroes out `rolling_std_7d`. XGBoost underperforms OLS (R²=−0.22), confirming linear relationship.

### Q1S4 — Post-Spike Spread Behaviour
Stats: Spike-window half-life 0.87 days vs 1.25 days steady-state — spreads close *faster* post-spike, not slower. 68.5% of episodes close within 3 days, 92.5% within 7 days.
ML: Interaction term `lag×days_since_spike` coefficient near zero (0.000149). R²=0.004 for linear regression. Hypothesis that post-spike reversion is slower is rejected.

### Q1S5 — Weekend Effect
Stats: Fri→Mon persist rate 68.5% vs Tue→Wed 80.1% — Friday spreads are less likely to persist into Monday, not more. Saturday has lowest mean |spread| (1.11%) and highest persistence (85.5%). Cohen's d=−0.088 (negligible effect size).
ML: XGBoost AUC=0.639. Weekend dummies (`dow_5`, `dow_6`) account for 24% of combined feature importance. Day-of-week adds modest signal but is not dominant.

### Q1S6 — Lead/Lag Between Protocols
Stats: Neither direction significant at lag 1. Aave→Compound significant only at lag 4; Compound→Aave from lag 2. Rolling Granger: Aave→Compound significant in only 25.8% of 180-day windows.
ML: Cross-lagged R² near zero (0.0002 for Compound~Aave lag-1). VAR(10) off-diagonal lag-1 coefficients tiny (0.012 and −0.031). Own-series negative autocorrelation (−0.43, −0.39) dominates.

### Q1S7 — Spread Direction Prediction
Stats: Near-balanced target (48/52). Direction ACF significant only at lag 1 (−0.148). Five features linearly significant; `rolling_std_7d`, `day_of_week`, `days_since_spike` insignificant.
ML: Logistic regression best (AUC=0.671, Brier=0.239). Tree models AUC 0.632–0.636 — relationship is linear. Signal dominated by `aave_rate_change_1d` and `spread_vs_net_lag_1d`.


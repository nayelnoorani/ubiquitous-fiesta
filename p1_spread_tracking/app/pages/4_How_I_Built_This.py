import streamlit as st

st.set_page_config(page_title="How I Built This", layout="wide")
st.title("How I Built This")

# ── Pipeline diagram ───────────────────────────────────────────────────────────
st.markdown("## Data Pipeline")

st.markdown("""
```
DefiLlama Yields API
        │
        │  GET /chart/{pool-id}  (no auth required)
        ▼
  src/data/ingest.py
  ┌─────────────────────────────────────────────────────────┐
  │  1. Fetch pool list for USDC/Ethereum                   │
  │  2. Filter to main markets (poolMeta == null)           │
  │     → Aave V3:     pool aa70268e  ($799M TVL)           │
  │     → Compound V3: pool 7da72d09  ($109M TVL)           │
  │  3. Pull daily chart data for each pool                 │
  │  4. Save raw JSON to data/raw/                          │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  src/data/loader.py
  ┌─────────────────────────────────────────────────────────┐
  │  Load raw JSON → combined long-format DataFrame         │
  │  Parse timestamps to UTC · drop null columns            │
  │  2,445 rows × 8 columns (both protocols)                │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  src/data/quality.py
  ┌─────────────────────────────────────────────────────────┐
  │  Five-check data quality gate:                          │
  │    1. Schema validation (expected dtypes)               │
  │    2. Row count (≥ 1,000 per protocol)                  │
  │    3. Null rates (only apyReward for Aave — structural) │
  │    4. Value ranges (APY 0–100%, TVL > 0)                │
  │    5. Group distribution (exactly 2 protocols)          │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  src/features/engineering.py
  ┌─────────────────────────────────────────────────────────┐
  │  Pivot long → wide (inner join on date)                 │
  │  Engineer 22 features across 6 groups:                  │
  │    Spread · Rate momentum · Reward · Liquidity          │
  │    Regime · Calendar                                    │
  │  Correlation filter (threshold 0.95) → drop 7          │
  │  Variance filter (< 1% of mean) → drop 5               │
  │  Selected: 10 features saved to data/features.csv      │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  src/models/  (one subfolder per study)
  ┌─────────────────────────────────────────────────────────┐
  │  Q1  · Q1S1 · Q1S2 · Q1S3 · Q1S4 · Q1S5 · Q1S6 · Q1S7 │
  │  Each: *_stats.py (writes results.md)                   │
  │        *_ml.py    (appends to results.md)               │
  │  Splits: 80/20 chronological — no random shuffling      │
  └─────────────────────────────────────────────────────────┘
```
""")

st.markdown("---")

# ── Data decisions ─────────────────────────────────────────────────────────────
st.markdown("## Data Decisions")

with st.expander("Why V2 markets were excluded"):
    st.markdown("""
After pulling the full USDC/Ethereum pool list from DefiLlama, four candidates appeared:

| Protocol | TVL | Decision |
|---|---|---|
| Aave V3 main market | $799M | **Used** |
| Aave V3 isolated markets (2) | < $10M | Skipped — niche, low TVL |
| Compound V3 main market | $109M | **Used** |
| Compound V2 | $2.67M | Skipped — near-deprecated |
| Aave V2 | Not listed | Fully deprecated |

Comparing V3 to V3 on the same asset gives the cleanest apples-to-apples view.
V2 markets attract little volume and would contaminate the spread analysis.
""")

with st.expander("Why Compound's effective rate = apyBase − apyReward"):
    st.markdown("""
DefiLlama returns two separate fields:
- `apyBase` — the base borrow APY set by the protocol's utilisation curve
- `apyReward` — token incentive APY currently running on the pool

For Compound V3, COMP rewards are paid **directly to borrowers**, not lenders.
This means a borrower's true cost is `apyBase − apyReward`, not `apyBase` alone.

Compound's mean `apyReward` over the sample is **0.43%**.
Using `apyBase` for comparison would systematically overstate Compound's cost by that amount —
enough to flip which protocol appears cheaper on a majority of days.

Aave V3 offers no reward programme on this pool, so `apyBase == apyNet` throughout.
""")

with st.expander("The .gitignore bug that silently excluded source files"):
    st.markdown("""
The initial `.gitignore` contained the pattern `data/` (without a leading `/`).
Git interprets this as matching any directory named `data/` anywhere in the repo tree —
including `src/data/`, which contained the pipeline source code.

All Python files in `src/data/` (`ingest.py`, `loader.py`, `quality.py`) were silently
excluded from version control. The fix was to anchor the pattern to the repo root: `/data/`.

This is a subtle gotcha with gitignore patterns: a pattern without a `/` prefix matches
anywhere in the tree, not just at the root.
""")

st.markdown("---")

# ── Feature engineering ────────────────────────────────────────────────────────
st.markdown("## Feature Engineering")

st.markdown("""
22 features were engineered from the 6 raw columns. Selection reduced this to 10 using two filters:

**Correlation filter (threshold 0.95):** 7 features dropped
- All `_vs_base` spread variants correlated > 0.95 with their `_vs_net` equivalents —
  because `compound_apyReward` is stable enough that both spread definitions carry
  near-identical information over the sample.
- `compound_base_change_1d` correlated 1.0 with `compound_net_change_1d` (same reason).

**Variance filter (< 1% of mean variance):** 5 features dropped
- `compound_apyReward` — reward rate is nearly constant (most signal already absorbed into `compound_net`)
- `is_spike` — rare binary flag (< 7% of days)
- `compound_tvl_change_pct_1d`, `rate_divergence_direction_vs_net`, `spread_vs_net_zscore_30d` — near-zero variance
""")

st.markdown("**Selected features (10):**")
st.markdown("""
| Feature | Group | Description |
|---|---|---|
| `spread_vs_net` | Spread | Aave base − Compound net borrow cost |
| `spread_vs_net_lag_1d` | Spread | Prior day's spread |
| `spread_vs_net_rolling_mean_7d` | Spread | 7-day rolling mean |
| `spread_vs_net_rolling_std_7d` | Spread | 7-day rolling volatility |
| `aave_rate_change_1d` | Rate momentum | Aave rate change vs prior day |
| `compound_net_change_1d` | Rate momentum | Compound net rate change vs prior day |
| `tvl_ratio` | Liquidity | Aave TVL / Compound TVL |
| `aave_tvl_change_pct_1d` | Liquidity | Aave TVL % change vs prior day |
| `days_since_spike` | Regime | Days since either protocol exceeded 10% APY |
| `day_of_week` | Calendar | 0=Monday … 6=Sunday |
""")

st.markdown("---")

# ── Stationarity deep-dive ─────────────────────────────────────────────────────
st.markdown("## Stationarity Deep-Dive")

st.markdown("""
The ADF and KPSS tests produced a conflicting result on the spread series:
ADF rejected a unit root (stationary, p=0.0); KPSS also rejected its null (non-stationary, p=0.019).
When both reject, the standard label is **trend-stationary** — a series with a slowly drifting mean.

That label turned out to be a false signal. A five-step investigation resolved the conflict:

1. **Visual inspection** — no directional trend visible in rolling means over 2023–2026.
2. **ADF + KPSS with trend term** — KPSS continued to reject even with a linear trend included.
3. **Zivot-Andrews test** — allows a single structural break in the mean. Stat = −7.61,
   clearing the 1% critical value (−5.58). Break identified at 2023-07-19. Unit root decisively rejected.
4. **Bai-Perron test** — 3 optimal breaks found (2023-07-30, 2024-01-16, 2024-09-23)
   by BIC minimisation. The first break is stable across all model orders.
5. **Within-regime stationarity** — all 4 segments individually stationary (ADF + KPSS both satisfied).

**Conclusion:** The spread is stationary within regimes. The KPSS rejection was a statistical
artifact of three small level shifts — the segment means are all within 3 basis points of zero,
economically negligible. There is no meaningful long-run drift in the equilibrium spread level.

The cointegration finding and OU half-life (0.95 days) hold without qualification.
""")

st.markdown("---")

# ── Modelling choices ──────────────────────────────────────────────────────────
st.markdown("## Modelling Choices")

with st.expander("Chronological splits — why no random shuffling"):
    st.markdown("""
All train/test splits follow an 80/20 chronological split (roughly 927 days train,
232 days test). Random shuffling is inappropriate for time series because:

1. It leaks future information into the training set (look-ahead bias).
2. It destroys the autocorrelation structure the models are trying to exploit.
3. It inflates out-of-sample performance estimates.

For hyperparameter tuning, walk-forward (expanding-window) cross-validation was used
on the training set in a 5-fold `TimeSeriesSplit` configuration.
""")

with st.expander("Why ARIMA underperforms naive persistence"):
    st.markdown("""
ARIMA(2,0,2) was fit once on the training data and asked to forecast 197 steps ahead.
Forecast uncertainty compounds with each step — a 1-step-ahead ARIMA is typically accurate;
a 197-step-ahead ARIMA degenerates toward the unconditional mean.

The result (MAE=0.96 vs naive 0.48) is therefore not surprising and is not evidence that
ARIMA is a poor model for this series. A rolling 1-step-ahead forecast would give a
fairer comparison. That exercise is covered in Q1S1, where AR(10) achieves MAE=0.57.
""")

with st.expander("The linear regression tautology — what it reveals"):
    st.markdown("""
Linear regression achieved perfect MAE (0.0000) and R²=1.0000 in Q1. The feature set
contained an algebraic identity:

```
spread_vs_net_t  =  spread_vs_net_lag_1d
                  + aave_rate_change_1d
                  − compound_net_change_1d
```

`aave_rate_change_1d` is defined as `aave_apyBase_t − aave_apyBase_{t-1}`, and similarly
for Compound. Together with the 1-day lag, they reconstruct today's spread exactly.
The regression found the exact coefficients (1.0, 1.0, −1.0) and solved the identity.

This is not a bug to fix — it is an illuminating result. The spread at time `t` is
*fully determined* by yesterday's spread plus today's rate moves on each protocol.
The interesting modelling question — predicting tomorrow's spread from today's features — is Q1S1.
""")

st.markdown("---")

# ── Stack ──────────────────────────────────────────────────────────────────────
st.markdown("## Stack")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**Data & modelling**
- Python 3.12
- pandas — data loading and feature engineering
- statsmodels — Engle-Granger, ADF, KPSS, Zivot-Andrews, Granger, OU fit
- scikit-learn — linear regression, logistic regression, random forest, Lasso
- XGBoost — gradient boosted trees

**Infrastructure**
- DefiLlama Yields API — no API key required
- Git + GitHub — version control
- Streamlit Community Cloud — deployment
""")

with col2:
    st.markdown("""
**App**
- Streamlit — multipage app framework
- Plotly — interactive charts

**Repository**
All code and data at:
[github.com/nayelnoorani/ubiquitous-fiesta](https://github.com/nayelnoorani/ubiquitous-fiesta)

Project lives in `p1_spread_tracking/`.
""")

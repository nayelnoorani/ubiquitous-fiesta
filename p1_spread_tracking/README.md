# DeFi Borrow Rate Analysis: Aave V3 vs Compound V3

Empirical analysis of USDC borrow rates across Aave V3 and Compound V3 on Ethereum mainnet, using historical data from the DefiLlama API.

---

## Research Question

**How closely do Aave and Compound track each other on the same asset?** For a given asset like USDC, do the two protocols' borrow rates move in lockstep, or are there persistent spreads? If spreads exist, how long do they last before arbitrage closes them?

**Supplementary questions** arising from feature engineering:

1. **Spread persistence** — Is today's spread predictive of tomorrow's? How many days does a spread take to close on average?
2. **Volatility as a leading indicator** — When rolling spread volatility spikes, does the spread subsequently narrow? Does high local volatility signal a regime transition?
3. **TVL shocks and rate divergence** — Do sudden inflows or outflows on either protocol drive same-day or next-day spread widening?
4. **Post-spike spread behaviour** — Do spreads take longer to close in the days following a rate spike than in steady state? Does market stress impair arbitrage?
5. **Weekend effect** — Are spreads opened on Friday more likely to persist into Monday, consistent with reduced arbitrageur activity on weekends?
6. **Lead/lag between protocols** — Does one protocol consistently move its rate before the other? If so, the lagging protocol's rate change is predictable.
7. **Spread direction prediction** — Given spread level, lag, rolling mean, volatility, and rate momentum, can a model predict whether the spread will widen or narrow the next day?

---

## EDA Findings (Question 1)

### Data

| Protocol | Pool | TVL (avg) | Records | Date range |
|---|---|---|---|---|
| Aave V3 | `aa70268e` | $397M | 1,163 | 2023-02-06 → present |
| Compound V3 | `7da72d09` | $60M | 1,282 | 2022-10-06 → present |

Overlapping window: **1,159 days**.

---

### Rates are broadly similar but not in lockstep

On the overlapping period, Aave's mean `apyBase` (4.87%) sits just below Compound's (4.99%). Compound was the higher-rate protocol on **55% of days**. The two series move together over long horizons but diverge sharply during demand shocks — the daily spread has a standard deviation of **2.75%** and a range of **−18.8% to +52.9%**.

---

### Compound is cheaper once rewards are accounted for

COMP rewards are paid **to borrowers**, reducing their net cost. Compound's effective borrow rate is `apyBase − apyReward`, not `apyBase` alone.

| Metric | Aave V3 | Compound V3 |
|---|---|---|
| Mean `apyBase` | 4.87% | 4.99% |
| Mean `apyReward` | — | 0.43% |
| Mean net borrow cost | 4.87% | 4.55% |

Compound is **32 bps cheaper** on a net basis. Aave is the more expensive protocol on **55% of overlapping days** once rewards are factored in, up from 45% on a base-vs-base comparison.

**Rule: compare Aave's `apyBase` against Compound's `apyBase − apyReward` for a fair effective-cost comparison.**

---

### Aave is more volatile; divergences are driven by spikes

Aave's `apyBase` has a higher standard deviation (3.45% vs 2.90%) and a much wider ceiling — Aave's maximum borrow rate reached **56.7%** versus Compound's **22.4%**. This is consistent with Aave holding ~6.6× more TVL: large protocols attract bigger absolute liquidity events, which register as sharper rate spikes on a deeper pool.

The protocols diverge most violently during demand shocks, not in steady-state. Characterising these events is the core of Question 1's analysis.

---

### Missing data is structural, not a quality issue

`apyReward` is null for all 1,163 Aave V3 records — Aave offers no token incentive programme on this pool. It is the only column with missing data; all other fields are 100% complete across both protocols.

---

## Exploratory Data Analysis

**Dataset:** 2,445 daily records × 8 columns across two protocols. Numeric features: `apyBase`, `apy`, `apyReward`, `tvlUsd`. One datetime (`timestamp`), three categoricals (`project`, `symbol`, `chain`). Overlapping window: 1,159 days.

**Key findings:**
- `apy` and `apyBase` are 0.99 correlated for Aave (no rewards) but diverge for Compound — use `apyBase` only in all models.
- `apyReward` is null for 100% of Aave records; structural, not missing at random — do not impute. The correct comparison target for Compound is `apyBase − apyReward` (rewards go to borrowers, reducing net cost by ~0.43% on average).
- `apyBase` is strongly right-skewed with extreme spikes (max 56.7%) — will need log transform or robust scaling before modelling.
- `tvlUsd` is weakly negatively correlated with `apyBase` (−0.17) — higher liquidity coincides with lower rates, consistent with utilisation mechanics, but the signal is weak in isolation.

**Modeling implications:**
- Net borrow cost on Compound (`apyBase − apyReward`) averages 4.55% vs Aave's 4.87% — models comparing protocols must use net cost, not raw base rate.
- Spike episodes (65 Aave days, 73 Compound days above 10%) will likely dominate error metrics; consider training separate models for spike vs non-spike regimes, or use quantile loss.

---

## Feature Engineering

22 features engineered across 6 groups (spread, rate momentum, reward, liquidity, regime, calendar). After correlation and variance filtering, **10 features** were selected for modelling:

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

Both `_vs_net` (Aave base − Compound net cost) and `_vs_base` (Aave base − Compound base) spread variants were engineered. Selection dropped all `_vs_base` variants after they correlated > 0.95 with their `_vs_net` equivalents — confirming that `compound_apyReward` is stable enough over the sample that both definitions carry near-identical information. See `feature_engineering.md` for full selection log.

---

## Model Results Summary

All models and results live in `src/models/`. Each study has a `results.md` with raw output and interpretation.

| Study | Key finding |
|---|---|
| **Q1 main** | Cointegrated (p=0.0). OU half-life 0.95 days. Mean rolling correlation 0.27 — linked long-run, independent short-run. |
| **Q1S1 Persistence** | Today's spread barely predicts tomorrow's (AR(10) R²=0.064). Naive persistence R²=−0.60. |
| **Q1S2 Volatility** | High-vol leads to more narrowing (Mann-Whitney p=0.003). XGBoost AUC=0.635; threshold rule fails. |
| **Q1S3 TVL Shocks** | Spread Granger-causes TVL flows, not vice versa. OLS R²=0.27 — best regression result. Outflows 4× more impactful than inflows. |
| **Q1S4 Post-Spike** | Spreads close *faster* post-spike (half-life 0.87 vs 1.25 days). 92.5% close within 7 days. Hypothesis rejected. |
| **Q1S5 Weekend** | Friday spreads less persistent into Monday (68.5%) than mid-week (80.1%). Saturday is the most stable day. |
| **Q1S6 Lead/Lag** | Neither protocol leads the other at lag 1. Relationship is intermittent (significant in <26% of rolling windows). |
| **Q1S7 Direction** | Logistic regression AUC=0.671. Signal is linear and concentrated in `aave_rate_change_1d` and `spread_vs_net_lag_1d`. |

---

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── ingest.py      # Fetch pool data from DefiLlama API
│   │   ├── loader.py      # Load raw JSON → cleaned DataFrame
│   │   └── quality.py     # Five-check data quality gate
│   ├── features/          # Feature engineering (upcoming)
│   └── models/            # Model training and prediction (upcoming)
├── app/                   # FastAPI + Streamlit (upcoming)
├── tests/                 # Unit tests
├── notebooks/
│   └── eda.ipynb          # Exploratory data analysis
├── data/                  # Raw and processed data (gitignored)
├── models/                # Saved model artefacts (gitignored)
├── requirements.txt
└── setup.py
```

## Setup

```bash
conda activate build-the-death-star
pip install -r requirements.txt
pip install -e .       # makes src/ importable as a package
```

## Data Ingestion

```bash
python src/data/ingest.py
```

Fetches the main-market USDC pools for Aave V3 and Compound V3 from the DefiLlama Yields API and saves raw JSON to `data/raw/`.

## Data Quality

```bash
cd src/data && python quality.py
```

Runs five checks (schema, row count, null rates, value ranges, group distribution) and prints a pass/fail report.

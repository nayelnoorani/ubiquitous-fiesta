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

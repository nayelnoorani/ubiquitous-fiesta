# DeFi Interest Rate Analytics

A research portfolio analyzing Aave and Compound borrow rates on Ethereum. Each project is a self-contained study answering one of the ten questions below.

---

## Projects

| # | Question | Folder |
|---|----------|--------|
| 1 | [How closely do Aave and Compound track each other on the same asset?](#1-cross-protocol-rate-tracking) | [p1_spread_tracking](./p1_spread_tracking) |
| 2 | [How sensitive are borrow rates to changes in utilization?](#2-utilization-curve-sensitivity) | — |
| 3 | [How do rates behave around major market events?](#3-event-driven-rate-behavior) | — |
| 4 | [Are stablecoin borrow rates mean-reverting, and on what timescale?](#4-mean-reversion) | — |
| 5 | [Which asset class shows the highest rate volatility?](#5-cross-asset-volatility) | — |
| 6 | [Can you predict next-day borrow rates?](#6-rate-forecasting) | — |
| 7 | [Do rate spikes predict or follow price volatility?](#7-rate-price-lead-lag) | — |
| 8 | [Is there a day-of-week or time-of-day pattern in rates?](#8-calendar-effects) | — |
| 9 | [How did Aave V3 affect rate behavior compared to V2?](#9-aave-v2-vs-v3) | — |
| 10 | [Is there evidence of strategic rate manipulation by large depositors?](#10-whale-detection) | — |

---

## Question Summaries

### 1. Cross-Protocol Rate Tracking
**[p1_spread_tracking](./p1_spread_tracking)**

For a given asset like USDC, do Aave and Compound borrow rates move in lockstep, or are there persistent spreads? If spreads exist, how long do they last before arbitrage closes them?

### 2. Utilization Curve Sensitivity
Each protocol defines a mathematical utilization curve in its smart contracts. Does the empirical data actually follow that curve, or are there deviations worth explaining?

### 3. Event-Driven Rate Behavior
Pick 3–5 known shocks — the USDC depeg (March 2023), the FTX collapse (November 2022), the ETH Merge (September 2022) — and characterize the rate response: magnitude, duration, and recovery speed.

### 4. Mean Reversion
USDC and DAI rates tend to spike and then normalize. Can this be modeled as a mean-reverting process (e.g. Ornstein-Uhlenbeck), and what is the half-life of a rate shock?

### 5. Cross-Asset Volatility
Compare the distribution of borrow APYs across asset types. Are stablecoin rates more volatile than ETH rates? What does that imply about borrower demand patterns?

### 6. Rate Forecasting
Build a regression or time-series model (ARIMA or gradient boosting) to forecast short-term rates. How much of the variance is explainable from available features?

### 7. Rate-Price Lead-Lag
When ETH borrow rates spike, does that precede or follow a large ETH price move? Is there a lead/lag relationship that could serve as a signal?

### 8. Calendar Effects
DeFi operates 24/7, but human behavior is cyclical. Do rates systematically differ on weekends versus weekdays, or during U.S. trading hours versus off-hours?

### 9. Aave V2 vs V3
Aave V3 launched with efficiency mode and improved risk parameters. Did rates on the same assets become more or less volatile after the migration? Did utilization patterns shift?

### 10. Whale Detection
Large liquidity events — a whale depositing $50M USDC — can sharply move rates. Can these events be identified in the data and their transient impact on the rate curve quantified?

---

## Data Sources

| Source | Link |
|--------|------|
| DefiLlama historical APY (Aave, Compound) | https://defillama.com/yields |
| DefiLlama API | https://yields.llama.fi/chart/{pool-id} |
| Compound interest rate model data (Dune) | https://dune.com/queries/compound |
| Aave V3 reserve data via The Graph | https://thegraph.com/explorer/subgraphs/GQFbb95cE6d8mV989mL5figjaGaKCQB3xqYrr1bRyXqF |
| Compound V3 subgraph | https://thegraph.com/explorer/subgraphs/5nwMCSHaTqG3Kd2gHznbTXEnZ9QNWsssQfbHhDqQSsy1 |

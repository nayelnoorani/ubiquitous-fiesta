**1. How closely do Aave and Compound track each other on the same asset?**
For a given asset like USDC, do the two protocols' borrow rates move in lockstep, or are there persistent spreads? If spreads exist, how long do they last before arbitrage closes them?

---

**2. How sensitive are borrow rates to changes in utilization — and does this match the theoretical model?**
Each protocol defines a mathematical utilization curve in its smart contracts. Does the empirical data actually follow that curve, or are there deviations worth explaining?

---

**3. How do rates behave around major market events?**
Pick 3–5 known shocks — the USDC depeg (March 2023), the FTX collapse (November 2022), the ETH Merge (September 2022) — and characterize the rate response: magnitude, duration, and recovery speed.

---

**4. Are stablecoin borrow rates mean-reverting, and on what timescale?**
USDC and DAI rates tend to spike and then normalize. Can you model this as a mean-reverting process (e.g. Ornstein-Uhlenbeck), and estimate the half-life of a rate shock?

---

**5. Which asset class shows the highest rate volatility — stablecoins, ETH, or wrapped BTC?**
Compare the distribution of borrow APYs across asset types. Are stablecoin rates more volatile than ETH rates? What does that imply about borrower demand patterns?

---

**6. Can you predict next-day borrow rates using utilization, price volatility, and recent rate history?**
Build a simple regression or time-series model (ARIMA or gradient boosting) to forecast short-term rates. How much of the variance is explainable from the available features?

---

**7. Do rate spikes predict or follow price volatility in the underlying asset?**
When ETH borrow rates spike, does that precede or follow a large ETH price move? Is there a lead/lag relationship that could serve as a signal?

---

**8. Is there a day-of-week or time-of-day pattern in interest rates?**
DeFi operates 24/7, but human behavior is cyclical. Do rates systematically differ on weekends versus weekdays, or during U.S. trading hours versus off-hours?

---

**9. How did the introduction of Aave V3 affect rate behavior compared to V2?**
Aave V3 launched with efficiency mode and improved risk parameters. Did rates on the same assets become more or less volatile after the migration? Did utilization patterns shift?

---

**10. Is there evidence of strategic rate manipulation or coordinated large deposits/withdrawals affecting rates?**
Large liquidity events — a whale depositing $50M USDC — can sharply move rates. Can you identify these events in the data and quantify their transient impact on the rate curve?

---

### Defined sequence

Questions **1, 2, and 3** are the most portfolio-friendly to start with — they're descriptive, visually compelling, and require no modeling assumptions. 
Questions **4, 5, and 6** are natural second-phase projects once you're comfortable with the data. 
Questions **7, 8, 9, and 10** are the most original angles and would stand out the most to a technical audience.


## Datasets & Links
Dataset Link
DefiLlama historical APY (Aave, Compound)   https://defillama.com/yields
DefiLlama API (programmatic access)         https://yields.llama.fi/chart/{pool-id}
Compound interest rate model data (Dune)    https://dune.com/queries/compound
Aave V3 reserve data via The Graph          https://thegraph.com/explorer/subgraphs/GQFbb95cE6d8mV989mL5figjaGaKCQB3xqYrr1bRyXqF
Compound V3 subgraph                        https://thegraph.com/explorer/subgraphs/5nwMCSHaTqG3Kd2gHznbTXEnZ9QNWsssQfbHhDqQSsy1
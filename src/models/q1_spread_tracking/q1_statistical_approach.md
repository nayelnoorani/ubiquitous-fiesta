# Q1 Spread Tracking — Statistical Approach

ML models are of limited value for Q1, which is fundamentally a descriptive question about the relationship between two rate series. The following statistical approaches give more direct answers.

---

## Proposed Tests

### 1. Cointegration (Engle-Granger)

**What it tests:** Whether two series that individually wander (non-stationary) share a long-run equilibrium relationship — i.e. they're "tied together" even if each one drifts on its own.

**How it works:** Regress one series on the other (e.g. Aave rate ~ Compound rate). If the residuals of that regression are stationary (tested with ADF, see below), the two series are cointegrated. The intuition is: even if both rates drift up or down over time, the *gap between them* stays bounded. If they're not cointegrated, the spread can wander off indefinitely.

**What the result means:**
- Cointegrated → the protocols are fundamentally linked; spreads are temporary and will close
- Not cointegrated → the rates can drift apart permanently; no guarantee of convergence

**In our context:** This is the single most important test for Q1. A positive result is what we'd expect if arbitrage is functioning — borrowers would migrate to the cheaper protocol, pushing rates back together.

**Deeper reading:** https://en.wikipedia.org/wiki/Cointegration | statsmodels implementation: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html

---

### 2. Spread Stationarity (ADF + KPSS)

**What they test:** Whether a single time series has a stable mean and variance over time (stationary), or whether it drifts and has no tendency to return to a fixed level (non-stationary).

**How they work:**

- **ADF (Augmented Dickey-Fuller):** Tests the null hypothesis that the series has a unit root (i.e. is non-stationary). A small p-value (< 0.05) rejects the null → the series is stationary. Think of a unit root as the series having "no memory" of where it should be — it just keeps drifting.
- **KPSS (Kwiatkowski-Phillips-Schmidt-Shin):** Tests the *opposite* null — that the series *is* stationary. A small p-value rejects stationarity. Running both catches cases where ADF gives a false positive.

**Why run both:** ADF and KPSS have different null hypotheses, so their results can conflict. The four possible combinations:
| ADF | KPSS | Conclusion |
|---|---|---|
| Reject (p < 0.05) | Fail to reject (p > 0.05) | Stationary ✓ |
| Fail to reject | Reject | Non-stationary ✓ |
| Reject | Reject | Trend-stationary (needs differencing) |
| Fail to reject | Fail to reject | Inconclusive |

**What the result means for us:** If `spread_vs_net` is stationary, it has a fixed long-run mean it always returns to — spreads are self-correcting. If non-stationary, spreads can drift indefinitely.

**Deeper reading:** ADF: https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test | KPSS: https://en.wikipedia.org/wiki/KPSS_test | statsmodels ADF: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html | statsmodels KPSS: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html

---

### 3. Ornstein-Uhlenbeck Fit

**What it is:** A mathematical model for a value that randomly fluctuates but is continuously pulled back toward a long-run mean. Originally developed to model the velocity of a particle in a fluid — now widely used in finance to model mean-reverting quantities like interest rate spreads.

**The equation:** `dX = θ(μ − X)dt + σ dW`

In plain English: at each moment, the spread is pushed toward its mean (μ) with a force proportional to how far it has strayed. θ controls how strong that pull is. σ is the random noise.

**What the parameters tell us:**
- **θ (theta):** Mean-reversion speed. High θ = snaps back quickly. Low θ = slow drift back.
- **μ (mu):** The long-run average spread the process reverts to.
- **σ (sigma):** How noisy/volatile the spread is around its mean path.
- **Half-life = ln(2) / θ:** The number of days it takes for a spread shock to decay to half its original size. This is the most interpretable output — e.g. a half-life of 5 days means a 2% spread spike typically narrows to 1% within 5 days.

**How we fit it:** Estimate θ, μ, σ by regressing the daily spread change on the spread level (OLS). The slope gives θ, the intercept gives θμ, and the residual variance gives σ.

**Deeper reading:** https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

---

### 4. Rolling Correlation
Compute Pearson correlation between Aave and Compound rates on a 30-day rolling window. Identifies whether the tracking relationship breaks down during specific periods (e.g. market stress) and how stable it is over time.

### 5. Spread Distribution and Episode Analysis
- Summary statistics: mean, std, skew, kurtosis, percentiles
- Define a spread "episode" as consecutive days where |spread| exceeds a threshold (e.g. 1%)
- Measure: episode count, mean duration, max duration, mean magnitude
- Directly answers how long spreads persist in practical terms

### 6. Cross-Correlation Function (CCF)
Plot cross-correlation between Aave and Compound rate changes at lags −5 to +5. Identifies whether one protocol consistently leads the other and by how many days. Overlaps with Q1S6.

---

## Suggested Execution Order

1. Cointegration → 2. Stationarity → 3. OU fit (logical chain: do they cointegrate? → is the spread stationary? → how fast does it revert?)
4. Rolling correlation — characterise stability of the relationship over time
5. Episode analysis — characterise spread persistence in practical terms
6. CCF — lead/lag as a bonus finding

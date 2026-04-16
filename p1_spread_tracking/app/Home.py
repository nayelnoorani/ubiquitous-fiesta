import streamlit as st

st.set_page_config(
    page_title="Aave vs Compound — Spread Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Aave V3 vs Compound V3")
st.subheader("USDC Borrow Rate Spread Analysis")

st.markdown("""
How closely do Aave and Compound track each other on the same asset?
For USDC on Ethereum, this project examines 1,159 days of borrow rate history to ask:
do the two protocols move in lockstep — and when they diverge, how fast does arbitrage close the gap?
""")

st.markdown("---")

# ── Key findings ──────────────────────────────────────────────────────────────
st.markdown("### Key Findings")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Cointegration", "Confirmed", "p = 0.0",
        help="Engle-Granger test stat = −5.32, clears the 1% critical value (−3.91). "
             "The two protocols share a long-run equilibrium — the spread cannot drift apart permanently.",
    )
    st.metric(
        "Spread half-life", "0.95 days",
        help="Ornstein-Uhlenbeck estimate. Spread shocks decay 50% within a single day on average. "
             "New shocks (σ = 2.7% daily) arrive faster than old ones close.",
    )

with col2:
    st.metric(
        "Mean 30-day rolling correlation", "0.27",
        help="Despite long-run cointegration, the protocols move largely independently day-to-day. "
             "74% of days the rolling correlation sits below 0.50; it reaches −0.70 at times.",
    )
    st.metric(
        "Spread episodes > 1%", "144 events",
        help="Mean duration 3.3 days, max 27 days. 92.5% of episodes close within 7 days. "
             "Almost always calm, occasionally explosive.",
    )

with col3:
    st.metric(
        "Compound net cost advantage", "−32 bps avg",
        help="COMP rewards (avg 0.43%) are paid to borrowers, reducing Compound's effective rate. "
             "Aave is the more expensive protocol on 55% of days once rewards are accounted for.",
    )
    st.metric(
        "Direction prediction AUC", "0.671",
        help="Logistic regression (Q1S7). Spread direction is weakly but genuinely predictable. "
             "Signal is almost entirely captured by today's Aave rate change and spread level. "
             "Tree models do not improve on the linear baseline.",
    )

st.markdown("---")

# ── Dataset ───────────────────────────────────────────────────────────────────
st.markdown("### Dataset")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
| | Aave V3 | Compound V3 |
|---|---|---|
| Pool | Main market | Main market |
| Avg TVL | \\$397M | \\$60M |
| Records | 1,163 | 1,282 |
| Start date | 2023-02-06 | 2022-10-06 |
| **Overlapping window** | **1,159 days** | **1,159 days** |
""")

with col2:
    st.markdown("""
**Source:** DefiLlama Yields API — no authentication required

**Rate comparison rule:**
Aave `apyBase` vs Compound `apyBase − apyReward`

COMP token rewards are paid directly to borrowers, reducing their effective borrow cost.
Comparing raw base rates would overstate Compound's cost by ~43 bps on average.
Aave offers no reward programme on this pool.
""")

st.markdown("---")

# ── Navigation guide ──────────────────────────────────────────────────────────
st.markdown("### Pages")

st.markdown("""
| Page | Content |
|---|---|
| **Explore the Data** | Interactive rate and spread charts, TVL, rolling correlation, episode browser |
| **Model & Statistical Results** | Q1 main study + 7 supplementary analyses — statistics, ML model performance, interpretations |
| **How I Built This** | Data pipeline, feature engineering decisions, modelling choices, stack |
""")

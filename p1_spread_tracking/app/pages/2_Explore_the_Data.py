import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(page_title="Explore the Data", layout="wide")


def placeholder(description: str) -> None:
    st.markdown(
        f'<div style="border:1px dashed rgba(180,180,180,0.3);border-radius:6px;'
        f'padding:2rem;margin:0.5rem 0;text-align:center;'
        f'color:rgba(200,200,200,0.5);font-style:italic;">'
        f'[ {description} ]</div>',
        unsafe_allow_html=True,
    )


# ── Section 1: Opening Hook ───────────────────────────────────────────────────

st.markdown("## Compound looks more expensive. It isn't.")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**The headline view**")
    placeholder(
        "Left panel — two large numbers: Aave 4.87% / Compound 4.99% · "
        "Label: Mean base borrow rate · Implication: Compound looks pricier"
    )

with col_right:
    st.markdown("**The full picture**")
    placeholder(
        "Right panel — two large numbers: Aave more expensive on 55% of days / "
        "Compound more expensive on 45% of days · "
        "Label: Days cheaper, net of COMP rewards · "
        "Implication: Aave is the pricier protocol most of the time"
    )

st.markdown(
    "Compound pays COMP token rewards directly to borrowers, reducing their effective cost by "
    "43 basis points on average. The protocol that looks more expensive by headline rate is "
    "actually the cheaper place to borrow on most days."
)

st.markdown("---")

# ── Section 2: The Two Rate Series ───────────────────────────────────────────

st.markdown("## Same asset. Same chain. Very different moments.")

st.markdown(
    "Sam Savage called it the Flaw of Averages — a plan built on average assumptions will fail "
    "on average. Aave's mean borrow rate is 4.87%. Its maximum was 56.7%. Use the toggle to see "
    "why the average is the wrong number to watch."
)

col_t1, col_t2 = st.columns(2)
with col_t1:
    rate_toggle = st.radio(
        "Rate view",
        ["Base rates", "Net cost (incl. COMP rewards)"],
        horizontal=True,
    )
with col_t2:
    scale_toggle = st.radio(
        "Y-axis scale",
        ["Capped (0–15%)", "Full scale"],
        horizontal=True,
    )

placeholder(
    "Line chart — Aave base rate vs Compound (base or net depending on toggle) · "
    "Shaded area between the two lines showing the spread · "
    "Capped view: Y-axis max 15%, note showing count of days exceeding range · "
    "Full scale: reveals spike magnitude · "
    "Annotations: Aave max spike 56.7%, Compound max spike 22.4%"
)

st.markdown("---")

# ── Section 3: The Spread Distribution ───────────────────────────────────────

st.markdown("## The spread is almost nothing. Until it isn't.")

st.markdown(
    "The median daily spread between Aave and Compound is 0.19% — on most days, the two "
    "protocols are essentially identical. But a kurtosis of 126.7 means the tails are extreme: "
    "when spreads move, they move violently."
)

col_hist, col_stats = st.columns([3, 2])

with col_hist:
    placeholder(
        "Histogram of daily spread values with a normal distribution overlay · "
        "The normal curve appears as a thin symmetrical bell barely visible above zero "
        "compared to the actual spike-dominated shape · "
        "Two callouts: Median 0.19% and Kurtosis 126.7 (note: normal distribution = 3)"
    )

with col_stats:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("### **144**")
    st.caption("episodes above 1%")
    st.markdown("### **3.3 days**")
    st.caption("mean episode duration")
    st.markdown("### **27 days**")
    st.caption("longest episode")
    st.markdown("### **3.9%**")
    st.caption("mean peak spread")
    st.markdown("### **54.2%**")
    st.caption("maximum peak spread")

st.markdown("**Episode browser**")

col_f1, col_f2 = st.columns(2)
with col_f1:
    min_peak = st.slider("Minimum peak spread (%)", 1.0, 20.0, 1.0, 0.5)
with col_f2:
    min_dur = st.slider("Minimum duration (days)", 1, 14, 1)

placeholder(
    f"Filterable table of spread episodes where |spread| > 1% · "
    f"Filtered to: peak >= {min_peak}%, duration >= {min_dur} day(s) · "
    f"Columns: start date, duration, peak spread"
)

st.markdown("---")

# ── Section 4: The Rolling Correlation Puzzle ─────────────────────────────────

st.markdown("## Aave & Compound rates: Linked over years. Strangers in any given month.")

st.markdown(
    "Aave and Compound offer the same product — USDC borrowing — on the same blockchain, "
    "so it is reasonable to expect their rates track closely. Cointegration confirms a long-run "
    "structural link: the two series share an equilibrium they never permanently escape. Yet a "
    "mean 30-day rolling correlation of 0.27 says they spend most of their time ignoring each "
    "other getting there."
)

col_chart, col_stats = st.columns([3, 1])

with col_chart:
    placeholder(
        "Line chart of 30-day rolling Pearson correlation over full 1,159-day window · "
        "Y-axis: −1 to +1 · "
        "Three horizontal reference lines: 0 (labeled), 0.27 (mean, labeled), 0.50 (moderate correlation) · "
        "Shaded red: periods where correlation is negative · "
        "Shaded green: periods where correlation exceeds 0.50"
    )

with col_stats:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("### **0.27**")
    st.caption("Mean")
    st.markdown("### **−0.70**")
    st.caption("Min")
    st.markdown("### **0.88**")
    st.caption("Max")
    st.markdown("### **74.3%**")
    st.caption("% of days below 0.50")

st.markdown("---")

# ── Section 5: Data Quality Note ─────────────────────────────────────────────

st.caption(
    "Aave's reward column is null for all 1,163 records — not missing data, but a structural "
    "fact: Aave runs no token incentive programme on this pool."
)

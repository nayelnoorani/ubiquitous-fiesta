import streamlit as st

st.set_page_config(
    page_title="Aave vs Compound — Spread Analysis",
    page_icon="📊",
    layout="wide",
)


def placeholder(description: str) -> None:
    st.markdown(
        f'<div style="border:1px dashed rgba(180,180,180,0.3);border-radius:6px;'
        f'padding:2rem;margin:0.5rem 0;text-align:center;'
        f'color:rgba(200,200,200,0.5);font-style:italic;">'
        f'[ {description} ]</div>',
        unsafe_allow_html=True,
    )


# ── Section 1: Opening Frame ──────────────────────────────────────────────────

col_text, col_visual = st.columns([2, 3])

with col_text:
    st.markdown(
        "<p style='font-size:1.25rem;line-height:1.7;padding-top:1rem;'>"
        "I've spent ten years analyzing interest rates in traditional finance. "
        "I came to DeFi with five reasonable hypotheses — the data overturned four of them."
        "</p>"
        "<p style='font-size:1rem;line-height:1.7;color:rgba(250,250,250,0.55);font-style:italic;'>"
        "Here's where each one landed."
        "</p>",
        unsafe_allow_html=True,
    )

with col_visual:
    st.markdown("""
| Hypothesis | Verdict |
|:---|:---:|
| Large liquidity flows drive rate divergence between protocols | ❌ |
| Spreads are harder to close after a rate spike | ❌ |
| Friday spreads get locked in over the weekend | ❌ |
| The bigger protocol moves first; the smaller one follows | ❌ |
| Yesterday's spread tells you something about tomorrow's | ⚠️ |
""")

st.markdown("---")

# ── Section 2: What Was Actually True ────────────────────────────────────────

st.markdown("## \"The market is more efficient than I expected, and stranger.\"")
st.markdown("*Arbitrage is alive. It's just constantly overwhelmed.*")

st.markdown(
    "The hypotheses didn't fail because the market is broken — "
    "they failed because it's faster and smarter than expected."
)

st.markdown("<br>", unsafe_allow_html=True)

# Card 1 — stat LEFT, text RIGHT
c_stat, c_text = st.columns([1, 2])
with c_stat:
    st.markdown("# **p = 0.0**")
    st.caption("Engle-Granger cointegration test")
with c_text:
    st.markdown(
        "Aave and Compound share a long-run equilibrium that neither protocol can permanently escape. "
        "No matter how far rates diverge, the market always closes the gap."
    )

st.markdown("<br>", unsafe_allow_html=True)

# Card 2 — stat RIGHT, text LEFT
c_text, c_stat = st.columns([2, 1])
with c_text:
    st.markdown(
        "The typical spread shock loses half its size within a single trading day. "
        "That's not slow arbitrage — that's near-instantaneous price discovery for a market that never closes."
    )
with c_stat:
    st.markdown("# **0.95 days**")
    st.caption("Ornstein-Uhlenbeck half-life")

st.markdown("<br>", unsafe_allow_html=True)

# Card 3 — stat LEFT, text RIGHT
c_stat, c_text = st.columns([1, 2])
with c_stat:
    st.markdown("# **0.19%**")
    st.caption("Median daily spread")
with c_text:
    st.markdown(
        "On a typical day, borrowing on Aave costs virtually the same as borrowing on Compound. "
        "The volatility that dominates the statistics lives almost entirely in rare, short-lived spikes."
    )

st.markdown("<br>", unsafe_allow_html=True)

# Card 4 — stat RIGHT, text LEFT
c_text, c_stat = st.columns([2, 1])
with c_text:
    st.markdown(
        "When spreads do blow out, they almost always close within a week. "
        "The larger the gap, the stronger the pull back toward equilibrium — "
        "stress doesn't break the market, it accelerates it."
    )
with c_stat:
    st.markdown("# **92.5%**")
    st.caption("Spike episodes resolved within 7 days")

st.markdown("---")

# ── Section 3: Five Things I Was Wrong About ──────────────────────────────────

st.markdown("## Five Things I Was Wrong About")
st.markdown("*Well, I got one. Kinda.*")

st.markdown(
    "Each card is a belief I held going in, stated exactly as I held it — "
    "the verdict is upfront, the full evidence is on page 3."
)

st.markdown("<br>", unsafe_allow_html=True)

cards = [
    {
        "hypothesis": "Large liquidity flows drive rate divergence between protocols",
        "verdict": "❌",
        "stat": "4×",
        "finding": "Outflows are 4× more impactful than inflows — and the causality runs backwards.",
    },
    {
        "hypothesis": "Spreads are harder to close after a rate spike",
        "verdict": "❌",
        "stat": "0.87 days",
        "finding": "Spreads close faster post-spike, not slower. Stress accelerates arbitrage.",
    },
    {
        "hypothesis": "Friday spreads get locked in over the weekend",
        "verdict": "❌",
        "stat": "68.5%",
        "finding": "Friday is the least sticky transition of the week — DeFi doesn't close for the weekend.",
    },
    {
        "hypothesis": "The bigger protocol moves first; the smaller one follows",
        "verdict": "❌",
        "stat": "R² = 0.0002",
        "finding": "Yesterday's Aave move explains essentially nothing about today's Compound move.",
    },
    {
        "hypothesis": "Yesterday's spread tells you something about tomorrow's",
        "verdict": "⚠️",
        "stat": "6.4%",
        "finding": "The level is nearly unpredictable. But the direction has a weak signal worth knowing.",
    },
]

cols = st.columns(5)
for col, card in zip(cols, cards):
    with col:
        st.markdown(f"**{card['hypothesis']}**")
        st.markdown(f"### {card['verdict']}")
        st.markdown(f"## {card['stat']}")
        st.markdown(card["finding"])
        st.page_link("pages/3_Model_Results.py", label="See full analysis →")

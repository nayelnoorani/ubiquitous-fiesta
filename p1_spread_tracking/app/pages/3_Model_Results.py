import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(page_title="Model & Statistical Results", layout="wide")


def placeholder(description: str) -> None:
    st.markdown(
        f'<div style="border:1px dashed rgba(180,180,180,0.3);border-radius:6px;'
        f'padding:2rem;margin:0.5rem 0;text-align:center;'
        f'color:rgba(200,200,200,0.5);font-style:italic;">'
        f'[ {description} ]</div>',
        unsafe_allow_html=True,
    )


def section_label(label: str) -> None:
    st.markdown(
        f'<p style="font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;'
        f'color:rgba(200,200,200,0.5);margin-bottom:0.25rem;">{label}</p>',
        unsafe_allow_html=True,
    )


# ── Page Introduction ─────────────────────────────────────────────────────────

st.title("Five Things I Was Wrong About")
st.markdown(
    "*Each hypothesis gets the same treatment: what I assumed, how I tested it, "
    "what the data said, and why it matters.*"
)

st.markdown("---")

# ── Round 1 — TVL Flows Drive Divergence ─────────────────────────────────────

st.markdown("### Round 1 — TVL Flows Drive Divergence")

col_text, col_visual = st.columns(2)

with col_text:
    section_label("Assumed")
    st.markdown(
        "Capital rushing in or out of a protocol drives rate divergence — inflows mean lower "
        "rates, outflows mean higher rates, and the spread widens as a result."
    )

    section_label("Tested")
    st.markdown(
        "Granger causality in both directions — does TVL change precede the spread, or does "
        "the spread precede TVL change?"
    )

    section_label("Found")
    st.markdown(
        "The causality runs backwards. Spreads Granger-cause TVL flows (F=593 at lag 1, p=0.0). "
        "TVL barely correlates with the spread on the same day (r=0.10). When TVL shocks do "
        "matter, outflows are 4× more impactful than inflows."
    )

    section_label("Why it matters")
    st.markdown(
        "Borrowers respond to price signals, not the other way around. The spread opens first — "
        "then capital migrates. Arbitrage is the mechanism, not the cause."
    )

with col_visual:
    placeholder(
        "Left panel: Scatter plot of same-day TVL change vs spread change — "
        "shows the weak contemporaneous relationship (r=0.10)"
    )
    placeholder(
        "Right panel: Directional diagram showing the Granger causality arrow "
        "running from spread → TVL, not TVL → spread"
    )

st.markdown("---")

# ── Round 2 — Post-Spike Spreads Close Slower ─────────────────────────────────

st.markdown("### Round 2 — Post-Spike Spreads Close Slower")

col_text, col_visual = st.columns(2)

with col_text:
    section_label("Assumed")
    st.markdown(
        "Market stress impairs arbitrage — dislocated borrowers, uncertain rates, reduced "
        "activity. Spreads take longer to close after a rate spike."
    )

    section_label("Tested")
    st.markdown(
        "Ornstein-Uhlenbeck fit split by regime — spike window (days since spike ≤ 7) vs "
        "steady state. Survival analysis on 146 detected spike episodes."
    )

    section_label("Found")
    st.markdown(
        "The opposite. Half-life in the spike window is 0.87 days vs 1.25 days in steady state. "
        "Spreads close faster post-spike, not slower. 92.5% of episodes resolve within 7 days; "
        "median closure is 2 days."
    )

    section_label("Why it matters")
    st.markdown(
        "Arbitrage force is proportional to spread size. The larger the gap, the stronger the "
        "pull. Stress doesn't impair the market — it activates it harder."
    )

with col_visual:
    placeholder(
        "Left panel: Side-by-side OU half-life comparison — two large numbers, "
        "spike window 0.87 days vs steady state 1.25 days, stark and immediate"
    )
    placeholder(
        "Right panel: Survival curve showing % of spike episodes still open by day — "
        "drops steeply, 92.5% closed by day 7 clearly visible"
    )

st.markdown("---")

# ── Round 3 — Friday Spreads Get Locked In ────────────────────────────────────

st.markdown("### Round 3 — Friday Spreads Get Locked In Over the Weekend")

col_text, col_visual = st.columns(2)

with col_text:
    section_label("Assumed")
    st.markdown(
        "Reduced weekend activity means fewer arbitrageurs watching the market. Spreads opened "
        "on Friday persist through Saturday and Sunday, closing only when traders return on Monday."
    )

    section_label("Tested")
    st.markdown(
        "Mann-Whitney U comparing weekend vs weekday spread levels. Friday→Monday persistence "
        "rate vs mid-week transition rates."
    )

    section_label("Found")
    st.markdown(
        "Wrong in both direction and magnitude. Friday→Monday persistence is 68.5% — lower than "
        "the mid-week rate of 80.1%. Saturday is the most stable day of the week. "
        "DeFi doesn't close on Friday."
    )

    section_label("Why it matters")
    st.markdown(
        "There is no human trading halt in DeFi. Protocols run continuously, rates update "
        "continuously, borrowers respond continuously. Weekend effects from traditional markets "
        "don't transplant."
    )

with col_visual:
    placeholder(
        "Left panel: Bar chart of persistence rate by day of week — Saturday visually the "
        "highest, Friday and Thursday the lowest, immediately contradicting the hypothesis"
    )
    placeholder(
        "Right panel: Two large numbers side by side — Fri→Mon 68.5% vs Tue→Wed 80.1%"
    )

st.markdown("---")

# ── Round 4 — The Bigger Protocol Moves First ─────────────────────────────────

st.markdown("### Round 4 — The Bigger Protocol Moves First")

col_text, col_visual = st.columns(2)

with col_text:
    section_label("Assumed")
    st.markdown(
        "Aave holds 6.6× more TVL than Compound. Larger pools attract bigger liquidity events, "
        "so Aave should move first and Compound should follow — the smaller protocol taking its "
        "cue from the dominant one."
    )

    section_label("Tested")
    st.markdown(
        "Cross-correlation function at lags −5 to +5. Granger causality in both directions at "
        "lags 1–5. Rolling 180-day Granger windows to test stability. Cross-lagged regressions."
    )

    section_label("Found")
    st.markdown(
        "Neither protocol leads the other at lag 1. Granger significance only appears at lags "
        "2–4, in both directions, and only in 12–26% of rolling windows. Yesterday's Aave move "
        "explains essentially nothing about today's Compound move (R²=0.0002). Each protocol is "
        "primarily driven by its own prior moves."
    )

    section_label("Why it matters")
    st.markdown(
        "There is no exploitable lead/lag signal. Both protocols respond to the same external "
        "conditions independently. Size doesn't confer timing advantage in a market with no "
        "closing bell."
    )

with col_visual:
    placeholder(
        "Left panel: CCF chart at lags −5 to +5 — all values small and noisy, no clean peak "
        "in either direction, visually confirming the absence of structure"
    )
    placeholder(
        "Right panel: R² = 0.0002 displayed large with label "
        "'variance in Compound's move explained by yesterday's Aave move'"
    )

st.markdown("---")

# ── Round 5 — Yesterday's Spread Tells You Something ─────────────────────────

st.markdown("### Round 5 — Yesterday's Spread Tells You Something About Tomorrow's")

col_text, col_visual = st.columns(2)

with col_text:
    section_label("Assumed")
    st.markdown(
        "With mean reversion this strong and a half-life under a day, autocorrelation should "
        "carry useful predictive power. Yesterday's spread should tell you something about "
        "tomorrow's."
    )

    section_label("Tested")
    st.markdown(
        "AR(10) model for next-day spread level. Logistic regression and tree models for "
        "next-day spread direction. ACF/PACF analysis."
    )

    section_label("Found")
    st.markdown(
        "Level prediction fails — AR(10) explains only 6.4% of next-day variance. Naive "
        "persistence is worse than predicting the mean every day. But direction is weakly "
        "predictable: logistic regression achieves AUC=0.671. The signal is almost entirely "
        "two features — today's Aave rate change and today's spread level. Tree models add nothing."
    )

    section_label("Why it matters")
    st.markdown(
        "The spread reverts fast but new shocks arrive faster. What little signal exists is "
        "linear and concentrated in mean reversion — not a tradeable edge, but an honest one."
    )

with col_visual:
    placeholder(
        "Left panel: Two large numbers — AR(10) R²=6.4% (level) vs Logistic AUC=0.671 "
        "(direction) — the contrast between the two tells the story"
    )
    placeholder(
        "Right panel: Feature importance bar chart from the logistic regression — two bars "
        "dominate (Aave rate change, spread lag), the rest are noise"
    )

st.markdown("---")

# ── Closing Section ───────────────────────────────────────────────────────────

st.markdown("## Four wrong. One partial. One honest conclusion.")
st.markdown(
    "*This market is structurally sound, operationally noisy, and largely unpredictable "
    "in the short run — which is exactly what efficient markets look like.*"
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    "The cointegration holds, the half-life is under a day, and 92.5% of spike episodes "
    "resolve within a week — the market works. But working efficiently and being predictable "
    "are different things: four of five hypotheses failed precisely because the market was "
    "faster and smarter than expected. The one partial signal that survived — direction at "
    "AUC=0.671 — is linear, interpretable, and concentrated in mean reversion. "
    "That's not a trading strategy. It's a description of how the market heals itself."
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from utils.data import inject_global_css

st.set_page_config(
    page_title="Aave vs Compound — Spread Analysis",
    page_icon="📊",
    layout="wide",
)

inject_global_css()


# ── Section 1: Opening Frame ──────────────────────────────────────────────────

col_text, col_visual = st.columns([2, 3])

with col_text:
    st.markdown(
        "<p style='font-size:1.7rem !important;font-weight:700;line-height:1.7;padding-top:1rem;'>"
        "I've spent ten years analyzing interest rates in traditional finance. "
        "I came to DeFi with five reasonable hypotheses — the data overturned four of them."
        "</p>"
        "<p style='font-size:1rem;line-height:1.7;color:rgba(250,250,250,0.55);font-style:italic;'>"
        "Here's where each one landed."
        "</p>",
        unsafe_allow_html=True,
    )

with col_visual:
    st.markdown(
        """
        <table style='width:100%;border-collapse:collapse;font-size:0.95rem;'>
          <thead>
            <tr>
              <th style='text-align:left;padding:0.5rem 0.75rem;
              color:rgba(250,250,250,0.4);font-weight:500;font-size:0.8rem;
              letter-spacing:0.05em;text-transform:uppercase;
              border-bottom:1px solid rgba(255,255,255,0.1);'>Hypothesis</th>
              <th style='text-align:center;padding:0.5rem 0.75rem;
              color:rgba(250,250,250,0.4);font-weight:500;font-size:0.8rem;
              letter-spacing:0.05em;text-transform:uppercase;
              border-bottom:1px solid rgba(255,255,255,0.1);'>Verdict</th>
            </tr>
          </thead>
          <tbody>
            <tr style='background:rgba(220,60,60,0.07);'>
              <td style='padding:0.55rem 0.75rem;'>Large liquidity flows drive rate divergence between protocols</td>
              <td style='text-align:center;padding:0.55rem 0.75rem;'>❌</td>
            </tr>
            <tr style='background:rgba(220,60,60,0.07);'>
              <td style='padding:0.55rem 0.75rem;'>Spreads are harder to close after a rate spike</td>
              <td style='text-align:center;padding:0.55rem 0.75rem;'>❌</td>
            </tr>
            <tr style='background:rgba(220,60,60,0.07);'>
              <td style='padding:0.55rem 0.75rem;'>Friday spreads get locked in over the weekend</td>
              <td style='text-align:center;padding:0.55rem 0.75rem;'>❌</td>
            </tr>
            <tr style='background:rgba(220,60,60,0.07);'>
              <td style='padding:0.55rem 0.75rem;'>The bigger protocol moves first; the smaller one follows</td>
              <td style='text-align:center;padding:0.55rem 0.75rem;'>❌</td>
            </tr>
            <tr style='background:rgba(240,178,122,0.08);'>
              <td style='padding:0.55rem 0.75rem;'>Today's spread tells you something about tomorrow's spread</td>
              <td style='text-align:center;padding:0.55rem 0.75rem;'>⚠️</td>
            </tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Section 2: What Was Actually True ────────────────────────────────────────

st.markdown("## \"The market is more efficient than I expected, and stranger.\"")
st.markdown("*Arbitrage is alive. It's just constantly overwhelmed.*")

st.markdown(
    "The hypotheses didn't fail because the market is broken — "
    "they failed because it's faster and smarter than expected."
)

CARD_STYLE = (
    "display:flex;align-items:center;gap:2rem;padding:1.25rem 1.5rem;"
    "border:1px solid rgba(255,255,255,0.08);border-radius:8px;margin-bottom:0.75rem;"
)
STAT_STYLE = "font-size:3.9rem;font-weight:700;margin:0;line-height:1;"
LABEL_STYLE = "font-size:1.05rem;color:rgba(250,250,250,0.55);margin:0.1rem 0 0;"
TEXT_STYLE = "font-size:1rem;line-height:1.6;margin:0;"

st.markdown(
    f"<div style='{CARD_STYLE}'>"
    f"<div style='flex:1;'>"
    f"<p style='{STAT_STYLE}'>p = 0.0</p>"
    f"<p style='{LABEL_STYLE}'>Engle-Granger cointegration test</p>"
    f"</div>"
    f"<div style='flex:2;'><p style='{TEXT_STYLE}'>"
    "Aave and Compound share a long-run equilibrium that neither protocol can permanently escape. "
    "No matter how far rates diverge, the market always closes the gap."
    f"</p></div></div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='{CARD_STYLE}'>"
    f"<div style='flex:2;'><p style='{TEXT_STYLE}'>"
    "The typical spread shock loses half its size within a single trading day. "
    "That's not slow arbitrage — that's near-instantaneous price discovery for a market that never closes."
    f"</p></div>"
    f"<div style='flex:1;text-align:right;'>"
    f"<p style='{STAT_STYLE}'>0.95 days</p>"
    f"<p style='{LABEL_STYLE}'>Ornstein-Uhlenbeck half-life</p>"
    f"</div></div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='{CARD_STYLE}'>"
    f"<div style='flex:1;'>"
    f"<p style='{STAT_STYLE}'>0.19%</p>"
    f"<p style='{LABEL_STYLE}'>Median daily spread</p>"
    f"</div>"
    f"<div style='flex:2;'><p style='{TEXT_STYLE}'>"
    "On a typical day, borrowing on Aave costs virtually the same as borrowing on Compound. "
    "The volatility that dominates the statistics lives almost entirely in rare, short-lived spikes."
    f"</p></div></div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='{CARD_STYLE}'>"
    f"<div style='flex:2;'><p style='{TEXT_STYLE}'>"
    "When spreads do blow out, they almost always close within a week. "
    "The larger the gap, the stronger the pull back toward equilibrium — "
    "stress doesn't break the market, it accelerates it."
    f"</p></div>"
    f"<div style='flex:1;text-align:right;'>"
    f"<p style='{STAT_STYLE}'>92.5%</p>"
    f"<p style='{LABEL_STYLE}'>Spike episodes resolved within 7 days</p>"
    f"</div></div>",
    unsafe_allow_html=True,
)

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
        "hypothesis": "Today's spread tells you something about tomorrow's spread",
        "verdict": "⚠️",
        "stat": "6.4%",
        "finding": "The level is nearly unpredictable. But the direction has a weak signal worth knowing.",
    },
]

cols = st.columns(5)
for col, card in zip(cols, cards):
    is_warning = card["verdict"] == "⚠️"
    border_color = "rgba(240,178,122,0.45)" if is_warning else "rgba(220,60,60,0.35)"
    bg_color = "rgba(240,178,122,0.05)" if is_warning else "rgba(220,60,60,0.05)"

    with col:
        st.markdown(
            f"""
            <div style='border:1px solid {border_color};border-radius:8px;
            background:{bg_color};padding:1rem;margin-bottom:0.5rem;
            min-height:280px;display:flex;flex-direction:column;'>
              <div style='min-height:5rem;'>
                <p style='font-size:0.9rem;color:rgba(250,250,250,0.6);
                margin:0;line-height:1.4;'>{card['hypothesis']}</p>
              </div>
              <p style='font-size:1.1rem;margin:0.4rem 0 0.15rem;text-align:center;'>{card['verdict']}</p>
              <p style='font-size:2rem;font-weight:700;margin:0 0 0.5rem;
              line-height:1;text-align:center;'>{card['stat']}</p>
              <p style='font-size:0.9rem;color:rgba(250,250,250,0.7);margin:0 0 0.75rem;
              line-height:1.4;'>{card['finding']}</p>
              <a href='/Model_Results' style='font-size:0.85rem;color:rgba(250,250,250,0.45);
              text-decoration:none;margin-top:auto;'>See full analysis →</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

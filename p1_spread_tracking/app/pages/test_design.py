import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.data import (
    AAVE_COLOR, COMPOUND_COLOR, SPREAD_COLOR,
    big_number, get_episodes, inject_global_css, load_wide,
)

st.set_page_config(page_title="Design Test", layout="wide")

inject_global_css()

# Hide this page from sidebar navigation
st.markdown(
    "<style>[data-testid='stSidebarNav'] a[href*='test_design']{display:none!important;}</style>",
    unsafe_allow_html=True,
)

wide = load_wide()

# ── Shared style ──────────────────────────────────────────────────────────────

PANEL_BG = "rgba(26,29,35,0.0)"
GRID_COLOR = "rgba(255,255,255,0.06)"
AXIS_LINE_COLOR = "rgba(255,255,255,0.2)"
AXIS_TICK_COLOR = "rgba(255,255,255,0.85)"


def base_layout(fig: go.Figure, yrange=None, height: int = 280) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font_color="#FAFAFA",
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="closest",
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR, linecolor=AXIS_LINE_COLOR, tickfont_color=AXIS_TICK_COLOR,
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR, linecolor=AXIS_LINE_COLOR, tickfont_color=AXIS_TICK_COLOR,
        range=yrange,
    )
    return fig


def section_label(label: str) -> None:
    st.markdown(
        f'<p style="font-size:0.85rem;letter-spacing:0.1em;text-transform:uppercase;'
        f'color:rgba(200,200,200,0.5);margin-bottom:0.25rem;">{label}</p>',
        unsafe_allow_html=True,
    )


def two_stats(
    lv: str, ll: str, ls: str,
    rv: str, rl: str, rs: str,
    la: str = "rgba(240,178,122,0.7)",
    ra: str = "rgba(255,255,255,0.15)",
    dim_right: bool = True,
) -> None:
    r_opacity = "opacity:0.55;" if dim_right else ""
    st.markdown(
        f"<div style='display:flex;gap:1.5rem;margin-bottom:0.75rem;'>"
        f"<div style='flex:1;padding:1rem;border-left:3px solid {la};'>"
        f"<div style='font-size:2.6rem;font-weight:700;margin:0;line-height:1;'>{lv}</div>"
        f"<div style='font-size:1.05rem;color:rgba(250,250,250,0.55);margin:0.05rem 0 0;'>{ll}</div>"
        f"<div style='font-size:0.9rem;color:rgba(250,250,250,0.35);margin:0;'>{ls}</div>"
        f"</div>"
        f"<div style='flex:1;padding:1rem;border-left:3px solid {ra};'>"
        f"<div style='font-size:2.6rem;font-weight:700;margin:0;line-height:1;{r_opacity}'>{rv}</div>"
        f"<div style='font-size:1.05rem;color:rgba(250,250,250,0.55);margin:0.05rem 0 0;'>{rl}</div>"
        f"<div style='font-size:0.9rem;color:rgba(250,250,250,0.35);margin:0;'>{rs}</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

col_text, col_visual = st.columns([2, 3])

with col_text:
    st.markdown(
        "<style>#opening-text{font-size:1.6rem!important;font-weight:700!important;line-height:1.4!important;}</style>"
        "<div id='opening-text' style='padding-top:1rem;'>"
        "I've spent ten years analyzing interest rates in traditional finance. "
        "I came to DeFi with five reasonable hypotheses — the data overturned four of them."
        "</div>"
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
STAT_STYLE = "font-size:2.6rem;font-weight:700;margin:0;line-height:1;"
LABEL_STYLE = "font-size:1.05rem;color:rgba(250,250,250,0.55);margin:0.05rem 0 0;"
TEXT_STYLE = "font-size:1rem;line-height:1.6;margin:0;"

st.markdown(
    f"<div style='{CARD_STYLE}'>"
    f"<div style='flex:1;'>"
    f"<div style='{STAT_STYLE}'>p = 0.0</div>"
    f"<div style='{LABEL_STYLE}'>Engle-Granger cointegration test</div>"
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
    f"<div style='{STAT_STYLE}'>0.95 days</div>"
    f"<div style='{LABEL_STYLE}'>Ornstein-Uhlenbeck half-life</div>"
    f"</div></div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='{CARD_STYLE}'>"
    f"<div style='flex:1;'>"
    f"<div style='{STAT_STYLE}'>0.19%</div>"
    f"<div style='{LABEL_STYLE}'>Median daily spread</div>"
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
    f"<div style='{STAT_STYLE}'>92.5%</div>"
    f"<div style='{LABEL_STYLE}'>Spike episodes resolved within 7 days</div>"
    f"</div></div>",
    unsafe_allow_html=True,
)

st.markdown("---")

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
              <div style='min-height:6rem;'>
                <div style='font-size:0.9rem;color:rgba(250,250,250,0.6);
                margin:0;line-height:1.4;'>{card['hypothesis']}</div>
              </div>
              <div style='font-size:0.75rem;margin:-1.5rem 0 0.15rem;text-align:center;'>{card['verdict']}</div>
              <div style='font-size:2rem;font-weight:700;margin:0 0 0.5rem;
              line-height:1;text-align:center;'>{card['stat']}</div>
              <div style='font-size:0.9rem;color:rgba(250,250,250,0.7);margin:0 0 0.75rem;
              line-height:1.4;'>{card['finding']}</div>
              <a href='/Model_Results' style='font-size:0.85rem;color:#FAFAFA;
              text-decoration:none;margin-top:auto;'>See full analysis →</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORE THE DATA
# ══════════════════════════════════════════════════════════════════════════════

st.title("Explore the Data")
st.markdown(
    "*The verdicts are on page 1. Here's the data that made the hypotheses seem reasonable "
    "in the first place.*"
)

st.markdown("---")

st.markdown("## Compound looks more expensive. It isn't.")

_HDR = "font-size:1.5rem;letter-spacing:0.08em;text-transform:uppercase;color:rgba(250,250,250,0.45);margin:0 0 0.2rem;"
_NUM = "font-size:2.6rem;font-weight:700;margin:0;line-height:1;"
_LBL = "font-size:1.05rem;color:rgba(250,250,250,0.55);margin:0.1rem 0 0;"
_SUB = "font-size:1rem;color:rgba(250,250,250,0.45);margin:0;"
_CAP = "font-size:0.9rem;color:rgba(250,250,250,0.45);margin:0.5rem 0 0;"

st.markdown(
    f"""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:3rem;'>
      <div>
        <div style='{_HDR}'>The headline view</div>
        <div style='display:flex;gap:2.5rem;'>
          <div>
            <div style='{_NUM}'>4.87%</div>
            <div style='{_LBL}'>Aave mean borrow rate</div>
          </div>
          <div>
            <div style='{_NUM}'>4.99%</div>
            <div style='{_LBL}'>Compound mean borrow rate</div>
          </div>
        </div>
        <div style='{_CAP}'>Base rates, no rewards adjustment — Compound appears more expensive</div>
      </div>
      <div>
        <div style='{_HDR}'>The full picture</div>
        <div style='display:flex;gap:2.5rem;'>
          <div>
            <div style='{_NUM}'>45%</div>
            <div style='{_LBL}'>of days Aave costs more</div>
            <div style='{_SUB}'>base rates only</div>
          </div>
          <div>
            <div style='{_NUM}'>55%</div>
            <div style='{_LBL}'>of days Aave costs more</div>
            <div style='{_SUB}'>net of COMP rewards</div>
          </div>
        </div>
        <div style='{_CAP}'>Once rewards are factored in, Aave is the pricier protocol most of the time</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='margin-top:1.5rem;margin-bottom:0.25rem;'>"
    "Compound pays COMP token rewards directly to borrowers, reducing their effective cost by "
    "43 basis points on average — making it the cheaper place to borrow on most days, despite the "
    "higher headline rate. That's why every spread in this analysis is calculated net of rewards, "
    "not off the headline rate."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown("## Same asset. Same chain. Very different moments.")

st.markdown(
    "Aave's mean borrow rate is 4.87%. Its maximum was 56.7%. A mean built from a distribution "
    "that wide isn't much of a guide — the chart defaults to a capped view for readability, "
    "but the full scale is the honest picture."
)

col_t1, col_t2 = st.columns(2)
with col_t1:
    rate_toggle = st.radio("Rate view", ["Base rates", "Net cost (incl. COMP rewards)"], horizontal=True)
with col_t2:
    scale_toggle = st.radio("Y-axis scale", ["Capped (0–15%)", "Full scale"], horizontal=True)

use_net = rate_toggle == "Net cost (incl. COMP rewards)"
use_full = scale_toggle == "Full scale"

compound_y = wide["compound_net"] if use_net else wide["compound_apyBase"]
compound_label = "Compound net cost" if use_net else "Compound base rate"
dates = wide.index

above15 = int(((wide["aave_apyBase"] > 15) | (compound_y > 15)).sum())

fig_rates = go.Figure()
fig_rates.add_trace(go.Scatter(
    x=dates, y=wide["aave_apyBase"], fill=None, mode="lines",
    line=dict(color=AAVE_COLOR, width=1.5), name="Aave base rate",
))
fig_rates.add_trace(go.Scatter(
    x=dates, y=compound_y, fill="tonexty",
    fillcolor="rgba(240,178,122,0.15)", mode="lines",
    line=dict(color="rgba(0,211,149,0.75)", width=1.5), name=compound_label,
))

if use_full:
    aave_max_idx = wide["aave_apyBase"].idxmax()
    comp_max_idx = compound_y.idxmax()
    fig_rates.add_annotation(
        x=aave_max_idx, y=wide["aave_apyBase"].max(),
        text=f"Aave max {wide['aave_apyBase'].max():.1f}%",
        showarrow=True, arrowhead=2, ax=60, ay=-30,
        font=dict(color=AAVE_COLOR, size=11), arrowcolor=AAVE_COLOR,
    )
    fig_rates.add_annotation(
        x=comp_max_idx, y=compound_y.max(),
        text=f"Compound max {compound_y.max():.1f}%",
        showarrow=True, arrowhead=2, ax=-60, ay=-30,
        font=dict(color=COMPOUND_COLOR, size=11), arrowcolor=COMPOUND_COLOR,
    )

yrange = None if use_full else [0, 15]
base_layout(fig_rates, yrange=yrange, height=340)
fig_rates.update_layout(hovermode="x unified", yaxis_title="Borrow rate (%)")
st.plotly_chart(fig_rates, use_container_width=True)

if not use_full and above15 > 0:
    st.caption(f"{above15} days exceed this range — toggle to full scale to see them")

st.markdown("---")

st.markdown("## The spread is almost nothing. Until it isn't.")

st.markdown(
    "The median daily spread between Aave and Compound is 0.19% — on most days, the two "
    "protocols are essentially identical. The normal distribution overlaid on the chart is what "
    "a well-behaved spread would look like. A kurtosis of 126.7 means this isn't one: the real "
    "distribution towers at the centre, and when spreads move, they move violently."
)

col_hist, col_stats = st.columns([3, 2])

spread = wide["spread_vs_net"].dropna()

with col_hist:
    hist_vals, bin_edges = np.histogram(spread, bins=120)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    mu, sigma = spread.mean(), spread.std()
    x_norm = np.linspace(spread.min(), spread.max(), 400)
    pdf_vals = stats.norm.pdf(x_norm, mu, sigma)
    y_norm = pdf_vals * (hist_vals.max() / pdf_vals.max())

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=bin_centers, y=hist_vals, width=bin_width,
        marker_color=SPREAD_COLOR, marker_opacity=0.7, name="Daily spread",
    ))
    fig_hist.add_trace(go.Scatter(
        x=x_norm, y=y_norm, mode="lines",
        line=dict(color="#FFFFFF", width=1.5, dash="dot"), name="Normal distribution",
    ))
    fig_hist.add_vline(x=spread.median(), line_color="rgba(255,255,255,0.5)", line_dash="dash", line_width=1)
    fig_hist.add_annotation(
        x=spread.median(), y=hist_vals.max() * 0.9,
        text=f"Median: {spread.median():.2f}%",
        showarrow=False, xanchor="left", xshift=6,
        font=dict(color="rgba(255,255,255,0.7)", size=11),
    )
    fig_hist.add_annotation(
        x=0.97, y=0.92, xref="paper", yref="paper",
        text="Kurtosis: 126.7<br><span style='font-size:10px;opacity:0.6'>(normal = 3)</span>",
        showarrow=False, xanchor="right",
        font=dict(color="rgba(255,255,255,0.7)", size=11),
        bgcolor="rgba(26,29,35,0.7)", borderpad=4,
    )
    base_layout(fig_hist, height=600)
    fig_hist.update_layout(
        hovermode="x unified", xaxis_title="Spread (%)", yaxis_title="Count",
        barmode="overlay", showlegend=True,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_stats:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    big_number("144", "episodes above 1%", compact=True)
    big_number("3.3 days", "mean episode duration", compact=True)
    big_number("27 days", "longest episode", compact=True)
    big_number("3.9%", "mean peak spread", compact=True)
    big_number("54.2%", "maximum peak spread", compact=True)

st.markdown(
    "The 144 episodes above aren't just a count — each one is a dated, measurable event. "
    "Filter by peak spread or duration below to pull up the specific periods that dominate "
    "the distribution on the left."
)

st.markdown("**Episode browser**")

col_f1, col_f2 = st.columns(2)
with col_f1:
    min_peak = st.slider("Minimum peak spread (%)", 1.0, 20.0, 1.0, 0.5)
with col_f2:
    min_dur = st.slider("Minimum duration (days)", 1, 14, 1)

episodes = get_episodes(1.0)
filtered = episodes[
    (episodes["Peak |spread| %"] >= min_peak) &
    (episodes["Duration (days)"] >= min_dur)
].reset_index(drop=True)

st.dataframe(
    filtered, use_container_width=True, hide_index=True,
    column_config={
        "Start": st.column_config.DateColumn("Start", format="YYYY-MM-DD"),
        "End": st.column_config.DateColumn("End", format="YYYY-MM-DD"),
        "Duration (days)": st.column_config.NumberColumn("Duration (days)"),
        "Peak |spread| %": st.column_config.NumberColumn("Peak |spread| %", format="%.2f%%"),
        "Mean spread %": st.column_config.NumberColumn("Mean spread %", format="%.2f%%"),
    },
)
st.caption(f"{len(filtered)} episode{'s' if len(filtered) != 1 else ''} shown")

st.markdown("---")

st.markdown("## Aave & Compound rates: Linked over years. Strangers in any given month.")

st.markdown(
    "Aave and Compound offer the same product — USDC borrowing — on the same blockchain, "
    "so you'd expect their rates to track closely. Cointegration confirms a long-run "
    "structural link: the two series share an equilibrium they never permanently escape. Yet a "
    "mean 30-day rolling correlation of 0.27 says they spend most of their time ignoring each "
    "other getting there."
)

col_chart, col_stats2 = st.columns([3, 1])
corr = wide["rolling_corr_30d"].dropna()
corr_dates = corr.index

with col_chart:
    fig_corr = go.Figure()
    neg_mask = corr < 0
    fig_corr.add_trace(go.Scatter(
        x=corr_dates, y=corr.where(neg_mask), fill="tozeroy",
        fillcolor="rgba(220,60,60,0.2)", mode="none", name="Negative correlation",
    ))
    high_mask = corr > 0.5
    fig_corr.add_trace(go.Scatter(
        x=corr_dates, y=corr.where(high_mask), fill="tozeroy",
        fillcolor="rgba(0,211,149,0.15)", mode="none", name="Correlation > 0.50",
    ))
    fig_corr.add_trace(go.Scatter(
        x=corr_dates, y=corr, mode="lines",
        line=dict(color="#FAFAFA", width=1.5), name="30-day rolling correlation",
    ))
    for y_val, label, color in [
        (0, "0", "rgba(255,255,255,0.35)"),
        (0.27, "mean 0.27", "rgba(240,178,122,0.7)"),
        (0.50, "moderate 0.50", "rgba(0,211,149,0.5)"),
    ]:
        fig_corr.add_hline(
            y=y_val, line_color=color, line_dash="dash", line_width=1,
            annotation_text=label, annotation_font_color=color, annotation_position="right",
        )
    base_layout(fig_corr, yrange=[-1, 1], height=340)
    fig_corr.update_layout(hovermode="x unified", yaxis_title="Pearson r (30-day)")
    st.plotly_chart(fig_corr, use_container_width=True)

with col_stats2:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    big_number("0.27", "Mean")
    big_number("−0.70", "Min")
    big_number("0.88", "Max")
    big_number("74.3%", "% of days below 0.50")

st.markdown("---")

st.markdown(
    "<p style='font-size:1rem;color:rgba(250,250,250,0.45);'>"
    "Aave's reward column is null for all 1,163 records — not missing data, but a structural "
    "fact: Aave runs no token incentive programme on this pool.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

st.title("Five Things I Was Wrong About")
st.markdown(
    "*Each hypothesis gets the same treatment: what I assumed, how I tested it, "
    "what the data said, and why it matters.*"
)

st.markdown("---")

# Round 1
st.markdown("## Round 1 — TVL Flows Drive Divergence")

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
        "The spread opens first and capital follows — arbitrage is the mechanism, not the cause."
    )

with col_visual:
    tvl_chg = wide["aave_tvlUsd"].pct_change() * 100
    valid = tvl_chg.notna() & wide["spread_vs_net"].notna() & (tvl_chg.abs() < 15)
    x_sc = tvl_chg[valid].values
    y_sc = wide["spread_vs_net"][valid].values
    m, b_int = np.polyfit(x_sc, y_sc, 1)
    x_line = np.linspace(x_sc.min(), x_sc.max(), 200)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=x_sc, y=y_sc, mode="markers",
        marker=dict(color=SPREAD_COLOR, opacity=0.22, size=4), name="Daily observation",
    ))
    fig_scatter.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b_int, mode="lines",
        line=dict(color="rgba(255,255,255,0.6)", width=1.5, dash="dot"), name="Linear fit",
    ))
    fig_scatter.add_annotation(
        x=0.97, y=0.92, xref="paper", yref="paper",
        text="r = 0.10", showarrow=False, xanchor="right",
        font=dict(color="rgba(255,255,255,0.8)", size=13),
        bgcolor="rgba(26,29,35,0.7)", borderpad=4,
    )
    base_layout(fig_scatter, height=240)
    fig_scatter.update_layout(xaxis_title="Aave TVL change (%)", yaxis_title="Spread (%)", showlegend=False)
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Same-day TVL change vs spread</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    _box = (
        "font-size:0.9rem;font-weight:700;border-radius:4px;"
        "padding:0.2rem 0.5rem;min-width:4.5rem;text-align:center;display:inline-block;"
    )
    _arrow = "font-size:1.2rem;min-width:6rem;text-align:center;display:inline-block;"
    _stat = "font-size:0.8rem;color:rgba(250,250,250,0.5);margin-left:0.35rem;"

    st.markdown(
        "<div style='padding:0.25rem 0.25rem 0;'>"
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0.65rem;'>Granger causality direction</p>"
        f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;'>"
        f"<span style='{_box}background:rgba(240,178,122,0.12);border:1px solid rgba(240,178,122,0.45);'>SPREAD</span>"
        f"<span style='{_arrow}color:rgba(0,211,149,0.9);'>──────→</span>"
        f"<span style='{_box}background:rgba(182,80,158,0.12);border:1px solid rgba(182,80,158,0.45);'>TVL</span>"
        f"<span style='{_stat}'>F = 593 &nbsp;·&nbsp; p = 0.0</span>"
        f"</div>"
        f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
        f"<span style='{_box}border:1px solid rgba(255,255,255,0.2);color:rgba(255,255,255,0.3);'>TVL</span>"
        f"<span style='{_arrow}color:rgba(255,255,255,0.25);'>──────→</span>"
        f"<span style='{_box}border:1px solid rgba(255,255,255,0.2);color:rgba(255,255,255,0.3);'>SPREAD</span>"
        f"<span style='{_stat}'>F = 0.0001 &nbsp;·&nbsp; p = 0.99</span>"
        f"</div>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# Round 2
st.markdown("## Round 2 — Post-Spike Spreads Close Slower")

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
        "steady state. Survival analysis on 146 OU-detected spike episodes (vs 144 identified "
        "by the >1% threshold on Page 2 — different detection methods)."
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
    two_stats(
        "0.87 days", "Spike window half-life", "days since spike ≤ 7",
        "1.25 days", "Steady-state half-life", "days since spike > 7",
        la="rgba(240,178,122,0.8)", ra="rgba(255,255,255,0.15)",
    )

    ep = get_episodes(1.0)
    durations = ep["Duration (days)"].values
    max_d = min(int(durations.max()), 30)
    days_arr = np.arange(0, max_d + 1)
    surv_pct = np.array([100.0 * np.mean(durations > d) for d in days_arr])

    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(
        x=days_arr, y=surv_pct, mode="lines",
        line=dict(color=SPREAD_COLOR, width=2),
        fill="tozeroy", fillcolor="rgba(240,178,122,0.1)",
    ))
    for x_ref, label, ax_offset in [(3, "68.5% closed by day 3", 40), (7, "92.5% closed by day 7", -40)]:
        idx = min(x_ref, len(surv_pct) - 1)
        fig_surv.add_vline(x=x_ref, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1)
        fig_surv.add_annotation(
            x=x_ref, y=surv_pct[idx], text=label,
            showarrow=True, arrowhead=2, ax=ax_offset, ay=-24,
            arrowcolor="rgba(255,255,255,0.3)",
            font=dict(color="rgba(255,255,255,0.6)", size=10),
            bgcolor="rgba(26,29,35,0.6)", borderpad=3,
        )
    base_layout(fig_surv, yrange=[0, 100], height=230)
    fig_surv.update_layout(xaxis_title="Days since spike started", yaxis_title="% still open", showlegend=False)
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Episode survival — % still open by day</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_surv, use_container_width=True)

st.markdown("---")

# Round 3
st.markdown("## Round 3 — Friday Spreads Get Locked In Over the Weekend")

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
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    persist_rates = [81.2, 78.2, 80.1, 76.5, 76.5, 85.5, 79.4]
    bar_colors = [
        COMPOUND_COLOR if d == "Sat" else
        "rgba(220,60,60,0.65)" if d in ("Fri", "Thu") else
        "rgba(240,178,122,0.6)"
        for d in dow_labels
    ]

    fig_dow = go.Figure()
    fig_dow.add_trace(go.Bar(
        x=dow_labels, y=persist_rates, marker_color=bar_colors,
        text=[f"{v}%" for v in persist_rates], textposition="outside",
        textfont=dict(color=AXIS_TICK_COLOR, size=11),
    ))
    avg = float(np.mean(persist_rates))
    fig_dow.add_hline(
        y=avg, line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1,
        annotation_text=f"avg {avg:.1f}%", annotation_font_color="rgba(255,255,255,0.45)",
        annotation_position="right",
    )
    base_layout(fig_dow, yrange=[60, 93], height=270)
    fig_dow.update_layout(yaxis_title="Persistence rate (%)", showlegend=False)
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Sign-persistence rate by day of week</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_dow, use_container_width=True)

    two_stats(
        "68.5%", "Fri → Mon persistence", "spread direction same sign",
        "80.1%", "Tue → Wed persistence", "mid-week reference",
        la="rgba(220,60,60,0.6)", ra="rgba(0,211,149,0.4)", dim_right=False,
    )

st.markdown("---")

# Round 4
st.markdown("## Round 4 — The Bigger Protocol Moves First")

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
        "Each test was chosen to probe a different aspect of the lead/lag claim: whether a signal "
        "exists at all, which direction it runs, and whether it holds up over time. "
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
        "Each protocol responds primarily to its own history, not to the other's. "
        "Size doesn't confer timing advantage in a market with no closing bell."
    )

with col_visual:
    lags = list(range(-5, 6))
    ccf_vals = [
        -0.1367, 0.0847, 0.0361, -0.0703, -0.0283,
         0.1708,
        -0.1052, -0.0746, 0.1566, -0.0853, -0.0027,
    ]
    sig_threshold = 1.96 / np.sqrt(len(wide))
    ccf_colors = [
        SPREAD_COLOR if abs(v) > sig_threshold else "rgba(255,255,255,0.2)"
        for v in ccf_vals
    ]

    fig_ccf = go.Figure()
    fig_ccf.add_trace(go.Bar(x=lags, y=ccf_vals, marker_color=ccf_colors))
    for threshold in [sig_threshold, -sig_threshold]:
        fig_ccf.add_hline(y=threshold, line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1)
    fig_ccf.add_hline(y=0, line_color="rgba(255,255,255,0.12)", line_width=1)
    fig_ccf.add_annotation(
        x=-5.5, y=sig_threshold + 0.006, text="95% CI",
        showarrow=False, font=dict(color="rgba(255,255,255,0.35)", size=10),
    )
    fig_ccf.add_annotation(
        x=-2.5, y=0.22, text="← Compound leads    Aave leads →",
        showarrow=False, font=dict(color="rgba(255,255,255,0.35)", size=10),
    )
    base_layout(fig_ccf, yrange=[-0.2, 0.25], height=265)
    fig_ccf.update_layout(
        xaxis=dict(tickvals=lags, ticktext=[str(l) for l in lags], tickfont_color=AXIS_TICK_COLOR),
        xaxis_title="Lag (days)", yaxis_title="Pearson r", showlegend=False,
    )
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Cross-correlation function — rate changes</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_ccf, use_container_width=True)

    st.markdown(
        "<div style='padding:0.85rem 1rem;border-left:3px solid rgba(255,255,255,0.15);'>"
        "<div style='font-size:2.6rem;font-weight:700;margin:0;line-height:1;'>R² = 0.0002</div>"
        "<div style='font-size:1.05rem;color:rgba(250,250,250,0.5);margin:0.05rem 0 0;'>"
        "variance in Compound's move explained by yesterday's Aave move</div>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# Round 5
st.markdown("## Round 5 — Today's Spread Tells You Something About Tomorrow's Spread")

col_text, col_visual = st.columns(2)

with col_text:
    section_label("Assumed")
    st.markdown(
        "With mean reversion this strong and a half-life under a day, autocorrelation should "
        "carry useful predictive power. Today's spread should tell you something about "
        "tomorrow's spread."
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
        "The spread reverts fast, but new shocks arrive faster. What survives that — a weak "
        "directional signal, linear and concentrated in mean reversion — is not a tradeable "
        "edge, but a real one."
    )

with col_visual:
    two_stats(
        "6.4%", "AR(10) R² — level", "next-day spread level variance explained",
        "0.671", "Logistic AUC — direction", "next-day spread direction",
        la="rgba(220,60,60,0.6)", ra="rgba(0,211,149,0.5)", dim_right=False,
    )

    features = ["Aave rate change", "Spread lag (t−1)", "Spread rolling mean",
                "Compound net change", "Days since spike", "Day of week"]
    importance = [0.44, 0.36, 0.10, 0.07, 0.02, 0.01]
    feat_colors = [AAVE_COLOR if i < 2 else "rgba(255,255,255,0.22)" for i in range(len(features))]

    fig_feat = go.Figure()
    fig_feat.add_trace(go.Bar(
        x=importance, y=features, orientation="h", marker_color=feat_colors,
        text=[f"{v:.0%}" for v in importance], textposition="outside",
        textfont=dict(color=AXIS_TICK_COLOR, size=11),
    ))
    base_layout(fig_feat, height=250)
    fig_feat.update_layout(
        xaxis=dict(tickformat=".0%", range=[0, 0.58], tickfont_color=AXIS_TICK_COLOR),
        xaxis_title="Relative importance",
        yaxis=dict(autorange="reversed", tickfont_color=AXIS_TICK_COLOR),
        showlegend=False,
    )
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Logistic regression — feature importance (next-day direction)</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_feat, use_container_width=True)

st.markdown("---")

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

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HOW I BUILT THIS
# ══════════════════════════════════════════════════════════════════════════════

st.title("How I Built This")
st.markdown("*The decisions, detours, and one result that looked perfect for thirty seconds.*")

st.markdown(
    "Traditional fixed income taught me how interest rates work — how policy decisions propagate "
    "through markets, how credit spreads price risk, how duration shapes a portfolio. Applying "
    "that knowledge to my own savings was where traditional finance drew a line: the vehicles "
    "that compound capital efficiently come with a minimum ticket. Save enough first, then access "
    "the tools. DeFi drops that barrier. I came in through that door — with a decade of rates "
    "intuition, a blank slate on DeFi mechanics, and enough curiosity to measure what I found "
    "rather than assume I understood it."
)

st.markdown("---")

st.markdown("## The stack")

st.markdown("""
- **Data:** DefiLlama Yields API — daily USDC pool data for Aave V3 and Compound V3 on Ethereum mainnet
- **Quality gate:** Five checks on every ingest — schema, row count, null rates, value ranges, group distribution
- **Language:** Python — pandas, statsmodels, scikit-learn, xgboost
- **App:** Streamlit
- **Storage:** gitignored — raw JSON and processed files not committed
""")

st.markdown("---")

st.markdown("## When the tests disagree, dig deeper")

col_text, col_visual = st.columns([3, 2])

with col_text:
    st.markdown(
        "Early in the analysis, a basic question needed answering: does the spread between Aave "
        "and Compound have a stable long-run average, or does it drift over time? The answer "
        "matters — a spread that drifts indefinitely is a different beast from one that always "
        "returns to the same level."
    )
    st.markdown(
        "Two standard tests were run to answer this — ADF and KPSS. ADF answered yes; KPSS "
        "answered no. Each test approaches the question from a different angle, and when they "
        "contradict each other like this, it usually means something specific is going on: the "
        "mean is probably drifting slowly over time rather than anchored to a fixed level — not "
        "collapsing into a trend, but not fully stable either. That's an unsatisfying answer, "
        "so the analysis went further."
    )
    st.markdown(
        "A third test — Zivot-Andrews — was run to allow for a single abrupt shift in the mean. "
        "It found one, dated to July 2023. A fourth test — Bai-Perron — allowed for multiple "
        "breaks and found three in total. Crucially, within each of the resulting four segments, "
        "both ADF and KPSS agree: the spread is cleanly stationary. The mean shifts between "
        "segments are economically negligible — all within 3 basis points of zero."
    )
    st.markdown(
        "The original conflict was a statistical artifact of three small regime shifts, not "
        "evidence of a genuine trend. Zivot-Andrews and Bai-Perron were not in the original "
        "plan. This is what following the data looks like."
    )

with col_visual:
    segments = [
        ("2023-02-06", "2023-07-29", -0.0072, "S1"),
        ("2023-07-30", "2024-01-15",  0.0227, "S2"),
        ("2024-01-16", "2024-09-22",  0.0074, "S3"),
        ("2024-09-23", None,          -0.0010, "S4"),
    ]
    seg_colors = [
        "rgba(182,80,158,0.12)", "rgba(0,211,149,0.10)",
        "rgba(240,178,122,0.12)", "rgba(100,149,237,0.10)",
    ]
    seg_line_colors = [
        "rgba(182,80,158,0.8)", "rgba(0,211,149,0.8)",
        "rgba(240,178,122,0.8)", "rgba(100,149,237,0.8)",
    ]

    spread_series = wide["spread_vs_net"].clip(-10, 10)
    seg_dates = wide.index

    fig_seg = go.Figure()
    fig_seg.add_trace(go.Scatter(
        x=seg_dates, y=spread_series, mode="lines",
        line=dict(color="rgba(255,255,255,0.18)", width=1), showlegend=False,
    ))

    for i, (start, end, mean, label) in enumerate(segments):
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end) if end else wide.index[-1]

        fig_seg.add_trace(go.Scatter(
            x=[start_dt, end_dt, end_dt, start_dt], y=[-10, -10, 10, 10],
            fill="toself", fillcolor=seg_colors[i], line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig_seg.add_shape(
            type="line", x0=start_dt, x1=end_dt, y0=mean, y1=mean,
            line=dict(color=seg_line_colors[i], width=2, dash="solid"),
        )
        mid_dt = start_dt + (end_dt - start_dt) / 2
        fig_seg.add_annotation(
            x=mid_dt, y=mean, text=f"{label}: {mean:+.3f}%",
            showarrow=False, yshift=12,
            font=dict(color=seg_line_colors[i], size=10),
            bgcolor="rgba(14,17,23,0.7)", borderpad=2,
        )

    for break_date in ["2023-07-30", "2024-01-16", "2024-09-23"]:
        fig_seg.add_vline(
            x=pd.Timestamp(break_date).value / 1e6,
            line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1,
        )

    base_layout(fig_seg, yrange=[-10, 10], height=370)
    fig_seg.update_layout(xaxis_title=None, yaxis_title="Spread (%, clipped ±10%)")
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Spread series — four stationary segments (Bai-Perron breaks)</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_seg, use_container_width=True)

st.markdown("---")

st.markdown("## Why I built 22 features and kept 10")

col_text, col_visual = st.columns([3, 2])

with col_text:
    st.markdown(
        "The spread between the two protocols — Aave's rate minus Compound's net cost — was the "
        "target. To predict it, or understand what drives it, you need features that capture the "
        "state of both protocols on any given day: how far apart they are right now, which "
        "direction each is moving, how much liquidity each holds, and whether the market is in a "
        "calm or stressed regime."
    )
    st.markdown(
        "That's four categories of information. The first decision was how to handle Compound's "
        "COMP token rewards, which reduce borrowers' effective cost. Were those rewards stable "
        "over time, or volatile? If volatile, a spread calculated with rewards included would "
        "behave differently from one calculated without. The answer was: unknown. So both "
        "versions of every spread feature were built — one net of rewards, one ignoring them — "
        "and the data was allowed to decide."
    )
    st.markdown(
        "It decided quickly. Every rewards-adjusted feature correlated above 0.95 with its "
        "unadjusted equivalent. Compound's rewards are stable enough that both definitions carry "
        "the same information. All seven base rate variants were dropped."
    )
    st.markdown(
        "Five more features were cut for a different reason: they barely changed over the dataset. "
        "A feature that doesn't vary can't explain why the spread varies. Two cases stand out. "
        "The spike flag — which marked days when either protocol's rate exceeded 10% — fired on "
        "only 12% of days, making it almost always zero; any regime information it carried is "
        "still present in the days-since-last-spike feature, which survived. And the COMP reward "
        "as a standalone feature failed for the same reason the base rate variants did: if the "
        "reward barely moves, it can't explain why the spread moves."
    )
    st.markdown("Of the 22 features built, 10 survived.")

with col_visual:
    kept = [
        ("spread_vs_net", "Spread"), ("spread_vs_net_lag_1d", "Spread"),
        ("spread_vs_net_rolling_mean_7d", "Spread"), ("spread_vs_net_rolling_std_7d", "Spread"),
        ("aave_rate_change_1d", "Rate momentum"), ("compound_net_change_1d", "Rate momentum"),
        ("tvl_ratio", "Liquidity"), ("aave_tvl_change_pct_1d", "Liquidity"),
        ("days_since_spike", "Regime"), ("day_of_week", "Calendar"),
    ]
    dropped_corr = [
        ("spread_vs_base", "r = 0.990"), ("spread_vs_base_lag_1d", "r = 0.990"),
        ("spread_vs_base_rolling_mean_7d", "r = 0.962"), ("spread_vs_base_rolling_std_7d", "r = 1.000"),
        ("spread_vs_base_zscore_30d", "r = 0.997"), ("compound_base_change_1d", "r = 1.000"),
        ("rate_divergence_direction_vs_base", "r = 0.968"),
    ]
    dropped_var = [
        ("spread_vs_net_zscore_30d", "low variance"), ("rate_divergence_direction_vs_net", "low variance"),
        ("compound_apyReward", "low variance"), ("compound_tvl_change_pct_1d", "low variance"),
        ("is_spike", "low variance"),
    ]

    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0.4rem;'>✓ Kept — 10 features</p>",
        unsafe_allow_html=True,
    )
    kept_df = pd.DataFrame(kept, columns=["Feature", "Group"])
    st.dataframe(kept_df, use_container_width=True, hide_index=True, height=230)

    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin:0.75rem 0 0.4rem;'>✗ Dropped — high correlation (7)</p>",
        unsafe_allow_html=True,
    )
    corr_df = pd.DataFrame(dropped_corr, columns=["Feature", "Reason"])
    st.dataframe(corr_df, use_container_width=True, hide_index=True, height=215)

    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin:0.75rem 0 0.4rem;'>✗ Dropped — low variance (5)</p>",
        unsafe_allow_html=True,
    )
    var_df = pd.DataFrame(dropped_var, columns=["Feature", "Reason"])
    st.dataframe(var_df, use_container_width=True, hide_index=True, height=180)

st.markdown("---")

st.markdown("## The model got everything right. That was the problem.")

col_text, col_visual = st.columns([3, 2])

with col_text:
    st.markdown(
        "The main Q1 model was a linear regression predicting the daily spread — Aave's borrow "
        "rate minus Compound's net cost — from a set of ten engineered features including the "
        "prior day's spread, each protocol's daily rate change, TVL movements, and calendar "
        "variables."
    )
    st.markdown("The results came back: MAE = 0.0, R² = 1.0, direction accuracy = 100%. Perfect on every metric.")
    st.markdown("That's a red flag.")
    st.markdown(
        "The coefficients made it obvious: the model assigned exactly 1.0 to the prior day's "
        "spread, exactly 1.0 to Aave's daily rate change, and exactly −1.0 to Compound's daily "
        "rate change. Everything else got zero. It had found an accounting identity:"
    )
    st.code("spread_t = spread_lag_1d + aave_rate_change_1d − compound_net_change_1d")
    st.markdown(
        "Today's spread is yesterday's spread plus whatever Aave moved, minus whatever Compound "
        "moved. That's not a prediction — it's arithmetic. The features contained a perfect "
        "reconstruction of the target by construction, and the model found it."
    )
    st.markdown(
        "The right response is not to fix it. The identity is real and it's logged. The "
        "interesting question — can you predict tomorrow's spread from today's features — is a "
        "separate problem, addressed in the five rounds on the previous page."
    )

with col_visual:
    feat_names = [
        "Spread lag", "Aave rate change", "Compound net change",
        "Spread rolling mean", "Spread rolling std",
        "TVL ratio", "Aave TVL change", "Days since spike", "Day of week",
    ]
    coef_vals = [1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bar_colors = [
        SPREAD_COLOR if abs(v) == 1.0 else "rgba(255,255,255,0.15)"
        for v in coef_vals
    ]

    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        x=coef_vals, y=feat_names, orientation="h", marker_color=bar_colors,
        text=[f"{v:+.1f}" if abs(v) == 1.0 else "0.0" for v in coef_vals],
        textposition="outside", textfont=dict(color=AXIS_TICK_COLOR, size=12),
    ))
    fig_coef.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    base_layout(fig_coef, yrange=None, height=320)
    fig_coef.update_layout(
        xaxis=dict(range=[-1.4, 1.4], tickvals=[-1, 0, 1], tickfont_color=AXIS_TICK_COLOR),
        yaxis=dict(autorange="reversed", tickfont_color=AXIS_TICK_COLOR),
        showlegend=False,
    )
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Linear regression coefficients</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_coef, use_container_width=True)

st.markdown("---")

st.markdown("## Nine more questions")

st.markdown(
    "This is the first of ten planned research questions on DeFi interest rate behaviour. "
    "The remaining nine:"
)

st.markdown("""
1. Utilization curve sensitivity
2. Event-driven rate behaviour
3. Mean reversion timescales
4. Cross-asset volatility
5. Rate forecasting
6. Rate-price lead-lag
7. Calendar effects
8. Aave V2 vs V3
9. Whale detection
""")

st.markdown(
    "Each will follow the same structure: a question, a dataset, a set of hypotheses, "
    "and an account of what the data said."
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.data import AAVE_COLOR, COMPOUND_COLOR, SPREAD_COLOR, big_number, get_episodes, inject_global_css, load_wide

st.set_page_config(page_title="Explore the Data", layout="wide")

inject_global_css()

wide = load_wide()

# ── Shared style helpers ──────────────────────────────────────────────────────

PANEL_BG = "rgba(26,29,35,0.0)"
GRID_COLOR = "rgba(255,255,255,0.06)"
AXIS_LINE_COLOR = "rgba(255,255,255,0.2)"
AXIS_TICK_COLOR = "rgba(255,255,255,0.85)"


def base_layout(fig: go.Figure, yrange=None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font_color="#FAFAFA",
        margin=dict(l=8, r=8, t=32, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        tickfont_color=AXIS_TICK_COLOR,
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        linecolor=AXIS_LINE_COLOR,
        tickfont_color=AXIS_TICK_COLOR,
        range=yrange,
    )
    return fig


# ── Page Introduction ─────────────────────────────────────────────────────────

st.title("Explore the Data")
st.markdown(
    "*The verdicts are on page 1. Here's the data that made the hypotheses seem reasonable "
    "in the first place.*"
)

st.markdown("---")

# ── Section 1: Opening Hook ───────────────────────────────────────────────────

st.markdown("## Compound looks more expensive. It isn't.")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(
        "<p style='font-size:1.5rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.45);margin-bottom:0.5rem;'>The headline view</p>",
        unsafe_allow_html=True,
    )
    r1, r2 = st.columns(2)
    with r1:
        big_number("4.87%", "Aave mean borrow rate")
    with r2:
        big_number("4.99%", "Compound mean borrow rate")
    st.caption("Base rates, no rewards adjustment — Compound appears more expensive")

with col_right:
    st.markdown(
        "<p style='font-size:1.5rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.45);margin-bottom:0.5rem;'>The full picture</p>",
        unsafe_allow_html=True,
    )
    r1, r2 = st.columns(2)
    with r1:
        big_number("55%", "of days Aave is more expensive", "net of COMP rewards")
    with r2:
        big_number("45%", "of days Compound is more expensive", "net of COMP rewards")
    st.caption("After subtracting COMP rewards — Aave is the pricier borrow on most days")

st.markdown(
    "Compound pays COMP token rewards directly to borrowers, reducing their effective cost by "
    "43 basis points on average — making it the cheaper place to borrow on most days, despite the "
    "higher headline rate. That's why every spread in this analysis is calculated net of rewards, "
    "not off the headline rate."
)

st.markdown("---")

# ── Section 2: The Two Rate Series ───────────────────────────────────────────

st.markdown("## Same asset. Same chain. Very different moments.")

st.markdown(
    "Aave's mean borrow rate is 4.87%. Its maximum was 56.7%. A mean built from a distribution "
    "that wide isn't much of a guide — the chart defaults to a capped view for readability, "
    "but the full scale is the honest picture."
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

use_net = rate_toggle == "Net cost (incl. COMP rewards)"
use_full = scale_toggle == "Full scale"

compound_y = wide["compound_net"] if use_net else wide["compound_apyBase"]
compound_label = "Compound net cost" if use_net else "Compound base rate"
dates = wide.index

above15 = int(((wide["aave_apyBase"] > 15) | (compound_y > 15)).sum())

fig_rates = go.Figure()

# Shaded area between the two series
fig_rates.add_trace(go.Scatter(
    x=dates,
    y=wide["aave_apyBase"],
    fill=None,
    mode="lines",
    line=dict(color=AAVE_COLOR, width=1.5),
    name="Aave base rate",
))
fig_rates.add_trace(go.Scatter(
    x=dates,
    y=compound_y,
    fill="tonexty",
    fillcolor="rgba(240,178,122,0.15)",
    mode="lines",
    line=dict(color=f"rgba(0,211,149,0.75)", width=1.5),
    name=compound_label,
))

# Spike annotations (only on full scale)
if use_full:
    aave_max_idx = wide["aave_apyBase"].idxmax()
    comp_max_idx = compound_y.idxmax()
    fig_rates.add_annotation(
        x=aave_max_idx, y=wide["aave_apyBase"].max(),
        text=f"Aave max {wide['aave_apyBase'].max():.1f}%",
        showarrow=True, arrowhead=2, ax=60, ay=-30,
        font=dict(color=AAVE_COLOR, size=11),
        arrowcolor=AAVE_COLOR,
    )
    fig_rates.add_annotation(
        x=comp_max_idx, y=compound_y.max(),
        text=f"Compound max {compound_y.max():.1f}%",
        showarrow=True, arrowhead=2, ax=-60, ay=-30,
        font=dict(color=COMPOUND_COLOR, size=11),
        arrowcolor=COMPOUND_COLOR,
    )

yrange = None if use_full else [0, 15]
base_layout(fig_rates, yrange=yrange)
fig_rates.update_layout(height=340, yaxis_title="Borrow rate (%)")

st.plotly_chart(fig_rates, use_container_width=True)

if not use_full and above15 > 0:
    st.caption(f"{above15} days exceed this range — toggle to full scale to see them")

st.markdown("---")

# ── Section 3: The Spread Distribution ───────────────────────────────────────

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
    # Histogram
    hist_vals, bin_edges = np.histogram(spread, bins=120)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Normal distribution overlay scaled to histogram
    mu, sigma = spread.mean(), spread.std()
    x_norm = np.linspace(spread.min(), spread.max(), 400)
    pdf_vals = stats.norm.pdf(x_norm, mu, sigma)
    y_norm = pdf_vals * (hist_vals.max() / pdf_vals.max())

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=bin_centers,
        y=hist_vals,
        width=bin_width,
        marker_color=SPREAD_COLOR,
        marker_opacity=0.7,
        name="Daily spread",
    ))
    fig_hist.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm,
        mode="lines",
        line=dict(color="#FFFFFF", width=1.5, dash="dot"),
        name="Normal distribution",
    ))

    # Callout: median line
    fig_hist.add_vline(
        x=spread.median(), line_color="rgba(255,255,255,0.5)",
        line_dash="dash", line_width=1,
    )
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

    base_layout(fig_hist)
    fig_hist.update_layout(
        height=600,
        xaxis_title="Spread (%)",
        yaxis_title="Count",
        barmode="overlay",
        showlegend=True,
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
    filtered,
    use_container_width=True,
    hide_index=True,
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

# ── Section 4: The Rolling Correlation Puzzle ─────────────────────────────────

st.markdown("## Aave & Compound rates: Linked over years. Strangers in any given month.")

st.markdown(
    "Aave and Compound offer the same product — USDC borrowing — on the same blockchain, "
    "so you'd expect their rates to track closely. Cointegration confirms a long-run "
    "structural link: the two series share an equilibrium they never permanently escape. Yet a "
    "mean 30-day rolling correlation of 0.27 says they spend most of their time ignoring each "
    "other getting there."
)

col_chart, col_stats = st.columns([3, 1])

corr = wide["rolling_corr_30d"].dropna()
corr_dates = corr.index

with col_chart:
    fig_corr = go.Figure()

    # Shaded bands — negative correlation (red)
    neg_mask = corr < 0
    fig_corr.add_trace(go.Scatter(
        x=corr_dates,
        y=corr.where(neg_mask),
        fill="tozeroy",
        fillcolor="rgba(220,60,60,0.2)",
        mode="none",
        name="Negative correlation",
        showlegend=True,
    ))

    # Shaded bands — high correlation > 0.5 (green)
    high_mask = corr > 0.5
    fig_corr.add_trace(go.Scatter(
        x=corr_dates,
        y=corr.where(high_mask),
        fill="tozeroy",
        fillcolor="rgba(0,211,149,0.15)",
        mode="none",
        name="Correlation > 0.50",
        showlegend=True,
    ))

    # Main correlation line
    fig_corr.add_trace(go.Scatter(
        x=corr_dates,
        y=corr,
        mode="lines",
        line=dict(color="#FAFAFA", width=1.5),
        name="30-day rolling correlation",
    ))

    # Reference lines
    for y_val, label, color in [
        (0, "0", "rgba(255,255,255,0.35)"),
        (0.27, "mean 0.27", "rgba(240,178,122,0.7)"),
        (0.50, "moderate 0.50", "rgba(0,211,149,0.5)"),
    ]:
        fig_corr.add_hline(
            y=y_val, line_color=color, line_dash="dash", line_width=1,
            annotation_text=label,
            annotation_font_color=color,
            annotation_position="right",
        )

    base_layout(fig_corr, yrange=[-1, 1])
    fig_corr.update_layout(height=340, yaxis_title="Pearson r (30-day)")
    st.plotly_chart(fig_corr, use_container_width=True)

with col_stats:
    st.markdown("&nbsp;", unsafe_allow_html=True)
    big_number("0.27", "Mean")
    big_number("−0.70", "Min")
    big_number("0.88", "Max")
    big_number("74.3%", "% of days below 0.50")

st.markdown("---")

# ── Section 5: Data Quality Note ─────────────────────────────────────────────

st.markdown(
    "<p style='font-size:1rem;color:rgba(250,250,250,0.45);'>"
    "Aave's reward column is null for all 1,163 records — not missing data, but a structural "
    "fact: Aave runs no token incentive programme on this pool.</p>",
    unsafe_allow_html=True,
)

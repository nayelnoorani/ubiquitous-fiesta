import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.data import AAVE_COLOR, COMPOUND_COLOR, SPREAD_COLOR, get_episodes, load_wide

st.set_page_config(page_title="Explore the Data", layout="wide")
st.title("Explore the Data")

wide = load_wide()
episodes = get_episodes(1.0)

# ── Date range filter (sidebar) ───────────────────────────────────────────────
st.sidebar.header("Date range")
min_date = wide.index.min().date()
max_date = wide.index.max().date()

start_date = st.sidebar.date_input("From", min_date, min_value=min_date, max_value=max_date)
end_date   = st.sidebar.date_input("To",   max_date, min_value=min_date, max_value=max_date)

mask = (wide.index.date >= start_date) & (wide.index.date <= end_date)
w = wide.loc[mask]

# ── 1. Borrow rates ───────────────────────────────────────────────────────────
st.markdown("### Borrow Rates Over Time")
st.caption("Aave `apyBase` vs Compound effective rate (`apyBase − apyReward`). "
           "Compound rewards reduce borrower cost — this is the fair comparison.")

fig_rates = go.Figure()
fig_rates.add_trace(go.Scatter(
    x=w.index, y=w["aave_apyBase"],
    name="Aave V3",
    line=dict(color=AAVE_COLOR, width=1.5),
    hovertemplate="%{x|%Y-%m-%d}<br>Aave: %{y:.2f}%<extra></extra>",
))
fig_rates.add_trace(go.Scatter(
    x=w.index, y=w["compound_net"],
    name="Compound V3 (net)",
    line=dict(color=COMPOUND_COLOR, width=1.5),
    hovertemplate="%{x|%Y-%m-%d}<br>Compound net: %{y:.2f}%<extra></extra>",
))
fig_rates.update_layout(
    yaxis_title="APY (%)",
    hovermode="x unified",
    height=360,
    margin=dict(t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig_rates, use_container_width=True)

# ── 2. Spread over time ───────────────────────────────────────────────────────
st.markdown("### Spread Over Time")
st.caption("Positive = Aave is more expensive. Shaded bands = episodes where |spread| > 1%.")

fig_spread = go.Figure()

# Shade episode regions that fall within the filtered window
ep_in_window = episodes[
    (pd.to_datetime(episodes["End"]) >= pd.to_datetime(start_date)) &
    (pd.to_datetime(episodes["Start"]) <= pd.to_datetime(end_date))
]
for _, ep in ep_in_window.iterrows():
    fig_spread.add_vrect(
        x0=str(ep["Start"]), x1=str(ep["End"]),
        fillcolor="rgba(240,178,122,0.18)",
        layer="below", line_width=0,
        annotation_text="", annotation_position="top left",
    )

fig_spread.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
fig_spread.add_trace(go.Scatter(
    x=w.index, y=w["spread_vs_net"],
    name="Spread",
    line=dict(color=SPREAD_COLOR, width=1.2),
    fill="tozeroy",
    fillcolor="rgba(240,178,122,0.06)",
    hovertemplate="%{x|%Y-%m-%d}<br>Spread: %{y:.2f}%<extra></extra>",
))
fig_spread.update_layout(
    yaxis_title="Spread (%)",
    hovermode="x unified",
    height=300,
    margin=dict(t=10, b=10),
    showlegend=False,
)
st.plotly_chart(fig_spread, use_container_width=True)

# ── 3. Rolling 30-day correlation ─────────────────────────────────────────────
st.markdown("### 30-Day Rolling Correlation")
st.caption("Despite long-run cointegration, the protocols correlate at only 0.27 on average "
           "and go negative at times — they respond to different short-term demand signals.")

fig_corr = go.Figure()
fig_corr.add_hline(y=0,    line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)
fig_corr.add_hline(
    y=0.2693,
    line_dash="dot", line_color="rgba(255,255,255,0.5)", line_width=1,
    annotation_text="mean 0.27", annotation_position="top right",
)
fig_corr.add_trace(go.Scatter(
    x=w.index, y=w["rolling_corr_30d"],
    name="30-day correlation",
    line=dict(color="#636EFA", width=1.5),
    fill="tozeroy",
    fillcolor="rgba(99,110,250,0.08)",
    hovertemplate="%{x|%Y-%m-%d}<br>Corr: %{y:.3f}<extra></extra>",
))
fig_corr.update_layout(
    yaxis_title="Correlation",
    yaxis_range=[-1, 1],
    hovermode="x unified",
    height=280,
    margin=dict(t=10, b=10),
    showlegend=False,
)
st.plotly_chart(fig_corr, use_container_width=True)

# ── 4. TVL + Spread distribution ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### TVL Over Time")
    st.caption("Aave holds ~6.6× more TVL, which explains its sharper rate spikes.")
    fig_tvl = go.Figure()
    fig_tvl.add_trace(go.Scatter(
        x=w.index, y=w["aave_tvlUsd"] / 1e6,
        name="Aave V3", line=dict(color=AAVE_COLOR, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Aave TVL: $%{y:.0f}M<extra></extra>",
    ))
    fig_tvl.add_trace(go.Scatter(
        x=w.index, y=w["compound_tvlUsd"] / 1e6,
        name="Compound V3", line=dict(color=COMPOUND_COLOR, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Compound TVL: $%{y:.0f}M<extra></extra>",
    ))
    fig_tvl.update_layout(
        yaxis_title="TVL ($M)",
        hovermode="x unified",
        height=300,
        margin=dict(t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_tvl, use_container_width=True)

with col2:
    st.markdown("### Spread Distribution")
    st.caption("Clipped to ±10% for readability. Actual range: −18.8% to +52.9%.")
    clipped = w["spread_vs_net"].clip(-10, 10)
    median_val = w["spread_vs_net"].median()
    fig_hist = px.histogram(
        clipped, nbins=80,
        color_discrete_sequence=[SPREAD_COLOR],
        labels={"value": "Spread (%)"},
    )
    fig_hist.add_vline(
        x=median_val,
        line_dash="dash", line_color="white", line_width=1.5,
        annotation_text=f"median {median_val:.2f}%",
        annotation_position="top right",
    )
    fig_hist.update_layout(
        xaxis_title="Spread (%)", yaxis_title="Days",
        showlegend=False,
        height=300,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── 5. Episode browser ────────────────────────────────────────────────────────
st.markdown("### Spread Episodes  —  |spread| > 1%")
st.caption(f"{len(episodes)} episodes detected over the full sample. "
           "Mean duration 3.3 days · Max 27 days · Mean peak 3.9%")

ep_display = episodes[
    (pd.to_datetime(episodes["Start"]) >= pd.to_datetime(start_date)) &
    (pd.to_datetime(episodes["End"])   <= pd.to_datetime(end_date))
].reset_index(drop=True)

st.dataframe(ep_display, use_container_width=True, hide_index=True)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data import AAVE_COLOR, COMPOUND_COLOR, SPREAD_COLOR, get_episodes, inject_global_css, load_wide

st.set_page_config(page_title="Model & Statistical Results", layout="wide")

inject_global_css()

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


# ── Page Introduction ─────────────────────────────────────────────────────────

st.title("Five Things I Was Wrong About")
st.markdown(
    "*Each hypothesis gets the same treatment: what I assumed, how I tested it, "
    "what the data said, and why it matters.*"
)

st.markdown("---")

# ── Round 1 — TVL Flows Drive Divergence ─────────────────────────────────────

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
    # Scatter: same-day Aave TVL % change vs spread level
    tvl_chg = wide["aave_tvlUsd"].pct_change() * 100
    valid = tvl_chg.notna() & wide["spread_vs_net"].notna() & (tvl_chg.abs() < 15)
    x_sc = tvl_chg[valid].values
    y_sc = wide["spread_vs_net"][valid].values
    m, b_int = np.polyfit(x_sc, y_sc, 1)
    x_line = np.linspace(x_sc.min(), x_sc.max(), 200)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=x_sc, y=y_sc, mode="markers",
        marker=dict(color=SPREAD_COLOR, opacity=0.22, size=4),
        name="Daily observation",
    ))
    fig_scatter.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b_int, mode="lines",
        line=dict(color="rgba(255,255,255,0.6)", width=1.5, dash="dot"),
        name="Linear fit",
    ))
    fig_scatter.add_annotation(
        x=0.97, y=0.92, xref="paper", yref="paper",
        text="r = 0.10", showarrow=False, xanchor="right",
        font=dict(color="rgba(255,255,255,0.8)", size=13),
        bgcolor="rgba(26,29,35,0.7)", borderpad=4,
    )
    base_layout(fig_scatter, height=240)
    fig_scatter.update_layout(
        xaxis_title="Aave TVL change (%)",
        yaxis_title="Spread (%)",
        showlegend=False,
    )
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Same-day TVL change vs spread</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Granger causality direction diagram
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

# ── Round 2 — Post-Spike Spreads Close Slower ─────────────────────────────────

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
    # OU half-life comparison
    two_stats(
        "0.87 days", "Spike window half-life", "days since spike ≤ 7",
        "1.25 days", "Steady-state half-life", "days since spike > 7",
        la="rgba(240,178,122,0.8)", ra="rgba(255,255,255,0.15)",
    )

    # Survival curve from episode durations
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
        name="% episodes still open",
    ))
    for x_ref, label, ax_offset in [(3, "68.5% closed by day 3", 40), (7, "92.5% closed by day 7", -40)]:
        idx = min(x_ref, len(surv_pct) - 1)
        fig_surv.add_vline(
            x=x_ref, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1,
        )
        fig_surv.add_annotation(
            x=x_ref, y=surv_pct[idx],
            text=label, showarrow=True, arrowhead=2, ax=ax_offset, ay=-24,
            arrowcolor="rgba(255,255,255,0.3)",
            font=dict(color="rgba(255,255,255,0.6)", size=10),
            bgcolor="rgba(26,29,35,0.6)", borderpad=3,
        )
    base_layout(fig_surv, yrange=[0, 100], height=230)
    fig_surv.update_layout(
        xaxis_title="Days since spike started",
        yaxis_title="% still open",
        showlegend=False,
    )
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Episode survival — % still open by day</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_surv, use_container_width=True)

st.markdown("---")

# ── Round 3 — Friday Spreads Get Locked In ────────────────────────────────────

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
    # Persistence rate by day of week (from Q1S5 results)
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
        x=dow_labels, y=persist_rates,
        marker_color=bar_colors,
        text=[f"{v}%" for v in persist_rates],
        textposition="outside",
        textfont=dict(color=AXIS_TICK_COLOR, size=11),
    ))
    avg = float(np.mean(persist_rates))
    fig_dow.add_hline(
        y=avg, line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1,
        annotation_text=f"avg {avg:.1f}%",
        annotation_font_color="rgba(255,255,255,0.45)",
        annotation_position="right",
    )
    base_layout(fig_dow, yrange=[60, 93], height=270)
    fig_dow.update_layout(
        yaxis_title="Persistence rate (%)",
        showlegend=False,
    )
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Sign-persistence rate by day of week</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_dow, use_container_width=True)

    # Fri→Mon vs Tue→Wed comparison
    two_stats(
        "68.5%", "Fri → Mon persistence", "spread direction same sign",
        "80.1%", "Tue → Wed persistence", "mid-week reference",
        la="rgba(220,60,60,0.6)", ra="rgba(0,211,149,0.4)",
        dim_right=False,
    )

st.markdown("---")

# ── Round 4 — The Bigger Protocol Moves First ─────────────────────────────────

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
    # CCF chart — lags -5 to +5 (values from Q1S6 results)
    lags = list(range(-5, 6))
    ccf_vals = [
        -0.1367, 0.0847, 0.0361, -0.0703, -0.0283,   # lags -5 to -1: Compound leads
         0.1708,                                         # lag 0
        -0.1052, -0.0746, 0.1566, -0.0853, -0.0027,   # lags +1 to +5: Aave leads
    ]
    sig_threshold = 1.96 / np.sqrt(len(wide))
    ccf_colors = [
        SPREAD_COLOR if abs(v) > sig_threshold else "rgba(255,255,255,0.2)"
        for v in ccf_vals
    ]

    fig_ccf = go.Figure()
    fig_ccf.add_trace(go.Bar(
        x=lags, y=ccf_vals, marker_color=ccf_colors, name="CCF",
    ))
    for threshold in [sig_threshold, -sig_threshold]:
        fig_ccf.add_hline(
            y=threshold, line_dash="dot",
            line_color="rgba(255,255,255,0.3)", line_width=1,
        )
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
        xaxis=dict(
            tickvals=lags, ticktext=[str(l) for l in lags],
            tickfont_color=AXIS_TICK_COLOR,
        ),
        xaxis_title="Lag (days)",
        yaxis_title="Pearson r",
        showlegend=False,
    )
    st.markdown(
        "<p style='font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;"
        "color:rgba(250,250,250,0.4);margin-bottom:0;'>Cross-correlation function — rate changes</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_ccf, use_container_width=True)

    # R² = 0.0002 large display
    st.markdown(
        "<div style='padding:0.85rem 1rem;border-left:3px solid rgba(255,255,255,0.15);'>"
        "<div style='font-size:2.6rem;font-weight:700;margin:0;line-height:1;'>R² = 0.0002</div>"
        "<div style='font-size:1.05rem;color:rgba(250,250,250,0.5);margin:0.05rem 0 0;'>"
        "variance in Compound's move explained by yesterday's Aave move</div>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Round 5 — Yesterday's Spread Tells You Something ─────────────────────────

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
    # Level vs direction: two contrasting stats
    two_stats(
        "6.4%", "AR(10) R² — level", "next-day spread level variance explained",
        "0.671", "Logistic AUC — direction", "next-day spread direction",
        la="rgba(220,60,60,0.6)", ra="rgba(0,211,149,0.5)",
        dim_right=False,
    )

    # Feature importance (from logistic regression on next-day direction)
    features = [
        "Aave rate change",
        "Spread lag (t−1)",
        "Spread rolling mean",
        "Compound net change",
        "Days since spike",
        "Day of week",
    ]
    importance = [0.44, 0.36, 0.10, 0.07, 0.02, 0.01]
    feat_colors = [
        AAVE_COLOR if i < 2 else "rgba(255,255,255,0.22)"
        for i in range(len(features))
    ]

    fig_feat = go.Figure()
    fig_feat.add_trace(go.Bar(
        x=importance,
        y=features,
        orientation="h",
        marker_color=feat_colors,
        text=[f"{v:.0%}" for v in importance],
        textposition="outside",
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

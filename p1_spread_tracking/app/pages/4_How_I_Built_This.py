import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import SPREAD_COLOR, load_wide

st.set_page_config(page_title="How I Built This", layout="wide")

wide = load_wide()

# ── Shared style ──────────────────────────────────────────────────────────────

PANEL_BG = "rgba(26,29,35,0.0)"
GRID_COLOR = "rgba(255,255,255,0.06)"
AXIS_LINE_COLOR = "rgba(255,255,255,0.2)"
AXIS_TICK_COLOR = "rgba(255,255,255,0.85)"


def base_layout(fig: go.Figure, yrange=None, height: int = 300) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font_color="#FAFAFA",
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
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


# ── Page Introduction ─────────────────────────────────────────────────────────

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

# ── Section 1: Stack and Pipeline ────────────────────────────────────────────

st.markdown("## The stack")

st.markdown("""
- **Data:** DefiLlama Yields API — daily USDC pool data for Aave V3 and Compound V3 on Ethereum mainnet
- **Quality gate:** Five checks on every ingest — schema, row count, null rates, value ranges, group distribution
- **Language:** Python — pandas, statsmodels, scikit-learn, xgboost
- **App:** Streamlit
- **Storage:** gitignored — raw JSON and processed files not committed
""")

st.markdown("---")

# ── Section 2: The Stationarity Investigation ─────────────────────────────────

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
        "rgba(182,80,158,0.12)",
        "rgba(0,211,149,0.10)",
        "rgba(240,178,122,0.12)",
        "rgba(100,149,237,0.10)",
    ]
    seg_line_colors = [
        "rgba(182,80,158,0.8)",
        "rgba(0,211,149,0.8)",
        "rgba(240,178,122,0.8)",
        "rgba(100,149,237,0.8)",
    ]

    spread = wide["spread_vs_net"].clip(-10, 10)
    dates = wide.index

    fig_seg = go.Figure()

    fig_seg.add_trace(go.Scatter(
        x=dates, y=spread, mode="lines",
        line=dict(color="rgba(255,255,255,0.18)", width=1),
        showlegend=False,
    ))

    for i, (start, end, mean, label) in enumerate(segments):
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end) if end else wide.index[-1]

        fig_seg.add_trace(go.Scatter(
            x=[start_dt, end_dt, end_dt, start_dt],
            y=[-10, -10, 10, 10],
            fill="toself",
            fillcolor=seg_colors[i],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig_seg.add_shape(
            type="line",
            x0=start_dt, x1=end_dt,
            y0=mean, y1=mean,
            line=dict(color=seg_line_colors[i], width=2, dash="solid"),
        )

        mid_dt = start_dt + (end_dt - start_dt) / 2
        fig_seg.add_annotation(
            x=mid_dt, y=mean,
            text=f"{label}: {mean:+.3f}%",
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
    fig_seg.update_layout(
        xaxis_title=None,
        yaxis_title="Spread (%, clipped ±10%)",
        title=dict(
            text="Spread series — four stationary segments (Bai-Perron breaks)",
            font=dict(size=12, color="rgba(255,255,255,0.45)"), x=0,
        ),
    )
    st.plotly_chart(fig_seg, use_container_width=True)

st.markdown("---")

# ── Section 3: Feature Engineering ───────────────────────────────────────────

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
        ("spread_vs_net", "Spread"),
        ("spread_vs_net_lag_1d", "Spread"),
        ("spread_vs_net_rolling_mean_7d", "Spread"),
        ("spread_vs_net_rolling_std_7d", "Spread"),
        ("aave_rate_change_1d", "Rate momentum"),
        ("compound_net_change_1d", "Rate momentum"),
        ("tvl_ratio", "Liquidity"),
        ("aave_tvl_change_pct_1d", "Liquidity"),
        ("days_since_spike", "Regime"),
        ("day_of_week", "Calendar"),
    ]
    dropped_corr = [
        ("spread_vs_base", "r = 0.990"),
        ("spread_vs_base_lag_1d", "r = 0.990"),
        ("spread_vs_base_rolling_mean_7d", "r = 0.962"),
        ("spread_vs_base_rolling_std_7d", "r = 1.000"),
        ("spread_vs_base_zscore_30d", "r = 0.997"),
        ("compound_base_change_1d", "r = 1.000"),
        ("rate_divergence_direction_vs_base", "r = 0.968"),
    ]
    dropped_var = [
        ("spread_vs_net_zscore_30d", "low variance"),
        ("rate_divergence_direction_vs_net", "low variance"),
        ("compound_apyReward", "low variance"),
        ("compound_tvl_change_pct_1d", "low variance"),
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

# ── Section 4: The Tautology Catch ───────────────────────────────────────────

st.markdown("## The model got everything right. That was the problem.")

col_text, col_visual = st.columns([3, 2])

with col_text:
    st.markdown(
        "The main Q1 model was a linear regression predicting the daily spread — Aave's borrow "
        "rate minus Compound's net cost — from a set of ten engineered features including the "
        "prior day's spread, each protocol's daily rate change, TVL movements, and calendar "
        "variables."
    )
    st.markdown(
        "The results came back: MAE = 0.0, R² = 1.0, direction accuracy = 100%. "
        "Perfect on every metric."
    )
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
        x=coef_vals,
        y=feat_names,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.1f}" if abs(v) == 1.0 else "0.0" for v in coef_vals],
        textposition="outside",
        textfont=dict(color=AXIS_TICK_COLOR, size=12),
    ))
    fig_coef.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    base_layout(fig_coef, yrange=None, height=320)
    fig_coef.update_layout(
        xaxis=dict(range=[-1.4, 1.4], tickvals=[-1, 0, 1], tickfont_color=AXIS_TICK_COLOR),
        yaxis=dict(autorange="reversed", tickfont_color=AXIS_TICK_COLOR),
        showlegend=False,
        title=dict(
            text="Linear regression coefficients",
            font=dict(size=12, color="rgba(255,255,255,0.45)"), x=0,
        ),
    )
    st.plotly_chart(fig_coef, use_container_width=True)

st.markdown("---")

# ── Section 5: What's Next ────────────────────────────────────────────────────

st.markdown("## Nine more questions")

st.markdown(
    "This is the first of ten planned research questions on DeFi interest rate behaviour. "
    "The remaining nine:"
)

st.markdown("""
- Utilization curve sensitivity
- Event-driven rate behaviour
- Mean reversion timescales
- Cross-asset volatility
- Rate forecasting
- Rate-price lead-lag
- Calendar effects
- Aave V2 vs V3
- Whale detection
""")

st.markdown(
    "Each will follow the same structure: a question, a dataset, a set of hypotheses, "
    "and an account of what the data said."
)

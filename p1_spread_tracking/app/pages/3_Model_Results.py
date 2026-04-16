import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import AAVE_COLOR, COMPOUND_COLOR, SPREAD_COLOR

st.set_page_config(page_title="Model & Statistical Results", layout="wide")
st.title("Model & Statistical Results")
st.markdown("""
Results for the main Q1 study and 7 supplementary analyses. Each section states the question,
the headline finding, key statistics, and model performance.
""")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def table(df: pd.DataFrame) -> None:
    st.dataframe(df, use_container_width=True, hide_index=True)


def single_bar(
    x: list,
    y: list,
    color: "str | list",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 280,
    orientation: str = "v",
) -> go.Figure:
    if orientation == "h":
        fig = go.Figure(go.Bar(x=y, y=x, orientation="h", marker_color=color))
        fig.update_layout(
            xaxis_title=xaxis_title, yaxis_title=yaxis_title,
            height=height, margin=dict(t=10, b=10, l=220),
        )
    else:
        fig = go.Figure(go.Bar(x=x, y=y, marker_color=color))
        fig.update_layout(
            xaxis_title=xaxis_title, yaxis_title=yaxis_title,
            height=height, margin=dict(t=10, b=10),
        )
    return fig


# ── Q1 MAIN ───────────────────────────────────────────────────────────────────
st.markdown("## Q1 — How Closely Do the Protocols Track Each Other?")
st.markdown("""
**Answer:** They are structurally linked (cointegrated, p=0.0) but behave independently
day-to-day (mean 30-day rolling correlation 0.27). The spread reverts with a half-life of
~0.95 days but is constantly hit by new shocks (σ=2.7% daily).
""")

tab_stats, tab_ml = st.tabs(["Statistical Analysis", "ML Models"])

with tab_stats:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Cointegration (Engle-Granger)**")
        table(pd.DataFrame({
            "Statistic":   [-5.3245],
            "p-value":     [0.0],
            "1% critical": [-3.9059],
            "Result":      ["Cointegrated ✓"],
        }))

        st.markdown("**Ornstein-Uhlenbeck Fit**")
        table(pd.DataFrame({
            "Parameter": ["θ (reversion speed)", "μ (long-run mean)", "σ (daily volatility)",
                          "Half-life", "R²"],
            "Value":     ["0.730", "0.32%", "2.70%", "0.95 days", "0.365"],
        }))

        st.markdown("**Spread Distribution**")
        table(pd.DataFrame({
            "Metric": ["Median", "Std", "Skew", "Kurtosis", "p5 / p95"],
            "Value":  ["0.19%", "2.81%", "7.03", "126.7", "-2.93% / 3.73%"],
        }))

    with col2:
        st.markdown("**Rolling 30-Day Correlation**")
        table(pd.DataFrame({
            "Metric": ["Mean", "Min", "Max", "% days corr < 0.50"],
            "Value":  ["0.27", "-0.70", "0.88", "74.3%"],
        }))

        st.markdown("**Spread Episodes  |spread| > 1%**")
        table(pd.DataFrame({
            "Metric": ["Count", "Mean duration", "Max duration", "Mean peak", "Max peak"],
            "Value":  ["144", "3.3 days", "27 days", "3.89%", "54.18%"],
        }))

        st.markdown("**Stationarity (ADF + KPSS)**")
        table(pd.DataFrame({
            "Test":       ["ADF", "KPSS"],
            "Statistic":  [-5.985, 0.642],
            "p-value":    ["0.0", "0.019"],
            "Conclusion": ["Stationary ✓", "Non-stationary ✗"],
        }))
        st.caption(
            "Conflicting result → trend-stationary. "
            "Resolved via Zivot-Andrews and Bai-Perron: "
            "3 structural breaks found; spread is stationary within each regime."
        )

with tab_ml:
    st.markdown("""
**Models:** Naive persistence · ARIMA(2,0,2) · Linear regression · XGBoost
**Split:** 788-day train / 197-day holdout (chronological)
**Target:** `spread_vs_net`
""")

    col1, col2 = st.columns(2)

    with col1:
        table(pd.DataFrame({
            "Model":         ["Naive persistence", "ARIMA(2,0,2)", "Linear regression †", "XGBoost"],
            "MAE":           [0.4796, 0.9629, "0.000 †", 0.1453],
            "R²":            [-0.2533, -1.0387, "1.000 †", 0.9172],
            "Direction Acc": ["—", "50.8%", "100% †", "79.7%"],
        }))

    with col2:
        fig = single_bar(
            x=["Naive", "ARIMA", "XGBoost"],
            y=[0.4796, 0.9629, 0.1453],
            color=[SPREAD_COLOR, "#636EFA", AAVE_COLOR],
            yaxis_title="MAE (%)",
        )
        fig.add_annotation(
            text="Linear regression excluded<br>(tautological identity)",
            x=1, y=0.7, showarrow=False,
            font=dict(size=10, color="rgba(255,255,255,0.6)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "† **Linear regression tautology:** the feature set contains an algebraic identity — "
        "`spread_t = spread_lag_1d + aave_rate_change_1d - compound_net_change_1d`. "
        "The model found the exact coefficients (1.0, 1.0, -1.0) and reconstructed today's spread perfectly. "
        "This is not predictive power. Predicting *tomorrow's* spread is Q1S1."
    )

    st.markdown("**XGBoost Feature Importance**")
    feat_imp = pd.DataFrame({
        "Feature": [
            "day_of_week", "spread_lag_1d", "spread_rolling_std_7d",
            "days_since_spike", "compound_net_change_1d", "tvl_ratio",
            "spread_rolling_mean_7d", "aave_rate_change_1d", "aave_tvl_change_pct_1d",
        ],
        "Importance": [0.026, 0.037, 0.050, 0.070, 0.077, 0.095, 0.100, 0.261, 0.285],
    })
    fig_fi = single_bar(
        x=feat_imp["Feature"].tolist(),
        y=feat_imp["Importance"].tolist(),
        color=AAVE_COLOR,
        xaxis_title="Importance",
        height=300,
        orientation="h",
    )
    st.plotly_chart(fig_fi, use_container_width=True)


st.markdown("---")


# ── Q1S1 ──────────────────────────────────────────────────────────────────────
with st.expander("Q1S1 — Spread Persistence: Is today's spread predictive of tomorrow's?"):
    st.markdown(
        "**Finding:** Today's spread level has almost no predictive power for tomorrow's. "
        "The 7-day rolling mean outperforms the 1-day lag because it proxies the regime mean."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ACF (autocorrelation function)**")
        table(pd.DataFrame({
            "Lag": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "ACF": [0.269, 0.159, 0.176, 0.080, 0.103, 0.112, 0.134, 0.038, 0.073, 0.091],
        }))
        st.caption("Ljung-Box p=0.0 at every lag — autocorrelation is real but small.")
        st.markdown("**OU Half-life:** 0.95 days  ·  **σ:** 2.7%  ·  **R²:** 0.365")

    with col2:
        acf_vals = [0.269, 0.159, 0.176, 0.080, 0.103, 0.112, 0.134, 0.038, 0.073, 0.091]
        fig_acf = go.Figure(go.Bar(
            x=list(range(1, 11)),
            y=acf_vals,
            marker_color=[SPREAD_COLOR if v > 0 else "#EF553B" for v in acf_vals],
        ))
        fig_acf.add_hline(y=0, line_color="white", line_width=0.5)
        fig_acf.add_hline(
            y=0.063, line_dash="dot",
            line_color="rgba(255,255,255,0.4)",
            annotation_text="95% band",
        )
        fig_acf.update_layout(
            xaxis_title="Lag (days)", yaxis_title="ACF",
            height=260, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_acf, use_container_width=True)

    st.markdown("**ML model performance** (target: next-day spread)")
    table(pd.DataFrame({
        "Model":         ["Naive persistence", "AR(10)", "Linear regression"],
        "MAE":           [0.6576, 0.5747, 0.6081],
        "R²":            [-0.603, 0.064, 0.012],
        "Direction Acc": ["—", "58.2%", "62.5%"],
    }))
    st.markdown(
        "AR(10) is the best model but explains only **6.4%** of next-day variance. "
        "Naive persistence R²=-0.60 — worse than always predicting the mean."
    )


# ── Q1S2 ──────────────────────────────────────────────────────────────────────
with st.expander("Q1S2 — Volatility as a Leading Indicator: Does high rolling volatility predict spread narrowing?"):
    st.markdown(
        "**Finding:** Elevated volatility is a statistically significant leading indicator "
        "of spread narrowing (Mann-Whitney p=0.003) — but the threshold rule completely fails. "
        "XGBoost captures diffuse non-linear signal (AUC=0.635)."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Regime comparison** (high-vol = rolling_std_7d > 1.84%)")
        table(pd.DataFrame({
            "Metric":          ["Days in regime", "Mean spread", "Mean |spread|",
                                "Std spread", "OU half-life"],
            "High-vol (25%)":  ["290", "+1.09%", "3.10%", "5.14%", "0.84 days"],
            "Low-vol (75%)":   ["868", "+0.06%", "0.88%", "1.21%", "1.80 days"],
        }))
        st.caption("Mean-reversion is faster when the spread is far from equilibrium.")

    with col2:
        fig_hl = single_bar(
            x=["Full sample", "High-vol", "Low-vol"],
            y=[0.95, 0.84, 1.80],
            color=[SPREAD_COLOR, AAVE_COLOR, COMPOUND_COLOR],
            yaxis_title="Half-life (days)",
        )
        st.plotly_chart(fig_hl, use_container_width=True)

    st.markdown("**Classifier performance** (target: does |spread| narrow in next 3 days?)")
    table(pd.DataFrame({
        "Model":              ["Threshold rule (vol > 75th pct)", "Logistic regression", "XGBoost"],
        "AUC-ROC":            [0.503, 0.571, 0.635],
        "Precision":          ["0.600", "0.714", "0.692"],
        "Recall":             ["0.025", "0.164", "0.295"],
        "F1":                 ["0.047", "0.267", "0.414"],
    }))
    st.markdown(
        "XGBoost feature importances are nearly equal across all 5 features (0.18–0.21 each) — "
        "the signal is non-linear and diffuse. No single feature dominates."
    )


# ── Q1S3 ──────────────────────────────────────────────────────────────────────
with st.expander("Q1S3 — TVL Shocks: Do sudden inflows/outflows drive spread widening?"):
    st.markdown(
        "**Finding:** The causal arrow mostly runs the other way — spread Granger-causes "
        "TVL flows (F=593 at lag 1), not vice versa. Outflow shocks are 4x more impactful "
        "than inflow shocks. OLS R²=0.27 — the best regression result in this project."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Granger causality summary**")
        table(pd.DataFrame({
            "Direction":     ["Aave TVL -> Spread", "Spread -> Aave TVL",
                              "Compound net -> Spread"],
            "Min p-value":   ["0.007 (lag 2)", "0.0 (lag 1)", "0.0 (lag 2)"],
            "Significant?":  ["Yes ✓", "Yes ✓", "Yes ✓"],
            "Note":          ["Lag 1 insignificant (p=0.99)",
                              "F=593 — dominant direction",
                              "Lag 1 insignificant"],
        }))

        st.markdown("**Inflow vs Outflow next-day spread change**")
        table(pd.DataFrame({
            "Shock type":            ["Inflow (TVL increase)", "Outflow (TVL decrease)"],
            "N days":                [158, 132],
            "Mean next-day Dspread": ["+0.42%", "-1.26%"],
        }))

    with col2:
        fig_shock = single_bar(
            x=["Inflow shock", "Outflow shock"],
            y=[0.4245, -1.2642],
            color=[COMPOUND_COLOR, AAVE_COLOR],
            yaxis_title="Mean next-day spread change (%)",
        )
        fig_shock.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.5)
        st.plotly_chart(fig_shock, use_container_width=True)

    st.markdown("**ML performance** (target: next-day spread change)")
    table(pd.DataFrame({
        "Model":     ["OLS", "Lasso (a=0.032)", "XGBoost"],
        "MAE":       [0.5573, 0.5474, 0.7540],
        "R²":        [0.273, 0.275, -0.220],
        "Direction": ["58.6%", "56.9%", "54.7%"],
    }))
    st.markdown("XGBoost underperforms OLS — the TVL-spread relationship is largely linear.")


# ── Q1S4 ──────────────────────────────────────────────────────────────────────
with st.expander("Q1S4 — Post-Spike Behaviour: Do spreads close more slowly after a rate spike?"):
    st.markdown(
        "**Finding:** The hypothesis is wrong — spreads close *faster* post-spike "
        "(half-life 0.87 vs 1.25 days in steady-state). 92.5% of episodes close within 7 days."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**OU half-life by regime**")
        table(pd.DataFrame({
            "Regime":        ["Full sample", "Spike window (<=7 days post-spike)", "Steady-state"],
            "Half-life":     ["0.92 days", "0.87 days  <- faster", "1.25 days"],
            "Mean |spread|": ["—", "3.07%", "0.98%"],
            "R²":            ["0.482", "0.525", "0.277"],
        }))

        st.markdown("**Episode survival**")
        table(pd.DataFrame({
            "Metric": ["Episodes detected", "Median days to close",
                       "% closed <=3 days", "% closed <=7 days", "Max days to close"],
            "Value":  ["146", "2 days", "68.5%", "92.5%", "24 days"],
        }))

    with col2:
        fig_hl = single_bar(
            x=["Spike window", "Steady-state"],
            y=[0.87, 1.25],
            color=[AAVE_COLOR, COMPOUND_COLOR],
            yaxis_title="OU Half-life (days)",
        )
        st.plotly_chart(fig_hl, use_container_width=True)

    st.markdown(
        "Arbitrage force is proportional to spread size — the larger the divergence, "
        "the stronger the incentive to close it."
    )


# ── Q1S5 ──────────────────────────────────────────────────────────────────────
with st.expander("Q1S5 — Weekend Effect: Are Friday spreads more likely to persist into Monday?"):
    st.markdown(
        "**Finding:** The hypothesis is wrong — Friday spreads are *less* persistent "
        "into Monday (68.5%) than mid-week (80.1%). Weekends are the most stable period; "
        "DeFi rates update continuously with no trading halt."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Day-of-week summary**")
        table(pd.DataFrame({
            "Day":           ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "Mean |spread|": ["1.57%", "1.53%", "1.31%", "1.70%", "1.41%", "1.11%", "1.43%"],
            "Persist rate":  ["81.2%", "78.2%", "80.1%", "76.5%", "76.5%", "85.5%", "79.4%"],
        }))
        st.caption(
            "Weekend mean |spread| is statistically lower (p=0.011) "
            "but effect size is negligible: Cohen's d = -0.088."
        )

    with col2:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        persist = [81.2, 78.2, 80.1, 76.5, 76.5, 85.5, 79.4]
        colors = [AAVE_COLOR if d in ("Sat", "Sun") else COMPOUND_COLOR for d in days]
        fig_dow = single_bar(x=days, y=persist, color=colors, yaxis_title="Persist rate (%)")
        fig_dow.update_layout(yaxis_range=[60, 90])
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown(
        "XGBoost AUC=0.639. Weekend dummies account for 24% of combined feature importance — "
        "day-of-week adds genuine signal despite the negligible effect size."
    )


# ── Q1S6 ──────────────────────────────────────────────────────────────────────
with st.expander("Q1S6 — Lead/Lag: Does one protocol consistently move its rate before the other?"):
    st.markdown(
        "**Finding:** Neither protocol reliably leads the other on a next-day basis. "
        "Granger significance only emerges at lags 2-4. Each protocol is primarily driven "
        "by its own prior moves (negative autocorrelation ~-0.40), not the other's."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Granger causality**")
        table(pd.DataFrame({
            "Direction":       ["Aave -> Compound", "Compound -> Aave"],
            "Lag 1 p-value":   ["0.181 (n.s.)", "0.237 (n.s.)"],
            "First sig. lag":  ["Lag 4 (p=0.0001)", "Lag 2 (p=0.006)"],
        }))
        st.caption(
            "Rolling Granger (180-day windows): "
            "Aave->Compound significant in only 25.8% of windows."
        )

        st.markdown("**VAR(10) lag-1 coefficients**")
        table(pd.DataFrame({
            "Equation":       ["Aave_t", "Compound_t"],
            "Own lag (diag)": [-0.754, -0.599],
            "Cross lag":      [-0.031, +0.012],
        }))
        st.caption("Own-series mean reversion dominates. Cross-series coefficients are negligible.")

    with col2:
        lags = [0, 1, 2, 3, 4, 5]
        aave_leads = [0.1708, -0.1052, -0.0746, 0.1566, -0.0853, -0.0027]
        comp_leads = [0.1708, -0.0283, -0.0703, 0.0361, 0.0847, -0.1367]

        fig_ccf = go.Figure()
        fig_ccf.add_trace(go.Bar(
            name="Aave leads", x=lags, y=aave_leads,
            marker_color=AAVE_COLOR, opacity=0.85,
        ))
        fig_ccf.add_trace(go.Bar(
            name="Compound leads", x=lags, y=comp_leads,
            marker_color=COMPOUND_COLOR, opacity=0.85,
        ))
        fig_ccf.add_hline(y=0, line_color="white", line_width=0.5)
        fig_ccf.update_layout(
            barmode="group",
            xaxis_title="Lag (days)", yaxis_title="CCF",
            height=280, margin=dict(t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_ccf, use_container_width=True)


# ── Q1S7 ──────────────────────────────────────────────────────────────────────
with st.expander("Q1S7 — Direction Prediction: Can we predict whether the spread will widen or narrow tomorrow?"):
    st.markdown(
        "**Finding:** Spread direction is predictable at AUC=0.671 using logistic regression. "
        "The signal is almost entirely linear and concentrated in two features: "
        "`aave_rate_change_1d` and `spread_vs_net_lag_1d`. Tree models do not improve on it."
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model performance** (target: spread widens tomorrow?)")
        table(pd.DataFrame({
            "Model":         ["Logistic regression", "Random forest", "XGBoost"],
            "AUC-ROC":       [0.671, 0.636, 0.632],
            "Accuracy":      ["53.3%", "59.4%", "59.4%"],
            "Brier score":   [0.239, 0.245, 0.251],
        }))
        st.caption(
            "Brier score: no-skill baseline ~0.25. "
            "Logistic regression is the only model with meaningful probability calibration."
        )

        st.markdown("**Feature correlations with direction label**")
        table(pd.DataFrame({
            "Feature":      ["aave_rate_change_1d", "rolling_mean_7d", "spread_lag_1d",
                             "tvl_ratio", "compound_net_change_1d"],
            "r":            [-0.114, -0.109, -0.102, +0.079, +0.077],
            "Significant?": ["Yes", "Yes", "Yes", "Yes", "Yes"],
        }))

    with col2:
        features = [
            "spread_rolling_std_7d", "day_of_week", "aave_tvl_change_pct_1d",
            "tvl_ratio", "days_since_spike",
            "spread_rolling_mean_7d", "compound_net_change_1d",
            "spread_lag_1d", "aave_rate_change_1d",
        ]
        coefs = [-0.007, 0.024, 0.025, 0.054, -0.076, 0.344, 0.998, -1.752, -2.024]
        colors = [COMPOUND_COLOR if c > 0 else AAVE_COLOR for c in coefs]

        fig_coef = go.Figure(go.Bar(
            x=coefs, y=features, orientation="h",
            marker_color=colors,
        ))
        fig_coef.update_layout(
            xaxis_title="Logistic regression coefficient",
            height=320, margin=dict(t=10, b=10, l=220),
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown(
        "Negative coefficient = feature predicts narrowing (mean reversion). "
        "The two dominant features capture the same signal: when Aave's rate is elevated "
        "and the spread is wide, it reverts toward Compound."
    )


# ── Cross-study summary ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Cross-Study Summary")

table(pd.DataFrame({
    "Study": [
        "Q1 — Spread tracking",
        "Q1S1 — Persistence",
        "Q1S2 — Volatility indicator",
        "Q1S3 — TVL shocks",
        "Q1S4 — Post-spike behaviour",
        "Q1S5 — Weekend effect",
        "Q1S6 — Lead/lag",
        "Q1S7 — Direction prediction",
    ],
    "Key finding": [
        "Cointegrated (p=0.0). OU half-life 0.95 days. Mean rolling correlation 0.27 — linked long-run, independent short-run.",
        "Today's spread barely predicts tomorrow's (AR(10) R²=0.064). 7-day rolling mean outperforms yesterday's value.",
        "High-vol leads to more narrowing (p=0.003). XGBoost AUC=0.635; threshold rule fails (AUC=0.503).",
        "Spread Granger-causes TVL flows, not vice versa. OLS R²=0.27. Outflows 4x more impactful than inflows.",
        "Spreads close faster post-spike (half-life 0.87 vs 1.25 days). 92.5% close within 7 days. Hypothesis rejected.",
        "Friday spreads less persistent into Monday (68.5%) than mid-week (80.1%). Saturday is the most stable day.",
        "Neither protocol leads the other at lag 1. Intermittent across rolling windows. Own-series autocorrelation dominates.",
        "Logistic regression AUC=0.671. Signal is linear, concentrated in aave_rate_change_1d and spread_lag_1d.",
    ],
}))

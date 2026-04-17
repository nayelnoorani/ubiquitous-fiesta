import streamlit as st

st.set_page_config(page_title="How I Built This", layout="wide")


def placeholder(description: str) -> None:
    st.markdown(
        f'<div style="border:1px dashed rgba(180,180,180,0.3);border-radius:6px;'
        f'padding:2rem;margin:0.5rem 0;text-align:center;'
        f'color:rgba(200,200,200,0.5);font-style:italic;">'
        f'[ {description} ]</div>',
        unsafe_allow_html=True,
    )


# ── Page Introduction ─────────────────────────────────────────────────────────

st.title("How I Built This")
st.markdown("*The decisions, detours, and one result that looked perfect for thirty seconds.*")

st.markdown(
    "The traditional path to capital efficiency has a cover charge. Save enough, then access "
    "the vehicles that help you save more efficiently. DeFi drops it. I came in through that "
    "door — with a rates background and no idea what I'd find."
)

st.markdown("---")

# ── Section 1: The Tautology Catch ───────────────────────────────────────────

st.markdown("## R² = 1.0 (not what it seems)")

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
    placeholder(
        "The three dominant coefficients displayed large — 1.0, 1.0, −1.0 — "
        "with the identity written out below them and all other feature coefficients shown as 0.0"
    )

st.markdown("---")

# ── Section 2: Feature Engineering ───────────────────────────────────────────

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
        "That's six categories of information. The first decision was how to handle Compound's "
        "COMP token rewards, which reduce borrowers' effective cost. Were those rewards stable "
        "over time, or volatile? If volatile, a spread calculated with rewards included would "
        "behave differently from one calculated without. The honest answer was: unknown. So both "
        "versions of every spread feature were built — one net of rewards, one ignoring them — "
        "and the data was allowed to decide."
    )

    st.markdown(
        "It decided quickly. Every rewards-adjusted feature correlated above 0.95 with its "
        "unadjusted equivalent. Compound's rewards are stable enough that both definitions carry "
        "the same information. All unadjusted variants were dropped. Of the 22 features built, "
        "10 survived selection."
    )

with col_visual:
    placeholder(
        "Two-column table — 22 features in, 10 features out, "
        "dropped features grouped by reason (high correlation vs low variance)"
    )

st.markdown("---")

# ── Section 3: The Stationarity Investigation ─────────────────────────────────

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
        "Two standard tests were run to answer this — ADF and KPSS. They gave opposite answers. "
        "ADF said the spread was stationary; KPSS said it wasn't. Both rejecting simultaneously "
        "points to a specific diagnosis: trend-stationary, meaning the mean is slowly drifting "
        "rather than fixed. That's an unsatisfying answer, so the analysis went further."
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
    placeholder(
        "Timeline of the spread series with four segments marked and their mean levels labeled — "
        "the near-identical means make the point visually · "
        "Segments: 2023-02-06→2023-07-29 (mean −0.007%), "
        "2023-07-30→2024-01-15 (mean +0.023%), "
        "2024-01-16→2024-09-22 (mean +0.007%), "
        "2024-09-23→present (mean −0.001%)"
    )

st.markdown("---")

# ── Section 4: Stack and Pipeline ────────────────────────────────────────────

st.markdown("## The stack")

st.markdown("""
- **Data:** DefiLlama Yields API — daily USDC pool data for Aave V3 and Compound V3 on Ethereum mainnet
- **Quality gate:** Five checks on every ingest — schema, row count, null rates, value ranges, group distribution
- **Language:** Python — pandas, statsmodels, scikit-learn, xgboost
- **App:** Streamlit
- **Data:** gitignored — raw JSON and processed files not committed
""")

st.markdown("---")

# ── Section 5: What's Next ────────────────────────────────────────────────────

st.markdown("## Nine more questions")

st.markdown(
    "This is the first of ten planned research questions on DeFi interest rate behaviour. "
    "The remaining nine cover utilization curve sensitivity, event-driven rate behaviour, "
    "mean reversion timescales, cross-asset volatility, rate forecasting, rate-price lead-lag, "
    "calendar effects, Aave V2 vs V3, and whale detection. Each will follow the same structure: "
    "a question, a dataset, a set of hypotheses, and an honest account of what the data said."
)

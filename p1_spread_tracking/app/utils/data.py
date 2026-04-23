import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve data directory relative to this file
# utils/data.py -> app/utils/ -> app/ -> p1_spread_tracking/ -> data/raw/
_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Brand colours shared across pages
AAVE_COLOR = "#B6509E"
COMPOUND_COLOR = "#00D395"
SPREAD_COLOR = "#F0B27A"


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        .stMarkdown h1 { margin-bottom: 0.4rem !important; }
        .stMarkdown h2 {
            margin-top: 1.5rem !important;
            margin-bottom: 0.5rem !important;
            border-left: 3px solid #B6509E;
            padding-left: 0.75rem;
        }
        .stMarkdown h3 { margin-top: 1.25rem !important; margin-bottom: 0.4rem !important; }
        hr {
            border: none !important;
            border-top: 1px solid rgba(255,255,255,0.10) !important;
            margin: 2rem 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def big_number(value: str, label: str, sub: str = "", compact: bool = False) -> None:
    padding = "0.5rem 1rem" if compact else "1.25rem 1rem"
    sub_html = f"<p style='font-size:1rem;color:rgba(250,250,250,0.45);margin:0;'>{sub}</p>" if sub else ""
    st.markdown(
        f"""
        <div style='padding:{padding};'>
          <p style='font-size:2.6rem;font-weight:700;margin:0;line-height:1;'>{value}</p>
          <p style='font-size:1.05rem;color:rgba(250,250,250,0.55);margin:0.35rem 0 0;'>{label}</p>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_wide() -> pd.DataFrame:
    """Load both raw JSON files and compute every series needed by the app."""

    # --- Aave V3 main market ---
    with open(_RAW_DIR / "chart_aave_v3_usdc_ethereum_aa70268e.json") as f:
        aave_raw = json.load(f)
    aave = pd.DataFrame(aave_raw)
    aave["date"] = pd.to_datetime(aave["timestamp"], utc=True).dt.date
    aave = (
        aave.set_index("date")[["apyBase", "tvlUsd"]]
        .rename(columns={"apyBase": "aave_apyBase", "tvlUsd": "aave_tvlUsd"})
    )

    # --- Compound V3 main market ---
    with open(_RAW_DIR / "chart_compound_v3_usdc_ethereum_7da72d09.json") as f:
        comp_raw = json.load(f)
    comp = pd.DataFrame(comp_raw)
    comp["date"] = pd.to_datetime(comp["timestamp"], utc=True).dt.date
    comp = (
        comp.set_index("date")[["apyBase", "apyReward", "tvlUsd"]]
        .rename(columns={
            "apyBase": "compound_apyBase",
            "apyReward": "compound_apyReward",
            "tvlUsd": "compound_tvlUsd",
        })
    )

    # --- Inner join on overlapping dates ---
    wide = aave.join(comp, how="inner")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    # Compound net borrow cost: COMP rewards are paid to borrowers
    wide["compound_apyReward"] = wide["compound_apyReward"].fillna(0)
    wide["compound_net"] = wide["compound_apyBase"] - wide["compound_apyReward"]

    # Spread: Aave base minus Compound net
    wide["spread_vs_net"] = wide["aave_apyBase"] - wide["compound_net"]

    # Rolling spread features
    wide["spread_rolling_mean_7d"] = wide["spread_vs_net"].rolling(7, min_periods=1).mean()
    wide["spread_rolling_std_7d"]  = wide["spread_vs_net"].rolling(7, min_periods=2).std()

    # Rate momentum
    wide["aave_rate_change_1d"]    = wide["aave_apyBase"].diff()
    wide["compound_net_change_1d"] = wide["compound_net"].diff()

    # Liquidity
    wide["tvl_ratio"] = wide["aave_tvlUsd"] / wide["compound_tvlUsd"]

    # Regime: days since a rate spike (either protocol > 10%)
    wide["is_spike"] = (
        (wide["aave_apyBase"] > 10) | (wide["compound_apyBase"] > 10)
    ).astype(int)
    spike_dates = wide.index.to_series().where(wide["is_spike"] == 1)
    last_spike  = spike_dates.ffill()
    wide["days_since_spike"] = (
        (wide.index - last_spike).dt.days.fillna(9999).astype(int)
    )

    # Calendar
    wide["day_of_week"] = wide.index.dayofweek

    # 30-day rolling correlation between Aave base and Compound net rates
    wide["rolling_corr_30d"] = (
        wide["aave_apyBase"]
        .rolling(30, min_periods=10)
        .corr(wide["compound_net"])
    )

    return wide


@st.cache_data
def get_episodes(threshold: float = 1.0) -> pd.DataFrame:
    """Return a DataFrame of spread episodes where |spread| > threshold%."""
    wide = load_wide()
    abs_spread = wide["spread_vs_net"].abs()
    in_ep = abs_spread > threshold

    rows: list[dict] = []
    start = None
    for dt, val in in_ep.items():
        if val and start is None:
            start = dt
        elif not val and start is not None:
            ep = wide.loc[start : dt - pd.Timedelta(days=1), "spread_vs_net"]
            rows.append({
                "Start":           start.date(),
                "End":             (dt - pd.Timedelta(days=1)).date(),
                "Duration (days)": len(ep),
                "Peak |spread| %": round(ep.abs().max(), 2),
                "Mean spread %":   round(ep.mean(), 2),
            })
            start = None

    # Close any episode still open at end of sample
    if start is not None:
        ep = wide.loc[start:, "spread_vs_net"]
        rows.append({
            "Start":           start.date(),
            "End":             wide.index[-1].date(),
            "Duration (days)": len(ep),
            "Peak |spread| %": round(ep.abs().max(), 2),
            "Mean spread %":   round(ep.mean(), 2),
        })

    return pd.DataFrame(rows)

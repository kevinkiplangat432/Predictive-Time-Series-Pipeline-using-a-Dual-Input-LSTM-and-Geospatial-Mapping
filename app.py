"""
Kenya Food Price Early Warning System (KFPEWS)
Streamlit dashboard - app.py

Run with:
    streamlit run app.py

DATA CONTRACT (matches the capstone notebook exactly)
-------------------------------------------------------
This app reads the two files your notebook writes at the end of Section 5
(cell that calls dashboard_data.to_csv(...) and price_history.to_csv(...)):

1. price_history.csv
   columns: market, commodity, date, price_per_kg, expected_price, flagged
   - price_per_kg   : actual observed price
   - expected_price : naive one-step-ahead baseline (previous month's price)
   - flagged        : bool, True if that month's residual exceeded
                       RESIDUAL_STD_MULTIPLIER x the pair's own prior
                       residual std (the anomaly detector from Section 5)

2. forecasts.csv
   columns: market, commodity, chosen_forecast, naive_forecast,
            display_forecast, latest_flagged
   - chosen_forecast : strategy label for this pair -> "model" | "blend" | "naive"
                        (how much weight the LSTM embedding model got vs naive,
                        from Section 4.21.6 per-pair selection)
   - naive_forecast  : last naive (previous-price) forecast on the test set
   - display_forecast: the blended forecast (weight*model + (1-weight)*naive),
                        this is the single number to show as "next expected price"
   - latest_flagged  : bool, whether the most recent observed month for this
                        pair was flagged as anomalous

IMPORTANT - what this system does NOT do:
This is a next-step (1-month-ahead) forecast + anomaly flag, not a multi-month
forecast with confidence intervals. The notebook's own conclusion (Section 6.2)
found no model reliably beats naive at the individual pair level, so the honest
UI here shows the blended forecast next to the naive baseline and the model's
own win-rate context, rather than implying more precision than the modelling
actually supports.

Put price_history.csv and forecasts.csv in ./data/ (or edit DATA_DIR below).
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

DATA_DIR = "data"
PRICE_HISTORY_FILE = os.path.join(DATA_DIR, "price_history.csv")
FORECASTS_FILE = os.path.join(DATA_DIR, "forecasts.csv")

# From the notebook's anomaly detection section (5.1) - shown in the UI as
# context for what "flagged" means.
RESIDUAL_STD_MULTIPLIER = 2
MIN_HISTORY_FOR_THRESHOLD = 6

# --------------------------------------------------------------------------
# PAGE CONFIG + STYLING
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Kenya Food Price Early Warning System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #eaeaea;
        border-radius: 10px;
        padding: 16px 18px;
        height: 100%;
    }
    .kpi-label { font-size: 0.85rem; color: #666; margin-bottom: 4px; }
    .kpi-value { font-size: 1.7rem; font-weight: 700; }
    .kpi-sub { font-size: 0.8rem; color: #888; }

    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        color: white;
    }
    .badge-flagged { background-color: #d9362f; }
    .badge-normal { background-color: #3d9a4f; }
    .badge-strategy-model { background-color: #1f4fd6; }
    .badge-strategy-blend { background-color: #7a5cc0; }
    .badge-strategy-naive { background-color: #888888; }

    .warning-card {
        background: #fdecea;
        border: 1px solid #f5c2be;
        border-radius: 10px;
        padding: 18px;
    }
    .warning-card-ok {
        background: #eaf6ec;
        border: 1px solid #bfe3c4;
        border-radius: 10px;
        padding: 18px;
    }
    .warning-title { color: #c0392b; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px; }
    .warning-title-ok { color: #256b34; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px; }

    .rec-card {
        background: #ffffff;
        border: 1px solid #eaeaea;
        border-radius: 10px;
        padding: 16px;
        height: 100%;
    }
    .rec-title { font-weight: 700; margin-bottom: 6px; }

    .note-box {
        background: #fff8e6;
        border: 1px solid #f2e0a8;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #6b5c1e;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# DEMO DATA (used only if the real CSVs aren't found, so the app still runs)
# --------------------------------------------------------------------------


@st.cache_data
def _generate_demo_price_history():
    markets = ["Nairobi", "Kisumu", "Nakuru", "Eldoret", "Mombasa"]
    commodities = ["Maize", "Beans", "Rice"]
    dates = pd.date_range("2021-01-01", "2024-12-01", freq="MS")
    rng = np.random.default_rng(42)
    rows = []
    for market in markets:
        for commodity in commodities:
            price = rng.uniform(45, 65)
            for d in dates:
                price *= 1 + rng.normal(0.01, 0.02)
                rows.append({"market": market, "commodity": commodity, "date": d,
                             "price_per_kg": round(price, 2)})
    df = pd.DataFrame(rows)
    df["expected_price"] = df.groupby(["market", "commodity"])["price_per_kg"].shift(1)
    df["residual"] = df["price_per_kg"] - df["expected_price"]
    df["residual_std_prior"] = df.groupby(["market", "commodity"])["residual"].transform(
        lambda s: s.expanding(min_periods=MIN_HISTORY_FOR_THRESHOLD).std().shift(1)
    )
    df["flagged"] = df["residual_std_prior"].notna() & (
        df["residual"].abs() > RESIDUAL_STD_MULTIPLIER * df["residual_std_prior"]
    )
    return df[["market", "commodity", "date", "price_per_kg", "expected_price", "flagged"]]


@st.cache_data
def _generate_demo_forecasts(history: pd.DataFrame):
    rng = np.random.default_rng(7)
    rows = []
    for (market, commodity), grp in history.groupby(["market", "commodity"]):
        grp = grp.sort_values("date")
        last_price = grp["price_per_kg"].iloc[-1]
        naive_forecast = last_price
        weight = round(rng.uniform(0, 1), 2)
        model_forecast = last_price * (1 + rng.normal(0, 0.04))
        display_forecast = weight * model_forecast + (1 - weight) * naive_forecast
        label = "model" if weight >= 0.5 else ("blend" if weight > 0 else "naive")
        rows.append({
            "market": market, "commodity": commodity,
            "chosen_forecast": label,
            "naive_forecast": round(naive_forecast, 2),
            "display_forecast": round(display_forecast, 2),
            "latest_flagged": bool(grp["flagged"].iloc[-1]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------


@st.cache_data
def load_price_history():
    if os.path.exists(PRICE_HISTORY_FILE):
        df = pd.read_csv(PRICE_HISTORY_FILE)
        df["date"] = pd.to_datetime(df["date"])
        df["flagged"] = df["flagged"].astype(bool)
        return df[["market", "commodity", "date", "price_per_kg", "expected_price", "flagged"]]
    return _generate_demo_price_history()


@st.cache_data
def load_forecasts(history: pd.DataFrame):
    if os.path.exists(FORECASTS_FILE):
        df = pd.read_csv(FORECASTS_FILE)
        df["latest_flagged"] = df["latest_flagged"].astype(bool)
        return df[["market", "commodity", "chosen_forecast", "naive_forecast",
                   "display_forecast", "latest_flagged"]]
    return _generate_demo_forecasts(history)


def strategy_badge(label: str) -> str:
    css_class = {"model": "badge-strategy-model", "blend": "badge-strategy-blend",
                 "naive": "badge-strategy-naive"}.get(label, "badge-strategy-naive")
    return f'<span class="badge {css_class}">{label.upper()}</span>'


def flag_badge(flagged: bool) -> str:
    css_class = "badge-flagged" if flagged else "badge-normal"
    text = "FLAGGED" if flagged else "NORMAL"
    return f'<span class="badge {css_class}">{text}</span>'


# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------

price_history = load_price_history()
forecasts = load_forecasts(price_history)

all_markets = sorted(price_history["market"].unique())
all_commodities = sorted(price_history["commodity"].unique())

# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🌾 KFPEWS")
    page = st.radio(
        "Navigate",
        ["Overview", "Price History & Anomalies", "Early Warnings",
         "Markets Monitor", "Decision Support", "Reports & Export", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.info(
        "Forecasts are a blend of a pooled LSTM (entity-embedding) model and a "
        "naive previous-price baseline, weighted per market-commodity pair by "
        "validation evidence. Anomaly flags use a residual vs. expanding "
        f"prior-std threshold ({RESIDUAL_STD_MULTIPLIER}\u03c3, min "
        f"{MIN_HISTORY_FOR_THRESHOLD} months of history).",
        icon="ℹ️",
    )

# --------------------------------------------------------------------------
# TOP FILTER ROW
# --------------------------------------------------------------------------

st.title("Kenya Food Price Early Warning System")
st.caption("Monitoring staple food prices and flagging unusual movements across Kenyan markets")

f1, f2, f3 = st.columns([1, 1, 1])
with f1:
    selected_market = st.selectbox("Select Market", all_markets, index=0)
with f2:
    selected_commodity = st.selectbox("Select Commodity", all_commodities, index=0)
with f3:
    st.markdown("**Data last updated**")
    st.write(datetime.today().strftime("%d %b %Y"))

hist_slice = price_history[
    (price_history["market"] == selected_market) & (price_history["commodity"] == selected_commodity)
].sort_values("date")

fc_row = forecasts[
    (forecasts["market"] == selected_market) & (forecasts["commodity"] == selected_commodity)
]
fc_row = fc_row.iloc[0] if not fc_row.empty else None

current_price = hist_slice["price_per_kg"].iloc[-1] if not hist_slice.empty else np.nan
display_forecast = fc_row["display_forecast"] if fc_row is not None else np.nan
naive_forecast = fc_row["naive_forecast"] if fc_row is not None else np.nan
chosen_strategy = fc_row["chosen_forecast"] if fc_row is not None else "naive"
latest_flagged = bool(fc_row["latest_flagged"]) if fc_row is not None else False

pct_change = ((display_forecast - current_price) / current_price * 100) if current_price else 0

st.markdown("---")

# --------------------------------------------------------------------------
# PAGE: OVERVIEW
# --------------------------------------------------------------------------

if page == "Overview":

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        last_date = hist_slice["date"].max()
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">Current Price ({last_date:%b %Y})</div>
            <div class="kpi-value" style="color:#1f4fd6;">{current_price:.2f}</div>
            <div class="kpi-sub">KES/kg</div>
            </div>""", unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">Next-Month Forecast (blended)</div>
            <div class="kpi-value" style="color:#2e9e46;">{display_forecast:.2f}</div>
            <div class="kpi-sub">KES/kg &nbsp;\u00b7&nbsp; naive: {naive_forecast:.2f}</div>
            </div>""", unsafe_allow_html=True,
        )

    with k3:
        arrow = "+" if pct_change >= 0 else ""
        color = "#c0392b" if pct_change >= 0 else "#2e9e46"
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">Expected Change</div>
            <div class="kpi-value" style="color:{color};">{arrow}{pct_change:.1f}%</div>
            <div class="kpi-sub">({arrow}{display_forecast - current_price:.2f} KES/kg)</div>
            </div>""", unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">Anomaly Status</div>
            <div style="margin-top:4px;">{flag_badge(latest_flagged)}</div>
            <div class="kpi-sub" style="margin-top:6px;">
                {"Latest month's price broke from its expected range" if latest_flagged
                 else "Latest month's price is within its expected range"}
            </div>
            </div>""", unsafe_allow_html=True,
        )

    with k5:
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">Forecast Strategy</div>
            <div style="margin-top:4px;">{strategy_badge(chosen_strategy)}</div>
            <div class="kpi-sub" style="margin-top:6px;">
                {"Model outweighs naive" if chosen_strategy == "model"
                 else "Partial blend with naive" if chosen_strategy == "blend"
                 else "Defaults to naive (no reliable model edge)"}
            </div>
            </div>""", unsafe_allow_html=True,
        )

    st.markdown("")

    c1, c2 = st.columns([1.7, 1])

    with c1:
        st.markdown("#### Actual Price vs. Naive Expected Price")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_slice["date"], y=hist_slice["price_per_kg"],
            mode="lines", name="Actual Price", line=dict(color="black", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=hist_slice["date"], y=hist_slice["expected_price"],
            mode="lines", name="Expected (naive)", line=dict(color="gray", width=1.5, dash="dash"),
        ))
        flagged_pts = hist_slice[hist_slice["flagged"]]
        fig.add_trace(go.Scatter(
            x=flagged_pts["date"], y=flagged_pts["price_per_kg"],
            mode="markers", name="Flagged (anomaly)",
            marker=dict(color="#d9362f", size=9, symbol="circle"),
        ))
        fig.update_layout(
            height=430, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            yaxis_title="Price (KES/kg)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"A month is flagged when its residual (actual \u2212 expected) exceeds "
            f"{RESIDUAL_STD_MULTIPLIER}\u00d7 the pair's own prior residual std "
            f"(computed on an expanding window, using only data available at the time)."
        )

    with c2:
        st.markdown("#### Early Warning Summary")
        if latest_flagged:
            reasons = [
                "Actual price broke past the pair's own historical volatility threshold",
                "This does not by itself predict direction \u2014 it flags an unusual move",
                f"Forecast strategy for this pair is currently: {chosen_strategy}",
            ]
            reasons_html = "".join(f"<li>{r}</li>" for r in reasons)
            st.markdown(
                f"""<div class="warning-card">
                <div class="warning-title">\u26a0\ufe0f ANOMALY FLAGGED</div>
                <div>{selected_commodity} prices in {selected_market} moved unusually in the
                most recent observed month.</div>
                <div style="margin-top:10px; font-weight:600;">Why this warning?</div>
                <ul>{reasons_html}</ul>
                </div>""", unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="warning-card-ok">
                <div class="warning-title-ok">\u2705 NO ANOMALY DETECTED</div>
                <div>{selected_commodity} prices in {selected_market} are within their
                expected historical range as of the latest observed month.</div>
                </div>""", unsafe_allow_html=True,
            )

    st.markdown("")

    d1, d2 = st.columns([1.4, 1])

    with d1:
        st.markdown("#### Decision Support Recommendations")
        r1, r2, r3 = st.columns(3)
        with r1:
            msg = ("Prices moved unusually this month \u2014 verify local conditions before "
                   "committing to a sale or purchase." if latest_flagged else
                   "No anomaly detected. Normal seasonal planning applies.")
            st.markdown(f"""<div class="rec-card"><div class="rec-title">\U0001F33E For Farmers</div>{msg}</div>""",
                        unsafe_allow_html=True)
        with r2:
            msg = ("Consider delaying non-urgent stock movement until the anomaly resolves."
                   if latest_flagged else
                   "Standard storage/release timing applies \u2014 no unusual signal present.")
            st.markdown(f"""<div class="rec-card"><div class="rec-title">\U0001F3E6 Storage</div>{msg}</div>""",
                        unsafe_allow_html=True)
        with r3:
            msg = ("This forecast currently relies mostly on the naive baseline for this pair "
                   "\u2014 treat the number as indicative, not precise." if chosen_strategy == "naive" else
                   "Model has some validated edge over naive for this pair.")
            st.markdown(f"""<div class="rec-card"><div class="rec-title">\U0001F4B0 Forecast Confidence</div>{msg}</div>""",
                        unsafe_allow_html=True)
        st.markdown(
            """<div class="note-box">\u26a0\ufe0f Note: No model in this system reliably beats the naive
            baseline at the individual market-commodity level (see notebook Section 6.2).
            Treat forecasts as directional context, not precise predictions.</div>""",
            unsafe_allow_html=True,
        )

    with d2:
        st.markdown("#### Markets at a Glance")
        glance_rows = []
        for market in all_markets:
            row = forecasts[(forecasts["market"] == market) & (forecasts["commodity"] == selected_commodity)]
            h = price_history[(price_history["market"] == market) & (price_history["commodity"] == selected_commodity)].sort_values("date")
            if row.empty or h.empty:
                continue
            row = row.iloc[0]
            cp = h["price_per_kg"].iloc[-1]
            chg = (row["display_forecast"] - cp) / cp * 100
            glance_rows.append({
                "Market": market, "Current": round(cp, 2), "Forecast": round(row["display_forecast"], 2),
                "Change (%)": round(chg, 1), "Strategy": row["chosen_forecast"],
                "Flagged": "Yes" if row["latest_flagged"] else "No",
            })
        st.dataframe(pd.DataFrame(glance_rows), use_container_width=True, hide_index=True, height=300)

    st.markdown("---")
    st.caption("Kenya Food Price Early Warning System | Data Sources: WFP Food Prices, NASA POWER Weather")

# --------------------------------------------------------------------------
# PAGE: PRICE HISTORY & ANOMALIES
# --------------------------------------------------------------------------

elif page == "Price History & Anomalies":
    st.markdown(f"### {selected_commodity} in {selected_market} \u2014 full history")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_slice["date"], y=hist_slice["price_per_kg"], name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=hist_slice["date"], y=hist_slice["expected_price"], name="Expected (naive)",
                              line=dict(color="gray", dash="dash")))
    flagged_pts = hist_slice[hist_slice["flagged"]]
    fig.add_trace(go.Scatter(x=flagged_pts["date"], y=flagged_pts["price_per_kg"], mode="markers",
                              name="Flagged", marker=dict(color="#d9362f", size=9)))
    fig.update_layout(height=480, yaxis_title="Price (KES/kg)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Flagged months")
    st.dataframe(
        flagged_pts[["date", "price_per_kg", "expected_price"]]
        .rename(columns={"date": "Date", "price_per_kg": "Actual", "expected_price": "Expected"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("##### Full history table")
    st.dataframe(
        hist_slice.rename(columns={"date": "Date", "price_per_kg": "Actual", "expected_price": "Expected",
                                    "flagged": "Flagged"}),
        use_container_width=True, hide_index=True, height=350,
    )

# --------------------------------------------------------------------------
# PAGE: EARLY WARNINGS
# --------------------------------------------------------------------------

elif page == "Early Warnings":
    st.markdown("### Anomaly flags across all markets and commodities")
    merged = forecasts.copy()
    merged["flag_label"] = merged["latest_flagged"].map({True: "FLAGGED", False: "Normal"})
    flagged_only = merged[merged["latest_flagged"]]
    st.write(f"**{len(flagged_only)}** of **{len(merged)}** market-commodity pairs are currently flagged.")
    st.dataframe(
        merged[["market", "commodity", "chosen_forecast", "naive_forecast", "display_forecast", "flag_label"]]
        .sort_values("flag_label")
        .rename(columns={"market": "Market", "commodity": "Commodity", "chosen_forecast": "Strategy",
                          "naive_forecast": "Naive Forecast", "display_forecast": "Blended Forecast",
                          "flag_label": "Status"}),
        use_container_width=True, hide_index=True,
    )

# --------------------------------------------------------------------------
# PAGE: MARKETS MONITOR
# --------------------------------------------------------------------------

elif page == "Markets Monitor":
    st.markdown("### Markets monitor")
    summary = price_history.sort_values("date").groupby(["market", "commodity"]).agg(
        last_price=("price_per_kg", "last"), last_date=("date", "last"), any_flagged=("flagged", "last")
    ).reset_index()
    st.dataframe(
        summary.rename(columns={"market": "Market", "commodity": "Commodity", "last_price": "Last Price",
                                 "last_date": "As Of", "any_flagged": "Flagged"}),
        use_container_width=True, hide_index=True,
    )

# --------------------------------------------------------------------------
# PAGE: DECISION SUPPORT
# --------------------------------------------------------------------------

elif page == "Decision Support":
    st.markdown("### Decision support")
    st.info(
        f"{selected_commodity} in {selected_market}: blended forecast is {display_forecast:.2f} KES/kg "
        f"({pct_change:+.1f}% vs current), using the '{chosen_strategy}' strategy. "
        f"{'This pair is currently flagged as anomalous.' if latest_flagged else 'No anomaly currently flagged for this pair.'}",
        icon="📊",
    )
    st.markdown(
        """
        **How to read this:** the notebook's own evaluation found no model reliably
        beats the naive (previous-price) baseline at the individual market-commodity
        level. The blended forecast only leans on the model where validation data
        supported it; where it didn't, the forecast defaults to naive. Use the
        anomaly flag as the primary early-warning signal, and the forecast number
        as secondary context.
        """
    )

# --------------------------------------------------------------------------
# PAGE: REPORTS & EXPORT
# --------------------------------------------------------------------------

elif page == "Reports & Export":
    st.markdown("### Export data")
    st.download_button(
        "Download price history + anomaly flags (CSV)",
        data=price_history.to_csv(index=False).encode("utf-8"),
        file_name="price_history_export.csv", mime="text/csv",
    )
    st.download_button(
        "Download forecasts + flags (CSV)",
        data=forecasts.to_csv(index=False).encode("utf-8"),
        file_name="forecasts_export.csv", mime="text/csv",
    )

# --------------------------------------------------------------------------
# PAGE: ABOUT
# --------------------------------------------------------------------------

elif page == "About":
    st.markdown("### About KFPEWS")
    st.write(
        """
        The Kenya Food Price Early Warning System (KFPEWS) combines historical
        food price data from the World Food Programme (WFP) with weather data
        from NASA POWER to monitor staple food prices across Kenyan markets.

        **Forecasting:** each market-commodity pair gets a one-month-ahead price
        forecast that blends a pooled LSTM model (trained across all pairs with
        shared market/commodity embeddings) with a naive previous-price baseline.
        The blend weight is set per pair based on validation evidence \u2014 pairs
        with little or weak evidence for the model default toward naive.

        **Anomaly detection:** a residual-based detector flags a month when the
        gap between actual and expected (naive) price exceeds twice that pair's
        own recent residual volatility, using only information available at the
        time (no lookahead).

        **Key finding:** pooling data with entity embeddings meaningfully improves
        forecasting versus independent per-pair models, but no model reliably
        beats naive at the individual pair level. This system is therefore built
        primarily as an anomaly / early-warning detector, with the forecast number
        shown as supporting context rather than a precise prediction.
        """
    )
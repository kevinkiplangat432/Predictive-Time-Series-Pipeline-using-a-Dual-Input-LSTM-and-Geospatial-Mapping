# app.py -- Streamlit dashboard
# Reads forecasts.csv and price_history.csv produced by generate_forecasts.py.
# Run with: streamlit run app.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Kenya Food Price Early Warning System", layout="wide")


@st.cache_data
def load_data():
    forecasts = pd.read_csv("forecasts.csv")
    price_history = pd.read_csv("price_history.csv", parse_dates=["date"])
    return forecasts, price_history


forecasts, price_history = load_data()

st.title("Kenya Food Price Early Warning System")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

all_markets = sorted(forecasts["market"].unique())
all_commodities = sorted(forecasts["commodity"].unique())

selected_markets = st.sidebar.multiselect("Market", all_markets, default=[])
selected_commodities = st.sidebar.multiselect("Commodity", all_commodities, default=[])
flagged_only = st.sidebar.checkbox("Show flagged pairs only", value=False)

filtered = forecasts.copy()
if selected_markets:
    filtered = filtered[filtered["market"].isin(selected_markets)]
if selected_commodities:
    filtered = filtered[filtered["commodity"].isin(selected_commodities)]
if flagged_only:
    filtered = filtered[filtered["latest_flagged"] == True]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pairs tracked", len(forecasts))
col2.metric("Currently flagged", int(forecasts["latest_flagged"].sum()))
col3.metric("Model-informed forecasts", int((forecasts["chosen_forecast"] != "naive").sum()))
col4.metric("Markets covered", forecasts["market"].nunique())

st.divider()

# ---------------------------------------------------------------------------
# Map: alert status by market
# ---------------------------------------------------------------------------
st.subheader("Alert Status by Market")

market_summary = (
    filtered.groupby(["market", "latitude", "longitude"])
    .agg(pairs_tracked=("commodity", "count"), pairs_flagged=("latest_flagged", "sum"))
    .reset_index()
)
market_summary["flag_rate"] = market_summary["pairs_flagged"] / market_summary["pairs_tracked"]

fig_map = px.scatter_mapbox(
    market_summary,
    lat="latitude", lon="longitude",
    size="pairs_tracked", color="flag_rate",
    color_continuous_scale="OrRd",
    hover_name="market",
    hover_data={"pairs_tracked": True, "pairs_flagged": True, "flag_rate": ":.0%", "latitude": False, "longitude": False},
    zoom=5, height=450,
    mapbox_style="carto-positron",
)
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Filtered table
# ---------------------------------------------------------------------------
st.subheader("All Tracked Pairs")

display_table = filtered[[
    "market", "commodity", "chosen_forecast", "naive_forecast", "display_forecast", "latest_flagged"
]].sort_values("latest_flagged", ascending=False)

st.dataframe(display_table, use_container_width=True, hide_index=True)

st.download_button(
    "Download this view as CSV",
    display_table.to_csv(index=False),
    file_name="filtered_forecasts.csv",
    mime="text/csv",
)

st.divider()

# ---------------------------------------------------------------------------
# Detail view: one pair's price history
# ---------------------------------------------------------------------------
st.subheader("Price History for a Specific Pair")

detail_market = st.selectbox("Market", all_markets, key="detail_market")
available_commodities = sorted(
    forecasts.loc[forecasts["market"] == detail_market, "commodity"].unique()
)
detail_commodity = st.selectbox("Commodity", available_commodities, key="detail_commodity")

pair_history = price_history[
    (price_history["market"] == detail_market) & (price_history["commodity"] == detail_commodity)
].sort_values("date")

pair_forecast_row = forecasts[
    (forecasts["market"] == detail_market) & (forecasts["commodity"] == detail_commodity)
]

if not pair_forecast_row.empty:
    row = pair_forecast_row.iloc[0]
    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.metric("Forecast source", row["chosen_forecast"])
    fcol2.metric("Naive forecast", f"{row['naive_forecast']:.2f} KES/kg")
    fcol3.metric("Displayed forecast", f"{row['display_forecast']:.2f} KES/kg")

fig_history = go.Figure()
fig_history.add_trace(go.Scatter(
    x=pair_history["date"], y=pair_history["price_per_kg"],
    mode="lines", name="Actual price", line=dict(color="black"),
))
fig_history.add_trace(go.Scatter(
    x=pair_history["date"], y=pair_history["expected_price"],
    mode="lines", name="Expected (naive)", line=dict(color="gray", dash="dash"),
))
flagged_points = pair_history[pair_history["flagged"] == True]
fig_history.add_trace(go.Scatter(
    x=flagged_points["date"], y=flagged_points["price_per_kg"],
    mode="markers", name="Flagged", marker=dict(color="red", size=9),
))
fig_history.update_layout(
    title=f"{detail_commodity} in {detail_market}: actual vs expected price",
    xaxis_title="Date", yaxis_title="Price per kg (KES)",
    height=450, hovermode="x unified",
)
st.plotly_chart(fig_history, use_container_width=True)

st.caption(
    "Expected price is naive (previous month's observed price) for every pair, per the routing decision in Section 7.1 — "
    "no pair's validation evidence has yet earned a model-based override."
)
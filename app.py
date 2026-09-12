
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Kenya Food Price Early Warning System", layout="wide", page_icon="🌾")

WARNING_COLORS = {"HIGH": "#DC2626", "MODERATE": "#F59E0B", "LOW": "#16A34A", "NORMAL": "#2563EB", "UNKNOWN": "#6B7280"}

CUSTOM_CSS = """
<style>
[data-testid="stSidebar"] { background-color: #0B1120; }
[data-testid="stSidebar"] * { color: #E5E7EB !important; }
div[data-testid="stMetric"] {
    background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px;
    padding: 14px 16px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data():
    forecasts = pd.read_csv("forecasts.csv", parse_dates=["forecast_date"])
    price_history = pd.read_csv("price_history.csv", parse_dates=["date"])
    return forecasts, price_history


forecasts, price_history = load_data()
data_last_updated = price_history["date"].max().strftime("%d %b %Y")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🌾 KFPEWS")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Price Forecast", "Early Warnings", "Markets Monitor", "Decision Support", "Reports & Export", "About"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.info(
    "This system uses food price data from WFP and weather data from NASA POWER "
    "to forecast prices and generate early warnings."
)


# ---------------------------------------------------------------------------
# Shared: market/commodity selector, used by Overview and Price Forecast
# ---------------------------------------------------------------------------
def pair_selector(key_prefix=""):
    all_markets = sorted(forecasts["market"].unique())
    market = st.selectbox("Select Market", all_markets, key=f"{key_prefix}_market")
    available_commodities = sorted(forecasts.loc[forecasts["market"] == market, "commodity"].unique())
    commodity = st.selectbox("Select Commodity", available_commodities, key=f"{key_prefix}_commodity")
    return market, commodity


def get_pair_row(market, commodity):
    match = forecasts[(forecasts["market"] == market) & (forecasts["commodity"] == commodity)]
    return match.iloc[0] if not match.empty else None


def model_label(row):
    if not row["model_available"]:
        return "Naive persistence (no model available)"
    return "Prophet (with weather)" if row["model_track"] == "prophet" else "Pooled LSTM (with weather)"


def render_price_forecast_chart(market, commodity, row):
    pair_history = price_history[
        (price_history["market"] == market) & (price_history["commodity"] == commodity)
    ].sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pair_history["date"], y=pair_history["price_per_kg"],
        mode="lines", name="Actual Price", line=dict(color="#1D4ED8", width=2),
    ))

    if row is not None and row["model_available"]:
        last_date = pair_history["date"].iloc[-1]
        last_price = pair_history["price_per_kg"].iloc[-1]
        forecast_x = [last_date, row["forecast_date"]]
        forecast_y = [last_price, row["display_forecast"]]
        fig.add_trace(go.Scatter(
            x=forecast_x, y=forecast_y, mode="lines+markers",
            name="Forecast (Next Month)", line=dict(color="#16A34A", width=2, dash="dot"),
        ))
        if pd.notna(row.get("forecast_lower")):
            fig.add_trace(go.Scatter(
                x=[last_date, row["forecast_date"]], y=[last_price, row["forecast_upper"]],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=[last_date, row["forecast_date"]], y=[last_price, row["forecast_lower"]],
                mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(22, 163, 74, 0.15)",
                name="Confidence Range (Prophet)", hoverinfo="skip",
            ))
        else:
            fig.add_annotation(
                x=row["forecast_date"], y=row["display_forecast"], text="Point forecast only — no confidence range for this model",
                showarrow=True, arrowhead=1, ax=-60, ay=-30, font=dict(size=10, color="#6B7280"),
            )

    flagged_points = pair_history[pair_history["flagged"] == True]
    fig.add_trace(go.Scatter(
        x=flagged_points["date"], y=flagged_points["price_per_kg"],
        mode="markers", name="Flagged (Anomaly)", marker=dict(color="#DC2626", size=9, symbol="circle-open", line=dict(width=2)),
    ))

    fig.update_layout(
        title=f"Historical Price and Forecast: {commodity} in {market}",
        xaxis_title="Date", yaxis_title="Price per kg (KES)",
        height=420, hovermode="x unified", legend=dict(orientation="h", y=-0.2),
    )
    return fig


def render_kpi_row(market, commodity, row):
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Current Price", f"{row['current_price']:.2f} KES/kg")

    if row["model_available"]:
        col2.metric(
            f"Forecast Price ({pd.Timestamp(row['forecast_date']).strftime('%b %Y')})",
            f"{row['display_forecast']:.2f} KES/kg",
            f"{row['expected_change_pct']:+.1f}%",
        )
    else:
        col2.metric("Forecast Price", "Not available", "naive fallback")

    change_color = WARNING_COLORS.get(row["warning_level"], "#6B7280")
    col3.markdown(f"**Expected Change**")
    col3.markdown(f"<span style='color:{change_color}; font-size: 1.5rem; font-weight:700;'>{row['expected_change_pct']:+.1f}%</span>", unsafe_allow_html=True)

    col4.markdown("**Warning Level**")
    col4.markdown(
        f"<span style='background-color:{change_color}; color:white; padding:4px 12px; "
        f"border-radius:6px; font-weight:600;'>{row['warning_level']}</span>",
        unsafe_allow_html=True,
    )

    col5.metric("Model", model_label(row))


def render_early_warning_summary(row):
    reasons = []
    if row["latest_flagged"]:
        reasons.append("The current price is flagged as outside this pair's own normal historical range.")
    if row["model_available"] and abs(row["expected_change_pct"]) >= 5:
        direction = "increase" if row["expected_change_pct"] > 0 else "decrease"
        reasons.append(f"The forecast expects a {direction} of {abs(row['expected_change_pct']):.1f}% next month.")
    if not reasons:
        reasons.append("No unusual signal currently detected for this pair.")

    level = row["warning_level"] if row["latest_flagged"] or abs(row["expected_change_pct"]) >= 5 else "NORMAL"
    color = WARNING_COLORS.get(level, "#6B7280")

    st.markdown(
        f"""
        <div style="background-color: {color}15; border-left: 4px solid {color}; padding: 16px; border-radius: 6px;">
        <strong style="color:{color};">{level} — Early Warning Summary</strong>
        <ul>{"".join(f"<li>{r}</li>" for r in reasons)}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_support(row):
    st.caption("General guidance based on this pair's current signal — not personalized financial advice.")
    col1, col2, col3 = st.columns(3)
    increasing = row["model_available"] and row["expected_change_pct"] > 2

    with col1:
        st.markdown("**📦 For Farmers**")
        if increasing:
            st.write("Prices are expected to rise. If safe storage is available, holding part of your stock may be worth considering. Avoid distress selling.")
        else:
            st.write("No strong upward signal currently. Normal selling decisions apply based on your own circumstances.")

    with col2:
        st.markdown("**🏬 Storage Consideration**")
        st.write("If affordable, safe storage is available, delaying sale of surplus stock can help you benefit from favorable price movement, when one is expected.")

    with col3:
        st.markdown("**💰 Financial Support**")
        st.write("If you face cash-flow pressure, explore available financial support options before selling all stock immediately at the current price.")

    st.warning("These are general observations, not individualized recommendations. Please weigh your own circumstances before acting.")


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview():
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title("Kenya Food Price Early Warning System")
        st.caption("Forecasting food prices and providing early warnings to support better decisions")
    with header_col2:
        st.metric("Data last updated", data_last_updated)

    market, commodity = pair_selector("overview")
    row = get_pair_row(market, commodity)
    if row is None:
        st.warning("No forecast available for this pair.")
        return

    render_kpi_row(market, commodity, row)
    st.divider()

    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        st.plotly_chart(render_price_forecast_chart(market, commodity, row), use_container_width=True)
    with table_col:
        st.subheader("Price by Month (Historical)")
        pair_history = price_history[(price_history["market"] == market) & (price_history["commodity"] == commodity)].copy()
        pair_history["year"] = pair_history["date"].dt.year
        pair_history["month"] = pair_history["date"].dt.strftime("%b")
        pivot = pair_history.pivot_table(index="month", columns="year", values="price_per_kg")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot.reindex([m for m in month_order if m in pivot.index])
        st.dataframe(pivot.style.background_gradient(cmap="RdYlGn_r", axis=None).format("{:.1f}"), use_container_width=True)

    st.divider()
    warn_col, market_col = st.columns(2)
    with warn_col:
        render_early_warning_summary(row)
    with market_col:
        st.subheader(f"Markets at a Glance — {commodity}")
        glance = forecasts[forecasts["commodity"] == commodity][
            ["market", "expected_change_pct", "warning_level"]
        ].sort_values("expected_change_pct", ascending=False)
        st.dataframe(glance, use_container_width=True, hide_index=True)

    st.divider()
    render_decision_support(row)


# ---------------------------------------------------------------------------
# Page: Price Forecast (deeper single-pair view)
# ---------------------------------------------------------------------------
def page_price_forecast():
    st.title("Price Forecast")
    market, commodity = pair_selector("forecast")
    row = get_pair_row(market, commodity)
    if row is None:
        st.warning("No forecast available for this pair.")
        return

    render_kpi_row(market, commodity, row)
    st.plotly_chart(render_price_forecast_chart(market, commodity, row), use_container_width=True)

    st.subheader("Last 12 Months: Actual vs Expected (Naive)")
    pair_history = price_history[
        (price_history["market"] == market) & (price_history["commodity"] == commodity)
    ].sort_values("date").tail(12)
    st.dataframe(
        pair_history[["date", "price_per_kg", "expected_price", "flagged"]].rename(
            columns={"price_per_kg": "actual_price", "expected_price": "naive_expected"}
        ),
        use_container_width=True, hide_index=True,
    )

    if not row["model_available"]:
        st.info("No trained model is available for this pair. The forecast shown is naive persistence (last observed price).")
    elif row["model_track"] == "lstm":
        st.info("This pair uses the pooled LSTM. Confidence bounds aren't available for this model track — point forecast only.")


# ---------------------------------------------------------------------------
# Page: Early Warnings
# ---------------------------------------------------------------------------
def page_early_warnings():
    st.title("Early Warnings")
    st.caption("Pairs where the current price is flagged as outside its own normal historical range.")

    flagged = forecasts[forecasts["latest_flagged"] == True].sort_values("expected_change_pct", key=abs, ascending=False)
    st.metric("Pairs currently flagged", len(flagged))

    if flagged.empty:
        st.success("No pairs are currently flagged.")
        return

    display_cols = ["market", "commodity", "current_price", "display_forecast", "expected_change_pct", "warning_level", "model_track"]
    st.dataframe(flagged[display_cols], use_container_width=True, hide_index=True)

    st.caption(
        "A flag means this pair's most recent price deviated more than 2 standard deviations from its own "
        "expanding historical pattern — validated in backtesting against three real documented Kenyan price shocks "
        "(27.6% flag rate during shock windows vs. a 9.7% baseline). See About for details."
    )


# ---------------------------------------------------------------------------
# Page: Markets Monitor
# ---------------------------------------------------------------------------
def page_markets_monitor():
    st.title("Markets Monitor")

    commodity_filter = st.multiselect("Filter by commodity", sorted(forecasts["commodity"].unique()))
    filtered = forecasts[forecasts["commodity"].isin(commodity_filter)] if commodity_filter else forecasts

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pairs tracked", len(forecasts))
    col2.metric("Currently flagged", int(forecasts["latest_flagged"].sum()))
    col3.metric("Markets covered", forecasts["market"].nunique())
    col4.metric("Pairs with a trained model", int(forecasts["model_available"].sum()))

    st.subheader("Market Location")
    map_data = filtered.dropna(subset=["latitude", "longitude"])
    fig_map = px.scatter_map(
        map_data, lat="latitude", lon="longitude", color="warning_level",
        color_discrete_map=WARNING_COLORS, hover_name="market",
        hover_data={"commodity": True, "expected_change_pct": ":.1f", "latitude": False, "longitude": False},
        zoom=5, height=420, map_style="carto-positron",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("All Tracked Pairs")
    st.dataframe(
        filtered[["market", "commodity", "current_price", "display_forecast", "expected_change_pct", "warning_level", "latest_flagged", "model_track"]],
        use_container_width=True, hide_index=True,
    )


def page_decision_support():
    st.title("Decision Support")
    market, commodity = pair_selector("decision")
    row = get_pair_row(market, commodity)
    if row is None:
        st.warning("No forecast available for this pair.")
        return
    render_kpi_row(market, commodity, row)
    st.divider()
    render_decision_support(row)



def page_reports_export():
    st.title("Reports & Export")

    st.subheader("Full Forecast Table")
    st.dataframe(forecasts, use_container_width=True, hide_index=True)
    st.download_button("Download forecasts.csv", forecasts.to_csv(index=False), "forecasts.csv", "text/csv")

    st.subheader("Full Price History")
    st.download_button("Download price_history.csv", price_history.to_csv(index=False), "price_history.csv", "text/csv")



def page_about():
    st.title("About")
    st.markdown("""
    **Kenya Food Price Early Warning System** forecasts staple food prices across 133 market-commodity
    pairs in Kenya and flags markets moving outside their own normal price range, built on WFP food price
    data (via HDX) and NASA POWER weather data.

    **What's validated:**
    - The anomaly detector is backtested against three real, independently documented price shocks (2022 Horn
      of Africa drought, Ukraine-linked grain and fertilizer shock, 2022-2023 fuel subsidy removal), showing a
      27.6% flag rate in shock windows vs. a 9.7% baseline.
    - Forecasting is validated one month ahead only. No model in this system reliably beats naive persistence
      at the individual pair level — this is reported as a real property of how persistent Kenyan staple food
      prices are, not a limitation of the modelling approach.

    **What's not independently validated:**
    - **Warning Level** (HIGH/MODERATE/LOW/NORMAL) is rule-based, derived from the magnitude of the forecasted
      change. Unlike the anomaly flag, it has not been backtested against real historical shocks.
    - Confidence ranges on the forecast chart are shown only for Prophet-track pairs, since Prophet produces
      them natively. The pooled LSTM does not have a calibrated uncertainty estimate, so its forecasts are
      shown as a point estimate only, never a fabricated range.
    - Decision Support suggestions are general, templated guidance based on the forecast direction — not
      personalized financial advice.

    **Coverage note:** the anomaly detector's strongest backtest coverage skews toward refugee-camp markets
    and informal Nairobi settlements, since these have the most consistent reporting reaching back to 2022.
    Results speak most directly to humanitarian and NGO use, not general smallholder farmer markets.

    Built as a capstone for Moringa School's Data Science program (DSF-FT16).
    """)


PAGES = {
    "Overview": page_overview,
    "Price Forecast": page_price_forecast,
    "Early Warnings": page_early_warnings,
    "Markets Monitor": page_markets_monitor,
    "Decision Support": page_decision_support,
    "Reports & Export": page_reports_export,
    "About": page_about,
}

PAGES[page]()
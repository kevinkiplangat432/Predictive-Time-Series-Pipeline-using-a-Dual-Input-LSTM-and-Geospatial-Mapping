import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kenya Food Price Early Warning", layout="centered")

@st.cache_data
def load_forecasts():
    return pd.read_csv("forecasts.csv")

@st.cache_data
def load_history():
    df = pd.read_csv("price_history.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

forecasts = load_forecasts()
history = load_history()

st.title("Kenya Food Price Early Warning")

st.subheader("Current Alerts")
alerts = forecasts[forecasts["latest_flagged"] == True]
if alerts.empty:
    st.caption("No pairs are currently flagged as unusual.")
else:
    st.dataframe(
        alerts[["market", "commodity", "chosen_forecast"]].rename(
            columns={"chosen_forecast": "forecast source"}
        ),
        hide_index=True,
        use_container_width=True,
    )

st.divider()

st.subheader("Look up a market and commodity")
market = st.selectbox("Market", sorted(forecasts["market"].unique()))
available_commodities = sorted(forecasts[forecasts["market"] == market]["commodity"].unique())
commodity = st.selectbox("Commodity", available_commodities)

row = forecasts[(forecasts["market"] == market) & (forecasts["commodity"] == commodity)].iloc[0]
pair_history = history[(history["market"] == market) & (history["commodity"] == commodity)].sort_values("date")

st.metric("Forecast price per kg (KES)", f"{row['display_forecast']:.2f}")

source_label = "Model-based forecast" if row["chosen_forecast"] == "model" else "Naive forecast (last observed price)"
st.caption(source_label)

if row["latest_flagged"]:
    st.warning("This market's most recently reported price was flagged as unusual relative to its own history.")
else:
    st.caption("No anomaly flagged in the most recently reported month.")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pair_history["date"], pair_history["price_per_kg"], label="Actual price", color="black")
ax.plot(pair_history["date"], pair_history["expected_price"], label="Expected (naive)", color="gray", linestyle="--")
flagged_points = pair_history[pair_history["flagged"] == True]
ax.scatter(flagged_points["date"], flagged_points["price_per_kg"], color="red", zorder=5, label="Flagged")
ax.set_title(f"{commodity} in {market}: price history")
ax.legend()
st.pyplot(fig)

st.caption("Anomaly status reflects the most recently reported month, not a prediction of future movement.")
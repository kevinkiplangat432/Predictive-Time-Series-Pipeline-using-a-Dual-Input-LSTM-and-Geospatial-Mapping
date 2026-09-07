import streamlit as st
import pandas as pd

st.set_page_config(page_title="Kenya Food Price Early Warning", layout="centered")

@st.cache_data
def load_data():
    return pd.read_csv("forecasts.csv")

data = load_data()

st.title("Kenya Food Price Early Warning")

market = st.selectbox("Market", sorted(data["market"].unique()))
available_commodities = sorted(data[data["market"] == market]["commodity"].unique())
commodity = st.selectbox("Commodity", available_commodities)

row = data[(data["market"] == market) & (data["commodity"] == commodity)].iloc[0]

st.metric("Forecast price per kg (KES)", f"{row['display_forecast']:.2f}")

source_label = "Model-based forecast" if row["chosen_forecast"] == "model" else "Naive forecast (last observed price)"
st.caption(source_label)

if row["latest_flagged"]:
    st.warning("This market's most recently reported price was flagged as unusual relative to its own history.")
else:
    st.caption("No anomaly flagged in the most recently reported month.")

st.caption("Anomaly status reflects the most recently reported month, not a prediction of future movement.")
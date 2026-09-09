"""
data_prep.py

Loads the WFP Kenya food prices dataset and NASA POWER weather data,
applies the scope filter and completeness floor, builds the shortlist
and master table, and saves both to disk.

This is a direct port of the Data Understanding and Data Preparation
sections of Kenyan_food_prices.ipynb. No modelling happens here, this
script's only job is producing master.parquet and shortlist.parquet
for train_model.py to consume.

Run standalone:
    python data_prep.py
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

DATA_URL = (
    "https://data.humdata.org/dataset/"
    "e0d3fba6-f9a2-45d7-b949-140c455197ff/"
    "resource/517ee1bf-2437-4f8c-aa1b-cb9925b9d437/"
    "download/wfp_food_prices_ken.csv"
)
FILENAME = "wfp_food_prices_ken.csv"

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
WEATHER_START = "20060101"
WEATHER_END = "20260815"

UNIT_TO_KG = {
    "KG": 1,
    "90 KG": 90,
    "64 KG": 64,
    "50 KG": 50,
    "26 KG": 26,
    "126 KG": 126,
    "13 KG": 13,
    "200 G": 0.2,
    "400 G": 0.4,
}

FUEL_COMMODITIES = ["Fuel (diesel)", "Fuel (kerosene)", "Fuel (petrol-gasoline)"]

MIN_YEARS_ACTIVE = 1.0
COMPLETENESS_THRESHOLD = 60  # percent, "fair" completeness against each pair's own active window


def load_food_prices():
    """Load the Kenya food prices dataset, preferring a local copy, falling back to the source URL."""
    if os.path.exists(FILENAME):
        print(f"Loading local dataset: {FILENAME}")
        return pd.read_csv(FILENAME)

    print("Local file not found. Downloading from source URL...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    with open(FILENAME, "wb") as f:
        f.write(response.content)
    print(f"Dataset downloaded and saved as '{FILENAME}'")
    return pd.read_csv(FILENAME)


def clean_price_data(prices_raw):
    """Drop a leading units/description row if present, fix date and price dtypes."""
    if prices_raw.iloc[0].astype(str).str.startswith("#").any():
        prices = prices_raw.iloc[1:].reset_index(drop=True)
    else:
        prices = prices_raw.copy()

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    if "usdprice" in prices.columns:
        prices["usdprice"] = pd.to_numeric(prices["usdprice"], errors="coerce")

    return prices


def build_shortlist(retail):
    """
    Compute each market-commodity pair's reporting span and completeness
    against its own active window, then split into Prophet-track (3+ years)
    and LSTM-track (1 to 3 years) pairs. Pairs under the 1-year floor are
    dropped entirely, they lack enough history to model reliably.
    """
    reporting_span = (
        retail.groupby(["market", "commodity"])["date"]
        .agg(first_reported="min", last_reported="max", months_reported="nunique")
        .reset_index()
    )

    reporting_span["active_months"] = (
        (reporting_span["last_reported"].dt.year - reporting_span["first_reported"].dt.year) * 12
        + (reporting_span["last_reported"].dt.month - reporting_span["first_reported"].dt.month)
        + 1
    )
    reporting_span["completeness_pct_fair"] = (
        reporting_span["months_reported"] / reporting_span["active_months"] * 100
    ).round(1)
    reporting_span["years_active"] = (
        (reporting_span["last_reported"] - reporting_span["first_reported"]).dt.days / 365.25
    ).round(1)

    long_history = reporting_span[
        (reporting_span["completeness_pct_fair"] >= COMPLETENESS_THRESHOLD)
        & (reporting_span["years_active"] >= 3)
    ].copy()
    recent_only = reporting_span[
        (reporting_span["completeness_pct_fair"] >= COMPLETENESS_THRESHOLD)
        & (reporting_span["years_active"] >= MIN_YEARS_ACTIVE)
        & (reporting_span["years_active"] < 3)
    ].copy()
    insufficient_data = reporting_span[
        (reporting_span["completeness_pct_fair"] >= COMPLETENESS_THRESHOLD)
        & (reporting_span["years_active"] < MIN_YEARS_ACTIVE)
    ]

    print(f"Long history pairs (3+ years): {len(long_history)}")
    print(f"Recent only pairs (1 to 3 years): {len(recent_only)}")
    print(f"Insufficient data pairs (under 1 year, excluded): {len(insufficient_data)}")

    long_history["model_track"] = "prophet"
    recent_only["model_track"] = "lstm"

    shortlist = pd.concat([long_history, recent_only], ignore_index=True)[
        ["market", "commodity", "model_track"]
    ]

    print(f"Total shortlisted market-commodity pairs: {len(shortlist)}")
    print(f"Unique markets: {shortlist['market'].nunique()}")
    print(f"Unique commodities: {shortlist['commodity'].nunique()}")

    return shortlist


def fetch_weather(market_coords):
    """
    Retrieve daily rainfall and temperature from the NASA POWER API for
    every shortlisted market's coordinates. A market that fails to return
    a 200 is skipped and logged, not silently dropped, since a partial
    weather join is expected (see README known limitations) but should
    always be visible when it happens.
    """
    weather_records = []
    failed_markets = []

    for _, row in market_coords.iterrows():
        params = {
            "parameters": "T2M,PRECTOTCORR",
            "community": "ag",
            "longitude": row["longitude"],
            "latitude": row["latitude"],
            "start": WEATHER_START,
            "end": WEATHER_END,
            "format": "JSON",
        }
        try:
            resp = requests.get(POWER_URL, params=params, timeout=60)
            if resp.status_code == 200:
                param_data = resp.json()["properties"]["parameter"]
                df_market = pd.DataFrame({
                    "date": pd.to_datetime(list(param_data["T2M"].keys()), format="%Y%m%d"),
                    "temperature": list(param_data["T2M"].values()),
                    "rainfall": list(param_data["PRECTOTCORR"].values()),
                })
                df_market["market"] = row["market"]
                weather_records.append(df_market)
            else:
                failed_markets.append((row["market"], resp.status_code))
        except requests.RequestException as error:
            failed_markets.append((row["market"], str(error)))

        time.sleep(1)  # avoid rate limiting, matches the notebook's original pacing

    if failed_markets:
        print(f"Weather retrieval failed for {len(failed_markets)} market(s): {failed_markets}")

    if not weather_records:
        raise RuntimeError("Weather retrieval failed for every market. Check network access to power.larc.nasa.gov.")

    print(f"Markets retrieved: {len(weather_records)} out of {len(market_coords)}")
    return pd.concat(weather_records, ignore_index=True)


def build_master(modeling_data, weather_all):
    """Aggregate weather to monthly, add lag features, and join to modeling_data."""
    weather_monthly = (
        weather_all.set_index("date")
        .groupby("market")
        .resample("ME")
        .agg({"rainfall": "sum", "temperature": "mean"})
        .reset_index()
    )

    weather_monthly = weather_monthly.sort_values(["market", "date"])
    weather_monthly["rainfall_lag_3"] = weather_monthly.groupby("market")["rainfall"].shift(3)
    weather_monthly["rainfall_lag_4"] = weather_monthly.groupby("market")["rainfall"].shift(4)
    weather_monthly["temperature_lag_3"] = weather_monthly.groupby("market")["temperature"].shift(3)
    weather_monthly["temperature_lag_4"] = weather_monthly.groupby("market")["temperature"].shift(4)

    modeling_data = modeling_data.copy()
    modeling_data["date_month"] = modeling_data["date"].values.astype("datetime64[M]")
    weather_monthly["date_month"] = weather_monthly["date"].values.astype("datetime64[M]")
    weather_features = weather_monthly.drop(columns=["date"])

    master = modeling_data.merge(weather_features, on=["market", "date_month"], how="left")
    master = master.sort_values("date")

    print(f"Master table rows: {len(master)}")
    print(f"Rows with matched weather data: {master['rainfall'].notna().sum()}")
    match_rate = master["rainfall"].notna().sum() / len(master) * 100
    print(f"Weather match rate: {match_rate:.2f}%")

    return master


def main():
    print(f"Data prep run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    prices_raw = load_food_prices()
    prices = clean_price_data(prices_raw)
    print(f"Rows: {prices.shape[0]}, Columns: {prices.shape[1]}")
    print(f"Date range: {prices['date'].min()} to {prices['date'].max()}")

    retail = prices[prices["pricetype"] == "Retail"].copy()
    print(f"Retail rows: {len(retail)} out of {len(prices)} total rows")

    excluded_units = sorted(set(retail["unit"]) - set(UNIT_TO_KG))
    unit_excluded_commodities = retail[retail["unit"].isin(excluded_units)]["commodity"].unique()
    print(f"Excluding {len(FUEL_COMMODITIES)} fuel commodities on scope grounds")
    print(f"Excluding {len(unit_excluded_commodities)} commodities priced in a non-weight unit: "
          f"{list(unit_excluded_commodities)}")

    retail = retail[
        ~retail["commodity"].isin(FUEL_COMMODITIES) & retail["unit"].isin(UNIT_TO_KG)
    ].copy()
    print(f"Retail rows after scope filtering: {len(retail)}")

    shortlist = build_shortlist(retail)

    retail["kg_equivalent"] = retail["unit"].map(UNIT_TO_KG)
    retail["price_per_kg"] = retail["price"] / retail["kg_equivalent"]
    modeling_data = retail.merge(shortlist, on=["market", "commodity"], how="inner")
    modeling_data = modeling_data[modeling_data["price_per_kg"].notna()].copy()

    market_coords = modeling_data[["market", "latitude", "longitude"]].drop_duplicates()
    weather_all = fetch_weather(market_coords)

    master = build_master(modeling_data, weather_all)

    os.makedirs("data", exist_ok=True)
    shortlist.to_parquet("data/shortlist.parquet", index=False)
    master.to_parquet("data/master.parquet", index=False)
    print("Saved data/shortlist.parquet")
    print("Saved data/master.parquet")

    print(f"Data prep run finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
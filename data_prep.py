import os
import time
import requests
import numpy as np
import pandas as pd

from pipeline_config import (
    RAW_PRICE_FILE, DATA_URL, UNIT_TO_KG, FUEL_COMMODITIES,
    MIN_YEARS_ACTIVE, LONG_HISTORY_YEARS, MIN_FAIR_COMPLETENESS_PCT,
    DATA_DIR, MASTER_PATH, SHORTLIST_PATH,
)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def load_food_prices():
    """Load the raw Kenya food prices dataset, local file first, then source URL."""
    if os.path.exists(RAW_PRICE_FILE):
        print(f"Loaded local dataset: {RAW_PRICE_FILE}")
        return pd.read_csv(RAW_PRICE_FILE)

    print("Local file not found, downloading from source...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    with open(RAW_PRICE_FILE, "wb") as f:
        f.write(response.content)
    print(f"Downloaded and saved as {RAW_PRICE_FILE}")
    return pd.read_csv(RAW_PRICE_FILE)


def clean_price_data(prices_raw):
    """Drop a units/description row if present, coerce date/price dtypes."""
    if prices_raw.iloc[0].astype(str).str.startswith("#").any():
        prices = prices_raw.iloc[1:].reset_index(drop=True)
    else:
        prices = prices_raw.copy()

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    if "usdprice" in prices.columns:
        prices["usdprice"] = pd.to_numeric(prices["usdprice"], errors="coerce")

    return prices

def standardize_units(prices):
    """Add kg_equivalent and price_per_kg. Rows with a non-weight unit get NaN, not an error."""
    prices = prices.copy()
    prices["kg_equivalent"] = prices["unit"].map(UNIT_TO_KG)
    prices["price_per_kg"] = prices["price"] / prices["kg_equivalent"]
    return prices


def select_scope_and_retail(prices):
    """Exclude fuel and non-weight-unit commodities, then split to retail only."""
    prices_scoped = prices[
        ~prices["commodity"].isin(FUEL_COMMODITIES) &
        prices["unit"].isin(UNIT_TO_KG)
    ].copy()

    retail = prices_scoped[prices_scoped["pricetype"] == "Retail"].copy()
    print(f"Scoped rows: {len(prices_scoped)} | Retail rows: {len(retail)}")
    return retail


def compute_shortlist(retail):
    """Classify each market-commodity pair by fair completeness and years of active history."""
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
        (reporting_span["completeness_pct_fair"] >= MIN_FAIR_COMPLETENESS_PCT) &
        (reporting_span["years_active"] >= LONG_HISTORY_YEARS)
    ].copy()
    recent_only = reporting_span[
        (reporting_span["completeness_pct_fair"] >= MIN_FAIR_COMPLETENESS_PCT) &
        (reporting_span["years_active"] >= MIN_YEARS_ACTIVE) &
        (reporting_span["years_active"] < LONG_HISTORY_YEARS)
    ].copy()

    long_history["model_track"] = "prophet"
    recent_only["model_track"] = "lstm"

    shortlist = pd.concat([long_history, recent_only], ignore_index=True)[["market", "commodity", "model_track"]]
    print(f"Shortlist: {len(shortlist)} pairs ({len(long_history)} prophet, {len(recent_only)} lstm)")
    return shortlist


def build_modeling_data(retail, shortlist):
    """Restrict retail rows to shortlisted pairs and require a valid price_per_kg."""
    modeling_data = retail.merge(shortlist, on=["market", "commodity"], how="inner")
    modeling_data = modeling_data[modeling_data["price_per_kg"].notna()].copy()
    print(f"Modelling-ready rows: {len(modeling_data)}")
    return modeling_data


def retrieve_weather(market_coords, start="20060101", end=None):
    """Pull daily rainfall and temperature for every shortlisted market's coordinates."""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y%m%d")

    weather_records = []
    for _, row in market_coords.iterrows():
        params = {
            "parameters": "T2M,PRECTOTCORR", "community": "ag",
            "longitude": row["longitude"], "latitude": row["latitude"],
            "start": start, "end": end, "format": "JSON",
        }
        try:
            resp = requests.get(POWER_URL, params=params, timeout=60)
            resp.raise_for_status()
            param_data = resp.json()["properties"]["parameter"]
            df_market = pd.DataFrame({
                "date": pd.to_datetime(list(param_data["T2M"].keys()), format="%Y%m%d"),
                "temperature": list(param_data["T2M"].values()),
                "rainfall": list(param_data["PRECTOTCORR"].values()),
            })
            df_market["market"] = row["market"]
            weather_records.append(df_market)
        except requests.exceptions.RequestException as e:
            print(f"Weather retrieval failed for {row['market']}: {e}")
        time.sleep(1)

    if not weather_records:
        raise RuntimeError("No weather data retrieved for any market.")

    weather_all = pd.concat(weather_records, ignore_index=True)
    print(f"Weather retrieved for {weather_all['market'].nunique()} of {len(market_coords)} markets")
    return weather_all


def aggregate_weather_monthly(weather_all):
    """Rainfall summed, temperature averaged, to match price's monthly grain."""
    return (
        weather_all.set_index("date")
        .groupby("market")
        .resample("ME")
        .agg({"rainfall": "sum", "temperature": "mean"})
        .reset_index()
    )


def merge_base_weather(modeling_data, weather_monthly):
    """Join raw (unlagged) monthly weather onto modeling_data. Lags are added in add_lag_features."""
    modeling_data = modeling_data.copy()
    weather_monthly = weather_monthly.copy()

    modeling_data["date_month"] = modeling_data["date"].values.astype("datetime64[M]")
    weather_monthly["date_month"] = weather_monthly["date"].values.astype("datetime64[M]")

    master = modeling_data.merge(
        weather_monthly.drop(columns=["date"]),
        on=["market", "date_month"], how="left"
    )
    print(f"Master rows: {len(master)} | matched weather: {master['rainfall'].notna().sum()}")
    return master

def add_target(master):
    """price_diff: month-over-month change in price_per_kg, per pair."""
    master = master.sort_values(["market", "commodity", "date"]).copy()
    master["price_diff"] = master.groupby(["market", "commodity"])["price_per_kg"].diff()
    return master


def add_lag_features(master, weather_monthly):
    """
    Shift rainfall/temperature on weather_monthly (one row per market per month),
    not on master directly -- master has multiple commodity rows per market-month,
    which would corrupt a row-position shift.
    """
    weather_monthly = weather_monthly.sort_values(["market", "date"]).copy()
    weather_monthly["rainfall_lag_3"] = weather_monthly.groupby("market")["rainfall"].shift(3)
    weather_monthly["rainfall_lag_4"] = weather_monthly.groupby("market")["rainfall"].shift(4)
    weather_monthly["temperature_lag_3"] = weather_monthly.groupby("market")["temperature"].shift(3)
    weather_monthly["temperature_lag_4"] = weather_monthly.groupby("market")["temperature"].shift(4)
    weather_monthly["date_month"] = weather_monthly["date"].values.astype("datetime64[M]")

    lag_cols = ["market", "date_month", "rainfall_lag_3", "rainfall_lag_4", "temperature_lag_3", "temperature_lag_4"]
    master = master.merge(weather_monthly[lag_cols], on=["market", "date_month"], how="left")
    print(f"Rows with matched lag features: {master['rainfall_lag_3'].notna().sum()} of {len(master)}")
    return master

def run_pipeline():
    """Run the full Data Preparation + Feature Engineering pipeline. Returns (master, shortlist)."""
    prices_raw = load_food_prices()
    prices = clean_price_data(prices_raw)
    prices = standardize_units(prices)

    retail = select_scope_and_retail(prices)
    shortlist = compute_shortlist(retail)
    modeling_data = build_modeling_data(retail, shortlist)

    market_coords = modeling_data[["market", "latitude", "longitude"]].drop_duplicates()
    weather_all = retrieve_weather(market_coords)
    weather_monthly = aggregate_weather_monthly(weather_all)

    master = merge_base_weather(modeling_data, weather_monthly)
    master = add_target(master)
    master = add_lag_features(master, weather_monthly)

    return master, shortlist


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    master, shortlist = run_pipeline()
    master.to_parquet(MASTER_PATH, index=False)
    shortlist.to_parquet(SHORTLIST_PATH, index=False)
    print(f"\nSaved {MASTER_PATH} ({len(master)} rows)")
    print(f"Saved {SHORTLIST_PATH} ({len(shortlist)} pairs)")
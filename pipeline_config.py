import os

SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RAW_PRICE_FILE = os.path.join(BASE_DIR, "wfp_food_prices_ken.csv")

MASTER_PATH = os.path.join(DATA_DIR, "master.parquet")
SHORTLIST_PATH = os.path.join(DATA_DIR, "shortlist.parquet")

FORECASTS_CSV = os.path.join(BASE_DIR, "forecasts.csv")
PRICE_HISTORY_CSV = os.path.join(BASE_DIR, "price_history.csv")

PROPHET_MODEL_DIR = os.path.join(MODEL_DIR, "prophet")  # one .json per prophet-track pair
POOLED_LSTM_PATH = os.path.join(MODEL_DIR, "pooled_lstm.keras")
SCALER_DIR = os.path.join(MODEL_DIR, "scalers")  # one .joblib per lstm-track pair
ROUTER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "router_weights.csv")

DATA_URL = (
    "https://data.humdata.org/dataset/"
    "e0d3fba6-f9a2-45d7-b949-140c455197ff/"
    "resource/517ee1bf-2437-4f8c-aa1b-cb9925b9d437/"
    "download/wfp_food_prices_ken.csv"
)


UNIT_TO_KG = {
    "KG": 1, "90 KG": 90, "64 KG": 64, "50 KG": 50, "26 KG": 26,
    "126 KG": 126, "13 KG": 13, "200 G": 0.2, "400 G": 0.4,
}

FUEL_COMMODITIES = ["Fuel (diesel)", "Fuel (kerosene)", "Fuel (petrol-gasoline)"]

MIN_YEARS_ACTIVE = 1.0          # floor for the recent-only (LSTM) track
LONG_HISTORY_YEARS = 3.0        # floor for the long-history (Prophet) track
MIN_FAIR_COMPLETENESS_PCT = 60  # completeness measured against each pair's own active window


WEATHER_FEATURES = ["rainfall_lag_3", "rainfall_lag_4", "temperature_lag_3", "temperature_lag_4"]
FEATURES = ["price_diff"] + WEATHER_FEATURES

LOOKBACK = 6
MIN_ROWS = LOOKBACK + 4


MIN_VAL_ROWS_FOR_ANY_WEIGHT = 4
FULL_CONFIDENCE_VAL_ROWS = 8
FORECAST_HORIZON = 1  # months ahead -- the only horizon this system is validated for

MIN_HISTORY_FOR_THRESHOLD = 6
RESIDUAL_STD_MULTIPLIER = 2

WEATHER_MONTHLY_PATH = os.path.join(DATA_DIR, "weather_monthly.parquet")
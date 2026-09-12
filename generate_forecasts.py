import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from prophet.serialize import model_from_json

import data_prep
from pipeline_config import (
    WEATHER_FEATURES, FEATURES, LOOKBACK,
    MIN_HISTORY_FOR_THRESHOLD, RESIDUAL_STD_MULTIPLIER,
    PROPHET_MODEL_DIR, POOLED_LSTM_PATH, SCALER_DIR, ROUTER_WEIGHTS_PATH,
    FORECASTS_CSV, PRICE_HISTORY_CSV,
)

WARNING_THRESHOLDS = [(10, "HIGH"), (5, "MODERATE"), (2, "LOW")]  # checked in order, first match wins


def classify_warning_level(expected_change_pct):
    if pd.isna(expected_change_pct):
        return "UNKNOWN"
    magnitude = abs(expected_change_pct)
    for threshold, label in WARNING_THRESHOLDS:
        if magnitude >= threshold:
            return label
    return "NORMAL"


def rebuild_anomaly_detection(master):
    """Identical logic to notebook Section 7 -- expanding, prior-only residual std."""
    master = master.sort_values("date").copy()
    master["expected_price"] = master.groupby(["market", "commodity"])["price_per_kg"].shift(1)
    master["residual"] = master["price_per_kg"] - master["expected_price"]
    master["residual_std_prior"] = master.groupby(["market", "commodity"])["residual"].transform(
        lambda s: s.expanding(min_periods=MIN_HISTORY_FOR_THRESHOLD).std().shift(1)
    )
    master["flagged"] = master["residual_std_prior"].notna() & (
        master["residual"].abs() > RESIDUAL_STD_MULTIPLIER * master["residual_std_prior"]
    )
    return master


def load_router_weights():
    if not os.path.exists(ROUTER_WEIGHTS_PATH):
        raise FileNotFoundError(f"{ROUTER_WEIGHTS_PATH} not found -- run train_model.py first.")
    weights = pd.read_csv(ROUTER_WEIGHTS_PATH)
    return weights.set_index(["market", "commodity"])["weight"].to_dict()


def get_future_regressors(weather_monthly, market, next_month_start):
    """next_month_start: a pandas Timestamp for the 1st of the month being forecast."""
    lag_specs = [
        (3, "rainfall_lag_3", "rainfall"), (4, "rainfall_lag_4", "rainfall"),
        (3, "temperature_lag_3", "temperature"), (4, "temperature_lag_4", "temperature"),
    ]
    lookups = {}
    for lag, feature_name, source_col in lag_specs:
        target_month = np.datetime64(next_month_start - pd.DateOffset(months=lag), "M")
        row = weather_monthly[(weather_monthly["market"] == market) & (weather_monthly["date_month"] == target_month)]
        if row.empty:
            return None
        lookups[feature_name] = row[source_col].iloc[0]
    return lookups


def forecast_prophet_pair(market, commodity, next_month_date, weather_monthly):
    path = os.path.join(PROPHET_MODEL_DIR, f"{market}__{commodity}.json".replace("/", "-"))
    if not os.path.exists(path):
        return None

    next_month_start = next_month_date.replace(day=1)
    regressors = get_future_regressors(weather_monthly, market, next_month_start)
    if regressors is None:
        return None

    with open(path) as f:
        model = model_from_json(f.read())

    future_row = pd.DataFrame([{"ds": next_month_date, **regressors}])
    forecast = model.predict(future_row)
    return {
        "point": float(forecast["yhat"].iloc[0]),
        "lower": float(forecast["yhat_lower"].iloc[0]),
        "upper": float(forecast["yhat_upper"].iloc[0]),
    }


def forecast_lstm_pair(market, commodity, pair_df, pooled_model):
    scaler_path = os.path.join(SCALER_DIR, f"{market}__{commodity}.joblib".replace("/", "-"))
    if not os.path.exists(scaler_path):
        return None

    recent = pair_df.sort_values("date").dropna(subset=FEATURES).tail(LOOKBACK)
    if len(recent) < LOOKBACK:
        return None

    scaler = joblib.load(scaler_path)
    scaled = scaler.transform(recent[FEATURES])
    X = scaled.reshape(1, LOOKBACK, len(FEATURES))

    predicted_diff_scaled = pooled_model.predict(X, verbose=0).flatten()[0]
    diff_matrix = np.zeros((1, len(FEATURES)))
    diff_matrix[:, 0] = predicted_diff_scaled
    predicted_diff = scaler.inverse_transform(diff_matrix)[0, 0]

    current_price = pair_df.sort_values("date")["price_per_kg"].iloc[-1]
    return current_price + predicted_diff


def build_forecasts(master, shortlist, weather_monthly, router_weights, pooled_model):
    rows = []

    for _, row in shortlist.iterrows():
        market, commodity, track = row["market"], row["commodity"], row["model_track"]
        pair_df = master[(master["market"] == market) & (master["commodity"] == commodity)].sort_values("date")
        if pair_df.empty:
            continue

        last_date = pair_df["date"].iloc[-1]
        next_month_date = last_date + pd.DateOffset(months=1)
        current_price = pair_df["price_per_kg"].iloc[-1]
        naive_forecast = current_price

        if track == "prophet":
            prophet_result = forecast_prophet_pair(market, commodity, next_month_date, weather_monthly)
            model_forecast = prophet_result["point"] if prophet_result else None
            forecast_lower = prophet_result["lower"] if prophet_result else np.nan
            forecast_upper = prophet_result["upper"] if prophet_result else np.nan
        else:
            model_forecast = forecast_lstm_pair(market, commodity, pair_df, pooled_model)
            forecast_lower, forecast_upper = np.nan, np.nan

        weight = router_weights.get((market, commodity), 0.0)
        model_available = model_forecast is not None
        if not model_available:
            weight = 0.0

        display_forecast = (
            weight * model_forecast + (1 - weight) * naive_forecast if model_available else naive_forecast
        )
        chosen_forecast = "model" if weight >= 0.5 else ("blend" if weight > 0 else "naive")

        expected_change_pct = (
            (display_forecast - current_price) / current_price * 100 if current_price else np.nan
        )

        rows.append({
            "market": market, "commodity": commodity,
            "latitude": pair_df["latitude"].iloc[-1], "longitude": pair_df["longitude"].iloc[-1],
            "model_track": track, "model_available": model_available,
            "current_price": current_price, "forecast_date": next_month_date.strftime("%Y-%m-%d"),
            "chosen_forecast": chosen_forecast, "weight": weight,
            "naive_forecast": naive_forecast, "display_forecast": display_forecast,
            "expected_change_pct": expected_change_pct,
            "warning_level": classify_warning_level(expected_change_pct),
        })

    return pd.DataFrame(rows)


def main():
    print("Refreshing data via data_prep.run_pipeline()...")
    master, shortlist, weather_monthly = data_prep.run_pipeline()
    master = rebuild_anomaly_detection(master)

    router_weights = load_router_weights()
    pooled_model = tf.keras.models.load_model(POOLED_LSTM_PATH)

    forecasts = build_forecasts(master, shortlist, weather_monthly, router_weights, pooled_model)

    latest_flag = (
        master.sort_values("date").groupby(["market", "commodity"])["flagged"].last()
        .reset_index().rename(columns={"flagged": "latest_flagged"})
    )
    forecasts = forecasts.merge(latest_flag, on=["market", "commodity"], how="left")

    forecasts.to_csv(FORECASTS_CSV, index=False)
    print(f"Saved {FORECASTS_CSV} ({len(forecasts)} rows)")

    price_history = master[["market", "commodity", "date", "price_per_kg", "expected_price", "flagged"]].copy()
    price_history.to_csv(PRICE_HISTORY_CSV, index=False)
    print(f"Saved {PRICE_HISTORY_CSV} ({len(price_history)} rows)")

    print(f"\nPairs with unavailable model (naive fallback): {(~forecasts['model_available']).sum()}")
    print(f"Warning level distribution:\n{forecasts['warning_level'].value_counts()}")


if __name__ == "__main__":
    main()
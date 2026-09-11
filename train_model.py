import os
import json
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import data_prep
from pipeline_config import (
    SEED, MASTER_PATH, SHORTLIST_PATH, WEATHER_FEATURES, FEATURES,
    LOOKBACK, MIN_ROWS, MIN_VAL_ROWS_FOR_ANY_WEIGHT, FULL_CONFIDENCE_VAL_ROWS,
    PROPHET_MODEL_DIR, POOLED_LSTM_PATH, SCALER_DIR, ROUTER_WEIGHTS_PATH, MODEL_DIR,
)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



def chronological_split(df, date_col="date"):
    """80/10/10 split by date quantile, computed per pair -- pairs don't share a start date."""
    train_end = df[date_col].quantile(0.8)
    val_end = df[date_col].quantile(0.9)
    train = df[df[date_col] <= train_end]
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)]
    test = df[df[date_col] > val_end]
    return train, val, test


def naive_metrics(df, price_col="price_per_kg"):
    """MAE/MAPE for predicting each month as the previous month's price."""
    naive_pred = df[price_col].shift(1)
    ready = naive_pred.notna()
    if not ready.any():
        return np.nan, np.nan
    actual = df.loc[ready, price_col]
    pred = naive_pred.loc[ready]
    mae = mean_absolute_error(actual, pred)
    nz = actual != 0
    mape = np.mean(np.abs((actual[nz] - pred[nz]) / actual[nz])) * 100
    return mae, mape


def compute_model_weight(val_rows, val_naive_mae, val_model_mae):
    if val_rows < MIN_VAL_ROWS_FOR_ANY_WEIGHT or np.isnan(val_naive_mae) or val_naive_mae == 0:
        return 0.0
    improvement = (val_naive_mae - val_model_mae) / val_naive_mae
    confidence = min(val_rows / FULL_CONFIDENCE_VAL_ROWS, 1.0)
    return max(0.0, min(improvement, 1.0)) * confidence


def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_prophet_pair(pair_df):
    """Validation pass for one prophet-track pair. Returns metrics needed for the router."""
    data = pair_df[["date", "price_per_kg"] + WEATHER_FEATURES].dropna()
    data = data.rename(columns={"date": "ds", "price_per_kg": "y"})
    if len(data) < MIN_ROWS:
        return None

    train, val, test = chronological_split(data, date_col="ds")
    if len(train) < 10 or val.empty:
        return None

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for feature in WEATHER_FEATURES:
        model.add_regressor(feature)
    model.fit(train)

    val_forecast = model.predict(val[["ds"] + WEATHER_FEATURES])
    val_mae = mean_absolute_error(val["y"], val_forecast["yhat"])

    val_naive_mae, _ = naive_metrics(
        pair_df[pair_df["date"] <= val["ds"].max()].rename(columns={"date": "ds"}).assign(price_per_kg=lambda d: d["price_per_kg"])
    )

    test_mae, test_mape = (np.nan, np.nan)
    if not test.empty:
        test_forecast = model.predict(test[["ds"] + WEATHER_FEATURES])
        test_mae = mean_absolute_error(test["y"], test_forecast["yhat"])
        nz = test["y"] != 0
        test_mape = np.mean(np.abs((test["y"][nz] - test_forecast["yhat"][nz]) / test["y"][nz])) * 100

    naive_test_mae, naive_test_mape = naive_metrics(pair_df.rename(columns={"date": "date"}))

    return {
        "val_rows": len(val), "val_mae": val_mae, "val_naive_mae": val_naive_mae,
        "test_mae": test_mae, "test_mape": test_mape,
        "naive_test_mae": naive_test_mae, "naive_test_mape": naive_test_mape,
    }


def fit_final_prophet(pair_df, market, commodity):
    """Refit on the pair's full history and save as JSON."""
    data = pair_df[["date", "price_per_kg"] + WEATHER_FEATURES].dropna()
    data = data.rename(columns={"date": "ds", "price_per_kg": "y"})

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for feature in WEATHER_FEATURES:
        model.add_regressor(feature)
    model.fit(data)

    os.makedirs(PROPHET_MODEL_DIR, exist_ok=True)
    path = os.path.join(PROPHET_MODEL_DIR, f"{market}__{commodity}.json".replace("/", "-"))
    with open(path, "w") as f:
        f.write(model_to_json(model))


def run_prophet_track(master, shortlist):
    prophet_pairs = shortlist[shortlist["model_track"] == "prophet"]
    results = []

    for _, row in prophet_pairs.iterrows():
        pair_df = master[(master["market"] == row["market"]) & (master["commodity"] == row["commodity"])].sort_values("date")
        metrics = train_prophet_pair(pair_df)
        if metrics is None:
            continue

        weight = compute_model_weight(metrics["val_rows"], metrics["val_naive_mae"], metrics["val_mae"])
        fit_final_prophet(pair_df, row["market"], row["commodity"])

        results.append({
            "market": row["market"], "commodity": row["commodity"], "model_track": "prophet",
            "weight": weight, **{k: v for k, v in metrics.items() if k != "val_rows"},
            "val_rows": metrics["val_rows"],
        })

    print(f"Prophet track: {len(results)} of {len(prophet_pairs)} pairs trained")
    return pd.DataFrame(results)



def build_pair_sequences(pair_df, lookback=LOOKBACK):
    """Validation-pass sequence builder -- train/val/test, scaler fit on train only."""
    pair_df = pair_df.sort_values("date").dropna(subset=FEATURES)
    if len(pair_df) < MIN_ROWS:
        return None

    train, val, test = chronological_split(pair_df)
    if len(train) <= lookback:
        return None

    naive_pred = pair_df["price_per_kg"].shift(1)
    val_mask = pair_df.index.isin(val.index)
    test_mask = pair_df.index.isin(test.index)

    val_naive_ready = val_mask & naive_pred.notna()
    val_naive_mae = (
        mean_absolute_error(pair_df.loc[val_naive_ready, "price_per_kg"], naive_pred.loc[val_naive_ready])
        if val_naive_ready.any() else np.nan
    )
    test_naive_ready = test_mask & naive_pred.notna()
    if not test_naive_ready.any():
        return None
    test_naive_mae = mean_absolute_error(pair_df.loc[test_naive_ready, "price_per_kg"], naive_pred.loc[test_naive_ready])

    scaler = MinMaxScaler()
    scaler.fit(train[FEATURES])
    train_s, val_s, test_s = scaler.transform(train[FEATURES]), scaler.transform(val[FEATURES]), scaler.transform(test[FEATURES])

    X_train, y_train = create_sequences(train_s, lookback)
    val_source = np.vstack([train_s[-lookback:], val_s]) if len(val_s) > 0 else train_s[-lookback:]
    X_val, y_val = create_sequences(val_source, lookback)
    test_source = np.vstack([val_s[-lookback:], test_s]) if len(val_s) >= lookback else np.vstack([train_s[-lookback:], test_s])
    X_test, y_test = create_sequences(test_source, lookback)

    if len(X_train) < 1 or len(X_test) == 0:
        return None

    return {
        "market": pair_df["market"].iloc[0], "commodity": pair_df["commodity"].iloc[0],
        "scaler": scaler, "val_naive_mae": val_naive_mae, "test_naive_mae": test_naive_mae,
        "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test,
        "last_price_val": pair_df["price_per_kg"].shift(1).loc[val_mask].values,
        "actual_price_val": pair_df.loc[val_mask, "price_per_kg"].values,
        "last_price_test": pair_df["price_per_kg"].shift(1).loc[test_mask].values,
        "actual_price_test": pair_df.loc[test_mask, "price_per_kg"].values,
    }


def build_pooled_lstm(n_features, lookback=LOOKBACK):
    model = Sequential([
        LSTM(16, input_shape=(lookback, n_features), return_sequences=False, recurrent_dropout=0.2),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def run_lstm_track(master, shortlist):
    lstm_pairs = shortlist[shortlist["model_track"] == "lstm"]
    bundles, results = [], []

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list, pair_idx_val_list, last_price_val_list, actual_price_val_list = [], [], [], [], []
    X_test_list, y_test_list, pair_idx_test_list, last_price_test_list, actual_price_test_list = [], [], [], [], []

    for _, row in lstm_pairs.iterrows():
        pair_df = master[(master["market"] == row["market"]) & (master["commodity"] == row["commodity"])]
        bundle = build_pair_sequences(pair_df)
        if bundle is None:
            continue
        pair_idx = len(bundles)
        bundles.append(bundle)

        X_train_list.append(bundle["X_train"]); y_train_list.append(bundle["y_train"])
        X_val_list.append(bundle["X_val"]); y_val_list.append(bundle["y_val"])
        pair_idx_val_list.append(np.full(len(bundle["X_val"]), pair_idx))
        last_price_val_list.append(bundle["last_price_val"]); actual_price_val_list.append(bundle["actual_price_val"])
        X_test_list.append(bundle["X_test"]); y_test_list.append(bundle["y_test"])
        pair_idx_test_list.append(np.full(len(bundle["X_test"]), pair_idx))
        last_price_test_list.append(bundle["last_price_test"]); actual_price_test_list.append(bundle["actual_price_test"])

    print(f"LSTM track: {len(bundles)} of {len(lstm_pairs)} pairs included")

    X_train_all, y_train_all = np.concatenate(X_train_list), np.concatenate(y_train_list)
    X_val_all, y_val_all = np.concatenate(X_val_list), np.concatenate(y_val_list)
    pair_idx_val_all = np.concatenate(pair_idx_val_list)
    last_price_val_all, actual_price_val_all = np.concatenate(last_price_val_list), np.concatenate(actual_price_val_list)
    X_test_all = np.concatenate(X_test_list)
    pair_idx_test_all = np.concatenate(pair_idx_test_list)
    last_price_test_all, actual_price_test_all = np.concatenate(last_price_test_list), np.concatenate(actual_price_test_list)

    n_features = len(FEATURES)
    model = build_pooled_lstm(n_features)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train_all, y_train_all, validation_data=(X_val_all, y_val_all),
              epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)

    predicted_val_delta = model.predict(X_val_all, verbose=0).flatten()
    predicted_test_delta = model.predict(X_test_all, verbose=0).flatten()

    for pair_idx, bundle in enumerate(bundles):
        val_rows_mask = pair_idx_val_all == pair_idx
        test_rows_mask = pair_idx_test_all == pair_idx

        val_mae = np.nan
        if val_rows_mask.sum() > 0:
            diff_matrix = np.zeros((val_rows_mask.sum(), n_features)); diff_matrix[:, 0] = predicted_val_delta[val_rows_mask]
            pred_diff = bundle["scaler"].inverse_transform(diff_matrix)[:, 0]
            val_pred = last_price_val_all[val_rows_mask] + pred_diff
            val_mae = mean_absolute_error(actual_price_val_all[val_rows_mask], val_pred)

        test_mae, test_mape = np.nan, np.nan
        if test_rows_mask.sum() > 0:
            diff_matrix = np.zeros((test_rows_mask.sum(), n_features)); diff_matrix[:, 0] = predicted_test_delta[test_rows_mask]
            pred_diff = bundle["scaler"].inverse_transform(diff_matrix)[:, 0]
            test_pred = last_price_test_all[test_rows_mask] + pred_diff
            actual = actual_price_test_all[test_rows_mask]
            test_mae = mean_absolute_error(actual, test_pred)
            nz = actual != 0
            test_mape = np.mean(np.abs((actual[nz] - test_pred[nz]) / actual[nz])) * 100

        weight = compute_model_weight(val_rows_mask.sum(), bundle["val_naive_mae"], val_mae)

        results.append({
            "market": bundle["market"], "commodity": bundle["commodity"], "model_track": "lstm",
            "weight": weight, "val_rows": val_rows_mask.sum(), "val_mae": val_mae, "val_naive_mae": bundle["val_naive_mae"],
            "test_mae": test_mae, "test_mape": test_mape,
            "naive_test_mae": bundle["test_naive_mae"], "naive_test_mape": np.nan,
        })

    return pd.DataFrame(results), lstm_pairs


def fit_final_pooled_lstm(master, lstm_pairs):
    """Refit the pooled LSTM on every lstm-track pair's FULL history, and save a full-data scaler per pair."""
    os.makedirs(SCALER_DIR, exist_ok=True)
    X_list, y_list = [], []

    for _, row in lstm_pairs.iterrows():
        pair_df = master[(master["market"] == row["market"]) & (master["commodity"] == row["commodity"])]
        pair_df = pair_df.sort_values("date").dropna(subset=FEATURES)
        if len(pair_df) <= LOOKBACK:
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(pair_df[FEATURES])
        X, y = create_sequences(scaled, LOOKBACK)
        if len(X) == 0:
            continue
        X_list.append(X); y_list.append(y)

        scaler_path = os.path.join(SCALER_DIR, f"{row['market']}__{row['commodity']}.joblib".replace("/", "-"))
        joblib.dump(scaler, scaler_path)

    X_all, y_all = np.concatenate(X_list), np.concatenate(y_list)
    model = build_pooled_lstm(len(FEATURES))
    early_stop = EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)
    model.fit(X_all, y_all, epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)
    model.save(POOLED_LSTM_PATH)
    print(f"Final pooled LSTM saved: {POOLED_LSTM_PATH} ({len(X_all)} sequences across {len(X_list)} pairs)")


def main():
    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError(f"{MASTER_PATH} not found -- run data_prep.py first.")

    master = pd.read_parquet(MASTER_PATH)
    shortlist = pd.read_parquet(SHORTLIST_PATH)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=== Prophet track ===")
    prophet_results = run_prophet_track(master, shortlist)

    print("\n=== LSTM track ===")
    lstm_results, lstm_pairs = run_lstm_track(master, shortlist)
    fit_final_pooled_lstm(master, lstm_pairs)

    router_weights = pd.concat([prophet_results, lstm_results], ignore_index=True)
    router_weights.to_csv(ROUTER_WEIGHTS_PATH, index=False)

    print(f"\nRouter weights saved: {ROUTER_WEIGHTS_PATH}")
    print(f"Pairs with zero weight: {(router_weights['weight'] == 0).sum()}")
    print(f"Pairs with partial weight: {((router_weights['weight'] > 0) & (router_weights['weight'] < 0.5)).sum()}")
    print(f"Pairs with weight >= 0.5: {(router_weights['weight'] >= 0.5).sum()}")
    print(f"Max weight: {router_weights['weight'].max():.3f}")


if __name__ == "__main__":
    main() 
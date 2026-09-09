"""
train_model.py

Loads master.parquet and shortlist.parquet from data_prep.py, builds
per-pair sequences, trains the pooled entity-embedding LSTM, computes
each pair's blended router weight, and saves everything downstream
scripts need: the trained model, the label encoders, the per-pair
scalers, and the router's own per-pair results.

This is a direct port of section 4.21 (Global Entity Embedding Model)
of Kenyan_food_prices.ipynb, including the finalized blended router
from section 4.21.6, confirmed against real notebook output before
being ported here.

Run standalone, after data_prep.py has produced data/master.parquet
and data/shortlist.parquet:
    python train_model.py
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Concatenate, Dense, Dropout, Embedding, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

SEED = 42

FEATURES = ["price_diff", "rainfall_lag_3", "rainfall_lag_4", "temperature_lag_3", "temperature_lag_4"]
LOOKBACK = 6
MIN_ROWS = LOOKBACK + 4

MIN_VAL_ROWS_FOR_ANY_WEIGHT = 4
FULL_CONFIDENCE_VAL_ROWS = 8


def set_seeds(seed=SEED):
    """Fix all sources of randomness so this script's results are reproducible run to run."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_sequences(data, lookback):
    """Convert a scaled time series into rolling input sequences, target is the next diff."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_pair_sequences(pair_df, market_id, commodity_id, model_track, lookback=LOOKBACK):
    """
    Build one pair's train, validation, and test sequences, its own scaler
    fit only on that pair's training rows, and naive baseline MAE for both
    validation and test. Returns None if the pair doesn't have enough
    usable history, callers are expected to skip those pairs.
    """
    pair_df = pair_df.sort_values("date").copy()
    pair_df["price_diff"] = pair_df["price_per_kg"].diff()
    pair_df = pair_df.dropna(subset=FEATURES)

    if len(pair_df) < MIN_ROWS:
        return None

    train_end = pair_df["date"].quantile(0.8)
    val_end = pair_df["date"].quantile(0.9)

    train_mask = pair_df["date"] <= train_end
    val_mask = (pair_df["date"] > train_end) & (pair_df["date"] <= val_end)
    test_mask = pair_df["date"] > val_end

    naive_pred = pair_df["price_per_kg"].shift(1)
    naive_ready = test_mask & naive_pred.notna()
    if not naive_ready.any():
        return None

    naive_actual = pair_df.loc[naive_ready, "price_per_kg"]
    naive_pred_vals = naive_pred.loc[naive_ready]
    naive_mae = mean_absolute_error(naive_actual, naive_pred_vals)
    naive_mape = np.mean(np.abs((naive_actual - naive_pred_vals) / naive_actual)) * 100

    val_naive_ready = val_mask & naive_pred.notna()
    if val_naive_ready.any():
        val_naive_actual = pair_df.loc[val_naive_ready, "price_per_kg"]
        val_naive_pred_vals = naive_pred.loc[val_naive_ready]
        val_naive_mae = mean_absolute_error(val_naive_actual, val_naive_pred_vals)
    else:
        val_naive_mae = np.nan

    scaler = MinMaxScaler()
    scaler.fit(pair_df.loc[train_mask, FEATURES])

    train_scaled = scaler.transform(pair_df.loc[train_mask, FEATURES])
    val_scaled = scaler.transform(pair_df.loc[val_mask, FEATURES])
    test_scaled = scaler.transform(pair_df.loc[test_mask, FEATURES])

    if len(train_scaled) <= lookback:
        return None

    X_train, y_train = create_sequences(train_scaled, lookback)

    val_source = np.vstack([train_scaled[-lookback:], val_scaled]) if len(val_scaled) > 0 else train_scaled[-lookback:]
    X_val, y_val = create_sequences(val_source, lookback)

    test_source = (
        np.vstack([val_scaled[-lookback:], test_scaled])
        if len(val_scaled) >= lookback
        else np.vstack([train_scaled[-lookback:], test_scaled])
    )
    X_test, y_test = create_sequences(test_source, lookback)

    if len(X_train) < 1 or len(X_test) == 0:
        return None

    last_price_val = pair_df["price_per_kg"].shift(1).loc[val_mask].values
    actual_price_val = pair_df.loc[val_mask, "price_per_kg"].values
    last_price_test = pair_df["price_per_kg"].shift(1).loc[test_mask].values
    actual_price_test = pair_df.loc[test_mask, "price_per_kg"].values

    return {
        "market": pair_df["market"].iloc[0],
        "commodity": pair_df["commodity"].iloc[0],
        "market_id": market_id,
        "commodity_id": commodity_id,
        "model_track": model_track,
        "scaler": scaler,
        "naive_mae": naive_mae,
        "naive_mape": naive_mape,
        "val_naive_mae": val_naive_mae,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "last_price_val": last_price_val,
        "actual_price_val": actual_price_val,
        "last_price_test": last_price_test,
        "actual_price_test": actual_price_test,
    }


def build_batch(master, shortlist, market_encoder, commodity_encoder):
    """Build every pair's sequences and stack them into one pooled training batch."""
    pair_bundles = []
    X_train_list, y_train_list, mkt_train_list, com_train_list = [], [], [], []
    X_val_list, y_val_list, mkt_val_list, com_val_list = [], [], [], []
    X_test_list, y_test_list, mkt_test_list, com_test_list = [], [], [], []
    pair_idx_val_list, pair_idx_test_list = [], []
    last_price_val_list, actual_price_val_list = [], []
    last_price_test_list, actual_price_test_list = [], []

    for _, row in shortlist.iterrows():
        pair_df = master[(master["market"] == row["market"]) & (master["commodity"] == row["commodity"])]

        market_id = market_encoder.transform([row["market"]])[0]
        commodity_id = commodity_encoder.transform([row["commodity"]])[0]

        bundle = build_pair_sequences(pair_df, market_id, commodity_id, row["model_track"])
        if bundle is None:
            continue

        pair_idx = len(pair_bundles)
        pair_bundles.append(bundle)

        n_tr, n_va, n_te = len(bundle["X_train"]), len(bundle["X_val"]), len(bundle["X_test"])

        X_train_list.append(bundle["X_train"]); y_train_list.append(bundle["y_train"])
        mkt_train_list.append(np.full(n_tr, market_id)); com_train_list.append(np.full(n_tr, commodity_id))

        X_val_list.append(bundle["X_val"]); y_val_list.append(bundle["y_val"])
        mkt_val_list.append(np.full(n_va, market_id)); com_val_list.append(np.full(n_va, commodity_id))
        pair_idx_val_list.append(np.full(n_va, pair_idx))
        last_price_val_list.append(bundle["last_price_val"])
        actual_price_val_list.append(bundle["actual_price_val"])

        X_test_list.append(bundle["X_test"]); y_test_list.append(bundle["y_test"])
        mkt_test_list.append(np.full(n_te, market_id)); com_test_list.append(np.full(n_te, commodity_id))
        pair_idx_test_list.append(np.full(n_te, pair_idx))
        last_price_test_list.append(bundle["last_price_test"])
        actual_price_test_list.append(bundle["actual_price_test"])

    print(f"Pairs included: {len(pair_bundles)} out of {len(shortlist)}")

    batch = {
        "pair_bundles": pair_bundles,
        "X_train_all": np.concatenate(X_train_list),
        "y_train_all": np.concatenate(y_train_list),
        "market_train_all": np.concatenate(mkt_train_list),
        "commodity_train_all": np.concatenate(com_train_list),
        "X_val_all": np.concatenate(X_val_list),
        "y_val_all": np.concatenate(y_val_list),
        "market_val_all": np.concatenate(mkt_val_list),
        "commodity_val_all": np.concatenate(com_val_list),
        "pair_idx_val_all": np.concatenate(pair_idx_val_list),
        "last_price_val_all": np.concatenate(last_price_val_list),
        "actual_price_val_all": np.concatenate(actual_price_val_list),
        "X_test_all": np.concatenate(X_test_list),
        "y_test_all": np.concatenate(y_test_list),
        "market_test_all": np.concatenate(mkt_test_list),
        "commodity_test_all": np.concatenate(com_test_list),
        "pair_idx_test_all": np.concatenate(pair_idx_test_list),
        "last_price_test_all": np.concatenate(last_price_test_list),
        "actual_price_test_all": np.concatenate(actual_price_test_list),
    }
    return batch


def build_embedding_model(n_markets, n_commodities, n_features):
    """Pooled entity-embedding LSTM, market and commodity embeddings concatenated with the LSTM output."""
    market_embed_dim = min(8, (n_markets + 1) // 2)
    commodity_embed_dim = min(8, (n_commodities + 1) // 2)

    sequence_input = Input(shape=(LOOKBACK, n_features), name="sequence_input")
    market_input = Input(shape=(1,), name="market_input")
    commodity_input = Input(shape=(1,), name="commodity_input")

    market_embed = Embedding(
        n_markets, market_embed_dim, embeddings_regularizer=l2(0.01), name="market_embedding"
    )(market_input)
    market_embed = Flatten()(market_embed)

    commodity_embed = Embedding(
        n_commodities, commodity_embed_dim, embeddings_regularizer=l2(0.01), name="commodity_embedding"
    )(commodity_input)
    commodity_embed = Flatten()(commodity_embed)

    lstm_out = LSTM(16, return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=l2(0.01))(sequence_input)
    lstm_out = Dropout(0.3)(lstm_out)

    merged = Concatenate()([lstm_out, market_embed, commodity_embed])
    dense_out = Dense(16, activation="relu", kernel_regularizer=l2(0.01))(merged)
    dense_out = Dropout(0.3)(dense_out)
    output = Dense(1)(dense_out)

    model = Model(inputs=[sequence_input, market_input, commodity_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def compute_model_weight(val_rows, val_naive_mae, val_model_mae):
    """
    Weighted blend, not a hard switch, informed by the forecast combination
    literature (Bates and Granger, 1969; repeatedly confirmed in the
    M-competitions). Weight scales with both the strength and the volume
    of validation evidence, a pair with no real evidence collapses to
    exactly 0, identical to pure naive.
    """
    if val_rows < MIN_VAL_ROWS_FOR_ANY_WEIGHT or np.isnan(val_naive_mae) or val_naive_mae == 0:
        return 0.0
    improvement = (val_naive_mae - val_model_mae) / val_naive_mae
    confidence = min(val_rows / FULL_CONFIDENCE_VAL_ROWS, 1.0)
    return max(0.0, min(improvement, 1.0)) * confidence


def run_router(embedding_model, batch, n_features):
    """
    Score every pair on validation, compute its blend weight, then score
    the blended forecast on test. This never touches test data to make
    the weighting decision, only to report the final honest number.
    """
    pair_bundles = batch["pair_bundles"]
    pair_idx_val_all = batch["pair_idx_val_all"]
    pair_idx_test_all = batch["pair_idx_test_all"]
    last_price_val_all = batch["last_price_val_all"]
    actual_price_val_all = batch["actual_price_val_all"]
    last_price_test_all = batch["last_price_test_all"]
    actual_price_test_all = batch["actual_price_test_all"]

    predicted_val_delta = embedding_model.predict(
        [batch["X_val_all"], batch["market_val_all"], batch["commodity_val_all"]], verbose=0
    ).flatten()
    predicted_test_delta = embedding_model.predict(
        [batch["X_test_all"], batch["market_test_all"], batch["commodity_test_all"]], verbose=0
    ).flatten()

    selection_results = []

    for pair_idx, bundle in enumerate(pair_bundles):
        val_mask_rows = pair_idx_val_all == pair_idx
        test_mask_rows = pair_idx_test_all == pair_idx

        if not test_mask_rows.any():
            continue

        scaler = bundle["scaler"]

        weight = 0.0
        if val_mask_rows.sum() >= MIN_VAL_ROWS_FOR_ANY_WEIGHT and not np.isnan(bundle["val_naive_mae"]):
            val_diff_matrix = np.zeros((val_mask_rows.sum(), n_features))
            val_diff_matrix[:, 0] = predicted_val_delta[val_mask_rows]
            val_predicted_diff = scaler.inverse_transform(val_diff_matrix)[:, 0]
            val_pred = last_price_val_all[val_mask_rows] + val_predicted_diff
            val_actual = actual_price_val_all[val_mask_rows]
            val_model_mae = mean_absolute_error(val_actual, val_pred)

            weight = compute_model_weight(val_mask_rows.sum(), bundle["val_naive_mae"], val_model_mae)

        test_diff_matrix = np.zeros((test_mask_rows.sum(), n_features))
        test_diff_matrix[:, 0] = predicted_test_delta[test_mask_rows]
        test_predicted_diff = scaler.inverse_transform(test_diff_matrix)[:, 0]
        model_pred = last_price_test_all[test_mask_rows] + test_predicted_diff
        naive_pred_test = last_price_test_all[test_mask_rows]
        actual = actual_price_test_all[test_mask_rows]

        blended_pred = weight * model_pred + (1 - weight) * naive_pred_test

        final_mae = mean_absolute_error(actual, blended_pred)
        nz = actual != 0
        final_mape = np.mean(np.abs((actual[nz] - blended_pred[nz]) / actual[nz])) * 100

        if weight >= 0.5:
            label = "model"
        elif weight > 0:
            label = "blend"
        else:
            label = "naive"

        selection_results.append({
            "market": bundle["market"],
            "commodity": bundle["commodity"],
            "model_track": bundle["model_track"],
            "weight": weight,
            "chosen_forecast": label,
            "final_mae": final_mae,
            "final_mape": final_mape,
            "naive_mae": bundle["naive_mae"],
            "naive_forecast": naive_pred_test[-1],
            "model_forecast": model_pred[-1],
            "blended_forecast": blended_pred[-1],
        })

    selection_results = pd.DataFrame(selection_results)
    selection_results["beats_naive"] = selection_results["final_mae"] <= selection_results["naive_mae"]
    return selection_results


def main():
    set_seeds()
    print(f"Train run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (seed={SEED})")

    master = pd.read_parquet("data/master.parquet")
    shortlist = pd.read_parquet("data/shortlist.parquet")
    print(f"Loaded master: {len(master)} rows, shortlist: {len(shortlist)} pairs")

    market_encoder = LabelEncoder()
    commodity_encoder = LabelEncoder()
    market_encoder.fit(shortlist["market"])
    commodity_encoder.fit(shortlist["commodity"])

    n_markets = len(market_encoder.classes_)
    n_commodities = len(commodity_encoder.classes_)
    n_features = len(FEATURES)

    batch = build_batch(master, shortlist, market_encoder, commodity_encoder)

    embedding_model = build_embedding_model(n_markets, n_commodities, n_features)
    embedding_model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    embedding_model.fit(
        [batch["X_train_all"], batch["market_train_all"], batch["commodity_train_all"]],
        batch["y_train_all"],
        validation_data=(
            [batch["X_val_all"], batch["market_val_all"], batch["commodity_val_all"]],
            batch["y_val_all"],
        ),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    selection_results = run_router(embedding_model, batch, n_features)

    print(f"Pairs with zero model weight (pure naive): {(selection_results['weight'] == 0).sum()}")
    print(f"Pairs with weight >= 0.5 (model-leaning): {(selection_results['weight'] >= 0.5).sum()}")
    print(f"Overall mean MAE with blending: {selection_results['final_mae'].mean():.2f}")
    print(f"Pairs at or better than naive: {selection_results['beats_naive'].mean() * 100:.1f}%")

    os.makedirs("model", exist_ok=True)
    embedding_model.save("model/embedding_model.keras")

    with open("model/encoders_and_scalers.pkl", "wb") as f:
        pickle.dump({
            "market_encoder": market_encoder,
            "commodity_encoder": commodity_encoder,
            "pair_scalers": {
                (b["market"], b["commodity"]): b["scaler"] for b in batch["pair_bundles"]
            },
            "features": FEATURES,
            "lookback": LOOKBACK,
        }, f)

    selection_results.to_parquet("model/selection_results.parquet", index=False)

    print("Saved model/embedding_model.keras")
    print("Saved model/encoders_and_scalers.pkl")
    print("Saved model/selection_results.parquet")
    print(f"Train run finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
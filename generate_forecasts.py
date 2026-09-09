"""
generate_forecasts.py

Third pipeline script for the Kenya Food Price Early Warning System.
Loads the master price table and the trained router's selection results,
rebuilds the anomaly detection layer (not persisted elsewhere), and writes
forecasts.csv and price_history.csv in the exact shape app.py expects.

Run after data_prep.py and train_model.py have produced:
    data/master.parquet
    data/shortlist.parquet
    model/selection_results.parquet
    model/encoders_and_scalers.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MIN_HISTORY_FOR_THRESHOLD = 6
RESIDUAL_STD_MULTIPLIER = 2


def load_inputs(data_dir: Path, model_dir: Path):
    master = pd.read_parquet(data_dir / "master.parquet")
    shortlist = pd.read_parquet(data_dir / "shortlist.parquet")
    selection_results = pd.read_parquet(model_dir / "selection_results.parquet")

    with open(model_dir / "encoders_and_scalers.pkl", "rb") as f:
        encoders_and_scalers = pickle.load(f)

    return master, shortlist, selection_results, encoders_and_scalers


def run_anomaly_detection(master: pd.DataFrame) -> pd.DataFrame:
    master = master.sort_values("date").copy()

    master["expected_price"] = master.groupby(["market", "commodity"])["price_per_kg"].shift(1)
    master["residual"] = master["price_per_kg"] - master["expected_price"]

    # expanding std on prior residuals only, shifted so the current row is never included
    master["residual_std_prior"] = master.groupby(["market", "commodity"])["residual"].transform(
        lambda s: s.expanding(min_periods=MIN_HISTORY_FOR_THRESHOLD).std().shift(1)
    )

    master["flagged"] = master["residual_std_prior"].notna() & (
        master["residual"].abs() > RESIDUAL_STD_MULTIPLIER * master["residual_std_prior"]
    )

    return master


def sanity_check_pairs(shortlist: pd.DataFrame, encoders_and_scalers: dict):
    n_markets_expected = shortlist["market"].nunique()
    n_commodities_expected = shortlist["commodity"].nunique()

    n_markets_fit = len(encoders_and_scalers["market_encoder"].classes_)
    n_commodities_fit = len(encoders_and_scalers["commodity_encoder"].classes_)

    if n_markets_fit != n_markets_expected or n_commodities_fit != n_commodities_expected:
        print(
            f"Warning, encoder entity counts ({n_markets_fit} markets, "
            f"{n_commodities_fit} commodities) do not match the current shortlist "
            f"({n_markets_expected} markets, {n_commodities_expected} commodities). "
            "Master or shortlist may have changed since train_model.py last ran."
        )


def build_forecasts_csv(selection_results: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    latest_flag_per_pair = (
        master.sort_values("date")
        .groupby(["market", "commodity"])["flagged"]
        .last()
        .reset_index()
        .rename(columns={"flagged": "latest_flagged"})
    )

    dashboard_data = selection_results.merge(
        latest_flag_per_pair, on=["market", "commodity"], how="left"
    )
    dashboard_data = dashboard_data.rename(columns={"blended_forecast": "display_forecast"})
    dashboard_data = dashboard_data[[
        "market", "commodity", "chosen_forecast", "naive_forecast",
        "display_forecast", "latest_flagged",
    ]]

    return dashboard_data


def build_price_history_csv(master: pd.DataFrame) -> pd.DataFrame:
    return master[["market", "commodity", "date", "price_per_kg", "expected_price", "flagged"]].copy()


def main():
    parser = argparse.ArgumentParser(description="Generate forecasts.csv and price_history.csv for the dashboard.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    master, shortlist, selection_results, encoders_and_scalers = load_inputs(args.data_dir, args.model_dir)

    sanity_check_pairs(shortlist, encoders_and_scalers)

    master = run_anomaly_detection(master)

    eligible_rows = master["residual_std_prior"].notna().sum()
    flag_rate = master["flagged"].sum() / eligible_rows * 100
    print(f"Anomaly detection, eligible rows: {eligible_rows}, flagged: {master['flagged'].sum()}, rate: {flag_rate:.1f}%")

    forecasts = build_forecasts_csv(selection_results, master)
    forecasts.to_csv(args.output_dir / "forecasts.csv", index=False)
    print(f"Saved {len(forecasts)} rows to forecasts.csv")

    price_history = build_price_history_csv(master)
    price_history.to_csv(args.output_dir / "price_history.csv", index=False)
    print(f"Saved {len(price_history)} rows to price_history.csv")


if __name__ == "__main__":
    main()
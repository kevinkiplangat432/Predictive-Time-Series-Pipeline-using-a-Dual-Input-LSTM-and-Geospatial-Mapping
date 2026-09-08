<!--markdownlint-disable-->
# Technical Report: Kenya Food Price Early Warning System

## 1. Project Overview

A Moringa School Data Science capstone (DSF-FT16), forecasting Kenyan staple food prices and building an early-warning system for unusual price movement. The original objective, forecasting prices two to three months ahead and beating naive persistence, was tested with three separate model families and none succeeded at the individual market-commodity level. This is documented throughout as a real finding about price persistence, not an unresolved failure, and the project's actual delivered value is a validated anomaly detection layer plus an honestly-scoped forecasting pipeline.

## 2. Data Sources

- **WFP Kenya Food Prices** (via HDX): retail and wholesale price observations by market and commodity, January 2006 through August 2026 in the extract used.
- **NASA POWER API**: monthly rainfall and temperature by market coordinates, no API key required.

## 3. Data Preparation

**Scope filtering.** Three fuel commodities (diesel, kerosene, petrol-gasoline) excluded as non-food. Twelve commodities priced in non-weight units excluded: four milk varieties, two vegetable oil varieties, bananas, and unit-incompatible kale and cabbage rows. All prices standardized to price-per-kilogram.

**Completeness floor.** Each market-commodity pair's completeness is measured against its own active reporting window (from first to last report), not the full dataset span, since most series only began consistent reporting in late 2023. Pairs below 60% completeness against their own window are excluded regardless of history length. A `MIN_YEARS_ACTIVE = 1.0` floor removes any pair with under one year of active history, this alone removed 542 pairs that would otherwise have been shortlisted with near-empty history.

**Resulting shortlist:** 133 market-commodity pairs, 26 markets, 14 commodities. Split by history depth: 100 pairs with 3+ years (Prophet track), 33 pairs with 1-3 years (LSTM track).

**Market composition, verified directly, not assumed:** 16 of the 26 markets are Kakuma/Dadaab refugee camp submarkets or Nairobi informal settlements. The remaining 10, Garissa, Marsabit, Mandera, Lodwar (Turkana), Hola (Tana River), Marigat (Baringo), Kajiado, Kilifi, Kitui, and Nairobi proper, are general Kenyan county markets concentrated in arid and semi-arid regions, the classic NDMA drought early-warning geography. This is a real, useful dataset, not a refugee-camp-only artifact, though the specific shock backtest below skews toward the first group due to which series have long enough history to test against a 2022 event.

**Weather join.** NASA POWER daily rainfall and temperature aggregated to monthly, joined to price data by market and month. Master table: 6,077 rows, 98.98% weather match rate (one market, Hola/Tana River, fails retrieval with an API 422 and is documented as a known gap, not silently dropped).

**Known documentation bug (unresolved as of this report):** Data Preparation contains a commented-out global chronological split (`cutoff_val`/`cutoff_test`/`train`/`val`/`test`), but an uncommented print statement immediately after it still executes, displaying contaminated leftover variable values from elsewhere in the notebook (nonsensical output: train data extending to 2026, test data starting in 2020). Section 3.16's markdown claims a chronological split happens here, this is not accurate as the notebook currently stands, the real, authoritative split is per-pair, computed inside `build_pair_sequences` during Modelling, not here.

## 4. Hypotheses (from Business Understanding, revisited honestly)

Of six stated hypotheses, two received a dedicated statistical test:

- **H1 (seasonality)**: mild, descriptive support only, not a formal decomposition.
- **H2 (rainfall lag)**: a preliminary one-year, Nairobi-only test suggested a 4-month lag relationship. The full-scale test, weather matched by market and date across the whole dataset, found essentially no relationship, rainfall correlation of -0.015 to -0.020 at 3 and 4 month lags. Temperature showed a weak 0.109-0.110. **The preliminary result did not hold up and H2 is not supported.**
- **H3 (market differences)**: addressed structurally by the entity-embedding architecture, not tested statistically.
- **H4 (maize-bean substitution)**: examined only via a price distribution plot, not correlation-tested, no conclusion drawn.
- **H5 (drought volatility)**: not directly tested.
- **H6 (wholesale-to-retail lag)**: not tested, wholesale analysis was explicitly deferred, retail chosen as the sole modelling series since it's the price point most named stakeholders transact on.

## 5. Modelling

**Prophet**: baseline for the 100 long-history pairs.

**LSTM, two variants compared**: an independent model per pair, versus a pooled model sharing market and commodity embeddings across all 33 short-history pairs, matched for capacity. Pooling roughly halves error (MAE ~6.3-6.5 pooled vs ~14.5-15 independent, exact figures vary slightly run to run since LSTM training was not seeded until a later stage of the project, see Section 8).

**Architecture (pooled model)**: sequence input (6-month lookback, 5 features: `price_diff`, `rainfall_lag_3`, `rainfall_lag_4`, `temperature_lag_3`, `temperature_lag_4`), market and commodity embeddings (dimension `min(8, (n+1)//2)` each), concatenated with a 16-unit LSTM's output (recurrent dropout 0.2, L2 kernel regularization 0.01), followed by a 16-unit dense layer and dropout, both regularized. Target is scaled month-over-month price change, not price level, differencing was necessary since trending price levels caused catastrophic-looking val loss that was actually a scaling artifact.

**Per-pair train/validation/test split**: chronological, per pair, at that pair's own 80th and 90th percentile date. Median validation window per pair is 4 rows (min 1, max 17), a genuine, structural data availability limit, not a modelling shortfall.

**The forecast selection router, finalized as a weighted blend.** An earlier hard-switch design (each pair fully routed to either model or naive based on a validation comparison) was tested, tuned across two thresholds, and found unreliable, pairs that won on validation consistently lost on test. This was diagnosed as a real property of the data (thin validation splits don't predict test performance at this volume), not a bug, and the hard-switch design was retired.

The current router instead computes a continuous weight per pair:
confidence = min(val_rows / 8, 1.0)
improvement = (val_naive_mae - val_model_mae) / val_naive_mae
weight = max(0, min(improvement, 1.0)) * confidence

grounded explicitly in forecast combination literature (Bates and Granger, 1969; repeatedly confirmed across the M-competitions), which finds that blending forecasts tends to outperform confidently selecting one, especially when the evidence for selection is thin. The `8`-row confidence ceiling was empirically calibrated against the real distribution of validation rows per pair, not guessed. Result, stable across two different calibrations and multiple seeded runs: no pair ever reaches 25% model weight, none cross the 50% threshold that would mark a pair as genuinely model-leaning. Aggregate "beats naive" sits around 82-85%, lower than the retired hard-switch's ~97%, because the blend allows small model influence into more pairs (versus a hard switch touching very few), and that small influence still tends to slightly underperform naive on test. This is treated as a doubly-confirmed finding: at this data volume, no per-pair selection or blending scheme currently produces a forecast that reliably improves on naive persistence.

## 6. The Validation Loss Plateau, Investigated

Training loss converges to ~0.04-0.06 while pooled validation loss plateaus flat around 0.49-0.5 and never closes the gap. Investigated directly for data leakage: scaler fit boundaries, feature construction, and sequence construction were all checked and found correct, no row sees its own target, no scaler is fit on validation or test data. The flat-and-elevated shape of the curve is itself evidence against leakage, which typically produces suspiciously good, not bad, validation performance.

The actual mechanism: `y_val_all` (scaled validation targets) ranges from -9.00 to 10.77 against a scaler calibrated to `[0, 1]` on training data, with 5.3% of validation targets falling outside that range, some by 9-11x. Under pooled MSE loss, a small number of unusually volatile pair-months dominates the aggregate validation loss regardless of overall model quality on the remaining ~95% of validation rows.

A test of heavier regularization (embedding dimensions halved, L2 penalty increased 5x, LSTM units halved, added dropout and batch normalization throughout) produced a smoother, slower-converging curve but did not move the plateau (0.49 before, ~0.5 after). This rules out excess model complexity as the cause and further supports the outlier-domination explanation. A per-pair validation loss breakdown was proposed as the next diagnostic step, to confirm whether a small number of pairs account for most of the pooled loss, not yet executed as of this report.

This gap is not treated as a defect to eliminate before deployment. It is consistent with, and arguably a visible symptom of, the same underlying finding as Section 5's router result, this data does not support confident per-pair model trust, and the pipeline (naive-default anomaly detection, low-weight router) is already built around that reality rather than in spite of it.

## 7. Anomaly Detection

Expected price defaults to naive (previous month's observed price) for every pair, consistent with the router finding that no pair earns meaningful model weight. A pair is flagged when its residual exceeds 2 standard deviations of its own expanding historical residual distribution, computed using only prior data.

**Baseline flag rate**: 9.7% of all eligible pair-months.

**Backtest against three real, independently documented shocks** (2022 Horn of Africa drought, Ukraine-linked grain and fertilizer shock, 2022-2023 fuel subsidy removal), checked across all 270 shortlisted pairs with any real data coverage in one of these windows, not a single cherry-picked pair: flag rate rises to 27.6% inside shock windows, roughly 3x the baseline, with the drought window at 28.3% and the fuel subsidy window at 29.1%. The Ukraine window shows 13.8% but on only one month of coverage per pair, too thin to draw a conclusion from. 116 of the 270 covered pairs saw no flag in their window, this is not a claim of high recall on every shock, but the elevated aggregate rate across 270 pairs is a real, independently checked signal.

**Coverage limitation**: the pairs with long enough history to test against a specific 2022 shock skew toward refugee camp and informal settlement markets (16 of 26 shortlisted markets). The remaining 10 general county markets mostly fall in the shorter-history LSTM track and were not part of this specific backtest, though they exist in the shortlist and are covered by the anomaly detector going forward.

## 8. Reproducibility

LSTM training was not seeded for most of the project's development. A fixed seed (`np.random.seed(42)`, `tf.random.set_seed(42)`) was added once the pipeline was formalized into standalone scripts. Even with a fixed seed, minor run-to-run variation remains possible due to non-deterministic CPU-level operations in TensorFlow, exact figures throughout this report (MAE, exact weight distributions, exact "beats naive" percentage) should be read as stable ranges (e.g. "82-85%"), not fixed constants.

## 9. Pipeline and Deployment Status

- `data_prep.py`: complete, verified against real execution, reproduces all Section 3 numbers exactly.
- `train_model.py`: complete, verified, seeded, reproduces Section 5 and 6 findings.
- `generate_forecasts.py`: not yet built, this is the current in-progress task.
- `app.py`: built, includes a live alerts panel and per-pair price history chart, not yet re-verified against `generate_forecasts.py`'s specific output.
- Deployment (Streamlit Community Cloud): not started, blocked on the above.

## 10. Environment

Python 3.12, Anaconda environment `gen-lab-env`. Key libraries: pandas 3.0 (note: `.groupby().apply()` drops grouping columns by default in this version, use `.transform()` or `include_groups=False`), numpy, scikit-learn, tensorflow/keras, prophet, matplotlib, streamlit. `requirements.txt` intentionally not yet pinned, deferred until all pipeline scripts exist.
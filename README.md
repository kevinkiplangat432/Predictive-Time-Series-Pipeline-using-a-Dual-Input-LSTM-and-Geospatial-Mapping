<!--markdownlint-disable-->
# Kenya Food Price Early Warning System

> **Status: notebook, data prep, model training, and forecast generation all complete and verified. Dashboard built and confirmed running locally. Deployment not yet live.**

## Overview

A system that forecasts Kenyan staple food prices and flags markets moving outside their own normal range, built on public WFP price data and NASA POWER weather data. Intended users are farmers timing sales, traders managing inventory, and NGOs and county offices watching for early signs of food-price stress.

## CRISP-DM Mapping

This project follows CRISP-DM, with two deliberate departures from the standard six phases, noted below. `kenyan_food_prices.ipynb` is organized section-by-section to match.

| CRISP-DM Phase | Notebook Section(s) | Covers |
|---|---|---|
| Business Understanding | 1 | Objectives, stakeholders, success criteria, hypotheses |
| Data Understanding | 2 | Data sources, structure, initial quality assessment (on raw, uncleaned data) |
| Data Preparation | 3 | Cleaning, unit standardization, scope and retail selection, historical coverage, outlier handling, weather integration |
| *(project-specific)* Exploratory Data Analysis | 4 | Deferred until after Section 3, not nested under Data Understanding as canonical CRISP-DM would place it — raw units and trade-level scope aren't yet comparable at the point Data Understanding runs |
| *(project-specific)* Feature Engineering | 5 | Broken out as its own section rather than folded into Data Preparation, since it involves the target transform and lag validation, not just data cleaning |
| Modelling | 6 | Naive baseline, Prophet (trained per pair across all 100 long-history pairs), pooled LSTM, validation-weighted router unifying both tracks |
| *(project-specific)* Anomaly Detection | 7 | Not a CRISP-DM phase — the early-warning component of the system, built on Modelling's output |
| Evaluation | 8 | Section 1's criteria checked against actual results, not restated as new findings |
| Deployment | 9 | Pipeline scripts, dashboard, current deployment status |
| *(project-specific)* Findings / Conclusion | 10, 11 | Cross-section synthesis and honest assessment against the original objectives |

## Architecture
WFP Kenya Food Prices (HDX) ─┐
NASA POWER API (weather) ─┴─→ data_prep.py ─→ cleaned, feature-engineered master table
│

train_model.py
(naive baseline, per-pair Prophet x100,
pooled LSTM, unified validation-weighted router)
│

generate_forecasts.py
(rebuilds anomaly detection on current data,
writes forecasts.csv + price_history.csv)
│

app.py (Streamlit dashboard)
Overview, Price Forecast, Early Warnings,
Markets Monitor, Decision Support, Reports, About


`train_model.py` is deliberately separate from `generate_forecasts.py` — training is expensive and only needs to rerun when there's meaningfully new data to learn from; regenerating forecasts and anomaly flags against already-trained models is cheap and can run far more often.

| Component | Status |
|---|---|
| `data_prep.py` | Complete, run successfully |
| `train_model.py` | Complete, run successfully — seeded (`SEED = 42`), 99 of 100 Prophet-track pairs trained individually, all 33 LSTM-track pairs pooled |
| `generate_forecasts.py` | Complete, run successfully — produces a genuine forward (next-month) forecast per pair |
| `app.py` | Complete, confirmed running locally via `streamlit run app.py` |

## Data Sources

- **WFP Kenya Food Prices** (via HDX) — retail and wholesale price observations by market and commodity. Covers January 2006 through the current extraction date.
- **NASA POWER API** — monthly rainfall and temperature by market location, joined to price data on market coordinates and date. No API key required.

## Methodology Summary

**Scope decisions.** Fuel commodities excluded as non-food. Milk, vegetable oil, bananas, and unit-incompatible kale and cabbage rows excluded because they're priced by volume or count, not weight — this system works in price-per-kilogram terms throughout. A minimum history floor of one year of active reporting was applied before any pair was shortlisted.

**Resulting shortlist:** 133 market-commodity pairs across 26 markets and 14 commodities — 100 pairs with 3+ years of history (Prophet track), 33 pairs with 1 to 3 years (LSTM track). Master table: 6,077 rows, 98.98% matched to weather data.

**Forecasting.** Every prophet-track pair now gets its own individually trained Prophet model (99 of 100 trained; one pair lost all rows to a single market's weather-retrieval gap), using the same four weather-lag regressors as every other model in this project. Every lstm-track pair goes through a single pooled LSTM, trained once across all 33 pairs with no market or commodity identity input — an entity-embedding version was tried and dropped after adding no measurable improvement. Each pair's final forecast is a weighted blend of its own model and naive persistence, following Bates and Granger's (1969) forecast-combination result — weight scales with the strength and volume of that pair's own validation evidence. An earlier hard-switch version is archived as `Kenyan_food_prices_hardswitch_archive.ipynb`.

In the current run: **118 of 132 trained pairs land at zero model weight (pure naive), 14 receive a partial blend, and 0 cross the 0.5 threshold** that would mark a pair as fully model-leaning — the highest single-pair weight is 0.477.

**Anomaly detection.** Expected price for every pair defaults to naive persistence, consistent with the router finding above. A pair is flagged when its residual exceeds 2 standard deviations of its own expanding historical residual distribution, computed using only prior data.

## Key Findings

- **Pooling improves forecasting substantially on short-history series.** The pooled LSTM achieves MAE 6.03 on the 33 lstm-track pairs, against MAE 15.43 for an independently trained model per pair on those same pairs — roughly a 60% reduction in error from pooling alone.
- **The router individually beats naive on most pairs, but not yet in aggregate.** Now that Prophet is genuinely trained per pair (not scored through the pooled LSTM, as an earlier version of this project did), **93.9% of the 132 trained pairs individually beat or tie naive** on the held-out test set — a real jump from an earlier 83.3% figure. But the **aggregate mean blended test MAE (6.57) is marginally worse than the aggregate mean naive test MAE (6.55)**. Both are genuine and not contradictory: 118 of 132 pairs are exact ties by construction (zero weight), so the win rate is dominated by ties, and the 14 blended pairs — including the highest-weighted pair in the shortlist, at 0.477 — don't collectively pay off on the untouched test set even where validation suggested they should. This is reported as the honest current state of the router.
- **The anomaly detector shows a real signal.** Across 851 pair-months of coverage inside three independently documented Kenyan price shocks (2022 Horn of Africa drought, Ukraine-linked grain and fertilizer shock, 2022–2023 fuel subsidy removal), the flag rate rises to 27.6% versus a 9.7% baseline — roughly 3x elevation. 154 of 270 shock-pair instances were flagged at least once; 116 saw no flag at all — not perfect recall, but a real, independently checked signal.
- **Scope limitation on the anomaly result.** Nearly all pairs with usable coverage this far back are refugee-camp markets (Kakuma, Kalobeyei, Dadaab) and informal Nairobi settlements. This speaks most directly to humanitarian and NGO use, not general smallholder farmer markets, which mostly lack sufficient history to have been tested here.

## Deployment

**Current state:** the full pipeline (`data_prep.py` → `train_model.py` → `generate_forecasts.py` → `app.py`) runs end-to-end locally without errors, seeded and reproducible. Not yet deployed to a public URL.

**Target state:**
python data_prep.py && python train_model.py && python generate_forecasts.py && streamlit run app.py


**Remaining steps, in order:**
1. Pin `requirements.txt` to exact installed versions, now that all four scripts exist.
2. Commit pre-generated `forecasts.csv`, `price_history.csv`, and the `model/` directory alongside the code, so the dashboard has data on first load without requiring a live pipeline run.
3. Deploy to Streamlit Community Cloud.
4. Confirm it loads correctly on both phone and laptop before calling it done.

## Environment

- Python 3.12, Anaconda environment (`gen-lab-env`).
- pandas 3.0 — `.groupby(...).apply()` no longer returns grouping columns by default; use `.transform()` or pass `include_groups=False` where relevant. Caused real bugs during development; worth knowing before modifying Data Preparation or Feature Engineering.
- Key libraries: pandas, numpy, scikit-learn, tensorflow/keras, prophet, matplotlib, streamlit, plotly, joblib, pyarrow.

## Known Limitations

- The router's aggregate result is mixed: strong per-pair win rate (93.9%) alongside a marginally worse aggregate MAE than naive (6.57 vs 6.55). The 14 blended pairs are where this gap lives — not yet individually diagnosed as of this writing.
- Two of the six hypotheses in Business Understanding (H1 seasonality, H2 rainfall lag) received a dedicated statistical test. H2's preliminary result did not hold up at full scale. H3–H6 were addressed only structurally or explicitly deferred, and are not reported as confirmed or refuted.
- Wholesale price forecasting was scoped out in favor of retail, the price point most named stakeholders actually transact on. Wholesale rows remain intact in the source data — a defined next phase, not a data gap.
- The anomaly backtest's strongest coverage skews toward refugee-camp and informal-settlement markets (16 of 26 shortlisted markets). The remaining 10 — general Kenyan county markets in arid and semi-arid regions — mostly fall in the shorter-history LSTM track and weren't part of the 2022 shock backtest. This reflects which series WFP has monitored most consistently, not a deliberate scoping choice.
- One shortlisted market, Hola (Tana River), has no usable weather match — it lacks coordinates in the source WFP data, so it's excluded from the weather join entirely rather than imputed.
- Confidence ranges on the dashboard's forecast chart are shown only for Prophet-track pairs, since Prophet produces them natively. The pooled LSTM has no calibrated uncertainty estimate, so its forecasts show as a point estimate only.
- The dashboard's "Warning Level" indicator (HIGH/MODERATE/LOW/NORMAL) is rule-based, derived from forecasted change magnitude — unlike the anomaly flag, it has not been backtested against real historical shocks.

## Acknowledgments

Price data from the World Food Programme via the Humanitarian Data Exchange. Weather data from NASA's POWER Project. Built as a capstone for Moringa School's Data Science program (DSF-FT16).
<!--markdownlint-disable-->
# Kenya Food Price Early Warning System

> **Status: modelling and anomaly detection validated, seeded for reproducibility. Dashboard built, deployment in progress.**

## Overview

A system that forecasts Kenyan staple food prices and flags markets moving outside their own normal range, built on public WFP price data and NASA POWER weather data. Intended users are farmers timing sales, traders managing inventory, and NGOs and county offices watching for early signs of food-price stress.

## CRISP-DM Mapping

This project follows CRISP-DM, with two deliberate departures from the standard six phases, noted below. `Kenyan_food_prices.ipynb` is organized section-by-section to match.

| CRISP-DM Phase | Notebook Section(s) | Covers |
|---|---|---|
| Business Understanding | 1 | Objectives, stakeholders, success criteria, hypotheses |
| Data Understanding | 2 | Data sources, structure, initial quality assessment (on raw, uncleaned data) |
| Data Preparation | 3 | Cleaning, unit standardization, scope and retail selection, historical coverage, outlier handling, weather integration |
| *(project-specific)* Exploratory Data Analysis | 4 | Deferred until after Section 3, not nested under Data Understanding as canonical CRISP-DM would place it — raw units and trade-level scope aren't yet comparable at the point Data Understanding runs |
| *(project-specific)* Feature Engineering | 5 | Broken out as its own section rather than folded into Data Preparation, since it involves the target transform and lag validation, not just data cleaning |
| Modelling | 6 | Naive baseline, Prophet, LSTM (per-pair and pooled), validation-weighted model selection |
| *(project-specific)* Anomaly Detection | 7 | Not a CRISP-DM phase — the early-warning component of the system, built on Modelling's output |
| Evaluation | 8 | Section 1's criteria checked against actual results, not restated as new findings |
| Deployment | 9 | Pipeline scripts, dashboard, current deployment status |
| *(project-specific)* Findings / Conclusion | 10, 11 | Cross-section synthesis and honest assessment against the original objectives |

## Architecture

```
WFP Kenya Food Prices (HDX)  ─┐
NASA POWER API (weather)     ─┴─→  data_prep.py  ─→  cleaned, feature-engineered master table
                                                            │
                                                            ▼
                                                     train_model.py
                                              (naive baseline, Prophet, pooled LSTM,
                                               per-pair LSTM, validation-weighted router)
                                                            │
                                                            ▼
                                                  generate_forecasts.py
                                        (rebuilds anomaly detection on current data,
                                         writes forecasts.csv + price_history.csv)
                                                            │
                                                            ▼
                                                 app.py (Streamlit dashboard)
                                          map, filterable table, per-pair detail view
```

`train_model.py` is deliberately separate from `generate_forecasts.py` — training is expensive and only needs to rerun when there's meaningfully new data to learn from; regenerating forecasts and anomaly flags against already-trained models is cheap and can run far more often.

| Component | Status |
|---|---|
| `data_prep.py` | Complete, verified against real execution |
| `train_model.py` | Complete, seeded (`SEED = 42`) for reproducible runs |
| `generate_forecasts.py` | Not yet written — the next concrete step |
| `app.py` | Written, reads `forecasts.csv` / `price_history.csv` — not yet tested against real pipeline output, since `generate_forecasts.py` doesn't exist yet |

## Data Sources

- **WFP Kenya Food Prices** (via HDX) — retail and wholesale price observations by market and commodity. Covers January 2006 through August 2026 in the current extract.
- **NASA POWER API** — monthly rainfall and temperature by market location, joined to price data on market coordinates and date. No API key required.

## Methodology Summary

**Scope decisions.** Fuel commodities excluded as non-food. Milk, vegetable oil, bananas, and unit-incompatible kale and cabbage rows excluded because they're priced by volume or count, not weight — this system works in price-per-kilogram terms throughout. A minimum history floor of one year of active reporting was applied before any pair was shortlisted.

**Resulting shortlist:** 133 market-commodity pairs across 26 markets and 14 commodities — 100 pairs with 3+ years of history (Prophet track), 33 pairs with 1 to 3 years (LSTM track). Master table: 6,077 rows, 98.98% matched to weather data.

**Forecasting.** Prophet for longer-history pairs. For shorter-history pairs, a single pooled LSTM, trained once across all pairs with no market or commodity identity input, compared against an independent per-pair LSTM of matched capacity. Each pair's final forecast is a weighted blend of the pooled model and naive persistence — weight scales with both the strength and volume of that pair's validation evidence, following Bates and Granger's (1969) forecast-combination result, rather than a hard switch between the two. An earlier hard-switch version is archived as `Kenyan_food_prices_hardswitch_archive.ipynb`. In the current seeded run, no pair reached even 25% model weight (17.5% was the highest); 108 of 132 pairs land at pure naive, the remaining 24 get a partial blend.

**Anomaly detection.** Expected price for every pair defaults to naive persistence, consistent with no pair earning meaningful model weight. A pair is flagged when its residual exceeds 2 standard deviations of its own expanding historical residual distribution, computed using only prior data.

## Key Findings

- **Pooling improves forecasting.** The pooled model achieves MAE 6.03 on the short-history (lstm-track) pairs, against MAE 15.43 for an independently trained model per pair on those same pairs — roughly a 60% reduction in error from pooling alone. An entity-embedding version of the pooled model was tried and dropped after adding no measurable improvement over this simpler version.
- **No model reliably beats naive at the individual pair level.** The blended router beats or ties naive on 83.3% of pairs, but that figure is driven mostly by pairs routed to pure naive tying it exactly, not by the model outperforming it. Three model families — Prophet, per-pair LSTM, pooled LSTM — have all failed to reliably beat naive at the individual pair level. Treated as a real property of the data (Kenyan staple prices are highly persistent month to month), not a modelling shortfall.
- **The anomaly detector shows a real signal.** Across 851 pair-months of coverage inside three independently documented Kenyan price shocks (2022 Horn of Africa drought, Ukraine-linked grain and fertilizer shock, 2022–2023 fuel subsidy removal), the flag rate rises to 27.6% versus a 9.7% baseline — roughly 3x elevation. Not perfect recall — 116 of 270 covered shock-pair instances saw no flag — but a real, independently checked signal.
- **Scope limitation on the anomaly result.** Nearly all pairs with usable coverage this far back are refugee-camp markets (Kakuma, Kalobeyei, Dadaab) and informal Nairobi settlements. This speaks most directly to humanitarian and NGO use, not general smallholder farmer markets, which mostly lack sufficient history to have been tested here.

## Deployment

**Current state:** notebook-only. Open `Kenyan_food_prices.ipynb`, restart the kernel, and run all cells top to bottom in one continuous pass — later sections depend on state built earlier in the same run. Results are seeded and reproducible; the numbers above should match exactly on a fresh `Run All`.

**Target state**, once `generate_forecasts.py` is written:
```
python data_prep.py && python train_model.py && python generate_forecasts.py && streamlit run app.py
```

**Remaining steps, in order:**
1. Write `generate_forecasts.py`.
2. Pin `requirements.txt` to exact installed versions — deferred until all pipeline scripts exist, so it reflects what the full pipeline actually ran against.
3. Commit pre-generated `forecasts.csv` and `price_history.csv` alongside the code, so the dashboard has data on first load.
4. Deploy to Streamlit Community Cloud; confirm it loads on both phone and laptop.

## Environment

- Python 3.12, Anaconda environment.
- pandas 3.0 — `.groupby(...).apply()` no longer returns grouping columns by default; use `.transform()` or pass `include_groups=False` where relevant. Caused real bugs during development; worth knowing before modifying Data Preparation or Feature Engineering.
- Key libraries: pandas, numpy, scikit-learn, tensorflow/keras, prophet, matplotlib, streamlit, plotly.

## Known Limitations

- Half of the 132 modelled pairs have 4 or fewer validation rows — the minimum this project treats as usable for any model weight at all. A structural data availability limit, not something a different model or router design can work around.
- Two of the six hypotheses in Business Understanding (H1 seasonality, H2 rainfall lag) received a dedicated statistical test. H2's preliminary result did not hold up at full scale. H3–H6 were addressed only structurally or explicitly deferred, and are not reported as confirmed or refuted.
- Wholesale price forecasting was scoped out in favor of retail, the price point most named stakeholders actually transact on. Wholesale rows remain intact in the source data — a defined next phase, not a data gap.
- The anomaly backtest's strongest coverage skews toward refugee-camp and informal-settlement markets (16 of 26 shortlisted markets). The remaining 10 — general Kenyan county markets in arid and semi-arid regions — mostly fall in the shorter-history LSTM track and weren't part of the 2022 shock backtest. This reflects which series WFP has monitored most consistently, not a deliberate scoping choice.
- One shortlisted market, Hola (Tana River), has no usable weather match — it lacks coordinates in the source WFP data, so it's excluded from the weather join entirely rather than imputed.

## Acknowledgments

Price data from the World Food Programme via the Humanitarian Data Exchange. Weather data from NASA's POWER Project. Built as a capstone for Moringa School's Data Science program (DSF-FT16).

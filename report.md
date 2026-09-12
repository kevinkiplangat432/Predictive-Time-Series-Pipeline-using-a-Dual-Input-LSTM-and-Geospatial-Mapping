<!--markdownlint-disable-->
# Kenya Food Price Early Warning System
## CRISP-DM Data Report

**Course:** DSF-FT16, Moringa School Data Science Capstone
**Author:** Kevin

---

## 1. Business / Problem Understanding

### 1.1 Background and Problem Statement

Food prices in Kenya move sharply and unevenly across markets, driven by harvest cycles, rainfall, and local supply conditions. Farmers, traders, and food security institutions largely act on current or historical prices, not forecasts — a gap that shows up as poor sell/buy timing for farmers, avoidable inventory risk for traders, and institutional responses that arrive only after a shortage or price spike is already visible.

The core problem is not a lack of data — WFP already publishes Kenyan market price data, and NASA POWER publishes weather data, both freely. What is missing is a system that combines them into a forward-looking signal at the market level.

### 1.2 Objectives

1. Forecast staple food prices one month ahead, per market and commodity.
2. Detect when a market's price begins to diverge meaningfully from its own expected pattern (early warning).
3. Deliver both through an accessible dashboard, not a static analysis.
4. Build a reproducible, seeded pipeline — not a one-off notebook run.

### 1.3 Stakeholders

Smallholder farmers and cooperatives (sell-timing decisions), traders and market intermediaries (inventory timing), county agricultural offices and NDMA (early localized food-stress signals), NGOs and humanitarian organizations (procurement and cash-based intervention planning), the National Cereals and Produce Board (reserve and stabilization decisions), and urban consumers and food processors (indirect beneficiaries of institutional planning).

### 1.4 Success Criteria

**Business criteria:** forecasts and alerts should surface information stakeholders don't already have from watching current prices; the anomaly detector should flag genuine shocks without excessive false alarms; the system should run on free, public data; the pipeline should be refreshable without significant manual rework.

**Data mining criteria:** a model earns its place only by beating a naive persistence baseline (previous month's price), not an arbitrary fixed threshold; the anomaly detector should show a meaningfully elevated flag rate during documented real shocks versus its own baseline rate; the pipeline must be reproducible (seeded) and the price-weather join should have minimal data loss.

### 1.5 Hypotheses

- **H1:** Maize prices follow a seasonal pattern tied to harvest periods.
- **H2:** Rainfall has a measurable lagged relationship with future commodity prices.
- **H3:** Price patterns differ meaningfully across markets.
- **H4:** Maize and bean prices move together as substitute staples.
- **H5:** Price volatility increases during drought periods.
- **H6:** Wholesale price changes are reflected in retail prices within one to two weeks.

---

## 2. Data Understanding

### 2.1 Data Sources

| Dataset | Source | Purpose |
|---|---|---|
| WFP Kenya Food Prices | HDX (Humanitarian Data Exchange) | Historical commodity prices — target variable |
| NASA POWER | Public API | Daily rainfall and temperature — explanatory variables |

Both sources are free and public. The raw WFP extract contains 27,763 rows and 16 columns, comfortably exceeding the capstone's minimum thresholds of 10,000 rows and 10 features, and is sourced from an international organization's open data portal rather than a Kaggle-style repository.

### 2.2 Dataset Description

The WFP dataset is one row per commodity, market, and date — a specific reported price at retail or wholesale level. It spans January 2006 to the current extraction date, covering 226 markets and 51 distinct commodity labels across 7 administrative regions. NASA POWER is one row per day for a given coordinate; the two are joined via each market's latitude/longitude, already present in the WFP data.

### 2.3 Initial Data Quality Findings

- Coordinates are complete for 225 of 226 markets — the exception, Hola (Tana River), is excluded from the weather join rather than imputed.
- No duplicate rows at the (date, market, commodity, pricetype) grain.
- Maize alone appears under five separate labels (`Maize`, `Maize (white)`, `Maize flour`, `Maize (white, dry)`, `Maize flour (white)`) — a real cleanup consideration addressed by scope decisions in Data Preparation, not by merging labels.
- 13 distinct price units are in use, and both retail and wholesale trade levels are mixed — both resolved explicitly in Data Preparation.

---

## 3. Data Preparation

### 3.1 Cleaning and Standardization

Every unit was mapped to a kilogram-equivalent multiplier, producing a standardized `price_per_kg` field. Units with no weight equivalent (litres, millilitres, count-based units) were identified and excluded on measurement grounds, alongside three fuel commodities excluded as out of scope — 12 commodities in total removed for stated reasons, not data failures.

### 3.2 Scope Selection

Retail was chosen as the sole modelling series, since it is the price point most stakeholders actually transact on. Wholesale rows remain intact in the source data as a defined next phase.

### 3.3 Historical Coverage and Shortlisting

Completeness was measured against each pair's own active reporting window, not the full dataset span, since most series only began consistent reporting in late 2023. Pairs meeting a 60% fair-completeness floor were split by years of active history: **100 pairs with 3+ years routed to a Prophet track**, **33 pairs with 1–3 years routed to an LSTM track**. A one-year minimum floor excluded 542 near-empty pairs from the 2,021 raw market-commodity combinations.

**Final shortlist: 133 pairs across 26 markets and 14 commodities.**

### 3.4 Outlier Handling

Outliers were identified per commodity on the standardized `price_per_kg` field (69 of 6,077 modelling rows) and classified as transient spikes or persistent shifts. No outlier was removed — the anomaly detection layer in Section 4 is designed specifically to act on this kind of divergence.

### 3.5 Weather Integration and Feature Engineering

Daily rainfall and temperature were retrieved from NASA POWER, aggregated to monthly, and joined to price data — a **98.98% match rate** (the shortfall is entirely the single market lacking coordinates).

Two feature engineering decisions matter methodologically:
- **Target transform:** the LSTM track predicts `price_diff` (month-over-month change), not the raw price level — far more stable across many pooled pairs with different long-run trends.
- **Lag features:** rainfall and temperature were lagged 3 and 4 months, reflecting that a harvest's price effect follows the rain that produced it by months, not immediately. A preliminary single-market test suggested a strong lag relationship; the full-scale, properly matched test across every shortlisted market found the rainfall correlation was effectively zero (-0.015 to -0.020), while temperature showed a weak positive correlation (~0.11). Both were retained as candidate features, not dropped.

An entity-embedding LSTM variant (market/commodity identity as learned embeddings) was tried and dropped after checking it added no measurable improvement over a simpler pooled model with no identity input.

---

## 4. Modelling

### 4.1 Approach

Three model families were built and compared, all judged by MAE against naive persistence:

| Model | Track | Scope |
|---|---|---|
| Naive persistence | All 132 trained pairs | Baseline, no exceptions |
| Prophet, with rainfall/temperature as external regressors | 100 long-history pairs | **One individually trained model per pair — 99 of 100 trained successfully** |
| Pooled LSTM (no entity identity), predicting `price_diff` | 33 short-history pairs | One shared model trained across all 33 |

An independent per-pair LSTM was also trained as a comparison baseline for the pooling decision: it achieved MAE 15.43 against naive, beating naive on only 6.1% of its pairs. The pooled LSTM on the same pairs achieved MAE 6.03 — pooling roughly halves the error on short-history series, the strongest single modelling result in this project.

**A methodological correction made mid-project is worth documenting explicitly:** an earlier version of the batch modelling pipeline scored every pair — prophet-track pairs included — through the pooled LSTM, never actually training a real Prophet model beyond a single demonstration series. This has since been corrected: every prophet-track pair now gets its own trained Prophet model, and the numbers in this report reflect that corrected pipeline, not the earlier placeholder state.

### 4.2 Forecast Selection

Each pair's production forecast is a **weighted blend** of its own model and naive persistence, following Bates and Granger's (1969) forecast-combination result: weight scales with both the strength and the volume of that pair's own validation evidence (minimum 4 validation rows for any weight, full confidence at 8+).

**Real, verified results from the corrected pipeline:** 99 of 100 Prophet-track pairs trained successfully (one lost all rows to a weather-data gap), and all 33 LSTM-track pairs were included. Across the full 132-pair shortlist: **118 pairs land at zero model weight (pure naive), 14 receive a partial blend, and 0 pairs cross the 0.5 threshold** that would mark a pair as fully model-leaning — the highest single-pair weight is 0.477.

**93.9% of pairs individually beat or tie naive** on the held-out test set — a real increase from an earlier 83.3% figure computed before Prophet was properly batch-trained. However, the **aggregate mean blended test MAE (6.57) is marginally worse than the aggregate mean naive test MAE (6.55)**. Both figures are genuine and not in conflict: 118 of 132 pairs are exact ties by construction, so the 93.9% win rate is dominated by ties rather than clear wins, and a handful of the 14 partially-weighted pairs — likely including the pair at 0.477 weight — are wrong by enough on the untouched test set to offset the gains elsewhere in the aggregate average.

### 4.3 Anomaly Detection

Expected price for every pair defaults to naive persistence, consistent with the finding above that no pair earns strong model trust. A pair is flagged when its residual exceeds 2 standard deviations of its own expanding historical residual distribution, computed using only prior data.

---

## 5. Evaluation

### 5.1 Business Criteria Revisited

| Criterion | Verdict |
|---|---|
| Forecasts/alerts add information beyond watching current prices | Partially met — see 5.2 for the nuanced modelling result; the anomaly alerts add clearer, independently verified information |
| Anomaly detector flags shocks without excessive false alarms | **Met** |
| Runs on free, public data | **Met** |
| Pipeline refreshable without significant rework | **Met**, with one documented exception (a single market's weather gap) |

### 5.2 Data Mining Criteria Revisited

- **Beats naive, not an arbitrary threshold:** genuinely mixed, now with real evidence rather than an approximation. 93.9% of pairs individually beat or tie naive — a real improvement now that Prophet is properly trained per pair. But the aggregate mean blended MAE (6.57) is not actually better than the aggregate mean naive MAE (6.55), because a small number of highly-weighted pairs' test-set errors offset the many exact ties. The router is functioning as designed — it stays conservative and rarely commits real weight — but where it does commit weight, that weight isn't yet reliably paying off. This is reported as the current, honest state of the evidence.
- **Price-weather join, minimal loss:** met — 98.98% match rate.
- **Anomaly detector, elevated flag rate in real shocks:** met — backtested against three independently documented Kenyan price shocks (2022 Horn of Africa drought, Ukraine-linked grain and fertilizer shock, 2022–2023 fuel subsidy removal). Across 851 pair-months of coverage inside these windows, the detector flagged 235 (**27.6%**) against its own **9.7%** baseline — roughly 3x elevation. 154 of 270 shock-pair instances were flagged at least once; 116 saw no flag — not perfect recall, but a real, independently checked signal built with no knowledge of the events it was tested against.
- **Reproducibility:** met — every stochastic step is seeded (`SEED = 42`).

### 5.3 Honest Limitations

- The router's aggregate result is mixed: a strong per-pair win rate alongside a marginally worse aggregate MAE than naive. The 14 blended pairs are where this gap lives, and have not yet been individually diagnosed to identify which specific pair's test-set error is driving the aggregate shortfall.
- Coverage skews heavily toward refugee-camp submarkets and informal Nairobi settlements. Results speak most directly to humanitarian and NGO use, not general smallholder farmer markets.
- Of the six hypotheses in Section 1.5, only H1 and H2 received a dedicated statistical test. H2's preliminary signal did not survive the full-scale, properly matched version of the test. H3–H6 were addressed only structurally or explicitly deferred.
- Wholesale price forecasting was deliberately scoped out in favor of retail.

---

## 6. Deployment

The system is built as four scripts plus a dashboard, not a notebook the user must run manually:

| Component | Role | Status |
|---|---|---|
| `data_prep.py` | Load, clean, scope, join weather, engineer features → `master.parquet` / `shortlist.parquet` | **Complete, run successfully** |
| `train_model.py` | Trains naive baseline, per-pair Prophet (99/100 pairs), pooled LSTM, and the unified validation-weighted router; persists all artifacts | **Complete, run successfully** |
| `generate_forecasts.py` | Refreshes data, produces a genuine forward (next-month) forecast per pair, rebuilds anomaly detection, writes `forecasts.csv` / `price_history.csv` | **Complete, run successfully** |
| `app.py` | Multi-page Streamlit dashboard — Overview, Price Forecast, Early Warnings, Markets Monitor, Decision Support, Reports & Export, About | **Complete, confirmed running locally** |

Model artifacts are persisted for reuse without retraining: Prophet models as JSON (one per pair, via Prophet's own serialization format), the pooled LSTM as a `.keras` file, per-pair scalers via `joblib`, and router weights as a CSV — all committed alongside the code so the pipeline runs from a fresh clone with no external state.

**Remaining steps:** pin `requirements.txt` now that all scripts exist; commit pre-generated CSVs and model artifacts; deploy to Streamlit Community Cloud; confirm it loads on both phone and laptop.

---

## 7. Recommendations and Conclusions

### 7.1 What This Project Delivers

A forecasting router that individually ties or beats naive persistence on 93.9% of pairs, while not yet producing a net aggregate improvement — evidence that pooling and per-pair Prophet training both help at the margins, without yet adding up to a system that reliably outperforms simply carrying the last observed price forward. Separately, and more decisively validated, an anomaly detector with a real, independently backtested signal (27.6% vs. 9.7% flag rate during genuine historical shocks). Both findings are reported as measured, not adjusted to tell a cleaner story.

### 7.2 Recommendations for Future Work

1. Identify the specific pairs among the 14 blended ones dragging the aggregate MAE above naive, and determine whether a stricter validation-evidence threshold would exclude them without losing the genuine wins among the other 13.
2. Extend wholesale-price modelling for trader- and NCPB-facing use cases.
3. Test H3–H6 with dedicated statistical procedures.
4. Investigate calibrated uncertainty bounds for the LSTM track.
5. Expand anomaly-detection backtest coverage beyond refugee-camp and informal-settlement markets as more general county markets accumulate sufficient history.

### 7.3 Conclusion

The original objective — beating naive persistence at multi-month price forecasting — was revised over the course of the project toward a narrower, better-evidenced claim. With Prophet now genuinely trained on every long-history pair, the picture is more nuanced than either "it works" or "it doesn't": most individual pairs are ties or modest wins, but the aggregate has not yet crossed into a net improvement. Reporting both halves of that finding, rather than the more flattering one alone, is the standard this project has held itself to throughout — and the same standard the anomaly detection result meets more cleanly, with a real, independently verified signal that adds genuine value for the stakeholders this system was built for.
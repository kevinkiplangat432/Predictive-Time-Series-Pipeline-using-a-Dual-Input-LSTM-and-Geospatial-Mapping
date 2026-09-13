<!--markdownlint-disable-->

# Kenya Food Price Early Warning System
## CRISP-DM Data Report

**Course:** DSF-FT16, Moringa School Data Science Capstone

---

## 1. Business / Problem Understanding

### 1.1 Background and Problem Statement

Food prices in Kenya move sharply and unevenly across markets, driven by harvest cycles, rainfall, and local supply conditions. Farmers, traders, and food security institutions largely act on current or historical prices, not forecasts, a gap that shows up as poor sell/buy timing for farmers, avoidable inventory risk for traders, and institutional responses that arrive only after a shortage or price spike is already visible.

The core problem is not a lack of data: WFP already publishes Kenyan market price data spanning **20 years and 226 markets**, and NASA POWER publishes weather data, both freely. What is missing is a system that combines them into a forward-looking signal at the market level.

### 1.2 Objectives

1. Forecast staple food prices one month ahead, per market and commodity.
2. Detect when a market's price begins to diverge meaningfully from its own expected pattern (early warning).
3. Deliver both through an accessible dashboard, not a static analysis.
4. Build a reproducible, seeded pipeline, not a one-off notebook run.

### 1.3 Stakeholders

Six named stakeholder groups: smallholder farmers and cooperatives (sell-timing decisions), traders and market intermediaries (inventory timing), county agricultural offices and NDMA (early localized food-stress signals), NGOs and humanitarian organizations (procurement and cash-based intervention planning), the National Cereals and Produce Board (reserve and stabilization decisions), and urban consumers and food processors (indirect beneficiaries of institutional planning).

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
| WFP Kenya Food Prices | HDX (Humanitarian Data Exchange) | Historical commodity prices, target variable |
| NASA POWER | Public API | Daily rainfall and temperature, explanatory variables |

Both sources are free and public. The raw WFP extract contains **27,763 rows and 16 columns**, comfortably exceeding the capstone's minimum thresholds of 10,000 rows and 10 features, and is sourced from an international organization's open data portal rather than a Kaggle-style repository.

### 2.2 Dataset Description

The WFP dataset is one row per commodity, market, and date: a specific reported price at retail or wholesale level. It spans **January 2006 to August 2026**, covering **226 markets** and **51 distinct commodity labels** across **7 administrative regions**. Raw price values range from KES 5 to KES 19,800, with a mean of KES 1,308.39 and a median of KES 130, a heavily right-skewed distribution, expected given the dataset mixes retail per-kilogram prices with wholesale bulk-unit prices before any cleaning. NASA POWER is one row per day for a given coordinate; the two are joined via each market's latitude/longitude, already present in the WFP data.

### 2.3 Initial Data Quality Findings

- **Missingness:** confined to four columns: `admin1`, `admin2`, `latitude`, `longitude`, each missing in exactly **62 rows (0.22%)**, all tied to the single market lacking coordinates.
- **Coordinates** are complete for **225 of 226 markets**. The exception, Hola (Tana River), is excluded from the weather join rather than imputed.
- **Duplicates:** zero, checked at the (date, market, commodity, pricetype) grain.
- **Commodity labelling:** maize alone appears under **five separate labels** (`Maize`, `Maize (white)`, `Maize flour`, `Maize (white, dry)`, `Maize flour (white)`), a real cleanup consideration addressed by scope decisions in Data Preparation, not by merging labels.
- **Units:** 13 distinct price units are in use, and both retail (14,450 rows after scoping) and wholesale (9,155 rows) trade levels are mixed in the raw data, both resolved explicitly in Data Preparation.

---

## 3. Data Preparation

### 3.1 Cleaning and Standardization

Every unit was mapped to a kilogram-equivalent multiplier. The distribution of units in the raw data:

| Unit | Rows | Convertible to kg? |
|---|---|---|
| KG | 14,595 | Yes (×1) |
| 90 KG | 4,118 | Yes (×90) |
| L | 2,715 | No |
| 50 KG | 1,905 | Yes (×50) |
| 200 G | 1,362 | Yes (×0.2) |
| 500 ML | 804 | No |
| Unit (count) | 591 | No |
| 64 KG | 518 | Yes (×64) |
| 13 KG | 470 | Yes (×13) |
| 126 KG | 459 | Yes (×126) |
| 400 G | 178 | Yes (×0.4) |
| Head | 26 | No |
| Bunch | 22 | No |

This produced a standardized `price_per_kg` field for every weight-based row. Rows priced in a genuinely non-weight unit (L, 500 ML, Unit, Head, Bunch, 5 unit types) were identified and excluded on measurement grounds in the next step, alongside three fuel commodities excluded as out of scope. **12 commodities in total** were removed for stated reasons, not data failures, taking the working dataset from 27,763 to **23,605 rows**.

### 3.2 Scope Selection

Retail was chosen as the sole modelling series: **14,450 rows**, against 9,155 wholesale rows left untouched in the source data as a defined next phase. Retail is the price point most named stakeholders actually transact on.

### 3.3 Historical Coverage and Shortlisting

Completeness was measured against each pair's own active reporting window, not the full dataset span, a fixed-span measure would unfairly penalize the majority of series that only began consistent reporting in late 2023. Of **2,021 raw market-commodity combinations**, applying a 60% fair-completeness floor and splitting by years of active history gave:

| Track | Years active | Pairs |
|---|---|---|
| Prophet | 3+ years | 100 |
| LSTM | 1 to 3 years | 33 |
| Excluded | Under 1 year | 542 |

**Final shortlist: 133 pairs across 26 markets and 14 commodities**, producing a modelling-ready dataset of **6,077 rows**.

### 3.4 Exploratory Analysis Highlights

Maize and beans dominate observation counts (`Beans (dry)`: 1,854 rows; `Maize (white)`: 1,656; `Maize`: 1,518, all pre-scoping), confirming sufficient depth for individual analysis. An IQR check on raw maize prices found bounds of [−7.88, 88.73] KES, flagging **24 of 1,518** observations as statistical outliers, concentrated in specific high-cost markets rather than spread randomly. A calendar-month seasonality check found a mild, recurring pattern across years, partially supporting H1. Coverage is highly uneven across markets and years, which is precisely what the tiered Prophet/LSTM track split above was built to accommodate.

### 3.5 Outlier Handling

Outliers were re-checked per commodity on the standardized `price_per_kg` field: **69 of 6,077** modelling rows, broken down as:

| Commodity | Outliers |
|---|---|
| Meat (camel) | 26 |
| Salt | 17 |
| Beans (dry) | 8 |
| Maize | 5 |
| Maize (white) | 4 |
| Rice | 3 |
| Potatoes (Irish) | 2 |
| Sugar | 2 |
| Meat (beef) | 1 |
| Sorghum | 1 |

Classified by before/after comparison: **45 persistent shifts**, **22 transient spikes**, 2 with insufficient surrounding data. None were removed; the anomaly detection layer in Section 4 is designed specifically to act on this kind of divergence.

### 3.6 Weather Integration and Feature Engineering

Daily rainfall and temperature were retrieved from NASA POWER for **25 of 26** shortlisted markets (Hola failed retrieval with a 422 error, consistent with its missing coordinates), producing 6,200 monthly market-weather rows. Joined to price data: a **98.98% match rate** (6,015 of 6,077 rows matched on base weather; 6,000 of 6,077 once 3- and 4-month lag features are included).

Two feature engineering decisions matter methodologically:
- **Target transform:** the LSTM track predicts `price_diff` (month-over-month change), not the raw price level; **5,944 of 6,077 rows** have a valid `price_diff` (the 133 nulls are each pair's first observation). Far more stable across many pooled pairs with different long-run trends than the level would be.
- **Lag features:** rainfall and temperature were lagged 3 and 4 months. A preliminary single-market test (one year of Nairobi weather against the full national price series) suggested a strong relationship: correlation as high as **0.639 at a 4-month lag**. The full-scale, properly matched test across every shortlisted market found the rainfall correlation was effectively zero (**−0.015 at lag 3, −0.020 at lag 4**), while temperature showed a weak positive correlation (**0.110 and 0.109**). Both were retained as candidate features, not dropped; the preliminary result was an artifact of comparing mismatched series, not a real signal the fuller test failed to detect.

An entity-embedding LSTM variant (market/commodity identity as learned embeddings) was tried and dropped after checking it added no measurable improvement over a simpler pooled model with no identity input.

---

## 4. Modelling

### 4.1 Proof of Concept: Single-Series Validation

Before scaling to the full shortlist, the pipeline was validated on one well-populated series: Kitui, Maize (white), 180 monthly observations:

| Model | Test MAE | Test MAPE |
|---|---|---|
| Naive persistence | 8.54 | 7.72% |
| Prophet | 6.95 | 19.53% |
| LSTM | 2.26 | 6.78% |

LSTM was the clear winner on this single series, a promising sign, but one series proving out is not the same claim as 132 pairs proving out, which is what the batch run below actually tests.

### 4.2 Batch Approach

Three model families were built and compared at scale, all judged by MAE against naive persistence:

| Model | Track | Scope |
|---|---|---|
| Naive persistence | All 132 trained pairs | Baseline, no exceptions |
| Prophet, with rainfall/temperature as external regressors | 100 long-history pairs | **One individually trained model per pair, 99 of 100 trained successfully** |
| Pooled LSTM (no entity identity), predicting `price_diff`, 16 LSTM units, 6-month lookback | 33 short-history pairs | One shared model trained across all 33 |

An independent per-pair LSTM was also trained as a comparison baseline for the pooling decision: it achieved **MAE 15.43, MAPE 12.19%**, beating naive on only **6.1%** of its pairs. The pooled LSTM on the same 33 pairs achieved **MAE 6.03, MAPE 6.33%**; pooling cut error by roughly 60%, the strongest single modelling result in this project.

**A methodological correction made mid-project is worth documenting explicitly:** an earlier version of the batch pipeline scored every pair, prophet-track pairs included, through the pooled LSTM, never actually training a real Prophet model beyond the single Kitui demonstration series. This has since been corrected: every prophet-track pair now gets its own trained Prophet model, and every number in this report reflects that corrected pipeline.

### 4.3 Forecast Selection

Each pair's production forecast is a **weighted blend** of its own model and naive persistence, following Bates and Granger's (1969) forecast-combination result: weight scales with both the strength and the volume of that pair's own validation evidence (minimum 4 validation rows for any weight, full confidence at 8+).

**Verified results from the corrected pipeline, all 132 trained pairs:**

| Weight category | Pairs |
|---|---|
| Zero weight (pure naive) | 118 |
| Partial weight (blend) | 14 |
| ≥ 0.5 (model-leaning) | 0 |
| Highest single weight | 0.477 |

**93.9%** of pairs individually beat or tie naive on the held-out test set, a real increase from an earlier 83.3% figure computed before Prophet was properly batch-trained. However, the **aggregate mean blended test MAE (6.57)** is marginally worse than the **aggregate mean naive test MAE (6.55)**. Both figures are genuine and not in conflict: 118 of 132 pairs are exact ties by construction, so the 93.9% win rate is dominated by ties rather than clear wins, and a handful of the 14 partially-weighted pairs, likely including the pair at 0.477 weight, are wrong by enough on the untouched test set to offset the gains elsewhere in the aggregate average.

### 4.4 Anomaly Detection

Expected price for every pair defaults to naive persistence, consistent with the router finding above. A pair is flagged when its residual exceeds 2 standard deviations of its own expanding historical residual distribution (minimum 6 months of prior history required), computed using only prior data. **5,146 of 6,077 rows** were eligible for flagging; **497** were flagged, a **9.7% baseline rate**.

---

## 5. Evaluation

### 5.1 Business Criteria Revisited

| Criterion | Verdict |
|---|---|
| Forecasts/alerts add information beyond watching current prices | Partially met, see 5.2 for the nuanced modelling result; the anomaly alerts add clearer, independently verified information |
| Anomaly detector flags shocks without excessive false alarms | **Met**: 27.6% vs. 9.7% baseline |
| Runs on free, public data | **Met** |
| Pipeline refreshable without significant rework | **Met**, with one documented exception (Hola's weather gap) |

### 5.2 Data Mining Criteria Revisited

- **Beats naive, not an arbitrary threshold:** genuinely mixed. 93.9% of pairs individually beat or tie naive, but the aggregate mean blended MAE (6.57) is not actually better than the aggregate mean naive MAE (6.55). The router is functioning as designed, conservative, rarely committing real weight (max 0.477, none above 0.5), but where it does commit weight, that weight isn't yet reliably paying off.
- **Price-weather join, minimal loss:** met: 98.98% match rate (6,015 of 6,077).
- **Anomaly detector, elevated flag rate in real shocks:** met: backtested against three independently documented Kenyan price shocks (2022 Horn of Africa drought, Ukraine-linked grain and fertilizer shock, 2022 to 2023 fuel subsidy removal). Across **851 pair-months** of coverage inside these windows, the detector flagged **235 (27.6%)** against its own **9.7%** baseline, roughly 3x elevation. By shock: drought 28.3% (94/332 months), fuel subsidy 29.1% (132/454 months), Ukraine-linked 13.8% (9/65 months, too thin to conclude alone). **154 of 270** shock-pair instances were flagged at least once; **116** saw no flag.
- **Reproducibility:** met: every stochastic step is seeded (`SEED = 42`).

### 5.3 Honest Limitations

- The router's aggregate result is mixed: a strong per-pair win rate (93.9%) alongside a marginally worse aggregate MAE than naive (6.57 vs. 6.55). The 14 blended pairs are where this gap lives, and have not yet been individually diagnosed to identify which specific pair's test-set error is driving the aggregate shortfall.
- Coverage skews heavily toward refugee-camp submarkets and informal Nairobi settlements: **16 of the 26** shortlisted markets. Results speak most directly to humanitarian and NGO use, not general smallholder farmer markets.
- Of the six hypotheses in Section 1.5, only H1 and H2 received a dedicated statistical test. H2's preliminary signal (0.639 correlation) did not survive the full-scale, properly matched version of the test (−0.015 to −0.020). H3 to H6 were addressed only structurally or explicitly deferred.
- Wholesale price forecasting (9,155 rows available, untouched) was deliberately scoped out in favor of retail.

---

## 6. Deployment

The system is built as four scripts plus a dashboard, not a notebook the user must run manually:

| Component | Role | Status |
|---|---|---|
| `data_prep.py` | Load, clean, scope, join weather, engineer features → `master.parquet` (6,077 rows) / `shortlist.parquet` (133 pairs) | **Complete, run successfully** |
| `train_model.py` | Trains naive baseline, per-pair Prophet (99/100 pairs), pooled LSTM, and the unified validation-weighted router; persists all artifacts | **Complete, run successfully** |
| `generate_forecasts.py` | Refreshes data, produces a genuine forward (next-month) forecast per pair, rebuilds anomaly detection, writes `forecasts.csv` (133 rows) / `price_history.csv` (6,077 rows) | **Complete, run successfully** |
| `app.py` | Multi-page Streamlit dashboard: Overview, Price Forecast, Early Warnings, Markets Monitor, Decision Support, Reports & Export, About | **Complete, confirmed running locally** |

Model artifacts persisted for reuse without retraining: **99 individual Prophet models** as JSON, **1 pooled LSTM** as a `.keras` file, **33 per-pair scalers** via `joblib`, and one router-weights CSV covering all 132 trained pairs, all committed alongside the code so the pipeline runs from a fresh clone with no external state.

**First real forecast batch, from the current pipeline run:** of 133 shortlisted pairs, 132 received a forecast (1 fell back to naive due to the Hola weather gap). Warning-level distribution: **126 NORMAL, 4 LOW, 2 MODERATE, 1 HIGH**, consistent with a system where, per Section 4.3, the model rarely diverges strongly from naive.

**Remaining steps:** pin `requirements.txt` now that all scripts exist; commit pre-generated CSVs and model artifacts; deploy to Streamlit Community Cloud; confirm it loads on both phone and laptop.

---

## 7. Recommendations and Conclusions

### 7.1 What This Project Delivers

A forecasting router that individually ties or beats naive persistence on **93.9%** of 132 trained pairs, while not yet producing a net aggregate improvement (6.57 vs. 6.55 MAE), evidence that pooling (60% error reduction on the short-history track) and per-pair Prophet training both help at the margins, without yet adding up to a system that reliably outperforms simply carrying the last observed price forward. Separately, and more decisively validated, an anomaly detector with a real, independently backtested signal (**27.6% vs. 9.7%** flag rate across 851 pair-months of genuine historical shocks). Both findings are reported as measured, not adjusted to tell a cleaner story.

### 7.2 Recommendations for Future Work

1. Identify the specific pairs among the 14 blended ones dragging the aggregate MAE above naive, and determine whether a stricter validation-evidence threshold would exclude them without losing the genuine wins among the other 13.
2. Extend wholesale-price modelling (9,155 untouched rows) for trader- and NCPB-facing use cases.
3. Test H3 to H6 with dedicated statistical procedures.
4. Investigate calibrated uncertainty bounds for the LSTM track, so confidence ranges aren't Prophet-exclusive (currently 99 of 132 pairs have them, 33 don't).
5. Expand anomaly-detection backtest coverage beyond the 16 refugee-camp/informal-settlement markets, as the remaining 10 general county markets accumulate sufficient history.

### 7.3 Conclusion

The original objective, beating naive persistence at multi-month price forecasting, was revised over the course of the project toward a narrower, better-evidenced claim. With Prophet now genuinely trained on all 99 of its available long-history pairs, the picture is more nuanced than either "it works" or "it doesn't": 93.9% of individual pairs are ties or modest wins, but the aggregate (6.57 vs. 6.55 MAE) has not yet crossed into a net improvement. Reporting both numbers, rather than the more flattering one alone, is the standard this project has held itself to throughout, and the same standard the anomaly detection result meets more cleanly, with a real, independently verified 3x signal elevation (27.6% vs. 9.7%) that adds genuine value for the stakeholders this system was built for.
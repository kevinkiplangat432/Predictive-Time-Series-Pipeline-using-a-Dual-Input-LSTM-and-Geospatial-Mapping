<!--markdownlint-disable-->
## Modelling Report

### Objective

Three model families were built and compared, all judged against naive persistence by MAE and MAPE, and never against an arbitrary fixed threshold. Every model reads from the same `WEATHER_FEATURES` list defined in Feature Engineering: `rainfall_lag_3`, `rainfall_lag_4`, `temperature_lag_3`, `temperature_lag_4`. Every stochastic step is seeded (`SEED = 42`) for reproducibility.

### Naive Baseline

The baseline for every pair is simply the previous month's observed price. On the single demo series (Kitui, Maize white, 180 monthly observations): MAE 8.54, MAPE 7.72%. At batch scale, aggregated across all 132 trained pairs, mean naive MAE is 6.55.

### Prophet Track

**Proof of concept.** Demonstrated first on the Kitui Maize series: 140 training rows, 18 validation, 18 test. Test MAE 6.95, test MAPE 19.53%, beating naive on MAE.

**Batch training.** Every one of the 100 long-history pairs gets its own individually trained Prophet model, using rainfall and temperature as external regressors, with a chronological 80/10/10 split per pair. 99 of 100 trained successfully; one lost all rows to a single market's weather retrieval gap.

This was a real methodological correction made partway through the project. An earlier version of the batch pipeline scored every pair, including prophet-track pairs, through the pooled LSTM instead, and Prophet only ever ran on the one demo series. That gap has since been fixed, and every number below reflects the corrected pipeline.

### LSTM Track

**Proof of concept.** The same Kitui series, but predicting `price_diff` (month over month change) rather than the price level, since a differenced target stays stable across pairs with different long run trends, which matters once many pairs share one scaler. Test MAE 2.26, test MAPE 6.78%, beating naive on both metrics.

**Pooled model.** All 33 short-history pairs are pooled into one shared LSTM, 16 units, 6-month lookback, trained once across all of them. No market or commodity identity feeds into the model. An entity-embedding version, giving the model a learned representation of which market and commodity each sequence belonged to, was tried and dropped after checking it added no measurable improvement over the simpler version.

**Pooling versus independent per-pair models.** As a comparison, not a production model, an independent LSTM was also trained separately for each of the 33 short-history pairs. Result: independent per-pair models achieve MAE 15.43, MAPE 12.19%, beating naive on only 6.1% of pairs. The pooled model on the same 33 pairs achieves MAE 6.03, MAPE 6.33%, a roughly 60% reduction in error. This is the strongest single result in modelling: pooling substantially helps where individual pairs lack enough history to train alone.

### Unified Validation-Weighted Router

Every pair, regardless of track, feeds into one selection rule. Rather than a hard switch between a model's forecast and naive, each pair's forecast is a weighted blend of the two, following Bates and Granger's 1969 result on forecast combination: weight scales with both the strength and the volume of that pair's own validation evidence. A pair needs at least 4 validation rows to receive any weight at all, and reaches full confidence at 8 or more.

Result across all 132 trained pairs: 118 land at zero weight, pure naive. 14 receive a partial blend. 0 cross the 0.5 threshold that would mark a pair as genuinely model-leaning. The highest single weight is 0.477, held by one prophet-track pair.

### Model Comparison and Selection

93.9% of pairs individually beat or tie naive on the held-out test set, a real increase from an earlier 83.3% figure computed before Prophet was properly trained per pair. But the aggregate mean blended test MAE, 6.57, is marginally worse than the aggregate mean naive test MAE, 6.55.

Both figures are genuine and do not contradict each other. 118 of 132 pairs are exact ties by construction since they carry zero weight, so the 93.9% win rate is dominated by ties rather than clear wins. A small number of the 14 blended pairs, likely including the 0.477 weight pair, are wrong by enough on the untouched test set to offset the gains elsewhere in the average.

**Selection:** the production forecast for every pair is this validation-weighted blend, Prophet for prophet-track pairs and the pooled LSTM for lstm-track pairs, each combined with naive in the proportion its own validation evidence earned. No model is allowed to fully override naive anywhere in the shortlist.

---

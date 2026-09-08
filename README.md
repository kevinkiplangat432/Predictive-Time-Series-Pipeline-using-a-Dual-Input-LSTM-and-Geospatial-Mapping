<!--markdownlint-disable-->
# Kenya Food Price Early Warning System

> **Project status: modelling and anomaly detection validated. Dashboard and deployment in progress.**

## Problem

Farmers, traders, and food security institutions in Kenya often act on current or historical prices rather than any forward signal. This project investigates whether historical price and weather data can support two things: a useful price forecast, and a system that flags when a market is starting to behave outside its own normal range early enough to matter.

## What this project actually delivers

The original objective aimed to forecast prices two to three months ahead and beat a naive persistence baseline. Over the course of the project, three separate model families, Prophet, an independent per-pair LSTM, and a pooled entity-embedding LSTM, were all tested against that goal, and none reliably beat naive at the individual market-commodity level. This is documented as a real finding about how persistent Kenyan staple food prices are month to month, not treated as an unresolved bug.

What the work does support, backed by a real backtest against independently documented events, is a residual-based anomaly detector that flags unusual price movement meaningfully more often during real shocks than during ordinary months. See Key Findings below for the numbers.

## Data Sources

- **WFP Kenya Food Prices** (via HDX), retail and wholesale price observations by market and commodity. Currently covers January 2006 through August 2026 in the raw extract used.
- **NASA POWER API**, monthly rainfall and temperature by market location, joined to price data on market coordinates and date, no API key required.

## Note on the Forecast Selection Router

An earlier version of the per-pair forecast selector used a hard switch, each pair's final forecast came entirely from either the model or naive persistence, chosen by a validation comparison. That version is archived as `Kenyan_food_prices_hardswitch_archive.ipynb` for reference and is superseded by the design described below.

The current router replaces that switch with a weighted blend, informed by a well documented finding in forecasting research, going back to Bates and Granger's 1969 work on forecast combination and repeatedly confirmed in the M-competitions, that combining forecasts tends to outperform confidently selecting a single one, particularly when the evidence available to make that selection is thin. Model weight for each pair scales with both the strength and the volume of its validation evidence, so a pair with little evidence collapses toward naive, and a pair with more consistent support earns more trust, rather than a single validation comparison deciding the whole forecast.

This design was checked against two different confidence calibrations before being finalized, and the result held both times, no pair reached even 25% model weight under either calibration, let alone the 50% threshold that would mark a pair as genuinely model-leaning. That stability across two different reasonable choices of calibration is itself the evidence this is a real property of the data, not an artifact of one arbitrary constant.

## Methodology Summary

**Scope decisions.** Fuel commodities excluded as non-food. Milk, vegetable oil, bananas, and unit-incompatible kale and cabbage rows excluded because they're priced by volume or count, not weight, and this system works in price-per-kilogram terms throughout. A minimum history floor of one year of active reporting was applied before any pair was shortlisted.

**Resulting shortlist:** 133 market-commodity pairs across 26 markets and 14 commodities, split into 100 pairs with 3+ years of history (Prophet track) and 33 pairs with 1 to 3 years (LSTM track). Master table: 6,077 rows, 98.98% matched to weather data.

**Forecasting.** Prophet as the baseline for longer-history pairs. For shorter-history pairs, a pooled entity-embedding LSTM, sharing market and commodity representations across all series, is compared against an independent per-pair LSTM of matched capacity. Each pair's final forecast is a weighted blend of the pooled model and naive persistence, with the weight determined by how much validation evidence supports the model and how strong that evidence was, rather than a hard switch between the two. Half of the 132 modelled pairs have 4 or fewer validation rows to base that judgment on, a real data availability constraint, not a modelling shortfall, and the weighting scheme is designed to reflect that scarcity directly rather than treat every pair's validation result as equally trustworthy.

**Anomaly detection.** Expected price for every pair defaults to naive, the previous month's observed price, consistent with the router finding that no pair earns meaningful model weight (see Key Findings). A pair is flagged when its residual exceeds 2 standard deviations of its own expanding historical residual distribution, computed using only prior data, never future information.

## Key Findings

*(exact figures below are representative of a typical run; the LSTM training is not currently seeded, so exact MAE/MAPE values and exact weight figures shift slightly between runs, see Known Limitations)*

- **Pooling improves forecasting.** A pooled entity-embedding model roughly halves error compared to training an independent model per pair on the same short-history pairs, MAE around 6.3 versus 14.5 in the most recent run.
- **No model reliably beats naive at the individual pair level, and this holds regardless of how forecasts are combined.** Under a weighted blend that scales model trust by both validation strength and the amount of validation evidence available, checked against two different confidence calibrations, no pair reached even 25% model weight, and none crossed the 50% threshold that would mark a pair as genuinely model-leaning. The blended approach beats or ties naive on 81.8% of pairs overall, lower than a hard-switch design would show, since a hard switch trivially ties naive on every pair it doesn't touch at all, while this design allows a small model influence into more pairs, and that small influence still tends to underperform naive on the untouched test set. This is treated as a real, doubly confirmed finding, not a design flaw, at this data volume, no per-pair selection or blending scheme, however designed, currently produces a forecast that reliably improves on simply carrying the last observed price forward.
- **The anomaly detector shows a real signal.** Across 270 shortlisted pairs with any real data coverage inside one of three independently documented Kenyan price shocks (2022 Horn of Africa drought, Ukraine-linked grain and fertilizer shock, 2022-2023 fuel subsidy removal), the flag rate rises to 27.6% during shock windows versus a 9.7% baseline rate, roughly 3x elevation. This is not perfect recall, 116 of the 270 covered pairs saw no flag in their window, but it is a real, independently checked signal.
- **Scope limitation on the anomaly result.** Nearly all pairs with usable coverage this far back in history are refugee camp markets (Kakuma, Kalobeyei, Daadab) and a small number of informal Nairobi settlements. This result speaks most directly to humanitarian and NGO use, not general smallholder farmer markets, since those series mostly lack sufficient history to have been tested here.

## How to Run (current, notebook-only state)

1. Open `Kenyan_food_prices.ipynb`.
2. Restart the kernel and run all cells top to bottom in one continuous pass. Do not run cells out of order, several later sections depend on state built earlier in the same session.
3. Confirm the printed summary numbers in the Modelling and Anomaly Detection sections before treating any of them as final, see Known Limitations on run-to-run variance.

Once the pipeline scripts exist, this section will be replaced with `python data_prep.py && python train_model.py && python generate_forecasts.py && streamlit run app.py`.

## Environment

- Python 3.12, Anaconda environment.
- pandas 3.0, note: `.groupby(...).apply()` no longer returns grouping columns by default in this version, use `.transform()` or pass `include_groups=False` where relevant, this has caused real bugs during development and is worth knowing before modifying section 5.
- Key libraries: pandas, numpy, scikit-learn, tensorflow/keras, prophet, matplotlib, streamlit (once dashboard work starts).
- `requirements.txt` to be pinned to exact installed versions once the pipeline scripts are written, not before, so the versions actually reflect what the code was run against.

## Known Limitations

- LSTM training is not currently seeded. Exact MAE/MAPE figures and exact per-pair weight values vary between full runs, though the qualitative conclusions, pooling helps, no pair earns meaningful model trust, are consistent every time. A fixed seed should be added before this project is considered fully reproducible.
- Half of the 132 modelled pairs have 4 or fewer validation rows, the minimum this project treats as usable for any model weight at all. This is a structural data availability limit, not something a different model or a different router design can work around.
- Two of the six hypotheses stated in Business Understanding (H1, seasonality; H2, rainfall lag) received a dedicated statistical test. H2's preliminary result did not hold up under the full-scale version of the test. The remaining four hypotheses (H3 through H6) were either addressed only structurally or explicitly deferred, and are not reported as confirmed or refuted.
- Wholesale price forecasting, more directly relevant to traders, millers, and NCPB, was scoped out in favor of retail, since retail is the price point most named stakeholders actually transact on. Wholesale rows remain intact in the source data and are a defined next phase, not a data gap.
- The anomaly detection backtest's strongest coverage (long enough history to test against a specific 2022 shock window) skews toward refugee camp and informal settlement markets (Kakuma, Kalobeyei, Daadab, and several Nairobi informal settlements), 16 of the 26 shortlisted markets. The remaining 10, including Garissa, Marsabit, Mandera, Turkana, and Baringo, are general Kenyan county markets concentrated in arid and semi-arid regions, but mostly fall in the shorter-history LSTM track and were not part of the specific 2022 shock backtest. Coverage across both groups reflects which series WFP has monitored most consistently, not a deliberate scoping choice.

## Acknowledgments

Price data from the World Food Programme via the Humanitarian Data Exchange. Weather data from NASA's POWER Project. Built as a capstone for Moringa School's Data Science program (DSF-FT16).
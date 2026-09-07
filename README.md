<!--markdownlint-disable-->
# Kenya Food Price Early Warning System

A Moringa School Data Science capstone project (DSF-FT16). Turns Kenya's public food price history into an early-warning signal for unusual price movement, rather than a precise multi-month price forecast, a scope decision explained below, not a shortfall.

## Problem

Farmers, traders, and food security institutions in Kenya often act on current or historical prices rather than any forward signal. This project investigates whether historical price and weather data can support two things: a useful price forecast, and a system that flags when a market is starting to behave outside its own normal range early enough to matter.

## What this project actually delivers

The original objective aimed to forecast prices two to three months ahead and beat a naive persistence baseline. Over the course of the project, three separate model families, Prophet, an independent per-pair LSTM, and a pooled entity-embedding LSTM, were all tested against that goal, and none reliably beat naive at the individual market-commodity level. This is documented as a real finding about how persistent Kenyan staple food prices are month to month, not treated as an unresolved bug.

What the work does support, backed by a real backtest against independently documented events, is a residual-based anomaly detector that flags unusual price movement meaningfully more often during real shocks than during ordinary months. See Key Findings below for the numbers.

## Data Sources

- **WFP Kenya Food Prices** (via HDX), retail and wholesale price observations by market and commodity. Currently covers January 2006 through August 2026 in the raw extract used.
- **NASA POWER API**, monthly rainfall and temperature by market location, joined to price data on market coordinates and date, no API key required.


## Methodology Summary

**Scope decisions.** Fuel commodities excluded as non-food. Milk, vegetable oil, bananas, and unit-incompatible kale and cabbage rows excluded because they're priced by volume or count, not weight, and this system works in price-per-kilogram terms throughout. A minimum history floor of one year of active reporting was applied before any pair was shortlisted.

**Resulting shortlist:** 133 market-commodity pairs across 26 markets and 14 commodities, split into 100 pairs with 3+ years of history (Prophet track) and 33 pairs with 1 to 3 years (LSTM track). Master table: 6,077 rows, 98.98% matched to weather data.

**Forecasting.** Prophet as the baseline for longer-history pairs. For shorter-history pairs, a pooled entity-embedding LSTM, sharing market and commodity representations across all series, is compared against an independent per-pair LSTM of matched capacity. A per-pair router then chooses, using validation performance only, never test, whether each pair's final forecast comes from the model or from naive persistence.

**Anomaly detection.** Expected price for every pair defaults to naive, the previous month's observed price, since no pair has independently earned a model-based substitution (see Key Findings). A pair is flagged when its residual exceeds 2 standard deviations of its own expanding historical residual distribution, computed using only prior data, never future information.

## Key Findings

*(exact figures below are representative of a typical run; the LSTM training is not currently seeded, so exact MAE/MAPE values and the precise count of model-routed pairs shift slightly between runs, see Known Limitations)*

- **Pooling improves forecasting.** A pooled entity-embedding model roughly halves error compared to training an independent model per pair on the same short-history pairs, MAE around 6.3 versus 14.5 in the most recent run.
- **No model reliably beats naive at the individual pair level.** Typically only a handful of the 132 modelled pairs (observed range: 3 to 5 across separate runs) get routed to the model by the validation-based selector, and none of those have held up on the untouched test set in any run so far. This is treated as a real property of the data, thin per-pair validation splits at this data volume, not a fixable bug.
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

- LSTM training is not currently seeded. Exact MAE/MAPE figures and the exact count of model-routed pairs vary between full runs, though the qualitative conclusions are consistent every time. A fixed seed should be added before this project is considered fully reproducible.
- Two of the six hypotheses stated in Business Understanding (H1, seasonality; H2, rainfall lag) received a dedicated statistical test. H2's preliminary result did not hold up under the full-scale version of the test. The remaining four hypotheses (H3 through H6) were either addressed only structurally or explicitly deferred, and are not reported as confirmed or refuted.
- Wholesale price forecasting, more directly relevant to traders, millers, and NCPB, was scoped out in favor of retail, since retail is the price point most named stakeholders actually transact on. Wholesale rows remain intact in the source data and are a defined next phase, not a data gap.
- The anomaly detection backtest's strongest coverage is concentrated in refugee camp and informal settlement markets, not general rural markets, purely because those are the series with consistent reporting reaching back into the 2022 shock windows.

## Acknowledgments

Price data from the World Food Programme via the Humanitarian Data Exchange. Weather data from NASA's POWER Project. Built as a capstone for Moringa School's Data Science program (DSF-FT16).
<!--markdownlint-disable-->
## Anomaly Detection Report

### Methodology

Expected price for every pair defaults to naive persistence, the previous month's observed price, consistent with the modelling finding that no pair earns strong model trust. A pair is flagged when its residual, actual price minus expected price, exceeds 2 standard deviations of its own expanding historical residual distribution. The standard deviation is computed using only residuals strictly before the current month, never the current or a future point, and a minimum of 6 prior residuals is required before a pair becomes eligible to be flagged at all.

Of 6,077 modelling rows, 5,944 have a defined residual, 5,146 are eligible for flagging once the 6-month history requirement is met, and 497 are actually flagged, a 9.7% baseline rate.

### Backtest Design

Three real, independently documented Kenyan price shocks serve as validation cases, none of which were used to build or tune the detector: the 2022 Horn of Africa drought, the Ukraine-linked grain and fertilizer shock, and the 2022 to 2023 fuel subsidy removal. A shock window only counts as usable if a shortlisted pair has genuine data coverage inside it, checked directly against the master table.

### Results

Across 851 pair-months of coverage inside the three shock windows, the detector flagged 235, a 27.6% month-level rate against its own 9.7% baseline, roughly 3 times elevation.

By shock: the drought window hit 28.3% (94 of 332 months), the fuel subsidy window hit 29.1% (132 of 454 months), and the Ukraine-linked window hit 13.8% (9 of 65 months), though that last figure covers only one month per pair on average and is too thin on its own to draw a conclusion from.

Counting differently, 154 of 270 shock-pair instances were flagged at least once during their window, and 116 saw no flag at all. This is not a claim of high recall on every shock. It is evidence that the signal is real and detectable during genuine documented events, built with no knowledge of the specific events it was later tested against.

### Illustrative Case

The single demo pair, chosen by hit rate among pairs with at least 4 months of coverage rather than by raw coverage alone, is Ethiopia (Kakuma), Sugar, during the fuel subsidy removal window: 4 of 5 months flagged.

### Scope Limitation

Nearly every pair with usable coverage this far back in history is a refugee-camp submarket, Kakuma, Kalobeyei, or Dadaab, or an informal Nairobi settlement, 16 of the 26 shortlisted markets. This result speaks most directly to humanitarian and NGO use, not to general smallholder farmer markets, which mostly lack sufficient history in this dataset to have been tested here at all.
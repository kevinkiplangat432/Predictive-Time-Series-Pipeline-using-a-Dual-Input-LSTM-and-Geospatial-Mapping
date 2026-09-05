<!--markdownlint-disable-->
## Standup Update: Modelling Section, Week 3

### Where we stand

Data prep (sections 1-3) is solid: 175 market-commodity pairs after fixing a completeness floor and a scope filter, weather joined at high match rate, chronological train/val/test split in place. The modelling section has been rebuilt from scratch and is now running clean end to end, no errors, no broken metrics. That's real progress, most of this week went into finding and fixing bugs that were producing numbers that looked wrong for reasons that weren't obvious.

### Problems we ran into, and what we fixed

1. **Shortlist included pairs with almost no history.** 743 of 872 pairs had under a year of data, guaranteed to fail any model. Fixed with a `MIN_YEARS_ACTIVE = 1.0` floor applied before the shortlist is built, not after.

2. **A pooled model across all pairs looked like severe overfitting** (validation loss frozen far above training loss). Root cause: per-pair price scalers fit only on training data were extrapolating wildly on trending series in validation and test. Fixed by predicting month-over-month price change instead of price level, differenced before scaling rather than after, so the scaler never has to extrapolate as badly.

3. **Two implementation bugs surfaced while fixing the above**, a leftover double-differencing line from an earlier fix attempt, and a shift-then-slice ordering bug that produced `NaN` predictions for the first test row of every pair. Both traced to exact lines, both fixed and confirmed against clean diagnostic output.

4. **43 of 175 shortlisted pairs were silently missing from training.** Diagnosed, not guessed: 8 commodities (milk in four forms, vegetable oil, bananas, and three fuel products) are priced in litres or bare units, not weight, so they were never convertible to a per-kg price and shouldn't have been in a food-price shortlist in the first place. Fixed with an explicit scope filter, applied upstream in section 3, documented as a deliberate decision rather than a silent gap.

5. **A generalization gap between training and validation loss was real**, not a measurement artifact this time. Fixed with regularization (smaller LSTM, recurrent dropout, L2 on the LSTM kernel and both entity embeddings). Training is now stable and the two curves track each other.

### The core finding, and the honest limitation

**Entity embeddings clearly help where data is scarce.** On the 33 shortest-history pairs, a pooled model with shared weights across all pairs cuts MAE by more than half compared to fitting a separate LSTM per pair (6.37 vs 14.95). That's a proven, controlled result and it's our strongest piece of evidence.

**Beating the naive baseline overall is not solved, and I don't think it's a bug.** Across all 132 modelled pairs, only 12.9% beat naive on MAE. This has now shown up across three separate model families (Prophet, per-pair LSTM, pooled embeddings), which points to something structural in monthly staple food prices, they're highly persistent month to month, not something a different architecture fixes.

### What we haven't fixed, and the proposed solution

**Multi-step forecasting.** Business objective 1 says "2-3 months ahead." Everything built so far forecasts one month ahead only. Proposed fix: either build a recursive forecast (feed each prediction back in for the next step) this week, or formally revise the objective the same way we already revised the original 2-4 week horizon, and say so explicitly in the report rather than let the gap go unaddressed.

**Shock detection / early warning.** Not built in this notebook, only prototyped elsewhere with an incomplete backtest. Proposed solution: build it properly against our now-fixed data, and validate it against at least one real, documented, citable Kenyan price shock before trusting it, that test is the one thing that turns this from a plausible idea into a proven feature.

**Dashboard and automated pipeline.** Not started. Proposed solution: minimal, not elaborate, a single view with a market/commodity selector, the forecast, and a shock flag. This is UI work, not research risk, lowest priority relative to the above.

### How we improve the modelling section: throwing out the old plan

Instead of continuing to force one model architecture to be the answer for every pair, the proposal is to route each pair to whichever forecasting approach actually earns its keep on that pair, decided by validation performance, not assumption:

- Compute how much each pair's price moves month to month on its own (naive MAPE). Pairs that barely move get naive as the forecast, since nothing beats "no change" there, and that's not a failure, it's a property of that market.
- Pairs with real month-to-month movement get the learned forecast, whichever of Prophet, per-pair LSTM, or pooled embeddings performs best for that pair on validation data.
- This replaces every existing "does the model beat naive overall" framing and every markdown cell built around that framing. The new report doesn't claim one model wins everywhere, it claims we can tell, per pair, which approach is right, and route accordingly. That's a stronger and more honest system than any single model forced across all 132 pairs, and it turns the 12.9% number from a discouraging headline into evidence for why the routing approach is the right design.

**Bottom line for standup:** the pipeline is finally trustworthy, the embeddings hypothesis is proven where it was meant to matter, and the plan going forward is to stop trying to make one model win everywhere and instead build a system that knows which pairs are worth modelling at all.


## Task Distribution: Modelling Section, Remaining Work

### Person 1: Router design and implementation
- Build the per-pair router (naive vs learned forecast, selected on validation MAE, not test)
- Add the linear regression and seasonal-naive-with-drift candidates to the router
- Rewrite the 4.21 markdown to describe the routing decision, not a single model's performance
- Owns: does the routing logic actually select correctly per pair

### Person 2: Multi-step forecasting
- Implement recursive forecasting (feed each month's prediction back in as input for the next) to actually produce a 2-3 month horizon
- Evaluate how much error compounds by step (month 1 vs month 2 vs month 3 accuracy)
- If recursive forecasting doesn't hold up, draft the alternative: a formal, documented revision of objective 1 to a 1-month horizon, same pattern as the earlier 2-4 week to 1-3 month revision
- Owns: whether the "2-3 months ahead" objective is actually deliverable or needs to be rescoped

### Person 3: Shock detection and backtest
- Rebuild the residual-based anomaly detection from the teammate's prototype, on top of the now-fixed shortlist and master table
- Identify 1-2 real, citable Kenyan price shocks (2022 Horn of Africa drought, Ukraine-linked grain/fertilizer shock, fuel subsidy removal knock-on effects) and confirm they fall inside a pair's test window
- Run the actual backtest this time, not a placeholder, and report false-positive rate honestly
- Owns: whether shock detection is a provable feature or another documented limitation

### Person 4: Dashboard
- Minimal Streamlit view: market/commodity selector, forecast line, naive-vs-model indicator, shock flag if Person 3's work is ready in time
- Don't wait on Person 3, build the forecast-only version first, add the shock flag as a fast follow
- Owns: the one artifact non-technical stakeholders will actually see

### Person 5: Report and reproducibility
- Consolidate the standup summary into the CRISP-DM report, including the honest framing on naive-vs-model performance and the scope decisions made along the way (fuel exclusion, milk/oil/bananas out of scope, the completeness floor)
- Turn the notebook into a script or clean pipeline (data prep to forecast to dashboard input), even a simple sequential runner, since "reproducible automated pipeline" is one of the five business objectives and currently only exists as a notebook
- Owns: whether a stranger could rerun this end to end and get the same result

### Sequencing note
Person 1's router depends on final numbers from the current notebook, once that's settled (it should be), Persons 1, 2, and 3 can work in parallel. Person 4 can start immediately with a placeholder forecast. Person 5 should start now too, report writing doesn't need to wait for everything else to finish.
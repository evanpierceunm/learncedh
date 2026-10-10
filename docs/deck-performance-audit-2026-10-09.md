# Performance review: October 9, 2026

The score has modest predictive value, but it does not isolate deck strength from pilot skill. We kept the current formula after testing pilot-aware alternatives.

The review covered source filtering, fixed-cut likelihood, sample-size shrinkage, score normalization, numerical accuracy, uncertainty, temporal validation and repeated pilots. The existing 14 model/data checks pass. The retained scoring window contains 243 eligible tournaments, 7,491 entries and 4,788 identified pilots; 36.1% of entries are beyond a pilot's first appearance. No missing IDs or duplicate pilot within an eligible event was found.

**Unique pilot counts currently describe the evidence; they do not change the score.** Changing identities without changing deck/event results produces identical estimates. Repeated pilots can therefore make the model ranges too narrow, and pilot skill can affect a deck's estimate. Rewarding decks merely for having more pilots would introduce a popularity bonus.

## Historical comparisons

Candidates were selected on three earlier months and compared on the following month: 49 events and 1,567 entrants. That final month had already been inspected in initial development, so this is retrospective evidence, not a new untouched holdout. Lower log loss is better.

| Method | Log loss |
|---|---:|
| Current model | 0.56345 |
| Discount repeated pilots by square-root frequency | 0.56362 |
| Equal total likelihood weight per pilot | 0.56382 |
| Regularized deck-only logistic model | 0.56330 |
| Separate deck/pilot model, including pilot history | 0.55159 |
| Separate deck/pilot model, deck component only | 0.56418 |
| Event cut fraction alone | 0.56539 |

The pilot-history model predicts a particular player plus deck better. Removing the player's contribution did not produce a better general deck predictor. A separate test withheld each test pilot's entire earlier history using five deterministic groups and chronological boundaries. Current log loss was 0.56481, versus 0.56505 for the pilot-separated deck estimate. The small deck-only logistic improvement was inconclusive under event resampling. Repetition discounts did not improve the later-month average.

The separate-effects prototype uses regularized logistic MAP estimates with an event-cut offset; it is not the current fixed-cut likelihood and does not supply production uncertainty estimates. Discounted-likelihood variants are sensitivity experiments, not proven correlation corrections.

## Ranking quality

Using only earlier results to rank decks, the higher-scoring deck was the qualifier in 54.5% of 13,499 same-event qualifier/nonqualifier pairs involving different decks, counting ties as half. An uninformative ordering is 50%. The event-resampled interval was 51.6% to 59.1%; repeated pilots across events remain a limitation.

The five highest-scoring primer decks made 56 later cuts versus 53.9 expected from their events' field rates. The top ten made 98 versus 89.3. These overlapping, retrospective comparisons do not establish reliable exact positions or certify the best five decks.

The score remains a sample- and event-adjusted performance estimate. A next version needs an explicitly defined common-pilot target, joint event/deck/pilot modeling, appropriate repeated-pilot uncertainty and fresh prospective validation. Current ranges and sample counts remain available in each card's details.

Methods were informed by [scikit-learn's grouped and temporal validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) and [statsmodels' generalized mixed-model documentation](https://www.statsmodels.org/stable/mixed_glm.html). Those methods do not themselves establish that deck and pilot effects can be cleanly separated in this dataset.

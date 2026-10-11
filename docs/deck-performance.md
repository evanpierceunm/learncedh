# Performance

**100 = the reference field baseline; higher is better.** Performance estimates how likely a deck is to make a tournament cut. It accounts for the share of entrants who qualify, the amount of data available and the number of different pilots playing the deck. It is the default sort on [LearnCEDH's Decks page](https://learncedh.com/decks).

Each score combines results from different lists and pilots using the same commander or commander pair. It does not measure game win rate or the chance of winning a tournament, and it cannot separate the deck’s strength from its pilots’ skill. Small score differences and overlapping ranges leave the order uncertain.

## Results included

We use a rolling 180-day window from [EDHTop16](https://edhtop16.com/). To allow time for reporting, the cutoff is two UTC dates before the update date. Events on or after that cutoff are excluded. Each card’s popup shows the period covered.

An event must have at least 16 entrants, qualification rounds, a complete entrant roster, a valid cut smaller than its field, and commander identities for at least **95% of entrants**. We leave missing commanders unidentified. To compare results, an event must include players with known commanders on both sides of the cut.

Final standings must identify exactly the stated number of qualifiers and one winner. Every qualifier must have a recorded bracket result, the winner must have a bracket win, and nonqualifiers cannot have bracket results. These checks help us screen for completed events, though the source does not guarantee that every result is final. Duplicate records, explicit test/custom-rule flags, invalid standings, out-of-window events and incomplete rosters are excluded.

The previous version required every commander to be known, which disproportionately excluded large tournaments. Version 2 includes events with at least 95% of commanders identified and keeps the other checks. The public feed reports entrant counts, how many commanders are known or missing, how many events have missing commanders, and why events were excluded. Each deck card’s sample counts include only entrants identified as playing that deck.

Some decks or finishes may be more likely to have their results reported, which can bias the sample. Events below 95% coverage are excluded even if some useful results remain. The 95% threshold lets us include more events while limiting missing information. We have not established that it is the best possible threshold. Conversion rate and metashare use EDHTop16’s separate three-month window, so their samples can differ from Performance’s.

## Base model

Within each event, we compare entrants whose commanders are known. Let `N` be their count, `c` their number of qualifiers, `m` the count on one deck, and `k` its qualifiers:

```
P(k | N,c,m,theta) ∝ choose(m,k) × choose(N-m,c-k) × exp(k × theta)
```

This is [Fisher's noncentral hypergeometric distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nchypergeom_fisher.html). Entrants with unknown commanders are left out of this comparison. When every commander is known, the calculation is the same as in the original model. It assumes that missing commander information does not skew one deck’s qualification odds relative to another’s.

`theta` is the deck's qualification log-odds advantage against the pooled identified field. A Normal prior centered at zero pulls estimates based on small samples toward neutral. Its spread is fitted by marginal likelihood across all observed commander groups, not only decks with primers. We integrate on the grid -5 to 5 in steps of 0.01. We fit each deck separately, so the model does not estimate individual matchups or pilot skill.

For scoring, all decks use the same entry-weighted mix of **full-event** cut fractions. The base reference probability `p_d` is the posterior mean single-entrant prediction across that mix. Full field sizes remain in the reference even when a few commander identities are missing. The field and cut sizes tell us what share of players can qualify. They do not tell us how strong those players are.

## Pilot count adjustment

The adjustment uses the number `u_d` of distinct identified pilots on that deck in its 180-day training window:

```
s_d = -0.0550359053 + 0.0395358518 × (log(1 + u_d) - 4.6918455687)
p'_d = logistic(logit(p_d) + s_d)
```

When a deck already has many pilots, doubling its pilot count increases its predicted odds by about 2.8%, with the base prediction held fixed. We fitted this adjustment using earlier forecasts and the results that followed them. A player counts once toward the pilot total, even if they enter several events. All their entries still contribute to the base model, which does not account for the statistical dependence between results from the same player. The adjustment does not account for pilot skill. Missing pilot IDs do not add to the count and cause the deck to be marked “Limited data.”

The coefficients and centering are **frozen**, stored in `scripts/ranking/adoption-calibration.json`. Daily updates refresh results and pilot counts. Changing the coefficients requires a new version and a separate evaluation.

Training used three 30-day validation periods from June 9 through September 6, 2026. Each forecast used only its preceding 180-day training window. Logistic calibration used the predicted event-specific log odds as an offset, an intercept and log pilot count, with fixed L2 penalty 1. We used the following month to compare results without fitting the coefficients to it. Because we had examined that month in earlier research, it was not a fresh test on unseen data. Pilot count helped predict results in these historical comparisons. That does not show that choosing a popular deck will improve a player’s results.

## Scores and uncertainty

Let `b` be the shared full-event qualification fraction. For every observed deck compute its neutral calibrated probability `logistic(logit(b) + s_d)`. Average those probabilities using identified entrant counts as weights to obtain one common neutral baseline `B`. The published index is:

```
Performance_d = 100 × p'_d / B
```

A neutral field with this adoption mix averages 100. This does not force the actual scores to average 100. Every deck shares `B`, including after website filters change. Sorting uses the score stored to three decimals; cards display one decimal, and alphabetical order resolves exact ties. Decks with no eligible results remain unranked.

The displayed 95% range transforms the base posterior's 2.5th and 97.5th percentiles through the same adoption adjustment and common scale. This range assumes the fitted prior, reference mix and calibration are fixed. It does not include uncertainty from fitting the calibration, missing commanders, repeated results from the same pilots or changes in opponents. It describes uncertainty in the score, not in a deck’s exact rank. We adjust the base posterior mean to calculate the score; that is not generally the same as averaging the adjusted posterior.

The popup also shows eligible entries, events and identified pilots. Fewer than five events, fewer than five pilots, or any missing pilot IDs produce a “Limited data” label. The label flags a small sample or missing pilot information; it is not a test of statistical significance.

## Historical evidence

Across four months of historical forecasts, we compared pairs of entrants in the same event who played different decks, with one making the cut and the other missing it. The broader-coverage model ranked the qualifier’s deck higher in 56.08% of pairs, compared with 54.21% under the original coverage policy. Sorting by a lower posterior percentile did not improve this measure across the four months, so we kept the mean-based score.

The table compares predictions for the same 2,657 entrants with known commanders in the final historical month. Every included event had at least 95% of commanders identified. Lower log loss means better probability predictions; rank concordance measures how often the qualifier’s deck ranked higher in the pairs described above.

| Candidate | Log loss, lower is better | Rank concordance |
|---|---:|---:|
| Original strict coverage | 0.538409 | 54.74% |
| Broader coverage | 0.536409 | 56.75% |
| Broader coverage, intercept-only calibration | 0.536138 | 56.75% |
| Broader coverage, learned pilot count | 0.534925 | 57.25% |

The paired event-bootstrap 95% interval for the final row's log-loss difference from intercept-only calibration was [-0.002170, -0.000244]. The improvement was small and came from historical data. Resampling events does not account for all uncertainty from repeated pilots, missing information or model selection. The interval for the ranking improvement from broader coverage alone included zero. These results leave uncertainty about the order of individual decks and about deck strength apart from pilot skill.

The [v2 validation summary](deck-performance-v2-validation.json) records the data sources, comparisons and calibration details. The [earlier pilot audit](deck-performance-audit-2026-10-09.md) and original validation files are historical v1 evidence, not fresh v2 evaluations. `validate.py` compares base likelihoods before adoption calibration; it is not a full v2 validation tool.

To reproduce the frozen coefficient fit using the archived full-year snapshot:

```
python scripts/ranking/fit_calibration.py --snapshot PATH/events.json --evaluation-end 2026-10-07 --output calibration-review.json
```

The command writes a review file without changing the live calibration. It needs source data covering the 300 days through that endpoint. Corrections to historical results can change the output of a later run.

## Daily updates

The scores update daily at 00:20 UTC through a GitHub workflow. It downloads every page of results, runs tests and publishes the new feed only if all checks pass. Raw snapshots and exclusions are retained as workflow artifacts for 14 days; the public score feed contains no player identities. The previous feed stays live if an update has fewer than 20 eligible events, insufficient coverage, invalid scores or dates earlier than the previous window. A drop of more than 30% in event coverage also stops publication when the model version has not changed.

If the website cannot load the Performance feed, it shows N/A and sorts decks by popularity. The popup flags data older than seven days. Conversion rate and metashare use separate feeds.

Model version: `conditional-cut-adoption-v2`. Implementation: `scripts/ranking/`; tests: `tests/ranking/`; website: `site/decks-index.html`.

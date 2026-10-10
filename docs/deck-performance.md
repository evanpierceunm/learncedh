# Performance

**100 = the reference field baseline; higher is better.** Performance estimates tournament-cut results using event context, sample-size shrinkage and a modest, historically learned adjustment for breadth of adoption. It is the default sort on [LearnCEDH's Decks page](https://learncedh.com/decks).

The score describes commander archetypes across their lists and pilots. It is not a game-win percentage, tournament-win probability, or causal measure of deck strength independent of pilot skill. Close scores and overlapping ranges do not establish an exact ordering.

## Results included

We use a rolling 180-day window from [EDHTop16](https://edhtop16.com/). The daily build ends two UTC days before its build date, exclusively, to allow reporting time. Each card's popup gives the actual included dates.

An event must have at least 16 entrants, qualification rounds, a complete entrant roster, a valid cut smaller than its field, and commander identities for at least **95% of entrants**. Missing commander identities remain unknown. An event needs both identified qualifiers and identified nonqualifiers to contribute comparative information.

Final standings must identify exactly the stated number of qualifiers and one winner. Every qualifier must have a recorded bracket result, the winner must have a bracket win, and nonqualifiers cannot have bracket results. These are completion checks, not a source-provided guarantee. Duplicate records, explicit test/custom-rule flags, invalid standings, out-of-window events and incomplete rosters are excluded.

The previous version required every commander to be known, which disproportionately excluded large tournaments. Version 2 retains mostly complete events while preserving the other checks. The public feed reports total entrants, identified and unknown entrants, partial-label event counts, coverage and exclusion reasons. Card sample counts include only identified entrants on that archetype.

Reporting can depend on results and archetype, so this policy does not eliminate selection bias. Events below 95% coverage are excluded even if some useful results remain. The threshold balances coverage and missing information; it is not a statistically proven optimum. Raw conversion and metashare retain their separate EDHTop16 three-month definitions and need not match the performance sample.

## Base model

Within each event, use only entrants with known commanders for the comparison. Let `N` be their count, `c` their number of qualifiers, `m` the count on one deck, and `k` its qualifiers:

```
P(k | N,c,m,theta) ∝ choose(m,k) × choose(N-m,c-k) × exp(k × theta)
```

This is [Fisher's noncentral hypergeometric distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nchypergeom_fisher.html). Unknown entrants are not assigned another deck or silently treated as opponents of every named deck. For complete-label events this is the original likelihood. The conditional comparison still assumes missing-label selection does not distort relative success odds differently by archetype.

`theta` is the deck's qualification log-odds advantage against the pooled identified field. A zero-centered Normal prior shrinks sparse results toward neutral. Its spread is fitted by marginal likelihood across all observed commander groups, not only decks with primers. We integrate on the grid -5 to 5 in steps of 0.01. Separate fits do not constitute a joint matchup or pilot-skill model.

For scoring, all decks use the same entry-weighted mix of **full-event** cut fractions. The base reference probability `p_d` is the posterior mean single-entrant prediction across that mix. Full field sizes remain in the reference even when a few commander identities are missing. Event size describes the qualification opportunity, not the strength of its players.

## Learned adoption adjustment

The versioned calibration uses the number `u_d` of distinct identified pilots on that deck in its 180-day training window:

```
s_d = -0.0550359053 + 0.0395358518 × (log(1 + u_d) - 4.6918455687)
p'_d = logistic(logit(p_d) + s_d)
```

The coefficient is small: at substantial pilot counts, doubling adoption increases predicted odds by about 2.8%, holding the base prediction fixed. It was learned from earlier chronological forecasts, not chosen to put specific decks higher. Repeat entries from one pilot do not increase the distinct-pilot feature. They still contribute base-model results, so this does not remove dependence between repeated observations or adjust for skill. Missing pilot IDs contribute no invented identities and mark a deck's evidence as limited.

The coefficients and centering are **frozen**, stored in `scripts/ranking/adoption-calibration.json`. Daily builds update results and pilot counts but do not silently refit the coefficients. New calibration requires explicit versioning and evaluation.

Training used three 30-day validation periods from June 9 through September 6, 2026. Each forecast used only its preceding 180-day training window. Logistic calibration used the predicted event-specific log odds as an offset, an intercept and log pilot count, with fixed L2 penalty 1. The following historical month was used for comparison, not coefficient fitting. It had already been examined in prior research and is not an untouched prospective test. The feature is a predictive association with breadth of adoption, not evidence that choosing a popular deck causes better results.

## Common scale and ranges

Let `b` be the shared full-event qualification fraction. For every observed deck compute its neutral calibrated probability `logistic(logit(b) + s_d)`. Average those probabilities using identified entrant counts as weights to obtain one common neutral baseline `B`. The published index is:

```
Performance_d = 100 × p'_d / B
```

A neutral field with this adoption mix averages 100. This does not force the actual scores to average 100. Every deck shares `B`, including after website filters change. Sorting uses the score stored to three decimals; cards display one decimal, and alphabetical order resolves exact ties. Unobserved decks remain unranked.

The displayed 95% range transforms the base posterior's 2.5th and 97.5th percentiles through the same adoption adjustment and common scale. It is conditional on the fitted prior, reference mix and fixed calibration. It excludes calibration-estimation uncertainty, unknown-label selection, repeated-pilot dependence and changing opponents. It is not a confidence interval for leaderboard position. The score is a calibrated point estimate, not a claim that transforming a posterior mean equals the mean of a transformed posterior.

The popup also shows eligible entries, events and identified pilots. Fewer than five events, fewer than five pilots, or any missing pilot IDs produce a “Limited data” label. This display rule is not a significance threshold.

## Historical evidence

On four chronological months, broader-coverage mean ranking distinguished a qualifier from a nonqualifier on a different deck within the same event in 56.08% of pairs, compared with 54.21% for the original coverage policy. Conservative lower-quantile ranking did not improve pooled discrimination, so it was not adopted.

In the final historical month, on the same 2,657 identified entrants in events with at least 95% labels:

| Candidate | Log loss, lower is better | Rank concordance |
|---|---:|---:|
| Original strict coverage | 0.538409 | 54.74% |
| Broader coverage | 0.536409 | 56.75% |
| Broader coverage, intercept-only calibration | 0.536138 | 56.75% |
| Broader coverage, learned pilot count | 0.534925 | 57.25% |

The paired event-bootstrap 95% interval for the final row's log-loss difference from intercept-only calibration was [-0.002170, -0.000244]. These are modest retrospective improvements. Event resampling does not remove shared-pilot dependence or all missingness and model-selection uncertainty. The broader-coverage ranking improvement alone had an interval spanning zero. These results do not validate a precise deck ordering or independently establish intrinsic strength.

The [v2 validation summary](deck-performance-v2-validation.json) retains provenance, comparisons and calibration details. The [earlier pilot audit](deck-performance-audit-2026-10-09.md) and original validation files are historical v1 evidence, not fresh v2 evaluations. `validate.py` compares base likelihoods before adoption calibration; it is not a full v2 validation tool.

To reproduce the frozen coefficient fit using the archived full-year snapshot:

```
python scripts/ranking/fit_calibration.py --snapshot PATH/events.json --evaluation-end 2026-10-07 --output calibration-review.json
```

This explicit research command does not overwrite the production calibration. Source data must cover the 300 days through that endpoint. Historical source corrections can change a later rerun.

## Updates and failure behavior

The GitHub workflow runs daily at 00:20 UTC. It fetches complete pages, runs tests, builds the feed and publishes only on success. Raw snapshots and exclusions are retained as workflow artifacts for 14 days; the public score feed contains no player identities. Fewer than 20 eligible events, insufficient coverage, invalid scores, backward windows or a greater-than-30% event-coverage drop within the same model version stop publication and retain the prior feed.

The website's Performance default, hover/tap popup, card styling and other sorts are unchanged. If the feed fails, Performance displays N/A and sorting falls back to popularity. Details flag feeds older than seven days. Original conversion/metashare feeds and primer content remain separate.

Model version: `conditional-cut-adoption-v2`. Implementation: `scripts/ranking/`; tests: `tests/ranking/`; website: `site/decks-index.html`.

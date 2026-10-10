# Performance

Performance estimates how a commander archetype performs at making tournament cuts, accounting for sample size and the fraction of each event that qualifies. It is the default sort on [LearnCEDH's Decks page](https://learncedh.com/decks).

**100 is the field baseline. Higher is better. This is an index, not a win percentage.** Every deck is compared on the same event mix. More entries give the model more evidence; popularity does not earn bonus points.

The score covers commander archetypes, including different lists and pilots. It does not isolate pilot skill, measure the chance of winning the whole tournament, or establish that one exact decklist is better than another. Similar scores with overlapping ranges should not be read as a settled ordering.

## Data included

We use a rolling **180-day** window from [EDHTop16](https://edhtop16.com/), whose tournament results include [TopDeck](https://topdeck.gg/) events. Dates use UTC. The daily update uses an exclusive end date two days before the UTC build date, leaving a short reporting buffer; the exact included dates appear in each card’s expandable performance details.

An event needs at least 16 entrants, qualification rounds, a real cut smaller than its field, a complete entrant roster and a known commander for every entry. Final standings must identify exactly the stated number of qualifiers. Every qualifier must have a recorded bracket result, the sole first-place finisher must have a bracket win, and nonqualifiers cannot have bracket results. These checks are evidence of a finished bracket; they are not a source-provided completion-status guarantee.

We exclude future/out-of-window dates, duplicate records, names explicitly flagged as test events, explicit custom-rules flags, missing or inconsistent fields, no-cut events, and incomplete commander labels. Empty names and “Unknown Commander” are missing data. We retain exclusion counts and build snapshots. The public score feed includes no player identities.

This strict policy excludes many otherwise real tournaments. The results describe this eligible sample, not every cEDH event. It reduces missing-label bias within included events but does not eliminate selection bias between included and excluded events. Undocumented custom formats and incorrect source records can still escape detection. Raw conversion and metagame share retain their separate EDHTop16 definitions and three-month window, so their totals need not match this score's sample.

At initial release, the window is April 10 through October 6, 2026: **243 events, 7,491 entries and 515 commander groups**. Of the 72 primers on LearnCEDH, 71 have eligible observations; Florian has no score in this window and remains searchable. These counts will change with daily updates.

## How the score works

For a deck at one event, let `N` be the field size, `c` the actual cut size, `m` the number of entrants on that deck and `k` its qualifiers. The model conditions on exactly `c` qualifying places:

```
P(k | N,c,m,theta) ∝ choose(m,k) × choose(N-m,c-k) × exp(k × theta)
```

This is Fisher's noncentral hypergeometric distribution. Its support prevents more qualifiers than the event's cut or the deck's entrant count. See the [SciPy distribution documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nchypergeom_fisher.html). Tests compare our probability calculation against SciPy's independent implementation.

`theta` is the deck's log qualification-odds advantage against the pooled remaining field. We use a zero-centered Normal prior and estimate its spread by marginal likelihood across all observed commander groups in the training window, not just decks with primers. A small sample is pulled more strongly toward neutral odds. The fitted spread is stored in each feed.

For each deck, we integrate over its posterior on a log-odds grid from -5 to 5 in increments of 0.01. We predict a single entrant's qualification chance across the same entry-weighted distribution of event cut fractions for every deck. The published score is 100 times this posterior mean prediction divided by the reference field's qualification fraction. Sorting uses the unrounded score; the page displays one decimal place.

This conditions on fixed cut capacity within each deck/event observation. Separate deck fits treat all other decks as a pooled field; they do not form a jointly normalized matchup model of every deck at the event. Event size is qualification context, not proof that a tournament has stronger opponents.

## Uncertainty and limited data

Each card’s expandable details show a **95% model range**, eligible entries, distinct events and distinct identified pilots. The range is conditional on the fitted prior and model assumptions. It is not a confidence interval for an exact leaderboard position. Repeated pilots across events, variation in opposing fields, unknown format details and fitting the prior can make actual uncertainty larger.

“Limited data” is a transparent display rule: fewer than five events, fewer than five identified pilots, or missing pilot IDs. It is not a statistical significance threshold. Such decks can still have a score. A deck with no eligible results is unranked, never assigned a zero, and sorts after rated decks. Alphabetical order breaks equal scores.

We sort the estimated average, rather than a conservative lower bound. A lower-bound sort would favor results with the strongest evidence and answer a different question. The distinction is discussed in [Bayesian ranking methods](https://www.evanmiller.org/bayesian-average-ratings.html).

## Pilot review

Unique pilot counts are currently descriptive; they do not change the score. An October 9 audit found modest predictive value and no demonstrated improvement from the tested pilot-aware deck-only alternatives. The separate pilot-history forecast improved, but it predicts the player plus deck rather than deck strength alone. See the [full audit](deck-performance-audit-2026-10-09.md).

## Historical validation

The initial fetch contains 4,985 events over a year. We selected the 180-day window over a 90-day candidate using three earlier, nonoverlapping 30-day validation periods. Prior fitting uses only each training period. The most recent 30 days, September 7 through October 6, were held out of window selection. The difference between candidate windows was small; this does not establish that 180 days will always be optimal.

The held-out period contains 49 eligible events and 1,567 entries. Lower values are better:

| Forecast | Log loss | Brier score |
|---|---:|---:|
| Raw prior conversion | 0.71279 | 0.20450 |
| Sample-adjusted conversion | 0.58003 | 0.19572 |
| Event cut fraction alone | 0.56539 | 0.19000 |
| Adjusted Performance model | **0.56345** | **0.18903** |

Whole-event bootstrap intervals for the log-loss improvement exclude zero against raw and sample-adjusted conversion. The smaller improvement against event cut fraction alone does **not** exclude zero. Much of the predictive gain therefore comes from event context; this test does not independently establish a precise strength ordering between decks.

Raw probabilities are clipped to `[0.000001, 0.999999]` for log-loss evaluation. Previously unseen decks receive a training-population estimate, not information from the held-out outcome. Event-level resampling preserves within-event dependence but does not remove repeated-pilot confounding. Historical source records may contain corrections unavailable at their original dates. Complete-label event selection is determined from the retrieved database, not a contemporaneous archive of what was known before each event.

The full [validation report](deck-performance-validation.json) includes all folds, calibration bins and event-bootstrap intervals. Model checks and retrospective forecasting are evidence for this implementation, not proof of intrinsic deck strength or causal effects.

We also checked sensitivity for all 515 scored commanders while holding the prior and reference events fixed. Among the top 20 estimates, dropping the most represented pilot's observations changed a score by as much as 9.3 points; dropping the largest contributing event changed a score by as much as 2.9 points. This is another reason to show sample counts and ranges rather than treating exact ranks as settled. The [sensitivity report](deck-performance-sensitivity.json) contains scores and counts, with no player identifiers.

## Updates, failure behavior and reproducibility

The GitHub workflow updates `deck-performance.json` daily at 00:20 UTC. It fetches all pages, runs model/data checks and writes the feed only after a successful build. Fetch errors, insufficient coverage, backward-dated windows and a drop of more than 30% in eligible-event coverage stop publication and retain the previous feed. Source snapshots and exclusions are retained as workflow artifacts for 14 days.

The website keeps scores unchanged when color, category, archetype or search filters narrow the list. Explicit sort parameters in links remain respected. Reset Filters selects Performance. Each card shows Performance, Conversion Rate and Metashare in a compact footer. Selecting that row opens the range, sample counts and data window. If the feed cannot load, Performance displays N/A, its details explain the failure, and the sort falls back to popularity; other filters and sorts continue working. A feed older than seven days is marked delayed in those details.

Implementation: `scripts/ranking/`, numerical requirements in `scripts/ranking/requirements.txt`, tests in `tests/ranking/`, and the Squarespace code block in `site/decks-index.html`. Model version: `conditional-cut-v1`. Each feed includes its parameters, window, source-snapshot hash and exclusions summary. Existing primer statistics, decklist recommendations and the original ranking feed remain separate and unchanged.

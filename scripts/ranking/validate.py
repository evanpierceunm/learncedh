"""Historical base-likelihood comparison, before adoption calibration.
Do not treat eventAdjusted here as a validation of the full v2 score.
"""
import argparse
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from model import (GRID, beta_prior, clean_events, fit, groups, predictive,
                   prior_weights, reference_curve)


def evaluate(events, cutoff, days, stop):
    start = cutoff - timedelta(days=days)
    train = [e for e in events if start.isoformat() <= e['tournamentDate'][:10] < cutoff.isoformat()]
    test = [e for e in events if cutoff.isoformat() <= e['tournamentDate'][:10] < stop.isoformat()]
    grouped, posteriors, sigma = fit(train)
    a, b = beta_prior(grouped)
    fallback = a / (a + b)
    scores = {m: {'n': 0, 'logLossSum': 0., 'brierSum': 0., 'bins': [[0., 0., 0] for _ in range(10)]}
              for m in ['raw', 'sampleAdjusted', 'eventOnly', 'eventAdjusted']}
    event_losses = {}
    prior = np.exp(prior_weights(sigma))
    for key, rows in groups(test, full_field=True).items():
        history = grouped.get(key, [])
        n, k = sum(r['m'] for r in history), sum(r['k'] for r in history)
        raw = k / n if n else fallback
        shrunk = (a + k) / (a + b + n)
        for row in rows:
            adjusted = predictive(posteriors.get(key, prior), row['N'], row['c'], row['m'])
            for name, prob in [('raw', raw), ('sampleAdjusted', shrunk),
                               ('eventOnly', row['c']/row['N']), ('eventAdjusted', adjusted)]:
                p = np.clip(prob, 1e-6, 1 - 1e-6)
                m, y = row['m'], row['k']
                loss = -y * np.log(p) - (m - y) * np.log1p(-p)
                score = scores[name]
                score['n'] += m
                score['logLossSum'] += float(loss)
                score['brierSum'] += float(y * (1-p)**2 + (m-y) * p**2)
                bucket = score['bins'][min(int(p*10), 9)]
                bucket[0] += float(m*p); bucket[1] += y; bucket[2] += m
                event_losses.setdefault(row['event'], {}).setdefault(name, [0., 0])
                event_losses[row['event']][name][0] += float(loss)
                event_losses[row['event']][name][1] += m
    for score in scores.values():
        score['logLoss'] = score.pop('logLossSum') / score['n']
        score['brier'] = score.pop('brierSum') / score['n']
        score['calibration'] = [{'predicted': p/n, 'observed': y/n, 'n': n}
                                for p, y, n in score.pop('bins') if n]
    return {'cutoff': cutoff.isoformat(), 'stopExclusive': stop.isoformat(), 'days': days,
            'trainingEvents': len(train), 'testEvents': len(test), 'sigma': sigma,
            'scores': scores, 'eventLosses': event_losses}


def compare_bootstrap(result):
    # Whole events, not shuffled entrant rows. Repeated pilots remain a limitation.
    rows = list(result['eventLosses'].values())
    rng = np.random.default_rng(20261009)
    comparisons = {}
    for baseline in ['raw', 'sampleAdjusted', 'eventOnly']:
        draws = []
        for _ in range(2000):
            selection = rng.integers(0, len(rows), len(rows))
            numerator = sum(rows[i]['eventAdjusted'][0] - rows[i][baseline][0] for i in selection)
            denominator = sum(rows[i]['eventAdjusted'][1] for i in selection)
            draws.append(numerator / denominator)
        comparisons[baseline] = {'logLossDifference95PercentileInterval': np.quantile(draws, [.025,.975]).tolist()}
    return comparisons


def run(snapshot):
    events, exclusions = clean_events(snapshot)
    end = date.fromisoformat(snapshot['end'])
    holdout = end - timedelta(days=30)
    folds = []
    # Three earlier, nonoverlapping validation months; last month held out.
    for offset in [120, 90, 60]:
        cutoff = end - timedelta(days=offset)
        for days in [90, 180]:
            folds.append(evaluate(events, cutoff, days, cutoff+timedelta(days=30)))
    candidates = {days: sum(f['scores']['eventAdjusted']['logLoss'] * f['scores']['eventAdjusted']['n']
                            for f in folds if f['days'] == days) /
                       sum(f['scores']['eventAdjusted']['n'] for f in folds if f['days'] == days)
                  for days in [90, 180]}
    selected = min(candidates, key=candidates.get)
    final = evaluate(events, holdout, selected, end)
    final['eventBootstrap'] = compare_bootstrap(final)
    passes = all(final['scores']['eventAdjusted'][metric] < final['scores'][baseline][metric]
                 for metric in ['logLoss', 'brier'] for baseline in ['raw','sampleAdjusted'])
    passes = passes and all(final['eventBootstrap'][b]['logLossDifference95PercentileInterval'][1] < 0
                           for b in ['raw', 'sampleAdjusted'])
    for fold in folds + [final]:
        fold.pop('eventLosses')
    return {'scope': 'Historical forecast evaluation, not causal deck strength or exact-rank certification',
            'selectedWindowDays': selected, 'validationLogLoss': candidates,
            'folds': folds, 'heldOut': final, 'releaseGatePassed': bool(passes),
            'limitations': ['Player repetition crosses events; event bootstrap does not remove pilot confounding.',
                           'Current historical database may include corrections unavailable at each historical cutoff.',
                           'Commander archetypes pool different decklists; source format metadata is limited.'],
            'eligibleEvents': len(events), 'excludedEvents': len(exclusions)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    report = run(json.loads(Path(args.snapshot).read_text()))
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['selectedWindowDays','validationLogLoss','releaseGatePassed']},indent=2))

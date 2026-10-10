"""Bayesian deck qualification odds, conditioning on each event's known-label cut counts.

Each deck/event observation follows Fisher's noncentral hypergeometric law:
P(k | N,c,m,theta) proportional to C(m,k) C(N-m,c-k) exp(k*theta).
Other decks form a pooled opponent field. Separate deck fits are not a joint
causal player/deck model. Calibration is frozen and ranges are conditional.
"""
from collections import Counter, defaultdict
from datetime import date
import re

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln, logsumexp

GRID = np.linspace(-5, 5, 1001)
VERSION = 'conditional-cut-adoption-v2'
MIN_COMMANDER_COVERAGE = 0.95
TEST_NAME = re.compile(r'\b(test|tester|testing|mock|dummy|sandbox)\b', re.I)
CUSTOM = re.compile(r'partner with (anyone|anything)|custom rules|house rules', re.I)


def known_commander(deck):
    name = (deck.get('name') or '').strip() if deck else ''
    return bool(deck and deck.get('id') and name and
                name.lower() not in {'unknown', 'unknown commander'})


def clean_events(snapshot):
    if snapshot.get('paginationComplete') is not True:
        raise ValueError('Incomplete snapshot')
    start, end = date.fromisoformat(snapshot['start']), date.fromisoformat(snapshot['end'])
    accepted, excluded, seen = [], [], set()
    for event in snapshot['events']:
        reason = None
        tid = event.get('TID')
        entries = event.get('entries') or []
        n, c = event.get('size'), event.get('topCut')
        try:
            day = date.fromisoformat(event['tournamentDate'][:10])
        except (TypeError, KeyError, ValueError):
            day = None
        if not tid or tid in seen:
            reason = 'missing_or_duplicate_event'
        elif day is None or not start <= day < end:
            reason = 'outside_date_bounds'
        elif TEST_NAME.search(event.get('name') or ''):
            reason = 'test_event_name'
        elif CUSTOM.search((event.get('name') or '') + ' ' + (event.get('editorsNote') or '')):
            reason = 'custom_rules_flag'
        elif type(n) is not int or n < 16 or len(entries) != n:
            reason = 'incomplete_field'
        elif type(c) is not int or not 0 < c < n:
            reason = 'no_valid_cut'
        elif not event.get('swissRounds'):
            reason = 'no_qualification_rounds'
        elif len({e.get('id') for e in entries}) != n or any(not e.get('id') for e in entries):
            reason = 'duplicate_or_missing_entry'
        elif any(type(e.get('standing')) is not int or not 1 <= e['standing'] <= n for e in entries):
            reason = 'invalid_standing'
        elif sum(known_commander(e.get('commander')) for e in entries) / n < MIN_COMMANDER_COVERAGE:
            reason = 'commander_coverage_below_95_percent'
        else:
            qualifiers = [e for e in entries if e['standing'] <= c]
            winners = [e for e in entries if e['standing'] == 1]
            if len(qualifiers) != c or len(winners) != 1:
                reason = 'unreconciled_cut'
            elif not winners[0].get('winsBracket') or any(
                (e.get('winsBracket') or 0) + (e.get('lossesBracket') or 0) < 1
                for e in qualifiers):
                reason = 'unconfirmed_finished_bracket'
            elif any((e.get('winsBracket') or 0) + (e.get('lossesBracket') or 0) > 0
                     for e in entries if e['standing'] > c):
                reason = 'bracket_standing_conflict'
        seen.add(tid)
        if reason:
            excluded.append({'event': tid, 'reason': reason})
        else:
            known = [e for e in entries if known_commander(e.get('commander'))]
            known_cut = sum(e['standing'] <= c for e in known)
            if not 0 < known_cut < len(known):
                excluded.append({'event': tid, 'reason': 'no_known_cut_comparison'})
            else:
                accepted.append(event)
    accepted.sort(key=lambda e: (e['tournamentDate'], e['TID']))
    return accepted, excluded


def groups(events, *, full_field=False):
    result = defaultdict(list)
    for event in events:
        decks = defaultdict(list)
        identified = [e for e in event['entries'] if known_commander(e.get('commander'))]
        N = event['size'] if full_field else len(identified)
        c = event['topCut'] if full_field else sum(e['standing'] <= event['topCut'] for e in identified)
        for entry in event['entries']:
            deck = entry.get('commander') or {}
            if known_commander(deck):
                decks[deck['id']].append(entry)
        for key, entries in decks.items():
            result[key].append({'event': event['TID'], 'date': event['tournamentDate'][:10],
                'name': entries[0]['commander']['name'], 'N': N, 'c': c, 'fieldN': event['size'], 'fieldC': event['topCut'], 'm': len(entries),
                'k': sum(e['standing'] <= event['topCut'] for e in entries),
                'pilots': [e['player']['id'] if e.get('player') else None for e in entries]})
    return dict(result)


def logchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def distribution_terms(N, c, m, grid=GRID):
    j = np.arange(max(0, c - (N - m)), min(c, m) + 1)
    logw = logchoose(m, j) + logchoose(N - m, c - j)
    terms = logw[:, None] + j[:, None] * grid[None, :]
    z = logsumexp(terms, axis=0)
    return j, terms, z


def event_loglik(row):
    _, _, z = distribution_terms(row['N'], row['c'], row['m'])
    ll = row['k'] * GRID - z
    return ll - ll.max()


def likelihoods(grouped):
    return {key: np.sum([event_loglik(r) for r in rows], axis=0)
            for key, rows in grouped.items()}


def prior_weights(sigma):
    logp = -0.5 * (GRID / sigma) ** 2
    return logp - logsumexp(logp)


def fit_sigma(lls):
    values = np.array(list(lls.values()))
    if not len(values):
        raise ValueError('No commander observations')
    def objective(log_sigma):
        return -float(logsumexp(values + prior_weights(np.exp(log_sigma)), axis=1).sum())
    fit = minimize_scalar(objective, bounds=(np.log(0.08), np.log(1.5)), method='bounded')
    if not fit.success:
        raise ValueError('Prior optimization failed')
    return float(np.exp(fit.x))


def posterior(ll, sigma):
    value = ll + prior_weights(sigma)
    return np.exp(value - logsumexp(value))


def fit(events, sigma=None):
    grouped = groups(events)
    lls = likelihoods(grouped)
    sigma = fit_sigma(lls) if sigma is None else sigma
    return grouped, {key: posterior(ll, sigma) for key, ll in lls.items()}, sigma


def predictive(posterior_mass, N, c, m):
    j, terms, z = distribution_terms(N, c, m)
    means = (j[:, None] * np.exp(terms - z)).sum(axis=0) / m
    return float(means @ posterior_mass)


def reference_curve(events):
    # Entry-weighted event opportunities, shared by every commander.
    total = sum(e['size'] for e in events)
    curve = np.zeros(len(GRID))
    base = 0.0
    for event in events:
        q = event['topCut'] / event['size']
        weight = event['size'] / total
        curve += weight * expit(np.log(q / (1 - q)) + GRID)
        base += weight * q
    return 100 * curve / base, base


def quantile(values, mass, p):
    return float(np.interp(p, np.cumsum(mass), values))


def records(events):
    grouped, posteriors, sigma = fit(events)
    curve, baseline = reference_curve(events)
    from calibration import adoption_shift, calibrate, CALIBRATION
    pilot_counts_by_deck = {key: Counter(p for r in rows for p in r['pilots'] if p)
                            for key, rows in grouped.items()}
    shifts = {key: adoption_shift(len(counts)) for key, counts in pilot_counts_by_deck.items()}
    total_known = sum(r['m'] for rows in grouped.values() for r in rows)
    # One shared field baseline, weighted by identified entrants. A neutral
    # field with this adoption mix averages exactly 100; filters cannot change it.
    neutral_baseline = sum(sum(r['m'] for r in rows) * calibrate(baseline, shifts[key])
                           for key, rows in grouped.items()) / total_known
    output = []
    for key, rows in grouped.items():
        mass = posteriors[key]
        n, k = sum(r['m'] for r in rows), sum(r['k'] for r in rows)
        pilot_counts = pilot_counts_by_deck[key]
        pilots = set(pilot_counts)
        missing = sum(p is None for r in rows for p in r['pilots'])
        output.append({'commanderId': key, 'name': rows[0]['name'],
            'score': round(100 * calibrate(float(curve @ mass) * baseline / 100, shifts[key]) / neutral_baseline, 3),
            'interval': [round(100 * calibrate(quantile(curve, mass, q) * baseline / 100, shifts[key]) / neutral_baseline, 2) for q in [0.025, 0.975]],
            'entries': n, 'topCuts': k, 'events': len(rows), 'pilots': len(pilots),
            'missingPilotIds': missing,
            'largestPilotShare': round(max(pilot_counts.values(), default=0) / n, 4),
            'largestEventShare': round(max(r['m'] for r in rows) / n, 4),
            'evidence': 'limited' if len(rows) < 5 or len(pilots) < 5 or missing else 'available'})
    output.sort(key=lambda x: (-x['score'], x['name']))
    return output, {'priorSigma': sigma, 'referenceQualificationRate': baseline,
                    'neutralCalibratedQualificationRate': neutral_baseline,
                    'minimumCommanderCoverage': MIN_COMMANDER_COVERAGE,
                    'adoptionCalibration': CALIBRATION,
                    'gridRange': [-5, 5], 'gridStep': 0.01}


def beta_prior(grouped):
    ns = np.array([sum(r['m'] for r in rs) for rs in grouped.values()])
    ks = np.array([sum(r['k'] for r in rs) for rs in grouped.values()])
    mean = ks.sum() / ns.sum()
    from scipy.special import betaln
    def objective(log_strength):
        a, b = np.exp(log_strength) * mean, np.exp(log_strength) * (1 - mean)
        return -float((betaln(a + ks, b + ns - ks) - betaln(a, b)).sum())
    opt = minimize_scalar(objective, bounds=(np.log(1), np.log(2000)), method='bounded')
    return mean * np.exp(opt.x), (1 - mean) * np.exp(opt.x)

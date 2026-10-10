"""Build the public feed; keep raw entrant IDs outside the public score feed."""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from model import VERSION, clean_events, records, known_commander


def build(snapshot, days=180):
    end = date.fromisoformat(snapshot['end'])
    start = end - timedelta(days=days)
    if date.fromisoformat(snapshot['start']) > start:
        raise ValueError('Snapshot does not cover requested window')
    scoped = dict(snapshot, start=start.isoformat())
    # Historical records outside the scoring window are not data-quality failures.
    scoped['events'] = [e for e in snapshot['events']
                        if (e.get('tournamentDate') or '')[:10] >= start.isoformat()]
    eligible, excluded = clean_events(scoped)
    if len(eligible) < 20:
        raise ValueError('Insufficient eligible events; retain last valid feed')
    data, parameters = records(eligible)
    total = sum(e['size'] for e in eligible)
    known = sum(r['entries'] for r in data)
    if known / total < 0.95:
        raise ValueError('Commander coverage below 95%; retain last valid feed')
    if not data or any(not 0 < r['score'] < 1000 for r in data):
        raise ValueError('Invalid scores')
    payload = {
        'schemaVersion': 1, 'modelVersion': VERSION,
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'source': 'https://edhtop16.com',
        'methodologyUrl': 'https://github.com/evanpierceunm/learncedh/blob/main/docs/deck-performance.md',
        'window': {'start': start.isoformat(), 'endExclusive': end.isoformat(), 'days': days},
        'reference': '100 = qualification opportunity of the shared event field; not a win percentage',
        'intervalScope': '95% conditional model range, holding the fitted prior and adoption calibration fixed; excludes calibration, repeated-pilot and missing-label uncertainty; not a rank interval or pilot-skill-adjusted estimate',
        'parameters': parameters,
        'coverage': {'events': len(eligible), 'entries': total, 'knownCommanderEntries': known,
                     'unknownCommanderEntries': total - known,
                     'commanderCoverage': known / total,
                     'partiallyLabeledEvents': sum(any(not e.get('commander') or not known_commander(e['commander']) for e in event['entries']) for event in eligible),
                     'excludedEvents': len(excluded),
                     'exclusionReasons': dict(Counter(e['reason'] for e in excluded))},
        'sourceSnapshotSha256': hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
        'data': data,
    }
    return payload, excluded


def validate_update(old, feed, days):
    if feed['window']['endExclusive'] < old['window']['endExclusive']:
        raise ValueError('Refusing to replace feed with an older scoring window')
    if old.get('modelVersion') == feed['modelVersion'] and old.get('window', {}).get('days') == days:
        if feed['coverage']['events'] < 0.7 * old['coverage']['events']:
            raise ValueError('Event coverage fell by more than 30%; review before publishing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--audit')
    args = parser.parse_args()
    feed, excluded = build(json.loads(Path(args.snapshot).read_text()), args.days)
    target = Path(args.output)
    if target.exists():
        old = json.loads(target.read_text())
        validate_update(old, feed, args.days)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(feed, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(target)
    if args.audit:
        Path(args.audit).write_text(json.dumps(excluded, indent=2) + '\n')
    print(json.dumps({'decks': len(feed['data']), 'coverage': feed['coverage'],
                      'parameters': feed['parameters']}, indent=2))

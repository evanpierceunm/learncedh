"""Fetch complete date-bounded EDHTop16 event pages, retaining raw responses."""
import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ENDPOINT = 'https://edhtop16.com/api/graphql'
QUERY = '''query($after:String,$start:String!,$end:String!) {
 tournaments(first:50,sortBy:DATE,after:$after,
 filters:{minDate:$start,maxDate:$end,minSize:16}) {
 edges { node { TID name size topCut swissRounds tournamentDate editorsNote
 entries {id standing winsSwiss lossesSwiss winsBracket lossesBracket draws
 commander {id name} player {id}} }} pageInfo {hasNextPage endCursor}
 }}'''


def fetch(directory, start, end):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'endpoint': ENDPOINT, 'query': QUERY, 'start': start, 'end': end}
    path = directory / 'request.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Snapshot directory belongs to another request')
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    after, seen, events, page = None, set(), [], 0
    while True:
        target = directory / f'page-{page:04}.json'
        if target.exists():
            response = json.loads(target.read_text())
        else:
            body = json.dumps({'query': QUERY, 'variables': {
                'after': after, 'start': start, 'end': end}}).encode()
            for attempt in range(4):
                try:
                    req = urllib.request.Request(ENDPOINT, data=body,
                        headers={'Content-Type': 'application/json',
                                 'User-Agent': 'LearnCEDH-performance/1.0'})
                    with urllib.request.urlopen(req, timeout=90) as stream:
                        response = json.load(stream)
                    if response.get('errors'):
                        raise ValueError(str(response['errors']))
                    break
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            target.write_text(json.dumps(response) + '\n')
        if response.get('errors'):
            raise ValueError(str(response['errors']))
        connection = response['data']['tournaments']
        nodes = [edge['node'] for edge in connection['edges']]
        if not nodes and connection['pageInfo']['hasNextPage']:
            raise ValueError('Empty nonterminal page')
        events.extend(nodes)
        info = connection['pageInfo']
        print(f'page={page} events={len(events)}', flush=True)
        if not info['hasNextPage']:
            break
        after = info['endCursor']
        if not after or after in seen:
            raise ValueError('Pagination did not advance')
        seen.add(after)
        page += 1
        time.sleep(0.15)
    result = dict(manifest, retrievedAt=datetime.now(timezone.utc).isoformat(),
                  paginationComplete=True, events=events)
    (directory / 'events.json').write_text(json.dumps(result) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--end', default=(datetime.now(timezone.utc).date() - timedelta(days=2)).isoformat())
    parser.add_argument('--days', type=int, default=365)
    args = parser.parse_args()
    start = (date.fromisoformat(args.end) - timedelta(days=args.days)).isoformat()
    fetch(args.output, start, args.end)

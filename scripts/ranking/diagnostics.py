"""Sensitivity to the most represented pilot/event, with the fitted prior fixed."""
import argparse
from collections import Counter
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from model import GRID, clean_events, event_loglik, fit, posterior, reference_curve, records


def run(snapshot, days):
    scoped = dict(snapshot, start=(date.fromisoformat(snapshot['end'])-timedelta(days=days)).isoformat())
    events, _ = clean_events(scoped)
    grouped, posts, sigma = fit(events)
    by_event = {e['TID']: e for e in events}
    curve, base = reference_curve(events)
    from calibration import adoption_shift, calibrate
    release, parameters = records(events)
    baseline = parameters['neutralCalibratedQualificationRate']
    def scored(mass, pilots):
        return 100 * calibrate(float(curve @ mass) * base / 100, adoption_shift(pilots)) / baseline
    output=[]
    for key, rows in grouped.items():
        pilots=Counter(p for r in rows for p in r['pilots'] if p)
        leading=pilots.most_common(1)[0][0] if pilots else None
        biggest=max(rows,key=lambda r:r['m'])['event']
        without_pilot=np.zeros(len(GRID));without_event=np.zeros(len(GRID))
        for row in rows:
            if row['event']!=biggest:without_event+=event_loglik(row)
            es=[e for e in by_event[row['event']]['entries']
                if (e.get('commander') or {}).get('id')==key and (e.get('player') or {}).get('id')!=leading]
            if es:
                adjusted=dict(row,m=len(es),k=sum(e['standing']<=row['fieldC'] for e in es))
                without_pilot+=event_loglik(adjusted)
        score=scored(posts[key],len(pilots))
        event_pilots={p for r in rows if r['event']!=biggest for p in r['pilots'] if p}
        output.append({'name':rows[0]['name'],'score':score,
            'entries':sum(r['m'] for r in rows),'largestPilotEntries':max(pilots.values(),default=0),
            'scoreWithoutLeadingPilot':scored(posterior(without_pilot,sigma),len(pilots)-(1 if leading else 0)),
            'scoreWithoutLargestEvent':scored(posterior(without_event,sigma),len(event_pilots))})
    output.sort(key=lambda r:-r['score'])
    return {'scope':'Sensitivity, not pilot-adjusted strength; reference events, calibration and empirical prior held fixed; omits selected labelled evidence while retaining its field context.',
            'windowDays':days,'data':output}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True)
    p.add_argument('--output',required=True);p.add_argument('--days',type=int,default=180)
    a=p.parse_args();report=run(json.loads(Path(a.snapshot).read_text()),a.days)
    Path(a.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Sensitivity checked for',len(report['data']),'commanders')

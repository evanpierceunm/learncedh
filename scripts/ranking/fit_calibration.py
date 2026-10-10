"""Reproduce adoption coefficients from three earlier chronological forecast folds.
This is an explicit research command, never run automatically by the daily build.
"""
import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from model import clean_events, fit, groups, predictive, prior_weights


def train(snapshot, evaluation_end):
    end = date.fromisoformat(evaluation_end)
    if date.fromisoformat(snapshot['start']) > end-timedelta(days=300) or date.fromisoformat(snapshot['end']) < end:
        raise ValueError('Need 300 days of source coverage through evaluation end')
    events, _ = clean_events(snapshot)
    rows=[]
    for offset in [120,90,60]:
        cutoff=end-timedelta(days=offset)
        training=[e for e in events if (cutoff-timedelta(days=180)).isoformat() <= e['tournamentDate'][:10] < cutoff.isoformat()]
        validation=[e for e in events if cutoff.isoformat() <= e['tournamentDate'][:10] < (cutoff+timedelta(days=30)).isoformat()]
        g,p,sigma=fit(training); prior=np.exp(prior_weights(sigma))
        pilots={k:len({p for row in rs for p in row['pilots'] if p}) for k,rs in g.items()}
        for key,rs in groups(validation,full_field=True).items():
            for r in rs:
                prediction=predictive(p.get(key,prior),r['N'],r['c'],r['m'])
                rows.append((r['m'],r['k'],np.log1p(pilots.get(key,0)),prediction))
    if not rows:raise ValueError('No validation observations')
    n,k,x,p=np.array(rows).T
    center=float(np.average(x,weights=n));X=np.c_[np.ones(len(n)),x-center]
    offset=logit(np.clip(p,1e-6,1-1e-6))
    def objective(b):
        z=offset+X@b
        return np.sum(n*np.logaddexp(0,z)-k*z)+.5*np.dot(b,b), X.T@(n*expit(z)-k)+b
    result=minimize(objective,np.zeros(2),jac=True,method='BFGS',options={'gtol':1e-6})
    if not result.success and np.max(abs(result.jac))>=1e-5:raise ValueError(result.message)
    return {'intercept':float(result.x[0]),'logPilotCoefficient':float(result.x[1]),
            'logPilotCenter':center,'trainingValidationEntries':int(n.sum()),
            'trainingValidationStart':(end-timedelta(days=120)).isoformat(),
            'trainingValidationEndExclusive':(end-timedelta(days=30)).isoformat()}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True)
    parser.add_argument('--evaluation-end',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=train(json.loads(Path(args.snapshot).read_text()),args.evaluation_end)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

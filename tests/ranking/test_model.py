import copy
from datetime import date
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import nchypergeom_fisher

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'ranking'))
from model import (GRID, clean_events, distribution_terms, event_loglik,
                   groups, posterior, predictive, records)


def event(tid='event', day='2026-09-01', n=16, cut=4):
    return {'TID': tid, 'name': 'Tournament', 'size': n, 'topCut': cut,
            'swissRounds': 3, 'tournamentDate': day+'T12:00:00Z',
            'entries': [{'id': str(i), 'standing': i+1,
                         'winsBracket': int(i==0), 'lossesBracket': int(0<i<cut),
                         'commander': {'id':'deck'+str(i%4), 'name':'Deck '+str(i%4)},
                         'player': {'id':'player'+str(i)}} for i in range(n)]}


def snapshot(events):
    return {'events': events, 'start':'2026-07-01','end':'2026-10-01','paginationComplete':True}


class DataTests(unittest.TestCase):
    def test_complete_event(self):
        good, bad = clean_events(snapshot([event()]))
        self.assertEqual((len(good), len(bad)), (1,0))

    def test_future_no_cut_and_test_event(self):
        a,b,c=event('future','2030-01-01'),event('no-cut',cut=0),event('test')
        c['name']='Tester'
        good,bad=clean_events(snapshot([a,b,c]))
        self.assertFalse(good)
        self.assertEqual({r['reason'] for r in bad}, {'outside_date_bounds','no_valid_cut','test_event_name'})

    def test_missing_entries_and_unfinished_winner(self):
        a,b=event('missing'),event('unfinished')
        a['entries'].pop();b['entries'][0]['winsBracket']=0
        _,bad=clean_events(snapshot([a,b]))
        self.assertEqual({r['reason'] for r in bad},{'incomplete_field','unconfirmed_finished_bracket'})

    def test_cut_capacity_and_conflicting_bracket(self):
        a,b=event('cut'),event('conflict')
        a['entries'][4]['standing']=4;b['entries'][4]['winsBracket']=1
        _,bad=clean_events(snapshot([a,b]))
        self.assertEqual({r['reason'] for r in bad},{'unreconciled_cut','bracket_standing_conflict'})

    def test_duplicate_snapshot_and_exclusive_end(self):
        good,bad=clean_events(snapshot([event(),event(),event('end','2026-10-01')]))
        self.assertEqual(len(good),1);self.assertEqual(len(bad),2)

    def test_incomplete_pagination_fails(self):
        s=snapshot([]);s['paginationComplete']=False
        with self.assertRaises(ValueError):clean_events(s)

    def test_unknown_commander_is_not_a_ranked_deck(self):
        e=event();e['entries'][0]['commander']={'id':'unknown','name':'Unknown Commander'}
        grouped=groups([e])
        self.assertNotIn('unknown',grouped)
        self.assertEqual(sum(r['m'] for rs in grouped.values() for r in rs),15)
        self.assertTrue(all(r['N']==15 and r['c']==3 for rs in grouped.values() for r in rs))

    def test_blank_labels_exclude_entire_event(self):
        e=event();e['entries'][0]['commander']['name']=''
        good,bad=clean_events(snapshot([e]))
        self.assertFalse(good)
        self.assertEqual(bad[0]['reason'],'commander_coverage_below_95_percent')

    def test_95_percent_boundary_and_full_completion_guards(self):
        a,b,c=event('boundary',n=20),event('too-missing',n=20),event('bad-bracket',n=20)
        for e in [a,b,c]:e['entries'][0]['commander']=None
        b['entries'][1]['commander']=None
        c['entries'][0]['winsBracket']=0
        good,bad=clean_events(snapshot([a,b,c]))
        self.assertEqual([e['TID'] for e in good],['boundary'])
        self.assertEqual({r['reason'] for r in bad},{'commander_coverage_below_95_percent','unconfirmed_finished_bracket'})
        rs=[r for rows in groups(good).values() for r in rows]
        self.assertEqual(sum(r['m'] for r in rs),19)
        self.assertEqual(sum(r['k'] for r in rs),3)
        self.assertTrue(all(r['N']==19 and r['c']==3 for r in rs))
        full=[r for rows in groups(good,full_field=True).values() for r in rows]
        self.assertTrue(all(r['N']==20 and r['c']==4 for r in full))

    def test_missing_nonqualifier_is_not_an_opponent(self):
        e=event(n=20);e['entries'][-1]['commander']=None
        rs=[r for rows in groups([e]).values() for r in rows]
        self.assertEqual(sum(r['m'] for r in rs),19)
        self.assertTrue(all(r['N']==19 and r['c']==4 for r in rs))

    def test_complete_labels_preserve_likelihood_inputs(self):
        self.assertEqual(groups([event()]),groups([event()],full_field=True))

    def test_no_known_cut_variation_rejected(self):
        e=event(n=20,cut=1);e['entries'][0]['commander']=None
        good,bad=clean_events(snapshot([e]))
        self.assertFalse(good)
        self.assertEqual(bad[0]['reason'],'no_known_cut_comparison')


class StatisticalTests(unittest.TestCase):
    def test_distribution_against_independent_scipy_implementation(self):
        for N,c,m,theta in [(16,4,3,.7),(120,16,30,-.4),(20,16,8,1.2)]:
            j, terms, z = distribution_terms(N,c,m,np.array([theta]))
            actual=np.exp(terms[:,0]-z[0])
            expected=nchypergeom_fisher.pmf(j,N,m,c,np.exp(theta))
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)

    def test_neutral_expectation_and_fixed_capacity(self):
        j,terms,z=distribution_terms(32,4,10,np.array([0.]))
        probabilities=np.exp(terms[:,0]-z[0])
        self.assertAlmostEqual(float(j@probabilities),10*4/32)
        self.assertLessEqual(j.max(),4)

    def test_single_entrant_matches_logistic(self):
        mass=np.zeros(len(GRID));mass[np.argmin(abs(GRID-.5))]=1
        p=predictive(mass,64,16,1)
        self.assertAlmostEqual(p,1/(1+np.exp(-(np.log(1/3)+.5))))

    def test_same_results_at_harder_cut_get_higher_estimate(self):
        easy={'N':32,'c':16,'m':4,'k':2}
        hard=dict(easy,c=4)
        pe,ph=posterior(event_loglik(easy),.5),posterior(event_loglik(hard),.5)
        self.assertGreater(float(GRID@ph),float(GRID@pe))

    def test_small_sample_shrinks_and_strong_repetition_narrows(self):
        row={'N':64,'c':16,'m':2,'k':1}
        small=posterior(event_loglik(row),.5)
        large=posterior(100*event_loglik(row),.5)
        self.assertGreater(float(GRID@large),float(GRID@small))
        def variance(p):return float((GRID**2)@p-(GRID@p)**2)
        self.assertLess(variance(large),variance(small))

    def test_labels_and_counts_are_preserved(self):
        rows,params=records([event()])
        self.assertEqual(sum(r['entries'] for r in rows),16)
        self.assertEqual(sum(r['topCuts'] for r in rows),4)
        self.assertTrue(all(r['evidence']=='limited' for r in rows))
        self.assertTrue(all(r['interval'][0]<r['score']<r['interval'][1] for r in rows))

    def test_shared_neutral_field_baseline(self):
        from calibration import adoption_shift,calibrate
        e=event(n=20);e['entries'][1]['player']=e['entries'][5]['player']
        rs,params=records([e]);base=params['referenceQualificationRate']
        neutral=sum(r['entries']*calibrate(base,adoption_shift(r['pilots'])) for r in rs)/sum(r['entries'] for r in rs)
        self.assertAlmostEqual(neutral,params['neutralCalibratedQualificationRate'])
        self.assertAlmostEqual(sum(r['entries']*100*calibrate(base,adoption_shift(r['pilots']))/neutral for r in rs)/20,100)

    def test_pilot_feature_deduplicates_and_handles_missing_ids(self):
        e=event();e['entries'][4]['player']=e['entries'][0]['player'];e['entries'][8]['player']=None
        rs,_=records([e]);r=next(r for r in rs if r['commanderId']=='deck0')
        self.assertEqual(r['pilots'],2);self.assertEqual(r['missingPilotIds'],1)
        self.assertEqual(r['evidence'],'limited')

    def test_calibration_monotonic_and_bounded(self):
        from calibration import adoption_shift,calibrate
        self.assertGreater(adoption_shift(500),adoption_shift(50))
        self.assertGreater(calibrate(.4,adoption_shift(50)),calibrate(.2,adoption_shift(50)))
        self.assertTrue(0<calibrate(.25,adoption_shift(0))<1)
        for bad in [-1,True,2.5]:
            with self.assertRaises(ValueError):adoption_shift(bad)
        for bad in [0,1,float('nan')]:
            with self.assertRaises(ValueError):calibrate(bad,0)


class FeedTests(unittest.TestCase):
    def test_public_counts_unknowns_and_conditional_range(self):
        from build import build
        import json
        events=[event(str(i),n=20) for i in range(20)]
        for e in events:e['entries'][0]['commander']=None
        feed,_=build(snapshot(events),90)
        self.assertEqual(feed['coverage']['entries'],400)
        self.assertEqual(feed['coverage']['knownCommanderEntries'],380)
        self.assertEqual(feed['coverage']['unknownCommanderEntries'],20)
        self.assertEqual(feed['coverage']['partiallyLabeledEvents'],20)
        self.assertIn('calibration',feed['intervalScope'])
        self.assertNotIn('player0',json.dumps(feed))
        self.assertTrue(all(r['interval'][0]<r['score']<r['interval'][1] for r in feed['data']))

    def test_rollback_and_coverage_guards(self):
        from build import validate_update
        old={'modelVersion':'v1','window':{'endExclusive':'2026-10-08','days':180},'coverage':{'events':400}}
        new=copy.deepcopy(old);new['modelVersion']='v2';new['window']['endExclusive']='2026-10-07'
        with self.assertRaises(ValueError):validate_update(old,new,180)
        new=copy.deepcopy(old);new['coverage']['events']=279
        with self.assertRaises(ValueError):validate_update(old,new,180)
        new['coverage']['events']=280;validate_update(old,new,180)


if __name__=='__main__':unittest.main()

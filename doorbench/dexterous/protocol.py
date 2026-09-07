"""Versioned scope inventory and conservative door-level coverage scoring."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

VERSION = 'doorbench.dexterous-sim.v1-draft'


def inventory(assets_root):
    rows=[]
    for path in sorted((Path(assets_root)/'doors').glob('*/spec.json')):
        raw=path.read_bytes();spec=json.loads(raw)
        scenarios=spec.get('benchmark',{}).get('scenarios',[])
        excluded=spec['family']=='pet_door'
        rows.append({'door_id':spec['id'],'family':spec['family'],
            'spec_sha256':hashlib.sha256(raw).hexdigest(),
            'collection':'supplementary_pet' if excluded else 'robotics',
            'human_feasibility':'excluded_pet' if excluded else 'pending_independent_review',
            'required_opening_scenarios':[s['name'] for s in scenarios
                if s['name'] in ('open_and_traverse','open_then_close','unlock_and_traverse')],
            'other_scenarios':[s['name'] for s in scenarios
                if s['name'] not in ('open_and_traverse','open_then_close','unlock_and_traverse')],
            'operator':spec.get('operator'), 'tags':spec.get('tags',[]),
            'review_note':'Scenario names are candidates only; they do not certify human feasibility.'})
    return {'protocol':VERSION,'status':'scope draft; not a frozen benchmark',
        'doors_total':len(rows),'robotics_collection':sum(r['collection']=='robotics' for r in rows),
        'families':dict(sorted(Counter(r['family'] for r in rows).items())),
        'human_feasibility_review_complete':False,'rows':rows}


def score(required, trials, *, seeds=10, minimum_fraction=.9):
    """required maps each frozen-scope door to its opening scenarios.

    Missing trials remain failures. Only explicitly physically valid successes
    count. A case cannot disappear because the policy did not attempt it.
    """
    if seeds<1 or not 0<minimum_fraction<=1:
        raise ValueError('Invalid reliability criterion')
    evidence={}
    for row in trials:
        key=(row['door_id'],row['scenario'],row['seed'])
        if key in evidence:
            raise ValueError('Duplicate evaluation case')
        if row['door_id'] not in required or row['scenario'] not in required[row['door_id']] or not 0<=row['seed']<seeds:
            raise ValueError('Trial outside frozen protocol')
        evidence[key]=row.get('success') is True and row.get('physically_valid') is True
    threshold=math.ceil(seeds*minimum_fraction);rows=[]
    for door,scenarios in sorted(required.items()):
        if not scenarios or len(set(scenarios))!=len(scenarios):
            raise ValueError('Each target door needs distinct declared scenarios')
        counts={s:sum(evidence.get((door,s,k),False) for k in range(seeds)) for s in scenarios}
        rows.append({'door_id':door,'successful_trials':counts,'reliably_solved':all(n>=threshold for n in counts.values())})
    solved=sum(r['reliably_solved'] for r in rows)
    return {'target_doors':len(required),'reliably_solved':solved,
            'coverage':solved/len(required) if required else None,
            'expected_trials':sum(len(x) for x in required.values())*seeds,
            'recorded_trials':len(evidence),'successful_trials':sum(evidence.values()),
            'minimum_successes_per_scenario':threshold,'rows':rows}

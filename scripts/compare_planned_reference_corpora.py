#!/usr/bin/env python3
"""Compare complete motion runs without conflating pending cases or changed gates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.build_planned_reference_corpus import identity
from scripts.export_planned_reference_web import verified_artifacts

ACCEPTED='accepted_kinematic'
SOURCES=('manifest_sha256','recording_index_sha256','recording_sha256','source_sha256','native_resources_sha256')
GATE_FILES=('doorbench/reference/rig.py','scripts/validate_planned_reference.py')


def require(ok,message):
    if not ok: raise ValueError(message)


def digest(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())


def load_complete(root):
    root=Path(root).resolve(); raw=(root/'index.json').read_bytes(); index=json.loads(raw)
    require(index.get('schema')=='doorbench.planned-reference-corpus.v1','Unsupported corpus schema')
    report=read(root/'report.json')
    require(report.get('snapshot_id')==index['snapshot_id'],'Corpus index/report snapshot mismatch')
    rows=index['doors']; ids=[r['door_id'] for r in rows]
    require(ids and len(set(ids))==len(ids) and set(ids)==set(index['selected_ids']),'Corpus IDs are incomplete or duplicated')
    require(all(r.get('action') in ('completed','resume') and r.get('result')==r['door_id']+'/result.json' for r in rows),
            'Comparison requires complete results; pending attempts are not unresolved outcomes')
    results={}; gates=set()
    for row in rows:
        door=row['door_id']; require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',door),'Unsafe door ID')
        directory=root/door; require(not directory.is_symlink(),'Attempt directory is a symlink')
        result_raw=(directory/'result.json').read_bytes(); result=json.loads(result_raw)
        provenance=result['provenance']; status=result['status']
        require(status in (ACCEPTED,'rejected','unresolved') and status==row['status'] and
                result['door_id']==door and identity(provenance)==result['identity_sha256']==row['identity_sha256'],
                'Result identity/status differs from its corpus index')
        require(provenance['generator_sha256']==index['generator']['sha256'] and
                provenance['manifest_sha256']==index['manifest_sha256'],'Attempt belongs to another generator or dataset')
        require(all(key in provenance for key in SOURCES),'Missing native source bindings')
        files=verified_artifacts(directory,result)
        record={'door_id':door,'family':row['family'],'status':status,'scenario':result['source_outcome']['scenario'],
                'source':{k:provenance[k] for k in SOURCES},'source_outcome':result['source_outcome'],
                'result_sha256':digest(result_raw),'reason_code':result.get('reason_code'),
                'failure_counts':result.get('failure_counts') or {},'duration_s':None}
        if status==ACCEPTED:
            require('failure.json' not in files and {'clip.json','trajectory.npz','validation.json'}<=files.keys(),
                    'Accepted result lacks complete motion artifacts')
            clip=json.loads(files['clip.json']); audit=json.loads(files['validation.json'])
            require(clip['door_id']==audit['door_id']==door and clip.get('complete_proposal') is True and
                    audit.get('accepted') is True and audit.get('kinematic_accepted') is True and
                    audit.get('status')==ACCEPTED and not audit.get('failure_counts') and
                    audit.get('task_completion',{}).get('evidence_pass') is True and
                    audit.get('task_completion',{}).get('complete_proposal') is True and
                    all(result.get('new_completion',{}).get(k) is True for k in
                        ('complete_proposal','artifact_bindings_verified','task_evidence_pass','source_success_declared')),
                    'Claimed accepted result did not pass independent complete task checks')
            require(audit['clip_sha256']==digest(files['clip.json']) and
                    audit['trajectory_sha256']==clip['trajectory_sha256']==digest(files['trajectory.npz']),
                    'Independent acceptance binds different motion bytes')
            require(clip['source_sha256']==audit['source_sha256']==provenance['source_sha256'] and
                    clip['proposal']['source_outcome']==result['source_outcome'] and
                    result['source_outcome'].get('success') is True and result['source_outcome'].get('outcome')=='success' and
                    not result['source_outcome'].get('error'),
                    'Accepted motion source outcome or geometry binding differs')
            require(clip['proposal']['scenario']==record['scenario']==audit['task_completion']['scenario'],
                    'Accepted source task differs from checked task')
            record['duration_s']=float(clip['duration']); gates.add(identity(audit['settings']))
        results[door]=record
    counts=dict(Counter(r['status'] for r in results.values()))
    require(counts==report['status_counts'],'Corpus report counts differ from checked results')
    require(len(gates)<=1,'Corpus mixes independent validation thresholds')
    require((root/'index.json').read_bytes()==raw,'Corpus snapshot changed while comparing')
    return {'root':root,'index':index,'index_sha256':digest(raw),'results':results,'counts':counts,'gates':gates}


def compare(before_path,after_path):
    before=load_complete(before_path); after=load_complete(after_path)
    require(before['results'].keys()==after['results'].keys(),'Corpora contain different door sets')
    for name in GATE_FILES:
        old=before['index']['generator']['files'].get(name); new=after['index']['generator']['files'].get(name)
        require(old is not None and old==new,'Comparison requires the same rig and independent validator: '+name)
    require(not before['gates'] or not after['gates'] or before['gates']==after['gates'],
            'Independent validation thresholds changed between runs')
    transitions=Counter(); changed=[]; durations=[]; families={}
    for door in sorted(before['results']):
        old=before['results'][door]; new=after['results'][door]
        require(old['source']==new['source'] and old['source_outcome']==new['source_outcome'] and
                old['scenario']==new['scenario'] and old['family']==new['family'],
                door+': native data or source task changed; this is not a planner-only comparison')
        key=old['status']+' -> '+new['status']; transitions[key]+=1
        families.setdefault(old['family'],Counter())[key]+=1
        if old['status']!=new['status']:
            changed.append({'door_id':door,'family':old['family'],'scenario':old['scenario'],
                            'before':old['status'],'after':new['status'],
                            'before_reason':old['reason_code'],'after_reason':new['reason_code'],
                            'before_failures':old['failure_counts'],'after_failures':new['failure_counts']})
        if old['status']==new['status']==ACCEPTED:
            durations.append({'door_id':door,'scenario':old['scenario'],'before_s':old['duration_s'],
                              'after_s':new['duration_s'],'change_s':new['duration_s']-old['duration_s']})
    def summary(run):
        accepted=[r for r in run['results'].values() if r['status']==ACCEPTED]
        return {'index_sha256':run['index_sha256'],'generator_sha256':run['index']['generator']['sha256'],
                'snapshot_id':run['index']['snapshot_id'],'counts':run['counts'],
                'accepted_scenarios':dict(Counter(r['scenario'] for r in accepted))}
    # Recheck both snapshots after both complete inventories were read.
    for run in (before,after):
        require(digest((run['root']/'index.json').read_bytes())==run['index_sha256'],'Corpus snapshot changed while comparing')
    return {'schema':'doorbench.planned-reference-comparison.v1','doors':len(before['results']),
            'scope':'Hash-bound complete-result comparison with identical native inputs, original rig and independent gates. No new physics or visual acceptance is inferred.',
            'before':summary(before),'after':summary(after),'transitions':dict(transitions),
            'newly_accepted':[r['door_id'] for r in changed if r['after']==ACCEPTED],
            'lost_acceptance':[r['door_id'] for r in changed if r['before']==ACCEPTED],
            'changed_statuses':changed,'family_transitions':{k:dict(v) for k,v in families.items()},
            'common_accepted_durations':durations}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',required=True);parser.add_argument('--after',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(); target=Path(args.out).resolve()
    require(all(not target.is_relative_to(Path(p).resolve()) for p in (args.before,args.after)),
            'Comparison output must be outside both input corpora')
    result=compare(args.before,args.after);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('doors','before','after','newly_accepted','lost_acceptance')}))


if __name__=='__main__':main()

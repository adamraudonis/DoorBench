#!/usr/bin/env python3
"""Audit actual external support contacts during upright panel continuation."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.json_record_stream import iter_json_object_array


def allowed_contact(bodies,geoms):
    robot=[i for i,n in enumerate(bodies) if n.startswith('robot/')]
    if len(robot)!=1:return True
    i=robot[0];other=1-i;name=bodies[i]
    if geoms[other]=='floor' and name in ('robot/left_ankle_link','robot/right_ankle_link'):return True
    return bodies[other]=='leaf' and (name=='robot/lh_palm' or name.startswith(tuple('robot/lh_'+digit for digit in ('ff','mf','rf','lf','th'))))


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.trial
    report=json.loads((b/'report.json').read_text());manifest=json.loads((b/'manifest.json').read_text());archive=json.loads((b/'raw-transitions/manifest.json').read_text())
    if not archive['complete']:raise ValueError('Require a complete physical archive')
    start=report['standing_panel']['started_s']
    if start is None:raise ValueError('No actual panel phase to audit')
    cfg=manifest['configuration'];robot=Path(cfg['robot']);door=Path(cfg['door'])
    if sha(robot)!=manifest['inputs']['robot']['sha256'] or sha(door/'door.xml')!=manifest['inputs']['door']['door.xml']:raise ValueError('Recorded model inputs changed')
    from doorbench.dexterous.environment import DexterousDoorEnv
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));m=sim.m
    bad=[];intervals=0;loads={};minimum_clearance=float('inf');clearance_intervals=0
    try:
        for chunk in archive['chunks']:
            if chunk['interval_end_s']<start:continue
            path=b/'raw-transitions'/chunk['file']
            if sha(path)!=chunk['sha256']:raise ValueError('Actual contact archive changed')
            with np.load(path) as z:
                for i,t in enumerate(z['interval_start_s']):
                    if t<start-1e-8:continue
                    intervals+=1;s,e=z['contact_offsets'][i:i+2]
                    for k in range(s,e):
                        force=float(z['contact_wrench_contact_frame'][k,0])
                        if force<=.1:continue
                        bodies=[m.body(int(j)).name for j in z['contact_body'][k]];geoms=[m.geom(int(j)).name for j in z['contact_geom'][k]]
                        label=' / '.join(bodies);loads[label]=max(loads.get(label,0.),force)
                        if not allowed_contact(bodies,geoms):bad.append(dict(time_s=float(t),bodies=bodies,geoms=geoms,normal_load_N=force))
        with gzip.open(b/'physics-steps.json.gz','rt') as f:
            for r in iter_json_object_array(f):
                if r['sim_time_s']>start+1e-8:
                    minimum_clearance=min(minimum_clearance,r['right_environment_clearance_m']);clearance_intervals+=1
    finally:sim.close()
    expected=round((report['expected_duration_s']-start)/report['physics_dt_s'])
    checks=dict(physical_report_passed=report['passed'],complete_panel_intervals=intervals==expected and clearance_intervals==expected,only_declared_external_support=not bad,right_hand_clear_throughout_panel=minimum_clearance>=.04)
    result=dict(passed=all(checks.values()),checks=checks,panel_start_s=start,intervals=intervals,expected_intervals=expected,minimum_right_clearance_m=minimum_clearance,loaded_contact_threshold_N=.1,unexpected_contacts=bad,maximum_contact_loads_N=loads,scope='Independent actual archived contact forces and recorded clearance; no stepping or recomputed contact forces. Allows feet/floor and left palm/fingers/leaf only for external robot support.',input_sha256={str(p.relative_to(b)):sha(p) for p in (b/'report.json',b/'manifest.json',b/'raw-transitions/manifest.json',b/'physics-steps.json.gz')},auditor_sha256=sha(Path(__file__)))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('unexpected_contacts','maximum_contact_loads_N','input_sha256')},indent=2))


if __name__=='__main__':main()

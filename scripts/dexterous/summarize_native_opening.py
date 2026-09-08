#!/usr/bin/env python3
"""Summarize preserved every-step opening evidence without changing its score."""
import argparse
import gzip
import json
from pathlib import Path


def summarize(rows,*,hold_seconds=.5,palm_floor_N=2.):
    if not rows:raise ValueError('Missing physics evidence')
    final=rows[-1];tail=[r for r in rows if r['sim_time_s']>=final['sim_time_s']-hold_seconds-1e-8]
    loads=sorted(r.get('left_surface_audit',{}).get('palm_normal_load_N',0.) for r in tail)
    contacts=[r for r in rows if r.get('contact_force_source')=='actual_mj_step_dynamics']
    return dict(schema='doorbench.native-opening-summary.v1',steps=len(rows),
        actual_transition_samples=len(contacts),duration_s=final['sim_time_s'],
        final_leaf_rad=final['door_q'],final_palm_load_N=final.get('left_surface_audit',{}).get('palm_normal_load_N',0.),
        final_hold_seconds=hold_seconds,final_hold_samples=len(tail),
        final_hold_palm_below_floor_samples=sum(f<palm_floor_N for f in loads),
        final_hold_palm_min_N=min(loads),final_hold_palm_median_N=loads[len(loads)//2],
        max_joint_limit_violation_rad=max(r['max_joint_limit_violation_rad'] for r in rows),
        joint_limit_violation_samples=sum(r['max_joint_limit_violation_rad']>.02 for r in rows),
        max_shadow_loopback_violation_rad=max(r['max_shadow_loopback_violation_rad'] for r in rows),
        max_nonfoot_penetration_m=max(r['max_nonfoot_penetration_m'] for r in rows),
        max_torso_tilt_deg=max(r['torso_tilt_deg'] for r in rows),
        final_contact_interval_s=[final.get('contact_interval_start_s'),final.get('contact_interval_end_s')],
        scope='Derived summary only; original frozen report and every-step evidence remain authoritative')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    with gzip.open(a.run/'physics-steps.json.gz','rt') as stream:result=summarize(json.load(stream))
    (a.run/'transition-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Recompute grasp qualification from actual-step forces and matching body frames.

This read-only evaluator does not trust archived pad labels. It never feeds
scene state, contact IDs or success labels to the sensor-based controller.
"""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import pad_opposition, shadow_surface_qualified


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_pad_evidence(model,raw,lever):
    contacts=[c for c in raw['contacts'] if lever in c['geom']]
    if not contacts:return pad_opposition([],np.zeros(3),np.array([1.,0,0])),[]
    poses=dict(zip(raw['body_ids'],zip(raw['body_positions_world_m'],raw['body_rotations_world'])))
    body=int(model.geom_bodyid[lever]);p,R=poses[body];R=np.asarray(R);p=np.asarray(p)
    local=np.zeros(9);mujoco.mju_quat2Mat(local,model.geom_quat[lever])
    center=p+R@model.geom_pos[lever];axis=(R@local.reshape(3,3))[:,2]
    patches=[]
    for c in contacts:
        other=c['geom'][1] if c['geom'][0]==lever else c['geom'][0]
        b=int(model.geom_bodyid[other]);name=model.body(b).name
        match=re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(.+)',name)
        if not match:continue
        position=np.asarray(c['position_world_m']);bp,br=poses[b];br=np.asarray(br)
        local_point=br.T@(position-bp)
        outward=br.T@((1 if c['geom'][0]==other else -1)*np.asarray(c['frame_world'])[0])
        relative=position-center;axial=float(relative@axis);radial=relative-axial*axis
        alignment=float((br@outward)@(-radial/np.linalg.norm(radial))) if np.linalg.norm(radial)>1e-8 else 0.
        margin=float(model.geom_size[lever,1]-abs(axial))
        pad=bool(shadow_surface_qualified(match.group(1),match.group(2),local_point,outward) and margin>=.001 and alignment>.8)
        patches.append(dict(digit=match.group(1),position=position.tolist(),normal_force_N=max(0.,float(c['wrench_contact_frame'][0])),pad_qualified=pad))
    return pad_opposition(patches,center,axis),patches


def audit(trial):
    trial=Path(trial);prov=json.loads((trial/'provenance.json').read_text());report=json.loads((trial/'report.json').read_text())
    robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door']);door=door if door.is_dir() else door.parent
    if sha(robot)!=prov['robot_xml_sha256'] or sha(door/'door.xml')!=prov['door_xml_sha256']:raise ValueError('Actual source asset bytes changed')
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()),frame_skip=1)
    m=sim.m;lever=m.geom('leaf_handle_lever_col_n').id
    n=valid_count=current=best=0;force_error=0.;mismatches=[];bad_patches=0;first_touch=first_valid=None;bestend=None
    min_final={k:float('inf') for k in ('ff','mf','rf','lf','th')};final_valid=True;final_count=0
    try:
        with gzip.open(trial/'physics.jsonl.gz','rt') as f,gzip.open(trial/'actual-transitions.jsonl.gz','rt') as g:
            for line,rawline in itertools.zip_longest(f,g):
                if line is None or rawline is None:raise ValueError('Endpoint and actual-interval arrays have different lengths')
                row=json.loads(line);raw=json.loads(rawline)
                t=n*.002;end=t+.002
                for value,wanted in ((raw['interval_start_s'],t),(raw['geometry_time_s'],t),(raw['interval_end_s'],end),
                                     (row['contact_geometry_time_s'],t),(row['measurement_pose_time_s'],end)):
                    if abs(value-wanted)>1e-9:raise ValueError('Contact or endpoint epoch mismatch')
                result,patches=raw_pad_evidence(m,raw,lever)
                if any(c['normal_force_N']>0 for c in patches) and first_touch is None:first_touch=t
                if result['valid_pad_grasp']!=row['pad_grasp']['valid_pad_grasp']:mismatches.append(n)
                for k in min_final:
                    force_error=max(force_error,abs(result['qualified_pad_forces_N'][k]-row['pad_grasp']['qualified_pad_forces_N'][k]))
                bad_patches+=sum(c['normal_force_N']>1e-6 and not c['pad_qualified'] for c in patches)
                if result['valid_pad_grasp']:
                    valid_count+=1;current+=1
                    if first_valid is None:first_valid=t
                    if current>best:best=current;bestend=end
                else:current=0
                if t>=report['requested_duration_s']-.502-1e-9:
                    final_count+=1;final_valid=final_valid and result['valid_pad_grasp']
                    for k in min_final:min_final[k]=min(min_final[k],result['qualified_pad_forces_N'][k])
                n+=1
    finally:sim.close()
    checks=dict(physical_report_passed=report['passed'],complete_actual_interval_record=n==round(report['requested_duration_s']/.002),
        exact_classification=not mismatches,matching_qualified_pad_loads=force_error<1e-8,
        all_loaded_patches_original_distal=bad_patches==0,final_original_opposed_window=final_count>=251 and final_valid and best*.002>=.5)
    return dict(passed=all(checks.values()),checks=checks,interval_count=n,maximum_qualified_pad_force_difference_N=force_error,
        classification_mismatch_steps=mismatches,invalid_loaded_patches=bad_patches,first_positive_pad_contact_s=first_touch,
        first_qualified_interval_start_s=first_valid,strongest_qualified_hold_s=best*.002,strongest_hold_end_s=bestend,
        final_half_second_minimum_pad_force_N={k:v if np.isfinite(v) else None for k,v in min_final.items()},final_window_intervals=final_count,
        scope='Independent actual-interval contact/frame reduction; no controller input, physics step or contact-force recomputation',
        input_sha256={name:sha(trial/name) for name in ('provenance.json','report.json','physics.jsonl.gz','actual-transitions.jsonl.gz')},
        auditor_source_sha256=sha(__file__))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh audit output required')
    result=audit(a.trial);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

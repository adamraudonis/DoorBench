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
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
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


def audit(trial,*,phase='grasp'):
    if phase not in ('grasp','standing-withdrawal'):raise ValueError('Explicit supported contact-audit phase required')
    trial=Path(trial);report=json.loads((trial/'report.json').read_text());array_format=(trial/'raw-transitions/manifest.json').exists()
    if array_format:
        provenance_name='manifest.json';physics_name='physics-steps.json.gz';prov=json.loads((trial/provenance_name).read_text());cfg=prov['configuration'];robot=Path(cfg['robot']);door=Path(cfg['door'])
        robot_hash=prov['inputs']['robot']['sha256'];door_hash=prov['inputs']['door']['door.xml'];duration=report['expected_duration_s']
    else:
        provenance_name='provenance.json';physics_name='physics.jsonl.gz';prov=json.loads((trial/provenance_name).read_text());robot=Path(prov['parameters']['robot']);door=Path(prov['parameters']['door']);door=door if door.is_dir() else door.parent
        robot_hash=prov['robot_xml_sha256'];door_hash=prov['door_xml_sha256'];duration=report['requested_duration_s']
    if sha(robot)!=robot_hash or sha(door/'door.xml')!=door_hash:raise ValueError('Actual source asset bytes changed')
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()),frame_skip=1)
    m=sim.m;lever=m.geom('leaf_handle_lever_col_n').id
    opposed_end=duration;clearance_pairs=[];clearance_count=0;minimum_clearance=float('inf');clearance_error=0.
    if phase=='standing-withdrawal':
        if not array_format or 'standing_withdrawal' not in report:raise ValueError('Explicit native withdrawal report required')
        opposed_end=report['standing_withdrawal']['release_started_s']
        if opposed_end is not None and (not np.isfinite(opposed_end) or not .5<=opposed_end<=duration):raise ValueError('Invalid declared release epoch')
        active=[g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        hand=[g for g in active if m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
        scene=[g for g in active if not m.body(m.geom_bodyid[g]).name.startswith('robot/')]
        if not hand or not scene:raise ValueError('Complete hand/environment collision geometry required')
        clearance_pairs=[(g,h) for g in hand for h in scene]
    n=valid_count=current=best=0;force_error=0.;mismatches=[];bad_patches=0;first_touch=first_valid=None;bestend=None
    min_final={k:float('inf') for k in ('ff','mf','rf','lf','th')};final_valid=True;final_count=0
    raw_name='raw-transitions/manifest.json' if array_format else 'actual-transitions/manifest.json' if (trial/'actual-transitions/manifest.json').exists() else 'actual-transitions.jsonl.gz'
    def actual_rows():
        if raw_name.endswith('manifest.json'):
            yield from NativeTransitionArchive.read(trial/Path(raw_name).parent,allow_incomplete=not array_format)
        else:
            with gzip.open(trial/raw_name,'rt') as stream:
                for text in stream:yield json.loads(text)
    def physical_rows():
        with gzip.open(trial/physics_name,'rt') as f:
            if array_format:
                rows=json.load(f)
                if rows[0]['sim_time_s']!=0:raise ValueError('Missing actual t=0 reset')
                yield from rows[1:]
            else:
                for line in f:yield json.loads(line)
    try:
        for line,rawline in itertools.zip_longest(physical_rows(),actual_rows()):
                if line is None or rawline is None:raise ValueError('Endpoint and actual-interval arrays have different lengths')
                row=line;raw=rawline
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
                if opposed_end is not None and opposed_end-.502-1e-9<=t<opposed_end-1e-9:
                    final_count+=1;final_valid=final_valid and result['valid_pad_grasp']
                    for k in min_final:min_final[k]=min(min_final[k],result['qualified_pad_forces_N'][k])
                if clearance_pairs and t>=duration-.502-1e-9:
                    # Rebuild endpoint geometry from raw coordinates; never trust
                    # the runtime's clearance label or recompute contact forces.
                    sim.d.qpos[:]=raw['qpos_after'];mujoco.mj_kinematics(m,sim.d)
                    gap=min(float(mujoco.mj_geomDistance(m,sim.d,g,h,.5,None)) for g,h in clearance_pairs)
                    minimum_clearance=min(minimum_clearance,gap);clearance_count+=1
                    recorded=row.get('right_environment_clearance_m')
                    clearance_error=max(clearance_error,abs(gap-recorded) if recorded is not None and np.isfinite(recorded) else float('inf'))
                n+=1
    finally:sim.close()
    checks=dict(physical_report_passed=report['passed'],complete_actual_interval_record=n==round(duration/.002),
        exact_classification=not mismatches,matching_qualified_pad_loads=force_error<1e-8,
        all_loaded_patches_original_distal=bad_patches==0,final_original_opposed_window=final_count>=251 and final_valid and best*.002>=.5)
    if phase=='standing-withdrawal':
        checks['opposed_window_before_intentional_release']=checks.pop('final_original_opposed_window')
        checks['final_hand_clear_of_environment']=clearance_count>=251 and minimum_clearance>=.04
        checks['matching_recorded_final_clearance']=clearance_error<1e-8
    result=dict(passed=all(checks.values()),checks=checks,interval_count=n,maximum_qualified_pad_force_difference_N=force_error,
        classification_mismatch_steps=mismatches,invalid_loaded_patches=bad_patches,first_positive_pad_contact_s=first_touch,
        first_qualified_interval_start_s=first_valid,strongest_qualified_hold_s=best*.002,strongest_hold_end_s=bestend,
        final_half_second_minimum_pad_force_N={k:v if np.isfinite(v) else None for k,v in min_final.items()},final_window_intervals=final_count,
        scope='Independent actual-interval contact/frame reduction; no controller input, physics step or contact-force recomputation',
        input_sha256={name:sha(trial/name) for name in (provenance_name,'report.json',physics_name,raw_name)},
        auditor_source_sha256=sha(__file__))
    if phase=='standing-withdrawal':
        result['pre_release_half_second_minimum_pad_force_N']=result.pop('final_half_second_minimum_pad_force_N')
        result['pre_release_window_intervals']=result.pop('final_window_intervals')
        result.update(release_started_s=opposed_end,minimum_final_hand_clearance_m=minimum_clearance if np.isfinite(minimum_clearance) else None,maximum_clearance_label_error_m=clearance_error if np.isfinite(clearance_error) else None,final_clearance_intervals=clearance_count)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=('grasp','standing-withdrawal'),default='grasp');a=p.parse_args()
    if a.output.exists():p.error('Fresh audit output required')
    result=audit(a.trial,phase=a.phase);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

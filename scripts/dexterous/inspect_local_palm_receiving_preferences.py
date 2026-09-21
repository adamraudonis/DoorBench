#!/usr/bin/env python3
"""Compare prospective receiving normals without executing or qualifying a route.

The original fixed-body generator and its gates remain unchanged. This search
freezes every non-left-arm coordinate from a qualified source and inspects
which original hand collider would reach the panel first. It exports no
standing-transfer runtime route and cannot demonstrate loaded palm support.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.dexterous.plan_local_standing_transfer import (
    sha,write_json,resample_preferences,source_qualification,load_snapshot,
    generate_fixed_body_path,
)
from doorbench.dexterous.landed_left_planner import LandedLeftScene,JOINT_NAMES


def hand_gaps(m,d):
    panels=[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name=='leaf'
        and (m.geom(g).name or '').startswith('leaf_slab') and (m.geom_contype[g] or m.geom_conaffinity[g])]
    gaps={}
    for g in range(m.ngeom):
        name=m.body(int(m.geom_bodyid[g])).name
        if not name.startswith('robot/lh_') or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        gap=min(float(mujoco.mj_geomDistance(m,d,g,p,1.,None)) for p in panels)
        gaps[name]=min(gaps.get(name,float('inf')),gap)
    return gaps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','robot','door','preferences','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--normal-fractions',type=float,nargs='+',default=[0.,.33,.67,1.])
    parser.add_argument('--receiving-normal-offset-m',type=float,default=-.002)
    parser.add_argument('--receiving-height-offset-m',type=float,default=0.,help='Prospective numeric panel-height preference, not a model/root displacement')
    parser.add_argument('--pitch-bump',type=float,default=.4)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh static output directory')
    if not np.isfinite(a.normal_fractions).all() or min(a.normal_fractions)<0 or max(a.normal_fractions)>1:
        raise ValueError('Explicit normal fractions must be within zero and one')
    if not np.isfinite(a.receiving_normal_offset_m) or abs(a.receiving_normal_offset_m)>.005:
        raise ValueError('Use the original bounded receiving-plane preference')
    if not np.isfinite(a.receiving_height_offset_m) or abs(a.receiving_height_offset_m)>.15:
        raise ValueError('Bound the explicit numeric height search to +/-15 cm')
    source=source_qualification(a.source_run)
    if not source['passed']:raise ValueError('A qualified source is required; failed transfer is not a seed')
    robot=a.robot.resolve();door=a.door.resolve()
    if door.is_dir():door=door/'door.xml'
    scene=LandedLeftScene(robot,door)
    frozen,binding=load_snapshot(a.source_run,robot,door,scene)
    original=json.loads(a.preferences.read_text());numeric=resample_preferences(original)
    inputs={str(p.resolve()):sha(p) for p in (Path(__file__),a.preferences,robot,door)}
    inputs.update(binding['input_sha256'])
    a.output.mkdir(parents=True)
    rows=[]
    for fraction in a.normal_fractions:
        preferences=copy.deepcopy(numeric)
        normals=(1-fraction)*numeric['normal']+fraction*np.array([0.,-1.,0.])
        preferences['normal']=normals/np.linalg.norm(normals,axis=1)[:,None]
        preferences['position'][:,1]+=a.receiving_normal_offset_m
        preferences['position'][:,2]+=a.receiving_height_offset_m
        path,checks,fit=generate_fixed_body_path(scene,frozen,preferences,pitch_bump=a.pitch_bump)
        scene.d.qpos[:]=path[-1];mujoco.mj_kinematics(scene.m,scene.d)
        gaps=hand_gaps(scene.m,scene.d);palm=gaps['robot/lh_palm']
        first=min(gaps,key=gaps.get)
        name='normal-'+str(fraction).replace('.','p')
        pref=dict(schema='doorbench.left-palm-targets.v1',joint_names=JOINT_NAMES,
            scope='Prospective numerical preferences only; original source qualification is not inherited',
            targets=[dict(phase='left_reach',position=p,normal=n,nominal=q) for p,n,q in
                zip(preferences['position'].tolist(),preferences['normal'].tolist(),preferences['nominal'].tolist())])
        write_json(a.output/(name+'-preferences.json'),pref)
        write_json(a.output/(name+'-path.json'),dict(physics_steps=0,route_qualified=False,path_qpos=path,samples=checks))
        row=dict(normal_fraction=fraction,requested_normal=preferences['normal'][-1],
            endpoint_fit=fit,sampled_geometry_passed=all(r['passed'] for r in checks),
            failed_sample_indices=[i for i,r in enumerate(checks) if not r['passed']],
            left_body_panel_gap_m=gaps,first_body=first,palm_gap_m=palm,
            palm_lead_over_other_hand_bodies_m=min(v for k,v in gaps.items() if k!='robot/lh_palm')-palm,
            preference_path=str((a.output/(name+'-preferences.json')).resolve()))
        rows.append(row)
        print(json.dumps({k:row[k] for k in ('normal_fraction','sampled_geometry_passed','first_body','palm_gap_m','palm_lead_over_other_hand_bodies_m')}),flush=True)
    if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('Inputs changed during static comparison')
    write_json(a.output/'comparison.json',dict(schema='doorbench.local-receiving-normal-comparison.v1',
        source_qualification=source,source_binding=binding,physics_steps=0,runtime_routes_exported=0,
        scope=__doc__,input_sha256=inputs,rows=rows))


if __name__=='__main__':main()

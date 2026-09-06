#!/usr/bin/env python3
"""Audit every authored hand input and closed aperture on the native export.

This inventory checks transforms and contact dispatch. It does not certify an
operation sequence, collision-free motion, human reach or structural strength.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from doorbench.build import build_model,write_hardware_meshes
from doorbench.export.mjcf import build_mjcf
from doorbench.spec import generate_all
from doorbench.benchmark.interactions import ContactSites
from doorbench.benchmark.runner import operator_reachable,HAND_LIFTED_LATCHES,ENV_DRIVEN_LOCK_PARTS
from doorbench.benchmark.scenarios import site_world_positions
from doorbench.benchmark.passage import Passage


def audit(spec,hardware):
    model=build_model(spec);ir=model.to_dict('full');meta=ir['meta']
    write_hardware_meshes(model,str(hardware))
    xml=ET.tostring(build_mjcf(model,mesh_dir_rel=str(hardware)),encoding='unicode')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
    env=SimpleNamespace(m=m,d=d,mj=mujoco,model_json=ir,meta=meta,spec=spec)
    contacts=ContactSites(env)
    site_errors=[]
    for name,site in site_world_positions(ir).items():
        sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,name)
        if sid<0:continue
        error=float(np.linalg.norm(d.site_xpos[sid]-site['pos']))
        if error>5e-5:site_errors.append({'site':name,'position_error_m':error})
    missing=[]
    for body in ir['bodies']:
        j=body.get('joint')
        if not j or not j.get('robot_interactive',True):continue
        name,role=j['name'],j.get('role')
        if role=='operator' and not operator_reachable(spec,name):continue
        if role=='lock' and (not spec['lock'].get('robot_side_release',True) or any(p in name for p in ENV_DRIVEN_LOCK_PARTS)):continue
        if role=='latch' and not any(p in name for p in HAND_LIFTED_LATCHES):continue
        if role not in ('operator','lock','latch'):continue
        if contacts.select(name) is None:missing.append({'joint':name,'role':role})
    passage=Passage(m,spec,meta,ir)
    return {'door_id':spec['id'],'family':spec['family'],'model_sha256':sha256(json.dumps(ir,sort_keys=True).encode()).hexdigest(),
        'xml_sha256':sha256(xml.encode()).hexdigest(),'site_transform_errors':site_errors,'unserved_declared_hand_inputs':missing,
        'declared_lock':spec['lock'],'contacts':{name:[m.site(sid).name for sid,_ in values] for name,values in contacts.by_joint.items()},
        'sampled_access_paths':contacts.access_paths,
        'closed_clear_intervals':passage.intervals(d),'mechanical_incomplete':meta.get('mechanical_incomplete',[])}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ids',default='');args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True);hardware=(args.out.parent/'hardware').resolve()
    paths=sorted((ROOT/'doorbench').rglob('*.py'))+[Path(__file__).resolve()]
    before={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths}
    specs=generate_all();wanted=set(args.ids.split(',')) if args.ids else None
    rows=[];start=time.monotonic()
    for spec in specs:
        if wanted and spec['id'] not in wanted:continue
        try:rows.append(audit(spec,hardware))
        except Exception as e:rows.append({'door_id':spec['id'],'family':spec['family'],'error':str(e)})
        if len(rows)%100==0:print(f'{len(rows)} native inventories in {time.monotonic()-start:.1f}s',flush=True)
    changed=[str(p.relative_to(ROOT)) for p in paths if before[str(p.relative_to(ROOT))]!=sha256(p.read_bytes()).hexdigest()]
    result={'schema':'doorbench.interaction-inventory.v1','scope':__doc__,'count':len(rows),'elapsed_s':time.monotonic()-start,
        'source_sha256':before,'source_files_changed_during_audit':changed,'rows':rows,
        'counts':{'errors':sum('error' in r for r in rows),'doors_with_transform_errors':sum(bool(r.get('site_transform_errors')) for r in rows),
            'doors_with_unserved_inputs':sum(bool(r.get('unserved_declared_hand_inputs')) for r in rows),
            'closed_apertures_reported_clear':sum(bool(r.get('closed_clear_intervals')) for r in rows)}}
    args.out.write_text(json.dumps(result,indent=1,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('count','elapsed_s','counts','source_files_changed_during_audit')},indent=1))


if __name__=='__main__':main()

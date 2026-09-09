#!/usr/bin/env python3
"""Reject loaded working-hand contacts outside the grasped lever collider.

Independent extra gate on lossless native contacts. Does not replace the pad,
stance, joint, motor or task audits and never upgrades an existing trial.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def extra_handle_patch_indices(pair,geoms,forces,*,handle,lever,hands):
    pair,geoms,forces=np.asarray(pair),np.asarray(geoms),np.asarray(forces,float)
    if pair.ndim!=2 or pair.shape[1]!=2 or geoms.shape!=pair.shape or forces.shape!=(len(pair),) or not np.isfinite(forces).all() or np.any(forces<0):
        raise ValueError('Complete finite actual contact arrays required')
    return np.flatnonzero(np.any(pair==handle,axis=1)&np.any(np.isin(pair,hands),axis=1)&~np.any(geoms==lever,axis=1)&(forces>1e-6))


def audit(trial):
    from doorbench.dexterous.environment import DexterousDoorEnv
    trial=Path(trial);manifest=json.loads((trial/'manifest.json').read_text());archive=json.loads((trial/'raw-transitions/manifest.json').read_text())
    if not archive['complete']:raise ValueError('Complete physical archive required')
    cfg=manifest['configuration'];robot=Path(cfg['robot']);door=Path(cfg['door'])
    if sha(robot)!=manifest['inputs']['robot']['sha256'] or sha(door/'door.xml')!=manifest['inputs']['door']['door.xml']:raise ValueError('Exact recorded robot and door required')
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));m=sim.m
    try:
        handle=m.body('leaf_handle').id;lever=m.geom('leaf_handle_lever_col_n').id
        hands=[i for i in range(m.nbody) if m.body(i).name.startswith('robot/rh_')]
        count=intervals=0;maximum=0.;examples=[];first=None;last=None
        for chunk in archive['chunks']:
            path=trial/'raw-transitions'/chunk['file']
            if sha(path)!=chunk['sha256']:raise ValueError('Archived contacts changed')
            with np.load(path,allow_pickle=False) as z:
                pair=z['contact_body'];geoms=z['contact_geom'];forces=z['contact_wrench_contact_frame'][:,0]
                if not np.isfinite(forces).all() or np.any(forces<0):raise ValueError('Invalid actual contact forces')
                bad=extra_handle_patch_indices(pair,geoms,forces,handle=handle,lever=lever,hands=hands)
                count+=len(bad);maximum=max(maximum,float(forces[bad].max(initial=0)))
                indices=np.searchsorted(z['contact_offsets'][1:],bad,side='right');intervals+=len(np.unique(indices))
                if len(bad):
                    if first is None:first=float(z['interval_start_s'][indices[0]])
                    last=float(z['interval_end_s'][indices[-1]])
                for k,i in zip(bad[:max(0,10-len(examples))],indices):
                    examples.append(dict(interval_s=[float(z['interval_start_s'][i]),float(z['interval_end_s'][i])],bodies=[m.body(int(b)).name for b in pair[k]],geoms=[m.geom(int(g)).name for g in geoms[k]],normal_force_N=float(forces[k])))
        return dict(schema='doorbench.native-whole-handle-audit.v1',passed=count==0,scope='Additional gate across the entire recorded episode: no loaded RH contacts against handle assembly colliders other than the declared lever. Existing pad and task audits remain required.',loaded_threshold_N=1e-6,extra_loaded_patches=count,affected_physics_intervals=intervals,maximum_extra_force_N=maximum,first_bad_interval_start_s=first,last_bad_interval_end_s=last,examples=examples,input_sha256={str(p):sha(p) for p in (trial/'manifest.json',trial/'raw-transitions/manifest.json',robot,door/'door.xml')},auditor_sha256=sha(__file__))
    finally:sim.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh audit output required')
    r=audit(a.trial);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('passed','extra_loaded_patches','affected_physics_intervals','maximum_extra_force_N')}))
    return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

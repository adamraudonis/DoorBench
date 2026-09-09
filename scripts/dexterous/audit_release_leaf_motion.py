#!/usr/bin/env python3
"""Screen hand clearance while the supported leaf moves within a declared envelope.

Unstepped collision geometry only. Negative clearance is a planning defect, not
an inferred contact force. This supplements, never replaces, the original dense
anatomy audit and complete motor-driven episode.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.operation_teacher import smooth_phase


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--leaf-envelope-rad',type=float,required=True)
    a=p.parse_args()
    if not np.isfinite(a.leaf_envelope_rad) or not 0<a.leaf_envelope_rad<=.03:
        raise ValueError('Explicit positive envelope up to 0.03 rad required')
    if a.output.exists():raise ValueError('Fresh evidence path required')
    s=json.loads(a.screen.read_text());source=a.source_run
    manifest=json.loads((source/'manifest.json').read_text());cfg=manifest['configuration']
    robot=Path(cfg['robot']);door=Path(cfg['door'])
    if sha(robot)!=manifest['inputs']['robot']['sha256'] or sha(door/'door.xml')!=manifest['inputs']['door']['door.xml']:
        raise ValueError('Exact source geometry required')
    if sha(source/'trajectory.npz')!=s['source_trajectory_sha256']:
        raise ValueError('Screen and source state differ')
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    try:
        m,d=sim.m,sim.d;rows=s['trials'][0]['rows'];times=np.array([r['time_s'] for r in rows]);qs=np.array([r['qpos'] for r in rows])
        if qs.shape!=(len(rows),m.nq) or not np.isfinite(qs).all() or not np.all(np.diff(times)>0):
            raise ValueError('Complete monotonic finite route required')
        root=int(m.joint('robot/free_base').qposadr[0]);leaf=int(m.joint('leaf_hinge').qposadr[0]);lever=m.geom('leaf_handle_lever_col_n').id
        rotation=Slerp(times,Rotation.from_quat(qs[:,root+3:root+7][:,[1,2,3,0]]))
        hand=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
        handle=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name=='leaf_handle']
        # Distal lever contact is allowed here, but still must pass the original
        # anatomy audit. Other hand surfaces and every hub contact need clearance.
        pairs=[(g,h) for g in hand for h in handle if h!=lever or not m.body(m.geom_bodyid[g]).name.endswith('distal')]
        failures=[];minimum=.1;total=0
        for t in np.linspace(times[0],times[-1],1001):
            i=min(len(times)-2,max(0,int(np.searchsorted(times,t,side='right')-1)));f=(t-times[i])/(times[i+1]-times[i]);q=(1-f)*qs[i]+f*qs[i+1]
            quat=rotation(t).as_quat();q[root+3:root+7]=quat[[3,0,1,2]]
            # Zero at the attained start; the envelope grows as fingers release.
            extent=a.leaf_envelope_rad*float(smooth_phase((t-times[0])/3.))
            for offset in (-extent,0.,extent):
                d.qpos[:]=q;d.qpos[leaf]+=offset;mujoco.mj_kinematics(m,d);total+=1
                nearest=min((float(mujoco.mj_geomDistance(m,d,g,h,.1,None)),g,h) for g,h in pairs)
                gap,g,h=nearest;minimum=min(minimum,gap)
                if gap<0:failures.append(dict(route_time_s=float(t),leaf_offset_rad=float(offset),clearance_m=gap,hand_body=m.body(m.geom_bodyid[g]).name,handle_geom=m.geom(h).name))
        inputs=[a.screen,source/'trajectory.npz',source/'manifest.json',robot,door/'door.xml',Path(__file__)]
        result=dict(schema='doorbench.release-leaf-motion-screen.v1',passed=not failures,scope=__doc__,physics_steps=0,leaf_envelope_rad=a.leaf_envelope_rad,envelope_ramp='quintic 0–3 route seconds',sampled_configurations=total,minimum_clearance_m=minimum,failed_configurations=len(failures),failures=failures,input_sha256={str(p):sha(p) for p in inputs})
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('failures','input_sha256','scope')}))
        return 0 if result['passed'] else 1
    finally:sim.close()


if __name__=='__main__':raise SystemExit(main())

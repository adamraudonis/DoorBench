#!/usr/bin/env python3
"""Frozen-pose IK diagnostic for hub clearance; never authorizes a physical grasp."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from doorbench.dexterous.environment import DexterousDoorEnv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trial','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--at',type=float,action='append',required=True)
    p.add_argument('--torso-adjustment-rad',type=float,default=0.,help='Optional bounded waist adjustment, at most0.03rad; no balance qualification')
    a=p.parse_args()
    if not np.isfinite(a.torso_adjustment_rad) or not 0<=a.torso_adjustment_rad<=.03:raise ValueError('Torso adjustment must be within0..0.03rad')
    manifest=json.loads((a.trial/'manifest.json').read_text());cfg=manifest['configuration']
    robot=Path(cfg['robot']);door=Path(cfg['door'])
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(robot)!=manifest['inputs']['robot']['sha256'] or digest(door/'door.xml')!=manifest['inputs']['door']['door.xml']:
        raise ValueError('Exact recorded XML required')
    times=np.array([r['sim_time_s'] for r in json.loads((a.trial/'trace.json').read_text())])
    with np.load(a.trial/'trajectory.npz') as z:qs=z['qpos'].copy()
    if len(times)!=len(qs) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Matching monotonic recorded states required')
    if any(not np.isfinite(t) or not times[0]<=t<=times[-1] for t in a.at):raise ValueError('Requested time outside recording')
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d;rows=[]
    try:
        names=['right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['rh_WRJ2','rh_WRJ1']
        if a.torso_adjustment_rad:names=['torso',*names]
        ids=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[ids]
        palm=m.site('robot/rh_palm_touch').id;lever=m.geom('leaf_handle_lever_col_n').id;hub=m.geom('leaf_handle_hub_col_n').id
        shapes=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
        tips={digit:[g for g in shapes if m.body(m.geom_bodyid[g]).name=='robot/rh_'+digit+'distal'] for digit in ('ff','mf','rf','lf','th')}
        if any(not v for v in tips.values()):raise ValueError('All five original distal bodies required')
        for requested in a.at:
            i=int(np.searchsorted(times,requested));base=qs[i].copy();d.qpos[:]=base;mujoco.mj_kinematics(m,d)
            origin=d.site_xpos[palm].copy();orientation=d.site_xmat[palm].reshape(3,3).copy()
            actual_hub_gap=float(min(mujoco.mj_geomDistance(m,d,g,hub,.2,None) for g in shapes))
            actual_arm_limit_violation=float(np.maximum(np.maximum(m.jnt_range[ids,0]-base[qa],base[qa]-m.jnt_range[ids,1]),0).max())
            axis=d.geom_xmat[lever].reshape(3,3)[:,2].copy()
            direction=-np.sign((d.geom_xpos[hub]-d.geom_xpos[lever])@axis)*axis
            if not np.isclose(np.linalg.norm(direction),1):raise ValueError('Ambiguous free end')
            lower=m.jnt_range[ids,0].copy();upper=m.jnt_range[ids,1].copy()
            if a.torso_adjustment_rad:
                lower[0]=max(lower[0],base[qa[0]]-a.torso_adjustment_rad)
                upper[0]=min(upper[0],base[qa[0]]+a.torso_adjustment_rad)
            for shift in (0.,.002,.004,.006,.008,.010):
                target=origin+shift*direction
                def residual(x):
                    d.qpos[:]=base;d.qpos[qa]=x;mujoco.mj_kinematics(m,d)
                    return np.r_[100*(d.site_xpos[palm]-target),10*Rotation.from_matrix(d.site_xmat[palm].reshape(3,3)@orientation.T).as_rotvec(),.001*(x-base[qa])]
                fit=least_squares(residual,np.clip(base[qa],lower,upper),bounds=(lower,upper),max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                residual(fit.x)
                gaps={digit:float(min(mujoco.mj_geomDistance(m,d,g,lever,.2,None) for g in geoms)) for digit,geoms in tips.items()}
                hubgap=float(min(mujoco.mj_geomDistance(m,d,g,hub,.2,None) for g in shapes))
                pe=float(np.linalg.norm(d.site_xpos[palm]-target));re=float(np.linalg.norm(Rotation.from_matrix(d.site_xmat[palm].reshape(3,3)@orientation.T).as_rotvec()))
                rows.append(dict(time_s=float(times[i]),torso_adjustment_rad=float(fit.x[0]-base[qa[0]]) if a.torso_adjustment_rad else 0.,actual_recorded_hub_gap_m=actual_hub_gap,actual_arm_limit_violation_rad=actual_arm_limit_violation,free_end_shift_m=shift,world_direction=direction.tolist(),position_error_m=pe,rotation_error_rad=re,solver_converged=bool(fit.success),minimum_hand_hub_gap_m=hubgap,distal_lever_gaps_m=gaps,
                    promising_geometry_only=bool(fit.success and pe<.0001 and re<.001 and hubgap>=.002 and all(-.003<=x<=.003 for x in gaps.values()))))
        result=dict(scope=__doc__,torso_adjustment_limit_rad=a.torso_adjustment_rad,physics_steps=0,grasp_qualified=False,limitations=['No swept-path collision audit','No contact-force or pad-material qualification','No balance or motor rollout'],inputs={str(x):digest(x) for x in (robot,door/'door.xml',a.trial/'trajectory.npz',a.trial/'manifest.json',Path(__file__))},rows=rows)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(rows))
    finally:sim.close()


if __name__=='__main__':main()

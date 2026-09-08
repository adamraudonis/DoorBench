#!/usr/bin/env python3
"""Optimize bounded thumb preload in physics from a declared initialized pose."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import differential_evolution
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.provenance import capture
from doorbench.dexterous.contact_audit import lever_contacts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--robot',default='out/dexterous/robot/h1-shadow.xml')
    p.add_argument('--door',default='out/dexterous/assets/doors/db0055_swing_single')
    p.add_argument('--seed-pose',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=15)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    capture(Path(__file__).resolve().parents[2],a.output,vars(a))
    path=Path(a.robot);env=DexterousDoorEnv(a.door,path,json.loads(path.with_suffix('.audit.json').read_text()))
    qpos=np.load(a.seed_pose)['qpos'];local={env.m.actuator(i).name:k for k,i in enumerate(env.actuators)}
    thumb=[local['robot/rh_A_THJ'+str(i)] for i in (5,4,3,2,1)]
    curl=[local['robot/rh_A_'+finger+'J0'] for finger in ('FF','MF','RF','LF')]
    best={'objective':float('inf')};evaluations=0
    def rollout(delta):
        nonlocal evaluations,best
        env.reset(randomize=False,images=False);env.d.qpos[:]=qpos;env.d.qvel[:]=0;mujoco.mj_forward(env.m,env.d)
        controls=env.d.actuator_length[env.actuators].copy();controls[thumb]+=delta[:5];controls[curl]+=delta[5]
        action=env.normalize(np.clip(controls,env.low,env.high));trace=[];poses=[];velocities=[];commands=[]
        for step in range(20):
            env.step(action,images=False);contact=lever_contacts(env.m,env.d,'leaf_handle_lever_col_n')
            trace.append(contact);poses.append(env.d.qpos.copy());velocities.append(env.d.qvel.copy());commands.append(env.d.ctrl.copy())
        loaded=np.array([[row['digit_forces_N'][digit] for digit in ('ff','mf','rf','lf','th')] for row in trace[4:]])
        opposed=sum(row['opposed'] for row in trace[4:])
        # Saturation discourages crushing harder instead of establishing all five contacts.
        score=float(np.minimum(loaded,1.).mean(axis=0).sum()+3*np.minimum(loaded.min(axis=1),1.).mean()+5*opposed/16)
        objective=-score;evaluations+=1
        if objective<best['objective']:
            best={'objective':objective,'evaluations':evaluations,'thumb_delta_and_curl_rad':delta.tolist(),
                  'opposed_frames_after_settling':opposed,'evaluated_frames':16,
                  'mean_digit_forces_N':loaded.mean(axis=0).tolist(),'final':env.diagnostics()}
            np.savez_compressed(a.output/'best-trajectory.npz',qpos=np.array(poses),qvel=np.array(velocities),ctrl=np.array(commands))
            (a.output/'best-contacts.json').write_text(json.dumps(trace)+'\n')
            (a.output/'best.json').write_text(json.dumps(best,indent=2)+'\n');print(json.dumps(best),flush=True)
        return objective
    try:
        differential_evolution(rollout,[(-.3,.3)]*5+[(0,.3)],popsize=6,maxiter=a.iterations,seed=93,polish=False,workers=1)
        report={'stage':'contact preload optimization from initialized grasp','qualifying_start':False,
                'door_opening_claim':False,'evaluations':evaluations,'best':best,
                'limitations':['Only 0.4 seconds per candidate','Initial grasp pose supplied','Fixed motor targets; no complete opening policy']}
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:env.close()

if __name__=='__main__':main()

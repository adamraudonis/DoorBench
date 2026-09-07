#!/usr/bin/env python3
"""Evaluate body reaching on deterministic held-out seeds; never a door score."""
import argparse
import json
import hashlib
import platform
import mujoco
import torch
import time
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from doorbench.dexterous.reach_training import ReachTeacherEnv


def main():
    p=argparse.ArgumentParser()
    for name in ('upstream','robot','door','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--checkpoint',help='Omit to evaluate frozen upstream actor')
    p.add_argument('--episodes',type=int,default=30)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--distance',type=float,default=.25)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    env=ReachTeacherEnv(a.door,a.robot,a.upstream,distance=a.distance)
    model=PPO.load(a.checkpoint,device='cpu') if a.checkpoint else None
    rows=[];start=time.time()
    try:
        for seed in range(a.seed,a.seed+a.episodes):
            obs,_=env.reset(seed=seed);poses=[];velocities=[];controls=[];trace=[];max_tilt=0.;done=False
            while not done:
                if model:
                    action,_=model.predict(obs,deterministic=True)
                else:
                    action=env.controller.predict_body()
                obs,reward,terminated,truncated,info=env.step(action)
                done=terminated or truncated
                poses.append(env.sim.d.qpos.copy())
                velocities.append(env.sim.d.qvel.copy());controls.append(env.sim.d.ctrl.copy())
                trace.append({k:float(info[k]) for k in ('root_height_m','torso_tilt_deg','max_reach_error_m','external_wrench_max','applied_generalized_force_max')})
                max_tilt=max(max_tilt,info['torso_tilt_deg'])
            row={'seed':seed,'success':bool(info['is_success']),'fell':bool(info['fell']),
                 'max_tilt_deg':max_tilt,'final_reach_error_m':info['max_reach_error_m'],
                 'seconds':float(env.sim.d.time)}
            rows.append(row);print(json.dumps(row),flush=True)
            # Retain every trial for later video and failure inspection.
            np.savez_compressed(out/f'trial-{seed}.npz',qpos=np.array(poses),qvel=np.array(velocities),ctrl=np.array(controls))
            (out/f'trial-{seed}.json').write_text(json.dumps(trace)+'\n')
        report={'stage':'held-out body reach evaluation','door_opening_claim':False,
                'checkpoint':a.checkpoint,'episodes':len(rows),
                'successes':sum(r['success'] for r in rows),'falls':sum(r['fell'] for r in rows),
                'elapsed_seconds':time.time()-start,'trials':rows}
        report['checkpoint_sha256']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.checkpoint else None
        report['checkpoint_transitions']=int(model.num_timesteps) if model else None
        report['robot_sha256']=hashlib.sha256(Path(a.robot).read_bytes()).hexdigest()
        report['distance_m']=a.distance
        report['evaluated_at_unix']=time.time()
        report['evaluation_platform']=platform.platform()
        report['mujoco_version']=mujoco.__version__;report['torch_version']=torch.__version__
        report['evaluation_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:env.close()

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Measure native body-skill throughput at fixed physics; no GPU-physics claim."""
import argparse
from functools import partial
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from doorbench.dexterous.reach_training import ReachTeacherEnv


def main():
    p=argparse.ArgumentParser()
    for key in ('upstream','robot','door','checkpoint','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--counts',type=int,nargs='+',default=[4,8,16])
    p.add_argument('--steps',type=int,default=300)
    a=p.parse_args();torch.set_num_threads(1);rows=[]
    for count in a.counts:
        if count<1 or count>128:raise ValueError('Profile counts must be between 1 and 128')
        make=partial(ReachTeacherEnv,a.door,a.robot,a.upstream)
        env=SubprocVecEnv([make]*count,start_method='spawn')
        try:
            policy=PPO.load(a.checkpoint,device='cpu');obs=env.reset()
            for i in range(30):
                actions,_=policy.predict(obs,deterministic=True);obs,_,_,_=env.step(actions)
            start=time.perf_counter()
            for i in range(a.steps):
                actions,_=policy.predict(obs,deterministic=True);obs,_,_,_=env.step(actions)
            elapsed=time.perf_counter()-start
            row={'envs':count,'transitions':a.steps*count,'seconds':elapsed,'transitions_per_second':a.steps*count/elapsed}
            rows.append(row);print(json.dumps(row),flush=True)
        finally:env.close()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'scope':'native physics plus CPU actor inference; no rendering or PPO updates',
        'physics':{'timestep':.002,'frame_skip':10},'results':rows,
        'best_tested_envs':max(rows,key=lambda r:r['transitions_per_second'])['envs']},indent=2)+'\n')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Train a tactile grasp reflex from declared initialized poses."""
import argparse
from functools import partial
import json
from pathlib import Path
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from doorbench.dexterous.grasp_training import GraspSkillEnv
from doorbench.dexterous.provenance import capture


class Progress(BaseCallback):
    def __init__(self,out):
        super().__init__();self.out=out;self.start=time.time();self.last=0.;self.saved=0;self.rows=[]
    def _on_step(self):
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done:self.rows.append({k:info[k] for k in ('is_success','fell','held_steps','torso_tilt_deg','root_height_m')})
        if time.time()-self.last>5:
            self.last=time.time();row={'stage':'initialized tactile grasp training','timesteps':self.num_timesteps,
                'wall_seconds':time.time()-self.start,'heartbeat_unix':time.time(),'completed_episodes':len(self.rows),
                'recent_episodes':self.rows[-30:],'door_opening_claim':False}
            tmp=self.out/'progress.tmp';tmp.write_text(json.dumps(row,indent=2));tmp.replace(self.out/'progress.json')
            print(json.dumps({k:v for k,v in row.items() if k!='recent_episodes'}),flush=True)
            with (self.out/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if self.num_timesteps-self.saved>=10000:
            self.saved=self.num_timesteps;self.model.save(self.out/'checkpoint-pending.zip');(self.out/'checkpoint-pending.zip').replace(self.out/'latest.zip')
        return True


def main():
    p=argparse.ArgumentParser()
    for key in ('robot','door','seed-pose','preload-json'):p.add_argument('--'+key,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--envs',type=int,default=6)
    p.add_argument('--steps',type=int,default=200000);p.add_argument('--device',default='cpu')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists():raise SystemExit('Use a new output directory')
    capture(Path(__file__).resolve().parents[2],a.output,vars(a));torch.set_num_threads(1)
    make=partial(GraspSkillEnv,a.door,a.robot,a.seed_pose,a.preload_json)
    env=SubprocVecEnv([make]*a.envs,start_method='spawn')
    try:
        (a.output/'interface.json').write_text(json.dumps(env.env_method('configuration_audit')[0],indent=2)+'\n')
        model=PPO('MlpPolicy',env,policy_kwargs={'net_arch':dict(pi=[128,128],vf=[128,128]),'log_std_init':-1.5},
                  learning_rate=1e-4,n_steps=256,batch_size=256,n_epochs=5,target_kl=.03,device=a.device,seed=37,verbose=1)
        with torch.no_grad():model.policy.action_net.weight.mul_(0);model.policy.action_net.bias.mul_(0)
        (a.output/'config.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
        model.learn(total_timesteps=a.steps,callback=Progress(a.output));model.save(a.output/'final.zip')
        (a.output/'completed.json').write_text(json.dumps({'completed_at_unix':time.time(),'timesteps':model.num_timesteps,'door_opening_claim':False})+'\n')
    except Exception as exc:
        (a.output/'failed.json').write_text(json.dumps({'error':type(exc).__name__+': '+str(exc)})+'\n');raise
    finally:
        if any(not process.is_alive() for process in env.processes):
            for process in env.processes:
                if process.is_alive():process.terminate()
            for process in env.processes:process.join(timeout=5)
        else:env.close()

if __name__=='__main__':main()

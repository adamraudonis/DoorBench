#!/usr/bin/env python3
"""Adapt the frozen upstream body skill using PPO in native MuJoCo."""
import argparse
from functools import partial
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from doorbench.dexterous.reach_training import ReachTeacherEnv

class Progress(BaseCallback):
    def __init__(self, out, every=10000):
        super().__init__(); self.out=out;self.every=every;self.last=0;self.start=time.time()
        self.results=[]
    def _on_step(self):
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done:self.results.append({k:info[k] for k in ('is_success','fell','max_reach_error_m','torso_tilt_deg')})
        if self.num_timesteps-self.last >= self.every:
            self.last=self.num_timesteps
            row={'stage':'privileged body reach training','timesteps':self.num_timesteps,
                 'wall_seconds':time.time()-self.start,'heartbeat_unix':time.time(),
                 'completed_episodes':len(self.results),'recent_episodes':self.results[-30:],
                 'door_opening_claim':False}
            tmp=self.out/'progress.tmp';tmp.write_text(json.dumps(row,indent=2));tmp.replace(self.out/'progress.json')
            self.model.save(self.out/'latest')
            print(json.dumps({k:v for k,v in row.items() if k!='recent_episodes'}),flush=True)
        return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',required=True);p.add_argument('--robot',required=True);p.add_argument('--door',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=500000)
    p.add_argument('--envs',type=int,default=8);p.add_argument('--distance',type=float,default=.25)
    p.add_argument('--device',default='cuda');p.add_argument('--checkpoint');p.add_argument('--seed',type=int,default=17)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    constructor=partial(ReachTeacherEnv,a.door,a.robot,a.upstream,distance=a.distance)
    env=SubprocVecEnv([constructor for _ in range(a.envs)],start_method='spawn') if a.envs>1 else DummyVecEnv([constructor])
    try:
        if a.checkpoint:
            model=PPO.load(a.checkpoint,env=env,device=a.device)
        else:
            model=PPO('MlpPolicy',env,policy_kwargs={'net_arch':dict(pi=[256,256],vf=[256,256]),
                'activation_fn':torch.nn.Tanh,'log_std_init':-2.5},learning_rate=1e-5,n_steps=256,
                batch_size=512 if a.envs>1 else 256,n_epochs=5,gamma=.99,gae_lambda=.95,
                clip_range=.1,target_kl=.02,ent_coef=0.,device=a.device,seed=a.seed,verbose=1)
            weights=torch.load(Path(a.upstream)/'data/reach_two_hands/torch_model.pt',map_location='cpu',weights_only=True)
            # Initialize the actor exactly; the value network learns the new posture/contact task.
            with torch.no_grad():
                for layer,name in [(model.policy.mlp_extractor.policy_net[0],'dense1'),
                                   (model.policy.mlp_extractor.policy_net[2],'dense2'),
                                   (model.policy.action_net,'dense3')]:
                    layer.weight.copy_(weights[name+'.weight']);layer.bias.copy_(weights[name+'.bias'])
        (a.output/'config.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
        model.learn(total_timesteps=a.steps,callback=Progress(a.output))
        model.save(a.output/'final')
    finally:
        env.close()

if __name__=='__main__':main()

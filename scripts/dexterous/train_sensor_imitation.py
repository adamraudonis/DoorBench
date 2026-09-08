#!/usr/bin/env python3
"""Train a sensor-only recurrent force actor from qualified robot episodes.

This reports supervised prediction loss only. A checkpoint is not a door-opening
result until independently executed in the simulator without teacher assistance.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from doorbench.dexterous.provenance import capture
from doorbench.dexterous.sensor_actor import SensorActor
from doorbench.dexterous.sensor_demonstrations import SensorDemonstration
from doorbench.dexterous.motor_contract_identity import SENSOR_ACTOR_CHECKPOINT_SCHEMA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,action='append',required=True)
    parser.add_argument('--validation-episode',type=Path,action='append',default=[])
    parser.add_argument('--qualification',choices=['operation-report.json','acquisition-report.json'],default='operation-report.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy-teacher-receipt',type=Path,help='Explicit audited acquisition-only compatibility; one training episode')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--reset-observation-run',type=Path)
    parser.add_argument('--episode-start-probability',type=float,default=0.,help='Supervise true episode-start windows with zero GRU state and no masked prefix')
    parser.add_argument('--iterations',type=int,default=1000)
    parser.add_argument('--sequence-length',type=int,default=32)
    parser.add_argument('--burn-in',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if min(args.iterations,args.sequence_length,args.batch_size)<=0 or args.burn_in<0 or not np.isfinite(args.learning_rate) or args.learning_rate<=0:
        parser.error('Use finite positive training settings and nonnegative burn-in')
    if not np.isfinite(args.episode_start_probability) or not 0<=args.episode_start_probability<=1:
        parser.error('Episode-start probability must be in [0,1]')
    if args.reset_observation_run and not args.episode_start_probability:
        parser.error('Cold-start augmentation requires explicit start-window supervision')
    if args.output.exists():raise FileExistsError('Use a new training output directory')
    if args.legacy_teacher_receipt and (len(args.episode)!=1 or args.validation_episode):
        parser.error('The legacy acquisition receipt supports one explicit training episode and no validation episodes')
    episodes=[SensorDemonstration(p,qualification=args.qualification,legacy_teacher_receipt=args.legacy_teacher_receipt,reset_observation_run=args.reset_observation_run) for p in args.episode]
    validation=[SensorDemonstration(p,qualification=args.qualification) for p in args.validation_episode]
    first=episodes[0];dimensions=first.dimensions
    identity=lambda e:e.metadata['files']['actor-sensors.npz']
    if {identity(e) for e in episodes}&{identity(e) for e in validation}:
        raise ValueError('Validation must use separate episodes, not renamed training archives')
    for episode in episodes+validation:
        if (episode.layout!=first.layout or episode.dimensions!=dimensions or
                episode.metadata['physics_dt_s']!=first.metadata['physics_dt_s'] or
                episode.motor_contract_sha256!=first.motor_contract_sha256):
            raise ValueError('Freeze one embodiment, calibration, action order and control timestep per checkpoint')
        if len(episode)<args.sequence_length+args.burn_in:
            raise ValueError('Episode is too short for the declared recurrent window')
    capture(Path(__file__).resolve().parents[2],args.output,{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);device=torch.device(args.device)
    model=SensorActor(dimensions).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    history=[];started=time.time()

    def batch(pool):
        windows=[];targets=[]
        start_window=bool(args.episode_start_probability and rng.random()<args.episode_start_probability)
        burn=0 if start_window else args.burn_in
        total=args.sequence_length+burn
        for _ in range(args.batch_size):
            episode=pool[int(rng.integers(len(pool)))];start=0 if start_window else int(rng.integers(len(episode)-total+1))
            values,target=episode.sequence(start,total);windows.append(values);targets.append(target)
        inputs={key:torch.as_tensor(np.stack([w[key] for w in windows]),device=device) for key in windows[0]}
        return inputs,torch.as_tensor(np.stack(targets),device=device),burn

    def prediction(inputs,burn):
        hidden=None
        if burn:
            with torch.no_grad():
                _,hidden=model(**{k:v[:,:burn] for k,v in inputs.items()})
        return model(**{k:v[:,burn:] for k,v in inputs.items()},hidden=hidden)[0]

    for iteration in range(args.iterations):
        model.train();inputs,labels,burn=batch(episodes);optimizer.zero_grad(set_to_none=True)
        estimate=prediction(inputs,burn);target=labels[:,burn:]
        loss=torch.mean((estimate-target)**2)
        if not torch.isfinite(loss):raise ValueError('Nonfinite imitation objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if iteration%25==0 or iteration==args.iterations-1:
            row=dict(iteration=iteration+1,training_normalized_force_mse=float(loss.detach()),elapsed_s=time.time()-started)
            if validation:
                model.eval()
                with torch.inference_mode():
                    vi,vl,vburn=batch(validation);row['separate_episode_prediction_mse']=float(torch.mean((prediction(vi,vburn)-vl[:,vburn:])**2))
            history.append(row);(args.output/'progress.json').write_text(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    checkpoint=dict(schema=SENSOR_ACTOR_CHECKPOINT_SCHEMA,dimensions=asdict(dimensions),model_state=model.state_dict(),
        motor_contract_sha256=first.motor_contract_sha256,
        sensor_layout=first.layout,physics_dt_s=first.metadata['physics_dt_s'],seed=args.seed,
        training_episodes=[e.metadata for e in episodes],validation_episodes=[e.metadata for e in validation])
    torch.save(checkpoint,args.output/'actor.pt')
    result=dict(scope=__doc__,closed_loop_evaluated=False,task_success_rate=None,
        iterations=args.iterations,parameters=sum(p.numel() for p in model.parameters()),history=history,
        recurrent_training='Truncated windows with sensor-only burn-in; optional explicitly weighted true-start windows have zero hidden state and no masked prefix',
        episode_start_probability=args.episode_start_probability,
        observations='Stereo RGB, local tactile bins, encoders, IMU, previous action, sensor age/validity; no absolute clock or task state')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

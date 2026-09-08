#!/usr/bin/env python3
"""Train a sensor-only recurrent force actor from qualified robot episodes.

This reports supervised prediction loss only. A checkpoint is not a door-opening
result until independently executed in the simulator without teacher assistance.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from doorbench.dexterous.provenance import capture
from doorbench.dexterous.sensor_actor import SensorActor
from doorbench.dexterous.sensor_demonstrations import SensorDemonstration
from doorbench.dexterous.correction_demonstrations import CorrectionDemonstration
from doorbench.dexterous.motor_contract_identity import SENSOR_ACTOR_CHECKPOINT_SCHEMA
from doorbench.dexterous.sensor_training_bundle import load_bundle, digest
from doorbench.dexterous.sensor_fit_evaluation import evaluate_frozen_fit
from doorbench.dexterous.recurrent_sampling import sample_windows


def atomic_json(path, value):
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def atomic_torch(path, value):
    temp=path.with_name(path.name+'.partial')
    torch.save(value,temp);os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,action='append',default=[])
    parser.add_argument('--dataset-manifest',type=Path,help='Hash-verified relative-path bundle; mutually exclusive with other dataset arguments')
    parser.add_argument('--correction-dataset',type=Path,action='append',default=[],help='Separate audited counterfactual teacher labels on real student sensor observations')
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
    parser.add_argument('--window-sampling',choices=['legacy_fixed_burn','prefix_complete_v1'],default='legacy_fixed_burn')
    parser.add_argument('--checkpoint-every',type=int,default=0,help='Atomically preserve periodic weights and latest optimizer/RNG state')
    parser.add_argument('--evaluate-every',type=int,default=0,help='Full-history four-source and actual-start fit audit; requires frozen bundle')
    parser.add_argument('--max-wall-seconds',type=float,default=0.,help='Bound training wall time; zero disables the bound')
    args=parser.parse_args()
    if min(args.iterations,args.sequence_length,args.batch_size)<=0 or args.burn_in<0 or not np.isfinite(args.learning_rate) or args.learning_rate<=0:
        parser.error('Use finite positive training settings and nonnegative burn-in')
    if not np.isfinite(args.episode_start_probability) or not 0<=args.episode_start_probability<=1:
        parser.error('Episode-start probability must be in [0,1]')
    if min(args.checkpoint_every,args.evaluate_every,args.max_wall_seconds)<0 or not np.isfinite(args.max_wall_seconds):
        parser.error('Periodic intervals and wall bound must be nonnegative')
    if args.dataset_manifest and (args.episode or args.correction_dataset or args.validation_episode or args.legacy_teacher_receipt or args.reset_observation_run):
        parser.error('Frozen bundle owns all dataset paths; do not mix independent inputs')
    if not args.dataset_manifest and (not args.episode or args.evaluate_every):
        parser.error('Supply an episode or a bundle; periodic fixed-fit evaluation requires a bundle')
    if args.reset_observation_run and not args.episode_start_probability:
        parser.error('Cold-start augmentation requires explicit start-window supervision')
    if args.output.exists():raise FileExistsError('Use a new training output directory')
    if args.legacy_teacher_receipt and (len(args.episode)!=1 or args.validation_episode):
        parser.error('The legacy acquisition receipt supports one explicit training episode and no validation episodes')
    bundle=None
    if args.dataset_manifest:
        episodes,bundle=load_bundle(args.dataset_manifest)
        for key,value in bundle['training_settings'].items():
            if getattr(args,key)!=value:
                parser.error(f'Frozen training setting differs: {key}')
    else:
        episodes=[SensorDemonstration(p,qualification=args.qualification,legacy_teacher_receipt=args.legacy_teacher_receipt,reset_observation_run=args.reset_observation_run) for p in args.episode]
        episodes += [CorrectionDemonstration(p) for p in args.correction_dataset]
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
        if len(episode)<args.sequence_length+(args.burn_in if args.window_sampling=='legacy_fixed_burn' else 0):
            raise ValueError('Episode is too short for the declared recurrent window')
    capture(Path(__file__).resolve().parents[2],args.output,{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);device=torch.device(args.device)
    model=SensorActor(dimensions).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    history=[];started=time.time()
    supervised_counts=[np.zeros(len(e),np.int64) for e in episodes]

    def checkpoint(step):
        coverage=[dict(dataset_index=i,examples=len(count),supervised_events=int(count.sum()),
            unique_supervised_examples=int(np.count_nonzero(count)),unsupervised_examples=int(np.count_nonzero(count==0)),
            never_supervised_first128=np.flatnonzero(count[:128]==0).tolist(),last_example_supervisions=int(count[-1]))
            for i,count in enumerate(supervised_counts)]
        atomic_json(args.output/f'coverage-step-{step:06d}.json',dict(completed_optimizer_steps=step,
            window_sampling=args.window_sampling,datasets=coverage))
        value=dict(schema=SENSOR_ACTOR_CHECKPOINT_SCHEMA,dimensions=asdict(dimensions),model_state=model.state_dict(),
            motor_contract_sha256=first.motor_contract_sha256,sensor_layout=first.layout,
            physics_dt_s=first.metadata['physics_dt_s'],seed=args.seed,completed_optimizer_steps=step,
            training_episodes=[e.metadata for e in episodes],validation_episodes=[e.metadata for e in validation])
        destination=args.output/f'actor-step-{step:06d}.pt'
        atomic_torch(destination,value)
        atomic_torch(args.output/'actor.pt',value)
        atomic_torch(args.output/'training-state.pt',dict(schema='doorbench.sensor-optimizer-state.v1',
            completed_optimizer_steps=step,actor=value,optimizer=optimizer.state_dict(),
            numpy_generator_state=rng.bit_generator.state,python_random_state=random.getstate(),
            supervised_counts=[torch.from_numpy(c.copy()) for c in supervised_counts],
            torch_rng_state=torch.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            settings=json.loads(json.dumps(vars(args),default=str)),
            note='Optimizer/RNG preservation only; resume is not exposed by this training command'))
        return destination

    def batch(pool):
        selected=sample_windows(rng,[len(e) for e in pool],batch_size=args.batch_size,
            supervised_length=args.sequence_length,burn_in=args.burn_in,
            episode_start_probability=args.episode_start_probability,mode=args.window_sampling)
        # Equal history lengths can share a batch. Different lengths use their
        # actual prefix, with no synthetic zero observations or padded GRU steps.
        grouped={}
        for window in selected:
            if pool is episodes:
                supervised_counts[window.episode][window.label_start:window.label_start+window.supervised_length]+=1
            values,target=pool[window.episode].sequence(window.observation_start,window.total_length)
            grouped.setdefault(window.burn_in,[]).append((values,target))
        result=[]
        for burn,rows in grouped.items():
            inputs={key:torch.as_tensor(np.stack([w[key] for w,_ in rows]),device=device) for key in rows[0][0]}
            result.append((inputs,torch.as_tensor(np.stack([target for _,target in rows]),device=device),burn))
        return result

    def prediction(inputs,burn):
        hidden=None
        if burn:
            with torch.no_grad():
                _,hidden=model(**{k:v[:,:burn] for k,v in inputs.items()})
        return model(**{k:v[:,burn:] for k,v in inputs.items()},hidden=hidden)[0]

    completed=0;last_evaluation=None
    for iteration in range(args.iterations):
        model.train();groups=batch(episodes);optimizer.zero_grad(set_to_none=True)
        loss=sum(torch.mean((prediction(inputs,burn)-labels[:,burn:])**2)*(len(labels)/args.batch_size)
                 for inputs,labels,burn in groups)
        if not torch.isfinite(loss):raise ValueError('Nonfinite imitation objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        completed=iteration+1
        if iteration%25==0 or iteration==args.iterations-1:
            row=dict(iteration=iteration+1,training_normalized_force_mse=float(loss.detach()),elapsed_s=time.time()-started)
            if validation:
                model.eval()
                with torch.inference_mode():
                    row['separate_episode_prediction_mse']=float(sum(torch.mean((prediction(vi,vburn)-vl[:,vburn:])**2)*(len(vl)/args.batch_size)
                        for vi,vl,vburn in batch(validation)))
            history.append(row);atomic_json(args.output/'progress.json',row);print(json.dumps(row),flush=True)
        bounded_stop=bool(args.max_wall_seconds and time.time()-started>=args.max_wall_seconds)
        evaluate_now=bool(args.evaluate_every and (completed%args.evaluate_every==0 or completed==args.iterations or bounded_stop))
        save_now=evaluate_now or completed==args.iterations or bounded_stop or bool(args.checkpoint_every and completed%args.checkpoint_every==0)
        if save_now:
            saved=checkpoint(completed)
        if evaluate_now:
            evaluated=time.time()
            last_evaluation=evaluate_frozen_fit(model,episodes,[r['name'] for r in bundle['datasets']],readiness_limits=bundle['readiness_limits'])
            last_evaluation.update(completed_optimizer_steps=completed,checkpoint_sha256=digest(saved),elapsed_s=time.time()-started,evaluation_duration_s=time.time()-evaluated)
            atomic_json(args.output/f'fit-step-{completed:06d}.json',last_evaluation)
            atomic_json(args.output/'latest-fit.json',last_evaluation)
            print(json.dumps({'completed_optimizer_steps':completed,'fit':last_evaluation}),flush=True)
        if bounded_stop or (args.max_wall_seconds and time.time()-started>=args.max_wall_seconds):break
    result=dict(scope=__doc__,closed_loop_evaluated=False,task_success_rate=None,
        iterations=completed,requested_iterations=args.iterations,completed=completed==args.iterations,
        elapsed_s=time.time()-started,parameters=sum(p.numel() for p in model.parameters()),history=history,
        final_fit=last_evaluation,dataset_manifest_sha256=digest(args.dataset_manifest) if args.dataset_manifest else None,
        recurrent_training='Truncated windows with sensor-only burn-in; optional explicitly weighted true-start windows have zero hidden state and no masked prefix',
        episode_start_probability=args.episode_start_probability,
        window_sampling=args.window_sampling,
        history_semantics=('Uniform supervised-start indices; warm up from max(0,label_start-burn_in). Early prefixes use all available real history. Equal-history groups are batched; no padded observations.'
            if args.window_sampling=='prefix_complete_v1' else 'Original fixed-burn random windows plus explicit true-start windows; if burn_in exceeds sequence_length, intermediate early labels are unreachable.'),
        dataset_sampling='Uniform per dataset; qualified teacher episodes and counterfactual correction prefixes remain separately identified',
        training_datasets=[e.metadata for e in episodes],
        observations='Stereo RGB, local tactile bins, encoders, IMU, previous action, sensor age/validity; no absolute clock or task state')
    atomic_json(args.output/'report.json',result)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Fit continuous sensor histories; this is not a physical robot rollout."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch

from doorbench.dexterous.continuous_history_training import accumulate_full_sources,AccumulationInterrupted
from doorbench.dexterous.sensor_training_bundle import load_bundle,bundled_path,digest
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
from doorbench.dexterous.motor_contract_identity import SENSOR_ACTOR_CHECKPOINT_SCHEMA
from doorbench.dexterous.sensor_fit_evaluation import evaluate_frozen_fit
from doorbench.dexterous.provenance import capture
from scripts.dexterous.train_sensor_imitation import atomic_json,atomic_torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=64);p.add_argument('--chunk-length',type=int,default=32)
    p.add_argument('--cold-prefix-length',type=int,default=32);p.add_argument('--cold-weight',type=float,default=.5)
    p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cpu');p.add_argument('--checkpoint-every',type=int,default=8)
    p.add_argument('--evaluate-every',type=int,default=8);p.add_argument('--max-wall-seconds',type=float,default=2700.)
    args=p.parse_args()
    if min(args.iterations,args.chunk_length,args.cold_prefix_length,args.checkpoint_every,args.evaluate_every)<1:
        p.error('Positive iteration, chunk, prefix, and periodic counts are required')
    if not np.isfinite([args.cold_weight,args.learning_rate,args.max_wall_seconds]).all() or not 0<=args.cold_weight<=1 or min(args.learning_rate,args.max_wall_seconds)<=0:
        p.error('Use finite positive learning rate/wall cap and a cold weight in [0,1]')
    if args.output.exists():raise FileExistsError('Use a new output directory')
    episodes,bundle=load_bundle(args.dataset_manifest)
    for key,value in bundle['training_settings'].items():
        if getattr(args,key)!=value:p.error(f'Frozen setting differs: {key}')
    first=episodes[0]
    for e in episodes:
        if e.layout!=first.layout or e.dimensions!=first.dimensions or e.motor_contract_sha256!=first.motor_contract_sha256 or e.metadata['physics_dt_s']!=first.metadata['physics_dt_s']:
            raise ValueError('All sources must share actual embodiment, timing, and calibration')
    initial=bundle.get('initial_checkpoint',{});name=initial.get('path')
    checkpoint_path=bundled_path(args.dataset_manifest.parent,name)
    if digest(checkpoint_path)!=initial.get('sha256') or bundle['files_sha256'].get(name)!=initial.get('sha256'):
        raise ValueError('Warm-start checkpoint must be part of the immutable verified bundle')
    motors=json.loads((first.path/'motor-contract.json').read_text())
    actor=SensorPolicyController(checkpoint_path,motor_contract=motors,sensor_layout=first.layout,
        physics_dt_s=first.metadata['physics_dt_s'],device=args.device)
    payload=torch.load(checkpoint_path,weights_only=True,map_location='cpu')
    if payload.get('completed_optimizer_steps')!=initial.get('source_optimizer_step'):
        raise ValueError('Warm-start checkpoint step differs from the declared source')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    model=actor._actor.requires_grad_(True).train()
    # Preserve009's AdamW defaults exactly. No optimizer state is imported.
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    configuration.update(initial_checkpoint=initial,optimizer_class='AdamW',optimizer_initialization='fresh',optimizer_defaults=optimizer.defaults)
    capture(Path(__file__).resolve().parents[2],args.output,configuration)
    started=time.monotonic();deadline=started+args.max_wall_seconds
    history=[];last_evaluation=None;completed=0;aborted=None;last_saved=None
    def checkpoint(step):
        nonlocal last_saved
        coverage=dict(completed_optimizer_steps=step,protocol='continuous_history_v1',loss_views_per_logical_label=2,
            cold_prefix_weight=args.cold_weight,datasets=[dict(dataset_index=i,examples=len(e),logical_supervisions_per_example=step,
                unique_supervised_examples=len(e) if step else 0,unsupervised_examples=0 if step else len(e)) for i,e in enumerate(episodes)])
        atomic_json(args.output/f'coverage-step-{step:06d}.json',coverage)
        value=dict(schema=SENSOR_ACTOR_CHECKPOINT_SCHEMA,dimensions=asdict(first.dimensions),model_state=model.state_dict(),
            motor_contract_sha256=first.motor_contract_sha256,sensor_layout=first.layout,physics_dt_s=first.metadata['physics_dt_s'],
            seed=args.seed,completed_optimizer_steps=step,initial_checkpoint=initial,training_protocol='continuous_history_v1',
            training_episodes=[e.metadata for e in episodes],validation_episodes=[])
        destination=args.output/f'actor-step-{step:06d}.pt';atomic_torch(destination,value);atomic_torch(args.output/'actor.pt',value)
        atomic_torch(args.output/'training-state.pt',dict(schema='doorbench.sensor-optimizer-state.v1',completed_optimizer_steps=step,
            actor=value,optimizer=optimizer.state_dict(),optimizer_class='AdamW',optimizer_initialization='fresh',settings=configuration,
            supervised_counts=[torch.full((len(e),),step,dtype=torch.int64) for e in episodes],
            torch_rng_state=torch.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
            note='Complete-source updates only; interrupted accumulated gradients are discarded. No resume interface.'))
        last_saved=step
        return destination
    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True);model.train();begin=time.monotonic();last_progress=begin
        def progress(row):
            nonlocal last_progress
            now=time.monotonic()
            if now-last_progress>=5 or row['next_sample']==row['examples']:
                atomic_json(args.output/'progress.json',dict(iteration=completed,requested_iterations=args.iterations,
                    accumulating_update=iteration+1,elapsed_s=now-started,source_progress=row,parameters_updated=False))
                last_progress=now
        try:
            result=accumulate_full_sources(model,episodes,chunk_length=args.chunk_length,cold_prefix_length=args.cold_prefix_length,
                cold_weight=args.cold_weight,deadline=deadline,progress=progress)
        except AccumulationInterrupted as error:
            optimizer.zero_grad(set_to_none=True)
            aborted=dict(attempted_update=iteration+1,logical_samples=error.logical_samples,completed_sources=error.completed_sources,
                gradients_discarded=True,parameters_updated=False)
            atomic_json(args.output/'interrupted-accumulation.json',aborted);break
        except Exception as error:
            optimizer.zero_grad(set_to_none=True)
            atomic_json(args.output/'accumulation-error.json',dict(attempted_update=iteration+1,
                completed_optimizer_steps=completed,gradients_discarded=True,parameters_updated=False,error=repr(error)))
            raise
        gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(gradient_norm):raise ValueError('Nonfinite accumulated gradient')
        optimizer.step();completed=iteration+1
        row=dict(iteration=completed,requested_iterations=args.iterations,elapsed_s=time.monotonic()-started,
            update_duration_s=time.monotonic()-begin,training_normalized_force_mse=result['mixed_loss'],
            training_recorded_history_mse=result['recorded_loss'],training_actor_history_mse=result['actor_loss'],
            preclip_gradient_norm=float(gradient_norm),source_passes=result['sources'],parameters_updated=True)
        history.append(row);atomic_json(args.output/'progress.json',row);print(json.dumps(row),flush=True)
        bounded_stop=time.monotonic()>=deadline
        evaluate_now=completed%args.evaluate_every==0 or completed==args.iterations or bounded_stop
        if evaluate_now or completed%args.checkpoint_every==0:saved=checkpoint(completed)
        if evaluate_now and time.monotonic()<deadline:
            evaluated=time.monotonic();last_evaluation=evaluate_frozen_fit(model,episodes,[r['name'] for r in bundle['datasets']],
                readiness_limits=bundle['readiness_limits'],include_actor_history_sources=True)
            last_evaluation.update(completed_optimizer_steps=completed,checkpoint_sha256=digest(saved),elapsed_s=time.monotonic()-started,
                evaluation_duration_s=time.monotonic()-evaluated)
            atomic_json(args.output/f'fit-step-{completed:06d}.json',last_evaluation);atomic_json(args.output/'latest-fit.json',last_evaluation)
        if bounded_stop or time.monotonic()>=deadline:break
    # Even a wall-bound stop during accumulation keeps exactly the last whole
    # update, not the incomplete gradients or a partially advanced optimizer.
    if last_saved!=completed:checkpoint(completed)
    result=dict(scope=__doc__,closed_loop_evaluated=False,task_success_rate=None,iterations=completed,requested_iterations=args.iterations,
        completed=completed==args.iterations,elapsed_s=time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters()),
        history=history,final_fit=last_evaluation if last_evaluation and last_evaluation['completed_optimizer_steps']==completed else None,last_available_fit=last_evaluation,initial_checkpoint=initial,optimizer_class='AdamW',optimizer_initialization='fresh',
        optimizer_defaults=optimizer.defaults,interrupted_accumulation=aborted,dataset_manifest_sha256=digest(args.dataset_manifest),
        training_datasets=[e.metadata for e in episodes],protocol='continuous_history_v1',
        history_semantics='Each source resets once. Separate recorded and actor-owned histories continue through every32-frame gradient chunk. Hidden and command gradients detach at boundaries; numeric states continue. Weights do not change until all sources complete.',
        objective='Equal source weights; equal separate history losses; per-source 50% first32-label mean plus50% complete-source mean. No synthetic observations or teacher commands inside actor-owned history.',
        observations='Stereo RGB, local tactile bins, encoders, IMU, previous action and sensor age/validity; no task/object state or absolute clock',
        limitation='Recorded sensors cannot react to predicted forces. This is fitting, never physically executed recovery or stable robot control.')
    atomic_json(args.output/'report.json',result)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Fit complete audited native sensor histories from fresh weights.

Reports supervised training loss only, never a physical task success. Recurrent
state spans the entire episode; weights update only after a full source pass.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from doorbench.dexterous.continuous_history_training import accumulate_full_sources, AccumulationInterrupted
from doorbench.dexterous.native_sensor_demonstrations import NativeSensorDemonstration
from doorbench.dexterous.sensor_actor import SensorActor
from doorbench.dexterous.motor_contract_identity import SENSOR_ACTOR_CHECKPOINT_SCHEMA
from doorbench.dexterous.provenance import capture
from scripts.dexterous.train_sensor_imitation import atomic_json, atomic_torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--camera-variant', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--iterations', type=int, default=1)
    p.add_argument('--chunk-length', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--max-wall-seconds', type=float, default=1800.)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if (min(a.iterations, a.chunk_length) < 1 or not np.isfinite([a.learning_rate, a.max_wall_seconds]).all()
            or min(a.learning_rate, a.max_wall_seconds) <= 0):
        p.error('Positive finite settings required')
    if a.output.exists():
        raise FileExistsError('Preserve previous training attempts')
    episode = NativeSensorDemonstration(a.run, a.camera_variant)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    model = SensorActor(episode.dimensions).to(a.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate)
    configuration = {k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
    configuration.update(initialization='fresh', dataset=episode.metadata,
        cold_prefix_length=32, cold_weight=.5, protocol='continuous_history_v1',
        limitation=__doc__)
    capture(Path(__file__).resolve().parents[2], a.output, configuration)
    started = time.monotonic(); deadline = started+a.max_wall_seconds
    history = []; interrupted = None; last_progress = 0.
    for iteration in range(a.iterations):
        optimizer.zero_grad(set_to_none=True)
        def progress(row):
            nonlocal last_progress
            now = time.monotonic()
            if now-last_progress >= 5 or row['next_sample'] == row['examples']:
                value = dict(completed_optimizer_steps=len(history), attempted_update=iteration+1,
                    elapsed_s=now-started, source_progress=row, parameters_updated=False)
                atomic_json(a.output/'progress.json', value)
                print(json.dumps(value), flush=True); last_progress = now
        try:
            result = accumulate_full_sources(model, [episode], chunk_length=a.chunk_length,
                cold_prefix_length=32, cold_weight=.5, deadline=deadline, progress=progress)
        except AccumulationInterrupted as error:
            optimizer.zero_grad(set_to_none=True)
            interrupted = dict(logical_samples=error.logical_samples, gradients_discarded=True,
                parameters_updated=False)
            break
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite gradient; no optimizer update accepted')
        optimizer.step()
        history.append(dict(update=iteration+1, elapsed_s=time.monotonic()-started,
            pre_update_loss=result, gradient_norm_before_clip=float(norm)))
        payload = dict(schema=SENSOR_ACTOR_CHECKPOINT_SCHEMA, dimensions=asdict(episode.dimensions),
            model_state=model.state_dict(), motor_contract_sha256=episode.motor_contract_sha256,
            sensor_layout=episode.layout, physics_dt_s=episode.metadata['physics_dt_s'],
            seed=a.seed, completed_optimizer_steps=len(history), training_protocol='continuous_history_v1',
            training_episodes=[episode.metadata], validation_episodes=[])
        atomic_torch(a.output/'actor.pt', payload)
        atomic_torch(a.output/'training-state.pt', dict(actor=payload, optimizer=optimizer.state_dict(),
            torch_rng_state=torch.get_rng_state(), configuration=configuration))
        atomic_json(a.output/'progress.json', dict(completed_optimizer_steps=len(history),
            logical_examples_per_update=len(episode), history=history, physical_rollout_evaluated=False))
        if time.monotonic() >= deadline:
            break
    report = dict(completed_optimizer_steps=len(history), requested_optimizer_steps=a.iterations,
        complete=len(history)==a.iterations, history=history, interrupted=interrupted,
        source_examples=len(episode), physical_rollout_evaluated=False, scope=__doc__)
    atomic_json(a.output/'report.json', report)
    print(json.dumps(report), flush=True)
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

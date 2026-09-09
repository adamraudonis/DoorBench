#!/usr/bin/env python3
"""Fit complete audited native sensor histories with full-source updates.

Reports supervised training loss only, never a physical task success. Recurrent
state spans the entire episode; weights update only after a full source pass.
"""
import argparse
import hashlib
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



def initialize_actor_weights(actor, model, episode):
    if (actor['schema'] != SENSOR_ACTOR_CHECKPOINT_SCHEMA
            or actor['dimensions'] != asdict(episode.dimensions)
            or actor['motor_contract_sha256'] != episode.motor_contract_sha256
            or actor['sensor_layout'] != episode.layout
            or actor['physics_dt_s'] != episode.metadata['physics_dt_s']):
        raise ValueError('Initial actor contract mismatch')
    model.load_state_dict(actor['model_state'], strict=True)


def restore_training_state(state, model, optimizer, configuration, episode):
    """Restore only completed full-source updates under the same data/protocol."""
    previous = state['configuration']; actor = state['actor']
    for key in ('seed', 'learning_rate', 'chunk_length', 'cold_prefix_length',
                'cold_weight', 'protocol', 'dataset', 'device'):
        if previous.get(key) != configuration.get(key):
            raise ValueError(f'Resume configuration mismatch: {key}')
    if (actor['schema'] != SENSOR_ACTOR_CHECKPOINT_SCHEMA
            or actor['dimensions'] != asdict(episode.dimensions)
            or actor['motor_contract_sha256'] != episode.motor_contract_sha256
            or actor['sensor_layout'] != episode.layout
            or actor['physics_dt_s'] != episode.metadata['physics_dt_s']
            or actor['training_protocol'] != 'continuous_history_v1'):
        raise ValueError('Resume actor contract mismatch')
    completed = actor['completed_optimizer_steps']
    if type(completed) is not int or completed < 1:
        raise ValueError('Resume requires a completed optimizer update')
    history = state.get('history', [])
    if any(type(row.get('update')) is not int or not 1 <= row['update'] <= completed for row in history):
        raise ValueError('Invalid checkpoint history')
    model.load_state_dict(actor['model_state'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng_state'].cpu())
    return completed, history

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
    p.add_argument('--correction-run', type=Path, action='append', default=[], help='Separately audited bounded native recovery source')
    p.add_argument('--correction-only', action='store_true', help='Explicit approach curriculum; does not retain complete-task imitation coverage')
    initial = p.add_mutually_exclusive_group()
    initial.add_argument('--resume-state', type=Path, help='Completed-update checkpoint; output must be a new directory')
    initial.add_argument('--initialize-actor', type=Path, help='Fine-tune existing sensor weights with fresh Adam; new experiment, not exact optimizer recovery')
    a = p.parse_args()
    if (min(a.iterations, a.chunk_length) < 1 or not np.isfinite([a.learning_rate, a.max_wall_seconds]).all()
            or min(a.learning_rate, a.max_wall_seconds) <= 0):
        p.error('Positive finite settings required')
    if a.output.exists():
        raise FileExistsError('Preserve previous training attempts')
    if a.correction_only and not a.correction_run:
        p.error('--correction-only requires at least one --correction-run')
    episode = NativeSensorDemonstration(a.run, a.camera_variant)
    from doorbench.dexterous.native_correction_demonstrations import NativeCorrectionDemonstration
    corrections = [NativeCorrectionDemonstration(path) for path in a.correction_run]
    for correction in corrections:
        if (correction.layout != episode.layout or correction.dimensions != episode.dimensions
                or correction.motor_contract_sha256 != episode.motor_contract_sha256):
            raise ValueError('Native correction contract differs from complete teacher source')
    episodes = ([] if a.correction_only else [episode]) + corrections
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    model = SensorActor(episode.dimensions).to(a.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate)
    configuration = json.loads(json.dumps(vars(a), default=str))
    configuration.update(initialization='fresh', dataset=[e.metadata for e in episodes] if corrections else episode.metadata,
        cold_prefix_length=32, cold_weight=.5, protocol='continuous_history_v1',
        limitation=__doc__)
    if a.initialize_actor is not None:
        actor = torch.load(a.initialize_actor, map_location=a.device, weights_only=False)
        initialize_actor_weights(actor, model, episode)
        configuration.update(initialization='pretrained_actor_fresh_adam',
            initial_actor_sha256=hashlib.sha256(a.initialize_actor.read_bytes()).hexdigest(),
            pretrained_optimizer_steps=actor.get('completed_optimizer_steps'),
            pretrained_episodes=actor.get('training_episodes', []),
            previous_optimizer_restored=False)
    completed = 0; inherited_history = []
    if a.resume_state is not None:
        state = torch.load(a.resume_state, map_location=a.device, weights_only=False)
        completed, inherited_history = restore_training_state(
            state, model, optimizer, configuration, episode)
        if completed >= a.iterations:
            raise ValueError('Requested total updates must exceed the saved completed count')
        configuration.update(initialization='resumed_completed_update',
            resume_state_sha256=hashlib.sha256(a.resume_state.read_bytes()).hexdigest(),
            resumed_optimizer_steps=completed,
            prior_history_available=len(inherited_history)==completed)
    capture(Path(__file__).resolve().parents[2], a.output, configuration)
    started = time.monotonic(); deadline = started+a.max_wall_seconds
    history = list(inherited_history); initial_completed = completed; interrupted = None; last_progress = 0.
    for iteration in range(completed, a.iterations):
        optimizer.zero_grad(set_to_none=True)
        def progress(row):
            nonlocal last_progress
            now = time.monotonic()
            if now-last_progress >= 5 or row['next_sample'] == row['examples']:
                value = dict(completed_optimizer_steps=completed, attempted_update=iteration+1,
                    elapsed_s=now-started, source_progress=row, parameters_updated=False)
                atomic_json(a.output/'progress.json', value)
                print(json.dumps(value), flush=True); last_progress = now
        try:
            result = accumulate_full_sources(model, episodes, chunk_length=a.chunk_length,
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
        completed = iteration+1
        history.append(dict(update=iteration+1, elapsed_s=time.monotonic()-started,
            pre_update_loss=result, gradient_norm_before_clip=float(norm)))
        payload = dict(schema=SENSOR_ACTOR_CHECKPOINT_SCHEMA, dimensions=asdict(episode.dimensions),
            model_state=model.state_dict(), motor_contract_sha256=episode.motor_contract_sha256,
            sensor_layout=episode.layout, physics_dt_s=episode.metadata['physics_dt_s'],
            seed=a.seed, completed_optimizer_steps=completed, training_protocol='continuous_history_v1',
            training_episodes=[e.metadata for e in episodes], validation_episodes=[])
        atomic_torch(a.output/'actor.pt', payload)
        atomic_torch(a.output/'training-state.pt', dict(actor=payload, optimizer=optimizer.state_dict(),
            torch_rng_state=torch.get_rng_state(), configuration=configuration, history=history))
        atomic_json(a.output/'progress.json', dict(completed_optimizer_steps=completed,
            logical_examples_per_update=sum(len(e) for e in episodes), history=history, physical_rollout_evaluated=False))
        if time.monotonic() >= deadline:
            break
    report = dict(completed_optimizer_steps=completed, requested_optimizer_steps=a.iterations,
        complete=completed==a.iterations, history=history, interrupted=interrupted,
        resumed_optimizer_steps=initial_completed, missing_historical_updates=initial_completed-len(inherited_history),
        source_examples=sum(len(e) for e in episodes), correction_only=a.correction_only,
        complete_task_examples_in_update=0 if a.correction_only else len(episode),
        physical_rollout_evaluated=False, scope=__doc__)
    atomic_json(a.output/'report.json', report)
    print(json.dumps(report), flush=True)
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

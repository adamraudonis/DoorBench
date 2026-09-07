#!/usr/bin/env python3
"""Development validation of initialized grasp; never an opening score."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from doorbench.dexterous.grasp_training import GraspSkillEnv
from doorbench.dexterous.provenance import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('robot', 'door', 'seed-pose', 'preload-json'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--checkpoint', help='Omit for constant optimized preload')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20000)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new evaluation directory')
    args.output.mkdir(parents=True)
    capture(Path(__file__).resolve().parents[2], args.output, vars(args))
    torch.set_num_threads(1)
    env = GraspSkillEnv(args.door, args.robot, args.seed_pose, args.preload_json)
    (args.output / 'interface.json').write_text(json.dumps(env.configuration_audit(), indent=2) + '\n')
    model = PPO.load(args.checkpoint, device='cpu') if args.checkpoint else None
    rows = []
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            obs, _ = env.reset(seed=seed)
            states = {key: [] for key in ('qpos', 'qvel', 'ctrl')}
            trace = []
            while True:
                action = model.predict(obs, deterministic=True)[0] if model else np.zeros(6)
                obs, _, terminated, truncated, info = env.step(action)
                trace.append(info)
                for key in states:
                    states[key].append(getattr(env.sim.d, key).copy())
                if terminated or truncated:
                    break
            row = {'seed': seed, 'success': info['is_success'], 'fell': info['fell'],
                   'max_contiguous_hold_steps': max(x['held_steps'] for x in trace),
                   'opposed_frames': sum(x['opposed'] for x in trace), 'frames': len(trace),
                   'max_torso_tilt_deg': max(x['torso_tilt_deg'] for x in trace),
                   'min_root_height_m': min(x['root_height_m'] for x in trace)}
            rows.append(row)
            np.savez_compressed(args.output / f'trial-{seed}.npz', **states)
            (args.output / f'trial-{seed}.json').write_text(json.dumps(trace) + '\n')
            print(json.dumps(row), flush=True)
        report = {'stage': 'development initialized grasp evaluation', 'door_opening_claim': False,
                  'controller': 'tactile PPO' if model else 'constant optimized preload',
                  'checkpoint_transitions': int(model.num_timesteps) if model else None,
                  'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
                  'falls': sum(r['fell'] for r in rows), 'trials': rows,
                  'evaluated_at_unix': time.time(),
                  'scope': 'One-second opposed contact from an initialized pose; no approach or opening'}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        env.close()


if __name__ == '__main__':
    main()

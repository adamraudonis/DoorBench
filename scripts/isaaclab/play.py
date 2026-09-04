#!/usr/bin/env python
"""Play a trained DoorBench policy (thin wrapper around Isaac Lab's rsl_rl/play.py).

  ./isaaclab.sh -p scripts/isaaclab/play.py --task DoorBench-Open-Hand-Play-v0 --num_envs 64 --doors easy-100 \
      [--checkpoint logs/rsl_rl/doorbench_hand/<run>/model_300.pt] [--video --video_length 600] [--headless]

Without --checkpoint the latest run of the task's experiment is used.  Exports the policy as TorchScript + ONNX
into <run>/exported/.

NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_door_args, apply_door_args, ensure_extension_importable  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Play a DoorBench RSL-RL policy.")
parser.add_argument("--task", type=str, default="DoorBench-Open-Hand-Play-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="path to model_<iter>.pt (default: latest run)")
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--random", action="store_true", help="random actions instead of a policy (smoke test)")
add_door_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

ensure_extension_importable()
import doorbench_isaaclab  # noqa: E402, F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    apply_door_args(env_cfg, args_cli)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = None
    if not args_cli.random:
        if args_cli.checkpoint:
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, args_cli.load_run or agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    else:
        log_dir = os.path.join(log_root_path, "random")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, video_folder=os.path.join(log_dir, "videos", "play"), step_trigger=lambda step: step == 0, video_length=args_cli.video_length, disable_logger=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = None
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        export_dir = os.path.join(log_dir, "exported")
        try:
            export_policy_as_jit(runner.alg.policy, normalizer=getattr(runner, "obs_normalizer", None), path=export_dir, filename="policy.pt")
            export_policy_as_onnx(runner.alg.policy, normalizer=getattr(runner, "obs_normalizer", None), path=export_dir, filename="policy.onnx")
        except Exception as e:  # export API differs between rsl-rl versions; not essential for playing
            print(f"[WARN] policy export skipped: {e}")
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    if isinstance(obs, tuple):  # older wrappers return (obs, extras)
        obs = obs[0]
    step = 0
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions = policy(obs) if policy is not None else torch.randn(env.num_envs, env.num_actions, device=env.unwrapped.device).clamp(-1, 1)
            obs, _, _, _ = env.step(actions)
        step += 1
        if args_cli.video and step >= args_cli.video_length:
            break
        if args_cli.real_time:
            time.sleep(max(0.0, dt - (time.time() - t0)))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

#!/usr/bin/env python
"""Train a DoorBench policy with RSL-RL PPO (thin wrapper around Isaac Lab's rsl_rl/train.py).

Run inside the Isaac Lab python:
  ./isaaclab.sh -p /path/to/DoorBench/scripts/isaaclab/train.py --task DoorBench-Open-Hand-v0 --num_envs 1024 \
      --max_iterations 300 --doors easy-100 --headless
  ... --task DoorBench-Open-G1-v0 --num_envs 2048 --doors easy-100 --headless --video

Extra flags over Isaac Lab's script: --doors / --door_seed / --door_random_choice (door subset per environment).
Logs: logs/rsl_rl/<experiment_name>/<date>/  (checkpoints model_<iter>.pt, params/env.yaml, params/agent.yaml,
params/env.pkl, params/agent.pkl, params/doors.txt)

Mirrors scripts/reinforcement_learning/rsl_rl/train.py of Isaac Lab **v2.3.2** (rsl-rl-lib 3.1.2): configs are
dumped with ``isaaclab.utils.io.dump_yaml`` (the pickle helpers were removed from Isaac Lab in isaaclab 0.47.0; the
``.pkl`` copies come from ``_common.dump_pickle``), the runner is chosen by ``agent_cfg.class_name`` and the env cfg
gets ``log_dir``.  In this repo training is a data-validation tool (isaaclab/cloud/run_all.sh: TRAIN=1).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_door_args, apply_door_args, dump_pickle, ensure_extension_importable, rsl_rl_version_check  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train a DoorBench policy with RSL-RL.")
parser.add_argument("--task", type=str, default="DoorBench-Open-Hand-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--video", action="store_true", help="record videos during training (needs --enable_cameras in headless mode; set automatically)")
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--video_interval", type=int, default=5000)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--logger", type=str, default=None, choices=[None, "wandb", "tensorboard", "neptune"])
parser.add_argument("--log_project_name", type=str, default=None)
add_door_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------------------------- after app start
import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

rsl_rl_version_check()
ensure_extension_importable()
import doorbench_isaaclab  # noqa: E402, F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    apply_door_args(env_cfg, args_cli)
    # CLI overrides (as in Isaac Lab's cli_args.update_rsl_rl_cfg)
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.experiment_name:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.logger:
        agent_cfg.logger = args_cli.logger
    if args_cli.log_project_name and agent_cfg.logger in ("wandb", "neptune"):
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    if hasattr(env_cfg, "log_dir"):  # v2.3.2: the env writes its own logs (IO descriptors, physics logs) here
        env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # resolve the resume path before the new log_dir exists (get_checkpoint_path picks the latest run)
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint) if agent_cfg.resume else None
    if args_cli.video:
        video_kwargs = {"video_folder": os.path.join(log_dir, "videos", "train"), "step_trigger": lambda step: step % args_cli.video_interval == 0,
                        "video_length": args_cli.video_length, "disable_logger": True}
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}.get(getattr(agent_cfg, "class_name", "OnPolicyRunner"))
    if runner_cls is None:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    with open(os.path.join(log_dir, "params", "doors.txt"), "w") as f:
        f.write("\n".join(os.path.basename(os.path.dirname(p)) for p in env_cfg.scene.door.spawn.usd_path))

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

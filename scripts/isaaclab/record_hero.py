#!/usr/bin/env python
"""Hero shot: hundreds of environments, each with a different DoorBench door, in one 3D scene.

  ./isaaclab.sh -p scripts/isaaclab/record_hero.py --task DoorBench-Open-Hand-v0 --num_envs 512 --doors all \
      --headless --enable_cameras [--checkpoint logs/rsl_rl/doorbench_hand/<run>/model_300.pt]
  -> docs/media/isaaclab_hero.png (wide shot), docs/media/isaaclab_hero_detail.png (close shot),
     docs/media/isaaclab_hero.mp4 (~12 s at 30 fps)

Random actions when no checkpoint is given.  Uses the env's ``rgb_array`` render (the viewport camera of
``cfg.viewer``) and writes the video with imageio (ffmpeg).

NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, add_door_args, apply_door_args, ensure_extension_importable, unwrap_obs  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Record the DoorBench Isaac Lab hero shot.")
parser.add_argument("--task", type=str, default="DoorBench-Open-Hand-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--out", type=str, default=os.path.join(ROOT, "docs", "media", "isaaclab_hero"))
parser.add_argument("--resolution", type=str, default="1920x1080")
parser.add_argument("--settle_steps", type=int, default=30, help="steps before the wide screenshot")
add_door_args(parser)
parser.set_defaults(doors="all")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

ensure_extension_importable()
import doorbench_isaaclab  # noqa: E402, F401


def main():
    w, h = (int(x) for x in args_cli.resolution.lower().split("x"))
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    apply_door_args(env_cfg, args_cli)
    env_cfg.viewer.resolution = (w, h)
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.seconds + 2.0)
    n = args_cli.num_envs
    side = math.ceil(math.sqrt(n))
    spacing = env_cfg.scene.env_spacing
    extent = side * spacing
    # wide shot: elevated 3/4 view over the whole grid (the cloner centres the grid on the origin)
    env_cfg.viewer.eye = (0.62 * extent, -0.95 * extent, 0.45 * extent)
    env_cfg.viewer.lookat = (0.0, 0.05 * extent, 0.0)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env)
    base = env.unwrapped
    policy = None
    if args_cli.checkpoint:
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=base.device)
        runner.load(os.path.abspath(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=base.device)
    obs = unwrap_obs(env.get_observations())   # TensorDict in Isaac Lab v2.3.x

    def act():
        if policy is not None:
            return policy(obs)
        return 0.6 * torch.randn(env.num_envs, env.num_actions, device=base.device).clamp(-1, 1)

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    frames = []
    n_frames = int(args_cli.seconds * args_cli.fps)
    steps_per_frame = max(1, round(1.0 / (args_cli.fps * base.step_dt)))
    with torch.inference_mode():
        for i in range(args_cli.settle_steps):
            obs, _, _, _ = env.step(act())
        base.render()  # warm up the render product
        wide = base.render()
        imageio.imwrite(args_cli.out + ".png", wide)
        print(f"[hero] wrote {args_cli.out}.png ({wide.shape[1]}x{wide.shape[0]}, {n} envs, {len(env_cfg.scene.door.spawn.usd_path)} distinct doors)")
        for f in range(n_frames):
            for _ in range(steps_per_frame):
                obs, _, _, _ = env.step(act())
            # slow orbit of the camera during the video
            ang = 2 * math.pi * f / n_frames * 0.25
            r = 1.1 * extent
            base.sim.set_camera_view(eye=(r * math.sin(ang + 0.6), -r * math.cos(ang + 0.6), 0.45 * extent), target=(0.0, 0.05 * extent, 0.0))
            frames.append(base.render())
        # detail shot: first env, camera at the robot's shoulder
        origin = base.scene.env_origins[0].cpu().numpy()
        base.sim.set_camera_view(eye=tuple(origin + np.array([2.2, -3.2, 1.9])), target=tuple(origin + np.array([0.0, 0.0, 1.0])))
        for _ in range(3):
            obs, _, _, _ = env.step(act())
            detail = base.render()
        imageio.imwrite(args_cli.out + "_detail.png", detail)
    if frames:
        imageio.mimwrite(args_cli.out + ".mp4", frames, fps=args_cli.fps, quality=8, macro_block_size=None)
        print(f"[hero] wrote {args_cli.out}.mp4 ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

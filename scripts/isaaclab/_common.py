"""Shared helpers for the Isaac Lab scripts (imported AFTER the AppLauncher started the simulator)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def ensure_extension_importable():
    """Make ``doorbench_isaaclab`` importable from the checkout even when it was not pip-installed."""
    try:
        import doorbench_isaaclab  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.join(ROOT, "isaaclab"))
        import doorbench_isaaclab  # noqa: F401
    os.environ.setdefault("DOORBENCH_ASSETS", os.path.join(ROOT, "assets"))


def add_door_args(parser):
    parser.add_argument("--doors", type=str, default=os.environ.get("DOORBENCH_DOORS", "easy-100"),
                        help="door subset: easy-100 | easy-300 | all | random-50 | family:saloon,pivot | db0002_swing_single,... | @ids.txt")
    parser.add_argument("--door_seed", type=int, default=0, help="seed for the door shuffle / random subset")
    parser.add_argument("--door_random_choice", action="store_true", help="random door per env instead of round-robin")


def apply_door_args(env_cfg, args):
    from doorbench_isaaclab.door_task_env_cfg import set_doors

    set_doors(env_cfg, args.doors, seed=args.door_seed, random_choice=args.door_random_choice)
    n = len(env_cfg.scene.door.spawn.usd_path)
    print(f"[doorbench] {n} doors selected by --doors {args.doors!r} (round-robin over {env_cfg.scene.num_envs} envs)")
    return env_cfg

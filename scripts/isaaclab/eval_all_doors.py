#!/usr/bin/env python
"""Evaluate a checkpoint over all (or a subset of) DoorBench doors, N seeds each, and write a results JSON.

  ./isaaclab.sh -p scripts/isaaclab/eval_all_doors.py --task DoorBench-Open-Hand-Play-v0 \
      --checkpoint logs/rsl_rl/doorbench_hand/<run>/model_300.pt --doors all --seeds 3 --batch 250 --headless

Doors are evaluated in batches: a scene with `batch` doors x `seeds` envs (round-robin assignment, so every env's door
is known), one full episode per env (until the termination manager ends it or the time limit), then the scene is
rebuilt for the next batch.  Output (default results/isaaclab_<task>_<timestamp>.json), compatible with
results/schema.json when present in the repo; otherwise:
  {"policy": ..., "simulator": "isaaclab", "per_door": {door_id: {"success": bool, "events": [...], "time_s": ...}}, "aggregate": {...}}

NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, ensure_extension_importable  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate a DoorBench policy over many doors.")
parser.add_argument("--task", type=str, default="DoorBench-Open-Hand-Play-v0")
parser.add_argument("--checkpoint", type=str, default=None, help="model_<iter>.pt (omit with --random)")
parser.add_argument("--random", action="store_true", help="random policy baseline")
parser.add_argument("--doors", type=str, default="all")
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--batch", type=int, default=250, help="doors per scene (batch * seeds environments)")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--policy_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

ensure_extension_importable()
import doorbench_isaaclab  # noqa: E402, F401
from doorbench_isaaclab import doors as D  # noqa: E402
from doorbench_isaaclab.door_task_env_cfg import set_doors  # noqa: E402
from doorbench_isaaclab.mdp import get_door_state  # noqa: E402


def run_batch(ids: list[str], seeds: int, agent_cfg, checkpoint: str | None):
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=len(ids) * seeds)
    set_doors(env_cfg, ids, seed=0, random_choice=False)
    env_cfg.scene.door.spawn.usd_path = D.door_usd_paths(ids)  # unshuffled: env i -> door i % len(ids)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    policy = None
    if checkpoint:
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=base.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=base.device)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    st = get_door_state(base)
    n = base.num_envs
    done_once = torch.zeros(n, dtype=torch.bool, device=base.device)
    records = [None] * n
    max_steps = base.max_episode_length + 5
    with torch.inference_mode():
        for _ in range(max_steps):
            actions = policy(obs) if policy is not None else torch.randn(n, env.num_actions, device=base.device).clamp(-1, 1)
            obs, _, dones, extras = env.step(actions)
            # env.step() already reset the finished environments; DoorState.reset() (run by the reset_door event
            # before the managers reset) snapshotted their final labels -> read the `last` record
            newly = dones.bool() & ~done_once
            for k in torch.nonzero(newly).flatten().tolist():
                records[k] = st.episode_record(k, last=True)
            done_once |= dones.bool()
            if bool(done_once.all()):
                break
    # envs that never terminated (should not happen with time_out): take current labels
    for k in range(n):
        if records[k] is None:
            records[k] = st.episode_record(k)
    env.close()
    return records


def main():
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    ids = D.select_ids(args_cli.doors)
    checkpoint = None if args_cli.random else os.path.abspath(args_cli.checkpoint)
    policy_name = args_cli.policy_name or ("random" if args_cli.random else os.path.basename(os.path.dirname(checkpoint)) + "/" + os.path.basename(checkpoint))
    t0 = time.time()
    per_door: dict[str, list] = {i: [] for i in ids}
    for b in range(0, len(ids), args_cli.batch):
        batch = ids[b: b + args_cli.batch]
        print(f"[eval] batch {b // args_cli.batch + 1}: {len(batch)} doors x {args_cli.seeds} seeds")
        recs = run_batch(batch, args_cli.seeds, agent_cfg, checkpoint)
        for k, r in enumerate(recs):
            per_door[batch[k % len(batch)]].append(r)
    # aggregate
    man = {d["id"]: d for d in D.manifest()["doors"]}
    out_doors = {}
    for i, recs in per_door.items():
        succ = [r["success"] for r in recs]
        out_doors[i] = {
            "success": bool(sum(succ) > len(succ) / 2) if succ else False,
            "success_rate": (sum(succ) / len(succ)) if succ else 0.0,
            "events": sorted({e for r in recs for e in r["events"]}),
            "time_s": float(sum(r["time_to_pass"] for r in recs if r["time_to_pass"] >= 0) / max(1, sum(1 for r in recs if r["time_to_pass"] >= 0))) if any(r["time_to_pass"] >= 0 for r in recs) else None,
            "episodes": recs,
            "family": man.get(i, {}).get("family"), "task": man.get(i, {}).get("task"), "difficulty": man.get(i, {}).get("difficulty"),
        }
    by_family: dict[str, list] = {}
    by_task: dict[str, list] = {}
    for i, r in out_doors.items():
        by_family.setdefault(r["family"], []).append(r["success_rate"])
        by_task.setdefault(r["task"], []).append(r["success_rate"])
    aggregate = {
        "n_doors": len(out_doors), "seeds": args_cli.seeds,
        "success_rate": sum(r["success_rate"] for r in out_doors.values()) / max(1, len(out_doors)),
        "n_doors_solved": sum(1 for r in out_doors.values() if r["success"]),
        "damage_rate": sum(1 for r in out_doors.values() for e in r["episodes"] if "door_damaged" in e["events"]) / max(1, sum(len(r["episodes"]) for r in out_doors.values())),
        "by_family": {f: sum(v) / len(v) for f, v in sorted(by_family.items())},
        "by_task": {t: sum(v) / len(v) for t, v in sorted(by_task.items())},
        "wall_time_s": round(time.time() - t0, 1),
    }
    result = {"policy": policy_name, "simulator": "isaaclab", "task": args_cli.task, "checkpoint": checkpoint, "doors": args_cli.doors,
              "generated": datetime.datetime.now().isoformat(timespec="seconds"), "per_door": out_doors, "aggregate": aggregate}
    schema = os.path.join(ROOT, "results", "schema.json")
    if os.path.exists(schema):
        try:
            with open(schema) as f:
                sch = json.load(f)
            result["schema"] = sch.get("$id") or sch.get("title") or "results/schema.json"
            # keep the same top-level keys as the schema when they exist
            for key in sch.get("required", []):
                result.setdefault(key, None)
        except Exception as e:
            print(f"[eval] results/schema.json present but unreadable: {e}")
    out = args_cli.out or os.path.join(ROOT, "results", f"isaaclab_{args_cli.task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[eval] {aggregate['n_doors_solved']}/{aggregate['n_doors']} doors solved, success rate {aggregate['success_rate']:.3f}, damage rate {aggregate['damage_rate']:.3f} -> {out}")


if __name__ == "__main__":
    main()
    simulation_app.close()

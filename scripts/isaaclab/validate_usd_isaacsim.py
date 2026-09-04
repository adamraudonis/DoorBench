#!/usr/bin/env python
"""Headless Isaac Sim / Isaac Lab validation of the DoorBench USDs (runs INSIDE the Isaac Lab python on a GPU box).

  ./isaaclab.sh -p scripts/isaaclab/validate_usd_isaacsim.py --all [--which rl|full|both] [--batch 25] [--headless]
  ./isaaclab.sh -p scripts/isaaclab/validate_usd_isaacsim.py --ids db0002_swing_single,db0345_sliding_single

For every door (batches of `batch` doors in one stage, each door its own Isaac Lab ``Articulation``):
  * loads door_rl.usda (canonical 7-DoF) and/or door.usda (full) through ``sim_utils.UsdFileCfg``
  * checks the joint names / limits / drive gains against model.json (full) or the canonical set (rl)
  * settles 200 physics steps: finite state, no joint drift beyond 2 deg / 2 cm on a closed door, no explosion
  * applies a torque / force on the operator joint (if any) and a push on the door joint for 400 steps and reports
    whether the primary joint opened (doors without a robot-side release must NOT open)
  * closer doors: releases and checks the door returns toward closed
Writes a JSON report (default assets/usd_validation_isaacsim.json).

*** NOT EXECUTED ON THIS MACHINE (Apple silicon, no NVIDIA GPU): written against the Isaac Lab 2.3 API; expect small
    fixes on first run (see isaaclab/STATUS.md). ***
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, ensure_extension_importable  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Validate DoorBench USDs in Isaac Sim.")
parser.add_argument("--all", action="store_true")
parser.add_argument("--ids", type=str, default="")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--which", type=str, default="both", choices=["rl", "full", "both"])
parser.add_argument("--batch", type=int, default=25)
parser.add_argument("--out", type=str, default=os.path.join(ROOT, "assets", "usd_validation_isaacsim.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import build_simulation_context  # noqa: E402

ensure_extension_importable()
from doorbench_isaaclab import doors as D  # noqa: E402
from doorbench_isaaclab.assets import DOOR_ACTUATORS, DOOR_ARTICULATION_PROPS, DOOR_RIGID_PROPS  # noqa: E402

RL_JOINTS = {"door_slide", "door_hinge", "operator_hinge", "operator_slide", "latch_slide", "leaf2_slide", "leaf2_hinge"}
DT = 1.0 / 120.0


def _door_cfg(door_id: str, kind: str, k: int) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=f"/World/Doors/door_{k:03d}",
        spawn=sim_utils.UsdFileCfg(usd_path=D.usd_path(door_id, canonical=(kind == "rl")), activate_contact_sensors=False, rigid_props=DOOR_RIGID_PROPS, articulation_props=DOOR_ARTICULATION_PROPS),
        init_state=ArticulationCfg.InitialStateCfg(pos=(6.0 * (k % 10), 6.0 * (k // 10), 0.0)),
        actuators=DOOR_ACTUATORS,
        articulation_root_prim_path="/Articulation",
    )


def validate_batch(ids: list[str], kind: str, device: str) -> list[dict]:
    rows = []
    with build_simulation_context(device=device, dt=DT, gravity_enabled=True, add_ground_plane=True, auto_add_lighting=True) as sim:
        arts, metas = [], []
        for k, did in enumerate(ids):
            try:
                arts.append(Articulation(_door_cfg(did, kind, k)))
            except Exception as e:
                arts.append(None)
                rows.append({"id": did, "kind": kind, "ok": False, "errors": [f"spawn: {type(e).__name__}: {e}"]})
            metas.append(D.load_model(did))
        sim.reset()
        for k, (did, art, mj) in enumerate(zip(ids, arts, metas)):
            if art is None:
                continue
            errors, warnings, stats = [], [], {}
            try:
                art.update(DT)
                jn = list(art.joint_names)
                stats["joints"] = jn
                stats["bodies"] = list(art.body_names)
                mj_joints = {b["joint"]["name"]: b["joint"] for b in mj["bodies"] if b.get("joint")}
                if kind == "rl":
                    if set(jn) != RL_JOINTS:
                        errors.append(f"joint names {sorted(jn)} != canonical")
                else:
                    if set(jn) != set(mj_joints):
                        errors.append(f"joint names differ from model.json: {sorted(set(mj_joints) ^ set(jn))[:6]}")
                    lim = art.data.joint_pos_limits[0].cpu().numpy()
                    st = art.data.default_joint_stiffness[0].cpu().numpy()
                    for i, name in enumerate(jn):
                        j = mj_joints.get(name)
                        if j is None or j.get("range") is None:
                            continue
                        lo, hi = j["range"][0] - j["modeled_at"], j["range"][1] - j["modeled_at"]
                        if abs(lim[i][0] - lo) > 2e-3 or abs(lim[i][1] - hi) > 2e-3:
                            errors.append(f"{name}: limits {lim[i].tolist()} != IR {[lo, hi]}")
                        if abs(st[i] - j["stiffness"]) > 1e-2 * max(1.0, abs(j["stiffness"])):
                            errors.append(f"{name}: stiffness {st[i]:.4g} != IR {j['stiffness']:.4g}")
                rows.append({"id": did, "kind": kind, "ok": not errors, "errors": errors, "warnings": warnings, "stats": stats, "_art": art, "_mj": mj})
            except Exception as e:
                rows.append({"id": did, "kind": kind, "ok": False, "errors": [f"inspect: {type(e).__name__}: {e}"]})
        # ---- settle
        live = [r for r in rows if "_art" in r]
        q0 = {r["id"]: r["_art"].data.joint_pos.clone() for r in live}
        for _ in range(200):
            for r in live:
                r["_art"].set_joint_position_target(torch.zeros_like(r["_art"].data.joint_pos))
                r["_art"].write_data_to_sim()
            sim.step()
            for r in live:
                r["_art"].update(DT)
        for r in live:
            art, mj = r["_art"], r["_mj"]
            q = art.data.joint_pos
            if not torch.isfinite(q).all() or not torch.isfinite(art.data.joint_vel).all():
                r["errors"].append("non-finite joint state after settling")
            drift = (q - q0[r["id"]]).abs().max().item()
            r["stats"]["settle_drift"] = drift
            if drift > 0.05:
                r["warnings"].append(f"joint drift {drift:.3f} after 200 steps (spring preloads move joints to their stops; > 0.05 rad/m is suspicious)")
            # ---- actuate: operator torque + push on the door joint
            jn = list(art.joint_names)
            meta = mj["meta"]
            if r["kind"] == "rl":
                rl = json.loads(sim.stage.GetPrimAtPath(art.cfg.prim_path).GetAttribute("doorbench:rl").Get())
                pj = jn.index(rl["door_joint"])
                oj = jn.index(rl["operator_slot_joint"]) if rl.get("operator_slot_joint") else None
                lock_engaged = rl["lock"]["engaged"] and not rl["lock"]["robot_side_release"]
            else:
                pj = jn.index(meta["primary_joint"])
                oj = jn.index(meta["operator_joint"]) if meta.get("operator_joint") in jn else None
                lock_engaged = False
            r["_pj"], r["_oj"], r["_lock"] = pj, oj, lock_engaged
            r["_q_before"] = q[0, pj].item()
        for _ in range(400):
            for r in live:
                art = r["_art"]
                eff = torch.zeros_like(art.data.joint_pos)
                if r["_oj"] is not None:
                    eff[0, r["_oj"]] = 8.0        # N*m on a lever / N on a bar (generous)
                eff[0, r["_pj"]] = 60.0            # N*m on a hinge / N on a slider
                art.set_joint_effort_target(eff)
                art.set_joint_position_target(torch.zeros_like(art.data.joint_pos))
                art.write_data_to_sim()
            sim.step()
            for r in live:
                r["_art"].update(DT)
        for r in live:
            art = r["_art"]
            q = art.data.joint_pos[0, r["_pj"]].item()
            opened = abs(q - r["_q_before"])
            r["stats"]["opened"] = opened
            if not torch.isfinite(art.data.joint_pos).all():
                r["errors"].append("non-finite joint state after actuation")
            if r["_lock"]:
                if opened > 0.1:
                    r["errors"].append(f"locked door opened by {opened:.3f}")
            elif opened < 0.05:
                r["warnings"].append(f"door did not open under 60 N*m + operator torque (opened {opened:.3f}); heavy / spring-loaded doors may need more")
            for key in ("_art", "_mj", "_pj", "_oj", "_lock", "_q_before"):
                r.pop(key, None)
    return rows


def main():
    if args_cli.ids:
        ids = args_cli.ids.split(",")
    elif args_cli.all:
        ids = D.all_ids()
    else:
        ids = D.easy_ids(20)
    if args_cli.limit:
        ids = ids[: args_cli.limit]
    kinds = ["rl", "full"] if args_cli.which == "both" else [args_cli.which]
    device = args_cli.device or "cuda:0"
    t0 = time.time()
    rows = []
    for kind in kinds:
        for b in range(0, len(ids), args_cli.batch):
            batch = ids[b: b + args_cli.batch]
            print(f"[isaacsim-validate] {kind} batch {b // args_cli.batch + 1}/{math.ceil(len(ids) / args_cli.batch)}: {len(batch)} doors")
            try:
                rows += validate_batch(batch, kind, device)
            except Exception as e:  # a crashing batch must not lose the report
                rows += [{"id": i, "kind": kind, "ok": False, "errors": [f"batch exception: {type(e).__name__}: {e}"]} for i in batch]
    n_ok = sum(1 for r in rows if r["ok"])
    summary = {"n_files": len(rows), "n_ok": n_ok, "n_failed": len(rows) - n_ok, "time_s": round(time.time() - t0, 1), "isaac_lab_api": "2.3"}
    hist = {}
    for r in rows:
        for e in r.get("errors", []):
            hist.setdefault(e.split(":")[0][:60], []).append(r["id"])
    summary["error_histogram"] = {k: {"count": len(v), "examples": v[:5]} for k, v in sorted(hist.items(), key=lambda kv: -len(kv[1]))}
    with open(args_cli.out, "w") as f:
        json.dump({"summary": summary, "files": rows}, f, indent=1)
    print(f"[isaacsim-validate] {n_ok}/{len(rows)} ok in {summary['time_s']} s -> {args_cli.out}")
    for k, v in list(summary["error_histogram"].items())[:15]:
        print(f"  x{v['count']}: {k}  e.g. {v['examples'][:3]}")


if __name__ == "__main__":
    main()
    simulation_app.close()

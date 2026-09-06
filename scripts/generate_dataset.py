#!/usr/bin/env python
"""Generate the full DoorBench dataset: 1000 doors x (MJCF, URDF, USD, JSON) + QA + thumbnails + manifest.

Usage:
  python scripts/generate_dataset.py --out assets --workers 8 [--limit N] [--ids db0001_swing_single,...] [--no-thumbs]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from doorbench.benchmark_eligibility import benchmark_eligibility, collection_counts
from doorbench.spec import generate_all
from doorbench.build import export_door, build_model
from doorbench import physics as P
from doorbench.qa import run_qa, render_thumbnails


def one(args):
    spec, out_root, hardware_dir, do_thumbs, formats = args
    t0 = time.time()
    try:
        summ = export_door(spec, out_root, hardware_dir, formats=formats)
        door_dir = os.path.join(out_root, spec["id"])
        with open(os.path.join(door_dir, "model.json")) as f:
            meta = json.load(f)["meta"]
        with open(os.path.join(door_dir, "spec.json")) as f:
            phys = json.load(f)["physics"]
        qa = run_qa(spec, door_dir, meta, summ["files"], phys)
        thumbs = []
        if do_thumbs:
            try:
                thumbs = render_thumbnails(summ["files"]["mjcf"]["full"], door_dir)
                thumbs += render_thumbnails(summ["files"]["mjcf"]["full"], door_dir, cams=("iso",), open_fraction=0.6, primary_joint=meta.get("primary_joint"))
            except Exception as e:
                qa["metrics"]["thumb_error"] = str(e)[:200]
        with open(os.path.join(door_dir, "qa.json"), "w") as f:
            json.dump(qa, f, indent=1)
        row = {
            "id": spec["id"], "index": spec["index"], "family": spec["family"], "context": spec.get("context", ""), "use_case": spec.get("use_case", ""),
            "task": spec.get("task"), "difficulty": None, "mass_kg": round(summ["mass_kg"], 2), "leaf": {k: spec["leaf"][k] for k in ("width", "height", "thickness", "slab", "panel_style")},
            "operator": spec["operator"]["model"], "latch": spec["latch"]["model"], "lock": spec["lock"]["model"], "lock_engaged": spec["lock"].get("engaged", False),
            "robot_side_release": spec["lock"].get("robot_side_release", True), "closer": spec["closer"]["model"], "hinge": spec["hinge"]["model"], "condition": spec["condition"],
            "swing": "push" if spec["robot"].get("is_push") else "pull", "hinge_side": spec["hinge"]["side"], "extras": spec.get("extras", []), "tags": spec.get("tags", []),
            "benchmark_eligibility": benchmark_eligibility(spec),
            "n_bodies": summ["n_bodies"], "n_joints": summ["n_joints"], "benchmark": summ.get("benchmark"), "signed_off": qa["signed_off"], "qa_failed": [k for k, v in qa["checks"].items() if not v],
            "thumbs": [os.path.relpath(t, out_root) for t in thumbs], "files": {k: ({kk: os.path.relpath(vv, out_root) for kk, vv in v.items()} if isinstance(v, dict) else (os.path.relpath(v, out_root) if isinstance(v, str) and os.path.exists(v) else v)) for k, v in summ["files"].items()},
            "physics_summary": {
                "hinge_friction_Nm": phys.get("hinge", {}).get("coulomb_torque_Nm"), "damping": phys.get("hinge", {}).get("total_damping_symmetric"),
                "closer_preload_Nm": phys.get("closer", {}).get("spring_preload_Nm"), "closer_k": phys.get("closer", {}).get("spring_stiffness_Nm_per_rad"), "closer_size": phys.get("closer", {}).get("en_size"),
                "roller_friction_N": phys.get("roller", {}).get("coulomb_force_N"), "latch_throw_m": phys.get("latch", {}).get("throw_m"),
                "opening_force_start_N": phys.get("compliance", {}).get("opening_force_start_N"), "ada_ok": phys.get("compliance", {}).get("ada_interior_5lbf_ok"),
            },
            "time_s": round(time.time() - t0, 2),
        }
        from doorbench.spec import difficulty
        row["difficulty"] = difficulty(spec, phys)
        return row
    except Exception as e:
        return {"id": spec["id"], "family": spec["family"], "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:], "signed_off": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--worker-timeout",type=float,default=900.,help="Seconds without a completed worker before serial recovery; detailed native chain tests can take several minutes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--families", default="")
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--formats", default="mjcf,urdf,usd,json")
    ap.add_argument("--seed", type=int, default=20260903)
    a = ap.parse_args()
    if a.workers<1 or not math.isfinite(a.worker_timeout) or a.worker_timeout<=0:
        ap.error('workers and worker-timeout must be positive and finite')
    specs = generate_all(a.seed)
    if a.ids:
        want = set(a.ids.split(","))
        specs = [s for s in specs if s["id"] in want]
    if a.families:
        want = set(a.families.split(","))
        specs = [s for s in specs if s["family"] in want]
    if a.limit:
        specs = specs[: a.limit]
    out_root = os.path.abspath(os.path.join(a.out, "doors"))
    hardware_dir = os.path.abspath(os.path.join(a.out, "hardware"))
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(hardware_dir, exist_ok=True)
    # pre-populate the shared hardware library serially (avoids write races)
    t0 = time.time()
    from doorbench.build import write_hardware_meshes
    for s in specs:
        try:
            write_hardware_meshes(build_model(s), hardware_dir)
        except Exception as e:
            print("hardware prepass error", s["id"], e)
    print(f"hardware library: {len(os.listdir(hardware_dir))} meshes in {time.time() - t0:.1f}s", flush=True)
    formats = tuple(a.formats.split(","))
    jobs = [(s, out_root, hardware_dir, not a.no_thumbs, formats) for s in specs]
    rows = []
    t0 = time.time()

    def report(i, row):
        if "error" in row:
            print(f"[{i + 1}/{len(jobs)}] ERROR {row['id']}: {row['error']}", flush=True)
        elif len(jobs)<=25 or (i + 1) % 25 == 0 or not row["signed_off"]:
            print(f"[{i + 1}/{len(jobs)}] {row['id']} signed_off={row['signed_off']} failed={row.get('qa_failed')} ({time.time() - t0:.0f}s)", flush=True)

    # crash-tolerant pool: a worker that dies (e.g. offscreen GL) loses only its current job, which is redone serially
    import multiprocessing
    pool = Pool(a.workers, maxtasksperchild=64)
    it = pool.imap_unordered(one, jobs, chunksize=1)
    while True:
        try:
            row = it.next(timeout=a.worker_timeout)
        except StopIteration:
            break
        except multiprocessing.TimeoutError:
            print("worker timeout (dead worker?) - terminating pool and finishing the rest serially", flush=True)
            break
        rows.append(row)
        report(len(rows) - 1, row)
    pool.terminate()
    pool.join()
    done = {r["id"] for r in rows}
    for job in jobs:
        if job[0]["id"] not in done:
            row = one(job)
            rows.append(row)
            report(len(rows) - 1, row)
    rows.sort(key=lambda r: r.get("index", 0))
    manifest = {
        "name": "DoorBench", "version": "0.1.0", "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": a.seed, "n_doors": len(rows),
        **collection_counts(rows),
        "n_signed_off": sum(1 for r in rows if r.get("signed_off")), "families": sorted({r["family"] for r in rows}), "doors": rows,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    ok = manifest["n_signed_off"]
    print(f"done: {len(rows)} doors, {ok} signed off, {len(rows) - ok} need attention, {time.time() - t0:.0f}s")
    fails = {}
    for r in rows:
        for k in r.get("qa_failed", []) or (["error"] if "error" in r else []):
            fails.setdefault(k, []).append(r["id"])
    for k, v in sorted(fails.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k}: {len(v)}  e.g. {v[:5]}")


if __name__ == "__main__":
    main()

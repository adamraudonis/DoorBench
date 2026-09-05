#!/usr/bin/env python
"""Run the two geometric gates over the generated dataset and print grouped failures.

  clearance          nothing interpenetrates anywhere in the travel
  running_clearance  nothing TOUCHES either: every moving collider keeps its running clearance from the static
                     structure at rest and through the sweep (3 mm at jambs / head, 6 mm over the floor, 10 mm on a
                     revolving / turnstile rotor; seals, bearings, latches and stops are allow-listed by semantics)

Usage: python scripts/clearance_report.py [--assets assets] [--workers 8] [--families a,b] [--ids id1,id2] [--top 40]
       [--gate both|clearance|running] [--near 0.002]   # --near also lists passing pairs within N m of their minimum
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from doorbench.clearance import run_clearance


def one(door_dir):
    r = run_clearance(door_dir)
    return os.path.basename(door_dir), r


def canon(name: str) -> str:
    return re.sub(r"_-?\d+", "_N", re.sub(r"^(leaf_[a-z]+_|leaf_|lower_|upper_|section_\d+_|panel_\d+_|wing_\d+_)", "L_", name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--families", default=""); ap.add_argument("--ids", default=""); ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--gate", default="both", choices=("both", "clearance", "running"))
    ap.add_argument("--near", type=float, default=0.0, help="also list passing moving/static pairs within this margin (m) of their required minimum")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.assets, "manifest.json")))
    doors = man["doors"]
    if a.families:
        doors = [d for d in doors if d["family"] in a.families.split(",")]
    if a.ids:
        doors = [d for d in doors if d["id"] in a.ids.split(",")]
    dirs = [os.path.join(a.assets, "doors", d["id"]) for d in doors]
    fam = {d["id"]: d["family"] for d in doors}
    with Pool(a.workers) as pool:
        res = dict(pool.map(one, dirs, chunksize=4))
    if a.gate in ("both", "clearance"):
        n_ok = sum(1 for r in res.values() if r["ok"])
        print(f"clearance: {n_ok}/{len(res)} doors clean")
        groups = collections.defaultdict(list)
        for sid, r in res.items():
            for f in r["failures"]:
                key = (fam[sid], f["config"].split(":")[0], canon(f["geoms"][0]) if f["geoms"] else "", canon(f["geoms"][1]) if len(f["geoms"]) > 1 else f.get("error", ""))
                groups[key].append((sid, f["depth"], f["config"], f["q"]))
        for key, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))[: a.top]:
            worst = max(v, key=lambda x: x[1])
            print(f"{len(v):4d}  {key[0]:18s} {key[1]:8s} {key[2]} vs {key[3]}   worst {worst[1]*1000:.0f} mm in {worst[0]} ({worst[2]} q={worst[3]})")
    if a.gate in ("both", "running"):
        rr = {sid: r["running"] for sid, r in res.items()}
        n_ok = sum(1 for r in rr.values() if r["ok"])
        print(f"running_clearance: {n_ok}/{len(rr)} doors clean")
        groups = collections.defaultdict(list)
        for sid, r in rr.items():
            for f in r["failures"]:
                key = (fam[sid], "/".join(f.get("sem", [])), canon(f.get("moving", "")), canon(f.get("static", "")) or f.get("error", ""))
                groups[key].append((sid, f["gap"], f["required"], f["config"]))
        for key, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))[: a.top]:
            worst = min(v, key=lambda x: x[1] - x[2])
            print(f"{len(v):4d}  {key[0]:18s} {key[1]:14s} {key[2]} vs {key[3]}   worst {worst[1]*1000:.2f} mm (need {worst[2]*1000:.0f}) in {worst[0]} ({worst[3]})")
    if a.near > 0:
        from doorbench.clearance import run_running_clearance
        tight = collections.defaultdict(list)
        with Pool(a.workers) as pool:
            for sid, det in zip([os.path.basename(d) for d in dirs], pool.map(_near_one, dirs, chunksize=4)):
                for p in det:
                    if 0 <= p["gap"] - p["required"] <= a.near:
                        tight[(fam[sid], canon(p["moving"]), canon(p["static"]))].append((sid, p["gap"] - p["required"]))
        print(f"pairs passing within {a.near*1000:.1f} mm of their minimum:")
        for key, v in sorted(tight.items(), key=lambda kv: -len(kv[1]))[: a.top]:
            print(f"{len(v):4d}  {key[0]:18s} {key[1]} vs {key[2]}   tightest {min(x[1] for x in v)*1000:.2f} mm margin")
    if a.json:
        json.dump(res, open(a.json, "w"))


def _near_one(door_dir):
    from doorbench.clearance import run_running_clearance
    return [p for p in run_running_clearance(door_dir, record_all=True).get("pairs", []) if p["required"] > 0]


if __name__ == "__main__":
    main()

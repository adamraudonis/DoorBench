#!/usr/bin/env python
"""Run the kinematic clearance gate over the generated dataset and print grouped failures.

Usage: python scripts/clearance_report.py [--assets assets] [--workers 8] [--families a,b] [--ids id1,id2] [--top 40]
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
    return re.sub(r"_-?\d+", "_N", re.sub(r"^(leaf_[a-z]+_|leaf_|lower_|upper_|section_\d+_)", "L_", name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--families", default=""); ap.add_argument("--ids", default=""); ap.add_argument("--top", type=int, default=40)
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
    if a.json:
        json.dump(res, open(a.json, "w"))


if __name__ == "__main__":
    main()

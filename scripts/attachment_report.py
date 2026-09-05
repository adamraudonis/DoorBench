#!/usr/bin/env python
"""Run the ATTACHMENT gate over the generated dataset and print grouped findings.

Nothing floats: every static geom lands on the structure, every body touches what carries it (at rest and through
its travel), every body's own geoms form one connected part, every connect/weld equality is authored closed, and no
geom is degenerate or duplicated.  See doorbench/attachment.py for the rules and their tolerances.

Usage: python scripts/attachment_report.py [--assets assets] [--workers 8] [--families a,b] [--ids id1,id2]
       [--top 40] [--rules static_detached,body_detached] [--json out.json] [--list]
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from doorbench.attachment import run_attachment


def one(door_dir):
    return os.path.basename(door_dir), run_attachment(door_dir)


def canon(name: str) -> str:
    """Collapse per-door numbering so the same defect in 130 doors groups into one line."""
    name = re.sub(r"^(leaf_[ab]_|leaf_|lower_|upper_|section_\d+_|panel_\d+_|wing_\d+_|slat_\d+_|strip_\d+_)", "L_", name)
    return re.sub(r"_-?\d+", "_N", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--families", default="")
    ap.add_argument("--ids", default="")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--rules", default="", help="only report these rules (comma separated)")
    ap.add_argument("--list", action="store_true", help="list every failing door id per group")
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
    rules = set(a.rules.split(",")) if a.rules else None
    with Pool(a.workers) as pool:
        res = dict(pool.map(one, dirs, chunksize=4))
    n_ok = sum(1 for r in res.values() if r["ok"])
    print(f"attachment: {n_ok}/{len(res)} doors clean")
    per_rule = collections.Counter()
    doors_per_rule = collections.defaultdict(set)
    for sid, r in res.items():
        for rule, n in r["by_rule"].items():
            per_rule[rule] += n
            doors_per_rule[rule].add(sid)
    for rule, n in per_rule.most_common():
        print(f"  {rule:20s} {n:6d} findings in {len(doors_per_rule[rule]):4d} doors")
    groups = collections.defaultdict(list)
    for sid, r in res.items():
        for f in r["findings"]:
            if rules and f["rule"] not in rules:
                continue
            key = (f["rule"], fam[sid], "+".join(canon(n) for n in f["names"][:3]))
            groups[key].append((sid, f.get("gap"), f["detail"]))
    print()
    for key, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))[: a.top]:
        worst = max(v, key=lambda x: x[1] if x[1] is not None else 0.0)
        gap = f"{worst[1] * 1000:.1f} mm" if worst[1] is not None else "-"
        print(f"{len(v):4d}  {key[0]:20s} {key[1]:18s} {key[2][:70]:70s} worst {gap:>10s} in {worst[0]}")
        if a.list:
            print("       " + " ".join(sorted({x[0] for x in v})))
    if a.json:
        json.dump(res, open(a.json, "w"))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run the attachment gate (doorbench/attachment.py) over the generated dataset and print grouped findings.

Usage: python scripts/attachment_report.py [--assets assets] [--workers 8] [--families a,b] [--ids id1,id2]
                                           [--top 40] [--json out.json] [--rules detached,static_floating] [--domain door|closer]

Findings on closer / power-operator / gas-strut parts carry domain="closer" and are summarised separately from the
rest of the door ("door" domain), because the closer mechanism model is maintained on its own.
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from doorbench.attachment import run_attachment


def one(door_dir):
    return os.path.basename(door_dir), run_attachment(door_dir)


def canon(name: str) -> str:
    """Collapse per-leaf / per-index name variants so findings group across doors."""
    return re.sub(r"_-?\d+", "_N", re.sub(r"^(leaf_[a-z]+_|leaf_|lower_|upper_|section_\d+_|panel_\d+_\d+_|strip_\d+_|wing_\d+_|dog_\d+_|bolt_\d+_)", "L_", name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--families", default=""); ap.add_argument("--ids", default=""); ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--json", default=""); ap.add_argument("--rules", default="", help="only report these rules (comma list)")
    ap.add_argument("--domain", default="", help="only report findings of this domain (door | closer)")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.assets, "manifest.json")))
    doors = man["doors"]
    if a.families:
        doors = [d for d in doors if d["family"] in a.families.split(",")]
    if a.ids:
        doors = [d for d in doors if d["id"] in a.ids.split(",")]
    dirs = [os.path.join(a.assets, "doors", d["id"]) for d in doors]
    fam = {d["id"]: d["family"] for d in doors}
    if a.workers > 1 and len(dirs) > 1:
        with Pool(a.workers) as pool:
            res = dict(pool.map(one, dirs, chunksize=2))
    else:
        res = dict(one(p) for p in dirs)
    rules = set(a.rules.split(",")) if a.rules else None
    n_ok = sum(1 for r in res.values() if r["ok"])
    n_ok_door = sum(1 for r in res.values() if r.get("ok_door", r["ok"]))
    n_closer = sum(1 for r in res.values() if r.get("n_closer_findings", 0) > 0)
    print(f"attachment: {n_ok}/{len(res)} doors clean; {n_ok_door}/{len(res)} clean outside the closer mechanism; {n_closer} doors with closer-domain findings")
    by_rule = collections.Counter()
    by_rule_fam = collections.defaultdict(collections.Counter)
    doors_by_rule = collections.defaultdict(set)
    groups = collections.defaultdict(list)
    for sid, r in res.items():
        for f in r["findings"]:
            if rules and f["rule"] not in rules:
                continue
            if a.domain and f.get("domain", "door") != a.domain:
                continue
            by_rule[f["rule"]] += 1
            by_rule_fam[f["rule"]][fam[sid]] += 1
            doors_by_rule[f["rule"]].add(sid)
            key = (f.get("domain", "door"), f["rule"], f.get("kind", ""), fam[sid], canon(f["body"]) if f["body"] else "", canon(f["geoms"][0]) if f["geoms"] else "")
            groups[key].append((sid, f["dist"], f["config"], f["why"]))
    print("per rule (findings / doors):", {k: f"{v}/{len(doors_by_rule[k])}" for k, v in by_rule.most_common()})
    print("per rule and family:", {k: dict(v.most_common(8)) for k, v in by_rule_fam.items()})
    for key, v in sorted(groups.items(), key=lambda kv: (kv[0][0] != "door", -len(kv[1])))[: a.top]:
        worst = max(v, key=lambda x: x[1])
        print(f"{len(v):4d}  {key[0]:6s} {key[1]:18s} {key[2]:14s} {key[3]:18s} {key[4]} / {key[5]}   e.g. {worst[0]} ({worst[2]}): {worst[3][:150]}")
    if a.json:
        json.dump(res, open(a.json, "w"))


if __name__ == "__main__":
    main()

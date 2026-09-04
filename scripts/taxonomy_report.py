#!/usr/bin/env python
"""Emit the DoorBench taxonomy hierarchy (motion class -> family -> variant) with per-node counts, representative
doors + thumbnails, hardware / size summaries and the family x mechanism-kind relationship matrices.

The JSON feeds the site's "Hierarchy" page (viewer/public/taxonomy.json, committed like assets/manifest.json and
regenerated after every dataset build); `--tree` prints the hierarchy as an indented text tree and `--md` prints the
markdown tables used in docs/TAXONOMY.md so the document can be refreshed from the data.

Usage:
  python scripts/taxonomy_report.py                       # -> viewer/public/taxonomy.json
  python scripts/taxonomy_report.py --tree                # print the tree with counts
  python scripts/taxonomy_report.py --md                  # print markdown tables (family cards, matrices)
  python scripts/taxonomy_report.py --assets assets --out viewer/public/taxonomy.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from doorbench import taxonomy as T  # noqa: E402


def load_specs(assets: str, rows: list[dict]) -> dict:
    specs = {}
    for r in rows:
        p = os.path.join(assets, "doors", r["id"], "spec.json")
        if os.path.exists(p):
            with open(p) as f:
                specs[r["id"]] = json.load(f)
    return specs


def fmt_range(lo: float, hi: float, digits: int = 2) -> str:
    return f"{lo:.{digits}f}" if abs(hi - lo) < 10 ** -digits else f"{lo:.{digits}f}-{hi:.{digits}f}"


def top(d: dict, n: int = 4) -> str:
    items = [(k, v) for k, v in d.items() if k != "none"]
    if not items:
        return "-"
    return ", ".join(f"{k.replace('_', ' ')} ({v})" for k, v in items[:n])


def print_tree(h: dict) -> None:
    print(f"DoorBench taxonomy: {h['n_doors']} doors, {len(h['motion_classes'])} motion classes, "
          f"{sum(len(c['families']) for c in h['motion_classes'])} families, "
          f"{sum(len(f['variants']) for c in h['motion_classes'] for f in c['families'])} variants")
    for c in h["motion_classes"]:
        print(f"{c['label']}  [{c['count']}]")
        for f in c["families"]:
            s = f["sizes"]
            print(f"  {f['label']} ({f['id']})  [{f['count']}]  {f['kinematics_type']}, leaves {f['leaves_note']}; "
                  f"{fmt_range(*s['leaf_width_m'])} x {fmt_range(*s['leaf_height_m'])} m, {fmt_range(*s['mass_kg'], 0)} kg")
            for v in f["variants"]:
                print(f"    {v['label']} ({v['id']})  [{v['count']}]  op: {top(v['hardware']['operator_kind'], 3)}; "
                      f"latch: {top(v['hardware']['latch_kind'], 2)}; lock: {top(v['hardware']['lock_kind'], 2)}; closer: {top(v['hardware']['closer_kind'], 2)}")


def print_markdown(h: dict) -> None:
    print("## Hierarchy with counts\n")
    print("| Motion class | Family | Variant | Doors | Kinematics | Leaves | Typical operators | Latch kinds | Lock kinds | Closers | Leaf size (m) | Mass (kg, per leaf) |")
    print("|---|---|---|---:|---|---|---|---|---|---|---|---|")
    for c in h["motion_classes"]:
        print(f"| **{c['label']}** | | | **{c['count']}** | | | | | | | | |")
        for f in c["families"]:
            s = f["sizes"]
            leaves = ", ".join(f"{k} ({v})" for k, v in f["leaves"].items())
            print(f"| | **{f['label']}** `{f['id']}` | | **{f['count']}** | {f['kinematics_type']} | {leaves} | {top(f['hardware']['operator_kind'])} | {top(f['hardware']['latch_kind'], 3)} | {top(f['hardware']['lock_kind'], 3)} | {top(f['hardware']['closer_kind'], 3)} | {fmt_range(*s['leaf_width_m'])} x {fmt_range(*s['leaf_height_m'])} | {fmt_range(*s['mass_kg'], 0)} |")
            for v in f["variants"]:
                s = v["sizes"]
                leaves = ", ".join(f"{k} ({n})" for k, n in v["leaves"].items())
                print(f"| | | {v['label']} `{v['id']}` | {v['count']} | {', '.join(v['kinematics'])} | {leaves} | {top(v['hardware']['operator_kind'], 3)} | {top(v['hardware']['latch_kind'], 2)} | {top(v['hardware']['lock_kind'], 2)} | {top(v['hardware']['closer_kind'], 2)} | {fmt_range(*s['leaf_width_m'])} x {fmt_range(*s['leaf_height_m'])} | {fmt_range(*s['mass_kg'], 0)} |")
    print()
    for mech in ("closer", "latch", "lock", "operator", "hinge"):
        rel = h["relations"][mech]
        print(f"## Families x {mech} kinds\n")
        print("| family | " + " | ".join(rel["cols"]) + " |")
        print("|---|" + "---:|" * len(rel["cols"]))
        for fam, rowv in zip(rel["rows"], rel["matrix"]):
            print(f"| {fam} | " + " | ".join(str(x) if x else "·" for x in rowv) + " |")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--out", default="viewer/public/taxonomy.json")
    ap.add_argument("--tree", action="store_true", help="print the hierarchy as text instead of writing JSON")
    ap.add_argument("--md", action="store_true", help="print markdown tables instead of writing JSON")
    ap.add_argument("--reps", type=int, default=6, help="representative doors per node")
    a = ap.parse_args()
    with open(os.path.join(a.assets, "manifest.json")) as f:
        manifest = json.load(f)
    rows = [r for r in manifest["doors"] if not r.get("error")]
    specs = load_specs(a.assets, rows)
    h = T.build_hierarchy(rows, specs, n_reps=a.reps)
    h["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    h["manifest_generated"] = manifest.get("generated")
    h["seed"] = manifest.get("seed")
    h["version"] = manifest.get("version")
    if a.tree:
        print_tree(h)
        return
    if a.md:
        print_markdown(h)
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(h, f, separators=(",", ":"))
    n_var = sum(len(f["variants"]) for c in h["motion_classes"] for f in c["families"])
    print(f"wrote {a.out}: {h['n_doors']} doors, {len(h['motion_classes'])} motion classes, "
          f"{sum(len(c['families']) for c in h['motion_classes'])} families, {n_var} variants, "
          f"{len(h['shared_mechanisms'])} shared mechanism kinds, {os.path.getsize(a.out) / 1024:.0f} kB")


if __name__ == "__main__":
    main()

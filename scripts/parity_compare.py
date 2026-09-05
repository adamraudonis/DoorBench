#!/usr/bin/env python
"""Compare the MuJoCo reference with the Isaac / PhysX parity runs and classify every door.

  PYTHONPATH=$PWD python scripts/parity_compare.py [--mujoco results/parity/mujoco.json] [--isaac results/parity/isaac_full.json results/parity/isaac_rl.json]

Writes results/parity/compare.json  {door_id: {"full": verdict, "rl": verdict, "grade": worst}} and
results/parity/compare_summary.json (grades, codes with examples, per-phase agreement, worst doors) and prints a table.
Verdict / codes: doorbench.parity.protocol.compare_door (OK, PHYSX_NO_OPEN, PHYSX_HOLD_FAIL, EXPORT_WELD_MISSING, SETTLE_DRIFT,
LIMIT_VIOLATION, NAN, CLOSER_NO_RETURN, LATCH_NO_RETURN, RELATCH_FAIL, MUJOCO_FAIL, METRIC_DELTA, RL_CANON, STRUCTURE_FAIL, LOAD_FAIL, MISSING).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from doorbench.parity import protocol as P  # noqa: E402

RANK = {"A": 0, "B": 1, "C": 2, "X": 3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mujoco", default=os.path.join(ROOT, "results", "parity", "mujoco.json"))
    ap.add_argument("--isaac", nargs="+", default=[os.path.join(ROOT, "results", "parity", "isaac_full.json"), os.path.join(ROOT, "results", "parity", "isaac_rl.json")])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "parity", "compare.json"))
    args = ap.parse_args()
    with open(args.mujoco) as f:
        mj = json.load(f)["doors"]
    isaac = {}
    for p in args.isaac:
        if not os.path.isfile(p):
            print(f"[parity-compare] missing {p}, skipped")
            continue
        with open(p) as f:
            data = json.load(f)
        kind = data.get("meta", {}).get("kind") or ("rl" if "rl" in os.path.basename(p) else "full")
        isaac[kind] = data["doors"]
    out, per_kind = {}, {k: [] for k in isaac}
    for kind, doors in isaac.items():
        pkind = "usd_rl" if kind == "rl" else "usd_full"
        for did, px in doors.items():
            ref = mj.get(did)
            if ref is None:
                v = {"door_id": did, "kind": pkind, "grade": "X", "codes": ["MISSING"], "phases": {}, "note": "no MuJoCo reference"}
            else:
                v = P.compare_door(ref["inputs"], ref, px, kind=pkind)
                v["family"] = ref["inputs"]["family"]
            out.setdefault(did, {})[kind] = v
            per_kind[kind].append(v)
    for did, row in out.items():
        row["grade"] = max((v["grade"] for k, v in row.items() if k in isaac), key=lambda g: RANK[g], default="X")
        row["codes"] = sorted({c for k, v in row.items() if k in isaac for c in v["codes"]})
    summary = {"n_doors": len(out), "per_kind": {k: P.summarize(v) for k, v in per_kind.items()},
               "grades": {g: sum(1 for r in out.values() if r["grade"] == g) for g in "ABCX"},
               "by_family": {}}
    for did, row in out.items():
        fam = next((v.get("family") for k, v in row.items() if isinstance(v, dict) and v.get("family")), "?")
        summary["by_family"].setdefault(fam, {"n": 0, "A": 0, "B": 0, "C": 0, "X": 0})
        summary["by_family"][fam]["n"] += 1
        summary["by_family"][fam][row["grade"]] += 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[parity-compare] {len(out)} doors: grades {summary['grades']} -> {args.out}")
    for kind, s in summary["per_kind"].items():
        print(f"  {kind}: grades {s['grades']}")
        for code, info in sorted(s["codes"].items(), key=lambda kv: -kv[1]["count"]):
            print(f"    x{info['count']:4d} {code:22s} e.g. {info['examples'][:4]}")
        for p, st in s["phases"].items():
            print(f"    {p:8s} agree {st['agree']:4d}  disagree {st['disagree']:4d}  n/a {st['na']:4d}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""MuJoCo reference run of the Isaac parity protocol over the dataset (CPU, local).

  PYTHONPATH=$PWD python scripts/parity_reference_mujoco.py --doors all --workers 8
  PYTHONPATH=$PWD python scripts/parity_reference_mujoco.py --doors db0002_swing_single,db0021_swing_single
  PYTHONPATH=$PWD python scripts/parity_reference_mujoco.py --doors family:swing_single --limit 20

Door selection strings: all | family:a,b | id,id,... | @ids.txt | random-50 | one-per-family[-N].
Runs ``doorbench.parity.protocol`` on assets/doors/<id>/door.xml (the same file and drive semantics as qa.py), writes

  results/parity/mujoco.json            {"meta": {...}, "doors": {door_id: record}} - inputs, per-phase status + metrics,
                                        5 Hz primary/operator/bolt curves; consumed by scripts/isaaclab/isaac_parity.py
  results/parity/mujoco_summary.json    counts per phase / family, QA-reproduction mismatches, failing doors
  results/parity/cache/mujoco/<id>.json full 30 Hz curves of every joint (resume cache, not committed)

Resumable: doors whose cache entry matches the protocol version + inputs hash are not re-simulated (--force to redo).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from doorbench.parity import protocol as P  # noqa: E402
from doorbench.parity.mujoco_runner import run_door, compact_record  # noqa: E402


def _git_commit() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _mujoco_version() -> str | None:
    try:
        import mujoco
        return mujoco.__version__
    except Exception:
        return None


def _dataset_stamp(assets: str) -> dict:
    """Which dataset the reference was measured on (the manifest's own stamp + its mtime), so a report can say so."""
    try:
        with open(os.path.join(assets, "manifest.json")) as f:
            man = json.load(f)
        return {"n_doors": man.get("n_doors"), "version": man.get("version"), "generated": man.get("generated"),
                "manifest_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(os.path.join(assets, "manifest.json"))))}
    except Exception:
        return {}


def select_ids(spec: str, assets: str, seed: int = 0) -> list[str]:
    with open(os.path.join(assets, "manifest.json")) as f:
        man = json.load(f)
    rows = man["doors"]
    spec = spec.strip()
    if spec == "all":
        return [d["id"] for d in rows]
    if spec.startswith("random-"):
        return sorted(random.Random(seed).sample([d["id"] for d in rows], int(spec.split("-")[1])))
    if spec.startswith("one-per-family"):
        n = int(spec.split("-")[-1]) if spec[-1].isdigit() else 1
        by = {}
        for d in rows:
            by.setdefault(d["family"], []).append(d["id"])
        return [i for fam in sorted(by) for i in by[fam][:n]]
    if spec.startswith("family:"):
        fams = set(spec[len("family:"):].split(","))
        return [d["id"] for d in rows if d["family"] in fams]
    if spec.startswith("@"):
        with open(spec[1:]) as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]
    known = {d["id"] for d in rows}
    ids = [s for s in spec.split(",") if s]
    bad = [i for i in ids if i not in known]
    if bad:
        raise SystemExit(f"unknown door ids: {bad[:5]}")
    return ids


def _work(args):
    door_dir, dt, cache_dir, force = args
    door_id = os.path.basename(door_dir)
    try:
        rec = run_door(door_dir, dt=dt, cache_dir=cache_dir, force=force)
        lite = compact_record(rec)
        lite.pop("cached", None)      # a resume detail, not a property of the door
        return door_id, lite, None
    except Exception as e:  # never lose the run to one door
        return door_id, None, f"{type(e).__name__}: {e}"


INFO_FAIL_METRICS = ("q_at_1s", "hold_displacement", "t_free", "relatch_closed_angle", "relatch_repush_angle", "bolt_after_release_m")


def summarize(doors: dict, errors: dict) -> dict:
    fam_stat, phase_stat = {}, {p: {} for p in P.PHASES}
    repro_bad, failing, info_fail = [], [], []
    for did, r in doors.items():
        fam = r["inputs"]["family"]
        fam_stat.setdefault(fam, {"n": 0, "ok": 0})
        fam_stat[fam]["n"] += 1
        fam_stat[fam]["ok"] += int(r["ok"])
        for p, row in r["phases"].items():
            phase_stat[p][row["status"]] = phase_stat[p].get(row["status"], 0) + 1
            if row["status"] == "fail" and row.get("informational"):
                # the reference itself says the door does not do what its family should (e.g. a revolving door that does
                # not turn under the push): not graded, but the whole point of the gate is to surface these
                info_fail.append({"door_id": did, "family": fam, "phase": p, "expected": row["expected"],
                                  **{k: row["metrics"].get(k) for k in INFO_FAIL_METRICS if row["metrics"].get(k) is not None}})
        if not r["qa_reproduction"]["ok"]:
            repro_bad.append({"door_id": did, "mismatches": r["qa_reproduction"]["mismatches"]})
        if not r["ok"]:
            failing.append({"door_id": did, "failed": {p: row["metrics"] and {k: row["metrics"].get(k) for k in ("hold_displacement", "opened", "bolt_after_release_m", "relatch_closed_angle", "closer_final_angle", "locked_displacement", "settle_drift", "warnings")} for p, row in r["phases"].items() if row["status"] == "fail"}})
    n_lim = sum(1 for r in doors.values() if r["limits"]["violations"])
    return {"protocol_version": P.PROTOCOL_VERSION, "n_doors": len(doors), "n_ok": sum(1 for r in doors.values() if r["ok"]), "n_errors": len(errors), "errors": errors,
            "phases": phase_stat, "families": fam_stat, "qa_reproduction": {"n_compared": sum(1 for r in doors.values() if r["qa_reproduction"]["available"]), "mismatching_doors": repro_bad},
            "doors_with_limit_overshoot": n_lim, "failing_doors": failing, "informational_fails": info_fail,
            "wall_time_s_total": round(sum(r.get("wall_time_s", 0) for r in doors.values()), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doors", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--assets", default=os.path.join(ROOT, "assets"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "parity", "mujoco.json"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "results", "parity", "cache", "mujoco"))
    ap.add_argument("--dt", type=float, default=None, help="override the MJCF timestep (sensitivity rerun, e.g. 0.001)")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    ap.add_argument("--no-resume", action="store_true", help="do not merge into an existing --out file")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ids = select_ids(args.doors, args.assets, seed=args.seed)
    if args.limit:
        ids = ids[: args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    doors, errors = {}, {}
    if not args.no_resume and os.path.isfile(args.out):
        with open(args.out) as f:
            prev = json.load(f)
        if prev.get("meta", {}).get("protocol_version") == P.PROTOCOL_VERSION and prev.get("meta", {}).get("metrics_version") == P.METRICS_VERSION:
            doors = prev.get("doors", {})
    todo = [(os.path.join(args.assets, "doors", i), args.dt, args.cache, args.force) for i in ids]
    t0 = time.time()
    n_done = 0

    def flush():
        meta = {"protocol_version": P.PROTOCOL_VERSION, "metrics_version": P.METRICS_VERSION, "sim": "mujoco", "kind": "mjcf", "dt": args.dt,
                "engine": {"mujoco": _mujoco_version()}, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_doors": len(doors), "sample_hz": P.SAMPLE_HZ,
                "commit": _git_commit(), "dataset": _dataset_stamp(args.assets)}
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"meta": meta, "doors": doors}, f)
        os.replace(tmp, args.out)
        summ = summarize(doors, errors)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
            json.dump(summ, f, indent=1)
        return summ

    if args.workers <= 1:
        results = map(_work, todo)
        for door_id, rec, err in results:
            n_done += 1
            if err:
                errors[door_id] = err
                print(f"[parity-mujoco] {door_id}: ERROR {err}")
            else:
                doors[door_id] = rec
            if n_done % 25 == 0:
                flush()
                print(f"[parity-mujoco] {n_done}/{len(todo)} doors, {time.time() - t0:.0f} s")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_work, w) for w in todo]
            for fut in as_completed(futs):
                door_id, rec, err = fut.result()
                n_done += 1
                if err:
                    errors[door_id] = err
                    print(f"[parity-mujoco] {door_id}: ERROR {err}")
                else:
                    doors[door_id] = rec
                if n_done % 25 == 0:
                    flush()
                    print(f"[parity-mujoco] {n_done}/{len(todo)} doors, {time.time() - t0:.0f} s")
    summ = flush()
    print(f"[parity-mujoco] {summ['n_ok']}/{summ['n_doors']} doors pass every applicable phase; {len(errors)} errors; "
          f"qa reproduction mismatches: {len(summ['qa_reproduction']['mismatching_doors'])}; "
          f"informational fails: {len(summ['informational_fails'])}; {time.time() - t0:.0f} s -> {args.out}")
    for p, st in summ["phases"].items():
        print(f"  {p:8s} {st}")
    for row in summ["failing_doors"][:15]:
        print(f"  FAIL {row['door_id']}: {list(row['failed'])}")
    for row in summ["qa_reproduction"]["mismatching_doors"][:10]:
        print(f"  QA-mismatch {row['door_id']}: {row['mismatches']}")
    by_fam = {}
    for row in summ["informational_fails"]:
        by_fam.setdefault((row["family"], row["phase"], row["expected"]), []).append(row["door_id"])
    for (fam, p, exp), ids in sorted(by_fam.items()):
        print(f"  INFO-FAIL {fam}/{p} ({exp}): {len(ids)} doors, e.g. {ids[:4]}")


if __name__ == "__main__":
    main()

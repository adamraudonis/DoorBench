#!/usr/bin/env python
"""Adapter: the legacy 40-door Isaac probe (scripts/isaaclab/validate_usd_isaacsim.py output) + the MuJoCo qa.json of
the same doors -> parity result files in the runner shape, so scripts/isaaclab/parity_report.py can render a first
docs/ISAAC_PARITY.md before the shared protocol has run on the GPU.

    PYTHONPATH=$PWD python scripts/isaaclab/probe_to_parity.py \
        --probe results/isaac_validation/usd_validation_isaacsim_40.json --out results/parity/probe
    PYTHONPATH=$PWD python scripts/isaaclab/parity_report.py --results results/parity/probe

Mapping (best effort - the probe is NOT the parity protocol, and the report says so in its header):
  MuJoCo side (qa.json checks / metrics): settle <- settle / settle_drift; hold <- hold | free_opens / hold_displacement;
    operate_open <- actuate_opens / actuate_displacement, operator_travel_reached; release <- latch_returns;
    relatch <- relatch; closer_return <- closer_returns; locked_holds <- locked_holds.
  Isaac side (probe rows): structure <- ok / errors; settle <- stats.settle_drift (max over ALL joints, so operator sag
    shows up here); the single "8 N*m operator + 60 N*m push for 400 steps" phase maps to the phase the MuJoCo QA ran
    for that door (operate_open when it has a releasable operator, locked_holds for locked doors, hold / free_opens for
    doors without an operator) with the QA thresholds.  No curves.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

PROBE_WARNING = ("These inputs come from the legacy 40-door probe (`validate_usd_isaacsim.py`: a fixed 60 N*m / 60 N push plus 8 N*m on the operator for "
                 "400 steps, position targets zeroed every step) adapted to the parity schema by `scripts/isaaclab/probe_to_parity.py`, with qa.json as the "
                 "MuJoCo side. The protocols differ (adaptive QA push, latch coupling, spring targets), so every class below is a hypothesis about the probe as "
                 "much as about the door; the numbers are replaced when `doorbench/parity/protocol.py` runs on the GPU.")


def _read(path: str):
    with open(path) as f:
        return json.load(f)


def mujoco_record(qa: dict, spec: dict | None) -> dict:
    ch, m = qa.get("checks") or {}, qa.get("metrics") or {}
    is_hinge = _is_hinge(spec)
    ph: dict = {}
    if "load_full" in ch:
        ph["structure"] = {"pass": bool(ch["load_full"]) and ch.get("usd_opens", True) is not False, "metrics": {"moving_mass_kg": m.get("moving_mass_kg")}}
    if "settle" in ch:
        ph["settle"] = {"pass": bool(ch["settle"]), "metrics": {"settle_drift": m.get("settle_drift"), "pen0_m": m.get("initial_penetration_m")}}
    if "hold" in ch:
        ph["hold"] = {"pass": bool(ch["hold"]), "expected": "hold", "metrics": {"hold_displacement": m.get("hold_displacement"), "qa_push": m.get("qa_push")}}
    elif "free_opens" in ch:
        ph["hold"] = {"pass": bool(ch["free_opens"]), "expected": "free_opens", "metrics": {"hold_displacement": m.get("hold_displacement"), "qa_push": m.get("qa_push")}}
    if "actuate_opens" in ch:
        ph["operate_open"] = {"pass": bool(ch["actuate_opens"]), "metrics": {"opened": m.get("actuate_displacement"), "operator_travel_reached": m.get("operator_travel_reached")}}
    if "latch_returns" in ch:
        ph["release"] = {"pass": bool(ch["latch_returns"]), "metrics": {"bolt_after_release_m": m.get("bolt_after_release_m")}}
    if "relatch" in ch:
        ph["relatch"] = {"pass": bool(ch["relatch"]), "metrics": {"relatch_closed_angle": m.get("relatch_closed_angle"), "relatch_repush_angle": m.get("relatch_repush_angle")}}
    if "closer_returns" in ch:
        ph["closer_return"] = {"pass": bool(ch["closer_returns"]), "metrics": {"closer_final_angle": m.get("closer_final_angle")}}
    if "locked_holds" in ch:
        ph["locked_holds"] = {"pass": bool(ch["locked_holds"]), "metrics": {"locked_displacement": m.get("locked_displacement")}}
    ph["sanity"] = {"pass": not (m.get("warnings") or []), "metrics": {}}
    return {"phases": ph, "curves": {}, "metrics": {"is_hinge": is_hinge, "qa_push": m.get("qa_push")}, "errors": [], "engine": {"mujoco": qa.get("mujoco_version")}}


def isaac_record(row: dict, qa: dict, spec: dict | None) -> dict:
    ch, m = qa.get("checks") or {}, qa.get("metrics") or {}
    is_hinge = _is_hinge(spec)
    st = row.get("stats") or {}
    errors = list(row.get("errors") or [])
    ph: dict = {"structure": {"pass": bool(row.get("ok")) and not errors, "metrics": {}}}
    drift = st.get("settle_drift")
    if drift is not None:
        ph["settle"] = {"pass": abs(float(drift)) < (0.05 if is_hinge else 0.01), "metrics": {"settle_drift": float(drift)}}
    opened = st.get("opened")
    thr = math.radians(2.0) if is_hinge else 0.015
    thr_free = math.radians(10.0) if is_hinge else 0.05
    max_open = ((spec or {}).get("kinematics") or {}).get("max_open_deg") or 90
    target = math.radians(min(20.0, 0.5 * max_open)) if is_hinge else 0.05
    if opened is not None:
        o = float(opened)
        if "actuate_opens" in ch:
            ph["operate_open"] = {"pass": o > target, "metrics": {"opened": o}}
            ph["hold"] = {"status": "skip", "reason": "probe applies operator torque and push together"}
        elif "locked_holds" in ch:
            ph["locked_holds"] = {"pass": o < thr, "metrics": {"locked_displacement": o}}
        elif "hold" in ch:
            ph["hold"] = {"pass": o < thr, "expected": "hold", "metrics": {"hold_displacement": o}}
        elif "free_opens" in ch:
            ph["hold"] = {"pass": o > thr_free, "expected": "free_opens", "metrics": {"hold_displacement": o}}
        else:
            ph["free_swing"] = {"pass": o > thr_free, "expected": "free_opens_fs", "metrics": {"hold_displacement": o}}
    finite = all(math.isfinite(float(v)) for v in (drift, opened) if v is not None)
    ph["sanity"] = {"pass": finite and not any("non-finite" in e for e in errors), "metrics": {}}
    return {"phases": ph, "curves": {}, "metrics": {"is_hinge": is_hinge, "probe_warnings": len(row.get("warnings") or [])}, "errors": errors}


def _is_hinge(spec: dict | None) -> bool:
    kin = str(((spec or {}).get("kinematics") or {}).get("type") or "hinge")
    return kin.startswith("hinge") or kin == "rotor"


def convert(probe_path: str, assets_dir: str, out_dir: str) -> dict:
    doc = _read(probe_path)
    rows = doc.get("files") or doc.get("rows") or []
    ids = sorted({r["id"] for r in rows if r.get("id")})
    mj, px = {}, {"full": {}, "rl": {}}
    for did in ids:
        ddir = os.path.join(assets_dir, "doors", did)
        qa = _read(os.path.join(ddir, "qa.json")) if os.path.exists(os.path.join(ddir, "qa.json")) else {}
        spec = _read(os.path.join(ddir, "spec.json")) if os.path.exists(os.path.join(ddir, "spec.json")) else None
        if qa:
            mj[did] = mujoco_record(qa, spec)
        for r in rows:
            if r.get("id") == did and r.get("kind") in px:
                px[r["kind"]][did] = isaac_record(r, qa, spec)
    os.makedirs(out_dir, exist_ok=True)
    summ = doc.get("summary") or {}
    mj_ver = next((v["engine"]["mujoco"] for v in mj.values() if v.get("engine", {}).get("mujoco")), None)
    with open(os.path.join(out_dir, "mujoco.json"), "w") as f:
        json.dump({"engine": {"mujoco": mj_ver}, "protocol": {"name": "qa.py checks (qa.json), adapted"}, "doors": mj}, f, indent=1)
    for kind in ("full", "rl"):
        with open(os.path.join(out_dir, f"isaac_{kind}.json"), "w") as f:
            json.dump({"engine": {"isaac_sim": "5.1.0", "isaac_lab": "2.3.2", "api": summ.get("isaac_lab_api")}, "protocol": {"name": "probe_v0 (validate_usd_isaacsim.py), adapted", "warning": PROBE_WARNING},
                       "source": os.path.relpath(probe_path, ROOT) if probe_path.startswith(ROOT) else probe_path, "doors": px[kind]}, f, indent=1)
    return {"n_doors": len(ids), "n_mujoco": len(mj), "n_full": len(px["full"]), "n_rl": len(px["rl"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert the legacy Isaac probe + qa.json into parity result files.")
    ap.add_argument("--probe", default=os.path.join(ROOT, "results", "isaac_validation", "usd_validation_isaacsim_40.json"))
    ap.add_argument("--assets", default=os.environ.get("DOORBENCH_ASSETS", os.path.join(ROOT, "assets")))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "parity", "probe"))
    a = ap.parse_args(argv)
    n = convert(a.probe, a.assets, a.out)
    print(f"[probe-to-parity] {n['n_doors']} doors: mujoco {n['n_mujoco']}, isaac full {n['n_full']}, rl {n['n_rl']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Independently verify the unchanged-timestep control/state replay exactly."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "_contact_replay_archive", ROOT / "doorbench/dexterous/native_transition_archive.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Use a new audit output; preserve existing receipts")
    report = json.loads((a.run / "report.json").read_text())
    dt = round(report["unchanged_physical_settings"]["source_dt_s"], 12)
    raw_path = a.run / f"dt-{dt:.4f}" / "raw-transitions"
    raw = list(archive.NativeTransitionArchive.read(raw_path))
    with np.load(a.run / "held-input.npz", allow_pickle=False) as source:
        pairs = {"qpos": (np.array([r["qpos_after"] for r in raw]), source["source_qpos_after"]),
                 "qvel": (np.array([r["qvel_after"] for r in raw]), source["source_qvel_after"]),
                 "controls": (np.array([r["controls"] for r in raw]), source["controls"])}
        comparisons = {key: {"bitwise_equal": bool(np.array_equal(x, y)),
                            "max_error": float(np.max(np.abs(x - y)))}
                       for key, (x, y) in pairs.items()}
    result = dict(schema="doorbench.convergence-replay-audit.v2",
        scope="Independent state/control equality; no task reclassification",
        comparisons=comparisons, samples=len(raw), physics_dt_s=dt,
        passed=all(r["bitwise_equal"] for r in comparisons.values()),
        sha256={str(path.relative_to(a.run)): hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                for path in (a.run / "held-input.npz", raw_path / "manifest.json", a.run / "report.json")})
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

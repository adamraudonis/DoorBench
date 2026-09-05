#!/usr/bin/env python3
"""Assemble reviewed baby gates and pet-door scope over the published snapshot.

Geometry and render files come from separate generator/Blender outputs. This
only merges their inventories and applies the central benchmark-scope policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doorbench.benchmark_eligibility import benchmark_eligibility, collection_counts, is_benchmark_eligible
from doorbench.benchmark.scenarios import benchmark_summary, build_benchmark
from site_assets import DOOR_FILES, read, verified_files


def prepare(base: Path, baby_gates: Path, out: Path):
    if out.exists():
        raise ValueError("Use a new output directory; the published base is preserved")
    manifest = read(base / "assets/manifest.json")
    gates = read(baby_gates / "assets/manifest.json")
    appearances = read(baby_gates / "appearance-public/index.json")
    gate_ids = {d["id"] for d in manifest["doors"] if d["family"] == "baby_gate"}
    rows = {d["id"]: d for d in gates["doors"]}
    if (len(gate_ids) != 10 or set(rows) != gate_ids or len(rows) != len(gates["doors"])
            or any(d["family"] != "baby_gate" or not d["signed_off"] for d in rows.values())):
        raise ValueError("Expected all 10 reviewed, signed-off baby gates")
    image_rows = {(r["door_id"], r["variant"]): r for r in appearances["renders"]}
    if set(image_rows) != {(door, 0) for door in gate_ids} or any(r.get("blend") for r in image_rows.values()):
        raise ValueError("Expected 10 image-only default Blender renders")
    shutil.copytree(base, out)
    for door in sorted(gate_ids):
        source = baby_gates / "assets/doors" / door
        target = out / "assets/doors" / door
        for name in DOOR_FILES:
            shutil.copy2(source / name, target / name)
        for path in source.glob("thumb_*.jpg"):
            shutil.copy2(path, target / path.name)
        shutil.copytree(baby_gates / "appearance-public" / door,
                        out / "appearance" / door, dirs_exist_ok=True)
        rows[door]["reference_motion_available"] = False
        rows[door]["reference_motion_unavailable_reason"] = (
            "Door geometry was revised; archived motion uses the earlier overhead wall.")
    manifest["doors"] = [rows.get(d["id"], d) for d in manifest["doors"]]
    for door in manifest["doors"]:
        door["benchmark_eligibility"] = benchmark_eligibility(door)
        if not is_benchmark_eligible(door):
            source = base / "assets/doors" / door["id"]
            spec = read(source / "spec.json")
            door["benchmark"] = benchmark_summary(build_benchmark(spec, spec["physics"], read(source / "model.json")))
            door["reference_motion_available"] = False
            door["reference_motion_unavailable_reason"] = door["benchmark_eligibility"]["reason"]
    manifest.update(collection_counts(manifest["doors"]))
    manifest["n_signed_off"] = sum(d.get("signed_off", False) for d in manifest["doors"])
    manifest["collection_revision"] = "baby-gate-headroom-and-pet-supplement-20260905"
    manifest["qa_scope"] = {"newly_verified_doors": sorted(gate_ids),
                            "other_signoffs": "Retained from the immutable base release; not a new all-door certification."}
    (out / "assets/manifest.json").write_text(json.dumps(manifest) + "\n")
    index = read(base / "appearance/index.json")
    index["renders"] = [image_rows.get((r["door_id"], r["variant"]), r) for r in index["renders"]]
    (out / "appearance/index.json").write_text(json.dumps(index, indent=2) + "\n")
    _, info = verified_files(out / "assets", out / "appearance")
    print(json.dumps({**collection_counts(manifest["doors"]), "corrected_baby_gates": sorted(gate_ids),
                      "signed_off": manifest["n_signed_off"], **info}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--baby-gates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base, args.baby_gates, args.out)


if __name__ == "__main__":
    main()

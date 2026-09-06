#!/usr/bin/env python3
"""Audit saved G1 goal outcomes against a sampled crossing of the actual wall plane.
Keep raw results. Horizontal hatches are attempted but are not upright-traversal
fixtures. Native errors and unimplemented spatial elements remain explicit.
"""

import argparse, hashlib, json, re, math
from pathlib import Path
from datetime import datetime, timezone

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--results", required=True)
p.add_argument("--assets", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
root = Path(a.results)
assets = Path(a.assets)
doc = json.loads((root / "results.json").read_text())
rows = []
suite = json.loads((assets / "demo-suite.json").read_text())
expected = {c["id"]: c for c in suite["cases"]}
if set(doc["per_door"]) - expected.keys():
    raise ValueError("Result contains doors outside the frozen suite")
if doc["complete"] and set(doc["per_door"]) != expected.keys():
    raise ValueError("Complete result is missing frozen suite cases")
if (
    doc["suite_sha256"]
    != hashlib.sha256((assets / "demo-suite.json").read_bytes()).hexdigest()
):
    raise ValueError("Suite checksum mismatch")
for id, r in doc["per_door"].items():
    folder = assets / "doors" / id
    spec = json.loads((folder / "spec.json").read_text())
    model = json.loads((folder / "model.json").read_text())
    match = re.search(
        r'string doorbench:rl = (".*")', (folder / "door_rl.usda").read_text()
    )
    rl = json.loads(json.loads(match.group(1))) if match else {}
    vertical = spec["family"] not in ("hatch_floor", "hatch_ceiling", "pet_door")
    native_supported = rl.get("mechanical_parity_supported") is not False
    path = root / r.get("evidence_directory", "") / (id + ".trace.json")
    source_errors = []
    for name, sha in expected[id].get("source_sha256", {}).items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != sha:
            source_errors.append(name + ": frozen input changed")
        if not r.get("simulator_error") and r.get("source_sha256", {}).get(name) != sha:
            source_errors.append(name + ": receipt source mismatch")
    crossed = False
    trace_error = None
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != r.get("trace_sha256"):
            trace_error = "trace_hash_mismatch"
        else:
            frames = json.loads(path.read_text())
            plane = float(model.get("meta", {}).get("wall_y", 0.0))
            half = spec["opening"]["width"] / 2 - 0.2
            base = float(spec["opening"].get("sill_height") or 0) + float(
                spec["opening"].get("elevation") or 0
            )
            for before, after in zip(frames, frames[1:]):
                p0, p1 = before["position"], after["position"]
                if not all(math.isfinite(v) for v in p0 + p1):
                    trace_error = "nonfinite_trace"
                    continue
                if p0[1] <= plane < p1[1]:
                    fraction = (plane - p0[1]) / (p1[1] - p0[1])
                    x = p0[0] + fraction * (p1[0] - p0[0])
                    z = p0[2] + fraction * (p1[2] - p0[2])
                    crossed = crossed or (
                        abs(x) < half and base < z < base + spec["opening"]["height"]
                    )
    else:
        trace_error = "missing_trace"
    rows.append(
        dict(
            door_id=id,
            family=spec["family"],
            raw_goal_success=bool(r.get("success")),
            vertical_traversal_applicable=vertical,
            native_spatial_elements_supported=native_supported,
            sampled_opening_crossing=crossed,
            trace_error=trace_error,
            source_errors=source_errors,
            simulator_error=r.get("simulator_error"),
            traversal_success=bool(
                r.get("success")
                and vertical
                and crossed
                and not trace_error
                and not source_errors
                and not r.get("simulator_error")
            ),
            failure_reason=r.get("failure_reason"),
        )
    )
report = {
    "scope": "Closed-start canonical-USD G1 locomotion diagnostic, audited root-plane crossing; not assigned core benchmark success or full-body clearance certification",
    "source_results_sha256": hashlib.sha256(
        (root / "results.json").read_bytes()
    ).hexdigest(),
    "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "started_at_utc": doc["started_at_utc"],
    "completed_at_utc": doc.get("completed_at_utc"),
    "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    "complete": doc["complete"],
    "eligible_collection": doc["eligible_doors"],
    "attempted": len(rows),
    "vertical_doors": sum(r["vertical_traversal_applicable"] for r in rows),
    "horizontal_hatches": sum(not r["vertical_traversal_applicable"] for r in rows),
    "successes": sum(r["traversal_success"] for r in rows),
    "native_spatial_limitations": sum(
        not r["native_spatial_elements_supported"] for r in rows
    ),
    "errors": sum(
        bool(r["simulator_error"] or r["trace_error"] or r["source_errors"])
        for r in rows
    ),
    "excluded_pets": doc["excluded"],
    "per_door": rows,
}
Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in report.items() if k not in ("per_door", "excluded_pets")},
        indent=2,
    )
)

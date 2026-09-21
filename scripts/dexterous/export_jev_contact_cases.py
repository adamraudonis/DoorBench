"""Extract actual native contact intervals for an offline Jev judgment probe.

The source is the privileged simulator contact audit, not a camera or a tactile
actor packet. Missing slip, joint margin and balance evidence stays unavailable;
use experiment_jev_advisor.py --contact-only for these records. Computed contact
labels check interpretation of measured loads, not independent perception skill.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.jev_advisor import AstraPlan, Telemetry
from doorbench.dexterous.json_record_stream import iter_json_object_array


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def case_from_row(row: dict, *, digit: str, index: int, episode_id: str, threshold_N: float) -> dict | None:
    if row.get("contact_force_source") != "actual_mj_step_dynamics":
        return None  # Initial reset/forward-dynamics samples are not real intervals.
    start, end = row.get("contact_interval_start_s"), row.get("contact_interval_end_s")
    geometry = row.get("contact_geometry_time_s")
    if not all(finite(v) for v in (start, end, geometry)) or start < 0 or end <= start or abs(geometry-start) > 1e-8:
        raise ValueError("Require a matching actual contact interval and geometry epoch")
    audit = row.get("pad_grasp", {})
    load = audit.get("digit_forces_N", {}).get(digit)
    if load is not None and (not finite(load) or load < 0):
        raise ValueError("Invalid measured digit contact load")
    contacts = audit.get("contacts")
    if load is not None and isinstance(contacts, list):
        patch_loads = [c["normal_force_N"] for c in contacts if c.get("digit") == digit]
        if not all(finite(v) and v >= 0 for v in patch_loads) or not math.isclose(sum(patch_loads), load, abs_tol=1e-7):
            raise ValueError("Digit load disagrees with recorded contact patches")
    operation = row.get("operation", {})
    target = operation.get("commanded_operator_reference_rad")
    actual = operation.get("actual_handle_rad")
    before = row.get("pre_integration_state", {})
    if finite(actual) and finite(before.get("handle_angle_rad")) and abs(actual-before["handle_angle_rad"]) > 1e-7:
        raise ValueError("Operation angle must match the contact interval's pre-integration state")
    angle_error = target-actual if finite(target) and finite(actual) else None
    sample = Telemetry(episode_id, index, start, 0., "native-telemetry", load, angle_error,
                       None, None, None, sensor_valid=bool(row.get("finite") is True and load is not None),
                       contact_scope="privileged_digit_handle_contact")
    return dict(name=f"{digit}-interval-{index:06d}", telemetry=asdict(sample),
        label=dict(contact=load >= threshold_N if load is not None else None),
        provenance=dict(
            measurement_source="privileged actual_mj_step contact-pair forces",
            selected_digit=digit, known_handle_geometry=audit.get("lever_geom"),
            contact_interval_start_s=start, contact_interval_end_s=end, contact_geometry_time_s=geometry,
            phase=operation.get("phase", "acquisition"),
            commanded_operator_reference_rad=target, preintegration_operator_angle_rad=actual,
            angle_error_definition="commanded_operator_reference_rad minus actual_handle_rad at interval start; not a measured wrist angle",
            label_definition=f"Same measured digit normal load >= {threshold_N} N; arithmetic consistency label, not an independent perception target",
            unavailable_fields=["slip_speed_mps", "joint_margin_rad", "balance_stable"],
            limitation="Torso tilt alone does not establish stability; maximum joint-limit violation does not supply an available joint margin; moving contact patch positions do not establish slip velocity."))


def select_indices(cases: list[dict], count: int) -> list[int]:
    """Include contact/phase transitions and cover the episode, deterministically."""
    if len(cases) <= count:
        return list(range(len(cases)))
    selected = {0, len(cases)-1}
    transitions = [i for i in range(1, len(cases)) if
                   cases[i]["label"]["contact"] != cases[i-1]["label"]["contact"] or
                   cases[i]["provenance"]["phase"] != cases[i-1]["provenance"]["phase"]]
    # Spread transition picks when force chatter produces many boundary crossings.
    slots = max(1, (count-2)//4)
    chosen = transitions if len(transitions) <= slots else [transitions[round(i*(len(transitions)-1)/max(1, slots-1))] for i in range(slots)]
    for i in chosen:
        if len(selected)+2 <= count:
            selected.update((i-1, i))
    for i in [round(j*(len(cases)-1)/max(1, count-1)) for j in range(count)]:
        if len(selected) >= count:
            break
        selected.add(i)
    # Uniform indices can collide with the chosen transitions.
    for i in range(len(cases)):
        if len(selected) >= count:
            break
        selected.add(i)
    return sorted(selected)


def export_cases(run: Path, output: Path, *, digit: str = "ff", count: int = 12,
                 threshold_N: float = .15, allow_incomplete: bool = False) -> dict:
    run = run.resolve()
    final = run/"physics-steps.json.gz"
    complete = final.exists() and (run/"report.json").exists()
    if not complete and not allow_incomplete:
        raise ValueError("Run has no finalized physics/report pair; --allow-incomplete explicitly permits durable prefix chunks")
    sources = [final] if final.exists() else sorted((run/"physics-chunks").glob("*.jsonl.gz"))
    if not sources:
        raise ValueError("No recorded native physics evidence found")
    cases, ledger = [], []
    row_index = 0
    for path in sources:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ledger.append(dict(file=str(path), sha256=digest))
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = iter_json_object_array(stream) if path == final else (json.loads(line) for line in stream if line.strip())
            for local_index, row in enumerate(rows):
                case = case_from_row(row, digit=digit, index=row_index, episode_id=run.name, threshold_N=threshold_N)
                if case is not None:
                    case["provenance"].update(source_file=str(path), source_sha256=digest,
                                              source_row_index=local_index, episode_complete=complete)
                    cases.append(case)
                row_index += 1
    if not cases:
        raise ValueError("No actual dynamics intervals available")
    selected = [cases[index] for index in select_indices(cases, count)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report = dict(schema="doorbench.jev-contact-export.v1", source_run=str(run), episode_complete=complete,
                  source_rows=row_index, actual_intervals=len(cases), exported_cases=len(selected),
                  digit=digit, contact_threshold_N=threshold_N, sources=ledger,
                  output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  scope=__doc__, selection="Contact/phase transition pairs and spread across the recorded episode",
                  phase_counts={phase: sum(c["provenance"]["phase"] == phase for c in selected)
                                for phase in sorted({c["provenance"]["phase"] for c in selected})},
                  contact_counts={str(value): sum(c["label"]["contact"] is value for c in selected) for value in (True, False, None)})
    output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--digit", choices=("ff", "mf", "rf", "lf", "th"), default="ff")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--contact-threshold-N", type=float, default=.15)
    parser.add_argument("--plan", type=Path, help="Explicit AstraPlan JSON; its contact threshold overrides the default")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.plan:
        plan = AstraPlan(**json.loads(args.plan.read_text(encoding="utf-8")))
        args.contact_threshold_N = plan.contact_threshold_N
    if not 2 <= args.count <= 100 or not finite(args.contact_threshold_N) or args.contact_threshold_N <= 0:
        parser.error("Use 2–100 cases and a finite positive contact threshold")
    report = export_cases(args.run, args.output, digit=args.digit, count=args.count,
                          threshold_N=args.contact_threshold_N, allow_incomplete=args.allow_incomplete)
    print(json.dumps({key: report[key] for key in ("episode_complete", "actual_intervals", "exported_cases", "phase_counts", "contact_counts")}))


if __name__ == "__main__":
    main()

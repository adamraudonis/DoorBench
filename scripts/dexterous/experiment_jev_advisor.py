"""Evaluate Jev's contact/alignment advice on synthetic or recorded telemetry.

This is an offline judgment experiment, never an H1 rollout or success claim.
The --live flag sends the compact telemetry summary to the official TypeSafe API.
The default only writes requests; it does not use a substitute/mock model.

Recorded input is a JSON list or JSONL, with each row containing ``telemetry``
matching the Telemetry dataclass. Optional ``label`` fields are evaluator-only.
Offline replay refreshes the process-clock timestamp at presentation; original
records remain in the receipt, and no advice is applied to a running simulation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import sys
import time

# Direct execution works from a checkout without installing simulator extras.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.jev_advisor import (
    AstraPlan, JevAdvisor, JevClient, MODEL, SCHEMA, Telemetry,
    build_request, reference_action,
)


def synthetic_cases() -> list[dict]:
    """Handwritten contract fixtures; these are not observations of a robot."""
    base = dict(episode_id="synthetic-contract", sample_id=0, simulation_time_s=0.,
                capture_monotonic_s=0., source="synthetic", normal_load_N=1.,
                angle_error_rad=.12, slip_speed_mps=0., joint_margin_rad=.2,
                balance_stable=True, sensor_valid=True)
    cases = [
        ("positive_angle_error", {}, True, "increase_angle"),
        ("negative_angle_error", {"angle_error_rad": -.12}, True, "decrease_angle"),
        ("aligned", {"angle_error_rad": .005}, True, "hold"),
        ("no_contact", {"normal_load_N": 0.}, False, "reacquire_contact"),
        ("contact_at_threshold", {"normal_load_N": .15}, True, "increase_angle"),
        ("contact_below_threshold", {"normal_load_N": .149}, False, "reacquire_contact"),
        ("slipping", {"slip_speed_mps": .05}, True, "reacquire_contact"),
        ("overloaded", {"normal_load_N": 20.}, True, "hold"),
        ("balance_lost", {"balance_stable": False}, True, "hold"),
        ("joint_limit_near", {"joint_margin_rad": .005}, True, "hold"),
        ("missing_touch", {"normal_load_N": None}, None, "request_astra"),
        ("invalid_sensor", {"sensor_valid": False}, None, "request_astra"),
    ]
    return [dict(name=name, telemetry={**base, **changes, "sample_id": index},
                 label=dict(contact=contact, angle_action=action))
            for index, (name, changes, contact, action) in enumerate(cases)]


def load_cases(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(content)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) and "telemetry" in row for row in rows):
        raise ValueError("Recorded input must contain a nonempty list of telemetry rows")
    # Validate the whole input before making any calls; provenance is mandatory.
    for row in rows:
        sample = Telemetry(**row["telemetry"])
        if sample.source == "synthetic":
            raise ValueError("Use --synthetic for synthetic fixtures, not --input")
    return rows


def run_experiment(cases: list[dict], plan: AstraPlan, *, advisor: JevAdvisor | None,
                   repeats: int = 1, max_cases: int = 12, contact_only: bool = False) -> dict:
    records = []
    for repeat in range(repeats):
        for index, case in enumerate(cases[:max_cases]):
            original = Telemetry(**case["telemetry"])
            sample = replace(original, capture_monotonic_s=time.monotonic())
            record = dict(name=case.get("name", f"sample-{index}"), repeat=repeat,
                          original_telemetry=asdict(original), request=build_request(sample, plan, contact_only=contact_only),
                          deterministic_comparator=reference_action(sample, plan),
                          label=case.get("label"), provenance=case.get("provenance"), replay_only=True)
            if advisor is not None:
                advice = advisor.evaluate(sample, plan)
                record["advice"] = advice.to_dict()
                record["proposed_matches_comparator"] = (advice.proposed_action == record["deterministic_comparator"]
                                                         if advice.proposed_action is not None else None)
                label = case.get("label") or {}
                if type(label.get("contact")) is bool and advice.contact_probability is not None:
                    record["contact_brier_score"] = (advice.contact_probability - float(label["contact"])) ** 2
                if isinstance(label.get("angle_action"), str) and advice.proposed_action is not None:
                    record["proposed_matches_label"] = advice.proposed_action == label["angle_action"]
                print(json.dumps(dict(name=record["name"], repeat=repeat, **advice.to_dict())), flush=True)
            records.append(record)
    latencies = [r["advice"]["latency_ms"] for r in records if r.get("advice", {}).get("latency_ms") is not None]
    comparator = [r["proposed_matches_comparator"] for r in records if r.get("proposed_matches_comparator") is not None]
    label_matches = [r["proposed_matches_label"] for r in records if "proposed_matches_label" in r]
    brier = [r["contact_brier_score"] for r in records if "contact_brier_score" in r]
    return dict(schema=SCHEMA, model_requested=MODEL,
        scope="Offline telemetry judgment experiment; no robot rollout, vision, physical control, or door-opening success measured.",
        mode="live_jev_offline_replay" if advisor else "request_preview_no_model_calls",
        sources=sorted({r["original_telemetry"]["source"] for r in records}),
        plan=asdict(plan), api_key_saved=False, contact_only=contact_only,
        summary=dict(records=len(records), accepted_advice=sum(r.get("advice", {}).get("accepted", False) for r in records),
                     contact_answers=sum(r.get("advice", {}).get("contact_probability") is not None for r in records),
                     model_answers=len(comparator), comparator_agreement=statistics.mean(comparator) if comparator else None,
                     label_agreement=statistics.mean(label_matches) if label_matches else None,
                     contact_brier_score=statistics.mean(brier) if brier else None,
                     latency_ms_median=statistics.median(latencies) if latencies else None,
                     latency_ms_max=max(latencies) if latencies else None), records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--input", type=Path)
    parser.add_argument("--plan", type=Path, help="JSON AstraPlan, otherwise the declared alignment probe plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="Use TYPESAFE_API_KEY to make real API calls")
    parser.add_argument("--contact-only", action="store_true", help="Judge measured contact only; all control advice abstains, missing angle/slip/balance is permitted")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--max-age-s", type=float, default=.75)
    parser.add_argument("--timeout-s", type=float, default=2.)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 1 <= args.max_cases <= 100:
        parser.error("Use 1–10 repeats and 1–100 cases for a bounded probe")
    cases = synthetic_cases() if args.synthetic else load_cases(args.input)
    plan = AstraPlan(**json.loads(args.plan.read_text(encoding="utf-8"))) if args.plan else AstraPlan(
        plan_id="astra-contact-alignment-probe-v1", phase="hold_and_align",
        objective="Maintain measured fingertip contact while aligning toward a locally computed target; request high-level recovery when evidence is insufficient.")
    advisor = JevAdvisor(JevClient(timeout_s=args.timeout_s), max_age_s=args.max_age_s, contact_only=args.contact_only) if args.live else None
    result = run_experiment(cases, plan, advisor=advisor, repeats=args.repeats, max_cases=args.max_cases, contact_only=args.contact_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(args.output.resolve()), mode=result["mode"], **result["summary"])))


if __name__ == "__main__":
    main()

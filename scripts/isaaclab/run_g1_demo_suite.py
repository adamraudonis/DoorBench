#!/usr/bin/env python3
"""Run explicit demo cases in fresh Isaac processes; retain simulator errors too."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", default="out/isaac-g1-demo/assets")
    p.add_argument(
        "--out", required=True, help="New output directory (must not already exist)"
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--duration", type=float, default=16.0)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--policy", default="robot_demo.isaac_policy_adapter:unitree_factory"
    )
    p.add_argument("--checkpoint")
    p.add_argument("--policy-config")
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    assets = Path(a.assets).resolve()
    cases = json.loads((assets / "demo-suite.json").read_text())["cases"]
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        for seed in a.seeds:
            stem = f"{case['id']}-seed{seed}"
            command = [
                sys.executable,
                str(ROOT / "scripts/isaaclab/demo_g1.py"),
                "--headless",
                "--assets",
                str(assets),
                "--out",
                str(out),
                "--door",
                case["id"],
                "--seed",
                str(seed),
                "--duration",
                str(a.duration),
                "--device",
                a.device,
                "--policy",
                a.policy,
            ]
            for flag, value in (
                ("--checkpoint", a.checkpoint),
                ("--policy-config", a.policy_config),
            ):
                if value:
                    command.extend([flag, value])
            if a.video:
                command.append("--video")
            log = out / (stem + ".log")
            failure = None
            with log.open("w") as stream:
                try:
                    proc = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=a.timeout,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    )
                    if proc.returncode:
                        failure = f"process_exit_{proc.returncode}"
                except subprocess.TimeoutExpired:
                    failure = "simulator_timeout"
            report = out / (stem + ".json")
            text = log.read_text(errors="replace")
            # Kit shutdown sometimes exits 0 after a Python exception: require fresh evidence.
            if (
                not report.exists()
                or "DOORBENCH_G1_RESULT " not in text
                or "Traceback (most recent call last)" in text
            ):
                failure = failure or "missing_or_invalid_native_result"
            row = (
                json.loads(report.read_text())
                if report.exists()
                else {"door_id": case["id"], "seed": seed}
            )
            row.update(command=command, log=log.name)
            if failure:
                row.update(success=False, simulator_error=failure)
            rows.append(row)
            summary = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Explicit integration demo, not a full DoorBench benchmark",
                "episodes": len(rows),
                "successful_episodes": sum(bool(r.get("success")) for r in rows),
                "simulator_errors": sum(bool(r.get("simulator_error")) for r in rows),
                "results": rows,
            }
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                stem,
                "PASS" if row.get("success") else "FAIL",
                row.get("simulator_error") or row.get("failure_reason"),
                flush=True,
            )
    return int(any(r.get("simulator_error") for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Generate the small explicit G1 demo suite, without thumbnails or benchmark claims."""

import argparse, json, sys, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from doorbench.spec import generate_all
from doorbench.build import export_door

CASES = [
    {
        "id": "db0119_swing_single",
        "task": "traverse_open",
        "initial_open_fraction": 1.0,
        "description": "Already-open doorway: initialization is not policy opening.",
    },
    {
        "id": "db0990_automatic_sliding",
        "task": "open_and_traverse",
        "initial_open_fraction": 0.0,
        "description": "Automatic sensor opens the door; policy supplies locomotion.",
    },
    {
        "id": "db0123_saloon",
        "task": "push_through",
        "initial_open_fraction": 0.0,
        "description": "Robot collision must push the passive leaves.",
    },
    {
        "id": "db0705_swing_single",
        "task": "open_and_traverse",
        "initial_open_fraction": 0.0,
        "description": "Closed latched door; locomotion alone is expected to fail.",
    },
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="out/isaac-g1-demo/assets")
    p.add_argument(
        "--isaac", action="store_true", help="Use the installed Isaac USD runtime"
    )
    a = p.parse_args()
    app = None
    if a.isaac:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    specs = {s["id"]: s for s in generate_all()}
    rows = []
    for case in CASES:
        s = specs[case["id"]]
        exported = export_door(
            s, str(out / "doors"), str(out / "hardware"), formats=("usd", "json")
        )
        errors = {
            key: value
            for key, value in exported["files"].items()
            if isinstance(value, str) and value.startswith("ERROR:")
        }
        if errors:
            raise RuntimeError(f"{s['id']} export failed: {errors}")
        folder = out / "doors" / s["id"]
        hashes = {
            n: hashlib.sha256((folder / n).read_bytes()).hexdigest()
            for n in ("spec.json", "model.json", "door.usda", "door_rl.usda")
        }
        rows.append({**case, "family": s["family"], "source_sha256": hashes})
        print(s["id"], flush=True)
    (out / "manifest.json").write_text(
        json.dumps({"schema_version": "g1-demo-1", "doors": rows}, indent=2) + "\n"
    )
    (out / "demo-suite.json").write_text(
        json.dumps(
            {
                "scope": "Four explicit integration examples, not the 1000-door benchmark",
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    if app:
        app.close()


if __name__ == "__main__":
    main()

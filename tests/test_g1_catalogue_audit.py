"""A goal behind a wall must not certify walking around or below its opening."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/isaaclab/summarize_g1_catalogue.py"
)


def test_audit_checks_crossing_sources_and_applicability(tmp_path):
    assets, results = tmp_path / "assets", tmp_path / "results"
    results.mkdir()
    cases, receipts = [], {}
    definitions = [
        ("pass", "swing_single", [[0, -1, 0.8], [0, 1, 0.8]], False),
        (
            "walk_around",
            "swing_single",
            [[2, -1, 0.8], [2, 1, 0.8], [0, 2, 0.8]],
            False,
        ),
        ("hatch", "hatch_floor", [[0, -1, 0.8], [0, 1, 0.8]], False),
        ("wrong_source", "swing_single", [[0, -1, 0.8], [0, 1, 0.8]], True),
    ]
    for id, family, positions, bad in definitions:
        folder = assets / "doors" / id
        folder.mkdir(parents=True)
        (folder / "spec.json").write_text(
            json.dumps({"family": family, "opening": {"width": 1, "height": 2}})
        )
        (folder / "model.json").write_text(json.dumps({"meta": {"wall_y": 0.1}}))
        (folder / "door_rl.usda").write_text('string doorbench:rl = "{}"')
        sha = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()
        }
        cases.append({"id": id, "source_sha256": sha})
        path = results / (id + ".trace.json")
        path.write_text(json.dumps([{"position": p} for p in positions]))
        receipts[id] = {
            "success": True,
            "source_sha256": {} if bad else sha,
            "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    suite = assets / "demo-suite.json"
    suite.write_text(json.dumps({"cases": cases}))
    (results / "results.json").write_text(
        json.dumps(
            {
                "per_door": receipts,
                "complete": True,
                "eligible_doors": 4,
                "excluded": [],
                "started_at_utc": "2026-09-06T00:00:00Z",
                "suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
            }
        )
    )
    out = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--assets",
            str(assets),
            "--results",
            str(results),
            "--out",
            str(out),
        ],
        check=True,
    )
    data = json.loads(out.read_text())
    assert data["successes"] == 1
    assert data["horizontal_hatches"] == 1
    assert data["errors"] == 1
    assert [r["door_id"] for r in data["per_door"] if r["traversal_success"]] == [
        "pass"
    ]


def test_hero_requires_audited_supported_successes(tmp_path):
    import importlib.util
    import pytest

    source = SCRIPT.with_name("select_g1_hero.py")
    spec = importlib.util.spec_from_file_location("select_hero", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for id, success, vertical, supported in [
        ("valid", True, True, True),
        ("raw_only", False, True, True),
        ("hatch", True, False, True),
        ("unsupported", True, True, False),
    ]:
        folder = tmp_path / "doors" / id
        folder.mkdir(parents=True)
        (folder / "spec.json").write_text(
            json.dumps({"opening": {"width": 1.0, "height": 2.0}})
        )
        rows.append(
            {
                "door_id": id,
                "family": id,
                "traversal_success": success,
                "vertical_traversal_applicable": vertical,
                "native_spatial_elements_supported": supported,
            }
        )
    assert module.select({"per_door": rows}, tmp_path, count=1) == ["valid"]
    with pytest.raises(ValueError, match="distinct audited successes"):
        module.select({"per_door": rows}, tmp_path, count=2)

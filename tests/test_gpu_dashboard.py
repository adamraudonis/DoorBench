import hashlib
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "gpu_collect", ROOT / "scripts/gpu_dashboard/collect.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def ledger(tmp_path, complete=True):
    data = {
        "started_at_utc": "2026-09-06T20:00:00+00:00",
        "completed_at_utc": "2026-09-06T20:10:00+00:00" if complete else None,
        "complete": complete,
        "eligible_doors": 2,
        "per_door": {
            "db0001_rollup": {
                "success": True,
                "completed_at_utc": "2026-09-06T20:09:00+00:00",
                "evidence_directory": "retry-db0001_rollup",
            }
        },
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(data))
    return p


def test_stale_audit_never_promotes_raw_goals(tmp_path):
    p = ledger(tmp_path)
    audit = {
        "source_results_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "successes": 0,
        "vertical_doors": 1,
        "per_door": [
            {
                "door_id": "db0001_rollup",
                "traversal_success": False,
                "vertical_traversal_applicable": True,
            }
        ],
    }
    (tmp_path / "traversal-audit.json").write_text(json.dumps(audit))
    d = module.collect(tmp_path)
    assert d["audited"] and d["audited_success"] == 0 and d["raw_goals"] == 1
    assert d["rows"][0]["outcome"] == "crossing rejected" and d["retries"] == 1
    p.write_text(p.read_text() + "\n")
    d = module.collect(tmp_path)
    assert not d["audited"] and d["audited_success"] is None
    assert d["rows"][0]["outcome"] == "raw goal reached"


def test_stalled_heartbeat_and_retry_activity(tmp_path):
    ledger(tmp_path, False)
    folder = tmp_path / "retry-db0001_rollup"
    folder.mkdir()
    (folder / "run.log").write_text("GRID_PROGRESS 3.5 0\n")
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "phase": "retrying",
                "folder": folder.name,
                "ids": ["db0001_rollup"],
                "heartbeat_at_utc": "2020-01-01T00:00:00Z",
            }
        )
    )
    d = module.collect(tmp_path)
    assert d["status"] == "not reporting" and d["simulation"]["seconds"] == 3.5
    assert d["attempted"] == 1  # active retry is not added a second time
    assert d["log_name"] == folder.name + "/run.log"


def test_heartbeat_atomic_state_and_finish(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "run_progress", ROOT / "scripts/isaaclab/run_progress.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tracker = mod.RunProgress(tmp_path, 985, 32)
    tracker.update("retrying", folder="retry-a", ids=["a"])
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == "retrying"
    tracker.update("completed")
    tracker.close()
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == "completed"


def test_remote_collector_quotes_path_and_keeps_credentials_server_side(monkeypatch):
    import shlex
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "gpu_server", ROOT / "scripts/gpu_dashboard/server.py"
    )
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    recorded = {}

    def run(args, **kwargs):
        recorded.update(args=args, kwargs=kwargs)
        return SimpleNamespace(returncode=0, stdout='{"attempted": 3}', stderr="")

    monkeypatch.setattr(server.subprocess, "run", run)
    path = "/work/results; echo unsafe"
    assert server.snapshot(
        {"results": path, "ssh_host": "root@example", "ssh_key": "/private/key"}
    ) == {"attempted": 3}
    assert shlex.split(recorded["args"][-1]) == ["python3", "-", path, "--gpu"]
    assert "BatchMode=yes" in recorded["args"]
    assert "/private/key" not in recorded["kwargs"]["input"]

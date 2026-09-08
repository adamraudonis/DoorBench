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


def test_finished_experiment_with_failed_checks_is_not_success(tmp_path):
    (tmp_path / 'pipeline.json').write_text(json.dumps(dict(
        completion_marker='TRIAL_FINISHED', result_passed=False)))
    (tmp_path / 'run.log').write_text('TRIAL_FINISHED\n')
    result = module.collect(tmp_path)
    assert result['status'] == 'failed'
    assert result['complete'] is False


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


def test_training_heartbeat_failure_and_completion(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'steps':500000}))
    progress={'stage':'privileged body reach training','heartbeat_unix':time.time(),'timesteps':2048}
    (tmp_path/'progress.json').write_text(json.dumps(progress))
    d=module.collect(tmp_path)
    assert d['training'] and d['status']=='running' and not d['complete']
    assert 'audited_success' not in d
    (tmp_path/'failed.json').write_text(json.dumps({'error':'test failure'}))
    assert module.collect(tmp_path)['status']=='failed'
    (tmp_path/'failed.json').unlink()
    progress['heartbeat_unix']=0
    (tmp_path/'progress.json').write_text(json.dumps(progress))
    assert module.collect(tmp_path)['status']=='not reporting'
    (tmp_path/'completed.json').write_text(json.dumps({'timesteps':500000}))
    assert module.collect(tmp_path)['status']=='completed'
    assert module.collect(tmp_path)['progress']['timesteps']==500000
    progress['checkpoint_timesteps']=progress['timesteps']+430000
    (tmp_path/'progress.json').write_text(json.dumps(progress))
    (tmp_path/'completed.json').write_text(json.dumps({'timesteps':930000}))
    assert module.collect(tmp_path)['progress']['timesteps']==500000


def test_pipeline_failure_overrides_stale_completion(tmp_path):
    (tmp_path/'pipeline.json').write_text(json.dumps(dict(completion_marker='ISAAC_ENVIRONMENT_READY')))
    (tmp_path/'run.log').write_text('ISAAC_ENVIRONMENT_READY\nREADINESS_FAILED exit=1\n')
    d=module.collect(tmp_path)
    assert d['status']=='failed' and not d['complete']


def test_pipeline_final_report_survives_missing_log_and_stale_status(tmp_path):
    (tmp_path/'pipeline.json').write_text(json.dumps(dict(
        status='running', report_file='report.json')))
    (tmp_path/'report.json').write_text(json.dumps(dict(
        passed=False, checks={'upright': False}, scope='Actual failed actor')))
    d=module.collect(tmp_path)
    assert d['status']=='failed' and d['result']['checks']=={'upright': False}
    (tmp_path/'report.json').write_text(json.dumps(dict(
        passed=True, checks={'upright': True})))
    assert module.collect(tmp_path)['status']=='completed'
    (tmp_path/'report.json').write_text(json.dumps(dict(
        passed=True, checks={'upright': False})))
    assert module.collect(tmp_path)['status']=='failed'


def test_pipeline_status_without_report_does_not_claim_success(tmp_path):
    (tmp_path/'pipeline.json').write_text(json.dumps(dict(
        status='passed', report_file='../report.json')))
    assert module.collect(tmp_path)['status']=='stopped'

"""Read-only, standard-library snapshot collector; also runs over SSH stdin."""

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def tail(path):
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 24000))
            return f.read().decode(errors="replace").splitlines()[-100:]
    except OSError:
        return []


def stamp(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, AttributeError):
        return None


def collect(directory, telemetry=False):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError("Results directory does not exist yet")
    raw = (
        (root / "results.json").read_bytes()
        if (root / "results.json").exists()
        else b"{}"
    )
    try:
        data = json.loads(raw)
    except ValueError:
        raise ValueError(
            "Result ledger is being written; retaining the previous snapshot"
        )
    progress = read(root / "progress.json", {})
    audit = read(root / "traversal-audit.json", {})
    verified = bool(
        audit and audit.get("source_results_sha256") == hashlib.sha256(raw).hexdigest()
    )
    ar = {r["door_id"]: r for r in audit.get("per_door", [])} if verified else {}
    rows = []
    events = []
    for id, r in data.get("per_door", {}).items():
        check = ar.get(id, {})
        error = r.get("simulator_error")
        outcome = (
            "native error"
            if error
            else "not applicable"
            if check.get("vertical_traversal_applicable") is False
            else "verified traversal"
            if check.get("traversal_success")
            else "crossing rejected"
            if verified and r.get("success")
            else "raw goal reached"
            if r.get("success")
            else "policy failure"
        )
        rows.append(
            dict(
                id=id,
                outcome=outcome,
                reason=error or r.get("failure_reason"),
                retry=r.get("evidence_directory", "").startswith("retry-"),
                family=check.get("family") or id.split("_", 1)[-1],
                duration=r.get("elapsed_sim_s"),
            )
        )
        if stamp(r.get("completed_at_utc")):
            events.append(stamp(r["completed_at_utc"]))
    logs = (
        list(root.glob("batch-*/run.log"))
        + list(root.glob("retry-*/run.log"))
        + list(root.glob("hero/run.log"))
    )
    active = root / progress.get("folder", "") / "run.log"
    if not active.exists() and logs and not progress.get("folder"):
        active = max(logs, key=lambda p: p.stat().st_mtime)
    lines = tail(active)
    sim = None
    for line in reversed(lines):
        if line.startswith("GRID_PROGRESS "):
            pieces = line.split()
            try:
                sim = {"seconds": float(pieces[1]), "finished": int(pieces[2])}
            except (ValueError, IndexError):
                pass
            break
    gpu = []
    if telemetry:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for values in csv.reader(proc.stdout.splitlines()):
                if len(values) == 6:
                    gpu.append(
                        dict(
                            zip(
                                [
                                    "name",
                                    "utilization",
                                    "memory_used",
                                    "memory_total",
                                    "temperature",
                                    "power",
                                ],
                                [v.strip() for v in values],
                            )
                        )
                    )
        except (OSError, subprocess.TimeoutExpired):
            pass
    started = data.get("started_at_utc") or progress.get("started_at_utc")
    ended = data.get("completed_at_utc")
    elapsed = (
        max(0, (stamp(ended) or time.time()) - stamp(started))
        if stamp(started)
        else None
    )
    count = len(rows)
    total = data.get("eligible_doors") or progress.get("eligible_doors", 0)
    rate = count * 60 / elapsed if elapsed and count else None
    freshness = max(
        [
            p.stat().st_mtime
            for p in [root / "results.json", root / "progress.json", active]
            if p.exists()
        ],
        default=None,
    )
    heartbeat = stamp(progress.get("heartbeat_at_utc"))
    status = progress.get("phase") or (
        "completed" if data.get("complete") else "running" if count else "waiting"
    )
    if (
        status not in ("completed", "stopped")
        and heartbeat
        and time.time() - heartbeat > 30
    ):
        status = "not reporting"
    elif (
        not progress
        and not data.get("complete")
        and freshness
        and time.time() - freshness > 120
    ):
        status = "not reporting"
    return dict(
        status=status,
        progress=progress,
        complete=bool(data.get("complete")),
        started=started,
        ended=ended,
        elapsed=elapsed,
        attempted=count,
        total=total,
        raw_goals=sum(
            bool(r.get("success")) for r in data.get("per_door", {}).values()
        ),
        native_errors=sum(
            bool(r.get("simulator_error")) for r in data.get("per_door", {}).values()
        ),
        retries=sum(r["retry"] for r in rows),
        audited=verified,
        audited_success=audit.get("successes") if verified else None,
        applicable=audit.get("vertical_doors") if verified else None,
        horizontal_hatches=audit.get("horizontal_hatches") if verified else None,
        rate=rate,
        eta=(total - count) * 60 / rate if rate and count < total else None,
        source_updated=freshness,
        heartbeat=heartbeat,
        gpu=gpu,
        recorded_gpu=next(
            (r.get("gpu") for r in data.get("per_door", {}).values() if r.get("gpu")),
            None,
        ),
        timeline=sorted(events),
        rows=rows,
        counts=dict(Counter(r["outcome"] for r in rows)),
        log_updated=active.stat().st_mtime if active.exists() else None,
        log=lines,
        log_name=str(active.relative_to(root)) if active.exists() else None,
        simulation=sim,
        scope=data.get("scope", "Native benchmark run"),
    )


if __name__ == "__main__":
    result = collect(sys.argv[1], "--gpu" in sys.argv)

    # Keep the browser API strict JSON even if a native receipt has invalid numbers.
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    print(json.dumps(clean(result), allow_nan=False))

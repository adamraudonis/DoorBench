#!/usr/bin/env python3
"""Run the frozen 55 s continuous privileged Door55 demo on a ready v2 host.

Uses the existing environment; does not allocate GPUs or change teardown guards.
The sequence ends at a small held opening, not traversal or sensor-only control.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from doorbench.dexterous.isaac_readiness import load_ready_receipt, ready_directory, file_sha256
from doorbench.dexterous.locomotion import POLICY_SHA256

PROTOCOL = "configs/dexterous/door55-readiness-v2/isaac-demo-protocol.json"
RESET = "configs/dexterous/door55-readiness-v2/body-reset.json"
PREPARATION = "configs/dexterous/door55-readiness-v2/reference.json"
REFERENCE = "configs/dexterous/door55-precurl-v2/reference.json"


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def build_plan(receipt_path, checkpoint, output, *, root=ROOT, asset_python=None,
               isaac_python=sys.executable, record=True, view="wide", device="cuda:0"):
    root = Path(root).resolve(); output = Path(output).resolve()
    receipt_path = Path(receipt_path).resolve(); checkpoint = Path(checkpoint).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh full-sequence output directory")
    receipt = load_ready_receipt(receipt_path, expected_profile="shadow-loopback-v2")
    protocol = json.loads((root / PROTOCOL).read_text())
    if (protocol.get("schema") != "doorbench.isaac-full-sequence-demo.v1" or
            protocol.get("seconds") != 55. or protocol.get("physics_dt_s") != .002 or
            protocol.get("mechanics_profile") != "shadow-loopback-v2"):
        raise ValueError("Unsupported frozen continuous-demo protocol")
    if set(protocol["frozen_inputs"]) != {RESET, PREPARATION, REFERENCE}:
        raise ValueError("Frozen walking, readiness and acquisition inputs are required")
    for name, digest in protocol["frozen_inputs"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("Frozen full-sequence input changed: " + name)
    if protocol["locomotion"]["sha256"] != POLICY_SHA256 or file_sha256(checkpoint) != POLICY_SHA256:
        raise ValueError("Expected the original pinned Unitree H1 checkpoint")
    native_door = Path(receipt["door_usd"]).resolve().with_name("door.xml")
    if file_sha256(native_door) not in protocol["native_door_xml_sha256"]:
        raise ValueError("Native door geometry differs from the frozen Door55 mechanism")
    trial = output / "trial"
    command = [str(isaac_python), str(root / "scripts/dexterous/isaac_opening.py"),
        "--robot-usd", receipt["robot_usd"], "--door-usd", receipt["door_usd"],
        "--motors", receipt["motor_contract"], "--native-robot", receipt["native_robot"],
        "--reference", str(root / REFERENCE), "--full-sequence-reset", str(root / RESET),
        "--preparation-reference", str(root / PREPARATION),
        "--locomotion-checkpoint", str(checkpoint), "--native-door", str(native_door),
        "--acquisition", "--operate-after-acquisition", "--open-on-latch-clear",
        "--operator-compliance-gain", str(protocol["operator_compliance_gain"]),
        "--seconds", str(protocol["seconds"]), "--output", str(trial),
        "--view", view, "--headless", "--device", device]
    if record:
        command += ["--record", "--enable_cameras"]
    inputs = {str(Path(p).resolve()): digest for p, digest in receipt["input_hashes"].items()}
    for p in [receipt_path, checkpoint, native_door, root / PROTOCOL,
              *(root / name for name in protocol["frozen_inputs"])]:
        inputs[str(p.resolve())] = file_sha256(p)
    # Preserve every local controller/auditor module rather than maintaining an
    # incomplete hand-written dependency list. Generated assets remain outside.
    sources = list((root / "doorbench/dexterous").glob("*.py"))
    sources += [root / "scripts/dexterous" / name for name in
                ("isaac_opening.py", "physx_teacher.py", "audit_isaac_reference.py", "setup_h1_walking.py")]
    sources += [root / "scripts/isaac/run_full_sequence_demo.py"]
    for p in sources:
        inputs[str(p.resolve())] = file_sha256(p)
    work = Path(os.environ.get("DOORBENCH_WORK", "/workspace"))
    preflight = [str(asset_python or work / "asset-venv/bin/python"),
        str(root / "scripts/dexterous/audit_isaac_reference.py"),
        "--robot", receipt["native_robot"], "--door", str(native_door.parent),
        "--reference", str(output / "initial-reference.json"),
        "--output", str(output / "initial-audit.json")]
    return dict(schema="doorbench.full-sequence-invocation.v1", scope=protocol["scope"],
        root=str(root), output=str(output), trial=str(trial), receipt=str(receipt_path),
        protocol=protocol, argv=command, preflight_argv=preflight, input_sha256=inputs,
        source_files=[str(p.resolve()) for p in sources], runtime_launched=False)


def verify_frozen_inputs(plan):
    for name, digest in plan["input_sha256"].items():
        if file_sha256(name) != digest:
            raise ValueError("Input changed after invocation was frozen: " + name)


def validate_result(report, protocol):
    checks = report.get("checks", {})
    duration = report.get("duration_s")
    if (report.get("passed") is not True or not isinstance(checks, dict) or
            not set(protocol["required_checks"]).issubset(checks) or
            any(value is not True for value in checks.values()) or
            not isinstance(duration, (int, float)) or not math.isfinite(duration) or
            abs(duration - protocol["seconds"]) > 1e-8 or
            report.get("physics_dt_s") != protocol["physics_dt_s"] or
            report.get("runtime_robot_pose_writes") != 0 or report.get("direct_door_commands") is not False):
        raise ValueError("Continuous full-sequence checks failed or evidence is incomplete")


def run_logged(command, *, cwd, env, log):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            log.write(line); log.flush()
            print(line, end="", flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        raise


def execute(plan, registry, *, check_only=False):
    output = Path(plan["output"]); root = Path(plan["root"])
    output.mkdir(parents=True, exist_ok=False)
    pipeline = dict(stage="Native walking-reset preflight", scope=plan["scope"],
        result_passed=None, started_at_unix=time.time(),
        completion_marker="FULL_SEQUENCE_PREFLIGHT_COMPLETE" if check_only else "FULL_SEQUENCE_DEMO_COMPLETE")
    write_json(output / "pipeline.json", pipeline)
    (output / "run.pid").write_text(str(os.getpid()))
    write_json(output / "invocation.json", plan)
    shutil.copy2(plan["receipt"], output / "ready-receipt.json")
    reset = json.loads((root / RESET).read_text())
    write_json(output / "initial-reference.json", dict(initial_root=reset["initial_root"], initial_joints=reset["joints"]))
    for source in plan["source_files"]:
        relative = Path(source).relative_to(root)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)
    for name in (PROTOCOL, RESET, PREPARATION, REFERENCE):
        destination = output / "inputs" / name
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root / name, destination)
    env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PRIVACY_CONSENT="Y")
    with (output / "run.log").open("w") as log:
        try:
            run_logged([sys.executable, str(root / "scripts/gpu_dashboard/server.py"), "--register-only",
                "--config", str(Path(registry).resolve()), "--id", "full-sequence-" + hashlib.sha256(str(output).encode()).hexdigest()[:16],
                "--name", "H1 Door55 continuous demo" + (" (preflight only)" if check_only else ""),
                "--results", str(output)], cwd=root, env=env, log=log)
            verify_frozen_inputs(plan)
            run_logged(plan["preflight_argv"], cwd=root, env=env, log=log)
            if json.loads((output / "initial-audit.json").read_text()).get("passed") is not True:
                raise ValueError("Native reset geometry did not pass")
            verify_frozen_inputs(plan)
            if check_only:
                pipeline.update(stage="Preflight passed; no Isaac trial launched", result_passed=True)
            else:
                pipeline.update(stage="Continuous approach, acquisition and partial opening")
                write_json(output / "pipeline.json", pipeline)
                plan["runtime_launched"] = True; write_json(output / "invocation.json", plan)
                run_logged(plan["argv"], cwd=root, env=env, log=log)
                report = json.loads((Path(plan["trial"]) / "full-sequence-report.json").read_text())
                validate_result(report, plan["protocol"])
                verify_frozen_inputs(plan)
                pipeline.update(stage="Continuous demo checks passed; partial opening only", result_passed=True)
            pipeline["finished_at_unix"] = time.time()
            write_json(output / "pipeline.json", pipeline)
            log.write(pipeline["completion_marker"] + "\n"); log.flush()
            print(pipeline["completion_marker"] + ": " + str(output), flush=True)
        except BaseException as exc:
            pipeline.update(stage="Failed; evidence retained", result_passed=False, finished_at_unix=time.time())
            write_json(output / "pipeline.json", pipeline)
            write_json(output / "failure.json", dict(error=str(exc), runtime_launched=plan["runtime_launched"]))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path); parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--registry", type=Path, default=ROOT / "out/isaac-launch/runs.json")
    parser.add_argument("--asset-python", type=Path); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--view", choices=("wide", "hand"), default="wide"); parser.add_argument("--no-record", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate existing files and print invocation; no files/processes/downloads")
    mode.add_argument("--check-only", action="store_true", help="Run native reset preflight and register evidence; no Isaac process")
    args = parser.parse_args()
    ready = Path(os.environ.get("DOORBENCH_READY_DIR", ready_directory(ROOT, "shadow-loopback-v2")))
    receipt_path = args.receipt or ready / "ready.json"
    load_ready_receipt(receipt_path, expected_profile="shadow-loopback-v2")
    checkpoint = args.checkpoint or ROOT / "out/h1-walking-upstream/deploy/pre_train/h1/motion.pt"
    if not checkpoint.exists():
        if args.checkpoint or args.dry_run:
            raise FileNotFoundError("Pinned checkpoint required for this check: " + str(checkpoint))
        asset_python = args.asset_python or Path(os.environ.get("DOORBENCH_WORK", "/workspace")) / "asset-venv/bin/python"
        subprocess.run([str(asset_python), str(ROOT / "scripts/dexterous/setup_h1_walking.py"),
            "--output", str(checkpoint.parents[3])], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), check=True)
    output = args.output or ROOT / "out/isaac-full-sequence" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    plan = build_plan(receipt_path, checkpoint, output, asset_python=args.asset_python,
                      record=not args.no_record, view=args.view, device=args.device)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    # Only prevent concurrent invocations of this demo; never inspect or stop
    # unrelated jobs or change an existing allocation's lifetime.
    with (receipt_path.parent / ".full-sequence-demo.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This ready environment already has a continuous demo running")
        execute(plan, args.registry, check_only=args.check_only)


if __name__ == "__main__":
    main()

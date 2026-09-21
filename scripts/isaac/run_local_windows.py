#!/usr/bin/env python3
"""Print or run local Windows H1 import, readiness and walking commands.

This launches existing simulator scripts directly. It never starts a cloud job.
Readiness and plane walking are component checks, not door-task success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "shadow-loopback-v2"
CREATE_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def capture(argv, env, timeout=45):
    try:
        result = subprocess.run(list(map(str, argv)), cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=timeout,
                                creationflags=CREATE_FLAGS)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def build_plan(args):
    ready_dir = args.prepared_dir
    robot = ready_dir / "h1-shadow-loopback-v2.xml"
    motors = ready_dir / "h1-import.motors.json"
    reference = ready_dir / "reference.json"
    imported = args.output / "import" / "robot.usda" if args.stage in ("import", "all") else args.robot_usd
    steps = []

    def add(name, python, script, *argv, report=None):
        steps.append({"name": name, "argv": list(map(str, [python, ROOT / script, *argv])),
                      "report": str(report) if report else None})

    if args.stage in ("import", "all"):
        add("import", args.isaac_python, "scripts/dexterous/isaac_import_audit.py",
            "--mjcf", ready_dir / "h1-import.xml", "--output", args.output / "import")
    if args.stage in ("smoke", "all"):
        if imported is None:
            raise ValueError("--robot-usd is required for a separate smoke stage")
        smoke = args.output / "smoke"
        add("smoke", args.isaac_python, "scripts/dexterous/isaac_opening.py",
            "--robot-usd", imported, "--door-usd", args.door_usd,
            "--motors", motors, "--reference", reference, "--output", smoke,
            "--joint-passive-profile", "backend-dry-v2", "--seconds", "1",
            "--headless", "--device", "cuda:0", "--record", "--enable_cameras")
        add("kinematics", args.asset_python, "scripts/dexterous/check_isaac_fk.py",
            "--robot", robot, "--run", smoke, report=smoke / "kinematics-audit.json")
        add("readiness", args.asset_python, "scripts/isaac/write_ready.py",
            "--trial", smoke, "--output", args.output / "ready.json",
            "--native-robot", robot, "--motors", motors, "--mechanics-profile", PROFILE,
            report=args.output / "ready.json")
    if args.stage in ("walking", "all"):
        if args.stage == "walking" and args.ready is None:
            raise ValueError("A separate walking stage requires --ready from a passed smoke stage")
        if imported is None and args.ready is not None:
            # Reading a receipt does not initialize Isaac or modify the filesystem.
            imported = Path(json.loads(args.ready.read_text(encoding="utf-8"))["robot_usd"])
        if imported is None:
            raise ValueError("A verified imported robot is required for walking")
        checkpoint = args.checkpoint or args.output / "upstream" / "deploy/pre_train/h1/motion.pt"
        if args.checkpoint is None:
            add("walking-checkpoint", args.asset_python, "scripts/dexterous/setup_h1_walking.py",
                "--output", args.output / "upstream")
        reset = args.output / "walking-reset.json"
        add("walking-reset", args.asset_python, "scripts/dexterous/export_h1_walking_reset.py",
            "--robot", robot, "--output", reset)
        add("walking", args.isaac_python, "scripts/dexterous/isaac_h1_walking.py",
            "--robot-usd", imported, "--motors", motors, "--checkpoint", checkpoint,
            "--reset", reset, "--output", args.output / "walking", "--seconds", "14",
            "--headless", "--device", "cuda:0", *([] if args.no_walking_video else ["--record"]),
            report=args.output / "walking" / "report.json")
    return steps


def validate_inputs(args):
    files = [args.asset_python, args.isaac_python,
             args.prepared_dir / "h1-shadow-loopback-v2.xml",
             args.prepared_dir / "h1-shadow-loopback-v2.audit.json",
             args.prepared_dir / "h1-import.motors.json"]
    if args.stage in ("import", "all"):
        files.append(args.prepared_dir / "h1-import.xml")
    if args.stage in ("smoke", "all"):
        files += [args.prepared_dir / "reference.json", args.door_usd]
    if args.robot_usd is not None:
        files.append(args.robot_usd)
    if args.checkpoint is not None:
        files.append(args.checkpoint)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    robot = args.prepared_dir / "h1-shadow-loopback-v2.xml"
    audit = json.loads(robot.with_suffix(".audit.json").read_text(encoding="utf-8"))
    motors = json.loads((args.prepared_dir / "h1-import.motors.json").read_text(encoding="utf-8"))
    if audit.get("mechanics_profile") != PROFILE:
        raise ValueError("The native model must have corrected shadow-loopback-v2 mechanics")
    if audit["robot_xml_sha256"] != sha(robot) or motors["source_xml_sha256"] != sha(robot):
        raise ValueError("Native model, audit and motor contract hashes differ")
    if args.stage == "walking":
        receipt = json.loads(args.ready.read_text(encoding="utf-8"))
        if receipt.get("ready") is not True or receipt.get("mechanics_profile") != PROFILE:
            raise ValueError("Walking requires a successful corrected-model readiness receipt")
        for path, digest in receipt["input_hashes"].items():
            if sha(path) != digest:
                raise ValueError("Readiness input changed: " + path)
        if Path(receipt["native_robot"]).resolve() != robot.resolve():
            raise ValueError("Readiness belongs to a different prepared native model")
        if args.robot_usd is not None and args.robot_usd.resolve() != Path(receipt["robot_usd"]).resolve():
            raise ValueError("--robot-usd differs from the readiness receipt")
    return {str(path): sha(path) for path in files if path not in (args.asset_python, args.isaac_python)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("import", "smoke", "walking", "all"), default="all")
    parser.add_argument("--asset-python", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "out/local-ready")
    parser.add_argument("--door-usd", type=Path, default=ROOT / "assets/doors/db0055_swing_single/door.usda")
    parser.add_argument("--robot-usd", type=Path, help="Imported robot from a prior local import")
    parser.add_argument("--ready", type=Path, help="Passed local readiness receipt for separate walking")
    parser.add_argument("--checkpoint", type=Path, help="Pinned official H1 checkpoint; downloaded if omitted")
    parser.add_argument("--output", type=Path, required=True, help="Fresh run directory; keep this path short")
    parser.add_argument("--timeout-seconds", type=float, default=7200, help="Wall-time limit per stage")
    parser.add_argument("--no-walking-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only (also the default)")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and args.dry_run:
        parser.error("Choose either --execute or --dry-run")
    if not 30 <= args.timeout_seconds <= 86400:
        parser.error("--timeout-seconds must be between 30 and 86400")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            # Preserve a short SUBST drive for the interpreter. Resolving it to
            # its backing directory can reintroduce Windows' 260-character limit.
            setattr(args, key, value.absolute() if key in ("asset_python", "isaac_python") else value.resolve())
    steps = build_plan(args)
    plan = {"execution_requested": args.execute, "mechanics_profile": PROFILE,
            "scope": "Local import/readiness/plane walking only; no complete door task claim",
            "walking_passive_profile": "Historical explicit damping/tanh friction in isaac_h1_walking.py",
            "output": str(args.output), "steps": steps}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if args.output.exists():
        raise FileExistsError("Use a fresh --output directory: " + str(args.output))
    inputs = validate_inputs(args)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", PYTHONUTF8="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", OMNI_KIT_ACCEPT_EULA="YES",
               ACCEPT_EULA="Y", PRIVACY_CONSENT="Y")
    # MuJoCo's Linux EGL backend is unsuitable for this native Windows runtime.
    if os.name == "nt":
        env.pop("MUJOCO_GL", None)
    args.output.mkdir(parents=True)
    save(args.output / "plan.json", plan)
    packages = ["isaacsim", "isaacsim-app", "isaaclab", "torch", "mujoco", "numpy", "scipy", "osqp", "Pillow"]
    version_code = "import importlib.metadata as m,json,sys; d={x.metadata['Name'].lower():x.version for x in m.distributions()}; print(json.dumps({'python':sys.version,'packages':{n:d.get(n.lower()) for n in " + repr(packages) + "}}))"
    manifest = {"started_unix": time.time(), "platform": platform.platform(),
                "hostname": platform.node(), "mechanics_profile": PROFILE,
                "input_sha256": inputs, "source_revision": capture(["git", "rev-parse", "HEAD"], env),
                "source_changes": capture(["git", "status", "--porcelain"], env),
                "source_entrypoint_sha256": {step["argv"][1]: sha(step["argv"][1]) for step in steps},
                "launcher_sha256": sha(__file__),
                "gpu": capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], env),
                "asset_runtime": capture([args.asset_python, "-c", version_code], env),
                "isaac_runtime": capture([args.isaac_python, "-c", version_code], env)}
    save(args.output / "manifest.json", manifest)
    results = {"complete": False, "passed": False, "scope": plan["scope"], "stages": []}
    try:
        for step in steps:
            result = {"name": step["name"], "started_unix": time.time(),
                      "log": str(args.output / (step["name"] + ".log"))}
            results["stages"].append(result)
            save(args.output / "result.json", results)
            print(json.dumps({"stage": step["name"], "status": "running", "log": result["log"]}), flush=True)
            with Path(result["log"]).open("w", encoding="utf-8") as log:
                process = subprocess.Popen(step["argv"], cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, creationflags=CREATE_FLAGS)
                result["pid"] = process.pid
                save(args.output / "result.json", results)
                try:
                    result["returncode"] = process.wait(timeout=args.timeout_seconds)
                except subprocess.TimeoutExpired:
                    result["timed_out"] = True
                    if step["name"] == "smoke" and (args.output / "smoke").is_dir():
                        (args.output / "smoke/stop.request").write_text("Local stage timeout\n", encoding="utf-8")
                        try:
                            process.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    result["returncode"] = process.returncode
                    raise TimeoutError(step["name"] + " exceeded its local wall-time budget")
            result["finished_unix"] = time.time()
            if result["returncode"] != 0:
                raise RuntimeError(step["name"] + " failed; inspect " + result["log"])
            # Kit can exit with code zero after an import-time Python exception.
            # Require the actual completed output before calling a stage passed.
            if step["name"] == "import":
                required = [args.output / "import" / name for name in ("robot.usda", "import-audit.json")]
            elif step["name"] == "smoke":
                required = [args.output / "smoke" / name for name in ("trace.json", "configuration.json")]
            else:
                required = []
            if any(not path.is_file() for path in required):
                raise RuntimeError(step["name"] + " exited without its completed artifacts; inspect " + result["log"])
            if step["name"] == "smoke" and (args.output / "smoke/error.txt").exists():
                raise RuntimeError("Smoke recorded an exception; inspect " + result["log"])
            if step["report"]:
                report = json.loads(Path(step["report"]).read_text(encoding="utf-8"))
                result["report"] = step["report"]
                result["report_sha256"] = sha(step["report"])
                if report.get("passed", report.get("ready")) is not True:
                    raise RuntimeError(step["name"] + " report did not pass")
            result["passed"] = True
            save(args.output / "result.json", results)
            print(json.dumps({"stage": step["name"], "status": "passed"}), flush=True)
        results.update(complete=True, passed=True)
    except Exception as error:
        results.update(complete=True, passed=False, error=str(error))
        if results["stages"]:
            results["stages"][-1].update(passed=False, finished_unix=time.time())
        print(str(error), file=sys.stderr, flush=True)
    finally:
        results["finished_unix"] = time.time()
        save(args.output / "result.json", results)
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

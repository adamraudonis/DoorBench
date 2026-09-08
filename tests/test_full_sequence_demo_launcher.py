"""Dry/local launcher contracts; no Isaac process or trained policy trial."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("full_sequence_launcher", SOURCE / "scripts/isaac/run_full_sequence_demo.py")
launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    from test_isaac_readiness_profiles import contract
    root = tmp_path / "source"; root.mkdir()
    runtime = tmp_path / "runtime"; runtime.mkdir()
    audit, motors = contract()
    robot = runtime / "h1-shadow-loopback-v2.xml"; robot.write_text("synthetic robot bytes")
    motors["source_xml_sha256"] = launcher.file_sha256(robot)
    motor_path = runtime / "motors.json"; motor_path.write_text(json.dumps(motors))
    robot.with_suffix(".audit.json").write_text("{}")
    robot_usd = runtime / "robot.usda"; robot_usd.write_text("synthetic robot USD")
    door_usd = runtime / "door.usda"; door_usd.write_text("synthetic door USD")
    door_xml = runtime / "door.xml"; door_xml.write_text("synthetic door XML")
    inputs = [robot, robot.with_suffix(".audit.json"), motor_path, robot_usd, door_usd]
    receipt = dict(ready=True, mechanics_profile="shadow-loopback-v2", native_robot=str(robot),
        native_robot_sha256=launcher.file_sha256(robot), motor_contract=str(motor_path),
        robot_usd=str(robot_usd), door_usd=str(door_usd),
        passive_tendons=dict(count=8, backend_verified=True, audit=audit),
        input_hashes={str(p):launcher.file_sha256(p) for p in inputs})
    receipt_path = runtime / "ready.json"; receipt_path.write_text(json.dumps(receipt))
    checkpoint = tmp_path / "motion.pt"; checkpoint.write_bytes(b"synthetic checkpoint")
    monkeypatch.setattr(launcher, "POLICY_SHA256", launcher.file_sha256(checkpoint))
    protocol = json.loads((SOURCE / launcher.PROTOCOL).read_text())
    for name in protocol["frozen_inputs"]:
        destination = root / name; destination.parent.mkdir(parents=True, exist_ok=True)
        value = dict(initial_root=[0, 0, 1, 1, 0, 0, 0], joints={"joint":0}) if name == launcher.RESET else {}
        destination.write_text(json.dumps(value)); protocol["frozen_inputs"][name] = launcher.file_sha256(destination)
    protocol["locomotion"]["sha256"] = launcher.POLICY_SHA256
    protocol["native_door_xml_sha256"] = [launcher.file_sha256(door_xml)]
    (root / launcher.PROTOCOL).write_text(json.dumps(protocol))
    files = ["scripts/isaac/run_full_sequence_demo.py", "doorbench/dexterous/controller.py",
        *["scripts/dexterous/" + n for n in ("isaac_opening.py", "physx_teacher.py", "audit_isaac_reference.py", "setup_h1_walking.py")]]
    for name in files:
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("# synthetic source\n")
    return dict(root=root, runtime=runtime, receipt=receipt_path, checkpoint=checkpoint,
                output=tmp_path / "demo", protocol=protocol, registry=tmp_path / "runs.json")


def plan(fixture):
    return launcher.build_plan(fixture["receipt"], fixture["checkpoint"], fixture["output"],
                               root=fixture["root"], asset_python="native-python", isaac_python="isaac-python")


def report(protocol):
    return dict(passed=True, checks={key:True for key in protocol["required_checks"]},
        duration_s=55., physics_dt_s=.002, runtime_robot_pose_writes=0, direct_door_commands=False)


def test_dry_plan_is_complete_and_writes_nothing(fixture):
    p = plan(fixture)
    assert not fixture["output"].exists() and not fixture["registry"].exists()
    assert p["runtime_launched"] is False
    argv = p["argv"]
    for flag in ("--acquisition", "--operate-after-acquisition", "--open-on-latch-clear", "--record", "--enable_cameras"):
        assert flag in argv
    assert argv[argv.index("--seconds") + 1] == "55.0"
    assert argv[argv.index("--operator-compliance-gain") + 1] == "0.5"
    assert argv[argv.index("--output") + 1] == str(fixture["output"] / "trial")
    assert argv[argv.index("--full-sequence-reset") + 1] == str(fixture["root"] / launcher.RESET)
    launcher.verify_frozen_inputs(p)


@pytest.mark.parametrize("change", ["v1", "readiness_input", "reference", "checkpoint", "door", "existing_output"])
def test_stale_or_wrong_input_refuses_plan(fixture, change):
    if change == "v1": fixture["receipt"].write_text(json.dumps(dict(ready=True, mechanics_profile="upstream-v1")))
    elif change == "readiness_input": (fixture["runtime"] / "robot.usda").write_text("changed")
    elif change == "reference": (fixture["root"] / launcher.PREPARATION).write_text("changed")
    elif change == "checkpoint": fixture["checkpoint"].write_bytes(b"other robot weights")
    elif change == "door": (fixture["runtime"] / "door.xml").write_text("other mechanism")
    elif change == "existing_output": fixture["output"].mkdir()
    with pytest.raises((ValueError, FileExistsError)): plan(fixture)


def test_change_after_freeze_is_rejected(fixture):
    p = plan(fixture)
    (fixture["root"] / "doorbench/dexterous/controller.py").write_text("# changed implementation")
    with pytest.raises(ValueError, match="after invocation"): launcher.verify_frozen_inputs(p)


@pytest.mark.parametrize("change", ["missing_gate", "failed_gate", "overall_false", "short", "nan", "wrong_dt", "pose_write", "door_command"])
def test_exit_zero_and_passed_field_cannot_replace_required_evidence(fixture, change):
    r = report(fixture["protocol"])
    if change == "missing_gate": del r["checks"]["all_pad_patches_valid"]
    elif change == "failed_gate": r["checks"]["motor_delivery_matches_command"] = False
    elif change == "overall_false": r["passed"] = False
    elif change == "short": r["duration_s"] = 10
    elif change == "nan": r["duration_s"] = float("nan")
    elif change == "wrong_dt": r["physics_dt_s"] = .004
    elif change == "pose_write": r["runtime_robot_pose_writes"] = 1
    elif change == "door_command": r["direct_door_commands"] = True
    with pytest.raises(ValueError, match="incomplete"): launcher.validate_result(r, fixture["protocol"])


def test_check_only_registers_pipeline_and_never_calls_isaac(fixture, monkeypatch):
    p = plan(fixture); commands = []
    def run(command, **kwargs):
        commands.append(command)
        if "--register-only" in command:
            fixture["registry"].write_text(json.dumps([dict(results=str(fixture["output"]))]))
        else:
            assert command[0] == "native-python"
            (fixture["output"] / "initial-audit.json").write_text(json.dumps(dict(passed=True)))
    monkeypatch.setattr(launcher, "run_logged", run)
    launcher.execute(p, fixture["registry"], check_only=True)
    assert len(commands) == 2 and all(command[0] != "isaac-python" for command in commands)
    assert not (fixture["output"] / "trial").exists()
    pipeline = json.loads((fixture["output"] / "pipeline.json").read_text())
    assert pipeline["result_passed"] is True and "no Isaac" in pipeline["stage"]
    assert "FULL_SEQUENCE_PREFLIGHT_COMPLETE" in (fixture["output"] / "run.log").read_text()


def test_failed_runtime_report_is_retained_and_propagated(fixture, monkeypatch):
    p = plan(fixture)
    def run(command, **kwargs):
        if command[0] == "native-python":
            (fixture["output"] / "initial-audit.json").write_text(json.dumps(dict(passed=True)))
        elif command[0] == "isaac-python":
            trial = fixture["output"] / "trial"; trial.mkdir()
            r = report(fixture["protocol"]); r["passed"] = False
            (trial / "full-sequence-report.json").write_text(json.dumps(r))
    monkeypatch.setattr(launcher, "run_logged", run)
    with pytest.raises(ValueError): launcher.execute(p, fixture["registry"])
    assert json.loads((fixture["output"] / "pipeline.json").read_text())["result_passed"] is False
    assert json.loads((fixture["output"] / "failure.json").read_text())["runtime_launched"] is True
    assert not (fixture["output"] / "run.log").read_text().endswith("FULL_SEQUENCE_DEMO_COMPLETE\n")


def test_complete_result_requires_all_expected_gates(fixture):
    launcher.validate_result(report(fixture["protocol"]), fixture["protocol"])


def test_failed_preflight_never_launches_runtime(fixture, monkeypatch):
    p = plan(fixture); commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "native-python":
            (fixture["output"] / "initial-audit.json").write_text(json.dumps(dict(passed=False)))
    monkeypatch.setattr(launcher, "run_logged", run)
    with pytest.raises(ValueError, match="reset geometry"): launcher.execute(p, fixture["registry"])
    assert all(command[0] != "isaac-python" for command in commands)
    assert json.loads((fixture["output"] / "failure.json").read_text())["runtime_launched"] is False


def test_subprocess_error_and_output_are_retained_without_shell(tmp_path):
    output = tmp_path / "log.txt"
    with output.open("w") as log, pytest.raises(subprocess.CalledProcessError):
        launcher.run_logged([sys.executable, "-c", "print('retained failure'); raise SystemExit(7)"],
                            cwd=tmp_path, env={}, log=log)
    assert "retained failure" in output.read_text()

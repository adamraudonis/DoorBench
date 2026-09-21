"""CPU-only synthetic fixtures for the real Isaac adapter; no simulator/API call."""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from doorbench.dexterous.isaac_jev_progress import (
    CONTACT_SOURCE, IsaacJevProgressGate, isaac_progress_snapshot, read_jev_plan,
    validate_isaac_jev_arguments, write_jev_protocol,
)
from doorbench.dexterous.jev_advisor import AstraPlan, MODEL
from doorbench.dexterous.jev_progress_advisor import JevProgressAdvisor, admissible_actions


def fixture(t=12.):
    plan = AstraPlan("synthetic-isaac-press", "lever_operation", "CPU fixture only", contact_threshold_N=.2)
    pad = dict(sim_time_s=t, physics_dt_s=.002, valid_pad_grasp=True,
        digit_forces_N=dict(ff=.5, mf=.6, rf=.7, lf=.8, th=.9),
        hand="rh", grasp_profile="volar-phalange-v1",
        distal_pad_grasp=dict(valid_pad_grasp=False, digit_forces_N=dict.fromkeys(("ff", "mf", "rf", "lf", "th"), 0.)),
        normal_pair_force_consistency_error_N=1e-7,
        contract_scope="Privileged PhysX handle-body patch audit; non-lever digit contacts count as misplaced",
        raw_evidence=dict(schema="doorbench.shadow-raw-pad-evidence.v1", clock="physx-interval-end",
            scope="complete-handle-body", interval_start_s=t-.002, interval_end_s=t, geometry_time_s=t))
    values = dict(episode_id="synthetic-isaac-fixture", sample_id=round(t/.002),
        simulation_time_s=t, phase="lever_operation", pad=pad, grasp_profile="volar-phalange-v1",
        root_state=[0., 0., .85, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        joint_position=[0.]*69, joint_velocity=[0.]*69, torso_tilt_deg=2., stance_status="solved",
        mechanical_audit=dict(max_joint_stop_penetration_rad=0., max_loopback_violation_rad=0.,
            max_self_penetration_m=0., max_nonfoot_environment_penetration_m=0., max_hand_door_penetration_m=0.),
        motor_delivery_error=1e-5, motor_caps_ok=True, physics_dt=.002)
    return values, plan


def test_selected_physx_profile_and_source_are_not_replaced_with_distal_diagnostic():
    values, plan = fixture()
    sample = isaac_progress_snapshot(now_s=10., **values)
    assert sample.normal_loads_N == (.5, .6, .7, .8, .9)
    assert sample.grip_stable and sample.sensor_valid and sample.contact_source == CONTACT_SOURCE
    assert "continue_press" in admissible_actions(sample, plan)


@pytest.mark.parametrize("field,value", [
    ("raw_evidence", {}), ("sim_time_s", 11.998), ("grasp_profile", "distal-pad-v1"),
    ("normal_pair_force_consistency_error_N", .01), ("hand", "lh"),
    ("digit_forces_N", {}), ("digit_forces_N", dict(ff=float("nan"))),
])
def test_missing_wrong_source_or_wrong_epoch_physx_evidence_closes_permission(field, value):
    values, plan = fixture()
    values["pad"][field] = value
    sample = isaac_progress_snapshot(now_s=10., **values)
    assert not sample.sensor_valid
    assert "continue_press" not in admissible_actions(sample, plan)


@pytest.mark.parametrize("field,value", [
    ("stance_status", None), ("stance_status", "failed"), ("motor_caps_ok", False),
    ("motor_delivery_error", .001), ("torso_tilt_deg", 14.), ("joint_velocity", [float("nan")]*69),
    ("mechanical_audit", {}),
])
def test_measured_posture_physical_guard_and_previous_solve_are_required(field, value):
    values, plan = fixture()
    values[field] = value
    sample = isaac_progress_snapshot(now_s=10., **values)
    assert not sample.local_continue_allowed
    assert "continue_press" not in admissible_actions(sample, plan)


def cli_args(**overrides):
    return SimpleNamespace(**dict(dict(jev_sample_period=.2, jev_progress_plan="plan.json",
        acquisition=True, operate_after_acquisition=True, native_robot="robot.xml",
        acquisition_stance_profile="landed-foot-v1", time_scale=1.), **overrides))


@pytest.mark.parametrize("name,value", [
    ("full_sequence_reset", "reset.json"), ("full_opening", True), ("traverse", True),
    ("standing_transfer_route", "route.json"), ("sensor_policy_checkpoint", "actor.pt"),
    ("sensor_layout", "sensors.json"),
    ("sensor_balance_calibration", "balance.json"), ("sensor_locomotion_calibration", "walk.json"),
    ("sensor_arm_schedule", "arms.json"), ("sensor_reach_protocol", "reach.json"),
    ("sensor_acquisition_protocol", "acquire.json"), ("time_scale", 2.),
    ("acquisition_stance_profile", None), ("acquisition", False), ("hold_attained_grasp", True),
])
def test_unsupported_paths_rejected_before_physics(name, value):
    with pytest.raises(ValueError):
        validate_isaac_jev_arguments(cli_args(**{name: value}))


def test_default_disabled_and_explicit_period_are_validated():
    validate_isaac_jev_arguments(cli_args(jev_progress_plan=None, acquisition=False))
    validate_isaac_jev_arguments(cli_args())
    for period in (0., .04, float("nan"), float("inf"), 10.01):
        with pytest.raises(ValueError):
            validate_isaac_jev_arguments(cli_args(jev_sample_period=period))
    with pytest.raises(ValueError):
        validate_isaac_jev_arguments(cli_args(jev_progress_plan=None, jev_sample_period=.3))


def test_real_parser_accepts_latch_clear_with_gate_and_optional_visual_profile(monkeypatch, tmp_path):
    from test_isaac_traversal_runner import run_parser
    _, plan = fixture()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(asdict(plan)))
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-parser-fixture-no-request")
    args = run_parser(monkeypatch, ["--acquisition", "--operate-after-acquisition",
        "--native-robot", "robot.xml", "--acquisition-stance-profile", "landed-foot-v1",
        "--open-on-latch-clear", "--jev-progress-plan", str(path),
        "--review-render-profile", "native-materials-v1"])
    assert args.open_on_latch_clear and args.jev_progress_plan == path
    assert args.review_render_profile == "native-materials-v1"


def test_plan_copy_hash_and_declared_privileged_profile_match_exact_used_bytes(tmp_path):
    _, plan = fixture()
    original = tmp_path / "source.json"
    content = (json.dumps(asdict(plan), indent=3)+"\n").encode()
    original.write_bytes(content)
    parsed, used = read_jev_plan(original)
    original.write_text("{}")  # A later source change cannot rewrite the plan used.
    receipt = write_jev_protocol(tmp_path, plan=parsed, plan_bytes=used, source_path=original,
        grasp_profile="volar-phalange-v1", sample_period=.2, physics_dt=.002)
    assert (tmp_path / "jev-progress-plan-input.json").read_bytes() == content
    assert receipt["plan_sha256"] == hashlib.sha256(content).hexdigest()
    assert receipt["contact_source"] == CONTACT_SOURCE and receipt["grasp_profile"] == "volar-phalange-v1"
    assert receipt["maximum_observation_age_simulation_s"] == .25
    original.write_text(json.dumps(dict(asdict(plan), phase="acquire_and_hold")))
    with pytest.raises(ValueError, match="lever_operation"):
        read_jev_plan(original)


class ImmediateWorker:
    """Fake worker that completes immediately; the gate still must poll first."""
    def __init__(self, now, action="continue_press"):
        self.events, self.latest, self.closed = [], None, False
        def transport(request):
            choices = request["questions"]["progress"]["criteria"]
            return dict(model=MODEL, answers=dict(progress=dict(type="choice", choice=action,
                confidence=.99, probabilities={key: float(key == action) for key in choices})))
        self.advisor = JevProgressAdvisor(transport, clock=lambda: now[0])

    def poll(self, snapshot, plan):
        self.events.append("poll")
        return None if self.latest is None else self.advisor.revalidate(self.latest, snapshot, plan)

    def submit(self, snapshot, plan):
        self.events.append("submit")
        self.latest = self.advisor.evaluate(snapshot, plan)
        return True

    def close(self):
        self.closed = True


INFO = dict(press_progress_s=.1, opening_progress_s=0., progress_rate=.5)


def test_poll_before_submit_and_unique_replies_separate_from_per_step_consumption(tmp_path):
    values, plan = fixture()
    now = [10.]
    worker = ImmediateWorker(now)
    gate = IsaacJevProgressGate(plan, worker, tmp_path, clock=lambda: now[0])
    allow, context = gate.choose(**values)
    assert not allow and worker.events == ["poll", "submit"]
    gate.record_submission(context, INFO)
    for _ in range(2):
        now[0] += .01
        allow, context = gate.choose(**values)
        assert allow
        gate.record_submission(context, INFO)
    gate.close()
    gate.close()
    result = json.loads((tmp_path / "jev-progress-summary.json").read_text())
    assert result["requests_submitted"] == result["model_replies_consumed"] == 1
    assert result["continue_intervals"] == 2 and result["pause_intervals"] == 1
    assert worker.closed and gate.stream.closed


@pytest.mark.parametrize("kind", ["wall", "simulation", "contact"])
def test_every_controller_step_revalidates_lease_and_current_physx_state(tmp_path, kind):
    values, plan = fixture()
    now = [10.]
    gate = IsaacJevProgressGate(plan, ImmediateWorker(now), tmp_path, clock=lambda: now[0])
    try:
        gate.choose(**values)
        now[0] = 10.01
        assert gate.choose(**values)[0]
        if kind == "wall":
            now[0] = 10.8
        elif kind == "simulation":
            values, _ = fixture(12.3)
        else:
            values["pad"]["digit_forces_N"]["ff"] = 0.
        assert not gate.choose(**values)[0]
    finally:
        gate.close()


def test_model_pause_is_honored_inside_safe_admissible_set(tmp_path):
    values, plan = fixture()
    now = [10.]
    gate = IsaacJevProgressGate(plan, ImmediateWorker(now, "pause_press"), tmp_path, clock=lambda: now[0])
    try:
        gate.choose(**values)
        allow, context = gate.choose(**values)
        assert not allow and context["local_continue_allowed"] and context["action"] == "pause_press"
    finally:
        gate.close()


def test_runtime_failure_is_persistent_pause_and_never_leaks_exception_text(tmp_path):
    values, plan = fixture()
    class BrokenWorker:
        closed = False
        def poll(self, *_):
            raise RuntimeError("sensitive transport diagnostic")
        def close(self):
            self.closed = True
    worker = BrokenWorker()
    gate = IsaacJevProgressGate(plan, worker, tmp_path, clock=lambda: 10.)
    for _ in range(2):
        allow, context = gate.choose(**values)
        assert not allow and context["reason"] == "advisor_runtime_error_RuntimeError"
        gate.record_submission(context, INFO)
    gate.close()
    assert worker.closed
    assert "sensitive transport diagnostic" not in (tmp_path / "jev-progress.jsonl").read_text()
    assert json.loads((tmp_path / "jev-progress-summary.json").read_text())["pause_intervals"] == 2


def test_actual_script_has_opt_in_boundary_before_engine_and_cleanup_in_finally():
    path = Path(__file__).parents[1] / "scripts/dexterous/isaac_opening.py"
    source = path.read_text()
    ast.parse(source)
    assert source.index("try:validate_isaac_jev_arguments(a)") < source.index("launcher=AppLauncher(a)")
    assert "jev_gate is not None and operation.started is not None and operation.open_started is None" in source
    assert "finally:\n        if jev_gate is not None:jev_gate.close()" in source
    assert "('jev_advisor.py','jev_progress_advisor.py','isaac_jev_progress.py')" in source
    assert "out/'jev-progress-plan-input.json'" in source

"""Opt-in Jev clock gate for standalone privileged PhysX lever operation.

No Isaac, GPU, physics-step or motor API is imported here. The caller supplies
copies of the preceding solved contact interval and the current physical state.
This is object-identified simulation evidence, not visual or tactile inference.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

from .jev_advisor import AstraPlan, MODEL, _finite
from .jev_progress_advisor import DIGITS, ProgressSnapshot

CONTACT_SOURCE = "privileged_physx_digit_handle_contact"


def validate_isaac_jev_arguments(args):
    period = args.jev_sample_period
    if not _finite(period) or not .05 <= period <= 10.:
        raise ValueError("Jev sample period must be 0.05..10 wall-clock seconds")
    if args.jev_progress_plan is None:
        if period != .2:
            raise ValueError("An explicit Jev sample period requires --jev-progress-plan")
        return
    forbidden = ("full_sequence_reset", "full_opening", "traverse", "standing_transfer_route",
                 "sensor_layout", "sensor_policy_checkpoint", "sensor_balance_calibration", "sensor_locomotion_calibration",
                 "sensor_arm_schedule", "sensor_reach_protocol", "sensor_acquisition_protocol",
                 "panel_push", "mechanism_test", "hold_attained_grasp")
    if (not args.acquisition or not args.operate_after_acquisition or not args.native_robot
            or args.acquisition_stance_profile != "landed-foot-v1" or args.time_scale != 1.
            or any(getattr(args, name, None) for name in forbidden)):
        raise ValueError("Jev progress requires standalone landed-foot acquisition/operation at the original clock; sensor, transfer, full-sequence and traversal paths are unsupported")


def read_jev_plan(path):
    """Parse once; retain exactly the bytes used for the controller and its hash."""
    content = Path(path).read_bytes()
    plan = AstraPlan(**json.loads(content))
    if plan.phase != "lever_operation":
        raise ValueError("Jev progress plan must explicitly authorize lever_operation")
    return plan, content


def write_jev_protocol(output, *, plan, plan_bytes, source_path, grasp_profile, sample_period, physics_dt):
    output = Path(output)
    (output / "jev-progress-plan-input.json").write_bytes(plan_bytes)
    receipt = dict(schema="doorbench.isaac-jev-progress.v1", plan=asdict(plan),
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), source_path=str(Path(source_path).resolve()),
        model=MODEL, sample_period_wall_s=sample_period, physics_dt_s=physics_dt,
        maximum_observation_age_wall_s=.75, maximum_observation_age_simulation_s=.25,
        permission_duration_wall_s=.35, minimum_model_confidence=.7,
        phase="lever_operation", default_other_phase_progress=True,
        contact_source=CONTACT_SOURCE, grasp_profile=grasp_profile,
        contact_epoch="preceding solved PhysX interval ending at current controller time",
        balance_source="current measured root posture and preceding completed stance-solver status",
        command_scope="permission to advance the existing press reference clock; all motor/physics loops continue",
        api_key_saved=False)
    (output / "jev-progress-plan.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    return receipt


def _number(value):
    return float(value) if _finite(value) else None


def isaac_progress_snapshot(*, episode_id, sample_id, simulation_time_s, now_s,
                            phase, pad, grasp_profile, root_state, joint_position,
                            joint_velocity, torso_tilt_deg, stance_status,
                            mechanical_audit, motor_delivery_error, motor_caps_ok,
                            physics_dt, prior_loads=None):
    """Missing/nonfinite/unmatched evidence removes permission at this step.

    Selected digit loads and selected grasp validity both come from the declared
    PhysX grasp profile. A volar-profile run must not substitute its nested distal
    diagnostic. Slip is not measured here and is never invented for the model.
    """
    pad = pad if isinstance(pad, dict) else {}
    raw = pad.get("raw_evidence", {})
    raw = raw if isinstance(raw, dict) else {}
    measured = pad.get("digit_forces_N", {})
    measured = measured if isinstance(measured, dict) else {}
    values = tuple(_number(measured.get(digit)) for digit in DIGITS)
    values = tuple(v if v is not None and v >= 0 else None for v in values)
    epsilon = max(1e-8, physics_dt*1e-5)
    def aligned(value, target):
        return _finite(value) and abs(value-target) <= epsilon
    time_aligned = (simulation_time_s >= physics_dt
        and aligned(pad.get("sim_time_s"), simulation_time_s)
        and aligned(pad.get("physics_dt_s"), physics_dt)
        and aligned(raw.get("interval_start_s"), simulation_time_s-physics_dt)
        and aligned(raw.get("interval_end_s"), simulation_time_s)
        and aligned(raw.get("geometry_time_s"), simulation_time_s))
    source_valid = (pad.get("hand") == "rh" and pad.get("grasp_profile") == grasp_profile
        and raw.get("schema") == "doorbench.shadow-raw-pad-evidence.v1"
        and raw.get("clock") == "physx-interval-end" and raw.get("scope") == "complete-handle-body"
        and str(pad.get("contract_scope", "")).startswith("Privileged PhysX handle-body patch audit")
        and _finite(pad.get("normal_pair_force_consistency_error_N"))
        and 0 <= pad["normal_pair_force_consistency_error_N"] <= .001)
    def finite_vector(value, size=None):
        return (isinstance(value, (tuple, list)) and len(value) > 0
            and (size is None or len(value) == size) and all(_finite(v) for v in value))
    finite_state = (finite_vector(root_state, 13) and finite_vector(joint_position)
        and finite_vector(joint_velocity) and len(joint_position) == len(joint_velocity)
        and _finite(torso_tilt_deg))
    valid = bool(time_aligned and source_valid and finite_state and all(v is not None for v in values))
    posture = bool(finite_state and root_state[2] > .7 and 0 <= torso_tilt_deg < 12)
    balance = None if stance_status is None else bool(posture and stance_status in ("solved", "solved inaccurate"))
    audit = mechanical_audit if isinstance(mechanical_audit, dict) else {}
    bounds = dict(max_joint_stop_penetration_rad=.02, max_loopback_violation_rad=.02,
        max_self_penetration_m=.003, max_nonfoot_environment_penetration_m=.003,
        max_hand_door_penetration_m=.003)
    physical = bool(valid and motor_caps_ok is True and _finite(motor_delivery_error)
        and 0 <= motor_delivery_error < 1e-4
        and all(_finite(audit.get(key)) and 0 <= audit[key] < bound for key, bound in bounds.items()))
    grip = pad.get("valid_pad_grasp")
    grip = grip if type(grip) is bool else None
    return ProgressSnapshot(episode_id=episode_id, sample_id=sample_id,
        simulation_time_s=simulation_time_s, capture_monotonic_s=now_s, phase=phase,
        normal_loads_N=values, grip_stable=grip, balance_ready=balance,
        local_continue_allowed=bool(physical and grip is True and balance is True),
        contact_source=CONTACT_SOURCE, sensor_valid=valid, prior_normal_loads_N=prior_loads)


class IsaacJevProgressGate:
    """Nonblocking controller boundary with separate reply and submission logs."""
    def __init__(self, plan, advisor, output, *, sample_period=.2, clock=time.monotonic):
        if not _finite(sample_period) or not .05 <= sample_period <= 10.:
            raise ValueError("Jev sample period must be 0.05..10 wall-clock seconds")
        self.plan, self.advisor, self.output = plan, advisor, Path(output)
        self.sample_period, self.clock = sample_period, clock
        self.last_submit = self.prior_loads = self.failure = None
        self.seen, self.latencies = set(), []
        self.closed = False
        self.counts = dict(requests_submitted=0, advisor_replies_consumed=0, model_replies_consumed=0,
            continue_intervals=0, pause_intervals=0, astra_requests=0)
        self.stream = (self.output / "jev-progress.jsonl").open("w", encoding="utf-8")

    def write(self, value):
        self.stream.write(json.dumps(value, allow_nan=False)+"\n")

    def choose(self, **measurements):
        if self.closed:
            raise RuntimeError("Jev gate is closed")
        now = self.clock()
        snapshot = isaac_progress_snapshot(now_s=now, prior_loads=self.prior_loads, **measurements)
        advice = None
        try:
            if self.failure is None:
                # A completed worker reply must be consumed before submitting again.
                advice = self.advisor.poll(snapshot, self.plan)
                if advice is not None and advice.sample_id not in self.seen:
                    self.seen.add(advice.sample_id)
                    self.counts["advisor_replies_consumed"] += 1
                    self.counts["model_replies_consumed"] += int(advice.model is not None)
                    if advice.model is not None and advice.latency_ms is not None:
                        self.latencies.append(advice.latency_ms)
                    self.counts["astra_requests"] += int(advice.proposed_action == "request_astra")
                    self.write(dict(event="reply_consumed", consumed_sample_id=snapshot.sample_id,
                        consumed_simulation_time_s=snapshot.simulation_time_s, advice=advice.to_dict()))
                if self.last_submit is None or now-self.last_submit >= self.sample_period:
                    if self.advisor.submit(snapshot, self.plan):
                        self.last_submit, self.prior_loads = now, tuple(snapshot.normal_loads_N)
                        self.counts["requests_submitted"] += 1
                        self.write(dict(event="request_submitted", snapshot=asdict(snapshot),
                            grasp_profile=measurements["grasp_profile"], preceding_stance_status=measurements["stance_status"]))
        except Exception as exc:
            # Never include exception text: provider diagnostics may contain secrets.
            self.failure = "advisor_runtime_error_"+type(exc).__name__
            advice = None
            self.write(dict(event="advisor_failure", reason=self.failure, sample_id=snapshot.sample_id))
        allow = bool(self.failure is None and advice is not None and advice.advance)
        return allow, dict(event="controller_submission", sample_id=snapshot.sample_id,
            simulation_time_s=snapshot.simulation_time_s, capture_monotonic_s=now,
            allow_progress=allow, advice_sample_id=None if advice is None else advice.sample_id,
            action="pause_press" if advice is None else advice.action,
            reason=self.failure or ("awaiting_advice" if advice is None else advice.reason),
            advice_age_wall_s=None if advice is None else now-advice.capture_monotonic_s,
            advice_age_simulation_s=None if advice is None else snapshot.simulation_time_s-advice.simulation_time_s,
            local_continue_allowed=snapshot.local_continue_allowed, sensor_valid=snapshot.sensor_valid,
            actual_grasp_qualified=snapshot.grip_stable, contact_source=CONTACT_SOURCE,
            grasp_profile=measurements["grasp_profile"], preceding_stance_status=measurements["stance_status"])

    def record_submission(self, context, info):
        self.counts["continue_intervals" if context["allow_progress"] else "pause_intervals"] += 1
        self.write(dict(context, progress={key: info[key] for key in
            ("press_progress_s", "opening_progress_s", "progress_rate")}))
        if (self.counts["continue_intervals"]+self.counts["pause_intervals"]) % 100 == 0:
            self.stream.flush()

    def summary(self):
        return dict(scope="Live asynchronous Jev decisions gate privileged PhysX lever progression only; no vision, release, traversal or learned-policy claim",
            plan_id=self.plan.plan_id, model=MODEL, sample_period_wall_s=self.sample_period,
            **self.counts, latency_ms_mean=statistics.mean(self.latencies) if self.latencies else None,
            latency_ms_median=statistics.median(self.latencies) if self.latencies else None,
            latency_ms_max=max(self.latencies) if self.latencies else None,
            contact_source=CONTACT_SOURCE, decision_log="jev-progress.jsonl",
            count_note="Requests count worker submissions; invalid evidence can abstain before HTTP. Model replies count unique parsed provider replies, separately from every physical submission reusing their lease.",
            failure=self.failure, api_key_saved=False)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.advisor.close()
        finally:
            result = self.summary()
            try:
                self.write(dict(event="closed", **result))
            finally:
                self.stream.close()
            (self.output / "jev-progress-summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")

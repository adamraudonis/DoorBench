"""Jev chooses whether an already planned press may progress, never motor force.

The independent local guard constructs an admissible action set. Jev may pause
even when the guard permits progress; the deterministic comparator is only an
experiment metric. Missing, expired, malformed or low-confidence advice pauses
the reference progression clock while the plant's grip/balance loops keep running.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import math
import time
import urllib.error

from .jev_advisor import AstraPlan, MODEL, _finite

PHASES = ("acquisition", "lever_operation", "partial_opening", "complete")
DIGITS = ("ff", "mf", "rf", "lf", "th")
PROGRESS_ACTIONS = ("continue_press", "pause_press", "request_astra")


@dataclass(frozen=True)
class ProgressSnapshot:
    episode_id: str
    sample_id: int
    simulation_time_s: float
    capture_monotonic_s: float
    phase: str
    normal_loads_N: tuple[float | None, ...]
    grip_stable: bool | None
    balance_ready: bool | None
    local_continue_allowed: bool
    contact_source: str = "privileged_digit_handle_contact"
    sensor_valid: bool = True
    prior_normal_loads_N: tuple[float | None, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id or self.phase not in PHASES:
            raise ValueError("Require an episode identity and declared operation phase")
        if type(self.sample_id) is not int or self.sample_id < 0:
            raise ValueError("Require a nonnegative sample identity")
        if not all(_finite(v) and v >= 0 for v in (self.simulation_time_s, self.capture_monotonic_s)):
            raise ValueError("Require finite nonnegative clocks")
        for field in ("normal_loads_N", "prior_normal_loads_N"):
            values = getattr(self, field)
            if values is None and field == "prior_normal_loads_N":
                continue
            if not isinstance(values, (tuple, list)) or len(values) != 5 or not all(v is None or (_finite(v) and v >= 0) for v in values):
                raise ValueError("Require five finite nonnegative digit normal loads or None")
            object.__setattr__(self, field, tuple(values))
        if any(v is not None and type(v) is not bool for v in (self.grip_stable, self.balance_ready)):
            raise ValueError("Grip/balance readiness must be measured bools or None")
        if type(self.local_continue_allowed) is not bool or type(self.sensor_valid) is not bool:
            raise ValueError("Explicit local admission and sensor validity required")
        if self.contact_source not in ("privileged_digit_handle_contact", "privileged_physx_digit_handle_contact", "calibrated_tactile_normal_load"):
            raise ValueError("Require explicit privileged or calibrated tactile contact provenance")


def admissible_actions(sample: ProgressSnapshot, plan: AstraPlan) -> tuple[str, ...]:
    """A missing/unsafe guard can never be overruled by model confidence."""
    allowed = ("pause_press", "request_astra")
    loaded = all(v is not None and plan.contact_threshold_N <= v < plan.excessive_load_N for v in sample.normal_loads_N)
    if (sample.phase == "lever_operation" and plan.phase == "lever_operation" and sample.sensor_valid and sample.local_continue_allowed
            and sample.grip_stable is True and sample.balance_ready is True and loaded):
        allowed = ("continue_press", *allowed)
    return allowed


def reference_progress_action(sample: ProgressSnapshot, plan: AstraPlan) -> str:
    """Comparator only: the advisor may legitimately choose a more cautious pause."""
    return "continue_press" if "continue_press" in admissible_actions(sample, plan) else "pause_press"


def build_progress_request(sample: ProgressSnapshot, plan: AstraPlan) -> dict:
    def bucket(value):
        return ("unavailable" if value is None else "overloaded" if value >= plan.excessive_load_N
                else "loaded" if value >= plan.contact_threshold_N else "unloaded")
    current = {digit: bucket(value) for digit, value in zip(DIGITS, sample.normal_loads_N)}
    prior = ({digit: bucket(value) for digit, value in zip(DIGITS, sample.prior_normal_loads_N)}
             if sample.prior_normal_loads_N is not None else None)
    descriptions = {
        "continue_press": "Permit the existing local planned lever press to progress briefly. Only when contact and balance evidence supports continuing; do not generate a new target.",
        "pause_press": "Pause the planned lever progression while existing local grip/balance motors continue. Use for uncertain, changing or inadequate contact evidence.",
        "request_astra": "Pause and request a new high-level plan because the situation cannot be handled confidently inside the current press primitive.",
    }
    return dict(model=MODEL,
        state=dict(plan=dict(plan_id=plan.plan_id, objective=plan.objective), phase=sample.phase,
                   contact_source=sample.contact_source, current_digit_contact=current,
                   prior_digit_contact=prior, measured_grip_stable=sample.grip_stable,
                   local_balance_ready=sample.balance_ready, sensor_valid=sample.sensor_valid,
                   local_progress_admission=sample.local_continue_allowed,
                   permitted_actions=list(admissible_actions(sample, plan))),
        questions=dict(progress=dict(type="choice",
            instructions="Choose one short-lived progress decision for the existing lever-press plan using the current and prior contact observations. Continue only if the observations support maintaining grip; a cautious pause is permitted even when the local guard allows continuation. This controls the reference progression clock, not grip forces, balance motors, hand release or walking.",
            criteria={action: descriptions[action] for action in admissible_actions(sample, plan)})))


@dataclass(frozen=True)
class ProgressAdvice:
    episode_id: str
    sample_id: int
    plan_id: str
    phase: str
    capture_monotonic_s: float
    received_monotonic_s: float
    permission_expires_monotonic_s: float
    accepted: bool = False
    action: str = "pause_press"
    reason: str = "awaiting_advice"
    proposed_action: str | None = None
    confidence: float | None = None
    model: str | None = None
    latency_ms: float | None = None
    comparator_action: str | None = None
    simulation_time_s: float = 0.
    # Immutable evaluation provenance stays distinct from per-tick permission.
    evaluation_reason: str | None = None
    permission_rejection_reason: str | None = None
    provider_error_streak: int = 0
    retry_backoff_s: float = 0.
    retry_not_before_monotonic_s: float = 0.

    @property
    def advance(self) -> bool:
        return self.accepted and self.action == "continue_press"

    def to_dict(self):
        return asdict(self)


class JevProgressAdvisor:
    def __init__(self, transport, *, max_age_s=.75, permission_duration_s=.35,
                 minimum_confidence=.7, max_simulation_age_s=.25, clock=time.monotonic):
        if not all(_finite(v) and v > 0 for v in (max_age_s, permission_duration_s, max_simulation_age_s)) or permission_duration_s > max_age_s:
            raise ValueError("Require finite bounded freshness and permission windows")
        if not _finite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
            raise ValueError("Confidence threshold must be in [0,1]")
        self.transport, self.clock = transport, clock
        self.max_age_s, self.permission_duration_s = max_age_s, permission_duration_s
        self.max_simulation_age_s = max_simulation_age_s
        self.minimum_confidence = minimum_confidence

    def evaluate(self, sample: ProgressSnapshot, plan: AstraPlan) -> ProgressAdvice:
        result = self._evaluate(sample, plan)
        return replace(result, evaluation_reason=result.reason)

    def _evaluate(self, sample: ProgressSnapshot, plan: AstraPlan) -> ProgressAdvice:
        started = self.clock()
        receipt = ProgressAdvice(sample.episode_id, sample.sample_id, plan.plan_id, sample.phase,
                                 sample.capture_monotonic_s, started, started,
                                 comparator_action=reference_progress_action(sample, plan),
                                 simulation_time_s=sample.simulation_time_s)
        if not 0 <= started-sample.capture_monotonic_s <= self.max_age_s:
            return replace(receipt, reason="stale_or_future_telemetry")
        if not sample.sensor_valid or any(v is None for v in sample.normal_loads_N):
            return replace(receipt, reason="missing_or_invalid_contact")
        request = build_progress_request(sample, plan)
        try:
            response = self.transport(request)
        except Exception as exc:
            received = self.clock()
            reason = f"provider_http_{exc.code}" if isinstance(exc, urllib.error.HTTPError) else f"provider_error_{type(exc).__name__}"
            return replace(receipt, received_monotonic_s=received, reason=reason, latency_ms=(received-started)*1000.)
        received = self.clock()
        receipt = replace(receipt, received_monotonic_s=received, latency_ms=(received-started)*1000.)
        try:
            answer = response["answers"]["progress"]
            choice, confidence, probabilities = answer["choice"], answer["confidence"], answer["probabilities"]
            requested = request["questions"]["progress"]["criteria"]
            if (response["model"] != MODEL or answer["type"] != "choice" or choice not in requested
                    or set(probabilities) != set(requested)
                    or not all(_finite(v) and 0 <= v <= 1 for v in (confidence, *probabilities.values()))
                    or not math.isclose(sum(probabilities.values()), 1., abs_tol=.02)
                    or probabilities[choice] < max(probabilities.values())):
                raise ValueError("Invalid progress response")
        except (KeyError, ValueError, TypeError, AttributeError):
            return replace(receipt, reason="invalid_provider_response")
        receipt = replace(receipt, proposed_action=choice, confidence=confidence, model=MODEL)
        if not 0 <= received-sample.capture_monotonic_s <= self.max_age_s:
            return replace(receipt, reason="response_expired")
        if confidence < self.minimum_confidence or probabilities[choice] < self.minimum_confidence:
            return replace(receipt, reason="low_confidence")
        if choice == "request_astra":
            return replace(receipt, reason="model_requested_astra")
        expires = min(sample.capture_monotonic_s+self.max_age_s, received+self.permission_duration_s)
        return replace(receipt, accepted=True, action=choice, permission_expires_monotonic_s=expires,
                       reason="accepted_progress_advice")

    def revalidate(self, advice: ProgressAdvice, latest: ProgressSnapshot, plan: AstraPlan) -> ProgressAdvice:
        now = self.clock()
        reason = None
        if advice.episode_id != latest.episode_id or advice.plan_id != plan.plan_id or advice.phase != latest.phase:
            reason = "episode_plan_or_phase_changed"
        elif latest.sample_id < advice.sample_id:
            reason = "out_of_order_telemetry"
        elif not 0 <= latest.simulation_time_s-advice.simulation_time_s <= self.max_simulation_age_s:
            reason = "simulation_observation_expired"
        elif not 0 <= now-latest.capture_monotonic_s <= self.max_age_s or not 0 <= now-advice.capture_monotonic_s <= self.max_age_s:
            reason = "advice_or_latest_telemetry_expired"
        elif advice.advance and now > advice.permission_expires_monotonic_s:
            reason = "progress_permission_expired"
        elif advice.advance and "continue_press" not in admissible_actions(latest, plan):
            reason = "local_guard_removed_progress_permission"
        if reason:
            return replace(advice, accepted=False, action="pause_press", reason=reason,
                           evaluation_reason=advice.evaluation_reason or advice.reason,
                           permission_rejection_reason=reason)
        return advice


class AsyncJevProgressAdvisor:
    """Single query in flight, reusable leased advice, nonblocking controller API.

    The controller calls poll at every physics decision to recheck permissions.
    It advances the progression clock only when the returned advice.advance is
    true. None means pause. submit never discards an unconsumed completed reply.
    Provider errors back off new submissions without sleeping or extending any
    advice lease. Errors still replace prior advice and keep progress paused.
    """

    def __init__(self, advisor: JevProgressAdvisor, *, error_backoff_initial_s=.25,
                 error_backoff_maximum_s=2.):
        if (not all(_finite(v) for v in (error_backoff_initial_s, error_backoff_maximum_s))
                or not .05 <= error_backoff_initial_s <= error_backoff_maximum_s <= 10.):
            raise ValueError("Provider retry backoff must be within 0.05..10 wall seconds")
        self.advisor = advisor
        self.error_backoff_initial_s = float(error_backoff_initial_s)
        self.error_backoff_maximum_s = float(error_backoff_maximum_s)
        self._provider_error_streak = 0
        self._retry_delay_s = self._retry_not_before_s = 0.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="doorbench-jev-progress")
        self._future = self._latest = None
        self._closed = False

    def submit(self, sample: ProgressSnapshot, plan: AstraPlan) -> bool:
        if (self._closed or self._future is not None
                or self.advisor.clock() < self._retry_not_before_s):
            return False
        self._future = self._pool.submit(self.advisor.evaluate, sample, plan)
        return True

    def poll(self, latest: ProgressSnapshot, plan: AstraPlan) -> ProgressAdvice | None:
        if self._closed:
            return None
        if self._future is not None and self._future.done():
            future, self._future = self._future, None
            self._latest = future.result()
            result_reason = self._latest.evaluation_reason or self._latest.reason
            if result_reason.startswith("provider_") or result_reason == "invalid_provider_response":
                self._provider_error_streak += 1
                self._retry_delay_s = (self.error_backoff_initial_s if self._provider_error_streak == 1
                    else min(self.error_backoff_maximum_s, self._retry_delay_s*2.))
                self._retry_not_before_s = self._latest.received_monotonic_s+self._retry_delay_s
            elif self._latest.model is not None:
                self._provider_error_streak = 0
                self._retry_delay_s = self._retry_not_before_s = 0.
            self._latest = replace(self._latest, provider_error_streak=self._provider_error_streak,
                retry_backoff_s=self._retry_delay_s, retry_not_before_monotonic_s=self._retry_not_before_s)
        if self._latest is None:
            return None
        return self.advisor.revalidate(self._latest, latest, plan)

    def close(self):
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)

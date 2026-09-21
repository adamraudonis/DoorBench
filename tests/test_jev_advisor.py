"""Contract and failure tests. All provider responses here are synthetic."""
from dataclasses import replace
import json
from threading import Event
import urllib.error

import pytest

from doorbench.dexterous.jev_advisor import (
    ACTIONS, MODEL, AsyncJevAdvisor, AstraPlan, JevAdvisor, Telemetry,
    build_request, reference_action,
)


def fixture():
    sample = Telemetry("test", 1, .1, 10., "synthetic", 1., .12, 0., .2, True)
    plan = AstraPlan("astra-test-v1", "align", "Maintain fingertip contact")
    return sample, plan


def test_old_plan_defaults_to_required_model_and_new_policy_is_explicit():
    _, plan = fixture()
    assert plan.progress_policy == "require_jev"
    assert replace(plan, progress_policy="local_guard_with_jev_advice").progress_policy == "local_guard_with_jev_advice"


@pytest.mark.parametrize("policy", ["", "continue", "local_guard", None, True, 1, [], {}])
def test_invalid_plan_progress_policy_is_rejected(policy):
    _, plan = fixture()
    with pytest.raises(ValueError, match="progress policy"):
        replace(plan, progress_policy=policy)


def response(action="increase_angle", contact=.99, confidence=.99):
    probabilities = {name: (1. if name == action else 0.) for name in ACTIONS}
    return {"model": MODEL, "answers": {
        "contact": {"type": "noul", "noul": contact},
        "angle_action": {"type": "choice", "choice": action,
                         "confidence": confidence, "probabilities": probabilities}},
        "usage": {"input_tokens": 250}}


def test_real_api_shape_and_no_pixels_or_audit_labels_in_request():
    sample, plan = fixture()
    payload = build_request(sample, plan)
    assert set(payload) == {"state", "model", "questions"}
    assert payload["model"] == MODEL
    assert payload["state"]["observation"]["alignment"] == "increase_needed"
    assert "normal_load_N" not in json.dumps(payload)
    assert "capture_monotonic_s" not in json.dumps(payload)
    assert "label" not in payload["state"]


@pytest.mark.parametrize("changes,expected", [
    ({}, "increase_angle"), ({"angle_error_rad": -.12}, "decrease_angle"),
    ({"angle_error_rad": .02}, "hold"), ({"normal_load_N": 0.}, "reacquire_contact"),
    ({"normal_load_N": .15}, "increase_angle"), ({"normal_load_N": 15.}, "hold"),
    ({"slip_speed_mps": .03}, "reacquire_contact"), ({"balance_stable": False}, "hold"),
    ({"joint_margin_rad": .001}, "hold"), ({"normal_load_N": None}, "request_astra"),
])
def test_local_arithmetic_boundaries(changes, expected):
    sample, plan = fixture()
    assert reference_action(replace(sample, **changes), plan) == expected


def test_only_bounded_signed_angle_advice_can_be_accepted():
    sample, plan = fixture()
    result = JevAdvisor(lambda _: response(), clock=lambda: 10.1).evaluate(sample, plan)
    assert result.accepted and result.angle_step_rad == .01
    assert result.input_tokens == 250
    sample = replace(sample, angle_error_rad=-.12)
    result = JevAdvisor(lambda _: response("decrease_angle"), clock=lambda: 10.1).evaluate(sample, plan)
    assert result.accepted and result.angle_step_rad == -.01


@pytest.mark.parametrize("changes,reason", [
    ({"sensor_valid": False}, "missing_or_invalid_evidence"),
    ({"normal_load_N": None}, "missing_or_invalid_evidence"),
    ({"capture_monotonic_s": 8.}, "stale_or_future_telemetry"),
    ({"capture_monotonic_s": 11.}, "stale_or_future_telemetry"),
])
def test_invalid_evidence_abstains_without_sending_a_request(changes, reason):
    sample, plan = fixture()
    def forbidden(_):
        pytest.fail("Invalid evidence must not trigger an API request")
    result = JevAdvisor(forbidden, clock=lambda: 10.1).evaluate(replace(sample, **changes), plan)
    assert not result.accepted and result.angle_step_rad == 0. and result.reason == reason


def test_response_expiry_is_independent_of_simulation_clock():
    sample, plan = fixture()
    clock = iter([10.1, 11.])
    result = JevAdvisor(lambda _: response(), clock=lambda: next(clock)).evaluate(sample, plan)
    assert not result.accepted and result.reason == "response_expired"
    assert result.latency_ms == pytest.approx(900.)
    assert result.model == MODEL


@pytest.mark.parametrize("bad", [
    {}, {"model": "other"}, {"noul": float("nan")}, {"choice": "invented_joint_command"},
    {"confidence": 2.}, {"probabilities": {"increase_angle": 1.}},
    {"probabilities": {name: .3 for name in ACTIONS}},
])
def test_malformed_outputs_never_authorize_advice(bad):
    sample, plan = fixture()
    body = response()
    if not bad:
        body = {}
    elif "model" in bad:
        body.update(bad)
    elif "noul" in bad:
        body["answers"]["contact"].update(bad)
    else:
        body["answers"]["angle_action"].update(bad)
    result = JevAdvisor(lambda _: body, clock=lambda: 10.1).evaluate(sample, plan)
    assert not result.accepted and result.reason == "invalid_provider_response"


def test_low_confidence_and_low_contact_probability_abstain():
    sample, plan = fixture()
    for body, reason in [(response(confidence=.5), "low_confidence"),
                         (response(contact=.5), "contact_probability_too_low")]:
        result = JevAdvisor(lambda _: body, clock=lambda: 10.1).evaluate(sample, plan)
        assert not result.accepted and result.reason == reason


def test_overload_guard_overrules_confident_wrong_model():
    sample, plan = fixture()
    result = JevAdvisor(lambda _: response(), clock=lambda: 10.1).evaluate(replace(sample, normal_load_N=20.), plan)
    assert result.proposed_action == "increase_angle"
    assert not result.accepted and result.reason == "measurement_guard_rejected"


def test_provider_errors_do_not_expose_response_or_key_in_receipt():
    sample, plan = fixture()
    def broken(_):
        raise urllib.error.HTTPError("https://secret.invalid/key", 429, "secret key", {}, None)
    result = JevAdvisor(broken, clock=lambda: 10.1).evaluate(sample, plan)
    assert result.reason == "provider_http_429" and "secret" not in json.dumps(result.to_dict())


def test_delivered_advice_rejected_after_contact_or_plan_or_episode_changes():
    sample, plan = fixture()
    advisor = JevAdvisor(lambda _: response(), clock=lambda: 10.1)
    advice = advisor.evaluate(sample, plan)
    for latest, current_plan in [
        (replace(sample, normal_load_N=0.), plan),
        (replace(sample, episode_id="reset"), plan),
        (sample, replace(plan, plan_id="new-plan")),
        (replace(sample, sensor_valid=False), plan),
    ]:
        current = advisor.revalidate(advice, latest, current_plan)
        assert not current.accepted and current.angle_step_rad == 0.


def test_revalidation_reduces_step_if_target_is_closer():
    sample, plan = fixture()
    plan = replace(plan, angle_tolerance_rad=.001)
    advisor = JevAdvisor(lambda _: response(), clock=lambda: 10.1)
    advice = advisor.evaluate(sample, plan)
    current = advisor.revalidate(advice, replace(sample, angle_error_rad=.004), plan)
    assert current.accepted and current.angle_step_rad == .004


def test_contact_only_can_judge_real_load_but_never_authorize_angle_control():
    sample, plan = fixture()
    sample = replace(sample, slip_speed_mps=None, balance_stable=None, joint_margin_rad=None,
                     contact_scope="privileged_digit_handle_contact")
    def transport(payload):
        assert set(payload["questions"]) == {"contact"}
        assert "Privileged" in payload["state"]["scope"]
        return {"model": MODEL, "answers": {"contact": {"type": "noul", "noul": .99}}}
    result = JevAdvisor(transport, contact_only=True, clock=lambda: 10.1).evaluate(sample, plan)
    assert result.contact_probability == .99
    assert not result.accepted and result.angle_step_rad == 0.
    assert result.reason == "contact_judgment_only_no_control"


def test_async_worker_never_blocks_poll_and_consumes_advice_once():
    sample, plan = fixture()
    entered, release = Event(), Event()
    def delayed(_):
        entered.set()
        assert release.wait(2.)
        return response()
    async_advisor = AsyncJevAdvisor(JevAdvisor(delayed, clock=lambda: 10.1))
    try:
        assert async_advisor.submit(sample, plan)
        assert entered.wait(1.)
        assert async_advisor.poll(sample, plan) is None
        assert not async_advisor.submit(sample, plan)
        release.set()
        # Wait in the test only; the controller uses nonblocking poll.
        async_advisor._future.result(timeout=2.)
        assert async_advisor.poll(sample, plan).accepted
        assert async_advisor.poll(sample, plan) is None
    finally:
        release.set()
        async_advisor.close()
    assert not async_advisor.submit(sample, plan)

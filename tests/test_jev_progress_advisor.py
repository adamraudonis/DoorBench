"""Fake transport tests for the bounded progression permission contract."""
from dataclasses import replace
from threading import Event

from doorbench.dexterous.jev_advisor import AstraPlan, MODEL
from doorbench.dexterous.jev_progress_advisor import (
    AsyncJevProgressAdvisor, JevProgressAdvisor, ProgressSnapshot, admissible_actions,
)


def fixture():
    plan = AstraPlan("astra-press-v1", "lever_operation", "Smoothly press the acquired lever", contact_threshold_N=.2)
    sample = ProgressSnapshot("test", 1, 12., 10., "lever_operation", (.5, .5, .5, .5, .8), True, True, True)
    return sample, plan


def transport(action):
    def call(request):
        keys = request["questions"]["progress"]["criteria"]
        return {"model": MODEL, "answers": {"progress": {"type": "choice", "choice": action,
            "confidence": .99, "probabilities": {key: float(key == action) for key in keys}}}}
    return call


def test_jev_pause_is_honored_even_when_comparator_prefers_continuation():
    sample, plan = fixture()
    advice = JevProgressAdvisor(transport("pause_press"), clock=lambda: 10.1).evaluate(sample, plan)
    assert advice.accepted and not advice.advance
    assert advice.comparator_action == "continue_press" and advice.action == "pause_press"


def test_allowed_continuation_has_a_short_permission_lease():
    sample, plan = fixture()
    now = [10.1]
    advisor = JevProgressAdvisor(transport("continue_press"), clock=lambda: now[0])
    advice = advisor.evaluate(sample, plan)
    assert advice.advance
    now[0] = 10.46
    assert not advisor.revalidate(advice, replace(sample, capture_monotonic_s=10.46), plan).advance


def test_local_guard_cannot_be_overridden_by_confident_model():
    sample, plan = fixture()
    sample = replace(sample, local_continue_allowed=False)
    assert "continue_press" not in admissible_actions(sample, plan)
    advice = JevProgressAdvisor(transport("continue_press"), clock=lambda: 10.1).evaluate(sample, plan)
    assert not advice.advance and advice.reason == "invalid_provider_response"


def test_actual_contact_loss_or_phase_change_revokes_existing_permission():
    sample, plan = fixture()
    advisor = JevProgressAdvisor(transport("continue_press"), clock=lambda: 10.1)
    advice = advisor.evaluate(sample, plan)
    for latest in (replace(sample, normal_loads_N=(0., .5, .5, .5, .8)),
                   replace(sample, balance_ready=None), replace(sample, phase="partial_opening")):
        assert not advisor.revalidate(advice, latest, plan).advance


def test_fast_simulation_cannot_apply_wallclock_fresh_but_old_state_advice():
    sample, plan = fixture()
    advisor = JevProgressAdvisor(transport("continue_press"), clock=lambda: 10.1)
    advice = advisor.evaluate(sample, plan)
    result = advisor.revalidate(advice, replace(sample, simulation_time_s=12.3, sample_id=151), plan)
    assert not result.advance and result.reason == "simulation_observation_expired"


def test_unknown_contact_does_not_trigger_request_or_authorize_motion():
    sample, plan = fixture()
    def forbidden(_):
        raise AssertionError("Must not send missing contact")
    result = JevProgressAdvisor(forbidden, clock=lambda: 10.1).evaluate(replace(sample, normal_loads_N=(None, .5, .5, .5, .8)), plan)
    assert result.reason == "missing_or_invalid_contact" and not result.advance


def test_missing_balance_and_overload_only_allow_pause_or_astra():
    sample, plan = fixture()
    for latest in (replace(sample, balance_ready=None), replace(sample, normal_loads_N=(20., .5, .5, .5, .8))):
        assert admissible_actions(latest, plan) == ("pause_press", "request_astra")
    assert "continue_press" not in admissible_actions(sample, replace(plan, phase="acquire_and_hold"))


def test_async_poll_does_not_wait_for_model_and_revokes_reused_advice():
    sample, plan = fixture()
    release = Event()
    def delayed(request):
        assert release.wait(2.)
        return transport("continue_press")(request)
    async_advisor = AsyncJevProgressAdvisor(JevProgressAdvisor(delayed, clock=lambda: 10.1))
    try:
        assert async_advisor.submit(sample, plan)
        assert async_advisor.poll(sample, plan) is None
        release.set()
        async_advisor._future.result(timeout=2.)
        assert async_advisor.poll(sample, plan).advance
        assert not async_advisor.poll(replace(sample, grip_stable=False), plan).advance
    finally:
        release.set()
        async_advisor.close()

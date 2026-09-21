"""Fake transport tests for the bounded progression permission contract."""
from dataclasses import replace
from threading import Event
import urllib.error

import pytest

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


def test_provider_error_survives_later_wall_expiry_without_exposing_diagnostics():
    sample, plan = fixture()
    now = [10.1]
    def unavailable(_):
        raise urllib.error.HTTPError('https://example.invalid', 503, 'sensitive detail', {}, None)
    advisor = JevProgressAdvisor(unavailable, clock=lambda: now[0])
    result = advisor.evaluate(sample, plan)
    assert result.reason == result.evaluation_reason == 'provider_http_503'
    assert result.permission_rejection_reason is None and not result.advance
    now[0] = 11.
    expired = advisor.revalidate(result, replace(sample, capture_monotonic_s=11.), plan)
    assert expired.reason == expired.permission_rejection_reason == 'advice_or_latest_telemetry_expired'
    assert expired.evaluation_reason == 'provider_http_503' and not expired.advance
    assert result.reason == 'provider_http_503'  # Frozen original is unchanged.
    assert 'sensitive detail' not in str(expired.to_dict())


@pytest.mark.parametrize('change,reason', [
    ({'simulation_time_s':12.3}, 'simulation_observation_expired'),
    ({'phase':'partial_opening'}, 'episode_plan_or_phase_changed'),
    ({'grip_stable':False}, 'local_guard_removed_progress_permission'),
])
def test_model_evaluation_and_later_permission_rejection_are_both_recorded(change, reason):
    sample, plan = fixture()
    advisor = JevProgressAdvisor(transport('continue_press'), clock=lambda:10.1)
    original = advisor.evaluate(sample, plan)
    result = advisor.revalidate(original, replace(sample, **change), plan)
    assert result.evaluation_reason == 'accepted_progress_advice'
    assert result.reason == result.permission_rejection_reason == reason
    assert not result.advance and result.model == MODEL and result.proposed_action == 'continue_press'


def test_provider_backoff_is_bounded_nonblocking_and_resets_only_after_a_model_reply():
    sample, plan = fixture()
    now = [10.]
    healthy = [False]
    calls = []
    def provider(request):
        calls.append(now[0])
        if not healthy[0]:raise urllib.error.HTTPError('https://example.invalid', 503, '', {}, None)
        return transport('continue_press')(request)
    worker = AsyncJevProgressAdvisor(JevProgressAdvisor(provider, clock=lambda:now[0]),
        error_backoff_initial_s=.25, error_backoff_maximum_s=1.)
    try:
        for index, delay in enumerate((.25,.5,1.,1.)):
            current = replace(sample, sample_id=index+1, capture_monotonic_s=now[0])
            assert worker.submit(current, plan)
            worker._future.result(timeout=2.)
            result = worker.poll(current, plan)
            assert not result.advance and result.evaluation_reason == 'provider_http_503'
            assert result.provider_error_streak == index+1 and result.retry_backoff_s == delay
            deadline = now[0]+delay
            assert result.retry_not_before_monotonic_s == deadline
            now[0] = deadline-.001
            # Neither waiting nor polling makes another HTTP call or advances.
            assert not worker.submit(replace(current,capture_monotonic_s=now[0]),plan)
            assert not worker.poll(replace(current,capture_monotonic_s=now[0]),plan).advance
            assert len(calls) == index+1
            now[0] = deadline
        healthy[0] = True
        current = replace(sample,sample_id=5,capture_monotonic_s=now[0])
        assert worker.submit(current,plan)
        worker._future.result(timeout=2.)
        result = worker.poll(current,plan)
        assert result.advance and result.provider_error_streak == 0 and result.retry_backoff_s == 0
        # Success did not expand the original lease or current-contact gates.
        assert result.permission_expires_monotonic_s == now[0]+.35
        assert not worker.poll(replace(current,grip_stable=False),plan).advance
        now[0] += .351
        expired = worker.poll(replace(current,capture_monotonic_s=now[0]),plan)
        assert not expired.advance and expired.permission_rejection_reason == 'progress_permission_expired'
    finally:worker.close()


def test_error_backoff_uses_immutable_evaluation_when_receipt_is_already_expired():
    sample,plan=fixture();now=[10.]
    def late_error(_):
        now[0]=11.
        raise TimeoutError('private transport detail')
    worker=AsyncJevProgressAdvisor(JevProgressAdvisor(late_error,clock=lambda:now[0]))
    try:
        assert worker.submit(sample,plan)
        worker._future.result(timeout=2.)
        result=worker.poll(replace(sample,capture_monotonic_s=11.),plan)
        assert result.evaluation_reason=='provider_error_TimeoutError'
        assert result.permission_rejection_reason=='advice_or_latest_telemetry_expired'
        assert result.retry_not_before_monotonic_s==11.25
        assert not result.advance and not worker.submit(replace(sample,capture_monotonic_s=11.),plan)
    finally:worker.close()


def test_new_provider_error_still_revokes_a_previous_unexpired_continue_reply():
    sample,plan=fixture();now=[10.1];calls=[]
    def provider(request):
        calls.append(True)
        if len(calls)>1:raise urllib.error.HTTPError('https://example.invalid',529,'',{},None)
        return transport('continue_press')(request)
    worker=AsyncJevProgressAdvisor(JevProgressAdvisor(provider,clock=lambda:now[0]))
    try:
        assert worker.submit(sample,plan)
        worker._future.result(timeout=2.)
        good=worker.poll(sample,plan)
        assert good.advance
        now[0]=10.2
        latest=replace(sample,sample_id=2,capture_monotonic_s=now[0])
        assert now[0]<good.permission_expires_monotonic_s
        assert worker.submit(latest,plan)
        worker._future.result(timeout=2.)
        error=worker.poll(latest,plan)
        assert not error.advance and error.evaluation_reason=='provider_http_529'
        assert error.retry_backoff_s==.25 and not worker.submit(latest,plan)
    finally:worker.close()


@pytest.mark.parametrize('initial,maximum',[(0.,2.),(.01,2.),(.5,.25),(.25,11.),(float('nan'),2.)])
def test_invalid_retry_bounds_rejected(initial,maximum):
    with pytest.raises(ValueError):
        AsyncJevProgressAdvisor(JevProgressAdvisor(transport('continue_press')),
            error_backoff_initial_s=initial,error_backoff_maximum_s=maximum)

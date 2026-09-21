from dataclasses import replace

import pytest

from doorbench.dexterous.jev_advisor import AstraPlan, MODEL
from doorbench.dexterous.jev_progress_advisor import ProgressAdvice, ProgressSnapshot
from doorbench.dexterous.jev_progress_advisor import ProgressArbitrator


def fixture():
    plan = AstraPlan("plan-1", "lever_operation", "Bounded press", contact_threshold_N=.2)
    snapshot = ProgressSnapshot("episode-1", 50, 1., 10., "lever_operation",
                                (1.,)*5, True, True, True)
    advice = ProgressAdvice("episode-1", 49, "plan-1", "lever_operation", 9.9,
        9.95, 10.2, accepted=True, action="continue_press", model=MODEL,
        confidence=.9, simulation_time_s=.998, proposed_action="continue_press",
        evaluation_reason="accepted_progress_advice")
    return snapshot, plan, advice


def test_absence_policy_is_explicit_and_default_still_requires_model():
    sample, plan, _ = fixture()
    assert not ProgressArbitrator().choose(sample, plan, None, now_s=10.).allow_progress
    result = ProgressArbitrator("local_guard_with_jev_advice").choose(sample, plan, None, now_s=10.)
    assert result.allow_progress and result.source == "local_fallback"
    assert not result.model_advice_used
    with pytest.raises(ValueError):
        ProgressArbitrator("continue_unconditionally")


@pytest.mark.parametrize("changes", [
    {"sensor_valid": False}, {"grip_stable": False}, {"balance_ready": False},
    {"normal_loads_N": (None, 1., 1., 1., 1.)},
    {"normal_loads_N": (20., 1., 1., 1., 1.)}, {"local_continue_allowed": False},
    {"phase": "partial_opening"}, {"capture_monotonic_s": 11.},
    {"capture_monotonic_s": 9.},
])
def test_live_guard_cannot_be_overridden_by_fallback_or_model(changes):
    sample, plan, advice = fixture()
    for reply in (None, advice):
        result = ProgressArbitrator("local_guard_with_jev_advice").choose(
            replace(sample, **changes), plan, reply, now_s=10.)
        assert not result.allow_progress and not result.model_advice_used


@pytest.mark.parametrize("changes", [
    {"episode_id": "other"}, {"plan_id": "other"}, {"phase": "acquisition"},
    {"sample_id": 51}, {"simulation_time_s": .5}, {"simulation_time_s": 1.1},
    {"capture_monotonic_s": 9.}, {"received_monotonic_s": 10.1},
    {"permission_expires_monotonic_s": 9.99}, {"confidence": .69},
    {"accepted": False}, {"model": "other"},
])
def test_expired_invalid_and_mismatched_advice_never_becomes_a_model_decision(changes):
    sample, plan, advice = fixture()
    result = ProgressArbitrator("local_guard_with_jev_advice").choose(
        sample, plan, replace(advice, **changes), now_s=10.)
    assert result.allow_progress and result.source == "local_fallback"
    assert not result.model_advice_used


def test_model_pause_overrides_local_fallback_only_for_its_existing_lease():
    sample, plan, advice = fixture()
    arb = ProgressArbitrator("local_guard_with_jev_advice")
    pause = replace(advice, action="pause_press", proposed_action="pause_press")
    result = arb.choose(sample, plan, pause, now_s=10.)
    assert not result.allow_progress and result.model_advice_used
    result = arb.choose(replace(sample, capture_monotonic_s=10.3), plan, pause, now_s=10.3)
    assert result.allow_progress and result.source == "local_fallback"


def test_valid_model_continue_is_attributed_to_model():
    sample, plan, advice = fixture()
    result = ProgressArbitrator("local_guard_with_jev_advice").choose(sample, plan, advice, now_s=10.)
    assert result.allow_progress and result.model_advice_used and result.source == "jev"


def test_astra_request_latches_beyond_lease_and_only_new_plan_releases_it():
    sample, plan, advice = fixture()
    arb = ProgressArbitrator("local_guard_with_jev_advice")
    request = replace(advice, accepted=False, action="pause_press", proposed_action="request_astra",
                      evaluation_reason="model_requested_astra")
    assert arb.choose(sample, plan, request, now_s=10.).astra_escalation_latched
    future = replace(sample, capture_monotonic_s=20., simulation_time_s=2., sample_id=500)
    assert arb.choose(future, plan, None, now_s=20.).astra_escalation_latched
    result = arb.choose(future, replace(plan, plan_id="plan-2"), None, now_s=20.)
    assert result.allow_progress and not result.astra_escalation_latched


def test_stale_or_low_confidence_astra_request_does_not_create_escalation():
    sample, plan, advice = fixture()
    for changes in ({"confidence": .4}, {"capture_monotonic_s": 9.}, {"episode_id": "other"}):
        request = replace(advice, accepted=False, action="pause_press", proposed_action="request_astra",
                          evaluation_reason="model_requested_astra", **changes)
        result = ProgressArbitrator("local_guard_with_jev_advice").choose(sample, plan, request, now_s=10.)
        assert result.allow_progress and not result.astra_escalation_latched


def advanced(sample,*,wall=10.1,index=51,simulation=1.002):
    return replace(sample,capture_monotonic_s=wall,sample_id=index,simulation_time_s=simulation)


@pytest.mark.parametrize('replacement',['missing','provider_error','low_confidence','wrong_episode'])
def test_live_pause_survives_missing_or_invalid_reply_without_extending_lease(replacement):
    sample,plan,advice=fixture();arb=ProgressArbitrator('local_guard_with_jev_advice')
    pause=replace(advice,action='pause_press',proposed_action='pause_press')
    arb.choose(sample,plan,pause,now_s=10.)
    reply=None
    if replacement=='provider_error':reply=replace(advice,accepted=False,action='pause_press',proposed_action=None,model=None,evaluation_reason='provider_http_529')
    if replacement=='low_confidence':reply=replace(advice,confidence=.5)
    if replacement=='wrong_episode':reply=replace(advice,episode_id='wrong')
    result=arb.choose(advanced(sample),plan,reply,now_s=10.1)
    assert not result.allow_progress and result.model_advice_used
    assert result.reason=='live_prior_model_pause'
    assert result.model_sample_id==pause.sample_id
    assert result.model_capture_monotonic_s==pause.capture_monotonic_s
    assert result.model_permission_expires_monotonic_s==10.2
    expired=arb.choose(advanced(sample,wall=10.201,index=52,simulation=1.004),plan,None,now_s=10.201)
    assert expired.allow_progress and expired.source=='local_fallback'
    assert not expired.model_advice_used and expired.model_sample_id is None


def test_newer_valid_directive_can_supersede_pause_but_older_or_same_epoch_cannot():
    sample,plan,advice=fixture();arb=ProgressArbitrator('local_guard_with_jev_advice')
    pause=replace(advice,action='pause_press',proposed_action='pause_press')
    arb.choose(sample,plan,pause,now_s=10.)
    for sample_id in (48,49):
        old_continue=replace(advice,sample_id=sample_id)
        result=arb.choose(advanced(sample),plan,old_continue,now_s=10.1)
        assert not result.allow_progress and result.model_sample_id==49
    newer=replace(advice,sample_id=50,capture_monotonic_s=10.01,received_monotonic_s=10.02,
        permission_expires_monotonic_s=10.3,simulation_time_s=1.)
    result=arb.choose(advanced(sample),plan,newer,now_s=10.1)
    assert result.allow_progress and result.model_advice_used and result.model_sample_id==50


def test_pause_observed_during_local_failure_still_applies_after_local_recovery():
    sample,plan,advice=fixture();arb=ProgressArbitrator('local_guard_with_jev_advice')
    pause=replace(advice,action='pause_press',proposed_action='pause_press')
    assert not arb.choose(replace(sample,grip_stable=False),plan,pause,now_s=10.).allow_progress
    result=arb.choose(advanced(sample),plan,None,now_s=10.1)
    assert not result.allow_progress and result.model_advice_used


@pytest.mark.parametrize('changes',[
    {'permission_expires_monotonic_s':float('inf')},
    {'permission_expires_monotonic_s':float('nan')},
    {'permission_expires_monotonic_s':10.31},
    {'permission_expires_monotonic_s':9.94},
    {'capture_monotonic_s':10.01,'received_monotonic_s':10.02},
    {'capture_monotonic_s':float('nan')},
    {'received_monotonic_s':float('inf')},
    {'simulation_time_s':float('nan')},
    {'sample_id':True}, {'sample_id':49.0},
    {'accepted':1}, {'accepted':'true'},
    {'proposed_action':'pause_press'},
    {'evaluation_reason':'provider_http_529'},
    {'permission_rejection_reason':'local_guard_removed_progress_permission'},
])
def test_malformed_receipt_cannot_be_attributed_to_model_or_grant_required_permission(changes):
    sample,plan,advice=fixture();bad=replace(advice,**changes)
    for policy in ('require_jev','local_guard_with_jev_advice'):
        result=ProgressArbitrator(policy).choose(sample,plan,bad,now_s=10.1)
        assert not result.model_advice_used
        assert result.allow_progress==(policy=='local_guard_with_jev_advice')


def test_advice_captured_after_latest_snapshot_is_not_a_fresh_observation():
    sample,plan,advice=fixture()
    bad=replace(advice,capture_monotonic_s=10.02,received_monotonic_s=10.03,permission_expires_monotonic_s=10.2)
    result=ProgressArbitrator().choose(sample,plan,bad,now_s=10.1)
    assert not result.allow_progress and not result.model_advice_used


@pytest.mark.parametrize('kind',['future','stale','phase','sample_regression','simulation_regression'])
def test_invalid_new_context_cannot_clear_astra_latch(kind):
    sample,plan,advice=fixture();arb=ProgressArbitrator('local_guard_with_jev_advice')
    request=replace(advice,accepted=False,action='pause_press',proposed_action='request_astra',evaluation_reason='model_requested_astra')
    assert arb.choose(sample,plan,request,now_s=10.).astra_escalation_latched
    bad=advanced(sample);new_plan=replace(plan,plan_id='plan-2')
    if kind=='future':bad=replace(bad,capture_monotonic_s=11.)
    if kind=='stale':bad=replace(bad,capture_monotonic_s=9.)
    if kind=='phase':bad=replace(bad,phase='acquisition')
    if kind=='sample_regression':bad=replace(bad,sample_id=49)
    if kind=='simulation_regression':bad=replace(bad,simulation_time_s=.99)
    assert arb.choose(bad,new_plan,None,now_s=10.1).astra_escalation_latched
    assert arb.choose(advanced(sample,wall=10.2),plan,None,now_s=10.2).astra_escalation_latched


def test_fresh_new_plan_clears_escalation_but_replayed_retired_plan_cannot_progress():
    sample,plan,advice=fixture();arb=ProgressArbitrator('local_guard_with_jev_advice')
    request=replace(advice,accepted=False,action='pause_press',proposed_action='request_astra',evaluation_reason='model_requested_astra')
    arb.choose(sample,plan,request,now_s=10.)
    plan2=replace(plan,plan_id='plan-2')
    assert arb.choose(advanced(sample),plan2,None,now_s=10.1).allow_progress
    replay=arb.choose(advanced(sample,wall=10.2,index=52,simulation=1.004),plan,None,now_s=10.2)
    assert not replay.allow_progress and replay.reason=='current_context_or_clock_invalid'


def test_fresh_new_episode_can_reset_its_physical_clock_and_clear_escalation():
    sample,plan,advice=fixture();arb=ProgressArbitrator('local_guard_with_jev_advice')
    request=replace(advice,accepted=False,action='pause_press',proposed_action='request_astra',evaluation_reason='model_requested_astra')
    arb.choose(sample,plan,request,now_s=10.)
    restarted=replace(sample,episode_id='episode-2',sample_id=1,simulation_time_s=.002,capture_monotonic_s=10.1)
    result=arb.choose(restarted,plan,None,now_s=10.1)
    assert result.allow_progress and not result.astra_escalation_latched


@pytest.mark.parametrize('changes,now',[
    ({'capture_monotonic_s':9.99},9.99),
    ({'sample_id':49,'capture_monotonic_s':10.1},10.1),
    ({'simulation_time_s':.99,'capture_monotonic_s':10.1},10.1),
    ({'sample_id':51,'capture_monotonic_s':10.1},10.1),
    ({'simulation_time_s':1.002,'capture_monotonic_s':10.1},10.1),
])
def test_controller_clock_reversal_or_inconsistent_epoch_never_falls_back(changes,now):
    sample,plan,_=fixture();arb=ProgressArbitrator('local_guard_with_jev_advice')
    arb.choose(sample,plan,None,now_s=10.)
    result=arb.choose(replace(sample,**changes),plan,None,now_s=now)
    assert not result.allow_progress and result.source=='local_guard'


@pytest.mark.parametrize('changes',[
    {'normal_loads_N':(0.,1.,1.,1.,1.)},
    {'normal_loads_N':(.199,1.,1.,1.,1.)},
    {'normal_loads_N':(15.,1.,1.,1.,1.)},
    {'grip_stable':None}, {'balance_ready':None},
])
def test_more_missing_or_false_local_evidence_blocks_even_valid_model_continue(changes):
    sample,plan,advice=fixture()
    for policy in ('require_jev','local_guard_with_jev_advice'):
        result=ProgressArbitrator(policy).choose(replace(sample,**changes),plan,advice,now_s=10.)
        assert not result.allow_progress and not result.model_advice_used

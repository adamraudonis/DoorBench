"""CPU-only prospective target timing; no physical support or speed claim."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous.isaac_withdrawal_support import (
    InheritedIsaacPalmSupport, ReleaseQualifiedPalmLoadProfile, PALM_LOAD_PROFILE)
from doorbench.dexterous import standing_withdrawal
from test_isaac_withdrawal_support import predecessor
from test_isaac_withdrawal_runtime import (
    runtime, admit, write,
    test_actual_constructor_uses_detached_isaac_state_and_prefix_delegates_exact_command as exercise_entry)


def test_real_inherited_feedback_keeps_history_and_exact_default_until_release():
    original,_=predecessor();selected,_=predecessor()
    baseline=InheritedIsaacPalmSupport(original,6.)
    profiled=InheritedIsaacPalmSupport(selected,6.,profile=PALM_LOAD_PROFILE,duration_s=8.)
    feedback=selected.support_feedback;surface=feedback.surface
    baseline.begin(44.);profiled.begin(44.)
    prior=feedback.previous
    assert feedback is profiled.feedback and feedback.previous is prior
    observations={}
    for step in range(4001):
        t=44.+step*.002;pose=[step*.00001,0.,0.,1.,0.,0.,0.];load=5.+.2*np.sin(step*.1)
        reference=baseline.update(t,None,{},pose,load)
        info=profiled.update(t,None,{},pose,load,release_started_s=None if t<45.5 else 45.5)
        assert selected.left.support_load_target==6. and feedback.maximum_target_N==6.
        assert feedback is selected.support_feedback and feedback.surface is surface
        assert feedback.started==32. and profiled.started==44.
        assert selected.left.filtered_palm_load==original.left.filtered_palm_load
        assert selected.left.hybrid_blend==original.left.hybrid_blend
        assert feedback.previous[0]==original.support_feedback.previous[0]==t
        for actual,expected in zip(feedback.previous[1:],original.support_feedback.previous[1:]):
            assert actual.tobytes()==expected.tobytes()
        assert selected.left.surface_velocity_world.tobytes()==original.left.surface_velocity_world.tobytes()
        assert info['active_target_N']==selected.left.hybrid_normal_target
        assert info['inherited_source_target_N']==info['inherited_maximum_target_N']==info['target_N']==6.
        assert 2.5<=info['active_target_N']<=6.
        if t<=45.5:
            assert selected.left.hybrid_normal_target==original.left.hybrid_normal_target==6.
            # The original telemetry fields are exactly unchanged before release.
            assert {k:info[k] for k in reference}==reference
        if step in (0,750,875,1000,3499,3500,3750,4000):observations[step]=copy.deepcopy(info)
    assert observations[0]['palm_load_phase']=='awaiting_qualified_release'
    assert observations[750]['active_target_N']==6.
    assert observations[875]['active_target_N']==4.25
    assert observations[1000]['active_target_N']==observations[3499]['active_target_N']==2.5
    assert observations[3500]['palm_load_phase']=='restoring_terminal_support'
    assert observations[3750]['active_target_N']==4.25
    assert observations[4000]['active_target_N']==6.
    assert observations[4000]['palm_load_phase']=='terminal_support'


def test_actual_requested_target_is_passed_to_existing_feedback_not_only_telemetry():
    transfer,_=predecessor();calls=[];original=transfer.support_feedback.update
    def observe(t,pose,load,target):
        calls.append((t,load,target));return original(t,pose,load,target)
    transfer.support_feedback.update=observe
    support=InheritedIsaacPalmSupport(transfer,6.,profile=PALM_LOAD_PROFILE,duration_s=8.)
    support.begin(44.)
    for step in range(1001):
        t=44.+step*.002
        info=support.update(t,None,{},[0,0,0,1,0,0,0],3.,release_started_s=None if t<45.5 else 45.5)
        assert calls[-1]==(t,3.,info['active_target_N'])
    assert calls[0][2]==6. and calls[-1][2]==2.5
    assert transfer.left.support_load_target==6. and transfer.support_feedback.maximum_target_N==6.


def test_continuous_endpoints_and_first_derivatives():
    p=ReleaseQualifiedPalmLoadProfile(PALM_LOAD_PROFILE,6.,8.)
    p.target(45.5,44.,45.5)
    for t,target in ((45.5,6.),(46.,2.5),(51.,2.5),(52.,6.)):
        assert p.target(t,44.,45.5)[0]==target
        eps=1e-5
        before=p.target(t-eps,44.,45.5)[0];after=p.target(t+eps,44.,45.5)[0]
        assert abs(after-before)/(2*eps)<1e-6


def test_finite_inputs_cannot_emit_overflowed_profile_epochs():
    profile=ReleaseQualifiedPalmLoadProfile(PALM_LOAD_PROFILE,6.,1e308)
    with pytest.raises(ValueError,match='Finite derived'):
        profile.target(1e308,1e308,None)


@pytest.mark.parametrize('profile',[None,False,6.,float('nan'),'', 'unknown',{},[]])
def test_explicit_runtime_profile_name_rejects_every_non_supported_value(runtime,profile):
    runtime.config['withdrawal_palm_load_profile']=profile;runtime.bind()
    with pytest.raises(ValueError,match='profile'):admit(runtime)
    assert runtime.calls==[]


@pytest.mark.parametrize('duration',[None,True,0.,1.5,-1.,float('nan'),float('inf')])
def test_profile_rejects_nonfinite_or_insufficient_duration(duration):
    with pytest.raises(ValueError):ReleaseQualifiedPalmLoadProfile(PALM_LOAD_PROFILE,6.,duration)


@pytest.mark.parametrize('target',[2.,2.5,4.,8.,6.000001,True,float('nan')])
def test_profile_keeps_exact_six_newton_source_identity(target):
    with pytest.raises(ValueError):ReleaseQualifiedPalmLoadProfile(PALM_LOAD_PROFILE,target,8.)


@pytest.mark.parametrize('release',[43.998,44.002,True,float('nan'),float('inf')])
def test_backdated_future_or_nonfinite_release_never_updates_feedback(release):
    transfer,reads=predecessor();support=InheritedIsaacPalmSupport(transfer,6.,profile=PALM_LOAD_PROFILE,duration_s=8.)
    support.begin(44.);previous=transfer.support_feedback.previous
    with pytest.raises(ValueError):support.update(44.,None,{},[0,0,0,1,0,0,0],3.,release_started_s=release)
    assert not reads and transfer.support_feedback.previous is previous
    assert transfer.left.hybrid_normal_target==6.
    with pytest.raises(ValueError):support.update(44.,None,{},[0,0,0,1,0,0,0],3.)


def test_late_release_overlap_is_rejected_before_feedback_changes():
    transfer,reads=predecessor();support=InheritedIsaacPalmSupport(transfer,6.,profile=PALM_LOAD_PROFILE,duration_s=2.)
    support.begin(44.)
    for step in range(252):
        t=44.+step*.002
        if step==251:
            old=transfer.support_feedback.previous;count=len(reads)
            with pytest.raises(ValueError,match='nonoverlapping'):
                support.update(t,None,{},[0,0,0,1,0,0,0],3.,release_started_s=t)
            assert transfer.support_feedback.previous is old and len(reads)==count
        else:support.update(t,None,{},[0,0,0,1,0,0,0],3.)
    assert transfer.left.hybrid_normal_target==6.


@pytest.mark.parametrize('changed',[None,44.002,True,float('nan')])
def test_latched_release_cannot_be_erased_or_changed(changed):
    transfer,_=predecessor();support=InheritedIsaacPalmSupport(transfer,6.,profile=PALM_LOAD_PROFILE,duration_s=8.)
    support.begin(44.);support.update(44.,None,{},[0,0,0,1,0,0,0],3.,release_started_s=44.)
    with pytest.raises(ValueError):support.update(44.002,None,{},[0,0,0,1,0,0,0],3.,release_started_s=changed)


def test_runtime_option_binds_profile_without_changing_source_target(runtime):
    runtime.config['withdrawal_palm_load_profile']=PALM_LOAD_PROFILE;runtime.bind();a=admit(runtime)
    transfer,_=predecessor();transfer.path=runtime.route;transfer.start_seconds=28.
    a.bind_predecessor(transfer)
    assert a.inherited_support.profile.name==PALM_LOAD_PROFILE and a.inherited_support.profile.duration_s==16.
    assert a.target_N==a.controller_config['left_support_target_N']==6.
    assert a.source_context.data['withdrawal_palm_load_profile']==PALM_LOAD_PROFILE
    assert a.inherited_support.started is None and transfer.support_feedback.previous[0]==43.998


def test_option_absent_keeps_original_runtime_context_and_controller_defaults(runtime):
    a=admit(runtime);transfer,_=predecessor();transfer.path=runtime.route;transfer.start_seconds=28.
    a.bind_predecessor(transfer)
    assert a.support_profile is None and a.inherited_support.profile is None
    assert 'withdrawal_palm_load_profile' not in a.source_context.data
    assert 'withdrawal_palm_load_profile' not in a.controller_config


def test_real_outer_entry_retains_source_returned_command_and_six_newton_first_update(runtime,monkeypatch):
    runtime.config['withdrawal_palm_load_profile']=PALM_LOAD_PROFILE;runtime.bind()
    # Existing actual-constructor seam exercises 250 untouched predecessor calls,
    # prefix authorization, first force/motor capture and real inherited feedback.
    exercise_entry(runtime,monkeypatch,True)


def test_existing_force_seam_passes_only_actual_release_event_for_selected_profile():
    tree=ast.parse(Path(standing_withdrawal.__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StandingWithdrawalTeacher')
    force=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='force')
    seam=next(n for n in force.body if isinstance(n,ast.If)
        and ast.unparse(n.test)=="getattr(self, 'inherited_support', None) is not None")
    compiled=compile(ast.fix_missing_locations(ast.Module(body=[seam],type_ignores=[])),'actual-support-seam','exec')
    for enabled in (False,True):
        calls=[];support=SimpleNamespace(profile=object() if enabled else None,
            update=lambda *a,**k:calls.append((a,k)))
        scope=dict(self=SimpleNamespace(inherited_support=support,release_started=45.5),
            t=45.75,root='root',joints='joints',leaf_pose='leaf',left_palm_load=3.)
        exec(compiled,scope)
        assert calls==[((45.75,'root','joints','leaf',3.),{'release_started_s':45.5} if enabled else {})]

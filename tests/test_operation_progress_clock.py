"""Causal pause/continue checks; no external model or simulator is substituted."""
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous.operation_teacher import DoorOperationTeacher


POSE = np.array([.7, .1, 1., 1., 0., 0., 0.])
GEOMETRY = dict(operator_origin=np.zeros(3), operator_axis=np.array([0., -1., 0.]),
                leaf_origin=np.zeros(3), leaf_axis=np.array([0., 0., 1.]))


class MeasuredAcquisition:
    def __init__(self):
        self.positions=np.array([[.8, .1, 1.]])
        self.rotations=np.eye(3)[None]
        self.position_integral=np.zeros(3)
        self.rotation_integral=np.zeros(3)
        self.palm=0
        self.d=SimpleNamespace(site_xpos=self.positions.copy(),site_xmat=np.eye(3).reshape(1,9))
        self.calls=[]

    def force(self,t,root,joints,velocities,handle_pose,hand_loads):
        self.calls.append((t,dict(joints),dict(hand_loads)))
        # Vary the returned feedback with real time and measured state. A pause
        # must not replace it with the previous force or a frozen timestamp.
        return np.array([t,joints.get('measured',0.)]),dict(path_fraction=1.)


def tick(wrapper,t,*,allow=True,valid=True,operator=0.,leaf=0.,latch=0.,measured=0.):
    return wrapper.force(t,None,{'measured':measured},{},POSE,POSE,
        dict(operator=operator,leaf=leaf,latch=latch),{'load':measured},
        grasp_qualified=valid,allow_progress=allow)


def prime(**kwargs):
    wrapper=DoorOperationTeacher(MeasuredAcquisition(),GEOMETRY,**kwargs)
    for t in np.arange(0.,.51,.01):tick(wrapper,float(t))
    assert wrapper.started==.5
    return wrapper


def quintic(value):
    # Independent analytic trajectory equation, not a call to the production helper.
    u=min(1.,max(0.,value))
    return u**3*(10+u*(-15+6*u))


def test_unpaused_clocks_match_historical_analytic_targets_exactly():
    wrapper=prime(press_seconds=1.,opening_seconds=3.)
    for t in (.501,.65,1.,1.49,1.5):
        _,info=tick(wrapper,t)
        assert info['goal_handle_rad']==.87*quintic((t-.5)/1.)
        assert info['press_progress_s']==t-.5
        assert info['progress_gate_active'] is False
    tick(wrapper,1.51,operator=.85,latch=.012)
    for t in (1.511,1.6,2.,3.,4.51):
        _,info=tick(wrapper,t,operator=.85,latch=.012)
        assert info['goal_leaf_rad']==.08*quintic((t-1.51)/3.)
        assert info['opening_progress_s']==t-1.51


def test_pause_holds_reference_but_real_time_and_measured_feedback_continue():
    wrapper=prime(operator_compliance_gain=.2)
    _,before=tick(wrapper,1.5)
    old_compliance=wrapper.operator_compliance
    for t in (1.6,1.7,1.8):
        force,info=tick(wrapper,t,allow=False,measured=3*t)
        assert info['goal_handle_rad']==before['goal_handle_rad']
        assert info['press_progress_s']==1.
        assert info['progress_rate']==0.
        assert info['progress_gate_active'] and not info['progress_allowed']
        np.testing.assert_array_equal(force,[t,3*t])
        assert wrapper.acquisition.calls[-1]==(t,{'measured':3*t},{'load':3*t})
        assert wrapper.last_time==t
    assert wrapper.operator_compliance>old_compliance


def test_resume_integrates_smooth_rate_without_catching_up_paused_time():
    wrapper=prime()
    tick(wrapper,1.5)
    _,paused=tick(wrapper,1.6,allow=False)
    _,same=tick(wrapper,1.6,allow=True)
    assert same['goal_handle_rad']==paused['goal_handle_rad']
    assert same['progress_rate']==0.
    _,middle=tick(wrapper,1.7)
    assert middle['press_progress_s']==pytest.approx(1.+.015625)
    assert middle['progress_rate']==pytest.approx(.5)
    _,end=tick(wrapper,1.8)
    assert end['press_progress_s']==pytest.approx(1.1)
    assert end['progress_rate']==pytest.approx(1.)
    _,later=tick(wrapper,1.9)
    assert later['press_progress_s']==pytest.approx(1.2)
    assert later['goal_handle_rad']==pytest.approx(.87*quintic(1.2/5.))
    assert later['press_progress_s']<1.9-wrapper.started


def test_resume_progress_is_independent_of_tick_partition():
    coarse,fine=prime(),prime()
    for wrapper in (coarse,fine):
        tick(wrapper,1.5)
        tick(wrapper,1.6,allow=False)
    tick(coarse,1.8)
    for t in np.linspace(1.602,1.8,100):tick(fine,float(t))
    assert coarse.press_progress_s==pytest.approx(fine.press_progress_s,abs=2e-15)
    assert coarse.press_progress_s==pytest.approx(1.1)


def test_paused_operation_cannot_spend_wall_time_to_complete_press_or_open():
    wrapper=prime(press_seconds=1.,opening_seconds=1.)
    tick(wrapper,1.)
    tick(wrapper,5.,allow=False,operator=.85,latch=.012)
    assert wrapper.press_progress_s==.5
    assert wrapper.open_started is None
    tick(wrapper,5.2,operator=.85,latch=.012)
    assert wrapper.open_started is None
    tick(wrapper,5.62,operator=.85,latch=.012)
    assert wrapper.open_started==5.62
    assert wrapper.open_progress_s==0.
    tick(wrapper,5.82,operator=.85,latch=.012)
    held=wrapper.info['goal_leaf_rad']
    tick(wrapper,7.,allow=False,operator=.85,latch=.012)
    assert wrapper.info['goal_leaf_rad']==held
    assert wrapper.open_progress_s==pytest.approx(.2)
    tick(wrapper,7.2,operator=.85,latch=.012)
    assert wrapper.open_progress_s==pytest.approx(.3)
    assert wrapper.info['goal_leaf_rad']<wrapper.leaf_target


def test_permission_never_substitutes_for_original_contact_or_latch_gates():
    wrapper=DoorOperationTeacher(MeasuredAcquisition(),GEOMETRY,press_seconds=.1)
    for t in np.arange(0.,.8,.01):tick(wrapper,float(t),allow=False)
    assert wrapper.started is None
    tick(wrapper,.8,valid=False)
    assert wrapper.started is None
    for t in np.arange(.81,1.32,.01):tick(wrapper,float(t))
    assert wrapper.started is not None
    tick(wrapper,2.,operator=.85,latch=.005)
    assert wrapper.open_started is None
    tick(wrapper,2.01,allow=False,operator=.85,latch=.012)
    assert wrapper.open_started is None
    tick(wrapper,2.012,operator=.85,latch=.012)
    assert wrapper.open_started==2.012


def test_invalid_progress_permission_and_backward_clock_are_rejected():
    wrapper=prime()
    tick(wrapper,1.,allow=False)
    for invalid in (None,0,1,'continue'):
        with pytest.raises(ValueError,match='boolean progress'):
            tick(wrapper,1.1,allow=invalid)
    with pytest.raises(ValueError,match='backwards'):
        tick(wrapper,.9)
    assert wrapper.last_time==1.


@pytest.mark.parametrize('value',[0.,.019,1.001,float('nan'),float('inf')])
def test_invalid_resume_duration_rejected(value):
    with pytest.raises(ValueError,match='Progress resumption'):
        prime(progress_resume_seconds=value)

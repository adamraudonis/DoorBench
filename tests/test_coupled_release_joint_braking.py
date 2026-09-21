"""Projection must remain stoppable inside the original audited joint set."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from doorbench.dexterous.coupled_release_motion import CoupledReferenceMotion
from doorbench.dexterous.coupled_release_reference import (
    CoupledReleaseReference, _joint_projection_velocity_interval,
)
import doorbench.dexterous.coupled_release_reference as reference_module


def interval(q,v,dt=.002,limits=(-.5,.5)):
    return _joint_projection_velocity_interval(q,v,dt,limits,name='robot/test_wrist')


@pytest.mark.parametrize('direction',[-1.,1.])
def test_analytic_discrete_braking_and_one_step_position_bound(direction):
    distance=.01;q=direction*(.52-distance);v=direction*.24
    lower,upper=interval(q,v)
    next_speed=upper if direction>0 else -lower
    expected=np.sqrt(2.*3.*distance)-3.*.002/2.
    assert next_speed==pytest.approx(expected,abs=1e-14)
    # Independent sum of this step plus maximal permitted future braking.
    speeds=np.maximum(0.,next_speed-3.*.002*np.arange(1000))
    assert .002*np.sum(speeds)<=distance+1e-14
    assert .002*next_speed<=distance
    assert abs(direction*next_speed-v)<=3.*.002+1e-12


@pytest.mark.parametrize('direction',[-1.,1.])
def test_unstoppable_outward_state_rejects_with_joint_diagnostics(direction):
    with pytest.raises(ValueError,match='test_wrist.*velocity_intersection=.*outward_caps='):
        interval(direction*.519999,direction*.1)


@pytest.mark.parametrize('q',[-.520001,.520001])
def test_prior_position_genuinely_outside_original_allowance_rejects(q):
    with pytest.raises(ValueError,match='prior position outside original audit interval.*test_wrist'):
        interval(q,0.)


def test_existing_allowance_and_motion_limits_are_unchanged():
    # Authored .5 is deliberately distinct from the pre-existing .52 audit set.
    assert interval(.51,0.)==(-.006,.006)
    assert interval(0.,1.199)==pytest.approx((1.193,1.2))
    assert interval(0.,-1.199)==pytest.approx((-1.2,-1.193))
    assert interval(.52,0.)==(-.006,0.)
    assert interval(-.52,0.)==(0.,.006)


@pytest.mark.parametrize('q,v,dt,limits',[
    (np.nan,0.,.002,(-.5,.5)),(0.,np.inf,.002,(-.5,.5)),
    (0.,0.,0.,(-.5,.5)),(0.,0.,.003,(-.5,.5)),
    (0.,0.,.002,(.5,-.5)),(0.,0.,.002,(-.5,np.inf)),
])
def test_nonfinite_or_invalid_interval_rejects(q,v,dt,limits):
    with pytest.raises(ValueError,match='Finite original projected joint interval'):
        interval(q,v,dt,limits)


def projector_fixture():
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body><freejoint name="robot/free_base"/>
      <geom type="sphere" size=".1"/><body>
      <joint name="robot/test_wrist" range="-.5 .5"/>
      <geom type="sphere" size=".02"/></body></body></worldbody></mujoco>''')
    reference=CoupledReleaseReference.__new__(CoupledReleaseReference)
    reference.geometry=SimpleNamespace(m=model,names=['test_wrist'])
    reference.body_columns=np.arange(7)
    reference._pose_residual_jacobian=lambda value,result,**kwargs:(np.zeros(24),np.zeros((24,7)),np.zeros(2),np.zeros((2,7)))
    return reference


@pytest.mark.parametrize('direction',[-1.,1.])
def test_fresh_projected_history_brakes_before_joint_audit_boundary(monkeypatch,direction):
    reference=projector_fixture();initial=np.zeros(7);original_ranges=reference.geometry.m.jnt_range.copy()
    # Adversarial pose solve always asks for the outward edge of its feasible
    # velocity interval, while the preferred nominal joint remains at zero.
    def outward(fun,x,*,bounds,**kwargs):
        value=np.zeros_like(x);value[-1]=bounds[-1][1 if direction>0 else 0]
        return SimpleNamespace(x=value,success=True,status=0,message='outward fixture',nit=1)
    monkeypatch.setattr(reference_module,'minimize',outward)
    motion=CoupledReferenceMotion(initial);motion.update(0.,initial)
    for index in range(1,1801):
        previous=motion.value.copy();previous_velocity=motion.velocity.copy()
        result,info=motion.update(index*.002,initial,
            project=lambda *args:reference._project_reference(*args,{}))
        assert -.52<=result[-1]<=.52
        assert abs(motion.velocity[-1])<=1.2+1e-12
        assert abs(motion.velocity[-1]-previous_velocity[-1])<=3.*(motion.time-(index-1)*.002)+1e-12
        assert info['projection']['joint_position_braking_bounds']==1
        assert direction*(result[-1]-previous[-1])>=-1e-14
    assert motion.velocity[-1]==0.
    assert 0.<.52-direction*motion.value[-1]<.002
    assert np.array_equal(reference.geometry.m.jnt_range,original_ranges)


def test_empty_intersection_rejects_before_optimizer_and_motion_commit(monkeypatch):
    reference=projector_fixture();initial=np.zeros(7);initial[-1]=.519999
    motion=CoupledReferenceMotion(initial);motion.update(0.,initial);motion.velocity[-1]=.1
    def forbidden(*args,**kwargs):raise AssertionError('optimizer must not run')
    monkeypatch.setattr(reference_module,'minimize',forbidden)
    with pytest.raises(ValueError,match='no original position/braking/motion intersection'):
        motion.update(.002,np.zeros(7),project=lambda *args:reference._project_reference(*args,{}))
    assert motion.time==0.
    assert np.array_equal(motion.value,initial)
    assert motion.velocity[-1]==.1

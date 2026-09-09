import ast
import copy
import inspect
import numpy as np
import pytest
from test_sensor_balance import authored,cold,valid
from test_sensor_digit_force_control import make as old
from test_sensor_thumb_flexion_force import make as flexion
from doorbench.dexterous.sensor_distal_touch_impedance import PROTOCOL
from doorbench.dexterous.sensor_index_touch_control import INDEX_PROTOCOL
from doorbench.dexterous.sensor_digit_force_control import FORCE_PROTOCOL
from doorbench.dexterous.sensor_hierarchical_digit_force import HIERARCHICAL_PROTOCOL
from doorbench.dexterous.sensor_thumb_flexion_force import THUMB_PROTOCOL
from doorbench.dexterous.tactile_contact_mode import THUMB_MODE_PROTOCOL
from doorbench.dexterous.thumb_normal_admittance import RobotThumbNormalAdmittance,PROFILE,NAMES,intersect_scalar_bounds,validate_profile
from doorbench.dexterous.thumb_normal_admittance import PROFILE_V2,PROFILE_V3,soft_interior_coefficient
from doorbench.dexterous.sensor_thumb_admittance import SensorThumbAdmittanceController


def setup(authored):
    original=old(authored);b=original.arm.balance
    calculator=RobotThumbNormalAdmittance(b.m,b.names,b.actions,b.matrix,PROFILE.copy(),authored[1])
    q=b.desired.copy()
    for n,value in zip(NAMES,[.91,1.18,-.154,-.668,-.18]):q[b.names.index(n)]=value
    goals=dict(zip(NAMES,q[calculator.columns]));goals['torso']=0.
    return calculator,q,goals


def test_original_effort_tangent_and_robot_local_jacobian(authored):
    c,q,g=setup(authored);B,p=c.basis(q)
    np.testing.assert_allclose(p@(c.gain[:,None]*B[3:]),0.,atol=1e-16)
    pos,R,jp,jr=c.geometry(q)
    for i in range(5):
        delta=np.zeros(69);delta[c.columns[i]]=1e-6
        plus=c.geometry(q+delta)[0];minus=c.geometry(q-delta)[0]
        np.testing.assert_allclose((plus-minus)/2e-6,jp[:,i],atol=1e-8,rtol=0)
    assert np.linalg.matrix_rank(jp@B)==3 and c.d.time==0. and c.mapper.d.time==0.


def test_actual_conflicting_soft_desires_keep_original_slew_bounds():
    column=np.array([.25959080439030596,-1.])
    lo=np.array([.006774682591264847,-.025959070536406448])
    hi=np.array([.6699066825912647,.8803404294635935])
    a,b=.0255692299974759,.0263692299974759
    with pytest.raises(ValueError):intersect_scalar_bounds(column,lo,hi)
    u=soft_interior_coefficient(column,lo,hi,a,b)
    assert a<=u<=b
    samples=np.linspace(a,b,10001)
    def cost(x):return np.sum((column*x-np.clip(column*x,lo,hi))**2)
    assert cost(u)<=min(map(cost,samples))+1e-16
    assert cost(u)>0  # Report the unavoidable soft deficit; do not call it satisfied.
    previous=np.array([.006741291269302396,-.0259692299974759])
    assert max(abs(column*u-previous))<=.0004
    assert validate_profile(PROFILE_V2)==PROFILE_V2


def test_soft_desires_never_widen_infeasible_hard_bounds():
    with pytest.raises(ValueError):soft_interior_coefficient([1,-1],[0,0],[1,1],.2,.1)


def test_resultant_profile_observes_shear_while_retaining_the_palmar_measurement(authored):
    c,q,goals=setup(authored);c.profile=validate_profile(PROFILE_V3)
    c.update(goals,q,[0,6.,3.],3.,now_s=23.)
    _,info=c.update(goals,q,[0,6.,3.],3.,now_s=23.002)
    assert info['thumb_admittance_measured_palmar_load_N']==3.
    assert info['thumb_admittance_feedback_load_N']==pytest.approx(np.sqrt(45.))
    assert np.linalg.norm(info['thumb_admittance_reference_offset_m'])>0


def test_normalized_calculator_and_original_xml_share_actual_policy_gain(authored):
    import mujoco
    c,q,g=setup(authored)
    np.testing.assert_array_equal(c.m.actuator_gainprm[:,0],np.ones(61))
    np.testing.assert_array_equal(c.gain,[24.,16.])
    model=mujoco.MjModel.from_xml_path(str(authored[0]))
    original=RobotThumbNormalAdmittance(model,c.names,c.mapper.actions,c.mapper.matrix,PROFILE.copy(),authored[1])
    np.testing.assert_array_equal(original.basis(q)[0],c.basis(q)[0])
    for calc in (c,original):calc.update(g,q,[0,0,3.],3.,now_s=23.)
    for i in range(1,51):
        a,ai=c.update(g,q,[0,0,6.],6.,now_s=23.+i*.002)
        b,bi=original.update(g,q,[0,0,6.],6.,now_s=23.+i*.002)
        assert a==b and ai==bi


def test_slew_bounds_other_joints_and_no_alias(authored):
    c,q,goals=setup(authored);original=goals.copy()
    result,_=c.update(goals,q,[0,0,3.],3.,now_s=23.)
    assert result==goals
    previous=np.array([goals[n] for n in NAMES]);velocity=np.zeros(5)
    for i in range(1,151):
        result,info=c.update(goals,q,[0,0,6.],6.,now_s=23.+i*.002)
        current=np.array([result[n] for n in NAMES]);v=(current-previous)/.002
        assert np.max(abs(v))<=.06+1e-12
        assert np.max(abs(v-velocity))/.002<=.2+1e-8
        assert result['torso']==0. and goals==original
        assert info['thumb_admittance_effort_tangent_residual']<1e-12
        assert np.linalg.norm(info['thumb_admittance_reference_offset_m'])<=.002
        previous=current;velocity=v


def test_interior_guard_and_rejection_terminal(authored):
    c,q,goals=setup(authored);c.update(goals,q,[0,0,3.],3.,now_s=23.)
    q[c.columns[3]]=c.limits[3,0]+.014
    with pytest.raises(ValueError,match='15mrad'):c.update(goals,q,[0,0,3.],3.,now_s=23.002)
    with pytest.raises(RuntimeError):c.update(goals,q,[0,0,3.],3.,now_s=23.002)
    c.reset()
    with pytest.raises(ValueError):c.update(goals,q,[0,0,3.],3.,now_s=True)
    with pytest.raises(ValueError):validate_profile(dict(PROFILE,door_pose=[0]*7))
    with pytest.raises(ValueError):intersect_scalar_bounds([1,-1],[.1,.1],[.2,.2])


def test_real_capped_force_owner_and_previous_action(authored):
    base=old(authored)
    c=SensorThumbAdmittanceController(base.arm,base._index_schedule.original,authored[2],PROTOCOL.copy(),authored[1],
        INDEX_PROTOCOL.copy(),FORCE_PROTOCOL.copy(),HIERARCHICAL_PROTOCOL.copy(),contact_mode_protocol=THUMB_MODE_PROTOCOL.copy(),
        thumb_flexion_protocol=THUMB_PROTOCOL.copy(),thumb_admittance_protocol=PROFILE.copy())
    previous=flexion(authored);b=c.arm.balance
    for t in (0.,.002):
        packet=cold(b) if t==0 else valid(b,t)
        f,info=c.force(packet,now_s=t);g,_=previous.force(copy.deepcopy(packet),now_s=t)
        np.testing.assert_array_equal(f,g);np.testing.assert_array_equal(f,b.last_force)
    bad=valid(b,.004);bad['previous_action'].fill(0)
    with pytest.raises(ValueError,match='Previous action'):c.force(bad,now_s=.004)
    with pytest.raises(RuntimeError):c.force(bad,now_s=.004)
    c.reset_episode();assert c.thumb_admittance.last_time is None
    assert c.arm.inner.owner is c
    import doorbench.dexterous.thumb_normal_admittance as source
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'mj_step','mj_forward','mj_collision','plant','environment','geom'}
                   for n in ast.walk(ast.parse(inspect.getsource(source))))

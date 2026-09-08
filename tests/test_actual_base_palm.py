import mujoco
import numpy as np
import pytest
from doorbench.dexterous.actual_base_palm import ActualBasePalmCorrection,bounded_next_velocity


def test_velocity_box_preserves_acceleration_and_joint_stop_over_many_steps():
    q=np.array([.48,-.48]);v=np.array([.1,-.1]);lo=np.array([-.5,-.5]);hi=-lo
    for _ in range(1500):
        low,high=bounded_next_velocity(q,v,lo,hi,.002,.2,.5)
        next_v=np.array([high[0],low[1]])
        assert np.max(abs(next_v-v))<=.0010000001
        q+=.001*(v+next_v);v=next_v
        assert np.all(q<=hi+1e-9) and np.all(q>=lo-1e-9)
    np.testing.assert_allclose(q,[.5,-.5],atol=1e-9)
    np.testing.assert_allclose(v,0,atol=1e-8)


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="base"><freejoint/><geom size=".03" mass="1"/><body><joint name="x" type="slide" axis="1 0 0" range="-.5 .5"/><joint name="y" type="slide" axis="0 1 0" range="-.5 .5"/><joint name="z" type="slide" axis="0 0 1" range="-.5 .5"/><geom size=".02" mass=".1"/><site name="palm"/></body></body></worldbody></mujoco>''')


def test_proprioceptive_cartesian_correction_tracks_without_base_or_object_state():
    correction=ActualBasePalmCorrection(model(),['x','y','z'],'palm',maximum_speed=.2,maximum_acceleration=.5)
    correction.begin(0.,np.zeros(3),np.zeros(3));joints=dict(x=0.,y=0.,z=0.)
    for t in np.arange(.002,3.,.002):
        target,velocity,info=correction.update(float(t),joints,[.1,-.05,.03],np.eye(3),np.zeros(3))
        joints=dict(zip(joints,target))
        assert max(abs(velocity))<=.200000001
        assert info['maximum_target_acceleration_rad_s2']<=.500000001
    np.testing.assert_allclose(target,[.1,-.05,.03],atol=1e-6)
    with pytest.raises(ValueError):correction.update(3.,dict(joints,door_angle=.3),[.1,-.05,.03],np.eye(3),target)


@pytest.mark.parametrize('fault',['nan','too_fast','no_stop'])
def test_invalid_target_state_cannot_be_hidden_by_clipping(fault):
    q=np.array([.49]);v=np.array([.1])
    if fault=='nan':q[0]=np.nan
    if fault=='too_fast':v[0]=2.
    if fault=='no_stop':v[0]=.19;q[0]=.4999
    with pytest.raises(ValueError):bounded_next_velocity(q,v,[-.5],[.5],.002,.2,.5)


def test_actual_uncontrolled_motion_is_compensated_without_changing_measured_arm_fk_claim():
    robot=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/><geom size=".03" mass="1"/><body><joint name="torso_shift" type="slide" axis="1 0 0" range="-.5 .5"/><geom size=".02" mass=".1"/><body><joint name="x" type="slide" axis="1 0 0" range="-.5 .5"/><joint name="y" type="slide" axis="0 1 0" range="-.5 .5"/><joint name="z" type="slide" axis="0 0 1" range="-.5 .5"/><geom size=".02" mass=".1"/><site name="palm"/></body></body></body></worldbody></mujoco>''')
    correction=ActualBasePalmCorrection(robot,['x','y','z'],'palm',maximum_speed=.2,maximum_acceleration=.5)
    correction.begin(0.,np.zeros(3),np.zeros(3))
    for t in np.arange(0.,3.,.002):
        shift=.01*np.sin(1.3*t)
        # Deliberately bogus controlled measurements document the scope:
        # this is target-FK correction; measured arm tracking is a separate gate.
        target,velocity,info=correction.update(float(t),dict(torso_shift=shift,x=.3,y=.2,z=.1),[0,0,0],np.eye(3),np.zeros(3))
    assert abs(target[0]+shift)<3e-6
    assert info['commanded_fk_position_residual_m']<3e-6
    assert max(abs(velocity))<=.200000001


@pytest.mark.parametrize('fault',['extra','missing','rotation','time_gap'])
def test_strict_finite_proprioception_and_clock_contract(fault):
    correction=ActualBasePalmCorrection(model(),['x','y','z'],'palm')
    correction.begin(0.,np.zeros(3),np.zeros(3));joints=dict(x=0.,y=0.,z=0.);rotation=np.eye(3);t=.002
    if fault=='extra':joints['object_id']=1.
    if fault=='missing':del joints['z']
    if fault=='rotation':rotation[0,0]=2.
    if fault=='time_gap':t=.004
    with pytest.raises(ValueError):correction.update(t,joints,[0,0,0],rotation,np.zeros(3))


def test_replay_admission_compares_to_original_plan_not_already_corrected_command():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('actual_base_screen',Path(__file__).resolve().parents[1]/'scripts/dexterous/screen_actual_base_palm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    plan={'joint_names':['torso','arm_a','arm_b']}
    row={'planned_coordinates':[0.]*6+[.1,.2,.3],'left_arm_targets':[.5,.6]}
    np.testing.assert_array_equal(module.screened_nominal_arm_targets(plan,row,['arm_b','arm_a']),[.3,.2])

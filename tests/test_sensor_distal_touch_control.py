"""Local finite touch controls goals; no oracle state or substituted motor force."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from test_sensor_balance import authored
from doorbench.dexterous.sensor_reach_balance import SensorReachBalanceController
from doorbench.dexterous.sensor_acquisition_schedule import ScriptedAcquisitionSchedule
from doorbench.dexterous.sensor_handle_operation_schedule import ScriptedHandleOperationSchedule
from doorbench.dexterous.sensor_distal_touch_control import SensorDistalTouchController
from doorbench.dexterous.sensor_contract import ActorObservationBuilder


class Arm:
    def __init__(self,actual):
        self.balance=actual.balance;self.shapes=actual.shapes;self.goal_names=actual.goal_names
        self.joint_names=actual.joint_names;self.action_names=actual.action_names;self.caps=actual.caps;self.last_force=np.zeros(61)
    def reset_episode(self):self.last_force=np.zeros(61)
    def force(self,p,*,now_s,joint_goals):
        self.last_force=np.ones(61)*.01
        return self.last_force.copy(),dict(goal_joint_names=list(joint_goals),goal_joint_position_rad=list(joint_goals.values()))


def create(authored):
    real=SensorReachBalanceController(*copy.deepcopy(authored),allow_torso_yaw=True,image_shape=(8,8,3))
    ref=json.loads((Path(__file__).resolve().parents[1]/'configs/dexterous/door55-precurl-v2/reference.json').read_text())['acquisition']
    acq=ScriptedAcquisitionSchedule(real.goal_names,ref['joint_names'],ref['path_qpos']);end=acq.goals(19.)
    route=ScriptedHandleOperationSchedule(acq,['torso'],[[end['torso']],[end['torso']+.05]])
    return SensorDistalTouchController(Arm(real),route,authored[2])


def packet(c,t,loads=None):
    p=ActorObservationBuilder(joint_count=69,action_count=61,tactile_dimension=c.shapes['tactile'][0],image_shape=(8,8,3)).observe(now_s=t,previous_action=np.zeros(61))
    p['sensor_valid'][:5]=True;p['sensor_time_s'][:5]=t
    q=c.schedule.goals(min(t,19.) if t<19 else c.virtual_s)
    for n,v in q.items():p['joint_position'][c.joint_names.index(n)]=v
    if loads is not None:
        for d,value in zip(c.digits,loads):p['tactile'][c.slices[d].start+1]=value
    return p


def advance_prefix(c):
    for i in range(9500):
        t=i*.002;f,info=c.force(packet(c,t),now_s=t)
        assert dict(zip(info['goal_joint_names'],info['goal_joint_position_rad']))==c.schedule.goals(t)
        np.testing.assert_array_equal(f,c.arm.last_force)


def test_local_palmar_bins_ignore_dorsal_bins_and_negative_projection(authored):
    c=create(authored);p=packet(c,0.,[1,2,3,4,5])
    for sl in c.slices.values():p['tactile'][sl.start]=100.;p['tactile'][sl.start+3]=100.
    np.testing.assert_array_equal(c.local_distal_loads(p),[1,2,3,4,5])
    p['tactile'].fill(-1.);np.testing.assert_array_equal(c.local_distal_loads(p),np.zeros(5))


def test_exact_prefix_then_touch_only_preload_bounded_progress_and_unload_pause(authored):
    c=create(authored);advance_prefix(c)
    for i in range(9500,10801):
        t=i*.002;c.force(packet(c,t,[2,2,2,2,3]),now_s=t)
    assert c.virtual_s>19. and c.virtual_s<20.
    for i in range(10801,10851):c.force(packet(c,i*.002,[2,0,2,2,3]),now_s=i*.002)
    before=c.virtual_s
    for i in range(10851,11101):c.force(packet(c,i*.002,[2,0,2,2,3]),now_s=i*.002)
    assert c.virtual_s==before and c.delta[1]>0
    assert np.all(c.delta<=c.maximum) and np.all(c.delta>=0.)
    assert c.info['finger_motor_closure_offsets_rad'][1]<=.06


def test_forbidden_object_identity_is_terminal(authored):
    c=create(authored);p=packet(c,0.);p['lever_angle']=np.zeros(1)
    with pytest.raises(ValueError):c.force(p,now_s=0.)
    with pytest.raises(RuntimeError):c.force(packet(c,0.),now_s=0.)


def test_frame_and_offset_corruption_rejected(authored):
    c=create(authored);layout=copy.deepcopy(authored[2])
    row=next(r for r in layout['sensors'] if r['name']=='rh_ffdistal_touch');row['quaternion_xyzw_body']=[0,0,0,1]
    with pytest.raises(ValueError):SensorDistalTouchController(c.arm,c.schedule,layout)
    layout=copy.deepcopy(authored[2]);layout['sensors'][0],layout['sensors'][1]=layout['sensors'][1],layout['sensors'][0]
    with pytest.raises(ValueError):SensorDistalTouchController(c.arm,c.schedule,layout)


def test_real_two_decision_force_ownership_and_bad_previous_command_are_terminal(authored):
    from test_sensor_balance import cold,valid
    template=create(authored)
    real=SensorReachBalanceController(*copy.deepcopy(authored),allow_torso_yaw=True,image_shape=(8,8,3))
    c=SensorDistalTouchController(real,template.schedule,authored[2])
    f,_=c.force(cold(real.balance),now_s=0.)
    np.testing.assert_array_equal(f,real.balance.last_force)
    p=valid(real.balance,.002);g,_=c.force(p,now_s=.002)
    np.testing.assert_array_equal(g,real.balance.last_force)
    p=valid(real.balance,.004);p['previous_action'].fill(0.)
    with pytest.raises(ValueError,match='Previous action'):c.force(p,now_s=.004)
    with pytest.raises(RuntimeError):c.force(p,now_s=.004)


def test_boolean_clock_is_not_a_valid_reset(authored):
    c=create(authored)
    with pytest.raises(ValueError,match='boolean'):c.force(packet(c,0.),now_s=False)

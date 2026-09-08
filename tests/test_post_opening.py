"""The post-opening geometric gate must reject anatomy and collision defects."""
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.post_opening import screen_state, smooth


def plant(obstacle=False):
    bodies=[]
    for hand_i,hand in enumerate(('lh','rh')):
        for i,digit in enumerate(('FF','MF','RF','LF')):
            bodies.append(f'<body name="robot/{hand}_{digit}" pos="{i} {hand_i} 1"><joint name="robot/{hand}_{digit}J1" axis="1 0 0" range="0 2"/><joint name="robot/{hand}_{digit}J2" axis="0 1 0" range="0 2"/><geom type="sphere" size=".05" mass="1"/></body>')
    obstacle_xml='<geom type="sphere" size=".05" pos="0 0 1"/>' if obstacle else ''
    model=mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody>'+obstacle_xml+''.join(bodies)+'</worldbody></mujoco>')
    return model,mujoco.MjData(model)


def test_screen_preserves_plant_data():
    m,planning=plant();actual=mujoco.MjData(m);before=actual.qpos.copy()
    assert screen_state(m,planning,before)['passed']
    assert np.array_equal(actual.qpos,before)
    assert actual.time==planning.time==0


def test_rejects_contact_penetration():
    m,d=plant(obstacle=True)
    result=screen_state(m,d,np.zeros(m.nq))
    assert not result['passed'] and result['collisions']


def test_rejects_loopback_excess_without_joint_stop_excess():
    m,d=plant();q=np.zeros(m.nq);q[0]=.1
    result=screen_state(m,d,q)
    assert not result['passed'] and not result['joint_violations']
    assert result['maximum_loopback_violation_rad']==pytest.approx(.1)


def test_rejects_joint_stop_excess():
    m,d=plant();q=np.zeros(m.nq);q[:2]=2.04
    result=screen_state(m,d,q)
    assert not result['passed'] and result['joint_violations']


@pytest.mark.parametrize('bad',[np.zeros(2),np.full(16,np.nan),np.full(16,np.inf)])
def test_rejects_malformed_pose(bad):
    m,d=plant()
    with pytest.raises(ValueError,match='Malformed'):screen_state(m,d,bad)


def test_reference_easing_is_bounded_and_resting_at_endpoints():
    assert smooth(-1)==0 and smooth(2)==1
    x=np.array([smooth(t) for t in np.linspace(0,1,1001)])
    assert np.all(np.diff(x)>=0) and x[1]<1e-7 and 1-x[-2]<1e-7

import mujoco
import numpy as np
import pytest
from doorbench.dexterous.destination_return_kinematics import (
    admit_destination_return_kinematics, DestinationKinematicsFailure,
)


def fixture():
    children=''.join(f'<body name="{name}" pos="{i*.1} 0 0"><geom size=".01"/>{site}</body>'
        for i,(name,site) in enumerate([
            ('robot/left_ankle_link',''),('robot/right_ankle_link',''),
            ('robot/torso_link',''),('robot/rh_palm','<site name="robot/rh_palm_touch"/>'),
            ('robot/lh_palm','<site name="robot/lh_palm_touch"/>')]))
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="root"><freejoint/>'
        '<geom size=".01"/>'+children+'</body><body name="leaf_handle"><geom size=".01"/>'
        '</body></worldbody></mujoco>')
    d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
    names=[m.body(i).name for i in range(1,m.nbody)]
    return m,dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),time_s=3.,geometry_time_s=3.,
        body_names=names,body_poses_xyz_wxyz=np.c_[d.xpos[1:],d.xquat[1:]])


def test_float32_mapping_is_explicit_unstepped_and_does_not_mutate_input(monkeypatch):
    m,state=fixture();state['qpos'][3]+=1e-7
    before=state['qpos'].copy()
    def forbidden(*args,**kwargs):raise AssertionError('No physics steps permitted')
    for name in ('mj_step','mj_step1','mj_step2','mj_forward'):monkeypatch.setattr(mujoco,name,forbidden)
    d,r=admit_destination_return_kinematics(m,**state)
    assert r['passed'] and r['simulation_steps']==0 and r['active_plant_writes']==0
    assert np.array_equal(state['qpos'],before) and d.qpos[3]==1
    assert r['maximum_root_quaternion_normalization_change']==pytest.approx(1e-7)


def test_changed_actual_body_pose_retains_failure_receipt():
    m,state=fixture();state['body_poses_xyz_wxyz'][-1,0]+=.001
    with pytest.raises(DestinationKinematicsFailure) as error:admit_destination_return_kinematics(m,**state)
    assert not error.value.receipt['passed']
    assert error.value.receipt['maximum_position_error_m']==pytest.approx(.001)


@pytest.mark.parametrize('change',['stale','missing','duplicate','unnormalized','incomplete'])
def test_invalid_or_partial_measurements_are_rejected(change):
    m,state=fixture()
    if change=='stale':state['geometry_time_s']-=.002
    if change=='missing':state['body_names']=state['body_names'][:-1];state['body_poses_xyz_wxyz']=state['body_poses_xyz_wxyz'][:-1]
    if change=='duplicate':state['body_names'][-1]=state['body_names'][0]
    if change=='unnormalized':state['qpos'][3]=.99
    if change=='incomplete':state['qvel']=state['qvel'][:-1]
    with pytest.raises(ValueError):admit_destination_return_kinematics(m,**state)

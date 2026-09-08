import copy
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.whole_body_ungrip_planner import copy_ungrip_inputs


def fixture():
    names=[f'rh_{digit}J{k}' for digit in ('FF','MF','RF','LF','TH') for k in (1,2)]
    joints=''.join(f'<body name="body{i}" pos="0 0 .1"><joint name="robot/{name}"/><geom size=".01"/></body>' for i,name in enumerate(names))
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint name="robot/free_base"/><geom size=".1"/>'+joints+'</body></worldbody></mujoco>')
    d=mujoco.MjData(m);samples=dict(finger_joint_names=names,finger_joint_delta_rad=np.zeros((3,len(names))),
        palm_position_handle=np.zeros((3,3)),time_s=[0.,4.1,4.5],palm_rotation_handle=np.tile(np.eye(3),(3,1,1)),
        source_finger_joint_positions=np.zeros((3,len(names))))
    return m,d,samples


def test_private_inputs_and_original_limits_are_not_mutated(monkeypatch):
    m,active,samples=fixture();q=active.qpos.copy();ranges=m.jnt_range.copy()
    def fail(*a,**kw):raise AssertionError('No planning physics step')
    monkeypatch.setattr(mujoco,'mj_step',fail);monkeypatch.setattr(mujoco,'mj_forward',fail)
    d,copied=copy_ungrip_inputs(m,active.qpos,active.qvel,53.,samples,withdrawal_profile='clearance-lift-v4',early_lift_m=.01)
    d.qpos[:]=9.;samples['source_finger_joint_positions'][:]=3.
    np.testing.assert_array_equal(active.qpos,q);np.testing.assert_array_equal(m.jnt_range,ranges)
    assert np.all(copied['source_finger_joint_positions']==0.)


@pytest.mark.parametrize('field', ['palm_position_handle','palm_rotation_handle','source_finger_joint_positions','finger_joint_delta_rad'])
def test_nonfinite_sample_rejected(field):
    m,d,samples=fixture();samples[field].flat[0]=np.nan
    with pytest.raises(ValueError,match='finite'):copy_ungrip_inputs(m,d.qpos,d.qvel,53.,samples)


def test_sample_time_requires_original_clearance_handoff():
    m,d,samples=fixture();samples['time_s']=[0.,4.09,4.5]
    with pytest.raises(ValueError,match='4.1s'):copy_ungrip_inputs(m,d.qpos,d.qvel,53.,samples,withdrawal_profile='clearance-lift-v4')
    samples['time_s']=[0.,4.1,4.1]
    with pytest.raises(ValueError,match='increasing'):copy_ungrip_inputs(m,d.qpos,d.qvel,53.,samples)


def test_rotation_and_bounds_cannot_be_silently_repaired():
    m,d,samples=fixture();samples['palm_rotation_handle'][1,0,0]=.99
    with pytest.raises(ValueError,match='proper'):copy_ungrip_inputs(m,d.qpos,d.qvel,53.,samples)
    m,d,samples=fixture()
    with pytest.raises(ValueError,match='bounded'):copy_ungrip_inputs(m,d.qpos,d.qvel,53.,samples,early_lift_m=.010001)
    samples['finger_joint_names'][0]='rh_WRJ1'
    with pytest.raises(ValueError,match='finger joint'):copy_ungrip_inputs(m,d.qpos,d.qvel,53.,samples)

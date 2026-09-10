import copy
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.destination_state_binding import freeze_destination_state, ROOT_CONVENTION
from doorbench.dexterous.destination_planning_coordinates import planning_coordinates


@pytest.fixture
def inputs():
    names = ['joint_' + str(i) for i in range(69)]
    bodies = ''.join(f'<body name="b{i}" pos="0 0 .01"><joint name="robot/{n}"/><geom size=".01"/>' for i, n in enumerate(names))
    xml = '<mujoco><worldbody><body><freejoint/><geom size=".01"/>' + bodies + '</body>' * 69 + '</body><body><joint name="leaf"/><geom size=".01"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    motors = dict(joint_names=names, source_xml_sha256='a'*64)
    quaternion = Rotation.from_euler('z', 90, degrees=True).as_quat()[[3,0,1,2]]
    binding = freeze_destination_state(motor_contract=motors, door_source_sha256='b'*64,
        time_s=36., measured_time_s=36., root_state_convention=ROOT_CONVENTION,
        root_state_world=[.1,.2,.9,*quaternion,.4,.5,.6,1.,0.,0.],
        joint_position={n:i*.001 for i,n in enumerate(names)},
        joint_velocity={n:i*.002 for i,n in enumerate(names)},
        door_joint_order=['leaf'], door_position={'leaf':.08}, door_velocity={'leaf':.003})
    return model, binding, dict(motor_contract=motors, door_source_sha256='b'*64)


def test_world_to_body_root_velocity_and_all_joint_values(inputs):
    m,b,kwargs=inputs; original=m.qpos0.copy()
    q,v,receipt=planning_coordinates(m,b,**kwargs)
    np.testing.assert_allclose(v[:6],[.4,.5,.6,0.,-1.,0.],atol=1e-14)
    for name,value in b['joint_position'].items():
        j=m.joint('robot/'+name)
        assert q[j.qposadr[0]]==value
        assert v[j.dofadr[0]]==b['joint_velocity'][name]
    assert q[m.joint('leaf').qposadr[0]]==.08
    assert v[m.joint('leaf').dofadr[0]]==.003
    np.testing.assert_array_equal(m.qpos0,original)
    assert receipt['active_plant_writes']==0


@pytest.mark.parametrize('fault',['checksum','door_identity','model_inventory'])
def test_rejects_unbound_or_incompatible_state(inputs,fault):
    m,b,kwargs=inputs;b=copy.deepcopy(b)
    if fault=='checksum':b['joint_velocity']['joint_0']=99.
    if fault=='door_identity':kwargs['door_source_sha256']='c'*64
    if fault=='model_inventory':m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError):planning_coordinates(m,b,**kwargs)

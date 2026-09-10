import copy
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.destination_state_binding import freeze_destination_state, ROOT_CONVENTION
from doorbench.dexterous.destination_planner_admission import admit_destination_planner
from doorbench.dexterous.standing_body_record import PLANNER_BODIES


def setup():
    names=['j'+str(i) for i in range(69)]
    joints=''.join(f'<body><joint name="robot/{n}"/><geom size=".01"/></body>' for n in names)
    bodies=''.join(f'<body name="{n}" pos="0 0 {i*.02}"><geom size=".01"/>'+
        (f'<site name="robot/{n.split("/")[-1]}_touch"/>' if n.endswith('palm') else '')+'</body>'
        for i,n in enumerate(PLANNER_BODIES[:-1]))
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/>'+joints+bodies+
        '</body><body name="leaf_handle"><joint name="leaf"/><geom size=".01"/></body></worldbody></mujoco>')
    d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
    motors=dict(joint_names=names,source_xml_sha256='a'*64)
    b=freeze_destination_state(motor_contract=motors,door_source_sha256='b'*64,
        time_s=36.,measured_time_s=36.,root_state_convention=ROOT_CONVENTION,
        root_state_world=[0,0,0,1,0,0,0,.1,0,0,0,0,0],
        joint_position=dict.fromkeys(names,0.),joint_velocity=dict.fromkeys(names,.002),
        door_joint_order=['leaf'],door_position={'leaf':0.},door_velocity={'leaf':.003})
    ids=[m.body(n).id for n in PLANNER_BODIES]
    extracted=dict(binding=b,measured_bodies=dict(geometry_time_s=36.,body_names=list(PLANNER_BODIES),
        body_poses_xyz_wxyz=np.c_[d.xpos[ids],d.xquat[ids]].tolist()))
    return m,extracted,dict(motor_contract=motors,door_source_sha256='b'*64)


def test_complete_admission_preserves_velocity_and_never_steps(monkeypatch):
    m,extracted,kw=setup();before=copy.deepcopy(extracted)
    def forbidden(*a,**k):raise AssertionError('No physical stepping')
    for n in ('mj_step','mj_forward','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,n,forbidden)
    d,r=admit_destination_planner(m,extracted,**kw)
    assert r['passed'] and d.time==36. and d.qvel[0]==.1
    assert d.qvel[m.joint('robot/j0').dofadr[0]]==.002
    assert extracted==before


@pytest.mark.parametrize('fault',['legacy','stale','body','state','identity'])
def test_no_planner_admission_when_either_evidence_layer_fails(fault):
    m,e,kw=setup()
    if fault=='legacy':del e['measured_bodies']
    if fault=='stale':e['measured_bodies']['geometry_time_s']-=.002
    if fault=='body':e['measured_bodies']['body_poses_xyz_wxyz'][0][0]+=.001
    if fault=='state':e['binding']['joint_position']['j0']=.01
    if fault=='identity':kw['door_source_sha256']='c'*64
    with pytest.raises(ValueError):admit_destination_planner(m,e,**kw)

"""Exercise the file-based admission boundary with a generated scalar-joint scene."""
import hashlib
import json
import runpy
from pathlib import Path
import sys
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.destination_state_binding import freeze_destination_state, ROOT_CONVENTION
from doorbench.dexterous.landed_left_planner import LandedLeftScene
from doorbench.dexterous.standing_body_record import PLANNER_BODIES


def test_cli_records_success_and_measured_pose_failure(tmp_path,monkeypatch):
    # Include the actual planner's named left-arm coordinates in 69 joints.
    from doorbench.dexterous.landed_left_planner import JOINT_NAMES
    names=JOINT_NAMES+['j'+str(i) for i in range(69-len(JOINT_NAMES))]
    joints=''.join(f'<body><joint name="{n}"/><geom size=".01"/></body>' for n in names)
    bodies=''.join(f'<body name="{n.removeprefix("robot/")}" pos="0 0 {i*.02}"><geom size=".01"/>'+
        (f'<site name="{n.split("/")[-1]}_touch"/>' if n.endswith('palm') else '')+'</body>'
        for i,n in enumerate(PLANNER_BODIES[:-1]))
    robot=tmp_path/'robot.xml';robot.write_text('<mujoco><worldbody><body><freejoint name="free_base"/>'
        '<geom size=".1"/>'+joints+bodies+'</body></worldbody></mujoco>')
    door=tmp_path/'door.xml';door.write_text('<mujoco><worldbody><body name="leaf"><joint name="leaf_hinge"/>'
        '<geom size=".01"/><body name="leaf_handle"><geom size=".01"/></body></body></worldbody></mujoco>')
    usd=tmp_path/'door.usda';usd.write_text('#usda 1.0\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    motors=dict(joint_names=names,source_xml_sha256=sha(robot));mf=tmp_path/'motors.json';mf.write_text(json.dumps(motors))
    scene=LandedLeftScene(robot,door);m,d=scene.m,scene.d;mujoco.mj_kinematics(m,d)
    root=d.qpos[scene.root:scene.root+7].tolist()
    binding=freeze_destination_state(motor_contract=motors,door_source_sha256=sha(usd),
        time_s=36.,measured_time_s=36.,root_state_convention=ROOT_CONVENTION,
        root_state_world=root+[.1,0,0,0,0,0],joint_position=dict.fromkeys(names,0.),
        joint_velocity=dict.fromkeys(names,0.),door_joint_order=['leaf_hinge'],
        door_position={'leaf_hinge':0.},door_velocity={'leaf_hinge':0.})
    ids=[m.body(n).id for n in PLANNER_BODIES]
    poses=np.c_[d.xpos[ids],d.xquat[ids]].tolist()
    extracted=dict(binding=binding,measured_bodies=dict(geometry_time_s=36.,body_names=list(PLANNER_BODIES),body_poses_xyz_wxyz=poses))
    state=tmp_path/'state.json';output=tmp_path/'success.json'
    argv=['audit','--robot',str(robot),'--door',str(door),'--door-usd',str(usd),'--extracted',str(state),'--motors',str(mf),'--output',str(output)]
    script=Path(__file__).resolve().parents[1]/'scripts/dexterous/audit_destination_planner.py'
    for failed in (False,True):
        if failed:poses[0][0]+=.001;argv[-1]=str(tmp_path/'failure.json')
        state.write_text(json.dumps(extracted));monkeypatch.setattr(sys,'argv',argv)
        with pytest.raises(SystemExit) as status:runpy.run_path(str(script),run_name='__main__')
        assert status.value.code==(1 if failed else 0)
        receipt=json.loads(Path(argv[-1]).read_text())
        assert receipt['passed'] is (not failed)
        assert receipt['input_sha256'][str(robot)]==sha(robot)
        assert len(receipt['source_designs'])==2

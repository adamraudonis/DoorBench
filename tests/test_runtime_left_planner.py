"""Real unstepped MuJoCo checks for runtime planning and stale-state rejection."""
import copy
import json

import mujoco
import numpy as np
import pytest

from doorbench.dexterous.landed_left_planner import LandedLeftScene,JOINT_NAMES
from doorbench.dexterous.robot_design_identity import robot_design_identity
from doorbench.dexterous.runtime_left_planner import plan_attained_left_contact,RuntimeLeftPlanFailure


@pytest.fixture
def inputs(tmp_path):
    bodies='<body name="pelvis"><freejoint name="free_base"/><geom size=".01" mass="1"/>'
    for i,name in enumerate(JOINT_NAMES):
        bodies+=f'<body name="arm_{i}" pos="0 0 .1"><joint name="{name}" range="-3 3" axis="0 1 0"/><geom size=".01" mass="1"/>'
    bodies+='<body name="lh_palm" pos="0 0 .1"><site name="lh_palm_touch"/><geom size=".01" mass="1"/></body>'+'</body>'*(len(JOINT_NAMES)+1)
    robot=tmp_path/'robot.xml';robot.write_text('<mujoco><compiler angle="radian"/><worldbody>'+bodies+'</worldbody></mujoco>')
    door=tmp_path/'door.xml';door.write_text('<mujoco><compiler angle="radian"/><worldbody><body name="leaf" pos="3 0 1"><joint name="leaf_hinge" range="-2 2"/><geom size=".1" mass="1"/><body name="leaf_handle" pos=".2 0 .1"><geom size=".01"/></body></body></worldbody></mujoco>')
    scene=LandedLeftScene(robot,door);q=scene.m.qpos0.copy();q[scene.root:scene.root+3]=[0,-.2,1]
    state=scene.state_from_qpos(q,pose_time_s=45.502);pos,normal=scene.palm_local(q)
    state['door_body_poses']={name:np.r_[scene.d.xpos[scene.m.body(name).id],scene.d.xquat[scene.m.body(name).id]].tolist() for name in ('leaf','leaf_handle')}
    row=dict(phase='left_reach',leaf_rad=0.,position=pos.tolist(),normal=normal.tolist(),nominal=q[scene.qa].tolist())
    config=dict(schema='doorbench.left-palm-targets.v1',passed=True,fixed_waist=True,joint_names=JOINT_NAMES.copy(),targets=[copy.deepcopy(row),copy.deepcopy(row)],source_design_identity=robot_design_identity(robot))
    targets=tmp_path/'targets.json';targets.write_text(json.dumps(config))
    return robot,door,targets,state


def test_public_planner_freezes_numeric_state_without_any_physics_step(inputs,monkeypatch):
    robot,door,targets,state=inputs;snapshot=copy.deepcopy(state)
    def forbidden(*args,**kwargs):raise AssertionError('No physics steps allowed')
    for name in ('mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,name,forbidden)
    config,receipt=plan_attained_left_contact(robot,door,targets,state,at_time_s=45.502,subdivisions=2)
    assert receipt['passed'] and receipt['actual_calculator_time_s']==0 and receipt['selected_profile']=='strict-v1'
    assert receipt['physical_execution_qualified'] is False and receipt['active_state_writes']==0
    assert state==snapshot and config['targets'][0]['nominal']==[state['joints'][n] for n in JOINT_NAMES]
    assert config['source']['runtime_attained_state_sha256']==receipt['attained_state_sha256']


@pytest.mark.parametrize('change',[
    lambda s:s['joints'].pop('torso'),
    lambda s:s['door_body_poses'].pop('leaf_handle'),
    lambda s:s['door_body_poses']['leaf'].__setitem__(0,3.004),
    lambda s:s['door_body_poses']['leaf_handle'].__setitem__(3,1.1),
    lambda s:s.update(pose_time_s=45.5),
    lambda s:s['door_positions'].__setitem__('leaf_hinge',.4),
    lambda s:s['root'].__setitem__(0,float('nan')),
    lambda s:s.update(active_plant='not an admitted input'),
])
def test_stale_partial_or_wrong_frame_fails_with_retained_receipt(inputs,change):
    robot,door,targets,state=inputs;change(state)
    with pytest.raises(RuntimeLeftPlanFailure) as raised:
        plan_attained_left_contact(robot,door,targets,state,at_time_s=45.502,subdivisions=2)
    assert raised.value.receipt['passed'] is False and raised.value.receipt['physics_steps']==0


def test_explicit_clearance_does_not_modify_a_passing_strict_path(inputs):
    robot,door,targets,state=inputs
    a,first=plan_attained_left_contact(robot,door,targets,state,at_time_s=45.502,subdivisions=2)
    b,second=plan_attained_left_contact(robot,door,targets,state,at_time_s=45.502,subdivisions=2,clearance_profile='intermediate-clearance-3mm-v1')
    assert a==b and len(second['attempts'])==1 and second['selected_profile']=='strict-v1'


def test_unsupported_clearance_is_never_implicitly_allowed(inputs):
    robot,door,targets,state=inputs
    with pytest.raises(RuntimeLeftPlanFailure):plan_attained_left_contact(robot,door,targets,state,at_time_s=45.502,clearance_profile='allow_collision')


def test_failed_screen_never_returns_a_usable_default_target(inputs,monkeypatch):
    import doorbench.dexterous.runtime_left_planner as module
    original=module.make_landed_left_plan
    def rejected(*args,**kwargs):
        config,report=original(*args,**kwargs)
        config['passed']=report['passed']=report['audit']['passed']=False
        return config,report
    monkeypatch.setattr(module,'make_landed_left_plan',rejected)
    robot,door,targets,state=inputs
    with pytest.raises(RuntimeLeftPlanFailure) as raised:
        plan_attained_left_contact(robot,door,targets,state,at_time_s=45.502,subdivisions=2)
    assert len(raised.value.receipt['attempts'])==1
    assert raised.value.receipt['attempts'][0]['candidate']['passed'] is False


def test_clearance_cannot_redeem_a_failed_endpoint_fit(inputs,monkeypatch):
    import doorbench.dexterous.runtime_left_planner as module
    original=module.make_landed_left_plan
    def failed_fit(*args,**kwargs):
        config,report=original(*args,**kwargs)
        config['passed']=report['passed']=report['fit']['passed']=False
        return config,report
    def no_clearance(*args,**kwargs):raise AssertionError('Failed endpoint must not try a clearance path')
    monkeypatch.setattr(module,'make_landed_left_plan',failed_fit)
    monkeypatch.setattr(module.left_approach_clearance,'add_left_approach_clearance',no_clearance)
    robot,door,targets,state=inputs
    with pytest.raises(RuntimeLeftPlanFailure) as raised:
        plan_attained_left_contact(robot,door,targets,state,at_time_s=45.502,subdivisions=2,clearance_profile='intermediate-clearance-3mm-v1')
    assert len(raised.value.receipt['attempts'])==1 and 'endpoint' in str(raised.value)

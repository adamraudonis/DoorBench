"""Small real MuJoCo scenes exercise actual-state and static planning guards."""
import copy

import mujoco
import numpy as np
import pytest

from doorbench.dexterous.landed_left_planner import LandedLeftScene, JOINT_NAMES, read_targets
from doorbench.dexterous.landed_left_audit import audit_landed_left_path, static_pose_check


@pytest.fixture
def scene(tmp_path):
    # Names mirror the planner contract; no external robot assets are required.
    bodies = '<body name="pelvis"><freejoint name="free_base"/><geom size=".01" mass="1"/>'
    for i, name in enumerate(JOINT_NAMES):
        bodies += f'<body name="arm_{i}" pos="0 0 .1"><joint name="{name}" range="-3 3" axis="0 1 0"/><geom size=".01" mass="1"/>'
    bodies += '<body name="lh_palm" pos="0 0 .1"><site name="lh_palm_touch"/><geom size=".01" mass="1"/></body>'
    bodies += '</body>'*(len(JOINT_NAMES)+1)
    robot = tmp_path/'robot.xml'
    robot.write_text('<mujoco><compiler angle="radian"/><worldbody>'+bodies+'</worldbody></mujoco>')
    door = tmp_path/'door.xml'
    door.write_text('<mujoco><compiler angle="radian"/><worldbody><body name="leaf" pos="3 0 1"><joint name="leaf_hinge" range="-2 2"/><geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>')
    s = LandedLeftScene(robot, door)
    q = s.m.qpos0.copy();q[s.root:s.root+3] = [0, 0, 1]
    state = s.state_from_qpos(q, pose_time_s=45.502)
    position, normal = s.palm_local(q)
    row = dict(phase='left_reach', leaf_rad=0., position=position.tolist(),
               normal=normal.tolist(), nominal=q[s.qa].tolist())
    config = dict(schema='doorbench.left-palm-targets.v1', passed=True, fixed_waist=True,
                  joint_names=JOINT_NAMES.copy(), targets=[copy.deepcopy(row), copy.deepcopy(row)])
    return s, state, config


def test_exact_named_state_and_no_physics_capability(scene, monkeypatch):
    s, state, config = scene
    before = s.d.qpos.copy()
    def fail(*args, **kwargs):
        raise AssertionError('Static planning must never step any plant')
    monkeypatch.setattr(mujoco, 'mj_step', fail)
    monkeypatch.setattr(mujoco, 'mj_step1', fail)
    monkeypatch.setattr(mujoco, 'mj_step2', fail)
    report = audit_landed_left_path(s, config, state, subdivisions=2)
    assert report['passed']
    assert np.array_equal(s.d.qpos, before)
    assert s.d.time == 0.
    assert not hasattr(s, 'plant')


@pytest.mark.parametrize('change', ['missing_joint', 'extra_joint', 'missing_door', 'nan', 'root_norm', 'clock'])
def test_rejects_incomplete_actual_state(scene, change):
    s, state, _ = scene
    if change == 'missing_joint': state['joints'].pop('torso')
    if change == 'extra_joint': state['joints']['unknown'] = 0.
    if change == 'missing_door': state['door_positions'].clear()
    if change == 'nan': state['joints']['torso'] = float('nan')
    if change == 'root_norm': state['root'][3] = 1.000005
    if change == 'clock': state['pose_time_s'] = -1.
    with pytest.raises(ValueError): s.freeze(state)


@pytest.mark.parametrize('change,gate', [
    ('leaf', 'actual_leaf_metadata'), ('position', 'target_fk_positions'),
    ('normal', 'target_fk_normals'), ('waist', 'fixed_actual_torso'), ('initial', 'exact_initial_arm')])
def test_independent_audit_rejects_stale_frames_or_hidden_relocation(scene, change, gate):
    s, state, config = scene
    if change == 'leaf': config['targets'][-1]['leaf_rad'] += .003
    if change == 'position': config['targets'][-1]['position'][0] += .001
    if change == 'normal': config['targets'][-1]['normal'] = [0., 1., 0.]
    if change == 'waist': config['targets'][-1]['nominal'][0] += .01
    if change == 'initial': config['targets'][0]['nominal'][1] += .01
    report = audit_landed_left_path(s, config, state, subdivisions=2)
    assert not report['passed'] and not report['checks'][gate]


def test_receiving_only_panel_collider_is_not_omitted():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="leaf"><geom name="receiving_strike" type="box" size=".1 .1 .1" contype="0" conaffinity="1"/></body>
      <body name="robot/lh_palm" pos="0 0 .199"><freejoint/><geom name="palm" type="sphere" size=".1" mass="1" contype="1" conaffinity="0"/></body>
    </worldbody></mujoco>''')
    d = mujoco.MjData(m)
    report = static_pose_check(m, d, coordinate=.5)
    assert not report['passed'] and report['left_environment_touches'] > 0
    assert any('receiving_strike' in c['geoms'] for c in report['contacts'])
    assert static_pose_check(m, d, coordinate=1.)['passed']


def test_off_panel_left_touch_never_becomes_allowed_by_progress():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="frame"><geom type="sphere" size=".1"/></body>
      <body name="robot/lh_palm" pos="0 0 .199"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>''')
    assert not static_pose_check(m, mujoco.MjData(m), coordinate=1.)['passed']


def test_passive_difference_constraint_is_audited():
    m = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
      <body name="robot/lh_palm"><joint name="J1" range="0 2"/><joint name="J2" axis="1 0 0" range="0 2"/><geom size=".1" mass="1"/></body>
    </worldbody><tendon><fixed name="robot/loopback" limited="true" range="-3 0">
      <joint joint="J1" coef="1"/><joint joint="J2" coef="-1"/>
    </fixed></tendon></mujoco>''')
    d = mujoco.MjData(m);d.qpos[:] = [.04, 0.]
    report = static_pose_check(m, d, coordinate=0.)
    assert not report['passed'] and report['maximum_joint_violation'] == 0.
    assert report['maximum_loopback_violation_rad'] == pytest.approx(.04)


def test_invalid_source_contract_is_not_promoted(scene):
    _, _, config = scene
    config['passed'] = False
    with pytest.raises(ValueError): read_targets(config)


def test_positive_gap_buffer_is_not_a_touch():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="leaf"><geom type="sphere" size=".1" margin=".02"/></body>
      <body name="robot/lh_palm" pos="0 0 .21"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>''')
    d = mujoco.MjData(m)
    report = static_pose_check(m, d, coordinate=.1)
    assert d.ncon > 0 and report['passed'] and report['left_environment_touches'] == 0

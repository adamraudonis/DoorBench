"""Fresh planning must not inherit geometry or physics claims from old inputs."""
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.dexterous.plan_local_standing_transfer import (
    can_export_route, generate_fixed_body_path, load_snapshot, resample_preferences, sha, source_qualification,
)
from doorbench.dexterous.landed_left_planner import LandedLeftScene, JOINT_NAMES


def preferences():
    rows = [dict(phase='left_reach',position=[x,-.05,1.],normal=[0.,-1.,0.],
                 nominal=[x]*8) for x in (0.,.2,.4)]
    return dict(schema='doorbench.left-palm-targets.v1',joint_names=JOINT_NAMES,
                targets=rows,passed=False,source_design_identity={'untrusted':'old'})


def test_only_numeric_preferences_consumed_and_endpoints_preserved():
    config = preferences()
    original = copy.deepcopy(config)
    value = resample_preferences(config)
    assert value['position'].shape == (101,3)
    np.testing.assert_array_equal(value['nominal'][0],config['targets'][0]['nominal'])
    np.testing.assert_array_equal(value['nominal'][-1],config['targets'][-1]['nominal'])
    np.testing.assert_allclose(np.linalg.norm(value['normal'],axis=1),1.)
    assert config == original and 'passed' not in value and 'source_design_identity' not in value
    config['passed'] = True
    other = resample_preferences(config)
    for key in value:
        np.testing.assert_array_equal(value[key],other[key])


@pytest.mark.parametrize('mutation', ['nan','normal','joint_order','phase','missing'])
def test_invalid_numeric_preferences_rejected(mutation):
    config = preferences()
    if mutation == 'nan': config['targets'][1]['nominal'][1] = float('nan')
    if mutation == 'normal': config['targets'][1]['normal'] = [0.,0.,0.]
    if mutation == 'joint_order': config['joint_names'] = list(reversed(JOINT_NAMES))
    if mutation == 'phase': config['targets'][0]['phase'] = 'right_release'
    if mutation == 'missing': config['targets'] = config['targets'][:1]
    with pytest.raises(ValueError):
        resample_preferences(config)


@pytest.mark.parametrize('source,sampled,dense,development,expected', [
    (True,True,True,False,True), (False,True,True,False,False),
    (True,False,True,False,False), (True,True,False,False,False),
    (True,True,True,True,False), (False,True,True,True,False),
    (1,True,True,False,False),
])
def test_failed_or_development_source_never_becomes_runtime_route(source,sampled,dense,development,expected):
    assert can_export_route(source_passed=source,sampled_passed=sampled,
        dense_passed=dense,development=development) is expected


def test_source_qualification_retains_missing_and_failed_reports(tmp_path):
    (tmp_path/'raw-transitions').mkdir()
    for name in ('manifest.json','raw-transitions/manifest.json','report.json'):
        (tmp_path/name).write_text(json.dumps({'passed':True,'checks':{'complete':True}}))
    inputs = {name:sha(tmp_path/name) for name in ('manifest.json','raw-transitions/manifest.json','report.json')}
    for name in ('independent-pad-audit.json','independent-whole-handle-audit.json'):
        (tmp_path/name).write_text(json.dumps({'passed':True,'checks':{'complete':True},'input_sha256':inputs}))
    assert source_qualification(tmp_path)['passed']
    (tmp_path/'report.json').write_text(json.dumps({'passed':True,'checks':{'complete':True},'changed':True}))
    assert not source_qualification(tmp_path)['passed']
    (tmp_path/'report.json').write_text(json.dumps({'passed':True,'checks':{'complete':False}}))
    result = source_qualification(tmp_path)
    assert not result['passed'] and not result['reports'][0]['passed']
    (tmp_path/'independent-pad-audit.json').unlink()
    result = source_qualification(tmp_path)
    assert not result['passed'] and not result['reports'][1]['exists']


@pytest.fixture
def recorded_scene(tmp_path):
    robot = tmp_path/'robot.xml'
    bodies = '<body name="pelvis"><freejoint name="free_base"/><geom size=".01" mass="1"/>'
    for index,name in enumerate(JOINT_NAMES):
        bodies += f'<body name="arm_{index}" pos="0 0 .1"><joint name="{name}" range="-3 3"/><geom size=".01" mass="1"/>'
    bodies += '<body name="lh_palm"><geom size=".01" mass="1"/><site name="lh_palm_touch"/></body>'
    bodies += '</body>'*(len(JOINT_NAMES)+1)
    robot.write_text('<mujoco><compiler angle="radian"/><worldbody>'+bodies+'</worldbody></mujoco>')
    door = tmp_path/'door.xml'
    door.write_text('<mujoco><worldbody><body name="leaf" pos="1 0 1"><joint name="leaf_hinge"/>'
        '<geom size=".01" mass="1"/></body></worldbody></mujoco>')
    scene = LandedLeftScene(robot,door)
    q = scene.m.qpos0.copy()
    scene.d.qpos[:] = q
    mujoco.mj_kinematics(scene.m,scene.d)
    arrays = dict(interval_start_s=np.array([.1,.102]), geometry_time_s=np.array([.1,.102]),
        qpos_before=np.array([q,q]),qvel_before=np.zeros((2,scene.m.nv)),
        body_offsets=np.array([0,2,4]),body_ids=np.tile([scene.leaf,scene.m.body('robot/lh_palm').id],2))
    ids = arrays['body_ids']
    arrays.update(body_positions_world_m=scene.d.xpos[ids].copy(),
                  body_rotations_world=scene.d.xmat[ids].reshape(-1,3,3).copy())
    trial = tmp_path/'trial'
    raw = trial/'raw-transitions'
    raw.mkdir(parents=True)
    chunk = raw/'transitions-00000.npz'
    np.savez(chunk,**arrays)
    (trial/'manifest.json').write_text(json.dumps({'inputs':{'robot':{'sha256':sha(robot)},'door':{'door.xml':sha(door)}}}))
    manifest = dict(schema='doorbench.native-transitions.v1',complete=True,
        chunks=[dict(file=chunk.name,sha256=sha(chunk),rows=2,interval_start_s=.1,interval_end_s=.104)])
    (raw/'manifest.json').write_text(json.dumps(manifest))
    return scene,trial,robot,door,arrays,manifest


def test_snapshot_uses_exact_recorded_epoch_without_physics(recorded_scene,monkeypatch):
    scene,trial,robot,door,arrays,_ = recorded_scene
    def no_step(*a,**k): raise AssertionError('Static planning cannot step physics')
    for name in ('mj_step','mj_step1','mj_step2'):
        monkeypatch.setattr(mujoco,name,no_step)
    q,receipt = load_snapshot(trial,robot,door,scene)
    np.testing.assert_array_equal(q,arrays['qpos_before'][-1])
    assert receipt['pose_time_s'] == .102 and receipt['maximum_recorded_fk_error'] == 0
    assert scene.d.time == 0
    with pytest.raises(ValueError,match='actual stored state'):
        load_snapshot(trial,robot,door,scene,at_time=.101)


def test_fixed_body_candidate_changes_only_left_arm_without_physics(recorded_scene,monkeypatch):
    scene,_,_,_,arrays,_ = recorded_scene
    frozen = arrays['qpos_before'][0]
    qa = scene.m.jnt_qposadr[[scene.m.joint('robot/'+n).id for n in JOINT_NAMES[1:]]]
    goal = frozen.copy()
    goal[qa[0]] = .2
    position,normal = scene.palm_local(goal)
    config = preferences()
    for row in config['targets']:
        row.update(position=position.tolist(),normal=normal.tolist(),nominal=goal[scene.qa].tolist())
    def no_step(*a,**k): raise AssertionError('Planner cannot integrate a plant')
    for name in ('mj_step','mj_step1','mj_step2'):
        monkeypatch.setattr(mujoco,name,no_step)
    path,rows,endpoint = generate_fixed_body_path(scene,frozen,resample_preferences(config))
    fixed = np.setdiff1d(np.arange(scene.m.nq),qa)
    assert path.shape == (101,scene.m.nq) and len(rows)==101
    np.testing.assert_array_equal(path[0],frozen)
    np.testing.assert_array_equal(path[:,fixed],np.tile(frozen[fixed],(101,1)))
    assert np.max(abs(path[-1,qa]-frozen[qa]))>.01 and scene.d.time==0


@pytest.mark.parametrize('mutation', ['robot','chunk_hash','body_pose','geometry_epoch','incomplete'])
def test_changed_or_unsynchronized_recorded_geometry_rejected(recorded_scene,mutation):
    scene,trial,robot,door,arrays,manifest = recorded_scene
    chunk = trial/'raw-transitions'/manifest['chunks'][0]['file']
    if mutation == 'robot': robot.write_text(robot.read_text()+'\n')
    if mutation == 'chunk_hash': chunk.write_bytes(chunk.read_bytes()+b'changed')
    if mutation in ('body_pose','geometry_epoch'):
        if mutation == 'body_pose': arrays['body_positions_world_m'][-1,0] += .001
        else: arrays['geometry_time_s'][-1] += .002
        np.savez(chunk,**arrays)
        manifest['chunks'][0]['sha256'] = sha(chunk)
    if mutation == 'incomplete': manifest['complete'] = False
    (trial/'raw-transitions/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_snapshot(trial,robot,door,scene)

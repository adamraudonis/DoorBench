"""Native articulation and calibration are authoritative for the vision bridge."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from doorbench.appearance.state import (capture_mujoco_state, capture_initial_state,
                                       export_state, validate_camera)
from doorbench.build import export_door
from doorbench.spec import generate_all


def _rotation(quaternion):
    from doorbench.ir import quat_to_mat
    return quat_to_mat(quaternion)


@pytest.fixture(scope='module')
def doors(tmp_path_factory):
    root = tmp_path_factory.mktemp('appearance_state')
    chosen = {'db0012_swing_single', 'db0079_sliding_single', 'db0188_cold_storage'}
    specs = [s for s in generate_all() if s['id'] in chosen]
    results = {}
    for spec in specs:
        export_door(spec, str(root / 'doors'), str(root / 'hardware'), formats=('mjcf', 'json'))
        results[spec['id']] = root / 'doors' / spec['id']
    return results


@pytest.mark.parametrize('door_id', ['db0012_swing_single', 'db0079_sliding_single', 'db0188_cold_storage'])
def test_capture_matches_native_after_steps_and_preserves_live_data(doors, door_id):
    mujoco = pytest.importorskip('mujoco')
    model = mujoco.MjModel.from_xml_path(str(doors[door_id] / 'door.xml'))
    data = mujoco.MjData(model)
    meta = json.loads((doors[door_id] / 'model.json').read_text())['meta']
    primary = model.joint(meta['primary_joint']).id
    data.qfrc_applied[model.jnt_dofadr[primary]] = 100
    for _ in range(31):
        mujoco.mj_step(model, data)
    before = {name: getattr(data, name).copy() for name in ('qpos', 'qvel', 'xpos', 'geom_xpos', 'qfrc_applied')}
    state = capture_mujoco_state(model, data, door_id=door_id)
    assert state['time_s'] == data.time
    assert state['state_kind'] == 'simulation_snapshot'
    assert not state['kinematic_inspection']
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(data, name), value)
    reference = mujoco.MjData(model)
    mujoco.mj_copyData(reference, model, data)
    mujoco.mj_forward(model, reference)
    for name, pose in state['body_world'].items():
        body = model.body(state['body_aliases'].get(name, name)).id
        np.testing.assert_allclose(pose['pos'], reference.xpos[body], atol=1e-12)
        np.testing.assert_allclose(_rotation(pose['quat_wxyz']), reference.xmat[body].reshape(3, 3), atol=1e-12)
    for name, pose in state['geom_world'].items():
        geom = model.geom(name).id
        np.testing.assert_allclose(pose['pos'], reference.geom_xpos[geom], atol=1e-12)
        np.testing.assert_allclose(_rotation(pose['quat_wxyz']), reference.geom_xmat[geom].reshape(3, 3), atol=1e-12)
    json.dumps(state, allow_nan=False)


def test_telescoping_and_reference_joint_poses_are_not_reconstructed_by_hand():
    mujoco = pytest.importorskip('mujoco')
    xml = '''<mujoco><compiler angle="radian"/><worldbody>
      <body name="rail" pos="1 2 3" quat="0.7071067811865476 0 0 0.7071067811865476">
      <joint name="outer" type="slide" axis="1 0 0" ref="0.3" range="0 1"/>
      <geom name="outer_geom" size=".1 .2 .3" type="box"/>
      <body name="extension" pos=".2 0 0"><joint name="inner" type="slide" axis="1 0 0" ref="-.2" range="-.5 .5"/>
      <geom name="inner_geom" pos=".3 0 0" type="sphere" size=".1"/></body></body>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = [.7, .1]
    data.qvel[:] = [.1, -.15]
    for _ in range(11):
        mujoco.mj_step(model, data)
    state = capture_mujoco_state(model, data, door_id='telescoping_fixture')
    reference = mujoco.MjData(model)
    mujoco.mj_copyData(reference, model, data)
    mujoco.mj_forward(model, reference)
    np.testing.assert_allclose(state['body_world']['extension']['pos'], reference.xpos[model.body('extension').id], atol=1e-12)
    assert state['qpos']['outer'] != model.qpos0[0]
    assert state['body_aliases'] == {'world_env': 'world'}


def test_initial_export_is_deterministic_and_overrides_are_explicit(doors, tmp_path):
    door = doors['db0012_swing_single']
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    state = export_state(door, first, camera='iso')
    assert capture_initial_state(door, camera='iso') == state
    export_state(door, second, camera='iso')
    assert first.read_bytes() == second.read_bytes()
    assert state['state_kind'] == 'authored_initial'
    assert state['camera']['resolution'] == [640, 480]
    inspected = export_state(door, second, qpos={'leaf_hinge': .2})
    assert inspected['kinematic_inspection']
    assert inspected['qpos']['leaf_hinge'] == .2
    for qpos in ({'does_not_exist': 0}, {'leaf_hinge': float('nan')}, {'leaf_hinge': [0.1]}, {'leaf_hinge': np.array([0.1])}, {'leaf_hinge': 100}, {'leaf_hinge': True}):
        with pytest.raises(ValueError):
            export_state(door, second, qpos=qpos)


def test_unknown_camera_and_invalid_pose_cannot_be_serialized(doors):
    mujoco = pytest.importorskip('mujoco')
    model = mujoco.MjModel.from_xml_path(str(doors['db0012_swing_single'] / 'door.xml'))
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match='Unknown MuJoCo camera'):
        capture_mujoco_state(model, data, camera='missing')
    data.qpos[0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        capture_mujoco_state(model, data)


def test_camera_intrinsics_match_native_asymmetric_frustum():
    mujoco = pytest.importorskip('mujoco')
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <camera name="calibrated" pos="1 2 3" sensorsize=".036 .024" resolution="600 400"
      focalpixel="500 510" principalpixel="12 16"/></worldbody></mujoco>''')
    state = capture_mujoco_state(model, mujoco.MjData(model), camera={'name': 'calibrated', 'resolution': [1200, 800]})
    camera = state['camera']
    np.testing.assert_allclose(camera['intrinsics'], [[1000, 0, 576], [0, 1020, 368], [0, 0, 1]], atol=1e-4)
    assert camera['pos'] == [1, 2, 3]
    assert camera['quat_wxyz'] == [1, 0, 0, 0]


def test_camera_manifest_rejects_bad_calibration_and_skew():
    valid = {'pos': [1, 2, 3], 'quat_wxyz': [1, 0, 0, 0], 'resolution': [640, 480],
             'intrinsics': [[500, 0, 320], [0, 510, 240], [0, 0, 1]]}
    assert validate_camera(valid)['resolution'] == [640, 480]
    reflected = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    variants = [dict(valid, resolution=[0, 480]), dict(valid, resolution=[640.5, 480]),
                dict(valid, quat_wxyz=[0, 0, 0, 0]), dict(valid, projection='orthographic'),
                dict(valid, intrinsics=[[500, 1, 320], [0, 510, 240], [0, 0, 1]]),
                dict(valid, intrinsics=[[500, 0, 320], [0, 510, 240], [0, 1, 1]]),
                dict(valid, pos=[0, float('inf'), 0]),
                {**{k: v for k, v in valid.items() if k != 'quat_wxyz'}, 'rotation_matrix': reflected}]
    for camera in variants:
        with pytest.raises(ValueError):
            validate_camera(camera)


def test_mesh_compiler_alignment_matches_authored_body_geometry():
    mujoco = pytest.importorskip('mujoco')
    # Deliberately offset, asymmetric mesh: compilation moves its origin and
    # principal axes, so placing the raw vertices with geom_world is incorrect.
    vertices = np.array([[2, 3, 4], [3, 3, 4], [2, 5, 4], [2, 3, 7]], dtype=float)
    vertex_text = ' '.join(str(v) for v in vertices.ravel())
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/><asset>
      <mesh name="offset" vertex="{vertex_text}" face="0 2 1 0 1 3 0 3 2 1 2 3"/>
      </asset><worldbody><body name="moving" pos="1 2 3"><joint name="hinge" axis="0 0 1" ref=".2"/>
      <geom name="mesh_geom" type="mesh" mesh="offset" pos=".4 .5 .6"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[0] = .7
    state = capture_mujoco_state(model, data)
    body = state['body_world']['moving']
    geom = state['geom_world']['mesh_geom']
    alignment = geom['mesh_alignment']
    local = vertices + [.4, .5, .6]
    authored_world = np.asarray(body['pos']) + local @ _rotation(body['quat_wxyz']).T
    compiled_local = (vertices - alignment['pos']) @ _rotation(alignment['quat_wxyz'])
    compiled_world = np.asarray(geom['pos']) + compiled_local @ _rotation(geom['quat_wxyz']).T
    np.testing.assert_allclose(compiled_world, authored_world, atol=1e-10)
    wrong_world = np.asarray(geom['pos']) + vertices @ _rotation(geom['quat_wxyz']).T
    assert np.max(np.abs(wrong_world - authored_world)) > 1.0


def test_import_does_not_require_mujoco():
    module = Path(__file__).parents[1] / 'doorbench/appearance/state.py'
    program = "import runpy,sys; sys.modules['mujoco']=None; runpy.run_path(sys.argv[1])"
    subprocess.run([sys.executable, '-c', program, str(module)], check=True)

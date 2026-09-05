"""External snapshot manifests cannot bypass the native bridge's validation."""
import copy
import json

import pytest

from doorbench.appearance.state import validate_snapshot


@pytest.fixture
def snapshot():
    identity = {'pos': [0, 0, 0], 'quat_wxyz': [1, 0, 0, 0]}
    return {'schema_version': 1, 'door_id': 'test_door', 'time_s': .2,
            'state_kind': 'simulation_snapshot', 'kinematic_inspection': False,
            'body_world': {'world': copy.deepcopy(identity), 'world_env': copy.deepcopy(identity),
                           'leaf': {'pos': [1, 2, 3], 'quat_wxyz': [1, 0, 0, 0]}},
            'body_aliases': {'world_env': 'world'}, 'geom_world': {},
            'qpos': {'leaf_hinge': .4}, 'qpos_vector': [.4],
            'camera': {'pos': [0, -2, 1], 'quat_wxyz': [1, 0, 0, 0],
                       'resolution': [640, 480], 'intrinsics': [[500, 0, 320], [0, 510, 240], [0, 0, 1]]},
            'source': {'engine': 'mujoco', 'model.json_sha256': 'A' * 64}}


def test_returns_independent_json_safe_copy_and_validates_expected_door(snapshot):
    original = copy.deepcopy(snapshot)
    result = validate_snapshot(snapshot, expected_door_id='test_door')
    assert snapshot == original
    assert result['source']['model.json_sha256'] == 'a' * 64
    result['body_world']['leaf']['pos'][0] = 100
    assert snapshot['body_world']['leaf']['pos'][0] == 1
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match='does not match expected door'):
        validate_snapshot(snapshot, expected_door_id='other_door')


@pytest.mark.parametrize('version', [None, True, '1', 2])
def test_unsupported_or_mistyped_version_rejected(snapshot, version):
    snapshot['schema_version'] = version
    with pytest.raises(ValueError, match='schema_version'):
        validate_snapshot(snapshot)


@pytest.mark.parametrize('pose', [
    {'pos': [0, 1], 'quat_wxyz': [1, 0, 0, 0]},
    {'pos': [0, float('nan'), 0], 'quat_wxyz': [1, 0, 0, 0]},
    {'pos': [0, 1, 2], 'quat_wxyz': [0, 0, 0, 0]},
    {'pos': [0, 1, 2], 'quat_wxyz': [2, 0, 0, 0]},
    {'pos': [0, 1, 2]},
])
def test_malformed_body_pose_has_actionable_body_name(snapshot, pose):
    snapshot['body_world']['leaf'] = pose
    with pytest.raises(ValueError, match='leaf'):
        validate_snapshot(snapshot)


def test_body_aliases_must_exist_and_match_pose(snapshot):
    snapshot['body_aliases']['world_env'] = 'missing'
    with pytest.raises(ValueError, match='missing body'):
        validate_snapshot(snapshot)
    snapshot['body_aliases']['world_env'] = 'leaf'
    with pytest.raises(ValueError, match='pose differs'):
        validate_snapshot(snapshot)
    snapshot['body_aliases'] = {'world_env': 'world', 'world': 'world_env'}
    with pytest.raises(ValueError, match='cycle'):
        validate_snapshot(snapshot)


def test_camera_skew_and_bad_rotation_are_rejected(snapshot):
    snapshot['camera']['intrinsics'][0][1] = 3
    with pytest.raises(ValueError, match='snapshot camera.*zero skew'):
        validate_snapshot(snapshot)
    snapshot['camera']['intrinsics'][0][1] = 0
    snapshot['camera']['quat_wxyz'] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match='snapshot camera.*unit'):
        validate_snapshot(snapshot)


@pytest.mark.parametrize('joint_value', [True, float('inf'), [1, 2], [2, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0]])
def test_invalid_joint_telemetry_rejected(snapshot, joint_value):
    snapshot['qpos']['leaf_hinge'] = joint_value
    with pytest.raises(ValueError, match='qpos'):
        validate_snapshot(snapshot)


def test_free_and_ball_telemetry_is_supported(snapshot):
    snapshot['qpos'] = {'ball': [-1, 0, 0, 0], 'free': [1, 2, 3, 1, 0, 0, 0]}
    snapshot['qpos_vector'] = [-1, 0, 0, 0, 1, 2, 3, 1, 0, 0, 0]
    result = validate_snapshot(snapshot)
    assert result['qpos']['ball'] == [1, 0, 0, 0]


def test_provenance_digest_and_extra_metadata_are_strict(snapshot):
    snapshot['source']['model.json_sha256'] = ''
    with pytest.raises(ValueError, match='SHA256'):
        validate_snapshot(snapshot)
    snapshot['source']['model.json_sha256'] = 'b' * 64
    snapshot['source']['extra_metric'] = float('nan')
    with pytest.raises(ValueError, match='source.extra_metric.*finite'):
        validate_snapshot(snapshot)


def test_minimal_pose_snapshot_is_allowed_without_optional_telemetry(snapshot):
    state = {key: snapshot[key] for key in ('schema_version', 'door_id', 'body_world')}
    assert validate_snapshot(state, expected_door_id='test_door')['body_world']['leaf']['pos'] == [1, 2, 3]


def test_coordinate_and_pixel_origin_conventions_cannot_silently_change(snapshot):
    snapshot['coordinate_system'] = 'left_handed_y_up_centimeters'
    with pytest.raises(ValueError, match='coordinate_system'):
        validate_snapshot(snapshot)
    snapshot['coordinate_system'] = 'right_handed_z_up_meters'
    snapshot['camera']['pixel_origin'] = 'pixel_center'
    with pytest.raises(ValueError, match='pixel_origin'):
        validate_snapshot(snapshot)

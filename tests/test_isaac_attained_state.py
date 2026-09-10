import numpy as np
import pytest
from doorbench.dexterous.isaac_attained_state import extract_attained_state, RECORDED_ROOT_CONVENTION


@pytest.fixture
def archive():
    names = ['joint_' + str(i) for i in range(69)]
    return dict(configuration=dict(root_state_convention=RECORDED_ROOT_CONVENTION,
                    robot_joint_names=names[::-1], door_joint_names=['leaf', 'handle'],
                    args=dict(door_usd='/original/door.usd')),
                motor_contract=dict(joint_names=names, source_xml_sha256='a' * 64),
                provenance=dict(files={'/original/door.usd': 'b' * 64}),
                physics=dict(time_s=np.array([.002, .004]),
                    root=np.tile([0, 0, .9, 1, 0, 0, 0, .1, 0, 0, 0, 0, .2], (2, 1)),
                    joints=np.tile(np.arange(69) * .01, (2, 1)),
                    joint_velocity=np.tile(np.arange(69) * .02, (2, 1)),
                    door=np.array([[.1, .2], [.3, .4]]),
                    door_velocity=np.array([[.5, .6], [.7, .8]])), time_s=.004)


def test_preserves_measured_velocities_and_actual_archive_order(archive):
    r = extract_attained_state(**archive)
    assert r['joint_position']['joint_0'] == .68
    assert r['joint_velocity']['joint_0'] == 1.36
    assert r['root_state_world'][7] == .1
    assert r['door_velocity'] == {'leaf': .7, 'handle': .8}
    assert r['door_position'] == {'leaf': .3, 'handle': .4}
    assert r['time_s'] == .004


@pytest.mark.parametrize('fault', ['missing_velocity', 'short_velocity', 'nan', 'duplicate_clock',
    'reversed_clock', 'interpolation', 'com_root', 'duplicate_joint', 'changed_inventory', 'bad_quaternion'])
def test_rejects_incomplete_or_ambiguous_state(archive, fault):
    p = archive['physics']; c = archive['configuration']
    if fault == 'missing_velocity': del p['joint_velocity']
    if fault == 'short_velocity': p['door_velocity'] = p['door_velocity'][:1]
    if fault == 'nan': p['joints'][0, 0] = np.nan
    if fault == 'duplicate_clock': p['time_s'][:] = .004
    if fault == 'reversed_clock': p['time_s'] = p['time_s'][::-1]
    if fault == 'interpolation': archive['time_s'] = .003
    if fault == 'com_root': c['root_state_convention'] = 'legacy COM velocity'
    if fault == 'duplicate_joint': c['robot_joint_names'][0] = c['robot_joint_names'][1]
    if fault == 'changed_inventory': c['robot_joint_names'][0] = 'unknown_joint'
    if fault == 'bad_quaternion': p['root'][1, 3] = .5
    with pytest.raises(ValueError): extract_attained_state(**archive)

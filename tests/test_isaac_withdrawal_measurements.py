import numpy as np
import pytest

from test_isaac_opening_measurements import calculator, measured_fixture
from doorbench.dexterous.isaac_withdrawal_measurements import IsaacWithdrawalMeasurements
from doorbench.dexterous.standing_withdrawal_audit import clearance_pairs, environment_clearance


def make_measurement(base):
    door, robot = list(base.sources)
    c = IsaacWithdrawalMeasurements(door, robot, ['wrist'])
    args = measured_fixture(c)
    for name in c.required_door_body_names:
        b = c.m.body(name).id
        args['body_poses'][name] = np.r_[c.d.xpos[b], c.d.xquat[b]]
    return c, args


def test_complete_clearance_uses_original_receiving_only_shapes_without_steps(calculator, monkeypatch):
    c, args = make_measurement(calculator)
    def reject(*a, **kw):raise AssertionError('No physics integration permitted')
    for name in ('mj_step', 'mj_step1', 'mj_step2'):monkeypatch.setattr(c.mujoco, name, reject)
    result = c.read(**args)
    assert c.d.time == 0 and result['native_mirror_steps'] == 0
    assert result['environment_pair_count'] == len(clearance_pairs(c.m))
    assert result['right_environment_clearance_m'] == environment_clearance(c.m, c.d, clearance_pairs(c.m))
    assert set(result['body_poses_xyz_wxyz']) == set(c.required_robot_body_names+c.required_door_body_names)
    assert 'bolt' in result['body_poses_xyz_wxyz']


def test_moving_bolt_pose_is_required_and_cannot_disagree(calculator):
    c, args = make_measurement(calculator)
    args['body_poses']['bolt'][0] += .004
    with pytest.raises(ValueError, match='environmental'):
        c.read(**args)
    del args['body_poses']['bolt']
    with pytest.raises(KeyError):c.read(**args)


def test_geometry_receipt_is_detached_from_actual_pose_buffers(calculator):
    c, args = make_measurement(calculator)
    result = c.read(**args)
    before = result['body_poses_xyz_wxyz']['rh_palm'].copy()
    args['body_poses']['rh_palm'][:] = 0
    assert result['body_poses_xyz_wxyz']['rh_palm'] == before

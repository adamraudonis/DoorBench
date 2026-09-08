import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from types import SimpleNamespace

path = Path(__file__).resolve().parents[1] / "doorbench/dexterous/right_hand_release.py"
spec = importlib.util.spec_from_file_location("_pressed_leaf_release", path)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def pose(p, rotation):
    q = rotation.as_quat()
    return np.r_[p, q[3], q[:3]]


def test_pressed_target_is_invariant_to_operator_spring_return():
    leaf = pose([.2, .3, 0], Rotation.from_euler("z", .1))
    pressed = pose([.7, .2, 1], Rotation.from_euler("xyz", [.2, .87, .1]))
    p, r = release.bind_handle_to_leaf(pressed, leaf)
    np.testing.assert_allclose(release.handle_from_leaf(leaf, p, r), pressed, atol=1e-12)
    # There is deliberately no input for the returning operator angle. Its
    # physical pose is evaluated separately, never overwritten by this target.
    returned = pose(pressed[:3], Rotation.from_euler("xyz", [.2, 0., .1]))
    assert not np.allclose(returned, release.handle_from_leaf(leaf, p, r))


def test_target_follows_actual_leaf_rigid_motion():
    leaf = pose([.2, .3, 0], Rotation.from_euler("z", .1))
    pressed = pose([.7, .2, 1], Rotation.from_euler("xyz", [.2, .87, .1]))
    p, r = release.bind_handle_to_leaf(pressed, leaf)
    world = Rotation.from_euler("xyz", [.3, -.2, .6])
    shift = np.array([.4, .1, -.2])
    leaf_r = Rotation.from_quat([*leaf[4:], leaf[3]])
    new_leaf = pose(world.apply(leaf[:3]) + shift, world * leaf_r)
    actual = release.handle_from_leaf(new_leaf, p, r)
    np.testing.assert_allclose(actual[:3], world.apply(pressed[:3]) + shift, atol=1e-12)
    actual_r = Rotation.from_quat([*actual[4:], actual[3]])
    expected_r = world * Rotation.from_quat([*pressed[4:], pressed[3]])
    np.testing.assert_allclose(actual_r.as_matrix(), expected_r.as_matrix(), atol=1e-12)


def test_unknown_stale_or_reversed_leaf_clock_rejected():
    candidate = release.PressedLeafFrameRightRelease.__new__(release.PressedLeafFrameRightRelease)
    candidate.leaf_time = None
    candidate.leaf_pose = None
    with pytest.raises(ValueError):
        candidate._current_leaf(1.)
    candidate.observe_leaf(1., [0, 0, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError):
        candidate._current_leaf(1.002)
    with pytest.raises(ValueError):
        candidate.observe_leaf(.99, [0, 0, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError):
        candidate.observe_leaf(2., [0, 0, 0, 0, 0, 0, 0])


def test_retained_preload_unloads_only_after_explicit_measured_clearance():
    candidate = release.PressedLeafFrameRightRelease.__new__(release.PressedLeafFrameRightRelease)
    candidate.started = 0.
    candidate.last_time = None
    candidate.leaf_time = None
    candidate.leaf_pose = None
    candidate.frozen = None
    candidate.clear_time = None
    candidate.retain_grip_until_clear = True
    candidate.pressed_position_leaf = np.zeros(3)
    candidate.pressed_rotation_leaf = np.eye(3)
    candidate.p_relative = np.zeros(3)
    candidate.r_relative = np.eye(3)
    candidate.duration = 6.
    candidate.distance = .14
    candidate.digit_forces = {'th': 3., 'ff': 2.}
    candidate.teacher = SimpleNamespace(positions=np.zeros((1, 3)), rotations=np.eye(3)[None])
    for t in (1., 2.):
        candidate.observe_leaf(t, [0, 0, 0, 1, 0, 0, 0])
        candidate.update(t)
        assert candidate.teacher.digit_forces == {'th': 3., 'ff': 2.}
    candidate.frozen = (np.zeros(3), np.eye(3))
    for t, scale in ((3., 1.), (3.2, .5), (3.4, 0.)):
        candidate.observe_leaf(t, [0, 0, 0, 1, 0, 0, 0])
        candidate.update(t)
        assert candidate.teacher.digit_forces['th'] == pytest.approx(3. * scale)
        assert candidate.info['grip_preload_scale'] == pytest.approx(scale)

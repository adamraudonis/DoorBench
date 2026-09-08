from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doorbench.dexterous.operation_teacher import DoorOperationTeacher, pose_components, reproject_grasp


GEOMETRY = dict(operator_origin=np.array([.01, .02, 0.]), operator_axis=np.array([0., -1., 0.]),
                leaf_origin=np.array([.03, 0., 0.]), leaf_axis=np.array([0., 0., 1.]))
POSE = np.array([.7, .1, 1., 1., 0., 0., 0.])


class Acquisition:
    def __init__(self):
        self.positions = np.array([[.8, .1, 1.]])
        self.rotations = np.eye(3)[None]
        self.position_integral = np.ones(3)
        self.rotation_integral = np.ones(3)
        self.palm = 0
        self.d = SimpleNamespace(site_xpos=self.positions.copy(), site_xmat=np.eye(3).reshape(1, 9))
        self.fraction = 1.

    def force(self, *args):
        return np.arange(6.), dict(path_fraction=self.fraction)


def tick(wrapper, t, valid=True, **angles):
    return wrapper.force(t, None, {}, {}, POSE, POSE,
                         dict(operator=angles.get('operator', 0.), leaf=angles.get('leaf', 0.),
                              latch=angles.get('latch', 0.)), {}, grasp_qualified=valid)


def test_qualification_resets_on_contact_loss_and_observation_gap():
    acq = Acquisition()
    wrapper = DoorOperationTeacher(acq, GEOMETRY)
    acq.fraction = .9
    for t in np.arange(0., .7, .01):
        tick(wrapper, t)
    assert wrapper.started is None
    acq.fraction = 1.
    for t in np.arange(.7, 1., .01):
        tick(wrapper, t)
    tick(wrapper, 1., False)
    for t in np.arange(1.01, 1.4, .01):
        tick(wrapper, t)
    tick(wrapper, 1.5)  # Missing observations cannot qualify a continuous hold.
    for t in np.arange(1.51, 2., .01):
        tick(wrapper, t)
    assert wrapper.started is None
    force, info = tick(wrapper, 2.01)
    assert wrapper.started == pytest.approx(2.01)
    np.testing.assert_array_equal(force, np.arange(6.))
    assert info['phase'] == 'lever_operation'
    np.testing.assert_array_equal(acq.position_integral, 0.)


def test_opening_requires_actual_release_and_keeps_goal_continuous():
    wrapper = DoorOperationTeacher(Acquisition(), GEOMETRY, press_seconds=1.)
    for t in np.arange(0., .51, .01):
        tick(wrapper, t)
    assert wrapper.started == pytest.approx(.5)
    tick(wrapper, 1.51, operator=.85, latch=.005, leaf=.006)
    assert wrapper.open_started is None
    tick(wrapper, 1.52, operator=.5, latch=.012, leaf=.006)
    assert wrapper.open_started is None
    tick(wrapper, 1.53, operator=.85, latch=.012, leaf=.006)
    assert wrapper.open_started == pytest.approx(1.53)
    assert wrapper.info['goal_leaf_rad'] == 0.  # Do not jump to soft leaf deflection.
    tick(wrapper, 1.54, valid=False, operator=.85, latch=.012, leaf=.006)
    assert wrapper.info['phase'] == 'partial_opening'  # Loads remain audited separately.
    assert 0. < wrapper.info['goal_leaf_rad'] < 1e-6


def test_measured_release_can_start_before_press_timer_without_a_goal_jump():
    wrappers=[DoorOperationTeacher(Acquisition(),GEOMETRY,press_seconds=5.,wait_for_press_completion=value) for value in (True,False)]
    for wrapper in wrappers:
        for t in np.arange(0.,.51,.01):tick(wrapper,t)
        tick(wrapper,1.,operator=.85,latch=.005)
        assert wrapper.open_started is None
        tick(wrapper,1.01,operator=.85,latch=.012)
    assert wrappers[0].open_started is None
    assert wrappers[1].open_started==pytest.approx(1.01)
    assert wrappers[1].info['goal_leaf_rad']==0.


def transformed_pose(pose, rotation, translation):
    p, r = pose_components(pose)
    q = Rotation.from_matrix(rotation @ r).as_quat()
    return np.r_[translation + rotation @ p, q[3], q[:3]]


def test_reprojection_preserves_bound_pose_and_is_world_frame_equivariant():
    relative = np.array([.1, -.03, .02])
    rr = Rotation.from_rotvec([.1, .2, .3]).as_matrix()
    angles = dict(operator=.3, leaf=.2)
    p, r = reproject_grasp(POSE, POSE, angles, angles, relative, rr, GEOMETRY)
    np.testing.assert_allclose(p, POSE[:3] + relative, atol=1e-12)
    np.testing.assert_allclose(r, rr, atol=1e-12)
    goals = dict(operator=.8, leaf=.6)
    p, r = reproject_grasp(POSE, POSE, angles, goals, relative, rr, GEOMETRY)
    world_r = Rotation.from_rotvec([.4, -.7, .2]).as_matrix()
    world_t = np.array([-2., 1., .2])
    moved = transformed_pose(POSE, world_r, world_t)
    p2, r2 = reproject_grasp(moved, moved, angles, goals, relative, rr, GEOMETRY)
    np.testing.assert_allclose(p2, world_t + world_r @ p, atol=1e-12)
    np.testing.assert_allclose(r2, world_r @ r, atol=1e-12)


def test_measured_interface_rejects_invalid_state():
    with pytest.raises(ValueError):
        pose_components([0.] * 7)
    with pytest.raises(ValueError):
        DoorOperationTeacher(Acquisition(), {**GEOMETRY, 'leaf_axis': [0., 0., 2.]})
    wrapper = DoorOperationTeacher(Acquisition(), GEOMETRY)
    with pytest.raises(ValueError):
        tick(wrapper, 0., valid=None)
    tick(wrapper, 1.)
    with pytest.raises(ValueError):
        tick(wrapper, .9)

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

path = Path(__file__).resolve().parents[1] / 'doorbench/dexterous/right_release_return.py'
spec = importlib.util.spec_from_file_location('doorbench.dexterous._return_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pose(position, rotation):
    q = rotation.as_quat()
    return np.r_[position, q[3], q[:3]]


def candidate():
    c = module.ControlledLeverReturn.__new__(module.ControlledLeverReturn)
    c.started = 0.
    c.last_time = None
    c.observed_time = None
    c.return_seconds = 4.
    c.initial_operator = .84
    c.initial_leaf = .1
    c.p_relative = np.array([-.05, -.1, .08])
    c.r_relative = np.eye(3)
    c.frozen = None
    c.digit_forces = {'ff': 2., 'mf': 2., 'rf': 2., 'lf': 2., 'th': 4.}
    c.teacher = SimpleNamespace(positions=np.zeros((1, 3)), rotations=np.eye(3)[None])
    return c


GEOMETRY = dict(operator_origin=[0, 0, 0], operator_axis=[0, 1, 0],
                leaf_origin=[0, 0, 0], leaf_axis=[0, 0, 1])


def test_return_target_is_continuous_at_binding_then_returns_about_joint_axis():
    c = candidate()
    lr = Rotation.from_euler('z', .1)
    hr = lr * Rotation.from_euler('y', .84)
    hp = np.array([.3, -.05, 1.])
    leaf = pose([0, 0, 0], lr)
    handle = pose(hp, hr)
    for t in (0., 2., 4., 6.):
        c.observe_operation(t, handle, leaf, dict(operator=.84, leaf=.1), GEOMETRY)
        c.update(t)
        expected_angle = {0.: .84, 2.: .42, 4.: 0., 6.: 0.}[t]
        expected = lr * Rotation.from_euler('y', expected_angle)
        np.testing.assert_allclose(c.teacher.positions[-1], hp + expected.apply(c.p_relative), atol=1e-12)
        np.testing.assert_allclose(c.teacher.rotations[-1], expected.as_matrix(), atol=1e-12)
        assert c.teacher.digit_forces == c.digit_forces
        assert c.info['release_fraction'] == 0.
        assert not c.info['intentional_ungrip_started']


def test_current_pose_and_unit_joint_axis_required():
    c = candidate()
    leaf = [0, 0, 0, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        c.update(1.)
    with pytest.raises(ValueError):
        c.observe_operation(1., leaf, leaf, dict(operator=.84, leaf=.1),
                            {**GEOMETRY, 'operator_axis': [0, 2, 0]})
    c.observe_operation(1., leaf, leaf, dict(operator=.84, leaf=.1), GEOMETRY)
    with pytest.raises(ValueError):
        c.update(1.002)

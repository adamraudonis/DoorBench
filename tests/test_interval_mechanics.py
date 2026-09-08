import numpy as np
import pytest
from doorbench.dexterous.interval_mechanics import contact_generalized_force, motor_power


def test_rotated_contact_sign_and_virtual_work():
    frame = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    wrench = np.array([4., 2., -1., .1, .3, .5])
    rng = np.random.default_rng(79)
    jp, jr, velocity = rng.normal(size=(3, 7)), rng.normal(size=(3, 7)), rng.normal(size=7)
    qforce, world = contact_generalized_force(frame, wrench, 1, jp, jr)
    assert np.allclose(world[:3], [-2., 4., -1.])
    assert np.isclose(qforce @ velocity, world[:3] @ (jp @ velocity) + world[3:] @ (jr @ velocity))
    other, opposite = contact_generalized_force(frame, wrench, 0, jp, jr)
    assert np.array_equal(other, -qforce)
    assert np.array_equal(opposite, -world)


def test_coupled_motor_power_preserves_virtual_work():
    matrix = np.array([[1., 1., 0.], [0., 0., -.5]])
    forces, velocity = np.array([2., 3.]), np.array([.1, .2, .4])
    actual = motor_power(forces, matrix, velocity)
    assert np.allclose(actual, [.6, -.6])
    assert np.isclose(actual.sum(), (matrix.T @ forces) @ velocity)


def test_invalid_contact_frame_is_rejected():
    with pytest.raises(ValueError):
        contact_generalized_force(np.zeros((3, 3)), np.zeros(6), 1, np.zeros((3, 1)), np.zeros((3, 1)))

"""Privileged mechanical diagnostics from already archived interval wrenches.

No simulator force recomputation is used. A MuJoCo contact frame has its axes
in rows; its archived wrench acts positively on the second contact body.
"""
import numpy as np


def contact_generalized_force(frame, wrench, body_index, jacobian_position,
                              jacobian_rotation):
    """Map one actual interval contact to generalized force and world wrench."""
    frame = np.asarray(frame, dtype=float)
    wrench = np.asarray(wrench, dtype=float)
    jp, jr = np.asarray(jacobian_position), np.asarray(jacobian_rotation)
    if (frame.shape != (3, 3) or wrench.shape != (6,) or jp.ndim != 2
            or jp.shape[0] != 3 or jr.shape != jp.shape or body_index not in (0, 1)
            or not np.isfinite(np.r_[frame.ravel(), wrench, jp.ravel(), jr.ravel()]).all()
            or not np.allclose(frame @ frame.T, np.eye(3), atol=1e-7)):
        raise ValueError('Require a finite orthonormal contact frame and matching Jacobians')
    sign = 1 if body_index == 1 else -1
    force = sign * (frame.T @ wrench[:3])
    torque = sign * (frame.T @ wrench[3:])
    return jp.T @ force + jr.T @ torque, np.r_[force, torque]


def motor_power(force, transmission, velocity):
    """Actual scalar motor force times transmission velocity, in watts."""
    force, transmission, velocity = map(np.asarray, (force, transmission, velocity))
    if (transmission.shape != (force.size, velocity.size) or force.ndim != 1
            or velocity.ndim != 1
            or not np.isfinite(np.r_[force, transmission.ravel(), velocity]).all()):
        raise ValueError('Invalid force, transmission or velocity')
    return force * (transmission @ velocity)

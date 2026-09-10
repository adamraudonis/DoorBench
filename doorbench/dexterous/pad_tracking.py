"""Privileged contact-pad tracking through the robot's original transmissions.

Pads are fixed material points selected in a verified grasp, not the nearest
surface of the current finger. This keeps the intended contact side continuous
while approaching. These task forces become motor commands; no external wrench
or simulator constraint is created.
"""
import mujoco
import numpy as np


def tangent_force(force, normal, maximum_force):
    """Project a requested pad force onto the measured surface tangent plane."""
    force, normal = np.asarray(force, float), np.asarray(normal, float)
    if (force.shape != (3,) or normal.shape != (3,)
            or not np.isfinite(np.r_[force, normal, maximum_force]).all()
            or maximum_force <= 0 or abs(np.linalg.norm(normal)-1.) > 1e-6):
        raise ValueError('Finite force, unit normal and positive force cap required')
    result = force - normal * (force @ normal)
    return result * min(1., maximum_force / max(1e-12, np.linalg.norm(result)))


class PadTracker:
    def __init__(self, model, reference_data, digit_geoms, lever, *, digits,
                 stiffness=300., damping=2., maximum_force=6.):
        self.m = model
        self.stiffness, self.damping, self.maximum_force = stiffness, damping, maximum_force
        self.pads = {}
        self.jp = np.zeros((3, model.nv)); self.jr = self.jp.copy()
        for digit in digits:
            nearest = None
            for geom in digit_geoms[digit]:
                pair = np.zeros(6)
                distance = mujoco.mj_geomDistance(model, reference_data, geom, lever, .2, pair)
                if nearest is None or distance < nearest[0]:
                    nearest = distance, geom, pair.copy()
            distance, geom, pair = nearest
            if abs(distance) > .01:
                raise ValueError(f'{digit}: reference pad is not near the lever ({distance})')
            body = int(model.geom_bodyid[geom])
            rotation = reference_data.xmat[body].reshape(3,3)
            local = rotation.T @ (pair[:3] - reference_data.xpos[body])
            self.pads[digit] = (body, local)

    def positions(self, data):
        return {digit: data.xpos[body] + data.xmat[body].reshape(3,3) @ local
                for digit, (body, local) in self.pads.items()}

    def generalized_force(self, data, targets, *, surface_normals=None):
        result = np.zeros(self.m.nv); errors = {}
        positions = self.positions(data)
        for digit, (body, _) in self.pads.items():
            position = positions[digit]
            mujoco.mj_jac(self.m, data, self.jp, self.jr, position, body)
            error = np.asarray(targets[digit]) - position
            force = self.stiffness * error - self.damping * (self.jp @ data.qvel)
            if surface_normals is not None:
                force = tangent_force(force, surface_normals[digit], self.maximum_force)
            magnitude = np.linalg.norm(force)
            if magnitude > self.maximum_force:
                force *= self.maximum_force / magnitude
            result += self.jp.T @ force
            errors[digit] = float(np.linalg.norm(error))
        return result, errors

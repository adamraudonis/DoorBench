"""Development component: return the lever before attempting to ungrip it.

Only measured-state analytic motor targets change. There is deliberately no
withdrawal in this component; qualification must establish a controlled return
with the original opposed grasp and loaded left support first.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .right_hand_release import AxialRightRelease, _pose_matrix


class ControlledLeverReturn(AxialRightRelease):
    def __init__(self, teacher, path, *, return_seconds=4.):
        super().__init__(teacher, path)
        if not np.isfinite(return_seconds) or not 1. <= return_seconds <= 10.:
            raise ValueError("Require a bounded positive lever-return duration")
        self.return_seconds = float(return_seconds)
        self.observed_time = None
        self.measurement = None

    def observe_operation(self, t, handle_pose, leaf_pose, angles, geometry):
        _pose_matrix(handle_pose)
        _pose_matrix(leaf_pose)
        if not np.isfinite(t) or t < 0 or (self.observed_time is not None and t < self.observed_time):
            raise ValueError("Require monotonic finite measured operation time")
        if set(geometry) != {'operator_origin', 'operator_axis', 'leaf_origin', 'leaf_axis'}:
            raise ValueError("Require the explicit operator and leaf joint geometry")
        vectors = {key: np.asarray(value, float) for key, value in geometry.items()}
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors.values()):
            raise ValueError("Require finite local joint vectors")
        if any(abs(np.linalg.norm(vectors[key]) - 1.) > 1e-6 for key in ('operator_axis', 'leaf_axis')):
            raise ValueError("Require unit local joint axes")
        if not np.isfinite([angles['operator'], angles['leaf']]).all():
            raise ValueError("Require finite measured joint angles")
        self.observed_time = float(t)
        self.measurement = (np.asarray(handle_pose, float).copy(),
                            np.asarray(leaf_pose, float).copy(),
                            {'operator': float(angles['operator']), 'leaf': float(angles['leaf'])},
                            {key: value.copy() for key, value in vectors.items()})

    def _current(self, t):
        if self.observed_time is None or abs(self.observed_time - t) > 1e-8:
            raise ValueError("Lever return requires current measured poses and angles")
        return self.measurement

    def begin(self, t, joints, root, handle_pose):
        measured_handle, _, angles, _ = self._current(t)
        if not np.allclose(handle_pose, measured_handle, atol=1e-8, rtol=0.):
            raise ValueError("Begin handle pose must match the current operation sample")
        super().begin(t, joints, root, handle_pose)
        self.initial_operator = angles['operator']
        self.initial_leaf = angles['leaf']

    def update(self, t):
        if self.started is None:
            return
        if not np.isfinite(t) or t < self.started or (self.last_time is not None and t < self.last_time):
            raise ValueError("Invalid lever-return clock")
        handle, leaf, angles, geometry = self._current(t)
        self.last_time = float(t)
        u = float(np.clip((t - self.started) / self.return_seconds, 0., 1.))
        blend = u ** 3 * (10 + u * (-15 + 6 * u))
        goal_h = self.initial_operator * (1. - blend)
        hp, hr = _pose_matrix(handle)
        lp, lr = _pose_matrix(leaf)
        ha = hp + hr @ geometry['operator_origin']
        la = lp + lr @ geometry['leaf_origin']
        dh = Rotation.from_rotvec(hr @ geometry['operator_axis'] * (goal_h - angles['operator'])).as_matrix()
        dl = Rotation.from_rotvec(lr @ geometry['leaf_axis'] * (self.initial_leaf - angles['leaf'])).as_matrix()
        target_hp = la + dl @ (ha + dh @ (hp - ha) - la)
        target_hr = dl @ dh @ hr
        if self.frozen is None:
            self.teacher.positions[-1] = target_hp + target_hr @ self.p_relative
            self.teacher.rotations[-1] = target_hr @ self.r_relative
        else:
            self.teacher.positions[-1], self.teacher.rotations[-1] = self.frozen
        self.teacher.digit_forces = self.digit_forces.copy()
        self.info = dict(phase='controlled_lever_return', return_fraction=u,
            goal_operator_rad=goal_h, actual_operator_rad=angles['operator'],
            goal_leaf_rad=self.initial_leaf, actual_leaf_rad=angles['leaf'],
            release_fraction=0., release_elapsed_s=t-self.started, grip_preload_scale=1.,
            intentional_ungrip_started=False, measurement_time_s=self.observed_time,
            unexpected_clearance=self.frozen is not None,
            controller_status='development_component_not_qualified')

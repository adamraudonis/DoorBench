"""Timed, measured-frame palm recontact; a privileged teacher component.

This supplies only targets to the existing capped motor controller. It cannot
qualify opening, and a later opening plan must use the actually attained loaded
state. In particular, flattening progresses even if the leaf is motionless.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .screened_panel_teacher import ScreenedWholeBodyPanel


def leaf_transform(pose):
    pose = np.asarray(pose, float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-6):
        raise ValueError('Require the current finite body-origin leaf pose')
    return pose[:3].copy(), Rotation.from_quat([*pose[4:], pose[3]]).as_matrix()


def reproject_palm(position, rotation, initial_leaf, current_leaf):
    initial_position, initial_rotation = leaf_transform(initial_leaf)
    current_position, current_rotation = leaf_transform(current_leaf)
    p, r = np.asarray(position, float), np.asarray(rotation, float)
    if p.shape != (3,) or r.shape != (3,3) or not np.isfinite(np.r_[p, r.ravel()]).all():
        raise ValueError('Require a finite analytic palm goal')
    return (current_position + current_rotation @ initial_rotation.T @ (p-initial_position),
            current_rotation @ initial_rotation.T @ r)


class NoOpeningPhase:
    @staticmethod
    def lead_at(angle):
        if not np.isfinite(angle):
            raise ValueError('Require a finite measured aperture')
        return 0.


class TimedPalmRecontact(ScreenedWholeBodyPanel):
    plan_schema = 'doorbench.palm-recontact-plan.v1'

    def __init__(self, *args, **kwargs):
        if not all(kwargs.get(name) is True for name in
                   ('actual_base_correction', 'palm_normal_admittance', 'correction_include_waist')):
            raise ValueError('This declared prelude requires actual-base waist/arm correction and tactile normal feedback')
        super().__init__(*args, **kwargs)
        self.initial_leaf_pose = None
        self.ready_for_opening = False
        self._advance_time = None

    def make_phase(self, plan, tracking_lead_rad, lead_start_angle, lead_ramp_rad):
        if (tracking_lead_rad != 0 or lead_start_angle is not None
                or plan['initial_leaf_angle_rad'] != plan['final_leaf_angle_rad']
                or not 0 < plan['normal_recontact_m'] <= .006):
            raise ValueError('Require a declared timed recontact plan with no opening-angle progress')
        return NoOpeningPhase()

    def begin(self, t, root, joints, leaf_pose, angle):
        leaf_transform(leaf_pose)
        self.initial_leaf_pose = np.asarray(leaf_pose, float).copy()
        super().begin(t, root, joints, leaf_pose, angle)

    def advance(self, t, angle):
        if self.started is None:
            return
        if (not np.isfinite([t, angle]).all() or t < self.started
                or (self._advance_time is not None and
                    (t < self._advance_time or t-self._advance_time > .00200001))):
            raise ValueError('Require a coherent consecutive prelude clock')
        sample = self.path.sample(t-self.started)
        velocity, acceleration = sample['velocity'], sample['acceleration']
        if (max(abs(velocity[6:])) > 1.2 or max(abs(acceleration[6:])) > 3.
                or np.linalg.norm(velocity[:3]) > .02 or np.linalg.norm(velocity[3:6]) > .03):
            raise ValueError('Timed prelude exceeded original target-rate bounds')
        self.latest = dict(time_s=float(t), coordinate=sample['position'], velocity=velocity,
                           acceleration=acceleration, aperture=float(angle),
                           progress=float(np.clip(sample['progress'], 0., 1.)))
        self._advance_time = float(t)

    def reference_palm_pose(self, position, rotation, leaf_pose):
        if self.initial_leaf_pose is None:
            raise ValueError('Bind the coherent actual start before reprojecting goals')
        return reproject_palm(position, rotation, self.initial_leaf_pose, leaf_pose)

    def update(self, t, root, joints, leaf_pose, palm_load, angle, right_clear=True):
        super().update(t, root, joints, leaf_pose, palm_load, angle, right_clear)
        self.ready_for_opening = bool(t-self.started >= self.path.duration_s and
                                      self.loaded_since is not None and t-self.loaded_since >= .5)
        self.left.info.update(phase='timed_palm_recontact', ready_for_fresh_opening_plan=self.ready_for_opening,
                              prelude_elapsed_s=float(t-self.started), prelude_duration_s=self.path.duration_s,
                              target_frame='current_measured_leaf', opening_phase_started=False)

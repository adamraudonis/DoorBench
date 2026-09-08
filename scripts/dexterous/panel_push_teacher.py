"""Privileged handle release followed by a bounded-motor palm push.

The analytic model is used for robot FK and Jacobians only. No state, force,
constraint, or contact is inserted into the physics plant.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from physx_teacher import HandleTeacher, smooth


class PanelPushTeacher(HandleTeacher):
    def __init__(self, *args, target_angle=1.2, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(target_angle) or target_angle < .7:
            raise ValueError('Keep a usable aperture target of at least 0.7 rad')
        self.target_angle = target_angle
        self.panel_start = None
        self.panel_offset = -.08
        self.panel_contact_steps = 0
        self.panel_done = None

    def command(self, t, root, joints, leaf, handle, handle_angle, door_angle,
                velocities=None, hand_forces=None, grasp=None):
        control, info = super().command(t, root, joints, leaf, handle,
            handle_angle, door_angle, velocities, hand_forces, grasp)
        if self.retract_start is None or t < self.retract_start + 1.0:
            return control, info
        m, d = self.m, self.d
        # Restore the measured state after the parent's analytic IK iteration.
        d.qpos[:7] = root[:7]
        for n, q in joints.items():
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = q
        mujoco.mj_forward(m, d)
        measured_q = d.qpos[self.qa].copy()
        measured_position = d.site_xpos[self.palm].copy()
        measured_rotation = d.site_xmat[self.palm].reshape(3, 3).copy()
        bias = d.qfrc_bias[self.va].copy()
        leaf_rotation = Rotation.from_quat([*leaf[4:7], leaf[3]]).as_matrix()
        normal = leaf_rotation[:, 1]
        if self.panel_start is None:
            self.panel_start = t
            self.panel_initial = measured_position.copy()
        elapsed = t - self.panel_start
        reaction = sum((np.asarray(f) for f in (hand_forces or {}).values()), np.zeros(3))
        loaded = max(0., float(-reaction @ normal))
        self.panel_contact_steps = self.panel_contact_steps + 1 if loaded > 1. else 0
        # First move across in free space, then approach the panel slowly.
        position = np.array(leaf[:3]) + leaf_rotation @ np.array([.25, -.23, .95])
        if elapsed < 1.2:
            position = self.panel_initial + smooth(elapsed/1.2)*(position-self.panel_initial)
        else:
            if elapsed < 2.4:
                offset = -.23 + smooth((elapsed-1.2)/1.2)*(.23+self.panel_offset)
            else:
                # A robot motor target advances until actual hand loading appears.
                self.panel_offset = float(np.clip(self.panel_offset + .0003*np.clip((8.-loaded)/8., -1., 1.), -.11, -.025))
                offset = self.panel_offset
            position = np.array(leaf[:3]) + leaf_rotation @ np.array([.25, offset, .95])
        if self.panel_done is None and door_angle >= self.target_angle and self.panel_contact_steps >= 3:
            self.panel_done = t
            self.panel_done_position = measured_position.copy()
            self.panel_done_normal = normal.copy()
        if self.panel_done is not None:
            position = self.panel_done_position - .15*smooth((t-self.panel_done)/1.0)*self.panel_done_normal
        # Keep the palm facing the panel, leaving rotation within its plane free.
        desired_z = -normal if self.panel_done is None else -self.panel_done_normal
        jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
        mujoco.mj_jacSite(m, d, jp, jr, self.palm)
        measured_jac = jp[:, self.va].copy()
        for _ in range(35):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            current_z = d.site_xmat[self.palm].reshape(3, 3)[:, 2]
            orient = np.cross(current_z, desired_z)
            blend = smooth(elapsed/1.2)
            error = np.r_[5*(position-d.site_xpos[self.palm]), blend*orient]
            mujoco.mj_jacSite(m, d, jp, jr, self.palm)
            tangent = np.eye(3)-np.outer(current_z, current_z)
            jac = np.vstack([5*jp[:, self.va], blend*tangent@jr[:, self.va]])
            inverse = jac.T@np.linalg.inv(jac@jac.T+.003*np.eye(6))
            change = inverse@error + (np.eye(len(self.joints))-inverse@jac)@(.03*(measured_q-d.qpos[self.qa]))
            d.qpos[self.qa] = np.clip(d.qpos[self.qa]+np.clip(change, -.05, .05), self.low, self.high)
            if np.linalg.norm(error) < 1e-4:
                break
        control[self.act] = d.qpos[self.qa]
        control[self.fingers] = 0.
        self.feedforward[:] = 0.
        push = 16.*smooth((elapsed-2.0)/.8)*np.clip((25.-loaded)/15., 0., 1.) if self.panel_done is None else 0.
        self.feedforward[self.act] = bias + measured_jac.T@(normal*push)
        self.control[:] = control
        info.update(phase='withdraw', panel_phase='complete' if self.panel_done is not None else 'push' if elapsed >= 2.4 else 'reach',
            release_fingers=True, panel_goal_angle_rad=self.target_angle, panel_measured_load_N=loaded,
            panel_offset_m=self.panel_offset, panel_tracking_error_m=float(np.linalg.norm(position-measured_position)),
            panel_normal_error=float(np.linalg.norm(measured_rotation[:, 2]-desired_z)))
        return control.copy(), info

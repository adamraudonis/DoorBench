"""Privileged loaded-panel continuation through the original H1 v2 motors.

Call only after the right hand has physically cleared the lever. The left palm
retains a full pose relative to the measured moving leaf, with bounded normal
admittance. The waist and both arms cooperate; the free right hand may change
posture while an analytic lever-clearance term keeps it away from the handle.

This changes only analytic reference targets. It never steps an analytic model,
sets a plant pose, commands the door or emits a helper force. Exact contact
loads and scene frames are privileged teacher inputs, never actor observations.
The qualified route begins in the recorded deep stance; another attained root
or robot requires a new workspace and physical qualification.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

class CoordinatedPanelPush:

    def __init__(self, left, *, target_palm_load=3.0, maximum_normal_offset=.012,
                 left_cup_seconds=2.0):
        if not np.isfinite(target_palm_load) or not 2.0 < target_palm_load <= 10.0:
            raise ValueError('Expected a finite palm-load target above the 2 N qualification floor')
        self.target_palm_load = float(target_palm_load)
        if not np.isfinite([maximum_normal_offset,left_cup_seconds]).all() or not 0 < maximum_normal_offset <= .05 or left_cup_seconds <= 0:
            raise ValueError('Invalid bounded panel controller settings')
        self.maximum_normal_offset = float(maximum_normal_offset)
        self.left_cup_seconds = float(left_cup_seconds)
        self.left = left
        self.teacher = left.teacher
        self.started = None
        self.last_update = None
        self.clear = False

    def begin(self, t, root, joints, leaf_pose, angle):
        if self.started is not None:
            raise ValueError('Panel continuation already began')
        self._validate(t, leaf_pose, 0.0, angle)
        if self.left.started is None or self.left.progress < 0.999:
            raise ValueError('Left contact approach must already be completed')
        self.started = t
        l = self.left
        l._read(root, joints)
        R = Rotation.from_quat([*leaf_pose[4:7], leaf_pose[3]]).as_matrix()
        self.local_position = R.T @ (l.d.site_xpos[l.palm] - leaf_pose[:3])
        self.initial_angle = float(angle)
        self.local_rotation = R.T @ l.d.site_xmat[l.palm].reshape(3, 3)
        self.initial_offset = 0.0
        l.contact_force = 3.5
        self.right_names = ['right_' + n for n in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_yaw')] + ['rh_WRJ2', 'rh_WRJ1']
        self.names = l.names + self.right_names
        self.offset = self.initial_offset
        self.loaded_since = None
        self.right_geoms = [g for g in range(l.m.ngeom) if l.m.geom_contype[g] and l.m.body(l.m.geom_bodyid[g]).name.startswith('rh_')]
        self.right_hold_position = l.d.site_xpos[l.right_palm].copy()
        self.right_hold_rotation = l.d.site_xmat[l.right_palm].reshape(3, 3).copy()
        self.finger_initial = {n: joints[n] for n in ('rh_LFJ5', 'lh_LFJ5')}
        self._set_joints(self.names)
        self.previous = l.d.qpos[self.qa].copy()
        keep = np.array([self.teacher.m.joint(int(j)).name != 'torso' for j in self.teacher.arm_joints])
        self.teacher.arm_joints = self.teacher.arm_joints[keep]
        self.teacher.arm_q = self.teacher.arm_q[keep]
        self.teacher.arm_v = self.teacher.arm_v[keep]

    def _set_joints(self, names):
        m = self.teacher.m
        self.names = names
        self.js = np.array([m.joint(n).id for n in names])
        self.qa = m.jnt_qposadr[self.js]
        self.indices = [self.teacher.names.index(n) for n in names]
        self.low = m.jnt_range[self.js, 0] + 0.01
        self.high = m.jnt_range[self.js, 1] - 0.01
        for name in ('rh_WRJ2', 'rh_WRJ1'):
            if name in names:
                i = names.index(name)
                self.low[i] += 0.03
                self.high[i] -= 0.03

    def _validate(self, t, leaf_pose, palm_load, angle):
        pose = np.asarray(leaf_pose, float)
        if pose.shape != (7,) or not np.isfinite(np.r_[pose, t, palm_load, angle]).all():
            raise ValueError('Expected finite clock, load, aperture and xyz/wxyz leaf pose')
        if t < 0 or palm_load < 0 or (not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-05)):
            raise ValueError('Invalid panel sensor state')
        if self.last_update is not None and t < self.last_update - 1e-09:
            raise ValueError('Panel clock moved backwards')

    def update(self, t, root, joints, leaf_pose, palm_load, angle, right_clear=True):
        if self.started is None:
            raise ValueError('Panel continuation has not begun')
        if right_clear is not True:
            raise ValueError('Require measured right-hand lever clearance before torso continuation')
        self._validate(t, leaf_pose, palm_load, angle)
        l = self.left
        teacher = self.teacher
        m, d = (l.m, l.d)
        l._read(root, joints)
        if right_clear and (not self.clear):
            self.clear = True
            for name in self.right_names:
                teacher.path[-1, teacher.names.index(name)] = joints[name]
            teacher.arm_joints = np.array([], dtype=int)
            teacher.arm_q = np.array([], dtype=int)
            teacher.arm_v = np.array([], dtype=int)
            self._set_joints(l.names + self.right_names)
            self.previous = d.qpos[self.qa].copy()
        if self.last_update is not None and t - self.last_update < 0.01 - 1e-08:
            return
        dt = 0.0 if self.last_update is None else t - self.last_update
        self.last_update = t
        R = Rotation.from_quat([*leaf_pose[4:7], leaf_pose[3]]).as_matrix()
        normal = R[:, 1]
        u = float(np.clip((angle - self.initial_angle) / 0.35, 0, 1))
        blend = u ** 3 * (10 + u * (-15 + 6 * u))
        local = self.local_position.copy()
        local[0] -= 0.04 * blend
        local[2] -= 0.15 * blend
        self.offset = float(np.clip(self.offset + dt * 0.016 * np.clip((self.target_palm_load - palm_load) / 4.0, -1.0, 1.0), -0.015, self.maximum_normal_offset))
        goal = leaf_pose[:3] + R @ local + normal * self.offset
        rotation = R @ self.local_rotation
        right_goal = self.right_hold_position
        right_rotation = self.right_hold_rotation
        seed = self.previous.copy()
        fu = float(np.clip((t - self.started) / 2.0, 0.0, 1.0))
        fb = fu ** 3 * (10 + fu * (-15 + 6 * fu))
        for name, value in self.finger_initial.items():
            cu = float(np.clip((t-self.started)/self.left_cup_seconds,0.,1.)) if name == 'lh_LFJ5' else fu
            cb = cu**3*(10+cu*(-15+6*cu))
            teacher.path[-1, teacher.names.index(name)] = value + cb * (0.12 - value)

        def fun(q):
            d.qpos[self.qa] = q
            mujoco.mj_kinematics(m, d)
            right = np.r_[0.05 * (d.site_xpos[l.right_palm] - right_goal), 0.02 * Rotation.from_matrix(right_rotation @ d.site_xmat[l.right_palm].reshape(3, 3).T).as_rotvec()]
            clearance = np.array([max(0.0, 0.025 - float(mujoco.mj_geomDistance(m, d, g, teacher.lever, 0.1, None))) for g in self.right_geoms])
            return np.r_[100 * (d.site_xpos[l.palm] - goal), 10 * Rotation.from_matrix(rotation @ d.site_xmat[l.palm].reshape(3, 3).T).as_rotvec(), right, 100 * clearance, 0.1 * (q - seed)]
        fit = least_squares(fun, np.clip(self.previous, self.low, self.high), bounds=(self.low, self.high), max_nfev=160)
        self.previous = fit.x.copy()
        l.target = fit.x[1:8].copy()
        teacher.path[-1, self.indices] = fit.x
        l._read(root, joints)
        error = float(np.linalg.norm(d.site_xpos[l.palm] - goal))
        rotation_error = float(np.linalg.norm(Rotation.from_matrix(rotation @ d.site_xmat[l.palm].reshape(3, 3).T).as_rotvec()))
        if palm_load >= 2.0:
            if self.loaded_since is None:
                self.loaded_since = t
        else:
            self.loaded_since = None
        l.normal = normal
        l.progress = 1.0
        l.info = dict(phase='full_palm_panel_push', left_progress=1.0, left_tracking_error_m=error, left_rotation_error_rad=rotation_error, left_panel_load_N=float(palm_load), left_offset_m=self.offset, left_ik_residual=float(np.linalg.norm(fit.fun[:6])), left_loaded_duration_s=0.0 if self.loaded_since is None else t - self.loaded_since, right_release_clear=self.clear, right_reference_mode='clearance-constrained free posture', contact_height_goal_m=float(local[2]), contact_radius_goal_m=float(local[0]))

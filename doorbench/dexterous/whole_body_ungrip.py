"""Opt-in, actual-state-specific whole-body lever return and measured ungrip.

This privileged development helper has no active plant handle. It changes only
analytic hand targets and existing landed-foot stance targets. Its frozen path
must be independently screened and its exact attained-state guard must pass;
it is not an Isaac-portable path or a sensor-only policy.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .whole_body_return import WholeBodyLeverReturn
from .right_hand_release import _pose_matrix


def interpolate_rotation(a, b, fraction):
    return Rotation.from_rotvec(fraction * Rotation.from_matrix(b @ a.T).as_rotvec()).as_matrix() @ a


def smooth_fraction(value):
    u = float(np.clip(value, 0., 1.))
    return u ** 3 * (10. + u * (-15. + 6. * u))


def validate_endpoint_clearance(receipt):
    """Validate the new declared40mm geometric endpoint without changing live gates."""
    if receipt is None:
        return  # Legacy development comparisons retain their original contract.
    values=np.array([receipt['required_clearance_m'],receipt['minimum_all_rh_environment_clearance_m']],float)
    if (not np.isfinite(values).all() or values[0]!=.04 or values[1]<.04
            or receipt.get('passed') is not True
            or receipt['hand_shapes']<=0 or receipt['environment_shapes']<=0):
        raise ValueError('Require the declared40mm clearance of every RH collision shape')


class WholeBodyMeasuredUngrip(WholeBodyLeverReturn):
    def __init__(self, teacher, path, whole_body_path, ungrip_path, *, goal_frame="attained-resting-world", handoff_posture_targets=False):
        super().__init__(teacher, path, whole_body_path)
        if type(handoff_posture_targets) is not bool:
            raise ValueError('Require an explicit posture ownership option')
        self.handoff_posture_targets = handoff_posture_targets
        if goal_frame not in ("attained-resting-world", "measured-handle"):
            raise ValueError("Unknown ungrip goal frame")
        self.goal_frame = goal_frame
        self.ungrip_plan = json.loads(Path(ungrip_path).read_text())
        plan = self.ungrip_plan
        validate_endpoint_clearance(plan.get('endpoint_clearance'))
        if plan.get('schema') != 'doorbench.whole-body-ungrip-plan.v1':
            raise ValueError('Require the declared whole-body ungrip path')
        audit = plan['interpolation_audit']
        if audit.get('passed') is not True or audit['samples'] < 2000:
            raise ValueError('Require a passed dense interpolation screen')
        self.ungrip_rows = plan['rows']
        self.ungrip_times = np.asarray([row['time_s'] for row in self.ungrip_rows], float)
        if len(self.ungrip_rows) < 2 or self.ungrip_times[0] != 0 or not np.all(np.diff(self.ungrip_times) > 0):
            raise ValueError('Require complete monotonic ungrip timing')
        self.ungrip_body_names = plan['body_names']
        self.ungrip_finger_names = plan['finger_names']
        if set(self.ungrip_body_names) != set(self.body_names):
            raise ValueError('Ungrip and return body contracts disagree')
        self.ungrip_finger_indices = [teacher.names.index(n) for n in self.ungrip_finger_names]
        self.ungrip_roots = np.asarray([row['root'] for row in self.ungrip_rows], float)
        self.ungrip_joints = np.asarray([[row['joints'][n] for n in self.ungrip_body_names] for row in self.ungrip_rows], float)
        self.ungrip_fingers = np.asarray([[row['finger_joints'][n] for n in self.ungrip_finger_names] for row in self.ungrip_rows], float)
        self.ungrip_positions = np.asarray([row['palm_position_handle'] for row in self.ungrip_rows], float)
        self.ungrip_rotations = np.asarray([row['palm_rotation_handle'] for row in self.ungrip_rows], float)
        arrays = [self.ungrip_times, self.ungrip_roots, self.ungrip_joints,
                  self.ungrip_fingers, self.ungrip_positions, self.ungrip_rotations]
        if not np.isfinite(np.concatenate([v.ravel() for v in arrays])).all():
            raise ValueError('Require finite screened ungrip targets')
        for root in self.ungrip_roots:
            if root.shape != (7,) or not np.isclose(np.linalg.norm(root[3:]), 1., atol=1e-6):
                raise ValueError('Invalid screened root quaternion')
        for rotation in self.ungrip_rotations:
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
                raise ValueError('Invalid screened palm rotation')
        self.ungrip_delay = float(plan['start_after_release_s'])
        if not np.isfinite(self.ungrip_delay) or self.ungrip_delay < self.return_seconds:
            raise ValueError('Ungrip must follow the completed lever return')
        self.ungrip_started = None
        self.state_time = None
        self.current_state = None
        self.ready_for_panel = False
        self.command_bridge_seconds = .5
        self.release_path_start = min(row['time_s'] for row in self.ungrip_rows if row['phase'] == 'measured_release')

    def observe_state(self, t, root, joints):
        root = np.asarray(root, float)
        if root.shape != (13,) or set(joints) != set(self.teacher.names):
            raise ValueError('Require complete actual measured robot state')
        values = np.r_[t, root, [joints[n] for n in self.teacher.names]]
        if not np.isfinite(values).all() or (self.state_time is not None and t < self.state_time):
            raise ValueError('Require finite monotonic robot measurements')
        self.state_time = float(t)
        self.current_state = (root.copy(), dict(joints))

    def _begin_ungrip(self, t):
        if self.state_time is None or abs(t - self.state_time) > 1e-8:
            raise ValueError('Ungrip requires coherent current robot state')
        root, joints = self.current_state
        plan = self.ungrip_plan
        if not np.allclose(root[:7], plan['initial_root'], atol=1e-5, rtol=0.):
            raise ValueError('Ungrip path requires the exact physically attained root')
        if not np.allclose([joints[n] for n in plan['initial_joints']], list(plan['initial_joints'].values()), atol=1e-5, rtol=0.):
            raise ValueError('Ungrip path requires the exact physically attained joint state')
        handle, _, angles, _ = self._current(t)
        if abs(angles['operator'] - plan['initial_operator_rad']) > 1e-5 or abs(angles['leaf'] - plan['initial_leaf_rad']) > 1e-5:
            raise ValueError('Ungrip path requires its screened actual door configuration')
        # Preserve continuity of desired commands as well as actual plant state.
        self.ungrip_handle_pose = handle.copy()
        self.bridge_position = self.teacher.positions[-1].copy()
        self.bridge_rotation = self.teacher.rotations[-1].copy()
        self.bridge_fingers = self.teacher.path[-1, self.ungrip_finger_indices].copy()
        self.bridge_body = super().body_goal(t)
        self.ungrip_started = float(t)

    def _sample(self, elapsed):
        t = float(np.clip(elapsed, 0., self.ungrip_times[-1]))
        i = min(len(self.ungrip_times)-2, max(0, int(np.searchsorted(self.ungrip_times, t, side='right')-1)))
        f = (t-self.ungrip_times[i])/(self.ungrip_times[i+1]-self.ungrip_times[i])
        a, b = self.ungrip_roots[i:i+2]
        ra = Rotation.from_quat([*a[4:], a[3]]).as_matrix()
        rb = Rotation.from_quat([*b[4:], b[3]]).as_matrix()
        return dict(position=(1-f)*a[:3]+f*b[:3], rotation=interpolate_rotation(ra, rb, f),
            joints=dict(zip(self.ungrip_body_names, (1-f)*self.ungrip_joints[i]+f*self.ungrip_joints[i+1])),
            fingers=(1-f)*self.ungrip_fingers[i]+f*self.ungrip_fingers[i+1],
            palm_position_handle=(1-f)*self.ungrip_positions[i]+f*self.ungrip_positions[i+1],
            palm_rotation_handle=interpolate_rotation(self.ungrip_rotations[i], self.ungrip_rotations[i+1], f),
            route_time_s=t)

    def body_goal(self, t):
        if self.ungrip_started is None:
            return super().body_goal(t)
        elapsed = t-self.ungrip_started
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Require current finite ungrip clock')
        target = self._sample(max(0., elapsed-self.command_bridge_seconds))
        if elapsed < self.command_bridge_seconds:
            f = smooth_fraction(elapsed/self.command_bridge_seconds)
            target['position'] = (1-f)*self.bridge_body['position']+f*target['position']
            target['rotation'] = interpolate_rotation(self.bridge_body['rotation'], target['rotation'], f)
            target['joints'] = {n:(1-f)*self.bridge_body['joints'][n]+f*v for n,v in target['joints'].items()}
        return target

    def update(self, t):
        if self.started is None:
            return
        if self.ungrip_started is None and t-self.started < self.ungrip_delay-1e-8:
            super().update(t)
            return
        if self.ungrip_started is None:
            self._begin_ungrip(t)
        if not np.isfinite(t) or t < self.last_time:
            raise ValueError('Require monotonic finite ungrip clock')
        self.last_time = float(t)
        handle, _, angles, _ = self._current(t)
        goal_handle = self.ungrip_handle_pose if self.goal_frame == "attained-resting-world" else handle
        hp, hr = _pose_matrix(goal_handle)
        elapsed = t-self.ungrip_started
        target = self.body_goal(t)
        position = hp+hr@target['palm_position_handle']
        rotation = hr@target['palm_rotation_handle']
        fingers = target['fingers']
        if elapsed < self.command_bridge_seconds:
            f = smooth_fraction(elapsed/self.command_bridge_seconds)
            position = (1-f)*self.bridge_position+f*position
            rotation = interpolate_rotation(self.bridge_rotation, rotation, f)
            fingers = (1-f)*self.bridge_fingers+f*fingers
        if self.frozen is None:
            self.teacher.positions[-1] = position
            self.teacher.rotations[-1] = rotation
        else:
            self.teacher.positions[-1], self.teacher.rotations[-1] = self.frozen
        posture_owned_by_panel = self.handoff_posture_targets and self.frozen is not None
        if posture_owned_by_panel and not self.ready_for_panel:
            raise ValueError('Posture handoff requires the complete cleared release')
        if not posture_owned_by_panel:
            self.teacher.path[-1,self.ungrip_finger_indices] = fingers
            self.teacher.path[-1,self.torso_index] = target['joints']['torso']
        release_elapsed = max(0., target['route_time_s']-self.release_path_start)
        scale = float(np.clip(1-release_elapsed/.4, 0., 1.))
        self.teacher.digit_forces = {n:v*scale for n,v in self.digit_forces.items()}
        self.ready_for_panel = bool(elapsed >= self.command_bridge_seconds+self.ungrip_times[-1])
        self.info = dict(phase='whole_body_measured_ungrip', release_elapsed_s=t-self.started, measurement_time_s=self.observed_time,
            attained_frame_time_s=self.ungrip_started, ungrip_elapsed_s=elapsed, route_time_s=target['route_time_s'],
            release_fraction=float(np.clip(release_elapsed/(self.ungrip_times[-1]-self.release_path_start),0,1)),
            ready_for_panel=self.ready_for_panel, grip_preload_scale=scale, goal_frame=self.goal_frame,
            actual_operator_rad=angles['operator'], actual_leaf_rad=angles['leaf'],
            target_pelvis_position_m=target['position'].tolist(),
            target_torso_rad=float(self.teacher.path[-1,self.torso_index]),
            posture_target_owner='panel' if posture_owned_by_panel else 'release',
            controller_status='development_actual_state_specific_unqualified')

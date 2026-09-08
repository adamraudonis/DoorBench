"""Force-only, privileged acquisition through loaded aperture for H1/Shadow v2.

All active-plant state arrives as numeric measurements. The owned MuJoCo model
is an unstepped kinematic/dynamic calculator. Only the original 61 capped motor
forces leave this interface. This is a teacher, never a sensor-only actor.
"""
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.acquisition_teacher import AcquisitionTeacher
from doorbench.dexterous.bimanual_transfer import LeftPalmContact, load_screen_targets
from doorbench.dexterous.right_hand_release import AxialRightRelease
from doorbench.dexterous.panel_continuation import CoordinatedPanelPush


def _smooth(value):
    u = np.clip(value, 0., 1.)
    return u**3*(10+u*(-15+6*u))


def _pose(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('Expected finite measured world xyz/wxyz pose')
    if not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-5):
        raise ValueError('Measured quaternion must be normalized')
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, pose[3:])
    return pose[:3], matrix.reshape(3, 3)


class FullOpeningTeacher:
    """Measured-state stage transitions; no active simulator object is accepted.

    Times are relative to the contact-free acquisition start. The default
    schedule follows native full-push005; coherent measurements use a 5 N palm
    target and establish the left palm cup during transfer. A different stance
    needs its own workspace and physical qualification. The consumer must end
    or hand off at the first declared aperture crossing; indefinite open-door
    holding and traversal are not supplied by this controller.
    """
    def __init__(self, robot_xml, motors, reference, joint_geometry, *, door_xml,
                 left_targets, release_screen, runtime_screen=None,
                 min_acquisition_seconds=10.6, min_left_seconds=22.,
                 min_release_seconds=30., press_seconds=5., opening_seconds=3.,
                 qualification_seconds=.5, physics_dt=.002, target_aperture=1.2,
                 open_on_latch_clear=False, operator_compliance_gain=0.,
                 operator_compliance_limit=.15, freeze_compliance_on_release=True):
        times = [min_acquisition_seconds, min_left_seconds, min_release_seconds,
                 press_seconds, opening_seconds, qualification_seconds, physics_dt,
                 target_aperture]
        if not np.isfinite(times).all() or min(times) <= 0:
            raise ValueError('Expected finite positive opening settings')
        if min_left_seconds < min_acquisition_seconds or min_release_seconds < min_left_seconds:
            raise ValueError('Stage minimum times must be ordered')
        if type(open_on_latch_clear) is not bool or type(freeze_compliance_on_release) is not bool:
            raise ValueError('Explicit operation transition and compliance policies required')
        if not np.isfinite([operator_compliance_gain,operator_compliance_limit]).all() or min(operator_compliance_gain,operator_compliance_limit)<0:
            raise ValueError('Invalid bounded operator compliance settings')
        self.open_on_latch_clear=open_on_latch_clear
        self.operator_compliance_gain=operator_compliance_gain
        self.operator_compliance_limit=operator_compliance_limit
        self.freeze_compliance_on_release=freeze_compliance_on_release
        self.operator_compliance=0.
        self.geometry = {k: np.asarray(joint_geometry[k], float) for k in
                         ('operator_origin', 'operator_axis', 'leaf_origin', 'leaf_axis')}
        for key, value in self.geometry.items():
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError('Expected finite mechanism geometry')
            if key.endswith('axis') and not np.isclose(np.linalg.norm(value), 1., atol=1e-6):
                raise ValueError('Joint axes must be normalized')
        self.acquisition = AcquisitionTeacher(robot_xml, motors, reference)
        self.names = self.acquisition.names
        self.caps = self.acquisition.caps
        if self.caps.shape != (61, 2):
            raise ValueError('This qualified teacher requires the 61 original H1/Shadow motors')
        self.left_cup_joint=self.names.index('lh_LFJ5')
        cup_motors=np.flatnonzero(self.acquisition.matrix[:,self.left_cup_joint])
        if len(cup_motors)!=1 or np.count_nonzero(self.acquisition.matrix[cup_motors[0]])!=1 or self.acquisition.matrix[cup_motors[0],self.left_cup_joint]!=1:
            raise ValueError('Expected the original independent left LFJ5 motor')
        self.left_cup_motor=int(cup_motors[0])
        targets = load_screen_targets(left_targets, robot_xml, door_xml,
                                      runtime_screen=runtime_screen)
        self.left = LeftPalmContact(self.acquisition, motors, targets, fixed_waist=True)
        self.release = AxialRightRelease(self.acquisition, Path(release_screen))
        self.push = CoordinatedPanelPush(self.left, target_palm_load=5.0,
                                        maximum_normal_offset=.025,left_cup_seconds=.25,
                                        flatten_palm=True,track_target_velocity=True)
        self.min_acquisition_seconds = float(min_acquisition_seconds)
        self.min_left_seconds = float(min_left_seconds)
        self.min_release_seconds = float(min_release_seconds)
        self.press_seconds, self.opening_seconds = press_seconds, opening_seconds
        self.qualification_seconds, self.physics_dt = qualification_seconds, physics_dt
        self.target_aperture = target_aperture
        self.history = deque()
        self.last_time = self.operation_started = self.open_started = None
        self.initial_contract = None
        self.all_physics_qualified = self.all_pad_patches_qualified = True
        self.handoffs = {}
        self.crossing = None
        self.operation_info = dict(phase='acquisition')
        self.info = dict(phase='acquisition', privileged_teacher=True)

    def _record_evidence(self, t, angles, evidence):
        booleans = ('grasp_qualified', 'physics_qualified', 'right_pad_patches_valid')
        required = set(booleans) | {'hand_contact_count','left_panel_load_N','left_palm_load_N','right_lever_clearance_m'}
        if not isinstance(evidence,dict) or set(evidence)!=required:
            raise ValueError('Expected only the explicit numeric/boolean opening evidence contract')
        for key in booleans:
            if not isinstance(evidence.get(key), (bool, np.bool_)):
                raise ValueError('Explicit actual boolean evidence required: '+key)
        count = evidence.get('hand_contact_count')
        if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 0:
            raise ValueError('Expected actual nonnegative hand contact count')
        loads = [evidence.get(k) for k in
                 ('left_panel_load_N', 'left_palm_load_N', 'right_lever_clearance_m')]
        if not np.isfinite(loads).all() or min(loads[:2]) < 0:
            raise ValueError('Expected finite actual surface loads and signed clearance')
        if self.last_time is not None and t <= self.last_time:
            raise ValueError('Opening evidence must advance the clock exactly once per sample')
        if self.last_time is not None and t-self.last_time > self.physics_dt+1e-7:
            # A gap cannot count toward a continuous-contact qualification.
            self.history.clear()
        if self.last_time is None:
            self.initial_contract = bool(abs(angles['leaf']) <= .001 and
                                         abs(angles['operator']) <= .001 and count == 0)
        self.last_time = t
        self.all_physics_qualified &= bool(evidence['physics_qualified'])
        self.all_pad_patches_qualified &= bool(evidence['right_pad_patches_valid'])
        self.history.append(dict(time=t, leaf=angles['leaf'], **evidence))
        lower = t-self.qualification_seconds-1e-8
        while self.history and self.history[0]['time'] < lower:
            self.history.popleft()

    def _qualified(self, predicate):
        if len(self.history) < round(self.qualification_seconds/self.physics_dt)+1:
            return False
        if self.history[-1]['time']-self.history[0]['time'] < self.qualification_seconds-1e-7:
            return False
        return all(predicate(row) for row in self.history)

    def _begin_operation(self, t, handle_pose, angles, right_palm_pose):
        hp, hr = _pose(handle_pose)
        pp,pr = _pose(right_palm_pose)
        self.p_relative = hr.T@(pp-hp)
        self.r_relative = hr.T@pr
        self.initial_handle = angles['operator']
        self.operation_started = t
        self.operation_last_time = t
        self.acquisition.position_integral[:] = 0.
        self.acquisition.rotation_integral[:] = 0.
        self.handoffs['qualified_grasp'] = float(t)

    def _operation_targets(self, t, handle_pose, leaf_pose, angles):
        goal_h = self.initial_handle+(.87-self.initial_handle)*_smooth(
            (t-self.operation_started)/self.press_seconds)
        elapsed=min(.05,max(0.,t-self.operation_last_time));self.operation_last_time=t
        if self.open_started is None or not self.freeze_compliance_on_release:
            self.operator_compliance=float(np.clip(self.operator_compliance+self.operator_compliance_gain*elapsed*(goal_h-angles['operator']),0.,self.operator_compliance_limit))
        press_ready=self.open_on_latch_clear or t >= self.operation_started+self.press_seconds
        if self.open_started is None and press_ready and angles['operator'] >= .80 and angles['latch'] >= .011:
            self.open_started = t
            self.initial_leaf_goal = self.operation_info.get('goal_leaf_rad', 0.)
            self.handoffs['latch_released'] = float(t)
        goal_l = 0. if self.open_started is None else self.initial_leaf_goal+(.08-self.initial_leaf_goal)*_smooth((t-self.open_started)/self.opening_seconds)
        hp, hr = _pose(handle_pose)
        lp, lr = _pose(leaf_pose)
        ha = hp+hr@self.geometry['operator_origin']
        la = lp+lr@self.geometry['leaf_origin']
        dh = Rotation.from_rotvec(hr@self.geometry['operator_axis']*(goal_h+self.operator_compliance-angles['operator'])).as_matrix()
        dl = Rotation.from_rotvec(lr@self.geometry['leaf_axis']*(goal_l-angles['leaf'])).as_matrix()
        hp = ha+dh@(hp-ha)
        hr = dh@hr
        hp = la+dl@(hp-la)
        hr = dl@hr
        self.acquisition.positions[-1] = hp+hr@self.p_relative
        self.acquisition.rotations[-1] = hr@self.r_relative
        self.operation_info = dict(phase='lever_operation' if self.open_started is None else 'partial_opening',
                                   goal_handle_rad=float(goal_h), goal_leaf_rad=float(goal_l),
                                   palm_compliance_rotation_rad=self.operator_compliance,
                                   actual_handle_rad=angles['operator'], actual_leaf_rad=angles['leaf'],
                                   actual_bolt_m=angles['latch'])

    def force(self, t, root, joints, velocities, handle_pose, leaf_pose, angles,
              hand_forces, *, evidence, right_palm_pose, pose_time_s):
        """Return (61 original capped forces, info), consuming actual state once.

        root13 is xyz+wxyz+world linear/angular velocity; body poses are xyz+wxyz.
        Angles contain operator/leaf radians and latch metres. Evidence contains
        current grasp/physics/pad-patch booleans, hand_contact_count, actual
        left_panel_load_N and left_palm_load_N, and signed minimum distance from
        every actual right-hand collision shape to the lever. Hand-force vectors
        are actual world forces on robot bodies for the unchanged stance QP.
        The right palm pose is measured xyz/wxyz at pose_time_s, which must
        match t. Pose and joint measurements may not silently mix simulator
        cache times. These privileged measurements must never reach an actor.
        """
        angles = {key: float(angles[key]) for key in ('operator', 'leaf', 'latch')}
        if not np.isfinite([t, *angles.values()]).all() or t < 0:
            raise ValueError('Expected finite nonnegative opening clock and mechanism state')
        _pose(handle_pose)
        _pose(leaf_pose)
        _pose(right_palm_pose)
        if not np.isfinite(pose_time_s) or abs(pose_time_s-t)>1e-7:
            raise ValueError('Measured body/palm poses and joint state must share the current step time')
        self._record_evidence(float(t), angles, evidence)
        teacher, left = self.acquisition, self.left
        left._read(root, joints)
        if self.operation_started is None and t >= self.min_acquisition_seconds-1e-8:
            if self._qualified(lambda row: row['grasp_qualified']) and teacher.info.get('path_fraction', 0) >= .999:
                self._begin_operation(float(t), handle_pose, angles, right_palm_pose)
        if self.operation_started is not None and self.release.started is None:
            self._operation_targets(t, handle_pose, leaf_pose, angles)
        if left.started is None and t >= self.min_left_seconds-1e-8 and self.open_started is not None:
            if self._qualified(lambda row: row['grasp_qualified'] and .075 <= row['leaf'] <= .10):
                left.begin(float(t), root, joints, leaf_pose, handle_pose)
                self.handoffs['left_approach'] = float(t)
        if self.release.started is None and t >= self.min_release_seconds-1e-8 and left.loaded_since is not None and t-left.loaded_since >= self.qualification_seconds:
            valid = self._qualified(lambda row: row['grasp_qualified'] and row['left_panel_load_N'] >= 2.)
            if valid and self.initial_contract and self.all_physics_qualified and self.all_pad_patches_qualified:
                self.release.begin(float(t), joints, root, handle_pose)
                self.handoffs['right_release'] = float(t)
        if self.release.started is not None:
            self.release.handle_pose = np.asarray(handle_pose).copy()
            if self.release.frozen is None and t-self.release.started > 1. and evidence['right_lever_clearance_m'] >= .02:
                left._read(root, joints)
                position,rotation = _pose(right_palm_pose)
                self.release.frozen = (position.copy(),rotation.copy())
                self.handoffs['right_clearance'] = float(t)
        self.release.update(float(t))
        if self.release.frozen is not None:
            if self.push.started is None:
                self.push.begin(float(t), root, joints, leaf_pose, angles['leaf'])
                left.damping_scale=2.0
                self.handoffs['panel_continuation'] = float(t)
            if self.push.started is not None:
                self.push.update(float(t), root, joints, leaf_pose,
                                 evidence['left_palm_load_N'], angles['leaf'], True)
        else:
            left.update_targets(float(t), root, joints, leaf_pose,
                                evidence['left_panel_load_N'], handle_pose)
        force, base_info = teacher.force(float(t), root, joints, velocities, handle_pose, hand_forces)
        force = left.apply_forces(force, joints, velocities)
        if self.release.frozen is not None:
            # The little-finger metacarpal forms part of the palm's loaded edge.
            # Its original 1 Nm motor can oppose that load, but the nominal
            # 1 Nm/rad position servo otherwise lets it press into the low stop.
            name='lh_LFJ5'
            target=teacher.path[-1,self.left_cup_joint]
            force[self.left_cup_motor] += 8.0*(target-joints[name])-.1*velocities[name]
        force = np.clip(force, self.caps[:, 0], self.caps[:, 1])
        if force.shape != (61,) or not np.isfinite(force).all():
            raise ValueError('Invalid opening motor command')
        if self.crossing is None and angles['leaf'] >= self.target_aperture:
            self.crossing = dict(time_s=float(t), aperture_rad=angles['leaf'],
                palm_load_N=float(evidence['left_palm_load_N']),
                final_palm_hold=self._qualified(lambda row: row['left_palm_load_N'] >= 2.),
                all_physics_qualified=self.all_physics_qualified,
                all_pad_patches_qualified=self.all_pad_patches_qualified,
                right_release_complete=bool(self.release.info.get('release_fraction', 0) >= .999 and evidence['right_lever_clearance_m'] >= .02))
        phase = self.operation_info['phase']
        if left.started is not None: phase = 'left_panel_contact'
        if self.release.started is not None: phase = 'right_axial_release'
        if self.push.started is not None: phase = 'loaded_panel_opening'
        self.operation_info.update(actual_handle_rad=angles['operator'],
            actual_leaf_rad=angles['leaf'],actual_bolt_m=angles['latch'],
            measurement_time_s=float(t))
        self.info = {**base_info, 'phase':phase, 'operation':self.operation_info.copy(),
                         'left':left.info.copy(), 'release':self.release.info.copy(),
                         'handoffs':self.handoffs.copy(), 'aperture_crossing':self.crossing,
                         'native_mirror_steps':0, 'privileged_teacher':True, 'pose_time_s':float(pose_time_s),
                         'measured_evidence':dict(evidence),'evidence_time_s':float(t),
                         'left_control_measurement_time_s':self.push.last_update if self.push.started is not None else left.last_update}
        return force.copy(), self.info.copy()

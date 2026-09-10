"""Engine-independent privileged acquisition, lever and partial-opening wrapper.

All poses are measured world [x,y,z,qw,qx,qy,qz]. Joint anchors/axes are expressed
in their measured child-body frames. Only the wrapped acquisition teacher emits
motor forces; this module changes reference goals, never a physical pose.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def smooth_phase(value):
    u = np.clip(value, 0., 1.)
    return u**3*(10+u*(-15+6*u))


def pose_components(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
        raise ValueError('Expected finite measured world position and wxyz quaternion')
    return pose[:3], Rotation.from_quat([*pose[4:7], pose[3]]).as_matrix()


def reproject_grasp(handle_pose, leaf_pose, angles, goals, relative_position, relative_rotation, geometry):
    """Apply desired hinge-angle differences about measured world joint frames."""
    hp, hr = pose_components(handle_pose)
    lp, lr = pose_components(leaf_pose)
    ha = hp+hr@geometry['operator_origin']
    la = lp+lr@geometry['leaf_origin']
    dh = Rotation.from_rotvec(hr@geometry['operator_axis']*(goals['operator']-angles['operator'])).as_matrix()
    dl = Rotation.from_rotvec(lr@geometry['leaf_axis']*(goals['leaf']-angles['leaf'])).as_matrix()
    hp = ha+dh@(hp-ha)
    hr = dh@hr
    hp = la+dl@(hp-la)
    hr = dl@hr
    return hp+hr@relative_position, hr@relative_rotation


class DoorOperationTeacher:
    def __init__(self, acquisition_teacher, joint_geometry, *, qualified_hold_seconds=.5,
                 min_acquisition_seconds=0., press_seconds=5., opening_seconds=3.,
                 operator_target=.87, release_operator_threshold=.80,
                 release_bolt_threshold=.011, leaf_target=.08, wait_for_press_completion=True,
                 operator_compliance_gain=0., operator_compliance_limit=.15,
                 freeze_compliance_on_release=True, grasp_offset_in_handle_m=(0.,0.,0.), index_proximal_offset_rad=0., index_tendon_offset_rad=0., fixed_pad_control=False, hold_attained_grasp=False, attained_hold_stage='opening', pad_control_profile='commanded-material-v1', leaf_lead_limit_rad=None, operator_follow_after_leaf_rad=None, handle_hub_avoidance=False):
        if type(fixed_pad_control) is not bool:raise ValueError('Explicit contact-controller flag required')
        if pad_control_profile not in ('commanded-material-v1','actual-material-v1','actual-material-v2','measured-pressure-v1'):raise ValueError('Unknown pad control profile')
        if type(handle_hub_avoidance) is not bool:raise ValueError('Explicit hub-avoidance flag required')
        self.hub_avoidance=None
        if handle_hub_avoidance:
            from .handle_hub_avoidance import HandleHubAvoidance
            self.hub_avoidance=HandleHubAvoidance(acquisition_teacher)
        self.pad_control_profile=pad_control_profile
        if leaf_lead_limit_rad is not None and (not np.isfinite(leaf_lead_limit_rad) or not .002<=leaf_lead_limit_rad<=.03):raise ValueError('Measured leaf lead must be .002..0.03 rad')
        self.leaf_lead_limit_rad=leaf_lead_limit_rad
        if operator_follow_after_leaf_rad is not None and (not np.isfinite(operator_follow_after_leaf_rad) or not .015<=operator_follow_after_leaf_rad<=.05 or hold_attained_grasp):raise ValueError('Operator follow requires .015..0.05 rad clearance and no fixed attained hold')
        self.operator_follow_after_leaf_rad=operator_follow_after_leaf_rad;self.operator_follow_started=None
        self.fixed_pad_control=fixed_pad_control;self.pad_control=None
        if type(hold_attained_grasp) is not bool or (hold_attained_grasp and fixed_pad_control):raise ValueError('Attained hold is a separate explicit hand controller')
        self.attained_hold=None
        if attained_hold_stage not in ('acquisition','operator','aperture','opening'):raise ValueError('Explicit attained hold stage required')
        self.attained_hold_stage=attained_hold_stage
        if hold_attained_grasp:
            from .qualified_hand_hold import QualifiedHandHold
            self.attained_hold=QualifiedHandHold(acquisition_teacher)
        self.grasp_offset=np.asarray(grasp_offset_in_handle_m,dtype=float)
        if self.grasp_offset.shape!=(3,) or not np.isfinite(self.grasp_offset).all() or np.linalg.norm(self.grasp_offset)>.01:
            raise ValueError('Grasp offset must be a finite handle-frame vector within 10 mm')
        if not np.isfinite(index_proximal_offset_rad) or abs(index_proximal_offset_rad)>.1:
            raise ValueError('Index proximal reference offset must be within 0.1 rad')
        if not np.isfinite(index_tendon_offset_rad) or abs(index_tendon_offset_rad)>.12:
            raise ValueError('Index tendon reference offset must be within 0.12 rad')
        self.index_tendon_offset=float(index_tendon_offset_rad)
        self.index_tendon_reference=None
        if self.index_tendon_offset:
            self.index_tendon_columns=[acquisition_teacher.names.index('rh_FFJ'+str(i)) for i in (1,2)]
            self.index_tendon_reference=acquisition_teacher.path[-1,self.index_tendon_columns].copy()
        self.index_proximal_offset=float(index_proximal_offset_rad)
        self.index_reference=None
        if self.index_proximal_offset:
            self.index_column=acquisition_teacher.names.index('rh_FFJ3')
            self.index_reference=float(acquisition_teacher.path[-1,self.index_column])
        self.acquisition = acquisition_teacher
        self.geometry = {k:np.asarray(joint_geometry[k], float) for k in
                         ('operator_origin','operator_axis','leaf_origin','leaf_axis')}
        for key, value in self.geometry.items():
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError('Joint geometry must contain finite 3D vectors')
            if key.endswith('axis') and not np.isclose(np.linalg.norm(value),1.,atol=1e-6):
                raise ValueError('Joint axes must be normalized')
        values = [qualified_hold_seconds,min_acquisition_seconds,press_seconds,opening_seconds,
                  operator_target,release_operator_threshold,release_bolt_threshold,leaf_target,
                  operator_compliance_gain,operator_compliance_limit]
        if not np.isfinite(values).all() or min(values) < 0 or min(qualified_hold_seconds,press_seconds,opening_seconds) <= 0:
            raise ValueError('Invalid operation timing or travel')
        self.qualified_hold_seconds = qualified_hold_seconds
        self.min_acquisition_seconds = min_acquisition_seconds
        self.press_seconds, self.opening_seconds = press_seconds, opening_seconds
        self.operator_target, self.leaf_target = operator_target, leaf_target
        self.release_operator_threshold = release_operator_threshold
        self.release_bolt_threshold = release_bolt_threshold
        if type(wait_for_press_completion) is not bool:
            raise ValueError('Explicit press-completion policy is required')
        self.wait_for_press_completion = wait_for_press_completion
        if type(freeze_compliance_on_release) is not bool:
            raise ValueError('Explicit compliance release policy is required')
        self.freeze_compliance_on_release=freeze_compliance_on_release
        self.operator_compliance_gain=operator_compliance_gain
        self.operator_compliance_limit=operator_compliance_limit
        self.operator_compliance=0.
        self.qualified_since = self.last_time = self.started = self.open_started = None
        self.info = dict(phase='acquisition')

    def _bind(self, t, handle_pose, angles):
        hp, hr = pose_components(handle_pose)
        teacher = self.acquisition
        # FK is from the current measured robot state just consumed by force().
        self.p_relative = hr.T@(teacher.d.site_xpos[teacher.palm]-hp)
        self.r_relative = hr.T@teacher.d.site_xmat[teacher.palm].reshape(3,3)
        if self.fixed_pad_control:
            from .operation_pad_control import OperationPadControl
            self.pad_control=OperationPadControl(teacher,handle_pose,profile=self.pad_control_profile)
        self.initial_handle = angles['operator']
        self.started = t
        teacher.position_integral[:] = 0.
        teacher.rotation_integral[:] = 0.
        self.info = dict(phase='lever_operation',operation_start_s=t,goal_leaf_rad=0.)

    def force(self, t, root, joints, velocities, handle_pose, leaf_pose, angles, hand_loads, *, grasp_qualified):
        """Return native-capped motors; qualify transitions using actual contact data.

        ``angles`` contains operator/leaf angles in radians and latch retraction
        in metres. ``grasp_qualified`` is supplied by the active plant's pad audit;
        no contact is inferred from the reference path. A >50 ms observation gap
        breaks the qualification hold. The physical per-step audit is separate.
        """
        angles = {key:float(angles[key]) for key in ('operator','leaf','latch')}
        if not np.isfinite([t,*angles.values()]).all():
            raise ValueError('Nonfinite operation measurements')
        if not isinstance(grasp_qualified,(bool,np.bool_)):
            raise ValueError('Explicit actual grasp qualification is required')
        if self.last_time is not None and t < self.last_time-1e-9:
            raise ValueError('Operation clock went backwards')
        stale = self.last_time is not None and t-self.last_time > .05+1e-9
        elapsed=0. if self.last_time is None else min(.05,t-self.last_time)
        self.last_time = t
        teacher = self.acquisition
        if self.started is None:
            force, info = teacher.force(t,root,joints,velocities,handle_pose,hand_loads)
            if stale or not grasp_qualified or info.get('path_fraction',0) < .999:
                self.qualified_since = None
            elif self.qualified_since is None:
                self.qualified_since = t
            if self.qualified_since is not None and t-self.qualified_since >= self.qualified_hold_seconds-1e-9 and t >= self.min_acquisition_seconds-1e-9:
                self._bind(t,handle_pose,angles)
            return force, {**info,**self.info}
        goal_h = self.initial_handle+(self.operator_target-self.initial_handle)*smooth_phase((t-self.started)/self.press_seconds)
        # Compensate compliant finger deflection with a bounded palm-reference
        # rotation. The actual operator target and joint limits do not change;
        # this is controller memory, never a door pose/force command.
        if self.open_started is None or not self.freeze_compliance_on_release:
            self.operator_compliance=float(np.clip(self.operator_compliance+
                self.operator_compliance_gain*elapsed*(goal_h-angles['operator']),
                0.,self.operator_compliance_limit))
        press_ready = not self.wait_for_press_completion or t >= self.started+self.press_seconds
        if self.open_started is None and press_ready and angles['operator'] >= self.release_operator_threshold and angles['latch'] >= self.release_bolt_threshold:
            self.open_started = t
            self.initial_leaf_goal = self.info.get('goal_leaf_rad',0.)
        goal_l = 0. if self.open_started is None else self.initial_leaf_goal+(self.leaf_target-self.initial_leaf_goal)*smooth_phase((t-self.open_started)/self.opening_seconds)
        requested_leaf_goal=goal_l
        if self.open_started is not None and self.leaf_lead_limit_rad is not None:
            goal_l=min(goal_l,angles['leaf']+self.leaf_lead_limit_rad)
        if self.operator_follow_after_leaf_rad is not None and self.open_started is not None and self.operator_follow_started is None and angles['leaf']>=self.operator_follow_after_leaf_rad:
            self.operator_follow_started=t
        follow=0. if self.operator_follow_started is None else float(smooth_phase(t-self.operator_follow_started))
        reference_h=(1-follow)*(goal_h+self.operator_compliance)+follow*angles['operator']
        # Ramp the optional reference recenter over one second. This moves a
        # bounded motor controller's target, never the physical hand or handle.
        offset=smooth_phase(t-self.started)*self.grasp_offset
        pos, rot = reproject_grasp(handle_pose,leaf_pose,angles,dict(operator=reference_h,leaf=goal_l),
                                  self.p_relative+offset,self.r_relative,self.geometry)
        if self.index_reference is not None:
            teacher.path[-1,self.index_column]=self.index_reference+smooth_phase(t-self.started)*self.index_proximal_offset
        if self.index_tendon_reference is not None:
            teacher.path[-1,self.index_tendon_columns]=self.index_tendon_reference+smooth_phase(t-self.started)*self.index_tendon_offset/2
        teacher.positions[-1] = pos
        teacher.rotations[-1] = rot
        force, info = teacher.force(t,root,joints,velocities,handle_pose,hand_loads)
        if self.pad_control is not None:
            force,pad_info=self.pad_control.force(force,t-self.started,handle_pose,leaf_pose,angles,
                dict(operator=reference_h,leaf=goal_l),self.geometry,hand_loads=hand_loads)
            info={**info,**pad_info}
        if self.attained_hold is not None:
            eligible=bool(grasp_qualified and (self.attained_hold_stage=='acquisition' or
                (self.attained_hold_stage=='operator' and self.open_started is not None
                 and angles['operator']>=.75 and angles['latch']>=.0105) or
                (self.open_started is not None and t>=self.open_started+self.opening_seconds
                and .075<=angles['leaf']<=.10 and (self.attained_hold_stage=='aperture' or
                 (angles['operator']>=.75 and angles['latch']>=.0105)))))
            force,hold_info=self.attained_hold.force(t,force,joints,velocities,eligible=eligible)
            info={**info,**hold_info}
        # Attained posture tracking replaces finger efforts. Apply clearance
        # feedback last so a captured hold cannot cancel live hub protection.
        if self.hub_avoidance is not None:
            force,hub_info=self.hub_avoidance.force(force,float(smooth_phase(t-self.started)))
            info={**info,**hub_info}
        self.info = dict(phase='lever_operation' if self.open_started is None else 'partial_opening',
                         operation_start_s=self.started,opening_start_s=self.open_started,
                         goal_handle_rad=float(goal_h),goal_leaf_rad=float(goal_l),
                         requested_leaf_goal_rad=float(requested_leaf_goal),leaf_lead_limit_rad=self.leaf_lead_limit_rad,
                         palm_compliance_rotation_rad=self.operator_compliance,
                         operator_follow_started_s=self.operator_follow_started,operator_follow_fraction=follow,
                         commanded_operator_reference_rad=float(reference_h),operator_follow_after_leaf_rad=self.operator_follow_after_leaf_rad,
                         grasp_offset_in_handle_m=offset.tolist(),
                         index_proximal_offset_rad=float(smooth_phase(t-self.started)*self.index_proximal_offset),
                         index_tendon_offset_rad=float(smooth_phase(t-self.started)*self.index_tendon_offset),
                         actual_handle_rad=angles['operator'],actual_leaf_rad=angles['leaf'],
                         actual_bolt_m=angles['latch'])
        return force, {**info,**self.info}

"""Read-only last-three controller snapshots for a future reference handoff.

Existing calculations supply transient goals explicitly. This observer never
evaluates a reference, updates a controller, reads a plant, or restores state.
A snapshot is accepted only after its existing command completes a 2 ms step.
"""
from collections import deque
import copy

import numpy as np

SCHEMA = 'doorbench.isaac-standing-continuation-reference-tail.v1'
DT = .002


def _finite(value, shape, name):
    value = np.asarray(value)
    if value.dtype.kind not in 'fiu' or value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Complete finite captured '+name+' required')
    return value.tolist()


def _time(value, name):
    array=np.asarray(value)
    if array.shape != () or array.dtype.kind not in 'fiu' or not np.isfinite(array):
        raise ValueError('Actual captured '+name+' required')
    return float(value)


def _optional_time(value, name):
    return None if value is None else _time(value, name)


def _plain(value):
    """Copy finite existing data only; no implicit numerical fallback."""
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all(): raise ValueError('Nonfinite captured array')
        return value.tolist()
    if isinstance(value, np.generic): return _plain(value.item())
    if value is None or type(value) in (str, bool, int): return value
    if type(value) is float:
        if not np.isfinite(value): raise ValueError('Nonfinite captured scalar')
        return value
    if type(value) in (list, tuple): return [_plain(v) for v in value]
    if type(value) is dict and all(type(k) is str for k in value): return {k:_plain(v) for k,v in value.items()}
    raise ValueError('Only detached existing numeric controller data may be captured')


class StandingContinuationReferenceTail:
    """Explicit Isaac-withdrawal-only observer; no work on the earlier prefix."""
    def __init__(self, controller, *, motor_names):
        if controller.isaac_runtime is None or controller.coupled is None:
            raise ValueError('Reference tail is opt-in actual Isaac coupled withdrawal only')
        self.controller = controller
        self.pending = None
        self.rows = deque(maxlen=3)
        self.completed = 0
        self.failure = None
        if len(motor_names)!=61 or len(set(motor_names))!=61 or not all(type(n) is str for n in motor_names):
            raise ValueError('Exact named original 61-motor inventory required')
        self.static=dict(motor_names=list(motor_names),motor_caps=_finite(controller.acquisition.caps,(61,2),'original motor caps'),
            left_path_at_construction=_plain(controller.left.path),left_path_names=list(controller.left.names),
            path_scope='Static authored left path; only its last nominal is replaced by the coupled reference. That current nominal is copied in every accepted row.')
        self.tracking_contract=None
        data = controller.source_context.data
        self.binding = {key:copy.deepcopy(data[key]) for key in
            ('runtime_path', 'runtime_sha256', 'motor_contract_sha256', 'source_state_sha256')}

    def capture_command(self, controller, time_s, *, post_feedback_targets,
                        right_goal_position, right_goal_rotation, returned_command,
                        finger_reference_velocity, grip_preload_scale,
                        measured_root, measured_joints, measured_velocities,
                        measured_handle_pose, measured_leaf_pose, measured_angles):
        """Copy already computed force-local values at the existing force return."""
        if self.failure is not None: raise ValueError('Reference capture already failed')
        try:
            if controller is not self.controller or controller.started_withdrawal is None or self.pending is not None:
                raise ValueError('One newly computed Isaac withdrawal command required')
            t = _time(time_s, 'command time')
            expected = controller.started_withdrawal if not self.rows else self.rows[-1]['post_step_time_s']
            if abs(t-expected)>1e-8: raise ValueError('Consecutive completed withdrawal commands required')
            teacher=controller.acquisition; coupled=controller.coupled; motion=coupled.motion
            names=list(teacher.names); motion_names=list(coupled.names)
            if len(names)!=69 or len(set(names))!=69 or set(names)!=set(motion_names) or set(post_feedback_targets)!=set(names):
                raise ValueError('Complete original joint and reference coordinate names required')
            if set(measured_joints)!=set(names) or set(measured_velocities)!=set(names) or set(measured_angles)!={'leaf','operator','latch'}:
                raise ValueError('Complete actual command-input coordinates required')
            if _time(motion.time,'coupled reference epoch') != t:
                raise ValueError('Coupled accepted reference and returned command epochs differ')
            stance=teacher.stance; left=controller.left; arm=controller.arm; palm=controller.palm; hand=controller.hand
            inherited=controller.inherited_support; feedback=inherited.feedback
            if (feedback is None or feedback is not inherited.transfer.support_feedback or feedback.left is not left
                    or inherited.left is not left or inherited.previous != t or feedback.previous[0] != t
                    or left.support_load_target != inherited.target_N or feedback.maximum_target_N != inherited.target_N):
                raise ValueError('Actual unchanged inherited palm feedback at command epoch required')
            if arm.last_call_time != t or palm.last_time != t:
                raise ValueError('Actual arm/palm feedback command epochs differ')
            force=_finite(returned_command,(61,),'returned motor command')
            if not np.array_equal(np.asarray(force),teacher.last_force):
                raise ValueError('Captured force must be the final outer withdrawal command')
            if np.any(np.asarray(force)<teacher.caps[:,0]) or np.any(np.asarray(force)>teacher.caps[:,1]):
                raise ValueError('Captured command exceeds original caps')
            nominal=_finite(motion.value,(75,),'accepted coupled coordinates')
            left_names=list(left.names); arm_names=list(arm.names); palm_names=list(palm.names)
            if not left.path or left.target is None or left.previous is None:
                raise ValueError('Actual attained left path, IK target and seed required')
            if self.tracking_contract is None:
                self.tracking_contract=dict(
                    arm=dict(names=arm_names,motor_indices=_plain(arm.act),joint_velocity_addresses=_plain(arm.va),
                        kp=_finite(arm.kp,(len(arm_names),),'actual arm stiffness'),kd=_finite(arm.kd,(len(arm_names),),'actual arm damping')),
                    hand=dict(motor_indices=_plain(hand.indices),stiffness_scale=_time(hand.scale,'actual attained hand scale'),
                        kp=_finite(hand.kp,(len(hand.indices),),'actual hand stiffness'),kd=_finite(hand.kd,(len(hand.indices),),'actual hand damping')),
                    scope='Actual enhanced tracking gains after predecessor handoff, not assumed equal to base motor gains')
            # Optional state is explicitly absent when the existing mode did not
            # create it. This is not a zero velocity or reset/filter substitute.
            target_velocity=dict(available=hasattr(left,'target_velocity'),
                value=_plain(left.target_velocity) if hasattr(left,'target_velocity') else None)
            previous_target=dict(available=hasattr(feedback,'previous_target'),
                value=_plain(feedback.previous_target) if hasattr(feedback,'previous_target') else None)
            row=dict(command_time_s=t,source_binding=copy.deepcopy(self.binding),
                observed_input=dict(time_s=t,root13_actor_origin=_finite(measured_root,(13,),'command input root'),
                    joint_names=names,joint_position=_finite([measured_joints[n] for n in names],(69,),'command input joints'),
                    joint_velocity=_finite([measured_velocities[n] for n in names],(69,),'command input velocity'),
                    handle_pose_xyz_wxyz=_finite(measured_handle_pose,(7,),'command input handle'),
                    leaf_pose_xyz_wxyz=_finite(measured_leaf_pose,(7,),'command input leaf'),
                    angles={k:_time(measured_angles[k],'command input '+k) for k in ('leaf','operator','latch')}),
                coupled=dict(joint_names=motion_names,value=nominal,
                    velocity=_finite(motion.velocity,(75,),'accepted coupled reference velocity'),time_s=t,
                    root_coordinate_origin_xyz_wxyz=_finite(coupled.geometry.initial[coupled.geometry.rq:coupled.geometry.rq+7],(7,),'coupled root origin'),
                    accepted_samples=int(coupled.accepted),
                    coordinate_convention='World root displacement/relative rotation-vector followed by named scalar joint targets',
                    acceleration_state_available=False),
                stance=dict(joint_names=list(controller.stance_names),
                    target_root=_finite(stance.target_root,(3,),'stance root'),
                    target_rotation=_finite(stance.target_rotation,(3,3),'stance rotation'),
                    joint_target=_finite(stance.joint_target,(len(controller.stance_names),),'stance joints'),
                    root_bias=_finite(controller.stance_root_bias,(3,),'stance root bias'),
                    rotation_bias=_finite(controller.stance_rotation_bias,(3,3),'stance rotation bias'),
                    joint_bias=_finite(controller.stance_joint_bias,(len(controller.stance_names),),'stance joint bias'),
                    qp_warm_start=_plain(stance.last),solver_metadata=_plain(stance.last_solver_metadata)),
                right=dict(goal_position_world=_finite(right_goal_position,(3,),'requested RH goal'),
                    goal_rotation_world=_finite(right_goal_rotation,(3,3),'requested RH orientation'),
                    post_feedback_joint_names=names,
                    post_feedback_joint_targets=_finite([post_feedback_targets[n] for n in names],(69,),'post-feedback joint targets'),
                    arm=dict(names=arm_names,previous_target=_finite(arm.previous_target,(len(arm_names),),'arm previous target'),
                        previous_target_time_s=_time(arm.previous_time,'arm previous target time'),
                        last_call_time_s=t,reference_velocity=_finite(arm.reference_velocity,(len(arm_names),),'arm reference velocity'),
                        preload=_finite(arm.preload,(len(arm_names),),'arm preload')),
                    palm=dict(names=palm_names,offset=_finite(palm.offset,(len(palm_names),),'RH palm correction'),last_time_s=t,
                        correction_gain_s_inv=_time(palm.correction_gain_s_inv,'palm gain'))),
                left=dict(names=left_names,started_s=_time(left.started,'left start'),
                    last_update_s=_time(left.last_update,'left IK epoch'),progress=_time(left.progress,'left progress'),
                    tracking_error_m=_time(left.tracking,'left tracking error'),loaded_since_s=_optional_time(left.loaded_since,'loaded since'),
                    nominal_joint_target=_finite(left.path[-1]['nominal'],(len(left_names),),'LH nominal target'),
                    target=_finite(left.target,(len(left_names)-1,),'left IK target'),
                    previous=_finite(left.previous,(len(left.solve_names),),'left IK seed'),solve_names=list(left.solve_names),
                    initial_position_delta=_finite(left.initial_position_delta,(3,),'initial left offset'),
                    waist_delta=_time(left.waist_delta,'left waist delta'),panel_palm_rotation=_finite(left.panel_palm_rotation,(3,3),'LH panel-local rotation'),
                    normal_offset_m=_time(left.offset,'left normal offset'),target_velocity=target_velocity,
                    support_target_N=_time(left.support_load_target,'support target'),filtered_palm_load_N=_time(left.filtered_palm_load,'filtered palm load'),
                    hybrid_normal_target_N=_time(left.hybrid_normal_target,'hybrid target'),hybrid_blend=_time(left.hybrid_blend,'hybrid blend'),
                    normal_world=_finite(left.normal,(3,),'left normal'),
                    surface_velocity_world=_finite(left.surface_velocity_world,(3,),'surface velocity'),
                    normal_contact_point_local=_finite(left.normal_contact_point_local,(3,),'normal contact point')),
                support=dict(inherited_started_s=_time(inherited.started,'inherited start'),previous_s=_time(inherited.previous,'inherited previous'),
                    feedback_started_s=_time(feedback.started,'feedback start'),maximum_target_N=_time(feedback.maximum_target_N,'maximum support target'),
                    previous_leaf_time_s=_time(feedback.previous[0],'previous leaf time'),
                    previous_leaf_position=_finite(feedback.previous[1],(3,),'previous leaf position'),
                    previous_leaf_rotation=_finite(feedback.previous[2],(3,3),'previous leaf rotation'),
                    previous_ik_target=previous_target,live_feedback_identity_retained=True),
                hand=dict(motor_indices=_plain(hand.indices),targets=_plain(hand.targets),preload=_plain(hand.preload),
                    original_preload=_plain(controller.preload),grip_preload_scale=_time(grip_preload_scale,'grip preload scale'),
                    finger_reference_velocity=_plain(finger_reference_velocity),
                    previous_finger_target=_plain(controller.previous_finger_target)),
                withdrawal=dict(started_s=_time(controller.started_withdrawal,'withdrawal start'),
                    release_started_s=_optional_time(controller.release_started,'release start'),
                    handoff_offset=_plain(controller.handoff.offset),handoff_seconds=_time(controller.handoff.seconds,'motor handoff duration'),
                    handoff_elapsed_s=_time(controller.handoff.last_elapsed,'motor handoff clock')),
                returned_motor_command=force,returned_motor_dtype=np.asarray(returned_command).dtype.str,
                private_generalized_bias=_finite(teacher.d.qfrc_bias,(teacher.m.nv,),'private inverse-dynamics bias'),
                private_bias_scope='Model qfrc_bias retained after the command calculation; not a measured actuator or contact force',
                authorized_stages=0)
            self.pending=row
        except Exception as exc:
            self.failure=type(exc).__name__+': '+str(exc)
            raise

    def observe_completed_interval(self, *, command_time_s, post_step_time_s, returned_command):
        """Commit only after the command's actual interval completed unchanged."""
        if self.failure is not None: raise ValueError('Reference capture already failed')
        try:
            if self.pending is None: raise ValueError('No accepted command reference awaiting its physical interval')
            t=_time(command_time_s,'executed command epoch'); end=_time(post_step_time_s,'completed physical epoch')
            force=np.asarray(returned_command)
            if (t != self.pending['command_time_s'] or abs(end-t-DT)>1e-8
                    or force.dtype.str != self.pending['returned_motor_dtype']
                    or force.shape != (61,)
                    or force.tobytes() != np.asarray(self.pending['returned_motor_command'],dtype=force.dtype).tobytes()):
                raise ValueError('Actual submitted command or completed epoch differs from captured return')
            row=self.pending;row['post_step_time_s']=end;row['completed_interval_s']=[t,end]
            self.rows.append(row);self.pending=None;self.completed+=1
        except Exception as exc:
            self.failure=type(exc).__name__+': '+str(exc)
            raise

    def receipt(self):
        return copy.deepcopy(dict(schema=SCHEMA,source_binding=self.binding,
            completed_reference_intervals=self.completed,tail_samples=len(self.rows),tail=list(self.rows),
            pending_unaccepted_command=self.pending,failure=self.failure,static_controller_data=self.static,
            attained_tracking_contract=self.tracking_contract,authorized_stages=0,
            physical_task_qualification=False,controller_state_restoration_supported=False,
            scope='Copied actual controller references after existing force calculations; accepted only after their physical interval. No reference evaluation, filter update, motor change, or state restoration.',
            derivative_scope='Last three accepted positions and stored reference velocities permit explicit finite-difference diagnostics; no unrecorded acceleration state is invented.'))

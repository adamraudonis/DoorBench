"""Privileged, force-only walking/opening -> stow/traversal composition.

The caller owns the active plant, every physics step and its independent audit.
This wrapper only routes numeric measurements and original-capped motor forces.
No failed opening prefix can be promoted by a successful continuation.
"""
from collections import deque
import copy
import hashlib
import json

import numpy as np

from .walking_opening_teacher import WalkingOpeningTeacher
from .post_opening_teacher import PostOpeningTeacher, contact_contract, numeric_mapping, pose_parts


def _detached_numeric(value):
    """Canonical JSON-safe numeric copy, with nonfinite values rejected."""
    def scalar(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError('Expected detached numeric measurements')
    return json.loads(json.dumps(value, default=scalar, sort_keys=True, allow_nan=False))


def handoff_state_sha256(state):
    """Hash exactly the receipt's detached state, independent of dict order."""
    data = json.dumps(_detached_numeric(state), sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


class ContinuousDoorFailure(ValueError):
    """The caller must stop and archive the failed physical episode."""


class ContinuousDoorTeacher:
    """One new instance per complete episode; never resets an active component.

    Opening is independently qualified from every preceding actual interval.
    On the first qualified aperture crossing, both components see the same
    actual pose/velocity/clock, and only the continuation's force is returned.
    The unapplied opening force at that boundary is never treated as a delivered
    command. The previous measured motor force seeds the new stance controller.
    """
    def __init__(self, robot_xml, motors, reference, preparation, reset, checkpoint,
                 joint_geometry, *, door_xml, left_targets, release_screen,
                 runtime_screen=None, opening_options=None, prepare_seconds=8.,
                 maximum_seconds=180., handoff_policy='first-crossing-v1'):
        if not np.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError('A finite positive episode bound is required')
        self.walking = WalkingOpeningTeacher(robot_xml, motors, reference, preparation,
            reset, checkpoint, joint_geometry, door_xml=door_xml, left_targets=left_targets,
            release_screen=release_screen, runtime_screen=runtime_screen,
            opening_options=opening_options, prepare_seconds=prepare_seconds)
        self.post = PostOpeningTeacher(robot_xml, motors, reset, checkpoint,
            door_xml=door_xml, stow_profile='sequential-v2', phase_seconds=5.,
            inward_roll=.07, passage=True)
        self._initialize(motors,reset,maximum_seconds,handoff_policy)

    @classmethod
    def from_components(cls,walking,post,motors,reset,*,maximum_seconds=180.,
                        handoff_policy='first-crossing-v1'):
        """Bind already constructed, unused components with the same contract."""
        if walking.acquisition_started is not None or post.last_time is not None:
            raise ValueError('Composition requires unused components, not resumed controller state')
        result=cls.__new__(cls);result.walking=walking;result.post=post
        result._initialize(motors,reset,maximum_seconds,handoff_policy)
        return result

    def _initialize(self,motors,reset,maximum_seconds,handoff_policy):
        if not np.isfinite(maximum_seconds) or maximum_seconds<=0:
            raise ValueError('A finite positive episode bound is required')
        if handoff_policy not in ('first-crossing-v1','loaded-hold-v2'):
            raise ValueError('Unknown explicit opening handoff policy')
        self.handoff_policy=handoff_policy
        self.names = tuple(self.post.names)
        self.motor_names = tuple(a['name'] for a in motors['actuators'])
        self.caps = np.asarray(self.post.caps).copy()
        if (tuple(self.walking.names) != self.names or
                tuple(self.post.motor_names) != self.motor_names or
                not np.array_equal(self.walking.caps, self.caps) or
                self.caps.shape != (61, 2) or len(set(self.motor_names)) != 61 or
                len(self.names) != 69 or len(set(self.names)) != 69):
            raise ValueError('Both components must share the exact ordered 61-motor contract')
        self.dt = float(self.walking.opening.physics_dt)
        self.hold_seconds = float(self.walking.opening.qualification_seconds)
        self.target_aperture = float(self.walking.opening.target_aperture)
        if self.dt != .002 or self.hold_seconds < .5 or self.target_aperture < 1.2:
            raise ValueError('Use the qualified 2ms clock, >=.5s contact holds and >=1.2rad aperture')
        self.maximum_seconds = float(maximum_seconds)
        self.goal_xy = np.asarray(reset['goal_xy'], float)
        if self.goal_xy.shape != (2,) or not np.isfinite(self.goal_xy).all():
            raise ValueError('An explicit finite approach goal is required')
        self.last_time = self.last_force = self.initial = self.handoff = self.failure = None
        self._opening_audit = None
        self.history = deque()
        self.quiet = deque()
        self.maximum_foot_lift = np.zeros(2)
        self.preparation_samples = 0
        self.milestones = {}
        self.phase = 'walking and opening'
        self.completed = False
        self.completed_time_s = None
        self.maximum_motor_delivery_error = 0.
        self.info = dict(phase=self.phase, completed=False, privileged_teacher=True)

    @property
    def opening(self):
        """Opening component, for runner diagnostics; never reset it in an episode."""
        return self.walking.opening

    @property
    def done(self):
        return self.completed

    @property
    def blocked_reason(self):
        return self.failure['reason'] if self.failure is not None else self.walking.blocked_reason

    @property
    def opening_audit(self):
        """Detached crossing receipt; later intentional release cannot rewrite it.

        These are composition checks over caller-audited physical intervals.
        The runner must additionally freeze its independent opening audit here.
        A later failure leaves this prefix receipt intact but clears ``done``.
        """
        return copy.deepcopy(self._opening_audit)

    @property
    def handoffs(self):
        """Detached chronology; opening event timestamps are global seconds."""
        return copy.deepcopy(dict(walking=self.walking.handoffs, opening=self.milestones,
            continuation_time_s=None if self.handoff is None else self.handoff['time_s'],
            completed_time_s=self.completed_time_s))

    def _fail(self, reason, t):
        if self.failure is None:
            try:
                failure_time = float(t)
                if not np.isfinite(failure_time):
                    failure_time = None
            except (TypeError, ValueError):
                failure_time = None
            self.failure = dict(reason=str(reason), time_s=failure_time)
        self.completed = False
        self.phase = 'failed'
        self.info = dict(phase=self.phase, completed=False, failure=copy.deepcopy(self.failure),
                         privileged_teacher=True, runtime_pose_writes=0, native_mirror_steps=0)
        raise ContinuousDoorFailure(str(reason))

    def _force_array(self, values, label, *, tolerance=0.):
        value = np.asarray(values, float)
        if (value.shape != (61,) or not np.isfinite(value).all() or
                np.any(value < self.caps[:, 0]-tolerance) or np.any(value > self.caps[:, 1]+tolerance)):
            raise ValueError(label+' must be 61 finite original-capped motor forces')
        return value.copy()

    def _held(self, predicate):
        return bool(len(self.history) == round(self.hold_seconds/self.dt)+1 and
            self.history[-1]['time_s']-self.history[0]['time_s'] >= self.hold_seconds-1e-7 and
            all(predicate(row) for row in self.history))

    def _milestones(self, t, info):
        offset = self.walking.acquisition_started
        declarations = info.get('handoffs', {})
        if not declarations:
            return
        if offset is None or not np.isfinite(offset):
            raise ValueError('Opening events require the actual acquisition clock offset')
        ordered = ('qualified_grasp', 'latch_released', 'left_approach',
                   'right_release', 'right_clearance', 'panel_continuation')
        predicates = {
            'qualified_grasp': lambda: self._held(lambda r:r['grasp']),
            'latch_released': lambda: self.history[-1]['operator'] >= .8 and self.history[-1]['latch'] >= .011,
            'left_approach': lambda: self._held(lambda r:r['grasp'] and .075 <= r['leaf'] <= .10),
            'right_release': lambda: self._held(lambda r:r['grasp'] and r['panel_load'] >= 2.),
            'right_clearance': lambda: self.history[-1]['right_gap'] >= .02,
            'panel_continuation': lambda: True,
        }
        for index, key in enumerate(ordered):
            if key not in declarations:
                continue
            actual = float(offset)+float(declarations[key])
            if not np.isfinite(actual):
                raise ValueError('Nonfinite opening event clock')
            if key in self.milestones:
                if abs(actual-self.milestones[key]) > 1e-7:
                    raise ValueError('An established opening event was rewritten')
                continue
            if abs(actual-t) > 1e-7:
                raise ValueError('Opening event was not first observed at its actual decision time')
            if any(previous not in self.milestones for previous in ordered[:index]):
                raise ValueError('Opening event sequence skipped a required predecessor')
            if not predicates[key]():
                raise ValueError('Independent actual-history qualification failed: '+key)
            self.milestones[key] = actual

    def _qualify_crossing(self, t, angles, evidence, continuation, info):
        opening = self.walking.opening
        crossing = info.get('aperture_crossing')
        offset = self.walking.acquisition_started
        if not isinstance(crossing, dict) or offset is None:
            raise ValueError('Aperture reached without an actual qualified opening crossing')
        first_crossing=copy.deepcopy(crossing)
        loaded_policy=getattr(self,'handoff_policy','first-crossing-v1')=='loaded-hold-v2'
        if loaded_policy:
            # Retain the original crossing receipt, including a failed hold.
            # This separately named event uses the current independent history;
            # it does not rewrite the first crossing or forgive bad physics.
            if float(crossing['time_s'])+offset>t+1e-7:
                raise ValueError('The first crossing cannot come from the future')
            crossing=dict(time_s=float(t-offset),aperture_rad=float(angles['leaf']),
                palm_load_N=float(evidence['left_palm_load_N']),
                final_palm_hold=self._held(lambda r:r['palm_load']>=2.),
                all_physics_qualified=opening.all_physics_qualified,
                all_pad_patches_qualified=opening.all_pad_patches_qualified,
                right_release_complete=bool(info.get('release',{}).get('release_fraction',0.)>=.999
                    and evidence['right_lever_clearance_m']>=.02))
        for key in ('final_palm_hold', 'all_physics_qualified', 'all_pad_patches_qualified', 'right_release_complete'):
            if not isinstance(crossing.get(key), (bool, np.bool_)) or not crossing[key]:
                raise ValueError('Opening crossing failed '+key)
        if (not opening.initial_contract or not opening.all_physics_qualified or
                not opening.all_pad_patches_qualified):
            raise ValueError('The opening component retained an invalid prefix')
        if (abs(float(crossing['time_s'])+offset-t) > 1e-7 or
                abs(float(crossing['aperture_rad'])-angles['leaf']) > 1e-8):
            raise ValueError('Crossing belongs to a stale pose or different clock')
        required = ('qualified_grasp','latch_released','left_approach','right_release','right_clearance','panel_continuation')
        if not all(k in self.milestones for k in required):
            raise ValueError('Incomplete independently qualified opening history')
        if (not self._held(lambda r:r['palm_load'] >= 2.) or evidence['left_palm_load_N'] < 2. or
                evidence['right_lever_clearance_m'] < .02 or continuation['right_environment_contacts'] != 0 or
                info.get('release', {}).get('release_fraction', 0.) < .999):
            raise ValueError('Aperture lacks sustained left palm support or complete actual right release')
        screen = self.walking.readiness_screen
        if (not isinstance(screen, dict) or screen.get('passed') is not True or
                not self.preparation_samples or not np.all(self.maximum_foot_lift > .015) or
                self.walking.blocked_reason is not None):
            raise ValueError('Physical walking/preparation prefix was not qualified')
        acquisition = self.walking.handoffs.get('acquisition', {})
        if acquisition.get('hand_contact_count') != 0:
            raise ValueError('Acquisition did not follow contact-free preparation')
        result=dict(time_s=float(t), aperture_rad=float(angles['leaf']),
                    opening_clock_offset_s=float(offset), opening_crossing=copy.deepcopy(crossing),
                    independently_qualified_events=copy.deepcopy(self.milestones),
                    maximum_actual_foot_lift_m=self.maximum_foot_lift.tolist(),
                    preparation_samples=self.preparation_samples,
                    opening_checks=dict(closed_contact_free_separated_start=True,
                        actual_both_foot_lift=True, contact_free_preparation=True,
                        actual_readiness_screen=True, ordered_measured_opening_events=True,
                        qualified_grasp_before_operation=True, actual_latch_release=True,
                        sustained_grasp_at_partial_aperture=True,
                        sustained_grasp_and_left_panel_support_before_right_release=True,
                        right_release_and_clearance=True, sustained_final_palm_support=True,
                        first_actual_aperture_crossing=True, all_prefix_physics_qualified=True,
                        all_prefix_right_pad_patches_valid=True))
        if loaded_policy:
            result['opening_checks'].pop('first_actual_aperture_crossing')
            result['opening_checks']['first_qualified_loaded_aperture_handoff']=True
            result['first_aperture_crossing']=first_crossing
            result['handoff_policy']='loaded-hold-v2'
        return result

    def _finish(self, t, root, feet, evidence, info):
        eligible = (info.get('passage_completed') is True and info['minimum_body_y_m'] > .2 and
                    np.linalg.norm(root[7:9]) < .03 and min(feet) >= 10. and
                    evidence['left_hand_contacts'] == 0 and evidence['left_hand_load_N'] < .1 and
                    evidence['right_environment_contacts'] == 0)
        if not eligible:
            self.quiet.clear()
            self.completed = False
            return
        self.quiet.append((float(t), root[:2].copy()))
        while self.quiet and self.quiet[0][0] < t-1.-1e-8:
            self.quiet.popleft()
        enough = len(self.quiet) == round(1./self.dt)+1 and t-self.quiet[0][0] >= 1.-1e-7
        excursion = max(float(np.linalg.norm(q-self.quiet[0][1])) for _,q in self.quiet)
        self.completed = bool(enough and excursion < .03)
        if self.completed and self.completed_time_s is None:
            self.completed_time_s = float(t)

    def force(self, t, root, joints, velocities, foot_loads, handle_pose, leaf_pose,
              angles, hand_forces, *, evidence, right_palm_pose, pose_time_s,
              contact_interval_s, body_poses, door_velocities, continuation_evidence,
              applied_motor_forces, release_normal_world=None):
        """Consume actual numeric state once per2ms, return the next61 forces.

        ``right_palm_pose`` is the actual touch-SITE pose used by opening.
        ``body_poses`` contains ankle/palm BODY origins, leaf and leaf_handle.
        Root13 is world xyz,wxyz,world linear/angular velocity. Native door
        velocity names are leaf_hinge,leaf_handle_hinge,leaf_latch_bolt_slide.
        Contact evidence and hand forces describe [t-.002,t], clipped at reset.
        Previous actual motor forces must match this wrapper's last command.
        """
        if self.failure is not None:
            self._fail('Failed episodes cannot resume or reset the controller', t)
        try:
            return self._advance(t, root, joints, velocities, foot_loads, handle_pose,
                leaf_pose, angles, hand_forces, evidence=evidence, right_palm_pose=right_palm_pose,
                pose_time_s=pose_time_s, contact_interval_s=contact_interval_s, body_poses=body_poses,
                door_velocities=door_velocities, continuation_evidence=continuation_evidence,
                applied_motor_forces=applied_motor_forces, release_normal_world=release_normal_world)
        except Exception as exc:
            self._fail(str(exc), t)

    def _advance(self, t, root, joints, velocities, foot_loads, handle_pose, leaf_pose,
                 angles, hand_forces, *, evidence, right_palm_pose, pose_time_s,
                 contact_interval_s, body_poses, door_velocities, continuation_evidence,
                 applied_motor_forces, release_normal_world):
        _, feet = contact_contract(t, pose_time_s, contact_interval_s, continuation_evidence,
                                  foot_loads, self.last_time, self.dt)
        if (self.last_time is None and abs(t) > 1e-8) or t > self.maximum_seconds:
            raise ValueError('Continuous episode must start at0 and finish within its declared bound')
        root = np.asarray(root, float)
        if root.shape != (13,) or not np.isfinite(root).all():
            raise ValueError('Complete actual root pose and world velocities are required')
        pose_parts(root[:7])
        numeric_mapping(joints, self.names, 'joint positions')
        numeric_mapping(velocities, self.names, 'joint velocities')
        for pose in (handle_pose, leaf_pose, right_palm_pose):
            pose_parts(pose)
        if not isinstance(body_poses, dict) or not set(self.post.pose_names) <= set(body_poses):
            raise ValueError('Actual ankle/palm body origins and mechanism body poses required')
        poses = {k:pose_parts(body_poses[k]) for k in self.post.pose_names}
        for name, expected in [('leaf',leaf_pose),('leaf_handle',handle_pose)]:
            pos,rot = pose_parts(expected)
            if np.linalg.norm(poses[name][0]-pos) > 1e-8 or np.linalg.norm(poses[name][1]-rot) > 1e-8:
                raise ValueError('Duplicate actual mechanism pose fields disagree')
        if set(angles) != {'operator','leaf','latch'} or not np.isfinite(list(angles.values())).all():
            raise ValueError('Complete finite actual mechanism angles required')
        door_positions = dict(leaf_hinge=angles['leaf'], leaf_handle_hinge=angles['operator'],
                              leaf_latch_bolt_slide=angles['latch'])
        numeric_mapping(door_velocities, self.post.door_names, 'door velocities')
        opening_keys = {'grasp_qualified','physics_qualified','right_pad_patches_valid','hand_contact_count',
                        'left_panel_load_N','left_palm_load_N','right_lever_clearance_m'}
        if not isinstance(evidence, dict) or set(evidence) != opening_keys:
            raise ValueError('Exact opening evidence contract is required throughout the episode')
        for key in ('grasp_qualified','physics_qualified','right_pad_patches_valid'):
            if not isinstance(evidence[key], (bool,np.bool_)):
                raise ValueError('Opening qualification must be an actual boolean')
        if not evidence['physics_qualified'] or not evidence['right_pad_patches_valid']:
            raise ValueError('Actual physics or right-pad gate failed; continuation cannot redeem it')
        count = evidence['hand_contact_count']
        if not isinstance(count,(int,np.integer)) or isinstance(count,(bool,np.bool_)) or count < 0:
            raise ValueError('Actual nonnegative total hand contact count required')
        loads = [evidence[k] for k in ('left_panel_load_N','left_palm_load_N','right_lever_clearance_m')]
        if not np.isfinite(loads).all() or min(loads[:2]) < 0:
            raise ValueError('Actual finite nonnegative palm/panel loads required')
        delivered = self._force_array(applied_motor_forces, 'Actual preceding motor delivery', tolerance=1e-5)
        if self.last_force is not None:
            error = float(np.max(abs(delivered-self.last_force)))
            self.maximum_motor_delivery_error = max(self.maximum_motor_delivery_error,error)
            if error > 1e-5:
                raise ValueError('Actual motor delivery differs from preceding wrapper command')
        foot_positions = np.array([poses[n][0] for n in self.post.feet])
        if self.initial is None:
            separation = float(np.linalg.norm(root[:2]-self.goal_xy))
            if separation < .5 or abs(angles['leaf']) > .001 or abs(angles['operator']) > .001 or count:
                raise ValueError('Start requires >=.5m separation, closed/resting door and contact-free hands')
            self.initial = dict(time_s=float(t), root=root.tolist(), foot_positions=foot_positions.copy(),
                                separated_distance_m=separation, applied_motor_forces=delivered.tolist())
        self.maximum_foot_lift = np.maximum(self.maximum_foot_lift,foot_positions[:,2]-self.initial['foot_positions'][:,2])
        self.history.append(dict(time_s=float(t),grasp=bool(evidence['grasp_qualified']),
            panel_load=float(loads[0]),palm_load=float(loads[1]),right_gap=float(loads[2]),
            leaf=float(angles['leaf']),operator=float(angles['operator']),latch=float(angles['latch'])))
        while self.history and self.history[0]['time_s'] < t-self.hold_seconds-1e-8:
            self.history.popleft()
        if self.handoff is None:
            force, info = self.walking.force(t,root,joints,velocities,feet,handle_pose,leaf_pose,
                angles,hand_forces,evidence=evidence,right_palm_pose=right_palm_pose,
                pose_time_s=pose_time_s,contact_interval_s=contact_interval_s)
            if self.walking.blocked_reason is not None or info.get('body_stance_solver_failures') != 0:
                raise ValueError('Walking/preparation stance or geometry qualification failed')
            if info.get('phase') == 'arm preparation':
                self.preparation_samples += 1
                if count:
                    raise ValueError('Actual hand contact occurred during preparation')
            self._milestones(t, info)
            ready_load=(getattr(self,'handoff_policy','first-crossing-v1')!='loaded-hold-v2'
                        or self._held(lambda r:r['palm_load']>=2.))
            if angles['leaf'] >= self.target_aperture and ready_load:
                handoff = self._qualify_crossing(t,angles,evidence,continuation_evidence,info)
                state = _detached_numeric(dict(time_s=t, pose_time_s=pose_time_s,
                    contact_interval_s=contact_interval_s, root=root, joint_positions=joints,
                    joint_velocities=velocities, foot_loads_N=feet,
                    body_poses={name:body_poses[name] for name in self.post.pose_names},
                    right_palm_site_pose=right_palm_pose, hand_forces_world=hand_forces,
                    door_positions=door_positions, door_velocities=door_velocities,
                    applied_motor_forces=delivered, release_normal_world=release_normal_world,
                    opening_evidence=evidence, continuation_evidence=continuation_evidence))
                handoff['state'] = state
                handoff['handoff_state_sha256'] = handoff_state_sha256(state)
                # Freeze the opening before continuation initialization. A valid
                # opening still deserves its receipt if the new stow plan fails.
                self._opening_audit = copy.deepcopy(dict(passed=True, time_s=handoff['time_s'],
                    checks=handoff['opening_checks'], crossing=handoff['opening_crossing'],
                    independently_qualified_events=handoff['independently_qualified_events'],
                    opening_clock_offset_s=handoff['opening_clock_offset_s'],
                    handoff_state_sha256=handoff['handoff_state_sha256'],
                    state=state,
                    measured_evidence=_detached_numeric(evidence),
                    qualification_window=_detached_numeric(list(self.history)),
                    scope='Composition checks over actual caller-audited intervals; independent physics archive audit required'))
                if self.handoff_policy=='loaded-hold-v2':
                    self._opening_audit.update(handoff_policy=self.handoff_policy,
                        first_aperture_crossing=copy.deepcopy(handoff['first_aperture_crossing']))
                if release_normal_world is None:
                    raise ValueError('An actual outward left-panel normal is required at crossing')
                force, post_info = self.post.force(t,root,joints,velocities,feet,hand_forces,
                    door_positions=door_positions,door_velocities=door_velocities,body_poses=body_poses,
                    evidence=continuation_evidence,pose_time_s=pose_time_s,contact_interval_s=contact_interval_s,
                    previous_motor_forces=delivered,release_normal_world=release_normal_world)
                handoff.update(root=root.tolist(),joint_positions=dict(joints),joint_velocities=dict(velocities),
                    actual_prior_motor_forces=delivered.tolist(),contact_interval_s=list(contact_interval_s),
                    opening_info=copy.deepcopy(info),post_initialization=copy.deepcopy(self.post.initialization))
                self.handoff = handoff
                self.phase = 'post-opening continuation'
                info = post_info
        else:
            force, info = self.post.force(t,root,joints,velocities,feet,hand_forces,
                door_positions=door_positions,door_velocities=door_velocities,body_poses=body_poses,
                evidence=continuation_evidence,pose_time_s=pose_time_s,contact_interval_s=contact_interval_s)
        if self.handoff is not None:
            if info.get('stance_solver_failures') != 0:
                raise ValueError('Post-opening stance solver failed')
            self._finish(t,root,feet,continuation_evidence,info)
        self.last_force = self._force_array(force,'Controller command')
        self.last_time = float(t)
        self.info = dict(phase=self.phase,component=copy.deepcopy(info),completed=self.completed,
            completed_time_s=self.completed_time_s,handoff_time_s=None if self.handoff is None else self.handoff['time_s'],
            maximum_actual_motor_delivery_error_Nm=self.maximum_motor_delivery_error,
            privileged_teacher=True, runtime_pose_writes=0, native_mirror_steps=0,
            scope='Continuous controller candidate; physical episode and independent archive audit still required')
        return self.last_force.copy(),copy.deepcopy(self.info)

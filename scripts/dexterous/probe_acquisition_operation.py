#!/usr/bin/env python3
"""Continuous privileged native grasp acquisition, lever operation and 0.08 rad opening.

Only the initial reset writes physical state. Subsequent goals modify the
acquisition teacher's analytic reference, and only capped motor forces reach the
plant. This diagnostic does not establish traversal or sensor-only control.
"""
import argparse
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.acquisition_teacher import AcquisitionTeacher
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import native_grasp_sample, audited_native_step, audit_grasp_steps
from doorbench.dexterous.provenance import capture


def validate_jev_progress_arguments(args):
    if not np.isfinite(args.jev_sample_period) or not .05<=args.jev_sample_period<=10.:
        raise ValueError('Jev sample period must be 0.05..10 wall-clock seconds')
    if args.jev_progress_plan is not None:
        if not args.portable_wrapper or not args.record_transitions or args.standing_transfer_path:
            raise ValueError('Jev progress requires standalone portable operation and full transition recording')
    elif args.jev_sample_period!=.2:
        raise ValueError('An explicit Jev sample period requires --jev-progress-plan')


def native_progress_snapshot(*,episode_id,sample_id,now_s,phase,physics_row,stance_status,
                             simulation_time_s,physics_dt,prior_loads=None):
    """Copy existing measured native evidence; never query or step the plant.

    Contact values are privileged digit/lever normal-force sums. They are not
    camera inference or finite tactile sensors. Balance combines the current
    measured posture with the preceding completed stance solve; that solve's
    status is not re-described as a fresh solution at this state.
    """
    from doorbench.dexterous.jev_progress_advisor import DIGITS,ProgressSnapshot
    row=physics_row;pad=row['pad_grasp']
    values=tuple(float(pad['digit_forces_N'][digit]) for digit in DIGITS)
    time_aligned=abs(float(row['sim_time_s'])-simulation_time_s)<=max(1e-8,physics_dt*1e-5)
    valid=bool(row['finite'] and time_aligned and all(np.isfinite(values)))
    posture=bool(row['root_height_m']>.7 and 0<=row['torso_tilt_deg']<12)
    balance=None if stance_status is None else bool(posture and stance_status in ('solved','solved inaccurate'))
    physical=bool(valid and row['numerical_warnings']==0 and row['native_motor_limits']
        and 0<=row['max_joint_limit_violation_rad']<=.02
        and 0<=row['max_shadow_loopback_violation_rad']<=.02
        and 0<=row['max_nonfoot_penetration_m']<=.003
        and row['external_wrench_max']==0 and row['applied_generalized_force_max']==0)
    grip=bool(pad['valid_pad_grasp'])
    return ProgressSnapshot(episode_id=episode_id,sample_id=sample_id,
        simulation_time_s=float(simulation_time_s),capture_monotonic_s=float(now_s),phase=phase,
        normal_loads_N=values,grip_stable=grip,balance_ready=balance,
        local_continue_allowed=bool(physical and grip and balance is True),
        contact_source='privileged_digit_handle_contact',sensor_valid=valid,
        prior_normal_loads_N=prior_loads)


class NativeJevProgressGate:
    """Optional nonblocking lever-progress experiment and causal JSONL record."""
    def __init__(self,plan,advisor,output,*,sample_period=.2,clock=time.monotonic):
        self.plan,self.advisor,self.output=plan,advisor,Path(output)
        self.sample_period,self.clock=sample_period,clock
        self.last_submit=None;self.prior_loads=None;self.seen=set();self.closed=False
        self.counts=dict(requests_submitted=0,model_replies_consumed=0,
                         continue_intervals=0,pause_intervals=0,astra_requests=0)
        self.latencies=[]
        self.stream=(self.output/'jev-progress.jsonl').open('w',encoding='utf-8')

    def write(self,value):
        self.stream.write(json.dumps(value,allow_nan=False)+'\n')

    def choose(self,**measurements):
        now=self.clock()
        snapshot=native_progress_snapshot(now_s=now,prior_loads=self.prior_loads,**measurements)
        advice=self.advisor.poll(snapshot,self.plan)
        if advice is not None and advice.sample_id not in self.seen:
            self.seen.add(advice.sample_id)
            self.counts['model_replies_consumed']+=1
            if advice.latency_ms is not None:self.latencies.append(advice.latency_ms)
            self.counts['astra_requests']+=int(advice.proposed_action=='request_astra')
            self.write(dict(event='reply_consumed',consumed_sample_id=snapshot.sample_id,
                consumed_simulation_time_s=snapshot.simulation_time_s,advice=advice.to_dict()))
        if self.last_submit is None or now-self.last_submit>=self.sample_period:
            if self.advisor.submit(snapshot,self.plan):
                from dataclasses import asdict
                self.last_submit=now;self.prior_loads=tuple(snapshot.normal_loads_N)
                self.counts['requests_submitted']+=1
                self.write(dict(event='request_submitted',snapshot=asdict(snapshot),
                    preceding_stance_status=measurements['stance_status']))
        allow=bool(advice is not None and advice.advance)
        context=dict(event='controller_submission',sample_id=snapshot.sample_id,
            simulation_time_s=snapshot.simulation_time_s,capture_monotonic_s=now,
            allow_progress=allow,advice_sample_id=None if advice is None else advice.sample_id,
            action='pause_press' if advice is None else advice.action,
            reason='awaiting_advice' if advice is None else advice.reason,
            advice_age_wall_s=None if advice is None else now-advice.capture_monotonic_s,
            advice_age_simulation_s=None if advice is None else snapshot.simulation_time_s-advice.simulation_time_s,
            local_continue_allowed=snapshot.local_continue_allowed,
            actual_grasp_qualified=snapshot.grip_stable,preceding_stance_status=measurements['stance_status'])
        return allow,context

    def record_submission(self,context,info):
        self.counts['continue_intervals' if context['allow_progress'] else 'pause_intervals']+=1
        context=dict(context,progress={key:info[key] for key in ('press_progress_s','opening_progress_s','progress_rate')})
        self.write(context)
        if sum(self.counts[k] for k in ('continue_intervals','pause_intervals'))%100==0:self.stream.flush()

    def summary(self):
        return dict(scope='Live asynchronous Jev decisions gate privileged native lever progression only; no vision, release, traversal or learned-policy claim',
            plan_id=self.plan.plan_id,sample_period_wall_s=self.sample_period,**self.counts,
            latency_ms_mean=float(np.mean(self.latencies)) if self.latencies else None,
            latency_ms_max=max(self.latencies) if self.latencies else None,
            contact_source='privileged_digit_handle_contact',
            balance_source='current measured posture and preceding completed stance-solver status',
            decision_log='jev-progress.jsonl',api_key_saved=False)

    def close(self):
        if self.closed:return
        self.closed=True
        self.advisor.close()
        result=self.summary();self.write(dict(event='closed',**result))
        self.stream.close()
        (self.output/'jev-progress-summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


def smooth(value):
    u = np.clip(value, 0., 1.)
    return u**3*(10+u*(-15+6*u))


class OperationGoals:
    """Bind the actual acquired palm to the measured handle; change goals only."""
    def __init__(self, teacher, model, data, *, press_seconds=4., opening_seconds=3.):
        self.teacher, self.m, self.d = teacher, model, data
        self.hb = model.body('leaf_handle').id
        self.palm = model.site('robot/rh_palm_touch').id
        self.hj = model.joint('leaf_handle_hinge').id
        self.lj = model.joint('leaf_hinge').id
        self.bj = model.joint('leaf_latch_bolt_slide').id
        self.press_seconds, self.opening_seconds = press_seconds, opening_seconds
        self.started = self.open_started = None
        self.info = dict(phase='acquisition')

    def begin(self):
        d = self.d
        self.started = float(d.time)
        rotation = d.xmat[self.hb].reshape(3, 3)
        self.p_relative = rotation.T@(d.site_xpos[self.palm]-d.xpos[self.hb])
        self.r_relative = rotation.T@d.site_xmat[self.palm].reshape(3, 3)
        self.initial_handle = float(d.qpos[self.m.jnt_qposadr[self.hj]])
        # Controller memory reset at a measured-pose handoff; no plant state changes.
        self.teacher.position_integral[:] = 0.
        self.teacher.rotation_integral[:] = 0.

    def update(self):
        if self.started is None:
            return self.info
        m, d = self.m, self.d
        t = float(d.time)
        actual_h = float(d.qpos[m.jnt_qposadr[self.hj]])
        actual_l = float(d.qpos[m.jnt_qposadr[self.lj]])
        actual_bolt = float(d.qpos[m.jnt_qposadr[self.bj]])
        goal_h = self.initial_handle+(.87-self.initial_handle)*smooth((t-self.started)/self.press_seconds)
        if self.open_started is None and t >= self.started+self.press_seconds and actual_h >= .80 and actual_bolt >= .011:
            self.open_started = t
            # Continue from the previous commanded goal. Jumping to the small
            # actual soft-contact deflection here creates a wrist target step.
            self.initial_leaf = self.info.get('goal_leaf_rad', 0.)
        goal_l = 0. if self.open_started is None else self.initial_leaf+(.08-self.initial_leaf)*smooth((t-self.open_started)/self.opening_seconds)
        dh = Rotation.from_rotvec(d.xaxis[self.hj]*(goal_h-actual_h)).as_matrix()
        dl = Rotation.from_rotvec(d.xaxis[self.lj]*(goal_l-actual_l)).as_matrix()
        hpos = d.xanchor[self.hj]+dh@(d.xpos[self.hb]-d.xanchor[self.hj])
        hrot = dh@d.xmat[self.hb].reshape(3, 3)
        hpos = d.xanchor[self.lj]+dl@(hpos-d.xanchor[self.lj])
        hrot = dl@hrot
        # Acquisition is already complete: its terminal Cartesian reference is
        # now the operation goal. The physical hand and door are never posed.
        self.teacher.positions[-1] = hpos+hrot@self.p_relative
        self.teacher.rotations[-1] = hrot@self.r_relative
        self.info = dict(phase='lever_operation' if self.open_started is None else 'partial_opening',
                         operation_start_s=self.started, opening_start_s=self.open_started,
                         goal_handle_rad=float(goal_h), goal_leaf_rad=float(goal_l),
                         actual_handle_rad=actual_h, actual_leaf_rad=actual_l,
                         actual_bolt_m=actual_bolt)
        return self.info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('robot', 'door', 'reference', 'motors', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=22.)
    parser.add_argument('--press-seconds', type=float, default=5.)
    parser.add_argument('--stance-profile',choices=['landed-foot-v1'])
    parser.add_argument('--record-transitions',action='store_true')
    parser.add_argument('--grasp-profile',choices=('distal-pad-v1','volar-phalange-v1'),default='distal-pad-v1',help='Prospectively selected anatomy contract; original distal score is retained')
    parser.add_argument('--grasp-profile-definition',type=Path,help='Frozen source-bound declaration for an opt-in grasp profile')
    parser.add_argument('--portable-wrapper', action='store_true')
    parser.add_argument('--open-on-latch-clear', action='store_true')
    parser.add_argument('--operator-compliance-gain',type=float,default=0.)
    parser.add_argument('--operation-fixed-pad-control',action='store_true')
    parser.add_argument('--operation-pad-control-profile',choices=('commanded-material-v1','actual-material-v1','actual-material-v2','actual-tangent-v1','measured-pressure-v1'),default='commanded-material-v1')
    parser.add_argument('--hold-attained-grasp',action='store_true',help='Capture coupled finger posture after qualified partial opening')
    parser.add_argument('--attained-hold-stage',choices=('acquisition','operator','aperture','opening'),default='opening')
    parser.add_argument('--index-tendon-offset-rad',type=float,default=0.)
    parser.add_argument('--index-proximal-offset-rad',type=float,default=0.)
    parser.add_argument('--pressure-segment',choices=['nearest','distal'],default='nearest')
    parser.add_argument('--standing-withdrawal-path',type=Path,help='Independently screened withdrawal after the qualified standing return')
    parser.add_argument('--standing-return-palm-feedback',action='store_true',help='Experimental bounded privileged palm correction during return')
    parser.add_argument('--standing-return-support-load',type=float,help='Explicit left support target during return, above the original 2 N gate')
    parser.add_argument('--standing-return-hold-finger-posture',action='store_true',help='Experimental attained coupled finger posture with original motor limits')
    parser.add_argument('--standing-return-path',type=Path,help='Source-bound measured-state lever return after the qualified transfer')
    parser.add_argument('--standing-transfer-leaf-relative-arm',action='store_true',help='Experimental panel-relative palm hold: follows leaf swing but resists lever return')
    parser.add_argument('--standing-transfer-handle-relative-arm',action='store_true',help='Experimental bounded handle-relative target compensation during attained-arm transfer')
    parser.add_argument('--standing-transfer-attained-arm',action='store_true',help='Experimental: track screened attained arm joints with measured initial motor preload')
    parser.add_argument('--standing-transfer-no-fixed-pads',action='store_true',help='Ablation: preserve operation digit controller without extra transfer pad tracking')
    parser.add_argument('--standing-transfer-start-seconds',type=float,default=22.)
    parser.add_argument('--standing-transfer-support-load',type=float,help='Prospective LH panel load target above 2 N and at most 8 N; default 4 N, original palm gate and motor caps unchanged')
    parser.add_argument('--standing-transfer-handoff-seconds',type=float,default=0.)
    parser.add_argument('--standing-transfer-hold-route',action='store_true',help='Diagnostic: hold body/left route at its initial pose; never a completed transfer')
    parser.add_argument('--standing-transfer-preload-profile',choices=('maintain','balanced-4n','index-6n'),default='maintain')
    parser.add_argument('--standing-transfer-grasp-shift',type=float,nargs=3,default=(0.,0.,0.))
    parser.add_argument('--standing-transfer-path',type=Path,help='Explicit attained-state screened bimanual transfer after partial opening')
    parser.add_argument('--index-finger-force',type=float,help='Explicit index preload through original motor limits')
    parser.add_argument('--middle-finger-force',type=float,help='Explicit middle-finger preload; original motor caps unchanged')
    parser.add_argument('--grasp-offset-in-handle-m',nargs=3,type=float,default=[0.,0.,0.])
    parser.add_argument('--min-acquisition-seconds', type=float, default=10.6,
                        help='Earliest event-triggered portable transition; 10.6 retains the native comparison protocol')
    parser.add_argument('--operation-opening-trigger-rad',type=float,default=.80,help='Controller transition only; final operator and latch acceptance thresholds remain unchanged')
    parser.add_argument('--operation-leaf-target-rad',type=float,default=.08,help='Commanded partial opening; physical acceptance bounds stay unchanged')
    parser.add_argument('--operation-operator-lead-limit-rad',type=float,help='Bound palm reference around measured handle angle; original release thresholds remain unchanged')
    parser.add_argument('--operation-leaf-lead-limit-rad',type=float,help='Explicit measured-door lead bound for a new standalone opening trial')
    parser.add_argument('--operation-operator-follow-after-leaf-rad',type=float,help='Blend toward the measured handle angle after the leaf clears the latch')
    parser.add_argument('--landed-foot-max-iterations',type=int,default=50000,help='Explicit solver work budget; physical limits and convergence tolerances remain unchanged')
    parser.add_argument('--operation-handle-hub-avoidance',action='store_true',help='Explicit bounded little-finger motor repulsion from the handle hub')
    parser.add_argument('--operation-hub-clearance-m',type=float,default=.004,help='Prospective 4–8 mm hub-avoidance activation; force cap remains 3 N')
    parser.add_argument('--jev-progress-plan',type=Path,help='Opt-in live Jev lever-progress decisions under this explicit AstraPlan; TYPESAFE_API_KEY required')
    parser.add_argument('--jev-sample-period',type=float,default=.2,help='Minimum wall-clock seconds between asynchronous Jev requests; no physics sleeps')
    parser.add_argument('--validate-arguments-only',action='store_true',help='Validate CLI combinations without reading assets or starting physics')
    args = parser.parse_args()
    if args.grasp_profile!='distal-pad-v1' and (args.grasp_profile_definition is None or not args.record_transitions):
        parser.error('An opt-in grasp profile requires a frozen declaration and actual transition recording')
    if args.grasp_profile_definition is not None and args.grasp_profile=='distal-pad-v1':
        parser.error('A grasp-profile declaration requires an explicit opt-in profile')
    try:validate_jev_progress_arguments(args)
    except ValueError as error:parser.error(str(error))
    if args.operation_handle_hub_avoidance and not args.portable_wrapper:parser.error('Hub avoidance requires portable operation wrapper')
    if args.operation_operator_follow_after_leaf_rad is not None and (not .015<=args.operation_operator_follow_after_leaf_rad<=.05 or not args.portable_wrapper or args.standing_transfer_path):parser.error('Operator follow requires standalone operation and .015..0.05 rad')
    lead_transfer_supported = (not args.standing_transfer_path or (
        args.standing_transfer_attained_arm and args.standing_transfer_no_fixed_pads
        and not args.standing_return_path and not args.standing_withdrawal_path))
    if args.operation_operator_lead_limit_rad is not None and (not .01<=args.operation_operator_lead_limit_rad<=.15 or not args.portable_wrapper or not lead_transfer_supported):
        parser.error('Operator lead requires portable opening or attained-arm transfer without fixed pads/return, and .01..0.15 rad')
    if args.operation_leaf_lead_limit_rad is not None and (not .002<=args.operation_leaf_lead_limit_rad<=.03 or not args.portable_wrapper or args.standing_transfer_path):parser.error('Leaf lead bound requires standalone operation and .002..0.03 rad')
    if not .075<=args.operation_leaf_target_rad<=.10:parser.error('Partial opening command must be .075..0.10 rad')
    if args.operation_leaf_target_rad!=.08 and not args.portable_wrapper:parser.error('Explicit leaf target requires portable wrapper')
    if not .70<=args.operation_opening_trigger_rad<=.80 or (not args.portable_wrapper and args.operation_opening_trigger_rad!=.80):parser.error('Opening trigger must be .70.. .80 rad in the portable controller')
    if args.hold_attained_grasp and (not args.portable_wrapper or args.standing_transfer_path):parser.error('Attained hold requires a standalone portable operation trial')
    if not args.portable_wrapper and (args.operation_fixed_pad_control or args.index_proximal_offset_rad or args.index_tendon_offset_rad):
        parser.error('Contact-control options require the portable operation wrapper')
    if args.standing_transfer_handle_relative_arm and args.standing_transfer_leaf_relative_arm:parser.error("Choose one relative arm reference frame")
    if (args.standing_transfer_handle_relative_arm or args.standing_transfer_leaf_relative_arm) and (not args.standing_transfer_path or not args.standing_transfer_attained_arm or not args.standing_transfer_no_fixed_pads or args.standing_return_path):
        parser.error('Handle-relative targets require explicit attained-arm transfer without return')
    if args.standing_withdrawal_path and not args.standing_return_path:parser.error('Withdrawal requires a standing return route')
    if args.standing_transfer_support_load is not None and (not args.standing_transfer_path or not np.isfinite(args.standing_transfer_support_load) or not 2<args.standing_transfer_support_load<=8):
        parser.error('Transfer support load requires an explicit transfer route and a target above 2 N and at most 8 N')
    if (args.standing_return_palm_feedback or args.standing_return_hold_finger_posture or args.standing_return_support_load is not None) and not args.standing_return_path:parser.error('Finger posture continuation requires an explicit return path')
    if not args.standing_transfer_path and (args.standing_return_path or args.standing_transfer_attained_arm or args.standing_transfer_no_fixed_pads or args.standing_transfer_start_seconds!=22. or args.standing_transfer_hold_route or args.standing_transfer_handoff_seconds or args.standing_transfer_preload_profile!='maintain' or any(args.standing_transfer_grasp_shift)):
        parser.error('Transfer options require an explicit transfer route')
    if not np.isfinite([args.seconds,args.press_seconds,args.min_acquisition_seconds]).all() or min(args.seconds,args.press_seconds) <= 0 or args.min_acquisition_seconds < 0:
        parser.error('Use finite positive operation durations')
    if not np.isfinite(args.operation_hub_clearance_m) or not .004<=args.operation_hub_clearance_m<=.008:parser.error('Hub clearance activation must be 4–8 mm')
    if args.validate_arguments_only:
        print(json.dumps(dict(arguments_valid=True,physics_started=False)));return
    jev_plan=jev_client=None
    if args.jev_progress_plan is not None:
        from doorbench.dexterous.jev_advisor import AstraPlan,JevClient
        jev_plan=AstraPlan(**json.loads(args.jev_progress_plan.read_text(encoding='utf-8')))
        jev_client=JevClient()  # Validate the credential before physics; no HTTP request here.
    from doorbench.dexterous.storage_budget import check_storage, check_retained_budget, EVIDENCE_BYTES_PER_SECOND
    storage_admission = check_storage(args.output, seconds_remaining=args.seconds)
    retained_roots=[args.output.parent]
    local_archive=Path.home()/"Desktop/Projects/DoorBench-runs"
    if local_archive.exists():retained_roots.append(local_archive)
    storage_admission.update(check_retained_budget(retained_roots,incoming_bytes=int(args.seconds*EVIDENCE_BYTES_PER_SECOND)))
    if args.output.exists():
        raise SystemExit('Use a new output directory')
    if not json.loads((args.reference.parent/'geometry-audit.json').read_text())['passed']:
        raise ValueError('Unscreened acquisition route')
    ref, motors = json.loads(args.reference.read_text()), json.loads(args.motors.read_text())
    if args.grasp_profile_definition is not None:
        declaration=json.loads(args.grasp_profile_definition.read_text())
        robot_sha=hashlib.sha256(args.robot.read_bytes()).hexdigest()
        if declaration.get('profile')!=args.grasp_profile or declaration.get('robot_xml_sha256')!=robot_sha or motors.get('source_xml_sha256')!=robot_sha:
            raise ValueError('Grasp declaration, robot and motor contract must identify the same embodiment')
    # Capture the actual imported controller tree even while this driver lives
    # in an isolated worktree, and retain the executed driver as an override.
    controller_root = Path(inspect.getfile(AcquisitionTeacher)).resolve().parents[2]
    capture(controller_root, args.output, {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    from doorbench.dexterous.controller_input_snapshot import snapshot_controller_inputs
    snapshot_controller_inputs([args.reference,args.motors,args.reference.parent/'geometry-audit.json',
        args.standing_transfer_path,args.standing_return_path,args.standing_withdrawal_path,args.jev_progress_plan,args.grasp_profile_definition],
        args.output/'controller-inputs')
    if args.grasp_profile_definition is not None:
        shutil.copy2(args.grasp_profile_definition,args.output/'grasp-profile-definition.json')
    (args.output/'storage-admission.json').write_text(json.dumps(storage_admission,indent=2)+'\n')
    (args.output/'run.pid').write_text(str(os.getpid()))
    def stage(label):
        temporary=args.output/'pipeline.json.tmp'
        temporary.write_text(json.dumps(dict(stage=label,report_file='report.json',scope=__doc__)))
        temporary.replace(args.output/'pipeline.json')
    stage('Native physics and full handle verification')
    shutil.copy2(__file__, args.output/'diagnostic-source.py')
    (args.output/'source-override.json').write_text(json.dumps(dict(entry_point='diagnostic-source.py',
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),controller_root=str(controller_root),scope=__doc__),indent=2)+'\n')
    shutil.copy2(args.reference, args.output/'reference.json')
    sim = DexterousDoorEnv(args.door, args.robot, json.loads(args.robot.with_suffix('.audit.json').read_text()))
    m, d = sim.m, sim.d
    hub_geometry=None
    if args.operation_handle_hub_avoidance:
        g=m.geom('leaf_handle_hub_col_n').id
        if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_CYLINDER or m.body(m.geom_bodyid[g]).name!='leaf_handle':raise ValueError('Expected original handle-body cylinder hub')
        hub_geometry=dict(size=m.geom_size[g].tolist(),position=m.geom_pos[g].tolist(),quaternion_wxyz=m.geom_quat[g].tolist())
        (args.output/'hub-geometry.json').write_text(json.dumps(hub_geometry,indent=2)+'\n')
    teacher = AcquisitionTeacher(args.robot, motors, ref,handle_hub_geometry=hub_geometry,landed_foot_max_iterations=args.landed_foot_max_iterations,stance_profile=args.stance_profile,pressure_segment=args.pressure_segment,middle_finger_force=args.middle_finger_force,index_finger_force=args.index_finger_force)
    sim.reset(randomize=False, images=False)
    d.qpos[sim.root_qadr:sim.root_qadr+7] = teacher.initial_root
    ids = np.array([m.joint('robot/'+n).id for n in teacher.names])
    qa, va = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
    d.qpos[qa] = teacher.path[0]
    d.qvel[:] = 0.
    mujoco.mj_forward(m, d)
    aids = np.array([m.actuator('robot/'+v['name']).id for v in motors['actuators']])
    m.actuator_gainprm[aids, 0] = 1.
    m.actuator_biasprm[aids, :3] = 0.
    m.actuator_ctrlrange[aids] = teacher.caps
    d.ctrl[aids] = 0.
    hj, lj, bj = [m.joint(n).id for n in ('leaf_handle_hinge','leaf_hinge','leaf_latch_bolt_slide')]
    hb, lb, palm = m.body('leaf_handle').id, m.body('leaf').id, m.site('robot/rh_palm_touch').id
    if args.portable_wrapper:
        from doorbench.dexterous.operation_teacher import DoorOperationTeacher
        operation = DoorOperationTeacher(teacher,dict(operator_origin=m.jnt_pos[hj],operator_axis=m.jnt_axis[hj],
            leaf_origin=m.jnt_pos[lj],leaf_axis=m.jnt_axis[lj]),release_operator_threshold=args.operation_opening_trigger_rad,handle_hub_avoidance=args.operation_handle_hub_avoidance,hub_clearance_m=args.operation_hub_clearance_m,leaf_target=args.operation_leaf_target_rad,leaf_lead_limit_rad=args.operation_leaf_lead_limit_rad,operator_lead_limit_rad=args.operation_operator_lead_limit_rad,operator_follow_after_leaf_rad=args.operation_operator_follow_after_leaf_rad,min_acquisition_seconds=args.min_acquisition_seconds,press_seconds=args.press_seconds,wait_for_press_completion=not args.open_on_latch_clear,operator_compliance_gain=args.operator_compliance_gain,grasp_offset_in_handle_m=args.grasp_offset_in_handle_m,index_proximal_offset_rad=args.index_proximal_offset_rad,index_tendon_offset_rad=args.index_tendon_offset_rad,fixed_pad_control=args.operation_fixed_pad_control,pad_control_profile=args.operation_pad_control_profile,hold_attained_grasp=args.hold_attained_grasp,attained_hold_stage=args.attained_hold_stage)
        wrapper_source = Path(inspect.getfile(DoorOperationTeacher))
        shutil.copy2(wrapper_source,args.output/'operation-teacher-source.py')
        (args.output/'operation-teacher-source.json').write_text(json.dumps(dict(
            source_path=str(wrapper_source),sha256=hashlib.sha256(wrapper_source.read_bytes()).hexdigest()),indent=2)+'\n')
    else:
        operation = OperationGoals(teacher, m, d, press_seconds=args.press_seconds)
    transfer=None
    if args.standing_transfer_path:
        if not args.portable_wrapper or not args.record_transitions:raise ValueError('Standing transfer requires portable operation and full physical evidence')
        from doorbench.dexterous.standing_transfer import StandingTransferTeacher
        transfer=StandingTransferTeacher(operation,motors,args.standing_transfer_path,start_seconds=args.standing_transfer_start_seconds,fixed_pad_tracking=not args.standing_transfer_no_fixed_pads,attained_arm_tracking=args.standing_transfer_attained_arm,preload_profile=args.standing_transfer_preload_profile,grasp_shift=args.standing_transfer_grasp_shift,hold_route=args.standing_transfer_hold_route,handoff_seconds=args.standing_transfer_handoff_seconds,handle_relative_arm=args.standing_transfer_handle_relative_arm,leaf_relative_arm=args.standing_transfer_leaf_relative_arm,support_load_target=4. if args.standing_transfer_support_load is None else args.standing_transfer_support_load)
    hand_names = {b:m.body(b).name.removeprefix('robot/') for b in range(m.nbody)
                  if m.body(b).name.startswith(('robot/rh_', 'robot/lh_'))}
    from doorbench.dexterous.bounded_evidence import BoundedEvidence
    from itertools import islice
    physics = BoundedEvidence(args.output/'physics-chunks')
    physics.append(native_grasp_sample(sim, 'leaf_handle_lever_col_n', handle_joint='leaf_handle_hinge',profile=args.grasp_profile))
    states = {key:[] for key in ('qpos', 'qvel', 'ctrl')}
    if args.standing_return_path:
        from doorbench.dexterous.standing_return import StandingReturnTeacher
        transfer=StandingReturnTeacher(transfer,motors,args.standing_return_path,hold_finger_posture=args.standing_return_hold_finger_posture,support_load_target=args.standing_return_support_load,palm_feedback=args.standing_return_palm_feedback)
    withdrawal_pairs=None
    if args.standing_withdrawal_path:
        from doorbench.dexterous.standing_withdrawal import StandingWithdrawalTeacher
        from doorbench.dexterous.standing_withdrawal_audit import clearance_pairs,environment_clearance,withdrawal_checks
        transfer=StandingWithdrawalTeacher(transfer,motors,args.standing_withdrawal_path)
        withdrawal_pairs=clearance_pairs(m)
    traces = [];recorder=archive=None;controller_steps=BoundedEvidence(args.output/'controller-chunks')
    if args.record_transitions:
        from doorbench.dexterous.native_transition_audit import NativeTransitionRecorder
        from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
        recorder=NativeTransitionRecorder(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge',profile=args.grasp_profile);archive=NativeTransitionArchive(args.output/'raw-transitions')
    jev_gate=None
    try:
        if jev_plan is not None:
            from doorbench.dexterous.jev_progress_advisor import JevProgressAdvisor,AsyncJevProgressAdvisor
            from dataclasses import asdict
            advisor=AsyncJevProgressAdvisor(JevProgressAdvisor(jev_client))
            jev_gate=NativeJevProgressGate(jev_plan,advisor,args.output,sample_period=args.jev_sample_period)
            (args.output/'jev-progress-plan.json').write_text(json.dumps(dict(plan=asdict(jev_plan),
                plan_sha256=hashlib.sha256(args.jev_progress_plan.read_bytes()).hexdigest(),
                sample_period_wall_s=args.jev_sample_period,physics_dt_s=float(m.opt.timestep),
                phase='lever_operation',default_other_phase_progress=True,api_key_saved=False),indent=2)+'\n')
        controller_error=None
        try:
            for step in range(round(args.seconds/m.opt.timestep)):
                if step % 500 == 0:
                    check_storage(args.output, seconds_remaining=max(0.,args.seconds-float(d.time)))
                if not args.portable_wrapper and operation.started is None and d.time >= 10.6-1e-8:
                    tail = [r for r in physics if r['sim_time_s'] >= d.time-.5-1e-8]
                    if len(tail) >= 250 and all(r['pad_grasp']['valid_pad_grasp'] for r in tail) and teacher.info.get('path_fraction',0) >= .999:
                        operation.begin()
                goal_info = operation.info if args.portable_wrapper else operation.update()
                loads = {name:np.zeros(3) for name in hand_names.values()}
                for index, contact in enumerate(d.contact[:d.ncon]):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(m, d, index, wrench)
                    force = contact.frame.reshape(3, 3).T@wrench[:3]
                    for sign, geom in zip((-1,1),contact.geom):
                        body = int(m.geom_bodyid[geom])
                        if body in hand_names:
                            loads[hand_names[body]] += sign*force
                rot = d.xmat[sim.pelvis].reshape(3, 3)
                root = np.r_[d.qpos[sim.root_qadr:sim.root_qadr+7],d.qvel[sim.root_vadr:sim.root_vadr+3],
                             rot@d.qvel[sim.root_vadr+3:sim.root_vadr+6]]
                measured_joints, measured_velocities = dict(zip(teacher.names,d.qpos[qa])),dict(zip(teacher.names,d.qvel[va]))
                progress_options={};jev_context=None
                if jev_gate is not None and operation.started is not None and operation.open_started is None:
                    allow,jev_context=jev_gate.choose(episode_id=str(args.output.resolve()),sample_id=step,
                        phase='lever_operation',physics_row=physics[-1],stance_status=teacher.info.get('stance_status'),
                        simulation_time_s=float(d.time),physics_dt=float(m.opt.timestep))
                    progress_options['allow_progress']=allow
                if args.portable_wrapper:
                    force,info = (transfer or operation).force(float(d.time),root,measured_joints,measured_velocities,
                        np.r_[d.xpos[hb],d.xquat[hb]],np.r_[d.xpos[lb],d.xquat[lb]],
                        dict(operator=d.qpos[m.jnt_qposadr[hj]],leaf=d.qpos[m.jnt_qposadr[lj]],latch=d.qpos[m.jnt_qposadr[bj]]),
                        loads,grasp_qualified=physics[-1]['pad_grasp']['valid_pad_grasp'],**progress_options,**({'left_panel_load':recorder.left_surface['total_normal_load_N']} if transfer else {}))
                    goal_info = transfer.info if transfer else operation.info
                else:
                    force, info = teacher.force(float(d.time),root,measured_joints,measured_velocities,np.r_[d.xpos[hb],d.xquat[hb]],loads)
                d.ctrl[aids] = force
                if jev_context is not None:
                    info=dict(info,jev_progress=jev_context)
                    jev_gate.record_submission(jev_context,info)
                if recorder:
                    recorder.before_step();sim.plant.step();row,raw=recorder.after_step();archive.write(raw);controller_steps.append(dict(time_s=float(d.time)-m.opt.timestep,**info))
                else:row = audited_native_step(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge',profile=args.grasp_profile)
                if transfer:row['left_surface']=recorder.left_surface.copy()
                if withdrawal_pairs is not None and d.time>=transfer.start_time-1e-8:
                    row['right_environment_clearance_m']=environment_clearance(m,d,withdrawal_pairs)
                row['bolt_slide_m'] = float(d.qpos[m.jnt_qposadr[bj]])
                row['operation'] = goal_info
                physics.append(row)
                if step % 10 == 0:
                    trace = dict(**sim.diagnostics(),teacher={**info,'phase':goal_info['phase']},operation=goal_info,
                                 palm_error_m=float(np.linalg.norm(d.site_xpos[palm]-teacher.positions[-1])))
                    traces.append(trace)
                    for key in states:
                        states[key].append(getattr(d,key).copy())
                    (args.output/'latest.json').write_text(json.dumps(trace)+'\n')
                    (args.output/'progress.json').write_text(json.dumps(dict(sim_time_s=float(d.time),expected_duration_s=args.seconds,door_angle_rad=row['door_q'],torso_tilt_deg=row['torso_tilt_deg'])))
                    if step % 500 == 0:
                        print(json.dumps(trace),flush=True)
                if not row['finite'] or row['torso_tilt_deg'] > 35:
                    break
        except Exception as exc:
            import traceback
            controller_error=type(exc).__name__+': '+str(exc)
            (args.output/'error.txt').write_text(traceback.format_exc())
            arm_target=getattr(transfer,'handle_target',None)
            snapshot=getattr(arm_target,'failure_snapshot',None)
            if snapshot is not None:
                (args.output/'relative-arm-failure.json').write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
        if jev_gate is not None:jev_gate.close()
        stage('Reducing recorded physical and contact checks')
        report = audit_grasp_steps(physics,physics_dt=m.opt.timestep,expected_duration=args.seconds)
        report['grasp_profile']=args.grasp_profile
        distal_tail=[r for r in physics if r['sim_time_s']>=args.seconds-.5-1e-8]
        report['original_distal_pad_hold']=bool(len(distal_tail)>=round(.5/m.opt.timestep)+1 and all(r['pad_grasp'].get('distal_pad_grasp',r['pad_grasp'])['valid_pad_grasp'] for r in distal_tail))
        if archive:
            archive.close(complete=controller_error is None)
            controller_steps.export(args.output/'controller-steps.json.gz')
            report['checks']['stance_solves_every_interval']=all(row['stance_status'] in ('solved','solved inaccurate') for row in controller_steps)
            report['checks']['no_warning_intervals']=all(row.get('mujoco_warning_interval',{}).get('passed',False) for row in islice(physics,1,None))
        if controller_error is not None:
            report['checks']['controller_completed']=False
            report['controller_error']=controller_error
        report['checks']['acquisition_precedes_operation'] = operation.started is not None
        report['checks']['operator_driven_to_release'] = max(r['handle_angle_rad'] for r in physics) >= .80 and max(r.get('bolt_slide_m',0) for r in physics) >= .011
        tail = [r for r in physics if r['sim_time_s'] >= args.seconds-.5-1e-8]
        report['checks']['partial_leaf_opening_held'] = bool(tail) and all(.075 <= r['door_q'] <= .10 for r in tail)
        report['checks']['opening_bounded_for_transfer'] = max(r['door_q'] for r in physics) <= .12
        if transfer:
            report['checks']['standing_transfer_started']=transfer.started is not None
            report['checks']['final_left_palm_support']=bool(tail) and all(r.get('left_surface',{}).get('palm_normal_load_N',0)>=2 for r in tail)
            report['standing_transfer']=dict(scope='Privileged physical partial opening and left-palm transfer only; no release/traversal',route=str(args.standing_transfer_path),started_s=transfer.started,final=transfer.info)
        if args.standing_return_path:
            report['checks']['standing_return_started']=transfer.return_started is not None
            report['checks']['operator_returned_to_rest']=bool(tail) and all(abs(r['handle_angle_rad'])<=.05 for r in tail)
            report['checks']['bolt_returned_to_rest']=bool(tail) and all(abs(r.get('bolt_slide_m',1.))<=.001 for r in tail)
            report['standing_return']=dict(route=str(args.standing_return_path),started_s=transfer.return_started,final=transfer.info)
        if args.standing_withdrawal_path:
            report['checks']=withdrawal_checks(report['checks'],physics,dt=m.opt.timestep,duration=args.seconds,started=transfer.started_withdrawal,release_started=transfer.release_started,completed=transfer.info.get('withdrawal_progress',0)>=.999)
            report['standing_withdrawal']=dict(route=str(args.standing_withdrawal_path),started_s=transfer.started_withdrawal,release_started_s=transfer.release_started,final=transfer.info)
            if transfer.panel is not None:
                target=transfer.panel.plan['final_leaf_angle_rad']
                report['checks']['standing_panel_started']=transfer.panel.started is not None
                report['checks']['standing_panel_reference_completed']=transfer.info.get('panel_progress',0)>=.999
                report['checks']['standing_panel_aperture_held']=bool(tail) and all(target-.02<=r['door_q']<=target+.05 for r in tail)
                panel_rows=lambda:(r for r in physics if r['sim_time_s']>=transfer.panel_schedule.panels[0].start_time)
                report['checks']['standing_panel_upright']=any(True for _ in panel_rows()) and all(r['torso_tilt_deg']<=5. for r in panel_rows())
                report['standing_panel']=dict(scope='Privileged upright continuation to screened partial aperture, not traversal',started_s=transfer.panel_schedule.panels[0].started,target_aperture_rad=target,final=transfer.info,completed_segments=transfer.panel_schedule.completed)
                if len(transfer.panel_schedule.panels)>1:report['checks']['all_panel_segments_executed']=len(transfer.panel_schedule.completed)==len(transfer.panel_schedule.panels)-1
        report.update(passed=all(report['checks'].values()),scope=__doc__,
            runtime_robot_pose_writes=0,direct_door_commands=False,
            maximum_handle_rad=max(r['handle_angle_rad'] for r in physics),
            maximum_leaf_rad=max(r['door_q'] for r in physics),final_leaf_rad=physics[-1]['door_q'],
            maximum_bolt_retraction_m=max(r.get('bolt_slide_m',0) for r in physics),
            final_contacts=physics[-1]['pad_grasp'])
        if jev_gate is not None:report['jev_progress_experiment']=jev_gate.summary()
        operation_rows = lambda:(r for r in physics if operation.started is not None and r['sim_time_s'] >= operation.started)
        report['operation_digit_unload_samples'] = sum(not r['pad_grasp']['valid_pad_grasp'] for r in operation_rows())
        report['operation_invalid_pad_patch_samples'] = sum(any(not c['pad_qualified'] for c in r['pad_grasp']['contacts']) for r in operation_rows())
        if operation.started is not None:
            report['operation_reference'] = dict(palm_position_in_handle_m=operation.p_relative.tolist(),
                palm_rotation_in_handle=operation.r_relative.tolist(),operation_start_s=operation.started,
                opening_start_s=operation.open_started,press_seconds=operation.press_seconds,
                opening_seconds=operation.opening_seconds,final_goals=operation.info)
        stage('Exporting lossless step records and trajectory')
        (args.output/'trace.json').write_text(json.dumps(traces)+'\n')
        physics.export(args.output/'physics-steps.json.gz')
        np.savez_compressed(args.output/'trajectory.npz',**states,
                            terminal_qpos=d.qpos.copy(),terminal_qvel=d.qvel.copy(),
                            terminal_ctrl=d.ctrl.copy(),terminal_time_s=float(d.time))
        # Publish completion only after every evidence stream is closed. A
        # reader must never mistake a still-writing gzip archive for a final run.
        whole_handle=dict(passed=False,scope='Complete archived handle contacts required')
        if archive is not None:
            try:
                stage('Auditing the complete handle assembly archive')
                from scripts.dexterous.audit_native_handle_assembly import audit as audit_handle_assembly
                whole_handle=audit_handle_assembly(args.output)
            except Exception as exc:
                whole_handle=dict(passed=False,error=type(exc).__name__+': '+str(exc),scope='Whole-handle audit could not complete')
        (args.output/'whole-handle-audit.json').write_text(json.dumps(whole_handle,indent=2)+'\n')
        report['checks']['whole_handle_assembly']=whole_handle['passed']
        report['passed']=all(report['checks'].values())
        report_tmp=args.output/'report.json.tmp'
        report_tmp.write_text(json.dumps(report,indent=2)+'\n')
        report_tmp.replace(args.output/'report.json')
        print(json.dumps({k:v for k,v in report.items() if k!='final_contacts'}),flush=True)
    finally:
        if jev_gate is not None:jev_gate.close()
        if archive and not archive.closed:archive.close(complete=False)
        sim.close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

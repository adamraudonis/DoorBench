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

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.acquisition_teacher import AcquisitionTeacher
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import native_grasp_sample, audited_native_step, audit_grasp_steps
from doorbench.dexterous.provenance import capture


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
    parser.add_argument('--portable-wrapper', action='store_true')
    parser.add_argument('--open-on-latch-clear', action='store_true')
    parser.add_argument('--operator-compliance-gain',type=float,default=0.)
    parser.add_argument('--operation-fixed-pad-control',action='store_true')
    parser.add_argument('--operation-pad-control-profile',choices=('commanded-material-v1','actual-material-v1','actual-material-v2','measured-pressure-v1'),default='commanded-material-v1')
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
    parser.add_argument('--standing-transfer-attained-arm',action='store_true',help='Experimental: track screened attained arm joints with measured initial motor preload')
    parser.add_argument('--standing-transfer-no-fixed-pads',action='store_true',help='Ablation: preserve operation digit controller without extra transfer pad tracking')
    parser.add_argument('--standing-transfer-start-seconds',type=float,default=22.)
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
    parser.add_argument('--operation-leaf-target-rad',type=float,default=.08,help='Commanded partial opening; physical acceptance bounds stay unchanged')
    parser.add_argument('--operation-leaf-lead-limit-rad',type=float,help='Explicit measured-door lead bound for a new standalone opening trial')
    parser.add_argument('--operation-operator-follow-after-leaf-rad',type=float,help='Blend toward the measured handle angle after the leaf clears the latch')
    parser.add_argument('--landed-foot-max-iterations',type=int,default=50000,help='Explicit solver work budget; physical limits and convergence tolerances remain unchanged')
    parser.add_argument('--operation-handle-hub-avoidance',action='store_true',help='Explicit bounded little-finger motor repulsion from the handle hub')
    args = parser.parse_args()
    if args.operation_handle_hub_avoidance and not args.portable_wrapper:parser.error('Hub avoidance requires portable operation wrapper')
    if args.operation_operator_follow_after_leaf_rad is not None and (not .015<=args.operation_operator_follow_after_leaf_rad<=.05 or not args.portable_wrapper or args.standing_transfer_path or args.hold_attained_grasp):parser.error('Operator follow requires standalone operation, no fixed hold, and .015..0.05 rad')
    if args.operation_leaf_lead_limit_rad is not None and (not .002<=args.operation_leaf_lead_limit_rad<=.03 or not args.portable_wrapper or args.standing_transfer_path):parser.error('Leaf lead bound requires standalone operation and .002..0.03 rad')
    if not .075<=args.operation_leaf_target_rad<=.10:parser.error('Partial opening command must be .075..0.10 rad')
    if args.operation_leaf_target_rad!=.08 and not args.portable_wrapper:parser.error('Explicit leaf target requires portable wrapper')
    if args.hold_attained_grasp and (not args.portable_wrapper or args.standing_transfer_path):parser.error('Attained hold requires a standalone portable operation trial')
    if not args.portable_wrapper and (args.operation_fixed_pad_control or args.index_proximal_offset_rad or args.index_tendon_offset_rad):
        parser.error('Contact-control options require the portable operation wrapper')
    if args.standing_withdrawal_path and not args.standing_return_path:parser.error('Withdrawal requires a standing return route')
    if (args.standing_return_palm_feedback or args.standing_return_hold_finger_posture or args.standing_return_support_load is not None) and not args.standing_return_path:parser.error('Finger posture continuation requires an explicit return path')
    if not args.standing_transfer_path and (args.standing_return_path or args.standing_transfer_attained_arm or args.standing_transfer_no_fixed_pads or args.standing_transfer_start_seconds!=22. or args.standing_transfer_hold_route or args.standing_transfer_handoff_seconds or args.standing_transfer_preload_profile!='maintain' or any(args.standing_transfer_grasp_shift)):
        parser.error('Transfer options require an explicit transfer route')
    if not np.isfinite([args.seconds,args.press_seconds,args.min_acquisition_seconds]).all() or min(args.seconds,args.press_seconds) <= 0 or args.min_acquisition_seconds < 0:
        parser.error('Use finite positive operation durations')
    if args.output.exists():
        raise SystemExit('Use a new output directory')
    if not json.loads((args.reference.parent/'geometry-audit.json').read_text())['passed']:
        raise ValueError('Unscreened acquisition route')
    ref, motors = json.loads(args.reference.read_text()), json.loads(args.motors.read_text())
    # Capture the actual imported controller tree even while this driver lives
    # in an isolated worktree, and retain the executed driver as an override.
    controller_root = Path(inspect.getfile(AcquisitionTeacher)).resolve().parents[2]
    capture(controller_root, args.output, {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    (args.output/'run.pid').write_text(str(os.getpid()))
    (args.output/'pipeline.json').write_text(json.dumps(dict(stage='Native physics and full handle verification',report_file='report.json',scope=__doc__)))
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
            leaf_origin=m.jnt_pos[lj],leaf_axis=m.jnt_axis[lj]),handle_hub_avoidance=args.operation_handle_hub_avoidance,leaf_target=args.operation_leaf_target_rad,leaf_lead_limit_rad=args.operation_leaf_lead_limit_rad,operator_follow_after_leaf_rad=args.operation_operator_follow_after_leaf_rad,min_acquisition_seconds=args.min_acquisition_seconds,press_seconds=args.press_seconds,wait_for_press_completion=not args.open_on_latch_clear,operator_compliance_gain=args.operator_compliance_gain,grasp_offset_in_handle_m=args.grasp_offset_in_handle_m,index_proximal_offset_rad=args.index_proximal_offset_rad,index_tendon_offset_rad=args.index_tendon_offset_rad,fixed_pad_control=args.operation_fixed_pad_control,pad_control_profile=args.operation_pad_control_profile,hold_attained_grasp=args.hold_attained_grasp,attained_hold_stage=args.attained_hold_stage)
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
        transfer=StandingTransferTeacher(operation,motors,args.standing_transfer_path,start_seconds=args.standing_transfer_start_seconds,fixed_pad_tracking=not args.standing_transfer_no_fixed_pads,attained_arm_tracking=args.standing_transfer_attained_arm,preload_profile=args.standing_transfer_preload_profile,grasp_shift=args.standing_transfer_grasp_shift,hold_route=args.standing_transfer_hold_route,handoff_seconds=args.standing_transfer_handoff_seconds)
    hand_names = {b:m.body(b).name.removeprefix('robot/') for b in range(m.nbody)
                  if m.body(b).name.startswith(('robot/rh_', 'robot/lh_'))}
    physics = [native_grasp_sample(sim, 'leaf_handle_lever_col_n', handle_joint='leaf_handle_hinge')]
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
    traces = [];recorder=archive=None;controller_steps=[]
    if args.record_transitions:
        from doorbench.dexterous.native_transition_audit import NativeTransitionRecorder
        from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
        recorder=NativeTransitionRecorder(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge');archive=NativeTransitionArchive(args.output/'raw-transitions')
    try:
        controller_error=None
        try:
            for step in range(round(args.seconds/m.opt.timestep)):
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
                if args.portable_wrapper:
                    force,info = (transfer or operation).force(float(d.time),root,measured_joints,measured_velocities,
                        np.r_[d.xpos[hb],d.xquat[hb]],np.r_[d.xpos[lb],d.xquat[lb]],
                        dict(operator=d.qpos[m.jnt_qposadr[hj]],leaf=d.qpos[m.jnt_qposadr[lj]],latch=d.qpos[m.jnt_qposadr[bj]]),
                        loads,grasp_qualified=physics[-1]['pad_grasp']['valid_pad_grasp'],**({'left_panel_load':recorder.left_surface['total_normal_load_N']} if transfer else {}))
                    goal_info = transfer.info if transfer else operation.info
                else:
                    force, info = teacher.force(float(d.time),root,measured_joints,measured_velocities,np.r_[d.xpos[hb],d.xquat[hb]],loads)
                d.ctrl[aids] = force
                if recorder:
                    recorder.before_step();sim.plant.step();row,raw=recorder.after_step();archive.write(raw);controller_steps.append(dict(time_s=float(d.time)-m.opt.timestep,**info))
                else:row = audited_native_step(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')
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
        report = audit_grasp_steps(physics,physics_dt=m.opt.timestep,expected_duration=args.seconds)
        if archive:
            archive.close(complete=controller_error is None)
            with gzip.open(args.output/'controller-steps.json.gz','wt') as f:json.dump(controller_steps,f)
            report['checks']['stance_solves_every_interval']=all(row['stance_status'] in ('solved','solved inaccurate') for row in controller_steps)
            report['checks']['no_warning_intervals']=all(row.get('mujoco_warning_interval',{}).get('passed',False) for row in physics[1:])
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
                panel_rows=[r for r in physics if r['sim_time_s']>=transfer.panel_schedule.panels[0].start_time]
                report['checks']['standing_panel_upright']=bool(panel_rows) and all(r['torso_tilt_deg']<=5. for r in panel_rows)
                report['standing_panel']=dict(scope='Privileged upright continuation to screened partial aperture, not traversal',started_s=transfer.panel_schedule.panels[0].started,target_aperture_rad=target,final=transfer.info,completed_segments=transfer.panel_schedule.completed)
                if len(transfer.panel_schedule.panels)>1:report['checks']['all_panel_segments_executed']=len(transfer.panel_schedule.completed)==len(transfer.panel_schedule.panels)-1
        report.update(passed=all(report['checks'].values()),scope=__doc__,
            runtime_robot_pose_writes=0,direct_door_commands=False,
            maximum_handle_rad=max(r['handle_angle_rad'] for r in physics),
            maximum_leaf_rad=max(r['door_q'] for r in physics),final_leaf_rad=physics[-1]['door_q'],
            maximum_bolt_retraction_m=max(r.get('bolt_slide_m',0) for r in physics),
            final_contacts=physics[-1]['pad_grasp'])
        operation_rows = [r for r in physics if operation.started is not None and r['sim_time_s'] >= operation.started]
        report['operation_digit_unload_samples'] = sum(not r['pad_grasp']['valid_pad_grasp'] for r in operation_rows)
        report['operation_invalid_pad_patch_samples'] = sum(any(not c['pad_qualified'] for c in r['pad_grasp']['contacts']) for r in operation_rows)
        if operation.started is not None:
            report['operation_reference'] = dict(palm_position_in_handle_m=operation.p_relative.tolist(),
                palm_rotation_in_handle=operation.r_relative.tolist(),operation_start_s=operation.started,
                opening_start_s=operation.open_started,press_seconds=operation.press_seconds,
                opening_seconds=operation.opening_seconds,final_goals=operation.info)
        (args.output/'trace.json').write_text(json.dumps(traces)+'\n')
        with gzip.open(args.output/'physics-steps.json.gz','wt') as stream:
            json.dump(physics,stream)
        np.savez_compressed(args.output/'trajectory.npz',**states,
                            terminal_qpos=d.qpos.copy(),terminal_qvel=d.qvel.copy(),
                            terminal_ctrl=d.ctrl.copy(),terminal_time_s=float(d.time))
        # Publish completion only after every evidence stream is closed. A
        # reader must never mistake a still-writing gzip archive for a final run.
        whole_handle=dict(passed=False,scope='Complete archived handle contacts required')
        if archive is not None:
            try:
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
        if archive and not archive.closed:archive.close(complete=False)
        sim.close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

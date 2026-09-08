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
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.acquisition_teacher import AcquisitionTeacher
from doorbench.dexterous.bimanual_transfer import LeftPalmContact, load_screen_targets
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
        self.hold_h=self.hold_l=None
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
        if self.hold_h is not None:goal_h=self.hold_h;goal_l=self.hold_l
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
    parser.add_argument('--seconds', type=float, default=34.)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--press-seconds', type=float, default=5.)
    args = parser.parse_args()
    if not np.isfinite([args.seconds,args.press_seconds]).all() or min(args.seconds,args.press_seconds) <= 0:
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
    shutil.copy2(__file__, args.output/'diagnostic-source.py')
    (args.output/'source-override.json').write_text(json.dumps(dict(entry_point='diagnostic-source.py',
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),controller_root=str(controller_root),scope=__doc__),indent=2)+'\n')
    shutil.copy2(args.reference, args.output/'reference.json')
    sim = DexterousDoorEnv(args.door, args.robot, json.loads(args.robot.with_suffix('.audit.json').read_text()))
    m, d = sim.m, sim.d
    teacher = AcquisitionTeacher(args.robot, motors, ref)
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
    operation = OperationGoals(teacher, m, d, press_seconds=args.press_seconds)
    left = LeftPalmContact(teacher,motors,load_screen_targets(args.plan,args.robot,args.door),fixed_waist=json.loads(args.plan.read_text()).get('fixed_waist',False))
    leaf_body=m.body('leaf').id
    shutil.copy2(inspect.getfile(LeftPalmContact),args.output/'bimanual-transfer-source.py')
    shutil.copy2(args.plan,args.output/'static-plan.json')
    hb, palm = operation.hb, operation.palm
    hand_names = {b:m.body(b).name.removeprefix('robot/') for b in range(m.nbody)
                  if m.body(b).name.startswith(('robot/rh_', 'robot/lh_'))}
    def left_load():
        normal=d.xmat[leaf_body].reshape(3,3)[:,1];total=0.
        for k,c in enumerate(d.contact[:d.ncon]):
            bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
            for hand_side in (0,1):
                if bodies[hand_side].startswith('robot/lh_') and bodies[1-hand_side]=='leaf':
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,k,wrench)
                    force=c.frame.reshape(3,3).T@wrench[:3]*(-1 if hand_side==0 else 1)
                    total+=max(0.,float(-force@normal))
        return total
    physics = [native_grasp_sample(sim, 'leaf_handle_lever_col_n', handle_joint='leaf_handle_hinge')]
    states = {key:[] for key in ('qpos', 'qvel', 'ctrl')}
    traces = []
    try:
        for step in range(round(args.seconds/m.opt.timestep)):
            if operation.started is None and d.time >= 10.6-1e-8:
                tail = [r for r in physics if r['sim_time_s'] >= d.time-.5-1e-8]
                if len(tail) >= 250 and all(r['pad_grasp']['valid_pad_grasp'] for r in tail) and teacher.info.get('path_fraction',0) >= .999:
                    operation.begin()
            goal_info = operation.update()
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
            current_joints=dict(zip(teacher.names,d.qpos[qa]));current_velocities=dict(zip(teacher.names,d.qvel[va]))
            leaf_pose=np.r_[d.xpos[leaf_body],d.xquat[leaf_body]]
            if left.started is None and d.time>=22.-1e-8 and operation.open_started is not None:
                tail=[r for r in physics if r['sim_time_s']>=d.time-.5-1e-8]
                if tail and all(r['pad_grasp']['valid_pad_grasp'] and .075<=r['door_q']<=.10 for r in tail):
                    if not left.fixed_waist:
                        operation.p_relative=d.xmat[hb].reshape(3,3).T@(d.site_xpos[palm]-d.xpos[hb])
                        operation.r_relative=d.xmat[hb].reshape(3,3).T@d.site_xmat[palm].reshape(3,3)
                        operation.hold_h=float(d.qpos[m.jnt_qposadr[operation.hj]])
                        operation.hold_l=float(d.qpos[m.jnt_qposadr[operation.lj]])
                        teacher.position_integral[:]=0.;teacher.rotation_integral[:]=0.
                        operation.update()
                    left.begin(float(d.time),root,current_joints,leaf_pose,np.r_[d.xpos[hb],d.xquat[hb]])
            left.update_targets(float(d.time),root,current_joints,leaf_pose,left_load(),np.r_[d.xpos[hb],d.xquat[hb]])
            force, info = teacher.force(float(d.time),root,dict(zip(teacher.names,d.qpos[qa])),
                dict(zip(teacher.names,d.qvel[va])),np.r_[d.xpos[hb],d.xquat[hb]],loads)
            force=left.apply_forces(force,current_joints,current_velocities)
            d.ctrl[aids] = force
            row = audited_native_step(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')
            row['bolt_slide_m'] = float(d.qpos[m.jnt_qposadr[operation.bj]])
            row['operation'] = goal_info
            row['left']=dict(left.info)
            row['left_panel_load_N']=left_load()
            physics.append(row)
            if step % 10 == 0:
                trace = dict(**sim.diagnostics(),teacher={**info,'phase':goal_info['phase']},operation=goal_info,
                             palm_error_m=float(np.linalg.norm(d.site_xpos[palm]-teacher.positions[-1])))
                trace['left']=dict(left.info);trace['left_panel_load_N']=row['left_panel_load_N']
                traces.append(trace)
                for key in states:
                    states[key].append(getattr(d,key).copy())
                (args.output/'latest.json').write_text(json.dumps(trace)+'\n')
                if step % 500 == 0:
                    print(json.dumps(trace),flush=True)
            if not row['finite'] or row['torso_tilt_deg'] > 35:
                break
        report = audit_grasp_steps(physics,physics_dt=m.opt.timestep,expected_duration=args.seconds)
        report['checks']['acquisition_precedes_operation'] = operation.started is not None
        report['checks']['operator_driven_to_release'] = max(r['handle_angle_rad'] for r in physics) >= .80 and max(r.get('bolt_slide_m',0) for r in physics) >= .011
        tail = [r for r in physics if r['sim_time_s'] >= args.seconds-.5-1e-8]
        report['checks']['partial_leaf_opening_held'] = bool(tail) and all(.075 <= r['door_q'] <= .10 for r in tail)
        report['checks']['opening_bounded_for_transfer'] = max(r['door_q'] for r in physics) <= .12
        report['checks']['left_contact_reached']=bool(left.started is not None and left.progress>=.999)
        report['checks']['sustained_left_panel_load']=bool(tail) and all(r.get('left_panel_load_N',0)>=2. for r in tail)
        report.update(passed=all(report['checks'].values()),scope='Continuous native acquisition, partial opening and left-palm contact only; no release or full opening',
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
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output/'trace.json').write_text(json.dumps(traces)+'\n')
        with gzip.open(args.output/'physics-steps.json.gz','wt') as stream:
            json.dump(physics,stream)
        np.savez_compressed(args.output/'trajectory.npz',**states,
                            terminal_qpos=d.qpos.copy(),terminal_qvel=d.qvel.copy(),
                            terminal_ctrl=d.ctrl.copy(),terminal_time_s=float(d.time))
        print(json.dumps({k:v for k,v in report.items() if k!='final_contacts'}),flush=True)
    finally:
        (args.output/'trace.json').write_text(json.dumps(traces)+'\n')
        with gzip.open(args.output/'physics-steps.json.gz','wt') as stream:json.dump(physics,stream)
        np.savez_compressed(args.output/'trajectory.npz',**states,terminal_qpos=d.qpos.copy(),terminal_qvel=d.qvel.copy(),terminal_ctrl=d.ctrl.copy(),terminal_time_s=float(d.time))
        sim.close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

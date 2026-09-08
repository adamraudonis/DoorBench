#!/usr/bin/env python3
"""Development run: uncut native walk, lower, prepare hand, acquire and operate.

This is a privileged teacher diagnostic. The initial reset is the only physical
pose write. All later phases use the same explicit-force adapter and native
force caps. The original intermediate heading error is reported, not waived.
"""
import argparse
import copy
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
from doorbench.dexterous.approach_lowering import ApproachLoweringController
from doorbench.dexterous.grasp_verification import native_grasp_sample, audited_native_step, audit_grasp_steps
from doorbench.dexterous.locomotion_approach import make_door_approach, wrap_angle
from doorbench.dexterous.operation_teacher import DoorOperationTeacher
from doorbench.dexterous.provenance import capture


def screen_preparation(sim, proposal):
    """Check planning copies at the actual landed root; never pose the plant."""
    m = sim.m
    planned = mujoco.MjData(m)
    planned.qpos[:] = sim.d.qpos
    names = proposal['acquisition']['joint_names']
    ids = np.array([m.joint('robot/'+n).id for n in names])
    qa = m.jnt_qposadr[ids]
    planned.qpos[sim.root_qadr:sim.root_qadr+7] = proposal['initial_root']
    bad = []
    for index,q in enumerate(proposal['acquisition']['path_qpos']):
        planned.qpos[qa] = q
        mujoco.mj_forward(m,planned)
        contacts = []
        for contact in planned.contact[:planned.ncon]:
            bodies = [m.body(m.geom_bodyid[g]).name for g in contact.geom]
            if any(name.startswith('robot/rh_') for name in bodies):
                contacts.append(dict(bodies=bodies,distance_m=float(contact.dist)))
        if contacts:
            bad.append(dict(sample=index,contacts=contacts))
    return dict(scope='All frozen planning samples at actual landed root/legs; no physical state writes',
                passed=not bad,samples=len(proposal['acquisition']['path_qpos']),bad_samples=bad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','reference','preparation','motors','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=55.)
    parser.add_argument('--prepare-seconds',type=float,default=8.)
    parser.add_argument('--portable-sequence',action='store_true')
    args = parser.parse_args()
    if not np.isfinite([args.seconds,args.prepare_seconds]).all() or min(args.seconds,args.prepare_seconds)<=0:
        parser.error('Use finite positive durations')
    if args.output.exists():
        raise SystemExit('Use a fresh evidence directory')
    ref = json.loads(args.reference.read_text())
    prep_ref = json.loads(args.preparation.read_text())
    motors = json.loads(args.motors.read_text())
    source_root = Path(inspect.getfile(ApproachLoweringController)).resolve().parents[2]
    capture(source_root,args.output,{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            timing='before_native_continuous_walk_grasp')
    shutil.copy2(__file__,args.output/'diagnostic-source.py')
    (args.output/'source-override.json').write_text(json.dumps(dict(
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=__doc__),indent=2)+'\n')
    for name in ('reference','preparation','motors'):
        shutil.copy2(getattr(args,name),args.output/(name+'.json'))
    sim,goal,yaw_goal = make_door_approach(args.door,args.robot,args.reference,distance=.7)
    m,d = sim.m,sim.d
    body = ApproachLoweringController(sim,args.checkpoint,goal,yaw_goal,
                                     height=ref['initial_root'][2],handoff_delay=3.,yaw_weight=2.)
    teacher = AcquisitionTeacher(args.robot,motors,ref)
    ids = np.array([m.joint('robot/'+n).id for n in teacher.names])
    qa,va = m.jnt_qposadr[ids],m.jnt_dofadr[ids]
    aids = np.array([m.actuator('robot/'+v['name']).id for v in motors['actuators']])
    original_gain = m.actuator_gainprm[aids,0].copy()
    original_bias = m.actuator_biasprm[aids,:3].copy()
    original_controls = m.actuator_ctrlrange[aids].copy()
    sequence = None
    if args.portable_sequence:
        from doorbench.dexterous.full_sequence_teacher import FullSequenceTeacher
        reset = dict(initial_root=d.qpos[sim.root_qadr:sim.root_qadr+7].tolist(),
            joints=dict(zip(teacher.names,d.qpos[qa])),
            motor_targets={v['name']:float(d.ctrl[a]) for v,a in zip(motors['actuators'],aids)},
            goal_xy=goal.tolist(),goal_yaw_rad=yaw_goal)
        hj,lj = m.joint('leaf_handle_hinge').id,m.joint('leaf_hinge').id
        sequence = FullSequenceTeacher(args.robot,motors,ref,prep_ref,reset,args.checkpoint,
            dict(operator_origin=m.jnt_pos[hj],operator_axis=m.jnt_axis[hj],leaf_origin=m.jnt_pos[lj],leaf_axis=m.jnt_axis[lj]),
            door_xml=args.door/'door.xml',prepare_seconds=args.prepare_seconds)
        source = Path(inspect.getfile(FullSequenceTeacher))
        shutil.copy2(source,args.output/'full-sequence-teacher-source.py')
        (args.output/'full-sequence-source.json').write_text(json.dumps(dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n')
    # Normalize only the adapter, once at reset; preserve original force caps.
    m.actuator_gainprm[aids,0] = 1.
    m.actuator_biasprm[aids,:3] = 0.
    m.actuator_ctrlrange[aids] = teacher.caps
    d.ctrl[aids] = 0.
    immutable = ('body_mass','body_inertia','body_gravcomp','jnt_range','tendon_range',
                 'tendon_solref_lim','tendon_solimp_lim','geom_friction','geom_contype',
                 'geom_conaffinity','actuator_gainprm','actuator_biasprm',
                 'actuator_ctrlrange','actuator_forcerange')
    originals = {key:getattr(m,key).copy() for key in immutable}
    hj,lj,bj = [m.joint(n).id for n in ('leaf_handle_hinge','leaf_hinge','leaf_latch_bolt_slide')]
    hb,lb,palm = m.body('leaf_handle').id,m.body('leaf').id,m.site('robot/rh_palm_touch').id
    operation = DoorOperationTeacher(teacher,dict(operator_origin=m.jnt_pos[hj],operator_axis=m.jnt_axis[hj],
        leaf_origin=m.jnt_pos[lj],leaf_axis=m.jnt_axis[lj]))
    hand_names = {b:m.body(b).name.removeprefix('robot/') for b in range(m.nbody)
                  if m.body(b).name.startswith(('robot/rh_','robot/lh_'))}
    leg_local = np.array([list(aids).index(a) for a in sim.adapter.actuators])
    foot_loads = np.zeros(2)
    physics = [native_grasp_sample(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')]
    traces,poses,velocities,controls = [],[],[],[]
    prep = prep_started = acquisition_started = quiet_since = None
    handoffs = {}
    readiness_screen = None
    try:
        for step in range(round(args.seconds/m.opt.timestep)):
            t = float(d.time)
            if sequence is None:
                walk_control = body.command(foot_loads)
                bias = original_bias[:,0]+original_bias[:,1]*d.actuator_length[aids]+original_bias[:,2]*d.actuator_velocity[aids]
                force = np.clip(original_gain*np.clip(walk_control[aids],original_controls[:,0],original_controls[:,1])+bias,
                                teacher.caps[:,0],teacher.caps[:,1])
                if body.stance_command is not None:
                    # After handoff, the QP reads the normalized model and already
                    # returns force units. Walking still uses original affine math.
                    force[leg_local] = walk_control[sim.adapter.actuators]
                if prep is None and body.stage == 'low stance hold':
                    ready = (t-body.stance_started >= 5. and min(foot_loads)>30 and
                             abs(d.qpos[sim.root_qadr+2]-ref['initial_root'][2])<.01 and
                             np.linalg.norm(d.qvel[sim.root_vadr:sim.root_vadr+3])<.02 and
                             physics[-1]['hand_contact_count']==0)
                    quiet_since = t if ready and quiet_since is None else quiet_since if ready else None
                    if quiet_since is not None and t-quiet_since >= .5:
                        actual = d.qpos[qa].copy()
                        proposal = copy.deepcopy(prep_ref)
                        names = proposal['acquisition']['joint_names']
                        path = np.asarray(proposal['acquisition']['path_qpos'])
                        order = [teacher.names.index(n) for n in names]
                        # Measured legs/root remain physical; only planning copies
                        # receive them. First goal is the actual neutral hand pose.
                        for i,n in enumerate(names):
                            if any(v in n for v in ('hip_','knee','ankle')):
                                path[:,i] = actual[order[i]]
                        path[0] = actual[order]
                        proposal['acquisition']['path_qpos'] = path.tolist()
                        proposal['initial_root'] = d.qpos[sim.root_qadr:sim.root_qadr+7].tolist()
                        (args.output/'actual-preparation-reference.json').write_text(json.dumps(proposal)+'\n')
                        readiness_screen = screen_preparation(sim,proposal)
                        (args.output/'actual-preparation-screen.json').write_text(json.dumps(readiness_screen,indent=2)+'\n')
                        if not readiness_screen['passed']:
                            break  # Preserve the complete executed prefix below.
                        prep = AcquisitionTeacher(args.robot,motors,proposal,reach_seconds=args.prepare_seconds,
                                                  grip_force=0.,palm_integral=0.)
                        prep_started = t
                        R = d.xmat[sim.pelvis].reshape(3,3)
                        actual_yaw = float(np.arctan2(R[1,0],R[0,0]))
                        handoffs['lowered'] = dict(time_s=t,root=d.qpos[sim.root_qadr:sim.root_qadr+7].tolist(),
                                                  original_heading_error_deg=abs(float(np.rad2deg(wrap_angle(actual_yaw-yaw_goal)))))
            else:
                force = np.zeros(len(aids))
            loads = {n:np.zeros(3) for n in hand_names.values()}
            for i,c in enumerate(d.contact[:d.ncon]):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m,d,i,wrench)
                world_force = c.frame.reshape(3,3).T@wrench[:3]
                for sign,g in zip((-1,1),c.geom):
                    b = int(m.geom_bodyid[g])
                    if b in hand_names:
                        loads[hand_names[b]] += sign*world_force
            R = d.xmat[sim.pelvis].reshape(3,3)
            root = np.r_[d.qpos[sim.root_qadr:sim.root_qadr+7],d.qvel[sim.root_vadr:sim.root_vadr+3],
                         R@d.qvel[sim.root_vadr+3:sim.root_vadr+6]]
            q,dq = dict(zip(teacher.names,d.qpos[qa])),dict(zip(teacher.names,d.qvel[va]))
            handle_pose,leaf_pose = np.r_[d.xpos[hb],d.xquat[hb]],np.r_[d.xpos[lb],d.xquat[lb]]
            info = dict(phase=body.stage)
            if sequence is not None:
                force,info = sequence.force(t,root,q,dq,foot_loads,handle_pose,leaf_pose,
                    dict(operator=d.qpos[m.jnt_qposadr[hj]],leaf=d.qpos[m.jnt_qposadr[lj]],latch=d.qpos[m.jnt_qposadr[bj]]),
                    loads,grasp_qualified=physics[-1]['pad_grasp']['valid_pad_grasp'],hand_contact_count=physics[-1]['hand_contact_count'])
                body = sequence.body.controller
                prep,prep_started,acquisition_started = sequence.prep,sequence.prep_started,sequence.acquisition_started
                handoffs = sequence.handoffs
                if readiness_screen is None and sequence.readiness_screen is not None:
                    readiness_screen = sequence.readiness_screen
                    (args.output/'actual-preparation-screen.json').write_text(json.dumps(readiness_screen,indent=2)+'\n')
                    (args.output/'actual-preparation-reference.json').write_text(json.dumps(sequence.actual_preparation)+'\n')
                if sequence.blocked_reason:
                    break
            else:
                if prep is not None and acquisition_started is None:
                    upper,info = prep.force(t-prep_started,root,q,dq,handle_pose,loads)
                    force[:] = upper
                    force[leg_local] = walk_control[sim.adapter.actuators]
                    info['phase'] = 'arm preparation'
                    if info.get('path_fraction',0)>=.999 and t-prep_started>=args.prepare_seconds+2 and info['tracking_error_m']<.005 and physics[-1]['hand_contact_count']==0:
                        acquisition_started = t
                        handoffs['acquisition'] = dict(time_s=t,root=root[:7].tolist(),
                                                      palm_error_m=info['tracking_error_m'],hand_contact_count=physics[-1]['hand_contact_count'])
                if acquisition_started is not None:
                    force,info = operation.force(t-acquisition_started,root,q,dq,handle_pose,leaf_pose,
                        dict(operator=d.qpos[m.jnt_qposadr[hj]],leaf=d.qpos[m.jnt_qposadr[lj]],latch=d.qpos[m.jnt_qposadr[bj]]),
                        loads,grasp_qualified=physics[-1]['pad_grasp']['valid_pad_grasp'])
                    force[leg_local] = walk_control[sim.adapter.actuators]
            d.ctrl[aids] = np.clip(force,teacher.caps[:,0],teacher.caps[:,1])
            row = audited_native_step(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')
            row.update(phase=info['phase'],bolt_slide_m=float(d.qpos[m.jnt_qposadr[bj]]))
            physics.append(row)
            foot_loads[:] = 0.
            for i,c in enumerate(d.contact[:d.ncon]):
                if m.geom(c.geom[0]).name != 'floor' and m.geom(c.geom[1]).name != 'floor':
                    continue
                for g in c.geom:
                    b = int(m.geom_bodyid[g])
                    if b in sim.feet:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(m,d,i,wrench)
                        foot_loads[sim.feet.index(b)] += wrench[0]
            if step%10==0:
                trace = dict(time_s=float(d.time),info=info,root=root[:7].tolist(),door_q=row['door_q'],
                             operator=row['handle_angle_rad'],foot_loads_N=foot_loads.tolist(),
                             strict_grasp=row['pad_grasp']['valid_pad_grasp'],body_solver=body.solver_status,
                             body_solver_failures=body.solver_failures)
                traces.append(trace);poses.append(d.qpos.copy());velocities.append(d.qvel.copy());controls.append(d.ctrl.copy())
                (args.output/'latest.json').write_text(json.dumps(trace)+'\n')
                if step%500==0:
                    print(json.dumps(trace),flush=True)
            if not row['finite'] or row['torso_tilt_deg']>35 or d.qpos[sim.root_qadr+2]<.6:
                break
        report = audit_grasp_steps(physics,physics_dt=m.opt.timestep,expected_duration=args.seconds)
        tail = [row for row in physics if row['sim_time_s']>=args.seconds-.5]
        preparation_rows = [row for row in physics if row.get('phase')=='arm preparation']
        operation_rows = [row for row in physics if row.get('phase') in ('lever_operation','partial_opening')]
        report['checks'].update(uncut_walk_lower_acquisition=acquisition_started is not None,
            actual_preparation_screen=bool(readiness_screen and readiness_screen['passed']),
            stance_solver=body.solver_failures==0,plant_adapter_unchanged=all(np.array_equal(value,getattr(m,key)) for key,value in originals.items()),
            partial_opening=bool(tail) and all(.075<=row['door_q']<=.10 for row in tail),
            contact_free_preparation=bool(preparation_rows) and all(row['hand_contact_count']==0 for row in preparation_rows),
            no_wrong_pad_contact=all(all(c['pad_qualified'] for c in row['pad_grasp']['contacts']) for row in physics))
        report.update(passed=all(report['checks'].values()),scope=__doc__,handoffs=handoffs,
            runtime_pose_writes=0,initial_adapter_normalization=True,source_motor_sha256=hashlib.sha256(args.motors.read_bytes()).hexdigest(),
            stance_solver_failures=body.solver_failures,stance_solver_events=body.solver_events,
            operation_digit_unload_samples=sum(not row['pad_grasp']['valid_pad_grasp'] for row in operation_rows),
            final_leaf_rad=physics[-1]['door_q'],final_contacts=physics[-1]['pad_grasp'])
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output/'trace.json').write_text(json.dumps(traces)+'\n')
        with gzip.open(args.output/'physics-steps.json.gz','wt') as stream:
            json.dump(physics,stream)
        np.savez_compressed(args.output/'trajectory.npz',qpos=poses,qvel=velocities,ctrl=controls,
                            terminal_qpos=d.qpos.copy(),terminal_qvel=d.qvel.copy(),terminal_time_s=float(d.time))
        print(json.dumps({k:v for k,v in report.items() if k!='final_contacts'}),flush=True)
    finally:
        sim.close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

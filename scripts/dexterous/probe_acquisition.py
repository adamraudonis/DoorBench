#!/usr/bin/env python3
"""Execute a screened acquisition candidate with free-base native robot motors.

MuJoCo development trial only. This does not establish Isaac parity, opening,
traversal, or sensor-only control. No robot pose writes occur after reset.
"""
import argparse
import gzip
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.contact_audit import lever_contacts
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.provenance import capture
from doorbench.dexterous.reset import check_joint_reset
from doorbench.dexterous.stance import StanceController
from doorbench.dexterous.grasp_route import reproject_attached_pose
from doorbench.dexterous.grasp_verification import (native_grasp_sample, audited_native_step, audit_grasp_steps, scalar_transmission_matrix)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('robot', 'door', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--mode', choices=('acquisition','initialized-release'), default='acquisition')
    p.add_argument('--landed-stance', action='store_true')
    p.add_argument('--reach-seconds', type=float, default=4.)
    p.add_argument('--hold-seconds', type=float, default=2.)
    p.add_argument('--grip-force', type=float, default=0.)
    p.add_argument('--grip-start', type=float, default=.8)
    p.add_argument('--track-pads', nargs='*', choices=('ff','mf','rf','lf','th'), default=[])
    p.add_argument('--pad-stiffness', type=float, default=300.)
    p.add_argument('--finger-grip-scale',type=float,default=1.)
    p.add_argument('--grip-reaction',action='store_true')
    p.add_argument('--palm-integral',type=float,default=0.,help='Bounded hold-phase Cartesian integral gain, per second')
    p.add_argument('--follow-operator',action='store_true',help='Reproject measured-release palm targets into the actual moving lever frame')
    p.add_argument('--operator-follow-start',type=float,default=.99,help='Path fraction at which to blend in measured lever-frame following')
    p.add_argument('--recorded-finger-forces',action='store_true',help='Use measured release forces as finger feedforward instead of nearest-point preload')
    p.add_argument('--hold-leaf-while-grasping',action='store_true',help='Keep the hand goal in the starting leaf frame while following the lever; motor control only')
    p.add_argument('--torso-impedance', type=float, default=1.)
    p.add_argument('--cartesian-tracking', action='store_true')
    p.add_argument('--tracking-gate', action='store_true')
    p.add_argument('--adaptive-thumb', action='store_true')
    p.add_argument('--motor-tracking-gate', action='store_true')
    p.add_argument('--limit-filter', action='store_true')
    p.add_argument('--explicit-motors', action='store_true', help='Apply the original servo law plus outer impedance as bounded motor torque, matching the Isaac adapter')
    args = p.parse_args()
    if not np.isfinite(args.grip_start) or not 0<=args.grip_start<=1 or not np.isfinite(args.pad_stiffness) or args.pad_stiffness<0:
        p.error('Grip start must be in [0,1] and pad stiffness finite and nonnegative')
    if not np.isfinite([args.finger_grip_scale,args.palm_integral]).all() or args.finger_grip_scale<0 or args.palm_integral<0:
        p.error('Grip scale and integral gain must be finite and nonnegative')
    if args.hold_leaf_while_grasping and not args.follow_operator:p.error('--hold-leaf-while-grasping requires --follow-operator')
    if not np.isfinite(args.operator_follow_start) or not 0<=args.operator_follow_start<=1:p.error('Operator following start must be in [0,1]')
    if not all(np.isfinite(x) for x in (args.reach_seconds,args.hold_seconds,args.grip_force,args.torso_impedance)) or args.reach_seconds<=0 or args.hold_seconds<0 or args.grip_force<0 or args.torso_impedance<1:
        p.error("Use finite positive reach duration, nonnegative hold/force, and impedance >= 1")
    if args.output.exists():
        raise SystemExit('Use a new output directory')
    if not json.loads((args.reference.parent/'geometry-audit.json').read_text())['passed']:
        raise SystemExit('Candidate failed its geometric screen')
    ref = json.loads(args.reference.read_text())
    path = np.asarray(ref['acquisition']['path_qpos'])
    root_path=np.asarray(ref['acquisition'].get('recorded_root_path',[ref['initial_root']]*len(path)))
    if root_path.shape!=(len(path),7) or not np.isfinite(root_path).all():raise ValueError('Invalid reference FK root path')
    recorded_path='recorded_root_path' in ref['acquisition']
    release_mode=args.mode=='initialized-release'
    if release_mode:
        if recorded_path:raise ValueError('Initialized release expects a geometric candidate, not a reversed physical recording')
        path=path[::-1].copy();root_path=root_path[::-1].copy()
    operator_positions=np.asarray(ref['acquisition'].get('recorded_lever_position_m',[]))
    operator_rotations=np.asarray(ref['acquisition'].get('recorded_lever_rotation',[]))
    if args.follow_operator:
        if operator_positions.shape!=(len(path),3) or operator_rotations.shape!=(len(path),3,3):raise ValueError('Recorded operator frames required')
        if not np.isfinite(operator_positions).all() or not np.isfinite(operator_rotations).all():raise ValueError('Nonfinite operator reference')
        if ref['acquisition'].get('recorded_lever_frame',{}).get('geometry_name')!='leaf_handle_lever_col_n':raise ValueError('Recorded operator geometry mismatch')
        if not np.allclose(operator_rotations.transpose(0,2,1)@operator_rotations,np.eye(3),atol=1e-6) or not np.allclose(np.linalg.det(operator_rotations),1,atol=1e-6):raise ValueError('Invalid operator rotations')
    names = ref['acquisition']['joint_names']
    capture(Path(__file__).resolve().parents[2], args.output,
            {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, timing='before_native_acquisition')
    (args.output/'reference.json').write_bytes(args.reference.read_bytes())
    run_scope='Initialized motor-driven release only; not acquisition, opening, traversal or sensor-only control' if release_mode else __doc__
    pipeline=dict(stage=args.mode+' native development trial',scope=run_scope,completion_marker='ACQUISITION_TRIAL_FINISHED',started_at_unix=time.time())
    (args.output/'pipeline.json').write_text(json.dumps(pipeline)+'\n')
    (args.output/'run.pid').write_text(str(os.getpid()))
    def emit(value):
        line=value if isinstance(value,str) else json.dumps(value)
        print(line,flush=True)
        with (args.output/'run.log').open('a') as stream:stream.write(line+'\n')
    sim = DexterousDoorEnv(args.door, args.robot, json.loads(args.robot.with_suffix('.audit.json').read_text()))
    m, d = sim.m, sim.d
    sim.reset(randomize=False, images=False)
    ids = [m.joint('robot/'+n).id for n in names]
    qa = m.jnt_qposadr[ids]
    for pose in path:
        check_joint_reset(names, pose, m.jnt_range[ids])
    if 'initial_plant_qpos' in ref['acquisition']:
        initial_plant=np.asarray(ref['acquisition']['initial_plant_qpos'])
        if initial_plant.shape!=(m.nq,) or not np.isfinite(initial_plant).all():raise ValueError('Invalid recorded reset')
        d.qpos[:]=initial_plant
    d.qpos[sim.root_qadr:sim.root_qadr+7] = ref['initial_root']
    d.qpos[qa] = path[0]
    d.qvel[:] = 0.
    mujoco.mj_forward(m,d)
    matrix = scalar_transmission_matrix(m,sim.actuators,ids)
    recorded_motor_force=np.asarray(ref['acquisition'].get('recorded_motor_force',[]))
    if args.recorded_finger_forces:
        if recorded_motor_force.shape!=(len(path),len(sim.actuators)) or not np.isfinite(recorded_motor_force).all():raise ValueError('Invalid recorded motor-force path')
        if ref['acquisition'].get('recorded_motor_force_names')!=[m.actuator(aid).name.removeprefix('robot/') for aid in sim.actuators]:raise ValueError('Recorded motor-force ordering mismatch')
        if args.grip_force:p.error('Recorded force feedforward and nearest-point preload are separate experiments')
    joint_index = {j:i for i,j in enumerate(ids)}
    if args.landed_stance:
        from doorbench.dexterous.locomotion_manipulation import LandedFootStanceController
        stance=LandedFootStanceController(sim)
    else:stance = StanceController(sim)
    kp = m.actuator_gainprm[sim.actuators,0].copy()
    native_bias=m.actuator_biasprm[sim.actuators,:3].copy()
    native_force_limits=m.actuator_forcerange[sim.actuators].copy()
    arm = np.array([(m.actuator(a).name.startswith('robot/right_') and
        not any(n in m.actuator(a).name for n in ('hip','knee','ankle'))) or
        m.actuator(a).name.startswith('robot/rh_A_WRJ') for a in sim.actuators])
    damping = np.array([(.8 if 'WRJ' in m.actuator(a).name else 10.) if is_arm else
        20. if m.actuator(a).name=='robot/torso' else 0. for a,is_arm in zip(sim.actuators,arm)])
    gain = 9*arm.astype(float)
    for i,aid in enumerate(sim.actuators):
        if m.actuator(aid).name=='robot/torso':
            gain[i]=args.torso_impedance-1.
    finger_motors = [i for i,aid in enumerate(sim.actuators) if m.actuator(aid).name.startswith('robot/rh_') and 'WRJ' not in m.actuator(aid).name]
    finger_inverse = np.linalg.pinv(matrix[finger_motors].T)
    arm_motors=np.flatnonzero(arm);arm_inverse=np.linalg.pinv(matrix[arm_motors].T)
    digit_geoms = {digit:[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_'+digit)] for digit in ('ff','mf','rf','lf','th')}
    thumb_joints=[j for j in ids if m.joint(j).name.startswith('robot/rh_TH')]
    thumb_q=m.jnt_qposadr[thumb_joints];thumb_v=m.jnt_dofadr[thumb_joints]
    thumb_bias=np.zeros(len(ids));thumb_indices=[joint_index[j] for j in thumb_joints]
    lever = m.geom('leaf_handle_lever_col_n').id
    leaf=m.body('leaf').id
    starting_leaf_position=d.xpos[leaf].copy();starting_leaf_rotation=d.xmat[leaf].reshape(3,3).copy()
    jp=np.zeros((3,m.nv));jr=jp.copy()
    target = matrix@path[0]
    d.ctrl[sim.actuators] = np.clip(target,sim.low,sim.high)
    if args.explicit_motors:
        # Only change command units from servo target to motor force. Preserve
        # transmissions, joint limits, dynamics, and native force caps.
        m.actuator_gainprm[sim.actuators,0]=1.
        m.actuator_biasprm[sim.actuators,:3]=0.
        m.actuator_ctrlrange[sim.actuators]=native_force_limits
        d.ctrl[sim.actuators]=0.
    limit_filter=None;limit_info={}
    if args.limit_filter:
        if not args.explicit_motors:p.error('--limit-filter requires --explicit-motors')
        from doorbench.dexterous.limit_filter import HandLimitFilter
        limit_filter=HandLimitFilter(m,d,sim.actuators,ids,matrix,finger_motors)
    rows = []; poses = []; velocities = []; controls = []
    initial = sim.diagnostics()
    initial_contacts = lever_contacts(m,d,'leaf_handle_lever_col_n')
    physics_steps=[native_grasp_sample(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge')]
    states = mujoco.MjData(m)
    states.qpos[:] = d.qpos
    states.qpos[qa] = path[-1]
    states.qpos[sim.root_qadr:sim.root_qadr+7]=root_path[-1]
    mujoco.mj_kinematics(m,states)
    palm = m.site('robot/rh_palm_touch').id
    goal = states.site_xpos[palm].copy()
    pad_tracker = None; pad_errors = {}; pad_paths = []
    if args.track_pads:
        from doorbench.dexterous.pad_tracking import PadTracker
        pad_tracker = PadTracker(m,states,digit_geoms,lever,digits=args.track_pads,
                                 stiffness=args.pad_stiffness)
    path_positions=[];path_rotations=[]
    for index,pose in enumerate(path):
        states.qpos[sim.root_qadr:sim.root_qadr+7]=root_path[index]
        states.qpos[qa]=pose;mujoco.mj_kinematics(m,states)
        path_positions.append(states.site_xpos[palm].copy());path_rotations.append(states.site_xmat[palm].reshape(3,3).copy())
        if pad_tracker:pad_paths.append(pad_tracker.positions(states))
    arm_ids=[m.joint('robot/'+n).id for n in ref['workspace_fit']['joint_names']]
    arm_q=m.jnt_qposadr[arm_ids];arm_v=m.jnt_dofadr[arm_ids]
    horizon = args.reach_seconds+args.hold_seconds+1.
    total_steps = round(horizon/m.opt.timestep)
    progress=0.;tracking_error=0.;motor_error=0.
    position_integral=np.zeros(3);rotation_integral=np.zeros(3)
    operator_follow_time=None
    try:
        for step in range(total_steps):
            if step%5 == 0:
                if args.tracking_gate:
                    if d.time>1.:
                        progress=min(1.,progress+5*m.opt.timestep/args.reach_seconds*np.clip((.015-tracking_error)/.010,0.,1.)*(np.clip((.06-motor_error)/.04,0.,1.) if args.motor_tracking_gate else 1.))
                    u=progress
                else:
                    u = np.clip((d.time-1.)/args.reach_seconds,0.,1.)
                if not recorded_path:u = u*u*u*(10+u*(-15+6*u))
                coordinate = u*(len(path)-1)
                i = min(int(coordinate),len(path)-2); f = coordinate-i
                desired = path[i]*(1-f)+path[i+1]*f
                target_position=path_positions[i]*(1-f)+path_positions[i+1]*f
                target_rotation=Rotation.from_rotvec(f*Rotation.from_matrix(path_rotations[i+1]@path_rotations[i].T).as_rotvec()).as_matrix()@path_rotations[i]
                if args.follow_operator:
                    reference_position=operator_positions[i]*(1-f)+operator_positions[i+1]*f
                    reference_rotation=Rotation.from_rotvec(f*Rotation.from_matrix(operator_rotations[i+1]@operator_rotations[i].T).as_rotvec()).as_matrix()@operator_rotations[i]
                    operator_position=d.geom_xpos[lever].copy();operator_rotation=d.geom_xmat[lever].reshape(3,3).copy()
                    if args.hold_leaf_while_grasping:
                        # A closed-leaf hand goal resists unintended swing through
                        # real contact. No leaf pose/velocity/force is written.
                        closing_rotation=starting_leaf_rotation@d.xmat[leaf].reshape(3,3).T
                        operator_position=starting_leaf_position+closing_rotation@(operator_position-d.xpos[leaf])
                        operator_rotation=closing_rotation@operator_rotation
                    attached_position,attached_rotation=reproject_attached_pose(target_position,target_rotation,
                        reference_position,reference_rotation,operator_position,operator_rotation)
                    # First seat along the measured free-space approach. Early
                    # rigid following can turn an incidental touch into a pull
                    # on the entire leaf before a grasp exists.
                    if operator_follow_time is None and u>=args.operator_follow_start:operator_follow_time=d.time
                    blend=0. if operator_follow_time is None else float(np.clip(d.time-operator_follow_time,0.,1.))
                    blend=blend*blend*(3-2*blend)
                    target_position=target_position*(1-blend)+attached_position*blend
                    target_rotation=Rotation.from_rotvec(blend*Rotation.from_matrix(attached_rotation@target_rotation.T).as_rotvec()).as_matrix()@target_rotation
                    if u>=.999:goal=target_position.copy()
                tracking_error=float(np.linalg.norm(target_position-d.site_xpos[palm]))
                motor_error=float(np.max(np.abs((matrix@desired-d.actuator_length[sim.actuators])[finger_motors])))
                if args.cartesian_tracking:
                    if args.palm_integral and u>.99:
                        caps=native_force_limits[arm_motors]
                        saturated=np.any(np.abs(d.actuator_force[sim.actuators[arm_motors]])>=.98*np.max(np.abs(caps),axis=1))
                        if not saturated:
                            position_integral+=args.palm_integral*5*m.opt.timestep*(target_position-d.site_xpos[palm])
                            rotation_integral+=args.palm_integral*5*m.opt.timestep*Rotation.from_matrix(target_rotation@d.site_xmat[palm].reshape(3,3).T).as_rotvec()
                        position_integral*=min(1.,.02/max(1e-12,np.linalg.norm(position_integral)))
                        rotation_integral*=min(1.,.05/max(1e-12,np.linalg.norm(rotation_integral)))
                    ik_position=target_position+position_integral
                    ik_rotation=Rotation.from_rotvec(rotation_integral).as_matrix()@target_rotation
                    states.qpos[:]=d.qpos;states.qpos[qa]=desired
                    for _ in range(25):
                        mujoco.mj_kinematics(m,states);mujoco.mj_comPos(m,states)
                        error=np.r_[5*(ik_position-states.site_xpos[palm]),Rotation.from_matrix(ik_rotation@states.site_xmat[palm].reshape(3,3).T).as_rotvec()]
                        mujoco.mj_jacSite(m,states,jp,jr,palm)
                        jac=np.vstack([5*jp[:,arm_v],jr[:,arm_v]])
                        change=jac.T@np.linalg.solve(jac@jac.T+.003*np.eye(6),error)
                        states.qpos[arm_q]=np.clip(states.qpos[arm_q]+np.clip(change,-.04,.04),m.jnt_range[arm_ids,0],m.jnt_range[arm_ids,1])
                        if np.linalg.norm(error)<1e-4:break
                    desired=states.qpos[qa].copy()
                if args.adaptive_thumb and u>.9:
                    nearest=None
                    for g in digit_geoms['th']:
                        pair=np.zeros(6);distance=mujoco.mj_geomDistance(m,d,g,lever,.08,pair)
                        if nearest is None or distance<nearest[0]:nearest=(distance,g,pair)
                    distance,g,pair=nearest
                    vector=pair[3:]-pair[:3];length=np.linalg.norm(vector)
                    if .0001<distance<.04 and length>1e-7:
                        mujoco.mj_jac(m,d,jp,jr,pair[:3],int(m.geom_bodyid[g]))
                        gradient=(vector/length)@jp[:,thumb_v]
                        correction=np.clip(.2*(distance+.0001)*gradient/(gradient@gradient+.0001),-.005,.005)
                        thumb_bias[thumb_indices]+=correction
                        thumb_bias[thumb_indices]=np.clip(desired[thumb_indices]+thumb_bias[thumb_indices],m.jnt_range[thumb_joints,0],m.jnt_range[thumb_joints,1])-desired[thumb_indices]
                target = np.clip(matrix@(desired+thumb_bias),sim.low,sim.high)
                stance_target, status = stance.command()
            command = target.copy()
            command += gain*(target-d.actuator_length[sim.actuators])
            command -= damping*d.actuator_velocity[sim.actuators]/kp
            for local, aid in enumerate(sim.actuators):
                if arm[local] and m.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:
                    command[local] += d.qfrc_bias[m.jnt_dofadr[m.actuator_trnid[aid,0]]]/kp[local]
            if pad_tracker:
                pad_targets={digit:pad_paths[i][digit]*(1-f)+pad_paths[i+1][digit]*f for digit in args.track_pads}
                generalized,pad_errors=pad_tracker.generalized_force(d,pad_targets)
                command[finger_motors]+=(finger_inverse@generalized[m.jnt_dofadr[ids]])/kp[finger_motors]
            preload_scale=float(np.clip((1.3-d.time)/.3,0.,1.)) if release_mode else float(u>args.grip_start)
            if args.grip_force and preload_scale:
                generalized = np.zeros(m.nv)
                for digit,geoms in digit_geoms.items():
                    nearest = None
                    for g in geoms:
                        pair = np.zeros(6)
                        distance = mujoco.mj_geomDistance(m,d,g,lever,.08,pair)
                        if nearest is None or distance<nearest[0]:
                            nearest = (distance,g,pair)
                    distance,g,pair = nearest
                    vector = pair[3:]-pair[:3]; length=np.linalg.norm(vector)
                    if length<1e-7 or distance>.05:
                        continue
                    inward=vector/length*(1 if distance>=0 else -1)
                    mujoco.mj_jac(m,d,jp,jr,pair[:3],int(m.geom_bodyid[g]))
                    generalized += jp.T@inward*args.grip_force*preload_scale*(1. if digit=='th' else args.finger_grip_scale)
                command[finger_motors] += (finger_inverse@generalized[m.jnt_dofadr[ids]])/kp[finger_motors]
                if args.grip_reaction:command[arm_motors]+=(arm_inverse@generalized[m.jnt_dofadr[ids]])/kp[arm_motors]
            if args.recorded_finger_forces:
                force_reference=recorded_motor_force[i]*(1-f)+recorded_motor_force[i+1]*f
                command[finger_motors]+=force_reference[finger_motors]/kp[finger_motors]
            if args.explicit_motors:
                force=kp*command+native_bias[:,0]+native_bias[:,1]*d.actuator_length[sim.actuators]+native_bias[:,2]*d.actuator_velocity[sim.actuators]
                if stance_target is not None:force[stance.local]=stance_target
                force=np.clip(force,native_force_limits[:,0],native_force_limits[:,1])
                if limit_filter is not None:force,limit_info=limit_filter.apply(force)
                d.ctrl[sim.actuators]=force
            else:
                if stance_target is not None:command[stance.local]=stance_target
                d.ctrl[sim.actuators] = np.clip(command,sim.low,sim.high)
            physics_steps.append(audited_native_step(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge'))
            if step%10 == 0:
                contact = lever_contacts(m,d,'leaf_handle_lever_col_n')
                joint_violation=float(np.maximum(m.jnt_range[ids,0]-d.qpos[qa],d.qpos[qa]-m.jnt_range[ids,1]).max())
                penetration=0.
                for c in d.contact[:d.ncon]:
                    bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
                    geoms=[m.geom(int(g)).name for g in c.geom]
                    if any(n.startswith('robot/') for n in bodies) and not ('floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)):
                        penetration=max(penetration,-float(c.dist))
                row = dict(**sim.diagnostics(), max_joint_limit_violation_rad=max(0.,joint_violation),max_nonfoot_penetration_m=penetration, contacts=contact, stance_status=status,
                    palm_error_m=float(np.linalg.norm(d.site_xpos[palm]-goal)),
                    path_fraction=float(u), tracking_error_m=tracking_error,motor_tracking_error_rad=motor_error,limit_filter=limit_info,pad_errors_m=pad_errors,
                    palm_integral_m=position_integral.tolist(),palm_rotation_integral_rad=rotation_integral.tolist(),
                    motor_force_Nm=d.actuator_force[sim.actuators].tolist())
                rows.append(row); poses.append(d.qpos.copy()); velocities.append(d.qvel.copy()); controls.append(d.ctrl.copy())
                (args.output/'latest.json').write_text(json.dumps(row)+'\n')
                if step%500 == 0:
                    emit({k:v for k,v in row.items() if k not in ('contacts','motor_force_Nm')})
                if not row['finite'] or row['root_height_m']<.6 or row['torso_tilt_deg']>35:
                    break
        tail = [r for r in rows if r['sim_time_s']>=horizon-.5]
        limits = m.actuator_forcerange[sim.actuators]
        forces = np.asarray([r['motor_force_Nm'] for r in rows])
        checks = dict(contact_free_start=not initial_contacts['contacts'],
            full_duration=rows[-1]['sim_time_s']>=horizon-.03,
            upright=all(r['torso_tilt_deg']<12 and r['root_height_m']>.7 for r in rows),
            finite=all(r['finite'] and not r['numerical_warnings'] for r in rows),
            native_motor_limits=bool(np.all(forces>=limits[:,0]-1e-5) and np.all(forces<=limits[:,1]+1e-5)),
            physical_joint_limits=all(r['max_joint_limit_violation_rad']<=.02 for r in rows),
            limit_filter_feasible=not args.limit_filter or all(r['limit_filter'].get('feasible',False) for r in rows),
            nonfoot_penetration=all(r['max_nonfoot_penetration_m']<=.003 for r in rows),
            path_completed=bool(rows[-1]['path_fraction']>=.999),
            reaches_grasp=bool(tail and max(r['palm_error_m'] for r in tail)<.02),
            holds_opposed_contacts=bool(tail and all(r['contacts']['opposed'] for r in tail)))
        strict_audit=audit_grasp_steps(physics_steps,physics_dt=m.opt.timestep,expected_duration=horizon)
        if release_mode:
            # The acquisition audit stays explicitly failed for an initialized
            # release. A separate contract checks its actual diagnostic scope.
            physical_keys=('complete_physics_step_evidence','closed_leaf_start','resting_operator_start','finite','upright','physical_joint_limits','nonfoot_penetration','native_motor_limits','no_external_assistance','documented_loopback_limits')
            release_checks={key:strict_audit['checks'].get(key,False) for key in physical_keys}
            initial_hold=[r for r in physics_steps if .4<=r['sim_time_s']<=.9]
            hand_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
            final_gap=min(float(mujoco.mj_geomDistance(m,d,g,lever,1.,None)) for g in hand_geoms)
            release_checks.update(initial_loaded_pad_grasp=bool(initial_hold and all(r['pad_grasp']['valid_pad_grasp'] for r in initial_hold)),
                ends_contact_free=bool(final_gap>=.02 and all(r['hand_contact_count']==0 for r in physics_steps if r['sim_time_s']>=horizon-.5)),path_completed=checks['path_completed'])
            release_audit=dict(scope=run_scope,passed=all(release_checks.values()),checks=release_checks,final_hand_gap_m=final_gap)
            (args.output/'release-audit.json').write_text(json.dumps(release_audit,indent=2)+'\n')
        checks.update({'strict_'+key:value for key,value in strict_audit['checks'].items()})
        report = dict(scope=run_scope, passed=all(checks.values()),checks=checks, initial=initial,strict_grasp_audit=strict_audit,
            max_torso_tilt_deg=max(r['torso_tilt_deg'] for r in rows),
            final_palm_error_m=rows[-1]['palm_error_m'], final_contacts=rows[-1]['contacts'],
            runtime_robot_pose_writes=0, direct_door_commands=False, explicit_motor_mode=args.explicit_motors, trials=1)
        np.savez_compressed(args.output/'trajectory.npz',qpos=poses,qvel=velocities,ctrl=controls)
        (args.output/'trace.json').write_text(json.dumps(rows)+'\n')
        with gzip.open(args.output/'physics-steps.json.gz','wt') as stream:json.dump(physics_steps,stream)
        (args.output/'strict-grasp-audit.json').write_text(json.dumps(strict_audit,indent=2)+'\n')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        emit({k:v for k,v in report.items() if k not in ('initial','final_contacts')})
        emit({k:v for k,v in report['final_contacts'].items() if k!='contacts'})
        effective_passed=release_audit['passed'] if release_mode else report['passed']
        pipeline.update(result_passed=effective_passed,stage=args.mode+(' checks passed' if effective_passed else ' failed; inspect report and trace'))
        (args.output/'pipeline.json').write_text(json.dumps(pipeline)+'\n')
        emit('ACQUISITION_TRIAL_FINISHED')
    finally:
        sim.close()
    raise SystemExit(0 if effective_passed else 1)


if __name__=='__main__':
    main()

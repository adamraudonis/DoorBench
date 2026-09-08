#!/usr/bin/env python3
"""Execute a screened acquisition candidate with free-base native robot motors.

MuJoCo development trial only. This does not establish Isaac parity, opening,
traversal, or sensor-only control. No robot pose writes occur after reset.
"""
import argparse
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('robot', 'door', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--reach-seconds', type=float, default=4.)
    p.add_argument('--hold-seconds', type=float, default=2.)
    p.add_argument('--grip-force', type=float, default=0.)
    p.add_argument('--torso-impedance', type=float, default=1.)
    p.add_argument('--cartesian-tracking', action='store_true')
    p.add_argument('--tracking-gate', action='store_true')
    p.add_argument('--adaptive-thumb', action='store_true')
    p.add_argument('--explicit-motors', action='store_true', help='Apply the original servo law plus outer impedance as bounded motor torque, matching the Isaac adapter')
    args = p.parse_args()
    if not all(np.isfinite(x) for x in (args.reach_seconds,args.hold_seconds,args.grip_force,args.torso_impedance)) or args.reach_seconds<=0 or args.hold_seconds<0 or args.grip_force<0 or args.torso_impedance<1:
        p.error("Use finite positive reach duration, nonnegative hold/force, and impedance >= 1")
    if args.output.exists():
        raise SystemExit('Use a new output directory')
    if not json.loads((args.reference.parent/'geometry-audit.json').read_text())['passed']:
        raise SystemExit('Candidate failed its geometric screen')
    ref = json.loads(args.reference.read_text())
    path = np.asarray(ref['acquisition']['path_qpos'])
    names = ref['acquisition']['joint_names']
    capture(Path(__file__).resolve().parents[2], args.output,
            {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, timing='before_native_acquisition')
    (args.output/'reference.json').write_bytes(args.reference.read_bytes())
    pipeline=dict(stage='Native open-hand acquisition development trial',scope=__doc__,completion_marker='ACQUISITION_TRIAL_FINISHED',started_at_unix=time.time())
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
    d.qpos[sim.root_qadr:sim.root_qadr+7] = ref['initial_root']
    d.qpos[qa] = path[0]
    d.qvel[:] = 0.
    mujoco.mj_forward(m,d)
    matrix = np.zeros((len(sim.actuators), len(ids)))
    joint_index = {j:i for i,j in enumerate(ids)}
    for i, aid in enumerate(sim.actuators):
        tid = int(m.actuator_trnid[aid,0])
        if m.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            matrix[i,joint_index[tid]] = m.actuator_gear[aid,0]
        elif m.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_TENDON:
            for k in range(m.tendon_adr[tid],m.tendon_adr[tid]+m.tendon_num[tid]):
                matrix[i,joint_index[int(m.wrap_objid[k])]] = m.wrap_prm[k]
        else:
            raise ValueError('Unsupported native motor transmission')
    stance = StanceController(sim)
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
    digit_geoms = {digit:[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_'+digit)] for digit in ('ff','mf','rf','lf','th')}
    thumb_joints=[j for j in ids if m.joint(j).name.startswith('robot/rh_TH')]
    thumb_q=m.jnt_qposadr[thumb_joints];thumb_v=m.jnt_dofadr[thumb_joints]
    thumb_bias=np.zeros(len(ids));thumb_indices=[joint_index[j] for j in thumb_joints]
    lever = m.geom('leaf_handle_lever_col_n').id
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
    rows = []; poses = []; velocities = []; controls = []
    initial = sim.diagnostics()
    initial_contacts = lever_contacts(m,d,'leaf_handle_lever_col_n')
    states = mujoco.MjData(m)
    states.qpos[:] = d.qpos
    states.qpos[qa] = path[-1]
    mujoco.mj_kinematics(m,states)
    palm = m.site('robot/rh_palm_touch').id
    goal = states.site_xpos[palm].copy()
    path_positions=[];path_rotations=[]
    for pose in path:
        states.qpos[qa]=pose;mujoco.mj_kinematics(m,states)
        path_positions.append(states.site_xpos[palm].copy());path_rotations.append(states.site_xmat[palm].reshape(3,3).copy())
    arm_ids=[m.joint('robot/'+n).id for n in ref['workspace_fit']['joint_names']]
    arm_q=m.jnt_qposadr[arm_ids];arm_v=m.jnt_dofadr[arm_ids]
    horizon = args.reach_seconds+args.hold_seconds+1.
    total_steps = round(horizon/m.opt.timestep)
    progress=0.;tracking_error=0.
    try:
        for step in range(total_steps):
            if step%5 == 0:
                if args.tracking_gate:
                    if d.time>1.:
                        progress=min(1.,progress+5*m.opt.timestep/args.reach_seconds*np.clip((.015-tracking_error)/.010,0.,1.))
                    u=progress
                else:
                    u = np.clip((d.time-1.)/args.reach_seconds,0.,1.)
                u = u*u*u*(10+u*(-15+6*u))
                coordinate = u*(len(path)-1)
                i = min(int(coordinate),len(path)-2); f = coordinate-i
                desired = path[i]*(1-f)+path[i+1]*f
                target_position=path_positions[i]*(1-f)+path_positions[i+1]*f
                tracking_error=float(np.linalg.norm(target_position-d.site_xpos[palm]))
                if args.cartesian_tracking:
                    target_rotation=Rotation.from_rotvec(f*Rotation.from_matrix(path_rotations[i+1]@path_rotations[i].T).as_rotvec()).as_matrix()@path_rotations[i]
                    states.qpos[:]=d.qpos;states.qpos[qa]=desired
                    for _ in range(25):
                        mujoco.mj_kinematics(m,states);mujoco.mj_comPos(m,states)
                        error=np.r_[5*(target_position-states.site_xpos[palm]),Rotation.from_matrix(target_rotation@states.site_xmat[palm].reshape(3,3).T).as_rotvec()]
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
            if args.grip_force and u>.8:
                generalized = np.zeros(m.nv)
                for geoms in digit_geoms.values():
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
                    generalized += jp.T@inward*args.grip_force
                command[finger_motors] += (finger_inverse@generalized[m.jnt_dofadr[ids]])/kp[finger_motors]
            if args.explicit_motors:
                force=kp*command+native_bias[:,0]+native_bias[:,1]*d.actuator_length[sim.actuators]+native_bias[:,2]*d.actuator_velocity[sim.actuators]
                if stance_target is not None:force[stance.local]=stance_target
                d.ctrl[sim.actuators]=np.clip(force,native_force_limits[:,0],native_force_limits[:,1])
            else:
                if stance_target is not None:command[stance.local]=stance_target
                d.ctrl[sim.actuators] = np.clip(command,sim.low,sim.high)
            sim.plant.step()
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
                    path_fraction=float(u), tracking_error_m=tracking_error,
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
            nonfoot_penetration=all(r['max_nonfoot_penetration_m']<=.003 for r in rows),
            path_completed=bool(rows[-1]['path_fraction']>=.999),
            reaches_grasp=bool(tail and max(r['palm_error_m'] for r in tail)<.02),
            holds_opposed_contacts=bool(tail and all(r['contacts']['opposed'] for r in tail)))
        report = dict(scope=__doc__, passed=all(checks.values()),checks=checks, initial=initial,
            max_torso_tilt_deg=max(r['torso_tilt_deg'] for r in rows),
            final_palm_error_m=rows[-1]['palm_error_m'], final_contacts=rows[-1]['contacts'],
            runtime_robot_pose_writes=0, direct_door_commands=False, explicit_motor_mode=args.explicit_motors, trials=1)
        np.savez_compressed(args.output/'trajectory.npz',qpos=poses,qvel=velocities,ctrl=controls)
        (args.output/'trace.json').write_text(json.dumps(rows)+'\n')
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        emit({k:v for k,v in report.items() if k not in ('initial','final_contacts')})
        emit({k:v for k,v in report['final_contacts'].items() if k!='contacts'})
        pipeline.update(result_passed=report['passed'],stage='Acquisition checks passed' if report['passed'] else 'Acquisition failed; inspect report and trace')
        (args.output/'pipeline.json').write_text(json.dumps(pipeline)+'\n')
        emit('ACQUISITION_TRIAL_FINISHED')
    finally:
        sim.close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Evaluate official H1 locomotion on the full free-base H1/dual-Shadow plant.

Plane-only CPU development trial; not Isaac parity, doorway traversal, a human
reference, or a vision/tactile door policy. Every step uses bounded robot motors.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from doorbench.dexterous.locomotion import H1WalkingPolicy, NativeH1MotorAdapter, DEFAULT_ANGLES, POLICY_SHA256, UPSTREAM_REVISION
from doorbench.dexterous.reset import check_joint_reset
from doorbench.dexterous.stance import PoseStanceController as SettlingStanceController
from doorbench.dexterous.provenance import capture


def make_plant(robot, *, initial_pose='standing', reference=None, seed=0, joint_noise=0.):
    spec=mujoco.MjSpec.from_file(str(robot))
    spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[10,10,.1],rgba=[.32,.34,.36,1],friction=[.7,.005,.0001])
    m=spec.compile();d=mujoco.MjData(m);audit=json.loads(robot.with_suffix('.audit.json').read_text())
    if hashlib.sha256(robot.read_bytes()).hexdigest()!=audit['robot_xml_sha256']:raise ValueError('Robot source differs from audit')
    if m.nmocap or m.neq or np.count_nonzero(m.body_gravcomp):raise ValueError('Unsupported root/support modification')
    d.qpos[2]=audit['free_root_height']
    for name,value in audit['nominal_joint_positions'].items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    for side in ('left','right'):
        d.qpos[m.jnt_qposadr[m.joint(side+'_elbow').id]]=.7
        d.qpos[m.jnt_qposadr[m.joint(side+'_shoulder_roll').id]]=.1 if side=='left' else -.1
    adapter=NativeH1MotorAdapter(m,prefix='')
    if initial_pose=='standing':d.qpos[adapter.qadr]=DEFAULT_ANGLES
    elif initial_pose=='opening':
        if reference is None:raise ValueError('opening initial pose needs --reference')
        ref=json.loads(reference.read_text())
        for name,value in ref['initial_joints'].items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    rng=np.random.default_rng(seed);d.qpos[adapter.qadr]+=rng.uniform(-joint_noise,joint_noise,10)
    joints=np.array([j for j in range(m.njnt) if m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE]);qa=m.jnt_qposadr[joints]
    check_joint_reset([m.joint(j).name for j in joints],d.qpos[qa],m.jnt_range[joints])
    mujoco.mj_forward(m,d)
    feet=[m.body(side+'_ankle_link').id for side in ('left','right')]
    bottoms=[]
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] not in feet or not m.geom_contype[gid]:continue
        if m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Audit non-mesh foot before use')
        mid=m.geom_dataid[gid];vertices=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
        world=vertices@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid];bottoms.append(float(world[:,2].min()))
    # Reset-only clearance of the actual collision sole, including joint-noise tilt.
    d.qpos[2]+=.0001-min(bottoms);mujoco.mj_forward(m,d)
    for aid in range(m.nu):d.ctrl[aid]=d.actuator_length[aid]
    return SimpleNamespace(m=m,d=d,root_qadr=0,root_vadr=0,joint_prefix='',actuators=np.arange(m.nu),joints=joints,pelvis=m.body('pelvis').id,feet=feet,adapter=adapter)


def schedule(mode,t,speed):
    if mode=='forward':return np.array([speed,0.,0.])
    if mode=='walk-stop':return np.array([speed if 1<=t<9 else 0.,0.,0.])
    if mode=='turn-stop':return np.array([speed if 1<=t<10 else 0.,0.,.3 if 4<=t<7 else 0.])
    if mode=='restart':return np.array([speed if 1<=t<5 or 7<=t<11 else 0.,0.,0.])
    if mode=='stand':return np.zeros(3)
    raise ValueError(mode)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('robot','checkpoint','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=('forward','walk-stop','turn-stop','restart','stand'),default='walk-stop');p.add_argument('--seconds',type=float,default=14.)
    p.add_argument('--speed',type=float,default=.4);p.add_argument('--seed',type=int,default=0);p.add_argument('--joint-noise',type=float,default=0.)
    p.add_argument('--initial-pose',choices=('standing','nominal','opening'),default='standing');p.add_argument('--reference',type=Path)
    p.add_argument('--phase-stop',action='store_true',help='Experimental fade of gait-phase observation after zero command')
    p.add_argument('--settle-stance',action='store_true',help='Privileged contact-gated handoff to stationary motor stance after a stop command')
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Use a fresh output directory')
    if not all(np.isfinite(x) for x in (args.seconds,args.speed,args.joint_noise)) or args.seconds<=0 or args.joint_noise<0:raise SystemExit('Invalid duration/speed/noise')
    capture(Path(__file__).resolve().parents[2],args.output,{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},timing='before_native_locomotion')
    pipeline=dict(stage='Native H1/Shadow locomotion development trial',completion_marker='LOCOMOTION_TRIAL_FINISHED',started_at_unix=time.time());(args.output/'pipeline.json').write_text(json.dumps(pipeline)+'\n');(args.output/'run.pid').write_text(str(os.getpid()))
    def emit(value):
        line=value if isinstance(value,str) else json.dumps(value);print(line,flush=True)
        with (args.output/'run.log').open('a') as stream:stream.write(line+'\n')
    sim=make_plant(args.robot,initial_pose=args.initial_pose,reference=args.reference,seed=args.seed,joint_noise=args.joint_noise);m,d=sim.m,sim.d;a=sim.adapter
    policy=H1WalkingPolicy(args.checkpoint);target=DEFAULT_ANGLES.copy();start=d.qpos[:3].copy();fixed=d.ctrl.copy();rows=[];poses=[];vel=[];controls=[];began=time.time()
    immutable=('body_mass','body_inertia','body_gravcomp','jnt_range','dof_damping','dof_armature','dof_frictionloss','geom_contype','geom_conaffinity','geom_friction','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange')
    originals={k:getattr(m,k).copy() for k in immutable};max_force_excess=0.;max_joint_violation=0.;external_max=0.;warnings=0
    stance=None;stance_command=None;stance_status='not active';zero_since=None;ever_moved=False
    initial_feet=d.xpos[sim.feet].copy();leg_count=round(.02/m.opt.timestep)
    if not np.isclose(leg_count*m.opt.timestep,.02):raise ValueError('Policy requires exact 20 ms update period')
    for step in range(round(args.seconds/m.opt.timestep)):
        if step%leg_count==0:
            command=schedule(args.mode,d.time,args.speed)
            if np.linalg.norm(command)>0:
                ever_moved=True;zero_since=None;stance=None
            elif zero_since is None:zero_since=d.time
            if args.settle_stance and stance is None and ever_moved and zero_since is not None and d.time-zero_since>1.:
                loads=np.zeros(2)
                for i,contact in enumerate(d.contact[:d.ncon]):
                    bodies=m.geom_bodyid[contact.geom]
                    if 0 in bodies and max(bodies) in sim.feet:
                        wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench);loads[sim.feet.index(max(bodies))]+=wrench[0]
                if min(loads)>30:
                    stance=SettlingStanceController(sim)
                    emit({'transition':'walking to stationary motor stance','time_s':d.time,'foot_loads_N':loads.tolist()})
            # Own-state observation contract. Body angular velocity and projected
            # gravity are IMU-equivalent quantities; no root position is passed.
            phase_amplitude=max(0.,1.-(d.time-zero_since)) if args.phase_stop and ever_moved and zero_since is not None else 1.
            target=policy.step(d.qpos[a.qadr],d.qvel[a.vadr],d.qvel[3:6],d.body('pelvis').xmat.reshape(3,3).T@[0.,0.,-1.],command,d.time,phase_amplitude=phase_amplitude)
        d.ctrl[:]=fixed;d.ctrl[a.actuators]=a.command(d,target)
        if stance is not None:
            if step%5==0:stance_command,stance_status=stance.command()
            if stance_command is not None:d.ctrl[stance.act]=np.clip(stance_command,m.actuator_ctrlrange[stance.act,0],m.actuator_ctrlrange[stance.act,1])
        mujoco.mj_step(m,d)
        max_force_excess=max(max_force_excess,float(np.maximum(m.actuator_forcerange[:,0]-d.actuator_force,d.actuator_force-m.actuator_forcerange[:,1]).max()))
        violation=np.maximum(m.jnt_range[sim.joints,0]-d.qpos[m.jnt_qposadr[sim.joints]],d.qpos[m.jnt_qposadr[sim.joints]]-m.jnt_range[sim.joints,1]);max_joint_violation=max(max_joint_violation,float(violation.max()))
        external_max=max(external_max,float(abs(d.xfrc_applied).max()),float(abs(d.qfrc_applied).max()))
        if step%10==0:
            up=d.body('torso_link').xmat.reshape(3,3)[:,2];loads=np.zeros(2);badcontact=0.;selfpenetration=0.
            for i,contact in enumerate(d.contact[:d.ncon]):
                bodies=m.geom_bodyid[contact.geom]
                if 0 in bodies:
                    other=max(bodies)
                    if other in sim.feet:
                        wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench);loads[sim.feet.index(other)]+=wrench[0]
                    else:badcontact=max(badcontact,-float(contact.dist))
                else:selfpenetration=max(selfpenetration,-float(contact.dist))
            row=dict(time_s=float(d.time),root=d.qpos[:3].tolist(),velocity=d.qvel[:3].tolist(),tilt_deg=float(np.rad2deg(np.arccos(np.clip(up[2],-1,1)))),foot_positions=d.xpos[sim.feet].tolist(),foot_loads_N=loads.tolist(),command=command.tolist(),phase_amplitude=phase_amplitude,controller='stationary stance' if stance is not None else 'H1 walking policy',stance_status=stance_status,joint_violation_rad=float(max(0,violation.max())),ground_collision_m=badcontact,self_penetration_m=selfpenetration,finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),motor_force_Nm=d.actuator_force.tolist())
            rows.append(row);poses.append(d.qpos.copy());vel.append(d.qvel.copy());controls.append(d.ctrl.copy())
            (args.output/'latest.json').write_text(json.dumps(row)+'\n')
            if step%500==0:emit({k:row[k] for k in ('time_s','root','tilt_deg','foot_loads_N','command')})
            if row['tilt_deg']>35 or d.qpos[2]<.55 or not row['finite']:break
    warnings=sum(int(d.warning[w].number) for w in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC,mujoco.mjtWarning.mjWARN_BADCTRL))
    tail=[r for r in rows if r['time_s']>args.seconds-1.]
    net_distance=float(np.linalg.norm(np.array(rows[-1]['root'][:2])-start[:2]));feet_lift=[any(r['foot_loads_N'][i]<10 and r['foot_positions'][i][2]>initial_feet[i,2]+.015 for r in rows) for i in range(2)]
    checks=dict(full_duration=rows[-1]['time_s']>=args.seconds-.025,upright=all(r['tilt_deg']<12 and r['root'][2]>.7 for r in rows),motor_limits=max_force_excess<1e-5,joint_limits=max_joint_violation<.02,no_nonfoot_ground_collision=all(r['ground_collision_m']<.003 for r in rows),self_collision=all(r['self_penetration_m']<.003 for r in rows),no_external_force=external_max==0,finite=all(r['finite'] for r in rows) and warnings==0,plant_unchanged=all(np.array_equal(v,getattr(m,k)) for k,v in originals.items()),moves_forward=args.mode=='stand' or net_distance>max(.5,abs(args.speed)*4),both_feet_swing=args.mode=='stand' or all(feet_lift),stops=args.mode=='forward' or bool(tail and max(np.linalg.norm(r['velocity'][:2]) for r in tail)<.15))
    if args.mode=='forward':checks.pop('stops')  # No stop was requested; do not label it a passed stop.
    checks={k:bool(v) for k,v in checks.items()}
    report=dict(scope=__doc__,passed=all(checks.values()),checks=checks,duration_s=rows[-1]['time_s'],net_distance_m=net_distance,max_tilt_deg=max(r['tilt_deg'] for r in rows),max_joint_violation_rad=max_joint_violation,max_motor_limit_excess_Nm=max_force_excess,max_self_penetration_m=max(r['self_penetration_m'] for r in rows),final_second_max_speed_m_s=max([np.linalg.norm(r['velocity'][:2]) for r in tail],default=None),both_feet_lifted=feet_lift,numerical_warnings=warnings,runtime_root_pose_writes=0,wall_seconds=time.time()-began,policy=dict(upstream_revision=UPSTREAM_REVISION,sha256=POLICY_SHA256,observation_count=41,action_count=10,period_s=.02),robot_audit=json.loads(args.robot.with_suffix('.audit.json').read_text()),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(args.output/'trace.json').write_text(json.dumps(rows)+'\n');np.savez_compressed(args.output/'trajectory.npz',qpos=poses,qvel=vel,ctrl=controls)
    emit({k:v for k,v in report.items() if k not in ('robot_audit','arguments')});pipeline.update(result_passed=report['passed'],stage='Locomotion checks passed' if report['passed'] else 'Locomotion failed; inspect report and trace');(args.output/'pipeline.json').write_text(json.dumps(pipeline)+'\n');emit('LOCOMOTION_TRIAL_FINISHED')
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Test an uncut H1 walk/stop/lower/hold/rise/restart sequence under native motors."""
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from probe_locomotion import make_plant
from doorbench.dexterous.locomotion import H1WalkingPolicy,DEFAULT_ANGLES
from doorbench.dexterous.locomotion_posture import posture_offset,transition_command
from doorbench.dexterous.provenance import capture
from doorbench.dexterous.stance import StanceController,PoseStanceController
from doorbench.dexterous.locomotion_manipulation import LandedFootStanceController
from doorbench.dexterous.locomotion_posture import minimum_jerk


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--controller',choices=('posture-offset','qp','pose-ik','landed-qp'),default='landed-qp')
    p.add_argument('--settle-foot-reference',action='store_true',help='Accept physically changed foot references during lowering/rising, then hold the landed frames')
    p.add_argument('--yaw',type=float,default=0.)
    p.add_argument('--landing-hip-spread',type=float,default=.07,help='Motor/proprioception hip-roll offset during final approach steps; meters are not teleported')
    p.add_argument('--knee-delta',type=float,default=1.03);p.add_argument('--seconds',type=float,default=37.);p.add_argument('--seed',type=int,default=0);p.add_argument('--joint-noise',type=float,default=0.)
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Use a fresh output directory')
    capture(Path(__file__).resolve().parents[2],args.output,{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},timing='before_native_walk_stance_transition')
    sim=make_plant(args.robot,seed=args.seed,joint_noise=args.joint_noise);m,d=sim.m,sim.d;a=sim.adapter;d.qpos[3:7]=[np.cos(args.yaw/2),0.,0.,np.sin(args.yaw/2)];mujoco.mj_forward(m,d);policy=H1WalkingPolicy(args.checkpoint)
    fixed=d.ctrl.copy();target=DEFAULT_ANGLES.copy();rows=[];poses=[];vel=[];controls=[];start=d.qpos[:3].copy();zero_since=None;ever_moved=False;max_violation=0.;max_force_excess=0.;external=0.;began=time.time();max_ground=0.;max_self=0.;max_tilt=0.;min_height=np.inf;all_finite=True
    immutable=('body_mass','body_inertia','body_gravcomp','jnt_range','dof_damping','dof_armature','dof_frictionloss','geom_contype','geom_conaffinity','geom_friction','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange');originals={n:getattr(m,n).copy() for n in immutable}
    def emit(value):
        line=json.dumps(value);print(line,flush=True)
        with (args.output/'run.log').open('a') as stream:stream.write(line+'\n')
    stance=None;stance_command=None;stance_status='not active';stance_initial_height=None;stance_failures=0;policy_was_active=True
    for step in range(round(args.seconds/m.opt.timestep)):
        if step%10==0:
            stage,command=transition_command(d.time)
            if np.linalg.norm(command)>0:ever_moved=True;zero_since=None
            elif zero_since is None:zero_since=d.time
            amplitude=max(0.,1.-(d.time-zero_since)) if ever_moved and zero_since is not None else 1.
            offset,offset_velocity=posture_offset(d.time,knee_delta=args.knee_delta)
            if args.controller!='posture-offset':offset[:]=0.;offset_velocity[:]=0.
            widen,widen_v=minimum_jerk(d.time,5.,3.)
            release,release_v=minimum_jerk(d.time,27.,3.)
            lateral=np.array([0.,1.,0.,0.,0.,0.,-1.,0.,0.,0.])*args.landing_hip_spread
            offset+=lateral*(widen-release);offset_velocity+=lateral*(widen_v-release_v)
            policy_active=args.controller=='posture-offset' or d.time<14 or d.time>=27
            if policy_active:
                if not policy_was_active:policy.reset()
                target=policy.step(d.qpos[a.qadr]-offset,d.qvel[a.vadr]-offset_velocity,d.qvel[3:6],d.body('pelvis').xmat.reshape(3,3).T@[0.,0.,-1.],command,d.time,phase_amplitude=amplitude)+offset
            policy_was_active=policy_active
        d.ctrl[:]=fixed;d.ctrl[a.actuators]=a.command(d,target)
        if args.controller!='posture-offset' and 14<=d.time<27:
            if stance is None:
                stance={'qp':StanceController,'pose-ik':PoseStanceController,'landed-qp':LandedFootStanceController}[args.controller](sim);stance_initial_height=float(stance.target_root[2])
            down,_=minimum_jerk(d.time,14.,4.);up,_=minimum_jerk(d.time,22.,4.)
            stance.target_root[2]=stance_initial_height+(0.87-stance_initial_height)*(down-up)
            if step%5==0:
                if args.settle_foot_reference and args.controller=='landed-qp' and (d.time<18 or 22<=d.time<26):stance.accept_physical_foot_repositioning()
                stance_command,stance_status=stance.command()
                if stance_command is None:stance_failures+=1
            if stance_command is not None:d.ctrl[stance.act]=np.clip(stance_command,m.actuator_ctrlrange[stance.act,0],m.actuator_ctrlrange[stance.act,1])
        mujoco.mj_step(m,d)
        max_force_excess=max(max_force_excess,float(np.maximum(m.actuator_forcerange[:,0]-d.actuator_force,d.actuator_force-m.actuator_forcerange[:,1]).max()))
        violation=np.maximum(m.jnt_range[sim.joints,0]-d.qpos[m.jnt_qposadr[sim.joints]],d.qpos[m.jnt_qposadr[sim.joints]]-m.jnt_range[sim.joints,1]);max_violation=max(max_violation,float(violation.max()));external=max(external,float(abs(d.xfrc_applied).max()),float(abs(d.qfrc_applied).max()))
        # Safety is audited every native 2 ms step; diagnostic traces are 20 ms.
        up=d.body('torso_link').xmat.reshape(3,3)[:,2];loads=np.zeros(2);ground=0.;selfpenetration=0.
        tilt=float(np.rad2deg(np.arccos(np.clip(up[2],-1,1))))
        for i,contact in enumerate(d.contact[:d.ncon]):
            bodies=m.geom_bodyid[contact.geom]
            if 0 in bodies:
                other=max(bodies)
                if other in sim.feet:
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench);loads[sim.feet.index(other)]+=wrench[0]
                else:ground=max(ground,-float(contact.dist))
            else:selfpenetration=max(selfpenetration,-float(contact.dist))
        max_ground=max(max_ground,ground);max_self=max(max_self,selfpenetration);max_tilt=max(max_tilt,tilt);min_height=min(min_height,float(d.qpos[2]));all_finite=all_finite and bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        if step%10==0:
            row=dict(time_s=float(d.time),stage=stage,stance_status=stance_status,root=d.qpos[:3].tolist(),velocity=d.qvel[:3].tolist(),tilt_deg=float(np.rad2deg(np.arccos(np.clip(up[2],-1,1)))),foot_loads_N=loads.tolist(),foot_positions=d.xpos[sim.feet].tolist(),phase_amplitude=amplitude,posture_offset=offset.tolist(),ground_collision_m=ground,self_penetration_m=selfpenetration,joint_violation_rad=max(0,float(violation.max())),finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()))
            rows.append(row);poses.append(d.qpos.copy());vel.append(d.qvel.copy());controls.append(d.ctrl.copy());(args.output/'latest.json').write_text(json.dumps(row)+'\n')
            if step%500==0:emit({k:row[k] for k in ('time_s','stage','root','tilt_deg','foot_loads_N')})
            if row['tilt_deg']>35 or d.qpos[2]<.55 or not row['finite']:break
    hold=[r for r in rows if 20<=r['time_s']<22];tail=[r for r in rows if r['time_s']>args.seconds-1]
    warnings=sum(int(d.warning[w].number) for w in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC,mujoco.mjtWarning.mjWARN_BADCTRL))
    stance_rows=[r for r in rows if 14<=r['time_s']<27]
    foot_positions=np.array([r['foot_positions'] for r in stance_rows]);foot_displacement=np.linalg.norm(foot_positions-foot_positions[0],axis=2).max(0).tolist() if len(stance_rows) else None
    hold_feet=np.array([r['foot_positions'] for r in hold]);hold_foot_excursion=float(np.linalg.norm(hold_feet-hold_feet[0],axis=2).max()) if hold else None
    checks=dict(full_duration=rows[-1]['time_s']>=args.seconds-.025,upright=max_tilt<12 and min_height>.7,motor_limits=max_force_excess<1e-5,joint_limits=max_violation<.02,no_ground_collision=max_ground<.003,no_self_collision=max_self<.003,no_external_force=external==0,finite=all_finite and warnings==0,stance_solver=stance_failures==0,plant_unchanged=all(np.array_equal(v,getattr(m,n)) for n,v in originals.items()),low_stance=bool(hold and all(.83<r['root'][2]<.91 for r in hold)),quiet_low_stance=bool(hold and max(np.linalg.norm(r['velocity'][:2]) for r in hold)<.03),quiet_low_feet=bool(hold and hold_foot_excursion<.01 and min(min(r['foot_loads_N']) for r in hold)>30),stopped_again=bool(tail and max(np.linalg.norm(r['velocity'][:2]) for r in tail)<.02))
    report=dict(scope=__doc__,passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},duration_s=rows[-1]['time_s'],safety_audit_period_s=float(m.opt.timestep),trace_period_s=float(m.opt.timestep*10),max_tilt_deg=max_tilt,max_nonfoot_ground_penetration_m=max_ground,max_self_penetration_m=max_self,max_joint_violation_rad=max_violation,max_motor_limit_excess_Nm=max_force_excess,low_stance_height_m=[min(r['root'][2] for r in hold),max(r['root'][2] for r in hold)] if hold else None,low_stance_max_speed_m_s=max([np.linalg.norm(r['velocity'][:2]) for r in hold],default=None),stance_max_foot_displacement_m=foot_displacement,low_stance_max_foot_excursion_m=hold_foot_excursion,low_stance_minimum_foot_load_N=min([min(r['foot_loads_N']) for r in hold],default=None),final_max_speed_m_s=max([np.linalg.norm(r['velocity'][:2]) for r in tail],default=None),runtime_root_pose_writes=0,stance_solver_failures=stance_failures,wall_seconds=time.time()-began,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(args.output/'trace.json').write_text(json.dumps(rows)+'\n');np.savez_compressed(args.output/'trajectory.npz',qpos=poses,qvel=vel,ctrl=controls);emit(report)
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()

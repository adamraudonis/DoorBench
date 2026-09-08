#!/usr/bin/env python3
"""Physically approach a fixed grasp-body waypoint in the complete native door scene."""
import argparse,json,time
from pathlib import Path
import mujoco,numpy as np
from doorbench.dexterous.locomotion import H1WalkingPolicy,DEFAULT_ANGLES
from doorbench.dexterous.locomotion_approach import make_door_approach,WaypointApproach,wrap_angle
from doorbench.dexterous.provenance import capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','reference','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    for name,default in [('distance',.7),('lateral',0.),('yaw-offset',0.),('joint-noise',0.),('seconds',25.),('gain',4.),('brake-prediction',.4),('brake-radius',.05),('max-speed',.3),('brake-velocity-window',0.)]:p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--precision-stance',action='store_true');p.add_argument('--seed',type=int,default=0);args=p.parse_args()
    if args.output.exists():raise SystemExit('Use a fresh output directory')
    if not np.isfinite([args.distance,args.lateral,args.yaw_offset,args.joint_noise,args.seconds,args.gain,args.brake_prediction,args.brake_radius,args.max_speed]).all() or args.distance<.5 or args.seconds<=1 or args.joint_noise<0:raise SystemExit('Invalid finite separated-start configuration')
    capture(Path(__file__).resolve().parents[2],args.output,{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},timing='before_native_door_waypoint_approach')
    (args.output/'reference.json').write_bytes(args.reference.read_bytes())
    sim,goal,yaw_goal=make_door_approach(args.door,args.robot,args.reference,distance=args.distance,lateral=args.lateral,yaw_offset=args.yaw_offset,seed=args.seed,joint_noise=args.joint_noise)
    m,d=sim.m,sim.d;a=sim.adapter;q=sim.root_qadr;v=sim.root_vadr;policy=H1WalkingPolicy(args.checkpoint)
    teacher=WaypointApproach(goal,yaw_goal,gain=args.gain,brake_prediction=args.brake_prediction,brake_radius=args.brake_radius,max_speed=args.max_speed,brake_velocity_window=args.brake_velocity_window)
    if not np.isclose(m.opt.timestep,.002):raise ValueError('Require the native 2 ms timestep')
    fixed=d.ctrl.copy();target=DEFAULT_ANGLES.copy();start=d.qpos[q:q+3].copy();rows=[];poses=[];vel=[];ctrl=[];began=time.time()
    robot_bodies={b for b in range(m.nbody) if m.body(b).name.startswith('robot/')};robot_v=np.r_[np.arange(v,v+6),sim.vadr];door_act=np.array([i for i in range(m.nu) if i not in sim.actuators],int)
    passive_tendons=[]
    for tendon in range(m.ntendon):
        if not m.tendon(tendon).name.startswith('robot/') or not m.tendon_limited[tendon]:continue
        terms=[]
        for k in range(m.tendon_adr[tendon],m.tendon_adr[tendon]+m.tendon_num[tendon]):
            if m.wrap_type[k]!=mujoco.mjtWrap.mjWRAP_JOINT:raise ValueError('Audit non-joint passive tendon before use')
            joint=int(m.wrap_objid[k])
            if m.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_HINGE:raise ValueError('Audit non-angular passive tendon units')
            terms.append((int(m.jnt_qposadr[joint]),float(m.wrap_prm[k])))
        passive_tendons.append((tendon,terms))
    immutable=('tendon_limited','tendon_range','tendon_stiffness','tendon_damping','tendon_solref_lim','tendon_solimp_lim','wrap_objid','wrap_prm','body_mass','body_inertia','body_gravcomp','jnt_range','dof_damping','dof_armature','dof_frictionloss','geom_contype','geom_conaffinity','geom_friction','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange');originals={n:getattr(m,n).copy() for n in immutable}
    worst={'passive_tendon_violation_rad':0.,'tilt_deg':0.,'joint_violation_rad':0.,'motor_excess_Nm':0.,'nonfoot_ground_penetration_m':0.,'scene_penetration_m':0.,'self_penetration_m':0.,'robot_external_force':0.,'door_motor_command':0.};min_height=np.inf;finite=True;collision_pairs=set();contact_steps=0
    def emit(value):
        line=json.dumps(value);print(line,flush=True)
        with (args.output/'run.log').open('a') as out:out.write(line+'\n')
    stance=None;stance_command=None;stance_started=None;stance_start_xy=None;stance_start_feet=None;stance_failures=0;stance_max_foot_displacement=0.;loads=np.zeros(2)
    for step in range(round(args.seconds/m.opt.timestep)):
        R=d.xmat[sim.pelvis].reshape(3,3);yaw=float(np.arctan2(R[1,0],R[0,0]))
        if step%10==0 and stance is None:
            command,amplitude,stage=teacher.step(d.qpos[q:q+2],yaw,d.qvel[v:v+2],d.time)
            target=policy.step(d.qpos[a.qadr],d.qvel[a.vadr],d.qvel[v+3:v+6],R.T@[0.,0.,-1.],command,d.time,phase_amplitude=amplitude)
        d.ctrl[:]=fixed;d.ctrl[a.actuators]=a.command(d,target)
        if args.precision_stance and stance is None and teacher.stop_time is not None and d.time-teacher.stop_time>=3. and np.linalg.norm(d.qvel[v:v+2])<.02 and min(loads)>30:
            from doorbench.dexterous.locomotion_manipulation import LandedFootStanceController
            stance=LandedFootStanceController(sim);stance_started=float(d.time);stance_start_xy=stance.target_root[:2].copy();stance_start_feet=d.xpos[sim.feet].copy()
        if stance is not None:
            stage='motor stance refinement';command=np.zeros(3);amplitude=0.
            blend=np.clip((d.time-stance_started)/3.,0.,1.);blend=blend**3*(10+blend*(-15+6*blend))
            stance.target_root[:2]=stance_start_xy+blend*(goal-stance_start_xy)
            if step%10==0:
                proposed,status=stance.command()
                if proposed is None:stance_failures+=1
                else:stance_command=proposed
            if stance_command is not None:d.ctrl[stance.act]=np.clip(stance_command,m.actuator_ctrlrange[stance.act,0],m.actuator_ctrlrange[stance.act,1])
            stance_max_foot_displacement=max(stance_max_foot_displacement,float(np.linalg.norm(d.xpos[sim.feet]-stance_start_feet,axis=1).max()))
        worst['robot_external_force']=max(worst['robot_external_force'],float(abs(d.xfrc_applied[list(robot_bodies)]).max()),float(abs(d.qfrc_applied[robot_v]).max()))
        sim.plant.step()
        worst['robot_external_force']=max(worst['robot_external_force'],float(abs(sim.plant.last_applied_qfrc[robot_v]).max()))
        worst['door_motor_command']=max(worst['door_motor_command'],float(abs(d.ctrl[door_act]).max()) if len(door_act) else 0.)
        tilt=float(np.rad2deg(np.arccos(np.clip(d.body('robot/torso_link').xmat.reshape(3,3)[2,2],-1,1))));worst['tilt_deg']=max(worst['tilt_deg'],tilt);min_height=min(min_height,float(d.qpos[q+2]))
        worst['motor_excess_Nm']=max(worst['motor_excess_Nm'],float(np.maximum(m.actuator_forcerange[sim.actuators,0]-d.actuator_force[sim.actuators],d.actuator_force[sim.actuators]-m.actuator_forcerange[sim.actuators,1]).max()))
        violation=np.maximum(m.jnt_range[sim.joints,0]-d.qpos[sim.qadr],d.qpos[sim.qadr]-m.jnt_range[sim.joints,1]);worst['joint_violation_rad']=max(worst['joint_violation_rad'],float(violation.max()))
        for tendon,terms in passive_tendons:
            length=sum(coefficient*d.qpos[adr] for adr,coefficient in terms)
            worst['passive_tendon_violation_rad']=max(worst['passive_tendon_violation_rad'],float(max(m.tendon_range[tendon,0]-length,length-m.tendon_range[tendon,1])))
        loads=np.zeros(2);touches=[]
        for i,c in enumerate(d.contact[:d.ncon]):
            bodies=list(m.geom_bodyid[c.geom]);robot=[b in robot_bodies for b in bodies]
            if not any(robot):continue
            penetration=max(0.,-float(c.dist));names=[m.geom(g).name for g in c.geom]
            if all(robot):worst['self_penetration_m']=max(worst['self_penetration_m'],penetration)
            else:
                rb=bodies[robot.index(True)];othergeom=int(c.geom[robot.index(False)])
                if m.geom(othergeom).name=='floor':
                    if rb in sim.feet:
                        wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench);loads[sim.feet.index(rb)]+=wrench[0]
                    else:worst['nonfoot_ground_penetration_m']=max(worst['nonfoot_ground_penetration_m'],penetration)
                else:
                    worst['scene_penetration_m']=max(worst['scene_penetration_m'],penetration)
                    if c.dist<0:touches.append(names);collision_pairs.add(tuple(names))
        contact_steps+=bool(touches);finite=finite and bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        if step%10==0 or step==round(args.seconds/m.opt.timestep)-1 or not finite or tilt>35 or d.qpos[q+2]<.55:
            R=d.xmat[sim.pelvis].reshape(3,3);yaw=float(np.arctan2(R[1,0],R[0,0]))
            row={'time_s':float(d.time),'stage':stage,'root':d.qpos[q:q+3].tolist(),'yaw_rad':yaw,'position_error_m':float(np.linalg.norm(goal-d.qpos[q:q+2])),'heading_error_deg':abs(float(np.rad2deg(wrap_angle(yaw_goal-yaw)))),'velocity':d.qvel[v:v+3].tolist(),'tilt_deg':tilt,'foot_loads_N':loads.tolist(),'foot_positions':d.xpos[sim.feet].tolist(),'command':command.tolist(),'phase_amplitude':amplitude,'door_q':float(sim.plant._door_q()),'scene_contacts':touches}
            rows.append(row);poses.append(d.qpos.copy());vel.append(d.qvel.copy());ctrl.append(d.ctrl.copy());(args.output/'latest.json').write_text(json.dumps(row)+'\n')
            if step%500==0:emit(row)
        if not finite or tilt>35 or d.qpos[q+2]<.55:break
    tail=[r for r in rows if r['time_s']>args.seconds-1];warnings=sum(int(d.warning[w].number) for w in (mujoco.mjtWarning.mjWARN_BADQPOS,mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC,mujoco.mjtWarning.mjWARN_BADCTRL))
    tail_speed=max([np.linalg.norm(r['velocity'][:2]) for r in tail],default=np.inf);tail_excursion=max([np.linalg.norm(np.array(r['root'][:2])-tail[0]['root'][:2]) for r in tail],default=np.inf);tail_load=min([min(r['foot_loads_N']) for r in tail],default=0.)
    checks={'precision_handoff':not args.precision_stance or stance is not None,'stance_solver':stance_failures==0,'full_duration':rows[-1]['time_s']>=args.seconds-.025,'separated_start':np.linalg.norm(start[:2]-goal)>=.5-1e-9,'upright':worst['tilt_deg']<12 and min_height>.7,'motor_limits':worst['motor_excess_Nm']<1e-5,'joint_limits':worst['joint_violation_rad']<.02,'passive_tendon_limits':worst['passive_tendon_violation_rad']<.02,'no_nonfoot_ground_collision':worst['nonfoot_ground_penetration_m']<.003,'no_self_collision':worst['self_penetration_m']<.003,'no_scene_contact':contact_steps==0,'no_robot_external_force':worst['robot_external_force']==0,'no_door_command':worst['door_motor_command']==0,'plant_unchanged':all(np.array_equal(value,getattr(m,name)) for name,value in originals.items()),'finite':finite and warnings==0,'position_accuracy':bool(tail and max(r['position_error_m'] for r in tail)<.03),'heading_accuracy':bool(tail and max(r['heading_error_deg'] for r in tail)<2.),'quiet_stop':bool(tail_speed<.02 and tail_excursion<.01 and tail_load>30)}
    report={'scope':__doc__,'precision_stance_started_s':stance_started,'precision_stance_failures':stance_failures,'precision_max_foot_displacement_m':stance_max_foot_displacement,'limited_robot_tendons':[m.tendon(t).name for t,_ in passive_tendons],'passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'worst':worst,'goal_xy':goal.tolist(),'goal_yaw_rad':yaw_goal,'start_root':start.tolist(),'last':rows[-1],'final_second_max_speed_m_s':float(tail_speed),'final_second_excursion_m':float(tail_excursion),'final_second_min_foot_load_N':float(tail_load),'scene_contact_steps':contact_steps,'scene_contact_pairs':sorted(collision_pairs),'stop_attempts':teacher.stops,'final_aim_offset_m':teacher.aim_offset.tolist(),'runtime_root_pose_writes':0,'safety_audit_period_s':float(m.opt.timestep),'trace_period_s':.02,'wall_seconds':time.time()-began,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(args.output/'trace.json').write_text(json.dumps(rows)+'\n');np.savez_compressed(args.output/'trajectory.npz',qpos=poses,qvel=vel,ctrl=ctrl);emit(report);sim.close()
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()

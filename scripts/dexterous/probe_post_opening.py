#!/usr/bin/env python3
"""Initialize from an attained Door55 terminal and test release/stow/rise only."""
import argparse,gzip,hashlib,inspect,json,shutil,time
from pathlib import Path
import mujoco,numpy as np
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import native_grasp_sample,audit_grasp_steps
from doorbench.dexterous.post_opening import plan_stow,StowRiseController


def hand_loads(m,d):
    count=0;load=0.
    for i,c in enumerate(d.contact[:d.ncon]):
        bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
        if any(n.startswith('robot/lh_') for n in bodies):
            if c.dist<=0:count+=1
            force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);load+=max(0.,float(force[0]))
    return count,load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('robot','door','motors','initial_trajectory','body_reset','output'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--seconds',type=float,default=34.);p.add_argument('--phase-seconds',type=float,default=4.);p.add_argument('--arm-gain',type=float,default=10.);p.add_argument('--no-rise',action='store_true');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use fresh output directory')
    if not a.no_rise and a.checkpoint is None:raise ValueError('Rise requires the audited H1 stabilization checkpoint')
    a.output.mkdir(parents=True);arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};(a.output/'arguments.json').write_text(json.dumps(arguments,indent=2)+'\n')
    files={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in vars(a).items() if isinstance(v,Path) and v.is_file()}
    for path in (Path(__file__),Path(inspect.getfile(StowRiseController)),Path(inspect.getfile(native_grasp_sample))):files[path.name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest());shutil.copy2(path,a.output/path.name)
    files['door.xml']=dict(path=str(a.door/'door.xml'),sha256=hashlib.sha256((a.door/'door.xml').read_bytes()).hexdigest())
    motors=json.loads(a.motors.read_text())
    if motors.get('source_xml_sha256')!=files['robot']['sha256'] or motors.get('hand_mechanics_profile')!='shadow-loopback-v2':raise ValueError('Expected exact corrected robot/motor contract')
    sim=DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()));sim.reset(randomize=False,images=False);m,d=sim.m,sim.d
    state=np.load(a.initial_trajectory)
    if state['terminal_qpos'].shape!=(m.nq,) or state['terminal_qvel'].shape!=(m.nv,) or state['terminal_ctrl'].shape!=(m.nu,):raise ValueError('Terminal state does not match actual plant')
    d.qpos[:]=state['terminal_qpos'];d.qvel[:]=state['terminal_qvel'];d.ctrl[:]=state['terminal_ctrl'];d.time=0.
    # Reset-only explicit-force adapter preserves original force caps and all
    # masses, joints, geometry and passive dynamics. Runtime emits motors only.
    aids=np.array([m.actuator('robot/'+v['name']).id for v in motors['actuators']]);caps=np.array([v['force_range'] for v in motors['actuators']])
    m.actuator_gainprm[aids,0]=1.;m.actuator_biasprm[aids,:3]=0.;m.actuator_ctrlrange[aids]=caps;mujoco.mj_forward(m,d)
    source_manifest=json.loads((a.initial_trajectory.parent/'manifest.json').read_text())
    if source_manifest['inputs']['robot']['sha256']!=files['robot']['sha256'] or source_manifest['inputs']['door']['door.xml']!=files['door.xml']['sha256']:raise ValueError('Attained trajectory source plant differs from continuation plant')
    source_time=float(state['terminal_time_s']);(a.output/'inputs.json').write_text(json.dumps(dict(files=files,source_terminal_time_s=source_time,scope='Initialized continuation; no uninterrupted opening/walk or sensor policy claim'),indent=2)+'\n')
    reset=json.loads(a.body_reset.read_text());began=time.time();plan=plan_stow(sim,reset)
    (a.output/'geometry-screen.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({k:v for k,v in plan.items() if k!='path_qpos' and k!='stage'}),flush=True)
    if not plan['passed'] or a.plan_only:sim.close();return 0 if plan['passed'] else 1
    controller=StowRiseController(sim,motors,plan,rise=not a.no_rise,phase_seconds=a.phase_seconds,arm_gain=a.arm_gain,checkpoint=a.checkpoint)
    immutable=('body_mass','body_inertia','body_gravcomp','jnt_range','dof_damping','dof_armature','dof_frictionloss','geom_contype','geom_conaffinity','geom_friction','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange');original={k:getattr(m,k).copy() for k in immutable}
    physics=[];traces=[];states={k:[] for k in ('qpos','qvel','ctrl')};actual=[]
    def sample(wrench=None):
        row=native_grasp_sample(sim,'leaf_handle_lever_col_n',handle_joint='leaf_handle_hinge',pre_step_external_wrench_max=wrench)
        count,load=hand_loads(m,d);row.update(left_hand_contacts=count,left_hand_load_N=load,measurement_pose_time_s=float(d.time),joint_state_time_s=float(d.time))
        limited=np.flatnonzero(m.jnt_limited);q=d.qpos[m.jnt_qposadr[limited]]
        row['root_xyz']=d.qpos[sim.root_qadr:sim.root_qadr+3].tolist();row['root_velocity']=d.qvel[sim.root_vadr:sim.root_vadr+3].tolist()
        row['all_joint_violation']=max(0.,float(np.maximum(m.jnt_range[limited,0]-q,q-m.jnt_range[limited,1]).max(initial=0)))
        return row
    physics.append(sample())
    try:
        for tick in range(round(a.seconds/m.opt.timestep)):
            count,load=hand_loads(m,d);force,info=controller.force(float(d.time),left_contacts=count,left_load=load)
            d.ctrl[aids]=force;pre=float(np.max(abs(d.xfrc_applied)));sim.plant.step();mujoco.mj_forward(m,d);row=sample(pre);row['controller']=info;physics.append(row)
            if tick%25==0:
                trace=dict(time_s=float(d.time),phase=info['phase'],root=d.qpos[sim.root_qadr:sim.root_qadr+7].tolist(),velocity=d.qvel[sim.root_vadr:sim.root_vadr+6].tolist(),door_q=row['door_q'],tilt=row['torso_tilt_deg'],left_hand_contacts=count,left_hand_load_N=load,arm_error=info['arm_motor_error_rad'],stance_status=info['stance_status']);traces.append(trace)
                for k in states:states[k].append(getattr(d,k).copy())
                actual.append(float(d.time));(a.output/'latest.json').write_text(json.dumps(trace)+'\n')
                if tick%500==0:print(json.dumps(trace),flush=True)
            if not row['finite'] or row['torso_tilt_deg']>35 or row['root_height_m']<.55:break
        end=float(d.time);report=audit_grasp_steps(physics,physics_dt=m.opt.timestep,expected_duration=end,require_contact_free_start=False)
        for k in ('closed_leaf_start','resting_operator_start','contact_free_start','sustained_pad_grasp'):report['checks'].pop(k)
        tail=[r for r in physics if r['sim_time_s']>=end-1.];tail_speed=max(np.linalg.norm(r['root_velocity'][:2]) for r in tail);tail_excursion=max(np.linalg.norm(np.asarray(r['root_xyz'][:2])-tail[0]['root_xyz'][:2]) for r in tail);completed=[x['completed'] for x in controller.events]
        report['checks'].update(full_requested_duration=abs(end-a.seconds)<m.opt.timestep/2,static_stow_screen=plan['passed'],left_released='release' in completed,left_contactfree_tail=all(r['left_hand_contacts']==0 and r['left_hand_load_N']<.1 for r in tail),arms_stowed=all(x in completed for x in ('left','right','torso')),risen_or_disabled=a.no_rise or 'rise' in completed,walking_stabilizer_started_or_disabled=a.no_rise or controller.walk_started is not None,quiet_terminal_body=tail_speed<.03 and tail_excursion<.03,stance_solver=controller.solver_failures==0,all_joints=all(r['all_joint_violation']<=.02 for r in physics),plant_unchanged=all(np.array_equal(v,getattr(m,k)) for k,v in original.items()))
        report.update(passed=all(report['checks'].values()),scope='Initialized continuation from exact attained qpos/qvel; release/stow/rise only, no passage or uninterrupted opening claim',events=controller.events,solver_failures=controller.solver_failures,walking_stabilizer_started_s=controller.walk_started,arguments=arguments,runtime_pose_writes=0,door_commands=False,physics_dt_s=m.opt.timestep,measurement_refresh='mj_forward after every physical step; no integrated-state writes',source_terminal_time_s=source_time,final_root=d.qpos[sim.root_qadr:sim.root_qadr+7].tolist(),final_door_q=physics[-1]['door_q'],final_second_max_horizontal_speed_m_s=float(tail_speed),final_second_horizontal_excursion_m=float(tail_excursion),maximum_tilt_deg=max(r['torso_tilt_deg'] for r in physics),max_nonfoot_penetration_m=max(r['max_nonfoot_penetration_m'] for r in physics),max_all_joint_violation_rad=max(r['all_joint_violation'] for r in physics),wall_seconds=time.time()-began,walking_blocked_reason='No corridor-qualified physical root alignment or measured aperture guarantee; only zero-velocity gait stabilization is requested, never a passage command')
        report['checks']={k:bool(v) for k,v in report['checks'].items()}
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    finally:
        (a.output/'trace.json').write_text(json.dumps(traces)+'\n')
        with gzip.open(a.output/'physics-steps.json.gz','wt') as f:json.dump(physics,f)
        np.savez_compressed(a.output/'trajectory.npz',**states,time_s=actual,terminal_qpos=d.qpos.copy(),terminal_qvel=d.qvel.copy(),terminal_ctrl=d.ctrl.copy(),terminal_time_s=float(d.time));sim.close()
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())

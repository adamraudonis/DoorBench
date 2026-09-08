#!/usr/bin/env python3
"""Live Isaac H1/dual-Shadow forward and phase-braking fixture on an empty plane.

The actor uses its own encoders, gyro and perfect projected gravity; it has no
vision/touch/door observations. This does not demonstrate doorway traversal.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def main():
    from isaaclab.app import AppLauncher
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot-usd','motors','checkpoint','reset','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=float,default=14.)
    p.add_argument('--record',action='store_true')
    AppLauncher.add_app_launcher_args(p)
    a=p.parse_args()
    a.enable_cameras=a.record
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    pipeline={'stage':'Initializing live Isaac H1 walking fixture',
        'completion_marker':'ISAAC_H1_WALKING_FINISHED','started_at_unix':time.time()}
    (out/'run.pid').write_text(str(os.getpid()))
    (out/'pipeline.json').write_text(json.dumps(pipeline)+'\n')
    def emit(value):
        line=value if isinstance(value,str) else json.dumps(value)
        print(line,flush=True)
        with (out/'run.log').open('a') as log:log.write(line+'\n')
    emit(pipeline)
    launcher=AppLauncher(a);app=launcher.app
    failed=False
    try:
        import numpy as np
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation,ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from pxr import Usd,UsdPhysics,UsdGeom,PhysxSchema
        from doorbench.dexterous.isaac_materials import bind_robot_contact_material,check_solver_materials,check_solver_offsets
        from doorbench.dexterous.sensor_contract import rotation_xyzw
        from doorbench.dexterous.isaac_sensors import all_scene_contact_paths
        from doorbench.dexterous.reset import check_joint_reset
        from doorbench.dexterous.locomotion import H1WalkingPolicy,JOINT_NAMES,DEFAULT_ANGLES,POLICY_SHA256
        motors=json.loads(a.motors.read_text());reset=json.loads(a.reset.read_text())
        dt=.002
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt,device=a.device,render_interval=20,
            physx=sim_utils.PhysxCfg(solver_type=1,min_position_iteration_count=32,min_velocity_iteration_count=8)))
        if getattr(sim,'_app_control_on_stop_handle',None) is not None:
            sim._app_control_on_stop_handle.unsubscribe();sim._app_control_on_stop_handle=None
        sim.carb_settings.set_bool('/physics/disableContactProcessing',False)
        stage=sim.stage
        root=UsdGeom.Xform.Define(stage,'/World/H1').GetPrim()
        root.GetReferences().AddReference(str(a.robot_usd.resolve()),'/H1')
        floor=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.7,dynamic_friction=.7,friction_combine_mode='max'))
        floor.func('/World/floor',floor)
        light=sim_utils.DomeLightCfg(intensity=1800.);light.func('/World/light',light)
        material_audit=bind_robot_contact_material(stage,'/World/H1',motors['contact_material'])
        roots=[];body_paths=[]
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                if prim.GetName()=='worldBody' and not any(x.HasAPI(UsdPhysics.RigidBodyAPI) for x in Usd.PrimRange(prim)):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                else:roots.append(str(prim.GetPath()))
            if prim.IsA(UsdPhysics.RevoluteJoint):
                drive=UsdPhysics.DriveAPI.Apply(prim,'angular');drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.)
                for schema in list(prim.GetAppliedSchemas()):
                    if 'Tendon' in schema:prim.RemoveAPI(getattr(PhysxSchema,schema.split(':')[0]),schema.split(':')[1])
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                rb.CreateDisableGravityAttr(False);rb.CreateMaxDepenetrationVelocityAttr(.5);rb.CreateMaxAngularVelocityAttr(5729.58)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
                body_paths.append(str(prim.GetPath()))
        assert len(roots)==1
        api=PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(roots[0]))
        api.CreateEnabledSelfCollisionsAttr(True);api.CreateSolverPositionIterationCountAttr(32);api.CreateSolverVelocityIterationCountAttr(8)
        effort_limits={name:0. for name in motors['joint_names']}
        for motor in motors['actuators']:
            for name,coefficient in motor['terms'].items():
                effort_limits[name]+=abs(coefficient)*max(abs(x) for x in motor['force_range'])
        robot=Articulation(ArticulationCfg(prim_path='/World/H1',spawn=None,
            articulation_root_prim_path=roots[0][len('/World/H1'):],actuators={'motors':ImplicitActuatorCfg(
                joint_names_expr=['.*'],stiffness=0.,damping=0.,effort_limit_sim=effort_limits)}))
        camera=None;writer=None
        if a.record:
            from isaaclab.sensors import Camera,CameraCfg
            import imageio.v2 as imageio
            camera=Camera(CameraCfg(prim_path='/World/Camera',update_period=0.,height=720,width=960,
                data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=18.,clipping_range=(.01,100.))))
        sim.reset();sim.carb_settings.set_bool('/physics/disableContactProcessing',False)
        offsets=robot.root_physx_view.get_contact_offsets()
        robot.root_physx_view.set_rest_offsets(torch.zeros_like(offsets),torch.tensor([0],dtype=torch.int32))
        robot.root_physx_view.set_contact_offsets(torch.full_like(offsets,.001),torch.tensor([0],dtype=torch.int32))
        material_audit.update(check_solver_offsets(robot.root_physx_view.get_contact_offsets().cpu().numpy(),robot.root_physx_view.get_rest_offsets().cpu().numpy()))
        material_audit.update(check_solver_materials(robot.root_physx_view.get_material_properties()[0].cpu().numpy(),motors['contact_material']))
        names=list(robot.joint_names);index={name:i for i,name in enumerate(names)}
        q0=np.array([reset['joints'][name] for name in names])
        limits=robot.root_physx_view.get_dof_limits()[0].cpu().numpy()
        check_joint_reset(names,q0,limits)
        robot.write_joint_state_to_sim(torch.tensor(q0[None],device=a.device,dtype=torch.float32),torch.zeros((1,len(names)),device=a.device))
        robot.write_root_pose_to_sim(torch.tensor([reset['initial_root']],device=a.device,dtype=torch.float32))
        robot.write_root_velocity_to_sim(torch.zeros((1,6),device=a.device))
        robot.write_joint_armature_to_sim(torch.tensor([[motors['passive'][name]['armature'] for name in names]],device=a.device))
        robot.write_joint_friction_coefficient_to_sim(torch.zeros((1,len(names)),device=a.device))
        matrix=np.zeros((len(motors['actuators']),len(names)))
        for i,motor in enumerate(motors['actuators']):
            for name,coefficient in motor['terms'].items():matrix[i,index[name]]=coefficient
        kp=np.array([m['kp'] for m in motors['actuators']]);bias=np.array([m['bias'] for m in motors['actuators']])
        force_range=np.array([m['force_range'] for m in motors['actuators']]);control_range=np.array([m['control_range'] for m in motors['actuators']])
        fixed=np.array([reset['motor_targets'][m['name']] for m in motors['actuators']])
        damp=np.array([motors['passive'][name]['damping'] for name in names]);friction=np.array([motors['passive'][name]['friction'] for name in names])
        leg_joints=np.array([index[name] for name in JOINT_NAMES])
        leg_motors=np.array([next(i for i,motor in enumerate(motors['actuators']) if motor['name']==name) for name in JOINT_NAMES])
        for ji,mi in zip(leg_joints,leg_motors):
            expected=np.zeros(len(names));expected[ji]=1.
            if not np.array_equal(matrix[mi],expected):raise ValueError('Walking leg motor transmission differs')
        actual_paths=all_scene_contact_paths(stage)
        contacts=sim.physics_sim_view.create_rigid_contact_view(body_paths,
            filter_patterns=[list(actual_paths) for _ in body_paths],max_contact_data_count=8192)
        contact_paths=list(contacts.sensor_paths);filters=np.array(contacts.filter_paths).reshape(len(contact_paths),-1)
        body_set=set(body_paths);feet=['left_ankle_link','right_ankle_link']
        foot_rows=[next(i for i,path in enumerate(contact_paths) if path.rsplit('/',1)[-1]==name) for name in feet]
        foot_bodies=[robot.body_names.index(name) for name in feet]
        if camera:
            camera.set_world_poses_from_view(eyes=torch.tensor([[3.,-5.,2.8]],device=a.device),targets=torch.tensor([[1.5,0.,1.]],device=a.device))
            writer=imageio.get_writer(out/'live-isaac.mp4',fps=25,codec='libx264',quality=8)
        policy=H1WalkingPolicy(a.checkpoint);target=DEFAULT_ANGLES.copy();ever_moved=False;zero_since=None
        invariant_getters={'mass':robot.root_physx_view.get_masses,
            'joint_limits':robot.root_physx_view.get_dof_limits,
            'effort_caps':robot.root_physx_view.get_dof_max_forces,
            'materials':robot.root_physx_view.get_material_properties,
            'contact_offsets':robot.root_physx_view.get_contact_offsets,
            'rest_offsets':robot.root_physx_view.get_rest_offsets}
        invariant_before={name:getter().cpu().numpy().copy() for name,getter in invariant_getters.items()}
        mass=float(robot.root_physx_view.get_masses().sum())
        if abs(mass-reset['mass_kg'])>1e-4:raise ValueError('Imported mass differs from native plant')
        source_paths=[Path(__file__),Path(__file__).resolve().parents[2]/'doorbench/dexterous/locomotion.py',a.motors,a.reset,a.robot_usd,a.checkpoint]
        (out/'manifest.json').write_text(json.dumps(dict(scope=__doc__,started_unix=time.time(),device=a.device,
            source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            dt=dt,policy_dt=.02,mass_kg=mass,checkpoint_sha256=POLICY_SHA256,material_audit=material_audit,
            runtime_pose_writes=0,external_wrenches=False),indent=2)+'\n')
        for path in source_paths[:2]:(out/('source-'+path.name)).write_bytes(path.read_bytes())
        pipeline['stage']='Live H1 walking and phase braking';(out/'pipeline.json').write_text(json.dumps(pipeline)+'\n')
        rows=[];max_limit=0.;max_force_excess=0.;max_motor_delivery_error=0.;max_self=0.;max_nonfoot=0.
        clock_origin=float(sim.current_time);begin=time.time();feet_initial=None
        for step in range(round(a.seconds/dt)):
            elapsed=step*dt
            q=robot.data.joint_pos[0].cpu().numpy();dq=robot.data.joint_vel[0].cpu().numpy();state=robot.data.root_state_w[0].cpu().numpy()
            if step%10==0:
                command=np.array([.4 if 1<=elapsed<9 else 0.,0.,0.])
                if np.linalg.norm(command)>0:ever_moved=True;zero_since=None
                elif zero_since is None:zero_since=elapsed
                amplitude=max(0.,1-(elapsed-zero_since)) if ever_moved and zero_since is not None else 1.
                rotation=rotation_xyzw([*state[4:7],state[3]])
                target=policy.step(q[leg_joints],dq[leg_joints],rotation.T@state[10:13],rotation.T@[0.,0.,-1.],command,elapsed,phase_amplitude=amplitude)
            lengths=matrix@q;speeds=matrix@dq
            servo_bias=bias[:,0]+bias[:,1]*lengths+bias[:,2]*speeds
            controls=fixed.copy()
            desired=np.clip(policy.torques(target,q[leg_joints],dq[leg_joints]),force_range[leg_motors,0],force_range[leg_motors,1])
            controls[leg_motors]=np.clip((desired-servo_bias[leg_motors])/kp[leg_motors],control_range[leg_motors,0],control_range[leg_motors,1])
            forces=np.clip(kp*controls+servo_bias,force_range[:,0],force_range[:,1])
            torque=matrix.T@forces-damp*dq-friction*np.tanh(dq/.001)
            robot.set_joint_effort_target(torch.tensor(torque[None],device=a.device,dtype=torch.float32));robot.write_data_to_sim()
            sim.step(render=False)
            if camera and step%20==0:sim.render()
            robot.update(dt)
            max_force_excess=max(max_force_excess,float(np.maximum(force_range[:,0]-forces,forces-force_range[:,1]).max()))
            sent=robot.root_physx_view.get_dof_actuation_forces()[0].cpu().numpy()
            max_motor_delivery_error=max(max_motor_delivery_error,float(abs(sent-torque).max()))
            qnew=robot.data.joint_pos[0].cpu().numpy()
            max_limit=max(max_limit,float(np.maximum(limits[:,0]-qnew,qnew-limits[:,1]).max()))
            if abs(float(sim.current_time)-clock_origin-(step+1)*dt)>1e-4:raise RuntimeError('Physics clock advanced outside motor steps')
            if step%10==0:
                force,point,normal,distance,count,start=[x.cpu().numpy().copy() for x in contacts.get_contact_data(dt)]
                if count.sum()>=8192:raise RuntimeError('Contact audit buffer exhausted')
                loads=np.zeros(2);nonfoot=0.;self_depth=0.
                for i,path in enumerate(contact_paths):
                    for j in range(count.shape[1]):
                        for k in range(int(start[i,j]),int(start[i,j]+count[i,j])):
                            depth=max(0.,-float(distance[k,0]));other=filters[i,j]
                            if other in body_set:self_depth=max(self_depth,depth)
                            elif i in foot_rows:loads[foot_rows.index(i)]+=float(force[k,0]*normal[k,2])
                            else:nonfoot=max(nonfoot,depth)
                max_self=max(max_self,self_depth);max_nonfoot=max(max_nonfoot,nonfoot)
                state=robot.data.root_state_w[0].cpu().numpy();up=-robot.data.projected_gravity_b[0].cpu().numpy()
                bodypose=robot.data.body_state_w[0].cpu().numpy();torso=bodypose[robot.body_names.index('torso_link')]
                rtorso=rotation_xyzw([*torso[4:7],torso[3]])
                feet_pose=bodypose[foot_bodies,:3]
                if feet_initial is None:feet_initial=feet_pose.copy()
                row=dict(time_s=(step+1)*dt,sim_time_s=float(sim.current_time)-clock_origin,root=state.tolist(),
                    tilt_deg=float(np.degrees(np.arccos(np.clip(rtorso[2,2],-1,1)))),foot_positions=feet_pose.tolist(),foot_loads_N=loads.tolist(),
                    command=command.tolist(),phase_amplitude=amplitude,joint_position=qnew.tolist(),joint_velocity=robot.data.joint_vel[0].cpu().tolist(),
                    motor_forces=forces.tolist(),joint_torque_command=torque.tolist(),joint_torque_delivered=sent.tolist(),
                    ground_collision_m=nonfoot,self_penetration_m=self_depth,
                    finite=bool(np.isfinite(state).all() and np.isfinite(qnew).all()))
                rows.append(row)
                if step%250==0:
                    (out/'progress.json').write_text(json.dumps({key:row[key] for key in ('time_s','root','tilt_deg','foot_loads_N','command')})+'\n')
                    emit({key:row[key] for key in ('time_s','tilt_deg','foot_loads_N','command')})
                if row['tilt_deg']>35 or state[2]<.55 or not row['finite']:break
            if camera and step%20==0:
                camera.update(dt*20);pixels=camera.data.output['rgb'][0].cpu().numpy()[...,:3];writer.append_data(pixels)
                if step%1000==0:imageio.imwrite(out/f'frame-{step:05d}.png',pixels)
        if writer:writer.close()
        (out/'trace.json').write_text(json.dumps(rows)+'\n')
        tail=[r for r in rows if r['time_s']>a.seconds-1]
        tail_speed=max([float(np.linalg.norm(r['root'][7:9])) for r in tail],default=float('inf'))
        excursion=max([float(np.linalg.norm(np.array(r['root'][:2])-tail[0]['root'][:2])) for r in tail],default=float('inf'))
        foot_load=min([min(r['foot_loads_N']) for r in tail],default=0.)
        distance=float(np.linalg.norm(np.array(rows[-1]['root'][:2])-reset['initial_root'][:2]))
        feet_swing=[any(r['foot_loads_N'][i]<10 and r['foot_positions'][i][2]>feet_initial[i,2]+.015 for r in rows) for i in range(2)]
        checks=dict(full_duration=rows[-1]['time_s']>=a.seconds-.025,upright=all(r['tilt_deg']<12 and r['root'][2]>.7 for r in rows),
            native_motor_caps=max_force_excess<1e-5,motor_delivery=max_motor_delivery_error<1e-4,joint_limits=max_limit<.02,
            no_self_penetration=max_self<.003,no_nonfoot_ground_penetration=max_nonfoot<.003,
            finite=all(r['finite'] for r in rows),moves_forward=distance>1.6,both_feet_swing=all(feet_swing),
            quiet_speed=tail_speed<.02,quiet_excursion=excursion<.01,both_feet_support=foot_load>30,
            physics_clock=max(abs(r['sim_time_s']-r['time_s']) for r in rows)<1e-4,
            plant_parameters_unchanged=all(np.array_equal(invariant_before[name],getter().cpu().numpy())
                for name,getter in invariant_getters.items()))
        report=dict(scope=__doc__,passed=all(checks.values()),checks=checks,duration_s=rows[-1]['time_s'],
            net_distance_m=distance,max_torso_tilt_deg=max(r['tilt_deg'] for r in rows),final_second_speed_m_s=tail_speed,
            final_second_excursion_m=excursion,final_second_min_foot_load_N=foot_load,max_joint_violation_rad=max_limit,
            max_self_penetration_m=max_self,max_nonfoot_penetration_m=max_nonfoot,max_motor_delivery_error_Nm=max_motor_delivery_error,
            runtime_root_pose_writes=0,external_wrenches=False,wall_seconds=time.time()-begin,
            limitation='Single plane development trial; contact penetration audited at 50 Hz, joint/motor limits at 500 Hz. Perfect attitude input; no vision/touch or door task.')
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');emit(report)
        failed=not report['passed']
    except BaseException:
        failed=True;(out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:app.close()
    pipeline.update(stage='Walking failed; inspect report/error' if failed else 'Walking checks passed',result_passed=not failed)
    (out/'pipeline.json').write_text(json.dumps(pipeline)+'\n')
    emit('ISAAC_H1_WALKING_FINISHED')
    if failed:raise SystemExit(1)


if __name__=='__main__':main()

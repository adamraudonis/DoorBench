#!/usr/bin/env python3
"""Isolated PhysX pose/velocity integration diagnostic, not a robot task.

One gravity-free rigid body and two identical two-foot articulations compare
unconstrained rotation against support contact. Articulations differ only by
1.736306 mm initial floor overlap. All pose/velocity writes occur at reset.
Solver variants are separate processes with one setting changed at a time.
"""
import argparse,hashlib,json,time,traceback
from pathlib import Path

VARIANTS={'baseline':dict(solver_type=1,velocity_iterations=8,stabilization=False),
    'with-stabilization':dict(solver_type=1,velocity_iterations=8,stabilization=True),
    'velocity-4':dict(solver_type=1,velocity_iterations=4,stabilization=False),
    'velocity-32':dict(solver_type=1,velocity_iterations=32,stabilization=False),
    'pgs':dict(solver_type=0,velocity_iterations=8,stabilization=False)}
for _settings in VARIANTS.values():_settings['position_iterations']=32
VARIANTS['position-4']={**VARIANTS['baseline'],'position_iterations':4}
DT=.002;DURATION=3.;OVERLAP=.0017363057918942104


def integrate_pose_errors(poses,velocities,dt):
    import numpy as np
    from scipy.spatial.transform import Rotation
    R=np.array([Rotation.from_quat(row[:,3:]).as_matrix() for row in poses])
    estimates={k:R[0].copy() for k in ('current','previous','trapezoid')};errors={k:[] for k in estimates};residual=[]
    for i in range(1,len(poses)):
        for key in estimates:
            alpha={'current':1.,'previous':0.,'trapezoid':.5}[key]
            omega=alpha*velocities[i,:,3:]+(1-alpha)*velocities[i-1,:,3:]
            estimates[key]=Rotation.from_rotvec(omega*dt).as_matrix()@estimates[key]
            errors[key].append(Rotation.from_matrix(estimates[key]@np.swapaxes(R[i],1,2)).as_rotvec())
        residual.append(Rotation.from_matrix(R[i]@np.swapaxes(R[i-1],1,2)).as_rotvec()-velocities[i,:,3:]*dt)
    return {k:np.asarray(v) for k,v in errors.items()},np.asarray(residual)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);settings=VARIANTS[a.variant]
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True});failed=True
    try:
        import numpy as np,torch
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,UsdShade
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        cfg=sim_utils.SimulationCfg(dt=DT,device='cuda:0',gravity=(0,0,-9.81),render_interval=1000,
            physx=sim_utils.PhysxCfg(solver_type=settings['solver_type'],min_position_iteration_count=settings['position_iterations'],
                min_velocity_iteration_count=settings['velocity_iterations'],enable_stabilization=settings['stabilization']))
        sim=sim_utils.SimulationContext(cfg)
        actual_dt=float(sim.get_physics_dt())
        if abs(actual_dt-DT)>1e-12:raise ValueError('Actual configured physics timestep differs')
        if getattr(sim,'_app_control_on_stop_handle',None) is not None:sim._app_control_on_stop_handle.unsubscribe();sim._app_control_on_stop_handle=None
        stage=sim.stage
        # Procedural static floor: no Nucleus/default-environment asset lookup.
        UsdGeom.Xform.Define(stage,'/World/ground')
        ground=UsdGeom.Cube.Define(stage,'/World/ground/shape');ground.CreateSizeAttr(1.)
        ground.AddTranslateOp().Set(Gf.Vec3d(0,0,-.05));ground.AddScaleOp().Set(Gf.Vec3f(10,10,.1));UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        material=UsdShade.Material.Define(stage,'/World/fixtureMaterial');mat=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat.CreateStaticFrictionAttr(.8);mat.CreateDynamicFrictionAttr(.7);mat.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material,materialPurpose='physics')
        paths=[];body_names=[]
        def body(path,pos,size,mass,inertia,collision=True,free=False):
            # Keep rigid-body/joint frames unscaled; only the child geometry is
            # scaled, so authored joint anchors have unambiguous SI coordinates.
            transform=UsdGeom.Xform.Define(stage,path);transform.AddTranslateOp().Set(Gf.Vec3d(*pos));prim=transform.GetPrim()
            cube=UsdGeom.Cube.Define(stage,path+'/shape');cube.CreateSizeAttr(1.);cube.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.RigidBodyAPI.Apply(prim);props=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            props.CreateDisableGravityAttr(free);props.CreateLinearDampingAttr(0.);props.CreateAngularDampingAttr(0.)
            props.CreateSleepThresholdAttr(0.);props.CreateMaxDepenetrationVelocityAttr(.5)
            props.CreateSolverPositionIterationCountAttr(settings['position_iterations']);props.CreateSolverVelocityIterationCountAttr(settings['velocity_iterations'])
            massapi=UsdPhysics.MassAPI.Apply(prim);massapi.CreateMassAttr(mass);massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
            if collision:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim());UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material,materialPurpose='physics')
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            paths.append(path);body_names.append(path.split('/World/')[1]);return prim
        body('/World/free',[0,0,1.5],[.2,.12,.1],1.,[.02,.02,.02],collision=False,free=True)
        free=RigidObject(RigidObjectCfg(prim_path='/World/free',spawn=None))
        robots=[];initial=[]
        for i,(label,overlap) in enumerate([('supported_clear',0.),('supported_overlap',OVERLAP)]):
            path='/World/'+label;x=1.+i;UsdGeom.Xform.Define(stage,path)
            root=body(path+'/root',[x,0,.60-overlap],[.24,.18,.25],8.,[.08,.08,.05])
            UsdPhysics.ArticulationRootAPI.Apply(root);art=PhysxSchema.PhysxArticulationAPI.Apply(root)
            art.CreateEnabledSelfCollisionsAttr(False);art.CreateSolverPositionIterationCountAttr(settings['position_iterations']);art.CreateSolverVelocityIterationCountAttr(settings['velocity_iterations']);art.CreateSleepThresholdAttr(0.)
            for side,y in [('left',.15),('right',-.15)]:
                foot=path+'/'+side+'_foot';body(foot,[x,y,.02-overlap],[.24,.08,.04],1.,[.001,.005,.005])
                joint=UsdPhysics.RevoluteJoint.Define(stage,path+'/'+side+'_ankle');joint.CreateAxisAttr('Y')
                joint.CreateBody0Rel().SetTargets([path+'/root']);joint.CreateBody1Rel().SetTargets([foot])
                joint.CreateLocalPos0Attr(Gf.Vec3f(0,y,-.55));joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,.03))
                joint.CreateLowerLimitAttr(-45.);joint.CreateUpperLimitAttr(45.)
                drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular');drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.)
            robots.append(Articulation(ArticulationCfg(prim_path=path,spawn=None,articulation_root_prim_path='/root',
                actuators={'fixture':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,effort_limit_sim=30.)})))
            initial.append([x,0,.60-overlap,1,0,0,0,0,0,0,.02,.01,.003])
        sim.reset()
        free_state=free.data.default_root_state.clone();free_state[0,:]=torch.tensor([0,0,1.5,1,0,0,0,0,0,0,.03,.01,.02],device=free.device)
        free.write_root_state_to_sim(free_state)
        for robot,state in zip(robots,initial):
            robot.write_root_state_to_sim(torch.tensor([state],dtype=torch.float32,device=robot.device))
            q=torch.zeros_like(robot.data.default_joint_pos);robot.write_joint_state_to_sim(q,q.clone());robot.update(DT)
        view=sim.physics_sim_view.create_rigid_body_view(paths)
        if set(view.prim_paths)!=set(paths):raise ValueError('Every fixture body must be measured')
        order=list(view.prim_paths);indices=[order.index(path) for path in paths]
        backend_properties={}
        for getter in ('get_angular_dampings','get_linear_dampings','get_max_angular_velocities','get_masses','get_inertias','get_coms'):
            if hasattr(view,getter):backend_properties[getter]=getattr(view,getter)().detach().cpu().numpy().copy()[indices].tolist()
            else:backend_properties[getter]={'unavailable':True}
        foot_paths=[path for path in paths if path.endswith('_foot')]
        ground_paths=[str(prim.GetPath()) for prim in stage.Traverse() if str(prim.GetPath()).startswith('/World/ground/') and prim.HasAPI(UsdPhysics.CollisionAPI)]
        if not ground_paths:raise ValueError('Actual floor collision paths required')
        contacts=sim.physics_sim_view.create_rigid_contact_view(foot_paths,filter_patterns=[ground_paths for _ in foot_paths],max_contact_data_count=256)
        if tuple(contacts.sensor_paths)!=tuple(foot_paths):raise ValueError('Foot contact row order differs')
        def read():
            # Copy each getter before the next backend call can reuse storage.
            pose=view.get_transforms().detach().cpu().numpy().copy()[indices]
            velocity=view.get_velocities().detach().cpu().numpy().copy()[indices]
            q=np.array([r.root_physx_view.get_dof_positions().detach().cpu().numpy()[0].copy() for r in robots])
            dq=np.array([r.root_physx_view.get_dof_velocities().detach().cpu().numpy()[0].copy() for r in robots])
            return pose,velocity,q,dq
        poses=[];velocities=[];qrows=[];dqrows=[];commands=[];floor_loads=[]
        def append():
            pose,velocity,q,dq=read();poses.append(pose);velocities.append(velocity);qrows.append(q);dqrows.append(dq)
        append();stage.GetRootLayer().Export(str(a.output/'fixture-initial.usda'))
        attr_terms=('solver','stabiliz','sleep','gravity','damping','depenet','mass','inertia','principalaxes','centerofmass','maxangularvelocity')
        authored={str(prim.GetPath()):{att.GetName():str(att.Get()) for att in prim.GetAttributes()
            if any(k in att.GetName().lower() for k in attr_terms)}
            for prim in stage.Traverse() if any(any(k in att.GetName().lower() for k in attr_terms) for att in prim.GetAttributes())}
        (a.output/'configuration.json').write_text(json.dumps(dict(variant=a.variant,settings=settings,dt=DT,duration=DURATION,
            physics_cfg=cfg.physx.to_dict(),actual_physics_dt_s=actual_dt,authored_attributes=authored,backend_properties=backend_properties,
            quaternion_tensor_dtype=str(poses[0].dtype),body_order=body_names,
            initial_state_writes=5,runtime_state_writes=0,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2,default=str)+'\n')
        started=time.monotonic()
        for step in range(round(DURATION/DT)):
            applied=[]
            for r in robots:
                q=r.root_physx_view.get_dof_positions();dq=r.root_physx_view.get_dof_velocities()
                effort=torch.clamp(-200*q-20*dq+.15*np.sin(2*np.pi*.7*step*DT),-30,30)
                r.set_joint_effort_target(effort);r.write_data_to_sim();applied.append(effort.detach().cpu().numpy()[0].copy())
            sim.step(render=False)
            for r in robots:r.update(DT)
            free.update(DT);commands.append(applied);append()
            floor_vectors=contacts.get_contact_force_matrix(DT).detach().cpu().numpy().copy()
            floor_loads.append(np.linalg.norm(floor_vectors,axis=-1).sum(axis=1))
            if not np.isfinite(np.r_[poses[-1].ravel(),velocities[-1].ravel()]).all():raise RuntimeError('Nonfinite physical state')
            if step%250==0:
                progress=dict(step=step+1,time_s=(step+1)*DT,wall_s=time.monotonic()-started)
                (a.output/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
        poses=np.array(poses);velocities=np.array(velocities);errors,residual=integrate_pose_errors(poses,velocities,DT)
        np.savez_compressed(a.output/'trace.npz',time_s=np.arange(len(poses))*DT,body_pose_xyzw=poses,body_velocity_world=velocities,
            joint_position=np.array(qrows),joint_velocity=np.array(dqrows),commands_Nm=np.array(commands),
            foot_floor_normal_load_N=np.array(floor_loads),
            root_increment_minus_current_omega_rad=residual,**{k+'_integration_error_rad':v for k,v in errors.items()})
        roots=[body_names.index(label+'/root') for label in ('supported_clear','supported_overlap')]
        from scipy.spatial.transform import Rotation
        tilt=np.arccos(np.clip(np.array([Rotation.from_quat(p[roots,3:]).as_matrix()[:,2,2] for p in poses]),-1,1))
        checks=dict(complete_duration=len(commands)==1500,finite=bool(np.isfinite(poses).all() and np.isfinite(velocities).all()),
            supported_roots_remain_up=bool(tilt.max()<.2 and poses[:,roots,2].min()>.5),
            both_feet_supported_late=bool(np.min(np.array(floor_loads)[-250:])>10.),
            joint_caps=bool(np.max(abs(np.asarray(commands)))<=30.),no_runtime_state_writes=True,
            free_body_endpoint_consistency=bool(np.linalg.norm(errors['current'][:,0],axis=1).max()<1e-4))
        metrics={name:{mode:dict(final_rotvec_rad=arr[-1,i].tolist(),maximum_error_rad=float(np.linalg.norm(arr[:,i],axis=1).max()),
            rms_error_rad=float(np.sqrt(np.mean(np.sum(arr[:,i]**2,axis=1))))) for mode,arr in errors.items()} for i,name in enumerate(body_names)}
        report=dict(schema='doorbench.actual-pose-velocity-fixture.v1',scope=__doc__,passed=all(checks.values()),checks=checks,
            variant=a.variant,settings=settings,physics_dt_s=DT,steps=1500,duration_s=DURATION,runtime_state_writes=0,
            body_order=body_names,metrics=metrics,maximum_supported_tilt_rad=float(tilt.max()),
            minimum_supported_height_m=float(poses[:,roots,2].min()),wall_time_s=time.monotonic()-started,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            physical_note='Bounded ankle PD is identical between supported fixtures and solver variants; no task/robot success claim')
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);failed=not report['passed']
    except BaseException:
        (a.output/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
        if not (a.output/'report.json').exists():
            (a.output/'report.json').write_text(json.dumps(dict(schema='doorbench.physics-fixture-failure.v1',passed=False,
                scope='Failed isolated fixture; no robot task claim',error=traceback.format_exc(),
                executed_physics_steps=len(commands) if 'commands' in locals() else 0,
                variant=a.variant,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
        if 'poses' in locals() and len(poses)>0:
            import numpy as np
            np.savez_compressed(a.output/'failed-prefix.npz',body_pose_xyzw=np.asarray(poses),body_velocity_world=np.asarray(velocities))
    finally:app.close()
    if failed:raise SystemExit(1)


if __name__=='__main__':main()

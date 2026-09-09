#!/usr/bin/env python3
"""Isolated GPU reset synchronization fixture; no physics integration after setup.

Two reset-only root/joint assignments exercise the documented tensor kinematic
update before an own-body gyro reset read. No robot task or sensor accuracy
claim follows from this fixture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True})
    try:
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation,ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from doorbench.dexterous.pose_gyro import OwnImuPoseGyroscope
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.002,device='cuda:0',gravity=(0.,0.,0.),
            physx=sim_utils.PhysxCfg(solver_type=1,min_position_iteration_count=32,min_velocity_iteration_count=8,enable_stabilization=False)))
        if getattr(sim,'_app_control_on_stop_handle',None) is not None:
            sim._app_control_on_stop_handle.unsubscribe();sim._app_control_on_stop_handle=None
        stage=sim.stage;UsdGeom.Xform.Define(stage,'/World/fixture')
        for name,z in [('root',1.),('imu_body',1.4)]:
            transform=UsdGeom.Xform.Define(stage,'/World/fixture/'+name);transform.AddTranslateOp().Set(Gf.Vec3d(0,0,z))
            prim=transform.GetPrim();UsdPhysics.RigidBodyAPI.Apply(prim)
            mass=UsdPhysics.MassAPI.Apply(prim);mass.CreateMassAttr(1.);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.02,.02,.02))
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
        root=stage.GetPrimAtPath('/World/fixture/root');UsdPhysics.ArticulationRootAPI.Apply(root)
        PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(False)
        joint=UsdPhysics.RevoluteJoint.Define(stage,'/World/fixture/hinge');joint.CreateAxisAttr('Y')
        joint.CreateBody0Rel().SetTargets(['/World/fixture/root']);joint.CreateBody1Rel().SetTargets(['/World/fixture/imu_body'])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0,0,.2));joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,-.2))
        joint.CreateLowerLimitAttr(-90.);joint.CreateUpperLimitAttr(90.)
        robot=Articulation(ArticulationCfg(prim_path='/World/fixture',spawn=None,articulation_root_prim_path='/root',
            actuators={'fixture':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,effort_limit_sim=10.)}))
        sim.reset();physics_steps=0
        view=sim.physics_sim_view.create_rigid_body_view('/World/fixture/imu_body')
        producer=OwnImuPoseGyroscope(view,expected_body_path='/World/fixture/imu_body',
            robot_body_paths=['/World/fixture/root','/World/fixture/imu_body'],robot_root_path='/World/fixture',
            imu_quaternion_wxyz_body=[1.,0.,0.,0.])
        records=[]
        for yaw,bend in [(.43,.71),(-.26,-.39)]:
            root_rotation=Rotation.from_rotvec([0,0,yaw]);q=root_rotation.as_quat()[[3,0,1,2]]
            root_state=torch.tensor([[.2,-.1,1.1,*q]],dtype=torch.float32,device='cuda:0')
            position=torch.tensor([[bend]],dtype=torch.float32,device='cuda:0')
            robot.write_root_pose_to_sim(root_state);robot.write_root_velocity_to_sim(torch.zeros((1,6),device='cuda:0'))
            robot.write_joint_state_to_sim(position,torch.zeros_like(position))
            def clock():return [float(sim.current_time),int(sim.current_time_step_index)]
            before=view.get_transforms().detach().cpu().numpy().copy();time_before=clock()
            sim.physics_sim_view.update_articulations_kinematic()
            after=view.get_transforms().detach().cpu().numpy().copy();time_after=clock()
            actual_q=robot.root_physx_view.get_dof_positions().detach().cpu().numpy().copy()
            actual_dq=robot.root_physx_view.get_dof_velocities().detach().cpu().numpy().copy()
            expected_rotation=root_rotation*Rotation.from_rotvec([0,bend,0])
            expected_position=np.array([.2,-.1,1.1])+root_rotation.apply([0,0,.2])+expected_rotation.apply([0,0,.2])
            error=(Rotation.from_quat(after[0,3:])*expected_rotation.inv()).magnitude()
            stale=(Rotation.from_quat(before[0,3:])*expected_rotation.inv()).magnitude()
            producer.reset_episode(now_s=0.);raw=producer.evidence()
            records.append(dict(root_yaw_rad=yaw,joint_bend_rad=bend,before=before.tolist(),after=after.tolist(),
                expected_position=expected_position.tolist(),expected_quaternion_xyzw=expected_rotation.as_quat().tolist(),
                rotation_error_rad=float(error),position_error_m=float(np.linalg.norm(after[0,:3]-expected_position)),
                stale_before_error_rad=float(stale),clock_before=time_before,clock_after=time_after,
                actual_joint_position=actual_q.tolist(),actual_joint_velocity=actual_dq.tolist(),
                producer_reset_exact_raw=bool(np.array_equal(raw['body_quaternion_xyzw_world'][0],after[0,3:].astype(float))),
                producer_receipt=producer.receipt()))
        checks=dict(two_distinct_reset_cases=len(records)==2,
            exact_no_physics_clock_advance=all(r['clock_before']==r['clock_after'] for r in records),
            updated_pose_matches_closed_form=all(r['rotation_error_rad']<1e-6 and r['position_error_m']<1e-6 for r in records),
            positive_stale_read_control=all(r['stale_before_error_rad']>.1 for r in records),
            reset_state_retained=all(abs(r['actual_joint_position'][0][0]-r['joint_bend_rad'])<1e-7 and np.max(np.abs(r['actual_joint_velocity']))==0 for r in records),
            producer_reads_synchronized_tensor=all(r['producer_reset_exact_raw'] for r in records),zero_physics_steps_after_setup=physics_steps==0)
        stage.GetRootLayer().Export(str(a.output/'fixture.usda'))
        report=dict(schema='doorbench.actual-reset-kinematic-fixture.v1',scope=__doc__,passed=all(checks.values()),checks=checks,
            cases=records,physics_steps_after_setup=physics_steps,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            synchronization='SimulationView.update_articulations_kinematic once after each reset-only assignment; no step/render/forward',
            primary_documentation='https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html')
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)
    except BaseException:
        (a.output/'error.txt').write_text(traceback.format_exc())
        (a.output/'report.json').write_text(json.dumps(dict(passed=False,scope=__doc__,error=traceback.format_exc()),indent=2)+'\n')
        raise
    finally:app.close()


if __name__=='__main__':main()

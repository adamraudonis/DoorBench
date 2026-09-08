#!/usr/bin/env python3
"""Live PhysX unilateral tendon fixture; not a robot task or grasp score.

Four identical fixed-base two-link mechanisms compare active/slack loopbacks
against untendoned controls. All state writes occur at fixture initialization.
"""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit-stiffness',type=float,default=10000.);p.add_argument('--duration',type=float,default=.25);p.add_argument('--test-torque',type=float,default=.001);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
failed=False
try:
 import torch,numpy as np
 from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
 import isaaclab.sim as sim_utils
 from isaaclab.assets import Articulation,ArticulationCfg
 from isaaclab.actuators import ImplicitActuatorCfg
 from doorbench.dexterous.isaac_tendons import author_passive_tendons
 sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.002,device='cuda:0',gravity=(0.,0.,0.),render_interval=1000,physx=sim_utils.PhysxCfg(solver_type=1,min_position_iteration_count=32,min_velocity_iteration_count=8)))
 if getattr(sim,'_app_control_on_stop_handle',None) is not None:sim._app_control_on_stop_handle.unsubscribe();sim._app_control_on_stop_handle=None
 stage=sim.stage;robots=[];names=('slack_off','slack_on','active_off','active_on');authoring=[]
 contract=[dict(name='rh_FF_loopback',profile='shadow-loopback-v2',root_joint='rh_FFJ2',terms={'rh_FFJ1':1.,'rh_FFJ2':-1.},range_rad=[-2.,0.],spring_stiffness=0.,damping=0.,physx_limit_stiffness=a.limit_stiffness,rest_length_rad=0.,offset_rad=0.)]
 for n,label in enumerate(names):
  path='/World/'+label;UsdGeom.Xform.Define(stage,path)
  for body,z in (('base',0.),('middle',.04),('distal',.10)):
   shape=UsdGeom.Cube.Define(stage,path+'/'+body);shape.CreateSizeAttr(.02);shape.AddTranslateOp().Set(Gf.Vec3d(n*.3,0,z));prim=shape.GetPrim()
   UsdPhysics.RigidBodyAPI.Apply(prim)
   mass=UsdPhysics.MassAPI.Apply(prim);mass.CreateMassAttr(.02);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.0002,.0002,.0002))
  root=stage.GetPrimAtPath(path+'/base');UsdPhysics.ArticulationRootAPI.Apply(root)
  settings=PhysxSchema.PhysxArticulationAPI.Apply(root);settings.CreateEnabledSelfCollisionsAttr(False);settings.CreateSolverPositionIterationCountAttr(32);settings.CreateSolverVelocityIterationCountAttr(8)
  anchor=UsdPhysics.FixedJoint.Define(stage,path+'/anchor');anchor.CreateBody1Rel().SetTargets([path+'/base']);anchor.CreateLocalPos0Attr(Gf.Vec3f(n*.3,0,0.))
  for joint,parent,child,pp,cp in (('rh_FFJ2','base','middle',.02,-.02),('rh_FFJ1','middle','distal',.04,-.02)):
   j=UsdPhysics.RevoluteJoint.Define(stage,path+'/'+joint);j.CreateAxisAttr('X');j.CreateBody0Rel().SetTargets([path+'/'+parent]);j.CreateBody1Rel().SetTargets([path+'/'+child]);j.CreateLocalPos0Attr(Gf.Vec3f(0,0,pp));j.CreateLocalPos1Attr(Gf.Vec3f(0,0,cp));j.CreateLowerLimitAttr(0.);j.CreateUpperLimitAttr(90.)
   drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular');drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.)
  authoring.append(author_passive_tendons(stage,path,contract if label.endswith('_on') else []))
  robots.append(Articulation(ArticulationCfg(prim_path=path,spawn=None,articulation_root_prim_path='/base',actuators={'test':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,effort_limit_sim=.1)})))
 sim.reset();indices=[];rows=[];readback=[]
 for label,r in zip(names,robots):
  i2=r.joint_names.index('rh_FFJ2');i1=r.joint_names.index('rh_FFJ1');indices.append((i2,i1))
  q=r.data.default_joint_pos.clone();q[:,i2]=.8 if label.startswith('slack') else .4;q[:,i1]=.2 if label.startswith('slack') else .4;r.write_joint_state_to_sim(q,torch.zeros_like(q));r.update(.002)
  view=r.root_physx_view
  values={}
  for method in ('get_fixed_tendon_stiffnesses','get_fixed_tendon_dampings','get_fixed_tendon_limit_stiffnesses','get_fixed_tendon_limits','get_fixed_tendon_rest_lengths','get_fixed_tendon_offsets'):
   try:values[method]=getattr(view,method)().detach().cpu().numpy().tolist()
   except Exception as e:values[method]=dict(error=type(e).__name__+': '+str(e))
  readback.append(values)
 start=time.time()
 for step in range(round(a.duration/.002)):
  for label,r,(i2,i1) in zip(names,robots,indices):
   u=torch.zeros_like(r.data.joint_pos);u[:,i2 if label.startswith('slack') else i1]=.04*a.test_torque if label.startswith('slack') else a.test_torque
   r.set_joint_effort_target(u);r.write_data_to_sim()
  sim.step(render=False)
  for r in robots:r.update(.002)
  stacked=torch.stack([r.data.joint_pos[0,list(ii)] for r,ii in zip(robots,indices)]).detach().cpu().numpy()
  rows.append(dict(t=(step+1)*.002,q_middle_distal=stacked.tolist()))
  if step%100==0:
   info=dict(sim_time_s=(step+1)*.002,elapsed_s=time.time()-start,positions=stacked.tolist());(a.output/'progress.json').write_text(json.dumps(info));print(json.dumps(info),flush=True)
 q=np.asarray([r['q_middle_distal'] for r in rows]);difference=q[:,:,1]-q[:,:,0]
 checks=dict(full_duration=len(rows)==round(a.duration/.002),finite=bool(np.isfinite(q).all()),slack_matches_unconstrained=bool(np.max(np.abs(q[:,0]-q[:,1]))<1e-5),active_inequality=bool(np.max(difference[:,3])<=.02),unconstrained_control_violates=bool(np.max(difference[:,2])>.1),middle_is_pulled=bool(q[-1,3,0]>q[-1,2,0]+.02),no_unrequested_spring=all(x['tendons']==[] for x in (authoring[0],authoring[2])))
 report=dict(scope=__doc__,passed=all(checks.values()),checks=checks,limit_stiffness=a.limit_stiffness,maximum_difference_rad={name:float(np.max(difference[:,i])) for i,name in enumerate(names)},final_positions={name:q[-1,i].tolist() for i,name in enumerate(names)},slack_max_difference_rad=float(np.max(np.abs(q[:,0]-q[:,1]))),test_torque_Nm=a.test_torque,initial_state_writes_per_fixture=1,runtime_state_writes=0,physics_dt=.002,readback=readback,authoring=authoring)
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'trace.json').write_text(json.dumps(rows)+'\n');stage.GetRootLayer().Export(str(a.output/'fixture.usda'));print(json.dumps(report,indent=2),flush=True);failed=not report['passed']
except BaseException:
 import traceback
 failed=True;(a.output/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
finally:app.close()
if failed:raise SystemExit(1)

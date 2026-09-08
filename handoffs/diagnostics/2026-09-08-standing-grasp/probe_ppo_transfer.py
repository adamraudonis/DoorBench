#!/usr/bin/env python3
"""Initialized tactile-PPO transfer diagnostic; not acquisition or a door task.

Restores a recorded failed-acquisition state once, then uses native capped motors.
Policy sees the exact historical 438-dimensional hand proprioception/tactile input
and emits its six residual actions; landed-body QP is a privileged teacher.
"""
import argparse, hashlib, importlib.util, json, shutil
from pathlib import Path
import mujoco
import numpy as np
from stable_baselines3 import PPO
from doorbench.dexterous.grasp_training import GraspSkillEnv
from doorbench.dexterous.contact_audit import lever_contacts
from doorbench.dexterous.limit_filter import HandLimitFilter

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--frame',type=int,default=-1);p.add_argument('--duration',type=float,default=3.)
p.add_argument('--limit-filter',action='store_true');p.add_argument('--base',choices=['snapshot','training'],default='snapshot')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
base=Path('/tmp/doorbench-dexterous-humanoid/out/dexterous')
checkpoint=base/'training/grasp-001/final.zip'
preload=base/'grasp-contact-optimization/best.json';seed=base/'grasp-seed-v6/pose.npz'
env=GraspSkillEnv(base/'assets/doors/db0055_swing_single',base/'robot/h1-shadow.xml',seed,preload,horizon=150,initialization_noise=0.)
env.reset(seed=0);s=env.sim;m,d=s.m,s.d
training_base=env.base.copy();source=np.load(a.source/'trajectory.npz');poses=source['qpos'];frame=a.frame%len(poses)
d.qpos[:]=poses[frame];d.qvel[:]=source['qvel'][frame];d.time=0.
mujoco.mj_forward(m,d)
env.base=d.actuator_length[s.actuators].copy() if a.base=='snapshot' else training_base
env.previous[:]=0.
policy=PPO.load(checkpoint,device='cpu')
interface=env.configuration_audit();original=json.loads((base/'evaluation/grasp-001-final/interface.json').read_text())
for key in ('observation_dimensions','observation_order','joint_order','tactile_sensor_order','thumb_action_order','shared_curl_actuators','residual_scale_rad','control_frequency_hz'):
 if interface[key]!=original[key]:raise ValueError(f'Historical interface mismatch {key}')
assert policy.observation_space.shape==(438,) and policy.action_space.shape==(6,)
(a.output/'interface.json').write_text(json.dumps(interface,indent=2)+'\n')
stance_path=Path('/tmp/doorbench-h1-locomotion/doorbench/dexterous/locomotion_manipulation.py')
spec=importlib.util.spec_from_file_location('landed_foot_stance',stance_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
stance=module.LandedFootStanceController(s)
kp=m.actuator_gainprm[s.actuators,0].copy();bias=m.actuator_biasprm[s.actuators,:3].copy();caps=m.actuator_forcerange[s.actuators].copy()
arm=np.array([(m.actuator(i).name.startswith('robot/right_') and not any(n in m.actuator(i).name for n in ('hip','knee','ankle'))) or m.actuator(i).name.startswith('robot/rh_A_WRJ') for i in s.actuators])
gain=9*arm.astype(float);damping=np.array([(.8 if 'WRJ' in m.actuator(i).name else 10.) if flag else 20. if m.actuator(i).name=='robot/torso' else 0. for i,flag in zip(s.actuators,arm)])
for i,aid in enumerate(s.actuators):
 if m.actuator(aid).name=='robot/torso':gain[i]=9.
m.actuator_gainprm[s.actuators,0]=1.;m.actuator_biasprm[s.actuators,:3]=0.;m.actuator_ctrlrange[s.actuators]=caps
d.ctrl[s.actuators]=0.
matrix=np.zeros((len(s.actuators),len(s.joints)));ji={int(j):i for i,j in enumerate(s.joints)}
for i,aid in enumerate(s.actuators):
 tid=int(m.actuator_trnid[aid,0])
 if m.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:matrix[i,ji[tid]]=m.actuator_gear[aid,0]
 else:
  for k in range(m.tendon_adr[tid],m.tendon_adr[tid]+m.tendon_num[tid]):matrix[i,ji[int(m.wrap_objid[k])]]=m.wrap_prm[k]
fingers=[i for i,aid in enumerate(s.actuators) if m.actuator(aid).name.startswith('robot/rh_') and 'WRJ' not in m.actuator(aid).name]
limit_filter=HandLimitFilter(m,d,s.actuators,s.joints,matrix,fingers) if a.limit_filter else None
rows=[];qs=[];vs=[];us=[];actions=[];observations=[];gate_rows=[];held=0;max_held=0
palm=m.site('robot/rh_palm_touch').id;goal=d.site_xpos[palm].copy();lever=m.geom('leaf_handle_lever_col_n').id
initial_q=d.qpos.copy();initial_v=d.qvel.copy();limit_info={};status='unstarted'
try:
 for step in range(round(a.duration/m.opt.timestep)):
  if step%10==0:
   obs=env.observation();action,_=policy.predict(obs,deterministic=True);action=np.clip(action,-1,1)
   observations.append(obs.copy());actions.append(action.copy());delta=env.preload+.20*action
   target=env.base.copy();target[env.thumb]+=delta[:5];target[env.curl]+=delta[5];target=np.clip(target,s.low,s.high)
   env.previous=action.copy()
  if step%5==0:stance_target,status=stance.command()
  command=target+gain*(target-d.actuator_length[s.actuators])-damping*d.actuator_velocity[s.actuators]/kp
  for local,aid in enumerate(s.actuators):
   if arm[local] and m.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:command[local]+=d.qfrc_bias[m.jnt_dofadr[m.actuator_trnid[aid,0]]]/kp[local]
  force=kp*command+bias[:,0]+bias[:,1]*d.actuator_length[s.actuators]+bias[:,2]*d.actuator_velocity[s.actuators]
  if stance_target is not None:force[stance.local]=stance_target
  if limit_filter is not None:force,limit_info=limit_filter.apply(force)
  d.ctrl[s.actuators]=np.clip(force,caps[:,0],caps[:,1]);s.plant.step()
  diag=s.diagnostics();violation=max(0.,float(np.maximum(m.jnt_range[s.joints,0]-d.qpos[s.qadr],d.qpos[s.qadr]-m.jnt_range[s.joints,1]).max()))
  penetration=0.
  for c in d.contact[:d.ncon]:
   bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom];geoms=[m.geom(int(g)).name for g in c.geom]
   if any(n.startswith('robot/') for n in bodies) and not ('floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)):penetration=max(penetration,-float(c.dist))
  contact=lever_contacts(m,d,'leaf_handle_lever_col_n')
  held=held+1 if contact['opposed'] else 0;max_held=max(max_held,held)
  force_ok=bool(np.all(d.actuator_force[s.actuators]>=caps[:,0]-1e-5) and np.all(d.actuator_force[s.actuators]<=caps[:,1]+1e-5))
  gate=dict(t=float(d.time),joint_violation=violation,penetration=penetration,force_ok=force_ok,finite=diag['finite'] and not diag['numerical_warnings'],upright=diag['torso_tilt_deg']<12 and diag['root_height_m']>.7,opposed=contact['opposed'],limit_feasible=limit_info.get('feasible',True))
  gate_rows.append(gate)
  if step%10==0:
   thumb=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_th')]
   gap=min(mujoco.mj_geomDistance(m,d,g,lever,.1,np.zeros(6)) for g in thumb)
   row=dict(**diag,contacts=contact,palm_error_m=float(np.linalg.norm(d.site_xpos[palm]-goal)),thumb_gap_m=float(gap),max_joint_limit_violation_rad=violation,max_nonfoot_penetration_m=penetration,stance_status=status,limit_filter=limit_info,handle_angle_rad=float(d.qpos[m.jnt_qposadr[m.joint('leaf_handle_hinge').id]]))
   rows.append(row);qs.append(d.qpos.copy());vs.append(d.qvel.copy());us.append(d.ctrl.copy())
   if step%500==0:print({k:v for k,v in row.items() if k!='contacts'},flush=True)
  if not diag['finite'] or diag['torso_tilt_deg']>35 or diag['root_height_m']<.6 or violation>.05 or penetration>.015:break
 checks=dict(full_duration=d.time>=a.duration-.003,upright=all(r['upright'] for r in gate_rows),finite=all(r['finite'] for r in gate_rows),native_motor_limits=all(r['force_ok'] for r in gate_rows),physical_joint_limits=all(r['joint_violation']<=.02 for r in gate_rows),nonfoot_penetration=all(r['penetration']<=.003 for r in gate_rows),holds_opposed_contacts=bool(gate_rows and all(r['opposed'] for r in gate_rows if r['t']>=a.duration-.5)),sustained_opposition_one_second=max_held*m.opt.timestep>=1.)
 if a.limit_filter:checks['limit_filter_feasible']=all(r['limit_feasible'] for r in gate_rows)
 report=dict(scope=__doc__,passed=all(checks.values()),checks=checks,source=str(a.source),source_frame=frame,base=a.base,duration_s=float(d.time),max_joint_limit_violation_rad=max(r['joint_violation'] for r in gate_rows),max_nonfoot_penetration_m=max(r['penetration'] for r in gate_rows),max_contiguous_opposition_s=max_held*m.opt.timestep,final_contacts=rows[-1]['contacts'],final_thumb_gap_m=rows[-1]['thumb_gap_m'],runtime_state_writes=0,direct_door_commands=False,policy_observation_dimension=438,policy_action_dimension=6)
 np.savez_compressed(a.output/'trajectory.npz',qpos=qs,qvel=vs,ctrl=us,actions=actions,observations=observations,initial_qpos=initial_q,initial_qvel=initial_v)
 for name,value in [('report',report),('trace',rows),('physical-gates',gate_rows)]: (a.output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
 records=[]
 for source_file,name in [(Path(__file__),'diagnostic-source.py'),(stance_path,'stance-source.py'),(Path('/tmp/doorbench-isaac-integration/doorbench/dexterous/grasp_training.py'),'grasp-training-source.py'),(Path('/tmp/doorbench-isaac-integration/doorbench/dexterous/limit_filter.py'),'limit-filter-source.py')]:
  shutil.copy2(source_file,a.output/name);records.append(dict(path=str(source_file),copy=name,sha256=hashlib.sha256(source_file.read_bytes()).hexdigest()))
 (a.output/'manifest.json').write_text(json.dumps(dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},source_files=records,checkpoint=dict(path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),source_trajectory_sha256=hashlib.sha256((a.source/'trajectory.npz').read_bytes()).hexdigest()),indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('scope','final_contacts')},indent=2));print(report['final_contacts']['digit_forces_N'])
finally:env.close()

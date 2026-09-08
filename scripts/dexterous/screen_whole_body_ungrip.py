#!/usr/bin/env python3
"""Screen an actual resting grasp adjustment and exact measured hand withdrawal.

All qpos writes are to a new, unstepped geometric model. This does not qualify
physical motion. Run audit_whole_body_ungrip_screen.py before motor execution.
"""
import argparse
from pathlib import Path
import sys,json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--measured-release',type=Path,required=True);parser.add_argument('--release-trajectory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--finger-lead-seconds',type=float,default=0.);parser.add_argument('--early-lift-m',type=float,default=.002);parser.add_argument('--withdrawal-profile',choices=('recorded','clearance-lift-v1','clearance-lift-v2','clearance-lift-v3','clearance-lift-v4'),default='recorded');args=parser.parse_args();
if not np.isfinite(args.finger_lead_seconds) or not 0<=args.finger_lead_seconds<=.5:raise ValueError('Require a bounded finger lead')
if not np.isfinite(args.early_lift_m) or not 0<=args.early_lift_m<=.01:raise ValueError('Require a bounded early geometric lift')
root=Path(__file__).resolve().parents[2];run=args.source_run.resolve();sys.path.insert(0,str(run.with_name(run.name+'-source')))
from doorbench.dexterous.environment import DexterousDoorEnv
c=json.load(open(run/'manifest.json'))['configuration'];r=Path(c['robot']);s=DexterousDoorEnv(c['door'],r,json.load(open(r.with_suffix('.audit.json'))));m,d=s.m,s.d
with np.load(run/'trajectory.npz') as a:base=a['terminal_qpos'].copy();initial_velocity=a['terminal_qvel'].copy();initial_time=float(a['terminal_time_s'])
d.qpos[:]=base;mujoco.mj_kinematics(m,d);hb=m.body('leaf_handle').id;rh=m.site('robot/rh_palm_touch').id;lh=m.site('robot/lh_palm_touch').id;H=d.xmat[hb].reshape(3,3).copy();P=d.site_xpos[rh].copy();R=d.site_xmat[rh].reshape(3,3).copy();PL=d.site_xpos[lh].copy();RL=d.site_xmat[lh].reshape(3,3).copy();feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')];FP=d.xpos[feet].copy();FR=d.xmat[feet].reshape(2,3,3).copy();rq=m.jnt_qposadr[m.joint('robot/free_base').id];rootP=base[rq:rq+3].copy();rootR=Rotation.from_quat([*base[rq+4:rq+7],base[rq+3]]);lever=m.geom('leaf_handle_lever_col_n').id;floor=m.geom('floor').id
right=['right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['rh_WRJ2','rh_WRJ1'];left=['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1'];legs=[side+'_'+j for side in ('left','right') for j in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')];names=legs+['torso']+right+left;js=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[js];start=base[qa].copy()
src=json.load(open(args.measured_release));fn=src['finger_joint_names'];fjs=np.array([m.joint('robot/'+n).id for n in fn]);fqa=m.jnt_qposadr[fjs];fds=np.asarray(src['finger_joint_delta_rad']);pp=np.asarray(src['palm_position_handle']);times=np.asarray(src['time_s']);out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);trials=[]
if hashlib.file_digest(r.open('rb'),'sha256').hexdigest()!=src['robot_sha256']:
 raise ValueError('Measured source requires the exact declared robot XML')
if hashlib.file_digest(args.release_trajectory.open('rb'),'sha256').hexdigest()!=src['source_files']['trajectory.npz']:
 raise ValueError('Measured source trajectory changed')
source_trajectory=np.load(args.release_trajectory)['qpos'];
if source_trajectory.shape[1]!=m.nq:raise ValueError('Measured source joint configuration contract differs')
source_fingers=source_trajectory[src['source_frames']][:,fqa];source_rotations=np.asarray(src['palm_rotation_handle']);source_initial_p=pp[0];source_initial_r=source_rotations[0];initial_relative_p=H.T@(P-d.xpos[hb]);initial_relative_r=H.T@R;handleP=d.xpos[hb].copy()
for dx,dy,roll in [(0,-.16056,0)]:
 rows=[];previous=np.r_[np.zeros(6),start]
 nodes=[('grasp_adjustment',float(u)) for u in np.linspace(0,1,31)]+[('measured_release',float(u)) for u in np.linspace(0,1,len(times))[1:] if args.withdrawal_profile=='recorded' or float(times[round(u*(len(times)-1))])<=4.1+1e-8]
 if args.withdrawal_profile.startswith('clearance-lift-'):nodes += [('clearance_lift',float(u)) for u in np.linspace(0,1,41)[1:]]
 for phase,u in nodes:
  if phase=='grasp_adjustment':
   fraction=u;relative_p=(1-u)*initial_relative_p+u*source_initial_p;relative_r=Rotation.from_rotvec(u*Rotation.from_matrix(source_initial_r@initial_relative_r.T).as_rotvec()).as_matrix()@initial_relative_r;f=(1-u)*base[fqa]+u*source_fingers[0];clock=4*u
  elif phase=='clearance_lift':
   start_i=int(np.argmin(abs(times-4.1)));fraction=u;blend=u**3*(10+u*(-15+6*u));relative_p=pp[start_i]+np.array([0.,0.,args.early_lift_m if args.withdrawal_profile in ('clearance-lift-v2','clearance-lift-v3','clearance-lift-v4') else 0.])+np.array([-.006 if args.withdrawal_profile=='clearance-lift-v4' else 0.,-.016 if args.withdrawal_profile in ('clearance-lift-v3','clearance-lift-v4') else -.010,.040])*blend;relative_r=Rotation.from_rotvec(blend*Rotation.from_matrix(source_rotations[-1]@source_rotations[start_i].T).as_rotvec()).as_matrix()@source_rotations[start_i];source_t=float(times[start_i])+blend*(times[-1]-times[start_i]);f=np.array([np.interp(source_t,times,source_fingers[:,k]) for k in range(len(fn))]);clock=4+float(times[start_i])+.7*u
  else:
   i=round(u*(len(fds)-1));fraction=u;relative_p=pp[i];relative_r=source_rotations[i];finger_t=float(times[i])+args.finger_lead_seconds*np.sin(np.pi*u)**2;f=np.array([np.interp(finger_t,times,source_fingers[:,k]) for k in range(len(fn))]);clock=4+float(times[i])
  if args.withdrawal_profile in ('clearance-lift-v2','clearance-lift-v3','clearance-lift-v4') and phase=='measured_release':
   lift_u=float(np.clip((clock-6.)/1.,0.,1.));relative_p=relative_p+np.array([0.,0.,args.early_lift_m*lift_u**3*(10+lift_u*(-15+6*lift_u))])
  PR=handleP+H@relative_p;RR=H@relative_r;f=np.clip(f,m.jnt_range[fjs,0],m.jnt_range[fjs,1])
  for digit in ('FF','MF','RF','LF'):
   a,b=[fn.index(f'rh_{digit}J{k}') for k in (1,2)]
   if f[a]>f[b]:f[[a,b]]=f[[a,b]].mean()
  target_base=base.copy();target_base[fqa]=f;lo=np.r_[[-.08,-.08,-.04],[-.12,-.12,-.12],m.jnt_range[js,0]+.001];hi=np.r_[[.08,.08,.04],[.12,.12,.12],m.jnt_range[js,1]-.001]
  def fun(x):
   d.qpos[:]=target_base;d.qpos[rq:rq+3]=rootP+x[:3];r=(Rotation.from_rotvec(x[3:6])*rootR).as_quat();d.qpos[rq+3:rq+7]=np.r_[r[3],r[:3]];d.qpos[qa]=x[6:];mujoco.mj_kinematics(m,d)
   hand=np.r_[100*(d.site_xpos[rh]-PR),10*Rotation.from_matrix(RR@d.site_xmat[rh].reshape(3,3).T).as_rotvec(),100*(d.site_xpos[lh]-PL),10*Rotation.from_matrix(RL@d.site_xmat[lh].reshape(3,3).T).as_rotvec()]
   foot=np.concatenate([np.r_[100*(d.xpos[b]-FP[i]),10*Rotation.from_matrix(FR[i]@d.xmat[b].reshape(3,3).T).as_rotvec()] for i,b in enumerate(feet)])
   return np.r_[hand,foot,.01*(x[6:]-start),.02*x[:6]]
  fit=least_squares(fun,np.clip(previous,lo,hi),bounds=(lo,hi),max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-11);previous=fit.x.copy();res=fun(fit.x);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d);cols={};invalid={}
  for cc in d.contact[:d.ncon]:
   bs=[m.body(m.geom_bodyid[g]).name for g in cc.geom]
   if cc.dist<-.003 and any(b.startswith('robot/') for b in bs) and not (floor in cc.geom and any(b.endswith('_ankle_link') for b in bs)):key=' + '.join(bs);cols[key]=max(cols.get(key,0),-float(cc.dist))
   if cc.dist<-.00005 and lever in cc.geom:
    g=int(cc.geom[1] if cc.geom[0]==lever else cc.geom[0]);b=m.geom_bodyid[g];name=m.body(b).name
    if name.startswith('robot/rh_'):
     br=d.xmat[b].reshape(3,3);p=br.T@(cc.pos-d.xpos[b]);normal=br.T@((1 if cc.geom[0]==g else -1)*cc.frame[:3]);axis=d.geom_xmat[lever].reshape(3,3)[:,2];v=cc.pos-d.geom_xpos[lever];a=np.dot(v,axis);rad=v-a*axis;align=np.dot(br@normal,-rad/max(np.linalg.norm(rad),1e-9));valid=name.endswith('distal') and p[1]<-.001 and .002<=p[2]<=.040 and -normal[1]>.5 and m.geom_size[lever,1]-abs(a)>=.001 and align>.8
     if not valid:invalid[name]=max(invalid.get(name,0),-float(cc.dist))
  up=d.xmat[m.body('robot/torso_link').id].reshape(3,3)[:,2];tilt=float(np.degrees(np.arccos(np.clip(up[2],-1,1))));rows.append(dict(phase=phase,time_s=clock,progress=float(u),fraction=fraction,palm_position=PR.tolist(),palm_rotation=RR.tolist(),right_position_error_m=float(np.linalg.norm(res[:3])/100),right_rotation_error_rad=float(np.linalg.norm(res[3:6])/10),left_position_error_m=float(np.linalg.norm(res[6:9])/100),left_rotation_error_rad=float(np.linalg.norm(res[9:12])/10),foot_position_errors_m=[float(np.linalg.norm(res[12+6*i:15+6*i])/100) for i in range(2)],foot_rotation_errors_rad=[float(np.linalg.norm(res[15+6*i:18+6*i])/10) for i in range(2)],root_delta_xyz_m=fit.x[:3].tolist(),root_delta_rotvec_rad=fit.x[3:6].tolist(),joints=dict(zip(names,fit.x[6:].tolist())),finger_joints=dict(zip(fn,f.tolist())),joint_margins_rad=dict(zip(names,np.minimum(fit.x[6:]-lo[6:],hi[6:]-fit.x[6:]).tolist())),forbidden_collisions=cols,invalid_patches=invalid,torso_tilt_deg=tilt,root_qpos_address=int(rq),qpos=d.qpos.tolist()))
 result=dict(dx=dx,dy=dy,roll=roll,maximum_palm_error_m=max(max(x['right_position_error_m'],x['left_position_error_m']) for x in rows),maximum_palm_rotation_error_rad=max(max(x['right_rotation_error_rad'],x['left_rotation_error_rad']) for x in rows),maximum_torso_tilt_deg=max(x['torso_tilt_deg'] for x in rows),maximum_forbidden_depth_m=max([v for x in rows for v in x['forbidden_collisions'].values()]+[0]),invalid_pad_samples=sum(bool(x['invalid_patches']) for x in rows),root_delta=rows[-1]['root_delta_xyz_m'],rows=rows);trials.append(result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True);(out/'report.json').write_text(json.dumps(dict(scope='Unstepped whole-body adjustment to canonical hand/lever grasp then exact measured release; no dynamics qualification',initial_time_s=initial_time,configuration={k:(str(v.resolve()) if isinstance(v,Path) else v) for k,v in vars(args).items()},robot_xml_sha256=hashlib.file_digest(r.open('rb'),'sha256').hexdigest(),source_trajectory_sha256=hashlib.file_digest((run/'trajectory.npz').open('rb'),'sha256').hexdigest(),trials=trials),indent=2))
s.close()

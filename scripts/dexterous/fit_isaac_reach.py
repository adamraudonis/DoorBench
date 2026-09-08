#!/usr/bin/env python3
"""Fit one upright reset and a continuous arm workspace across handle and leaf travel.

Analytic kinematics only. Outputs a reset/reference, never runtime pose control.
"""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from doorbench.dexterous.reset import check_joint_reset

p=argparse.ArgumentParser();p.add_argument('--robot',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--flip-grip',action='store_true');p.add_argument('--grip-shift-m',type=float,default=-.008);p.add_argument('--push-angle',type=float,default=.12);p.add_argument('--stance-lean',type=float,default=.08)
a=p.parse_args();m=mujoco.MjModel.from_xml_path(a.robot);d=mujoco.MjData(m);r=json.loads(Path(a.reference).read_text())
d.qpos[:7]=r['initial_root']
for n,v in r['initial_joints'].items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
idle_arm={'left_elbow':1.,'left_shoulder_roll':.18}
for n,v in idle_arm.items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
mujoco.mj_kinematics(m,d);palm=m.site('rh_palm_touch').id;pp=d.site_xpos[palm].copy();pr=d.site_xmat[palm].reshape(3,3).copy()
feet=[m.body(n+'_ankle_link').id for n in ('left','right')];footheight=d.xpos[feet,2].mean();base=d.qpos.copy()
names=['torso','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_yaw','rh_WRJ2','rh_WRJ1']
jids=[m.joint(n).id for n in names];qa=m.jnt_qposadr[jids];low=m.jnt_range[jids,0]+.04;high=m.jnt_range[jids,1]-.04
# The upper-arm shell contacts the torso before the permissive imported joint
# stop. Keep it outboard; the live motor limits themselves remain unchanged.
high[2]=-.04
k=len(names)
poses=[(h,0.) for h in np.linspace(0,.87,5)]+[(.87,float(leaf)) for leaf in np.linspace(a.push_angle/3,a.push_angle,3)]
def goals_for(roll):
 goals=[]
 Rg=Rotation.from_rotvec([roll,0,0]).as_matrix()@Rotation.from_rotvec([0,np.pi if a.flip_grip else 0,0]).as_matrix();center=np.array([.26,-.077,.914]);grip_pos=center+Rg@(pp-center)+[a.grip_shift_m,0,0];grip_rot=Rg@pr
 for handle,leaf in poses:
  Rl=Rotation.from_rotvec([0,0,leaf]).as_matrix();Rh=Rotation.from_rotvec([0,-handle,0]).as_matrix();pivot=np.array([-.385,0,0]);hp=np.array([.320,0,.914])
  goals.append((pivot+Rl@(hp-pivot+Rh@(grip_pos-hp)),Rl@Rh@grip_rot))
 return goals

def setbase(params):
 d.qpos[:]=base;x,y,yaw,bend=params;d.qpos[:2]=[x,y];d.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
 for side in ('left','right'):
  for n,v in [('hip_pitch',-bend+a.stance_lean),('knee',2*bend),('ankle',-bend-a.stance_lean)]:d.qpos[m.jnt_qposadr[m.joint(side+'_'+n).id]]=v
 mujoco.mj_kinematics(m,d);d.qpos[2]+=footheight-d.xpos[feet,2].mean()

prior=np.r_[base[:2],2*np.arctan2(base[6],base[3]),.55,0.]
def residual(x):
 setbase(x[:4]);errors=[];arms=x[5:].reshape(-1,k);goals=goals_for(x[4])
 for i,(position,rotation) in enumerate(goals):
  d.qpos[qa]=arms[i];mujoco.mj_kinematics(m,d)
  errors.extend(100*(d.site_xpos[palm]-position));errors.extend(10*Rotation.from_matrix(rotation@d.site_xmat[palm].reshape(3,3).T).as_rotvec())
  # Use the redundant arm DOF to keep the wrist away from a hard stop.
  errors.extend(np.array([.08,.003,.003,.003,.003,.003,.03,.08])*(arms[i]-(low+high)/2))
 errors.extend(.02*(x[:5]-prior))
 for i in range(1,len(arms)):errors.extend(.015*(arms[i]-arms[i-1]))
 return np.array(errors)

n=len(poses);block=6+k;sparsity=lil_matrix((block*n+5+k*(n-1),5+k*n))
for i in range(n):sparsity[block*i:block*(i+1),:5]=1;sparsity[block*i:block*(i+1),5+k*i:5+k*(i+1)]=1
sparsity[block*n:block*n+5,:5]=1
for i in range(1,n):sparsity[block*n+5+k*(i-1):block*n+5+k*i,5+k*(i-1):5+k*(i+1)]=1
max_bend=min(.75,*[-float(m.jnt_range[m.joint(side+'_ankle').id,0])-.08-a.stance_lean for side in ('left','right')],
             *[float(m.jnt_range[m.joint(side+'_knee').id,1])/2-.04 for side in ('left','right')])
lo=np.r_[[-.65,-1.,-3.2,.2,-.4],np.tile(low,n)];hi=np.r_[[.75,-.3,3.2,max_bend,.8],np.tile(high,n)]
best=None
rng=np.random.default_rng(412)
for seed in range(18):
 initial=np.r_[prior,np.tile(base[qa],n)]
 if seed:
  initial[:5]=rng.uniform([-.2,-.8,.5,.2,.2],[.4,-.45,2.8,.8,2.6])
  initial[5:]=np.tile(rng.uniform(low,high),n)
 fit=least_squares(residual,np.clip(initial,lo,hi),bounds=(lo,hi),jac_sparsity=sparsity.tocsr(),max_nfev=400,ftol=1e-8,xtol=1e-8,gtol=1e-8)
 score=np.max(np.abs(fit.fun[:block*n].reshape(n,block)[:,:6]));print('candidate',seed,'base',fit.x[:5].tolist(),'max weighted error',score,flush=True)
 if best is None or score<best[0]:best=(score,fit)
 if score<.01:break
score,fit=best;arms=fit.x[5:].reshape(n,k);setbase(fit.x[:4]);d.qpos[qa]=arms[0];mujoco.mj_kinematics(m,d)
check_joint_reset([m.joint(j).name for j in range(1,m.njnt)],d.qpos[m.jnt_qposadr[1:]],m.jnt_range[1:])
r['initial_root']=d.qpos[:7].tolist();r['initial_joints']={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(1,m.njnt)}
control=np.array(r['controls'][40])
for n,v in idle_arm.items():control[m.actuator(n).id]=v
for side in ('left','right'):
 for n in ('hip_pitch','knee','ankle'):control[m.actuator(side+'_'+n).id]=d.qpos[m.jnt_qposadr[m.joint(side+'_'+n).id]]
for name,q in zip(names,arms[0]):control[m.actuator('rh_A_'+name[3:] if name.startswith('rh_') else name).id]=q
r['controls']=[control.tolist()]*600
r['idle_arm_pose']=idle_arm
r['stance_lean_rad']=a.stance_lean
r['workspace_fit']={'root_parameters':fit.x[:5].tolist(),'flip_grip':a.flip_grip,'grip_shift_m':a.grip_shift_m,'poses':poses,'joint_names':names,'arm_qpos':arms.tolist(),'weighted_pose_residual_max':float(score),'scope':'Kinematic candidate, requires live free-base motor validation'}
Path(a.output).write_text(json.dumps(r)+'\n')
print('OUTPUT',a.output,flush=True)
if score>.01:raise RuntimeError('Candidate cannot preserve the grasp across the planned workspace')

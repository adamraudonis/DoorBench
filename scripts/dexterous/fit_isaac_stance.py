#!/usr/bin/env python3
"""Fit a reachable reset and bounded gravity feedforward; no forces in the plant."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser();p.add_argument('--robot',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--bend',type=float,default=.55)
a=p.parse_args();m=mujoco.MjModel.from_xml_path(a.robot);d=mujoco.MjData(m)
r=json.loads(Path(a.reference).read_text());d.qpos[:7]=r['initial_root']
for n,v in r['initial_joints'].items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
mujoco.mj_forward(m,d);palm=m.site('rh_palm_touch').id;position=d.site_xpos[palm].copy();rotation=d.site_xmat[palm].reshape(3,3).copy()
feet=[m.body(n+'_ankle_link').id for n in ('left','right')];height=d.xpos[feet,2].mean()
for side in ('left','right'):
 for n,v in [('hip_pitch',-a.bend),('knee',2*a.bend),('ankle',-a.bend)]:d.qpos[m.jnt_qposadr[m.joint(side+'_'+n).id]]=v
mujoco.mj_forward(m,d);d.qpos[2]+=height-d.xpos[feet,2].mean()
names=['right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_yaw','rh_WRJ2','rh_WRJ1']
joints=[m.joint(n).id for n in names];qa=m.jnt_qposadr[joints];prior=d.qpos[qa].copy();low=m.jnt_range[joints,0]+.03;high=m.jnt_range[joints,1]-.03
rng=np.random.default_rng(0)
def residual(x):
 d.qpos[qa]=x;mujoco.mj_kinematics(m,d)
 return np.r_[100*(d.site_xpos[palm]-position),10*Rotation.from_matrix(rotation@d.site_xmat[palm].reshape(3,3).T).as_rotvec(),.01*(x-prior)]
results=[]
for i in range(12):
 start=prior if i==0 else prior+rng.normal(0,.3,7)
 fit=least_squares(residual,np.clip(start,low,high),bounds=(low,high),max_nfev=400,ftol=1e-10,gtol=1e-10,xtol=1e-10)
 results.append(fit)
fit=min(results,key=lambda f:np.linalg.norm(f.fun[:6]));d.qpos[qa]=fit.x;mujoco.mj_forward(m,d)
error=float(np.linalg.norm(d.site_xpos[palm]-position));angle=float(np.linalg.norm(Rotation.from_matrix(rotation@d.site_xmat[palm].reshape(3,3).T).as_rotvec()))
if error>.001 or angle>.01:raise RuntimeError(f'Unreachable reset: {error}m, {angle}rad')
J=[]
for b in feet:
 jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,b);J.append(np.vstack([jp,jr]))
J=np.vstack(J);wrench=np.linalg.lstsq(J[:,:6].T,d.qfrc_bias[:6],rcond=None)[0]
leg=[m.actuator(side+'_'+n).id for side in ('left','right') for n in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]
controls=d.actuator_length.copy();gravity={}
for act in leg:
 joint=int(m.actuator_trnid[act,0]);v=m.jnt_dofadr[joint]
 torque=d.qfrc_bias[v]-J[:,v]@wrench
 controls[act]+=torque/m.actuator_gainprm[act,0];gravity[m.actuator(act).name]=float(torque)
# Retain the optimized physical grasp targets, replacing the reset's arm and leg targets.
base=np.array(r['controls'][40]);base[leg]=controls[leg]
for n,value in zip(names,fit.x):
 act=m.actuator('rh_A_'+n[3:] if n.startswith('rh_') else n).id;base[act]=value
base=np.clip(base,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
r['initial_root']=d.qpos[:7].tolist();r['initial_joints']={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(1,m.njnt)}
r['controls']=[base.tolist()]*600
r['reset_fit']=dict(bend_rad=a.bend,palm_error_m=error,palm_rotation_error_rad=angle,gravity_feedforward_Nm=gravity,planned_foot_wrenches=wrench.tolist())
Path(a.output).write_text(json.dumps(r)+'\n');print(json.dumps(r['reset_fit'],indent=2));print('root',r['initial_root'])

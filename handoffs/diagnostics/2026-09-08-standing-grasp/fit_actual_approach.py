"""Geometric fit only: actual Door55 approach-005 stopped root and legs fixed, canonical hand; arm/waist IK only."""
import json,copy
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares,minimize
from scipy.spatial.transform import Rotation
from doorbench.dexterous.environment import DexterousDoorEnv
from scripts.dexterous.plan_acquisition import collision_failures

out=Path('/tmp/doorbench-standing-grasp/fit-approach-005-v2');out.mkdir(exist_ok=False)
root=Path('/tmp/doorbench-dexterous-humanoid/out/dexterous');robot=root/'robot/h1-shadow.xml';rm=mujoco.MjModel.from_xml_path(str(robot));snapshot=np.load('/tmp/doorbench-h1-locomotion/out/locomotion/approach-005/trajectory.npz')['qpos'][-1]
s=DexterousDoorEnv(root/'assets/doors/db0055_swing_single',robot,json.loads(robot.with_suffix('.audit.json').read_text()));s.reset(images=False,randomize=False);m,d=s.m,s.d
ref=json.loads(Path('/tmp/doorbench-isaac-integration/configs/dexterous/isaac-door55-reference.json').read_text())
d.qpos[s.root_qadr:s.root_qadr+7]=ref['initial_root']
for n,v in ref['initial_joints'].items():d.qpos[m.jnt_qposadr[m.joint('robot/'+n).id]]=v
mujoco.mj_forward(m,d);palm=m.site('robot/rh_palm_touch').id;goal_pos=d.site_xpos[palm].copy();goal_rot=d.site_xmat[palm].reshape(3,3).copy();canonical=d.qpos.copy()
d.qpos[:]=snapshot
# Exact canonical finger geometry is frozen. WRJ is part of the fitted arm.
for n,v in ref['initial_joints'].items():
 if n.startswith('rh_') and not n.startswith('rh_WRJ'):d.qpos[m.jnt_qposadr[m.joint('robot/'+n).id]]=v
base=d.qpos.copy();names=ref['workspace_fit']['joint_names'];ids=[m.joint('robot/'+n).id for n in names];qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids];n=len(ids);lo=np.r_[m.jnt_range[ids,0],[-1e-10,-1e-10]];hi=np.r_[m.jnt_range[ids,1],[1e-10,1e-10]]
prior=np.r_[canonical[qa],[0.,0.]];jp=np.zeros((3,m.nv));jr=jp.copy()
def setx(x):
 d.qpos[:]=base;d.qpos[qa]=x[:n];d.qpos[s.root_qadr:s.root_qadr+2]+=x[n:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
def residual(x):
 setx(x);p=d.site_xpos[palm]-goal_pos;rv=Rotation.from_matrix(d.site_xmat[palm].reshape(3,3)@goal_rot.T).as_rotvec();radial=max(0,np.linalg.norm(x[n:])-.15);return np.r_[100*p,10*rv,.003*(x[:n]-prior[:n]),.03*x[n:],100*radial]
best=None;rows=[];rng=np.random.default_rng(43)
for k in range(16):
 x=prior.copy()
 if k:x[:n]+=rng.normal(0,.45,n);x[n:]=rng.uniform(-.1,.1,2)
 fit=least_squares(residual,np.clip(x,lo+np.minimum(1e-8,(hi-lo)*.1),hi-np.minimum(1e-8,(hi-lo)*.1)),bounds=(lo,hi),max_nfev=400,ftol=1e-11,xtol=1e-11,gtol=1e-10)
 setx(fit.x);mujoco.mj_collision(m,d);pos=np.linalg.norm(goal_pos-d.site_xpos[palm]);rot=np.linalg.norm(Rotation.from_matrix(goal_rot@d.site_xmat[palm].reshape(3,3).T).as_rotvec());cf=collision_failures(m,d);row=dict(seed=k,position_error_m=float(pos),orientation_error_rad=float(rot),root_displacement_m=float(np.linalg.norm(fit.x[n:])),arm_joint_values=fit.x[:n].tolist(),root_xy_offset=fit.x[n:].tolist(),collision_failures=cf)
 rows.append(row);np.save(out/f'candidate-{k:02d}.npy',d.qpos.copy());print(json.dumps(row),flush=True)
 score=100*pos+10*rot+100*max(0,np.linalg.norm(fit.x[n:])-.1501)+sum(f['depth_m'] for f in cf)*10
 if best is None or score<best[0]:best=(score,k,d.qpos.copy(),row)
 if pos<.0001 and rot<.001 and not cf and np.linalg.norm(fit.x[n:])<=.1501:break
score,k,q,row=best;result=copy.deepcopy(ref);result['initial_root']=q[s.root_qadr:s.root_qadr+7].tolist();result['initial_joints']={m.joint(j).name.removeprefix('robot/'):float(q[m.jnt_qposadr[j]]) for j in s.joints};result['standing_fit']=dict(scope=__doc__,source_snapshot='/tmp/doorbench-h1-locomotion/out/locomotion/approach-005/trajectory.npz',source_frame=-1,source_height_m=float(snapshot[s.root_qadr+2]),fitted_joint_names=names,fixed_leg_posture=True,goal_position=goal_pos.tolist(),goal_rotation=goal_rot.tolist(),best=row)
(out/'reference.json').write_text(json.dumps(result,indent=2)+'\n');(out/'report.json').write_text(json.dumps(dict(scope=__doc__,best=row,candidates=rows),indent=2)+'\n');print('BEST',json.dumps(row));s.close()

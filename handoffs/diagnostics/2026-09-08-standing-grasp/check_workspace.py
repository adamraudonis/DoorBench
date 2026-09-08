import json,copy
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from doorbench.dexterous.environment import DexterousDoorEnv
from scripts.dexterous.plan_acquisition import collision_failures
root=Path('/tmp/doorbench-dexterous-humanoid/out/dexterous');r=root/'robot/h1-shadow.xml';s=DexterousDoorEnv(root/'assets/doors/db0055_swing_single',r,json.loads(r.with_suffix('.audit.json').read_text()));s.reset(images=False,randomize=False);m,d=s.m,s.d
p=Path('/tmp/doorbench-standing-grasp/fit-001');ref=json.loads((p/'hold-reference.json').read_text());d.qpos[s.root_qadr:s.root_qadr+7]=ref['initial_root']
for n,v in ref['initial_joints'].items():d.qpos[m.jnt_qposadr[m.joint('robot/'+n).id]]=v
mujoco.mj_forward(m,d);base=d.qpos.copy();palm=m.site('robot/rh_palm_touch').id;lever=m.geom('leaf_handle_lever_col_n').id;R=d.geom_xmat[lever].reshape(3,3);relative_pos=R.T@(d.site_xpos[palm]-d.geom_xpos[lever]);relative_rot=R.T@d.site_xmat[palm].reshape(3,3)
names=ref['workspace_fit']['joint_names'];ids=[m.joint('robot/'+n).id for n in names];qa=m.jnt_qposadr[ids];lo=m.jnt_range[ids,0];hi=m.jnt_range[ids,1];start=d.qpos[qa].copy();rows=[];poses=[]
for h,leaf in [(float(v),0.) for v in np.linspace(0,.87,10)]+[(.87,float(v)) for v in np.linspace(.03,.15,5)]:
 d.qpos[:]=base;d.qpos[m.jnt_qposadr[m.joint('leaf_handle_hinge').id]]=h;d.qpos[m.jnt_qposadr[m.joint('leaf_hinge').id]]=leaf;mujoco.mj_forward(m,d);R=d.geom_xmat[lever].reshape(3,3).copy();goal_pos=d.geom_xpos[lever]+R@relative_pos;goal_rot=R@relative_rot
 def fun(x):
  d.qpos[qa]=x;mujoco.mj_kinematics(m,d);return np.r_[100*(d.site_xpos[palm]-goal_pos),10*Rotation.from_matrix(d.site_xmat[palm].reshape(3,3)@goal_rot.T).as_rotvec(),.001*(x-start)]
 fit=least_squares(fun,np.clip(start,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=400,ftol=1e-11,xtol=1e-11,gtol=1e-10);start=fit.x.copy();d.qpos[qa]=fit.x;mujoco.mj_forward(m,d);cf=collision_failures(m,d);row=dict(handle_rad=h,leaf_rad=leaf,position_error_m=float(np.linalg.norm(goal_pos-d.site_xpos[palm])),orientation_error_rad=float(np.linalg.norm(Rotation.from_matrix(goal_rot@d.site_xmat[palm].reshape(3,3).T).as_rotvec())),minimum_joint_margin_rad=float(np.min(np.minimum(fit.x-lo,hi-fit.x))),collisions=cf,arm_qpos=fit.x.tolist());rows.append(row);poses.append(d.qpos.copy());print(json.dumps(row))
(p/'workspace-report.json').write_text(json.dumps(dict(scope='Kinematic handle and initial-leaf workspace only, fixed stopped-walking legs/root',samples=rows),indent=2)+'\n');np.savez_compressed(p/'workspace.npz',qpos=poses)
rm=mujoco.MjModel.from_xml_path(str(r));raw=np.load('/tmp/doorbench-h1-locomotion/out/locomotion/phase-stop-001/trajectory.npz')['qpos'][-1];d.qpos[:]=base
for j in range(rm.njnt):
 if rm.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:
  jj=m.joint('robot/'+rm.joint(j).name).id;d.qpos[m.jnt_qposadr[jj]]=np.clip(raw[rm.jnt_qposadr[j]],*m.jnt_range[jj])
mujoco.mj_forward(m,d);fail=collision_failures(m,d);print('NEUTRAL',json.dumps(fail));nr=copy.deepcopy(ref);nr['initial_joints']={m.joint(j).name.removeprefix('robot/'):float(d.qpos[m.jnt_qposadr[j]]) for j in s.joints};nr['scope']='Actual stopped-walking neutral arm posture, relocated geometrically to fittedDoor55bodytarget; no physical approach claim';nr.pop('acquisition',None);(p/'neutral-reference.json').write_text(json.dumps(nr,indent=2)+'\n');(p/'neutral-audit.json').write_text(json.dumps(dict(scope=nr['scope'],passed=not fail,failures=fail),indent=2)+'\n');s.close()

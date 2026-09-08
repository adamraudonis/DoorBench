#!/usr/bin/env python3
"""Screen a bounded H1 whole-body lever return from an exact archived state.

This creates and modifies only an unstepped geometry model. Root and foot poses
are not imposed on an active plant. Use the source package frozen with the run,
then independently validate interpolated geometry and physical motor execution.
"""
import argparse
from pathlib import Path
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-run',type=Path,required=True)
p.add_argument('--state-run',type=Path,required=True)
p.add_argument('--source-package',type=Path,required=True)
p.add_argument('--at',type=float,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
if args.output.exists():raise ValueError('Use a new evidence directory')
args.output.mkdir(parents=True)
sys.path.insert(0,str(args.source_package.resolve()))
import json,hashlib,importlib.util,numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_archive import unpacked
baseline=args.source_run.resolve();conf=json.loads((baseline/'manifest.json').read_text())['configuration'];r=Path(conf['robot']);s=DexterousDoorEnv(conf['door'],r,json.loads(r.with_suffix('.audit.json').read_text()));m,d=s.m,s.d
manifest=json.loads((args.state_run/'raw-transitions/manifest.json').read_text());time=args.at
chunk=next(c for c in manifest['chunks'] if c['interval_start_s']-1e-8<=time<c['interval_end_s']-1e-8);file=args.state_run/'raw-transitions'/chunk['file'];assert hashlib.file_digest(file.open('rb'),'sha256').hexdigest()==chunk['sha256']
with np.load(file) as a:rows=list(unpacked({k:a[k] for k in a.files}))
row=min(rows,key=lambda r:abs(r['interval_start_s']-time));base=np.asarray(row['qpos_before']);d.qpos[:]=base;mujoco.mj_kinematics(m,d);assert np.allclose(d.xpos[row['body_ids']],row['body_positions_world_m'],atol=1e-9,rtol=0)
hb=m.body('leaf_handle').id;rh=m.site('robot/rh_palm_touch').id;lh=m.site('robot/lh_palm_touch').id;H=d.xmat[hb].reshape(3,3);pr=H.T@(d.site_xpos[rh]-d.xpos[hb]);rr=H.T@d.site_xmat[rh].reshape(3,3);PL=d.site_xpos[lh].copy();RL=d.site_xmat[lh].reshape(3,3).copy()
hj=m.joint('leaf_handle_hinge').id;d.qpos[m.jnt_qposadr[hj]]=0.;mujoco.mj_kinematics(m,d);PR=d.xpos[hb]+d.xmat[hb].reshape(3,3)@pr;RR=d.xmat[hb].reshape(3,3)@rr;target_base=d.qpos.copy()
right=['right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['rh_WRJ2','rh_WRJ1'];left=['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1'];reports=[]

from scipy.spatial.transform import Rotation
# base is exact attained state; target_base sets only unstepped operator to rest.
d.qpos[:]=base;mujoco.mj_kinematics(m,d);feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')];FP=d.xpos[feet].copy();FR=d.xmat[feet].copy().reshape(2,3,3);free_ids=np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE);assert len(free_ids)==1;rq=int(m.jnt_qposadr[free_ids[0]]);rootR=Rotation.from_quat([*base[rq+4:rq+7],base[rq+3]]);rootP=base[rq:rq+3].copy()
legs=[side+'_'+j for side in ('left','right') for j in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')];names=legs+['torso']+right+left;js=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[js];start=base[qa].copy();results=[]
initial_h=float(base[m.jnt_qposadr[hj]]);previous=np.r_[np.zeros(6),start];extent=.02
for blend in np.linspace(0,1,41):
 target_base=base.copy();target_base[m.jnt_qposadr[hj]]=initial_h*(1-blend);d.qpos[:]=target_base;mujoco.mj_kinematics(m,d);PR=d.xpos[hb]+d.xmat[hb].reshape(3,3)@pr;RR=d.xmat[hb].reshape(3,3)@rr
 lo=np.r_[[-extent]*3,[-.06,-.06,-.08],m.jnt_range[js,0]+.01];hi=np.r_[[extent]*3,[.06,.06,.08],m.jnt_range[js,1]-.01];x0=previous.copy()
 def fun(x):
  d.qpos[:]=target_base;d.qpos[rq:rq+3]=rootP+x[:3];rot=(Rotation.from_rotvec(x[3:6])*rootR).as_quat();d.qpos[rq+3:rq+7]=np.r_[rot[3],rot[:3]];d.qpos[qa]=x[6:];mujoco.mj_kinematics(m,d)
  hand=np.r_[100*(d.site_xpos[rh]-PR),10*Rotation.from_matrix(RR@d.site_xmat[rh].reshape(3,3).T).as_rotvec(),100*(d.site_xpos[lh]-PL),10*Rotation.from_matrix(RL@d.site_xmat[lh].reshape(3,3).T).as_rotvec()]
  foot=np.concatenate([np.r_[100*(d.xpos[b]-FP[i]),10*Rotation.from_matrix(FR[i]@d.xmat[b].reshape(3,3).T).as_rotvec()] for i,b in enumerate(feet)])
  return np.r_[hand,foot,.01*(x[6:]-start),.02*x[:6]]
 fit=least_squares(fun,np.clip(x0,lo,hi),bounds=(lo,hi),max_nfev=2000,ftol=1e-12,xtol=1e-12,gtol=1e-12);previous=fit.x.copy();res=fun(fit.x);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d);cols=[]
 for c in d.contact[:d.ncon]:
  bs=[m.body(m.geom_bodyid[g]).name for g in c.geom];gs=[m.geom(int(g)).name for g in c.geom]
  if any(b.startswith('robot/') for b in bs) and not ('floor' in gs and any(b.endswith('_ankle_link') for b in bs)) and c.dist<-.003:cols.append(dict(depth=-float(c.dist),bodies=bs))
 torso=m.body('robot/torso_link').id;root_body=m.jnt_bodyid[free_ids[0]];robot_mass=m.body_subtreemass[root_body];com=d.subtree_com[root_body].copy();up=d.xmat[torso].reshape(3,3)[:,2]
 result=dict(progress=float(blend),goal_operator_rad=initial_h*(1-float(blend)),root_qpos_address=rq,torso_tilt_deg=float(np.degrees(np.arccos(np.clip(up[2],-1,1)))),robot_com_world_m=com.tolist(),robot_mass_kg=float(robot_mass),root_translation_bound_m=extent,right_position_error_m=float(np.linalg.norm(res[:3])/100),right_rotation_error_rad=float(np.linalg.norm(res[3:6])/10),left_position_error_m=float(np.linalg.norm(res[6:9])/100),left_rotation_error_rad=float(np.linalg.norm(res[9:12])/10),foot_position_errors_m=[float(np.linalg.norm(res[12+6*i:15+6*i])/100) for i in range(2)],foot_rotation_errors_rad=[float(np.linalg.norm(res[15+6*i:18+6*i])/10) for i in range(2)],root_delta_xyz_m=fit.x[:3].tolist(),root_delta_rotvec_rad=fit.x[3:6].tolist(),joints=dict(zip(names,fit.x[6:].tolist())),joint_margins_rad=dict(zip(names,np.minimum(fit.x[6:]-(m.jnt_range[js,0]+.01),(m.jnt_range[js,1]-.01)-fit.x[6:]).tolist())),forbidden_collisions=cols,nfev=fit.nfev,qpos=d.qpos.tolist());results.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ('qpos','joints','joint_margins_rad','forbidden_collisions')}|{'forbidden_collision_count':len(cols)}),flush=True)
out=args.output;(out/'report.json').write_text(json.dumps(dict(scope='Unstepped whole-body IK; actual landed feet fixed; no dynamics qualification; original limits plus0.01rad planning margin',input_files={str(Path(p)/'door.xml' if Path(p).is_dir() else Path(p)):hashlib.file_digest((Path(p)/'door.xml' if Path(p).is_dir() else Path(p)).open('rb'),'sha256').hexdigest() for p in (conf['robot'],conf['door'])},source_time_s=row['interval_start_s'],source_raw_chunk_sha256=chunk['sha256'],mujoco_version=mujoco.__version__,results=results),indent=2));s.close()

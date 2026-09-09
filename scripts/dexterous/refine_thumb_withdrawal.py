"""Unstepped thumb withdrawal: separate radially before retracting the arm."""
from pathlib import Path
import argparse,shutil
import json,hashlib
import numpy as np,mujoco
from scipy.optimize import least_squares
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.operation_teacher import smooth_phase
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--little-finger-clearance-rad',type=float,default=0.,help='Explicit smooth LFJ4 abduction during free withdrawal; geometry requires re-audit')
parser.add_argument('--screen',type=Path,required=True);parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
if not np.isfinite(args.little_finger_clearance_rad) or not 0<=args.little_finger_clearance_rad<=.08:raise ValueError('Bounded little-finger clearance angle required')
source=args.screen;screen=json.loads(source.read_text());trial=args.source_run
checks=[trial/n for n in ('report.json','independent-pad-audit.json','independent-whole-handle-audit.json')]
if not all(json.loads(p.read_text()).get('passed') is True for p in checks):raise ValueError('Complete physical and whole-handle source qualification required')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((trial/'manifest.json').read_text());cfg=manifest['configuration'];robot=Path(cfg['robot']);door=Path(cfg['door'])
if sha(robot)!=manifest['inputs']['robot']['sha256'] or sha(door/'door.xml')!=manifest['inputs']['door']['door.xml']:raise ValueError('Exact source model required')
if screen['source_trajectory_sha256']!=sha(trial/'trajectory.npz'):raise ValueError('Candidate belongs to another attained state')
out=args.output;out.mkdir(parents=True,exist_ok=False);frozen=out/'thumb-planner-source.py';shutil.copy2(__file__,frozen)
s=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));s.reset(randomize=False,images=False);m,d=s.m,s.d
rows=screen['trials'][0]['rows'];names=['rh_THJ'+str(i) for i in (5,4,3,2,1)];js=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[js]
thumb=m.body('robot/rh_thdistal').id;palm=m.site('robot/rh_palm_touch').id;lever=m.geom('leaf_handle_lever_col_n').id
geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==thumb and m.geom_contype[g]]
d.qpos[:]=rows[0]['qpos'];mujoco.mj_kinematics(m,d);initial=d.qpos[qa].copy();near=[]
for g in geoms:
 pair=np.zeros(6);distance=mujoco.mj_geomDistance(m,d,g,lever,.2,pair);near.append((distance,pair.copy()))
_,pair=min(near,key=lambda x:x[0]);local=d.xmat[thumb].reshape(3,3).T@(pair[:3]-d.xpos[thumb]);axis=d.geom_xmat[lever].reshape(3,3)[:,2];radial=pair[:3]-d.geom_xpos[lever];radial-=axis*(radial@axis);radial/=np.linalg.norm(radial)
low=np.minimum(initial,m.jnt_range[js,0]+.005);high=np.maximum(initial,m.jnt_range[js,1]-.005);high[1]=m.jnt_range[js[1],1]-.02
previous=np.clip(initial,low+1e-10,high-1e-10);worst=0.;initial_pad=pair[:3].copy();toward_center=-np.sign((initial_pad-d.geom_xpos[lever])@axis)
for row in rows:
 t=row['time_s'];q=np.array(row['qpos']);authored=q[qa].copy();d.qpos[:]=q;d.qpos[qa]=initial;mujoco.mj_kinematics(m,d)
 goal=d.xpos[thumb]+d.xmat[thumb].reshape(3,3)@local+radial*.025*float(smooth_phase(t/1.5))
 goal+=axis*(-float((goal-initial_pad)@axis)+toward_center*.004*float(smooth_phase(t/1.)))
 def residual(v):
  d.qpos[qa]=v;mujoco.mj_kinematics(m,d)
  pos=d.xpos[thumb]+d.xmat[thumb].reshape(3,3)@local
  return np.r_[100*(pos-goal),.06*(v-initial),.02*(v-previous)]
 if t<=4:
  fit=least_squares(residual,previous,bounds=(low,high),max_nfev=100);v=fit.x;worst=max(worst,float(np.linalg.norm(fit.fun[:3]))/100);held=v.copy()
 else:v=held+float(smooth_phase((t-4)/2))*(authored-held)
 if t==0:v=initial.copy()
 q[qa]=v;previous=v.copy()
 if args.little_finger_clearance_rad:
  j=m.joint('robot/rh_LFJ4').id;address=m.jnt_qposadr[j]
  blend=float(smooth_phase((t-4)/1.5))*(1-float(smooth_phase((t-8)/1.5)))
  q[address]+=args.little_finger_clearance_rad*blend
  if not m.jnt_range[j,0]<=q[address]<=m.jnt_range[j,1]:raise ValueError('Clearance candidate exceeds original finger joint range')
  row['finger_joints']['rh_LFJ4']=float(q[address])
 row['qpos']=q.tolist();row['finger_joints'].update(zip(names,v.tolist()));d.qpos[:]=q;mujoco.mj_kinematics(m,d)
 row['source_requested_palm_position']=row['palm_position'];row['source_requested_palm_rotation']=row['palm_rotation'];row['palm_position']=d.site_xpos[palm].tolist();row['palm_rotation']=d.site_xmat[palm].reshape(3,3).tolist()
screen['scope']='Feasible wrist route with radial thumb material-pad withdrawal; unstepped candidate only'
screen['postprocess']=dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,frozen,*checks]},radial_direction_world=radial.tolist(),radial_goal_m=.025,axial_scope='Hold initial pad axial coordinate plus 4 mm toward shaft center before arm retreat',maximum_thumb_position_residual_m=worst,little_finger_clearance_rad=args.little_finger_clearance_rad,little_finger_clearance_schedule_s=[4,5.5,8,9.5])
(out/'report.json').write_text(json.dumps(screen,indent=2)+'\n');print(screen['postprocess']);s.close()

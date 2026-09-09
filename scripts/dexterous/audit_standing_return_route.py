"""Independently audit an unstepped return route, including the whole handle."""
from pathlib import Path
import argparse,shutil
import json,hashlib,re,numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.landed_left_audit import static_pose_check
from doorbench.dexterous.grasp_verification import shadow_surface_qualified
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--robot',type=Path,required=True);p.add_argument('--door',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
if args.output.exists():p.error('Fresh audit output required')
robot=args.robot;door=args.door;source=args.plan
r=json.loads(source.read_text())
if r.get('passed') is not True or len(r.get('rows',[]))!=41:raise ValueError('Complete candidate required')
for name,digest in r['source_sha256'].items():
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Source evidence changed: '+name)
for model_path in (robot,door/'door.xml'):
 if r['source_sha256'].get(str(model_path))!=hashlib.sha256(model_path.read_bytes()).hexdigest():raise ValueError('Audit model differs from source')
frozen=args.output.parent/'return-audit-source.py'
if frozen.exists():raise ValueError('Fresh frozen audit source required')
shutil.copy2(__file__,frozen)
s=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));s.reset(randomize=False,images=False);m,d=s.m,s.d;path=np.array([x['qpos'] for x in r['rows']]);rq=s.root_qadr;roots=Slerp(np.linspace(0,1,41),Rotation.from_quat(path[:,rq+3:rq+7][:,[1,2,3,0]]));lh=m.site('robot/lh_palm_touch').id;rh=m.site('robot/rh_palm_touch').id;hb=m.body('leaf_handle').id;feet=[m.body('robot/'+n+'_ankle_link').id for n in ['left','right']];lever=m.geom('leaf_handle_lever_col_n').id
d.qpos[:]=path[0];mujoco.mj_kinematics(m,d);H=d.xmat[hb].reshape(3,3);hp=H.T@(d.site_xpos[rh]-d.xpos[hb]);hr=H.T@d.site_xmat[rh].reshape(3,3);lp=d.site_xpos[lh].copy();lr=d.site_xmat[lh].reshape(3,3).copy();fp=d.xpos[feet].copy();fr=d.xmat[feet].reshape(2,3,3).copy();bad=[];peakp=peakr=peakt=0.;qs=[]
for index,t in enumerate(np.linspace(0,8,401)):
 v=t/8;u=v**3*(10+v*(-15+6*v));coordinate=u*40;i=min(int(coordinate),39);f=coordinate-i;q=path[i]*(1-f)+path[i+1]*f;qr=roots(u).as_quat();q[rq+3:rq+7]=qr[[3,0,1,2]];d.qpos[:]=q;check=static_pose_check(m,d,coordinate=1.);H=d.xmat[hb].reshape(3,3)
 pe=max(np.linalg.norm(H.T@(d.site_xpos[rh]-d.xpos[hb])-hp),np.linalg.norm(d.site_xpos[lh]-lp),max(np.linalg.norm(d.xpos[b]-p) for b,p in zip(feet,fp)))
 er=max(np.linalg.norm(Rotation.from_matrix(H.T@d.site_xmat[rh].reshape(3,3)@hr.T).as_rotvec()),np.linalg.norm(Rotation.from_matrix(d.site_xmat[lh].reshape(3,3)@lr.T).as_rotvec()),max(np.linalg.norm(Rotation.from_matrix(d.xmat[b].reshape(3,3)@a.T).as_rotvec()) for b,a in zip(feet,fr)))
 tilt=float(np.degrees(np.arccos(np.clip(d.xmat[m.body('robot/torso_link').id].reshape(3,3)[2,2],-1,1))));invalid=[]
 for c in d.contact[:d.ncon]:
  bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
  if c.dist<=0 and 'leaf_handle' in bodies and any(b.startswith('robot/rh_') for b in bodies) and lever not in c.geom:invalid.append('contact outside grasped lever')
  if lever not in c.geom or c.dist>0:continue
  side=0 if c.geom[1]==lever else 1;b=int(m.geom_bodyid[c.geom[side]]);name=m.body(b).name
  if not name.startswith('robot/rh_'):continue
  match=re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(distal|middle|proximal)',name);R=d.xmat[b].reshape(3,3);local=R.T@(c.pos-d.xpos[b]);normal=R.T@(c.frame[:3]*(1 if side==0 else -1));center=d.geom_xpos[lever];axis=d.geom_xmat[lever].reshape(3,3)[:,2];rel=c.pos-center;a=float(rel@axis);radial=rel-a*axis;align=float((R@normal)@(-radial/max(np.linalg.norm(radial),1e-12)))
  if not match or not shadow_surface_qualified(*match.groups(),local,normal) or m.geom_size[lever,1]-abs(a)<.001 or align<=.8:invalid.append(name)
 peakp=max(peakp,float(pe));peakr=max(peakr,float(er));peakt=max(peakt,tilt);qs.append(q.tolist())
 if not check['passed'] or invalid or pe>.001 or er>.01 or tilt>4:bad.append(dict(index=index,time_s=float(t),position_error_m=float(pe),rotation_error_rad=float(er),torso_tilt_deg=tilt,invalid_contact_surfaces=invalid,collision=check))
qa=[m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE and (m.joint(j).name or '').startswith('robot/')];velocity=np.diff(np.array(qs)[:,qa],axis=0)/.02;peakvelocity=float(np.max(abs(velocity)))
out=dict(passed=not bad and peakvelocity<=2.,samples=401,physics_steps=0,duration_s=8.,maximum_position_error_m=peakp,maximum_rotation_error_rad=peakr,maximum_torso_tilt_deg=peakt,maximum_joint_reference_velocity_rad_s=peakvelocity,bad_samples=bad,scope='Independent dense FK/collision/anatomical-surface and reference-rate screen. No dynamic force or actual release qualification.',input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [robot,door/'door.xml',source,frozen]})
args.output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['bad_samples','input_sha256']}));s.close()

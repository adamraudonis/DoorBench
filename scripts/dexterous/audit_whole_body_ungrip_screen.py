#!/usr/bin/env python3
"""Densely check a screened ungrip route including the attained-state prelude.

This checks geometry only, never forces from an unstepped/counterfactual solve.
The physical trial must still apply its unchanged actual loaded-patch gates.
"""
import argparse
from pathlib import Path
import sys,json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--screen',type=Path,required=True);args=parser.parse_args();run=args.source_run.resolve();sys.path.insert(0,str(run.with_name(run.name+'-source')))
from doorbench.dexterous.environment import DexterousDoorEnv
source=args.screen.resolve();report=json.load(open(source));rows=report['trials'][0]['rows'];config=json.load(open(run/'manifest.json'))['configuration'];rp=Path(config['robot']);s=DexterousDoorEnv(config['door'],rp,json.load(open(rp.with_suffix('.audit.json'))));m,d=s.m,s.d
with np.load(run/'trajectory.npz') as a:actual=a['terminal_qpos'].copy();actualv=a['terminal_qvel'].copy()
rq=m.jnt_qposadr[m.joint('robot/free_base').id];rh=m.site('robot/rh_palm_touch').id;lh=m.site('robot/lh_palm_touch').id;hb=m.body('leaf_handle').id;lever=m.geom('leaf_handle_lever_col_n').id;floor=m.geom('floor').id;feet=[m.body('robot/'+v+'_ankle_link').id for v in ('left','right')];torso=m.body('robot/torso_link').id
d.qpos[:]=actual;mujoco.mj_kinematics(m,d);actualp=d.site_xpos[rh].copy();actualr=d.site_xmat[rh].reshape(3,3).copy();leftp=d.site_xpos[lh].copy();leftr=d.site_xmat[lh].reshape(3,3).copy();footp=d.xpos[feet].copy();footr=d.xmat[feet].reshape(2,3,3).copy();hp=d.xpos[hb].copy();hr=d.xmat[hb].reshape(3,3).copy()
planrows=[dict(time_s=0.,phase='attained_start',qpos=actual.tolist(),palm_position=actualp.tolist(),palm_rotation=actualr.tolist())]+[dict(time_s=x['time_s']+.5,phase=x['phase'],qpos=x['qpos'],palm_position=x['palm_position'],palm_rotation=x['palm_rotation']) for x in rows]
qs=np.asarray([x['qpos'] for x in planrows]);ts=np.array([x['time_s'] for x in planrows]);ps=np.array([x['palm_position'] for x in planrows]);rs=np.array([x['palm_rotation'] for x in planrows]);limited=np.flatnonzero(m.jnt_limited);scalar=[j for j in limited if m.jnt_type[j] in (2,3)];qa=m.jnt_qposadr[scalar];bad=[];maxvals=dict(palm_position_m=0.,palm_rotation_rad=0.,left_position_m=0.,left_rotation_rad=0.,foot_position_m=0.,foot_rotation_rad=0.,joint_violation_rad=0.,torso_tilt_deg=0.,forbidden_depth_m=0.,loopback_violation_rad=0.);samples=[]
def rinterp(a,b,f):return Rotation.from_rotvec(f*Rotation.from_matrix(b@a.T).as_rotvec()).as_matrix()@a
for t in np.unique(np.r_[np.linspace(0,ts[-1],2001),ts]):
 i=min(len(ts)-2,max(0,int(np.searchsorted(ts,t,side='right')-1)));f=float((t-ts[i])/(ts[i+1]-ts[i]));d.qpos[:]=(1-f)*qs[i]+f*qs[i+1];ar=Rotation.from_quat([*qs[i,rq+4:rq+7],qs[i,rq+3]]).as_matrix();br=Rotation.from_quat([*qs[i+1,rq+4:rq+7],qs[i+1,rq+3]]).as_matrix();q=Rotation.from_matrix(rinterp(ar,br,f)).as_quat();d.qpos[rq+3:rq+7]=np.r_[q[3],q[:3]];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d)
 ep=float(np.linalg.norm(d.site_xpos[rh]-((1-f)*ps[i]+f*ps[i+1])));er=float(np.linalg.norm(Rotation.from_matrix(rinterp(rs[i],rs[i+1],f)@d.site_xmat[rh].reshape(3,3).T).as_rotvec()));lp=float(np.linalg.norm(d.site_xpos[lh]-leftp));lr=float(np.linalg.norm(Rotation.from_matrix(leftr@d.site_xmat[lh].reshape(3,3).T).as_rotvec()));fp=max(float(np.linalg.norm(d.xpos[b]-footp[k])) for k,b in enumerate(feet));fr=max(float(np.linalg.norm(Rotation.from_matrix(footr[k]@d.xmat[b].reshape(3,3).T).as_rotvec())) for k,b in enumerate(feet));violation=float(np.maximum(m.jnt_range[scalar,0]-d.qpos[qa],d.qpos[qa]-m.jnt_range[scalar,1]).max());tilt=float(np.degrees(np.arccos(np.clip(d.xmat[torso].reshape(3,3)[2,2],-1,1))));cols={};invalid={}
 for cc in d.contact[:d.ncon]:
  bs=[m.body(m.geom_bodyid[g]).name for g in cc.geom]
  if cc.dist<-.003 and any(b.startswith('robot/') for b in bs) and not (floor in cc.geom and any(b.endswith('_ankle_link') for b in bs)):key=' + '.join(bs);cols[key]=max(cols.get(key,0),-float(cc.dist))
  if cc.dist<-.00005 and lever in cc.geom:
   g=int(cc.geom[1] if cc.geom[0]==lever else cc.geom[0]);b=m.geom_bodyid[g];name=m.body(b).name
   if name.startswith('robot/rh_'):
    mat=d.xmat[b].reshape(3,3);p=mat.T@(cc.pos-d.xpos[b]);normal=mat.T@((1 if cc.geom[0]==g else -1)*cc.frame[:3]);axis=d.geom_xmat[lever].reshape(3,3)[:,2];v=cc.pos-d.geom_xpos[lever];ax=np.dot(v,axis);rad=v-ax*axis;align=np.dot(mat@normal,-rad/max(np.linalg.norm(rad),1e-9));valid=name.endswith('distal') and p[1]<-.001 and .002<=p[2]<=.040 and -normal[1]>.5 and m.geom_size[lever,1]-abs(ax)>=.001 and align>.8
    if not valid:invalid[name]=max(invalid.get(name,0),-float(cc.dist))
 loop=max(float(d.qpos[m.jnt_qposadr[m.joint(f'robot/{side}_{digit}J1').id]]-d.qpos[m.jnt_qposadr[m.joint(f'robot/{side}_{digit}J2').id]]) for side in ('lh','rh') for digit in ('FF','MF','RF','LF'))
 vals=dict(palm_position_m=ep,palm_rotation_rad=er,left_position_m=lp,left_rotation_rad=lr,foot_position_m=fp,foot_rotation_rad=fr,joint_violation_rad=violation,torso_tilt_deg=tilt,forbidden_depth_m=max(list(cols.values())+[0]),loopback_violation_rad=loop)
 for k,v in vals.items():maxvals[k]=max(maxvals[k],v)
 if cols or invalid or ep>.0001 or er>.001 or lp>.0001 or lr>.001 or fp>.0001 or fr>.001 or violation>.02 or tilt>12 or loop>.02:bad.append(dict(time_s=float(t),forbidden_collisions=cols,invalid_patches=invalid,**vals))
 samples.append(dict(time_s=float(t),**vals))
endpoint=None
if report.get('configuration',{}).get('withdrawal_profile','').startswith('clearance-lift-'):
 d.qpos[:]=qs[-1];mujoco.mj_kinematics(m,d)
 hand_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
 environment_geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and not m.body(m.geom_bodyid[g]).name.startswith('robot/')]
 pairs=[(float(mujoco.mj_geomDistance(m,d,g,h,.5,None)),m.geom(g).name,m.geom(h).name) for g in hand_geoms for h in environment_geoms]
 distance,hand_name,scene_name=min(pairs)
 endpoint=dict(required_clearance_m=.04,minimum_all_rh_environment_clearance_m=distance,closest_hand_geom=hand_name,closest_environment_geom=scene_name,hand_shapes=len(hand_geoms),environment_shapes=len(environment_geoms),passed=distance>=.04)
 if not endpoint['passed']:bad.append(dict(endpoint_clearance=endpoint))
result=dict(schema='doorbench.regrasp-interpolation-screen.v1',source_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),scope='Unstepped geometry only; static penetration patch diagnostic at50um is not the actual loaded-patch gate',passed=not bad,samples=len(samples),maxima=maxvals,failures=bad,endpoint_clearance=endpoint)
(source.parent/'interpolation-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2));print('first failures',bad[:3]);
if not bad:
 names=list(rows[0]['joints']);fingers=list(rows[0]['finger_joints']);
 for row in planrows:
  q=np.array(row.pop('qpos'));row['root']=q[rq:rq+7].tolist();row['joints']={n:float(q[m.jnt_qposadr[m.joint('robot/'+n).id]]) for n in names};row['finger_joints']={n:float(q[m.jnt_qposadr[m.joint('robot/'+n).id]]) for n in fingers};row['palm_position_handle']=(hr.T@(np.array(row.pop('palm_position'))-hp)).tolist();row['palm_rotation_handle']=(hr.T@np.array(row.pop('palm_rotation'))).tolist()
 source_report=json.load(open(run/'report.json'));delay=report['initial_time_s']-(source_report['opening_clock_offset_s']+source_report['handoffs']['right_release'])
 plan=dict(start_after_release_s=delay,schema='doorbench.whole-body-ungrip-plan.v1',scope='Actual-state-specific unstepped candidate; not physical qualification',initial_episode_time_s=report['initial_time_s'],source_trajectory_sha256=report['source_trajectory_sha256'],initial_root=actual[rq:rq+7].tolist(),initial_joints={n:float(actual[m.jnt_qposadr[m.joint('robot/'+n).id]]) for n in names+fingers},initial_operator_rad=float(actual[m.jnt_qposadr[m.joint('leaf_handle_hinge').id]]),initial_leaf_rad=float(actual[m.jnt_qposadr[m.joint('leaf_hinge').id]]),body_names=names,finger_names=fingers,rows=planrows,interpolation_audit=result,endpoint_clearance=endpoint)
 (source.parent/'target-plan.json').write_text(json.dumps(plan,indent=2))
s.close()

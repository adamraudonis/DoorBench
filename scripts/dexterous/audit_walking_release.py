#!/usr/bin/env python3
"""Independently reconstruct actual interval contact anatomy after hand transfer."""
import argparse
from pathlib import Path
import sys,json,hashlib,importlib.util,numpy as np,mujoco
from scipy.spatial.transform import Rotation
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--from-time',type=float,default=53.5);args=parser.parse_args();root=Path(__file__).resolve().parents[2];run=args.run.resolve();sys.path.insert(0,str(run.with_name(run.name+'-source')))
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
spec=importlib.util.spec_from_file_location('doorbench.dexterous._independent_pad',root/'doorbench/dexterous/isaac_pad_audit.py');pad=importlib.util.module_from_spec(spec);spec.loader.exec_module(pad)
c=json.load(open(run/'manifest.json'))['configuration'];robot=Path(c['robot']);s=DexterousDoorEnv(c['door'],robot,json.load(open(robot.with_suffix('.audit.json'))));m,d=s.m,s.d
lever=m.geom('leaf_handle_lever_col_n').id;hqa=m.jnt_qposadr[m.joint('leaf_handle_hinge').id];lqa=m.jnt_qposadr[m.joint('leaf_hinge').id];caps=m.actuator_forcerange
count=0;post_count=0;worstforce=0.;worstjoint=0.;frameerr=0.;wrong=[];tail=[];previous=None;continuity=True
for raw in NativeTransitionArchive.read(run/'raw-transitions'):
 count+=1;q=np.asarray(raw['qpos_before']);post=np.asarray(raw['qpos_after']);v=np.asarray(raw['qvel_before']);forces=np.asarray(raw['actuator_force']);controls=np.asarray(raw['controls']);worstforce=max(worstforce,float(np.max(forces-caps[:,1])),float(np.max(caps[:,0]-forces)));continuity&=previous is None or np.array_equal(previous,q);previous=post.copy()
 js=np.flatnonzero(m.jnt_limited);qa=m.jnt_qposadr[js];worstjoint=max(worstjoint,float(np.max(m.jnt_range[js,0]-q[qa])),float(np.max(q[qa]-m.jnt_range[js,1])),float(np.max(m.jnt_range[js,0]-post[qa])),float(np.max(post[qa]-m.jnt_range[js,1])))
 if raw['interval_start_s']<args.from_time:continue
 post_count+=1;d.qpos[:]=q;mujoco.mj_kinematics(m,d);ids=raw['body_ids'];bp=np.asarray(raw['body_positions_world_m']);br=np.asarray(raw['body_rotations_world']);frameerr=max(frameerr,float(np.max(np.abs(d.xpos[ids]-bp))),float(np.max(np.abs(d.xmat[ids].reshape(-1,3,3)-br))));poses={m.body(b).name:np.r_[p,Rotation.from_matrix(r).as_quat()] for b,p,r in zip(ids,bp,br)};contacts=[]
 for contact in raw['contacts']:
  if lever not in contact['geom']:continue
  other=1 if contact['geom'][0]==lever else 0;body=contact['body'][other];name=m.body(body).name
  if not name.startswith('robot/rh_'):continue
  contacts.append(dict(body=name,position=contact['position_world_m'],normal=(np.asarray(contact['frame_world'])[0]*(1 if other==1 else -1)),normal_force_N=max(0,contact['wrench_contact_frame'][0])))
 audit=pad.shadow_physx_pad_grasp(contacts,poses,d.geom_xpos[lever],d.geom_xmat[lever].reshape(3,3)[:,2],half_length=m.geom_size[lever,1],radius=m.geom_size[lever,0]);bad=[x for x in audit['contacts'] if x['normal_force_N']>.05 and not x['pad_qualified']]
 if bad:wrong.append(dict(t=raw['interval_start_s'],contacts=bad))
 if 59.5-1e-8 <= raw['interval_start_s'] < 60.-1e-8:tail.append(dict(time=raw['interval_start_s'],operator=float(q[hqa]),leaf=float(q[lqa]),valid=audit['valid_pad_grasp'],forces=audit['digit_forces_N']))
result=dict(scope='Independent actual-contact/body-frame archive audit; task qualification remains in the separately frozen report',all_intervals=count,post_return_start_intervals=post_count,raw_state_continuity=bool(continuity),maximum_motor_cap_violation_N_or_Nm=worstforce,maximum_all_scene_joint_violation_rad=worstjoint,maximum_actual_body_frame_error=frameerr,invalid_distal_patches=wrong,return_final_half_second_samples=len(tail),return_final_half_second_all_five_distal_grasp=all(x['valid'] for x in tail),return_final_half_second_maximum_abs_operator_rad=max(abs(x['operator']) for x in tail),return_final_half_second_minimum_leaf_rad=min(x['leaf'] for x in tail),return_final_half_second_minimum_digit_forces_N={k:min(x['forces'][k] for x in tail) for k in ('ff','mf','rf','lf','th')},original_trial_passed=json.load(open(run/'report.json'))['passed'])
(run/'independent-release-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));s.close()

#!/usr/bin/env python3
"""Independent native collision screen for a proposed Isaac reset/workspace.

This does not certify PhysX contacts or runtime stability. It rejects obvious
intersections before spending a live simulation trial on an impossible grasp.
"""
import argparse,json
from pathlib import Path
import mujoco
from doorbench.dexterous.environment import DexterousDoorEnv

p=argparse.ArgumentParser();p.add_argument('--robot',required=True);p.add_argument('--door',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
a=p.parse_args();robot=Path(a.robot);r=json.loads(Path(a.reference).read_text())
s=DexterousDoorEnv(a.door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));m,d=s.m,s.d
s.reset(images=False,randomize=False);d.qpos[s.root_qadr:s.root_qadr+7]=r['initial_root']
for n,q in r['initial_joints'].items():d.qpos[m.jnt_qposadr[m.joint('robot/'+n).id]]=q
workspace=r.get('workspace_fit');samples=workspace['poses'] if workspace else [(0,0)];failures=[];limit_failures=[];worst=0.
for i,(handle,leaf) in enumerate(samples):
 if workspace:
  for n,q in zip(workspace['joint_names'],workspace['arm_qpos'][i]):d.qpos[m.jnt_qposadr[m.joint('robot/'+n).id]]=q
 for n,q in [('leaf_handle_hinge',handle),('leaf_hinge',leaf),('leaf_latch_bolt_slide',.0127*handle/.87)]:d.qpos[m.jnt_qposadr[m.joint(n).id]]=q
 for j in s.joints:
  q=float(d.qpos[m.jnt_qposadr[j]]);low,high=m.jnt_range[j]
  if m.jnt_limited[j] and not low-1e-6<=q<=high+1e-6:limit_failures.append(dict(sample=i,joint=m.joint(j).name,q=q,range=[float(low),float(high)]))
 mujoco.mj_forward(m,d)
 for c in d.contact[:d.ncon]:
  bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom];geoms=[m.geom(int(g)).name for g in c.geom]
  if not any(n.startswith('robot/') for n in bodies):continue
  foot='floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)
  threshold=.003 if foot else .001
  worst=max(worst,-float(c.dist) if not foot else 0)
  if c.dist < -threshold:failures.append(dict(sample=i,handle_rad=handle,leaf_rad=leaf,depth_m=-float(c.dist),bodies=bodies,geoms=geoms))
report=dict(passed=not failures and not limit_failures,max_nonfoot_penetration_m=worst,samples=len(samples),failures=failures,limit_failures=limit_failures,scope='Native geometric rejection screen; live PhysX validation still required')
Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='failures'}));print('intersections',len(failures));s.close()
raise SystemExit(int(bool(failures or limit_failures)))

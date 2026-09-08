#!/usr/bin/env python3
"""Replay actual base/proprioception through bounded palm target correction.

This is an unstepped geometry/rate screen, never a physical contact or success
claim. Archived actual dynamics supply the current body and scalar joints.
Only the seven commanded left arm joints are changed on a separate FK model.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.run.resolve();source=run.with_name(run.name+'-source');own=Path(__file__).resolve().parents[2];sys.path.insert(0,str(own))
 from doorbench.dexterous.environment import DexterousDoorEnv
 from doorbench.dexterous.actual_base_palm import ActualBasePalmCorrection
 cfg=json.load(open(run/'manifest.json'))['configuration'];robot=Path(cfg['robot']);sim=DexterousDoorEnv(cfg['door'],robot,json.load(open(robot.with_suffix('.audit.json'))));m,d=sim.m,sim.d
 robot_model=mujoco.MjModel.from_xml_path(str(robot));names=['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1'];correction=ActualBasePalmCorrection(robot_model,names,'lh_palm_touch')
 scalar={m.joint(j).name.removeprefix('robot/'):int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.joint(j).name.startswith('robot/') and m.jnt_type[j] in (2,3)};cq=np.array([scalar[n] for n in names]);rq=m.jnt_qposadr[m.joint('robot/free_base').id];palm=m.site('robot/lh_palm_touch').id
 plan=json.load(open(source/'whole-body-panel-plan.json'));q0=np.asarray(plan['initial_qpos']);qa=np.asarray(plan['joint_qpos_addresses']);r0=Rotation.from_quat([*q0[rq+4:rq+7],q0[rq+3]]);rd=mujoco.MjData(m);cd=mujoco.MjData(m)
 with gzip.open(run/'panel-phase.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
 correction.begin(rows[0]['episode_time_s'],rows[0]['left_arm_targets'],rows[0]['left_arm_target_velocity'])
 manifest=json.load(open(run/'raw-transitions/manifest.json'));assert manifest['complete'];count=0;geometric=0;maxima={};failures=[];values=[];floor=m.geom('floor').id
 for chunk in manifest['chunks']:
  if chunk['interval_end_s']<rows[0]['episode_time_s']:continue
  file=run/'raw-transitions'/chunk['file']
  if hashlib.file_digest(file.open('rb'),'sha256').hexdigest()!=chunk['sha256']:raise ValueError('Changed archived source')
  with np.load(file) as raw:
   for i,t in enumerate(raw['interval_start_s']):
    if t<rows[0]['episode_time_s']-1e-9:continue
    row=rows[count]
    if abs(row['episode_time_s']-t)>1e-9:raise ValueError('Actual and target timestamps differ')
    q=raw['qpos_before'][i];d.qpos[:]=q;mujoco.mj_kinematics(m,d)
    begin,end=raw['body_offsets'][i:i+2];ids=raw['body_ids'][begin:end]
    if max(float(np.max(abs(d.xpos[ids]-raw['body_positions_world_m'][begin:end]))),float(np.max(abs(d.xmat[ids].reshape(-1,3,3)-raw['body_rotations_world'][begin:end]))))>1e-9:raise ValueError('Actual FK/body frames differ')
    x=np.asarray(row['planned_coordinates']);rd.qpos[:]=q0;rd.qpos[rq:rq+3]+=x[:3];quat=(Rotation.from_rotvec(x[3:6])*r0).as_quat();rd.qpos[rq+3:rq+7]=np.r_[quat[3],quat[:3]];rd.qpos[qa]=x[6:];mujoco.mj_kinematics(m,rd)
    base=Rotation.from_quat([*q[rq+4:rq+7],q[rq+3]]).as_matrix();goalp=base.T@(rd.site_xpos[palm]-q[rq:rq+3]);goalr=base.T@rd.site_xmat[palm].reshape(3,3)
    target,velocity,info=correction.update(float(t),{n:float(q[address]) for n,address in scalar.items()},goalp,goalr,row['left_arm_targets'])
    cd.qpos[:]=q;cd.qpos[cq]=target;mujoco.mj_kinematics(m,cd)
    vals={k:v for k,v in info.items() if k not in ('time_s','minimum_joint_margin_rad')};vals['target_change_from_nominal_rad']=float(np.max(abs(target-row['left_arm_targets'])))
    vals['maximum_world_fk_consistency_error']=max(float(np.max(abs(cd.site_xpos[palm]-(q[rq:rq+3]+base@correction.data.site_xpos[correction.palm])))),float(np.max(abs(cd.site_xmat[palm].reshape(3,3)-base@correction.data.site_xmat[correction.palm].reshape(3,3)))))
    if count%5==0 or count==len(rows)-1:
     geometric+=1;mujoco.mj_collision(m,cd);penetration=0.
     for contact in cd.contact[:cd.ncon]:
      bodies=[m.body(m.geom_bodyid[g]).name for g in contact.geom];allowed=floor in contact.geom and any(b.endswith('_ankle_link') for b in bodies)
      if not allowed and any(b.startswith('robot/') for b in bodies):penetration=max(penetration,-float(contact.dist))
     vals['nonfoot_robot_penetration_m']=penetration
     if penetration>.003:failures.append(dict(time_s=float(t),nonfoot_robot_penetration_m=penetration))
    for k,v in vals.items():maxima[k]=max(maxima.get(k,0.),float(v))
    if not np.isfinite(list(vals.values())).all():raise ValueError('Nonfinite proposed geometry')
    if vals['target_change_from_nominal_rad']>.1 or vals['maximum_world_fk_consistency_error']>1e-9:failures.append(dict(time_s=float(t),values=vals))
    values.append(np.r_[t,target,velocity,list(info.values())[1:]]);count+=1
  if count%1000<250:print(json.dumps(dict(samples=count,geometry_samples=geometric,failures=len(failures),maxima=maxima)),flush=True)
 if count!=len(rows):raise ValueError('Incomplete target replay')
 a.output.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.output/'targets.npz',values=np.array(values));result=dict(passed=not failures,scope='Unstepped actual-base target geometry/rate screen only. This replay starts at the archived first panel target; physical handoff continuity is checked separately.',samples=count,geometry_samples=geometric,maximum_target_correction_rad=.1,maxima=maxima,failures=failures,source_run=str(run),source_phase_sha256=hashlib.file_digest((run/'panel-phase.jsonl.gz').open('rb'),'sha256').hexdigest());(a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(a.output/'screen-source.py').write_bytes(Path(__file__).read_bytes());(a.output/'correction-source.py').write_bytes((own/'doorbench/dexterous/actual_base_palm.py').read_bytes());sim.close();print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Compare exact panel targets with actual palm geometry at an archived step."""
import argparse,gzip,json,hashlib,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.run.resolve();source=run.with_name(run.name+'-source');sys.path.insert(0,str(source))
 from doorbench.dexterous.environment import DexterousDoorEnv
 from doorbench.dexterous.screened_panel_path import ScreenedPanelPath
 cfg=json.load(open(run/'manifest.json'))['configuration'];robot=Path(cfg['robot']);sim=DexterousDoorEnv(cfg['door'],robot,json.load(open(robot.with_suffix('.audit.json'))));m,d=sim.m,sim.d
 with gzip.open(run/'panel-phase.jsonl.gz','rt') as f:
  for line in f:row=json.loads(line)
 manifest=json.load(open(run/'raw-transitions/manifest.json'));chunk=manifest['chunks'][-1];file=run/'raw-transitions'/chunk['file']
 if hashlib.file_digest(file.open('rb'),'sha256').hexdigest()!=chunk['sha256']:raise ValueError('Changed actual source')
 with np.load(file) as raw:actual=raw['qpos_before'][-1].copy();time=float(raw['interval_start_s'][-1]);begin,end=raw['body_offsets'][-2:];ids=raw['body_ids'][begin:end];bp=raw['body_positions_world_m'][begin:end];br=raw['body_rotations_world'][begin:end]
 if abs(row['episode_time_s']-time)>1e-9:raise ValueError('Target and actual state clocks differ')
 d.qpos[:]=actual;mujoco.mj_kinematics(m,d);error=max(float(np.max(abs(d.xpos[ids]-bp))),float(np.max(abs(d.xmat[ids].reshape(-1,3,3)-br))))
 if error>1e-9:raise ValueError('Actual model/body frames differ')
 palm=m.site('robot/lh_palm_touch').id;leaf=m.body('leaf').id;lr=d.xmat[leaf].reshape(3,3).copy();lp=d.xpos[leaf].copy();actual_p=d.site_xpos[palm].copy()
 plan=json.load(open(source/'whole-body-panel-plan.json'));q0=np.asarray(plan['initial_qpos']);rq=plan['root_qpos_address'];rotation=Rotation.from_quat([*q0[rq+4:rq+7],q0[rq+3]]);qa=np.asarray(plan['joint_qpos_addresses']);curve=ScreenedPanelPath(plan['progress'],plan['coordinates'],plan['duration_s']);delta=plan['final_leaf_angle_rad']-plan['initial_leaf_angle_rad'];data=mujoco.MjData(m)
 def desired(angle):
  x=curve.spline((angle-plan['initial_leaf_angle_rad'])/delta);data.qpos[:]=q0;data.qpos[rq:rq+3]+=x[:3];quat=(Rotation.from_rotvec(x[3:6])*rotation).as_quat();data.qpos[rq+3:rq+7]=np.r_[quat[3],quat[:3]];data.qpos[qa]=x[6:];mujoco.mj_kinematics(m,data);return data.site_xpos[palm].copy()
 target=desired(row['reference_aperture_rad']);h=1e-6;derivative=(desired(row['reference_aperture_rad']+h)-desired(row['reference_aperture_rad']-h))/(2*h)
 names=['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1'];data.qpos[:]=actual
 for name,value in zip(names,row['left_arm_targets']):data.qpos[m.jnt_qposadr[m.joint('robot/'+name).id]]=value
 mujoco.mj_kinematics(m,data);arm_target=data.site_xpos[palm].copy()
 result=dict(scope='Unstepped FK of archived same-time targets and actual state; no forces inferred from target errors.',episode_time_s=time,maximum_actual_body_frame_error=error,actual_leaf_rad=row['actual_aperture_rad'],reference_leaf_rad=row['reference_aperture_rad'],actual_palm_leaf_m=(lr.T@(actual_p-lp)).tolist(),planned_palm_leaf_m=(lr.T@(target-lp)).tolist(),whole_body_target_error_leaf_m=(lr.T@(target-actual_p)).tolist(),seven_arm_target_error_with_actual_base_leaf_m=(lr.T@(arm_target-actual_p)).tolist(),planned_palm_derivative_leaf_m_per_reference_rad=(lr.T@derivative).tolist(),source_chunk_sha256=chunk['sha256'])
 if row.get('corrected_chain_targets') is not None:
  chain_names=row['corrected_chain_joint_names'];chain=np.asarray(row['corrected_chain_targets'],float)
  if chain_names not in (names,['torso']+names) or chain.shape!=(len(chain_names),) or not np.isfinite(chain).all():raise ValueError('Require the explicit finite corrected chain')
  data.qpos[:]=actual
  for name,value in zip(chain_names,chain):data.qpos[m.jnt_qposadr[m.joint('robot/'+name).id]]=value
  mujoco.mj_kinematics(m,data)
  result['corrected_chain_joint_names']=chain_names
  result['full_corrected_chain_target_error_with_actual_base_leaf_m']=(lr.T@(data.site_xpos[palm]-actual_p)).tolist()
  offset=0. if row.get('normal_admittance') is None else row['normal_admittance']['normal_offset_m']
  result['commanded_goal_with_normal_offset_error_leaf_m']=(lr.T@(target+lr[:,1]*offset-actual_p)).tolist()
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'diagnostic-source.py').write_bytes(Path(__file__).read_bytes());sim.close();print(json.dumps(result,indent=2))


if __name__=='__main__':main()

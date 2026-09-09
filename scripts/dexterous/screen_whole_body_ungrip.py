#!/usr/bin/env python3
"""Screen an actual resting grasp adjustment and exact measured hand withdrawal.

All qpos writes are to a new, unstepped geometric model. This does not qualify
physical motion. Run audit_whole_body_ungrip_screen.py before motor execution.
"""
import argparse
from pathlib import Path
import sys,json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--measured-release',type=Path,required=True);parser.add_argument('--release-trajectory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--finger-lead-seconds',type=float,default=0.);parser.add_argument('--early-lift-m',type=float,default=.002);parser.add_argument('--withdrawal-profile',choices=('recorded','clearance-lift-v1','clearance-lift-v2','clearance-lift-v3','clearance-lift-v4'),default='recorded');parser.add_argument('--retarget-attained-grasp',action='store_true');parser.add_argument('--maximum-torso-tilt-deg',type=float);parser.add_argument('--right-wrist-margin-rad',type=float,default=.001);parser.add_argument('--right-orientation-weight',type=float,default=10.);args=parser.parse_args();
if not np.isfinite(args.finger_lead_seconds) or not 0<=args.finger_lead_seconds<=.5:raise ValueError('Require a bounded finger lead')
if not np.isfinite(args.early_lift_m) or not 0<=args.early_lift_m<=.01:raise ValueError('Require a bounded early geometric lift')
root=Path(__file__).resolve().parents[2];run=args.source_run.resolve();sys.path.insert(0,str(run.with_name(run.name+'-source')))
from doorbench.dexterous.environment import DexterousDoorEnv
c=json.load(open(run/'manifest.json'))['configuration'];r=Path(c['robot']);s=DexterousDoorEnv(c['door'],r,json.load(open(r.with_suffix('.audit.json'))));m,d=s.m,s.d
with np.load(run/'trajectory.npz') as a:base=a['terminal_qpos'].copy();initial_velocity=a['terminal_qvel'].copy();initial_time=float(a['terminal_time_s'])
src=json.load(open(args.measured_release));fn=src['finger_joint_names'];fqa=np.array([m.jnt_qposadr[m.joint('robot/'+n).id] for n in fn])
if hashlib.file_digest(r.open('rb'),'sha256').hexdigest()!=src['robot_sha256']:
 raise ValueError('Measured source requires the exact declared robot XML')
if hashlib.file_digest(args.release_trajectory.open('rb'),'sha256').hexdigest()!=src['source_files']['trajectory.npz']:
 raise ValueError('Measured source trajectory changed')
with np.load(args.release_trajectory) as arrays:source_trajectory=arrays['qpos'].copy()
if source_trajectory.ndim!=2 or source_trajectory.shape[1]!=m.nq:raise ValueError('Measured source joint configuration contract differs')
frames=np.asarray(src['source_frames'])
if frames.ndim!=1 or not np.issubdtype(frames.dtype,np.integer) or np.any(frames<0) or np.any(frames>=len(source_trajectory)):
 raise ValueError('Require valid recorded source frame indices')
samples={key:src[key] for key in ('finger_joint_names','finger_joint_delta_rad','palm_position_handle','time_s','palm_rotation_handle')}
samples['source_finger_joint_positions']=source_trajectory[frames][:,fqa]
import importlib.util
planner_path=root/'doorbench/dexterous/whole_body_ungrip_planner.py'
spec=importlib.util.spec_from_file_location('doorbench_numeric_ungrip_planner',planner_path)
planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);trials=[]
for result in planner.iter_whole_body_ungrip(m,qpos=base,qvel=initial_velocity,time_s=initial_time,
        release_samples=samples,finger_lead_seconds=args.finger_lead_seconds,
        early_lift_m=args.early_lift_m,withdrawal_profile=args.withdrawal_profile,maximum_torso_tilt_deg=args.maximum_torso_tilt_deg,retarget_attained_grasp=args.retarget_attained_grasp,right_wrist_margin_rad=args.right_wrist_margin_rad,right_orientation_weight=args.right_orientation_weight):
 trials.append(result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
 (out/'report.json').write_text(json.dumps(dict(scope='Unstepped whole-body adjustment to canonical hand/lever grasp then exact measured release; no dynamics qualification',initial_time_s=initial_time,configuration={k:(str(v.resolve()) if isinstance(v,Path) else v) for k,v in vars(args).items()},robot_xml_sha256=hashlib.file_digest(r.open('rb'),'sha256').hexdigest(),source_trajectory_sha256=hashlib.file_digest((run/'trajectory.npz').open('rb'),'sha256').hexdigest(),planner_source_sha256=hashlib.file_digest(planner_path.open('rb'),'sha256').hexdigest(),environment_source_sha256=hashlib.file_digest(Path(sys.modules[DexterousDoorEnv.__module__].__file__).open('rb'),'sha256').hexdigest(),trials=trials),indent=2))
s.close()

"""Detached named-joint/FK comparison of two physically recorded acquisitions."""
import argparse,gzip,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from doorbench.dexterous.environment import DexterousDoorEnv

p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--isaac',type=Path,required=True);p.add_argument('--robot',type=Path,required=True);p.add_argument('--door',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def no_steps(*args,**kwargs):raise RuntimeError('Saved-state geometry audit must not step physics')
mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=no_steps
sim=DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()))
m=mujoco.MjModel.from_xml_path(str(a.robot));d=mujoco.MjData(m)
trajectory=np.load(a.native/'trajectory.npz');rows=json.load(gzip.open(a.isaac/'balance-steps.json.gz','rt'))
native_reset=json.loads((a.native/'reset.json').read_text());isaac_reset=json.loads((a.isaac/'balance-acquisition-reset.json').read_text())
motors=json.loads((a.isaac/'motor-contract.json').read_text());names=motors['joint_names']
qa=np.array([m.joint(n).qposadr[0] for n in names]);native_qa=np.array([sim.m.joint('robot/'+n).qposadr[0] for n in names]);native_root=sim.root_qadr
assert trajectory['qpos'].shape==(9501,sim.m.nq) and len(rows)==9500
assert sha(a.robot)==motors['source_xml_sha256']
assert sha(a.robot)==json.loads((a.native/'provenance.json').read_text())['robot_xml_sha256']
assert sha(a.door/'door.xml')==json.loads((a.native/'provenance.json').read_text())['door_xml_sha256']
assert np.array_equal(trajectory['qpos'][0,native_qa],[native_reset['joints'][n] for n in names])
assert np.max(abs(trajectory['qpos'][0,native_qa]-[isaac_reset['joint_position'][n] for n in names]))<1e-6
with gzip.open(a.native/'physics.jsonl.gz','rt') as f:
 for line in f:pass
last_native=json.loads(line)['pad_grasp'];anchors={}
for digit in ('ff','mf','rf','lf','th'):
 patches=[c for c in last_native['contacts'] if c['digit']==digit and c['normal_force_N']>1e-6 and c['pad_qualified']]
 assert patches and all(c['body']=='robot/rh_'+digit+'distal' for c in patches)
 anchors[digit]=np.average([c['body_position_m'] for c in patches],axis=0,weights=[c['normal_force_N'] for c in patches])
def fk(root,q):
 d.qpos[:7]=root[:7];d.qpos[qa]=q;mujoco.mj_kinematics(m,d)
 return {m.body(i).name:(d.xpos[i].copy(),d.xmat[i].reshape(3,3).copy()) for i in range(m.nbody) if m.body(i).name.startswith('rh_')}
def points(poses):return {digit:poses['rh_'+digit+'distal'][0]+poses['rh_'+digit+'distal'][1]@anchor for digit,anchor in anchors.items()}
def gap(point,lever):
 rel=point-np.array(lever['center']);axis=np.array(lever['axis']);axial=float(rel@axis)
 return dict(radial_gap_m=float(np.linalg.norm(rel-axial*axis)-lever['radius']),axial_end_clearance_m=float(lever['half_length']-abs(axial)))
samples=[];fk_error_m=0.;fk_error_angle=0.
for t in (0.,.002,1.,8.,11.,15.,15.246,16.,17.,19.):
 i=round(t/.002);native_q=trajectory['qpos'][i,native_qa];native_pose=trajectory['qpos'][i,native_root:native_root+7]
 measured=isaac_reset if i==0 else rows[i-1];isaac_q=np.array([measured['joint_position' if i==0 else 'actual_joint_position'][n] for n in names]);isaac_pose=measured['root13_actororigin']
 native_fk=fk(native_pose,native_q);isaac_fk=fk(isaac_pose,isaac_q);npad=points(native_fk);ipad=points(isaac_fk)
 split={};counterfactual=isaac_q.copy()
 for digit in ('FF','MF','RF','LF'):
  j1,j2=[names.index('rh_'+digit+j) for j in ('J1','J2')];n=native_q[[j1,j2]];s=isaac_q[[j1,j2]]
  split[digit]=dict(native_J1_J2_rad=n.tolist(),isaac_J1_J2_rad=s.tolist(),sum_difference_rad=float(s.sum()-n.sum()),split_difference_rad=float(s[0]-s[1]-n[0]+n[1]))
  total=s.sum();difference=n[0]-n[1];counterfactual[j1]=(total+difference)/2;counterfactual[j2]=(total-difference)/2
 altered=points(fk(isaac_pose,counterfactual));entry=dict(time_s=t,pairs=split,material_pad_world_displacement_m={k:float(np.linalg.norm(ipad[k]-npad[k])) for k in ipad})
 if i:
  raw=rows[i-1]['pad_evidence']
  for path,value in raw['body_transforms_xyzw'].items():
   xyz,rot=isaac_fk[path.rsplit('/',1)[-1]];v=np.array(value)
   fk_error_m=max(fk_error_m,float(np.linalg.norm(xyz-v[:3])))
   fk_error_angle=max(fk_error_angle,float(Rotation.from_matrix(rot.T@Rotation.from_quat(v[3:]).as_matrix()).magnitude()))
  entry['actual_isaac_anchor_gaps']={k:gap(v,raw['lever']) for k,v in ipad.items()}
  entry['detached_native_split_only_same_motor_sums_anchor_gaps']={k:gap(v,raw['lever']) for k,v in altered.items()}
  entry['counterfactual_is_physical_execution']=False
 samples.append(entry)
result=dict(schema='doorbench.recorded-acquisition-pose-comparison.v1',scope='Read-only actual named-joint and robot-only FK comparison; split substitution is a detached geometric diagnostic, never an executed grasp',source_sha256={str(p):sha(p) for p in [a.native/'trajectory.npz',a.native/'reset.json',a.native/'physics.jsonl.gz',a.isaac/'balance-steps.json.gz',a.isaac/'balance-acquisition-reset.json',a.robot,a.door/'door.xml']},native_model_resolved_root_qpos_address=native_root,native_named_reset_mapping_exact=True,maximum_isaac_fk_vs_recorded_body_position_error_m=fk_error_m,maximum_isaac_fk_vs_recorded_body_rotation_error_rad=fk_error_angle,material_anchors_native_verified_final_patch_force_weighted_local_m={k:v.tolist() for k,v in anchors.items()},samples=samples,physics_steps=0,plant_parameters_modified=False,limitation='Ten sampled FK frames; radial distances are fixed native material-point probes, not minimum surface distances or a contact simulation. Geometry cannot prove contact-force causality.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('samples','source_sha256','material_anchors_native_verified_final_patch_force_weighted_local_m')}));print(json.dumps(samples[-1]))

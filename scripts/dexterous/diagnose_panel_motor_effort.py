#!/usr/bin/env python3
"""Reconstruct frozen arm effort terms; actual forces stay from the archive.

The separate unstepped robot model computes only state-dependent mass, gravity,
and Jacobians. It never substitutes freshly solved contact forces for the
recorded actual interval loads.
"""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.run.resolve();source=run.with_name(run.name+'-source');sys.path.insert(0,str(source))
 from doorbench.dexterous.environment import DexterousDoorEnv
 cfg=json.load(open(run/'manifest.json'))['configuration'];robot=Path(cfg['robot']);sim=DexterousDoorEnv(cfg['door'],robot,json.load(open(robot.with_suffix('.audit.json'))));m,d=sim.m,sim.d
 spec=mujoco.MjSpec.from_file(str(robot));spec.worldbody.add_body(name='analytic_lever').add_geom(name='analytic_lever_capsule',type=mujoco.mjtGeom.mjGEOM_CAPSULE,size=[.007,.053,0],quat=[2**-.5,0,2**-.5,0],contype=0,conaffinity=0);am=spec.compile();ad=mujoco.MjData(am)
 with gzip.open(run/'panel-phase.jsonl.gz','rt') as f:
  for line in f:row=json.loads(line)
 chunk=json.load(open(run/'raw-transitions/manifest.json'))['chunks'][-1];file=run/'raw-transitions'/chunk['file']
 if hashlib.file_digest(file.open('rb'),'sha256').hexdigest()!=chunk['sha256']:raise ValueError('Changed actual source')
 with np.load(file) as raw:
  q=raw['qpos_before'][-1].copy();v=raw['qvel_before'][-1].copy();forces=raw['actuator_force'][-1].copy();t=float(raw['interval_start_s'][-1]);b,e=raw['body_offsets'][-2:];ids=raw['body_ids'][b:e];expected_p=raw['body_positions_world_m'][b:e];expected_r=raw['body_rotations_world'][b:e]
 if abs(t-row['episode_time_s'])>1e-9:raise ValueError('Effort/target clock mismatch')
 d.qpos[:]=q;mujoco.mj_kinematics(m,d)
 if max(float(np.max(abs(d.xpos[ids]-expected_p))),float(np.max(abs(d.xmat[ids].reshape(-1,3,3)-expected_r))))>1e-9:raise ValueError('Actual FK/frame mismatch')
 rq=m.jnt_qposadr[m.joint('robot/free_base').id];rv=m.jnt_dofadr[m.joint('robot/free_base').id];ad.qpos[:7]=q[rq:rq+7];ad.qvel[:6]=v[rv:rv+6]
 for j in range(am.njnt):
  if am.jnt_type[j] in (2,3):
   k=m.joint('robot/'+am.joint(j).name).id;ad.qpos[am.jnt_qposadr[j]]=q[m.jnt_qposadr[k]];ad.qvel[am.jnt_dofadr[j]]=v[m.jnt_dofadr[k]]
 mujoco.mj_forward(am,ad)
 names=['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1'];motors=json.load(open(run/'motors-input.json'))['actuators'];indices=[next(i for i,x in enumerate(motors) if set(x['terms'])=={n}) for n in names];ji=np.array([am.joint(n).id for n in names]);qa=am.jnt_qposadr[ji];va=am.jnt_dofadr[ji];kp=np.array([motors[i]['kp'] for i in indices]);bias=np.array([motors[i]['bias'] for i in indices]);target=np.asarray(row['left_arm_targets']);targetv=np.asarray(row['left_arm_target_velocity']);damping=2*np.array([.8 if 'WRJ' in n else 10 for n in names]);palm=am.site('lh_palm_touch').id;jac=np.zeros((3,am.nv));mujoco.mj_jacSite(am,ad,jac,None,palm);normal=d.xmat[m.body('leaf').id].reshape(3,3)[:,1];jn=normal@jac[:,va]
 terms=dict(position_servo=kp*target+bias[:,0]+bias[:,1]*ad.qpos[qa]+kp*9*(target-ad.qpos[qa]),velocity_servo=bias[:,2]*ad.qvel[va]+damping*(targetv-ad.qvel[va]),gravity=ad.qfrc_bias[va],declared_normal_feedforward=jn*row['normal_feedforward_N']);assembled=sum(terms.values());actual=forces[[m.actuator('robot/'+motors[i]['name']).id for i in indices]];caps=np.array([motors[i]['force_range'] for i in indices]);error=float(np.max(abs(np.clip(assembled,caps[:,0],caps[:,1])-actual)))
 mass=np.zeros((am.nv,am.nv));mujoco.mj_fullM(am,ad,mass);inverse=np.linalg.solve(mass[np.ix_(va,va)],jn);denominator=float(jn@inverse)
 # These are equivalent free-chain effort coordinates, not measured loads.
 equivalent={name:float(inverse@value/denominator) for name,value in terms.items() if name!='gravity'}
 result=dict(scope='Actual archived capped forces reconstructed from frozen controller terms. Equivalent free-chain normal efforts are analytic coordinates, not measured contact loads.',episode_time_s=t,joint_names=names,terms={k:v.tolist() for k,v in terms.items()},actual_capped_forces=actual.tolist(),reconstructed_capped_forces=np.clip(assembled,caps[:,0],caps[:,1]).tolist(),maximum_actual_force_reconstruction_error=error,equivalent_normal_efforts_N=equivalent,normal_jacobian=jn.tolist(),source_chunk_sha256=chunk['sha256'])
 if error>1e-9:raise ValueError('Frozen effort reconstruction differs from actual consumed force: '+str(error))
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(a.output/'diagnostic-source.py').write_bytes(Path(__file__).read_bytes());sim.close();print(json.dumps(result,indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Sum actual archived contact wrenches about the physical leaf hinge axis.

The kinematic model supplies only the actual hinge frame; contacts and their
wrenches are read verbatim from the preceding physical interval. This does not
recompute solver forces or infer a hidden friction-constraint multiplier.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,mujoco


def contact_moment_about_axis(anchor,axis,point,frame,wrench,side):
    anchor=np.asarray(anchor,float);axis=np.asarray(axis,float);point=np.asarray(point,float);frame=np.asarray(frame,float);wrench=np.asarray(wrench,float)
    if (anchor.shape!=(3,) or axis.shape!=(3,) or point.shape!=(3,) or frame.shape!=(3,3) or wrench.shape!=(6,)
            or type(side) is not int or side not in (0,1) or not np.isfinite(np.r_[anchor,axis,point,frame.ravel(),wrench]).all()
            or not np.isclose(np.linalg.norm(axis),1.,atol=1e-8)
            or not np.allclose(frame@frame.T,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(frame),1.,atol=1e-8)):
        raise ValueError('Require finite actual unit hinge/contact frames and an explicit contact side')
    sign=1 if side==1 else -1
    force=sign*(frame.T@wrench[:3]);torque=sign*(frame.T@wrench[3:]);normal_force=sign*frame[0]*wrench[0]
    return float(axis@(np.cross(point-anchor,force)+torque)),float(axis@np.cross(point-anchor,normal_force)),force


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--time',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 run=a.run.resolve();sys.path.insert(0,str(run.with_name(run.name+'-source')))
 from doorbench.dexterous.environment import DexterousDoorEnv
 cfg=json.loads((run/'manifest.json').read_text())['configuration'];robot=Path(cfg['robot']);sim=DexterousDoorEnv(cfg['door'],robot,json.loads(robot.with_suffix('.audit.json').read_text()));m,d=sim.m,sim.d
 manifest=json.loads((run/'raw-transitions/manifest.json').read_text())
 if not manifest['complete']:raise ValueError('Require a complete actual dynamics archive')
 c=next(c for c in manifest['chunks'] if c['interval_start_s']-1e-8<=a.time<c['interval_end_s']-1e-8);file=run/'raw-transitions'/c['file']
 if hashlib.file_digest(file.open('rb'),'sha256').hexdigest()!=c['sha256']:raise ValueError('Changed archived dynamics')
 with np.load(file,allow_pickle=False) as raw:
  i=int(np.argmin(abs(raw['interval_start_s']-a.time)));d.qpos[:]=raw['qpos_before'][i];d.qvel[:]=raw['qvel_before'][i];after=raw['qvel_after'][i];times=[float(raw['interval_start_s'][i]),float(raw['interval_end_s'][i])];begin,end=raw['contact_offsets'][i:i+2]
  body_start,body_end=raw['body_offsets'][i:i+2];body_ids=raw['body_ids'][body_start:body_end];actual_body_positions=raw['body_positions_world_m'][body_start:body_end];actual_body_rotations=raw['body_rotations_world'][body_start:body_end]
  bodies=raw['contact_body'][begin:end];points=raw['contact_position_world_m'][begin:end];frames=raw['contact_frame_world'][begin:end];wrenches=raw['contact_wrench_contact_frame'][begin:end]
 mujoco.mj_kinematics(m,d)
 actual_fk_error=max(float(np.max(abs(d.xpos[body_ids]-actual_body_positions))),float(np.max(abs(d.xmat[body_ids].reshape(-1,3,3)-actual_body_rotations))))
 if actual_fk_error>1e-9:raise ValueError('Current geometry differs from the recorded physical body frames')
 j=m.joint('leaf_hinge').id;va=m.jnt_dofadr[j];leaf=m.body('leaf').id;anchor=d.xanchor[j].copy();axis=d.xaxis[j].copy();leaf_rotation=d.xmat[leaf].reshape(3,3)
 def descends(body):
  while body:
   if body==leaf:return True
   body=m.body_parentid[body]
  return False
 moment={};normal_moment={};rows=[]
 for pair,point,frame,wrench in zip(bodies,points,frames,wrenches):
  membership=[descends(int(b)) for b in pair]
  if sum(membership)!=1:continue
  side=membership.index(True);other=m.body(int(pair[1-side])).name;sign=1 if side==1 else -1
  value,normal_value,force=contact_moment_about_axis(anchor,axis,point,frame,wrench,side);moment[other]=moment.get(other,0.)+value;normal_moment[other]=normal_moment.get(other,0.)+normal_value
  if np.linalg.norm(wrench)>1e-8:rows.append(dict(other_body=other,moment_about_hinge_Nm=value,normal_force_moment_Nm=normal_value,tangential_force_moment_Nm=float(value-normal_value-axis@(sign*(frame.T@wrench[3:]))),contact_couple_moment_Nm=float(axis@(sign*(frame.T@wrench[3:]))),position_leaf_m=(leaf_rotation.T@(point-d.xpos[leaf])).tolist(),force_on_leaf_world_N=force.tolist(),force_on_leaf_local_N=(leaf_rotation.T@force).tolist()))
 result=dict(scope='Actual interval contact moment only; listed frictionloss is the original model limit, not an inferred solver multiplier.',interval_s=times,source_sha256=c['sha256'],maximum_actual_fk_error=actual_fk_error,hinge_anchor_world_m=anchor.tolist(),hinge_axis_world=axis.tolist(),leaf_angle_rad=float(d.qpos[m.jnt_qposadr[j]]),leaf_velocity_before_rad_s=float(d.qvel[va]),leaf_velocity_after_rad_s=float(after[va]),original_hinge_frictionloss_Nm=float(m.dof_frictionloss[va]),original_hinge_viscous_damping_Nm_s=float(m.dof_damping[va]),contact_moment_by_other_body_Nm=moment,normal_contact_moment_by_other_body_Nm=normal_moment,total_contact_moment_Nm=sum(moment.values()),contacts=rows)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));sim.close()


if __name__=='__main__':main()

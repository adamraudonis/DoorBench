#!/usr/bin/env python3
"""Fit feet, both hands and upright body to a source-bound release/door arc.

Geometry only. The measured failed rollout supplies a test aperture trajectory,
never a qualified restart state. The initial robot state comes from the passed
transfer. No physical support, tracking or release success is inferred.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.json_record_stream import iter_json_object_array
from doorbench.dexterous.operation_teacher import smooth_phase
from doorbench.dexterous.release_source_admission import admit_release_source


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--right-hand-route',type=Path,required=True)
    p.add_argument('--measured-arc-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=16.)
    p.add_argument('--nodes',type=int,default=161)
    p.add_argument('--torso-yaw-preference-rad',type=float,default=.15)
    p.add_argument('--follow-handle-through-route-seconds',type=float,help='Explicit handle-relative RH phase, followed by smooth world retreat; independent4mm separation check required')
    a=p.parse_args()
    if a.nodes<101 or not np.isfinite(a.duration) or a.duration<16.:raise ValueError('Dense slow release path required')
    if not np.isfinite(a.torso_yaw_preference_rad) or abs(a.torso_yaw_preference_rad)>.3:raise ValueError('Bounded upright torso-yaw preference required')
    if a.follow_handle_through_route_seconds is not None and not 3<=a.follow_handle_through_route_seconds<=5:raise ValueError('Explicit separation epoch must be3..5 route seconds')
    source=a.source_run.resolve();arc=a.measured_arc_run.resolve();right=a.right_hand_route.resolve()
    admission=admit_release_source(source,profile='volar-phalange-v1',contact_audit_name='independent-contact-audit.json',measured_rest=True)
    ref=json.loads(right.read_text())
    if ref['source_trajectory_sha256']!=sha(source/'trajectory.npz'):raise ValueError('RH route must begin at the same actual source')
    cfg=json.loads((source/'manifest.json').read_text())['configuration']
    arc_manifest=json.loads((arc/'manifest.json').read_text())
    if arc_manifest['inputs']!=json.loads((source/'manifest.json').read_text())['inputs']:raise ValueError('Measured aperture trajectory must use original source assets')
    if not json.loads((arc/'raw-transitions/manifest.json').read_text())['complete']:raise ValueError('Complete measured aperture record required')
    robot=Path(cfg['robot']);door=Path(cfg['door'])
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));m,d=sim.m,sim.d
    with np.load(source/'trajectory.npz') as z:base=z['terminal_qpos'].copy();epoch=float(z['terminal_time_s'])
    d.qpos[:]=base;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    rq=m.joint('robot/free_base').qposadr[0];leaf=m.body('leaf').id;lq=m.joint('leaf_hinge').qposadr[0]
    robotbody=m.jnt_bodyid[m.joint('robot/free_base').id];initial_com=d.subtree_com[robotbody].copy()
    rp=base[rq:rq+3].copy();rr=Rotation.from_quat(base[rq+3:rq+7][[1,2,3,0]])
    rh=m.site('robot/rh_palm_touch').id;lh=m.site('robot/lh_palm_touch').id
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    fp=d.xpos[feet].copy();fr=d.xmat[feet].reshape(2,3,3).copy()
    leafr=d.xmat[leaf].reshape(3,3).copy();leafp=d.xpos[leaf].copy()
    localp=leafr.T@(d.site_xpos[lh]-leafp);localr=leafr.T@d.site_xmat[lh].reshape(3,3)
    initial=dict(time_s=0.,qpos=base.tolist(),palm_position=d.site_xpos[rh].tolist(),palm_rotation=d.site_xmat[rh].reshape(3,3).tolist())
    refs=[initial]+[{**r,'time_s':r['time_s']+.5} for r in ref['trials'][0]['rows']]
    times=np.array([r['time_s'] for r in refs]);qs=np.array([r['qpos'] for r in refs])
    # Consume the actual independently screened FK palm, not its optimizer goal.
    positions=np.array([r['palm_position'] for r in refs]);rotations=Slerp(times,Rotation.from_matrix([r['palm_rotation'] for r in refs]))
    names=['torso']+[side+'_'+n for side in ('left','right') for n in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]+[side+'_'+n for side in ('right','left') for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+[side+'_'+n for side in ('rh','lh') for n in ('WRJ2','WRJ1')]
    joints=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[joints]
    lower=np.r_[[-.03,-.03,-.03],[-.04,-.04,-.12],m.jnt_range[joints,0]+.01]
    upper=np.r_[[.03,.03,.012],[.04,.04,.12],m.jnt_range[joints,1]-.01]
    initial_coordinate=np.r_[np.zeros(6),base[qa]]
    lower=np.minimum(lower,initial_coordinate);upper=np.maximum(upper,initial_coordinate)
    arc_times=[];arc_angles=[];arc_operators=[];arc_latches=[]
    with gzip.open(arc/'physics-steps.json.gz','rt') as stream:
        for r in iter_json_object_array(stream):
            if r['sim_time_s']>=epoch-1e-8:
                arc_times.append(r['sim_time_s']);arc_angles.append(r['door_q']);arc_operators.append(r['handle_angle_rad']);arc_latches.append(r['bolt_slide_m'])
    if abs(arc_times[0]-epoch)>1e-8 or arc_times[-1]<epoch+a.duration-1e-8:raise ValueError('Complete same-epoch measured aperture history required')
    a.output.mkdir(parents=True,exist_ok=False);frozen=a.output/'planner-source.py';shutil.copy2(__file__,frozen)
    previous=initial_coordinate.copy();rows=[]
    for node,elapsed in enumerate(np.linspace(0,a.duration,a.nodes)):
        clock=float(smooth_phase(elapsed/a.duration))*times[-1]
        i=min(len(times)-2,max(0,int(np.searchsorted(times,clock,side='right')-1)));f=(clock-times[i])/(times[i+1]-times[i])
        reference=(1-f)*qs[i]+f*qs[i+1]
        target_rhp=(1-f)*positions[i]+f*positions[i+1];target_rhr=rotations(clock).as_matrix()
        angle=float(np.interp(epoch+elapsed,arc_times,arc_angles))
        operator=float(np.interp(epoch+elapsed,arc_times,arc_operators));latch=float(np.interp(epoch+elapsed,arc_times,arc_latches))
        oq=m.joint('leaf_handle_hinge').qposadr[0];bq=m.joint('leaf_latch_bolt_slide').qposadr[0];handle=m.body('leaf_handle').id
        d.qpos[:]=reference;mujoco.mj_kinematics(m,d);reference_hp=d.xpos[handle].copy();reference_hr=d.xmat[handle].reshape(3,3).copy()
        d.qpos[lq]=angle;d.qpos[oq]=operator;d.qpos[bq]=latch;mujoco.mj_kinematics(m,d)
        follow=0.
        if a.follow_handle_through_route_seconds is not None:
            through=a.follow_handle_through_route_seconds;route_clock=max(0.,clock-.5)
            follow=1.-float(smooth_phase((route_clock-through)/(times[-1]-.5-through)))
            transformed_p=d.xpos[handle]+d.xmat[handle].reshape(3,3)@reference_hr.T@(target_rhp-reference_hp)
            transformed_r=d.xmat[handle].reshape(3,3)@reference_hr.T@target_rhr
            target_rhp=target_rhp+follow*(transformed_p-target_rhp)
            target_rhr=Rotation.from_rotvec(follow*Rotation.from_matrix(transformed_r@target_rhr.T).as_rotvec()).as_matrix()@target_rhr
        target_lhp=d.xpos[leaf]+d.xmat[leaf].reshape(3,3)@localp
        target_lhr=d.xmat[leaf].reshape(3,3)@localr
        preference=np.r_[np.zeros(6),reference[qa]]
        preference[6+names.index('torso')]=base[qa[names.index('torso')]]+a.torso_yaw_preference_rad*float(smooth_phase(elapsed/a.duration))
        def install(x):
            d.qpos[:]=reference;d.qpos[lq]=angle;d.qpos[oq]=operator;d.qpos[bq]=latch;d.qpos[rq:rq+3]=rp+x[:3]
            quat=(Rotation.from_rotvec(x[3:6])*rr).as_quat();d.qpos[rq+3:rq+7]=quat[[3,0,1,2]]
            d.qpos[qa]=x[6:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        def residual(x):
            install(x)
            hands=np.r_[100*(d.site_xpos[lh]-target_lhp),10*Rotation.from_matrix(target_lhr@d.site_xmat[lh].reshape(3,3).T).as_rotvec(),100*(d.site_xpos[rh]-target_rhp),10*Rotation.from_matrix(target_rhr@d.site_xmat[rh].reshape(3,3).T).as_rotvec()]
            foot=np.concatenate([np.r_[100*(d.xpos[b]-fp[k]),10*Rotation.from_matrix(fr[k]@d.xmat[b].reshape(3,3).T).as_rotvec()] for k,b in enumerate(feet)])
            torso=m.body('robot/torso_link').id;tilt=np.arccos(np.clip(d.xmat[torso].reshape(3,3)[2,2],-1,1))
            bounds=50*np.maximum([np.linalg.norm(x[:3])-.029,np.linalg.norm(d.subtree_com[robotbody,:2]-initial_com[:2])-.014,tilt-np.radians(3.9)],0.)
            return np.r_[hands,foot,.008*(x-preference),.006*(x-previous),bounds]
        if node==0:
            fit_x=initial_coordinate.copy();install(fit_x)
        else:
            # Reserve interpolation headroom below the original vector speed
            # limits; prevent a redundant IK branch from abruptly moving root.
            dt=elapsed-rows[-1]['elapsed_s']
            step=np.r_[np.full(3,.008),np.full(3,.012),np.full(len(names),1.)]*dt
            node_lower=np.maximum(lower,previous-step);node_upper=np.minimum(upper,previous+step)
            fit=least_squares(residual,np.clip(previous,node_lower+1e-12,node_upper-1e-12),bounds=(node_lower,node_upper),max_nfev=220,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            fit_x=fit.x;install(fit_x)
        previous=fit_x.copy()
        rows.append(dict(elapsed_s=float(elapsed),episode_time_s=epoch+float(elapsed),release_clock_s=clock,leaf_rad=angle,operator_rad=operator,latch_m=latch,right_hand_follow_weight=follow,qpos=(base.copy() if node==0 else d.qpos.copy()).tolist(),coordinates=fit_x.tolist(),right_palm_position=target_rhp.tolist(),right_palm_rotation=target_rhr.tolist(),left_palm_position=target_lhp.tolist(),left_palm_rotation=target_lhr.tolist(),fit_residual_norm=float(np.linalg.norm(residual(fit_x)[:24]))))
        if node%20==0:print(json.dumps(dict(node=node,elapsed_s=float(elapsed),fit_residual=rows[-1]['fit_residual_norm'])),flush=True)
    inputs=[source/'manifest.json',source/'trajectory.npz',arc/'manifest.json',arc/'physics-steps.json.gz',arc/'raw-transitions/manifest.json',right,robot,door/'door.xml',frozen]
    result=dict(schema='doorbench.coupled-release-support-screen.v1',scope=__doc__,configuration={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()},source_admission=admission,initial_qpos=base.tolist(),initial_episode_time_s=epoch,duration_s=a.duration,robot_path=str(robot.resolve()),door_path=str(door.resolve()),joint_names=names,root_qpos_address=int(rq),left_palm_in_leaf_position=localp.tolist(),left_palm_in_leaf_rotation=localr.tolist(),initial_feet_positions=fp.tolist(),initial_feet_rotations=fr.tolist(),initial_com=initial_com.tolist(),rows=rows,input_sha256={str(v.resolve()):sha(v) for v in inputs},physics_steps=0,physical_admission=False)
    (a.output/'candidate.json').write_text(json.dumps(result,indent=2)+'\n');sim.close()
    print(json.dumps(dict(candidate=str(a.output/'candidate.json'),nodes=len(rows),physics_steps=0)),flush=True)


if __name__=='__main__':main()

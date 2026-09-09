#!/usr/bin/env python3
"""Separate attained fingertip pads radially, then lift the free hand.

Unstepped geometry only. No canonical regrasp, plant actuation or task success
is inferred. The independent dense audit and full motor rollout remain required.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.operation_teacher import smooth_phase


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--radial-clearance-m',type=float,default=.018)
    parser.add_argument('--finger-profile',choices=('radial','extend'),default='radial')
    parser.add_argument('--retreat-profile',choices=('lift','slide-lift'),default='lift')
    parser.add_argument('--whole-body',action='store_true',help='Permit bounded upright root/limb adjustment with feet and left palm fixed')
    parser.add_argument('--coordinated-release',action='store_true',help='Begin the free-end slide during pad separation')
    parser.add_argument('--thumb-j3-margin-rad',type=float,default=.001)
    parser.add_argument('--early-palm-clearance-m',type=float,default=0.,help='Move palm away from leaf during finger separation')
    parser.add_argument('--early-palm-direction',choices=('away','up'),default='away')
    a=parser.parse_args()
    if not np.isfinite(a.early_palm_clearance_m) or not 0<=a.early_palm_clearance_m<=.03:
        raise ValueError('Early palm clearance must be within 0–30 mm')
    if not np.isfinite(a.radial_clearance_m) or not .01<=a.radial_clearance_m<=.03:
        raise ValueError('Explicit radial separation must be 10–30 mm')
    if not np.isfinite(a.thumb_j3_margin_rad) or not .001<=a.thumb_j3_margin_rad<=.08:
        raise ValueError('Explicit thumb J3 planning margin must be 1–80 mrad')
    if a.coordinated_release and a.retreat_profile!='slide-lift':
        raise ValueError('Coordinated release requires the free-end slide')
    source=a.source_run
    evidence=[source/n for n in ('report.json','independent-pad-audit.json','independent-whole-handle-audit.json')]
    if not all(json.loads(p.read_text()).get('passed') is True for p in evidence):
        raise ValueError('Fully qualified physical resting source required')
    manifest=json.loads((source/'manifest.json').read_text());cfg=manifest['configuration']
    robot=Path(cfg['robot']);door=Path(cfg['door'])
    if sha(robot)!=manifest['inputs']['robot']['sha256'] or sha(door/'door.xml')!=manifest['inputs']['door']['door.xml']:
        raise ValueError('Exact recorded model required')
    with np.load(source/'trajectory.npz') as z:
        base=z['terminal_qpos'].copy();start=float(z['terminal_time_s'])
    a.output.mkdir(parents=True,exist_ok=False)
    frozen=a.output/'direct-release-planner-source.py';shutil.copy2(__file__,frozen)
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d;d.qpos[:]=base;mujoco.mj_kinematics(m,d)
    palm=m.site('robot/rh_palm_touch').id;lever=m.geom('leaf_handle_lever_col_n').id
    initial_p=d.site_xpos[palm].copy();initial_R=d.site_xmat[palm].reshape(3,3).copy()
    axis=d.geom_xmat[lever].reshape(3,3)[:,2].copy();center=d.geom_xpos[lever].copy()
    hub=m.geom('leaf_handle_hub_col_n').id
    free_end_direction=-np.sign((d.geom_xpos[hub]-center)@axis)*axis
    outward=-d.xmat[m.body('leaf').id].reshape(3,3)[:,1].copy()
    arm_names=['right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['rh_WRJ2','rh_WRJ1']
    body_names=[side+'_'+n for side in ('left','right') for n in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]+['torso']+arm_names+['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1']
    arms=np.array([m.joint('robot/'+n).id for n in arm_names]);armqa=m.jnt_qposadr[arms]
    body_ids=np.array([m.joint('robot/'+n).id for n in body_names]);bodyqa=m.jnt_qposadr[body_ids]
    rootqa=m.joint('robot/free_base').qposadr[0];rootP=base[rootqa:rootqa+3].copy()
    rootR=Rotation.from_quat(base[rootqa+3:rootqa+7][[1,2,3,0]])
    left=m.site('robot/lh_palm_touch').id;leftP=d.site_xpos[left].copy();leftR=d.site_xmat[left].reshape(3,3).copy()
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    footP=d.xpos[feet].copy();footR=d.xmat[feet].reshape(2,3,3).copy()
    tilt=np.arccos(np.clip(d.xmat[m.body('robot/torso_link').id].reshape(3,3)[2,2],-1,1))
    rotation_margin=.99*(np.radians(4)-tilt)/np.sqrt(2)
    if rotation_margin<=0:raise ValueError('Source needs upright rotation margin')
    obstacles=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name=='leaf_handle']
    digits=[];finger_names=[]
    for digit in ('FF','MF','RF','LF','TH'):
        names=[m.joint(j).name.removeprefix('robot/') for j in range(m.njnt) if m.joint(j).name.startswith('robot/rh_'+digit+'J')]
        ids=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[ids]
        body=m.body('robot/rh_'+digit.lower()+'distal').id
        shapes=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_'+digit.lower())]
        near=[]
        for g in shapes:
            if m.geom_bodyid[g]!=body:continue
            points=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,g,lever,.2,points)
            near.append((gap,points.copy()))
        gap,points=min(near,key=lambda x:x[0])
        if abs(gap)>.01:raise ValueError('Every attained fingertip must be near the lever')
        local=d.xmat[body].reshape(3,3).T@(points[:3]-d.xpos[body])
        radial=points[:3]-center;radial-=axis*(radial@axis);radial/=np.linalg.norm(radial)
        low=np.minimum(base[qa],m.jnt_range[ids,0]+.001);high=np.maximum(base[qa],m.jnt_range[ids,1]-.001)
        relaxed=base[qa].copy()
        if digit!='TH':
            for j,name in enumerate(names):
                if name.endswith(('J1','J2','J3')):relaxed[j]=.1
        d.qpos[qa]=relaxed;mujoco.mj_kinematics(m,d)
        relaxed_point=d.xpos[body]+d.xmat[body].reshape(3,3)@local
        d.qpos[:]=base;mujoco.mj_kinematics(m,d)
        digits.append(dict(names=names,qa=qa,body=body,shapes=shapes,local=local,point=points[:3],radial=radial,low=low,high=high,previous=base[qa].copy(),relaxed=relaxed,relaxed_point=relaxed_point))
        finger_names+=names
    rows=[];previous_arm=base[armqa].copy();previous_body=np.r_[np.zeros(6),base[bodyqa]];worst=0.
    for time in np.linspace(0,8,161):
        separation=float(smooth_phase(time/3));lift=float(smooth_phase((time-3)/5))
        slide=0.
        if a.retreat_profile=='slide-lift':
            slide=float(smooth_phase((time-3)/2));lift=float(smooth_phase((time-5)/3))
            if a.coordinated_release:
                slide=float(smooth_phase((time-1)/3));lift=float(smooth_phase((time-4)/4))
        d.qpos[:]=base
        for digit in digits:
            qa=digit['qa'];goal=digit['point']+digit['radial']*a.radial_clearance_m*separation
            extending=a.finger_profile=='extend' and not digit['names'][0].startswith('rh_TH')
            preference=base[qa]
            if extending:
                goal=digit['point']+separation*(digit['relaxed_point']-digit['point'])
                preference=base[qa]+separation*(digit['relaxed']-base[qa])
            def residual(v):
                d.qpos[qa]=v;mujoco.mj_kinematics(m,d)
                point=d.xpos[digit['body']]+d.xmat[digit['body']].reshape(3,3)@digit['local']
                gaps=[min(0.,mujoco.mj_geomDistance(m,d,g,h,.01,None)-.0005*separation) for g in digit['shapes'] for h in obstacles]
                loop=[]
                if digit['names'][0].startswith('rh_TH') is False:
                    i,j=[digit['names'].index('rh_'+digit['names'][0][3:5]+'J'+str(k)) for k in (1,2)]
                    loop=[10*(v[i]-v[j]-(1-separation)*(base[qa[i]]-base[qa[j]]))]
                return np.r_[100*(point-goal),1000*np.asarray(gaps),loop,(.25 if extending else .01)*(v-preference),.01*(v-digit['previous'])]
            if time>0:
                low=digit['low'].copy();high=digit['high'].copy()
                if 'rh_THJ3' in digit['names']:
                    j=digit['names'].index('rh_THJ3');limits=m.jnt_range[m.joint('robot/rh_THJ3').id]
                    low[j]+=separation*(limits[0]+a.thumb_j3_margin_rad-low[j])
                    high[j]+=separation*(limits[1]-a.thumb_j3_margin_rad-high[j])
                fit=least_squares(residual,np.clip(digit['previous'],low+1e-12,high-1e-12),bounds=(low,high),max_nfev=100)
                digit['previous']=fit.x.copy();worst=max(worst,float(np.linalg.norm(fit.fun[:3]))/100)
            d.qpos[qa]=digit['previous']
        fingers=d.qpos.copy()
        early=a.early_palm_clearance_m*separation
        early_direction=outward if a.early_palm_direction=='away' else np.array([0.,0.,1.])
        target=initial_p+.12*slide*free_end_direction+lift*(np.array([0.,0.,.1])+.04*outward)+early*early_direction
        def arm_residual(v):
            d.qpos[:]=fingers;d.qpos[armqa]=v;mujoco.mj_kinematics(m,d)
            return np.r_[100*(d.site_xpos[palm]-target),10*Rotation.from_matrix(initial_R@d.site_xmat[palm].reshape(3,3).T).as_rotvec(),.01*(v-base[armqa])]
        def body_residual(v):
            d.qpos[:]=fingers;d.qpos[rootqa:rootqa+3]=rootP+v[:3]
            quat=(Rotation.from_rotvec(v[3:6])*rootR).as_quat();d.qpos[rootqa+3:rootqa+7]=quat[[3,0,1,2]]
            d.qpos[bodyqa]=v[6:];mujoco.mj_kinematics(m,d)
            hand=np.r_[100*(d.site_xpos[palm]-target),10*Rotation.from_matrix(initial_R@d.site_xmat[palm].reshape(3,3).T).as_rotvec(),100*(d.site_xpos[left]-leftP),10*Rotation.from_matrix(leftR@d.site_xmat[left].reshape(3,3).T).as_rotvec()]
            foot=np.concatenate([np.r_[100*(d.xpos[b]-footP[i]),10*Rotation.from_matrix(footR[i]@d.xmat[b].reshape(3,3).T).as_rotvec()] for i,b in enumerate(feet)])
            return np.r_[hand,foot,.01*(v[6:]-base[bodyqa]),.02*v[:6]]
        if a.whole_body and (lift>0 or slide>0 or early>0):
            lo=np.r_[[-.06,-.06,-.03],[-rotation_margin,-rotation_margin,-.12],np.minimum(base[bodyqa],m.jnt_range[body_ids,0]+.001)]
            hi=np.r_[[.06,.06,.03],[rotation_margin,rotation_margin,.12],np.maximum(base[bodyqa],m.jnt_range[body_ids,1]-.001)]
            fit=least_squares(body_residual,np.clip(previous_body,lo+1e-12,hi-1e-12),bounds=(lo,hi),max_nfev=400,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            previous_body=fit.x.copy();body_residual(previous_body)
        elif lift>0 or slide>0 or early>0:
            lo=np.minimum(base[armqa],m.jnt_range[arms,0]+.001);hi=np.maximum(base[armqa],m.jnt_range[arms,1]-.001)
            fit=least_squares(arm_residual,np.clip(previous_arm,lo+1e-12,hi-1e-12),bounds=(lo,hi),max_nfev=200)
            previous_arm=fit.x.copy()
            arm_residual(previous_arm)
        else:arm_residual(previous_arm)
        q=d.qpos.copy();actual_p=d.site_xpos[palm].copy();actual_R=d.site_xmat[palm].reshape(3,3).copy()
        rows.append(dict(time_s=float(time),phase='grasp_adjustment' if time==0 else 'measured_release' if time<=3 else 'clearance_lift',qpos=q.tolist(),palm_position=actual_p.tolist(),palm_rotation=actual_R.tolist(),requested_palm_position=target.tolist(),joints={n:float(q[m.joint('robot/'+n).qposadr[0]]) for n in body_names},finger_joints={n:float(q[m.joint('robot/'+n).qposadr[0]]) for n in finger_names}))
    report=dict(scope=__doc__,initial_time_s=start,source_trajectory_sha256=sha(source/'trajectory.npz'),robot_xml_sha256=sha(robot),planner_source_path=str(frozen),planner_source_sha256=sha(frozen),configuration=dict(source_run=str(source),radial_clearance_m=a.radial_clearance_m,finger_profile=a.finger_profile,retreat_profile=a.retreat_profile,whole_body=a.whole_body,coordinated_release=a.coordinated_release,thumb_j3_margin_rad=a.thumb_j3_margin_rad,early_palm_clearance_m=a.early_palm_clearance_m,early_palm_direction=a.early_palm_direction),source_evidence_sha256={str(p):sha(p) for p in evidence},maximum_pad_goal_residual_m=worst,trials=[dict(rows=rows)])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(nodes=len(rows),maximum_pad_goal_residual_m=worst,physics_steps=0)),flush=True)
    sim.close()


if __name__=='__main__':main()

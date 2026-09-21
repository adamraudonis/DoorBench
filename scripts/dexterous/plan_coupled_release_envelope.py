#!/usr/bin/env python3
"""Prospective unstepped progress x measured-aperture release/support map."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.operation_teacher import smooth_phase
from doorbench.dexterous.release_source_admission import admit_release_source


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coupled-candidate',type=Path,required=True)
    p.add_argument('--coupled-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--time-nodes',type=int,default=81)
    a=p.parse_args()
    if a.time_nodes<81:raise ValueError('At least81 progress nodes required')
    c=json.loads(a.coupled_candidate.read_text());audit=json.loads(a.coupled_audit.read_text())
    if audit.get('passed') is not True or audit.get('samples')!=2001:raise ValueError('Original full coupled-path qualification required')
    for path,digest in audit['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Coupled source changed: '+path)
    if audit['input_sha256'].get(str(a.coupled_candidate.resolve()))!=sha(a.coupled_candidate):raise ValueError('Audit must bind this candidate')
    admission=admit_release_source(c['configuration']['source_run'],profile='volar-phalange-v1',contact_audit_name='independent-contact-audit.json',measured_rest=True)
    if admission!=c['source_admission']:raise ValueError('Source qualification changed')
    sim=DexterousDoorEnv(Path(c['door_path']),Path(c['robot_path']),json.loads(Path(c['robot_path']).with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d;base=np.array(c['initial_qpos']);rq=c['root_qpos_address'];names=c['joint_names']
    joints=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[joints]
    leaf=m.body('leaf').id;handle=m.body('leaf_handle').id;rh=m.site('robot/rh_palm_touch').id;lh=m.site('robot/lh_palm_touch').id
    lq=m.joint('leaf_hinge').qposadr[0];oq=m.joint('leaf_handle_hinge').qposadr[0];bq=m.joint('leaf_latch_bolt_slide').qposadr[0]
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')];torso=m.body('robot/torso_link').id;robotbody=m.jnt_bodyid[m.joint('robot/free_base').id]
    rr=Rotation.from_quat(base[rq+3:rq+7][[1,2,3,0]]);initial_coordinate=np.r_[np.zeros(6),base[qa]]
    lower=np.minimum(np.r_[[-.03,-.03,-.03],[-.04,-.04,-.12],m.jnt_range[joints,0]+.01],initial_coordinate)
    upper=np.maximum(np.r_[[.03,.03,.012],[.04,.04,.12],m.jnt_range[joints,1]-.01],initial_coordinate)
    fp=np.array(c['initial_feet_positions']);fr=np.array(c['initial_feet_rotations']);localp=np.array(c['left_palm_in_leaf_position']);localr=np.array(c['left_palm_in_leaf_rotation']);com=np.array(c['initial_com'])
    rows=c['rows'];ct=np.array([r['elapsed_s'] for r in rows]);preferred=CubicSpline(ct,[r['coordinates'] for r in rows],bc_type='clamped')
    ref=json.loads(Path(c['configuration']['right_hand_route']).read_text())
    d.qpos[:]=base;mujoco.mj_kinematics(m,d)
    initial=dict(time_s=0.,qpos=base.tolist(),palm_position=d.site_xpos[rh].tolist(),palm_rotation=d.site_xmat[rh].reshape(3,3).tolist())
    refs=[initial]+[{**r,'time_s':r['time_s']+.5} for r in ref['trials'][0]['rows']]
    rt=np.array([r['time_s'] for r in refs]);rqs=np.array([r['qpos'] for r in refs]);positions=np.array([r['palm_position'] for r in refs]);rotations=Slerp(rt,Rotation.from_matrix([r['palm_rotation'] for r in refs]))
    elapsed=np.linspace(0,c['duration_s'],a.time_nodes);angles=np.sort(np.unique(np.r_[np.linspace(.08,.4,17),base[lq]]))
    coordinates=np.zeros((len(elapsed),len(angles),len(initial_coordinate)));residuals=np.zeros((len(elapsed),len(angles)))
    a.output.mkdir(parents=True,exist_ok=False);frozen=a.output/'planner-source.py';shutil.copy2(__file__,frozen)
    through=c['configuration']['follow_handle_through_route_seconds']
    for j,angle in enumerate(angles):
        previous=None
        for i,t in enumerate(elapsed):
            clock=float(smooth_phase(t/c['duration_s']))*rt[-1];k=min(len(rt)-2,max(0,int(np.searchsorted(rt,clock,side='right')-1)));f=(clock-rt[k])/(rt[k+1]-rt[k])
            reference=(1-f)*rqs[k]+f*rqs[k+1];rhp=(1-f)*positions[k]+f*positions[k+1];rhr=rotations(clock).as_matrix()
            d.qpos[:]=reference;mujoco.mj_kinematics(m,d);hp=d.xpos[handle].copy();hr=d.xmat[handle].reshape(3,3).copy()
            d.qpos[lq]=angle;d.qpos[oq]=base[oq];d.qpos[bq]=base[bq];mujoco.mj_kinematics(m,d)
            follow=1.-float(smooth_phase((max(0.,clock-.5)-through)/(8.-through)))
            transformedp=d.xpos[handle]+d.xmat[handle].reshape(3,3)@hr.T@(rhp-hp);transformedr=d.xmat[handle].reshape(3,3)@hr.T@rhr
            rhp=rhp+follow*(transformedp-rhp);rhr=Rotation.from_rotvec(follow*Rotation.from_matrix(transformedr@rhr.T).as_rotvec()).as_matrix()@rhr
            lhp=d.xpos[leaf]+d.xmat[leaf].reshape(3,3)@localp;lhr=d.xmat[leaf].reshape(3,3)@localr
            preference=preferred(t);neighbor=coordinates[i,j-1] if j else preference
            if previous is None:previous=neighbor.copy()
            def install(x):
                d.qpos[:]=reference;d.qpos[lq]=angle;d.qpos[oq]=base[oq];d.qpos[bq]=base[bq];d.qpos[rq:rq+3]=base[rq:rq+3]+x[:3]
                quat=(Rotation.from_rotvec(x[3:6])*rr).as_quat();d.qpos[rq+3:rq+7]=quat[[3,0,1,2]];d.qpos[qa]=x[6:]
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            def residual(x):
                install(x)
                hands=np.r_[100*(d.site_xpos[lh]-lhp),10*Rotation.from_matrix(lhr@d.site_xmat[lh].reshape(3,3).T).as_rotvec(),100*(d.site_xpos[rh]-rhp),10*Rotation.from_matrix(rhr@d.site_xmat[rh].reshape(3,3).T).as_rotvec()]
                foot=np.concatenate([np.r_[100*(d.xpos[b]-fp[k]),10*Rotation.from_matrix(fr[k]@d.xmat[b].reshape(3,3).T).as_rotvec()] for k,b in enumerate(feet)])
                tilt=np.arccos(np.clip(d.xmat[torso].reshape(3,3)[2,2],-1,1))
                barriers=50*np.maximum([np.linalg.norm(x[:3])-.029,np.linalg.norm(d.subtree_com[robotbody,:2]-com[:2])-.014,tilt-np.radians(3.9)],0.)
                return np.r_[hands,foot,.008*(x-preference),.006*(x-previous),.01*(x-neighbor),barriers]
            if i==0 and angle==base[lq]:x=initial_coordinate.copy()
            else:
                seed=neighbor if j else previous
                fit=least_squares(residual,np.clip(seed,lower+1e-12,upper-1e-12),bounds=(lower,upper),max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                x=fit.x
            coordinates[i,j]=x;residuals[i,j]=np.linalg.norm(residual(x)[:24]);previous=x.copy()
        print(json.dumps(dict(angle_rad=float(angle),column=j,columns=len(angles),maximum_pose_residual=float(residuals[:,j].max()))),flush=True)
        np.savez_compressed(a.output/'partial-map.npz',elapsed_s=elapsed,leaf_rad=angles,coordinates=coordinates,residuals=residuals,completed_columns=j+1)
    inputs={**audit['input_sha256'],str(a.coupled_audit.resolve()):sha(a.coupled_audit),str(frozen.resolve()):sha(frozen)}
    result=dict(schema='doorbench.coupled-release-envelope.v1',scope=__doc__,source_candidate=str(a.coupled_candidate.resolve()),source_admission=admission,elapsed_s=elapsed.tolist(),leaf_rad=angles.tolist(),coordinates=coordinates.tolist(),pose_residual_norm=residuals.tolist(),operator_reference_rad=float(base[oq]),operator_envelope_rad=[-.01,.01],duration_s=c['duration_s'],input_sha256=inputs,physics_steps=0,physical_admission=False,interpolation='Tensor cubic spline of progress time and measured leaf angle; bounded arm correction in measured leaf/handle frames')
    (a.output/'envelope.json').write_text(json.dumps(result,indent=2)+'\n');sim.close()
    print(json.dumps(dict(envelope=str(a.output/'envelope.json'),solves=len(elapsed)*len(angles),physics_steps=0)),flush=True)


if __name__=='__main__':main()

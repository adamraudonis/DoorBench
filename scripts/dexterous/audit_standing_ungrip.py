#!/usr/bin/env python3
"""Independently screen an upright withdrawal from a qualified physical return.

This uses unstepped geometry, never inferred forces. It admits a candidate for a
separate motor test; it does not establish physical release or traversal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.landed_left_audit import static_pose_check
from doorbench.dexterous.grasp_verification import shadow_surface_qualified


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def release_surface_scores(digit,segment,point,normal,axial_clearance,alignment,*,profile):
    """Selected anatomy and the unchanged distal counter-score share side gates."""
    geometry_ok=axial_clearance>=.001 and alignment>.8
    selected=shadow_surface_qualified(digit,segment,point,normal,profile=profile)
    distal=shadow_surface_qualified(digit,segment,point,normal)
    return bool(geometry_ok and selected),bool(geometry_ok and distal)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--duration',type=float,default=16.)
    p.add_argument('--grasp-profile',choices=('distal-pad-v1','volar-phalange-v1'),default='distal-pad-v1')
    p.add_argument('--contact-audit-name',default='independent-pad-audit.json')
    p.add_argument('--measured-rest-transfer',action='store_true')
    a=p.parse_args()
    if not np.isfinite(a.duration) or a.duration<=0:raise ValueError('Positive finite duration required')
    source=a.source_run.resolve()
    if (a.screen.parent/'standing-audit.json').exists():raise ValueError('Fresh audit output required')
    whole_path=source/'independent-whole-handle-audit.json'
    if json.loads(whole_path.read_text()).get('passed') is not True:raise ValueError('Whole-handle source qualification required')
    from doorbench.dexterous.release_source_admission import admit_release_source
    admission=admit_release_source(source,profile=a.grasp_profile,
        contact_audit_name=a.contact_audit_name,measured_rest=a.measured_rest_transfer)
    manifest=json.loads((source/'manifest.json').read_text());cfg=manifest['configuration']
    robot=Path(cfg['robot']);door=Path(cfg['door'])
    if sha(robot)!=manifest['inputs']['robot']['sha256'] or sha(door/'door.xml')!=manifest['inputs']['door']['door.xml']:raise ValueError('Physical model bytes changed')
    screen=json.loads(a.screen.read_text())
    if screen.get('grasp_profile','distal-pad-v1')!=a.grasp_profile:raise ValueError('Dense screen must retain the planned source profile')
    if a.measured_rest_transfer and screen.get('source_admission')!=admission:raise ValueError('Dense screen belongs to another measured-rest source')
    if screen['source_trajectory_sha256']!=sha(source/'trajectory.npz'):raise ValueError('Candidate belongs to another attained state')
    planner=Path(screen.get('planner_source_path',Path(__file__).resolve().parents[2]/'doorbench/dexterous/whole_body_ungrip_planner.py'))
    if screen['planner_source_sha256']!=sha(planner):raise ValueError('Planner changed since candidate generation')
    with np.load(source/'trajectory.npz') as z:actual=z['terminal_qpos'].copy();start_time=float(z['terminal_time_s'])
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    sim.reset(randomize=False,images=False);m,d=sim.m,sim.d;rq=sim.root_qadr
    rh=m.site('robot/rh_palm_touch').id;lh=m.site('robot/lh_palm_touch').id
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    torso=m.body('robot/torso_link').id;lever=m.geom('leaf_handle_lever_col_n').id
    d.qpos[:]=actual;mujoco.mj_kinematics(m,d)
    lp=d.site_xpos[lh].copy();lr=d.site_xmat[lh].reshape(3,3).copy()
    fp=d.xpos[feet].copy();fr=d.xmat[feet].reshape(2,3,3).copy()
    initial=dict(time_s=0.,qpos=actual.tolist(),palm_position=d.site_xpos[rh].tolist(),palm_rotation=d.site_xmat[rh].reshape(3,3).tolist())
    rows=[initial]+[{**r,'time_s':r['time_s']+.5} for r in screen['trials'][0]['rows']]
    times=np.array([r['time_s'] for r in rows]);qs=np.array([r['qpos'] for r in rows])
    if qs.shape!=(len(rows),m.nq) or not np.isfinite(qs).all() or not np.all(np.diff(times)>0):raise ValueError('Finite complete monotonic candidate required')
    rotations=Slerp(times,Rotation.from_quat(qs[:,rq+3:rq+7][:,[1,2,3,0]]))
    palms=Slerp(times,Rotation.from_matrix(np.array([r['palm_rotation'] for r in rows])))
    positions=np.array([r['palm_position'] for r in rows])
    failures=[];sampled=[];peakp=peakr=peakt=0.;distal_invalid_samples=distal_invalid_patches=0
    for elapsed in np.linspace(0,a.duration,2001):
        u=elapsed/a.duration;clock=times[-1]*u**3*(10+u*(-15+6*u))
        i=min(len(times)-2,max(0,int(np.searchsorted(times,clock,side='right')-1)));f=(clock-times[i])/(times[i+1]-times[i])
        q=(1-f)*qs[i]+f*qs[i+1];quat=rotations(clock).as_quat();q[rq+3:rq+7]=quat[[3,0,1,2]]
        d.qpos[:]=q;check=static_pose_check(m,d,coordinate=1.);check['passed']=bool(check['passed'])
        pe=max(np.linalg.norm(d.site_xpos[rh]-((1-f)*positions[i]+f*positions[i+1])),np.linalg.norm(d.site_xpos[lh]-lp),max(np.linalg.norm(d.xpos[b]-p) for b,p in zip(feet,fp)))
        rotation_pairs=[(palms(clock).as_matrix(),d.site_xmat[rh].reshape(3,3)),(lr,d.site_xmat[lh].reshape(3,3))]+list(zip(fr,d.xmat[feet].reshape(2,3,3)))
        er=max(np.linalg.norm(Rotation.from_matrix(x@y.T).as_rotvec()) for x,y in rotation_pairs)
        tilt=float(np.degrees(np.arccos(np.clip(d.xmat[torso].reshape(3,3)[2,2],-1,1))))
        invalid=[];distal_invalid=[]
        for c in d.contact[:d.ncon]:
            bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
            if c.dist<0 and 'leaf_handle' in bodies and any(b.startswith('robot/rh_') for b in bodies) and lever not in c.geom:invalid.append('contact outside grasped lever')
            if lever not in c.geom or c.dist>=0:continue
            side=0 if c.geom[1]==lever else 1;b=int(m.geom_bodyid[c.geom[side]]);name=m.body(b).name
            if not name.startswith('robot/rh_'):continue
            match=re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(distal|middle|proximal)',name)
            R=d.xmat[b].reshape(3,3);local=R.T@(c.pos-d.xpos[b]);normal=R.T@(c.frame[:3]*(1 if side==0 else -1))
            axis=d.geom_xmat[lever].reshape(3,3)[:,2];rel=c.pos-d.geom_xpos[lever];axial=float(rel@axis);radial=rel-axial*axis
            alignment=float((R@normal)@(-radial/max(np.linalg.norm(radial),1e-12)))
            selected,distal=(release_surface_scores(*match.groups(),local,normal,
                m.geom_size[lever,1]-abs(axial),alignment,profile=a.grasp_profile) if match else (False,False))
            if not selected:invalid.append(name)
            if not distal:distal_invalid.append(name)
        distal_invalid_samples+=bool(distal_invalid);distal_invalid_patches+=len(distal_invalid)
        peakp=max(peakp,float(pe));peakr=max(peakr,float(er));peakt=max(peakt,tilt);sampled.append(q)
        if not check['passed'] or invalid or pe>.001 or er>.01 or tilt>4:failures.append(dict(time_s=float(elapsed),invalid_surfaces=invalid,position_error_m=float(pe),rotation_error_rad=float(er),torso_tilt_deg=tilt,collision=check))
    hand=[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
    scene=[g for g in range(m.ngeom) if m.geom_contype[g] and not m.body(m.geom_bodyid[g]).name.startswith('robot/')]
    clearance=min(float(mujoco.mj_geomDistance(m,d,g,h,.5,None)) for g in hand for h in scene)
    scalar=[m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE and m.joint(j).name.startswith('robot/')]
    speed=float(np.max(abs(np.diff(np.array(sampled)[:,scalar],axis=0)/(a.duration/2000))))
    inputs=[robot,door/'door.xml',source/'trajectory.npz',source/'report.json',source/a.contact_audit_name,whole_path,a.screen,planner,Path(__file__)]+[Path(p) for p in admission['input_sha256']]
    result=dict(passed=not failures and clearance>=.04 and speed<=2.,samples=2001,physics_steps=0,duration_s=a.duration,initial_episode_time_s=start_time,maximum_position_error_m=peakp,maximum_rotation_error_rad=peakr,maximum_torso_tilt_deg=peakt,maximum_joint_reference_velocity_rad_s=speed,final_rh_environment_clearance_m=clearance,failures=failures,input_sha256={str(p):sha(p) for p in inputs},scope='Unstepped independent geometry and anatomy screen only; physical release remains unqualified')
    result.update(grasp_profile=a.grasp_profile,
        original_distal_counter_score=dict(invalid_anatomy_samples=distal_invalid_samples,invalid_anatomy_patches=distal_invalid_patches),
        source_admission=admission)
    (a.screen.parent/'standing-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('failures','input_sha256')}),flush=True)
    sim.close()
    return 0 if result['passed'] else 1


if __name__=='__main__':sys.exit(main())

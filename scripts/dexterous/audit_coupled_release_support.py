#!/usr/bin/env python3
"""Independently screen coupled support geometry; never infer contact forces."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import shadow_surface_qualified
from doorbench.dexterous.landed_left_audit import static_pose_check
from doorbench.dexterous.release_source_admission import admit_release_source
from doorbench.dexterous.operation_teacher import smooth_phase


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Fresh independent evidence required')
    c=json.loads(a.candidate.read_text());cfg=c['configuration']
    if c['schema']!='doorbench.coupled-release-support-screen.v1':raise ValueError('Explicit coupled support geometry schema required')
    for name,digest in c['input_sha256'].items():
        if sha(name)!=digest:raise ValueError('Coupled candidate input changed: '+name)
    admission=admit_release_source(cfg['source_run'],profile='volar-phalange-v1',contact_audit_name='independent-contact-audit.json',measured_rest=True)
    if admission!=c['source_admission']:raise ValueError('Candidate belongs to another qualified source')
    sim=DexterousDoorEnv(Path(c['door_path']),Path(c['robot_path']),json.loads(Path(c['robot_path']).with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d;rows=c['rows'];times=np.array([r['elapsed_s'] for r in rows]);qs=np.array([r['qpos'] for r in rows]);coords=np.array([r['coordinates'] for r in rows])
    if len(times)<101 or not np.all(np.diff(times)>0) or not np.isfinite(qs).all():raise ValueError('Finite monotonic dense candidate required')
    def spline(y):return CubicSpline(times,y,bc_type=((1,np.zeros_like(y[0])),(1,np.zeros_like(y[-1]))))
    path=spline(coords);fingers=spline(qs);angles=spline(np.array([r['leaf_rad'] for r in rows]))
    right_positions=spline(np.array([r['right_palm_position'] for r in rows]));right_rotations=Slerp(times,Rotation.from_matrix([r['right_palm_rotation'] for r in rows]))
    initial=np.array(c['initial_qpos']);rq=c['root_qpos_address'];initial_rotation=Rotation.from_quat(initial[rq+3:rq+7][[1,2,3,0]])
    names=c['joint_names'];joints=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[joints]
    leaf=m.body('leaf').id;lq=m.joint('leaf_hinge').qposadr[0];lh=m.site('robot/lh_palm_touch').id;rh=m.site('robot/rh_palm_touch').id
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    robotbody=m.jnt_bodyid[m.joint('robot/free_base').id];torso=m.body('robot/torso_link').id;lever=m.geom('leaf_handle_lever_col_n').id
    palm_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==m.site_bodyid[lh] and (m.geom_contype[g] or m.geom_conaffinity[g])]
    panel_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==leaf and (m.geom_contype[g] or m.geom_conaffinity[g])]
    right_geoms=[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith('robot/rh_') and (m.geom_contype[g] or m.geom_conaffinity[g])]
    handle_geoms=[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name=='leaf_handle' and (m.geom_contype[g] or m.geom_conaffinity[g])]
    d.qpos[:]=initial;mujoco.mj_kinematics(m,d)
    initial_gap=min(float(mujoco.mj_geomDistance(m,d,g,h,.1,None)) for g in palm_geoms for h in panel_geoms)
    maximum={};failures=[];right_failures=[];minimum_wrist_margin=float('inf');distal_invalid=0
    minimum_blend_handle_gap=float('inf');blend_samples=0;right_failed_samples=0;support_failed_samples=0
    def rotation_error(a,b):return float(np.linalg.norm(Rotation.from_matrix(a@b.T).as_rotvec()))
    for sample,t in enumerate(np.linspace(0,c['duration_s'],2001)):
        x=path(t);velocity=path(t,1);acceleration=path(t,2);d.qpos[:]=fingers(t);d.qpos[lq]=angles(t)
        d.qpos[rq:rq+3]=initial[rq:rq+3]+x[:3];q=(Rotation.from_rotvec(x[3:6])*initial_rotation).as_quat();d.qpos[rq+3:rq+7]=q[[3,0,1,2]];d.qpos[qa]=x[6:]
        check=static_pose_check(m,d,coordinate=1.);mujoco.mj_comPos(m,d)
        left_position=d.xpos[leaf]+d.xmat[leaf].reshape(3,3)@np.array(c['left_palm_in_leaf_position'])
        left_rotation=d.xmat[leaf].reshape(3,3)@np.array(c['left_palm_in_leaf_rotation'])
        gap=min(float(mujoco.mj_geomDistance(m,d,g,h,.1,None)) for g in palm_geoms for h in panel_geoms)
        values=dict(left_position_error_m=float(np.linalg.norm(d.site_xpos[lh]-left_position)),left_rotation_error_rad=rotation_error(left_rotation,d.site_xmat[lh].reshape(3,3)),right_position_error_m=float(np.linalg.norm(d.site_xpos[rh]-right_positions(t))),right_rotation_error_rad=rotation_error(right_rotations(t).as_matrix(),d.site_xmat[rh].reshape(3,3)),foot_position_error_m=max(float(np.linalg.norm(d.xpos[b]-np.array(c['initial_feet_positions'][i]))) for i,b in enumerate(feet)),foot_rotation_error_rad=max(rotation_error(np.array(c['initial_feet_rotations'][i]),d.xmat[b].reshape(3,3)) for i,b in enumerate(feet)),torso_tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[torso].reshape(3,3)[2,2],-1,1)))),root_translation_m=float(np.linalg.norm(x[:3])),com_xy_displacement_m=float(np.linalg.norm(d.subtree_com[robotbody,:2]-np.array(c['initial_com'][:2]))),joint_velocity_rad_s=float(np.max(abs(velocity[6:]))),joint_acceleration_rad_s2=float(np.max(abs(acceleration[6:]))),root_velocity_m_s=float(np.linalg.norm(velocity[:3])),root_rotation_velocity_rad_s=float(np.linalg.norm(velocity[3:6])),palm_panel_gap_change_m=abs(gap-initial_gap))
        for key,val in values.items():maximum[key]=max(maximum.get(key,0.),val)
        wrists=[m.joint('robot/'+n).id for n in ('left_wrist_yaw','lh_WRJ2','lh_WRJ1')]
        margin=min(float(min(d.qpos[m.jnt_qposadr[j]]-m.jnt_range[j,0],m.jnt_range[j,1]-d.qpos[m.jnt_qposadr[j]])) for j in wrists);minimum_wrist_margin=min(minimum_wrist_margin,margin)
        limits=dict(left_position_error_m=.001,left_rotation_error_rad=.01,right_position_error_m=.001,right_rotation_error_rad=.01,foot_position_error_m=.001,foot_rotation_error_rad=.01,torso_tilt_deg=4.,root_translation_m=.03,com_xy_displacement_m=.015,joint_velocity_rad_s=1.2,joint_acceleration_rad_s2=3.,root_velocity_m_s=.02,root_rotation_velocity_rad_s=.03,palm_panel_gap_change_m=.001)
        bad=[key for key,val in values.items() if not np.isfinite(val) or val>limits[key]]
        # Keep the original full collision check. Identify separately whether
        # a failed RH/door route prevents otherwise valid receiving/body work.
        right_collisions=[v for v in check['contacts'] if any(b.startswith('robot/rh_') for b in v['bodies']) and not all(b.startswith('robot/') for b in v['bodies'])]
        other_collisions=[v for v in check['contacts'] if v not in right_collisions]
        if (other_collisions or not check['finite'] or check['maximum_joint_violation']>.02
                or check['maximum_loopback_violation_rad']>.02):bad.append('original_body_collision_joint_loopback_limits')
        if bad:
            support_failed_samples+=1
            if len(failures)<100:failures.append(dict(elapsed_s=float(t),failed=bad,values=values,collision=check))
        invalid=['RH/environment penetration'] if right_collisions else []
        through=cfg.get('follow_handle_through_route_seconds')
        route_clock=max(0.,8.5*float(smooth_phase(t/c['duration_s']))-.5)
        if through is not None and route_clock>=through:
            gap=min(float(mujoco.mj_geomDistance(m,d,g,h,.2,None)) for g in right_geoms for h in handle_geoms)
            minimum_blend_handle_gap=min(minimum_blend_handle_gap,gap);blend_samples+=1
            if gap<.004:invalid.append('all-handle clearance below literal4mm during world blend')
        for contact in d.contact[:d.ncon]:
            if contact.dist>=0:continue
            bodies=[m.body(m.geom_bodyid[g]).name for g in contact.geom]
            if 'leaf_handle' not in bodies or not any(b.startswith('robot/rh_') for b in bodies):continue
            if lever not in contact.geom:invalid.append('off-lever RH contact');continue
            side=0 if contact.geom[1]==lever else 1;b=int(m.geom_bodyid[contact.geom[side]]);name=m.body(b).name
            match=re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(distal|middle|proximal)',name)
            R=d.xmat[b].reshape(3,3);point=R.T@(contact.pos-d.xpos[b]);normal=R.T@(contact.frame[:3]*(1 if side==0 else -1))
            axis=d.geom_xmat[lever].reshape(3,3)[:,2];rel=contact.pos-d.geom_xpos[lever];axial=float(rel@axis);radial=rel-axial*axis;alignment=float((R@normal)@(-radial/max(np.linalg.norm(radial),1e-12)))
            geometry=m.geom_size[lever,1]-abs(axial)>=.001 and alignment>.8
            if not match or not geometry or not shadow_surface_qualified(*match.groups(),point,normal,profile='volar-phalange-v1'):invalid.append(name)
            if not match or not geometry or not shadow_surface_qualified(*match.groups(),point,normal):distal_invalid+=1
        if invalid:
            right_failed_samples+=1
            if len(right_failures)<100:right_failures.append(dict(elapsed_s=float(t),route_clock_s=route_clock,invalid_surfaces=invalid))
    exact_initial=np.array_equal(qs[0],initial)
    frozen=a.output.with_name(a.output.stem+'-source.py');shutil.copy2(__file__,frozen)
    inputs={**c['input_sha256'],str(a.candidate.resolve()):sha(a.candidate),str(frozen.resolve()):sha(frozen)}
    result=dict(schema='doorbench.coupled-release-support-audit.v1',passed=exact_initial and not failures and not right_failures,support_body_geometry_passed=exact_initial and not failures,right_hand_geometry_passed=not right_failures,exact_initial_state=exact_initial,samples=2001,physics_steps=0,physical_admission=False,maximum=maximum,minimum_left_wrist_physical_margin_rad=minimum_wrist_margin,minimum_world_blend_all_handle_gap_m=minimum_blend_handle_gap if blend_samples else None,world_blend_samples=blend_samples,support_failed_samples=support_failed_samples,right_hand_failed_samples=right_failed_samples,initial_actual_palm_panel_gap_m=initial_gap,failures=failures,right_hand_failures=right_failures,original_distal_invalid_geometry_patches=distal_invalid,input_sha256=inputs,scope=__doc__,interpolation='Clamped CubicSpline of root displacement/rotvec, scalar joints and measured leaf angle; original root quaternion rebuilt from rotvec; RH rotation Slerp')
    a.output.write_text(json.dumps(result,indent=2)+'\n');sim.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('failures','right_hand_failures','input_sha256')},indent=2),flush=True)


if __name__=='__main__':main()

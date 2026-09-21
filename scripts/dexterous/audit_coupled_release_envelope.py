#!/usr/bin/env python3
"""Independent dense geometry over release progress and actual mechanism angles."""
import argparse
import json
from pathlib import Path
import re
import shutil

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from doorbench.dexterous.coupled_release_geometry import CoupledReleaseGeometry,sha
from doorbench.dexterous.grasp_verification import shadow_surface_qualified
from doorbench.dexterous.landed_left_audit import static_pose_check


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envelope',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coarse',action='store_true',help='Diagnostic search only; never runtime admission')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh independent envelope evidence required')
    model=CoupledReleaseGeometry(a.envelope);m,d=model.m,model.d;c=model.c
    feet=[m.body('robot/'+s+'_ankle_link').id for s in ('left','right')];torso=m.body('robot/torso_link').id;robotbody=m.jnt_bodyid[m.joint('robot/free_base').id]
    lever=m.geom('leaf_handle_lever_col_n').id
    active=[g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    right=[g for g in active if m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')];handle=[g for g in active if m.geom_bodyid[g]==model.handle]
    palm=[g for g in active if m.geom_bodyid[g]==m.site_bodyid[model.lh]];panel=[g for g in active if m.geom_bodyid[g]==model.leaf]
    d.qpos[:]=model.initial;mujoco.mj_kinematics(m,d)
    initial_gap=min(float(mujoco.mj_geomDistance(m,d,g,h,.1,None)) for g in palm for h in panel)
    maximum={};failed_count=0;failures=[];samples=0;minimum_blend_gap=float('inf');minimum_wrist_margin=float('inf');distal_invalid=0
    limits=dict(left_position_error_m=.001,left_rotation_error_rad=.01,right_position_error_m=.001,right_rotation_error_rad=.01,foot_position_error_m=.001,foot_rotation_error_rad=.01,torso_tilt_deg=4.,root_translation_m=.03,com_xy_displacement_m=.015,palm_panel_gap_change_m=.001)
    def error(a,b):return float(np.linalg.norm(Rotation.from_matrix(a@b.T).as_rotvec()))
    def inspect(t,angle,operator,latch):
        nonlocal samples,failed_count,minimum_blend_gap,minimum_wrist_margin,distal_invalid
        result=model.evaluate(float(t),float(angle),float(operator),float(latch));check=static_pose_check(m,d,coordinate=1.);mujoco.mj_comPos(m,d);x=result['coordinates'];samples+=1
        gap=min(float(mujoco.mj_geomDistance(m,d,g,h,.1,None)) for g in palm for h in panel)
        values=dict(left_position_error_m=float(np.linalg.norm(d.site_xpos[model.lh]-result['left_palm_position'])),left_rotation_error_rad=error(result['left_palm_rotation'],d.site_xmat[model.lh].reshape(3,3)),right_position_error_m=float(np.linalg.norm(d.site_xpos[model.rh]-result['right_palm_position'])),right_rotation_error_rad=error(result['right_palm_rotation'],d.site_xmat[model.rh].reshape(3,3)),foot_position_error_m=max(float(np.linalg.norm(d.xpos[b]-c['initial_feet_positions'][i])) for i,b in enumerate(feet)),foot_rotation_error_rad=max(error(np.array(c['initial_feet_rotations'][i]),d.xmat[b].reshape(3,3)) for i,b in enumerate(feet)),torso_tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[torso].reshape(3,3)[2,2],-1,1)))),root_translation_m=float(np.linalg.norm(x[:3])),com_xy_displacement_m=float(np.linalg.norm(d.subtree_com[robotbody,:2]-np.array(c['initial_com'][:2]))),palm_panel_gap_change_m=abs(gap-initial_gap))
        for key,val in values.items():maximum[key]=max(maximum.get(key,0.),val)
        bad=[key for key,value in values.items() if not np.isfinite(value) or value>limits[key]]
        if not check['passed']:bad.append('original_collision_joint_loopback_limits')
        wrists=[m.joint('robot/'+n).id for n in ('left_wrist_yaw','lh_WRJ2','lh_WRJ1')]
        minimum_wrist_margin=min(minimum_wrist_margin,min(float(min(d.qpos[m.jnt_qposadr[j]]-m.jnt_range[j,0],m.jnt_range[j,1]-d.qpos[m.jnt_qposadr[j]])) for j in wrists))
        if result['release_clock_s']-.5>=model.through:
            gap=min(float(mujoco.mj_geomDistance(m,d,g,h,.2,None)) for g in right for h in handle);minimum_blend_gap=min(minimum_blend_gap,gap)
            if gap<.004:bad.append('literal4mm_all_handle_blend_clearance')
        for contact in d.contact[:d.ncon]:
            if contact.dist>=0:continue
            bodies=[m.body(m.geom_bodyid[g]).name for g in contact.geom]
            if 'leaf_handle' not in bodies or not any(b.startswith('robot/rh_') for b in bodies):continue
            if lever not in contact.geom:bad.append('RH contact outside lever');continue
            side=0 if contact.geom[1]==lever else 1;b=int(m.geom_bodyid[contact.geom[side]]);name=m.body(b).name
            match=re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(distal|middle|proximal)',name)
            R=d.xmat[b].reshape(3,3);point=R.T@(contact.pos-d.xpos[b]);normal=R.T@(contact.frame[:3]*(1 if side==0 else -1))
            axis=d.geom_xmat[lever].reshape(3,3)[:,2];rel=contact.pos-d.geom_xpos[lever];axial=float(rel@axis);radial=rel-axial*axis;alignment=float((R@normal)@(-radial/max(np.linalg.norm(radial),1e-12)))
            geometry=m.geom_size[lever,1]-abs(axial)>=.001 and alignment>.8
            if not match or not geometry or not shadow_surface_qualified(*match.groups(),point,normal,profile='volar-phalange-v1'):bad.append('Original RH selected anatomy/side gates: '+name)
            if not match or not geometry or not shadow_surface_qualified(*match.groups(),point,normal):distal_invalid+=1
        if bad:
            failed_count+=1
            if len(failures)<100:failures.append(dict(elapsed_s=float(t),leaf_rad=float(angle),operator_rad=float(operator),latch_m=float(latch),failed=bad,values=values,collisions=check['contacts']))
    # Off-grid points cover each tensor cell, including operator extrema. The
    # independent original2,001-sample coupled path remains bound in provenance.
    lo,hi=model.plan['operator_envelope_rad']
    schedules=[(model.plan['operator_reference_rad'],81,18),(lo,81,18),(hi,81,18)] if a.coarse else [(model.plan['operator_reference_rad'],501,65),(lo,201,33),(hi,201,33)]
    if not a.coarse and hi-lo>.04:
        schedules.extend((value,101,33) for value in np.linspace(lo,hi,5)[1:-1])
    for operator,nt,na in schedules:
        for j,t in enumerate(np.linspace(0,model.plan['duration_s'],nt)):
            for angle in np.linspace(.08,model.upper_angle(t),na):inspect(t,angle,operator,model.initial[model.bq])
            if j%80==0:print(json.dumps(dict(samples=samples,failed_samples=failed_count,elapsed_s=float(t),operator_rad=float(operator))),flush=True)
    # Door-bolt motion is still a measured domain, even though it is far from
    # the palms: explicitly inspect both extrema rather than assuming this.
    for latch in (-.001,.001):
        for operator in (lo,hi):
            for t in np.linspace(0,model.plan['duration_s'],21 if a.coarse else 81):
                for angle in np.linspace(.08,model.upper_angle(t),9 if a.coarse else 17):inspect(t,angle,operator,latch)
    exact=np.array_equal(model.evaluate(0.,float(model.initial[model.lq]),float(model.initial[model.oq]),float(model.initial[model.bq]))['qpos'],model.initial)
    frozen=a.output.with_name(a.output.stem+'-source.py');shutil.copy2(__file__,frozen)
    import doorbench.dexterous.coupled_release_geometry as geometry
    result=dict(schema='doorbench.coupled-release-envelope-audit.v1',passed=not failed_count and exact and not a.coarse,coarse_diagnostic=a.coarse,samples=samples,failed_samples=failed_count,exact_initial_state=exact,physics_steps=0,physical_admission=False,geometry_domain=dict(elapsed_s=[0,model.plan['duration_s']],leaf_rad=[.08,.4],admitted_leaf_upper_nodes=model.plan.get('admitted_leaf_upper_nodes'),operator_rad=[lo,hi],latch_m=[-.001,.001]),maximum=maximum,limits=limits,minimum_blend_all_handle_clearance_m=minimum_blend_gap,minimum_left_wrist_physical_margin_rad=minimum_wrist_margin,original_distal_invalid_geometry_patches=distal_invalid,failures=failures,input_sha256={**model.plan['input_sha256'],str(a.envelope.resolve()):sha(a.envelope),str(Path(geometry.__file__).resolve()):sha(geometry.__file__),str(frozen.resolve()):sha(frozen)},runtime_motion_contract='Separate online original1.2rad/s,3rad/s² joint and0.02m/s,0.03rad/s root reference gates; this static2D screen does not qualify arbitrary mechanism speeds',scope=__doc__)
    a.output.write_text(json.dumps(result,indent=2)+'\n');model.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('failures','input_sha256')},indent=2),flush=True)


if __name__=='__main__':main()

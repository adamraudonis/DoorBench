#!/usr/bin/env python3
"""Dense audit of a recontact path with a cleared RH held in joint posture.

No physics is stepped and no contact loads are inferred. This cannot qualify
support, motor tracking, a release, or a full opening. The exact measured initial
state is included rather than silently using the optimizer's adjusted first row.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,required=True)
    parser.add_argument('--duration-s',type=float,default=20.)
    parser.add_argument('--samples',type=int,default=2001)
    parser.add_argument('--actual-leaf-lag-rad',type=float,default=0.,help='Conservative static door lag relative to the unchanged planned pose; initial state remains exact')
    parser.add_argument('--lag-start-angle-rad',type=float,help='Conservative envelope: retain5mrad below this reference angle, allow full declared lag above it')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 0<=args.actual_leaf_lag_rad<=.02:raise ValueError('Require declared static leaf lag in [0,.02] rad')
    if args.samples<2001:raise ValueError('Require >=2001 dense samples')
    report=json.loads(args.screen.read_text())
    if report['configuration'].get('right_target_mode')!='fixed-attained-joints':raise ValueError('Require the explicitly declared cleared-arm joint posture contract')
    run=Path(report['configuration']['source_run']).resolve()
    config=json.loads((run/'manifest.json').read_text())['configuration']
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from doorbench.dexterous.screened_panel_path import ScreenedPanelPath
    from doorbench.dexterous.palm_panel_geometry import original_palm_vertices,flatten_palm_goal
    from doorbench.dexterous.environment import DexterousDoorEnv
    robot=Path(config['robot'])
    sim=DexterousDoorEnv(config['door'],robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d
    initial=np.array(report['initial_qpos']);names=report['names']
    joint_ids=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[joint_ids]
    right_indices=[i for i,n in enumerate(names) if n.startswith('right_') and any(k in n for k in ('shoulder','elbow','wrist')) or n.startswith('rh_WRJ')]
    if len(right_indices)!=7:raise ValueError('Require exactly seven cleared RH arm joints')
    rq=m.jnt_qposadr[m.joint('robot/free_base').id]
    r0=Rotation.from_quat([*initial[rq+4:rq+7],initial[rq+3]])
    leafq=m.jnt_qposadr[m.joint('leaf_hinge').id];leafbody=m.body('leaf').id
    angles=np.array([r['leaf_angle_rad'] for r in report['rows']])
    coords=np.array([r['root_delta']+[r['joint_targets'][n] for n in names] for r in report['rows']])
    optimizer_initial_change=float(np.max(abs(coords[0]-np.r_[np.zeros(6),initial[qa]])))
    coords[0]=np.r_[np.zeros(6),initial[qa]]
    path=ScreenedPanelPath((angles-angles[0])/(angles[-1]-angles[0]),coords,args.duration_s)
    d.qpos[:]=initial;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    rh,lh=[m.site('robot/'+side+'_palm_touch').id for side in ('rh','lh')]
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    feetp=d.xpos[feet].copy();feetr=d.xmat[feet].reshape(2,3,3).copy()
    rhp=d.site_xpos[rh].copy();rhr=d.site_xmat[rh].reshape(3,3).copy()
    lr=d.xmat[leafbody].reshape(3,3).copy();lp=d.xpos[leafbody].copy()
    lhp=lr.T@(d.site_xpos[lh]-lp);lhr=lr.T@d.site_xmat[lh].reshape(3,3)
    palm_vertices=original_palm_vertices(m,d,lh) if report['configuration'].get('flatten_palm') else None
    robot_body=m.jnt_bodyid[m.joint('robot/free_base').id];com=d.subtree_com[robot_body].copy()
    torso=m.body('robot/torso_link').id;floor=m.geom('floor').id
    scalar=np.array([j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] in (2,3)])
    sq=m.jnt_qposadr[scalar]
    initial_violation=np.maximum(m.jnt_range[scalar,0]-initial[sq],initial[sq]-m.jnt_range[scalar,1])
    hand=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
    scene=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and not m.body(m.geom_bodyid[g]).name.startswith('robot/')]
    # Conservative, explicit enclosing radii for broad-phase pruning. Exact
    # mj_geomDistance still evaluates every pair that can be within 40 mm.
    radii=np.zeros(m.ngeom)
    for g in hand+scene:
        kind=int(m.geom_type[g]);size=m.geom_size[g]
        if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh=m.geom_dataid[g];start=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh]
            radii[g]=float(np.linalg.norm(m.mesh_vert[start:start+count],axis=1).max())
        elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):radii[g]=float(np.linalg.norm(size))
        elif kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):radii[g]=float(size[0])
        elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):radii[g]=float(size[0]+size[1])
        elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):radii[g]=float(np.hypot(size[0],size[1]))
        elif kind==int(mujoco.mjtGeom.mjGEOM_PLANE):radii[g]=np.inf
        else:raise ValueError('Unsupported collision shape; cannot prune safely')
    maxima={};failures=[];minimum_clearance=.04;exact_pairs=0;traces=[]
    times=np.linspace(0,args.duration_s,args.samples)
    for index,t in enumerate(times):
        sample=path.sample(float(t));x=sample['position'];d.qpos[:]=initial
        d.qpos[rq:rq+3]=initial[rq:rq+3]+x[:3]
        quat=(Rotation.from_rotvec(x[3:6])*r0).as_quat();d.qpos[rq+3:rq+7]=np.r_[quat[3],quat[:3]]
        d.qpos[qa]=x[6:];reference_angle=angles[0]+sample['progress']*(angles[-1]-angles[0])
        lag=args.actual_leaf_lag_rad
        if args.lag_start_angle_rad is not None and reference_angle<args.lag_start_angle_rad:lag=min(.005,lag)
        if index==0:lag=0.
        d.qpos[leafq]=reference_angle-lag
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d)
        leafr=d.xmat[leafbody].reshape(3,3);leafp=d.xpos[leafbody]
        if lag:
            leafj=m.joint('leaf_hinge').id;rotation=Rotation.from_rotvec(d.xaxis[leafj]*lag).as_matrix();anchor=d.xanchor[leafj]
            leafp=anchor+rotation@(leafp-anchor);leafr=rotation@leafr
        u=np.clip((reference_angle-angles[0])/.35,0,1);blend=u**3*(10+u*(-15+6*u))
        local=lhp.copy();local[0]+=report['configuration'].get('radius_shift_m',-.04)*blend;local[2]-=report['configuration']['height_drop_m']*blend
        localr=lhr
        if palm_vertices is not None:
            fu=float(np.clip((reference_angle-angles[0])/report['configuration'].get('flatten_over_rad',.2),0,1));fb=fu**3*(10+fu*(-15+6*fu))
            local,localr,_=flatten_palm_goal(local,lhr,palm_vertices,fb)
            local[1]+=report['configuration']['normal_recontact_m']*fb
        targetp=leafp+leafr@local;targetr=leafr@localr
        rotation_error=lambda target,actual:float(np.linalg.norm(Rotation.from_matrix(target@actual.T).as_rotvec()))
        max_joint_increase=float(np.max(np.maximum(m.jnt_range[scalar,0]-d.qpos[sq],d.qpos[sq]-m.jnt_range[scalar,1])-np.maximum(initial_violation,0)))
        vals=dict(left_position_m=float(np.linalg.norm(d.site_xpos[lh]-targetp)),left_rotation_rad=rotation_error(targetr,d.site_xmat[lh].reshape(3,3)),right_joint_posture_rad=float(max(abs(d.qpos[qa[right_indices]]-initial[qa[right_indices]]))),foot_position_m=max(float(np.linalg.norm(d.xpos[b]-feetp[k])) for k,b in enumerate(feet)),foot_rotation_rad=max(rotation_error(feetr[k],d.xmat[b].reshape(3,3)) for k,b in enumerate(feet)),joint_violation_increase_rad=max(0.,max_joint_increase),torso_tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[torso].reshape(3,3)[2,2],-1,1)))),root_translation_m=float(np.linalg.norm(x[:3])),root_rotation_rad=float(np.linalg.norm(x[3:6])),com_xy_displacement_m=float(np.linalg.norm(d.subtree_com[robot_body,:2]-com[:2])),joint_velocity_rad_s=float(np.max(abs(sample['velocity'][6:]))),joint_acceleration_rad_s2=float(np.max(abs(sample['acceleration'][6:]))),root_velocity_m_s=float(np.linalg.norm(sample['velocity'][:3])),root_rotvec_velocity_rad_s=float(np.linalg.norm(sample['velocity'][3:6])))
        collisions=[]
        for contact in d.contact[:d.ncon]:
            bodies=[m.body(m.geom_bodyid[g]).name for g in contact.geom]
            allowed=floor in contact.geom and any(b.endswith('_ankle_link') for b in bodies)
            if not allowed and any(b.startswith('robot/') for b in bodies) and contact.dist<-.003:
                collisions.append(dict(bodies=bodies,depth_m=-float(contact.dist)))
        center_dist=np.linalg.norm(d.geom_xpos[hand,None,:]-d.geom_xpos[None,scene,:],axis=2)
        candidates=np.argwhere(center_dist-radii[hand,None]-radii[None,scene]<.040001)
        nearest=.040001;pair=None
        for hi,si in candidates:
            g,h=hand[hi],scene[si];distance=float(mujoco.mj_geomDistance(m,d,g,h,.040001,None));exact_pairs+=1
            if distance<nearest:nearest=distance;pair=[m.geom(g).name,m.geom(h).name]
        minimum_clearance=min(minimum_clearance,nearest)
        for key,value in vals.items():maxima[key]=max(maxima.get(key,0.),value)
        limits=dict(left_position_m=.0001,left_rotation_rad=.001,right_joint_posture_rad=1e-7,foot_position_m=.0001,foot_rotation_rad=.001,joint_violation_increase_rad=.000001,torso_tilt_deg=12.,root_translation_m=.03,root_rotation_rad=.05,com_xy_displacement_m=.015,joint_velocity_rad_s=1.2,joint_acceleration_rad_s2=3.,root_velocity_m_s=.02,root_rotvec_velocity_rad_s=.03)
        bad={key:vals[key] for key,limit in limits.items() if vals[key]>limit}
        if bad or collisions or nearest<.04:failures.append(dict(time_s=float(t),violations=bad,collisions=collisions,right_clearance_m=nearest,closest_pair=pair))
        traces.append(np.r_[t,sample['progress'],sample['position'],sample['velocity'],sample['acceleration']])
        if index%200==0:print(json.dumps(dict(index=index,time_s=float(t),failures=len(failures),right_clearance_capped_m=nearest)),flush=True)
    bounds=path.derivative_bounds();delta=angles[-1]-angles[0];speed=.149;acceleration=.08
    envelope=dict(maximum_joint_speed_rad_s=float(np.max(bounds['first'][6:])*speed/delta),maximum_joint_acceleration_rad_s2=float(np.max(bounds['second'][6:]*(speed/delta)**2+bounds['first'][6:]*acceleration/delta)),maximum_root_speed_m_s=float(bounds['root_translation_first_norm']*speed/delta),maximum_root_rotvec_speed_rad_s=float(bounds['root_rotvec_first_norm']*speed/delta),aperture_speed_limit_rad_s=speed,aperture_acceleration_limit_rad_s2=acceleration,method='Exact cubic-segment derivative extrema and conservative chain-rule acceleration bound')
    envelope['passed']=envelope['maximum_joint_speed_rad_s']<=1.2 and envelope['maximum_joint_acceleration_rad_s2']<=3. and envelope['maximum_root_speed_m_s']<=.02 and envelope['maximum_root_rotvec_speed_rad_s']<=.03
    if not envelope['passed']:failures.append(dict(measured_aperture_reference_envelope=envelope))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'screened-panel-path-source.py').write_bytes((Path(__file__).resolve().parents[2]/'doorbench/dexterous/screened_panel_path.py').read_bytes())
    receipt=dict(schema='doorbench.whole-body-panel-screen.v1',right_target_mode='fixed-attained-joints',normal_recontact_m=report['configuration']['normal_recontact_m'],passed=not failures,scope='Unstepped geometry and target-rate qualification only. No loaded support, force tracking, release, or opening success claim.',screen_sha256=hashlib.file_digest(args.screen.open('rb'),'sha256').hexdigest(),source_chunk_sha256=report['source_chunk_sha256'],source_time_s=report['source_time_s'],reference_phase_envelope=envelope,actual_leaf_lag_rad=args.actual_leaf_lag_rad,lag_start_angle_rad=args.lag_start_angle_rad,lag_scope='Static collision screen only; does not assume or guarantee the physical tracking lag. The exact initial pose is unshifted.',duration_s=args.duration_s,samples=args.samples,exact_initial_state=True,optimizer_initial_coordinate_adjustment_removed=optimizer_initial_change,limits=limits,maxima=maxima,minimum_all_rh_scene_clearance_capped_m=minimum_clearance,hand_shapes=len(hand),scene_shapes=len(scene),exact_distance_pairs=exact_pairs,failures=failures)
    (args.output/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    np.savez_compressed(args.output/'target-traces.npz',values=np.array(traces))
    plan=dict(schema='doorbench.whole-body-panel-plan.v1',initial_qpos=initial.tolist(),initial_qvel=report['initial_qvel'],initial_robot_joints={m.joint(j).name.removeprefix('robot/'):float(initial[m.jnt_qposadr[j]]) for j in range(m.njnt) if m.joint(j).name.startswith('robot/') and m.jnt_type[j] in (2,3)},robot_xml_sha256=hashlib.file_digest(robot.open('rb'),'sha256').hexdigest(),initial_leaf_velocity_rad_s=float(report['initial_qvel'][m.jnt_dofadr[m.joint('leaf_hinge').id]]),initial_leaf_angle_rad=float(angles[0]),final_leaf_angle_rad=float(angles[-1]),initial_episode_time_s=report['source_time_s'],root_qpos_address=int(rq),joint_names=names,joint_qpos_addresses=qa.tolist(),progress=((angles-angles[0])/(angles[-1]-angles[0])).tolist(),coordinates=coords.tolist(),duration_s=args.duration_s,screen_receipt=receipt)
    (args.output/'target-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='failures'},indent=2));sim.close()


if __name__=='__main__':main()

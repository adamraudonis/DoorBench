"""Independent dense geometry/rate audit for an actual-Isaac panel candidate.

Only a freshly re-admitted successful released Isaac source is accepted. The
original native dense geometry and derivative checks are retained; no physical
support, tracking, aperture success or runtime stage authority is exported.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .isaac_panel_planning import (SCHEMA,JOINT_NAMES,admit_isaac_panel_context,
    _options,panel_source_paths)
from .palm_panel_geometry import original_palm_vertices,flatten_palm_goal
from .qualified_isaac_grasp import digest
from .screened_panel_path import ScreenedPanelPath

AUDIT_SCHEMA='doorbench.isaac-panel-geometry-audit.v1'
PLAN_SCHEMA='doorbench.isaac-panel-geometry-plan.v1'


def validate_panel_candidate(report,context,scene):
    source=context.admission;initial=context.qpos;m=scene.m
    if (report.get('schema')!=SCHEMA or report.get('source_engine')!='isaac-physx'
            or report.get('source_admission')!=source or report.get('source_context_sha256')!=context.sha256
            or report.get('source_run')!=source['source_run'] or report.get('source_time_s')!=source['source_time_s']):
        raise ValueError('Candidate must bind the freshly admitted actual released Isaac source')
    for key in ('authorized_stages','physics_steps','source_sample_playback','active_state_writes'):
        if type(report.get(key)) is not int or report[key]!=0:raise ValueError('Geometry candidate cannot claim stage/plant authority')
    for key in ('geometric_admission','physical_admission','runtime_route_exported'):
        if report.get(key) is not False:raise ValueError('Unadmitted candidate cannot claim prior qualification')
    if report.get('exact_initial_state') is not True or report.get('preferences_are_numeric_only') is not True:
        raise ValueError('Exact source and numeric-only preference contract required')
    if report.get('names')!=JOINT_NAMES or not np.array_equal(report.get('initial_qpos'),initial) or not np.array_equal(report.get('initial_qvel'),context.qvel):
        raise ValueError('Complete unchanged normalized source state and original joint order required')
    config=report['configuration'];target=report['target_aperture_rad']
    expected=vars(_options(config,target_aperture_rad=target,solver_method=config.get('solver_method'),flatten_palm=config.get('flatten_palm')))
    if config!=expected:raise ValueError('Candidate changed the original fixed planning modes or limits')
    if report.get('passage_aperture_target_rad')!=1.57 or report.get('target_reaches_passage_aperture') is not (target>=1.57):
        raise ValueError('Explicit truthful passage-aperture scope required')
    hashes=report.get('input_sha256',{})
    required={**source['input_sha256'],**{str(p):digest(p) for p in panel_source_paths()}}
    if any(hashes.get(p)!=value for p,value in required.items()):raise ValueError('Source/planner input binding differs')
    if any(digest(p)!=value for p,value in hashes.items()):raise ValueError('Candidate input changed')
    rq=int(m.joint('robot/free_base').qposadr[0]);lq=int(m.joint('leaf_hinge').qposadr[0])
    qa=np.array([m.joint('robot/'+name).qposadr[0] for name in JOINT_NAMES])
    if not float(initial[lq])<target<=float(m.jnt_range[m.joint('leaf_hinge').id,1]):
        raise ValueError('Prospective aperture must advance source within original leaf limit')
    rows=report['rows']
    if not isinstance(rows,list) or len(rows)!=config['nodes']:raise ValueError('Complete solver knot set required')
    angles=np.asarray([r['leaf_angle_rad'] for r in rows],float)
    if not np.array_equal(angles,np.linspace(initial[lq],target,len(rows))):raise ValueError('Original complete aperture grid required')
    rotation=Rotation.from_quat(initial[rq+3:rq+7][[1,2,3,0]])
    for index,row in enumerate(rows):
        if set(row['joint_targets'])!=set(JOINT_NAMES):raise ValueError('Complete named reference required')
        x=np.asarray(row['root_delta']+[row['joint_targets'][n] for n in JOINT_NAMES],float)
        q=np.asarray(row['qpos'],float)
        if x.shape!=(6+len(JOINT_NAMES),) or q.shape!=initial.shape or not np.isfinite(np.r_[x,q,angles[index]]).all():
            raise ValueError('Finite complete panel coordinates required')
        reconstructed=initial.copy();reconstructed[rq:rq+3]+=x[:3]
        reconstructed[rq+3:rq+7]=(Rotation.from_rotvec(x[3:6])*rotation).as_quat()[[3,0,1,2]]
        reconstructed[qa]=x[6:];reconstructed[lq]=angles[index]
        if not np.allclose(q,reconstructed,rtol=0,atol=1e-12):raise ValueError('Hidden mechanism/finger/state edits in panel knot')
        if index==0 and (not np.array_equal(q,initial) or not np.array_equal(x,np.r_[np.zeros(6),initial[qa]])):
            raise ValueError('Exact first normalized source knot required; no audit repair')


def audit_panel_candidate(candidate,*,duration_s=40.,samples=2001,
        aperture_speed_limit_rad_s=.149,aperture_acceleration_limit_rad_s2=.08,
        actual_leaf_lag_rad=0.,lag_start_angle_rad=None,progress=None):
    if (type(samples) is not int or not 2001<=samples<=20001
            or not np.isfinite([duration_s,aperture_speed_limit_rad_s,aperture_acceleration_limit_rad_s2,actual_leaf_lag_rad]).all()
            or not 3<=duration_s<=600 or not 0<aperture_speed_limit_rad_s<=.149
            or not 0<aperture_acceleration_limit_rad_s2<=.08 or not 0<=actual_leaf_lag_rad<=.02):
        raise ValueError('Original dense count, finite duration, aperture-rate and lag limits required')
    candidate=Path(candidate).resolve();candidate_hash=digest(candidate);report=json.loads(candidate.read_text())
    if report.get('schema')!=SCHEMA:raise ValueError('Distinct actual-Isaac candidate required; native screens are not proofs')
    source=report['source_admission']
    context=admit_isaac_panel_context(source['source_run'],robot=source['robot_path'],
        door_xml=source['door_xml_path'],door_usd=source['door_usd_path'])
    scene=context.scene();validate_panel_candidate(report,context,scene)
    initial=context.qpos;lq=int(scene.m.joint('leaf_hinge').qposadr[0])
    if lag_start_angle_rad is not None and (type(lag_start_angle_rad) not in (int,float)
            or not np.isfinite(lag_start_angle_rad) or not initial[lq]<=lag_start_angle_rad<=report['target_aperture_rad']):
        raise ValueError('Finite lag transition inside prospective aperture domain required')
    args=SimpleNamespace(duration_s=float(duration_s),samples=samples,
        aperture_speed_limit_rad_s=float(aperture_speed_limit_rad_s),
        aperture_acceleration_limit_rad_s2=float(aperture_acceleration_limit_rad_s2),
        actual_leaf_lag_rad=float(actual_leaf_lag_rad),lag_start_angle_rad=lag_start_angle_rad)
    hashes={**report['input_sha256'],str(candidate):candidate_hash,str(Path(__file__).resolve()):digest(__file__)}
    receipt,plan,traces=_dense_geometry(report,scene,args,progress)
    context.verify_inputs()
    if any(digest(p)!=value for p,value in hashes.items()):raise ValueError('Panel source/candidate/audit input changed during screen')
    binding=dict(source_admission=context.admission,source_context_sha256=context.sha256,
        input_sha256=hashes,candidate_path=str(candidate),candidate_sha256=candidate_hash,
        source_time_s=context.admission['source_time_s'],source_engine='isaac-physx',
        authorized_stages=0,physics_steps=0,source_sample_playback=0,active_state_writes=0,
        physical_admission=False,runtime_route_exported=False)
    receipt.update(binding);plan.update(binding);plan['screen_receipt']=receipt
    return receipt,plan,traces


def _dense_geometry(report,scene,args,progress=None):
    """Unchanged native geometry/rate arithmetic; strict upright proof profile."""
    m,d=scene.m,scene.d;relaxed=False;yaw=None
    initial=np.array(report['initial_qpos']);names=report['names']
    joint_ids=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[joint_ids]
    rq=m.jnt_qposadr[m.joint('robot/free_base').id]
    r0=Rotation.from_quat([*initial[rq+4:rq+7],initial[rq+3]])
    leafq=m.jnt_qposadr[m.joint('leaf_hinge').id];leafbody=m.body('leaf').id
    angles=np.array([r['leaf_angle_rad'] for r in report['rows']])
    coords=np.array([r['root_delta']+[r['joint_targets'][n] for n in names] for r in report['rows']])
    optimizer_initial_change=float(np.max(abs(coords[0]-np.r_[np.zeros(6),initial[qa]])))
    if not np.array_equal(coords[0],np.r_[np.zeros(6),initial[qa]]):raise ValueError('Audit cannot repair a changed source knot')
    path=ScreenedPanelPath((angles-angles[0])/(angles[-1]-angles[0]),coords,args.duration_s)
    d.qpos[:]=initial;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    rh,lh=[m.site('robot/'+side+'_palm_touch').id for side in ('rh','lh')]
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    feetp=d.xpos[feet].copy();feetr=d.xmat[feet].reshape(2,3,3).copy()
    rhp=d.site_xpos[rh].copy();rhr=d.site_xmat[rh].reshape(3,3).copy()
    lr=d.xmat[leafbody].reshape(3,3).copy();lp=d.xpos[leafbody].copy()
    lhp=lr.T@(d.site_xpos[lh]-lp);lhr=lr.T@d.site_xmat[lh].reshape(3,3)
    from doorbench.dexterous.released_hand_goal import released_hand_goal, released_hand_phase
    right_outward=np.sign((rhp-lp)@lr[:,1])*lr[:,1]
    palm_vertices=original_palm_vertices(m,d,lh) if report['configuration'].get('flatten_palm') else None
    robot_body=m.jnt_bodyid[m.joint('robot/free_base').id];com=d.subtree_com[robot_body].copy()
    torso=m.body('robot/torso_link').id;floor=m.geom('floor').id
    scalar=np.array([j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] in (2,3)])
    sq=m.jnt_qposadr[scalar]
    initial_violation=np.maximum(m.jnt_range[scalar,0]-initial[sq],initial[sq]-m.jnt_range[scalar,1])
    hand=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
    scene=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and not m.body(m.geom_bodyid[g]).name.startswith('robot/')]
    elbows=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name in ('robot/left_elbow_link','robot/right_elbow_link')]
    if not hand or not elbows or not scene:raise ValueError('Original hand/elbow/environment collision geometry required')
    minimum_elbow_clearance=.003001
    # Conservative, explicit enclosing radii for broad-phase pruning. Exact
    # mj_geomDistance still evaluates every pair that can be within 40 mm.
    radii=np.zeros(m.ngeom)
    for g in hand+elbows+scene:
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
        if not all(np.isfinite(value).all() for value in (d.qpos,d.xpos,d.xmat,d.site_xpos,d.site_xmat,d.geom_xpos,d.geom_xmat,d.subtree_com)):
            raise ValueError('Nonfinite private geometry cannot pass an independent screen')
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
        twist=report['configuration'].get('palm_twist_rad',0.)
        if twist:
            from doorbench.dexterous.palm_panel_geometry import twist_palm_goal
            localr=twist_palm_goal(localr,twist*blend)
        targetp=leafp+leafr@local;targetr=leafr@localr
        reference_progress=float(sample['progress']);retreat_phase=released_hand_phase(reference_progress)
        right_target_p,right_target_r=released_hand_goal(rhp,rhr,initial[rq:rq+3],x[:6],right_outward,retreat_phase,frame=report['configuration'].get('right_hand_frame','world'),retreat_m=report['configuration'].get('right_hand_retreat_m',0.))
        rotation_error=lambda target,actual:float(np.linalg.norm(Rotation.from_matrix(target@actual.T).as_rotvec()))
        max_joint_increase=float(np.max(np.maximum(m.jnt_range[scalar,0]-d.qpos[sq],d.qpos[sq]-m.jnt_range[scalar,1])-np.maximum(initial_violation,0)))
        vals=dict(left_position_m=float(np.linalg.norm(d.site_xpos[lh]-targetp)),left_rotation_rad=rotation_error(targetr,d.site_xmat[lh].reshape(3,3)),right_position_m=float(np.linalg.norm(d.site_xpos[rh]-right_target_p)),right_rotation_rad=rotation_error(right_target_r,d.site_xmat[rh].reshape(3,3)),foot_position_m=max(float(np.linalg.norm(d.xpos[b]-feetp[k])) for k,b in enumerate(feet)),foot_rotation_rad=max(rotation_error(feetr[k],d.xmat[b].reshape(3,3)) for k,b in enumerate(feet)),joint_violation_increase_rad=max(0.,max_joint_increase),torso_tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[torso].reshape(3,3)[2,2],-1,1)))),root_translation_m=float(np.linalg.norm(x[:3])),root_rotation_rad=float(np.linalg.norm(x[3:6])),com_xy_displacement_m=float(np.linalg.norm(d.subtree_com[robot_body,:2]-com[:2])),joint_velocity_rad_s=float(np.max(abs(sample['velocity'][6:]))),joint_acceleration_rad_s2=float(np.max(abs(sample['acceleration'][6:]))),root_velocity_m_s=float(np.linalg.norm(sample['velocity'][:3])),root_rotvec_velocity_rad_s=float(np.linalg.norm(sample['velocity'][3:6])))
        collisions=[]
        for contact in d.contact[:d.ncon]:
            if not np.isfinite(contact.dist):raise ValueError('Nonfinite collision gap cannot be screened')
            bodies=[m.body(m.geom_bodyid[g]).name for g in contact.geom]
            allowed=floor in contact.geom and any(b.endswith('_ankle_link') for b in bodies)
            if not allowed and any(b.startswith('robot/') for b in bodies) and contact.dist<-.003:
                collisions.append(dict(bodies=bodies,depth_m=-float(contact.dist)))
        center_dist=np.linalg.norm(d.geom_xpos[hand,None,:]-d.geom_xpos[None,scene,:],axis=2)
        candidates=np.argwhere(center_dist-radii[hand,None]-radii[None,scene]<.040001)
        nearest=.040001;pair=None
        for hi,si in candidates:
            g,h=hand[hi],scene[si];distance=float(mujoco.mj_geomDistance(m,d,g,h,.040001,None));exact_pairs+=1
            if not np.isfinite(distance):raise ValueError('Nonfinite hand distance cannot pass clearance')
            if distance<nearest:nearest=distance;pair=[m.geom(g).name,m.geom(h).name]
        minimum_clearance=min(minimum_clearance,nearest)
        elbow_nearest=.003001;elbow_pair=None
        elbow_centers=np.linalg.norm(d.geom_xpos[elbows,None,:]-d.geom_xpos[None,scene,:],axis=2)
        for ei,si in np.argwhere(elbow_centers-radii[elbows,None]-radii[None,scene]<.003001):
            g,h=elbows[ei],scene[si]
            gap=float(mujoco.mj_geomDistance(m,d,g,h,.003001,None));exact_pairs+=1
            if not np.isfinite(gap):raise ValueError('Nonfinite elbow distance cannot pass clearance')
            if gap<elbow_nearest:elbow_nearest=gap;elbow_pair=[int(g),int(h)]
        minimum_elbow_clearance=min(minimum_elbow_clearance,elbow_nearest)
        if elbow_nearest<.003:
            failures.append(dict(time_s=float(t),elbow_clearance_m=elbow_nearest,elbow_geom_pair=elbow_pair,required_elbow_clearance_m=.003))
        for key,value in vals.items():maxima[key]=max(maxima.get(key,0.),value)
        limits=dict(left_position_m=.0001,left_rotation_rad=.001,right_position_m=.0001,right_rotation_rad=.001,foot_position_m=.0001,foot_rotation_rad=.001,joint_violation_increase_rad=.000001,torso_tilt_deg=4.,root_translation_m=.03,root_rotation_rad=.05,com_xy_displacement_m=.015,joint_velocity_rad_s=1.2,joint_acceleration_rad_s2=3.,root_velocity_m_s=.02,root_rotvec_velocity_rad_s=.03)
        if relaxed:limits['right_rotation_rad']=.35
        if yaw is not None:
            vals.update(root_tilt_delta_rad=float(np.linalg.norm(x[3:5])),root_yaw_delta_rad=float(abs(x[5])))
            limits.update(root_rotation_rad=float(np.hypot(.05,yaw)),root_tilt_delta_rad=.05,root_yaw_delta_rad=yaw)
            for key in ('root_tilt_delta_rad','root_yaw_delta_rad'):maxima[key]=max(maxima.get(key,0.),vals[key])
        bad={key:vals[key] for key,limit in limits.items() if vals[key]>limit}
        if bad or collisions or nearest<.04:failures.append(dict(time_s=float(t),violations=bad,collisions=collisions,right_clearance_m=nearest,closest_pair=pair))
        traces.append(np.r_[t,sample['progress'],sample['position'],sample['velocity'],sample['acceleration']])
        if progress is not None and index%200==0:progress(dict(index=index,time_s=float(t),failures=len(failures),right_clearance_capped_m=nearest))
    bounds=path.derivative_bounds();delta=angles[-1]-angles[0];speed=args.aperture_speed_limit_rad_s;acceleration=args.aperture_acceleration_limit_rad_s2
    envelope=dict(maximum_joint_speed_rad_s=float(np.max(bounds['first'][6:])*speed/delta),maximum_joint_acceleration_rad_s2=float(np.max(bounds['second'][6:]*(speed/delta)**2+bounds['first'][6:]*acceleration/delta)),maximum_root_speed_m_s=float(bounds['root_translation_first_norm']*speed/delta),maximum_root_rotvec_speed_rad_s=float(bounds['root_rotvec_first_norm']*speed/delta),aperture_speed_limit_rad_s=speed,aperture_acceleration_limit_rad_s2=acceleration,method='Exact cubic-segment derivative extrema and conservative chain-rule acceleration bound')
    envelope['passed']=envelope['maximum_joint_speed_rad_s']<=1.2 and envelope['maximum_joint_acceleration_rad_s2']<=3. and envelope['maximum_root_speed_m_s']<=.02 and envelope['maximum_root_rotvec_speed_rad_s']<=.03
    if not envelope['passed']:failures.append(dict(measured_aperture_reference_envelope=envelope))
    receipt=dict(schema=AUDIT_SCHEMA,passed=not failures,geometric_admission=not failures,
        scope='Dense static geometry and target-rate checks only; no load, tracking, full opening or passage qualification.',
        reference_phase_envelope=envelope,actual_leaf_lag_rad=args.actual_leaf_lag_rad,
        lag_start_angle_rad=args.lag_start_angle_rad,
        lag_scope='One declared static lag slice, not a guaranteed physical lag envelope; exact initial pose retained.',
        duration_s=args.duration_s,samples=args.samples,exact_initial_state=True,
        optimizer_initial_coordinate_adjustment_removed=optimizer_initial_change,
        limits=limits,maxima=maxima,minimum_all_rh_scene_clearance_capped_m=minimum_clearance,
        minimum_elbow_scene_clearance_capped_m=minimum_elbow_clearance,required_elbow_clearance_m=.003,
        elbow_shapes=len(elbows),hand_shapes=len(hand),scene_shapes=len(scene),exact_distance_pairs=exact_pairs,
        target_aperture_rad=float(angles[-1]),passage_aperture_target_rad=1.57,
        target_reaches_passage_aperture=bool(angles[-1]>=1.57),failures=failures)
    plan=dict(schema=PLAN_SCHEMA,initial_qpos=initial.tolist(),initial_qvel=report['initial_qvel'],
        initial_leaf_velocity_rad_s=float(report['initial_qvel'][m.jnt_dofadr[m.joint('leaf_hinge').id]]),
        initial_leaf_angle_rad=float(angles[0]),final_leaf_angle_rad=float(angles[-1]),
        initial_episode_time_s=report['source_time_s'],root_qpos_address=int(rq),joint_names=names,
        joint_qpos_addresses=qa.tolist(),progress=((angles-angles[0])/(angles[-1]-angles[0])).tolist(),
        coordinates=coords.tolist(),duration_s=args.duration_s,geometric_admission=not failures,
        scope='Detached source-bound geometry artifact; not consumed by native or live Isaac panel constructors.')
    return receipt,plan,np.asarray(traces)

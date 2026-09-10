#!/usr/bin/env python3
"""Screen a new panel path from an exact attained snapshot, without physics.

Both foot frames and the cleared RH pose are planning objectives only. This
script owns a fresh unstepped model and never receives an active plant handle.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


def pose_constraint_barrier(hands,feet,foot_weight):
    """Prioritize an80% interior margin of unchanged dense pose tolerances."""
    hands=np.asarray(hands,float);feet=np.asarray(feet,float)
    if hands.shape!=(12,) or feet.shape!=(12,) or not np.isfinite(np.r_[hands,feet,foot_weight]).all() or not 10<=foot_weight<=100:
        raise ValueError('Complete finite weighted pose residuals required')
    errors=[]
    for residual,weight in ((hands,10.),(feet,foot_weight)):
        for i in (0,6):
            errors.extend((np.linalg.norm(residual[i:i+3])/100/.0001,np.linalg.norm(residual[i+3:i+6])/weight/.001))
    return 10*np.maximum(np.asarray(errors)-.8,0.)**2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--at',type=float,default=69.8)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--height-drop-m',type=float,default=0.)
    p.add_argument('--radius-shift-m',type=float,default=-.04)
    p.add_argument('--root-extent-m',type=float,default=.05)
    p.add_argument('--root-rotation-rad',type=float,default=.15)
    p.add_argument('--root-rotation-norm-rad',type=float)
    p.add_argument('--right-hand-relaxed-orientation',action='store_true',help='Released root-following hand may rotate within0.35rad; contact and joint gates remain unchanged')
    p.add_argument('--right-hand-frame',choices=('world','root'),default='world')
    p.add_argument('--right-hand-retreat-m',type=float,default=0.)
    p.add_argument('--warm-start-screen',type=Path,help='Numerical guesses only from the same exact source state and aperture grid')
    p.add_argument('--pose-tolerance-barrier',action='store_true',help='Prioritize interior hand/foot pose feasibility over posture regularization')
    p.add_argument('--balance-envelope-barrier',action='store_true',help='Fit inside existing3cm root and1.5cm COM audit bounds; does not change audit limits')
    p.add_argument('--joint-margin-rad',type=float,default=.001,help='Interior joint margin for fitting, to reserve interpolation clearance')
    p.add_argument('--foot-orientation-weight',type=float,default=10.,help='Declared geometric objective weight; dense foot tolerances stay unchanged')
    p.add_argument('--root-yaw-target-rad',type=float,help='Smooth prescribed upright yaw preference with continuity regularization and3cm root bound')
    p.add_argument('--root-yaw-extent-rad',type=float,help='Explicit upright pivot: allow yaw separately while retaining a0.05rad roll/pitch increment bound')
    p.add_argument('--maximum-torso-tilt-deg',type=float,help='Optional absolute upright planning bound, independent of the initial root orientation')
    p.add_argument('--target-aperture-rad',type=float,default=1.2)
    p.add_argument('--nodes',type=int,default=61)
    p.add_argument('--solver-method',choices=('least-squares','constrained'),default='least-squares')
    p.add_argument('--solver-scaling',choices=('unit','jac'),default='unit',help='Trust-region scaling diagnostic; acceptance tolerances unchanged')
    p.add_argument('--flatten-over-rad',type=float,default=.2)
    p.add_argument('--keep-elbow-in-front',action='store_true',help='Conservative original elbow-mesh clearance from the moving panel plane')
    p.add_argument('--elbow-clearance-m',type=float,default=.003,help='Interior elbow-to-panel planning margin; exact dense scene clearance remains mandatory')
    p.add_argument('--elbow-clearance-ramp-rad',type=float,default=0.,help='Smoothly increase elbow margin from4mm over a declared aperture interval')
    p.add_argument('--elbow-barrier-weight',type=float,default=1000.,help='Soft planning weight; never substitutes for dense acceptance')
    p.add_argument('--palm-twist-rad',type=float,default=0.,help='Smooth palm rotation about the panel normal; preserves collision vertex depths')
    p.add_argument('--flatten-palm',action='store_true',help='Rotate the actual palm face toward the panel over0.2rad while preserving its collision support plane')
    p.add_argument('--admit-exact-soft-limit-start',action='store_true',help='Retain only the measured initial solver-limit excursion, then smoothly regain the 1mm/rad numeric joint margin within 0.1rad aperture')
    a=p.parse_args()
    if a.elbow_clearance_ramp_rad and not .05<=a.elbow_clearance_ramp_rad<=.3:raise ValueError('Elbow margin ramp must be0 or0.05..0.3rad')
    if not np.isfinite(a.palm_twist_rad) or abs(a.palm_twist_rad)>.6:raise ValueError('Bounded palm twist required')
    if not .001<=a.joint_margin_rad<=.01:raise ValueError('Joint fit margin must be1..10mrad')
    if not -.04<=a.radius_shift_m<=.04 or not .1<=a.flatten_over_rad<=.6 or not 0<=a.height_drop_m<=.15 or not 0<=a.root_extent_m<=.08 or not 0<=a.root_rotation_rad<=.2 or a.nodes<21:
        raise ValueError('Require bounded declared geometry settings')
    if a.root_rotation_norm_rad is not None and not .01<=a.root_rotation_norm_rad<=.05:raise ValueError('Rotation norm must remain inside the original0.05rad screen')
    if a.maximum_torso_tilt_deg is not None and not 0<a.maximum_torso_tilt_deg<=12:raise ValueError('Require a positive torso bound within the physical safety limit')
    if a.root_yaw_extent_rad is not None and (not .05<=a.root_yaw_extent_rad<=.3 or a.root_rotation_norm_rad is not None):raise ValueError('Upright yaw profile requires .05..0.3rad yaw and no isotropic rotation-norm override')
    if a.root_yaw_target_rad is not None and (a.root_yaw_extent_rad is None or not abs(a.root_yaw_target_rad)<a.root_yaw_extent_rad):raise ValueError('Yaw target must lie inside its declared extent')
    if not 10<=a.foot_orientation_weight<=100:raise ValueError('Foot orientation weight must be10..100')
    if not 0<=a.right_hand_retreat_m<=.12 or (a.right_hand_frame=='world' and a.right_hand_retreat_m):raise ValueError('Right-hand retreat requires a root-relative goal within12cm')
    if a.right_hand_relaxed_orientation and a.right_hand_frame!='root':raise ValueError('Released orientation freedom requires root-following hand goal')
    if not .003<=a.elbow_clearance_m<=.05 or not 1000<=a.elbow_barrier_weight<=100000:raise ValueError('Require bounded positive elbow margin and weight')
    run=a.source_run.resolve();config=json.loads((run/'manifest.json').read_text())['configuration']
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from doorbench.dexterous.environment import DexterousDoorEnv
    from doorbench.dexterous.palm_panel_geometry import original_palm_vertices,flatten_palm_goal
    robot=Path(config['robot']);sim=DexterousDoorEnv(config['door'],robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d
    manifest=json.loads((run/'raw-transitions/manifest.json').read_text())
    if not manifest['complete']:raise ValueError('Require a complete measured source')
    chunk=next(c for c in manifest['chunks'] if c['interval_start_s']-1e-8<=a.at<c['interval_end_s']-1e-8)
    path=run/'raw-transitions'/chunk['file']
    if hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=chunk['sha256']:raise ValueError('Changed actual source')
    with np.load(path,allow_pickle=False) as raw:
        i=int(np.argmin(abs(raw['interval_start_s']-a.at)))
        base=raw['qpos_before'][i].copy();velocity=raw['qvel_before'][i].copy();time=float(raw['interval_start_s'][i])
        start,end=raw['body_offsets'][i:i+2];ids=raw['body_ids'][start:end]
        expected_p=raw['body_positions_world_m'][start:end];expected_r=raw['body_rotations_world'][start:end]
    d.qpos[:]=base;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    error=max(float(np.max(abs(d.xpos[ids]-expected_p))),float(np.max(abs(d.xmat[ids].reshape(-1,3,3)-expected_r))))
    if error>1e-9:raise ValueError('Destination FK differs from measured source; requalify geometry')
    rh,lh=[m.site('robot/'+s+'_palm_touch').id for s in ('rh','lh')]
    root_joint=m.joint('robot/free_base').id;rq=m.jnt_qposadr[root_joint]
    rp=base[rq:rq+3].copy();rr=Rotation.from_quat([*base[rq+4:rq+7],base[rq+3]])
    robot_body=m.jnt_bodyid[root_joint];com=d.subtree_com[robot_body].copy()
    leaf=m.body('leaf').id;leafj=m.joint('leaf_hinge').id;leafq=m.jnt_qposadr[leafj]
    leaf_p=d.xpos[leaf].copy();leaf_r=d.xmat[leaf].reshape(3,3).copy()
    local_p=leaf_r.T@(d.site_xpos[lh]-leaf_p);local_r=leaf_r.T@d.site_xmat[lh].reshape(3,3)
    palm_vertices=original_palm_vertices(m,d,lh) if a.flatten_palm else None
    elbow=m.body('robot/left_elbow_link').id;elbow_vertices=[];slab_vertices=[]
    if a.keep_elbow_in_front:
        for g in range(m.ngeom):
            if not m.geom_contype[g]:continue
            if m.geom_bodyid[g]==elbow:
                if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Original elbow mesh required')
                mesh=m.geom_dataid[g];start=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh]
                world=m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                elbow_vertices.extend((world-d.xpos[elbow])@d.xmat[elbow].reshape(3,3))
            elif m.geom(g).name.startswith('leaf_slab'):
                if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX:raise ValueError('Original slab box required')
                import itertools
                corners=np.asarray(list(itertools.product((-1.,1.),repeat=3)))*m.geom_size[g]
                world=corners@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                slab_vertices.extend((world-leaf_p)@leaf_r)
        elbow_vertices=np.asarray(elbow_vertices);slab_vertices=np.asarray(slab_vertices)
        if not len(elbow_vertices) or not len(slab_vertices):raise ValueError('Missing collision surfaces')
        elbow_side=float(np.sign((d.xpos[elbow]-leaf_p)@leaf_r[:,1]))
        slab_front=float(np.max(slab_vertices[:,1]*elbow_side))
    right_p=d.site_xpos[rh].copy();right_r=d.site_xmat[rh].reshape(3,3).copy()
    from doorbench.dexterous.released_hand_goal import released_hand_goal
    outward=np.sign((right_p-leaf_p)@leaf_r[:,1])*leaf_r[:,1]
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    feet_p=d.xpos[feet].copy();feet_r=d.xmat[feet].reshape(2,3,3).copy()
    names=[side+'_'+n for side in ('left','right') for n in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]+['torso']
    names += [side+'_'+n for side in ('right','left') for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]
    names += ['rh_WRJ2','rh_WRJ1','lh_WRJ2','lh_WRJ1']
    js=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[js];initial=base[qa].copy()
    joint_margin=max(a.joint_margin_rad,.005 if a.root_yaw_target_rad is not None else .001)
    low=np.r_[np.full(3,-max(1e-12,a.root_extent_m)),np.full(3,-max(1e-12,a.root_rotation_rad)),m.jnt_range[js,0]+joint_margin]
    high=np.r_[np.full(3,max(1e-12,a.root_extent_m)),np.full(3,max(1e-12,a.root_rotation_rad)),m.jnt_range[js,1]-joint_margin]
    if a.root_yaw_extent_rad is not None:low[5]=-a.root_yaw_extent_rad;high[5]=a.root_yaw_extent_rad
    previous=np.r_[np.zeros(6),initial];rows=[]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'screen-source.py').write_bytes(Path(__file__).read_bytes())
    warm=None;warm_hash=None
    if a.warm_start_screen is not None:
        data=a.warm_start_screen.read_bytes();seed=json.loads(data);warm_hash=hashlib.sha256(data).hexdigest()
        if seed['source_chunk_sha256']!=chunk['sha256'] or not np.array_equal(seed['initial_qpos'],base) or seed['names']!=names or len(seed['rows'])!=a.nodes:raise ValueError('Warm start must use the exact source state, joint order and grid')
        angles=np.linspace(base[leafq],a.target_aperture_rad,a.nodes)
        if not np.allclose([r['leaf_angle_rad'] for r in seed['rows']],angles,atol=1e-12,rtol=0):raise ValueError('Warm-start aperture grid differs')
        warm=np.array([np.r_[r['root_delta'],[r['joint_targets'][n] for n in names]] for r in seed['rows']])
        if not np.isfinite(warm).all():raise ValueError('Finite warm-start guesses required')
        (a.output/'warm-start-source.json').write_bytes(data)
    (a.output/'palm-panel-geometry-source.py').write_bytes((Path(__file__).resolve().parents[2]/'doorbench/dexterous/palm_panel_geometry.py').read_bytes())
    for angle in np.linspace(base[leafq],a.target_aperture_rad,a.nodes):
        if a.admit_exact_soft_limit_start:
            progress=float(np.clip((angle-base[leafq])/.1,0,1));ramp=progress**3*(10+progress*(-15+6*progress))
            low[6:]=(1-ramp)*np.minimum(m.jnt_range[js,0]+joint_margin,initial-1e-8)+ramp*(m.jnt_range[js,0]+joint_margin)
            high[6:]=(1-ramp)*np.maximum(m.jnt_range[js,1]-joint_margin,initial+1e-8)+ramp*(m.jnt_range[js,1]-joint_margin)
        state=base.copy();state[leafq]=angle;d.qpos[:]=state;mujoco.mj_kinematics(m,d)
        lr=d.xmat[leaf].reshape(3,3).copy();lp=d.xpos[leaf].copy()
        u=float(np.clip((angle-base[leafq])/.35,0,1));blend=u**3*(10+u*(-15+6*u))
        local=local_p.copy();local[0]+=a.radius_shift_m*blend;local[2]-=a.height_drop_m*blend
        goal_rotation=local_r
        if a.flatten_palm:
            fu=float(np.clip((angle-base[leafq])/a.flatten_over_rad,0,1));fb=fu**3*(10+fu*(-15+6*fu))
            local,goal_rotation,_=flatten_palm_goal(local,local_r,palm_vertices,fb)
        if a.palm_twist_rad:
            from doorbench.dexterous.palm_panel_geometry import twist_palm_goal
            goal_rotation=twist_palm_goal(goal_rotation,a.palm_twist_rad*blend)
        goal_p=lp+lr@local;goal_r=lr@goal_rotation
        phase=float(np.clip((angle-base[leafq])/(a.target_aperture_rad-base[leafq]),0,1));phase=phase**3*(10+phase*(-15+6*phase))
        elbow_margin=a.elbow_clearance_m
        if a.elbow_clearance_ramp_rad:
            fraction=float(np.clip((angle-base[leafq])/a.elbow_clearance_ramp_rad,0,1))
            fraction=fraction**3*(10+fraction*(-15+6*fraction))
            start_margin=min(.004,a.elbow_clearance_m)
            elbow_margin=start_margin+(a.elbow_clearance_m-start_margin)*fraction
        anchor=previous.copy()
        def evaluate(x):
            d.qpos[:]=state;d.qpos[rq:rq+3]=rp+x[:3]
            quat=(Rotation.from_rotvec(x[3:6])*rr).as_quat();d.qpos[rq+3:rq+7]=np.r_[quat[3],quat[:3]]
            d.qpos[qa]=x[6:];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            right_goal_p,right_goal_r=released_hand_goal(right_p,right_r,rp,x[:6],outward,phase,frame=a.right_hand_frame,retreat_m=a.right_hand_retreat_m)
            hands=np.r_[100*(d.site_xpos[lh]-goal_p),10*Rotation.from_matrix(goal_r@d.site_xmat[lh].reshape(3,3).T).as_rotvec(),
                        100*(d.site_xpos[rh]-right_goal_p),(.1 if a.right_hand_relaxed_orientation else 10)*Rotation.from_matrix(right_goal_r@d.site_xmat[rh].reshape(3,3).T).as_rotvec()]
            foot=np.concatenate([np.r_[100*(d.xpos[b]-feet_p[j]),a.foot_orientation_weight*Rotation.from_matrix(feet_r[j]@d.xmat[b].reshape(3,3).T).as_rotvec()] for j,b in enumerate(feet)])
            up=d.xmat[m.body('robot/torso_link').id].reshape(3,3)[:,2]
            upright=0. if a.maximum_torso_tilt_deg is None else 1000.*max(0.,np.arccos(np.clip(up[2],-1,1))-np.radians(max(0.,a.maximum_torso_tilt_deg-.01)))
            elbow_barrier=0.
            if a.keep_elbow_in_front:
                vertices=elbow_vertices@d.xmat[elbow].reshape(3,3).T+d.xpos[elbow]
                gap=float(np.min((vertices-lp)@lr[:,1]*elbow_side))-slab_front
                elbow_barrier=a.elbow_barrier_weight*max(0.,elbow_margin-gap)
            yaw_tilt_barrier=0. if a.root_yaw_extent_rad is None else 1000.*max(0.,np.linalg.norm(x[3:5])-.0499)
            pivot=[] if a.root_yaw_target_rad is None else np.r_[5.*(x[5]-a.root_yaw_target_rad*phase),1000.*max(0.,np.linalg.norm(x[:3])-.0299),.2*(x-anchor)]
            pose_barrier=pose_constraint_barrier(hands,foot,a.foot_orientation_weight) if a.pose_tolerance_barrier else []
            balance_barrier=[] if not a.balance_envelope_barrier else [10000.*max(0.,np.linalg.norm(x[:3])-.029),10000.*max(0.,np.linalg.norm(d.subtree_com[robot_body,:2]-com[:2])-.014)]
            if a.right_hand_relaxed_orientation:
                if a.pose_tolerance_barrier:pose_barrier[3]=0.
                pose_barrier=np.r_[pose_barrier,1000.*max(0.,np.linalg.norm(hands[9:12])/.1-.30)]
            return np.r_[hands,foot,pose_barrier,balance_barrier,upright,elbow_barrier,yaw_tilt_barrier,pivot,5*(d.subtree_com[robot_body,:2]-com[:2]),.015*(x[6:]-initial),.05*x[:6],0. if a.root_rotation_norm_rad is None else 1000.*max(0.,np.linalg.norm(x[3:6])-(a.root_rotation_norm_rad-.0001))]
        guess=np.clip(previous if warm is None else warm[len(rows)],low,high)
        if a.solver_method=='constrained':
            def constraints(x):
                residual=evaluate(x)
                # Interior planning bounds; the independent dense audit still
                # enforces its original 0.1mm/1mrad tolerances and scene gaps.
                values=[]
                for offset,rotation_weight,rotation_limit in [(0,10.,.0009),(6,.1 if a.right_hand_relaxed_orientation else 10.,.30 if a.right_hand_relaxed_orientation else .0009),(12,a.foot_orientation_weight,.0009),(18,a.foot_orientation_weight,.0009)]:
                    values.extend([1.-float(np.sum((residual[offset:offset+3]/.008)**2)),1.-float(np.sum((residual[offset+3:offset+6]/(rotation_weight*rotation_limit))**2))])
                values.extend([1.-float(np.sum((x[:3]/.029)**2)),1.-float(np.sum(((d.subtree_com[robot_body,:2]-com[:2])/.014)**2))])
                if a.maximum_torso_tilt_deg is not None:
                    up=d.xmat[m.body('robot/torso_link').id].reshape(3,3)[2,2]
                    values.append((up-np.cos(np.radians(a.maximum_torso_tilt_deg-.01)))/.01)
                if a.keep_elbow_in_front:
                    vertices=elbow_vertices@d.xmat[elbow].reshape(3,3).T+d.xpos[elbow]
                    gap=float(np.min((vertices-lp)@lr[:,1]*elbow_side))-slab_front
                    values.append((gap-elbow_margin)/.01)
                if a.root_yaw_extent_rad is not None:values.append(1.-float(np.sum((x[3:5]/.0495)**2)))
                elif a.root_rotation_norm_rad is not None:values.append(1.-float(np.sum((x[3:6]/(a.root_rotation_norm_rad-.0001))**2)))
                return np.asarray(values)
            def preference(x):
                return float(np.sum((x-anchor)**2)+(.1*(x[5]-a.root_yaw_target_rad*phase)**2 if a.root_yaw_target_rad is not None else 0.))
            fit=minimize(preference,guess,method='SLSQP',bounds=list(zip(low,high)),constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':800,'ftol':1e-10})
            fit.cost=float(fit.fun);fit.optimality=None
            fit.minimum_constraint=float(np.min(constraints(fit.x)))
        else:
            fit=least_squares(evaluate,guess,bounds=(low,high),x_scale='jac' if a.solver_scaling=='jac' else 1.,max_nfev=800,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        previous=fit.x.copy();res=evaluate(previous);mujoco.mj_collision(m,d)
        collisions=[]
        for c in d.contact[:d.ncon]:
            bodies=[m.body(m.geom_bodyid[g]).name for g in c.geom]
            foot_contact='floor' in [m.geom(int(g)).name for g in c.geom] and any(b.endswith('_ankle_link') for b in bodies)
            if not foot_contact and any(b.startswith('robot/') for b in bodies) and c.dist<-.003:
                collisions.append(dict(bodies=bodies,depth_m=-float(c.dist)))
        up=d.xmat[m.body('robot/torso_link').id].reshape(3,3)[:,2]
        row=dict(leaf_angle_rad=float(angle),qpos=d.qpos.tolist(),root_delta=fit.x[:6].tolist(),joint_targets=dict(zip(names,fit.x[6:].tolist())),
            left_position_error_m=float(np.linalg.norm(res[:3])/100),left_rotation_error_rad=float(np.linalg.norm(res[3:6])/10),
            right_position_error_m=float(np.linalg.norm(res[6:9])/100),right_rotation_error_rad=float(np.linalg.norm(res[9:12])/(.1 if a.right_hand_relaxed_orientation else 10)),
            maximum_foot_position_error_m=max(float(np.linalg.norm(res[12+6*j:15+6*j])/100) for j in range(2)),
            maximum_foot_rotation_error_rad=max(float(np.linalg.norm(res[15+6*j:18+6*j])/a.foot_orientation_weight) for j in range(2)),
            torso_tilt_deg=float(np.degrees(np.arccos(np.clip(up[2],-1,1)))),com_displacement_xy_m=(d.subtree_com[robot_body,:2]-com[:2]).tolist(),
            forbidden_collisions=collisions,nfev=int(fit.nfev),solver_status=int(fit.status),solver_optimality=None if fit.optimality is None else float(fit.optimality),solver_cost=float(fit.cost),solver_message=str(fit.message))
        if a.solver_method=='constrained':
            row['minimum_normalized_constraint']=fit.minimum_constraint
            row['solver_feasible']=fit.minimum_constraint>=-1e-7
        if a.keep_elbow_in_front:
            vertices=elbow_vertices@d.xmat[elbow].reshape(3,3).T+d.xpos[elbow]
            row['elbow_panel_plane_gap_m']=float(np.min((vertices-lp)@lr[:,1]*elbow_side))-slab_front
            row['required_elbow_planning_margin_m']=elbow_margin
            row['elbow_planning_margin_satisfied']=row['elbow_panel_plane_gap_m']>=elbow_margin
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('qpos','joint_targets')}),flush=True)
    summary=dict(scope='Fresh unstepped attained-state panel workspace screen. No physical or loaded-palm qualification; target-rate resampling and dense collision audit are still required.',
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},warm_start_screen_sha256=warm_hash,source_code_sha256=hashlib.file_digest((a.output/'screen-source.py').open('rb'),'sha256').hexdigest(),source_time_s=time,source_chunk_sha256=chunk['sha256'],
        maximum_actual_fk_error=error,initial_qpos=base.tolist(),initial_qvel=velocity.tolist(),names=names,rows=rows,
        maximum_palm_position_error_m=max(max(r['left_position_error_m'],r['right_position_error_m']) for r in rows),
        maximum_foot_position_error_m=max(r['maximum_foot_position_error_m'] for r in rows),maximum_torso_tilt_deg=max(r['torso_tilt_deg'] for r in rows),
        forbidden_collision_samples=sum(bool(r['forbidden_collisions']) for r in rows))
    (a.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n');sim.close()


if __name__=='__main__':main()

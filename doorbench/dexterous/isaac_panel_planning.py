"""Detached actual-Isaac panel geometry planning, with no live stage authority.

The numerical objectives/constraints are extracted from the native panel
screen. Source loading is distinct: a newly qualified released PhysX endpoint.
No native archive, old coordinates, plant interface or physics step is used.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from .landed_left_planner import LandedLeftScene
from .palm_panel_geometry import original_palm_vertices, flatten_palm_goal
from .qualified_isaac_grasp import digest

SCHEMA = 'doorbench.isaac-panel-candidate.v1'
SOURCE_SCHEMA = 'doorbench.isaac-released-panel-source.v1'
JOINT_NAMES = ([side+'_'+n for side in ('left','right') for n in
    ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]+['torso']+
    [side+'_'+n for side in ('right','left') for n in
    ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+
    ['rh_WRJ2','rh_WRJ1','lh_WRJ2','lh_WRJ1'])
DEFAULT_PREFERENCES = dict(nodes=61,max_nfev=800,height_drop_m=0.,radius_shift_m=-.04,
    root_extent_m=.03,root_rotation_rad=.05,left_palm_rotation_fit_rad=.0009,
    joint_margin_rad=.001,foot_orientation_weight=10.,flatten_over_rad=.2,
    elbow_clearance_m=.003,elbow_clearance_ramp_rad=0.,elbow_barrier_weight=1000.,
    palm_twist_rad=0.)
PREFERENCE_BOUNDS = dict(nodes=(21,401),max_nfev=(1,800),height_drop_m=(0.,.15),
    radius_shift_m=(-.04,.04),root_extent_m=(0.,.03),root_rotation_rad=(.01,.05),
    left_palm_rotation_fit_rad=(.0005,.0009),joint_margin_rad=(.001,.01),
    foot_orientation_weight=(10.,100.),flatten_over_rad=(.1,.6),
    elbow_clearance_m=(.003,.05),elbow_clearance_ramp_rad=(0.,.3),
    elbow_barrier_weight=(1000.,100000.),palm_twist_rad=(-.6,.6))


def numeric_panel_preferences(document):
    """Old documents contribute bounded scalar numbers only, never proof/state."""
    if type(document) is not dict:raise ValueError('Explicit numeric preference object required')
    values=document.get('configuration',document)
    if type(values) is not dict:raise ValueError('Explicit preference configuration required')
    result={k:values.get(k,v) for k,v in DEFAULT_PREFERENCES.items()}
    for key,value in result.items():
        low,high=PREFERENCE_BOUNDS[key]
        if (type(value) not in (int,float) or not np.isfinite(value) or not low<=value<=high
                or (key in ('nodes','max_nfev') and type(value) is not int)):
            raise ValueError('Bounded finite numeric panel preference required: '+key)
    if 0<result['elbow_clearance_ramp_rad']<.05:
        raise ValueError('Elbow margin ramp must be zero or at least0.05rad')
    return result


def panel_source_paths():
    """Actual planner/helper sources; evidence helper registries add their own."""
    directory=Path(__file__).resolve().parent
    return tuple(directory/name for name in ('isaac_panel_planning.py','isaac_panel_source.py',
        'landed_left_planner.py','palm_panel_geometry.py','released_hand_goal.py',
        'screened_panel_path.py'))


@dataclass(frozen=True)
class IsaacPanelPlanningContext:
    _admission_json: str

    @property
    def admission(self):return json.loads(self._admission_json)

    @property
    def qpos(self):return np.asarray(self.admission['initial_qpos'],float)

    @property
    def qvel(self):return np.asarray(self.admission['initial_qvel'],float)

    @property
    def sha256(self):return hashlib.sha256(self._admission_json.encode()).hexdigest()

    def verify_inputs(self):
        for name,expected in self.admission['input_sha256'].items():
            if digest(name)!=expected:raise ValueError('Actual released Isaac source changed: '+name)

    def scene(self):
        self.verify_inputs();source=self.admission
        scene=LandedLeftScene(source['robot_path'],source['door_xml_path'])
        if self.qpos.shape!=(scene.m.nq,) or self.qvel.shape!=(scene.m.nv,):
            raise ValueError('Complete admitted scene state required')
        scene.d.qpos[:]=self.qpos
        mujoco.mj_kinematics(scene.m,scene.d);mujoco.mj_comPos(scene.m,scene.d)
        return scene


def admit_isaac_panel_context(source,*,robot,door_xml,door_usd):
    from .isaac_panel_source import admit_isaac_panel_source
    admission=admit_isaac_panel_source(source,robot=robot,door_xml=door_xml,door_usd=door_usd)
    if (admission.get('schema')!=SOURCE_SCHEMA or admission.get('source_engine')!='isaac-physx'
            or admission['source_qualification'].get('passed') is not True
            or admission['coordinate_admission'].get('passed') is not True
            or admission.get('authorized_stages')!=0):
        raise ValueError('Qualified actual released Isaac source required')
    encoded=json.dumps(admission,sort_keys=True,separators=(',',':'),allow_nan=False)
    context=IsaacPanelPlanningContext(encoded)
    if not np.isfinite(np.r_[context.qpos,context.qvel,admission['source_time_s']]).all() or admission['source_time_s']<=0:
        raise ValueError('Finite actual endpoint and source epoch required')
    context.verify_inputs();return context


def _options(preferences,*,target_aperture_rad,solver_method,flatten_palm):
    values=numeric_panel_preferences(preferences)
    if (type(target_aperture_rad) not in (int,float) or not np.isfinite(target_aperture_rad)
            or not .075<target_aperture_rad<=1.75):raise ValueError('Explicit bounded aperture target required')
    if solver_method not in ('constrained','least-squares') or type(flatten_palm) is not bool:
        raise ValueError('Explicit known solver and palm geometry mode required')
    return SimpleNamespace(**values,target_aperture_rad=target_aperture_rad,
        solver_method=solver_method,flatten_palm=flatten_palm,solver_scaling='unit',
        root_rotation_norm_rad=.05,right_hand_relaxed_orientation=False,right_hand_frame='world',
        right_hand_retreat_m=0.,right_hand_scene_margin_m=.04 if solver_method=='constrained' else 0.,
        root_yaw_extent_rad=None,root_yaw_target_rad=None,maximum_torso_tilt_deg=4.,
        keep_elbow_in_front=True,pose_tolerance_barrier=True,balance_envelope_barrier=True,
        admit_exact_soft_limit_start=True)


def generate_panel_candidate(context,preferences=None,*,target_aperture_rad=1.62,
                             solver_method='constrained',flatten_palm=False,progress=None):
    if not isinstance(context,IsaacPanelPlanningContext):raise ValueError('Admitted actual panel context required')
    a=_options({} if preferences is None else preferences,target_aperture_rad=target_aperture_rad,
               solver_method=solver_method,flatten_palm=flatten_palm)
    context.verify_inputs();source=context.admission
    hashes={**source['input_sha256'],**{str(p):digest(p) for p in panel_source_paths()}}
    scene=context.scene();m,d=scene.m,scene.d;base=context.qpos;velocity=context.qvel
    leafq=int(m.joint('leaf_hinge').qposadr[0])
    if not float(base[leafq])<a.target_aperture_rad<=float(m.jnt_range[m.joint('leaf_hinge').id,1]):
        raise ValueError('Target must advance the actual aperture within original joint range')
    rows,names=_solve_panel(m,d,base,a,progress)
    if names!=JOINT_NAMES or not np.array_equal(rows[0]['qpos'],base):
        raise ValueError('Panel candidate changed the exact admitted initial state or joint order')
    context.verify_inputs()
    if any(digest(p)!=value for p,value in hashes.items()):raise ValueError('Panel input changed during solve')
    return dict(schema=SCHEMA,source_engine='isaac-physx',source_run=source['source_run'],
        source_admission=source,source_context_sha256=context.sha256,source_time_s=source['source_time_s'],
        input_sha256=hashes,configuration=vars(a),initial_qpos=base.tolist(),initial_qvel=velocity.tolist(),
        names=names,rows=rows,exact_initial_state=True,preferences_are_numeric_only=True,
        passage_aperture_target_rad=1.57,target_aperture_rad=a.target_aperture_rad,
        target_reaches_passage_aperture=bool(a.target_aperture_rad>=1.57),
        segment_scope='One prospective source-bound segment; its endpoint is not an admitted physical source',
        authorized_stages=0,physics_steps=0,source_sample_playback=0,active_state_writes=0,
        geometric_admission=False,physical_admission=False,runtime_route_exported=False,
        scope='Unstepped geometry candidate only. Independent dense audit and actual physical qualification required.')


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


def clearance_candidate_pairs(hand_positions,hand_radii,scene_positions,scene_radii,planes,cap):
    """Conservative sphere bounds, with signed bounds for infinite planes.

    A point below a plane remains a candidate. The plane's infinite enclosing
    sphere must not force every distant hand shape into an exact query.
    """
    delta=hand_positions[:,None]-scene_positions
    lower=np.linalg.norm(delta,axis=2)-hand_radii[:,None]-scene_radii
    for column,normal in planes:
        lower[:,column]=delta[:,column]@normal-hand_radii
    return np.argwhere(lower<=cap)


def _solve_panel(m,d,base,a,progress=None):
    """Original native numerical objectives, with exact source knot retained."""
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
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
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
    if a.right_hand_scene_margin_m:
        hand_geoms=np.array([g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')])
        scene_geoms=np.array([g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and not m.body(m.geom_bodyid[g]).name.startswith('robot/')])
        radii=np.zeros(m.ngeom)
        for g in np.r_[hand_geoms,scene_geoms]:
            kind=int(m.geom_type[g]);size=m.geom_size[g]
            if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh=m.geom_dataid[g];start=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh]
                radii[g]=float(np.linalg.norm(m.mesh_vert[start:start+count],axis=1).max())
            elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):radii[g]=float(np.linalg.norm(size))
            elif kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):radii[g]=float(size[0])
            elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):radii[g]=float(size[0]+size[1])
            elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):radii[g]=float(np.hypot(size[0],size[1]))
            elif kind==int(mujoco.mjtGeom.mjGEOM_PLANE):radii[g]=np.inf
            else:raise ValueError('Unsupported original collider for exact hand constraint')
        def hand_clearance():
            cap=a.right_hand_scene_margin_m+.005
            planes=[(j,d.geom_xmat[g].reshape(3,3)[:,2]) for j,g in enumerate(scene_geoms) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE]
            pairs=clearance_candidate_pairs(d.geom_xpos[hand_geoms],radii[hand_geoms],d.geom_xpos[scene_geoms],radii[scene_geoms],planes,cap)
            return min([cap]+[float(mujoco.mj_geomDistance(m,d,int(hand_geoms[i]),int(scene_geoms[j]),cap,None)) for i,j in pairs])
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
    warm=None
    for angle in np.linspace(base[leafq],a.target_aperture_rad,a.nodes):
        if a.admit_exact_soft_limit_start:
            fit_progress=float(np.clip((angle-base[leafq])/.1,0,1));ramp=fit_progress**3*(10+fit_progress*(-15+6*fit_progress))
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
        if angle==base[leafq]:
            fit=SimpleNamespace(x=np.r_[np.zeros(6),initial],nfev=0,status=0,optimality=0.,cost=0.,message='Exact admitted source; not optimized',minimum_constraint=0.)
        elif a.solver_method=='constrained':
            def constraints(x):
                residual=evaluate(x)
                # Interior planning bounds; the independent dense audit still
                # enforces its original 0.1mm/1mrad tolerances and scene gaps.
                values=[]
                for offset,rotation_weight,rotation_limit in [(0,10.,a.left_palm_rotation_fit_rad),(6,.1 if a.right_hand_relaxed_orientation else 10.,.30 if a.right_hand_relaxed_orientation else .0009),(12,a.foot_orientation_weight,.0009),(18,a.foot_orientation_weight,.0009)]:
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
                if a.right_hand_scene_margin_m:values.append((hand_clearance()-a.right_hand_scene_margin_m)/.01)
                return np.asarray(values)
            def preference(x):
                return float(np.sum((x-anchor)**2)+(.1*(x[5]-a.root_yaw_target_rad*phase)**2 if a.root_yaw_target_rad is not None else 0.))
            fit=minimize(preference,guess,method='SLSQP',bounds=list(zip(low,high)),constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':a.max_nfev,'ftol':1e-10})
            fit.cost=float(fit.fun);fit.optimality=None
            fit.minimum_constraint=float(np.min(constraints(fit.x)))
        else:
            fit=least_squares(evaluate,guess,bounds=(low,high),x_scale='jac' if a.solver_scaling=='jac' else 1.,max_nfev=a.max_nfev,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        if not np.isfinite(fit.x).all():raise ValueError('Nonfinite panel solve cannot be exported')
        previous=fit.x.copy();res=evaluate(previous)
        if angle==base[leafq]:
            d.qpos[:]=base;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        mujoco.mj_collision(m,d)
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
        if a.right_hand_scene_margin_m:row['right_hand_scene_clearance_capped_m']=hand_clearance()
        if a.keep_elbow_in_front:
            vertices=elbow_vertices@d.xmat[elbow].reshape(3,3).T+d.xpos[elbow]
            row['elbow_panel_plane_gap_m']=float(np.min((vertices-lp)@lr[:,1]*elbow_side))-slab_front
            row['required_elbow_planning_margin_m']=elbow_margin
            row['elbow_planning_margin_satisfied']=row['elbow_panel_plane_gap_m']>=elbow_margin
        rows.append(row)
        if progress is not None:progress({k:v for k,v in row.items() if k not in ('qpos','joint_targets')})
    return rows,names

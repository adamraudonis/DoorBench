"""CPU constrained IK on a private native-door + original adult MuJoCo scene.

Target schema (world metres and WXYZ quaternions)::
    {"pelvis": {"pos": [x,y,z], "quat_wxyz": [w,x,y,z]},
     "left_foot": {"pos": [...], "quat_wxyz": [...], "contact": True},
     "right_foot": {"pos": [...], "quat_wxyz": [...], "contact": False},
     "right_hand": {"pos": [...], "grip_geoms": ["specific_handle_geom"]},
     "com": {"pos": [x,y], "cost": 0.2}}

Feet refer to ANKLE frames, normally z=.055 above a level floor. Their rigid
foot box center is at ankle+(0,.04,-.0275); its sole is .055m below the
ankle when upright. Contact=True requires
all six pose coordinates and creates a hard QP equality; swing feet are soft.
Other keys: chest, head, left_hand, left_elbow, right_elbow. Pose position_cost/orientation_cost and
position_tolerance_m/orientation_tolerance_rad may be supplied explicitly.

Every solve uses a fixed native door pose. A trajectory planner must validate
the moving door between states. Geometric feasibility is not dynamic balance.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import numpy as np
import mujoco
import mink
from .humanoid import JOINTS
from .rig import combine_with_door, TARGET_SITES, DIMENSIONS


@dataclass
class IKResult:
    qpos: np.ndarray
    actor_qpos: np.ndarray
    native_qpos: np.ndarray
    joint_positions: np.ndarray
    foot_poses: dict
    com: np.ndarray
    diagnostics: dict
    success: bool


class _ActorComTask(mink.Task):
    """Mink's built-in COM task uses body1; our door occupies that subtree."""
    def __init__(self,body_id,target,cost):
        super().__init__(np.full(2,float(cost)),gain=1.,lm_damping=1e-5)
        self.body_id=body_id;self.target=np.asarray(target,float)
    def compute_error(self,configuration):return configuration.data.subtree_com[self.body_id,:2]-self.target
    def compute_jacobian(self,configuration):
        jac=np.zeros((3,configuration.nv));mujoco.mj_jacSubtreeCom(configuration.model,configuration.data,jac,self.body_id)
        return jac[:2]


class _PositionTask(mink.Task):
    """World-space point target, independent of the site's orientation.

    A zero rotation *cost* on an SE(3) log task does not make its translation
    error independent of rotation. Hands/elbows without a requested quaternion
    need a true Cartesian point residual and its native positional Jacobian.
    """
    def __init__(self,site_id,target,cost):
        super().__init__(np.full(3,float(cost)),gain=1.,lm_damping=1e-4)
        self.site_id=site_id;self.target=np.asarray(target,float)
    def compute_error(self,configuration):
        return configuration.data.site_xpos[self.site_id]-self.target
    def compute_jacobian(self,configuration):
        jac=np.zeros((3,configuration.nv))
        mujoco.mj_jacSite(configuration.model,configuration.data,jac,None,self.site_id)
        return jac


class _StepVelocityLimit(mink.limits.Limit):
    """Bound TOTAL displacement from the input, not each nonlinear iteration."""
    def __init__(self,model,start,indices,maximum,dt):
        self.model=model;self.start=start.copy();self.indices=indices;self.bound=maximum*dt
        eye=np.eye(model.nv)[indices];self.G=np.concatenate([eye,-eye])
    def compute_qp_inequalities(self,configuration,dt):
        displacement=np.empty(self.model.nv)
        mujoco.mj_differentiatePos(self.model,displacement,1.,self.start,configuration.q)
        used=displacement[self.indices]
        return mink.limits.Constraint(self.G,np.r_[self.bound-used,self.bound+used])


class _ActorConfigurationLimit(mink.ConfigurationLimit):
    def __init__(self,model,actor_dofs):
        super().__init__(model,gain=1.)
        self.actor_dofs=actor_dofs
    def compute_qp_inequalities(self,configuration,dt):
        result=super().compute_qp_inequalities(configuration,dt)
        if result.inactive:return result
        rows=np.any(result.G[:,self.actor_dofs]!=0,axis=1)
        return mink.limits.Constraint(result.G[rows],result.h[rows])


class _ClearanceLimit(mink.CollisionAvoidanceLimit):
    """Distance rows in displacement units, with bounded separation recovery.

    Mink 1.3's collision bound divides by dt, while its QP variable is delta-q.
    Evaluate that bound over unit time so a 60 Hz solve cannot spend 60 times the
    available gap. Below the desired gap Mink otherwise clamps the bound to zero;
    a small separating step avoids sticking at the nonlinear acceptance boundary.
    This is an optimization buffer, not a relaxation of final geometry checks.
    """
    def compute_qp_inequalities(self,configuration,dt):
        result=super().compute_qp_inequalities(configuration,1.)
        for i in np.flatnonzero(result.h==0.):
            a,b=self.geom_id_pairs[i]
            distance=float(mujoco.mj_geomDistance(self.model,configuration.data,a,b,
                self.minimum_distance_from_collisions,None))
            deficit=self.minimum_distance_from_collisions-distance
            if deficit>0:result.h[i]=-min(self.gain*deficit,.00005)
        return result


class _FootGroundLimit(mink.limits.Limit):
    """Exact box-corner half spaces for support surfaces, including zero gap.

    Coincident closest-point witnesses have no usable direction. In particular,
    normalizing that zero vector in a generic distance constraint can constrain
    horizontal foot motion instead of the floor normal. Plane normals come from
    the authored geometry here. Finite boxes use their top only when the whole
    foot footprint is inside it; edges/other shapes retain native distance rows.
    The nonlinear acceptance check always uses the original finite geometries.
    """
    def __init__(self,model,pairs):
        self.model=model;self.pairs=pairs;self.fixed_feet=set()
        self.corners={a:np.array(list(itertools.product((-1.,1.),repeat=3)))*model.geom_size[a] for a,_ in pairs}
        self.fallback={pair:_ClearanceLimit(model,[([pair[0]],[pair[1]])],gain=.85,
            minimum_distance_from_collisions=0.,collision_detection_distance=.06) for pair in pairs}

    def compute_qp_inequalities(self,configuration,dt):
        model=self.model;data=configuration.data;rows=[];bounds=[]
        jac_foot=np.zeros((3,model.nv));jac_floor=np.zeros((3,model.nv))
        for foot,floor in self.pairs:
            # A six-dimensional foot equality already fixes every corner. Mixing
            # its SE(3) log linearization with redundant corner rows can make a
            # return to exact ground contact infeasible after a nonlinear step.
            # Such feet are still subject to the identical exact geometry check.
            if foot in self.fixed_feet:continue
            rotation=data.geom_xmat[floor].reshape(3,3)
            points=self.corners[foot]@data.geom_xmat[foot].reshape(3,3).T+data.geom_xpos[foot]
            local=(points-data.geom_xpos[floor])@rotation
            kind=model.geom_type[floor]
            plane=kind==mujoco.mjtGeom.mjGEOM_PLANE
            box_top=kind==mujoco.mjtGeom.mjGEOM_BOX and np.all(np.abs(local[:,:2])<=model.geom_size[floor,:2]+1e-9)
            if not (plane or box_top):
                fallback=self.fallback[(foot,floor)].compute_qp_inequalities(configuration,dt)
                if not fallback.inactive:rows.extend(fallback.G);bounds.extend(fallback.h)
                continue
            heights=local[:,2]-(0. if plane else model.geom_size[floor,2])
            normal=rotation[:,2]
            for point,height in zip(points,heights):
                if height>.06:continue
                mujoco.mj_jac(model,data,jac_foot,None,point,int(model.geom_bodyid[foot]))
                mujoco.mj_jac(model,data,jac_floor,None,point,int(model.geom_bodyid[floor]))
                rows.append(-normal@(jac_foot-jac_floor))
                # Mink's QP variable is displacement, not velocity: no /dt.
                bounds.append(height)
        return mink.limits.Constraint(np.asarray(rows),np.asarray(bounds)) if rows else mink.limits.Constraint()


class DoorHumanoidIK:
    """Constrained local solver; `success` requires independently checked residuals.

    `previous_q` accepts full combined qpos or actor-only qpos in
    `actor_qpos_indices` order. Native coordinates always come from the latest
    `set_door_state` call, never from the warm start. No benchmark object is used.
    """
    def __init__(self,door_dir,*,native_qpos=None,root_pos=(0.,-1.,.94),root_yaw=0.,
                 clearance=.003,max_iterations=12):
        if not math.isfinite(clearance) or clearance<0:raise ValueError('clearance must be finite and nonnegative')
        if type(max_iterations) is not int or max_iterations<1:raise ValueError('max_iterations must be a positive integer')
        self.rig=combine_with_door(door_dir,root_pos=root_pos,root_yaw=root_yaw,native_qpos=native_qpos)
        self.model=self.rig.model;self.native_nq=self.rig.native_model.nq
        self.native_qpos_indices=self.rig.native_qpos_indices;self.actor_qpos_indices=self.rig.actor_qpos_indices
        self.actor_dof_indices=self.rig.actor_dof_indices
        self.home_qpos=self.rig.home_qpos.copy();self.qpos=self.home_qpos.copy()
        self._native_qpos=self.qpos[self.native_qpos_indices].copy()
        self.configuration=mink.Configuration(self.model,self.qpos)
        self.data=self.configuration.data
        self.clearance=float(clearance);self.max_iterations=max_iterations
        self._site_ids={key:int(self.model.site('actor_site_'+value).id) for key,value in TARGET_SITES.items()}
        self._landmark_ids=[int(self.model.site('actor_site_'+name).id) for name in JOINTS]
        self.actor_geom_ids=[i for i in range(self.model.ngeom) if self.model.geom(i).name.startswith('actor_geom_')]
        self.scene_geom_ids=[i for i in range(self.model.ngeom) if i not in self.actor_geom_ids and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])]
        ir=json.loads((Path(door_dir)/'model.json').read_text())
        floor_names={g['name'] for b in ir['bodies'] for g in b['geoms'] if g.get('semantic')=='floor'}
        self.floor_geom_ids={i for i in self.scene_geom_ids if self.model.geom(i).name in floor_names}
        self._foot_geom_ids={int(self.model.geom('actor_geom_foot_'+s).id) for s in ('l','r')}
        self._hand_geom_ids={'left_hand':int(self.model.geom('actor_geom_hand_l').id),'right_hand':int(self.model.geom('actor_geom_hand_r').id)}
        self._base_pairs=[];self._ground_pairs=[]
        for a,b in itertools.combinations(self.actor_geom_ids,2):
            ba,bb=int(self.model.geom_bodyid[a]),int(self.model.geom_bodyid[b])
            if ba==bb or self.model.body_parentid[ba]==bb or self.model.body_parentid[bb]==ba:continue
            self._base_pairs.append((a,b))
        for a,b in itertools.product(self.actor_geom_ids,self.scene_geom_ids):
            (self._ground_pairs if a in self._foot_geom_ids and b in self.floor_geom_ids else self._base_pairs).append((a,b))
        self._pairs=self._base_pairs.copy();self._grip_exclusions=set()
        self._distance_cache={}
        self._collision_limits_key=None;self._collision_limits=[]
        self._ground_limit=_FootGroundLimit(self.model,self._ground_pairs) if self._ground_pairs else None
        self._velocity=np.full(len(self.actor_dof_indices),2.5)
        rootadr=int(self.model.joint('actor_root').dofadr[0])
        for j,dof in enumerate(self.actor_dof_indices):
            if rootadr<=dof<rootadr+3:self._velocity[j]=.8
            elif rootadr+3<=dof<rootadr+6:self._velocity[j]=1.5
        self._freeze=mink.DofFreezingTask(self.model,self.rig.native_dof_indices.tolist()) if len(self.rig.native_dof_indices) else None

    def set_door_state(self,qpos):
        q=np.asarray(qpos,float)
        if q.shape!=(self.native_nq,) or not np.isfinite(q).all():raise ValueError(f'native_qpos must have {self.native_nq} finite coordinates')
        self._native_qpos=q.copy();self.qpos[self.native_qpos_indices]=q
        self.configuration.update(self.qpos)

    def _merge(self,qpos=None):
        q=self.qpos.copy()
        if qpos is not None:
            value=np.asarray(qpos,float)
            if value.shape==q.shape:q=value.copy()
            elif value.shape==(len(self.actor_qpos_indices),):q[self.actor_qpos_indices]=value
            else:raise ValueError('previous_q must be full combined qpos or actor-only qpos')
            if not np.isfinite(q).all():raise ValueError('qpos must be finite')
        q[self.native_qpos_indices]=self._native_qpos
        adr=int(self.model.joint('actor_root').qposadr[0]);norm=np.linalg.norm(q[adr+3:adr+7])
        if abs(norm-1)>1e-4:raise ValueError('Actor free-root quaternion must have unit norm')
        return q

    def _fresh_data(self,qpos=None):
        data=mujoco.MjData(self.model);data.qpos[:]=self._merge(qpos)
        mujoco.mj_kinematics(self.model,data);mujoco.mj_comPos(self.model,data)
        return data

    def joint_positions(self,qpos=None):return self._fresh_data(qpos).site_xpos[self._landmark_ids].copy()

    def _pose(self,data,key):
        i=self._site_ids[key];quat=np.empty(4);mujoco.mju_mat2Quat(quat,data.site_xmat[i])
        return {'pos':data.site_xpos[i].copy(),'quat_wxyz':quat}

    def foot_poses(self,qpos=None):
        data=self._fresh_data(qpos);return {k:self._pose(data,k) for k in ('left_foot','right_foot')}

    def collision_geometries(self,qpos=None):
        """Actual actor geometry for an independent viewer/auditor, in world space."""
        data=self._fresh_data(qpos);rows=[]
        for i in self.actor_geom_ids:
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,data.geom_xmat[i])
            rows.append({'name':self.model.geom(i).name,'body_name':self.model.body(int(self.model.geom_bodyid[i])).name,
                         'type':mujoco.mjtGeom(self.model.geom_type[i]).name.removeprefix('mjGEOM_').lower(),
                         'size':self.model.geom_size[i].tolist(),'pos':data.geom_xpos[i].tolist(),'quat_wxyz':quat.tolist()})
        return rows

    def _minimum(self,data,pairs):
        """Exact minimum with conservative bounding-sphere pruning, no contact masks."""
        if not pairs:return math.inf,None
        key=id(pairs)
        if key not in self._distance_cache:
            indices=np.asarray(pairs,dtype=int)
            a,b=indices[:,0],indices[:,1]
            planes=(self.model.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE)|(self.model.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE)
            self._distance_cache[key]=(a,b,self.model.geom_rbound[a]+self.model.geom_rbound[b],planes)
        a,b,radii,planes=self._distance_cache[key]
        lower=np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b],axis=1)-radii
        lower[planes]=-math.inf
        best=math.inf;pair=None
        for i in np.argsort(lower,kind='stable'):
            if lower[i]>best:break
            dist=float(mujoco.mj_geomDistance(self.model,data,int(a[i]),int(b[i]),1e6,None))
            if dist<best:best=dist;pair=(self.model.geom(int(a[i])).name,self.model.geom(int(b[i])).name)
        return best,pair

    def diagnostics(self,qpos=None):
        data=self._fresh_data(qpos);q=data.qpos
        violations=[]
        for i in range(self.model.njnt):
            name=self.model.joint(i).name
            if not name.startswith('actor_') or not self.model.jnt_limited[i]:continue
            value=float(q[self.model.jnt_qposadr[i]]);lo,hi=self.model.jnt_range[i]
            amount=max(0.,lo-value,value-hi)
            if amount:violations.append({'joint':name,'violation_rad':float(amount)})
        distance,pair=self._minimum(data,self._pairs)
        ground,ground_pair=self._minimum(data,self._ground_pairs)
        return {'min_noncontact_distance_m':None if math.isinf(distance) else distance,'closest_pair':pair,
                'min_foot_ground_distance_m':None if math.isinf(ground) else ground,'closest_foot_ground_pair':ground_pair,
                'clearance_m':self.clearance,'joint_limit_violation_rad':max([v['violation_rad'] for v in violations],default=0.),
                'joint_limit_violations':violations,'allowed_grip_pairs':[[self.model.geom(a).name,self.model.geom(b).name] for a,b in sorted(self._grip_exclusions)],
                'com':data.subtree_com[self.rig.actor_body_id].tolist(),
                'collision_scope':'All declared actor pairs except adjacent links; collidable native scene; exact hand/geom grip exceptions; feet may touch floor.'}

    def _collision_violations(self,qpos):
        """Exact individual violations, used only while restoring an invalid start."""
        data=self._fresh_data(qpos);violations={}
        for pairs,bound in ((self._pairs,self.clearance-1e-5),(self._ground_pairs,-1e-5)):
            for a,b in pairs:
                distance=float(mujoco.mj_geomDistance(self.model,data,a,b,max(.06,bound+.02),None))
                if distance<bound:violations[(a,b)]=bound-distance
        return violations

    @staticmethod
    def _restoration_improves(before,after):
        # No newly invalid pair, no worsening existing violation, and genuine
        # improvement. A better global minimum alone could hide another contact.
        return bool(before) and set(after)<=set(before) and all(
            value<=before[pair]+1e-10 for pair,value in after.items()) and (
            sum(after.values())<sum(before.values())-1e-10)

    def _targets(self,targets):
        if not isinstance(targets,dict) or set(targets)-set(TARGET_SITES)-{'com'}:raise ValueError('Unknown target name')
        tasks=[];constraints=[];clean={};excluded=set()
        if self._freeze is not None:constraints.append(self._freeze)
        posture=mink.PostureTask(self.model,cost=.025,lm_damping=1e-4)
        posture.set_target(self.home_qpos);tasks.append(posture)
        for key,raw in targets.items():
            if not isinstance(raw,dict):raise ValueError(f'{key} target must be a dictionary')
            if key=='com':
                value=np.asarray(raw.get('pos'),float);cost=float(raw.get('cost',.2))
                if value.shape not in ((2,),(3,)) or not np.isfinite(value).all() or not math.isfinite(cost) or cost<0:raise ValueError('com requires finite XY position and nonnegative cost')
                tasks.append(_ActorComTask(self.rig.actor_body_id,value[:2],cost));continue
            pos=np.asarray(raw.get('pos'),float)
            if pos.shape!=(3,) or not np.isfinite(pos).all():raise ValueError(f'{key}.pos must contain three finite coordinates')
            has_quat='quat_wxyz' in raw
            quat=np.asarray(raw.get('quat_wxyz',[1,0,0,0]),float)
            if quat.shape!=(4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1)>1e-5:raise ValueError(f'{key}.quat_wxyz must be a unit WXYZ quaternion')
            contact=raw.get('contact',False)
            if type(contact) is not bool or (contact and (key not in ('left_foot','right_foot') or not has_quat)):raise ValueError('Hard contact requires an explicit 6D left_foot/right_foot pose')
            pc=float(raw.get('position_cost',1.));oc=float(raw.get('orientation_cost',1. if has_quat else 0.))
            if min(pc,oc)<0 or not np.isfinite([pc,oc]).all() or pc==0:raise ValueError('Pose costs must be finite/nonnegative with positive position cost')
            if has_quat:
                task=mink.FrameTask(self._site_ids[key],'site',pc,oc,gain=1.,lm_damping=1e-4)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quat),pos))
            else:task=_PositionTask(self._site_ids[key],pos,pc)
            (constraints if contact else tasks).append(task)
            clean[key]={**raw,'pos':pos,'quat_wxyz':quat,'contact':contact,'has_quat':has_quat,'position_cost':pc,'orientation_cost':oc}
            grip=raw.get('grip_geoms',[])
            if not isinstance(grip,(list,tuple)) or (grip and key not in self._hand_geom_ids):raise ValueError('grip_geoms is a list of specific scene geometry names on a hand target only')
            for name in grip:
                if not isinstance(name,str):raise ValueError('grip_geoms must contain geometry names')
                try:g=int(self.model.geom(name).id)
                except KeyError as e:raise ValueError(f'Unknown grip geometry {name}') from e
                if g not in self.scene_geom_ids or g in self.floor_geom_ids:raise ValueError('A grip exception must name a collidable, non-floor native scene geom')
                excluded.add((self._hand_geom_ids[key],g))
        self._grip_exclusions=excluded
        cache_key=tuple(sorted(excluded))
        if self._collision_limits_key!=cache_key:
            self._pairs=[p for p in self._base_pairs if p not in excluded]
            self._distance_cache.clear()
            self._collision_limits=[]
            if self._pairs:self._collision_limits.append(_ClearanceLimit(self.model,[([a],[b]) for a,b in self._pairs],gain=.85,minimum_distance_from_collisions=self.clearance+.00005,collision_detection_distance=max(.06,self.clearance+.02)))
            if self._ground_limit is not None:self._collision_limits.append(self._ground_limit)
            self._collision_limits_key=cache_key
        if self._ground_limit is not None:
            self._ground_limit.fixed_feet={int(self.model.geom('actor_geom_foot_'+side).id)
                for key,side in [('left_foot','l'),('right_foot','r')] if clean.get(key,{}).get('contact')}
        return tasks,constraints,clean

    def solve(self,targets,dt,previous_q=None):
        if not math.isfinite(dt) or not 0<dt<=.5:raise ValueError('dt must be finite and in (0,.5] seconds')
        start=self._merge(previous_q);self.configuration.update(start)
        tasks,constraints,clean=self._targets(targets)
        step_limit=_StepVelocityLimit(self.model,start,self.actor_dof_indices,self._velocity,dt)
        limits=[_ActorConfigurationLimit(self.model,self.actor_dof_indices),step_limit,*self._collision_limits]
        solver_error=None;iterations=0;restoration_steps=0;current_check=self.diagnostics(start)
        for iterations in range(1,self.max_iterations+1):
            old=self.configuration.q.copy()
            old_check=current_check
            old_general=old_check['min_noncontact_distance_m'];old_ground=old_check['min_foot_ground_distance_m']
            restoring=(old_general is not None and old_general<self.clearance-1e-5) or (old_ground is not None and old_ground<-1e-5)
            old_violations=self._collision_violations(old) if restoring else {}
            try:velocity=mink.solve_ik(self.configuration,tasks,dt,'daqp',damping=1e-5,safety_break=False,limits=limits,constraints=constraints)
            except (mink.exceptions.NoSolutionFound,mink.exceptions.NotWithinConfigurationLimits) as exc:
                solver_error=str(exc);break
            candidate=old.copy();mujoco.mj_integratePos(self.model,candidate,velocity,dt)
            # Recheck nonlinear collision/joint geometry after the linearized QP.
            accepted=False
            for scale in (1.,.5,.25,.125,.0625,.03125):
                candidate=old.copy();mujoco.mj_integratePos(self.model,candidate,velocity,dt*scale)
                candidate[self.native_qpos_indices]=self._native_qpos
                check=self.diagnostics(candidate)
                general=check['min_noncontact_distance_m'];ground=check['min_foot_ground_distance_m']
                actual_velocity=np.empty(self.model.nv);mujoco.mj_differentiatePos(self.model,actual_velocity,dt,start,candidate)
                velocity_ok=np.all(np.abs(actual_velocity[self.actor_dof_indices])<=self._velocity*1.00001)
                collision_ok=(general is None or general>=self.clearance-1e-5) and (ground is None or ground>=-1e-5)
                if not collision_ok and restoring:
                    collision_ok=self._restoration_improves(old_violations,self._collision_violations(candidate))
                if velocity_ok and check['joint_limit_violation_rad']<=1e-7 and collision_ok:
                    accepted=True
                    if restoring:restoration_steps+=1
                    break
            if not accepted:solver_error='Nonlinear collision/limit check rejected step';break
            current_check=check
            self.configuration.update(candidate)
            displacement=np.empty(self.model.nv);mujoco.mj_differentiatePos(self.model,displacement,1.,old,candidate)
            if np.max(np.abs(displacement))<1e-7:break
        self.qpos=self.configuration.q.copy();data=self._fresh_data(self.qpos)
        check=self.diagnostics(self.qpos);residuals={};converged=True;foot_ok=True
        for key,target in clean.items():
            pose=self._pose(data,key);pe=float(np.linalg.norm(pose['pos']-target['pos']))
            dot=min(1.,abs(float(np.dot(pose['quat_wxyz'],target['quat_wxyz']))));oe=2*math.acos(dot)
            residuals[key]={'position_m':pe,'orientation_rad':oe if target['has_quat'] else None,'hard_contact':target['contact']}
            if target['contact']:foot_ok &= pe<=.001 and oe<=math.radians(.5)
            else:
                pt=float(target.get('position_tolerance_m',.01));ot=float(target.get('orientation_tolerance_rad',math.radians(5)))
                converged &= pe<=pt and (not target['has_quat'] or target['orientation_cost']==0 or oe<=ot)
        velocities=np.empty(self.model.nv);mujoco.mj_differentiatePos(self.model,velocities,dt,start,self.qpos)
        ratio=float(np.max(np.abs(velocities[self.actor_dof_indices])/self._velocity))
        general=check['min_noncontact_distance_m'];ground=check['min_foot_ground_distance_m']
        feasible=foot_ok and check['joint_limit_violation_rad']<=1e-7 and (general is None or general>=self.clearance-1e-5) and (ground is None or ground>=-1e-5) and ratio<=1.0001
        check.update(iterations=iterations,restoration_steps=restoration_steps,solver_error=solver_error,target_residuals=residuals,converged=bool(converged),kinematically_feasible=bool(feasible),
                     max_velocity_limit_ratio=ratio,actor_velocity=velocities[self.actor_dof_indices].tolist(),
                     velocity_limits=self._velocity.tolist(),velocity_limit_scope='Per-axis free-root translation .8m/s, rotation1.5rad/s; hinge2.5rad/s; total step across all IK iterations.',
                     dimensions=DIMENSIONS.copy())
        return IKResult(self.qpos.copy(),self.qpos[self.actor_qpos_indices].copy(),self._native_qpos.copy(),data.site_xpos[self._landmark_ids].copy(),
                        {k:self._pose(data,k) for k in ('left_foot','right_foot')},data.subtree_com[self.rig.actor_body_id].copy(),check,bool(feasible and converged and solver_error is None))

"""Opt-in, independently admitted measured-frame withdrawal motor references."""
import json
from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .coupled_release_geometry import CoupledReleaseGeometry,sha
from .coupled_release_motion import CoupledReferenceMotion
from .grasp_verification import shadow_surface_qualified
from .landed_left_audit import static_pose_check


def _joint_projection_velocity_interval(position,velocity,dt,joint_range,*,name):
    """Original scalar motion set intersected with position/braking bounds.

    The .02 rad allowance is exactly the existing static joint audit, not an
    expanded actuator or authored joint limit. For next outward speed v,
    v**2/(2*a)+v*dt/2+a*dt**2/8 bounds discrete stopping distance. Requiring
    that bound to fit in the remaining distance starts braking before an
    acceleration-limited reference becomes impossible to stop inside it.
    """
    values=np.asarray([position,velocity,dt,*joint_range],float)
    if values.shape!=(5,) or not np.isfinite(values).all() or not 0<dt<=.00200001 or joint_range[0]>=joint_range[1]:
        raise ValueError('Finite original projected joint interval required: '+name)
    lower_position=float(joint_range[0]-.02);upper_position=float(joint_range[1]+.02)
    if position<lower_position-1e-12 or position>upper_position+1e-12:
        raise ValueError(f'Projected joint prior position outside original audit interval: {name}; q={position!r}; allowed=[{lower_position!r},{upper_position!r}]')
    distance=np.maximum(0.,[position-lower_position,upper_position-position])
    stopping=np.maximum(0.,np.sqrt(6.*distance)-1.5*dt)
    outward=np.minimum(distance/dt,stopping)
    lower=max(-1.2,velocity-3.*dt,-float(outward[0]))
    upper=min(1.2,velocity+3.*dt,float(outward[1]))
    if lower>upper:
        if lower-upper>1e-12:
            raise ValueError(f'Projected joint has no original position/braking/motion intersection: {name}; q={position!r}; v={velocity!r}; dt={dt!r}; allowed=[{lower_position!r},{upper_position!r}]; velocity_intersection=[{lower!r},{upper!r}]; outward_caps={outward.tolist()!r}')
        # Only resolve arithmetic noise within the motion gate's existing
        # 1e-12 velocity tolerance. No state or authored limit is changed.
        lower=upper=(lower+upper)/2.
    return lower,upper


def validate_admission(config):
    envelope=Path(config['coupled_envelope_path']);audit_path=Path(config['coupled_audit_path'])
    if sha(envelope)!=config['coupled_envelope_sha256'] or sha(audit_path)!=config['coupled_audit_sha256']:raise ValueError('Coupled withdrawal evidence bytes changed')
    audit=json.loads(audit_path.read_text())
    if audit.get('schema')!='doorbench.coupled-release-envelope-audit.v1' or audit.get('passed') is not True or audit.get('coarse_diagnostic') is not False or audit.get('samples',0)<10000 or not audit.get('exact_initial_state') or audit.get('physics_steps')!=0:
        raise ValueError('Independent dense coupled envelope admission required')
    if audit['input_sha256'].get(str(envelope.resolve()))!=sha(envelope):raise ValueError('Coupled audit belongs to another envelope')
    import doorbench.dexterous.coupled_release_geometry as geometry
    if audit['input_sha256'].get(str(Path(geometry.__file__).resolve()))!=sha(geometry.__file__):raise ValueError('Coupled audit must bind the consumed geometry evaluator')
    limits=dict(left_position_error_m=.001,left_rotation_error_rad=.01,right_position_error_m=.001,right_rotation_error_rad=.01,foot_position_error_m=.001,foot_rotation_error_rad=.01,torso_tilt_deg=4.,root_translation_m=.03,com_xy_displacement_m=.015,palm_panel_gap_change_m=.001)
    if audit.get('limits')!=limits:raise ValueError('Original coupled geometric limits required')
    for name,digest in audit['input_sha256'].items():
        if sha(name)!=digest:raise ValueError('Coupled admitted input changed: '+name)
    return envelope,audit


class CoupledReleaseReference:
    def __init__(self,config,source_admission,initial_qpos,duration):
        if config.get('capture_returned_motor_command') is not True:raise ValueError('Coupled withdrawal requires actual predecessor motor capture')
        envelope,self.audit=validate_admission(config);self.geometry=CoupledReleaseGeometry(envelope);g=self.geometry;m,d=g.m,g.d
        if g.plan['source_admission']!=source_admission or not np.array_equal(g.initial,initial_qpos) or g.plan['duration_s']!=duration:raise ValueError('Coupled envelope must bind exact withdrawal source and duration')
        domain=self.audit['geometry_domain']
        if (domain.get('admitted_leaf_upper_nodes')!=g.plan.get('admitted_leaf_upper_nodes')
                or domain['operator_rad']!=g.plan['operator_envelope_rad']
                or domain.get('elapsed_s')!=[0,g.plan['duration_s']]
                or domain.get('leaf_rad')!=[float(g.angles[0]),float(g.angles[-1])]
                or domain.get('latch_m')!=[-.001,.001]):raise ValueError('Coupled audit and live domain differ')
        self.names=[m.joint(j).name[6:] for j in range(m.njnt) if m.joint(j).name.startswith('robot/') and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE]
        self.qa=np.array([m.joint('robot/'+n).qposadr[0] for n in self.names]);self.motion=CoupledReferenceMotion(np.r_[np.zeros(6),g.initial[self.qa]])
        self.projection=config.get('coupled_motion_projection')
        if self.projection not in (None,'fixed-poses-v1'):raise ValueError('Unknown coupled reference projection')
        self.body_columns=np.r_[np.arange(6),[6+self.names.index(n) for n in g.names]].astype(int)
        self.feet=[m.body('robot/'+s+'_ankle_link').id for s in ('left','right')];self.torso=m.body('robot/torso_link').id;self.robotbody=m.jnt_bodyid[m.joint('robot/free_base').id]
        self.lever=m.geom('leaf_handle_lever_col_n').id;active=[v for v in range(m.ngeom) if m.geom_contype[v] or m.geom_conaffinity[v]]
        self.right=[v for v in active if m.body(m.geom_bodyid[v]).name.startswith('robot/rh_')];self.handle=[v for v in active if m.geom_bodyid[v]==g.handle]
        self.palm=[v for v in active if m.geom_bodyid[v]==m.site_bodyid[g.lh]];self.panel=[v for v in active if m.geom_bodyid[v]==g.leaf]
        d.qpos[:]=g.initial;mujoco.mj_kinematics(m,d)
        self.initial_gap=min(float(mujoco.mj_geomDistance(m,d,a,b,.1,None)) for a in self.palm for b in self.panel)
        self.info={};self.accepted=0;self.limited=0;self.failure_snapshot=None

    def _install_reference(self,value):
        """Install only in the separate, unstepped evaluator model."""
        g=self.geometry;d=g.d
        d.qpos[g.rq:g.rq+3]=g.initial[g.rq:g.rq+3]+value[:3]
        rotation=Rotation.from_rotvec(value[3:6])*g.initial_rotation
        quat=rotation.as_quat();d.qpos[g.rq+3:g.rq+7]=quat[[3,0,1,2]]
        d.qpos[self.qa]=value[6:]
        mujoco.mj_kinematics(g.m,d);mujoco.mj_comPos(g.m,d)
        return rotation

    def _pose_residual_jacobian(self,value,result):
        """World-frame palm/foot errors and derivatives in reference coordinates.

        MuJoCo free-joint angular velocity is not a world-relative rotvec
        derivative. Differentiate the three quaternion coordinates explicitly
        instead of assuming those conventions coincide.
        """
        g=self.geometry;m,d=g.m,g.d;self._install_reference(value)
        mapping=np.zeros((m.nv,len(self.body_columns)))
        root_dof=int(m.joint('robot/free_base').dofadr[0])
        mapping[root_dof:root_dof+3,:3]=np.eye(3)
        base=d.qpos.copy();epsilon=1e-6
        for i in range(3):
            minus=base.copy();plus=base.copy();delta=np.zeros(3);delta[i]=epsilon
            for q,sign in ((minus,-1.),(plus,1.)):
                quat=(Rotation.from_rotvec(value[3:6]+sign*delta)*g.initial_rotation).as_quat()
                q[g.rq+3:g.rq+7]=quat[[3,0,1,2]]
            velocity=np.empty(m.nv)
            mujoco.mj_differentiatePos(m,velocity,2.*epsilon,minus,plus)
            mapping[:,3+i]=velocity
        for col,name in enumerate(g.names,6):mapping[m.joint('robot/'+name).dofadr[0],col]=1.
        tasks=[('site',g.lh,result['left_palm_position'],result['left_palm_rotation']),
               ('site',g.rh,result['right_palm_position'],result['right_palm_rotation'])]
        tasks += [('body',body,np.array(g.c['initial_feet_positions'][i]),np.array(g.c['initial_feet_rotations'][i])) for i,body in enumerate(self.feet)]
        errors=[];jacobians=[];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
        for kind,body,position,rotation in tasks:
            if kind=='site':
                actualp=d.site_xpos[body];actualr=d.site_xmat[body].reshape(3,3)
                mujoco.mj_jacSite(m,d,jp,jr,body)
            else:
                actualp=d.xpos[body];actualr=d.xmat[body].reshape(3,3)
                mujoco.mj_jacBody(m,d,jp,jr,body)
            angle=Rotation.from_matrix(rotation@actualr.T).as_rotvec()
            theta=float(np.linalg.norm(angle));x,y,z=angle
            skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            factor=1./12.+theta*theta/720. if theta<1e-4 else (1.-.5*theta/np.tan(.5*theta))/(theta*theta)
            inverse_right=np.eye(3)+.5*skew+factor*(skew@skew)
            errors.extend(((position-actualp)/.001,angle/.01))
            jacobians.extend((-jp@mapping/.001,-inverse_right@jr@mapping/.01))
        return np.concatenate(errors),np.concatenate(jacobians)

    def _project_reference(self,candidate,previous,previous_velocity,dt,result):
        """Fit all four poses together within the original per-step motion set.

        This small convex subproblem changes only body references, never fingers
        or mechanism state. It minimizes linearized pose error plus a small
        preference for the independently limited proposal. Exact nonlinear
        geometry/contact checks remain mandatory after this operation.
        """
        columns=self.body_columns
        scale=np.r_[np.full(3,.02),np.full(3,.03),np.full(len(columns)-6,1.2)]*dt
        origin=previous[columns];preferred=(candidate[columns]-origin)/scale
        lower=np.full(len(columns),-1.);upper=np.ones(len(columns))
        lower[6:]=np.maximum(-1.2,previous_velocity[columns[6:]]-3.*dt)/1.2
        upper[6:]=np.minimum(1.2,previous_velocity[columns[6:]]+3.*dt)/1.2
        bounded_joints=0
        for i,name in enumerate(self.geometry.names,6):
            joint=self.geometry.m.joint('robot/'+name)
            if not self.geometry.m.jnt_limited[joint.id]:continue
            low,high=_joint_projection_velocity_interval(origin[i],previous_velocity[columns[i]],dt,
                self.geometry.m.jnt_range[joint.id],name=joint.name)
            lower[i]=low/1.2;upper[i]=high/1.2;bounded_joints+=1
        preferred=np.clip(preferred,lower,upper);choice=preferred.copy()
        def balls(x):return np.array([1.-x[:3]@x[:3],1.-x[3:6]@x[3:6]])
        def ball_jac(x):
            jac=np.zeros((2,len(x)));jac[0,:3]=-2.*x[:3];jac[1,3:6]=-2.*x[3:6]
            return jac
        iterations=0;statuses=[]
        # Two differential solves remove linearization error without relaxing
        # motion bounds; both stay relative to the same previous accepted step.
        for _ in range(2):
            trial=candidate.copy();trial[columns]=origin+scale*choice
            error,jac=self._pose_residual_jacobian(trial,result);matrix=jac*scale
            center=choice.copy();regularizer=1e-5
            def objective(x):
                residual=error+matrix@(x-center);delta=x-preferred
                return 5000.*float(residual@residual+regularizer*(delta@delta))
            def gradient(x):return 10000.*(matrix.T@(error+matrix@(x-center))+regularizer*(x-preferred))
            def pose_balls(x):
                residual=(error+matrix@(x-center)).reshape(-1,3)
                return .99**2-np.sum(residual*residual,axis=1)
            def pose_jac(x):
                residual=(error+matrix@(x-center)).reshape(-1,3)
                return -2.*np.einsum('ij,ijk->ik',residual,matrix.reshape(-1,3,len(x)))
            solved=minimize(objective,choice,jac=gradient,bounds=list(zip(lower,upper)),
                constraints=[dict(type='ineq',fun=balls,jac=ball_jac),dict(type='ineq',fun=pose_balls,jac=pose_jac)],method='SLSQP',
                options=dict(ftol=1e-12,maxiter=160))
            if not np.isfinite(solved.x).all():raise ValueError('Coupled pose projection produced nonfinite coordinates')
            # Optimizer convergence is not kinematic admission. A finite
            # approximate minimum can be useful even if a line search stalls.
            # Normalize back into the *same* convex motion set, then let the
            # authoritative motion and exact nonlinear geometry gates decide.
            feasible=np.clip(solved.x,lower,upper)
            for sl in (slice(0,3),slice(3,6)):
                norm=np.linalg.norm(feasible[sl])
                if norm>1.:feasible[sl]/=norm
            statuses.append(dict(success=bool(solved.success),status=int(solved.status),message=str(solved.message)))
            iterations+=solved.nit;change=np.max(abs(feasible-choice));choice=feasible
            if change<1e-8:break
        projected=candidate.copy();projected[columns]=origin+scale*choice
        return projected,dict(method='fixed-poses-v1',iterations=int(iterations),body_coordinates=len(columns),maximum_coordinate_change=float(np.max(abs(projected-candidate))),solver_status=statuses,joint_position_braking_bounds=bounded_joints)

    def _inspect(self,result):
        g=self.geometry;m,d=g.m,g.d;c=g.c;check=static_pose_check(m,d,coordinate=1.);mujoco.mj_comPos(m,d)
        def er(a,b):return float(np.linalg.norm(Rotation.from_matrix(a@b.T).as_rotvec()))
        bad=[]
        if not check['passed']:bad.append('original collision/joint/loopback gates')
        pairs=[(g.lh,result['left_palm_position'],result['left_palm_rotation']),(g.rh,result['right_palm_position'],result['right_palm_rotation'])]
        pe=max(float(np.linalg.norm(d.site_xpos[s]-p)) for s,p,r in pairs);rotation_error=max(er(r,d.site_xmat[s].reshape(3,3)) for s,p,r in pairs)
        if pe>.001 or rotation_error>.01:bad.append('measured-frame palm pose gate')
        if max(float(np.linalg.norm(d.xpos[b]-c['initial_feet_positions'][i])) for i,b in enumerate(self.feet))>.001 or max(er(np.array(c['initial_feet_rotations'][i]),d.xmat[b].reshape(3,3)) for i,b in enumerate(self.feet))>.01:bad.append('fixed-foot pose gate')
        if np.degrees(np.arccos(np.clip(d.xmat[self.torso].reshape(3,3)[2,2],-1,1)))>4.:bad.append('upright4degree gate')
        if np.linalg.norm(d.qpos[g.rq:g.rq+3]-g.initial[g.rq:g.rq+3])>.03 or np.linalg.norm(d.subtree_com[self.robotbody,:2]-np.array(c['initial_com'][:2]))>.015:bad.append('root/COM gate')
        gap=min(float(mujoco.mj_geomDistance(m,d,a,b,.1,None)) for a in self.palm for b in self.panel)
        if abs(gap-self.initial_gap)>.001:bad.append('palm/panel gap gate')
        clear=None
        if result['release_clock_s']-.5>=g.through:
            clear=min(float(mujoco.mj_geomDistance(m,d,a,b,.2,None)) for a in self.right for b in self.handle)
            if clear<.004:bad.append('literal4mm all-handle blend gate')
        for contact in d.contact[:d.ncon]:
            if contact.dist>=0:continue
            bodies=[m.body(m.geom_bodyid[a]).name for a in contact.geom]
            if 'leaf_handle' not in bodies or not any(b.startswith('robot/rh_') for b in bodies):continue
            if self.lever not in contact.geom:bad.append('RH contact outside lever');continue
            side=0 if contact.geom[1]==self.lever else 1;b=int(m.geom_bodyid[contact.geom[side]]);name=m.body(b).name
            match=re.fullmatch(r'robot/rh_(ff|mf|rf|lf|th)(distal|middle|proximal)',name);R=d.xmat[b].reshape(3,3)
            point=R.T@(contact.pos-d.xpos[b]);normal=R.T@(contact.frame[:3]*(1 if side==0 else -1))
            axis=d.geom_xmat[self.lever].reshape(3,3)[:,2];rel=contact.pos-d.geom_xpos[self.lever];axial=float(rel@axis);radial=rel-axial*axis;alignment=float((R@normal)@(-radial/max(np.linalg.norm(radial),1e-12)))
            if not match or m.geom_size[self.lever,1]-abs(axial)<.001 or alignment<=.8 or not shadow_surface_qualified(*match.groups(),point,normal,profile='volar-phalange-v1'):bad.append('original selected RH anatomy/normal/end gate')
        if bad:raise ValueError('Coupled reference stopped before motor submission: '+', '.join(sorted(set(bad))))
        return dict(coupled_reference_palm_position_error_m=pe,coupled_reference_palm_rotation_error_rad=rotation_error,coupled_reference_all_handle_clearance_m=clear)

    def update(self,t,elapsed,angles,leaf_pose,handle_pose):
        if self.failure_snapshot is not None:raise ValueError('Coupled reference is terminal after its first failed diagnostic')
        try:return self._update(t,elapsed,angles,leaf_pose,handle_pose)
        except Exception as exc:
            self.failure_snapshot=dict(time_s=float(t),elapsed_s=float(elapsed),angles=dict(angles),error=type(exc).__name__+': '+str(exc),accepted_samples=self.accepted)
            raise

    def _update(self,t,elapsed,angles,leaf_pose,handle_pose):
        g=self.geometry;m,d=g.m,g.d
        elapsed=min(elapsed,g.plan['duration_s'])
        result=g.evaluate(elapsed,angles['leaf'],angles['operator'],angles['latch'])
        # Verify the actual poses used by the plant and the private mechanism
        # agree. Only measured angles enter the private model; none are outputs.
        for body,pose in [(g.leaf,leaf_pose),(g.handle,handle_pose)]:
            pose=np.asarray(pose,float)
            if pose.shape!=(7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]),1.,atol=1e-6):raise ValueError('Finite normalized measured mechanism pose required')
            rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
            if np.linalg.norm(pose[:3]-d.xpos[body])>1e-5 or Rotation.from_matrix(rotation@d.xmat[body].reshape(3,3).T).magnitude()>1e-5:raise ValueError('Measured mechanism frames differ from admitted model')
        desired=np.r_[result['coordinates'][:6],result['qpos'][self.qa]]
        project=None if self.projection is None else lambda *args:self._project_reference(*args,result)
        value,motion=self.motion.update(t,desired,project=project)
        rotation=self._install_reference(value)
        checked=self._inspect(result);self.accepted+=1;self.limited+=int(motion['limited'])
        self.info={**checked,'coupled_reference':True,'coupled_reference_accepted_samples':self.accepted,'coupled_reference_limited_samples':self.limited,'coupled_reference_leaf_rad':float(angles['leaf']),'coupled_reference_leaf_upper_rad':g.upper_angle(elapsed),'coupled_reference_operator_rad':float(angles['operator']),'coupled_reference_handle_follow_weight':result['right_follow_weight'],'coupled_reference_release_clock_s':result['release_clock_s'],'coupled_reference_motion':motion,'coupled_reference_plant_pose_writes':0}
        return dict(zip(self.names,value[6:])),d.qpos[g.rq:g.rq+3].copy(),rotation.as_matrix(),result['right_palm_position'],result['right_palm_rotation'],self.info.copy()

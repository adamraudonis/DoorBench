"""Privileged inverse-dynamics stance controller with finite foot support polygons.

Only robot motor targets are returned. Contact constraints here are planning
assumptions, never constraints or anchors inserted into the simulator.
"""
import numpy as np
import mujoco
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation


def validate_stance_solver_settings(settings):
    """Allow numerical iteration/penalty choices; never relax residual bounds."""
    settings={} if settings is None else dict(settings)
    if set(settings)-{'max_iter','rho'}:
        raise ValueError('Only stance solver max_iter and rho may be configured')
    if 'max_iter' in settings and (type(settings['max_iter']) is not int or settings['max_iter']<=0):
        raise ValueError('Stance solver max_iter must be a positive integer')
    if 'rho' in settings and (not np.isscalar(settings['rho']) or not np.isfinite(settings['rho']) or settings['rho']<=0):
        raise ValueError('Stance solver rho must be finite and positive')
    return settings


class StanceController:
    def __init__(self,sim,*,solver_settings=None):
        self.solver_settings=validate_stance_solver_settings(solver_settings)
        self.last_solver_metadata=None
        self.sim=sim;m,d=sim.m,sim.d
        prefix=getattr(sim,'joint_prefix','robot/')
        names=[side+'_'+joint for side in ('left','right') for joint in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]
        self.joints=[m.joint(prefix+n).id for n in names]
        self.act=np.array([m.actuator(prefix+n).id for n in names])
        self.local=np.array([list(sim.actuators).index(i) for i in self.act])
        self.v=np.r_[np.arange(sim.root_vadr,sim.root_vadr+6),m.jnt_dofadr[self.joints]]
        self.qa=m.jnt_qposadr[self.joints]
        self.feet=[m.body(prefix+side+'_ankle_link').id for side in ('left','right')]
        self.target_root=d.qpos[sim.root_qadr:sim.root_qadr+3].copy()
        yaw=np.arctan2(d.xmat[sim.pelvis].reshape(3,3)[1,0],d.xmat[sim.pelvis].reshape(3,3)[0,0])
        self.target_rotation=Rotation.from_euler('z',yaw).as_matrix()
        self.fixed_foot_rotations=None
        self.foot_positions=d.xpos[self.feet].copy()
        self.joint_target=d.qpos[self.qa].copy()
        self.support=[]
        for body in self.feet:
            points=[]
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid]!=body or not m.geom_contype[gid]:continue
                if m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Audit non-mesh foot footprint before use')
                mid=m.geom_dataid[gid];verts=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
                rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,m.geom_quat[gid])
                points.extend(verts@rotation.reshape(3,3).T+m.geom_pos[gid])
            points=np.asarray(points);sole=points[points[:,2]<points[:,2].min()+.002]
            self.support.append((sole[:,:2].min(0)+.003,sole[:,:2].max(0)-.003,-float(sole[:,2].min())))
        self.last=None

    def foot_target_rotation(self,index):
        """Historical root-coupled targets unless explicitly frozen at handoff."""
        return self.target_rotation if self.fixed_foot_rotations is None else self.fixed_foot_rotations[index]

    def freeze_foot_targets(self):
        """Preserve current targets continuously while allowing pelvis rotation.

        These are controller references, never anchors or simulator pose writes.
        Preserve an existing per-foot reference when crossing another segment.
        """
        self.fixed_foot_rotations=np.array([self.foot_target_rotation(i).copy() for i in range(2)])

    def command(self):
        s=self.sim;m,d=s.m,s.d;n=16;nt=10;nf=12;N=n+nt+nf
        M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M)
        J=[];accelerations=[]
        for foot_index,(body,pos) in enumerate(zip(self.feet,self.foot_positions)):
            jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,body)
            R=self.foot_target_rotation(foot_index);T=np.zeros((6,6));T[:3,:3]=R.T;T[3:,3:]=R.T
            jac=T@np.vstack((jp,jr));J.append(jac[:,self.v])
            orient=Rotation.from_matrix(R@d.xmat[body].reshape(3,3).T).as_rotvec()
            error=T@np.r_[pos-d.xpos[body],orient]
            accelerations.extend(100*error-20*(jac@d.qvel))
        J=np.vstack(J)
        # Subtract measured non-foot contacts, including the real hand load.
        external=np.zeros(m.nv)
        # Native constraint vector includes contacts and joint limits. Foot forces
        # are removed explicitly, leaving measured loads on the robot elsewhere.
        supplied=getattr(s,'external_generalized_force',None)
        external[:]=d.qfrc_constraint if supplied is None else supplied
        for i,c in enumerate(d.contact[:d.ncon] if supplied is None else []):
            bodies=[int(m.geom_bodyid[g]) for g in c.geom]
            if not any(b in self.feet for b in bodies):continue
            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench)
            R=c.frame.reshape(3,3).T;f=R@wrench[:3];tau=R@wrench[3:]
            for sign,b in zip((-1,1),bodies):
                if b==0:continue
                jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,c.pos,b)
                external-=sign*(jp.T@f+jr.T@tau)
        h=(d.qfrc_bias-d.qfrc_passive-external)[self.v]
        S=np.zeros((n,nt));S[6:,:]=np.eye(nt)
        eq=np.block([[M[np.ix_(self.v,self.v)],-S,-J.T],[J,np.zeros((nf,nt+nf))]])
        rhs=np.r_[-h,accelerations]
        root=d.qpos[s.root_qadr:s.root_qadr+3];R=d.xmat[s.pelvis].reshape(3,3)
        orient=Rotation.from_matrix(R.T@self.target_rotation).as_rotvec()
        desired=np.r_[60*(self.target_root-root)-15*d.qvel[s.root_vadr:s.root_vadr+3],
            80*orient-18*d.qvel[s.root_vadr+3:s.root_vadr+6],
            20*(self.joint_target-d.qpos[self.qa])-6*d.qvel[self.v[6:]]]
        weights=getattr(s,'stance_weights',np.r_[[200,200,1000,300,300,2],np.full(10,.01)])
        H=np.diag(np.r_[weights,np.full(nt,.002),np.full(nf,.00001)])
        linear=np.r_[-weights*desired,np.zeros(nt+nf)]
        limits=[];lo=[];hi=[]
        low=m.actuator_forcerange[self.act,0].copy();high=m.actuator_forcerange[self.act,1].copy()
        bias=m.actuator_biasprm[self.act,0]+m.actuator_biasprm[self.act,1]*d.actuator_length[self.act]+m.actuator_biasprm[self.act,2]*d.actuator_velocity[self.act]
        low=np.maximum(low,m.actuator_gainprm[self.act,0]*m.actuator_ctrlrange[self.act,0]+bias)
        high=np.minimum(high,m.actuator_gainprm[self.act,0]*m.actuator_ctrlrange[self.act,1]+bias)
        for k in range(nt):
            row=np.zeros(N);row[n+k]=1;limits.append(row);lo.append(low[k]);hi.append(high[k])
        for foot in range(2):
            offset=n+nt+6*foot
            row=np.zeros(N);row[offset+2]=1;limits.append(row);lo.append(0.);hi.append(1000.)
            for axis,ratio in ((0,.7),(1,.7),(5,.015)):
                for sign in (-1,1):
                    row=np.zeros(N);row[offset+axis]=sign;row[offset+2]=-ratio
                    limits.append(row);lo.append(-np.inf);hi.append(0.)
            lower_cop,upper_cop,height=self.support[foot]
            # Moments are about the ankle origin, not the ground plane.
            for moment,force,force_coef,minimum,maximum in (
                (3,1,-height,lower_cop[1],upper_cop[1]),
                (4,0,height,-upper_cop[0],-lower_cop[0])):
                for sign,limit in ((1,maximum),(-1,-minimum)):
                    row=np.zeros(N);row[offset+moment]=sign;row[offset+force]=sign*force_coef;row[offset+2]=-limit
                    limits.append(row);lo.append(-np.inf);hi.append(0.)
        A=sparse.csc_matrix(np.vstack([eq,limits]));lower=np.r_[rhs,lo];upper=np.r_[rhs,hi]
        modern=int(osqp.__version__.split('.')[0])>=1
        settings={'polishing' if modern else 'polish':False}
        settings.update(self.solver_settings);maximum_iterations=settings.pop('max_iter',4000)
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(H),q=linear,A=A,l=lower,u=upper,
             verbose=False,eps_abs=1e-4,eps_rel=1e-4,max_iter=maximum_iterations,**settings)
        if self.last is not None:solver.warm_start(x=self.last)
        result=solver.solve(raise_error=False) if modern else solver.solve()
        self.last_solver_metadata=dict(status=result.info.status,iterations=int(result.info.iter),
            primal_residual=float(getattr(result.info,'prim_res',getattr(result.info,'pri_res',np.nan))),
            dual_residual=float(getattr(result.info,'dual_res',getattr(result.info,'dua_res',np.nan))),
            eps_abs=1e-4,eps_rel=1e-4,max_iter=maximum_iterations,rho_requested=self.solver_settings.get('rho'),
            polishing=False)
        if result.info.status_val not in (1,2):return None,result.info.status
        self.last=result.x
        controls=(result.x[n:n+nt]-bias)/m.actuator_gainprm[self.act,0]
        return controls,result.info.status


class PoseStanceController(StanceController):
    """Foot-constrained IK, native joint PD, and bounded static-load feedforward."""
    def __init__(self,sim):
        super().__init__(sim)
        self.ik=mujoco.MjData(sim.m)

    def command(self):
        s=self.sim;m,d=s.m,s.d;k=self.ik;k.qpos[:]=d.qpos;k.qvel[:]=0
        k.qpos[s.root_qadr:s.root_qadr+3]=self.target_root
        q=Rotation.from_matrix(self.target_rotation).as_quat()
        k.qpos[s.root_qadr+3:s.root_qadr+7]=q[[3,0,1,2]]
        jp=np.zeros((3,m.nv));jr=jp.copy()
        def jacobians():
            rows=[];errors=[]
            for foot_index,(body,pos) in enumerate(zip(self.feet,self.foot_positions)):
                mujoco.mj_jacBody(m,k,jp,jr,body)
                rows.append(np.vstack([jp*5,jr]))
                orient=Rotation.from_matrix(self.foot_target_rotation(foot_index)@k.xmat[body].reshape(3,3).T).as_rotvec()
                errors.extend(np.r_[5*(pos-k.xpos[body]),orient])
            return np.vstack(rows),np.asarray(errors)
        for _ in range(20):
            mujoco.mj_kinematics(m,k);mujoco.mj_comPos(m,k)
            jac,error=jacobians();jac=jac[:,self.v[6:]]
            change=np.linalg.solve(jac.T@jac+.0001*np.eye(10),jac.T@error)
            k.qpos[self.qa]=np.clip(k.qpos[self.qa]+np.clip(change,-.05,.05),m.jnt_range[self.joints,0],m.jnt_range[self.joints,1])
            if np.linalg.norm(error)<1e-4:break
        mujoco.mj_forward(m,k)
        J=[]
        for foot_index,body in enumerate(self.feet):
            mujoco.mj_jacBody(m,k,jp,jr,body)
            R=self.foot_target_rotation(foot_index)
            T=np.zeros((6,6));T[:3,:3]=R.T;T[3:,3:]=R.T
            J.append(T@np.vstack([jp,jr]))
        J=np.vstack(J)
        rootJ=J[:,self.v[:6]].T
        h=k.qfrc_bias
        # Minimum-norm stationary support loads satisfy the unactuated root's
        # force/torque balance; the real engine still determines actual contacts.
        wrench=np.linalg.lstsq(rootJ,h[self.v[:6]],rcond=None)[0]
        tau=h[self.v[6:]]-J[:,self.v[6:]].T@wrench
        tau=np.clip(tau,m.actuator_forcerange[self.act,0],m.actuator_forcerange[self.act,1])
        targets=k.qpos[self.qa]+tau/m.actuator_gainprm[self.act,0]
        return targets,'pose IK and static load feedforward'

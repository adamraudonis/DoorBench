"""Manipulation-height motor stance preserving independently landed foot frames.

This is a privileged inverse-dynamics teacher, not a change to contact physics.
The zero margin uses the actual sole bounds; physical rollouts must still verify
that the planned support is realizable rather than trusting the planner alone.
"""
import numpy as np
import mujoco
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from doorbench.dexterous.stance import StanceController


class LandedFootStanceController(StanceController):
    def __init__(self,sim):
        super().__init__(sim)
        self.foot_rotations=[sim.d.xmat[b].reshape(3,3).copy() for b in self.feet]
        self.support=[(lo-.003,hi+.003,height) for lo,hi,height in self.support]
    def accept_physical_foot_repositioning(self):
        """Refresh planner references only; never move or constrain physical feet."""
        self.foot_positions=self.sim.d.xpos[self.feet].copy()
        self.foot_rotations=[self.sim.d.xmat[b].reshape(3,3).copy() for b in self.feet]

    def command(self):
        s=self.sim;m,d=s.m,s.d;n=16;nt=10;nf=12;N=n+nt+nf
        M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,M)
        J=[];accelerations=[]
        for body,pos,foot_rotation in zip(self.feet,self.foot_positions,self.foot_rotations):
            jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,body)
            R=foot_rotation;T=np.zeros((6,6));T[:3,:3]=R.T;T[3:,3:]=R.T
            jac=T@np.vstack((jp,jr));J.append(jac[:,self.v])
            orient=Rotation.from_matrix(foot_rotation@d.xmat[body].reshape(3,3).T).as_rotvec()
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
        solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(H),q=linear,A=A,l=lower,u=upper,
             verbose=False,eps_abs=1e-4,eps_rel=1e-4,max_iter=16000,**settings)
        if self.last is not None:solver.warm_start(x=self.last)
        result=solver.solve(raise_error=False) if modern else solver.solve()
        if result.info.status_val not in (1,2):return None,result.info.status
        self.last=result.x
        controls=(result.x[n:n+nt]-bias)/m.actuator_gainprm[self.act,0]
        return controls,result.info.status

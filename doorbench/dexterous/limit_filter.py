"""Bounded motor-space joint-limit filter for a privileged native teacher.

Uses measured generalized contact loads and full dynamics, so this implementation
belongs to the privileged teacher. It applies motor forces only; it never changes
joint states, joint limits, actuator strength, or simulator constraints.
"""
import mujoco
import numpy as np
import osqp
from scipy import sparse


class HandLimitFilter:
    def __init__(self, model, data, robot_actuators, joint_ids, transmission,
                 finger_motors, *, margin=.015, frequency=30.):
        self.m, self.d = model, data
        self.actuators = np.asarray(robot_actuators)
        self.local = np.asarray(finger_motors)
        self.joints = np.asarray([j for j in joint_ids
            if model.joint(j).name.startswith('robot/rh_') and 'WRJ' not in model.joint(j).name])
        self.q = model.jnt_qposadr[self.joints]
        self.v = model.jnt_dofadr[self.joints]
        self.B = np.zeros((model.nv, len(self.actuators)))
        self.B[model.jnt_dofadr[joint_ids]] = transmission.T
        self.margin, self.frequency = margin, frequency
        self.last = None

    def apply(self, nominal):
        m,d=self.m,self.d
        force=np.asarray(nominal).copy()
        mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,mass)
        # Remove previous robot actuator loads, preserve non-robot mechanism loads.
        other=d.qfrc_actuator-self.B@d.actuator_force[self.actuators]
        fixed=force.copy();fixed[self.local]=0.
        external=d.qfrc_passive+d.qfrc_constraint-d.qfrc_bias+other+self.B@fixed
        response=np.linalg.solve(mass,np.column_stack([external,self.B[:,self.local]]))[self.v]
        offset=response[:,0];mapping=response[:,1:]
        q=d.qpos[self.q];v=d.qvel[self.v];omega=self.frequency
        lower=-2*omega*v-omega**2*(q-m.jnt_range[self.joints,0]-self.margin)-offset
        upper=-2*omega*v+omega**2*(m.jnt_range[self.joints,1]-self.margin-q)-offset
        limits=m.actuator_forcerange[self.actuators[self.local]]
        A=sparse.csc_matrix(np.vstack([mapping,np.eye(len(self.local))]))
        solver=osqp.OSQP()
        modern=int(osqp.__version__.split('.')[0])>=1
        solver.setup(P=sparse.eye(len(self.local),format='csc'),q=-force[self.local],A=A,
            l=np.r_[lower,limits[:,0]],u=np.r_[upper,limits[:,1]],verbose=False,
            eps_abs=1e-5,eps_rel=1e-5,max_iter=3000,**({'polishing':False} if modern else {'polish':False}))
        if self.last is not None:solver.warm_start(x=self.last)
        result=solver.solve(raise_error=False) if modern else solver.solve()
        if result.info.status_val in (1,2):
            force[self.local]=np.clip(result.x,limits[:,0],limits[:,1]);self.last=force[self.local].copy()
        # Infeasibility is exposed to the evaluator/controller, not called success.
        return force,dict(status=result.info.status,feasible=result.info.status_val in (1,2),
            max_limit_violation_rad=float(np.maximum(m.jnt_range[self.joints,0]-q,q-m.jnt_range[self.joints,1]).max()))

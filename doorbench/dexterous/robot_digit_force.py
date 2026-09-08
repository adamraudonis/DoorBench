"""Robot-local pad virtual work projected onto original finger motors."""
import mujoco
import numpy as np


class RobotDigitForce:
    def __init__(self,robot_only_model,joint_names,action_names,transmission):
        self.m=m=robot_only_model;self.names=tuple(joint_names);self.actions=tuple(action_names)
        self.matrix=np.asarray(transmission,float).copy()
        if m.nq!=76 or m.nv!=75 or m.nu!=61 or self.matrix.shape!=(61,69):raise ValueError('Original robot-only motor/joint model required')
        if tuple(m.joint(i).name for i in range(1,m.njnt))!=self.names or tuple(m.actuator(i).name for i in range(m.nu))!=self.actions:raise ValueError('Exact authored motor/encoder order required')
        self.qa=np.array([m.jnt_qposadr[m.joint(n).id] for n in self.names]);self.va=np.array([m.jnt_dofadr[m.joint(n).id] for n in self.names])
        self.d=mujoco.MjData(m);self.jp=np.zeros((3,m.nv));self.jr=np.zeros((3,m.nv));self.groups={}
        for digit in ('ff','mf','rf','lf','th'):
            columns=np.array([i for i,n in enumerate(self.names) if n.startswith('rh_'+digit.upper()+'J')])
            rows=np.flatnonzero(np.any(abs(self.matrix[:,columns])>0,axis=1))
            A=self.matrix[np.ix_(rows,columns)]
            if len(columns)!=(5 if digit in ('lf','th') else 4) or len(rows)!=(5 if digit=='th' else len(columns)-1):raise ValueError('Original finger actuation rank changed')
            if np.any(np.delete(self.matrix[rows],columns,axis=1)!=0) or np.linalg.matrix_rank(A)!=len(rows):raise ValueError('Finger force projection cannot couple into other joints')
            self.groups[digit]=(columns,rows,A,m.site('rh_'+digit+'distal_touch').id)

    def motor_bias(self,joint_positions,correction_force_N):
        q=np.asarray(joint_positions,float);force=np.asarray(correction_force_N,float)
        if q.shape!=(69,) or force.shape!=(5,) or not np.isfinite(np.r_[q,force]).all() or np.any(abs(force)>2.+1e-12):raise ValueError('Finite encoder vector and bounded five-digit correction required')
        d,m=self.d,self.m;d.qpos[:7]=[0,0,1,1,0,0,0];d.qpos[self.qa]=q
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        bias=np.zeros(61);diagnostics={}
        for k,(digit,(columns,rows,A,site)) in enumerate(self.groups.items()):
            # Site -Z is the calibrated outward palmar direction. Its origin
            # defines the nominal pad force line; shifting along this normal
            # onto the palmar surface leaves virtual-work moments unchanged.
            normal=-d.site_xmat[site].reshape(3,3)[:,2]
            mujoco.mj_jacSite(m,d,self.jp,self.jr,site)
            requested=self.jp[:,self.va[columns]].T@(force[k]*normal)
            # tau = A.T @ u. Equal J1/J2 moment is the only actuated split for
            # each original J0 motor. Least-squares projection never invents
            # the unavailable difference torque; it is recorded explicitly.
            u=np.linalg.solve(A@A.T,A@requested);realized=A.T@u
            bias[rows]=u
            diagnostics[digit]=dict(joint_names=[self.names[i] for i in columns],motor_names=[self.actions[i] for i in rows],
                requested_joint_moment_Nm=requested.tolist(),actuated_joint_moment_Nm=realized.tolist(),
                unavailable_passive_split_moment_Nm=(requested-realized).tolist(),motor_bias_Nm=u.tolist(),
                correction_force_N=float(force[k]),force_line_site=m.site(site).name,
                actuator_coordinate_work_projection=True)
        return bias,diagnostics

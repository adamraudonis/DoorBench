"""Bounded, warm-started whole-body targets for measured contact goals.

This privileged planning helper owns unstepped data. It returns root/ordinary
joint targets; the existing stance and joint motors must realize them. It never
steps physics, writes an active pose, applies force, or qualifies contact.
"""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .actual_base_palm import bounded_next_velocity


class WholeBodyContactTargets:
    def __init__(self, model, names):
        self.model = model
        self.data = mujoco.MjData(model)
        self.names = list(names)
        self.scalar = {model.joint(j).name: int(model.jnt_qposadr[j])
                       for j in range(model.njnt) if model.jnt_type[j] in (2,3)}
        if len(self.names) != 25 or len(set(self.names)) != 25 or not set(self.names) <= set(self.scalar):
            raise ValueError('Require the original 25 leg, waist and arm joint targets')
        roots = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(roots) != 1:
            raise ValueError('Require exactly one robot free base')
        self.root_address = int(model.jnt_qposadr[roots[0]])
        self.root_body = int(model.jnt_bodyid[roots[0]])
        self.qa = np.array([self.scalar[n] for n in self.names])
        js = [model.joint(n).id for n in self.names]
        if not np.all(model.jnt_limited[js]):
            raise ValueError('Require the original limited body joints')
        # Inscribe the position boxes in the declared vector-norm envelopes.
        # Per-axis bounds of the full radius could pass the solve and then
        # violate the root norm check; conservative boxes make the boundary
        # available to the optimizer before another command is produced.
        self.lower = np.r_[np.full(3,-.03/np.sqrt(3)),np.full(3,-.05/np.sqrt(3)),model.jnt_range[js,0]]
        self.upper = np.r_[np.full(3,.03/np.sqrt(3)),np.full(3,.05/np.sqrt(3)),model.jnt_range[js,1]]
        self.palms = [model.site(side+'_palm_touch').id for side in ('lh','rh')]
        self.feet = [model.body(side+'_ankle_link').id for side in ('left','right')]
        self.time = None
        self.previous_residual = None

    def _kinematics(self, coordinate, joints):
        d,m = self.data,self.model
        d.qpos[:] = m.qpos0
        for name,address in self.scalar.items():
            d.qpos[address] = joints[name]
        d.qpos[self.root_address:self.root_address+3] = self.initial_root[:3]+coordinate[:3]
        quat = (Rotation.from_rotvec(coordinate[3:6])*self.initial_rotation).as_quat()
        d.qpos[self.root_address+3:self.root_address+7] = np.r_[quat[3],quat[:3]]
        d.qpos[self.qa] = coordinate[6:]
        mujoco.mj_kinematics(m,d)
        mujoco.mj_comPos(m,d)

    def begin(self, time_s, root, joints, coordinate=None, velocity=None):
        root = np.asarray(root,float)
        if (self.time is not None or root.shape!=(13,) or set(joints)!=set(self.scalar)
                or not np.isfinite(np.r_[time_s,root,list(joints.values())]).all()
                or not np.isclose(np.linalg.norm(root[3:7]),1.,atol=1e-6)):
            raise ValueError('Require the complete coherent measured initial robot state')
        self.initial_root = root[:7].copy()
        self.initial_rotation = Rotation.from_quat([*root[4:7],root[3]])
        actual = np.r_[np.zeros(6),[joints[n] for n in self.names]]
        self._kinematics(actual,joints)
        self.foot_positions = self.data.xpos[self.feet].copy()
        self.foot_rotations = self.data.xmat[self.feet].reshape(2,3,3).copy()
        self.right_position = self.data.site_xpos[self.palms[1]].copy()
        self.com_position = self.data.subtree_com[self.root_body,:2].copy()
        self.target = actual if coordinate is None else np.asarray(coordinate,float).copy()
        self.velocity = np.zeros(31) if velocity is None else np.asarray(velocity,float).copy()
        if self.target.shape!=(31,) or self.velocity.shape!=(31,):
            raise ValueError('Require complete body target position and velocity')
        self._bounds(.002)
        self.time = float(time_s)

    def _residual(self, coordinate, joints, position, rotation):
        self._kinematics(coordinate,joints)
        d = self.data
        residual = [100*(position-d.site_xpos[self.palms[0]]),
                    10*Rotation.from_matrix(rotation@d.site_xmat[self.palms[0]].reshape(3,3).T).as_rotvec(),
                    100*(self.right_position-d.site_xpos[self.palms[1]])]
        for i,body in enumerate(self.feet):
            residual += [100*(self.foot_positions[i]-d.xpos[body]),
                         10*Rotation.from_matrix(self.foot_rotations[i]@d.xmat[body].reshape(3,3).T).as_rotvec()]
        residual.append(5*(self.com_position-d.subtree_com[self.root_body,:2]))
        return np.concatenate(residual)

    def _bounds(self, dt):
        low,high = [],[]
        # Component limits conservatively guarantee the original vector speed
        # norms. Position vector norms are independently checked after solving.
        for start,end,speed,acceleration in ((0,3,.02/np.sqrt(3),.1),(3,6,.03/np.sqrt(3),.1),(6,31,1.2,3.)):
            a,b = bounded_next_velocity(self.target[start:end],self.velocity[start:end],
                                       self.lower[start:end],self.upper[start:end],dt,
                                       speed*(1.-1e-9),acceleration*(1.-1e-9))
            low.extend(a);high.extend(b)
        return np.array(low),np.array(high)

    def update(self,time_s,joints,palm_position_world,palm_rotation_world,nominal_coordinate):
        p,r,nominal = [np.asarray(v,float) for v in (palm_position_world,palm_rotation_world,nominal_coordinate)]
        if (self.time is None or set(joints)!=set(self.scalar) or p.shape!=(3,) or r.shape!=(3,3)
                or nominal.shape!=(31,) or not np.isfinite(np.r_[time_s,list(joints.values()),p,r.ravel(),nominal]).all()
                or not np.allclose(r@r.T,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(r),1.,atol=1e-7)):
            raise ValueError('Require complete measured joints and finite proper world-frame goals')
        dt = float(time_s-self.time)
        if dt<0 or (dt>0 and abs(dt-.002)>1e-9):
            raise ValueError('Require consecutive 2ms whole-body planning intervals')
        residual = self._residual(self.target,joints,p,r)
        acceleration = np.zeros(31)
        if dt:
            step = 1e-6
            jacobian = np.empty((len(residual),31))
            for j in range(31):
                perturbed = self.target.copy();perturbed[j]+=step
                jacobian[:,j] = -(self._residual(perturbed,joints,p,r)-residual)/step
            feedforward = np.zeros_like(residual) if self.previous_residual is None else (residual-self.previous_residual)/dt
            task_velocity = 6*(residual-dt*feedforward)+feedforward
            # A metre/radian of free-base motion has much more reach than one
            # arm joint. Equal coordinate penalties otherwise spend the small
            # root envelope first. Prefer the nominal base while allowing the
            # arms to follow the contact; all existing hard bounds still apply.
            posture_weight = np.r_[np.full(6,.5),np.full(25,.02)]
            matrix = np.vstack([jacobian,np.diag(posture_weight)])
            rhs = np.r_[task_velocity,6*posture_weight*(nominal-self.target)]
            low,high = self._bounds(dt)
            fixed = high-low<1e-12
            next_velocity = (low+high)/2
            if np.any(~fixed):
                fit = lsq_linear(matrix[:,~fixed],rhs-matrix[:,fixed]@next_velocity[fixed],
                                 bounds=(low[~fixed],high[~fixed]),method='bvls',tol=1e-11,max_iter=100)
                if not fit.success or not np.isfinite(fit.x).all():
                    raise ValueError('Bounded whole-body target solve failed')
                next_velocity[~fixed] = fit.x
            acceleration = (next_velocity-self.velocity)/dt
            self.target += .5*dt*(self.velocity+next_velocity)
            self.velocity = next_velocity
            self.time = float(time_s)
        self.previous_residual = self._residual(self.target,joints,p,r).copy()
        if (np.linalg.norm(self.target[:3])>.03+1e-10 or np.linalg.norm(self.target[3:6])>.05+1e-10
                or np.linalg.norm(self.data.subtree_com[self.root_body,:2]-self.com_position)>.015):
            raise ValueError('Whole-body target exceeded the original root/COM geometry envelope')
        info = dict(time_s=self.time,commanded_lh_position_error_m=float(np.linalg.norm(self.previous_residual[:3])/100),
                    commanded_lh_rotation_error_rad=float(np.linalg.norm(self.previous_residual[3:6])/10),
                    commanded_rh_position_error_m=float(np.linalg.norm(self.previous_residual[6:9])/100),
                    maximum_commanded_foot_position_error_m=max(float(np.linalg.norm(self.previous_residual[9+6*i:12+6*i])/100) for i in range(2)),
                    maximum_target_joint_speed_rad_s=float(max(abs(self.velocity[6:]))),
                    maximum_target_joint_acceleration_rad_s2=float(max(abs(acceleration[6:]))),
                    root_translation_m=float(np.linalg.norm(self.target[:3])),root_rotation_rad=float(np.linalg.norm(self.target[3:6])))
        return self.target.copy(),self.velocity.copy(),acceleration,info

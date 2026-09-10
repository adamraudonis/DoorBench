"""Bounded privileged arm targets preserving an attained handle-relative palm.

Owns unstepped kinematic data only. Targets still require motor-driven physical
validation; IK convergence does not establish contact or collision safety.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ARM_NAMES = tuple('right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw'))+('rh_WRJ2','rh_WRJ1')


def pose_rotation(pose):
    pose=np.asarray(pose,float)
    if pose.shape!=(7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1)>2e-6:
        raise ValueError('Finite measured XYZ/WXYZ pose required')
    return pose,Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()


class HandleRelativeArmTarget:
    def __init__(self, model, names, root, joints, handle_pose):
        self.m=model;self.d=mujoco.MjData(model);self.names=list(names)
        free=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        if len(free)!=1 or model.jnt_qposadr[free[0]]!=0:
            raise ValueError('One original free-root robot model required')
        self.qa=np.array([model.joint(n).qposadr[0] for n in self.names])
        if len(set(self.qa))!=len(self.names) or model.nq!=len(self.names)+7:
            raise ValueError('Complete unique scalar robot coordinates required')
        self.ids=np.array([model.joint(n).id for n in ARM_NAMES]);self.armqa=model.jnt_qposadr[self.ids]
        if not np.all(model.jnt_limited[self.ids]):raise ValueError('Explicit arm joint bounds required')
        self.site=model.site('rh_palm_touch').id
        self.read(root,joints)
        handle,rotation=pose_rotation(handle_pose)
        self.relative_position=rotation.T@(self.d.site_xpos[self.site]-handle[:3])
        self.relative_rotation=rotation.T@self.d.site_xmat[self.site].reshape(3,3)
        self.correction=np.zeros(7);self.goal=np.zeros(7);self.last_time=None;self.last_solve=None
        self.solve_info={}
        self.previous_nominal=None;self.nominal_time=None
        self.nominal_velocity=np.zeros(len(self.names));self.target_velocity=dict.fromkeys(self.names,0.)

    def read(self,root,joints):
        root=np.asarray(root,float)
        if root.shape!=(13,) or not np.isfinite(root).all() or set(joints)!=set(self.names):
            raise ValueError('Complete measured robot state required')
        pose_rotation(root[:7])
        values=np.array([joints[n] for n in self.names],float)
        if not np.isfinite(values).all():raise ValueError('Finite robot joints required')
        self.d.qpos[:7]=root[:7];self.d.qpos[self.qa]=values
        mujoco.mj_kinematics(self.m,self.d)

    def target(self,t,root,joints,handle_pose,nominal):
        if not np.isfinite(t) or t<0 or (self.last_time is not None and t<self.last_time):
            raise ValueError('Monotonic finite controller clock required')
        if set(nominal)!=set(self.names):raise ValueError('Complete nominal joint target required')
        base=np.array([nominal[n] for n in ARM_NAMES],float)
        if not np.isfinite(base).all():raise ValueError('Finite nominal arm target required')
        self.read(root,joints);handle,rotation=pose_rotation(handle_pose)
        position=handle[:3]+rotation@self.relative_position;orientation=rotation@self.relative_rotation
        # Preserve an already attained nominal coordinate at a soft stop, but
        # never allow compensation farther beyond that nominal value.
        lower=np.maximum(np.minimum(self.m.jnt_range[self.ids,0]+.01,base),base-.08)
        upper=np.minimum(np.maximum(self.m.jnt_range[self.ids,1]-.01,base),base+.08)
        if self.last_solve is None or t-self.last_solve>=.02-1e-10:
            def residual(x):
                self.d.qpos[self.armqa]=x;mujoco.mj_kinematics(self.m,self.d)
                error=(Rotation.from_matrix(self.d.site_xmat[self.site].reshape(3,3))*Rotation.from_matrix(orientation).inv()).as_rotvec()
                return np.r_[100*(self.d.site_xpos[self.site]-position),10*error,.001*(x-base)]
            fit=least_squares(residual,np.clip(base+self.goal,lower,upper),bounds=(lower,upper),max_nfev=60,ftol=1e-8,xtol=1e-8,gtol=1e-7)
            error=residual(fit.x)
            if not fit.success or np.linalg.norm(error[:3])/100>.0005 or np.linalg.norm(error[3:6])/10>.005:
                raise ValueError(f'Handle-relative arm target is outside bounded reachability: {fit.message}; position={np.linalg.norm(error[:3])/100:.6g}m rotation={np.linalg.norm(error[3:6])/10:.6g}rad')
            self.goal=fit.x-base;self.last_solve=t
            self.solve_info=dict(position_error_m=float(np.linalg.norm(error[:3])/100),rotation_error_rad=float(np.linalg.norm(error[3:6])/10),evaluations=fit.nfev)
        # Differentiate the 10 ms route and 2 ms correction on their own
        # clocks. Differentiating their sum treats a route step as a 2 ms
        # change whenever compensation changed in the intervening substeps.
        nominal_values=np.array([nominal[n] for n in self.names],float)
        if not np.isfinite(nominal_values).all():raise ValueError('Finite nominal joint target required')
        if self.previous_nominal is None or not np.array_equal(nominal_values,self.previous_nominal):
            if self.nominal_time is not None and t<=self.nominal_time:
                raise ValueError('Changed nominal target requires advancing controller clock')
            self.nominal_velocity=np.zeros(len(self.names)) if self.nominal_time is None else (nominal_values-self.previous_nominal)/(t-self.nominal_time)
            self.previous_nominal=nominal_values.copy();self.nominal_time=t
        elif t-self.nominal_time>.02:
            self.nominal_velocity=np.zeros(len(self.names))
        dt=0. if self.last_time is None else t-self.last_time
        previous=self.correction.copy()
        self.correction+=np.clip(self.goal-self.correction,-.5*dt,.5*dt)
        corrected=np.clip(base+self.correction,lower,upper)
        if np.max(abs(corrected-base-previous))>.5*dt+1e-10:
            raise ValueError('Moving nominal bounds conflict with compensation rate limit')
        self.correction=corrected-base;self.last_time=t
        velocity=self.nominal_velocity.copy()
        if dt>0:
            for i,n in enumerate(ARM_NAMES):velocity[self.names.index(n)]+=(self.correction[i]-previous[i])/dt
        self.target_velocity=dict(zip(self.names,map(float,velocity)))
        result=dict(nominal);result.update(zip(ARM_NAMES,map(float,corrected)))
        return result,dict(handle_relative_arm=True,maximum_correction_rad=float(np.max(abs(self.correction))),correction_limit_rad=.08,correction_rate_limit_rad_s=.5,solve=self.solve_info.copy())

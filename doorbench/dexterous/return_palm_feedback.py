"""Bounded privileged palm-error integration during a motor-driven return.

Only an unstepped private FK model is used. The correction adjusts original arm
motor references; contact qualification remains an independent plant measurement.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .operation_teacher import pose_components, reproject_grasp


def integrate_correction(previous, jacobian, error, dt, lower, upper):
    arrays = [np.asarray(x, float) for x in (previous, jacobian, error, lower, upper)]
    previous, jacobian, error, lower, upper = arrays
    if not all(np.isfinite(x).all() for x in arrays) or not np.isfinite(dt) or not 0 <= dt <= .05:
        raise ValueError('Finite feedback and a monotonic bounded update interval required')
    # Rotation rows use a 0.1 m characteristic hand length. Damping avoids
    # unbounded corrections near the wrist stops or a singular arm posture.
    velocity = 3. * jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6)*1e-5, error)
    velocity = np.clip(velocity, -.06, .06)
    return np.clip(previous + dt*velocity, np.maximum(lower, -.06), np.minimum(upper, .06))


class ReturnPalmFeedback:
    def __init__(self, teacher, root, joints, handle_pose, geometry):
        self.teacher = teacher
        self.m = teacher.m
        self.d = mujoco.MjData(self.m)
        self.geometry = geometry
        self.names = [self.m.joint(int(self.m.actuator_trnid[teacher.act[i], 0])).name for i in teacher.arm_motors]
        ids = np.array([self.m.joint(n).id for n in self.names])
        self.qa = self.m.jnt_qposadr[ids]
        self.va = self.m.jnt_dofadr[ids]
        self.limits = self.m.jnt_range[ids].copy()
        self.offset = np.zeros(len(ids))
        self.last_time = None
        self.jp = np.zeros((3,self.m.nv)); self.jr = self.jp.copy()
        self._read(root,joints)
        hp, hr = pose_components(handle_pose)
        self.position = hr.T @ (self.d.site_xpos[teacher.palm]-hp)
        self.rotation = hr.T @ self.d.site_xmat[teacher.palm].reshape(3,3)

    def _read(self,root,joints):
        self.d.qpos[:7] = np.asarray(root)[:7]
        self.d.qpos[self.teacher.qa] = [joints[n] for n in self.teacher.names]
        mujoco.mj_kinematics(self.m,self.d)
        mujoco.mj_comPos(self.m,self.d)

    def targets(self,t,nominal,root,joints,handle_pose,leaf_pose,angles,operator_goal):
        self._read(root,joints)
        p,r = reproject_grasp(handle_pose,leaf_pose,angles,
            dict(operator=operator_goal,leaf=angles['leaf']),self.position,self.rotation,self.geometry)
        pe = p-self.d.site_xpos[self.teacher.palm]
        re = Rotation.from_matrix(r@self.d.site_xmat[self.teacher.palm].reshape(3,3).T).as_rotvec()
        mujoco.mj_jacSite(self.m,self.d,self.jp,self.jr,self.teacher.palm)
        jacobian = np.vstack([self.jp[:,self.va],.1*self.jr[:,self.va]])
        q = np.array([nominal[n] for n in self.names])
        dt = 0. if self.last_time is None else t-self.last_time
        self.offset = integrate_correction(self.offset,jacobian,np.r_[pe,.1*re],dt,self.limits[:,0]-q,self.limits[:,1]-q)
        self.last_time = t
        result = dict(nominal);result.update(zip(self.names,q+self.offset))
        return result,dict(return_palm_feedback=True,return_palm_error_m=float(np.linalg.norm(pe)),
            return_palm_rotation_error_rad=float(np.linalg.norm(re)),return_arm_offset_rad=self.offset.tolist())

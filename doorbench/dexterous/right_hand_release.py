"""Privileged axial hand release; only analytic teacher targets change.

Requires another already loaded physical contact. The qualified fixture slides
along a straight lever toward its free end while retaining the attained legal
finger configuration and unloading active grip force. This is neither a
sensor-only actor nor proof of loaded substantial opening.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def axial_release_goal(handle_pose, palm_position_handle, palm_rotation_handle,
                       elapsed, *, distance=.14, duration=6.):
    """Return a smooth world palm goal in the measured handle's current frame."""
    h=np.asarray(handle_pose,float);p=np.asarray(palm_position_handle,float)
    r=np.asarray(palm_rotation_handle,float)
    if h.shape!=(7,) or p.shape!=(3,) or r.shape!=(3,3):
        raise ValueError('Expected xyz/wxyz handle pose and local palm transform')
    values=np.r_[h,p,r.ravel(),elapsed,distance,duration]
    if not np.isfinite(values).all() or elapsed<0 or not 0<distance<=.2 or duration<=0:
        raise ValueError('Invalid finite axial release geometry or clock')
    if not np.isclose(np.linalg.norm(h[3:]),1.,atol=1e-5) or not np.allclose(r.T@r,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(r),1.,atol=1e-6):
        raise ValueError('Expected proper unit rotations')
    u=float(np.clip(elapsed/duration,0,1));blend=u**3*(10+u*(-15+6*u))
    hr=Rotation.from_quat([*h[4:7],h[3]]).as_matrix()
    return h[:3]+hr@(p+np.array([-distance*blend,0,0])),hr@r,u,blend


class AxialRightRelease:
    """Unloaded axial slide toward a straight lever's free end; privileged teacher."""
    def __init__(self,teacher,path):
        self.teacher=teacher;self.started=None;self.info={};self.frozen=None
        data=json.loads(Path(path).read_text())
        good=[r for r in data['rows'] if r['dx']==-.14 and r['dz']==0. and r['max_error']<.001 and r['max_collision']==0 and r['gap']>.02]
        if len(good)!=1:raise ValueError('Axial release requires the qualified geometric screen')
        self.distance=.14;self.duration=6.;self.last_time=None

    def begin(self,t,joints,root,handle_pose):
        if self.started is not None:raise ValueError('Release already began')
        if not np.isfinite(t) or t<0:raise ValueError('Invalid release clock')
        if set(joints)!=set(self.teacher.names) or not np.isfinite([joints[n] for n in self.teacher.names]).all():raise ValueError('Expected complete finite measured joints')
        root=np.asarray(root,float)
        if root.shape!=(13,) or not np.isfinite(root).all() or not np.isclose(np.linalg.norm(root[3:7]),1.,atol=1e-5):raise ValueError('Expected finite xyz/wxyz measured root')
        axial_release_goal(handle_pose,[0,0,0],np.eye(3),0)
        self.started=t;teacher=self.teacher;m,d=teacher.m,teacher.d
        d.qpos[:7]=root[:7]
        for name,value in joints.items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
        mujoco.mj_kinematics(m,d)
        hr=Rotation.from_quat([*handle_pose[4:7],handle_pose[3]]).as_matrix()
        self.p_relative=hr.T@(d.site_xpos[teacher.palm]-handle_pose[:3]);self.r_relative=hr.T@d.site_xmat[teacher.palm].reshape(3,3)
        self.digit_forces=teacher.digit_forces.copy();self.handle_pose=np.asarray(handle_pose).copy()
        finger_indices=[i for i,n in enumerate(teacher.names) if n.startswith('rh_') and 'WRJ' not in n]
        teacher.path[-1,finger_indices]=[joints[teacher.names[i]] for i in finger_indices]
        teacher.position_integral[:]=0.;teacher.rotation_integral[:]=0.
        keep=np.array([m.joint(int(j)).name!='torso' for j in teacher.arm_joints]);teacher.arm_joints=teacher.arm_joints[keep];teacher.arm_q=teacher.arm_q[keep];teacher.arm_v=teacher.arm_v[keep]
        teacher.path[-1,teacher.names.index('torso')]=joints['torso']

    def update(self,t):
        if self.started is None:return
        if not np.isfinite(t) or t<self.started or (self.last_time is not None and t<self.last_time):raise ValueError('Invalid release clock')
        self.last_time=t;elapsed=t-self.started
        position,rotation,u,blend=axial_release_goal(self.handle_pose,self.p_relative,self.r_relative,elapsed,distance=self.distance,duration=self.duration)
        if self.frozen is None:self.teacher.positions[-1]=position;self.teacher.rotations[-1]=rotation
        else:self.teacher.positions[-1]=self.frozen[0];self.teacher.rotations[-1]=self.frozen[1]
        scale=float(np.clip(1-elapsed/.4,0,1));self.teacher.digit_forces={d:f*scale for d,f in self.digit_forces.items()}
        self.info=dict(phase='right_axial_release',release_fraction=u,release_elapsed_s=elapsed,grip_preload_scale=scale,axial_slide_m=self.distance*blend,goal_frozen_after_measured_clearance=self.frozen is not None)


def _pose_matrix(pose):
    value = np.asarray(pose, dtype=float)
    axial_release_goal(value, np.zeros(3), np.eye(3), 0.)
    return value[:3], Rotation.from_quat([*value[4:7], value[3]]).as_matrix()


def bind_handle_to_leaf(handle_pose, leaf_pose):
    """Bind the attained pressed handle frame to the measured leaf, once."""
    hp, hr = _pose_matrix(handle_pose)
    lp, lr = _pose_matrix(leaf_pose)
    return lr.T @ (hp - lp), lr.T @ hr


def handle_from_leaf(leaf_pose, relative_position, relative_rotation):
    """Reconstruct a target frame; this never writes the physical handle."""
    lp, lr = _pose_matrix(leaf_pose)
    p = np.asarray(relative_position, float)
    r = np.asarray(relative_rotation, float)
    # Reuse the same strict relative-transform validation as the axial goal.
    axial_release_goal(leaf_pose, p, r, 0.)
    q = Rotation.from_matrix(lr @ r).as_quat()
    return np.r_[lp + lr @ p, q[3], q[:3]]


class PressedLeafFrameRightRelease(AxialRightRelease):
    """Development opt-in: retain pressed orientation during axial withdrawal.

    The original release follows the measured springing operator. This variant
    keeps its attained pressed frame relative to the *measured moving leaf*,
    so releasing grip preload does not also command the wrist to follow the
    returning lever. Original distances, timing, finger targets, motor limits,
    and independent clearance/patch gates remain unchanged. A geometric screen
    of a frozen pressed lever is insufficient to qualify this physical route.

    The caller must deliver the current measured leaf via ``observe_leaf``
    before each begin/update, on the same local teacher clock. This remains a
    privileged teacher and is unqualified until a fresh physical trial passes.
    """
    def __init__(self, teacher, path, *, retain_grip_until_clear=False):
        super().__init__(teacher, path)
        if type(retain_grip_until_clear) is not bool:
            raise ValueError("Require an explicit boolean grip-retention option")
        self.retain_grip_until_clear = retain_grip_until_clear
        self.clear_time = None
        self.leaf_time = None
        self.leaf_pose = None

    def observe_leaf(self, t, pose):
        _pose_matrix(pose)
        if not np.isfinite(t) or t < 0 or (self.leaf_time is not None and t < self.leaf_time):
            raise ValueError("Require a monotonic finite measured leaf clock")
        self.leaf_time = float(t)
        self.leaf_pose = np.asarray(pose, float).copy()

    def _current_leaf(self, t):
        if self.leaf_time is None or abs(self.leaf_time - t) > 1e-8:
            raise ValueError("Release requires current measured leaf pose")
        return self.leaf_pose

    def begin(self, t, joints, root, handle_pose):
        leaf = self._current_leaf(t)
        super().begin(t, joints, root, handle_pose)
        self.pressed_position_leaf, self.pressed_rotation_leaf = bind_handle_to_leaf(handle_pose, leaf)

    def update(self, t):
        if self.started is None:
            return
        leaf = self._current_leaf(t)
        self.handle_pose = handle_from_leaf(leaf, self.pressed_position_leaf, self.pressed_rotation_leaf)
        super().update(t)
        if self.retain_grip_until_clear:
            if self.frozen is not None and self.clear_time is None:
                self.clear_time = float(t)
            scale = 1. if self.clear_time is None else float(np.clip(1 - (t - self.clear_time) / .4, 0., 1.))
            self.teacher.digit_forces = {digit: force * scale for digit, force in self.digit_forces.items()}
            self.info['grip_preload_scale'] = scale
        self.info.update(release_frame="attained_pressed_handle_relative_to_measured_leaf",
                         leaf_pose_time_s=self.leaf_time, controller_status="development_unqualified",
                         retain_grip_until_clear=self.retain_grip_until_clear,
                         measured_clear_time_s=self.clear_time)

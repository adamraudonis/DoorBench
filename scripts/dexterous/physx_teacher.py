"""Privileged task-space teacher: FK/IK only, never steps a second simulator.

Isaac is the plant. MuJoCo provides independent analytic robot kinematics.
Only bounded motor targets leave this module. Not a vision/tactile policy.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


class HandleTeacher:
    def __init__(self,robot_xml,motors,reference):
        self.m=mujoco.MjModel.from_xml_path(robot_xml);self.d=mujoco.MjData(self.m)
        self.motors=motors;self.joint_names=motors['joint_names']
        self.initial=np.array(reference['controls'][40]);self.control=self.initial.copy()
        names=['right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_yaw','rh_WRJ2','rh_WRJ1']
        self.joints=[self.m.joint(n).id for n in names];self.qa=self.m.jnt_qposadr[self.joints];self.va=self.m.jnt_dofadr[self.joints]
        self.act=[next(i for i,m in enumerate(motors['actuators']) if set(m['terms'])=={n}) for n in names]
        self.fingers=[i for i,m in enumerate(motors['actuators']) if m['name'].startswith('rh_') and 'WRJ' not in m['name']]
        self.low=self.m.jnt_range[self.joints,0]+.015;self.high=self.m.jnt_range[self.joints,1]-.015
        self.palm=self.m.site('rh_palm_touch').id
        self.relative_position=None;self.release=None
        self.jp=np.zeros((3,self.m.nv));self.jr=np.zeros_like(self.jp)

    def command(self,t,root,joints,leaf,handle,handle_angle,door_angle):
        m,d=self.m,self.d;d.qpos[:7]=root[:7]
        for n,q in joints.items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=q
        mujoco.mj_kinematics(m,d)
        leaf_pos=np.array(leaf[:3]);leaf_rot=Rotation.from_quat([*leaf[4:7],leaf[3]]).as_matrix()
        hpos=np.array(handle[:3]);hrot=Rotation.from_quat([*handle[4:7],handle[3]]).as_matrix()
        if self.relative_position is None:
            self.relative_position=hrot.T@(d.site_xpos[self.palm]-hpos)
            self.relative_rotation=hrot.T@d.site_xmat[self.palm].reshape(3,3)
            self.handle_local=leaf_rot.T@(hpos-leaf_pos)
        if self.release is None and door_angle>.30:
            self.release=t;self.release_control=self.control.copy()
        if self.release is not None:
            blend=smooth((t-self.release-.4)/1.2)
            self.control=self.release_control*(1-blend)+self.initial*blend
            self.control[self.fingers]*=1-smooth((t-self.release)/.4)
            return self.control.copy(),dict(phase='release',ik_error_m=0.)
        goal=.80*smooth((t-1.)/1.5)
        lead=.09 if handle_angle>.65 or door_angle>.015 else 0.
        future_leaf=leaf_rot@Rotation.from_rotvec([0,0,lead]).as_matrix()
        origin=leaf_pos+future_leaf@self.handle_local
        operator_rotation=future_leaf@Rotation.from_rotvec([0,-goal,0]).as_matrix()
        position=origin+operator_rotation@self.relative_position
        rotation=operator_rotation@self.relative_rotation
        for _ in range(35):
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            error=np.r_[(position-d.site_xpos[self.palm])*5,Rotation.from_matrix(rotation@d.site_xmat[self.palm].reshape(3,3).T).as_rotvec()]
            mujoco.mj_jacSite(m,d,self.jp,self.jr,self.palm)
            jac=np.vstack([self.jp[:,self.va]*5,self.jr[:,self.va]])
            change=jac.T@np.linalg.solve(jac@jac.T+.003*np.eye(6),error)
            d.qpos[self.qa]=np.clip(d.qpos[self.qa]+np.clip(change,-.08,.08),self.low,self.high)
            if np.linalg.norm(error)<1e-4:break
        mujoco.mj_kinematics(m,d)
        self.control[self.act]=d.qpos[self.qa]
        return self.control.copy(),dict(phase='push' if lead else 'press',ik_error_m=float(np.linalg.norm(position-d.site_xpos[self.palm])),goal_handle_rad=float(goal))

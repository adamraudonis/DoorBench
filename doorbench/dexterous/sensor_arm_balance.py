"""Opt-in arm targets over sensor-only lowered balance; no task-state inputs.

High-level joint goals are explicitly additional scripted/learned commands, not
sensor measurements. The original stationary controller and its torso/leg
posture targets are unchanged. No environment model or active plant is owned.
"""
import numpy as np
from .sensor_balance import SensorBalanceController


class SensorArmBalanceController:
    def __init__(self, robot_xml, motor_contract, sensor_layout, desired_posture,
                 *, physics_dt_s=.002, image_shape=(128,128,3), gravity_correction=.2,
                 maximum_goal_speed_radps=.5):
        self.balance=SensorBalanceController(robot_xml,motor_contract,sensor_layout,desired_posture,
            physics_dt_s=physics_dt_s,image_shape=image_shape,gravity_correction=gravity_correction)
        self.goal_names=tuple(n for n in self.balance.names if any(k in n for k in ('shoulder','elbow','WRJ')))
        if len(self.goal_names)!=12:raise ValueError('Expected twelve H1/Shadow arm and wrist joints')
        self.maximum_goal_speed_radps=float(maximum_goal_speed_radps)
        if not np.isfinite(self.maximum_goal_speed_radps) or not 0<self.maximum_goal_speed_radps<=1.:
            raise ValueError('Use an explicit finite arm-goal slew limit in (0,1]rad/s')
        self.indices=np.array([self.balance.names.index(n) for n in self.goal_names])
        self.joint_ids=self.balance.joints[self.indices]
        self.arm_motors=np.flatnonzero(np.any(abs(self.balance.matrix[:,self.indices])>0,axis=1))
        if len(self.arm_motors)!=12 or np.any(abs(np.delete(self.balance.matrix[self.arm_motors],self.indices,axis=1))>0):
            raise ValueError('Arm goals must not couple to torso, leg or finger motors')
        self.reset_episode()

    @property
    def joint_names(self):return tuple(self.balance.names)
    @property
    def action_names(self):return tuple(self.balance.actions)
    @property
    def shapes(self):return dict(self.balance.shapes)
    @property
    def caps(self):return self.balance.caps
    @property
    def last_force(self):return self.balance.last_force.copy()
    @property
    def last_info(self):return dict(self.info)

    def reset_episode(self):
        self.balance.reset_episode()
        self.balance.target=np.clip(self.balance.matrix@self.balance.desired,self.balance.limits[:,0],self.balance.limits[:,1])
        self.last_goals=self.balance.desired[self.indices].copy()
        self.failed_reason=None;self.info={}

    def force(self, packet, *, now_s, joint_goals=None):
        """Return61 capped forces from sensor packet and12 numeric joint targets.

        Omit goals to hold the last accepted target (initially stationary). Explicit
        goals require all twelve names, original joint bounds, a matching t0
        posture, and bounded change per2ms call. Rejections require reset.
        """
        if self.failed_reason is not None:raise RuntimeError('Arm inference requires reset_episode: '+self.failed_reason)
        try:
            b=self.balance
            if joint_goals is None:goals=self.last_goals.copy()
            else:
                if type(joint_goals) is not dict or set(joint_goals)!=set(self.goal_names):
                    raise ValueError('Arm command must contain exactly twelve arm/wrist names')
                if any(type(v) not in (int,float,np.float32,np.float64) for v in joint_goals.values()):
                    raise ValueError('Arm goals must be scalar numeric joint angles')
                goals=np.array([joint_goals[n] for n in self.goal_names],float)
            limits=b.m.jnt_range[self.joint_ids]
            if not np.isfinite(goals).all() or np.any(goals<limits[:,0]) or np.any(goals>limits[:,1]):
                raise ValueError('Arm goal exceeds the original authored joint bounds')
            bound=1e-8 if b.last_time is None else self.maximum_goal_speed_radps*b.dt+1e-8
            if np.max(abs(goals-self.last_goals))>bound:raise ValueError('Arm command exceeds initial calibration or goal slew bound')
            q=b.desired.copy();q[self.indices]=goals
            target=b.matrix@q
            if np.any(target[self.arm_motors]<b.limits[self.arm_motors,0]) or np.any(target[self.arm_motors]>b.limits[self.arm_motors,1]):
                raise ValueError('Arm command exceeds native actuator control limits')
            b.target[self.arm_motors]=target[self.arm_motors]
            force,info=b.force(packet,now_s=now_s)
            self.last_goals=goals.copy()
            self.info=dict(info,controller='sensor_arm_balance_v1',high_level_input='explicit arm/wrist joint commands',
                goal_joint_names=list(self.goal_names),goal_joint_position_rad=goals.tolist(),
                unchanged_balance_targets='calibrated torso/legs; estimated body state from encoders/IMU/touch only')
            return force,dict(self.info)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc)
            raise

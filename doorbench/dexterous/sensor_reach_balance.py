"""Opt-in coordinated right-hand reach over sensor-only pelvis/leg balance.

High-level joint goals are explicitly additional scripted/learned commands, not
sensor measurements. The original stationary controller source is unchanged. Pelvis/leg targets stay
fixed; torso yaw must be explicitly enabled as an upper-body joint command. No environment model or active plant is owned.
"""
import numpy as np
from .sensor_balance import SensorBalanceController


class SensorReachBalanceController:
    def __init__(self, robot_xml, motor_contract, sensor_layout, desired_posture,
                 *, physics_dt_s=.002, image_shape=(128,128,3), gravity_correction=.2,
                 maximum_goal_speed_radps=1.5, allow_torso_yaw=False,
                 finger_impedance_multiplier=1., finger_velocity_damping=0.,
                 finger_target_velocity_damping=False, tactile_reflex_profile=None):
        self.balance=SensorBalanceController(robot_xml,motor_contract,sensor_layout,desired_posture,
            physics_dt_s=physics_dt_s,image_shape=image_shape,gravity_correction=gravity_correction)
        if type(allow_torso_yaw) is not bool:raise ValueError('Declare torso yaw scope explicitly')
        self.allow_torso_yaw=allow_torso_yaw
        self.goal_names=tuple(n for n in self.balance.names if n.startswith('rh_') or (n.startswith('right_') and not any(k in n for k in ('hip_','knee','ankle'))) or (allow_torso_yaw and n=='torso'))
        if len(self.goal_names)!=(30 if allow_torso_yaw else 29):raise ValueError('Unexpected right-hand/arm joint scope')
        self.maximum_goal_speed_radps=float(maximum_goal_speed_radps)
        if not np.isfinite(self.maximum_goal_speed_radps) or not 0<self.maximum_goal_speed_radps<=2.:
            raise ValueError('Use an explicit finite arm-goal slew limit in (0,2]rad/s')
        self.indices=np.array([self.balance.names.index(n) for n in self.goal_names])
        self.joint_ids=self.balance.joints[self.indices]
        self.arm_motors=np.flatnonzero(np.any(abs(self.balance.matrix[:,self.indices])>0,axis=1))
        if len(self.arm_motors)!=(26 if allow_torso_yaw else 25) or np.any(abs(np.delete(self.balance.matrix[self.arm_motors],self.indices,axis=1))>0):
            raise ValueError('Reach goals must not couple outside the declared right-upper-body scope')
        if type(finger_target_velocity_damping) is not bool:raise ValueError('Declare finger target-velocity damping explicitly')
        self.finger_target_velocity_damping=finger_target_velocity_damping
        self.finger_impedance_multiplier=float(finger_impedance_multiplier)
        self.finger_velocity_damping=float(finger_velocity_damping)
        if not np.isfinite([self.finger_impedance_multiplier,self.finger_velocity_damping]).all() or not 1<=self.finger_impedance_multiplier<=4 or not 0<=self.finger_velocity_damping<=.1:
            raise ValueError('Declare bounded finger controller gains within1..4 and0..0.1Nm*s/rad')
        # This changes controller feedback only, never active-plant gains,
        # transmissions, passive mechanics or the original motor force caps.
        finger=np.array([n.startswith('rh_') and 'WRJ' not in n for n in self.balance.actions])
        self.finger_motors=finger
        self.original_constant_bias=self.balance.bias[:,0].copy()
        self.balance.kp[finger]*=self.finger_impedance_multiplier
        self.balance.bias[finger,1]*=self.finger_impedance_multiplier
        self.balance.bias[finger,2]-=self.finger_velocity_damping
        self.reflex=None
        if tactile_reflex_profile is not None:
            from .tactile_grasp_reflex import TactileGraspReflex, PROFILE
            if tactile_reflex_profile!=PROFILE:raise ValueError('Unknown tactile reflex profile')
            limits={n:self.balance.m.jnt_range[j].copy() for n,j in zip(self.goal_names,self.joint_ids)}
            self.reflex=TactileGraspReflex(sensor_layout,limits,physics_dt_s=physics_dt_s)
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
        self.balance.bias[:,0]=self.original_constant_bias
        self.last_goals=self.balance.desired[self.indices].copy()
        self.failed_reason=None;self.info={}
        if self.reflex is not None:self.reflex.reset()

    def force(self, packet, *, now_s, joint_goals=None):
        """Return61 capped forces from sensors and explicitly scoped joint targets.

        Omit goals to hold the last accepted target (initially stationary). Explicit
        goals require every declared name, original joint bounds, a matching t0
        posture, and bounded change per2ms call. Rejections require reset.
        """
        if self.failed_reason is not None:raise RuntimeError('Arm inference requires reset_episode: '+self.failed_reason)
        try:
            b=self.balance
            if self.reflex is not None:
                if joint_goals is None:raise ValueError('Tactile reflex requires explicit nominal targets every tick')
                joint_goals=self.reflex.apply(packet,now_s,joint_goals)
            if joint_goals is None:goals=self.last_goals.copy()
            else:
                if type(joint_goals) is not dict or set(joint_goals)!=set(self.goal_names):
                    raise ValueError('Arm command must contain exactly the declared right-upper-body names')
                if any(type(v) not in (int,float,np.float32,np.float64) for v in joint_goals.values()):
                    raise ValueError('Arm goals must be scalar numeric joint angles')
                goals=np.array([joint_goals[n] for n in self.goal_names],float)
            for digit in ('FF','MF','RF','LF'):
                if goals[self.goal_names.index('rh_'+digit+'J1')]>goals[self.goal_names.index('rh_'+digit+'J2')]+1e-7:
                    raise ValueError('Right-hand goal violates passive J1<=J2 anatomy')
            limits=b.m.jnt_range[self.joint_ids]
            if not np.isfinite(goals).all() or np.any(goals<limits[:,0]) or np.any(goals>limits[:,1]):
                raise ValueError('Arm goal exceeds the original authored joint bounds')
            bound=1e-8 if b.last_time is None else self.maximum_goal_speed_radps*b.dt+1e-8
            if np.max(abs(goals-self.last_goals))>bound:raise ValueError('Arm command exceeds initial calibration or goal slew bound')
            q=b.desired.copy();q[self.indices]=goals
            target=b.matrix@q
            if np.any(target[self.arm_motors]<b.limits[self.arm_motors,0]) or np.any(target[self.arm_motors]>b.limits[self.arm_motors,1]):
                raise ValueError('Arm command exceeds native actuator control limits')
            target_velocity=np.zeros(61) if b.last_time is None else (target-b.target)/b.dt
            b.bias[:,0]=self.original_constant_bias
            if self.finger_target_velocity_damping:
                b.bias[self.finger_motors,0]+=self.finger_velocity_damping*target_velocity[self.finger_motors]
            b.target[self.arm_motors]=target[self.arm_motors]
            force,info=b.force(packet,now_s=now_s)
            self.last_goals=goals.copy()
            self.info=dict(info,controller='sensor_reach_balance_v1',high_level_input='explicit right-hand/arm joint commands with declared torso yaw',torso_yaw_enabled=self.allow_torso_yaw,
                goal_joint_names=list(self.goal_names),goal_joint_position_rad=goals.tolist(),
                goal_motor_names=[b.actions[i] for i in self.arm_motors],goal_motor_coordinates=b.target[self.arm_motors].tolist(),
                coupled_joint_goals_are_nominal=True,finger_impedance_multiplier=self.finger_impedance_multiplier,
                finger_velocity_damping_Nm_s_per_rad=self.finger_velocity_damping,
                finger_target_velocity_damping=self.finger_target_velocity_damping,
                goal_motor_velocity_radps=target_velocity[self.arm_motors].tolist(),
                unchanged_balance_targets='calibrated pelvis/legs and left upper body; body estimate from encoders/IMU/touch only')
            if self.reflex is not None:self.info['tactile_grasp_reflex']=dict(self.reflex.info)
            return force,dict(self.info)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc)
            raise

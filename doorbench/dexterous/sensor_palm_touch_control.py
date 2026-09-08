"""Opposing local tactile feedback drives a tiny own-robot palm translation."""
import numpy as np
from .sensor_contract import validate_actor_packet
from .sensor_index_touch_control import SensorIndexTouchController
from .robot_palm_translation import RobotPalmTranslation


PALM_PROTOCOL={
    'schema':'doorbench.sensor-palm-touch.v1',
    'start_after_s':19.,
    'maximum_translation_m':.00025,
    'integral_gain_m_per_s_per_normalized_load':.00005,
    'maximum_translation_rate_mps':.0001,
    'finger_normalization_N':2.,
    'thumb_normalization_N':3.,
    'load_error':'thumb/3 - mean(four_fingers)/2',
    'direction':'mean own-robot distal palmar normals',
    'feedback_epoch':'preceding validated decision filtered distal projection',
    'base_profile':'doorbench.sensor-index-touch.v1',
    'duration_s':36.,
}


def validate_palm_protocol(protocol):
    if type(protocol) is not dict or set(protocol)!=set(PALM_PROTOCOL):raise ValueError('Exact declared palm coordination profile required')
    for key,expected in PALM_PROTOCOL.items():
        value=protocol[key]
        if isinstance(expected,float):
            if type(value) not in (int,float) or not np.isfinite(value) or value!=expected:raise ValueError('Unsupported palm coordination parameter: '+key)
        elif type(value) is not type(expected) or value!=expected:raise ValueError('Unsupported palm coordination parameter: '+key)
    return dict(protocol)


class _PalmGoalSchedule:
    def __init__(self,original,translator):
        self.original=original;self.translator=translator;self.reset()
    def __getattr__(self,name):return getattr(self.original,name)
    def reset(self):self.offset=0.;self.joints=None;self.cache={};self.info={}
    def goals(self,t):
        goals=dict(self.original.goals(t))
        if t<19.-1e-9 or self.offset==0.:return goals
        key=(float(t),tuple(goals.items()))
        if key not in self.cache:
            self.cache[key]=self.translator.goals(goals,self.joints,self.offset)
        result,self.info=self.cache[key]
        return dict(result)


class SensorPalmTouchController(SensorIndexTouchController):
    def __init__(self,arm_controller,operation_schedule,sensor_layout,
                 impedance_protocol,motor_contract,index_protocol,palm_protocol):
        self.palm_protocol=validate_palm_protocol(palm_protocol)
        super().__init__(arm_controller,operation_schedule,sensor_layout,impedance_protocol,motor_contract,index_protocol)
        self.palm_translator=RobotPalmTranslation(self.arm.balance.m,self.joint_names)
        self._palm_schedule=_PalmGoalSchedule(self.schedule,self.palm_translator)
        self.schedule=self._palm_schedule
        self.palm_delta=0.

    def reset_episode(self):
        super().reset_episode();self.palm_delta=0.
        if hasattr(self,'_palm_schedule'):self._palm_schedule.reset()

    def force(self,packet,*,now_s):
        if self.failed_reason is not None:raise RuntimeError('Palm controller requires reset_episode: '+self.failed_reason)
        try:
            validate_actor_packet(packet,self.shapes,61)
            if isinstance(now_s,(bool,np.bool_)) or not np.isfinite(now_s) or now_s<0:raise ValueError('Finite nonnegative numeric clock required')
            now=float(now_s);error=None;used_decision=None
            if now>19.+1e-9:
                if self.last_time is None or abs(now-self.last_time-self.dt)>1e-8:raise ValueError('Exact2ms episode clock required')
                error=float(self.filtered[4]/3.-np.mean(self.filtered[:4])/2.)
                used_decision=self.last_time
                self.palm_delta=float(np.clip(self.palm_delta+self.dt*np.clip(.00005*error,-.0001,.0001),0.,.00025))
            s=self._palm_schedule;s.offset=self.palm_delta;s.joints=packet['joint_position'].copy();s.cache={};s.info={}
            force,info=super().force(packet,now_s=now)
            self.info=dict(info,high_level_controller='local_palm_coordination_v1',
                palm_translation_m=self.palm_delta,palm_opposing_normalized_load_error=error,palm_feedback_decision_s=used_decision,
                palm_translation_kinematics=dict(s.info),
                palm_command_scope='own-robot encoder FK/Jacobian, static arm goals, and previous local tactile loads; no object/root/world inputs')
            return force,dict(self.info)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc);raise

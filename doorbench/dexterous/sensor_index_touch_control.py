"""Opt-in index proximal coordination using only prior local tactile feedback.

This preserves the first19s acquisition, original touch progression gates and
smooth finger stiffness profile. It adds one bounded FFJ3 joint-goal offset.
"""
import numpy as np
from .sensor_contract import validate_actor_packet
from .sensor_distal_touch_impedance import SensorDistalTouchImpedanceController


INDEX_PROTOCOL = {
    'schema': 'doorbench.sensor-index-touch.v1',
    'joint_name': 'rh_FFJ3',
    'start_after_s': 19.,
    'minimum_offset_rad': 0.,
    'maximum_offset_rad': .01,
    'integral_gain_rad_per_N_s': .005,
    'maximum_offset_rate_radps': .01,
    'load_target_N': 2.,
    'feedback_epoch': 'preceding validated decision filtered distal projection',
    'base_profile': 'doorbench.sensor-distal-touch-impedance.v1',
    'duration_s': 36.,
}


def validate_index_protocol(protocol):
    if type(protocol) is not dict or set(protocol) != set(INDEX_PROTOCOL):
        raise ValueError('Exact declared index coordination profile required')
    for key, expected in INDEX_PROTOCOL.items():
        value=protocol[key]
        if isinstance(expected,float):
            if type(value) not in (int,float) or not np.isfinite(value) or value!=expected:
                raise ValueError('Unsupported index coordination parameter: '+key)
        elif type(value) is not type(expected) or value!=expected:
            raise ValueError('Unsupported index coordination parameter: '+key)
    return dict(protocol)


class _IndexGoalSchedule:
    """Static joint route plus one numeric offset; no scene or plant state."""
    def __init__(self, schedule):self.original=schedule;self.offset=0.
    def __getattr__(self,name):return getattr(self.original,name)
    def goals(self,t):
        goals=self.original.goals(t)
        if t>=19.-1e-9:goals['rh_FFJ3']+=self.offset
        return goals


class SensorIndexTouchController(SensorDistalTouchImpedanceController):
    def __init__(self,arm_controller,operation_schedule,sensor_layout,
                 impedance_protocol,motor_contract,index_protocol):
        self.index_protocol=validate_index_protocol(index_protocol)
        if 'rh_FFJ3' not in arm_controller.goal_names:
            raise ValueError('Independent index proximal joint must be commanded')
        self._index_schedule=_IndexGoalSchedule(operation_schedule)
        super().__init__(arm_controller,self._index_schedule,sensor_layout,
                         impedance_protocol,motor_contract)

    def reset_episode(self):
        self.index_delta=0.;self._index_schedule.offset=0.
        super().reset_episode()

    def force(self,packet,*,now_s):
        if self.failed_reason is not None:
            raise RuntimeError('Index controller requires reset_episode: '+self.failed_reason)
        try:
            validate_actor_packet(packet,self.shapes,61)
            if isinstance(now_s,(bool,np.bool_)) or not np.isfinite(now_s) or now_s<0:
                raise ValueError('Finite nonnegative numeric clock required')
            now=float(now_s)
            used_load=None;used_decision=None
            if now>19.+1e-9:
                if self.last_time is None or abs(now-self.last_time-self.dt)>1e-8:
                    raise ValueError('Exact2ms episode clock required')
                # One decision of additional delay is explicit. This is the
                # previous accepted local filtered projection, never a current
                # evaluator contact count, object identity or geometric force.
                used_load=float(self.filtered[0]);used_decision=self.last_time
                rate=float(np.clip(.005*(2.-used_load),-.01,.01))
                self.index_delta=float(np.clip(self.index_delta+self.dt*rate,0.,.01))
            self._index_schedule.offset=self.index_delta
            force,info=super().force(packet,now_s=now)
            self.info=dict(info,high_level_controller='local_index_coordination_v1',
                index_proximal_offset_rad=self.index_delta,
                index_feedback_projection_N=used_load,index_feedback_decision_s=used_decision,
                index_command_scope='rh_FFJ3 only, scripted route plus bounded local touch coordination')
            return force,dict(self.info)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc)
            raise

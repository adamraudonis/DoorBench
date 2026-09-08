"""Bounded local tactile force correction through original coupled motors."""
import numpy as np
from .sensor_contract import validate_actor_packet
from .sensor_index_touch_control import SensorIndexTouchController
from .robot_digit_force import RobotDigitForce


FORCE_PROTOCOL={
    'schema':'doorbench.sensor-digit-force.v1',
    'start_after_s':19.,'ramp_seconds':2.,'ramp_shape':'quintic-smoothstep',
    'proportional_gain':.5,'integral_gain_per_s':1.,
    'maximum_integral_correction_N':2.,'maximum_virtual_correction_N':2.,
    'maximum_virtual_force_rate_Nps':2.,'full_contact_projection_N':.2,
    'targets_N':[2.,2.,2.,2.,3.],
    'direction':'calibrated distal tactile site minus-Z; positive palmar load gates correction',
    'transmission':'least-squares projection onto original finger motors; unavailable J1/J2 difference recorded',
    'base_profile':'doorbench.sensor-index-touch.v1','duration_s':36.,
}


def validate_force_protocol(protocol):
    if type(protocol) is not dict or set(protocol)!=set(FORCE_PROTOCOL):raise ValueError('Exact declared digit-force profile required')
    for key,expected in FORCE_PROTOCOL.items():
        value=protocol[key]
        if isinstance(expected,float):
            if type(value) not in (int,float) or not np.isfinite(value) or value!=expected:raise ValueError('Unsupported force parameter: '+key)
        elif isinstance(expected,list):
            if type(value) is not list or len(value)!=5 or any(type(x) not in (int,float) for x in value) or value!=expected:raise ValueError('Original local load targets required')
        elif type(value) is not type(expected) or value!=expected:raise ValueError('Unsupported force parameter: '+key)
    return dict(protocol)


class SensorDigitForceController(SensorIndexTouchController):
    def __init__(self,arm_controller,operation_schedule,sensor_layout,
                 impedance_protocol,motor_contract,index_protocol,force_protocol):
        self.force_protocol=validate_force_protocol(force_protocol)
        self._base_motor_bias=arm_controller.original_constant_bias.copy()
        super().__init__(arm_controller,operation_schedule,sensor_layout,impedance_protocol,motor_contract,index_protocol)
        b=self.arm.balance
        self.pad_force=RobotDigitForce(b.m,self.joint_names,self.action_names,b.matrix)

    def reset_episode(self):
        if hasattr(self,'arm'):self.arm.original_constant_bias[:]=self._base_motor_bias
        super().reset_episode()
        self.force_integral=np.zeros(5);self.virtual_force=np.zeros(5)

    def force(self,packet,*,now_s):
        if self.failed_reason is not None:raise RuntimeError('Digit force controller requires reset_episode: '+self.failed_reason)
        try:
            validate_actor_packet(packet,self.shapes,61)
            if isinstance(now_s,(bool,np.bool_)) or not np.isfinite(now_s) or now_s<0:raise ValueError('Finite nonnegative numeric clock required')
            now=float(now_s);bias=np.zeros(61);projection={};applied=np.zeros(5);weight=np.zeros(5);error=None;ramp=0.
            if now>=19.-1e-9:
                if self.last_time is None or abs(now-self.last_time-self.dt)>1e-8:raise ValueError('Exact2ms episode clock required')
                if not packet['sensor_valid'][4] or not 0<=now-packet['sensor_time_s'][4]<=.006+1e-9:raise ValueError('Fresh local tactile contact required')
                raw=self.local_distal_loads(packet);weight=np.clip(raw/.2,0.,1.)
                filtered=raw if self.last_time<19.-1e-9 else self.filtered.copy()
                error=self.target-filtered
                proposed=np.clip(self.force_integral+self.dt*error,-2.,2.)
                # Clamped anti-windup and contact loss reset. The local weight
                # is zero at zero measured palmar load; no object ID is used.
                total=.5*error+proposed
                admissible=(abs(total)<=2.)|(total*error<0.)
                self.force_integral=np.where(weight>0,np.where(admissible,proposed,self.force_integral),0.)
                desired=np.where(weight>0,np.clip(.5*error+self.force_integral,-2.,2.),0.)
                self.virtual_force+=np.clip(desired-self.virtual_force,-2.*self.dt,2.*self.dt)
                u=float(np.clip((now-19.)/2.,0.,1.));ramp=u**3*(10.+u*(-15.+6.*u))
                applied=weight*ramp*self.virtual_force
                bias,projection=self.pad_force.motor_bias(packet['joint_position'],applied)
            # This is a policy feedforward term, before the existing native
            # motor cap and previous-action update. No returned force is edited.
            self.arm.original_constant_bias[:]=self._base_motor_bias+bias
            force,info=super().force(packet,now_s=now)
            self.info=dict(info,high_level_controller='local_digit_force_v1',
                local_contact_weight=weight.tolist(),digit_force_error_N=None if error is None else error.tolist(),
                digit_force_integral_N=self.force_integral.tolist(),virtual_digit_force_state_N=self.virtual_force.tolist(),
                applied_virtual_digit_force_N=applied.tolist(),digit_force_ramp=ramp,
                requested_additional_motor_bias_Nm=bias.tolist(),digit_actuation_projection=projection,
                force_feedback_scope='local finite palmar projection + encoders; original coupled motors; no object/scene/root inputs')
            return force,dict(self.info)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc);raise

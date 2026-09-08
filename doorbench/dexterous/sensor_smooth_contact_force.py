"""Opt-in smooth pressure-mode recovery; whole-thumb projection stays unchanged.

This isolates a contact-switching repair from the separate thumb-subspace issue.
Only validated sensor packets, fixed robot calibration and local clocks enter.
"""
import numpy as np
from .sensor_contract import validate_actor_packet
from .sensor_index_touch_control import SensorIndexTouchController
from .sensor_hierarchical_digit_force import SensorHierarchicalDigitForceController
from .tactile_contact_mode import TactileContactMode,validate_mode_protocol


class SensorSmoothContactForceController(SensorHierarchicalDigitForceController):
    def __init__(self,*args,contact_mode_protocol,**kwargs):
        self.contact_mode_protocol=validate_mode_protocol(contact_mode_protocol)
        self.contact_mode=TactileContactMode(contact_mode_protocol)
        super().__init__(*args,**kwargs)

    def reset_episode(self):
        super().reset_episode();self.contact_mode.reset()

    def normal_contact_weight(self,packet):
        return self.contact_mode.weight.copy()

    def closure_integration_mask(self,raw):
        return raw>0.

    def progression_contact_ready(self,raw):
        return bool(np.all(raw>=.2))

    def index_integration_allowed(self,packet):
        return bool(self.local_distal_loads(packet)[0]>0.)

    def force(self,packet,*,now_s):
        if self.failed_reason is not None:raise RuntimeError('Smooth force controller requires reset_episode: '+self.failed_reason)
        try:
            validate_actor_packet(packet,self.shapes,61)
            if isinstance(now_s,(bool,np.bool_)) or not np.isfinite(now_s) or now_s<0:
                raise ValueError('Finite nonnegative numeric clock required')
            now=float(now_s);bias=np.zeros(61);projection={};applied=np.zeros(5);weight=np.zeros(5);error=None;ramp=0.
            mode={}
            if now>=19.-1e-9:
                if self.last_time is None or abs(now-self.last_time-self.dt)>1e-8:raise ValueError('Exact2ms episode clock required')
                if not packet['sensor_valid'][4] or not 0<=now-packet['sensor_time_s'][4]<=.006+1e-9:
                    raise ValueError('Fresh local tactile contact required; mode retention never retains a stale packet')
                raw=self.local_distal_loads(packet);weight,mode=self.contact_mode.update(raw,now_s=now)
                filtered=raw if self.last_time<19.-1e-9 else self.filtered.copy();error=self.target-filtered
                proposed=np.clip(self.force_integral+self.dt*error,-2.,2.)
                total=.5*error+proposed;admissible=(abs(total)<=2.)|(total*error<0.)
                # Preserve established integral and virtual effort during a
                # brief fresh unload; no unseen error is integrated. Mode
                # weight alone makes the bounded transition back to posture.
                self.force_integral=np.where(raw>0,np.where(admissible,proposed,self.force_integral),self.force_integral)
                desired=np.where(raw>0,np.clip(.5*error+self.force_integral,-2.,2.),self.virtual_force)
                self.virtual_force+=np.clip(desired-self.virtual_force,-2.*self.dt,2.*self.dt)
                u=float(np.clip((now-19.)/2.,0.,1.));ramp=u**3*(10.+u*(-15.+6.*u))
                applied=weight*ramp*self.virtual_force
                bias,projection=self.pad_force.motor_bias(packet['joint_position'],applied)
            self.arm.original_constant_bias[:]=self._base_motor_bias+bias
            # Skip the old instantaneous-force wrapper. The same index,
            # impedance, touch and original capped arm adapters remain in use.
            force,info=SensorIndexTouchController.force(self,packet,now_s=now)
            self.info=dict(info,high_level_controller='smooth_contact_hierarchy_v1',
                local_contact_weight=weight.tolist(),digit_force_error_N=None if error is None else error.tolist(),
                digit_force_integral_N=self.force_integral.tolist(),virtual_digit_force_state_N=self.virtual_force.tolist(),
                applied_virtual_digit_force_N=applied.tolist(),digit_force_ramp=ramp,
                requested_additional_motor_bias_Nm=bias.tolist(),digit_actuation_projection=projection,
                **mode,force_feedback_scope='fresh local touch plus encoders; smooth mode is not a physical contact claim')
            return force,dict(self.info)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc);raise

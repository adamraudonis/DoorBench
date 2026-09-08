"""Bounded contact-mode history from fresh, finite local touch projections only.

Retained mode is a controller state, never a claim that contact still exists.
The caller must independently admit packet freshness and original physical gates.
"""
import numpy as np


MODE_PROTOCOL={
    'schema':'doorbench.sensor-contact-mode.v1',
    'start_after_s':19.,'physics_dt_s':.002,
    'full_contact_projection_N':.2,
    'zero_load_retention_s':.02,'maximum_weight_rate_per_s':5.,
    'maximum_zero_load_recovery_s':.25,
    'zero_load_action':'freeze progression and all preload integrators; retain then slew mode toward posture',
    'timeout_action':'terminal controller failure; no inferred contact or automatic continuation',
    'thumb_projection':'unchanged whole-thumb pressure direction',
    'base_profile':'doorbench.sensor-hierarchical-digit-force.v1','duration_s':36.,
}

THUMB_MODE_PROTOCOL=dict(MODE_PROTOCOL,
    schema='doorbench.sensor-contact-mode-thumb-flexion.v1',
    thumb_projection='THJ1/THJ2 pressure; THJ3/THJ4/THJ5 original posture')


def validate_mode_protocol(protocol):
    if type(protocol) is not dict or set(protocol)!=set(MODE_PROTOCOL):
        raise ValueError('Exact declared contact-mode profile required')
    expected=THUMB_MODE_PROTOCOL if protocol.get('schema')==THUMB_MODE_PROTOCOL['schema'] else MODE_PROTOCOL
    for key,wanted in expected.items():
        value=protocol[key]
        if type(wanted) is float:
            if type(value) not in (int,float) or not np.isfinite(value) or value!=wanted:
                raise ValueError('Unsupported contact-mode parameter: '+key)
        elif type(value) is not type(wanted) or value!=wanted:
            raise ValueError('Unsupported contact-mode parameter: '+key)
    return dict(protocol)


class TactileContactMode:
    def __init__(self,protocol):
        self.protocol=validate_mode_protocol(protocol);self.reset()

    def reset(self):
        self.time=None;self.weight=np.zeros(5);self.loss_started=np.full(5,np.nan)
        self.failed_reason=None;self.info={}

    def update(self,loads,*,now_s):
        if self.failed_reason is not None:raise RuntimeError('Contact mode requires reset: '+self.failed_reason)
        try:
            if isinstance(now_s,(bool,np.bool_)) or type(now_s) not in (int,float,np.float32,np.float64) or not np.isfinite(now_s):
                raise ValueError('Finite scalar numeric clock required')
            now=float(now_s);raw=np.asarray(loads,float)
            if raw.shape!=(5,) or not np.isfinite(raw).all() or np.any(raw<0):
                raise ValueError('Five finite nonnegative local loads required')
            if (self.time is None and abs(now-19.)>1e-9) or (self.time is not None and abs(now-self.time-.002)>1e-8):
                raise ValueError('Exact2ms mode clock starting at19s required')
            present=raw>0.;requested=np.clip(raw/.2,0.,1.)
            loss=self.loss_started.copy();loss[present]=np.nan
            loss[(~present)&np.isnan(loss)]=now
            duration=np.where(present,0.,now-loss)
            if np.any(duration>=.25-1e-9):
                raise ValueError('Fresh zero-touch recovery exceeded250ms')
            retained=(~present)&(duration<.02-1e-9)
            target=np.where(retained,self.weight,requested)
            if self.time is None:
                # Outer force transfer is exactly zero at19s. Initializing its
                # mode here therefore produces no initial effort discontinuity.
                weight=requested.copy()
            else:weight=self.weight+np.clip(target-self.weight,-.01,.01)
            self.weight=weight;self.loss_started=loss;self.time=now
            self.info=dict(raw_contact_projection_weight=requested.tolist(),retained_contact_mode=retained.tolist(),
                contact_mode_weight=weight.tolist(),zero_load_duration_s=duration.tolist(),
                pressure_integration_allowed=present.tolist(),immediate_contact_progression_ready=bool(np.all(raw>=.2)),
                scope='Retained pressure mode is not evidence of continued physical touch')
            return weight.copy(),dict(self.info)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc);raise

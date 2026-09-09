"""Bounded four-finger preload using only local distal touch cells.

An opt-in scripted acquisition correction, not a learned visual policy. It
starts after the declared approach ends; no object identity or geometry enters.
Independent contact anatomy checks still decide whether a grasp is acceptable.
"""
import numpy as np
import json
from pathlib import Path
from .sensor_contract import SENSOR_KEYS

PROFILE = 'four-finger-preload-v1'
PROFILES={'four-finger-preload-v1':.4,'four-finger-preload-v2':1.5,'four-finger-preload-v3':.8}
INPUT_KEYS=('tactile','sensor_time_s','sensor_valid')


def archived_reflex_inputs(run, *, engine):
    """Read actual decision-epoch touch, never the later endpoint as input."""
    run=Path(run)
    if engine=='native':
        layout=json.loads((run/'sensor-layout.json').read_text())
        with np.load(run/'actor-inputs.npz',allow_pickle=False) as z:
            packets={k:z[k].copy() for k in INPUT_KEYS}
    elif engine=='isaac':
        layout=json.loads((run/'sensors/layout.json').read_text())
        with np.load(run/'sensors/actor-initial-decision.npz',allow_pickle=False) as initial, np.load(run/'sensors/actor-sensors.npz',allow_pickle=False) as z:
            packets={k:np.concatenate([initial[k][None],z[k][:-1]]) for k in INPUT_KEYS}
    else:raise ValueError('Unknown recorded sensor engine')
    return layout,packets


class TactileGraspReflex:
    def __init__(self, layout, joint_limits, *, physics_dt_s=.002, profile=PROFILE):
        if profile not in PROFILES:raise ValueError('Unknown declared distal pressure profile')
        self.profile=profile;self.target_load=PROFILES[profile]
        if physics_dt_s != .002 or layout['channel_order'] != ['z', 'x', 'y']:
            raise ValueError('Require the original 2ms local normal-first tactile contract')
        self.dt=physics_dt_s;self.limits=dict(joint_limits);self.cells={};offset=0
        for sensor in layout['sensors']:
            for digit in ('ff','mf','rf','lf'):
                if sensor['body_name']=='rh_'+digit+'distal':
                    count=sensor['width']*sensor['height']
                    if sensor['dimension'] != count*3 or digit in self.cells:
                        raise ValueError('Invalid distal tactile grid')
                    self.cells[digit]=slice(offset,offset+count)
            offset+=sensor['dimension']
        if set(self.cells)!={'ff','mf','rf','lf'} or offset!=layout['tactile_dimension']:
            raise ValueError('Require all four original distal sensors')
        self.dimension=offset;self.reset()

    def reset(self):
        self.offsets={d:0. for d in self.cells};self.filtered={d:0. for d in self.cells}
        self.last_time=None;self.info={}

    def apply(self, packet, now_s, nominal):
        if not np.isfinite(now_s) or now_s<0 or (self.last_time is not None and abs(now_s-self.last_time-self.dt)>1e-8):
            raise ValueError('Reflex requires consecutive episode-local decisions')
        touch=np.asarray(packet['tactile']);stream=SENSOR_KEYS.index('tactile')
        if touch.shape!=(self.dimension,) or not np.isfinite(touch).all():
            raise ValueError('Finite local touch cells required')
        valid=bool(packet['sensor_valid'][stream]);age=now_s-float(packet['sensor_time_s'][stream])
        if valid and not 0<=age<=.004+1e-8:
            raise ValueError('Reflex cannot use stale or future touch')
        result=dict(nominal);active=now_s>=17. and valid
        for digit,cells in self.cells.items():
            load=float(np.maximum(0.,touch[cells]).sum()) if valid else None
            if valid:self.filtered[digit]+=self.dt/(.04+self.dt)*(load-self.filtered[digit])
            if active:
                error=self.target_load-self.filtered[digit]
                rate=0. if abs(error)<=.05 else float(np.clip(.2*error,-.08,.08))
                names=['rh_'+digit.upper()+'J'+j for j in ('1','2')]
                allowed=min(.08,*(2*(self.limits[n][1]-1e-4-float(nominal[n])) for n in names))
                self.offsets[digit]=float(np.clip(self.offsets[digit]+rate*self.dt,0.,max(0.,allowed)))
            for joint in ('1','2'):
                name='rh_'+digit.upper()+'J'+joint
                result[name]=float(nominal[name])+self.offsets[digit]/2
        self.last_time=float(now_s)
        self.info=dict(profile=self.profile,active=active,filtered_distal_normal_load_N=dict(self.filtered),
            coupled_motor_preload_rad=dict(self.offsets),target_normal_load_N=self.target_load,
            maximum_coupled_preload_rad=.08,maximum_coupled_slew_radps=.08,
            dynamic_inputs='local distal tactile cells and validity; nominal scripted joint targets and local clock',
            object_geometry_input=False,thumb_target_changed=False)
        return result

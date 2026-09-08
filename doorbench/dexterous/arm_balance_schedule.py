"""Explicit bounded high-level arm schedule, independent of physics and sensors.

These scripted joint goals are not a learned/vision policy or an observation.
The separate sensor-feedback controller enforces native joint/control bounds.
"""
import numpy as np


def validate_schedule(schedule,names,duration):
    wanted={'schema','scope','start_s','outward_seconds','hold_seconds','return_seconds','duration_s','deltas_rad'}
    if type(schedule) is not dict or set(schedule)!=wanted or schedule['schema']!='doorbench.scripted-arm-balance.v1':raise ValueError('Explicit scripted arm schedule required')
    if type(schedule['deltas_rad']) is not dict or set(schedule['deltas_rad'])-set(names):raise ValueError('Script may move only arm/wrist joints')
    values=[schedule[k] for k in ('start_s','outward_seconds','hold_seconds','return_seconds','duration_s')]
    if not np.isfinite(values).all() or min(values)<=0 or duration!=schedule['duration_s'] or not 0<duration<=10 or sum(values[:4])+1>duration:raise ValueError('Schedule requires positive phases and at least1s final quiet')
    if not np.isfinite(list(schedule['deltas_rad'].values())).all() or any(abs(v)>.3 for v in schedule['deltas_rad'].values()):raise ValueError('This schedule only supports modest bounded arm motions')
    if any(1.875*abs(v)/min(schedule['outward_seconds'],schedule['return_seconds'])>.5 for v in schedule['deltas_rad'].values()):raise ValueError('Quintic goal slew exceeds0.5rad/s')


def scripted_goals(t,desired,names,schedule):
    if not np.isscalar(t) or not np.isfinite(t) or t<0:raise ValueError('Use a finite nonnegative local schedule clock')
    local=t-schedule['start_s'];outward=schedule['outward_seconds'];hold=schedule['hold_seconds'];returning=schedule['return_seconds']
    if local<=0:u=0.
    elif local<=outward:u=local/outward
    elif local<=outward+hold:u=1.
    else:u=max(0.,1-(local-outward-hold)/returning)
    blend=u**3*(10+u*(-15+6*u))
    return {name:float(desired[name]+blend*schedule['deltas_rad'].get(name,0.)) for name in names}


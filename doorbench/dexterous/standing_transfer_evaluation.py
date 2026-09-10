"""Streaming checks for measured Isaac standing-transfer surface support."""
import math
import numpy as np
from .isaac_opening_measurements import pose_parts


def standing_transfer_checks(rows,*,seconds,dt,started_s):
    if not all(math.isfinite(x) and x>0 for x in (seconds,dt)):raise ValueError('Positive finite clocks required')
    count=0;clock=True;loads_match=True;stance=True;tail=0;support=True
    for row in rows:
        count+=1;t=row['time_s'];clock &= math.isfinite(t) and abs(t-count*dt)<1e-8
        _,rotation=pose_parts(row['leaf_pose']);surface=row['surface']
        vector=np.asarray(surface['body_panel_forces_world_N']['lh_palm'],float)
        if vector.shape!=(3,) or not np.isfinite(vector).all():raise ValueError('Finite measured palm force required')
        actual=max(0.,float(-rotation[:,1]@vector));reported=surface['palm_normal_load_N']
        loads_match &= math.isfinite(reported) and abs(actual-reported)<1e-8
        stance &= row['stance_status'] in ('solved','solved inaccurate')
        if t>=seconds-.5-1e-8:
            tail+=1;support &= actual>=2.
    return dict(complete_transfer_clock=bool(clock and count==round(seconds/dt)),
        standing_transfer_started=bool(started_s is not None and math.isfinite(started_s) and 0<started_s<=seconds-.5),
        measured_palm_load_accounting=bool(loads_match),
        final_left_palm_support=bool(tail>=round(.5/dt) and support),
        stance_solves_every_interval=bool(stance))

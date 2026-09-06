"""Electrical coil force on a physical track-closer detent plunger only."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TrackHold:
    name:str
    plunger_qpos:int
    plunger_dof:int
    button_qpos:int
    button_threshold:float
    force:float
    gap:float
    powered:bool


def compile_track_holds(model,metadata):
    result=[]
    for row in metadata.get('closer_track_holds',[]):
        j=model.joint(row['plunger_joint']).id;b=model.joint(row['button_joint']).id
        force=float(row['coil_force_at_seat_N']);gap=float(row['magnetic_gap_scale_m']);threshold=float(row['button_release_threshold_m'])
        if not np.isfinite([force,gap,threshold]).all() or force<=0 or gap<=0 or not 0<threshold<float(model.jnt_range[b,1]):raise ValueError('Invalid track solenoid field or release threshold')
        result.append(TrackHold(row['plunger_joint'],int(model.jnt_qposadr[j]),int(model.jnt_dofadr[j]),int(model.jnt_qposadr[b]),threshold,force,gap,bool(row['powered_by_default'])))
    return tuple(result)


def track_coil_force(rule,plunger_q,button_q,powered=None):
    """Signed force for the authored seat-gap and physical switch state."""
    on=rule.powered if powered is None else (bool(powered.get(rule.name,rule.powered)) if isinstance(powered,dict) else bool(powered))
    if not on or button_q>=rule.button_threshold:return 0.
    return -rule.force/(1.+max(0.,float(plunger_q))/rule.gap)**2


def apply_track_holds(model,data,rules,powered=None):
    """Add gap-dependent attraction; all holding load passes native cam contact.

    powered may be a bool or per-plunger-joint dict. A physical depressed test
    switch always interrupts power. No qpos/model/constraints are modified.
    """
    for r in rules:
        data.qfrc_passive[r.plunger_dof]+=track_coil_force(r,data.qpos[r.plunger_qpos],data.qpos[r.button_qpos],powered)

"""External power supplies only an actual drop-release plunger force."""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class DropRelease:
    name:str
    qpos:int
    dof:int
    stroke:float
    force:float
    gap:float
    powered:bool

def compile_turnstile_drop(model,metadata):
    row=metadata.get('turnstile_drop_arm')
    if not row:return ()
    j=model.joint(row['release_joint']).id
    stroke,force,gap=(float(row[k]) for k in ('release_stroke_m','coil_force_at_seat_N','gap_scale_m'))
    import mujoco
    if (model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_SLIDE or not model.jnt_limited[j]
        or not np.allclose(model.jnt_axis[j],(0,-1,0),atol=1e-9,rtol=0)
        or not all(math.isfinite(x) and x>0 for x in (stroke,force,gap))
        or abs(float(model.jnt_range[j,0]))>1e-9 or abs(float(model.jnt_range[j,1])-stroke)>1e-6
        or not isinstance(row['powered_by_default'],bool)):
        raise ValueError('Invalid native drop-arm solenoid binding')
    return (DropRelease(row['release_joint'],int(model.jnt_qposadr[j]),int(model.jnt_dofadr[j]),stroke,force,gap,bool(row['powered_by_default'])),)

def apply_turnstile_drop(model,data,rules,powered=None):
    for r in rules:
        on=r.powered if powered is None else bool(powered.get(r.name,r.powered)) if isinstance(powered,dict) else bool(powered)
        if on:data.qfrc_passive[r.dof]-=r.force/(1.+max(0.,float(data.qpos[r.qpos]))/r.gap)**2

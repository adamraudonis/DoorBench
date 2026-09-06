"""Ideal electrical input drives an actual native turnstile bolt armature."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class TurnstileLock:
    name:str
    qpos:int
    dof:int
    stroke:float
    force:float
    gap:float
    powered:bool


def compile_turnstile_locks(model,metadata):
    row=metadata.get('turnstile_locks')
    if not row:return ()
    j=model.joint(row['bolt_joint']).id;stroke=float(row['stroke_m']);force=float(row['coil_force_at_seat_N']);gap=float(row['gap_scale_m'])
    if not all(math.isfinite(x) and x>0 for x in (stroke,force,gap)) or abs(float(model.jnt_range[j,1])-stroke)>1e-6:
        raise ValueError('Invalid native turnstile solenoid binding')
    return (TurnstileLock(row['bolt_joint'],int(model.jnt_qposadr[j]),int(model.jnt_dofadr[j]),stroke,force,gap,bool(row['powered_by_default'])),)


def apply_turnstile_locks(model,data,rules,powered=None):
    """Apply force only to the linear armature; native rotor state is untouched."""
    for r in rules:
        on=r.powered if powered is None else bool(powered.get(r.name,r.powered)) if isinstance(powered,dict) else bool(powered)
        if on:
            gap=max(0.,r.stroke-float(data.qpos[r.qpos]));data.qfrc_passive[r.dof]+=r.force/(1.+gap/r.gap)**2

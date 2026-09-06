"""Bounded ideal credential actuator on an actual exterior locking catch."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class RotaryCatch:
    name:str
    qpos:int
    dof:int
    stroke:float
    release_threshold:float
    force:float
    released:bool


def compile_rotary_catches(model,metadata):
    rules=[]
    for row in metadata.get('rotary_locksets',[]):
        j=model.joint(row['catch_joint']).id
        stroke=float(row['catch_stroke_m']);threshold=float(row['released_threshold_m'])
        force=float(row['catch_force_cap_N'])
        if not all(math.isfinite(v) and v>0 for v in (stroke,threshold,force)) or not threshold<stroke:
            raise ValueError('Invalid rotary catch actuation parameters')
        if (int(model.jnt_type[j])!=2 or not model.jnt_range[j,0]<=0<stroke<model.jnt_range[j,1]
                or model.geom(row['catch_geom']).bodyid[0]!=model.jnt_bodyid[j]):
            raise ValueError('Invalid native rotary catch binding')
        rules.append(RotaryCatch(row['catch_joint'],int(model.jnt_qposadr[j]),int(model.jnt_dofadr[j]),
            stroke,threshold,force,bool(row['released_by_default'])))
    return tuple(rules)


def apply_rotary_catches(model,data,rules,released=None):
    """Only actual catch DOF receives force; no handle range or pose writes."""
    for r in rules:
        active=r.released if released is None else bool(released.get(r.name,r.released)) if isinstance(released,dict) else bool(released)
        if active:
            # Constant bounded ideal electromagnetic pull. Actual collar and
            # housing contact carries the terminal reaction; no pose servo.
            data.qfrc_passive[r.dof]+=r.force

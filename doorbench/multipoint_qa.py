"""Repeated native lift/key/depress tests for original multipoint boltwork."""
from __future__ import annotations
import copy
import numpy as np


def _native_cycles(model,metadata):
    import mujoco
    rows=metadata.get('multipoint_locks',[])
    if not rows:return {'ok':True,'applicable':False,'failures':[]}
    m=model;results=[];failures=[]
    # Hardware isolation keeps the original complete door, all collisions and
    # locks. No state or parameter is changed to unlock a stuck mechanism.
    for row in rows:
        d=mujoco.MjData(m)
        names=[row['leaf_joint'],row['lever_joint'],row['drivebar_joint'],row['central_bolt_joint'],row['thumbturn_joint']]+[b['joint'] for b in row['auxiliary']]
        js=[m.joint(n).id for n in names];aa=m.jnt_qposadr[js];vv=m.jnt_dofadr[js]
        latch_names=[m.joint(j).name for j in range(m.njnt) if m.joint(j).name.startswith(row['leaf_body']+'_') and m.joint(j).name.endswith('latch_bolt_slide')]
        released={}
        stroke=row['stroke_m'];mujoco.mj_forward(m,d);depth=0.;stages=[]
        for cycle in range(2):
            for phase,seconds,lever,key in [('unlock',1,0,2),('lift',1.2,-.9,2),('neutral',.6,0,2),('key_lock',1,0,-.2),('blocked',.6,.9,-.2),('unlock',1,0,2),('depress',1.2,.9,2),('release',.6,0,2)]:
                for _ in range(round(seconds/m.opt.timestep)):
                    d.qfrc_applied[:]=0.
                    # A bounded external hand torque at each real spindle;
                    # the internal followers receive no direct actuation.
                    if phase not in ('neutral','release'):
                        d.qfrc_applied[vv[1]]=np.clip(20*(lever-d.qpos[aa[1]])-.3*d.qvel[vv[1]],-3,3)
                    d.qfrc_applied[vv[4]]=np.clip(3*(key-d.qpos[aa[4]])-.15*d.qvel[vv[4]],-.8,.8)
                    mujoco.mj_step(m,d)
                    depth=max(depth,max((max(0.,-float(c.dist)) for c in d.contact),default=0.))
                if phase=='depress':
                    released={n:float(d.qpos[m.jnt_qposadr[m.joint(n).id]]) for n in [*names[1:],*latch_names]}
                q=d.qpos[aa].copy();stages.append({'cycle':cycle,'phase':phase,'qpos':q.tolist()})
                if phase in ('lift','neutral','key_lock') and max(abs(q[2]),*abs(q[5:]))>.001:
                    failures.append({'phase':phase,'cycle':cycle,'reason':'Auxiliary points did not extend and remain extended','qpos':q.tolist()})
                if phase=='key_lock' and q[3]>.001:
                    failures.append({'phase':phase,'reason':'Central bolt could not engage through the aligned bar window'})
                if phase=='blocked' and (abs(q[1])>.06 or q[2]>.001 or max(q[5:])>.0015):
                    failures.append({'phase':phase,'reason':'Locked bolt tongue did not arrest lever and auxiliary bar'})
                if phase in ('depress','release') and min(q[2],*q[5:])<stroke-.001:
                    failures.append({'phase':phase,'reason':'Auxiliary points failed to withdraw and remain withdrawn','qpos':q.tolist()})
                if phase in ('neutral','release') and abs(q[1])>.04:
                    failures.append({'phase':phase,'reason':'Released lever did not return under its own spring','qpos':q.tolist()})
        if depth>.001:failures.append({'reason':'Contact penetration exceeds 1 mm','depth_m':depth})
        if np.any(d.warning.number) or not np.isfinite(d.qpos).all():failures.append({'reason':'Native warning or nonfinite state'})
        results.append({'leaf_joint':row['leaf_joint'],'joint_names':names,'stages':stages,'maximum_contact_depth_m':depth,
                        'released_joints':released,'lever_torque_limit_Nm':3.,'thumbturn_torque_limit_Nm':.8,'warnings':d.warning.number.tolist()})
    return {'ok':not failures,'applicable':True,'failures':failures,'results':results,
            'scope':'Two native lift/key-block/unlock/depress/neutral cycles with bounded external spindle torques; no human grasp or key-insertion claim'}


def run_multipoint_qa(model,metadata):
    """Private native cycles with the installed closer's real pinion law."""
    import mujoco
    from .closer_pinion import compile_pinion_closers,apply_pinion_closers
    native=copy.copy(model);rules=compile_pinion_closers(native,metadata)
    previous=mujoco.get_mjcb_passive()
    def callback(m,d):
        if m is native:apply_pinion_closers(m,d,rules)
    try:
        mujoco.set_mjcb_passive(callback)
        return _native_cycles(native,metadata)
    finally:mujoco.set_mjcb_passive(previous)

"""Finite native force controls for physical turnstile pawls and index bolts."""
from __future__ import annotations
import copy
import math
import numpy as np


def run_turnstile_mount_qa(model,metadata):
    """Every fixed load-carrying component must reach its structural anchor."""
    import mujoco
    row=metadata.get('turnstile_locks')
    if not row:return {'ok':True,'applicable':False,'failures':[]}
    d=mujoco.MjData(model);mujoco.mj_kinematics(model,d)
    names=[row['frame_anchor_geom'],*row['fixed_support_geoms']]
    ids=[model.geom(n).id for n in names];connected={0};edges=[]
    while True:
        extra=set()
        for i in connected:
            for k in range(len(ids)):
                if k in connected or k in extra:continue
                gap=float(mujoco.mj_geomDistance(model,d,ids[i],ids[k],.001,None))
                if gap<=.0005:extra.add(k);edges.append({'geoms':[names[i],names[k]],'gap_m':gap})
        if not extra:break
        connected|=extra
    missing=[names[k] for k in range(len(ids)) if k not in connected]
    return {'ok':not missing,'applicable':True,'failures':[{'detached_supports':missing}] if missing else [],
        'connected_support_count':len(connected)-1,'attachment_edges':edges,
        'scope':'Exact native component-to-anchor continuity within0.5mm; not a structural strength certificate'}


def run_turnstile_lock_qa(model,metadata,*,duration_s=3.,torque_Nm=80.):
    import mujoco
    from .turnstile_locks import compile_turnstile_locks,apply_turnstile_locks
    from .turnstile_drop import compile_turnstile_drop,apply_turnstile_drop
    row=metadata.get('turnstile_locks')
    if not row:return {'ok':True,'applicable':False,'failures':[]}
    m=copy.copy(model);rules=compile_turnstile_locks(m,metadata);drop_rules=compile_turnstile_drop(m,metadata);j=m.joint(row['rotor_joint']).id;a=int(m.jnt_qposadr[j]);v=int(m.jnt_dofadr[j])
    mount=run_turnstile_mount_qa(m,metadata)
    failures=list(mount['failures']);probes=[]
    if m.jnt_limited[j]:failures.append('Physical turnstile must not retain a primary angular range lock')
    previous=mujoco.get_mjcb_passive();powered=False
    def callback(native,data):
        if native is m:
            apply_turnstile_locks(native,data,rules,powered)
            apply_turnstile_drop(native,data,drop_rules,True)
    try:
        mujoco.set_mjcb_passive(callback)
        for powered in (False,True):
            for direction in (-1.,1.):
                d=mujoco.MjData(m);pairs=set();peak_depth=0.;peak_bolt=0.;min_pawl=0.;max_pawl=0.
                for _ in range(round(duration_s/m.opt.timestep)):
                    d.qfrc_applied[v]=0. if d.time<.4 else direction*torque_Nm-10.*d.qvel[v]
                    mujoco.mj_step(m,d)
                    for contact in d.contact:
                        pairs.add(tuple(sorted((m.geom(contact.geom1).name,m.geom(contact.geom2).name))))
                        peak_depth=max(peak_depth,-float(contact.dist))
                    peak_bolt=max(peak_bolt,float(d.qpos[rules[0].qpos]))
                    if row['pawl_joint']:
                        pa=int(m.jnt_qposadr[m.joint(row['pawl_joint']).id]);value=float(d.qpos[pa]);min_pawl=min(min_pawl,value);max_pawl=max(max_pawl,value)
                q=float(d.qpos[a]);bolt=float(d.qpos[rules[0].qpos]);blocked=not powered or (direction<0 and row['one_way'])
                has_bolt_contact=any(row['bolt_geom'] in pair and any(g in row['index_geoms'] for g in pair) for pair in pairs)
                has_tooth_contact=any(row['pawl_tip_geom'] in pair and any(g in row['ratchet_teeth'] for g in pair) for pair in pairs) if row['one_way'] else False
                has_stop_contact=any(row['pawl_stop_geom'] in pair for pair in pairs) if row['one_way'] else False
                item={'powered':powered,'direction':direction,'applied_torque_Nm':torque_Nm,'duration_s':duration_s,'rotor_angle_rad':q,
                    'bolt_travel_m':bolt,'peak_bolt_travel_m':peak_bolt,'pawl_min_max_rad':[min_pawl,max_pawl],
                    'index_bolt_contact':has_bolt_contact,'ratchet_tooth_contact':has_tooth_contact,'pawl_load_stop_contact':has_stop_contact,
                    'max_contact_penetration_m':peak_depth,'native_warnings':d.warning.number.tolist()};probes.append(item)
                if blocked and abs(q)>.06:failures.append({'physical_arrest_failed':item})
                if not blocked and direction*q<row['sector_angle_rad']:failures.append({'released_rotation_failed':item})
                if not powered and not has_bolt_contact:failures.append({'missing_index_load_contact':item})
                if powered and direction<0 and row['one_way'] and not (has_tooth_contact and has_stop_contact):failures.append({'missing_reverse_load_path':item})
                if powered and direction>0 and row['one_way'] and max_pawl<.02:failures.append({'pawl_did_not_lift_over_teeth':item})
                if powered and bolt<row['stroke_m']-.0005:failures.append({'solenoid_did_not_withdraw_bolt':item})
                if peak_bolt>row['stroke_m']+.001 or peak_depth>.001:failures.append({'excess_native_penetration_or_travel':item})
                if np.any(d.warning.number) or not np.isfinite(d.qpos).all():failures.append({'native_warning_or_nonfinite':item})
        return {'ok':not failures,'applicable':True,'failures':failures,'probes':probes,'mount':mount,
            'scope':'Finite80Nm native arrest/release and actual named load contacts; no primary range proxy, structural strength, OEM safety, drop-arm or traversal certificate'}
    finally:mujoco.set_mjcb_passive(previous)

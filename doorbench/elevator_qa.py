"""Native landing-door interlock sequence, including failure controls.

Each cycle continues the actual preceding state. No joint range, equality,
position or velocity is modified to release, open, close or re-latch a door.
"""
from __future__ import annotations
import copy
import hashlib
import json
from collections import OrderedDict
import numpy as np

_CACHE=OrderedDict()

def run_elevator_qa(model,metadata,*,cycles=2,negative_controls=True):
    import mujoco
    binary=np.zeros(mujoco.mj_sizeModel(model),dtype=np.uint8);mujoco.mj_saveModel(model,buffer=binary)
    key=hashlib.sha256(binary.tobytes()+json.dumps({'version':3,'cycles':cycles,'negatives':negative_controls,
        'assembly':metadata.get('elevator_interlocks'),'supports':metadata.get('sliding_track_supports')},sort_keys=True).encode()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key);result=copy.deepcopy(_CACHE[key]);result['cache_hit']=True;return result
    result=_run_elevator_qa(model,metadata,cycles=cycles,negative_controls=negative_controls)
    result.update(cache_hit=False,compiled_model_sha256=hashlib.sha256(binary.tobytes()).hexdigest())
    _CACHE[key]=copy.deepcopy(result)
    while len(_CACHE)>32:_CACHE.popitem(last=False)
    return result


def _run_elevator_qa(model,metadata,*,cycles=2,negative_controls=True):
    import mujoco
    assembly=metadata.get('elevator_interlocks')
    if not assembly:return {'ok':True,'applicable':False,'failures':[]}
    m=copy.copy(model);rows=assembly['leaves'];bindings=[]
    for row in rows:
        ids={key:m.joint(row[key]).id for key in ('joint','hook_joint','cam_joint')}
        qa={key:int(m.jnt_qposadr[j]) for key,j in ids.items()}
        va={key:int(m.jnt_dofadr[j]) for key,j in ids.items()}
        for key in ('joint','cam_joint'):
            if int(m.jnt_type[ids[key]])!=int(mujoco.mjtJoint.mjJNT_SLIDE):raise ValueError('Elevator translation binding is not a slider')
        if not (m.jnt_range[ids['joint'],0]<-.005 and m.jnt_range[ids['joint'],1]>row['stroke_m']+.005):raise ValueError('Elevator safety limits must lie outside actual rail stops')
        bindings.append((row,qa,va,m.actuator(row['leaf']+'_drive').id))
    # Check support continuity separately from native load/contact operation.
    initial=mujoco.MjData(m);mujoco.mj_kinematics(m,initial)
    groups=[]
    for row in rows:
        for suffix,anchor in (('_hanger_assembly',row['bar_geom']),('_interlock_hook',row['hook_geom']),('_interlock_cam',row['cam_geom'])):
            bid=m.body(row['leaf']+suffix).id
            names=[m.geom(g).name for g in range(m.ngeom) if m.geom_bodyid[g]==bid]
            if suffix=='_hanger_assembly':
                # The two hangers and locking bracket attach independently
                # to the panel. The real panel stock joins those mounts.
                parent=m.body(row['leaf']).id
                names += [m.geom(g).name for g in range(m.ngeom) if m.geom_bodyid[g]==parent]
            groups.append((anchor,names))
    failures=[];mounts=[]
    for anchor,names in groups:
        connected={anchor}
        while True:
            added={n for n in names if n not in connected and any(mujoco.mj_geomDistance(m,initial,m.geom(n).id,m.geom(c).id,.001,None)<=.0005 for c in connected)}
            if not added:break
            connected|=added
        missing=[n for n in names if n not in connected]
        mounts.append({'anchor':anchor,'detached':missing})
        if missing:failures.append({'check':'detached_moving_stock','anchor':anchor,'parts':missing})
    from .sliding_track_qa import run_sliding_track_qa
    tracks=run_sliding_track_qa(m,metadata,samples=61)
    if not tracks['ok']:failures.append({'check':'track_support','details':tracks})
    # An ideal cam slide must still have a physical, unfilled stem guide.
    for row in rows:
        stem=m.geom(row['leaf']+'_interlock_cam_stem').id
        qa=int(m.jnt_qposadr[m.joint(row['cam_joint']).id])
        for q in np.linspace(0,row['cam_travel_m'],25):
            initial.qpos[qa]=q;mujoco.mj_kinematics(m,initial)
            gap=min(float(mujoco.mj_geomDistance(m,initial,stem,m.geom(g).id,1.,None)) for g in row['cam_guide_geoms'])
            if gap<.0006:failures.append({'check':'cam_guide_stock','leaf':row['leaf'],'gap_m':gap});break
    d=mujoco.MjData(m);mujoco.mj_forward(m,d);probes=[];released_configuration={}
    def phase(name,duration,cam_on=False,goal=None,hand_force=0.,ramp=False,free_until=None):
        starts={r['leaf']:float(d.qpos[q['joint']]) for r,q,_,_ in bindings}
        depth=0.;worst=None;hook_load=cam_load=hook_stop=press_stop=return_stop=terminal_stop=0.;motor=0.
        for step in range(round(duration/m.opt.timestep)):
            d.qfrc_applied[:]=0.;d.ctrl[:]=0.
            for row,qa,va,aid in bindings:
                if cam_on:d.qfrc_applied[va['cam_joint']]=row['max_cam_force_N']
                if hand_force:d.qfrc_applied[va['joint']]=hand_force
                if goal is not None:
                    target=row['stroke_m']+.02 if goal=='open' else -.02
                    if ramp:
                        begin=starts[row['leaf']];t=step*m.opt.timestep
                        target=begin+float(np.clip(target-begin,-.30*t,.40*t))
                    d.ctrl[aid]=float(np.clip(400.*(target-d.qpos[qa['joint']])-60.*d.qvel[va['joint']],-135.,135.))
            mujoco.mj_step(m,d)
            motor=max(motor,max((abs(float(d.actuator_force[a])) for _,_,_,a in bindings),default=0.))
            for ci,c in enumerate(d.contact):
                names={m.geom(c.geom1).name,m.geom(c.geom2).name}
                if -c.dist>depth:depth=-float(c.dist);worst=sorted(names)
                if any(any(g in names for g in support['rollers']) and any(g in names for g in support['end_stops'])
                       for support in metadata['sliding_track_supports']):
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                    terminal_stop=max(terminal_stop,float(np.linalg.norm(force[:3])))
                for row in rows:
                    hook_pair=names=={row['hook_geom'],row['bar_geom']}
                    cam_pair=names=={row['cam_geom'],row['roller_geom']}
                    stop_pair=names==set(row['hook_stop_geoms'])
                    pressed=row['cam_press_collar_geom'] in names and any(g in names for g in row['cam_guide_geoms'])
                    returned=row['cam_return_collar_geom'] in names and any(g in names for g in row['cam_guide_geoms'])
                    if hook_pair or cam_pair or stop_pair or pressed or returned:
                        force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                        load=float(np.linalg.norm(force[:3]))
                        if hook_pair:hook_load=max(hook_load,load)
                        if cam_pair:cam_load=max(cam_load,load)
                        if stop_pair:hook_stop=max(hook_stop,load)
                        if pressed:press_stop=max(press_stop,load)
                        if returned:return_stop=max(return_stop,load)
            if free_until is not None and all(d.qpos[q['joint']]>=free_until for _,q,_,_ in bindings):break
        result={'phase':name,'time_s':float(d.time),'leaf_m':[float(d.qpos[q['joint']]) for _,q,_,_ in bindings],
            'hook_rad':[float(d.qpos[q['hook_joint']]) for _,q,_,_ in bindings],
            'cam_m':[float(d.qpos[q['cam_joint']]) for _,q,_,_ in bindings],
            'hook_contact_N':hook_load,'cam_contact_N':cam_load,'max_motor_N':motor,
            'hook_stop_contact_N':hook_stop,'cam_press_stop_contact_N':press_stop,'cam_return_stop_contact_N':return_stop,
            'wheel_terminal_stop_contact_N':terminal_stop,
            'max_cam_input_N':max(r['max_cam_force_N'] for r in rows) if cam_on else 0.,
            'max_hand_input_N':hand_force,'max_penetration_m':depth,'worst_pair':worst,'warnings':d.warning.number.tolist()}
        probes.append(result)
        if depth>.001 or np.any(d.warning.number) or not np.isfinite(d.qpos).all():failures.append({'check':'native_penetration_or_warning','phase':name,'details':result})
        return result
    def check(condition,name,report):
        if not condition:failures.append({'check':name,'details':report})
    held=phase('initial_locked_load',1.,hand_force=120.)
    check(max(abs(q) for q in held['leaf_m'])<.006 and held['hook_contact_N']>1. and held['hook_stop_contact_N']>1.,'hook_did_not_carry_initial_load',held)
    for cycle in range(cycles):
        seated=phase(f'{cycle}_seat',1.,goal='closed')
        check(max(abs(q) for q in seated['leaf_m'])<.0005,'not_seated_before_release',seated)
        release=phase(f'{cycle}_cam_release',.8,cam_on=True,goal='closed')
        check(min(release['hook_rad'])>=.65 and release['cam_contact_N']>1. and release['cam_press_stop_contact_N']>1.,'cam_did_not_release_hooks',release)
        released_configuration={r[k]:float(d.qpos[q[k]]) for r,q,_,_ in bindings for k in ('hook_joint','cam_joint')}
        opened=phase(f'{cycle}_open',max(r['stroke_m'] for r in rows)/.4+1.5,cam_on=True,goal='open',ramp=True)
        check(all(q>=r['stroke_m']-.001 for q,r in zip(opened['leaf_m'],rows)) and opened['wheel_terminal_stop_contact_N']>.1,'incomplete_full_travel_or_unloaded_terminal_stop',opened)
        closed=phase(f'{cycle}_close',max(r['stroke_m'] for r in rows)/.3+1.5,cam_on=True,goal='closed',ramp=True)
        check(max(abs(q) for q in closed['leaf_m'])<.0005 and closed['wheel_terminal_stop_contact_N']>.1,'not_seated_on_stop_before_relock',closed)
        locked=phase(f'{cycle}_retire_cam',1.,goal='closed')
        check(max(abs(q) for q in locked['hook_rad'])<.015 and locked['hook_stop_contact_N']>0.1 and locked['cam_return_stop_contact_N']>.1,'hook_did_not_return',locked)
        phase(f'{cycle}_hands_off',.6)
        held=phase(f'{cycle}_relocked_load',1.,hand_force=120.)
        check(max(abs(q) for q in held['leaf_m'])<.006 and held['hook_contact_N']>1. and held['hook_stop_contact_N']>1.,'hook_did_not_carry_relocked_load',held)
    negatives=[]
    if negative_controls:
        for row in rows:
            cam=m.body(row['leaf']+'_interlock_cam').id
            for g in range(m.ngeom):
                if m.geom_bodyid[g]==cam:m.geom_contype[g]=m.geom_conaffinity[g]=0
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        failed_release=phase('removed_cam_contact',1.2,cam_on=True,goal='open')
        check(max(failed_release['hook_rad'])<.03 and max(failed_release['leaf_m'])<.006,'cam_contact_removal_did_not_prevent_release',failed_release)
        negatives.append('complete_cam_assembly_contact_removal')
        for row in rows:
            g=m.geom(row['hook_geom']).id;m.geom_contype[g]=m.geom_conaffinity[g]=0
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        free=phase('removed_hook',2.,hand_force=120.,free_until=.15)
        check(min(free['leaf_m'])>=.15,'removed_hook_still_arrested_door',free)
        negatives.append('load_hook_removal')
    return {'ok':not failures,'applicable':True,'cycles':cycles,'failures':failures,'probes':probes,'mounts':mounts,
        'tracks':tracks,'negative_controls':negatives,'released_configuration':released_configuration,
        'locked_leaf_positions':{r['joint']:q for r,q in zip(rows,held['leaf_m'])},
        'scope':'Native 120 N per-leaf locked load, 30 N retiring cam, force-limited powered full travel, repeated measured seating/relocking and removed-contact controls. Stationary car only; no human, safety circuit, strength or moving-car certification.'}

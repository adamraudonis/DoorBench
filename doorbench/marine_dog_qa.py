"""Native component gate for marine dog mounts and wheel/rod transmissions.

This is a bounded mechanism test, not a human motion, pressure seal, durability
or traversal certificate. It never calls the inspection linkage resolver.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from collections import OrderedDict
import numpy as np

_CACHE=OrderedDict()


def run_marine_dog_qa(model,meta):
    """Apply actual grip forces, release/return dogs and measure native errors."""
    import mujoco
    if meta.get('family')!='ship_watertight':
        return {'applicable':False,'ok':True,'scope':'No marine dogging mechanism'}
    mounts=meta.get('marine_dog_mounts',[]);linkage=meta.get('marine_dog_linkage')
    if not isinstance(mounts,list) or len(mounts)<4:
        raise ValueError('Marine native gate requires every dog mount')
    dogs=[row['joint'] for row in mounts]
    for row in mounts:
        body=model.body(row['body']).id;joint=model.joint(row['joint']).id
        geom=model.geom(row['spindle']).id
        if model.jnt_bodyid[joint]!=body or model.geom_bodyid[geom]!=body or not model.geom_contype[geom]:
            raise ValueError('Marine spindle/body/joint binding differs from physical model')
    if linkage:
        if len(linkage['connect_equalities'])!=4 or set(linkage['dog_joints'])!=set(dogs):
            raise ValueError('Marine wheel requires four connected dogs')
        for name in [linkage['gear_equality'],*linkage['connect_equalities']]:
            if not model.eq_active0[model.equality(name).id]:
                raise ValueError('Marine required transmission constraint is inactive')
        inputs=[(linkage['input_joint'],'wheel_grip_n',6.)]
    else:inputs=[(row['joint'],row['body']+'_grip',2.) for row in mounts]
    binary=np.zeros(mujoco.mj_sizeModel(model),dtype=np.uint8);mujoco.mj_saveModel(model,buffer=binary)
    digest=hashlib.sha256(binary).hexdigest()
    key=hashlib.sha256(digest.encode()+json.dumps({'version':3,'mounts':mounts,'linkage':linkage},sort_keys=True).encode()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key);result=copy.deepcopy(_CACHE[key]);result['cache_hit']=True;return result
    addr=lambda name:int(model.jnt_qposadr[model.joint(name).id])
    leaf=model.joint(meta['primary_joint']).id;depth=loop=gear=peak_force=0.;failures=[];trace=[]
    held=mujoco.MjData(model);hold_peak=0.;cleat_load=False
    for _ in range(round(1/model.opt.timestep)):
        held.qfrc_applied[:]=0;held.qfrc_applied[model.jnt_dofadr[leaf]]=80
        mujoco.mj_step(model,held);hold_peak=max(hold_peak,abs(float(held.qpos[model.jnt_qposadr[leaf]])))
        for k,c in enumerate(held.contact):
            depth=max(depth,-float(c.dist))
            if any((model.geom(g).name or '').startswith('cleat_') for g in (c.geom1,c.geom2)):
                force=np.zeros(6);mujoco.mj_contactForce(model,held,k,force);cleat_load|=force[0]>1
    if hold_peak>.01 or not cleat_load:failures.append('closed_dogs_did_not_arrest_leaf_at_80Nm')
    # Separate native component trial from the closed-lock load trial. Both
    # start at the source's own initial state; no invented released qpos.
    data=mujoco.MjData(model);released=None;next_sample=0.;aborted=False;release_holds=[]
    for opening in (True,False):
        for name,site_name,duration in inputs:
            j=model.joint(name).id;a=int(model.jnt_qposadr[j]);v=int(model.jnt_dofadr[j]);site=model.site(site_name).id
            body=int(model.site_bodyid[site]);initial=float(data.qpos[a]);goal=float(model.jnt_range[j,1] if opening else model.jnt_range[j,0])
            start=float(data.time)
            for _ in range(round(duration/model.opt.timestep)):
                mujoco.mj_forward(model,data);elapsed=float(data.time-start)
                f=min(1.,elapsed/(4. if linkage else 1.4));s=f**3*(10-15*f+6*f*f)
                target=initial+(goal-initial)*s
                kp,kd=(12.,1.5) if linkage else (50.,3.)
                effort=kp*(target-data.qpos[a])-kd*data.qvel[v]
                if not linkage:effort+=model.dof_frictionloss[v]*(1 if opening else -1)
                tangent=np.cross(data.xaxis[j],data.site_xpos[site]-data.xanchor[j]);radius=float(np.linalg.norm(tangent))
                if radius<.05:raise ValueError('Marine hand site lacks a physical turning radius')
                force=tangent*np.clip(effort,-120*radius,120*radius)/(radius*radius)
                data.qfrc_applied[:]=0
                mujoco.mj_applyFT(model,data,force,np.zeros(3),data.site_xpos[site],body,data.qfrc_applied)
                if linkage:
                    for follower in [linkage['output_joint'],*dogs,*linkage['rod_joints']]:
                        if abs(data.qfrc_applied[model.jnt_dofadr[model.joint(follower).id]])>1e-10:
                            failures.append('remote_follower_force')
                mujoco.mj_step(model,data);peak_force=max(peak_force,float(np.linalg.norm(force)))
                depth=max(depth,max((-float(c.dist) for c in data.contact),default=0.))
                if linkage:
                    for ename in linkage['connect_equalities']:
                        e=model.equality(ename).id;ba,bb=model.eq_obj1id[e],model.eq_obj2id[e]
                        residual=data.xpos[ba]+data.xmat[ba].reshape(3,3)@model.eq_data[e,:3]-data.xpos[bb]-data.xmat[bb].reshape(3,3)@model.eq_data[e,3:6]
                        loop=max(loop,float(np.linalg.norm(residual)))
                    gear=max(gear,abs(float(data.qpos[addr(linkage['output_joint'])]-data.qpos[addr(linkage['input_joint'])]/6)))
                if data.time>=next_sample:
                    trace.append({'time_s':float(data.time),'phase':'release' if opening else 'return',
                        'input':name,'site':site_name,'input_q':float(data.qpos[a]),'force_N':force.tolist()});next_sample+=.25
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    failures.append('nonfinite_native_state');aborted=True;break
            if any(w.number for w in data.warning):
                failures.append('native_solver_warning');aborted=True
            if aborted:break
        if aborted:break
        # A hand-held returned pose is not a functioning lock. Retain the
        # actual velocities and wait after releasing every input force.
        before={name:float(data.qpos[addr(name)]) for name in dogs}
        data.qfrc_applied[:]=0;data.xfrc_applied[:]=0;data.ctrl[:]=0
        for _ in range(round(2/model.opt.timestep)):
            mujoco.mj_step(model,data)
            depth=max(depth,max((-float(c.dist) for c in data.contact),default=0.))
            if linkage:
                for ename in linkage['connect_equalities']:
                    e=model.equality(ename).id;ba,bb=model.eq_obj1id[e],model.eq_obj2id[e]
                    residual=data.xpos[ba]+data.xmat[ba].reshape(3,3)@model.eq_data[e,:3]-data.xpos[bb]-data.xmat[bb].reshape(3,3)@model.eq_data[e,3:6]
                    loop=max(loop,float(np.linalg.norm(residual)))
                gear=max(gear,abs(float(data.qpos[addr(linkage['output_joint'])]-data.qpos[addr(linkage['input_joint'])]/6)))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                failures.append('nonfinite_native_state');aborted=True;break
            if any(w.number for w in data.warning):
                failures.append('native_solver_warning');aborted=True;break
        release_holds.append({'phase':'released' if opening else 'returned','duration_s':2.,
            'before_rad':before,'after_rad':{name:float(data.qpos[addr(name)]) for name in dogs}})
        if opening:released={name:float(data.qpos[addr(name)]) for name in dogs}
        if aborted:break
    returned={name:float(data.qpos[addr(name)]) for name in dogs}
    returned_leaf_peak=abs(float(data.qpos[model.jnt_qposadr[leaf]]))
    if not aborted:
        for _ in range(round(1/model.opt.timestep)):
            data.qfrc_applied[:]=0;data.qfrc_applied[model.jnt_dofadr[leaf]]=80
            mujoco.mj_step(model,data)
            returned_leaf_peak=max(returned_leaf_peak,abs(float(data.qpos[model.jnt_qposadr[leaf]])))
            depth=max(depth,max((-float(c.dist) for c in data.contact),default=0.))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                failures.append('nonfinite_native_state');break
            if any(w.number for w in data.warning):
                failures.append('native_solver_warning');break
    if returned_leaf_peak>.01:failures.append('returned_dogs_did_not_arrest_leaf_at_80Nm')
    if released is None or min(released.values())<1.45:failures.append('not_all_dogs_released')
    if max(abs(q) for q in returned.values())>.1:failures.append('not_all_dogs_returned')
    if depth>.001:failures.append('native_penetration_exceeds_1mm')
    if loop>.001:failures.append('native_pin_residual_exceeds_1mm')
    if gear>.001:failures.append('native_gear_residual_exceeds_0_001rad')
    if peak_force>120+1e-9:failures.append('manual_force_limit_exceeded')
    if any(w.number for w in held.warning):failures.append('native_lock_solver_warning')
    result={'applicable':True,'ok':not failures,'schema_version':1,'cache_hit':False,'compiled_model_sha256':digest,
        'scope':'Native closed-lock load and actual-grip release/return, including two-second hand-release retention at both endpoints. Individual dogs operate sequentially. No embodied-human, full passage, gasket pressure, strength or durability certification.',
        'locked_leaf_peak_rad':hold_peak,'locked_leaf_torque_Nm':80.,'cleat_contact_load_observed':bool(cleat_load),
        'returned_leaf_peak_rad':returned_leaf_peak,'returned_leaf_torque_Nm':80.,
        'released_dogs_rad':released,'returned_dogs_rad':returned,'peak_force_N':peak_force,
        'max_penetration_m':depth,'max_loop_residual_m':loop,'max_gear_residual_rad':gear,
        'elapsed_native_s':float(data.time),'warnings':[int(w.number) for w in data.warning],
        'hand_release_holds':release_holds,
        'failures':sorted(set(failures)),'trace':trace}
    _CACHE[key]=copy.deepcopy(result)
    while len(_CACHE)>32:_CACHE.popitem(last=False)
    return result

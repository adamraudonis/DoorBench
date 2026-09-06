"""Short native transmission gate for physical rolling-door hand chains.

This exercises real material-link forces in both directions. It does not
certify full-height opening, human reach, mechanism strength or traversal.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from collections import OrderedDict
import numpy as np
from .rollup_hoist import compile_hoist,hoist_control
from .hoist_keeper import compile_keeper,begin_keeper_transition,keeper_transition_action,keeper_open_force

_CACHE: OrderedDict[str,dict]=OrderedDict()


def run_rollup_hoist_qa(model,meta,*,phase_duration_s=4.):
    """Check native opening/return transmission on the supplied source model.

    The caller must explicitly build any released-lock fixture. This function
    never changes the supplied model, counterbalance, friction, lock state or
    force cap. Observed force-limited inability to move is a failed attempt.
    """
    import mujoco
    if 'rollup_hoist' not in meta:return {'applicable':False,'ok':True,'scope':'No hand-chain mechanism'}
    if isinstance(phase_duration_s,bool) or not isinstance(phase_duration_s,(int,float)) or not math.isfinite(phase_duration_s) or phase_duration_s<2.:
        raise ValueError('Each physical hand-chain QA phase must be at least two finite seconds')
    rules=compile_hoist(model,meta);keeper=compile_keeper(model,meta)
    binary=np.zeros(mujoco.mj_sizeModel(model),dtype=np.uint8);mujoco.mj_saveModel(model,buffer=binary)
    digest=hashlib.sha256(binary).hexdigest()
    key=hashlib.sha256(digest.encode()+json.dumps({'version':2,'duration':phase_duration_s,'hoist':meta['rollup_hoist'],'curtain':meta['rollup_curtain']},sort_keys=True).encode()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key);result=copy.deepcopy(_CACHE[key]);result['cache_hit']=True;return result
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    initial_z=float(data.site_xpos[rules.bottom_site,2]);initial_input=float(data.qpos[rules.input_qpos])
    peak_z=initial_z;opening_end_z=initial_z;opening_end_input=initial_input
    depth=0.;loop=0.;gear=0.;peak_force=0.;sites=set();contacts=set();trace=[];failures=[]
    next_sample=0.;phase='release';phase_start=0.;release_elapsed=None
    transition=begin_keeper_transition(model,data,rules,keeper,mode='release')
    for _ in range(math.ceil((2*phase_duration_s+15.)/model.opt.timestep)):
        mujoco.mj_forward(model,data);t=float(data.time)
        if phase=='release':
            action=keeper_transition_action(model,data,rules,keeper,transition);transition=action['next_state']
            if action['failed']:failures.append(action['reason']);break
            if action['done']:
                phase='opening';phase_start=t;release_elapsed=t
        if phase=='opening' and t-phase_start>=phase_duration_s:
            phase='return';phase_start=t
            opening_end_z=float(data.site_xpos[rules.bottom_site,2]);opening_end_input=float(data.qpos[rules.input_qpos])
        if phase=='return' and t-phase_start>=phase_duration_s:break
        if phase=='release':forces=action['site_forces']
        else:
            control=hoist_control(model,data,rules,opening=phase=='opening',elapsed_s=t-phase_start)
            forces={control['site']:control['force_N'],**keeper_open_force(model,data,keeper)}
        data.qfrc_applied[:]=0
        for name,vector in forces.items():
            site=model.site(name).id;force=np.asarray(vector);sites.add(name)
            mujoco.mj_applyFT(model,data,force,np.zeros(3),data.site_xpos[site],model.site_bodyid[site],data.qfrc_applied)
            peak_force=max(peak_force,float(np.linalg.norm(force)))
        # The force Jacobian belongs to the material chain. Neither geared
        # shaft is directly actuated by this independent gate.
        for name in ('hoist_input_hinge','curtain_drum_hinge'):
            if abs(data.qfrc_applied[model.joint(name).dofadr[0]])>1e-10:failures.append('unexpected_direct_shaft_drive')
        mujoco.mj_step(model,data)
        peak_z=max(peak_z,float(data.site_xpos[rules.bottom_site,2]))
        loop=max(loop,float(np.linalg.norm(data.site_xpos[rules.loop_start]-data.site_xpos[rules.loop_end])))
        gear=max(gear,abs(float(data.qpos[rules.output_qpos]-rules.ratio*data.qpos[rules.input_qpos])))
        for c in data.contact:
            depth=max(depth,-float(c.dist));a=model.geom(c.geom1).name or '';b=model.geom(c.geom2).name or ''
            if c.dist<=0 and (a.startswith('hoist_hand_wheel') and b.startswith('hoist_link_') or b.startswith('hoist_hand_wheel') and a.startswith('hoist_link_')):
                contacts.add(tuple(sorted((a,b))))
        if t>=next_sample:
            trace.append({'time_s':t,'phase':phase,'bottom_z_m':float(data.site_xpos[rules.bottom_site,2]),'keeper_q_m':float(data.qpos[keeper.qpos]),'site_forces_N':forces});next_sample+=.1
        if any(w.number for w in data.warning):failures.append('native_solver_warning');break
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():failures.append('nonfinite_native_state');break
    mujoco.mj_forward(model,data);final_z=float(data.site_xpos[rules.bottom_site,2]);final_input=float(data.qpos[rules.input_qpos])
    if release_elapsed is None:failures.append('positive_keeper_not_physically_released')
    if opening_end_z-initial_z<.05:failures.append('native_opening_transmission_not_observed')
    if opening_end_z-final_z<.02:failures.append('native_return_transmission_not_observed')
    if initial_input-opening_end_input<.05 or final_input-opening_end_input<.05:failures.append('opposite_input_shaft_motion_not_observed')
    if not contacts:failures.append('missing_material_to_pocket_wheel_contact')
    if depth>.001:failures.append('native_penetration_exceeds_1mm')
    if loop>.001:failures.append('material_loop_residual_exceeds_1mm')
    if gear>.005:failures.append('gear_constraint_residual_exceeds_0_005rad')
    if peak_force>rules.force_limit+1e-9:failures.append('manual_force_limit_exceeded')
    result={'applicable':True,'ok':not failures,'schema_version':1,'cache_hit':False,'compiled_model_sha256':digest,
        'scope':'Actual keeper unload/withdrawal, short native material-chain transmission and opposite-strand return with a second input holding the keeper; not full opening, hands-free retention, embodied-human reach, strength, power-loss behavior or traversal certification',
        'keeper_release_elapsed_s':release_elapsed,
        'phase_duration_s':phase_duration_s,'initial_bottom_z_m':initial_z,'opening_end_bottom_z_m':opening_end_z,
        'peak_bottom_z_m':peak_z,'final_bottom_z_m':final_z,'opening_input_delta_rad':opening_end_input-initial_input,
        'return_input_delta_rad':final_input-opening_end_input,'max_penetration_m':depth,'max_loop_residual_m':loop,
        'max_gear_residual_rad':gear,'peak_force_N':peak_force,'material_sites_used':sorted(sites-{keeper.grip_name}),
        'keeper_grip_site':keeper.grip_name,
        'pocket_contact_pairs':[list(pair) for pair in sorted(contacts)],'elapsed_native_s':float(data.time),
        'warnings':[int(w.number) for w in data.warning],'failures':sorted(set(failures)),'trace':trace}
    _CACHE[key]=copy.deepcopy(result)
    while len(_CACHE)>24:_CACHE.popitem(last=False)
    return result

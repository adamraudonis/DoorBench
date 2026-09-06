"""Native-only rolling-curtain initialization, with model-bound result caching.

The articulated curtain has many contact-selected configurations at a given
barrel angle. A requested open state must therefore come from integrated
native motion. No analytic slat interpolation, lock release, mass/friction
change, or stronger force is used here.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from collections import OrderedDict
import numpy as np

_CACHE: OrderedDict[str,dict]=OrderedDict()
_MAX_CACHE=24


def rollup_handle_force(height_m,velocity_m_s,*,start_m,goal_m,elapsed_s,mass_kg,force_limit_N=120.,duration_s=12.):
    """Shared force controller for native initialization and reference policy.

    Inputs are measured bottom height, actual grip vertical speed, and the
    reflected grip mass returned by rollup_grip_dynamics. Return a scalar
    world-Z force plus its target/gains. Quintic
    position and velocity have zero first/second endpoint derivatives; the
    force is still capped independently of the mass-based damping choice.
    """
    values=(height_m,velocity_m_s,start_m,goal_m,elapsed_s,mass_kg,force_limit_N,duration_s)
    if any(isinstance(v,bool) or not isinstance(v,(int,float,np.number)) or not math.isfinite(float(v)) for v in values):
        raise ValueError('Rollup controller inputs must be finite numbers')
    if mass_kg<=0 or duration_s<=0 or not 0<force_limit_N<=120:raise ValueError('Invalid mass, duration or manual force limit')
    u=min(1.,max(0.,elapsed_s/duration_s));smooth=u**3*(10+u*(-15+6*u))
    desired=start_m+(goal_m-start_m)*smooth
    speed=(goal_m-start_m)/duration_s*30*u*u*(1-u)*(1-u)
    stiffness=3500.;damping=2*math.sqrt(stiffness*float(mass_kg))
    force=float(np.clip(stiffness*(desired-height_m)+damping*(speed-velocity_m_s),-force_limit_N,force_limit_N))
    return {'force_N':force,'target_z_m':desired,'target_speed_m_s':speed,'stiffness_N_m':stiffness,'damping_N_s_m':damping}


def rollup_grip_dynamics(model,data,mechanism):
    """Read physical grip velocity and its unconstrained reflected inertia.

    Native factorization evaluates 1/(J M^-1 J^T) along world Z. Contact
    constraints are not removed or altered; the unconstrained mass is the
    conservative fast-mode scale for the explicitly applied hand damping.
    Whole-curtain mass is unsuitable: the offset grip can rock a light bottom
    slat even when the complete assembly is heavy.
    """
    import mujoco
    grip=model.site(mechanism['manual_grip_site']).id;jac=np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,data,jac,None,grip)
    direction=np.ascontiguousarray(jac[2:3]);inverse=np.zeros_like(direction)
    mujoco.mj_solveM(model,data,inverse,direction)
    inverse_mass=float(direction[0]@inverse[0]);bound=float(model.body_subtreemass[model.body('curtain_barrel').id])
    mass=min(bound,1/max(inverse_mass,1e-12))
    return {'grip_speed_m_s':float(jac[2]@data.qvel),'grip_effective_mass_kg':mass,'carried_mass_kg':bound}


def prepare_rollup_open(model,meta,initial_qpos=None,*,ramp_duration_s=12.,time_limit_s=18.):
    """Return a native full-state open initialization or an explicit failure.

    Caller supplies the actual scenario model. If its initial lock is off,
    build the released fixture first; this function never removes a lock.
    Successful qpos/qvel describe one integrated state, not an equilibrium or
    a guarantee that the door remains open after the hand force is removed.
    The supplied model and any existing live MjData remain unchanged.
    """
    import mujoco
    mechanism=meta.get('rollup_curtain')
    if not isinstance(mechanism,dict):raise ValueError('rollup_curtain metadata is required')
    for name,value in [('ramp_duration_s',ramp_duration_s),('time_limit_s',time_limit_s)]:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f'{name} must be positive and finite')
    initial=np.asarray(model.qpos0 if initial_qpos is None else initial_qpos,dtype=np.float64)
    if initial.shape!=(model.nq,) or not np.isfinite(initial).all():raise ValueError('initial_qpos must be a finite full model state')
    initial=initial.copy()
    authored_limit=mechanism['drive']['manual_max_force_N']
    if isinstance(authored_limit,bool) or not isinstance(authored_limit,(int,float)):
        raise ValueError('Manual force limit must be a finite number')
    force_limit=float(authored_limit)
    if not math.isfinite(force_limit) or not 0<force_limit<=120:raise ValueError('Manual rollup initialization requires an authored force limit in (0,120] N')
    binary=np.zeros(mujoco.mj_sizeModel(model),dtype=np.uint8);mujoco.mj_saveModel(model,buffer=binary)
    model_hash=hashlib.sha256(binary).hexdigest()
    key=hashlib.sha256(model_hash.encode()+initial.tobytes()+json.dumps({'algorithm_version':3,'mechanism':mechanism,'ramp_duration_s':ramp_duration_s,'time_limit_s':time_limit_s},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key);cached=copy.deepcopy(_CACHE[key]);cached['cache_hit']=True;return cached
    report={'schema_version':1,'ok':False,'reason':'native_goal_not_reached','compiled_model_sha256':model_hash,
        'initial_qpos_sha256':hashlib.sha256(initial.tobytes()).hexdigest(),'cache_hit':False,
        'scope':'Native force-limited open initialization; no source lock release, geometry change, pose interpolation or equilibrium guarantee',
        'force_site':mechanism['manual_grip_site'],'force_limit_N':force_limit,'peak_force_N':0.,'max_penetration_m':0.,'warnings':[],
        'elapsed_native_s':0.,'peak_bottom_z_m':None,'final_bottom_z_m':None,'trace':[]}
    def finish():
        _CACHE[key]=copy.deepcopy(report)
        while len(_CACHE)>_MAX_CACHE:_CACHE.popitem(last=False)
        return report
    if mechanism['drive'].get('opener')=='chain_hoist':
        report['reason']='chain_hoist_drive_incomplete' if mechanism['drive'].get('chain_hoist_supported') is not True else 'chain_hoist_requires_material_link_initializer'
        return finish()
    barrel_id=model.body('curtain_barrel').id
    external=[]
    for jid in range(model.njnt):
        bid=int(model.jnt_bodyid[jid])
        while bid not in (0,barrel_id):bid=int(model.body_parentid[bid])
        if bid!=barrel_id:external.append(model.joint(jid).name)
    if external:
        report['reason']='additional_dynamic_bodies_require_door_only_initialization'
        report['unsupported_joint_names']=external;return finish()
    data=mujoco.MjData(model);data.qpos[:]=initial;mujoco.mj_forward(model,data)
    point=model.site(mechanism['progress']['site']).id;grip=model.site(mechanism['manual_grip_site']).id;body=model.site_bodyid[grip]
    start=float(data.site_xpos[point,2]);target=float(mechanism['progress']['open_z_m']);report['peak_bottom_z_m']=start
    carried_mass=float(model.body_subtreemass[model.body('curtain_barrel').id])
    grip_state=rollup_grip_dynamics(model,data,mechanism)
    initial_control=rollup_handle_force(start,0.,start_m=start,goal_m=target,elapsed_s=0.,mass_kg=grip_state['grip_effective_mass_kg'],force_limit_N=force_limit,duration_s=ramp_duration_s)
    report['controller']={k:initial_control[k] for k in ('stiffness_N_m','damping_N_s_m')}
    report['controller'].update({'mass_bound_kg':carried_mass,'initial_effective_mass_kg':grip_state['grip_effective_mass_kg'],'quintic_velocity_feedforward':True,'damping_velocity_site':mechanism['manual_grip_site'],'mass_basis':'Native unconstrained reflected grip inertia, capped by total carried mass'})
    jac=np.zeros((3,model.nv));rot=np.zeros_like(jac);stable=0.;stalled=0.;next_sample=0.
    for _ in range(math.ceil(time_limit_s/model.opt.timestep)):
        mujoco.mj_forward(model,data);t=float(data.time)
        mujoco.mj_jacSite(model,data,jac,rot,point);velocity=float((jac@data.qvel)[2]);height=float(data.site_xpos[point,2])
        grip_state=rollup_grip_dynamics(model,data,mechanism)
        control=rollup_handle_force(height,grip_state['grip_speed_m_s'],start_m=start,goal_m=target,elapsed_s=t,mass_kg=grip_state['grip_effective_mass_kg'],force_limit_N=force_limit,duration_s=ramp_duration_s)
        desired=control['target_z_m'];force=control['force_N'];data.qfrc_applied[:]=0
        mujoco.mj_applyFT(model,data,np.array([0.,0.,force]),np.zeros(3),data.site_xpos[grip],body,data.qfrc_applied);mujoco.mj_step(model,data)
        depth=max((-float(c.dist) for c in data.contact),default=0.)
        report['peak_force_N']=max(report['peak_force_N'],abs(force));report['max_penetration_m']=max(report['max_penetration_m'],depth)
        report['peak_bottom_z_m']=max(report['peak_bottom_z_m'],height)
        if t>=next_sample:
            report['trace'].append({'time_s':t,'bottom_z_m':height,'vertical_speed_m_s':velocity,'force_N':force});next_sample+=.1
        if any(w.number for w in data.warning):report['reason']='native_solver_warning';break
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():report['reason']='nonfinite_native_state';break
        # Once the target is appreciably above a stationary curtain, more
        # elapsed time cannot increase a saturated hand force. Report this
        # observed stall without attributing it to a particular lock/spring.
        stalled=stalled+model.opt.timestep if desired-height>.10 and force>=force_limit*.995 and abs(velocity)<.003 else 0.
        if stalled>=2.:
            report['reason']='native_lift_stalled_at_force_limit';break
        # Settling is measured from native velocity, not by zeroing it. A
        # millimetre is the same contact-depth gate as the winding regression.
        stable=stable+model.opt.timestep if t>=ramp_duration_s and height>=target-.025 and abs(velocity)<.05 else 0.
        if stable>=.25:
            if report['max_penetration_m']>.001:report['reason']='native_contact_depth_exceeded';break
            report.update({'ok':True,'reason':'native_open_state_reached','qpos':data.qpos.tolist(),'qvel':data.qvel.tolist()});break
    mujoco.mj_forward(model,data)
    report.update({'elapsed_native_s':float(data.time),'final_bottom_z_m':float(data.site_xpos[point,2]),'warnings':[int(w.number) for w in data.warning]})
    return finish()

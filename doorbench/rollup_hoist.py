"""Measured material-link control for the physical rolling-door hand chain.

The abstract hand regrips actual circulating links. It applies a bounded force
at that material point; the pocket contacts and ideal spur gears transmit the
load. This is a mechanism/controller test, not an embodied human reach test.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import numpy as np


@dataclass(frozen=True)
class HoistRules:
    site_ids: tuple[int, ...]
    site_names: tuple[str, ...]
    bottom_site: int
    loop_start: int
    loop_end: int
    input_qpos: int
    output_qpos: int
    ratio: float
    wheel_y: float
    opening_side: int
    closing_side: int
    grip_height: float
    open_z: float
    closed_z: float
    force_limit: float
    owned_roots: tuple[int, ...]
    excluded_grip_z: tuple[float,float] | None = None


def compile_hoist(model, meta):
    """Bind authored physical sites, root bodies and gear joints to one model."""
    import mujoco
    h=meta.get('rollup_hoist');curtain=meta.get('rollup_curtain')
    if not isinstance(h,dict) or not isinstance(curtain,dict):
        raise ValueError('Physical rollup hoist and curtain metadata are required')
    if h.get('schema_version')!=1 or h.get('kind')!='guided_roller_hand_chain_with_ideal_4_to_1_spur_gears':
        raise ValueError('Unsupported physical hand-chain schema')
    names=h['material_grip_sites'];bodies=h['material_bodies']
    if not names or len(names)!=len(bodies) or len(set(names))!=len(names) or len(set(bodies))!=len(bodies):
        raise ValueError('Material link/site inventory is incomplete or duplicated')
    sites=tuple(model.site(n).id for n in names)
    for site,body in zip(sites,bodies):
        if model.site_bodyid[site]!=model.body(body).id:
            raise ValueError('Hand-chain grip must be attached to its declared material link')
    root=model.joint(h['free_root_joint']).id
    if model.jnt_type[root]!=mujoco.mjtJoint.mjJNT_FREE or model.jnt_bodyid[root]!=model.body(h['free_root_body']).id:
        raise ValueError('Physical material chain requires its authored free root')
    joints=[model.joint(h[k]).id for k in ('input_joint','output_joint')]
    if any(model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE for j in joints):
        raise ValueError('Hand-chain gear shafts must be native hinges')
    values=[h['output_per_input'],*h['wheel_center_m'],h['nominal_regrasp_height_m'],
            h['hand_force_limit_N'],curtain['progress']['open_z_m'],curtain['progress']['closed_z_m']]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError('Hand-chain parameters must be finite numbers')
    if h['output_per_input']!=-.25 or not 0<h['hand_force_limit_N']<=120:
        raise ValueError('Unsupported gear ratio or manual force limit')
    equality=model.equality('hoist_gear_ratio').id
    if (not model.eq_active0[equality] or model.eq_type[equality]!=mujoco.mjtEq.mjEQ_JOINT or
        model.eq_obj1id[equality]!=joints[1] or model.eq_obj2id[equality]!=joints[0] or
        not np.array_equal(model.eq_data[equality,:5],[0.,-.25,0.,0.,0.])):
        raise ValueError('Native geared constraint differs from its physical hand-chain metadata')
    if h['opening_pull_strand_y_sign']!=-1 or h['closing_pull_strand_y_sign']!=1:
        raise ValueError('Physical hand-chain strand directions do not match the gearing')
    if not .55<h['nominal_regrasp_height_m']<1.7:
        raise ValueError('Regrasp height must lie inside the authored reachable band')
    roots=tuple(model.body(n).id for n in ('curtain_barrel','hoist_input','hoist_return_idler',h['free_root_body']))
    keeper=h.get('keeper');excluded=None;grip_height=float(h['nominal_regrasp_height_m'])
    if keeper:
        roots=(*roots,model.body(keeper['body']).id)
        excluded=tuple(map(float,keeper['excluded_chain_grip_z_m']));grip_height=.95
        if len(excluded)!=2 or not all(math.isfinite(v) for v in excluded) or excluded[0]>=excluded[1]:
            raise ValueError('Invalid keeper chain-grip exclusion')
    if any(model.body_parentid[b]!=0 for b in roots):raise ValueError('Unexpected hand-chain assembly parent')
    return HoistRules(sites,tuple(names),model.site(curtain['progress']['site']).id,
        model.site('hoist_chain_node_0').id,model.site('hoist_chain_loop_end').id,
        int(model.jnt_qposadr[joints[0]]),int(model.jnt_qposadr[joints[1]]),float(h['output_per_input']),
        float(h['wheel_center_m'][1]),-1,1,grip_height,
        float(curtain['progress']['open_z_m']),float(curtain['progress']['closed_z_m']),
        float(h['hand_force_limit_N']),roots,excluded)


def hoist_control(model, data, rules, *, opening=True, elapsed_s=0.):
    """Return a world-Z force on the nearest actual material-link grip.

    Downward hand travel on opposite strands opens/closes the geared curtain.
    Upward force is allowed to brake a descending link. The 0.45 m/s desired
    hand speed ramps smoothly over one second; load-dependent speed reduction
    and force-limited failure remain explicit. No native state is modified.
    """
    import mujoco
    if not isinstance(opening,bool):raise ValueError('opening must be boolean')
    if isinstance(elapsed_s,bool) or not isinstance(elapsed_s,(int,float,np.number)) or not math.isfinite(elapsed_s) or elapsed_s<0:
        raise ValueError('elapsed_s must be finite and nonnegative')
    sites=np.asarray(rules.site_ids,dtype=int);xyz=data.site_xpos[sites]
    side=rules.opening_side if opening else rules.closing_side
    allowed=(side*(xyz[:,1]-rules.wheel_y)>.06)&(xyz[:,2]>.55)&(xyz[:,2]<1.7)
    if rules.excluded_grip_z:
        lo,hi=rules.excluded_grip_z;allowed&=(xyz[:,2]<lo)|(xyz[:,2]>hi)
    candidates=np.flatnonzero(allowed)
    if not len(candidates):raise ValueError('No physical chain material grip is inside the reachable band')
    index=int(min(candidates,key=lambda k:abs(xyz[k,2]-rules.grip_height)))
    site=int(sites[index]);jac=np.zeros((3,model.nv));mujoco.mj_jacSite(model,data,jac,None,site)
    speed=float(jac[2]@data.qvel);height=float(data.site_xpos[rules.bottom_site,2])
    target=rules.open_z if opening else rules.closed_z
    error=target-height if opening else height-target
    u=min(1.,float(elapsed_s));ramp=u**3*(10+u*(-15+6*u))
    desired=-min(.45,max(0.,error*3.))*ramp
    force=float(np.clip(250.*(desired-speed),-rules.force_limit,rules.force_limit))
    return {'site':rules.site_names[index],'site_id':site,'body_id':int(model.site_bodyid[site]),
        'force_N':[0.,0.,force],'material_link_index':index,'grip_speed_m_s':speed,
        'target_grip_speed_m_s':desired,'bottom_z_m':height,'target_bottom_z_m':target,
        'scope':'Abstract regrasp of actual material links; no embodied hand trajectory or joint drive'}


_CACHE: OrderedDict[str,dict]=OrderedDict()


def prepare_hoist_open(model, meta, initial_qpos=None, *, time_limit_s=120.):
    """Integrate keeper release, complete opening and a hands-free held state.

    Locks, original counterbalance, friction and masses stay unchanged. Other
    dynamic roots (for example a robot) are explicitly unsupported, so this
    cannot silently transplant a fallen robot from a private initialization.
    Every transition uses real material-link/pin forces. The keeper must retain
    the final curtain for two seconds without either hand before state return.
    Returned state is sampled, not an equilibrium or dynamic certification.
    """
    import mujoco
    from .native_warnings import capture_native_warnings
    from .hoist_keeper import (compile_keeper,begin_keeper_transition,
        keeper_transition_action,keeper_open_force,keeper_pin_load)
    rules=compile_hoist(model,meta)
    if isinstance(time_limit_s,bool) or not isinstance(time_limit_s,(int,float)) or not math.isfinite(time_limit_s) or time_limit_s<=0:
        raise ValueError('time_limit_s must be finite and positive')
    initial=np.asarray(model.qpos0 if initial_qpos is None else initial_qpos,dtype=np.float64)
    if initial.shape!=(model.nq,) or not np.isfinite(initial).all():raise ValueError('Expected finite full native qpos')
    initial=initial.copy()
    for jid in range(model.njnt):
        if model.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE:
            start=model.jnt_qposadr[jid]+3
            if abs(float(np.linalg.norm(initial[start:start+4]))-1)>1e-6:raise ValueError('Initial free-root quaternion must be unit length')
    binary=np.zeros(mujoco.mj_sizeModel(model),dtype=np.uint8);mujoco.mj_saveModel(model,buffer=binary)
    model_hash=hashlib.sha256(binary).hexdigest()
    opts={'algorithm_version':3,'hoist':meta['rollup_hoist'],'curtain':meta['rollup_curtain'],'time_limit_s':time_limit_s}
    key=hashlib.sha256(model_hash.encode()+initial.tobytes()+json.dumps(opts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key);result=copy.deepcopy(_CACHE[key]);result['cache_hit']=True;return result
    report={'schema_version':1,'ok':False,'reason':'native_goal_not_reached','compiled_model_sha256':model_hash,
        'initial_qpos_sha256':hashlib.sha256(initial.tobytes()).hexdigest(),'cache_hit':False,
        'scope':'Native material-link and positive-keeper force initialization with two seconds of hands-free retention; no security-lock release, source property changes, pose interpolation, robot support or equilibrium guarantee',
        'force_limit_N':rules.force_limit,'peak_force_N':0.,'max_penetration_m':0.,'max_loop_residual_m':0.,
        'max_gear_residual_rad':0.,'regrasps':0,'elapsed_native_s':0.,'trace':[],
        'algorithm_version':3,'transitions':[],'hands_free_hold_s':0.,'peak_keeper_force_N':0.,
        'hands_free_pin_load_peak_N':0.,'hands_free_pin_load_observed_s':0.,
        'hands_free_up_stop_load_peak_N':0.,'hands_free_up_stop_load_observed_s':0.,
        'native_warning_messages':[],'native_warning_events':[]}
    def finish():
        _CACHE[key]=copy.deepcopy(report)
        while len(_CACHE)>24:_CACHE.popitem(last=False)
        return report
    external=[]
    for jid in range(model.njnt):
        body=int(model.jnt_bodyid[jid])
        while model.body_parentid[body]!=0:body=int(model.body_parentid[body])
        if body not in rules.owned_roots:external.append(model.joint(jid).name)
    if external:
        report.update(reason='additional_dynamic_bodies_require_door_only_initialization',unsupported_joint_names=external)
        return finish()
    if not meta['rollup_hoist'].get('keeper'):
        report['reason']='physical_keeper_required_for_hands_free_open_state'
        return finish()
    keeper=compile_keeper(model,meta)
    stop_pairs=[]
    up_stops=meta['rollup_curtain'].get('up_stops')
    if up_stops:
        names=up_stops.get('lug_names',[]);fixed=up_stops.get('stop_names',[])
        if len(names)!=2 or len(fixed)!=2 or len(set(names+fixed))!=4:
            raise ValueError('Actual paired upper-stop metadata is required')
        for lug_name,stop_name in zip(names,fixed):
            lug=model.geom(lug_name).id;stop=model.geom(stop_name).id
            body=int(model.geom_bodyid[lug]);bottom=int(model.site_bodyid[rules.bottom_site])
            while body and body!=bottom:body=int(model.body_parentid[body])
            if body!=bottom:raise ValueError('Upper-stop lug is detached from the actual bottom bar')
            body=int(model.geom_bodyid[stop])
            while body:
                if model.body_jntnum[body]:raise ValueError('Upper-stop reaction requires an actual fixed support')
                body=int(model.body_parentid[body])
            if any(not(model.geom_contype[g] and model.geom_conaffinity[g]) for g in (lug,stop)):
                raise ValueError('Upper-stop reaction requires active physical contacts')
            stop_pairs.append((lug,stop))
    def stop_reaction(data):
        force_z=0.
        for i,c in enumerate(data.contact):
            for lug,stop in stop_pairs:
                if {int(c.geom1),int(c.geom2)}!={lug,stop}:continue
                force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
                world=np.asarray(c.frame).reshape(3,3).T@force[:3]
                on_lug=world if c.geom2==lug else -world
                force_z+=max(0.,-float(on_lug[2]))
        return force_z
    with capture_native_warnings() as messages:
        data=mujoco.MjData(model);data.qpos[:]=initial;mujoco.mj_forward(model,data)
        stable=0.;stalled=0.;next_sample=0.;previous=None;jac=np.zeros((3,model.nv))
        phase='release';phase_start=0.
        transition=begin_keeper_transition(model,data,rules,keeper,mode='release')
        for _ in range(math.ceil(time_limit_s/model.opt.timestep)):
            mujoco.mj_forward(model,data);t=float(data.time)
            if messages:
                report.update(reason='native_solver_message',native_warning_events=[{'time_s':t,'phase':phase,'messages':list(messages)}]);break
            mujoco.mj_jacSite(model,data,jac,None,rules.bottom_site);speed=float(jac[2]@data.qvel)
            height=float(data.site_xpos[rules.bottom_site,2]);forces={};force=0.;site=None
            try:
                if phase in ('release','engage'):
                    action=keeper_transition_action(model,data,rules,keeper,transition)
                    transition=action['next_state'];forces=action['site_forces']
                    if phase=='engage' and transition['phase']=='verify_hold' and not forces:
                        load=keeper_pin_load(model,data,keeper)
                        report['hands_free_pin_load_peak_N']=max(report['hands_free_pin_load_peak_N'],load)
                        if load>5.:report['hands_free_pin_load_observed_s']+=model.opt.timestep
                        upper_load=stop_reaction(data)
                        report['hands_free_up_stop_load_peak_N']=max(report['hands_free_up_stop_load_peak_N'],upper_load)
                        if upper_load>5.:report['hands_free_up_stop_load_observed_s']+=model.opt.timestep
                    if action['failed']:
                        report.update(reason=action['reason'],failed_transition=transition);break
                    if action['done']:
                        report['transitions'].append({'phase':phase,'time_s':t,'bottom_z_m':height,'state':transition})
                        if phase=='release':
                            phase='open';phase_start=t;forces=keeper_open_force(model,data,keeper)
                        else:
                            report['hands_free_hold_s']=t-transition['phase_start_s']
                            report['hand_free_drift_m']=transition['hand_free_drift_m']
                            # Gap seating may lose a few millimetres after the
                            # measured full-open dwell, but cannot hide loss of
                            # opening by accepting an arbitrary raised pose.
                            if height<rules.open_z-.04 or abs(speed)>.05:
                                report['reason']='keeper_held_state_below_open_goal_or_unsettled';break
                            if report['hands_free_pin_load_observed_s']>=.1:
                                report['hands_free_support_kind']='positive_keeper'
                            elif report['hands_free_up_stop_load_observed_s']>=.1:
                                report['hands_free_support_kind']='fixed_upper_stops_with_seated_keeper'
                            else:
                                report['reason']='hands_free_physical_load_path_not_observed';break
                            if report['max_penetration_m']>.001 or report['max_loop_residual_m']>.001 or report['max_gear_residual_rad']>.005:
                                report['reason']='native_contact_or_linkage_tolerance_exceeded';break
                            report.update(ok=True,reason='native_open_state_reached_hands_free_with_keeper_seated',
                                qpos=data.qpos.tolist(),qvel=data.qvel.tolist());break
                elif phase=='open':
                    control=hoist_control(model,data,rules,opening=True,elapsed_s=t-phase_start)
                    site=control['site_id'];force=float(control['force_N'][2])
                    forces={control['site']:control['force_N'],**keeper_open_force(model,data,keeper)}
                    stalled=stalled+model.opt.timestep if rules.open_z-height>.1 and force<=-rules.force_limit*.995 and abs(speed)<.003 else 0.
                    if stalled>=2.:report['reason']='native_chain_pull_stalled_at_force_limit';break
                    stable=stable+model.opt.timestep if height>=rules.open_z-.01 and abs(speed)<.03 else 0.
                    if stable>=.25:
                        report['full_open_dwell_bottom_z_m']=height
                        report['transitions'].append({'phase':'full_open_dwell','time_s':t,'bottom_z_m':height,'dwell_s':stable})
                        phase='engage';phase_start=t
                        transition=begin_keeper_transition(model,data,rules,keeper,mode='engage')
                else:raise ValueError('Unknown native hoist initialization phase')
            except ValueError as error:
                report.update(reason='physical_input_unavailable',detail=str(error));break
            data.qfrc_applied[:]=0
            for name,vector in forces.items():
                norm=float(np.linalg.norm(vector))
                if not math.isfinite(norm) or norm>rules.force_limit+1e-9:raise ValueError('Native initializer input exceeds the manual force cap')
                sid=model.site(name).id
                mujoco.mj_applyFT(model,data,np.array(vector),np.zeros(3),data.site_xpos[sid],model.site_bodyid[sid],data.qfrc_applied)
                report['peak_force_N']=max(report['peak_force_N'],norm)
                if sid==keeper.grip_site:report['peak_keeper_force_N']=max(report['peak_keeper_force_N'],norm)
                else:
                    report['regrasps']+=int(previous is not None and sid!=previous);previous=sid
            mujoco.mj_step(model,data)
            report['max_penetration_m']=max(report['max_penetration_m'],max((-float(c.dist) for c in data.contact),default=0.))
            report['max_loop_residual_m']=max(report['max_loop_residual_m'],float(np.linalg.norm(data.site_xpos[rules.loop_start]-data.site_xpos[rules.loop_end])))
            report['max_gear_residual_rad']=max(report['max_gear_residual_rad'],abs(float(data.qpos[rules.output_qpos]-rules.ratio*data.qpos[rules.input_qpos])))
            if t>=next_sample:
                report['trace'].append({'time_s':t,'phase':phase,'keeper_phase':transition['phase'],
                    'bottom_z_m':height,'bottom_speed_m_s':speed,'site_forces':forces})
                next_sample+=.1
            if messages:
                report.update(reason='native_solver_message',native_warning_events=[{'time_s':float(data.time),'phase':phase,'messages':list(messages)}]);break
            if any(w.number for w in data.warning):report['reason']='native_solver_warning';break
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():report['reason']='nonfinite_native_state';break
        mujoco.mj_forward(model,data)
        # The final forward pass refreshes contact forces and may itself expose a
        # warning. No pre-forward success may bypass these final acceptance gates.
        report['max_penetration_m']=max(report['max_penetration_m'],max((-float(c.dist) for c in data.contact),default=0.))
        report['max_loop_residual_m']=max(report['max_loop_residual_m'],float(np.linalg.norm(data.site_xpos[rules.loop_start]-data.site_xpos[rules.loop_end])))
        report['max_gear_residual_rad']=max(report['max_gear_residual_rad'],abs(float(data.qpos[rules.output_qpos]-rules.ratio*data.qpos[rules.input_qpos])))
        final_failure=None
        report['native_warning_messages']=list(messages)
        if messages:final_failure='native_solver_message'
        elif any(w.number for w in data.warning):final_failure='native_solver_warning'
        elif not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():final_failure='nonfinite_native_state'
        elif report['max_penetration_m']>.001 or report['max_loop_residual_m']>.001 or report['max_gear_residual_rad']>.005:
            final_failure='native_contact_or_linkage_tolerance_exceeded'
        if final_failure:
            report.update(ok=False,reason=final_failure);report.pop('qpos',None);report.pop('qvel',None)
        report.update(elapsed_native_s=float(data.time),final_bottom_z_m=float(data.site_xpos[rules.bottom_site,2]),
            warnings=[int(w.number) for w in data.warning],final_phase=phase,
            final_keeper_q_m=float(data.qpos[keeper.qpos]))
        return finish()

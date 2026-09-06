"""Actual-site handoff for the positive pin hand-chain keeper.

This is an abstract two-input mechanism controller. It does not certify human
reach, hand trajectories, structural strength, or autonomous brake behavior.
All forces act at authored material sites; it never modifies native state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np

from .rollup_hoist import hoist_control


@dataclass(frozen=True)
class KeeperRules:
    joint: int
    qpos: int
    dof: int
    grip_site: int
    grip_name: str
    pin_geom: int
    chain_geoms: frozenset[int]
    withdrawn_q: float
    force_limit: float
    excluded_grip_z: tuple[float, float]


def compile_keeper(model, meta):
    """Fail closed when the positive pin, real grip or return spring is absent."""
    import mujoco
    row=meta.get('rollup_hoist',{}).get('keeper')
    if not isinstance(row,dict) or row.get('schema_version')!=1 or row.get('kind')!='spring_return_positive_roller_chain_pin':
        raise ValueError('Supported physical chain-keeper metadata is required')
    j=model.joint(row['joint']).id;site=model.site(row['grip_site']).id;body=model.body(row['body']).id
    if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_SLIDE or model.jnt_bodyid[j]!=body or model.site_bodyid[site]!=body:
        raise ValueError('Keeper input must be its actual sliding pin and attached grip')
    if not np.array_equal(model.jnt_axis[j],[0.,1.,0.]) or not np.allclose(model.jnt_range[j],[0.,.08],atol=1e-12):
        raise ValueError('Keeper native axis/stroke differs from the supported mechanism')
    if model.body_parentid[body]!=0 or not np.allclose(model.body_quat[body],[1.,0.,0.,0.]):
        raise ValueError('Keeper world force requires its authored fixed world orientation')
    pin=model.geom(row['pin_geom']).id
    if model.geom_bodyid[pin]!=body or not (model.geom_contype[pin] and model.geom_conaffinity[pin]):
        raise ValueError('Keeper pin must carry load through active native contact')
    spring=model.tendon(row['spring']).id
    if not np.isclose(model.tendon_stiffness[spring],200.) or not np.allclose(model.tendon_lengthspring[spring],[.15,.15]):
        raise ValueError('Keeper requires its actual 2–18 N guided return spring')
    if row.get('withdrawn_q_m')!=.08 or row.get('hand_force_limit_N')!=120.:
        raise ValueError('Unexpected keeper stroke or manual force limit')
    interval=tuple(row['excluded_chain_grip_z_m'])
    if len(interval)!=2 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in interval) or interval[0]>=interval[1]:
        raise ValueError('Keeper hand exclusion interval is malformed')
    geoms=frozenset(g for g in range(model.ngeom) if model.geom(g).name.startswith('hoist_link_'))
    if not geoms:raise ValueError('Keeper requires the original physical material hand chain')
    return KeeperRules(j,int(model.jnt_qposadr[j]),int(model.jnt_dofadr[j]),site,row['grip_site'],pin,geoms,.08,120.,interval)


def keeper_pin_load(model, data, rules):
    """Sum measured compressive pin/chain contact loads, excluding its bearings."""
    import mujoco
    load=0.
    for i,c in enumerate(data.contact):
        other=c.geom2 if c.geom1==rules.pin_geom else c.geom1 if c.geom2==rules.pin_geom else -1
        if other in rules.chain_geoms:
            force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force);load+=max(0.,float(force[0]))
    return load


def keeper_site_force(model, data, rules, target_q):
    """Bounded actual pull-knob force; native passive spring force is retained."""
    if isinstance(target_q,bool) or not isinstance(target_q,(int,float,np.number)) or not math.isfinite(target_q) or not 0<=target_q<=rules.withdrawn_q:
        raise ValueError('Keeper target must be inside its actual finite stroke')
    force=float(np.clip(600.*(target_q-data.qpos[rules.qpos])-40.*data.qvel[rules.dof]-data.qfrc_passive[rules.dof],-rules.force_limit,rules.force_limit))
    return {rules.grip_name:[0.,force,0.]}


def keeper_open_force(model, data, rules):
    """Second-hand force while operating the chain with the pin withdrawn."""
    return keeper_site_force(model,data,rules,rules.withdrawn_q)


def _hand_rules(data, hoist, keeper):
    # Never ask a hand to grip inside the steel keeper/receiver channel.
    allowed=[i for i,s in enumerate(hoist.site_ids)
        if not keeper.excluded_grip_z[0]<=data.site_xpos[s,2]<=keeper.excluded_grip_z[1]]
    if not allowed:raise ValueError('No material chain grip clears the keeper hand exclusion')
    return replace(hoist,site_ids=tuple(hoist.site_ids[i] for i in allowed),
        site_names=tuple(hoist.site_names[i] for i in allowed),grip_height=.95)


def _closed_floor_support(model, data, hoist):
    """Measured upward floor reaction on the actual bottom-bar subtree.

    Height alone is not support: removed-floor and suspended-curtain states
    must still take the chain load before withdrawing the keeper.
    """
    import mujoco
    floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    reaction=0.;contact_support=False
    bottom=int(model.site_bodyid[hoist.bottom_site])
    if floor>=0 and model.geom_contype[floor] and model.geom_conaffinity[floor]:
        for i,c in enumerate(data.contact):
            other=c.geom2 if c.geom1==floor else c.geom1 if c.geom2==floor else -1
            if other<0:continue
            body=int(model.geom_bodyid[other])
            while body and body!=bottom:body=int(model.body_parentid[body])
            # Contact normal points from geom1 toward geom2. Count only the
            # upward force on the bottom bar, not side friction or a roof.
            vertical=float(c.frame[2])*(1. if c.geom1==floor else -1.)
            if body==bottom and vertical>.95:
                force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
                reaction+=max(0.,float(force[0]))*vertical
                contact_support|=float(c.dist)<=0.
        if not contact_support:
            # Unilateral contact can chatter across zero by sub-micrometre
            # amounts after settling. Bind that numerical band to the actual
            # floor/seal surfaces, never merely the reported closed height.
            seal=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'curtain_astragal')
            if seal>=0 and model.geom_contype[seal] and model.geom_conaffinity[seal]:
                body=int(model.geom_bodyid[seal])
                while body and body!=bottom:body=int(model.body_parentid[body])
                if body==bottom:
                    points=np.zeros(6)
                    distance=float(mujoco.mj_geomDistance(model,data,floor,seal,.001,points))
                    upward=points[5]>=points[2] and np.linalg.norm(points[3:5]-points[:2])<=1e-7
                    contact_support=upward and 0.<=distance<=1e-5
    jac=np.zeros((3,model.nv));mujoco.mj_jacSite(model,data,jac,None,hoist.bottom_site)
    supported=(contact_support and abs(float(data.site_xpos[hoist.bottom_site,2])-hoist.closed_z)<.02
        and abs(float(jac[2]@data.qvel))<.01)
    return bool(supported),reaction


def begin_keeper_transition(model, data, hoist, keeper, *, mode):
    if mode not in ('engage','release'):raise ValueError('Keeper transition is engage or release')
    now=float(data.time)
    if mode=='engage' and data.site_xpos[hoist.bottom_site,2]<hoist.closed_z+.08:
        raise ValueError('A floor-supported closed curtain does not require a seated keeper hold; do not lift it merely to align a roller gap')
    floor_supported,floor_force=_closed_floor_support(model,data,hoist)
    floor_release=mode=='release' and floor_supported and floor_force>=5. and keeper_pin_load(model,data,keeper)<5.
    return {'mode':mode,'phase':'engage' if mode=='engage' else 'settle_on_floor' if floor_release else 'unload',
        'start_time_s':now,'phase_start_s':now,'last_time_s':now,'stable_since_s':None,
        'initial_bottom_m':float(data.site_xpos[hoist.bottom_site,2]),
        'initial_floor_reaction_N':floor_force,'release_support':'measured_floor' if floor_release else 'chain_hand',
        'initial_keeper_q_m':float(data.qpos[keeper.qpos]),'done':False,'failed':False,'reason':None}


def _smooth(t):
    u=min(1.,max(0.,t));return min(1.,max(0.,u**3*(10.+u*(-15.+6.*u))))


def keeper_transition_action(model, data, hoist, keeper, state):
    """Return both real-site inputs and a new JSON state; never writes physics.

    Release unloads the positive pin before withdrawing it. At the closed
    floor-supported state, measured floor reaction already carries the load;
    the pin is withdrawn without an unnecessary opposing chain preload.
    Engagement searches
    a nearby roller gap with a bounded chain hold, transfers load gradually to
    the captured pin, then verifies two seconds with both hands absent. A failed
    transition is explicit; no coordinate lock or continuing invisible hand is
    substituted for the missing load path.
    """
    import mujoco
    s=dict(state);now=float(data.time)
    if now<s['last_time_s']-1e-9:raise ValueError('Keeper transition native clock moved backwards')
    dt=max(0.,now-s['last_time_s']);s['last_time_s']=now
    if s['done'] or s['failed']:
        return {'next_state':s,'site_forces':{},'phase':s['phase'],'done':s['done'],'failed':s['failed'],'reason':s['reason']}
    elapsed=now-s['start_time_s'];phase_elapsed=now-s['phase_start_s'];phase=s['phase']
    if phase not in ('engage','transfer','verify_hold','unload','withdraw','settle_on_floor','withdraw_from_floor'):raise ValueError('Unknown keeper transition phase')
    height=float(data.site_xpos[hoist.bottom_site,2]);q=float(data.qpos[keeper.qpos]);speed=float(data.qvel[keeper.dof]);forces={};chain_force=0.
    if phase not in ('verify_hold','settle_on_floor','withdraw_from_floor'):
        h=_hand_rules(data,hoist,keeper)
        action=hoist_control(model,data,h,opening=True,elapsed_s=max(0.,elapsed))
        jac=np.zeros((3,model.nv));mujoco.mj_jacSite(model,data,jac,None,hoist.bottom_site)
        velocity=float(jac[2]@data.qvel)
        if phase in ('unload','withdraw'):
            target=s['initial_bottom_m']+.007*_smooth(now-s.get('load_start_s',s['start_time_s']))
        else:
            target=max(hoist.closed_z+.002,s['initial_bottom_m']-min(.02,max(0.,elapsed-1.)*.007))
        effort=60.+3000.*(target-height)-400.*velocity
        if phase=='engage':
            # A fixed 60 N load guess can exactly cancel the small seating
            # descent on a well-counterbalanced curtain. Integrate measured
            # height error so a roller gap can actually pass the waiting pin.
            # Keep the same material-chain input and 120 N cap; do not move
            # the pin or chain coordinates to manufacture alignment.
            integral=float(s.get('engage_integral_force_N',0.))
            delta=1500.*(target-height)*dt
            raw=effort+integral
            if abs(raw)<hoist.force_limit or raw*delta<0.:
                integral=float(np.clip(integral+delta,-hoist.force_limit,hoist.force_limit))
            s['engage_integral_force_N']=integral
            s['engage_target_bottom_m']=target
            effort+=integral
        chain_force=-float(np.clip(effort,-hoist.force_limit,hoist.force_limit))
        if phase=='transfer':chain_force=s['transfer_force_N']*(1.-_smooth(phase_elapsed/.5))
        forces[action['site']]=[0.,0.,chain_force]
        if phase=='engage':
            target_q=min(keeper.withdrawn_q,max(0.,s['initial_keeper_q_m']))*(1.-_smooth(elapsed))
            forces.update(keeper_site_force(model,data,keeper,target_q))
        elif phase=='unload' and 'handoff_keeper_q_m' in s:
            forces.update(keeper_site_force(model,data,keeper,s['handoff_keeper_q_m']))
        elif phase=='withdraw':
            start_q=s.get('handoff_keeper_q_m',0.)
            forces.update(keeper_site_force(model,data,keeper,start_q+(keeper.withdrawn_q-start_q)*_smooth(phase_elapsed)))
    load=keeper_pin_load(model,data,keeper)
    supported=False;reaction=0.
    if phase in ('settle_on_floor','withdraw_from_floor'):
        supported,reaction=_closed_floor_support(model,data,hoist)
        s['floor_reaction_N']=reaction
        # Reaction can momentarily become zero while the bottom bar remains
        # in actual unilateral floor contact. Entry required measured load;
        # continuation requires that same physical support and an unloaded pin.
        floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
        floor_active=floor>=0 and model.geom_contype[floor] and model.geom_conaffinity[floor]
        if not floor_active:
            s.update(failed=True,reason='measured_floor_support_lost_during_keeper_withdrawal')
        elif load>=5.:
            # A weak counterbalance may load the pin as chain slack settles.
            # Stop withdrawing, hold the actual pin, and take that load with
            # the chain before continuing under the unchanged unloading gate.
            hold_q=min(keeper.withdrawn_q,max(0.,q))
            s.update(phase='unload',phase_start_s=now,stable_since_s=None,
                initial_bottom_m=height,load_start_s=now,handoff_keeper_q_m=hold_q,
                floor_to_chain_handoff_s=elapsed,floor_handoff_pin_load_N=load,
                release_support='floor_then_chain_hand')
            forces.update(keeper_site_force(model,data,keeper,hold_q))
            return {'next_state':s,'site_forces':forces,'phase':'unload','done':False,'failed':False,
                'reason':None,'pin_chain_normal_force_N':load,'keeper_q_m':q,
                'scope':'Withdrawal paused while the actual chain hand takes the newly measured pin load'}
        elif phase=='withdraw_from_floor':
            # Pause withdrawal during actual floor-contact chatter. No chain
            # preload is introduced and no unsupported time advances the pin
            # target. The finite timeout still rejects genuinely lost support.
            progress=s.get('supported_withdrawal_s',0.)
            if supported:
                progress+=dt;s['supported_withdrawal_s']=progress
                target_q=keeper.withdrawn_q*_smooth(progress)
            else:
                s['floor_pause_s']=s.get('floor_pause_s',0.)+dt
                target_q=min(keeper.withdrawn_q,max(0.,q))
            forces.update(keeper_site_force(model,data,keeper,target_q))
    stable=False;duration=.15
    if phase=='settle_on_floor':stable=elapsed>.25 and supported and reaction>=5. and load<5.;duration=.1
    elif phase=='unload':stable=phase_elapsed>.1 and load<5.;duration=.1
    elif phase in ('withdraw','withdraw_from_floor'):
        clock=s.get('supported_withdrawal_s',0.) if phase=='withdraw_from_floor' else phase_elapsed
        stable=clock>1. and q>keeper.withdrawn_q-.002 and abs(speed)<.03;duration=.1
    elif phase=='engage':stable=elapsed>1. and q<.002 and abs(speed)<.02
    if stable:
        if s['stable_since_s'] is None:s['stable_since_s']=now
    else:s['stable_since_s']=None
    confirmed=s['stable_since_s'] is not None and now-s['stable_since_s']>=duration
    next_phase=None
    if phase=='settle_on_floor' and confirmed and not s['failed']:
        next_phase='withdraw_from_floor';s['floor_settled_time_s']=elapsed
    elif phase=='unload' and confirmed:
        next_phase='withdraw';s['unloaded_time_s']=now-s['start_time_s'];s['unloaded_pin_load_N']=load
    elif phase in ('withdraw','withdraw_from_floor') and confirmed and not s['failed']:
        s.update(done=True,reason='actual_pin_withdrawn_under_measured_floor_support' if phase=='withdraw_from_floor' else 'actual_pin_withdrawn_after_chain_unload')
    elif phase=='engage' and confirmed:next_phase='transfer';s['transfer_force_N']=chain_force
    elif phase=='transfer' and phase_elapsed>=.5:next_phase='verify_hold';s['hand_free_start_bottom_m']=height
    elif phase=='verify_hold':
        drift=height-s['hand_free_start_bottom_m'];s['hand_free_drift_m']=drift
        if abs(drift)>.04 or q>.02:s.update(failed=True,reason='positive_pin_failed_to_retain_curtain')
        elif phase_elapsed>=2. and q<.002 and abs(speed)<.02:s.update(done=True,reason='positive_pin_two_second_hands_free_hold')
    if next_phase:s.update(phase=next_phase,phase_start_s=now,stable_since_s=None)
    if not s['done'] and not s['failed'] and ((phase in ('unload','withdraw','withdraw_from_floor') and phase_elapsed>5.) or (phase=='settle_on_floor' and phase_elapsed>2.) or (phase=='engage' and elapsed>8.) or elapsed>15.):
        s.update(failed=True,reason='force_limited_keeper_transition_not_completed')
    if s['failed']:forces={}
    return {'next_state':s,'site_forces':forces,'phase':s['phase'],'done':s['done'],'failed':s['failed'],
        'reason':s['reason'],'pin_chain_normal_force_N':load,'keeper_q_m':q,
        'scope':'Abstract two-site mechanism input; no embodied hand, structural strength or task certification'}

"""Automated QA / sign-off for a generated door.

Checks (all tiers where applicable):
  load        MJCF loads in MuJoCo (full / simple / minimal)
  clearance   geometric gate: nothing interpenetrates anywhere in the travel (doorbench/clearance.py)
  running_clearance  geometric gate: no moving collider ever TOUCHES static structure - every structural
              moving/static pair keeps a real running clearance at rest and through the sweep (seals, bearings,
              latches and stops are allow-listed by semantics; see clearance.required_gap)
  settle      1 s free simulation: no warnings, no deep initial penetrations, primary joint drift small
  hold        latched door resists a strong opening torque/force (if it has a latch/lock, or is a locked rotor / bolted flap)
  free_opens  a leaf that nothing holds (no latch, no lock; every free-swing family) must move past a threshold under
              the same push - a leaf that stays shut is jammed by its own geometry or couplings
  actuate     driving the operator retracts the latch and the door opens (if robot-side release exists)
  return      releasing the operator lets the spring latch re-extend
  relatch     closing the door re-latches (spring latches with strike lip)
  closer      self-closing doors return to closed from 60 deg
  free_opens  free-swing / rotary families (saloon, revolving, turnstiles, folding, bypass, flaps, strips): the QA push
              moves the primary joint past 10 deg / 50 mm (locked rotors: it holds within its locked play instead)
  no_jam      ... and while it moves no static geometry presses on a moving part with more than JAM_FORCE_N: a zero-gap
              touch or a sub-tolerance interpenetration that the geometric clearance gate cannot see stalls the door
  urdf        URDF loads in MuJoCo (structure check)
  usd         USD stage opens; joint & rigid-body counts match the IR
Writes qa.json with pass/fail per check, metrics, and a signed_off flag.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import hardware as H


from contextvars import ContextVar
_QA_FIELDS = ContextVar("doorbench_qa_fields", default=None)


def _qa_native(model, data, step):
    import mujoco
    from .geometry.gate_hardware import apply_magnetic_latches
    from .closer_pinion import apply_pinion_closers
    from .closer_track_hold import apply_track_holds
    from .turnstile_locks import apply_turnstile_locks
    from .turnstile_drop import apply_turnstile_drop
    from .rotary_lockset import apply_rotary_catches
    rules = (_QA_FIELDS.get() or {}).get(id(model), {})
    fn = mujoco.mj_step if step else mujoco.mj_forward
    if not rules:
        return fn(model, data)
    previous = mujoco.get_mjcb_passive()
    def callback(m, d):
        if previous is not None:
            previous(m, d)
        if m is model:
            apply_magnetic_latches(m, d, rules.get('magnetic',()))
            apply_pinion_closers(m, d, rules.get('pinion',()))
            apply_track_holds(m,d,rules.get('track_holds',()))
            apply_turnstile_locks(m,d,rules.get('turnstile_locks',()))
            apply_turnstile_drop(m,d,rules.get('turnstile_drop',()))
            apply_rotary_catches(m,d,rules.get('rotary_catches',()))
    mujoco.set_mjcb_passive(callback)
    try:
        return fn(model, data)
    finally:
        for i in range(mujoco.mjtWarning.mjNWARNING):
            if data.warning[i].number:
                rules['warnings'].add(mujoco.mjtWarning(i).name)
        mujoco.set_mjcb_passive(previous)


def _qa_step(model, data):
    return _qa_native(model, data, True)


def _qa_forward(model, data):
    return _qa_native(model, data, False)


def _jid(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def _q(m, d, jid):
    return float(d.qpos[m.jnt_qposadr[jid]])


def _steps(m, duration_s):
    """Keep physical QA durations independent of a mechanism's integration step."""
    return range(int(math.ceil(float(duration_s) / float(m.opt.timestep))))


def _prepare_closer_service(m, d, spec, metadata, door_dir, multipoint=None):
    """Prepare an unlocked service fixture before prescribing an open angle.

    An extended deadbolt hitting its keeper says nothing about closer return.
    Modeled inputs are operated through native forces while the leaf remains
    closed. Missing keyed inputs remain unsupported, never silently removed.
    """
    import mujoco
    mujoco.mj_resetData(m,d)
    if not spec['lock'].get('engaged'):
        return {'ok':True,'scope':'Authored unlocked initial state'}
    if metadata.get('multipoint_locks'):
        if not multipoint or not multipoint['ok']:
            return {'ok':False,'reason':'Native multipoint release cycle failed'}
        for row in multipoint['results']:
            for name,value in row['released_joints'].items():
                d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
        return {'ok':True,'scope':'Unlocked service fixture from the successful native multipoint depression cycle'}
    ir=json.load(open(os.path.join(door_dir,'model.json')))
    driven=[]
    egress=False
    for body in ir['bodies']:
        joint=body.get('joint') or {}
        if joint.get('role')!='lock':continue
        name=joint['name']
        if not any(term in name for term in ('thumbturn_hinge','slide_bolt_slide','slide_latch_slide','pin_slide','aux_bolt_slide')):continue
        jid=_jid(m,name)
        if jid<0 or not m.jnt_limited[jid]:continue
        limit=1.2 if int(m.jnt_type[jid])==int(mujoco.mjtJoint.mjJNT_HINGE) else 60.
        driven.append((jid,limit))
    if not driven:
        # An exit device releases its spring latch from the inside even when
        # the outside trim is locked. Operate the actual installed input;
        # neither a permission flag nor a changed joint range is a release.
        inputs=set(metadata.get('inside_egress_inputs',[]))
        inputs.update(body['joint']['name'] for body in ir['bodies']
                      if (body.get('joint') or {}).get('role')=='operator'
                      and '_exit_device_' in body['joint']['name'])
        for name in sorted(inputs):
            jid=_jid(m,name)
            if jid>=0 and m.jnt_limited[jid] and m.jnt_range[jid,1]>m.jnt_range[jid,0]+.002:
                limit=4. if int(m.jnt_type[jid])==int(mujoco.mjtJoint.mjJNT_HINGE) else 120.
                driven.append((jid,limit))
        egress=bool(driven)
    if not driven:
        return {'ok':False,'reason':'The engaged lock has no modeled service-release input; an unlocked open fixture cannot be established'}
    targets=[]
    lock_bodies={body['name'] for body in ir['bodies'] if (body.get('joint') or {}).get('role')=='lock'}
    for row in metadata.get('lock_stock',[]):
        if not egress and row['bolt_body'] not in lock_bodies:continue
        jid=_jid(m,row['name']+'_slide')
        if jid>=0 and m.jnt_range[jid,1]>.002:
            targets.append(jid)
    if not targets:targets=[j for j,_ in driven]
    for _ in _steps(m,2.):
        d.qfrc_applied[:]=0.
        for j,limit in driven:
            a,v=int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j])
            gain=8. if int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_HINGE) else 2000.
            damping=.15 if gain==8. else 10.
            # An egress input has a return spring. Maintain bounded effort at
            # the stop rather than asking a zero-error PD to hold its preload.
            overdrive=limit/gain if egress else 0.
            d.qfrc_applied[v]=np.clip(gain*(m.jnt_range[j,1]+overdrive-d.qpos[a])-damping*d.qvel[v],-limit,limit)
        _qa_step(m,d)
    d.qfrc_applied[:]=0.
    states={m.joint(j).name:_q(m,d,j) for j in targets}
    ok=all(_q(m,d,j)>=m.jnt_range[j,1]-.001 for j in targets)
    return {'ok':ok,'scope':('Two-second native inside egress input withdraws the latch; the outside lock remains engaged'
            if egress else 'Two-second native service release; access from the installed input face, independent of robot approach'),
            'inputs':[{'joint':m.joint(j).name,'effort_limit':limit} for j,limit in driven],
            'released_joint_positions':states,'reason':None if ok else 'Modeled input did not fully withdraw the lock'}


def _max_pen(m, d):
    worst = (0.0, None, None)
    import mujoco
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist < worst[0]:
            worst = (float(c.dist), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
    return worst


G = 9.81
PUSH_BASE_MAX = {"hinge": 60.0, "slide": 80.0}   # N*m / N: a strong human or robot leaning on a full-size door leaf
PUSH_BASE_MIN = 2.0                              # N*m / N: a cat on a flap - below this nothing would open at all
PUSH_CAP = {"hinge": 800.0, "slide": 4000.0}


def push_base(unit: str, mass_kg: float | None = None, width_m: float | None = None) -> float:
    """Base term of the adaptive QA push, scaled by the leaf's own weight moment.

    A flat 60 N*m is the effort a person applies to a 20-100 kg door leaf; on a 0.14-1.4 kg pet flap it is ~100x the
    mechanism's own scale.  The flap's inertia about its hinge is ~m W^2 / 3, so 60 N*m accelerates it at some
    2000 rad/s^2 and it reaches 30-85 rad/s (1700-4900 deg/s) before it hits its stop - MuJoCo only survives that
    because its limits are soft, PhysX's articulation limit explodes within a few steps and the door reads NaN.

    ``0.5 * m * g * W`` is half the moment gravity would exert on the leaf if it lay horizontal: the effort scale of
    the mechanism itself, in the same units as the push (N*m about a hinge, and ``0.5 * m * g`` newtons on a slide,
    where there is no lever arm).  Clamped to [2, 60] N*m ([2, 80] N), so every door of 20 kg and up keeps exactly the
    push it had, and only leaves too light to justify it get less.

    Known approximation (verified 2026-09, 211 doors get a reduced base and none of them changes a QA verdict).
    ``mass_kg`` is the whole leaf assembly and ``width_m`` the spec's leaf width, which is the gravity moment arm only
    for a leaf hinged on a VERTICAL axis carrying its whole mass.  It is not the arm for the 48 ``hinge_horizontal``
    doors, and on the 8 strip curtains it is wrong in both factors at once: the primary joint carries ONE 0.58 kg
    strip whose COM hangs H/2 = 1.19 m below the hinge, so the physical moment is m_strip * g * H/2 = 6.7 N*m while
    the formula returns 0.5 * m_curtain * g * W = 4.2 N*m - within a factor 1.6 because the two errors cancel.  The
    obvious repair (subtree mass times the perpendicular distance from the axis to the subtree COM) is NOT correct
    either: a revolving door or turnstile is balanced about its axis, that distance is 0, and the formula would
    collapse to the 2 N*m floor on a 100 kg rotor.  Sizing a balanced rotor needs inertia, not gravity, so this is
    left as is and documented rather than replaced with a formula that is wrong somewhere else.
    Measured effect of the approximation: 18 doors differ by more than 5 % from the height-arm value, and on 10 of
    them (hatches, pet flaps) the base is a small part of the push anyway (the bias term is 44-420 N*m)."""
    cap = PUSH_BASE_MAX["hinge" if unit == "hinge" else "slide"]
    if not mass_kg or mass_kg <= 0:
        return cap
    if unit == "hinge":
        if not width_m or width_m <= 0:
            return cap
        scale = 0.5 * float(mass_kg) * G * float(width_m)
    else:
        scale = 0.5 * float(mass_kg) * G
    return float(min(cap, max(PUSH_BASE_MIN, scale)))


def qa_push(m, d, pj, mass_kg: float | None = None, width_m: float | None = None, model_meta: dict | None = None) -> dict:
    """The adaptive QA push on the primary joint (N*m for hinges, N for slides): twice the static resistance at rest
    (gravity bias + Coulomb friction + spring preload) plus a base effort sized by the leaf (``push_base``), capped -
    a strong human / robot.  Mirrored by ``parity.protocol`` (which imports ``push_base`` / ``PUSH_CAP`` from here)."""
    import mujoco
    is_hinge = int(m.jnt_type[pj]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    dof = m.jnt_dofadr[pj]
    mujoco.mj_resetData(m, d)
    _qa_forward(m, d)
    bias = abs(float(d.qfrc_bias[dof] - d.qfrc_passive[dof]))
    fl = float(m.dof_frictionloss[dof])
    preload = abs(float(m.jnt_stiffness[pj] * m.qpos_spring[m.jnt_qposadr[pj]])) if m.jnt_stiffness[pj] > 0 else 0.0
    projected = None
    if model_meta and model_meta.get("garage_tiltup_linkage"):
        from .geometry.garage_tiltup import projected_static_resistance
        projected = projected_static_resistance(m, d, model_meta)
    elif model_meta and model_meta.get('closer_pinion_laws'):
        from .closer_pinion import projected_static_resistance
        projected = projected_static_resistance(m,d,model_meta)
    if projected is not None:
        bias = max(0., float(projected["static_resistance"]))
        fl = float(projected["frictionloss"])
        preload = 0.  # Already included in J · (bias - passive).
    unit = "hinge" if is_hinge else "slide"
    base = push_base(unit, mass_kg, width_m)
    push = min(2.0 * (bias + fl + preload) + base, PUSH_CAP[unit])
    return {"push": push, "bias": bias, "frictionloss": fl, "preload": preload, "is_hinge": is_hinge, "push_base": base, "mechanism_projection": projected}


def push_primary(m, d, pj, push: float, has_holding: bool, thr_free: float) -> float:
    """The ``hold`` / ``free_opens`` drive from the reset state: push for 1 s (a held leaf) or up to 6 s (a free leaf,
    stopping once it is past ``thr_free``); returns the primary joint value at exit."""
    import mujoco
    mujoco.mj_resetData(m, d)
    for k in _steps(m, 1. if has_holding else 6.):
        d.qfrc_applied[:] = 0
        d.qfrc_applied[m.jnt_dofadr[pj]] = push
        _qa_step(m, d)
        if (k + 1) * m.opt.timestep >= 1. and not has_holding and _q(m, d, pj) > thr_free:
            break   # heavy leaves (big gates, vault doors) need longer to accelerate
    return _q(m, d, pj)


SPRING_LATCH_KINDS = ("tubular_latch", "deadlatch", "mortise_latch", "rim_latch", "vertical_rods", "hook", "gravity_bar", "dogs", "multi_bolt", "electric_bolt")
ENV_RELEASE_LOCK_KINDS = ("mag_lock", "delayed_egress", "card_reader", "electric_strike", "interlock", "credential_index_bolt")
FREE_SWING_FAMILIES = ("saloon", "strip_curtain", "pet_door", "turnstile_tripod", "turnstile_fullheight", "revolving", "bifold", "accordion", "sliding_bypass")
JAM_FORCE_N = 20.0       # N; largest contact normal force static geometry may exert on a moving part while a free door is pushed
#                          (all 147 free-swing doors read exactly 0 N after the 2026-09 fixes: a free leaf is carried by its joint;
#                          20 N already means a leaf resting on the floor or scraping a jamb without stalling - a visible defect)
FREE_PUSH_S = 6.0        # s; a free door is pushed for up to this long (stops once past thr_free after MIN_PUSH_S)
MIN_PUSH_S = 1.0


def jam_sweep(m, d, pj: int, push: float, thr_free: float, duration_s: float = FREE_PUSH_S, min_push_s: float = MIN_PUSH_S, end_position: float | None = None) -> dict:
    """Push the primary joint of a door that nothing holds shut and watch what static geometry does to it.

    The push runs for up to ``duration_s`` and stops once the joint is past ``thr_free`` after ``min_push_s`` (the
    free-door hold schedule of qa.py / the parity protocol).  Every step, every contact between a body that can move
    and static geometry (welded to the world) is measured with ``mj_contactForce``; the largest normal force and its
    geom pair are returned.  A free-swinging or rotating door is carried by its joint and has nothing static pressing
    on it while it moves (brush seals aside), so a large force here is a jam: an interpenetration too shallow for the
    geometric clearance gate, or a zero-gap touch whose degenerate contact normal is orthogonal to the only DOF
    (coplanar box faces) - either one stalls the door under the push.
    """
    import mujoco
    dof, qadr = m.jnt_dofadr[pj], m.jnt_qposadr[pj]
    static = np.asarray(m.body_weldid)[np.asarray(m.geom_bodyid)] == 0
    f6 = np.zeros(6)
    peak, pair, t_free = 0.0, None, None
    dt = float(m.opt.timestep)
    n, n_min = int(round(duration_s / dt)), int(round(min_push_s / dt))
    mujoco.mj_resetData(m, d)
    k = -1
    for k in range(n):
        d.qfrc_applied[:] = 0
        effort = push
        if end_position is not None:
            # A constant force accelerating a light panel into a real endstop
            # measures an impact, not a track jam. Brake before the verified
            # endpoint; still measure EVERY contact with the same force limit.
            mass = max(.1, float(m.body_mass[m.jnt_bodyid[pj]]))
            remaining = max(0., end_position-.010-float(d.qpos[qadr]))
            acceleration = max(.1, (push-float(m.dof_frictionloss[dof]))/mass)
            velocity = min(.4, math.sqrt(2*acceleration*remaining))
            effort = float(np.clip(push/.08*(velocity-d.qvel[dof]), -push, push))
        d.qfrc_applied[dof] = effort
        _qa_step(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if static[c.geom1] == static[c.geom2]:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(float(f6[0]))
            if fn > peak:
                peak = fn
                pair = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
        q = float(d.qpos[qadr])
        if t_free is None and q > thr_free:
            t_free = (k + 1) * dt
        if k >= n_min - 1 and q > thr_free:
            break
    return {"moved": float(d.qpos[qadr]), "t_free": t_free, "t_end": (k + 1) * dt, "peak_force_N": peak, "peak_pair": list(pair) if pair else None}


def door_flags(spec: dict) -> dict:
    """What the QA expects of a door, derived from its spec (shared with the tests).

    spring_latch      a spring latch holds the leaf shut until the operator retracts it
    lock_engaged      an engaged lock that physically holds (not a child cover / jammed hardware)
    has_holding       latched or locked: a push on the closed leaf must not open it   -> check "hold"
    env_release_only  the lock is released by environment logic (badge / REX / timer), not by the operator
    can_release       driving the operator (and any thumbturn / bolt / dogs) must open the door -> "actuate_opens"
    free_swing        families without a latch that a push must open                    -> "free_opens"
    """
    lk = H.LOCKS[spec["lock"]["model"]]
    lt = H.LATCHES[spec["latch"]["model"]]
    spring_latch = lt.throw > 0 and lt.kind in SPRING_LATCH_KINDS
    lock_engaged = bool(spec["lock"]["engaged"]) and lk.kind not in ("none", "child_lock_cover", "jam_stuck")
    env_release_only = bool(spec["lock"]["engaged"]) and lk.kind in ENV_RELEASE_LOCK_KINDS
    can_release = not spec['lock'].get('engaged', False) or bool(spec["lock"].get("robot_side_release", True)) or lk.kind == "jam_stuck"
    return {"spring_latch": spring_latch, "lock_engaged": lock_engaged, "has_holding": spring_latch or lock_engaged or H.OPERATORS[spec["operator"]["model"]].kind == "cremone", "env_release_only": env_release_only,   # cremone shoot bolts are the door's latch
            "can_release": can_release, "free_swing": spec["family"] in FREE_SWING_FAMILIES, "lock_kind": lk.kind, "latch_kind": lt.kind}


def run_qa(spec: dict, door_dir: str, model_meta: dict, files: dict, phys: dict) -> dict:
    token = _QA_FIELDS.set({})
    try:
        from .native_warnings import capture_native_warnings
        with capture_native_warnings() as native_messages:
            report = _run_qa(spec, door_dir, model_meta, files, phys)
        report['checks']['native_warning_messages_absent']=not native_messages
        report['metrics']['native_warning_messages']=native_messages
        report['signed_off']=bool(report['signed_off'] and not native_messages)
        import hashlib
        from pathlib import Path
        report['source_sha256'] = {name: hashlib.sha256((Path(door_dir)/name).read_bytes()).hexdigest()
                                   for name in ('spec.json','model.json','door.xml') if (Path(door_dir)/name).exists()}
        return report
    finally:
        _QA_FIELDS.reset(token)


def _run_qa(spec: dict, door_dir: str, model_meta: dict, files: dict, phys: dict) -> dict:
    import mujoco
    t0 = time.time()
    checks = {}
    metrics = {}
    fam = spec["family"]
    kin = spec["kinematics"]["type"]
    incomplete=list(model_meta.get('mechanical_incomplete',[]))
    for closer in model_meta.get('closer_pinion_calibration',[]):
        incomplete.extend({'component':closer['pinion_joint'],'reason':f'{feature} is not modeled'}
                          for feature in closer.get('unmodeled_features',[]))
    checks['declared_mechanisms_complete']=not incomplete
    if incomplete:metrics['mechanical_incomplete']=incomplete
    # ---- load all tiers
    models = {}
    for tier, path in files.get("mjcf", {}).items():
        try:
            models[tier] = mujoco.MjModel.from_xml_path(path)
            from .geometry.gate_hardware import compile_magnetic_latches
            from .closer_pinion import compile_pinion_closers
            from .closer_track_hold import compile_track_holds
            from .turnstile_locks import compile_turnstile_locks
            from .turnstile_drop import compile_turnstile_drop
            from .rotary_lockset import compile_rotary_catches
            _QA_FIELDS.get()[id(models[tier])] = {'magnetic':compile_magnetic_latches(models[tier], model_meta),
                                                 'pinion':compile_pinion_closers(models[tier], model_meta),
                                                 'track_holds':compile_track_holds(models[tier],model_meta),
                                                 'turnstile_locks':compile_turnstile_locks(models[tier],model_meta),
                                                 'turnstile_drop':compile_turnstile_drop(models[tier],model_meta),
                                                 'rotary_catches':compile_rotary_catches(models[tier],model_meta),'warnings':set()}
            checks[f"load_{tier}"] = True
        except Exception as e:
            checks[f"load_{tier}"] = False
            metrics[f"load_{tier}_error"] = str(e)[:300]
    if "full" not in models:
        return {"checks": checks, "metrics": metrics, "signed_off": False, "time_s": time.time() - t0}
    # ---- deterministic kinematic clearance gates (all geometry collidable, every joint swept)
    #   clearance          nothing INTERPENETRATES anywhere in the travel
    #   running_clearance  and nothing TOUCHES either: every moving collider keeps its running clearance from the
    #                      static structure (3 mm at jambs / head, 6 mm over the floor, 10 mm on a rotor), because a
    #                      0.000 m touch is free in MuJoCo at margin 0 and a jam in PhysX inside its contact offset
    try:
        from .clearance import run_clearance
        cl = run_clearance(door_dir, "full")
        checks["clearance"] = bool(cl["ok"])
        metrics["clearance_n_failures"] = cl["n_failures"]
        metrics["clearance_failures"] = cl["failures"][:10]
        rc = cl["running"]
        checks["running_clearance"] = bool(rc["ok"])
        metrics["running_clearance_n_failures"] = rc["n_failures"]
        metrics["running_clearance_failures"] = rc["failures"][:10]
        metrics["running_clearance_n_pairs"] = rc.get("n_pairs", 0)
    except Exception as e:
        checks["clearance"] = False
        checks["running_clearance"] = False
        metrics["clearance_error"] = str(e)[:200]
    m = models["full"]
    if model_meta.get('wall_switches'):
        from .wall_switch_qa import run_wall_switch_qa
        for tier,native in models.items():
            switches=run_wall_switch_qa(native,model_meta,step=_qa_step,forward=_qa_forward)
            checks[f'wall_switches_{tier}']=bool(switches['ok'])
            metrics[f'wall_switches_{tier}']=switches
    if model_meta.get('lock_stock'):
        from .lock_stock_qa import run_lock_stock_qa
        for tier,native in models.items():
            stock = run_lock_stock_qa(native,model_meta,tier=tier)
            checks[f'lock_stock_{tier}'] = bool(stock['ok'])
            metrics[f'lock_stock_{tier}'] = stock
    from .baby_gate_qa import run_baby_gate_qa
    headroom = run_baby_gate_qa(m, spec)
    checks["baby_gate_headroom"] = bool(headroom["ok"])
    metrics["baby_gate_headroom"] = headroom
    # Full-travel rail span plus actual tread contact where rollers are modeled.
    # The returned rail-only scope explicitly records incomplete suspension geometry.
    from .sliding_track_qa import run_sliding_track_qa
    track_support = run_sliding_track_qa(m, model_meta)
    checks["sliding_track_support"] = bool(track_support["ok"])
    metrics["sliding_track_support"] = track_support
    from .sliding_mechanics_qa import run_sliding_mechanics_qa
    sliding = run_sliding_mechanics_qa(m, model_meta)
    checks["sliding_mechanical_access"] = bool(sliding["ok"])
    metrics["sliding_mechanical_access"] = sliding
    from .gate_hardware_qa import run_gate_hardware_qa
    gate = run_gate_hardware_qa(m, spec, model_meta)
    checks["gate_hardware_operation"] = bool(gate["ok"])
    metrics["gate_hardware_operation"] = gate
    from .rotating_hardware_qa import run_rotating_hardware_qa
    rotating = run_rotating_hardware_qa(m, model_meta)
    checks['rotating_hardware_operation'] = bool(rotating['ok'])
    metrics['rotating_hardware_operation'] = rotating
    if model_meta.get('turnstile_locks'):
        from .turnstile_lock_qa import run_turnstile_lock_qa
        rotor_lock = run_turnstile_lock_qa(m, model_meta)
        checks['turnstile_lock_operation'] = bool(rotor_lock['ok'])
        metrics['turnstile_lock_operation'] = rotor_lock
    if model_meta.get('turnstile_drop_arm'):
        from .turnstile_drop_qa import run_turnstile_drop_qa
        drop = run_turnstile_drop_qa(m, model_meta)
        checks['turnstile_drop_operation'] = bool(drop['ok'])
        metrics['turnstile_drop_operation'] = drop
    if model_meta.get('closer_track_holds'):
        from .closer_track_qa import run_closer_track_qa
        hold = run_closer_track_qa(m, model_meta)
        checks['closer_track_operation'] = bool(hold['ok'])
        metrics['closer_track_operation'] = hold

    if model_meta.get('multipoint_locks'):
        from .multipoint_qa import run_multipoint_qa
        multipoint=run_multipoint_qa(m,model_meta)
        checks['multipoint_operation']=bool(multipoint['ok'])
        metrics['multipoint_operation']=multipoint

    if model_meta.get('elevator_interlocks'):
        from .elevator_qa import run_elevator_qa
        elevator=run_elevator_qa(m,model_meta)
        checks['elevator_operation']=bool(elevator['ok'])
        metrics['elevator_operation']=elevator

    if model_meta.get('dutch_joining_bolt'):
        from .paired_mechanics_qa import run_dutch_join_qa
        joining=run_dutch_join_qa(m,model_meta)
        checks['dutch_joining_operation']=bool(joining['ok'])
        metrics['dutch_joining_operation']=joining

    if model_meta.get('rotary_locksets'):
        from .rotary_lockset_qa import run_rotary_lockset_qa
        rotary=run_rotary_lockset_qa(m,model_meta,cycles=2)
        checks['rotary_lockset_operation']=bool(rotary['ok'])
        metrics['rotary_lockset_operation']=rotary
    if model_meta.get('vault_boltwork'):
        from .vault_hardware_qa import run_vault_mount_qa,run_vault_native_qa
        for tier,native in models.items():
            mounts=run_vault_mount_qa(native,model_meta)
            checks[f'vault_mounts_{tier}']=bool(mounts['ok'])
            metrics[f'vault_mounts_{tier}']=mounts
        vault=run_vault_native_qa(m,model_meta,cycles=2,negative_controls=True)
        checks['vault_service_operation']=bool(vault['ok'])
        metrics['vault_service_operation']=vault

    if model_meta.get('security_guards'):
        from .security_mechanics_qa import run_security_service_qa
        closer=phys.get('closer',{})
        security=run_security_service_qa(m,model_meta,
            opening_preload=closer.get('spring_preload_Nm',0.),
            opening_stiffness=closer.get('spring_stiffness_Nm_per_rad',0.))
        checks['security_guard_service_operation']=bool(security['ok'])
        metrics['security_guard_service_operation']=security

    if model_meta.get('rollup_hoist'):
        from .rollup_hoist_qa import run_rollup_hoist_qa
        hoist=run_rollup_hoist_qa(m,model_meta)
        checks['rollup_hoist_transmission']=bool(hoist['ok'])
        metrics['rollup_hoist_transmission']=hoist

    if model_meta.get('knob_covers'):
        from .knob_cover_qa import run_knob_cover_qa
        covered = run_knob_cover_qa(files['mjcf']['full'],model_meta)
        checks['covered_knob_operation'] = bool(covered['ok'])
        metrics['covered_knob_operation'] = covered
    if model_meta.get('marine_dog_mounts') or model_meta.get('marine_dog_linkage'):
        from .marine_dog_qa import run_marine_dog_qa
        marine=run_marine_dog_qa(m,model_meta)
        checks['marine_dog_operation']=bool(marine['ok'])
        metrics['marine_dog_operation']=marine
    flexible_strip = bool(model_meta.get('strip_curtain'))
    if flexible_strip:
        from .strip_mechanics_qa import run_strip_mechanics_qa
        strip = run_strip_mechanics_qa(m, model_meta)
        checks['strip_mechanics'] = bool(strip['ok'])
        metrics['strip_mechanics'] = strip
        metrics['motion_check_scope'] = 'Repeated 20 N native sheet-face loads with all neighboring contacts, material energy and clamp attachment checks.'
    # Collision clearance alone cannot detect impossible closed-loop mechanisms.
    from .linkage_qa import run_linkage_qa
    linkage = run_linkage_qa(door_dir)
    checks["linkage_feasibility"] = bool(linkage["ok"])
    metrics["linkage_feasibility"] = linkage
    d = mujoco.MjData(m)
    # ---- mass gate: simulated moving mass must match the derived door mass (slab + glass + hardware)
    moving_mass = float(sum(m.body_mass[b] for b in range(1, m.nbody) if m.body_dofnum[b] > 0 or m.body_parentid[b] != 0))
    tgt_mass = float(phys["mass"]["total_kg"])
    metrics["moving_mass_kg"] = moving_mass
    checks["mass"] = bool(abs(moving_mass - tgt_mass) <= max(0.2 * tgt_mass, 0.5))
    pj = _jid(m, model_meta.get("primary_joint") or "")
    oj = _jid(m, model_meta.get("operator_joint") or "") if model_meta.get("operator_joint") else -1
    bj = _jid(m, "leaf_latch_bolt_slide")
    # ---- settle
    _qa_forward(m, d)
    pen0 = _max_pen(m, d)
    for _ in _steps(m, 1.):
        _qa_step(m, d)
    warn = [mujoco.mjtWarning(i).name for i in range(mujoco.mjtWarning.mjNWARNING) if d.warning[i].number > 0]
    drift = abs(_q(m, d, pj)) if pj >= 0 else 0.0
    settle_ok = not warn and pen0[0] > -0.012 and (drift < (0.05 if kin.startswith("hinge") or kin == "rotor" else 0.01) or bool(spec["kinematics"].get("rest_angle_deg")))
    checks["settle"] = bool(settle_ok)
    metrics.update({"initial_penetration_m": pen0[0], "initial_penetration_pair": [pen0[1], pen0[2]], "settle_drift": drift, "warnings": warn})
    # ---- latch / lock behaviour (hinged & sliding single leaf with an operator joint)
    lk = H.LOCKS[spec["lock"]["model"]]
    lt = H.LATCHES[spec["latch"]["model"]]
    flags = door_flags(spec)
    has_holding, env_release_only, free_swing = flags["has_holding"], flags["env_release_only"], flags["free_swing"]
    if model_meta.get('dutch_operation')=='upper_only' and not spec['kinematics']['joining_bolt_engaged']:
        # The independent upper leaf has a ball catch, while the positive
        # spring latch in the specification belongs to the closed lower leaf.
        has_holding=False
    HINGE, SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
    # ---- free-swing / rotary families: nothing holds them, so the push must move them and nothing static may press on
    # them while they move (the "jam" gate; a locked rotor - turnstile awaiting a credential, pet flap with its locking
    # panel in - must instead hold within its locked play)
    if pj >= 0 and free_swing and not flexible_strip and int(m.jnt_type[pj]) in (HINGE, SLIDE):
        is_hinge = int(m.jnt_type[pj]) == HINGE
        push = qa_push(m, d, pj, phys["mass"].get("dynamics_mass_kg", phys["mass"]["total_kg"]), spec["leaf"]["width"], model_meta)["push"]
        metrics["qa_push"] = push
        thr_free = math.radians(10) if is_hinge else 0.05
        lo, hi = (m.jnt_range[pj] if m.jnt_limited[pj] else (-math.inf, math.inf))
        locked = bool(model_meta.get("locked")) or (bool(m.jnt_limited[pj]) and (hi - lo) < thr_free)
        endpoint = None
        if not locked and sliding["ok"] and int(m.jnt_type[pj]) == SLIDE:
            support = next((s for s in model_meta.get("sliding_track_supports", [])
                            if s.get("joint") == model_meta.get("primary_joint") and s.get("suspension_model") and s.get("end_stops")), None)
            if support:
                endpoint = float(support["nominal_range"][1])
        jam = jam_sweep(m, d, pj, push, thr_free, duration_s=MIN_PUSH_S if locked else FREE_PUSH_S, end_position=endpoint)
        metrics["jam_braking_endpoint_m"] = endpoint
        metrics["hold_displacement"] = jam["moved"]
        metrics.update({"jam_t_free": jam["t_free"], "jam_push_s": jam["t_end"], "jam_peak_force_N": jam["peak_force_N"], "jam_peak_pair": jam["peak_pair"]})
        if locked:
            thr_l = max(math.radians(2.0), hi + math.radians(1.0)) if is_hinge else max(0.015, hi + 0.005)
            checks["locked_holds"] = bool(jam["moved"] < thr_l)
        else:
            checks["free_opens"] = bool(jam["moved"] > thr_free)
        checks["no_jam"] = bool(jam["peak_force_N"] < JAM_FORCE_N)
    if pj >= 0 and not free_swing and not flexible_strip and int(m.jnt_type[pj]) in (HINGE, SLIDE):
        is_hinge = int(m.jnt_type[pj]) == HINGE
        mass = phys["mass"].get("dynamics_mass_kg", phys["mass"]["total_kg"])
        W = spec["leaf"]["width"]
        # adaptive push: gravity bias at rest + friction + spring preload, with margin (a strong human / robot)
        dof = m.jnt_dofadr[pj]
        pf = qa_push(m, d, pj, mass, W, model_meta)
        push, bias, fl, preload = pf["push"], pf["bias"], pf["frictionloss"], pf["preload"]
        metrics["qa_push"] = push
        thr = math.radians(2.0) if is_hinge else 0.015
        thr_free = math.radians(10) if is_hinge else 0.05
        if free_swing and has_holding and m.jnt_limited[pj]:
            thr = max(thr, float(m.jnt_range[pj][1]) + math.radians(1.0))   # locked rotor / bolted flap: play inside its locked range
        moved = push_primary(m, d, pj, push, has_holding, thr_free)
        metrics["hold_displacement"] = moved
        if has_holding:
            checks["hold"] = bool(moved < thr)
        else:
            # nothing holds this leaf (no latch / lock; the free-swing families: saloon, revolving, turnstile, bifold,
            # accordion, bypass, pet flap, strip curtain) - the push must actually move it.  A leaf that stays shut is
            # jammed by its own geometry or couplings (a fold whose driven hinges sit on a limit, a wing rubbing the header).
            checks["free_opens"] = bool(moved > thr_free)
    if pj >= 0 and not free_swing and int(m.jnt_type[pj]) in (HINGE, SLIDE):
        # actuate operator (if any) with generous effort; for deadbolt/thumbturn drive it too
        can_release = flags["can_release"]
        if model_meta.get('security_guards'):
            # A security chain/guard deliberately permits partial opening.
            # Its contact-driven release and re-engagement use the dedicated
            # native service sequence above, not a generic2-degree lock test
            # or an arbitrary pose with the chain left in its closed shape.
            metrics['security_actuation_scope']='Ordinary latch hold is tested separately; guard retention, release and reinsertion require the native service cycles. These permit the actual guard slack and do not certify approach access.'
        elif model_meta.get('rotary_locksets'):
            metrics['rotary_actuation_scope']='Separate bounded inside/outside surface-force and catch cycles; credential operation and complete opening are evaluated by native tasks, not by forcing a locked exterior handle.'
        elif env_release_only:
            metrics["note"] = "lock released by environment logic (badge / REX / timer); actuation not tested by QA"
        elif oj >= 0 and can_release:
            mujoco.mj_resetData(m, d)
            ojn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, oj) or ""
            eff = (14.0 if ojn.startswith("dog_") else (10.0 if "wheel" in ojn else (8.0 if "exit_device" in ojn else 4.0))) if int(m.jnt_type[oj]) == HINGE else 120.0
            tt = _jid(m, "leaf_deadbolt_thumbturn_hinge")
            if tt < 0:
                tt = _jid(m, "leaf_a_deadbolt_thumbturn_hinge")
            aux = [_jid(m, n) for n in ("leaf_aux_bolt_slide", "slide_latch_slide", "leaf_slide_bolt_slide", "leaf_pin_slide", "leaf_thumb_hinge", "hatch_bolt_slide", "join_bolt_slide", "garage_slide_lock_slide", "leaf_hook_thumbturn_hinge", "leaf_a_hook_thumbturn_hinge") if _jid(m, n) >= 0]
            dogs = [i for i in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith("dog_") and "hinge" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "")]
            peak_opened = _q(m, d, pj)
            for k in _steps(m, 6.4):
                elapsed = k * m.opt.timestep
                d.qfrc_applied[:] = 0
                if tt >= 0 and elapsed < 1.2:
                    d.qfrc_applied[m.jnt_dofadr[tt]] = 2.0
                for a in aux:
                    d.qfrc_applied[m.jnt_dofadr[a]] = 3.0 if int(m.jnt_type[a]) == HINGE else 60.0
                for dg in dogs:
                    d.qfrc_applied[m.jnt_dofadr[dg]] = 14.0
                if elapsed >= .6:
                    d.qfrc_applied[m.jnt_dofadr[oj]] = eff
                if elapsed >= 1.2 and (not is_hinge or _q(m, d, pj) < math.radians(50)):
                    d.qfrc_applied[m.jnt_dofadr[pj]] = push   # stop pushing past 50 deg (levers would be pressed against the wall)
                _qa_step(m, d)
                peak_opened = max(peak_opened, _q(m, d, pj))
            opened = _q(m, d, pj)
            metrics["actuate_displacement"] = opened
            metrics["actuate_peak_displacement"] = peak_opened
            metrics["operator_travel_reached"] = _q(m, d, oj)
            target = math.radians(min(20.0, 0.5 * (spec["kinematics"].get("max_open_deg") or 90))) if is_hinge else 0.05
            if spec["lock"]["engaged"] and lk.kind in ("chain", "swing_bar_guard") and is_hinge:
                # chain / swing bar: the door opens only to the slack limit
                lim = math.asin(min(0.99, lk.chain_slack / max(W - 0.1, 0.2)))
                checks["actuate_opens"] = bool(math.radians(1.5) < opened < lim + math.radians(4))
                metrics["chain_limit_rad"] = lim
            else:
                # The drive stops above 50 degrees; a functioning closer or
                # counterbalanced linkage may return before the final sample.
                # Opening attainment and subsequent return are separate facts.
                checks["actuate_opens"] = bool(peak_opened > target)
            # release operator: spring latch re-extends
            if bj >= 0:
                q_hold = d.qpos[m.jnt_qposadr[pj]]
                for _ in _steps(m, .8):
                    d.qfrc_applied[:] = 0
                    # A bounded holding effort lets every passive linkage
                    # respond naturally while the operator is released.
                    pdof=int(m.jnt_dofadr[pj])
                    d.qfrc_applied[pdof]=np.clip(1000.*(q_hold-_q(m,d,pj))-80.*d.qvel[pdof],-push,push)
                    _qa_step(m, d)
                metrics["bolt_after_release_m"] = _q(m, d, bj)
                if lt.kind not in ("roller", "ball_catch", "magnetic"):
                    checks["latch_returns"] = bool(_q(m, d, bj) < 0.006 or opened < math.radians(3))
                # relatch: drive closed (only for hinged doors that actually opened)
                if is_hinge and opened > math.radians(5):
                    for _ in _steps(m, 6.):
                        d.qfrc_applied[:] = 0
                        d.qfrc_applied[m.jnt_dofadr[pj]] = -min(0.5 * push, 1.5 * (bias + fl + preload) + 40.0)
                        _qa_step(m, d)
                    closed = _q(m, d, pj)
                    for _ in _steps(m, 1.):
                        d.qfrc_applied[:] = 0
                        d.qfrc_applied[m.jnt_dofadr[pj]] = push
                        _qa_step(m, d)
                    metrics["relatch_closed_angle"] = closed
                    metrics["relatch_repush_angle"] = _q(m, d, pj)
                    if lt.kind not in ("roller", "ball_catch", "magnetic") and lk.kind != "jam_stuck":
                        checks["relatch"] = bool(abs(closed) < math.radians(2.0) and _q(m, d, pj) < math.radians(2.5))
        elif oj >= 0 and spec['lock'].get('engaged') and not can_release and not env_release_only:
            # locked: operator must not free the door
            mujoco.mj_resetData(m, d)
            eff = 6.0 if int(m.jnt_type[oj]) == HINGE else 150.0
            for _ in _steps(m, 2.):
                d.qfrc_applied[:] = 0
                d.qfrc_applied[m.jnt_dofadr[oj]] = eff
                d.qfrc_applied[m.jnt_dofadr[pj]] = push
                _qa_step(m, d)
            metrics["locked_displacement"] = _q(m, d, pj)
            thr_l = thr + (math.asin(min(0.99, lk.chain_slack / max(W - 0.1, 0.2))) if lk.chain_slack else 0.0)
            checks["locked_holds"] = bool(_q(m, d, pj) < thr_l)
        # closer returns from 60 deg (not applicable to gates with a gravity fork latch: the fork is not self-latching
        # and must be lifted to close the gate, so a closer cannot bring the gate home on its own)
        if lt.id == "fork_gravity":
            metrics["closer_note"] = "fork latch: gate closes only with the fork lifted; closer return not applicable"
        elif is_hinge and spec["closer"]["model"] not in ("none", "gas_strut") and phys["closer"].get("spring_preload_Nm", 0) > 0 and not spec["kinematics"].get("both_ways") and not env_release_only and not model_meta.get('security_guards') and not (spec["lock"]["engaged"] and lk.kind in ("chain", "swing_bar_guard", "padlock")):
            service=_prepare_closer_service(m,d,spec,model_meta,door_dir,metrics.get('multipoint_operation'))
            metrics['closer_service_configuration']=service
            if not service['ok']:
                checks['closer_returns']=False
            else:
                qa = m.jnt_qposadr[pj]
                d.qpos[qa] = math.radians(min(60.0, (spec["kinematics"].get("max_open_deg") or 90) * 0.8))
                if bj >= 0:
                    d.qpos[m.jnt_qposadr[bj]] = 0.0
                from .initial_configuration import resolve_joint_followers
                resolve_joint_followers(m,d.qpos,[model_meta['primary_joint']])
                if model_meta.get('closer_mounts'):
                    from .geometry.closer_mounts import resolve_closer_configuration
                    resolve_closer_configuration(m,d.qpos,model_meta)
                _qa_forward(m, d)
                return_duration = 12. + max([0.] + [row.get('delay_time_target_s', 0.)
                                                   for row in model_meta.get('closer_pinion_laws', [])])
                for _ in _steps(m, return_duration):
                    _qa_step(m, d)
                metrics["closer_final_angle"] = _q(m, d, pj)
                # A nearly shut leaf with a bolt still riding the strike is not
                # closed and latched. Preserve degraded self-closing failures.
                latch_state = {m.joint(j).name: _q(m,d,j) for j in range(m.njnt)
                               if m.joint(j).name.endswith(('latch_bolt_slide','top_latch_slide','bottom_latch_slide'))}
                metrics['closer_final_latches_m'] = latch_state
                metrics['closer_test_duration_s'] = return_duration
                checks["closer_returns"] = bool(abs(_q(m, d, pj)) < math.radians(1.0)
                                                 and all(abs(q)<.002 for q in latch_state.values()))
                if not checks['closer_returns']:
                    metrics['closer_return_condition'] = spec.get('condition')
    # ---- simple & minimal tiers settle
    for tier in ("simple", "minimal"):
        if tier in models:
            mm = models[tier]
            dd = mujoco.MjData(mm)
            for _ in _steps(mm, .6):
                _qa_step(mm, dd)
            w = [mujoco.mjtWarning(i).name for i in range(mujoco.mjtWarning.mjNWARNING) if dd.warning[i].number > 0]
            checks[f"settle_{tier}"] = not w
    for tier,mm in models.items():
        warnings=sorted(_QA_FIELDS.get().get(id(mm),{}).get('warnings',()))
        checks[f'native_dynamics_no_warnings_{tier}']=not warnings
        if warnings:metrics[f'native_dynamics_warnings_{tier}']=warnings
    # ---- URDF loads
    if "urdf" in files:
        try:
            mu = mujoco.MjModel.from_xml_path(files["urdf"]["full"])
            checks["urdf_loads"] = True
            metrics["urdf_nbody"] = mu.nbody
        except Exception as e:
            checks["urdf_loads"] = False
            metrics["urdf_error"] = str(e)[:300]
    # ---- USD opens
    if "usd" in files and isinstance(files["usd"], str) and files["usd"].endswith(".usda"):
        try:
            from pxr import Usd, UsdPhysics
            st = Usd.Stage.Open(files["usd"])
            prims = list(st.Traverse())
            nj = sum(1 for p in prims if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint))
            checks["usd_opens"] = nj == m.njnt
            metrics["usd_joints"] = nj
        except Exception as e:
            checks["usd_opens"] = False
            metrics["usd_error"] = str(e)[:300]
    # ---- canonical RL USD opens with the fixed 7-DoF structure (Isaac Lab multi-door spawning)
    if "usd_rl" in files and isinstance(files["usd_rl"], str) and files["usd_rl"].endswith(".usda"):
        try:
            from pxr import Usd, UsdPhysics
            from .export.usd import RL_DOF_JOINTS
            st = Usd.Stage.Open(files["usd_rl"])
            names = {p.GetName() for p in st.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)}
            checks["usd_rl_opens"] = names == set(RL_DOF_JOINTS)
            metrics["usd_rl_joints"] = sorted(names)
        except Exception as e:
            checks["usd_rl_opens"] = False
            metrics["usd_rl_error"] = str(e)[:300]
    signed = all(v for k, v in checks.items())
    return {"checks": checks, "metrics": metrics, "signed_off": bool(signed), "time_s": time.time() - t0, "mujoco_version": mujoco.__version__}


def render_thumbnails(path_xml: str, out_dir: str, cams=("robot_view", "iso", "detail_handle", "far_view"), size=(640, 480), open_fraction=0.0, primary_joint=None) -> list:
    """Offscreen renders for the catalogue.  Returns list of written files."""
    import mujoco
    from PIL import Image
    m = mujoco.MjModel.from_xml_path(path_xml)
    d = mujoco.MjData(m)
    if open_fraction and primary_joint:
        j = _jid(m, primary_joint)
        if j >= 0 and m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
            d.qpos[m.jnt_qposadr[j]] = lo + open_fraction * (hi - lo)
    _qa_forward(m, d)
    r = mujoco.Renderer(m, height=size[1], width=size[0])
    out = []
    for cam in cams:
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
            continue
        r.update_scene(d, camera=cam)
        img = r.render()
        fn = os.path.join(out_dir, f"thumb_{cam}{'_open' if open_fraction else ''}.jpg")
        Image.fromarray(img).save(fn, quality=82)
        out.append(fn)
    r.close()
    return out

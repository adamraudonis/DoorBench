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


def _jid(m, name):
    import mujoco
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)


def _q(m, d, jid):
    return float(d.qpos[m.jnt_qposadr[jid]])


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


def qa_push(m, d, pj, mass_kg: float | None = None, width_m: float | None = None) -> dict:
    """The adaptive QA push on the primary joint (N*m for hinges, N for slides): twice the static resistance at rest
    (gravity bias + Coulomb friction + spring preload) plus a base effort sized by the leaf (``push_base``), capped -
    a strong human / robot.  Mirrored by ``parity.protocol`` (which imports ``push_base`` / ``PUSH_CAP`` from here)."""
    import mujoco
    is_hinge = int(m.jnt_type[pj]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    dof = m.jnt_dofadr[pj]
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    bias = abs(float(d.qfrc_bias[dof] - d.qfrc_passive[dof]))
    fl = float(m.dof_frictionloss[dof])
    preload = abs(float(m.jnt_stiffness[pj] * m.qpos_spring[m.jnt_qposadr[pj]])) if m.jnt_stiffness[pj] > 0 else 0.0
    unit = "hinge" if is_hinge else "slide"
    base = push_base(unit, mass_kg, width_m)
    push = min(2.0 * (bias + fl + preload) + base, PUSH_CAP[unit])
    return {"push": push, "bias": bias, "frictionloss": fl, "preload": preload, "is_hinge": is_hinge, "push_base": base}


def push_primary(m, d, pj, push: float, has_holding: bool, thr_free: float) -> float:
    """The ``hold`` / ``free_opens`` drive from the reset state: push for 1 s (a held leaf) or up to 6 s (a free leaf,
    stopping once it is past ``thr_free``); returns the primary joint value at exit."""
    import mujoco
    mujoco.mj_resetData(m, d)
    for k in range(500 if has_holding else 3000):
        d.qfrc_applied[:] = 0
        d.qfrc_applied[m.jnt_dofadr[pj]] = push
        mujoco.mj_step(m, d)
        if k >= 499 and not has_holding and _q(m, d, pj) > thr_free:
            break   # heavy leaves (big gates, vault doors) need longer to accelerate
    return _q(m, d, pj)


def operator_effort(m, j: int, name: str) -> float:
    """Effort the QA drive puts on one operator joint (N*m on a hinge, N on a slide).  Mirrored by parity.protocol."""
    import mujoco
    if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
        return 120.0
    return 14.0 if name.startswith("dog_") else (10.0 if "wheel" in name else (8.0 if "exit_device" in name else 4.0))


def drive_operators(m, d, pj: int, op_ids: list, aux_ids: list, tt: int, push: float, is_hinge: bool, steps: int = 3200) -> float:
    """Work a SET of operators and push the leaf on the same schedule the ``actuate_opens`` drive uses (thumbturn for
    the first 0.6 s, aux bolts throughout, operators from 0.3 s, push from 0.6 s and stopped past 50 deg).  Returns the
    primary joint at the end.  Driving a subset is how ``all_latches_release`` asks "does one dog still hold it?"."""
    import mujoco
    HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
    names = {j: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "") for j in op_ids}
    mujoco.mj_resetData(m, d)
    for k in range(steps):
        d.qfrc_applied[:] = 0
        if tt >= 0 and k < 600:
            d.qfrc_applied[m.jnt_dofadr[tt]] = 2.0
        for a in aux_ids:
            d.qfrc_applied[m.jnt_dofadr[a]] = 3.0 if int(m.jnt_type[a]) == HINGE else 60.0
        if k >= 300:
            for j in op_ids:
                d.qfrc_applied[m.jnt_dofadr[j]] = operator_effort(m, j, names[j])
        if k >= 600 and (not is_hinge or _q(m, d, pj) < math.radians(50)):
            d.qfrc_applied[m.jnt_dofadr[pj]] = push
        mujoco.mj_step(m, d)
    return _q(m, d, pj)


def hold_with_one_point(m, d, pj: int, keep: int, release: list, push: float, steps: int = 1200) -> float:
    """Push the leaf with exactly ONE lock point still engaged: every joint in ``release`` is driven to its retracted
    end and the joint equalities that drive them are switched off, so the point under test is the only thing holding
    the leaf.  Returns the primary joint at the end."""
    import mujoco
    SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
    mujoco.mj_resetData(m, d)
    for e in range(m.neq):
        if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT) and int(m.eq_obj1id[e]) in release:
            d.eq_active[e] = 0
    for _ in range(steps):
        d.qfrc_applied[:] = 0
        for j in release:
            d.qfrc_applied[m.jnt_dofadr[j]] = 200.0 if int(m.jnt_type[j]) == SLIDE else 30.0
        d.qfrc_applied[m.jnt_dofadr[pj]] = push
        mujoco.mj_step(m, d)
    return _q(m, d, pj)


SPRING_LATCH_KINDS = ("tubular_latch", "deadlatch", "mortise_latch", "rim_latch", "vertical_rods", "hook", "gravity_bar", "dogs", "multi_bolt", "electric_bolt")
ENV_RELEASE_LOCK_KINDS = ("mag_lock", "delayed_egress", "card_reader", "electric_strike", "interlock")
FREE_SWING_FAMILIES = ("saloon", "strip_curtain", "pet_door", "turnstile_tripod", "turnstile_fullheight", "revolving", "bifold", "accordion", "sliding_bypass")
JAM_FORCE_N = 20.0       # N; largest contact normal force static geometry may exert on a moving part while a free door is pushed
#                          (all 147 free-swing doors read exactly 0 N after the 2026-09 fixes: a free leaf is carried by its joint;
#                          20 N already means a leaf resting on the floor or scraping a jamb without stalling - a visible defect)
FREE_PUSH_S = 6.0        # s; a free door is pushed for up to this long (stops once past thr_free after MIN_PUSH_S)
MIN_PUSH_S = 1.0


def jam_sweep(m, d, pj: int, push: float, thr_free: float, duration_s: float = FREE_PUSH_S, min_push_s: float = MIN_PUSH_S) -> dict:
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
        d.qfrc_applied[dof] = push
        mujoco.mj_step(m, d)
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
    can_release = bool(spec["lock"].get("robot_side_release", True)) or lk.kind == "jam_stuck"
    return {"spring_latch": spring_latch, "lock_engaged": lock_engaged, "has_holding": spring_latch or lock_engaged or H.OPERATORS[spec["operator"]["model"]].kind == "cremone", "env_release_only": env_release_only,   # cremone shoot bolts are the door's latch
            "can_release": can_release, "free_swing": spec["family"] in FREE_SWING_FAMILIES, "lock_kind": lk.kind, "latch_kind": lt.kind}


def run_qa(spec: dict, door_dir: str, model_meta: dict, files: dict, phys: dict) -> dict:
    import mujoco
    t0 = time.time()
    checks = {}
    metrics = {}
    fam = spec["family"]
    kin = spec["kinematics"]["type"]
    # ---- load all tiers
    models = {}
    for tier, path in files.get("mjcf", {}).items():
        try:
            models[tier] = mujoco.MjModel.from_xml_path(path)
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
    # Full-travel rail span plus actual tread contact where rollers are modeled.
    # The returned rail-only scope explicitly records incomplete suspension geometry.
    from .sliding_track_qa import run_sliding_track_qa
    track_support = run_sliding_track_qa(m, model_meta)
    checks["sliding_track_support"] = bool(track_support["ok"])
    metrics["sliding_track_support"] = track_support
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
    mujoco.mj_forward(m, d)
    pen0 = _max_pen(m, d)
    for _ in range(500):
        mujoco.mj_step(m, d)
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
    HINGE, SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
    # ---- free-swing / rotary families: nothing holds them, so the push must move them and nothing static may press on
    # them while they move (the "jam" gate; a locked rotor - turnstile awaiting a credential, pet flap with its locking
    # panel in - must instead hold within its locked play)
    if pj >= 0 and free_swing and int(m.jnt_type[pj]) in (HINGE, SLIDE):
        is_hinge = int(m.jnt_type[pj]) == HINGE
        push = qa_push(m, d, pj, phys["mass"]["total_kg"], spec["leaf"]["width"])["push"]
        metrics["qa_push"] = push
        thr_free = math.radians(10) if is_hinge else 0.05
        lo, hi = (m.jnt_range[pj] if m.jnt_limited[pj] else (-math.inf, math.inf))
        locked = bool(model_meta.get("locked")) or (bool(m.jnt_limited[pj]) and (hi - lo) < thr_free)
        jam = jam_sweep(m, d, pj, push, thr_free, duration_s=MIN_PUSH_S if locked else FREE_PUSH_S)
        metrics["hold_displacement"] = jam["moved"]
        metrics.update({"jam_t_free": jam["t_free"], "jam_push_s": jam["t_end"], "jam_peak_force_N": jam["peak_force_N"], "jam_peak_pair": jam["peak_pair"]})
        if locked:
            thr_l = max(math.radians(2.0), hi + math.radians(1.0)) if is_hinge else max(0.015, hi + 0.005)
            checks["locked_holds"] = bool(jam["moved"] < thr_l)
        else:
            checks["free_opens"] = bool(jam["moved"] > thr_free)
        checks["no_jam"] = bool(jam["peak_force_N"] < JAM_FORCE_N)
    if pj >= 0 and not free_swing and int(m.jnt_type[pj]) in (HINGE, SLIDE):
        is_hinge = int(m.jnt_type[pj]) == HINGE
        mass = phys["mass"]["total_kg"]
        W = spec["leaf"]["width"]
        # adaptive push: gravity bias at rest + friction + spring preload, with margin (a strong human / robot)
        dof = m.jnt_dofadr[pj]
        pf = qa_push(m, d, pj, mass, W)
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
        if env_release_only:
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
            for k in range(3200):
                d.qfrc_applied[:] = 0
                if tt >= 0 and k < 600:
                    d.qfrc_applied[m.jnt_dofadr[tt]] = 2.0
                for a in aux:
                    d.qfrc_applied[m.jnt_dofadr[a]] = 3.0 if int(m.jnt_type[a]) == HINGE else 60.0
                for dg in dogs:
                    d.qfrc_applied[m.jnt_dofadr[dg]] = 14.0
                if k >= 300:
                    d.qfrc_applied[m.jnt_dofadr[oj]] = eff
                if k >= 600 and (not is_hinge or _q(m, d, pj) < math.radians(50)):
                    d.qfrc_applied[m.jnt_dofadr[pj]] = push   # stop pushing past 50 deg (levers would be pressed against the wall)
                mujoco.mj_step(m, d)
            opened = _q(m, d, pj)
            metrics["actuate_displacement"] = opened
            metrics["operator_travel_reached"] = _q(m, d, oj)
            target = math.radians(min(20.0, 0.5 * (spec["kinematics"].get("max_open_deg") or 90))) if is_hinge else 0.05
            if spec["lock"]["engaged"] and lk.kind in ("chain", "swing_bar_guard") and is_hinge:
                # chain / swing bar: the door opens only to the slack limit
                lim = math.asin(min(0.99, lk.chain_slack / max(W - 0.1, 0.2)))
                checks["actuate_opens"] = bool(math.radians(1.5) < opened < lim + math.radians(4))
                metrics["chain_limit_rad"] = lim
            else:
                checks["actuate_opens"] = bool(opened > target)
            # release operator: spring latch re-extends
            if bj >= 0:
                q_hold = d.qpos[m.jnt_qposadr[pj]]
                for _ in range(400):
                    d.qfrc_applied[:] = 0
                    d.qpos[m.jnt_qposadr[pj]] = q_hold
                    d.qvel[m.jnt_dofadr[pj]] = 0.0
                    mujoco.mj_step(m, d)
                metrics["bolt_after_release_m"] = _q(m, d, bj)
                if lt.kind not in ("roller", "ball_catch", "magnetic"):
                    checks["latch_returns"] = bool(_q(m, d, bj) < 0.006 or opened < math.radians(3))
                # relatch: drive closed (only for hinged doors that actually opened)
                if is_hinge and opened > math.radians(5):
                    for _ in range(3000):
                        d.qfrc_applied[:] = 0
                        d.qfrc_applied[m.jnt_dofadr[pj]] = -min(0.5 * push, 1.5 * (bias + fl + preload) + 40.0)
                        mujoco.mj_step(m, d)
                    closed = _q(m, d, pj)
                    for _ in range(500):
                        d.qfrc_applied[:] = 0
                        d.qfrc_applied[m.jnt_dofadr[pj]] = push
                        mujoco.mj_step(m, d)
                    metrics["relatch_closed_angle"] = closed
                    metrics["relatch_repush_angle"] = _q(m, d, pj)
                    if lt.kind not in ("roller", "ball_catch", "magnetic") and lk.kind != "jam_stuck":
                        checks["relatch"] = bool(abs(closed) < math.radians(2.0) and _q(m, d, pj) < math.radians(2.5))
        elif oj >= 0 and not can_release and not env_release_only:
            # locked: operator must not free the door
            mujoco.mj_resetData(m, d)
            eff = 6.0 if int(m.jnt_type[oj]) == HINGE else 150.0
            for _ in range(1000):
                d.qfrc_applied[:] = 0
                d.qfrc_applied[m.jnt_dofadr[oj]] = eff
                d.qfrc_applied[m.jnt_dofadr[pj]] = push
                mujoco.mj_step(m, d)
            metrics["locked_displacement"] = _q(m, d, pj)
            thr_l = thr + (math.asin(min(0.99, lk.chain_slack / max(W - 0.1, 0.2))) if lk.chain_slack else 0.0)
            checks["locked_holds"] = bool(_q(m, d, pj) < thr_l)
        # closer returns from 60 deg (not applicable to gates with a gravity fork latch: the fork is not self-latching
        # and must be lifted to close the gate, so a closer cannot bring the gate home on its own)
        if lt.id == "fork_gravity":
            metrics["closer_note"] = "fork latch: gate closes only with the fork lifted; closer return not applicable"
        elif is_hinge and spec["closer"]["model"] not in ("none", "gas_strut") and phys["closer"].get("spring_preload_Nm", 0) > 0 and not spec["kinematics"].get("both_ways") and not env_release_only and not (spec["lock"]["engaged"] and lk.kind in ("chain", "swing_bar_guard", "padlock")):
            mujoco.mj_resetData(m, d)
            qa = m.jnt_qposadr[pj]
            d.qpos[qa] = math.radians(min(60.0, (spec["kinematics"].get("max_open_deg") or 90) * 0.8))
            if bj >= 0:
                d.qpos[m.jnt_qposadr[bj]] = 0.0
            mujoco.mj_forward(m, d)
            for _ in range(int(12.0 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            metrics["closer_final_angle"] = _q(m, d, pj)
            checks["closer_returns"] = bool(_q(m, d, pj) < math.radians(6.0))
    # ---- all_latches_release: a door held by SEVERAL independent latches (watertight dog levers, blast-door lever
    # bolts) must behave like the real hardware: each latch holds the leaf on its own, so releasing all but one must
    # NOT free it, and only releasing every one of them opens it.  Without this gate a "multi-dog" door is decoration:
    # one dog turns, the leaf swings, and the other five never move.  Doors whose lock points are driven together from
    # one operator (handwheel -> dogs / boltwork, cremone knob -> shoot bolts) cannot be partially released, and are
    # covered by "hold" + "actuate_opens" instead; the coupling itself is recorded in the metrics.
    op_names = [n for n in (model_meta.get("operator_joints") or []) if _jid(m, n) >= 0]
    metrics["operator_joints"] = op_names
    metrics["operator_coupling"] = model_meta.get("operator_coupling", "coupled")
    if (pj >= 0 and not free_swing and int(m.jnt_type[pj]) in (HINGE, SLIDE) and len(op_names) > 1
            and model_meta.get("operator_coupling") == "individual" and flags["can_release"] and not env_release_only):
        is_hinge = int(m.jnt_type[pj]) == HINGE
        push = float(metrics.get("qa_push") or qa_push(m, d, pj, phys["mass"]["total_kg"], spec["leaf"]["width"])["push"])
        tt = _jid(m, "leaf_deadbolt_thumbturn_hinge")
        if tt < 0:
            tt = _jid(m, "leaf_a_deadbolt_thumbturn_hinge")
        aux = [_jid(m, n) for n in ("leaf_aux_bolt_slide", "slide_latch_slide", "leaf_slide_bolt_slide", "leaf_pin_slide", "leaf_thumb_hinge", "hatch_bolt_slide", "join_bolt_slide", "garage_slide_lock_slide", "leaf_hook_thumbturn_hinge", "leaf_a_hook_thumbturn_hinge") if _jid(m, n) >= 0]
        ids = [_jid(m, n) for n in op_names]
        target = math.radians(min(20.0, 0.5 * (spec["kinematics"].get("max_open_deg") or 90))) if is_hinge else 0.05
        # same "still shut" threshold the `hold` gate uses; the worst single latch left engaged today is a hinge-stile
        # watertight dog at 0.95 deg (it sits 34 mm from the hinge pin, so it takes the most leaf rotation to bite).
        thr_hold = min(math.radians(2.0) if is_hinge else 0.015, 0.5 * target)
        partial = []
        for keep in range(len(ids)):        # every latch on its own must still hold the leaf
            partial.append(drive_operators(m, d, pj, [j for i, j in enumerate(ids) if i != keep], aux, tt, push, is_hinge))
        full = drive_operators(m, d, pj, ids, aux, tt, push, is_hinge)
        metrics["all_latches_partial_displacement"] = partial
        metrics["all_latches_worst_partial"] = max(partial) if partial else 0.0
        metrics["all_latches_full_displacement"] = full
        metrics["all_latches_thresholds"] = [thr_hold, target]
        checks["all_latches_release"] = bool(all(p < thr_hold for p in partial) and full > target)
    # ---- rod_points_hold: a two-point rod mechanism throws a bolt into the head AND one into the floor - a cremone /
    # espagnolette knob drives both its rods, a surface vertical rod exit device both of its latches.  Each of the two
    # points has to hold the leaf ON ITS OWN, or the second rod is a drawn cylinder that latches nothing: which is
    # exactly what both mechanisms were (the down rod and the bottom rod were visuals with no bolt behind them).
    if pj >= 0 and int(m.jnt_type[pj]) == HINGE:
        lname = (model_meta.get("primary_joint") or "").rsplit("_hinge", 1)[0]
        push_r = float(metrics.get("qa_push") or 0.0)
        for tag, a, b in (("vertical_rods", f"{lname}_top_latch_slide", f"{lname}_bottom_latch_slide"),
                          ("cremone", f"{lname}_cremone_top_bolt_slide", f"{lname}_cremone_bottom_bolt_slide")):
            ja, jb = _jid(m, a), _jid(m, b)
            if ja < 0 or jb < 0 or push_r <= 0:
                continue
            top_only = hold_with_one_point(m, d, pj, ja, [jb], push_r)
            bot_only = hold_with_one_point(m, d, pj, jb, [ja], push_r)
            metrics["rod_points"] = {"mechanism": tag, "top_only_rad": top_only, "bottom_only_rad": bot_only, "joints": [a, b]}
            checks["rod_points_hold"] = bool(max(top_only, bot_only) < math.radians(2.0))
    # ---- simple & minimal tiers settle
    for tier in ("simple", "minimal"):
        if tier in models:
            mm = models[tier]
            dd = mujoco.MjData(mm)
            for _ in range(300):
                mujoco.mj_step(mm, dd)
            w = [mujoco.mjtWarning(i).name for i in range(mujoco.mjtWarning.mjNWARNING) if dd.warning[i].number > 0]
            checks[f"settle_{tier}"] = not w
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
    mujoco.mj_forward(m, d)
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

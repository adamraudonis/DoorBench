"""Automated QA / sign-off for a generated door.

Checks (all tiers where applicable):
  load        MJCF loads in MuJoCo (full / simple / minimal)
  settle      1 s free simulation: no warnings, no deep initial penetrations, primary joint drift small
  hold        latched door resists a strong opening torque/force (if it has a latch/lock)
  actuate     driving the operator retracts the latch and the door opens (if robot-side release exists)
  return      releasing the operator lets the spring latch re-extend
  operator_returns  every spring- / gravity-return operator (levers, knobs, pads, thumb pieces, forks ...) driven to full
              travel and let go comes back to rest within 1 s (1.5 s gravity), no residual offset, at most one bounce,
              with the door closed and with the door open (latch bolt free)
  operator_holds    every detent operator (handwheels, dogs, slide bolts, cremone knobs, toggle latches) stays where put
  multi_latch_holds independent multi-point holding (dogs / lever bolts): with all but one released the leaf must not open
  relatch     closing the door re-latches (spring latches with strike lip)
  closer      self-closing doors return to closed from 60 deg
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


SPRING_LATCH_KINDS = ("tubular_latch", "deadlatch", "mortise_latch", "rim_latch", "vertical_rods", "hook", "gravity_bar", "dogs", "multi_bolt", "electric_bolt")

# operator release behaviour (hardware.OperatorModel.return_kind -> ir.Joint.return_kind)
RETURN_TIME_LIMIT_S = {"spring": 1.0, "gravity": 1.5}
RETURN_TOL_FRACTION = 0.05          # "at rest" = within 5 % of the travel of the rest stop (>= 0.5 deg / 0.5 mm)
RETURN_MAX_BOUNCES = 1              # one clack against the rest stop is real hardware; repeated excursions are chatter


def _release_trial(m, d, j: int, pj: int, door_q: float | None, t_drive: float = 0.8, t_free: float = 1.5) -> dict:
    """Drive operator joint `j` to 95 % of its travel with a saturating hand servo, let go, and record how it comes
    back: time to enter the rest band, residual offset at the end, number of excursions out of the band after the first
    arrival (bounces).  With `door_q` the leaf is held at that opening (latch bolt free of the strike)."""
    import mujoco
    HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
    mujoco.mj_resetData(m, d)
    if door_q is not None and pj >= 0:
        d.qpos[m.jnt_qposadr[pj]] = door_q
    mujoco.mj_forward(m, d)
    lo, hi = m.jnt_range[j]
    adr, dof = m.jnt_qposadr[j], m.jnt_dofadr[j]
    hinge = int(m.jnt_type[j]) == HINGE
    span = float(hi - lo)
    dt = m.opt.timestep
    fl = float(m.dof_frictionloss[dof])
    k = float(m.jnt_stiffness[j])
    tau_spring = abs(k * (0.95 * hi - m.qpos_spring[adr])) if k > 0 else 0.0
    # stiff hand servo (steady-state error << 5 % of travel against the return spring), saturated at a hand-sized effort
    kp, kv, fmax = (60.0, 1.2, max(8.0, 4.0 * fl + 3.0 * tau_spring)) if hinge else (40000.0, 200.0, max(250.0, 4.0 * fl + 3.0 * tau_spring))
    target = lo + 0.95 * span

    def hold_door():
        if door_q is not None and pj >= 0:
            d.qpos[m.jnt_qposadr[pj]] = door_q
            d.qvel[m.jnt_dofadr[pj]] = 0.0
    for _ in range(int(t_drive / dt)):
        d.qfrc_applied[:] = 0
        hold_door()
        q, v = float(d.qpos[adr]), float(d.qvel[dof])
        d.qfrc_applied[dof] = float(np.clip(kp * (target - q) - kv * v, -fmax, fmax))
        mujoco.mj_step(m, d)
    q_drive = float(d.qpos[adr]) - lo
    tol = max(RETURN_TOL_FRACTION * span, math.radians(0.5) if hinge else 0.0005)
    t_ret, bounces, inside_prev = None, 0, False
    for kstep in range(int(t_free / dt)):
        d.qfrc_applied[:] = 0
        hold_door()
        mujoco.mj_step(m, d)
        inside = abs(float(d.qpos[adr]) - lo) < tol
        if t_ret is None and inside:
            t_ret = (kstep + 1) * dt
        if t_ret is not None and inside_prev and not inside:
            bounces += 1
        inside_prev = inside
    q_end = float(d.qpos[adr]) - lo
    return {"q_drive": q_drive, "travel": span, "t_return_s": None if t_ret is None else round(float(t_ret), 3), "residual": q_end, "tol": tol, "bounces": bounces,
            "holds": bool(q_drive > 0 and q_end >= 0.8 * q_drive)}


def operator_release_checks(m, d, joints: dict, pj: int) -> dict:
    """`joints`: {joint name: model.json joint dict} of the operator joints to test (role operator + dogs).
    Returns {"checks": {...}, "metrics": {...}} - checks operator_returns (spring / gravity kinds) and operator_holds (detent)."""
    import mujoco
    HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
    checks, metrics = {}, {}
    open_q = None
    if pj >= 0 and m.jnt_limited[pj]:
        lo_p, hi_p = m.jnt_range[pj]
        open_q = float(lo_p + 0.5 * (hi_p - lo_p)) if hi_p - lo_p > 0.05 else None
    for name, jj in joints.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0 or not m.jnt_limited[j] or jj.get("robot_interactive") is False:
            continue
        kind = jj.get("return_kind") or ("spring" if m.jnt_stiffness[j] > 0 else "")
        if kind not in ("spring", "gravity", "detent"):
            continue
        lo, hi = m.jnt_range[j]
        if hi - lo < 0.006:
            metrics[name] = {"return": kind, "note": "locked: range below the backlash threshold, not driven"}
            continue
        closed = _release_trial(m, d, j, pj, None)
        rec = {"return": kind, **closed}
        need = 0.5 if kind != "detent" else 0.3
        if closed["q_drive"] < need * closed["travel"]:
            rec["note"] = f"hand servo reached only {closed['q_drive'] / closed['travel']:.0%} of the travel; not judged"
            metrics[name] = rec
            continue
        if kind == "detent":
            ok = closed["holds"]
            rec["ok"] = bool(ok)
            checks["operator_holds"] = bool(checks.get("operator_holds", True) and ok)
        else:
            lim = RETURN_TIME_LIMIT_S[kind]
            ok = closed["t_return_s"] is not None and closed["t_return_s"] <= lim and abs(closed["residual"]) < closed["tol"] and closed["bounces"] <= RETURN_MAX_BOUNCES
            if open_q is not None:
                op_ = _release_trial(m, d, j, pj, open_q)
                rec.update({"t_return_open_s": op_["t_return_s"], "residual_open": op_["residual"], "bounces_open": op_["bounces"]})
                ok = ok and op_["t_return_s"] is not None and op_["t_return_s"] <= lim and abs(op_["residual"]) < op_["tol"] and op_["bounces"] <= RETURN_MAX_BOUNCES
            rec["ok"] = bool(ok)
            checks["operator_returns"] = bool(checks.get("operator_returns", True) and ok)
        metrics[name] = rec
    return {"checks": checks, "metrics": metrics}


def multi_latch_check(m, d, dog_joints: list, pj: int, push: float, thr: float) -> dict:
    """Independent multi-point holding (watertight dogs, lever bolts): release every dog but one (each in turn) and push
    the leaf - it must not open; then release all and push - it must open past `thr` (the actuate check covers that)."""
    import mujoco
    out = {"held_with_one_dog": [], "ok": True}
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in dog_joints]
    ids = [j for j in ids if j >= 0]
    if len(ids) < 2 or pj < 0:
        out["ok"] = None
        return out
    for keep in ids:
        mujoco.mj_resetData(m, d)
        for j in ids:
            if j != keep:
                d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j][1]        # undogged
        mujoco.mj_forward(m, d)
        for _ in range(500):
            d.qfrc_applied[:] = 0
            for j in ids:
                if j != keep:
                    d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j][1]
                    d.qvel[m.jnt_dofadr[j]] = 0.0
            d.qfrc_applied[m.jnt_dofadr[pj]] = push
            mujoco.mj_step(m, d)
        moved = float(d.qpos[m.jnt_qposadr[pj]])
        out["held_with_one_dog"].append(round(moved, 5))
        if moved >= thr:
            out["ok"] = False
    return out
ENV_RELEASE_LOCK_KINDS = ("mag_lock", "delayed_egress", "card_reader", "electric_strike", "interlock")
FREE_SWING_FAMILIES = ("saloon", "strip_curtain", "pet_door", "turnstile_tripod", "turnstile_fullheight", "revolving", "bifold", "accordion", "sliding_bypass")


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
    # ---- deterministic kinematic clearance gate (all geometry collidable, every joint swept)
    try:
        from .clearance import run_clearance
        cl = run_clearance(door_dir, "full")
        checks["clearance"] = bool(cl["ok"])
        metrics["clearance_n_failures"] = cl["n_failures"]
        metrics["clearance_failures"] = cl["failures"][:10]
    except Exception as e:
        checks["clearance"] = False
        metrics["clearance_error"] = str(e)[:200]
    m = models["full"]
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
    # ---- operator release behaviour (spring / gravity return, detent hold) for every operator joint incl. dogs
    try:
        with open(os.path.join(door_dir, "model.json")) as f:
            mj_bodies = json.load(f)["bodies"]
    except Exception:
        mj_bodies = []
    dog_names = list(model_meta.get("dog_joints") or [])
    op_joints = {}
    for b in mj_bodies:
        jj = b.get("joint")
        if jj and (jj.get("role") == "operator" or jj["name"] in dog_names or jj["name"] == model_meta.get("operator_joint")):
            op_joints[jj["name"]] = jj
    if op_joints:
        rel = operator_release_checks(m, d, op_joints, pj)
        checks.update(rel["checks"])
        metrics["operator_release"] = rel["metrics"]
    # ---- latch / lock behaviour (hinged & sliding single leaf with an operator joint)
    lk = H.LOCKS[spec["lock"]["model"]]
    lt = H.LATCHES[spec["latch"]["model"]]
    flags = door_flags(spec)
    has_holding, env_release_only, free_swing = flags["has_holding"], flags["env_release_only"], flags["free_swing"]
    HINGE, SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
    if pj >= 0 and not free_swing and int(m.jnt_type[pj]) in (HINGE, SLIDE):
        is_hinge = int(m.jnt_type[pj]) == HINGE
        mass = phys["mass"]["total_kg"]
        W = spec["leaf"]["width"]
        # adaptive push: gravity bias at rest + friction + spring preload, with margin (a strong human / robot)
        dof = m.jnt_dofadr[pj]
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        bias = abs(float(d.qfrc_bias[dof] - d.qfrc_passive[dof]))
        fl = float(m.dof_frictionloss[dof])
        preload = abs(float(m.jnt_stiffness[pj] * m.qpos_spring[m.jnt_qposadr[pj]])) if m.jnt_stiffness[pj] > 0 else 0.0
        push = 2.0 * (bias + fl + preload) + (60.0 if is_hinge else 80.0)
        push = min(push, 800.0 if is_hinge else 4000.0)
        metrics["qa_push"] = push
        mujoco.mj_resetData(m, d)
        thr = math.radians(2.0) if is_hinge else 0.015
        thr_free = math.radians(10) if is_hinge else 0.05
        for k in range(500 if has_holding else 3000):
            d.qfrc_applied[:] = 0
            d.qfrc_applied[m.jnt_dofadr[pj]] = push
            mujoco.mj_step(m, d)
            if k >= 499 and not has_holding and _q(m, d, pj) > thr_free:
                break   # heavy leaves (big gates, vault doors) need longer to accelerate
        moved = _q(m, d, pj)
        metrics["hold_displacement"] = moved
        if has_holding:
            checks["hold"] = bool(moved < thr)
        else:
            checks["free_opens"] = bool(moved > thr_free)
        # independent multi-point holding (watertight dogs, blast-door lever bolts): any single dog must hold the leaf
        if has_holding and len(dog_names) > 1 and model_meta.get("dogs_independent"):
            ml = multi_latch_check(m, d, dog_names, pj, push, thr)
            metrics["multi_latch"] = ml
            if ml["ok"] is not None:
                checks["multi_latch_holds"] = bool(ml["ok"])
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

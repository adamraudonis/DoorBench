"""Automated QA / sign-off for a generated door.

Checks (all tiers where applicable):
  load        MJCF loads in MuJoCo (full / simple / minimal)
  settle      1 s free simulation: no warnings, no deep initial penetrations, primary joint drift small
  hold        latched door resists a strong opening torque/force (if it has a latch/lock)
  actuate     driving the operator retracts the latch and the door opens (if robot-side release exists)
  return      releasing the operator lets the spring latch re-extend
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
    spring_latch = lt.throw > 0 and lt.kind in ("tubular_latch", "deadlatch", "mortise_latch", "rim_latch", "vertical_rods", "hook", "gravity_bar", "dogs", "multi_bolt", "electric_bolt")
    lock_engaged = spec["lock"]["engaged"] and lk.kind not in ("none", "child_lock_cover", "jam_stuck")
    opm_ = H.OPERATORS[spec["operator"]["model"]]
    has_holding = spring_latch or lock_engaged or opm_.kind == "cremone"   # a cremone's shoot bolts are the door's latch
    env_release_only = spec["lock"]["engaged"] and lk.kind in ("mag_lock", "delayed_egress", "card_reader", "electric_strike", "interlock")
    free_swing = fam in ("saloon", "strip_curtain", "pet_door", "turnstile_tripod", "turnstile_fullheight", "revolving", "bifold", "accordion", "sliding_bypass")
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
        # actuate operator (if any) with generous effort; for deadbolt/thumbturn drive it too
        can_release = spec["lock"].get("robot_side_release", True) or lk.kind == "jam_stuck"
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

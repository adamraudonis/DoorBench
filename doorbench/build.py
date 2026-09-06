"""Build pipeline: spec -> physics -> IR model -> exports."""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from .ir import Model, quat_to_mat
from . import physics as P
from . import hardware as H
from .geometry import hinged as GH
from .geometry import other as GO
from .geometry import meshes as MESH

G = 9.81


def _world_rotation(model: Model, body):
    """Orientation of a body's frame in world axes at the modelled (as-built, joints at their reference) pose."""
    R = np.eye(3)
    chain, b, seen = [], body, 0
    while b is not None and seen < 16:
        chain.append(b)
        b = model.body(b.parent) if b.parent else None
        seen += 1
    for b in reversed(chain):
        R = R @ quat_to_mat(b.quat)
    return R


def _subtree_about_axis(model: Model, body, axis, anchor):
    """Rotational (or translational) inertia of a joint's whole subtree about its axis, and the subtree's mass and
    centre of mass relative to the joint anchor, in the joint's own (body-local) frame."""
    a = np.asarray(axis, dtype=float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    I_axis, m_tot, first_moment = 0.0, 0.0, np.zeros(3)
    stack = [(body, -np.asarray(anchor, dtype=float), np.eye(3))]
    while stack:
        b, off, R = stack.pop()
        m_, com, Ic = b.inertial("full")
        if m_ > 0:
            c = off + R @ np.asarray(com, dtype=float)
            Iw = R @ np.asarray(Ic, dtype=float) @ R.T
            I_axis += float(a @ (Iw + m_ * (float(np.dot(c, c)) * np.eye(3) - np.outer(c, c))) @ a)
            m_tot += m_
            first_moment += m_ * c
        for ch in model.bodies:
            if ch.parent == b.name:
                stack.append((ch, off + R @ np.asarray(ch.pos, dtype=float), R @ quat_to_mat(ch.quat)))
    com_rel = first_moment / m_tot if m_tot > 1e-9 else np.zeros(3)
    return I_axis, m_tot, com_rel, a


def _gravity_moment_fn(axis_unit, mass, com_rel, hinge: bool, g_dir):
    """Gravity moment about the joint axis as a function of the joint value (positive = toward actuation).

    Rotating the whole subtree about the axis by q moves its centre of mass with it, so the moment is exact for a
    rigid operator; a slide joint just translates its mass, and gravity along it is constant.  ``g_dir`` is the
    (unit) direction of gravity expressed in the joint's own frame."""
    w = np.asarray(axis_unit, dtype=float)
    d0 = np.asarray(com_rel, dtype=float)
    F = np.asarray(g_dir, dtype=float) * (mass * G)
    if not hinge:
        return lambda q: float(np.dot(w, F))
    d_par = float(np.dot(d0, w)) * w
    d_perp = d0 - d_par

    def tau(q):
        # Rodrigues rotation of the perpendicular part about the axis
        d = d_par + d_perp * math.cos(q) + np.cross(w, d_perp) * math.sin(q)
        return float(np.dot(w, np.cross(d, F)))
    return tau


def _over_centre(tau_g_fn, lo: float, hi: float, n: int = 400):
    """The angle at which an unsprung part's own weight moment stops restoring and starts carrying it further open
    (the over-centre point), or None if it never does inside the range."""
    step = (hi - lo) / n
    prev = tau_g_fn(lo)
    if prev > 0:
        return None                  # already over centre at rest (a ring hanging off a ceiling hatch)
    for i in range(1, n + 1):
        q = lo + i * step
        cur = tau_g_fn(q)
        if cur > 0:
            return q - step * 0.5
        prev = cur
    return None


def _gravity_rest(tau_g_fn, lo: float, hi: float, q0: float, n: int = 400) -> float:
    """Where a part with no spring settles after being released at ``q0``: the first stable equilibrium of its own
    weight moment in the direction it starts moving, clamped to the joint range.

    A fork or a teardrop falls back to its drawn rest (the weight moment is restoring everywhere).  A plain ring on
    the UNDERSIDE of a ceiling hatch does not: gravity holds it hanging down, and that - not the recess it was drawn
    in - is where it comes to rest."""
    step = (hi - lo) / n
    down = tau_g_fn(q0) <= 0.0
    q = q0
    for _ in range(n + 1):
        nxt = q + (-step if down else step)
        if nxt < lo:
            return lo
        if nxt > hi:
            return hi
        if (tau_g_fn(nxt) > 0) != (tau_g_fn(q) > 0):
            return 0.5 * (q + nxt)     # sign change: the weight moment vanishes here and reverses beyond it
        q = nxt
    return lo if down else hi


def tune_operator_returns(model: Model, spec: dict, phys: dict) -> dict:
    """Give every operator joint the release behaviour its hardware actually has, and record it.

    The catalogue fixes WHAT the operator does when the hand lets go (hardware.OperatorModel.return_kind); the
    numbers that make it happen depend on the geometry that was just built, so they are derived here:

      spring   damping ``b = 2*zeta*sqrt(k*I)`` with zeta = 1 (critical) and ``I`` the joint's real inertia
               (armature + the handle's own inertia about the spindle), so the handle settles onto its rest stop
               instead of ringing; the preload is raised to ``GRAVITY_MARGIN`` x the handle's own gravity moment
               wherever the catalogue spring could not lift the modelled handle (a lever that hangs off horizontal
               is a failed A156.2 return); the bearing friction is capped at a fraction of the restoring torque at
               rest so friction can never park the handle short of its stop.
      gravity  no spring at all: only bearing friction (a small fraction of the weight moment, so it can never
               lock the part) and near-critical damping against an equivalent rate ``|tau_g| / travel``.
      detent   no spring: Coulomb friction ``detent_friction`` from the catalogue holds it where it was put.

    Returns the ``phys["operator"]["joints"]`` block it writes (one entry per operator joint, with units).
    """
    dyn = phys.get("operator") or P.operator_dynamics(H.OPERATORS[spec["operator"]["model"]])
    phys["operator"] = dyn
    joints = dyn.setdefault("joints", {})
    for b in model.bodies:
        j = b.joint
        if j is None or j.role != "operator" or j.return_kind in ("", "none"):
            continue
        hinge = j.type == "hinge"
        op = H.OPERATORS.get(j.operator_model)
        I_body, m_sub, com_rel, axis = _subtree_about_axis(model, b, j.axis, j.pos)
        I = j.armature + (I_body if hinge else m_sub)
        g_dir = _world_rotation(model, b).T @ np.array([0.0, 0.0, -1.0])
        tau_g_fn = _gravity_moment_fn(axis, m_sub, com_rel, hinge, g_dir)
        tau_g0 = tau_g_fn(0.0)
        travel = float((j.range[1] - j.range[0]) if j.range else (op.travel if op else 1.0)) or 1.0
        rec = {"joint": j.name, "type": j.type, "return_kind": j.return_kind, "operator_model": j.operator_model,
               "travel": travel, "inertia": I, "armature": j.armature, "moving_mass_kg": m_sub,
               "gravity_moment_at_rest": tau_g0, "units": dyn["units"]}
        if j.return_kind == "spring" and j.stiffness > 0:
            k = float(j.stiffness)
            preload_cat = -float(j.springref) * k
            fl_cat = (P.OPERATOR_BEARING_FRICTION + 0.02 * (op.mass if op else m_sub)) if hinge else 0.5
            # the spring has to beat the handle's own weight AND the bearing friction, with margin
            preload = max(preload_cat, P.OPERATOR_GRAVITY_MARGIN * max(tau_g0, 0.0) + 2.0 * fl_cat)
            fl = min(fl_cat, P.OPERATOR_FRICTION_FRACTION * (preload - max(tau_g0, 0.0)))
            fl = max(fl, 0.0)
            j.stiffness, j.springref = k, -preload / k
            j.damping = 2.0 * P.OPERATOR_DAMPING_RATIO * math.sqrt(k * I)
            j.frictionloss = fl
            tol = max(0.01 * travel, math.radians(0.25) if hinge else 2e-4)
            j.return_rest = float(j.range[0]) if j.range else 0.0
            j.return_time_s = P.operator_return_time(I, k, preload, j.damping, fl, tau_g_fn, travel, tol)
            rec.update({"spring_preload": preload, "spring_preload_catalogue": preload_cat, "spring_rate": k,
                        "damping": j.damping, "damping_ratio": P.OPERATOR_DAMPING_RATIO, "frictionloss": fl,
                        "omega_n_rad_s": math.sqrt(k / I), "rest": j.return_rest,
                        "expected_return_time_s": j.return_time_s, "return_tolerance": tol})
            if preload > preload_cat + 1e-9:
                rec["preload_raised"] = (f"catalogue {preload_cat:.2f} -> {preload:.2f}: "
                                         f"{P.OPERATOR_GRAVITY_MARGIN:g} x the {tau_g0:.2f} weight moment of the modelled handle "
                                         f"plus 2 x {fl_cat:.2f} bearing friction")
        elif j.return_kind == "gravity":
            lo = float(j.range[0]) if j.range else 0.0
            # An unsprung part goes over centre if you lift it far enough: past that angle its own weight carries it
            # ON, and a ring pull left standing vertically against its stop balances there for ever.  A real ring on a
            # staple would simply flop over the other way, which a one-sided joint range cannot represent, so the
            # travel stops just short of the over-centre angle and the part always falls back.
            q_over = _over_centre(tau_g_fn, lo, lo + travel) if j.range else None
            if q_over is not None:
                hi_new = max(lo + 0.05 * travel, q_over - 0.02 * travel)
                rec["travel_clamped"] = (f"travel {travel:.3f} -> {hi_new - lo:.3f}: beyond {q_over:.3f} rad the part's "
                                         "own weight carries it over centre and it would balance against its stop")
                j.range = (lo, hi_new)
                travel = hi_new - lo
                rec["travel"] = travel
            q0 = lo + P.GRAVITY_DRIVE_FRACTION * travel
            k_equiv = abs(tau_g0) / max(travel, 1e-6)
            j.stiffness, j.springref = 0.0, 0.0
            # a gravity return has only its own weight to work with, and the weight moment of a lifted fork or ring
            # falls to almost nothing at full lift - so the pin friction is a fraction of the WEAKEST moment between
            # the release point and rest, not of the moment at rest, or the part sticks where it was left
            tau_min = min(abs(tau_g_fn(lo + i * (q0 - lo) / 12.0)) for i in range(13))
            j.frictionloss = min(j.frictionloss, 0.15 * tau_min)
            j.damping = 2.0 * P.GRAVITY_DAMPING_RATIO * math.sqrt(max(k_equiv, 1e-9) * I)
            tol = max(P.GRAVITY_RETURN_TOL_FRACTION * travel, math.radians(0.5) if hinge else 5e-4)
            j.return_rest = _gravity_rest(tau_g_fn, lo, lo + travel, q0)
            j.return_time_s = P.operator_return_time(I, 0.0, 0.0, j.damping, j.frictionloss, tau_g_fn,
                                                     q0, tol, rest=j.return_rest)
            rec.update({"spring_preload": 0.0, "spring_rate": 0.0, "damping": j.damping,
                        "damping_ratio": P.GRAVITY_DAMPING_RATIO, "frictionloss": j.frictionloss,
                        "rest": j.return_rest, "released_from": q0, "expected_return_time_s": j.return_time_s,
                        "return_tolerance": tol, "gravity_moment_at_release": tau_g_fn(q0),
                        "note": ("gravity is restoring: the part falls back to its drawn rest" if abs(j.return_rest - lo) < 1e-6 else
                                 f"gravity holds this part away from its drawn rest and it settles at {j.return_rest:.3f} "
                                 "(a plain ring on the underside of a ceiling hatch hangs down; there is no spring to pull it flush)")})
        elif j.return_kind == "detent":
            hold = (op.detent_friction if op else 0.0) or j.frictionloss
            j.stiffness, j.springref = 0.0, 0.0
            j.frictionloss = max(hold, abs(tau_g0) * 1.5)      # it must also hold its own weight
            j.damping = P.DETENT_DAMPING["hinge" if hinge else "slide"]
            j.return_rest, j.return_time_s = None, None
            rec.update({"detent_friction": j.frictionloss, "damping": j.damping, "spring_preload": 0.0,
                        "spring_rate": 0.0, "rest": None, "expected_return_time_s": None,
                        "note": "stays where it is put; QA `operator_holds` asserts it does not creep back"})
        joints[j.name] = rec
    return joints


MIN_MODELLED_T = 0.003   # m; the thinnest a leaf may be MODELLED.  spec["leaf"]["thickness"] is the mass parameter
#                          (materials.SlabConstruction.area_density smears the whole leaf into it), and for a
#                          chain-link gate that is the 0.3 mm of the mesh wire - which builds a gate leaf, its
#                          stiles, rails and pickets and its collision proxy 0.3 mm thick: a degenerate collider in
#                          both engines and a membrane on screen.  The clamp is applied to the geometry only, after
#                          physics has been derived, so no mass or QA number moves.


def primary_assembly(model: Model):
    """What the primary joint carries: the mass of its whole subtree, and that subtree's lever about the axis.

    mass  one leaf of a pair or a bypass set, but the WHOLE rotor of a revolving door or turnstile and the whole
          stack of a fold - the mass a push on the primary joint has to accelerate.
    arm   the perpendicular distance from the joint axis to that subtree's centre of mass: the lever the door's
          own weight works through when the leaf is laid horizontal, which is what sizes the QA push.  It is
          W/2 for a leaf on a vertical hinge, H/2 for a strip hanging from a horizontal rod, and ~0 for a
          balanced rotor.  Returns (mass, arm) or None.
    """
    pj = model.meta.get("primary_joint")
    if not pj:
        return None
    host = next((b for b in model.bodies if b.joint is not None and b.joint.name == pj), None)
    if host is None:
        return None
    kids = {}
    for b in model.bodies:
        kids.setdefault(b.parent, []).append(b.name)
    total, stack = 0.0, [host.name]
    while stack:
        nm = stack.pop()
        b = model.body(nm)
        if b is None:
            continue
        total += float(b.inertial("full")[0])
        stack.extend(kids.get(nm, []))
    arm = 0.0
    if host.joint.type == "hinge":
        _, _, com_rel, axis = _subtree_about_axis(model, host, host.joint.axis, host.joint.pos)
        perp = np.asarray(com_rel, dtype=float) - float(np.dot(com_rel, axis)) * np.asarray(axis, dtype=float)
        arm = float(np.linalg.norm(perp))
    return total, arm


def build_model(spec: dict, phys: dict | None = None) -> Model:
    phys = phys or P.derive(spec)
    if float(spec["leaf"].get("thickness", 1.0)) < MIN_MODELLED_T:
        spec = {**spec, "leaf": {**spec["leaf"], "thickness": MIN_MODELLED_T, "mass_thickness": spec["leaf"]["thickness"]}}
    model = Model(spec["id"])
    model.meta.update({"door_id": spec["id"], "family": spec["family"], "task": spec.get("task"), "notes": []})
    fam = spec["family"]
    if fam in ("swing_single", "automatic_swing", "cold_storage", "baby_gate"):
        GH.build_swing_single(spec, phys, model)
    elif fam == "swing_double":
        GH.build_swing_double(spec, phys, model)
    elif fam == "dutch":
        GH.build_dutch(spec, phys, model)
    elif fam == "saloon":
        GH.build_saloon(spec, phys, model)
    elif fam == "pivot":
        GH.build_swing_single(spec, phys, model)
    elif fam == "ship_watertight":
        GH.build_ship(spec, phys, model)
    elif fam in ("vault", "blast"):
        GH.build_vault(spec, phys, model)
    elif fam == "gate_swing":
        GH.build_gate_or_fence(spec, phys, model)
    elif fam == "stall":
        GH.build_stall(spec, phys, model)
    elif fam in ("sliding_single", "sliding_bypass", "automatic_sliding", "elevator", "gate_sliding"):
        GO.build_sliding(spec, phys, model)
    elif fam in ("bifold", "accordion"):
        GO.build_folding(spec, phys, model)
    elif fam == "revolving":
        GO.build_revolving(spec, phys, model)
    elif fam == "turnstile_tripod":
        GO.build_turnstile(spec, phys, model, full_height=False)
    elif fam == "turnstile_fullheight":
        GO.build_turnstile(spec, phys, model, full_height=True)
    elif fam in ("garage_sectional", "rollup"):
        GO.build_vertical(spec, phys, model)
    elif fam in ("hatch_floor", "hatch_ceiling", "pet_door", "strip_curtain", "garage_tiltup"):
        GO.build_horizontal(spec, phys, model)
    else:
        raise ValueError(f"unknown family {fam}")
    from .geometry.common import brace_pending
    brace_pending(model)          # parts placed before the member they are screwed to existed
    if fam == "automatic_swing":
        act = spec["kinematics"].get("actuator", {})
        model.meta.setdefault("actuators", []).append({"name": "swing_operator", "joint": model.meta["primary_joint"], "kind": "position", "kp": 150.0, "kv": 40.0, "forcerange": (-act.get("max_torque_Nm", 60), act.get("max_torque_Nm", 60)), "ctrlrange": (0.0, 1.6)})
    # Armature floors: reflected inertia of internal lock/latch mechanisms (gears, springs, spindles).  Also required so
    # MuJoCo's mass-scaled soft constraints (equalities, tendon & joint limits) can act on very light mechanism bodies.
    ARM_HINGE = {"operator": 0.01, "lock": 0.005, "latch": 0.005, "mechanism": 0.002, "secondary": 0.005, "decor": 0.002, "primary": 0.01}
    ARM_SLIDE = {"operator": 0.15, "lock": 0.1, "latch": 0.1, "mechanism": 0.05, "secondary": 0.1, "decor": 0.05, "primary": 0.5}
    for b in model.bodies:
        j = b.joint
        if j is None:
            continue
        floor = (ARM_HINGE if j.type == "hinge" else ARM_SLIDE).get(j.role, 0.005)
        if j.role == "operator" and j.return_kind == "gravity":
            # a ring on a staple, a fork on a plain pin, a teardrop on a screw: no spindle, no spring cassette and
            # no gearing to reflect, so the 0.01 kg*m^2 lock-spindle floor is 10-25x the part's own inertia and
            # turns a 0.4 s drop into a 2 s creep
            floor = 1e-4 if j.type == "hinge" else 0.02
        if j.role == "primary" and j.type == "hinge":
            floor = 0.02
        if j.role == "primary" and j.type == "slide":
            floor = 0.5
        j.armature = max(j.armature, floor)
    op = spec["opening"]
    # --- mass reconciliation.  physics.mass_budget states what ONE leaf is made of (slab area density x its own
    # area, plus its glazing) and how many leaves the door has; the leaf bodies together must weigh ALL of them.
    # Splitting one leaf's mass across the leaves - what this used to do - made every multi-leaf door 2-8x too
    # light (a four-wing revolving door 110 kg where its wings alone are 440).  On top of the material, the leaf
    # carries any declared hardware the geometry did not model as its own body (tracks, hangers, straps, plates);
    # where the geometry models MORE hardware than the catalogue charged, the rider is zero and the leaf keeps
    # exactly its material.  qa.leaf_mass_checks re-derives all of this from the spec and gates it.
    leaf_bodies = [b for b in model.bodies if getattr(b, "semantic", "") == "leaf" and not b.static]
    hw_now = float(sum(b.inertial("full")[0] for b in model.bodies if not b.static and getattr(b, "semantic", "") != "leaf"))
    leaf_material = float(phys["mass"].get("slab_kg", 0.0) + phys["mass"].get("glass_kg", 0.0))   # ALL leaves
    rider = max(float(phys["mass"].get("hardware_kg", 0.0)) - hw_now, 0.0)
    tgt_mass = leaf_material + rider
    if tgt_mass > 0 and leaf_bodies:
        vols = [max(sum((g.volume() or 0.0) for g in b.geoms), 1e-6) for b in leaf_bodies]
        vt = sum(vols)
        for b, vol in zip(leaf_bodies, vols):
            b.mass_override = tgt_mass * vol / vt
        model.meta["mass_reconciled_kg"] = tgt_mass
    model.meta["mass"] = {"leaf_material_kg": leaf_material, "leaf_hardware_rider_kg": rider,
                          "hardware_modelled_kg": hw_now, "leaf_bodies": len(leaf_bodies),
                          "total_moving_kg": tgt_mass + hw_now if (tgt_mass > 0 and leaf_bodies) else leaf_material + hw_now}
    phys["mass"]["model_total_moving_kg"] = model.meta["mass"]["total_moving_kg"]
    phys["mass"]["leaf_hardware_rider_kg"] = rider
    model.meta["scene_extent"] = max(1.5, 0.75 * max(op["width"], op["height"]) + 0.5)
    model.meta["cam_target_z"] = 0.5 * op["height"] + float(op.get("elevation", 0.0) or 0.0) + float(op.get("sill_height", 0.0) or 0.0) * 0.5
    model.meta["cam_target_x"] = 0.0
    model.meta["handle_cam_x"] = float(model.meta.get("hinge_x", 0.0) + model.meta.get("u", 1.0) * (spec["leaf"]["width"] - 0.1)) if model.meta.get("u") is not None else 0.3
    # handle-detail camera: aim at the first grip site (world position via the parent chain; rotations are identity)
    def _world_pos(body_name):
        p = [0.0, 0.0, 0.0]
        seen = 0
        while body_name and seen < 12:
            b = model.body(body_name)
            p = [p[i] + float(b.pos[i]) for i in range(3)]
            body_name = b.parent
            seen += 1
        return p
    grip = None
    for b in model.bodies:
        for s_ in b.sites:
            if getattr(s_, "role", "") == "grip":
                wp = _world_pos(b.name)
                grip = [wp[i] + float(s_.pos[i]) for i in range(3)]
                break
        if grip:
            break
    if grip is None:
        grip = [model.meta["handle_cam_x"], float(model.meta.get("wall_y", 0.0)), float(model.meta.get("handle_height", 1.0))]
    fam = spec["family"]
    if fam == "hatch_floor":
        off = (0.35, -0.55, 0.85)
    elif fam == "hatch_ceiling":
        off = (0.35, -0.55, -0.85)
    elif fam in ("garage_sectional", "garage_tiltup", "rollup", "gate_sliding", "turnstile_tripod", "turnstile_fullheight", "revolving"):
        off = (0.25, -1.1, 0.35)
    else:
        off = (0.18, -0.8, 0.28)
    model.meta["handle_cam_target"] = grip
    model.meta["handle_cam_pos"] = [grip[0] + off[0], grip[1] + off[1], grip[2] + off[2]]
    # --- multi-latch doors: EVERY operator the robot has to work, not just the first one.
    # `operator_joints` lists them (a single-operator door lists its one joint, so every consumer reads one key) and
    # `operator_coupling` says how they relate:
    #   "individual"  independent releases - watertight dog levers, blast-door lever bolts: each one holds the leaf on
    #                 its own, so the door frees only when ALL of them are released (QA gate "all_latches_release").
    #   "coupled"     one operator drives every lock point through the mechanism (ship handwheel -> dogs, vault
    #                 handwheel -> boltwork, cremone knob -> shoot bolts, multipoint lever -> hooks).
    have = {b.joint.name for b in model.bodies if b.joint is not None}
    ops = [n for n in (model.meta.get("operator_joints") or []) if n in have]
    if not ops and model.meta.get("operator_joint"):
        ops = [model.meta["operator_joint"]]
    model.meta["operator_joints"] = ops
    if len(ops) < 2:
        model.meta["operator_coupling"] = "coupled"
    else:
        model.meta.setdefault("operator_coupling", "individual")
    # operator release behaviour: needs the finished geometry (real inertia, real weight moment) and the armature floors
    tune_operator_returns(model, spec, phys)
    # what the robot actually has to move: the mass hanging on the primary joint, MEASURED on the finished model.
    # physics.leaves_on_primary only estimates it (the geometry is what decides whether a fold stacks to one jamb
    # or two), and everything that sizes an effort from it - the QA push, the parity protocol, the benchmark's
    # transit time - should use the measured number.
    pa = primary_assembly(model)
    if pa is not None and pa[0] > 0:
        phys["mass"]["primary_assembly_estimated_kg"] = float(phys["mass"].get("primary_assembly_kg", 0.0))
        phys["mass"]["primary_assembly_kg"] = pa[0]
        phys["mass"]["primary_com_arm_m"] = pa[1]
        model.meta["primary_assembly_kg"] = pa[0]
    model.bake_initial()
    model.uniquify()
    model.validate()
    return model


def export_door(spec: dict, out_root: str, hardware_dir: str, formats=("mjcf", "urdf", "usd", "json"), tiers=("full", "simple", "minimal")) -> dict:
    """Export one door.  Returns a summary dict."""
    from .export import mjcf as XM
    t0 = time.time()
    phys = P.derive(spec)
    model = build_model(spec, phys)
    out_dir = os.path.join(out_root, spec["id"])
    os.makedirs(out_dir, exist_ok=True)
    rel_hw = os.path.relpath(hardware_dir, out_dir)
    rel_tex = os.path.relpath(os.path.join(os.path.dirname(hardware_dir), "textures"), out_dir)
    summary = {"id": spec["id"], "family": spec["family"], "files": {}, "mass_kg": phys["mass"]["total_kg"], "n_bodies": len(model.bodies_in_tier("full")), "n_joints": len(model.joints("full"))}
    # meshes -> shared hardware library
    written = write_hardware_meshes(model, hardware_dir)
    summary["meshes"] = written
    if "mjcf" in formats:
        summary["files"]["mjcf"] = XM.write_mjcf(model, out_dir, tiers=tiers, mesh_dir_rel=rel_hw, texture_dir_rel=rel_tex)
    if "urdf" in formats:
        from .export import urdf as XU
        summary["files"]["urdf"] = XU.write_urdf(model, out_dir, mesh_dir_rel=rel_hw)
    if "usd" in formats:
        from .export import usd as XS
        try:
            summary["files"]["usd"] = XS.write_usd(model, out_dir, hardware_dir=hardware_dir)
        except Exception as e:  # pragma: no cover
            summary["files"]["usd"] = f"ERROR: {e!r}"
        try:
            # canonical articulation for Isaac Lab multi-door training (same link/joint names for every door)
            summary["files"]["usd_rl"] = XS.write_usd_rl(model, out_dir, hardware_dir=hardware_dir, spec={**spec, "physics": phys})
        except Exception as e:  # pragma: no cover
            summary["files"]["usd_rl"] = f"ERROR: {e!r}"
    if "json" in formats:
        from .benchmark.scenarios import build_benchmark, benchmark_summary
        model_dict = json.loads(json.dumps(model.to_dict("full"), default=_json_default))
        bench = build_benchmark(spec, phys, model_dict)      # scenarios + rewards (docs/BENCHMARK.md)
        summary["benchmark"] = benchmark_summary(bench)
        with open(os.path.join(out_dir, "spec.json"), "w") as f:
            json.dump({**spec, "physics": phys, "benchmark": bench}, f, indent=1, default=_json_default)
        with open(os.path.join(out_dir, "model.json"), "w") as f:
            json.dump(model_dict, f)
    summary["build_time_s"] = time.time() - t0
    return summary


def write_hardware_meshes(model: Model, hardware_dir: str) -> list:
    """Write every shared mesh used by the model to hardware_dir/<key>.obj (once)."""
    os.makedirs(hardware_dir, exist_ok=True)
    out = []
    for b in model.bodies:
        for g in b.geoms:
            if g.type == "mesh" and g.mesh is not None:
                path = os.path.join(hardware_dir, f"{g.mesh_name}.obj")
                if not os.path.exists(path):
                    g.mesh.export(path, include_normals=False, include_texture=False)
                    out.append(g.mesh_name)
    return out


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(repr(o))

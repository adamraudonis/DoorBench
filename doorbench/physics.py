"""Physics derivations for a door specification.

All quantities are derived from first principles + catalog data and returned
with the formula / source so researchers can audit every number.
"""
from __future__ import annotations

import math
from typing import Optional

from . import materials as M
from . import hardware as H
from .taxonomy import CONDITIONS

G = 9.80665
AIR_DENSITY = 1.204


def leaf_count(spec: dict) -> int:
    """How many leaves the door has.  ``spec["leaf"]`` describes ONE of them (width, height, thickness, slab); a
    pair, a bypass set, a fold, a revolving rotor and a strip curtain all repeat that leaf ``count`` times."""
    return max(1, int(spec["leaf"].get("count", 1) or 1))


def leaves_on_primary(spec: dict) -> float:
    """How many leaves ride the door's PRIMARY degree of freedom - the one the robot drives.

    A pair, a bypass set and a bi-parting slider give every leaf its own joint, so the robot moves one of them.
    A revolving door or a turnstile hangs every wing on one rotor; a fold carries a whole stack on its pivot
    panel; a dutch door splits its one leaf into two independently hinged halves, so the primary joint carries
    about half of it.  ``build.build_model`` overwrites the resulting ``primary_assembly_kg`` with the mass it
    measures on the finished model, so this is only the estimate the geometry is built from."""
    fam = spec["family"]
    kin = spec["kinematics"]
    n = float(leaf_count(spec))
    if kin["type"] == "rotor":
        return n                                   # every wing / arm turns with the rotor
    if kin.get("fold"):
        from .folding import fold_groups
        return n / fold_groups(int(n), bool(kin.get("accordion")))
    if fam == "dutch":
        return 0.5                                 # top half on the primary joint, bottom half on its own
    return 1.0


def one_leaf_material(spec: dict) -> dict:
    """Slab + glazing of ONE leaf, from the slab's area density and the glazing's own material."""
    leaf = spec["leaf"]
    slab = M.SLABS[leaf["slab"]]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    area = W * Hh
    glaz = leaf.get("glazing") or {}
    glass_area = float(glaz.get("area_fraction", 0.0)) * area
    slab_area = area - glass_area
    ad = slab.area_density(t)
    slab_mass = ad * slab_area
    fam = spec.get("family", "")
    if fam == "turnstile_tripod":
        # ONE arm: 38 mm OD x 1.5 mm wall stainless tube, 0.5 m, plus its third of the hub casting.  There are
        # `count` (= 3) of them and the budget multiplies by that, so this must be per arm, not per tripod.
        slab_mass = math.pi * (0.019 ** 2 - 0.0175 ** 2) * W * 7900 + 1.0
    elif fam == "turnstile_fullheight":
        arms = leaf.get("arms_per_wing", 8)
        slab_mass = arms * (math.pi * (0.019 ** 2 - 0.0175 ** 2) * W * 7900) + 4.0  # per wing incl. share of rotor column
    elif fam == "strip_curtain":
        slab_mass = W * Hh * t * M.MATERIALS["pvc_flexible"].density   # one strip
    glass_mass = 0.0
    if glass_area > 0:
        gm = M.MATERIALS[glaz.get("material", "glass_clear")]
        gt = float(glaz.get("thickness", 0.006))
        glass_mass = glass_area * gt * gm.density
    return {"slab_kg": slab_mass, "glass_kg": glass_mass, "slab_area_density_kg_m2": ad, "source": slab.source}


def door_hardware(spec: dict) -> dict:
    """Hardware mass that moves with the door: operator, lock, latch, door-mounted closer, leaf-side hinge halves
    and the extras bolted to the leaf.

    Charged ONCE for the door, not once per leaf: every one of these is sampled once in the spec and
    ``spec["hinge"]["count"]`` is already the door's hinge count (one rotor bearing for a revolving door, the
    n-1 piano hinges between an accordion's panels, the pair of butts a bifold set hangs on).  Only the leaf
    MATERIAL repeats per leaf - that is what ``mass_budget`` multiplies by ``leaf_count``.

    Frame-side hardware (strike, keeper, mag-lock magnet, closer body on the frame) is not here: it never moves.
    """
    parts = {}
    op = H.OPERATORS[spec["operator"]["model"]]
    parts["operator"] = op.mass
    lk = H.LOCKS[spec["lock"]["model"]]
    parts["lock"] = lk.mass
    lt = H.LATCHES[spec["latch"]["model"]]
    if lt.mass:
        parts["latch"] = lt.mass          # the case, bolt, rods, dogs or boltwork the leaf carries
    cl = H.CLOSERS[spec["closer"]["model"]]
    if cl.mounts_on in ("door_push", "door_pull"):
        parts["closer"] = cl.mass
    hg = H.HINGES[spec["hinge"]["model"]]
    parts["hinges_half"] = 0.5 * hg.mass_each * spec["hinge"]["count"]
    for e in spec.get("extras", []):
        parts[e] = EXTRA_MASS.get(e, 0.0)
    return {"parts": parts, "total_kg": sum(parts.values())}


def mass_budget(spec: dict) -> dict:
    """The door's mass budget, at both levels, with the level in every name.

    ``spec["leaf"]`` describes ONE leaf and the door has ``leaf_count`` of them, so there are two different
    masses here and using either where the other belongs is a physics bug:

      per_leaf_kg     one leaf, its glazing and the hardware bolted to it - what ONE hinge set carries, what a
                      closer is sized for, what slams into the frame.
      total_kg        the whole door: every leaf's material plus the door's hardware.  This is what the model's
                      moving bodies must weigh (``build.build_model`` reconciles them to it and ``qa`` gates it).
      primary_assembly_kg  what the robot actually has to move: one leaf of a pair, but the whole rotor of a
                      revolving door and the whole stack of a fold.  ``build`` replaces the estimate here with
                      the mass it measures on the finished model.
    """
    n = leaf_count(spec)
    mat = one_leaf_material(spec)
    hw = door_hardware(spec)
    per_leaf = mat["slab_kg"] + mat["glass_kg"] + hw["total_kg"]
    door_hw = hw["total_kg"]
    total = n * (mat["slab_kg"] + mat["glass_kg"]) + door_hw
    k = leaves_on_primary(spec)
    return {
        "leaf_count": n,
        # ---- one leaf
        "leaf_slab_kg": mat["slab_kg"], "leaf_glass_kg": mat["glass_kg"],
        "leaf_hardware_kg": hw["total_kg"], "per_leaf_kg": per_leaf,
        "slab_area_density_kg_m2": mat["slab_area_density_kg_m2"], "hardware_parts": hw["parts"],
        # ---- the whole door
        "slab_kg": n * mat["slab_kg"], "glass_kg": n * mat["glass_kg"], "hardware_kg": door_hw,
        "total_kg": total,
        # ---- what the primary joint carries (estimate; build.py measures it on the model)
        "leaves_on_primary": k,
        "primary_assembly_kg": k * (mat["slab_kg"] + mat["glass_kg"]) + hw["total_kg"],
        "formula": ("per leaf: slab_area_density(t) * (W*H - A_glass) + rho_glass * t_glass * A_glass + hardware;  "
                    "door: leaf_count * (slab + glass) + the door's hardware set (charged once)"),
        "source": mat["source"],
    }


EXTRA_MASS = {
    "kick_plate": 1.4, "armor_plate": 4.5, "push_plate": 0.4, "peephole": 0.05, "mail_slot": 0.5, "knocker": 0.6,
    "house_number": 0.1, "pet_flap": 1.8, "chain_lock": 0.15, "swing_bar_guard": 0.25, "exit_sign": 0.0,
    "push_pull_sign": 0.05, "vision_lite_grille": 0.8, "door_stop_floor": 0.0, "door_stop_wall": 0.0,
    "hold_open_kickdown": 0.3, "wreath": 0.8, "keypad_reader_wall": 0.0, "rex_button": 0.0, "wave_sensor": 0.0,
    "call_button": 0.0, "threshold_saddle": 0.0, "weather_drip_cap": 0.3, "door_viewer_camera": 0.2, "coat_hook": 0.15,
    "bumper_rail": 1.2, "louver_vent": 0.9, "transom_window": 0.0, "sidelite": 0.0, "warning_placard": 0.1,
    "floor_guide": 0.0, "soft_close_damper": 0.0,
}


def hinge_friction(spec: dict, mass_kg: float) -> dict:
    """Coulomb friction torque about the hinge line.

    Load model: vertical load m*g on knuckle thrust faces (radius r_t);
    horizontal reactions at top/bottom hinges from the leaf's weight moment:
    F_h = m g (W/2 - x_hinge) / L_span  (each), acting on pin (radius r_p).
    tau_f = mu * (m g r_t + 2 F_h r_p) * condition_multiplier + seal drag.
    """
    hinge = spec["hinge"]
    hg = H.HINGES[hinge["model"]]
    mu, mu_note = M.HINGE_BEARING_MU[hg.bearing]
    W = spec["leaf"]["width"]
    Hh = spec["leaf"]["height"]
    kin = spec["kinematics"]["type"]
    cond = CONDITIONS[spec["condition"]]
    if kin in ("hinge_vertical",):
        span = max(0.3, Hh - 0.45) if hg.count_default >= 2 else 0.3
        com_arm = W / 2
        F_h = mass_kg * G * com_arm / span
        tau = mu * (mass_kg * G * hg.thrust_radius + 2 * F_h * hg.pin_radius)
    elif kin in ("hinge_horizontal",):
        # hatch: gravity load mostly radial on pins while closed; use full weight on pins
        tau = mu * mass_kg * G * hg.pin_radius * 1.5
    elif kin in ("rotor",):
        tau = mu * mass_kg * G * hg.thrust_radius
    else:
        tau = mu * mass_kg * G * hg.pin_radius
    tau *= cond["friction_mult"]
    # seal drag (converted to torque at the leaf edge, active only near closed; we add 30% as steady)
    seal = H.SEALS[spec["seal"]]
    seal_len = 2 * Hh + W
    seal_torque = min(3.0, 0.03 * seal["closing_resistance_N_per_m"] * seal_len * 0.5 * W)
    total = tau + seal_torque
    return {
        "coulomb_torque_Nm": total, "bearing_mu": mu, "bearing_note": mu_note,
        "pin_radius_m": hg.pin_radius, "thrust_radius_m": hg.thrust_radius,
        "condition_multiplier": cond["friction_mult"], "seal_contribution_Nm": seal_torque,
        "stick_torque_Nm": cond["stick_torque"],
        "formula": "mu*(m*g*r_thrust + 2*F_h*r_pin)*k_cond + seal_steady; F_h = m g (W/2)/span; seal_steady = min(3, 0.03*R*L*W/2) (gasket compression itself is the soft stop contact)",
    }


def air_damping(W: float, Hh: float) -> float:
    """Linearised aerodynamic damping torque coefficient (N*m*s/rad) at omega ~ 1 rad/s.
    tau = 0.5*rho*Cd*H*omega^2*W^4/4 with Cd ~ 1.2 -> linearize at omega=1."""
    return 0.5 * AIR_DENSITY * 1.2 * Hh * W ** 4 / 4


def closer_params(spec: dict, mass_kg: float, friction_Nm: float = 0.0) -> dict:
    cl = H.CLOSERS[spec["closer"]["model"]]
    W = spec["leaf"]["width"]
    need = 1.3 * friction_Nm   # closing moment must beat hinge + steady seal friction with margin
    out = {"model": cl.id, "kind": cl.kind, "spring_stiffness_Nm_per_rad": 0.0, "spring_preload_Nm": 0.0,
           "damping_closing": 0.0, "damping_opening": 0.0, "en_size": None, "hold_open_rad": cl.hold_open,
           "backcheck_angle_rad": cl.backcheck_angle, "backcheck_damping": cl.backcheck_damping, "latch_boost": cl.latch_boost}
    if cl.kind == "none":
        return out
    if cl.kind in ("surface_overhead", "concealed_overhead", "floor_spring", "electromagnetic_hold", "auto_operator_low_energy", "auto_operator_full", "pneumatic", "gate"):
        size = cl.en_size or spec["closer"].get("en_size") or H.closer_size_for(mass_kg, W)
        size = int(max(1, min(7, size)))
        adj = spec["closer"].get("spring_adjust", 1.15)
        while size < 7 and H.EN1154_SIZES[size].closing_moment_min * adj < need and cl.kind not in ("pneumatic", "gate"):
            size += 1   # installer picks the next size up when the door is stiff
        cs = H.EN1154_SIZES[size]
        # closers are set 10-20% above the minimum closing moment; opening moment ~85% of the max allowed
        tau0 = max(cs.closing_moment_min * adj, need)
        tau90 = min(cs.opening_moment_max * 0.85, tau0 * 2.8)
        k = max((tau90 - tau0) / (math.pi / 2), 0.5)
        if cl.kind == "pneumatic":
            tau0, k = max(3.0 * adj, need), 3.0
        if cl.kind == "gate":
            tau0, k = max(4.0 * adj, need), 5.0
        if tau0 > cs.closing_moment_min * adj + 1e-9:
            out["note"] = f"spring tension raised to {tau0:.1f} N*m to overcome {friction_Nm:.1f} N*m hinge/seal friction"
        out.update({"en_size": size, "spring_stiffness_Nm_per_rad": k, "spring_preload_Nm": tau0,
                    "damping_closing": cl.closing_damping * CONDITIONS[spec["condition"]]["damping_mult"],
                    "damping_opening": cl.opening_damping,
                    "closing_time_est_s": _closing_time(mass_kg, W, tau0, k, cl.closing_damping),
                    "formula": "tau(theta) = tau0 + k*theta; tau0 = 1.15*EN1154 closing moment(size); tau90 = 0.85*EN1154 opening moment",
                    "source": H.SOURCES_EN if hasattr(H, 'SOURCES_EN') else "EN 1154:1996 Table 1"})
    elif cl.kind == "spring_hinge":
        n = spec["hinge"]["count"]
        # Bommer 4310 class: ~2.5-4 N*m per hinge at 90 deg, preload ~1 N*m
        k_each = spec["closer"].get("spring_hinge_k", 2.2)
        out.update({"spring_stiffness_Nm_per_rad": k_each * n, "spring_preload_Nm": max(0.9 * n, need),
                    "damping_closing": cl.closing_damping, "damping_opening": cl.opening_damping,
                    "formula": "n_hinges * (0.9 N*m + 2.2 N*m/rad * theta)", "source": "Bommer 4310 adjustable spring hinge"})
    elif cl.kind == "gas_strut":
        F = spec["closer"].get("gas_force_N", 250.0)
        arm = 0.25
        out.update({"spring_stiffness_Nm_per_rad": -F * arm * 0.3, "spring_preload_Nm": -F * arm,
                    "damping_closing": 30.0, "damping_opening": 30.0,
                    "formula": "lift assist: tau = -F*arm (negative = assists opening)", "source": "Gas spring 150-400 N"})
    return out


def _closing_time(m, W, tau0, k, b):
    """Crude estimate of 90->0 closing time under spring, viscous damping, hinge inertia."""
    I = m * W * W / 3
    dt, th, w, t = 0.002, math.pi / 2, 0.0, 0.0
    while th > 0.01 and t < 30:
        tau = -(tau0 + k * th) - b * w
        a = tau / I
        w += a * dt
        th += w * dt
        t += dt
    return t


def latch_params(spec: dict) -> dict:
    lt = H.LATCHES[spec["latch"]["model"]]
    op = H.OPERATORS[spec["operator"]["model"]]
    return {
        "model": lt.id, "kind": lt.kind, "throw_m": lt.throw, "bolt_spring_preload_N": lt.spring_preload,
        "bolt_spring_rate_N_per_m": lt.spring_rate, "holding_force_N": lt.holding_force, "yield_force_N": lt.yield_force,
        "backset_m": lt.backset,
        "operator_travel": op.travel, "operator_dead_travel": op.dead_travel,
        "operator_spring_preload": op.spring_torque_preload, "operator_spring_rate": op.spring_rate,
        "operator_yield": op.yield_torque, "operator_grip_offset_m": op.grip_offset,
        "coupling": None if not op.unlatches or lt.throw == 0 else {
            "type": "polynomial", "bolt_q = c0 + c1*op_q": [0.0, -lt.throw / max(op.travel - op.dead_travel, 1e-6)],
            "dead_travel": op.dead_travel,
        },
    }


# ---------------------------------------------------------------------------
# Operator return dynamics: what the handle does when the hand lets go
# ---------------------------------------------------------------------------
OPERATOR_DAMPING_RATIO = 1.0      # critical: the handle comes home without ringing against its rest stop
GRAVITY_DAMPING_RATIO = 0.6       # a ring on a staple / fork on a pin has no spring cassette to damp it, only its
#                                   pin and the air: under-critical, so it still falls at a believable speed
GRAVITY_DRIVE_FRACTION = 0.90     # a hand lifts a gravity latch this far; at 100 % a ring stands vertical, where its
#                                   own weight moment is exactly zero and it balances (a real quirk, not a bug)
GRAVITY_RETURN_TOL_FRACTION = 0.05  # "back at rest" for an unsprung part: within 5 % of the travel (no spring holds
#                                     it hard against a stop, so the last millimetre is not meaningful)
OPERATOR_GRAVITY_MARGIN = 1.35    # the return spring must beat the handle's own weight moment by this factor
OPERATOR_BEARING_FRICTION = 0.02  # N*m base Coulomb friction of a spindle bearing (+0.02 per kg of hardware)
OPERATOR_FRICTION_FRACTION = 0.25   # ... capped at this share of the restoring torque at rest, so friction can
#                                     never park the handle short of its stop (that is a residual offset)
ROTARY_MOTIONS = ("rotate_normal", "rotate_horizontal", "rotate_vertical")
DETENT_DAMPING = {"hinge": 0.5, "slide": 2.0}


def operator_return_time(I: float, k: float, preload: float, b: float, fl: float, grav, q0: float, tol: float,
                         dt: float = 0.0005, t_max: float = 3.0, rest: float = 0.0) -> float | None:
    """Time for a released operator to come home, from a 1-D integration of the joint alone.

    Released at ``q0`` with the joint at rest, under the return spring ``-(preload + k*q)``, the gravity
    moment ``grav`` of the handle itself (a constant or a callable of q; positive = toward actuation), viscous ``b`` and Coulomb ``fl``, with a
    hard rest stop at q = 0.  Returns the time until ``|q| < tol`` (None if it never gets there).  This is the
    number recorded as ``expected_return_time_s``; QA measures the real one in MuJoCo."""
    if I <= 0 or q0 <= rest:
        return 0.0
    g_of = grav if callable(grav) else (lambda _q: grav)
    q, w, t = q0, 0.0, 0.0
    while t < t_max:
        tau = g_of(q) - (preload + k * q) - b * w
        if abs(w) < 1e-6:
            tau -= math.copysign(min(fl, abs(tau)), tau)      # stiction: friction cancels the drive up to fl
        else:
            tau -= math.copysign(fl, w)
        w += tau / I * dt
        q += w * dt
        t += dt
        if q <= 0.0:
            q, w = 0.0, 0.0        # the rest stop (joint limit); the spring holds it there
        if abs(q - rest) < tol:
            return round(t, 4)
    return None


def operator_dynamics(op: H.OperatorModel, preload_override: float | None = None) -> dict:
    """Catalogue-level return behaviour of one operator, with units.

    ``build.tune_operator_returns`` completes this per joint once the geometry is known (real rotational inertia,
    gravity moment, near-critical damping, measured-in-1D return time) and writes the result back here under
    ``joints``.  Units: rotary joints N*m, N*m/rad, N*m*s/rad, kg*m^2, rad; linear joints N, N/m, N*s/m, kg, m."""
    rotary = op.motion in ROTARY_MOTIONS
    kind = op.return_kind if op.motion != "none" or op.return_kind in ("gravity", "detent") else "none"
    pre = op.spring_torque_preload if preload_override is None else preload_override
    units = ({"preload": "N*m", "rate": "N*m/rad", "damping": "N*m*s/rad", "friction": "N*m", "inertia": "kg*m^2", "travel": "rad"}
             if rotary else {"preload": "N", "rate": "N/m", "damping": "N*s/m", "friction": "N", "inertia": "kg", "travel": "m"})
    return {
        "model": op.id, "kind": op.kind, "motion": op.motion, "rotary": rotary,
        "return_kind": kind, "return_note": op.return_note, "source": op.source,
        "travel": op.travel, "dead_travel": op.dead_travel,
        "spring_preload": pre if kind == "spring" else 0.0,
        "spring_rate": op.spring_rate if kind == "spring" else 0.0,
        "detent_friction": op.detent_friction if kind == "detent" else 0.0,
        "damping_ratio": OPERATOR_DAMPING_RATIO if kind == "spring" else (GRAVITY_DAMPING_RATIO if kind == "gravity" else None),
        "gravity_margin": OPERATOR_GRAVITY_MARGIN if kind == "spring" else None,
        "units": units,
        "joints": {},   # filled by build.tune_operator_returns: one entry per operator joint actually built
        "formula": ("tau = -(preload + k*q) - b*dq - fl*sign(dq); b = 2*zeta*sqrt(k*I) with zeta = "
                    f"{OPERATOR_DAMPING_RATIO} (critical) and I the joint-space inertia (armature + the handle's own "
                    "inertia about the spindle); preload raised to gravity_margin x the handle's gravity moment where "
                    "the catalogue spring would not lift it; fl capped at "
                    f"{OPERATOR_FRICTION_FRACTION:g} of the restoring torque at rest so it cannot park the handle short of rest"
                    if kind == "spring" else
                    "no spring: the part's own weight returns it; light bearing friction and near-critical damping only"
                    if kind == "gravity" else
                    "no spring: Coulomb friction detent_friction holds the part where it was left"
                    if kind == "detent" else "no moving operator part"),
    }


def lock_params(spec: dict) -> dict:
    lk = H.LOCKS[spec["lock"]["model"]]
    engaged = bool(spec["lock"].get("engaged", False))
    robot_can_release = spec["lock"].get("robot_side_release", False)
    return {
        "model": lk.id, "kind": lk.kind, "engaged": engaged, "robot_side_release": robot_can_release,
        "outside_release": lk.outside_release, "inside_release": lk.inside_release,
        "handle_backlash_locked_rad": lk.handle_backlash_locked + CONDITIONS[spec["condition"]]["backlash_add"],
        "deadbolt_throw_m": lk.deadbolt_throw, "thumbturn_travel_rad": lk.thumbturn_travel, "thumbturn_torque_Nm": lk.thumbturn_torque,
        "chain_slack_m": lk.chain_slack, "code": spec["lock"].get("code"),
    }


def compliance(spec: dict, phys: dict) -> dict:
    """ADA / IBC style checks on the *simulated* door as built."""
    W = spec["leaf"]["width"]
    op = H.OPERATORS[spec["operator"]["model"]]
    arm = max(W - (H.LATCHES[spec["latch"]["model"]].backset or 0.06) - 0.02, 0.3)
    tau_static = phys["hinge"]["coulomb_torque_Nm"] + phys["closer"]["spring_preload_Nm"] + phys["hinge"]["stick_torque_Nm"]
    tau_90 = phys["hinge"]["coulomb_torque_Nm"] + phys["closer"]["spring_preload_Nm"] + phys["closer"]["spring_stiffness_Nm_per_rad"] * math.pi / 2
    F_start = tau_static / arm
    F_90 = tau_90 / arm
    op_force = 0.0
    if op.kind == "paddle" and op.motion == "rotate_horizontal" and op.grip_offset > 0:
        # Face-normal force about the horizontal rocker pin.  Match the
        # generator's retained return-spring floor; this is a spring estimate,
        # excluding latch/cam friction and gravity, not a certification.
        op_force = (max(op.spring_torque_preload, 1.5) + op.spring_rate * op.travel) / op.grip_offset
    elif op.motion == "rotate_normal" and op.grip_offset > 0:
        op_force = (op.spring_torque_preload + op.spring_rate * op.travel) / op.grip_offset
    elif op.motion == "push_in":
        op_force = op.spring_torque_preload + op.spring_rate * op.travel
    is_fire = M.SLABS[spec["leaf"]["slab"]].fire_rating_min > 0 or spec.get("context") == "fire_egress"
    exterior = spec.get("context") in ("residential_exterior", "storefront_glass", "fire_egress") or spec["family"].startswith("gate")
    return {
        "opening_force_start_N": F_start, "opening_force_90deg_N": F_90, "operator_force_N": op_force,
        "ada_interior_5lbf_ok": (F_start <= 22.2 and F_90 <= 22.2) if not (is_fire or exterior) else None,
        "ibc_fire_exterior_ok": (F_start <= 133.4 and F_90 <= 66.7) if (is_fire or exterior) else None,
        "hardware_operable_5lbf_ok": op_force <= 22.2,
        "panic_unlatch_15lbf_ok": (op_force <= 66.7) if op.kind.startswith("panic") else None,
        "lever_or_ada_hardware": op.kind in ("lever", "pull", "push_plate", "panic_touchbar", "panic_crossbar", "paddle", "keypad_lever", "card_lever", "none", "flush_pull"),
        "handle_height_ada_ok": 0.864 <= spec["operator"]["height"] <= 1.219 if spec["operator"]["height"] > 0 else None,
        "clear_width_ada_ok": (W - spec["leaf"]["thickness"] - 0.03) >= 0.815,
        "notes": "ADA 2010 §404.2.9 (5 lbf interior), IBC §1010.1.3 (30 lbf set in motion / 15 lbf full open), §1010.1.10 (panic 15 lbf)",
    }


def damage_thresholds(spec: dict, mass_kg: float) -> dict:
    slab = M.SLABS[spec["leaf"]["slab"]]
    face = M.slab_face_material(slab)
    op = H.OPERATORS[spec["operator"]["model"]]
    lt = H.LATCHES[spec["latch"]["model"]]
    hg = H.HINGES[spec["hinge"]["model"]]
    glaz = spec["leaf"].get("glazing") or {}
    g = M.MATERIALS.get(glaz.get("material", "glass_clear")) if glaz else None
    return {
        "leaf_dent_force_N": face.dent_force_N * (2.0 if not slab.monolithic and slab.core_fill_fraction >= 1.0 else 1.0),
        "leaf_puncture_force_N": face.puncture_force_N,
        "glass_break_force_N": (g.puncture_force_N if g else None),
        "operator_yield_torque_Nm": op.yield_torque,
        "latch_shear_yield_N": lt.yield_force,
        "hinge_tearout_force_N": min(hg.max_door_mass * G * 4.0, 20000.0),
        "slam_velocity_rad_s": 4.0 if spec["kinematics"]["type"].startswith("hinge") else 2.0,
        "slam_impulse_Ns": 0.4 * mass_kg,
        "frame_impact_force_N": 3000.0,
        "notes": "Damage events are labelled when contact force / joint torque exceed these thresholds (see benchmark/labels.py)",
    }


def roller_friction(spec: dict, mass_kg: float) -> dict:
    kin = spec["kinematics"]
    mu, note = M.ROLLER_FRICTION[kin.get("roller", "ball_bearing_nylon")]
    cond = CONDITIONS[spec["condition"]]
    F = mu * mass_kg * G * cond["friction_mult"]
    # counterbalance for vertical doors
    cb = kin.get("counterbalance_fraction", 0.0)
    return {"coulomb_force_N": F, "mu_rolling": mu, "note": note, "condition_multiplier": cond["friction_mult"],
            "counterbalance_fraction": cb, "net_lift_force_N": mass_kg * G * (1 - cb) if cb else None,
            "formula": "F = mu_roll * m * g * k_cond", "viscous_damping_N_s_per_m": 2.0 + 0.05 * mass_kg}


def derive(spec: dict) -> dict:
    """Full physics block for a spec."""
    lm = mass_budget(spec)
    # Two masses, never interchangeable: ONE leaf (what a hinge set carries, what a closer is sized for, what
    # slams) and the PRIMARY ASSEMBLY (what the robot drives: one leaf of a pair, but the whole rotor of a
    # revolving door and the whole stack of a fold, which is also what the bearings and the track carry).
    m = lm["per_leaf_kg"]
    m_assy = lm["primary_assembly_kg"]
    W, Hh = spec["leaf"]["width"], spec["leaf"]["height"]
    phys = {"mass": lm}
    kin = spec["kinematics"]["type"]
    if kin.startswith("hinge") or kin == "rotor":
        # a rotor's thrust bearing carries every wing at once; a leaf hinge carries its own leaf
        hf = hinge_friction(spec, m_assy if kin == "rotor" else m)
        phys["hinge"] = hf
        phys["hinge"]["air_damping_Nms_per_rad"] = air_damping(W, Hh) if kin != "rotor" else 0.5 * air_damping(W, Hh) * lm["leaves_on_primary"]
        phys["closer"] = closer_params(spec, m, hf["coulomb_torque_Nm"] + 0.5 * hf["stick_torque_Nm"])
        phys["hinge"]["total_damping_symmetric"] = phys["hinge"]["air_damping_Nms_per_rad"] + (
            phys["closer"]["damping_opening"] if phys["closer"]["kind"] != "none" else 0.0)
        # moment of inertia about the primary axis (a rotor's is every wing's; a leaf's is its own)
        phys["inertia_about_hinge_kg_m2"] = m_assy * W * W / 3 + m_assy * spec["leaf"]["thickness"] ** 2 / 12
    else:
        phys["roller"] = roller_friction(spec, m_assy)   # the rollers carry what hangs on the track
        phys["closer"] = closer_params(spec, m) if spec["closer"]["model"] != "none" else {"model": "none", "kind": "none", "spring_stiffness_Nm_per_rad": 0.0, "spring_preload_Nm": 0.0, "damping_closing": 0.0, "damping_opening": 0.0}
        phys["hinge"] = {"coulomb_torque_Nm": 0.0, "stick_torque_Nm": 0.0, "air_damping_Nms_per_rad": 0.0, "total_damping_symmetric": 0.0}
    if "roller" not in phys and spec["kinematics"].get("roller"):
        phys["roller"] = roller_friction(spec, m_assy)
    phys["latch"] = latch_params(spec)
    op_ = H.OPERATORS[spec["operator"]["model"]]
    phys["operator"] = operator_dynamics(op_, preload_override=(max(op_.spring_torque_preload, 1.5) if op_.kind == "paddle" else None))
    phys["lock"] = lock_params(spec)
    phys["damage"] = damage_thresholds(spec, m)
    phys["compliance"] = compliance(spec, phys) if (kin.startswith("hinge") and spec["family"] not in ("pet_door", "hatch_floor", "hatch_ceiling")) else {}
    phys["gravity"] = G
    return phys

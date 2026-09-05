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


def leaf_mass(spec: dict) -> dict:
    """Mass breakdown of one leaf: slab + glazing + hardware."""
    leaf = spec["leaf"]
    slab = M.SLABS[leaf["slab"]]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    area = W * Hh
    glaz = leaf.get("glazing") or {}
    glass_area = float(glaz.get("area_fraction", 0.0)) * area
    slab_area = area - glass_area
    if slab.monolithic:
        ad = slab.area_density(t)
    else:
        ad = slab.area_density(t)
    slab_mass = ad * slab_area
    fam = spec.get("family", "")
    if fam == "turnstile_tripod":
        # 3 arms: 38 mm OD x 1.5 mm wall stainless tube, 0.5 m + hub
        slab_mass = 3 * (math.pi * (0.019 ** 2 - 0.0175 ** 2) * W * 7900) + 3.0
    elif fam == "turnstile_fullheight":
        arms = leaf.get("arms_per_wing", 8)
        slab_mass = arms * (math.pi * (0.019 ** 2 - 0.0175 ** 2) * W * 7900) + 4.0  # per wing incl. share of rotor column
    elif fam == "strip_curtain":
        slab_mass = W * Hh * t * M.MATERIALS["pvc_flexible"].density
    glass_mass = 0.0
    if glass_area > 0:
        gm = M.MATERIALS[glaz.get("material", "glass_clear")]
        gt = float(glaz.get("thickness", 0.006))
        glass_mass = glass_area * gt * gm.density
    # louvers: open area reduces mass (already in fill fraction for louver slab)
    hw = 0.0
    parts = {}
    op = H.OPERATORS[spec["operator"]["model"]]
    parts["operator"] = op.mass
    lk = H.LOCKS[spec["lock"]["model"]]
    parts["lock"] = lk.mass
    cl = H.CLOSERS[spec["closer"]["model"]]
    if cl.mounts_on in ("door_push", "door_pull"):
        parts["closer"] = cl.mass
    hg = H.HINGES[spec["hinge"]["model"]]
    parts["hinges_half"] = 0.5 * hg.mass_each * spec["hinge"]["count"]
    for e in spec.get("extras", []):
        parts[e] = EXTRA_MASS.get(e, 0.0)
    hw = sum(parts.values())
    total = slab_mass + glass_mass + hw
    return {
        "slab_kg": slab_mass, "slab_area_density_kg_m2": ad, "glass_kg": glass_mass, "hardware_kg": hw,
        "hardware_parts": parts, "total_kg": total,
        "formula": "slab_area_density(t) * (W*H - A_glass) + rho_glass * t_glass * A_glass + hardware",
        "source": slab.source,
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
    lm = leaf_mass(spec)
    m = lm["total_kg"]
    W, Hh = spec["leaf"]["width"], spec["leaf"]["height"]
    phys = {"mass": lm}
    kin = spec["kinematics"]["type"]
    if kin.startswith("hinge") or kin == "rotor":
        hf = hinge_friction(spec, m)
        phys["hinge"] = hf
        phys["hinge"]["air_damping_Nms_per_rad"] = air_damping(W, Hh) if kin != "rotor" else 0.5 * air_damping(W, Hh)
        phys["closer"] = closer_params(spec, m, hf["coulomb_torque_Nm"] + 0.5 * hf["stick_torque_Nm"])
        phys["hinge"]["total_damping_symmetric"] = phys["hinge"]["air_damping_Nms_per_rad"] + (
            phys["closer"]["damping_opening"] if phys["closer"]["kind"] != "none" else 0.0)
        # moment of inertia about hinge line for reporting
        phys["inertia_about_hinge_kg_m2"] = m * W * W / 3 + m * spec["leaf"]["thickness"] ** 2 / 12
    else:
        phys["roller"] = roller_friction(spec, m)
        phys["closer"] = closer_params(spec, m) if spec["closer"]["model"] != "none" else {"model": "none", "kind": "none", "spring_stiffness_Nm_per_rad": 0.0, "spring_preload_Nm": 0.0, "damping_closing": 0.0, "damping_opening": 0.0}
        phys["hinge"] = {"coulomb_torque_Nm": 0.0, "stick_torque_Nm": 0.0, "air_damping_Nms_per_rad": 0.0, "total_damping_symmetric": 0.0}
    if "roller" not in phys and spec["kinematics"].get("roller"):
        phys["roller"] = roller_friction(spec, m)
    phys["latch"] = latch_params(spec)
    phys["lock"] = lock_params(spec)
    phys["damage"] = damage_thresholds(spec, m)
    phys["compliance"] = compliance(spec, phys) if (kin.startswith("hinge") and spec["family"] not in ("pet_door", "hatch_floor", "hatch_ceiling")) else {}
    phys["gravity"] = G
    return phys

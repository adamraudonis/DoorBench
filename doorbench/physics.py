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
    """Declared leaf/wing/strip count, not the number of simulated bodies."""
    return max(1,int(spec['leaf'].get('count',1) or 1))


def leaves_on_primary(spec: dict) -> float:
    """Compatibility estimate in declared leaf units; build measures actual mass."""
    n=leaf_count(spec)
    if spec['kinematics']['type']=='rotor':return float(n)
    if spec['kinematics'].get('fold'):
        from .folding import fold_groups
        return n/fold_groups(n,bool(spec['kinematics'].get('accordion')))
    if spec['family']=='dutch':return .5
    return 1.


def one_leaf_material(spec: dict) -> dict:
    """First physical material panel/wing, excluding its installed hardware.

    Unequal panels, strip segments and prepared gaps are explicit in per_body;
    multiplying this representative value by leaf.count is not an assembly budget.
    """
    row=leaf_mass(spec)['per_body'][0]
    units=row.get('material_units',1)
    return {'slab_kg':row['slab_kg']/units,'glass_kg':row['glass_kg']/units,
            'width':row['width'],'height':row['height'],
            'scope':'first physical material panel or rotor wing; before geometry stock preparation'}


def door_hardware(spec: dict) -> dict:
    """Installed per-panel allowances before geometry-backed mechanism replacement."""
    mass=leaf_mass(spec)
    return {'parts':mass['hardware_parts'],'total_kg':mass['hardware_kg'],
            'scope':'Installed allowances; build replaces explicit mechanisms with their geometry BOM'}


def mass_budget(spec: dict) -> dict:
    """Assembly budget with explicit per-body ownership, before stock preparation.

    The compatibility single-leaf fields describe the first physical panel.
    Use per_body for unequal leaves, folded panels, segments and shared rotors.
    """
    mass=leaf_mass(spec)
    row=mass['per_body'][0];units=row.get('material_units',1)
    material=one_leaf_material(spec)
    hardware=row['hardware_kg']/units
    return {**mass,'leaf_count':leaf_count(spec),
            'leaf_slab_kg':material['slab_kg'],'leaf_glass_kg':material['glass_kg'],
            'leaf_hardware_kg':hardware,'per_leaf_kg':material['slab_kg']+material['glass_kg']+hardware,
            'leaves_on_primary':leaves_on_primary(spec),
            'primary_assembly_kg':mass['dynamics_mass_kg'],
            'single_leaf_fields_scope':material['scope']}


def _unit_leaf_mass(spec: dict) -> dict:
    """One declared panel, before physical assembly/ownership expansion."""
    leaf = spec["leaf"]
    slab = M.SLABS[leaf["slab"]]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    area = W * Hh
    glaz = leaf.get("glazing") or {}
    glass_area = float(glaz.get("area_fraction", 0.0)) * area
    slab_area = area - glass_area
    if leaf["slab"] == "chain_link_gate" and leaf.get("infill_thickness") is not None:
        # Structural tube diameter must not be interpreted as a solid sheet.
        # 1.6 mm galvanized tube wall; mesh area mass and perimeter tube mass
        # are separate contributions, including the infill-free edge band.
        radius, wall = t / 2, 0.0016
        tube_per_m = math.pi * (radius ** 2 - (radius - wall) ** 2) * M.MATERIALS["steel_galvanized"].density
        ad = (2 * (W + Hh) * tube_per_m + 2.4 * max(0, W - 2*t) * max(0, Hh - 2*t)) / area
    elif slab.monolithic:
        ad = slab.area_density(t)
    else:
        ad = slab.area_density(t)
    slab_mass = ad * slab_area
    fam = spec.get("family", "")
    if fam == "turnstile_tripod":
        # Drop arms begin at the physical hinge 65 mm from the rotor axis.
        # Their material transfers to moving tube bodies; it is not added twice.
        arm_length = W - .065 if spec['kinematics'].get('drop_arm') else W
        slab_mass = 3 * (math.pi * (0.019 ** 2 - 0.0175 ** 2) * arm_length * 7900) + 3.0
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
    if leaf["slab"] in M.FRAMED_GLASS_SLABS:
        profile = M.framed_glass_profile(leaf["slab"], W, Hh, t)
        slab_mass = profile["frame_kg"] + profile["gasket_kg"]
        glass_mass = profile["glass_kg"]
        ad = slab_mass / area
    from .glazing import uses_ordinary_glazing, construction
    glazing_profile = None
    if uses_ordinary_glazing(leaf):
        # The sectional builder leaves a real 6 mm articulation gap in each
        # panel. Rising hinges trim the slab at its actual lifted envelope.
        height=Hh-(.006 if fam=='garage_sectional' else 0.)
        if fam in ('swing_single','swing_double','automatic_swing','pivot','cold_storage') and not spec['opening'].get('outdoor'):
            opn=spec['opening'];zb=leaf.get('bottom_clearance',.012) or .012
            if opn.get('ground_clearance'):zb=opn['ground_clearance']
            if opn.get('threshold') in ('saddle','sill','sill_step'):
                zb=max(zb,.045 if opn['threshold']=='sill_step' else .017)
            else:zb=max(.005,min(zb,opn['height']-Hh-.004))
            if H.OPERATORS[spec['operator']['model']].kind=='cremone' or H.LATCHES[spec['latch']['model']].kind=='vertical_rods':
                from .construction_dimensions import FLOOR_STRIKE_TOP_M
                zb=max(zb,FLOOR_STRIKE_TOP_M.get(opn.get('threshold','none'),0.)+.016)
            height=min(height,opn['height']-.004-zb)
        if spec['hinge'].get('axis_tilt_deg') and fam in ('swing_single','swing_double','automatic_swing','pivot','cold_storage'):
            height-=1.3*{'rising_butt':.008,'cam_lift':.012,'gravity_pivot':.010}.get(H.HINGES[spec['hinge']['model']].kind,0.)
        glazing_profile=construction(leaf,W,height,spec=spec)
        slab_mass,glass_mass=glazing_profile['slab_kg'],glazing_profile['glass_kg']
    # louvers: open area reduces mass (already in fill fraction for louver slab)
    hw = 0.0
    parts = {}
    if glazing_profile:
        parts['glazing_retainers']=glazing_profile['retainer_kg']
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
    formula = "slab_area_density(t) * (W*H - A_glass) + rho_glass * t_glass * A_glass + hardware"
    if leaf["slab"] in M.FRAMED_GLASS_SLABS:
        formula = "hollow frame-wall volumes*rho_frame + actual glass-ply volumes*rho_glass + edge gaskets + hardware"
    elif glazing_profile:
        formula = "retained stock volume*effective slab density + true pane volume*glass density + separate stop/tape BOM + hardware"
    elif fam == "turnstile_tripod":
        formula = "3*pi*(0.019^2-0.0175^2)*arm_length*7900 + 3 kg hub + shared hardware (three arms already included)"
    elif fam == "turnstile_fullheight":
        formula = "arms_per_wing*pi*(0.019^2-0.0175^2)*arm_length*7900 + 4 kg column allocation per wing + shared hardware"
    return {
        "slab_kg": slab_mass, "slab_area_density_kg_m2": ad, "glass_kg": glass_mass, "hardware_kg": hw,
        "hardware_parts": parts, "total_kg": total,
        "formula": formula,
        "source": slab.source,
    }


def leaf_mass(spec: dict) -> dict:
    """Assembly mass plus explicit physical panel and driven-unit budgets.

    A material unit may be a panel, a full hatch or one rotor wing. Hardware is
    assigned by its installation scope rather than multiplying every catalogue
    entry by a generic count. Geometry reconciliation consumes per_body.
    """
    from .mass_layout import mass_panels
    reference = _unit_leaf_mass(spec)
    rows = []
    for panel in mass_panels(spec):
        sub = {**spec, "leaf": {**spec["leaf"], "width":panel["width"], "height":panel["height"]}}
        if panel.get("embedded_slab"):
            sub = {**sub, "leaf":{**sub["leaf"],"slab":panel["embedded_slab"],"thickness":panel["thickness"],"glazing":None},
                   "operator":{**spec["operator"],"model":"none"},"lock":{**spec["lock"],"model":"none"},
                   "closer":{**spec["closer"],"model":"none"},"hinge":{**spec["hinge"],"count":0},"extras":[]}
        if panel.get("inactive"):
            sub = {**sub, "operator":{**spec["operator"],"model":"none"}, "latch":{"model":"none"}, "lock":{**spec["lock"],"model":"none"}, "closer":{**spec["closer"],"model":"none"}, "extras":[e for e in spec.get("extras",[]) if e=="kick_plate"]}
        elif panel.get("secondary_pair"):
            sub = {**sub, "lock":{**spec["lock"],"model":"none"}}
        unit = _unit_leaf_mass(sub)
        units = panel.get("material_units",1)
        parts = {k:v*panel.get("hardware_fraction",1) for k,v in unit["hardware_parts"].items()}
        if 'glazing_retainers' in parts:
            # Retainers are material for this exact physical panel, not a
            # shared operator allowance split between Dutch halves/sections.
            parts['glazing_retainers']=unit['hardware_parts']['glazing_retainers']*units
        if "operator_fraction" in panel:
            parts["operator"] = unit["hardware_parts"]["operator"]*panel["operator_fraction"]
        replaced_operator = 0.
        replaced_rotary_bearing = 0.
        if spec['family']=='ship_watertight':
            # Every marine dog/shaft/handle and wheel transmission is now an
            # explicit geometry-backed mechanism. Do not also hide the old
            # single-operator allowance in the leaf's residual mass. Bearings,
            # locks and hinge allowances are otherwise unchanged.
            replaced_operator = parts.get('operator', 0.)
            parts['operator'] = 0.
        if spec['family']=='turnstile_tripod' and spec['kinematics'].get('drop_arm'):
            # The extended rotating steel journal is now its own exact BOM
            # body. Fixed bearing housings are outside the moving assembly.
            replaced_rotary_bearing = parts.get('hinges_half', 0.)
            parts['hinges_half'] = 0.
        if spec["kinematics"].get("track")=="top_hung_pocket":
            parts["edge_pull_fitting"] = .20  # original 98 mm case/rocker, separate from face grip
        slab, glass = unit["slab_kg"]*units, unit["glass_kg"]*units
        if panel.get("cutout_area_m2"):
            from .glazing import uses_ordinary_glazing
            if uses_ordinary_glazing(sub['leaf']):
                # The construction raises its glazing above the pet opening;
                # remove actual solid rail stock, never a fraction of glass.
                slab-=panel['cutout_area_m2']*M.SLABS[sub['leaf']['slab']].area_density(sub['leaf']['thickness'])
                if slab<=0:raise ValueError('Pet preparation exceeds retained glazed-door stock')
            else:
                retained=max(0.,1-panel["cutout_area_m2"]/(panel["width"]*panel["height"]))
                slab*=retained;glass*=retained
        hardware = sum(parts.values())
        rows.append({**panel,"slab_kg":slab,"glass_kg":glass,"hardware_kg":hardware,"hardware_parts":parts,"total_kg":slab+glass+hardware})
        if replaced_operator:
            rows[-1]['catalogue_operator_replaced_kg']=replaced_operator
            rows[-1]['operator_mass_source']='Explicit marine operator/shaft/bearing/transmission geometry BOM'
        if replaced_rotary_bearing:
            rows[-1]['catalogue_rotary_bearing_replaced_kg']=replaced_rotary_bearing
            rows[-1]['rotary_bearing_mass_source']='Explicit steel journal geometry BOM; fixed bearing housing excluded from moving mass'
    if spec['family']=='strip_curtain':
        for row in rows:
            row['carried_mass_kg']=sum(p['total_kg'] for p in rows if p['strip_index']==row['strip_index'] and p['segment_index']>=row['segment_index'])
    slab=sum(p["slab_kg"] for p in rows);glass=sum(p["glass_kg"] for p in rows);hardware=sum(p["hardware_kg"] for p in rows)
    parts={}
    for row in rows:
        for name,value in row["hardware_parts"].items():parts[name]=parts.get(name,0.)+value
    return {"leaf_count":max(1,int(spec["leaf"].get("count",1))),"leaf_slab_kg":reference["slab_kg"],"leaf_glass_kg":reference["glass_kg"],"per_leaf_kg":reference["total_kg"],"slab_kg":slab,"glass_kg":glass,"hardware_kg":hardware,"hardware_parts":parts,"total_kg":slab+glass+hardware,"slab_area_density_kg_m2":reference["slab_area_density_kg_m2"],"reference_unit":reference,"per_body":rows,"scope":"moving_assembly","dynamics_mass_kg":rows[0].get('carried_mass_kg',rows[0]["total_kg"]),"dynamics_mass_scope":"primary carried panel or complete strip chain; rotor/lift records contain their complete moving assembly","formula":"sum(per_body material mass + installed hardware budget); panel dimensions and count follow each family construction","source":reference["source"]}


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
        if spec['family'] in ('hatch_floor', 'hatch_ceiling'):
            out.update({'spring_stiffness_Nm_per_rad': 0., 'spring_preload_Nm': 0.,
                        'damping_closing': 0., 'damping_opening': 0., 'axial_force_N': F,
                        'formula': 'Native axial spring between frame and lid pivots; moment varies with linkage geometry.',
                        'source': 'Original telescopic gas-spring model; inspect model.meta.hatch_support for anchors and force law.'})
            return out
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
    """Limited analytical estimates; legacy compliance booleans are unassessed.

    A spring/friction calculation is not a continuous measured opening-force
    curve, a clear-width survey or a regulatory assessment.
    """
    W = spec["leaf"]["width"]
    op = H.OPERATORS[spec["operator"]["model"]]
    arm = max(W - (H.LATCHES[spec["latch"]["model"]].backset or 0.06) - 0.02, 0.3)
    tau_static = phys["hinge"]["coulomb_torque_Nm"] + phys["closer"]["spring_preload_Nm"] + phys["hinge"]["stick_torque_Nm"]
    tau_90 = phys["hinge"]["coulomb_torque_Nm"] + phys["closer"]["spring_preload_Nm"] + phys["closer"]["spring_stiffness_Nm_per_rad"] * math.pi / 2
    F_start = tau_static / arm
    F_90 = tau_90 / arm
    hinge = H.HINGES[spec['hinge']['model']]
    simple_hinge = spec['family'] in ('swing_single','swing_double','dutch','saloon') and not spec['hinge'].get('axis_tilt_deg') and not hinge.axis_tilt_deg and hinge.kind not in ('rising_butt','cam_lift','gravity_pivot')
    if not simple_hinge:
        F_start = F_90 = None
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
    return {
        "opening_force_start_N": F_start, "opening_force_90deg_N": F_90, "operator_force_N": op_force,
        "ada_interior_5lbf_ok": None, "ibc_fire_exterior_ok": None,
        "hardware_operable_5lbf_ok": None, "panic_unlatch_15lbf_ok": None,
        "lever_or_ada_hardware": None, "handle_height_ada_ok": None, "clear_width_ada_ok": None,
        "opening_force_model": "single_vertical_hinge_friction_and_spring" if simple_hinge else "requires_native_mechanism_force_curve",
        "assessment": "not_assessed",
        "notes": "Analytical spring/friction estimates after latch release, excluding contact friction and pressure loads. Nonlinear lifting, folding, sliding and coupled mechanisms need native force curves. Regulatory compliance is not assessed.",
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


def derive(spec: dict) -> dict:
    """Assembly report with per-carried-panel joint/closer dynamics."""
    lm = leaf_mass(spec)
    result = _derive_for_mass(spec, lm)
    result["per_body_dynamics"] = {}
    for panel in lm["per_body"]:
        sub = {**spec, "leaf":{**spec["leaf"],"width":panel["width"],"height":panel["height"]}}
        if panel.get("inactive") or panel.get("embedded_slab"):
            sub = {**sub,"closer":{**spec["closer"],"model":"none"},"operator":{**spec["operator"],"model":"none"},"lock":{**spec["lock"],"model":"none"}}
        elif panel.get("secondary_pair"):
            sub = {**sub,"lock":{**spec["lock"],"model":"none"}}
        local = {**panel,"dynamics_mass_kg":panel.get('carried_mass_kg',panel["total_kg"])}
        result["per_body_dynamics"][panel["body"]] = _derive_for_mass(sub, local)
    return result


def _derive_for_mass(spec: dict, lm: dict) -> dict:
    m = lm["dynamics_mass_kg"]
    W, Hh = spec["leaf"]["width"], spec["leaf"]["height"]
    phys = {"mass": lm, "operator": operator_dynamics(H.OPERATORS[spec['operator']['model']])}
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

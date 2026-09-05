"""Hinged (vertical-axis) door families -> IR Model."""
from __future__ import annotations

import math

import numpy as np

from ..ir import (Body, Geom, Joint, Site, Equality, Tendon, Model, ALL_TIERS, FULL_ONLY, FULL_SIMPLE, QUAT_ID,
                  quat_from_axis_angle, quat_z_to)
from .. import materials as M
from .. import hardware as H
from . import common as C
from . import meshes as MESH


def _uv(spec):
    side = spec["hinge"]["side"]
    u = 1.0 if side in ("left", "center", "bottom", "top", "far") else -1.0
    v = 1.0 if spec["robot"]["is_push"] else -1.0
    return u, v


def hinge_joint(spec, phys, u, v, y_pin, name="door_hinge", both_ways=False, max_open_deg=None, tilt_deg=0.0, axis_override=None):
    kin = spec["kinematics"]
    mo = math.radians(max_open_deg if max_open_deg is not None else (kin.get("max_open_deg") or 90))
    hf = phys["hinge"]
    cl = phys["closer"]
    axis = np.array([0.0, 0.0, u * v]) if axis_override is None else np.asarray(axis_override, float)
    if tilt_deg:
        # rising-butt / cam-lift / gravity hinge: tilt the axis so the leaf climbs as it opens (gravity closes it).
        a = math.radians(tilt_deg)
        W = spec["leaf"]["width"]
        best = None
        for s_ in (1.0, -1.0):
            cand = np.array([0.0, s_ * math.sin(a), math.cos(a)]) * (u * v)
            cand = cand / np.linalg.norm(cand)
            com = np.array([u * W / 2, 0.0, 0.0])
            from ..ir import quat_from_axis_angle as _qa, quat_rotate as _qr
            z1 = _qr(_qa(cand, 0.3), com)[2]
            if best is None or z1 > best[0]:
                best = (z1, cand)
        axis = best[1]
    rng = (-mo, mo) if both_ways else (0.0, mo)
    stiffness = cl.get("spring_stiffness_Nm_per_rad", 0.0) or 0.0
    preload = cl.get("spring_preload_Nm", 0.0) or 0.0
    springref = -preload / stiffness if stiffness > 1e-9 else 0.0
    if both_ways:
        springref = 0.0
    damping = hf.get("total_damping_symmetric", 0.0) + (0.05 if stiffness == 0 else 0.0)
    j = Joint(name, "hinge", tuple(axis), (0.0, y_pin, 0.0), rng, damping=damping, frictionloss=hf.get("coulomb_torque_Nm", 0.0) + hf.get("stick_torque_Nm", 0.0) * 0.5,
              stiffness=stiffness, springref=springref, armature=0.01, role="primary", label="Door hinge (0 = closed, + = opening)",
              damping_closing=cl.get("damping_closing"), damping_opening=cl.get("damping_opening"), backcheck_angle=cl.get("backcheck_angle_rad"), backcheck_damping=cl.get("backcheck_damping"))
    if H.SEALS[spec["seal"]]["compression_m"] > 0:
        j.limit_solref = (0.02, 1.0)
    return j


def _lock_state(spec):
    lk = H.LOCKS[spec["lock"]["model"]]
    engaged = bool(spec["lock"].get("engaged"))
    release = bool(spec["lock"].get("robot_side_release"))
    return lk, engaged, release


def _add_door_hasp(model, world, leaf_body, spec, u, hx, x_edge, t, z_h, locked, leaf_name, Wo, material="steel_galvanized", v=1.0):
    """Hasp & staple (+ padlock when locked) across the leaf / post (or jamb) joint on the OUTSIDE face.  Posts and
    jambs stand proud of the leaf face, so the hasp hinge plate is a packing block that brings the strap plane 4 mm
    clear of the post / jamb face (and of the stop moulding on the stop side) where the staple is screwed on."""
    op = spec["opening"]
    outdoor = bool(op.get("outdoor"))
    face = -1.0 if (outdoor or spec["robot"]["robot_outside"]) else 1.0
    is_gate = op["frame"]["kind"] in ("gate_posts", "pressure_frame")
    jt = C.frame_jamb_thickness(spec)
    sx_w = u * (Wo / 2)
    if is_gate:
        y_surf = face * jt / 2
        x_st = sx_w + u * min(0.028, jt * 0.5)
    else:
        depth = op["wall_thickness"] if op["frame"]["kind"] != "aluminum_storefront" else max(0.114, op["wall_thickness"])
        y_surf = float(model.meta.get("wall_y", 0.0)) + face * depth / 2
        x_st = sx_w + u * min(0.028, max(0.012, jt * 0.6))
    plane_h = max(0.006, abs(y_surf) - t / 2 + 0.004)
    if not is_gate and face == -v and op["frame"].get("stop_depth", 0) > 0:
        plane_h = max(plane_h, 0.032 + 0.006)          # strap clears the 32 mm stop moulding on the stop side
    hinge_x = x_edge - u * 0.055
    strap_len = abs(x_st - (hx + hinge_x)) + 0.022
    y_eye = face * (t / 2 + plane_h + 0.003 + 0.008)
    mat = C.mat_from_material(model, material, f"mat_op_{material}")
    pm = C.mat_from_material(model, "brass", "mat_padlock")
    C.add_hasp_assembly(model, leaf_body, world, leaf_name, (hinge_x, face * t / 2, z_h), (0, face, 0), (u, 0, 0), strap_len, plane_h, (x_st, y_eye, z_h), (x_st, y_surf, z_h), (0, face, 0), locked, mat, pm)


def build_swing_single(spec, phys, model: Model, leaf_name="leaf", pair=None):
    """Single hinged leaf (also used per-leaf for pairs via `pair`)."""
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    u, v = _uv(spec)
    if pair:
        u = pair["u"]
        v = pair["v"]
    zb = leaf.get("bottom_clearance", C.BOTTOM_CLEAR) or C.BOTTOM_CLEAR
    if op.get("ground_clearance"):
        zb = op["ground_clearance"]
    # keep >= 4 mm head clearance (real doors: ~3 mm at the head, 10-19 mm at the floor)
    if not op.get("outdoor") and spec["family"] not in ("stall", "baby_gate"):
        if op.get("threshold") in ("saddle", "sill", "sill_step"):
            zb = max(zb, 0.017 if op.get("threshold") != "sill_step" else 0.045)
        else:
            zb = max(0.005, min(zb, Ho - Hh - 0.004))
        if zb + Hh > Ho - 0.004:
            Hh = Ho - 0.004 - zb
            leaf = {**leaf, "height": Hh}
    outdoor = bool(op.get("outdoor"))
    jt_ = C.frame_jamb_thickness(spec)
    # the leaf hangs with its swing-side face C.LEAF_FACE_INSET behind the jamb's swing-side face, so the hinge
    # knuckle lands on that face and the leaf can swing back past 90 deg without the reveal arris fouling it
    depth_ = op["wall_thickness"] if op["frame"]["kind"] != "aluminum_storefront" else max(0.114, op["wall_thickness"])
    y_wall = -v * max(0.0, depth_ / 2 - t / 2 - C.LEAF_FACE_INSET) if not outdoor else 0.0
    hole_ = C.frame_hole(spec, u, jt_)
    world = pair["world"] if pair else C.add_floor_and_wall(model, spec, outdoor=outdoor, hole=hole_, y_wall=y_wall)
    fam = spec["family"]
    # --- hinge position
    hg = H.HINGES[spec["hinge"]["model"]]
    if hg.kind in ("pivot_center", "pivot_center_heavy") and spec["hinge"].get("pivot_offset_m"):
        x_axis_rel = u * spec["hinge"]["pivot_offset_m"]   # axis inside the leaf
        y_pin = 0.0
    elif hg.kind == "pivot_offset":
        x_axis_rel = u * 0.019
        y_pin = v * C.hinge_throw(t, depth_, y_wall, v, W, knuckle=0.010)
    elif hg.kind in ("gravity_pivot",):
        x_axis_rel = u * 0.02
        y_pin = 0.0
    else:
        # butt hinge: pin at the door edge line, its knuckle proud of the frame's swing-side face (see hinge_throw)
        x_axis_rel = u * C.GAP
        y_pin = v * C.hinge_throw(t, depth_, y_wall, v, W)
    hx = pair["hx"] if pair else u * (-Wo / 2)      # hinge jamb inner face x (world)
    x_leaf0 = u * C.GAP                                # leaf hinge edge in body frame (body origin at jamb face)
    if hg.kind in ("pivot_center", "pivot_center_heavy"):
        x_leaf0 = u * 0.006
    if abs(y_pin) < 1e-9:
        # centre-hung pivot (the axis lies in the leaf's own centre plane): the heel corner sweeps a circle about it,
        # so the heel gap has to be solved from the pivot setback, not fixed (see C.pivot_heel_gap)
        x_leaf0 = u * max(abs(x_leaf0), C.pivot_heel_gap(abs(x_axis_rel), t))
    both_ways = bool(spec["kinematics"].get("both_ways"))
    if both_ways:
        # double-acting: centre-hung pivot at the leaf edge, gap t/2 + 6 mm so the edge corners clear the post
        x_leaf0 = u * (t / 2 + 0.006)
        x_axis_rel = x_leaf0
        y_pin = 0.0
    tilt = spec["hinge"].get("axis_tilt_deg", 0.0) or 0.0
    rise_per_90 = {"rising_butt": 0.008, "cam_lift": 0.012, "gravity_pivot": 0.010}.get(hg.kind, 0.0) if tilt else 0.0
    leaf_parent = None
    if rise_per_90 > 0:
        # helical (rising) hinge = hinge + coupled vertical slide (screw joint); gravity closes the door with m*g*pitch
        riser = Body(f"{leaf_name}_riser", None, (hx, 0.0, 0.0), QUAT_ID, None, [], [], ALL_TIERS, "hinge", "Rising hinge carrier")
        riser.joint = Joint(f"{leaf_name}_rise", "slide", (0, 0, 1), (0, 0, 0), (0.0, rise_per_90 * 2.2), damping=1.0, frictionloss=0.0, armature=0.5, role="mechanism", label="Rising-hinge lift (coupled to hinge angle)", robot_interactive=False)
        # the carrier collar sits ON the hinge jamb it rides (a 4 mm bead floated 3.6 mm off it)
        riser.geoms.append(C.cyl(f"{leaf_name}_riser_marker", (x_axis_rel, y_pin, 0.055), 0.014, 0.018, C.mat_from_material(model, "steel", "mat_hinge"), (0, 0, 1), 7850, False, True, FULL_ONLY, "hinge", "Rising hinge carrier collar"))
        model.add_body(riser)
        leaf_parent = riser.name
    leaf_body = Body(leaf_name, leaf_parent, (hx, 0.0, 0.0) if leaf_parent is None else (0.0, 0.0, 0.0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Door leaf")
    j = hinge_joint(spec, phys, u, v, y_pin, name=f"{leaf_name}_hinge", both_ways=both_ways, tilt_deg=0.0)
    j.pos = (x_axis_rel, y_pin, 0.0)
    if rise_per_90 > 0:
        model.equalities.append(Equality("joint", f"{leaf_name}_rise_couple", f"{leaf_name}_rise", f"{leaf_name}_hinge", (0.0, rise_per_90 / (math.pi / 2), 0, 0, 0), tiers=ALL_TIERS, label=f"rise = {rise_per_90 * 1000:.0f} mm per 90 deg (helical hinge)"))
        j.notes = (j.notes + " " if j.notes else "") + f"rising hinge: gravity closing torque ~ m*g*{rise_per_90 / (math.pi / 2):.4f} N*m/rad"
    lk, engaged, release = _lock_state(spec)
    # chain / swing bar: limit opening
    if engaged and lk.kind in ("chain", "swing_bar_guard"):
        th = math.asin(min(0.99, lk.chain_slack / max(W - 0.1, 0.2)))
        j.range = (0.0, th)
        j.notes = f"{lk.name}: opening limited to {math.degrees(th):.1f} deg"
    if engaged and lk.kind in ("padlock", "jam_stuck") and not release:
        if lk.kind == "padlock":
            j.range = (0.0, 0.0015)
            j.notes = "Padlocked hasp: leaf effectively fixed (2 mm rattle)"
        else:
            j.frictionloss += phys["hinge"].get("stick_torque_Nm", 0.0) * 2
            j.notes = "Stuck/swollen door: high breakaway friction"
    if spec["kinematics"].get("stop") == "wedge_jammed":
        j.range = (0.0, math.radians(2))
    leaf_body.joint = j
    # rest angle for gravity-pivot stall doors
    if spec["kinematics"].get("rest_angle_deg"):
        j.initial = math.radians(spec["kinematics"]["rest_angle_deg"])
    model.add_body(leaf_body)
    # --- leaf slab & decoration (rising hinges: leaf trimmed so it clears the head when lifted)
    leaf_geom = leaf
    if rise_per_90 > 0:
        Hh = Hh - rise_per_90 * 1.3
        leaf_geom = {**leaf, "height": Hh}
    pf_ = spec["leaf"].get("pet_flap")
    hole_ = (x_leaf0 + u * W / 2, zb + 0.05 + pf_["height"] / 2, pf_["width"] + 0.002, pf_["height"] + 0.002) if pf_ else None
    C.add_leaf_geoms(model, leaf_body, spec, leaf_geom, u, x_leaf0, zb, phys, name_prefix=leaf_name, edge_pockets=(pair or {}).get("edge_pockets") if (pair and leaf_name == "leaf_b") else None, v_edge=v, hole=hole_)
    C.add_hinge_visuals(model, world, leaf_body, spec, (x_axis_rel, y_pin), Hh, zb, v, u)
    x_edge = x_leaf0 + u * W                           # latch edge (body frame)
    hz = spec["operator"]["height"] or Hh * 0.5
    hz = min(max(hz, 0.3), Hh - 0.2)
    opm = H.OPERATORS[spec["operator"]["model"]]
    lt = H.LATCHES[spec["latch"]["model"]]
    pockets = []
    handle_joint = None
    coupling = phys["latch"]["coupling"]
    scale = coupling["bolt_q = c0 + c1*op_q"][1] * -1 if coupling else 0.0   # positive scale: bolt retract per operator unit
    faces, far_op = C.operator_faces(spec, v)
    head_pockets = []
    locked_backlash = None
    if engaged and not release and lk.kind in ("privacy_button", "keyed_cylinder", "keypad_code", "card_reader", "electric_strike", "mortise_deadbolt", "multipoint", "night_latch", "vault_wheel"):
        locked_backlash = phys["lock"]["handle_backlash_locked_rad"]
    # --- operators
    x_spindle = x_edge - u * (lt.backset if lt.backset > 0 else 0.065)
    outside_face = 1.0 if not spec["robot"]["robot_outside"] else -1.0
    keypad_face = outside_face if lk.kind == "keypad_code" else None     # keypad on the OUTSIDE face; thumbturn inside
    # lock trims on the operator: key cylinder outside (keyed lever / knob), turn button inside (privacy / keyed)
    cyl_face = outside_face if lk.kind == "keyed_cylinder" else None
    btn_face = -outside_face if lk.kind in ("privacy_button", "keyed_cylinder") else None
    if opm.kind in ("lever", "knob", "keypad_lever", "card_lever", "keypad_deadbolt", "paddle", "t_handle", "cremone", "wheel") and faces:
        hb = C.add_rotary_operator(model, leaf_body, spec, phys, opm, u, v, x_spindle, hz, t, faces, locked_backlash, name=f"{leaf_name}_handle", cylinder_face=cyl_face, button_face=btn_face, rim_case_face=(-outside_face if opm.style_params.get("rim_box") else None))
        handle_joint = hb.joint.name
        if opm.kind == "wheel":
            j.notes += " handwheel drives boltwork"
        if opm.kind == "cremone":
            # cremone bolt: the knob turns a pinion that drives a surface rod up to a shoot bolt in the head and a rod
            # down toward the sill (visual); rods and guides live on the swing-side face (no stop moulding there)
            f_r = v
            cm_ = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
            y_r = f_r * (t / 2 + 0.012)
            x_r = x_spindle - u * 0.045                       # rod beside the knob's backplate (clear of any thumbturn)
            z_top_r, z_bot_r = zb + Hh - 0.075, zb + 0.03
            leaf_body.geoms.append(C.cyl(f"{leaf_name}_cremone_rod_up", (x_r, y_r, (hz + 0.04 + z_top_r) / 2), 0.006, (z_top_r - hz - 0.04) / 2, cm_, (0, 0, 1), 7850, False, True, FULL_SIMPLE, "lock", "Cremone rod (up)"))
            leaf_body.geoms.append(C.cyl(f"{leaf_name}_cremone_rod_down", (x_r, y_r, (hz - 0.04 + z_bot_r) / 2), 0.006, (hz - 0.04 - z_bot_r) / 2, cm_, (0, 0, 1), 7850, False, True, FULL_SIMPLE, "lock", "Cremone rod (down)"))
            leaf_body.geoms.append(C.box(f"{leaf_name}_cremone_gearbox", (x_r, y_r, hz), (0.012, 0.010, 0.045), cm_, 7850, False, True, FULL_SIMPLE, "lock", "Cremone rod junction (driven by the knob's gearbox)"))
            for k_, zg in enumerate((hz + 0.25, z_top_r - 0.10, hz - 0.25, z_bot_r + 0.10)):
                C.add_guide_loop(leaf_body.geoms, f"{leaf_name}_cremone_guide_{k_}", (x_r, f_r * t / 2, zg), (0, 0, 1), (0, f_r, 0), 0.0, 0.009, 0.021, cm_, 0.004, 0.014, False, FULL_SIMPLE, "lock", "Rod guide")
            sbolt = Body(f"{leaf_name}_cremone_top_bolt", leaf_body.name, (x_r, y_r, zb + Hh), QUAT_ID, None, [], [], FULL_SIMPLE, "lock", "Cremone shoot bolt (top)")
            sbolt.joint = Joint(f"{leaf_name}_cremone_top_bolt_slide", "slide", (0, 0, -1), (0, 0, 0), (0.0, 0.02), damping=2.0, frictionloss=0.5, role="lock", label="Shoot bolt (0 = thrown into the head, + = retracted)", robot_interactive=False)
            sbolt.geoms.append(Geom(f"{leaf_name}_cremone_top_bolt_geom", "capsule", (0.006, 0.03), (0, 0, -0.016), (1, 0, 0, 0), cm_, True, True, 7850.0, None, (0.4, 0.005, 0.0001), None, None, False, None, None, 0.0, FULL_SIMPLE, "lock", "Shoot bolt"))
            model.add_body(sbolt)
            model.equalities.append(Equality("joint", f"{leaf_name}_cremone_couple", sbolt.joint.name, hb.joint.name, (0, 0.02 / max(opm.travel, 1e-6), 0, 0, 0), tiers=FULL_SIMPLE, label="shoot bolt = knob * 0.02/travel"))
            model.meta.setdefault("clearance_allow", []).append([f"{leaf_name}_cremone_rod_up", f"{leaf_name}_cremone_top_bolt_geom", "the shoot bolt is the end of the rod"])
            head_pockets.append({"x": hx + x_r, "hx": 0.010, "w": 0.015, "depth": 0.026, "y": y_r})
    if lk.kind == "keypad_code" and not pair:
        # keypad / pushbutton unit above the handle on the outside face, whatever the operator
        z_min = None
        if opm.style_params.get("escutcheon"):
            z_min = hz + opm.style_params["escutcheon"][0] / 2
        if opm.kind == "handleset":
            z_min = hz + 0.175 + 0.03
        C.add_keypad(model, leaf_body, spec, u, x_spindle, hz, t, keypad_face, mechanical=(lk.id == "keypad_mechanical"), keys=opm.style_params.get("keys", 10), name=f"{leaf_name}_keypad", z_min=z_min, z_max=zb + Hh - 0.03)
    elif opm.kind in ("pull", "flush_pull", "ring_pull", "push_plate", "handleset"):
        for f in (faces if opm.kind != "handleset" else [-1.0]):
            C.add_pull(model, leaf_body, opm, u, x_edge - u * 0.105, hz, t, f, name=f"{leaf_name}_{opm.kind}")
        if opm.kind == "handleset":
            # exterior grip + thumb latch on robot face; interior knob on +1 face is a rotary operator
            knob = H.OPERATORS["knob_round"]
            hb = C.add_rotary_operator(model, leaf_body, spec, phys, knob, u, v, x_spindle, hz, t, [1.0], locked_backlash, name=f"{leaf_name}_handle")
            handle_joint = hb.joint.name
            # thumb piece (robot side) drives the same latch: small body with hinge about x on the -1 face, sitting at
            # the top of the grip plate (the deadbolt cylinder is above it, as on a Kwikset / Schlage handleset)
            tp = Body(f"{leaf_name}_thumbpiece", leaf_body.name, (x_edge - u * 0.105, -1.0 * (t / 2 + 0.018), hz + 0.112), QUAT_ID, None, [], [], FULL_SIMPLE, "operator", "Thumb latch")
            tp.joint = Joint(f"{leaf_name}_thumbpiece_hinge", "hinge", (-1, 0, 0), (0, 0, 0.02), (0.0, opm.travel), damping=0.02, frictionloss=0.02, stiffness=opm.spring_rate, springref=-opm.spring_torque_preload / max(opm.spring_rate, 1e-6), role="operator", label="Thumb piece (press in)")
            tm = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
            tp.geoms.append(C.box(f"{leaf_name}_thumbpiece_geom", (0, 0, 0), (0.018, 0.004, 0.02), tm, 3000, True, True, FULL_SIMPLE, "operator", "Thumb piece"))
            tp.sites.append(Site(f"{leaf_name}_thumb_push", (0, -0.005, 0.01), QUAT_ID, 0.01, "push"))
            model.add_body(tp)
            if lt.throw > 0:
                model.tendons.append(Tendon(f"{leaf_name}_thumb_coupling", [(f"{leaf_name}_latch_bolt_slide", 1.0), (tp.joint.name, -lt.throw / max(opm.travel - opm.dead_travel, 1e-6))], (0.0, 10.0), tiers=FULL_SIMPLE, label="bolt_q >= scale*thumb_q"))
                model.tendons[-1].kind = "fixed"
    elif opm.kind in ("panic_touchbar", "panic_crossbar"):
        face = -v
        rim_in_case = lt.kind == "rim_latch" and opm.kind == "panic_touchbar" and not pair
        pad, handle_joint = C.add_touchbar(model, leaf_body, spec, opm, u, v, x_edge, x_leaf0, hz, t, W, face, name=f"{leaf_name}_exit_device", z_top=zb + Hh, z_bot=zb, case_end_gap=0.012 if rim_in_case else 0.03)
        if far_op and far_op != "none":
            fo = H.OPERATORS[far_op]
            if fo.kind in ("pull",):
                C.add_pull(model, leaf_body, fo, u, x_edge - u * 0.09, hz, t, v, name=f"{leaf_name}_far_pull")
            elif fo.kind == "lever":
                C.add_rotary_operator(model, leaf_body, spec, phys, fo, u, v, x_spindle, hz, t, [v], phys["lock"]["handle_backlash_locked_rad"] if engaged else None, name=f"{leaf_name}_far_lever")
    elif opm.kind == "thumb_latch":
        # Suffolk: thumb press on robot face (hinge about x), latch bar on far face (hinge about y) resting in keeper on jamb
        mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
        key, mesh = MESH.thumb_latch_mesh(handle_length=opm.style_params.get("handle_length", 0.2), bar_length=opm.style_params.get("bar_length", 0.18))
        leaf_body.geoms.append(C.mesh_geom(f"{leaf_name}_suffolk_grip", key, mesh, (x_edge - u * 0.08, -1.0 * t / 2, hz - 0.05), C.q_face(-1.0, u), mat, 7000, False, ALL_TIERS, "operator", "Suffolk grip"))
        leaf_body.geoms.append(Geom(f"{leaf_name}_suffolk_grip_col", "capsule", (0.008, 0.09), (x_edge - u * 0.08, -1.0 * (t / 2 + 0.035), hz - 0.05), (1, 0, 0, 0), mat, True, False, 7000, None, (0.7, 0.01, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", "Grip"))
        leaf_body.sites.append(Site(f"{leaf_name}_grip_n", (x_edge - u * 0.08, -1.0 * (t / 2 + 0.035), hz - 0.05), QUAT_ID, 0.012, "grip"))
        # thumb piece: a horizontal pad standing off the plate top, pivoting on a pin along x at the door face; pressing
        # the pad DOWN (+q) rotates its tang, which passes through the door at pivot level, UP under the latch bar
        thumb = Body(f"{leaf_name}_thumb", leaf_body.name, (x_edge - u * 0.08, -1.0 * t / 2, hz + 0.095), QUAT_ID, None, [], [], ALL_TIERS, "operator", "Thumb press")
        thumb.joint = Joint(f"{leaf_name}_thumb_hinge", "hinge", (1, 0, 0), (0, 0, 0), (0.0, opm.travel), damping=0.02, frictionloss=0.05, stiffness=opm.spring_rate, springref=-opm.spring_torque_preload / max(opm.spring_rate, 1e-6), role="operator", label="Thumb press (+ = pad pressed down)")
        thumb.geoms.append(C.box(f"{leaf_name}_thumb_geom", (0, -0.020, 0.0), (0.015, 0.014, 0.003), mat, 7000, True, True, ALL_TIERS, "operator", "Thumb pad"))
        thumb.geoms.append(C.box(f"{leaf_name}_thumb_boss", (0, -0.006, 0.0), (0.012, 0.006, 0.008), mat, 7000, False, True, FULL_SIMPLE, "operator", "Pivot boss"))
        thumb.geoms.append(C.box(f"{leaf_name}_thumb_lifter", (0, (0.002 + t + 0.010) / 2, -0.017), (0.004, (t + 0.008) / 2, 0.004), mat, 7000, False, True, FULL_SIMPLE, "operator", "Lifter tang (through the door, under the latch bar)"))
        thumb.sites.append(Site(f"{leaf_name}_thumb_push", (0, -0.024, 0.005), QUAT_ID, 0.01, "push"))
        model.add_body(thumb)
        # latch bar on the far face at the thumb-press height (a Suffolk latch bar pivots level with the thumb press)
        bar_y = 1.0 * (t / 2 + 0.012)
        bar_z = hz + 0.09
        bar = Body(f"{leaf_name}_latch_bar", leaf_body.name, (x_edge - u * 0.20, bar_y, bar_z), QUAT_ID, None, [], [], ALL_TIERS, "latch", "Latch bar")
        bar.joint = Joint(f"{leaf_name}_latch_bar_hinge", "hinge", (0, -u, 0), (0, 0, 0), (0.0, 0.6), damping=0.01, frictionloss=0.01, role="latch", label="Latch bar (0 = in keeper, + = lifted)", robot_interactive=False)
        bar.geoms.append(Geom(f"{leaf_name}_latch_bar_geom", "capsule", (0.006, 0.10), (u * 0.106, 0, 0), tuple(quat_z_to((u, 0, 0))), mat, True, True, 7850.0, 0.25, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "latch", "Latch bar"))
        model.add_body(bar)
        # pivot plate on the far face (the bar's hinge pin) - leaf fixed, visual
        leaf_body.geoms.append(C.box(f"{leaf_name}_latch_bar_pivot", (x_edge - u * 0.20, 1.0 * (t / 2 + 0.002), bar_z), (0.018, 0.002, 0.018), mat, 7000, False, True, FULL_SIMPLE, "latch", "Latch bar pivot plate"))
        leaf_body.geoms.append(C.cyl(f"{leaf_name}_latch_bar_pin", (x_edge - u * 0.20, 1.0 * (t / 2 + 0.010), bar_z), 0.005, 0.010, mat, (0, 1, 0), 7000, False, True, FULL_ONLY, "latch", "Latch bar pivot pin"))
        model.meta.setdefault("clearance_allow", []).append([f"{leaf_name}_latch_bar_geom", f"{leaf_name}_latch_bar_pin", "bar on its pivot pin"])
        model.equalities.append(Equality("joint", f"{leaf_name}_thumb_couple", bar.joint.name, thumb.joint.name, (0, 0.6 / max(opm.travel, 1e-6), 0, 0, 0), tiers=ALL_TIERS, label="bar lift = thumb press"))
        # the bar tip (12 mm past the edge) drops into a keeper pocket in the jamb / post; keeper plate around the mouth
        pockets.append({"z": bar_z + 0.045, "h": 0.14, "w": 0.02, "depth": 0.02, "ramp": False, "y": bar_y})
        C.add_keeper_ring(world.geoms, f"{leaf_name}_latch_bar_keeper", (u * (Wo / 2), bar_y, bar_z + 0.045), (-u, 0, 0), (0, 0, 1), 0.010, 0.070, mat, bar=0.005)
        handle_joint = None
    elif opm.kind in ("gate_latch_fork",):
        # chain-link fork latch (e.g. "chain link fence fork latch 1-3/8 in frame / 2-3/8 in post"): a flat two-prong
        # fork strapped to the GATE's latch stile, lying horizontally with its prongs straddling the post.  Lifting the
        # fork (hinge about the gate normal) swings the prongs up and back off the post; dropped, the post sits between
        # the prongs and blocks the gate in both directions.  A padlock through the arm eye locks it down.
        mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
        ps_ = C.frame_jamb_thickness(spec)
        sx_w = u * (Wo / 2)
        x_piv = x_edge - u * 0.14
        A = abs(sx_w + u * ps_ / 2 - (hx + x_piv))       # pivot -> post centre
        ty = ps_ / 2 + 0.006                              # prong inner faces 6 mm clear of the post
        y_arm = -1.0 * (t / 2 + 0.012)                    # flat arm on the robot face, strapped around the stile
        h_f = 0.05                                        # the arm rises slightly to the fork (prongs above the pivot)
        dens = 2500.0                                     # pressed-steel fork, ~0.25 kg
        fork = Body(f"{leaf_name}_fork", leaf_body.name, (x_piv, 0.0, hz), QUAT_ID, None, [], [], ALL_TIERS, "latch", "Fork latch")
        fork.joint = Joint(f"{leaf_name}_fork_hinge", "hinge", (0, -u, 0), (0, 0, 0), (0.0, 1.2), damping=0.02, frictionloss=0.02, role="operator", label="Fork latch (0 = dropped over the post, + = lifted; lift to open AND to close)")
        x_root = A - ps_ / 2 - 0.020                      # prong root / bridge, 20 mm short of the post face
        L_pr = A + 0.03 - x_root                          # prongs reach 30 mm past the post centre
        # sloping arm from the pivot eye up to the prong root
        a_len = math.hypot(x_root, h_f)
        fork.geoms.append(C.box(f"{leaf_name}_fork_arm", (u * x_root / 2, y_arm, h_f / 2), (a_len / 2, 0.004, 0.012), mat, dens, True, True, ALL_TIERS, "latch", "Fork arm", quat=quat_from_axis_angle([0, 1, 0], -u * math.atan2(h_f, x_root))))
        fork.geoms.append(C.cyl(f"{leaf_name}_fork_eye", (0, y_arm, 0), 0.014, 0.005, mat, (0, 1, 0), dens, False, True, FULL_SIMPLE, "latch", "Pivot eye"))
        # straight parallel prongs: the post is captured without back-driving the fork.  Like the real product the
        # fork is NOT self-latching: the gate is closed with the fork lifted and the fork dropped over the post.
        for sy in (-1, 1):
            fork.geoms.append(C.box(f"{leaf_name}_fork_tine_{'p' if sy > 0 else 'n'}", (u * (x_root + L_pr / 2), sy * (ty + 0.003), h_f), (L_pr / 2, 0.003, 0.012), mat, dens, True, True, ALL_TIERS, "latch", "Fork prong"))
        fork.geoms.append(C.box(f"{leaf_name}_fork_bridge", (u * x_root, 0.0, h_f), (0.004, ty + 0.006, 0.012), mat, dens, True, True, ALL_TIERS, "latch", "Fork bridge"))
        fork.geoms.append(Geom(f"{leaf_name}_fork_handle", "capsule", (0.006, 0.018), (u * x_root, y_arm, h_f + 0.030), (1, 0, 0, 0), mat, True, True, dens, None, (0.7, 0.01, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", "Fork lift handle"))
        fork.sites.append(Site(f"{leaf_name}_fork_grip", (u * x_root, y_arm, h_f + 0.055), QUAT_ID, 0.012, "grip"))
        model.add_body(fork)
        handle_joint = fork.joint.name
        # strap plates around the stile + pivot pin through the stile into the arm's eye (leaf-fixed, visual)
        for sy in (-1, 1):
            leaf_body.geoms.append(C.box(f"{leaf_name}_fork_strap_{'p' if sy > 0 else 'n'}", (x_piv, sy * (t / 2 + 0.002), hz), (0.020, 0.002, 0.030), mat, 7800, False, True, FULL_SIMPLE, "latch", "Fork strap"))
        leaf_body.geoms.append(C.cyl(f"{leaf_name}_fork_pin", (x_piv, (y_arm - 0.008 + t / 2 + 0.004) / 2, hz), 0.005, (t / 2 + 0.004 - y_arm + 0.008) / 2, mat, (0, 1, 0), 7800, False, True, FULL_ONLY, "latch", "Pivot pin"))
        model.meta.setdefault("clearance_allow", []).extend([[f"{leaf_name}_fork_eye", f"{leaf_name}_fork_pin", "pivot pin through the fork eye"], [f"{leaf_name}_fork_arm", f"{leaf_name}_fork_pin", "pivot pin through the arm"]])
        if lk.kind == "padlock" and engaged:
            pm = C.mat_from_material(model, "brass", "mat_padlock")
            leaf_body.geoms.append(C.box(f"{leaf_name}_fork_lug", (x_piv + u * 0.05, y_arm, hz - 0.024), (0.006, 0.004, 0.012), mat, 7800, False, True, FULL_SIMPLE, "lock", "Padlock lug"))
            C.add_padlock(fork.geoms, f"{leaf_name}_fork_padlock", (u * 0.05, y_arm, 0.0), (0, 1, 0), (0, 0, -1), pm, ALL_TIERS, "lock", "Padlock (fork locked down)")
            fork.joint.range = (0.0, 0.001)
            model.meta.setdefault("clearance_allow", []).extend([[f"{leaf_name}_fork_padlock*", f"{leaf_name}_fork_arm", "shackle through the arm eye"], [f"{leaf_name}_fork_padlock*", f"{leaf_name}_fork_lug", "shackle through the lug"]])
    elif opm.kind in ("slide_bolt_handle",):
        # surface-mounted barrel / slide bolt on the robot face (e.g. National Hardware 4 in barrel bolt, 12 in heavy
        # gate slide bolt): mounting plate + guide loops + rod with L-handle on the leaf; U keeper on the post face
        # (standoff so the rod clears a post thicker than the leaf) or a keeper plate over a jamb pocket.
        mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
        L = opm.style_params.get("length", 0.1)
        d = opm.style_params.get("diameter", 0.012)
        ps_ = C.frame_jamb_thickness(spec)
        is_gate = spec["opening"]["frame"]["kind"] in ("gate_posts", "pressure_frame")
        face = -1.0
        standoff = max(d, 0.012, (ps_ / 2 - t / 2 + d / 2 + 0.006) if is_gate else 0.0)
        sx_w = u * (Wo / 2)
        gap_edge = abs(sx_w - (hx + x_edge))
        protrusion = min(opm.travel - 0.004, max(0.036, gap_edge + 0.032))
        sb, info = C.add_barrel_bolt(model, leaf_body, f"{leaf_name}_slide_bolt", (x_edge, face * t / 2, hz), (u, 0, 0), (0, face, 0), L, d, opm.travel, engaged, mat, protrusion=protrusion, standoff=standoff, role="lock", label="Slide bolt (0 = engaged, + = withdrawn)", frictionloss=opm.spring_torque_preload, joint_name=f"{leaf_name}_slide_bolt_slide", grip_site=f"{leaf_name}_grip_n")
        y_rod = face * (t / 2 + standoff)
        if is_gate:
            x_keep = sx_w + u * min(0.020, max(0.012, protrusion - gap_edge - 0.012))
            C.add_keeper_loop(world.geoms, f"{leaf_name}_slide_bolt_keeper", (x_keep, face * ps_ / 2, hz), (x_keep, y_rod, hz), (u, 0, 0), (0, face, 0), d / 2, mat)
        else:
            pockets.append({"z": hz, "h": d + 0.008, "w": d + 0.004, "depth": protrusion + 0.01, "ramp": False, "y": y_rod})
            C.add_keeper_ring(world.geoms, f"{leaf_name}_slide_bolt_keeper", (sx_w, y_rod, hz), (-u, 0, 0), (0, 0, 1), (d + 0.004) / 2, (d + 0.008) / 2, mat)
        if lk.kind == "padlock" and engaged:
            # padlockable bolt: lug on the mounting plate behind the handle; padlock through it blocks the handle
            pm = C.mat_from_material(model, "brass", "mat_padlock")
            s_lug = info["s_knob"] - 0.020
            leaf_body.geoms.append(C.obox(f"{leaf_name}_slide_bolt_lug", (x_edge, face * t / 2, hz), (u, 0, 0), (0, face, 0), s_lug, 0.0, (standoff + d / 2 + 0.012) / 2, 0.003, 0.006, (standoff + d / 2 + 0.012) / 2, mat, False, FULL_SIMPLE, "lock", "Padlock lug"))
            bar_pt = (x_edge + u * s_lug, face * (t / 2 + standoff + d / 2 + 0.014), hz)
            C.add_padlock(leaf_body.geoms, f"{leaf_name}_slide_bolt_padlock", bar_pt, (u, 0, 0), (0, 0, -1), pm, ALL_TIERS, "lock", "Padlock (bolt locked)")
            sb.joint.range = (0.0, 0.001)
            model.meta.setdefault("clearance_allow", []).extend([[f"{leaf_name}_slide_bolt_padlock*", f"{leaf_name}_slide_bolt_lug", "shackle through the lug"], [f"{leaf_name}_slide_bolt_padlock*", f"{leaf_name}_slide_bolt_knob*", "padlock against the handle"], [f"{leaf_name}_slide_bolt_padlock*", f"{leaf_name}_slide_bolt_rod", "padlock beside the rod"]])
    elif opm.kind == "hasp":
        _add_door_hasp(model, world, leaf_body, spec, u, hx, x_edge, t, hz, engaged and lk.kind == "padlock", leaf_name, Wo, opm.material, v=v)
    elif opm.kind == "lift_latch":
        # vertical pin latch (MagnaLatch / baby gate): spring-loaded pin on the POST drops into a striker cup on the
        # gate face; the cup has an approach ramp so the gate self-latches when the closer swings it shut.
        mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
        y_pin = -v * (t / 2 + 0.035)            # on the face the gate swings away from
        x_pin_w = hx + x_edge - u * 0.04         # world x, 40 mm inside the gate's latch edge (striker clears the post while swinging)
        sx_w = u * (Wo / 2)
        pin = Body(f"{leaf_name}_pin", None, (x_pin_w, y_pin, hz), QUAT_ID, None, [], [], ALL_TIERS, "latch", "Lift pin")
        pin.joint = Joint(f"{leaf_name}_pin_slide", "slide", (0, 0, 1), (0, 0, 0), (0.0, opm.travel + 0.03), damping=1.0, frictionloss=0.2, stiffness=60.0, springref=-1.0 / 60.0, role="operator", label="Lift pin (+ = lifted; weak return spring, magnet-assisted drop)")
        pin.geoms.append(Geom(f"{leaf_name}_pin_geom", "capsule", (0.006, 0.03), (0, 0, -0.02), (1, 0, 0, 0), mat, True, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "latch", "Latch pin"))
        pin.geoms.append(Geom(f"{leaf_name}_pin_knob", "capsule", (0.012, 0.01), (0, 0, 0.065), (1, 0, 0, 0), mat, True, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", "Lift knob"))
        pin.sites.append(Site(f"{leaf_name}_grip_pin", (0, 0, 0.065), QUAT_ID, 0.012, "grip"))
        model.add_body(pin)
        handle_joint = pin.joint.name
        # latch body: a housing on the post (bracketed out over the gate edge, like the D&D MagnaLatch / a pressure
        # gate latch housing) that the pin runs through; the pull knob sits on top of the housing
        is_mag = opm.id == "gate_latch_magnetic"
        hw, hd, htop, hbot = (0.022, 0.020, 0.045, 0.015) if is_mag else (0.018, 0.016, 0.045, 0.015)   # bottom clear of the striker cup
        world.geoms.append(C.box(f"{leaf_name}_pin_housing", (x_pin_w, y_pin, hz + (htop - hbot) / 2), (hw, hd, (htop + hbot) / 2), mat, 1800, False, True, FULL_SIMPLE, "latch", "Latch housing"))
        xb0, xb1 = x_pin_w + u * hw, sx_w + u * 0.006     # bracket from the housing's edge-side face to the post face
        for k, (zc_, hz_) in enumerate(((hz + 0.024, 0.010), (hz - 0.006, 0.006))):
            world.geoms.append(C.box(f"{leaf_name}_pin_bracket{'' if k == 0 else '_2'}", ((xb0 + xb1) / 2, y_pin, zc_), (abs(xb1 - xb0) / 2, hd, hz_), mat, 1800, False, True, FULL_SIMPLE if k == 0 else FULL_ONLY, "latch", "Housing bracket to the post"))
        world.geoms.append(C.box(f"{leaf_name}_pin_post_plate", (sx_w + u * 0.003, y_pin, hz + 0.01), (0.003, hd + 0.006, 0.05), mat, 1800, False, True, FULL_ONLY, "latch", "Post mounting plate"))
        model.meta.setdefault("clearance_allow", []).extend([[f"{leaf_name}_pin_geom", f"{leaf_name}_pin_housing", "pin slides inside its housing"], [f"{leaf_name}_pin_knob", f"{leaf_name}_pin_housing", "knob seats on the housing"], [f"{leaf_name}_pin_geom", f"{leaf_name}_pin_bracket*", "pin passes the bracket"]])
        # striker cup on the gate (leaf-local): walls +-y, floor, bracket to the face, approach ramp on the -v side
        km = C.mat_from_material(model, "steel_galvanized", "mat_keeper")
        xc_ = x_edge - u * 0.04
        for sy in (-1, 1):
            leaf_body.geoms.append(C.box(f"{leaf_name}_cup_{'p' if sy > 0 else 'n'}", (xc_, y_pin + sy * 0.012, hz - 0.045), (0.016, 0.004, 0.02), km, 7800, True, True, ALL_TIERS, "latch", "Cup wall"))
        leaf_body.geoms.append(C.box(f"{leaf_name}_cup_b", (xc_, y_pin, hz - 0.069), (0.016, 0.016, 0.004), km, 7800, True, True, ALL_TIERS, "latch", "Cup floor"))
        leaf_body.geoms.append(C.box(f"{leaf_name}_cup_bracket", (xc_, -v * (t / 2 + 0.0095), hz - 0.045), (0.016, 0.0095, 0.004), km, 7800, False, True, FULL_SIMPLE, "latch", "Cup bracket"))
        y_w = y_pin - v * 0.016
        d_ = (0.0, -v * 0.747, -0.664)                       # down the ramp (run 45 mm, drop 40 mm)
        n_up = (0.0, -v * 0.664, 0.747)
        phi = math.atan2(d_[2], d_[1])
        mid = (xc_, y_w - v * 0.0225, hz - 0.045)
        leaf_body.geoms.append(C.box(f"{leaf_name}_cup_ramp", (mid[0] - n_up[0] * 0.004, mid[1] - n_up[1] * 0.004, mid[2] - n_up[2] * 0.004), (0.016, 0.0301, 0.004), km, 7800, True, True, ALL_TIERS, "latch", "Cup ramp", quat=C.quat_from_axis_angle([1, 0, 0], phi), friction=(0.12, 0.005, 0.0001)))
    elif opm.kind == "push_button_screen":
        # push-button latch: small button on the robot face (slide), lever-ish; couples to a small latch bolt
        mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
        btn = Body(f"{leaf_name}_pushbutton", leaf_body.name, (x_spindle, -1.0 * (t / 2 + 0.01), hz), QUAT_ID, None, [], [], ALL_TIERS, "operator", "Push button")
        btn.joint = Joint(f"{leaf_name}_pushbutton_slide", "slide", (0, 1, 0), (0, 0, 0), (0.0, opm.travel), damping=1.0, frictionloss=0.2, stiffness=opm.spring_rate, springref=-opm.spring_torque_preload / max(opm.spring_rate, 1e-6), role="operator", label="Push button (+ = pressed)")
        btn.geoms.append(C.cyl(f"{leaf_name}_pushbutton_geom", (0, -0.006, 0), 0.01, 0.006, mat, (0, 1, 0), 2700, True, True, ALL_TIERS, "operator", "Button"))
        btn.sites.append(Site(f"{leaf_name}_push", (0, -0.012, 0), QUAT_ID, 0.01, "push"))
        model.add_body(btn)
        handle_joint = btn.joint.name
        # far side: small pull handle
        C.add_pull(model, leaf_body, H.OPERATORS["pull_d"], u, x_edge - u * 0.09, hz, t, 1.0, name=f"{leaf_name}_inside_pull")
        leaf_body.geoms.append(C.box(f"{leaf_name}_pushbutton_housing", (x_spindle, -1.0 * (t / 2 + 0.006), hz), (0.02, 0.006, 0.045), mat, 2700, False, True, FULL_SIMPLE, "operator", "Latch housing"))
    # --- latches
    if lt.throw > 0 and lt.kind in ("tubular_latch", "deadlatch", "mortise_latch", "rim_latch", "roller", "ball_catch", "hook") and fam not in ("stall",) and opm.kind not in ("lift_latch", "slide_bolt_handle", "gate_latch_fork", "thumb_latch", "hasp"):
        scale_eff = scale if handle_joint else 0.0
        # rim exit device: the Pullman latch bolt lives in the surface case on the push face (not in the slab) and
        # shoots into a pocket + lip strike at that offset in the jamb (Von Duprin 299 strike)
        y_bolt = -v * (t / 2 + 0.022) if (lt.kind == "rim_latch" and opm.kind == "panic_touchbar" and not pair) else 0.0
        res = C.add_spring_latch(model, leaf_body, spec, phys, u, v, x_edge, hz, t, lt, handle_joint, scale_eff, name=f"{leaf_name}_latch_bolt", y=y_bolt, faceplate=(lt.kind not in ("rim_latch", "roller", "ball_catch")))
        for pk in res.pockets:
            if lt.kind in ("roller", "ball_catch"):
                pk["ramp_both"] = True
            if abs(y_bolt) > 1e-6:
                pk["stop_cut_half"] = opm.style_params.get("bar_height", 0.05) * 0.75 + 0.010   # stop cut for the rim case
            pockets.append(pk)
        model.tendons += res.tendons
    elif lt.kind == "vertical_rods" and handle_joint:
        # top rod latch: bolt at the leaf top going up into a pocket in the head
        rl = H.LATCHES["rim_exit"]
        bw, bh = rl.bolt_size
        inside_ = 0.05
        bm = C.mat_from_material(model, "stainless", "mat_bolt")
        rb = Body(f"{leaf_name}_top_latch", leaf_body.name, (x_edge - u * 0.06, 0.0, zb + Hh), QUAT_ID, None, [], [], ALL_TIERS, "latch", "Top rod latch")
        rb.joint = Joint(f"{leaf_name}_top_latch_slide", "slide", (0, 0, -1), (0, 0, 0), (0.0, rl.throw), damping=2.0, frictionloss=0.3, stiffness=rl.spring_rate, springref=-rl.spring_preload / rl.spring_rate, armature=1e-4, role="latch", label="Top rod latch (0 = extended up, + = retracted)", robot_interactive=False)
        r = bw / 2
        rb.geoms.append(Geom(f"{leaf_name}_top_latch_capsule", "capsule", (r, (rl.throw + inside_) / 2 - r), (0, 0, (rl.throw - inside_) / 2), (1, 0, 0, 0), bm, True, True, 8500.0, None, (0.2, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "latch", "Top rod bolt"))
        model.add_body(rb)
        model.tendons.append(Tendon(f"{leaf_name}_top_latch_coupling", [(rb.joint.name, 1.0), (handle_joint, -scale)], (0.0, 10.0), tiers=ALL_TIERS, label="rod_q >= scale*bar_q (one-sided)"))
        model.tendons[-1].kind = "fixed"
        head_pockets.append({"x": hx + x_edge - u * 0.06, "hx": bw / 2 + 0.004, "w": bw + 0.003, "depth": rl.throw + 0.004})
    elif lt.kind == "gravity_bar" and opm.kind == "ring_pull":
        # latch bar lifted by... nothing on robot side except the ring; the bar is on the far side -> robot side must lift via a thumb? Use a simple gravity bar with a lift knob through the door.
        pass
    elif lt.kind == "magnetic":
        # magnetic catch: weak holding via a soft roller-style catch (ramp both ways)
        mag = H.LATCHES["roller_latch"]
        res = C.add_spring_latch(model, leaf_body, spec, phys, u, v, x_edge, Hh + zb - 0.05, t, mag, None, 0.0, name=f"{leaf_name}_mag_catch", tiers=FULL_SIMPLE)
        res.bolt_body.joint.stiffness = max(lt.holding_force, 5.0) / 0.006
        res.bolt_body.joint.springref = -0.006
        for pk in res.pockets:
            pk["ramp_both"] = True
            pockets.append(pk)
    # far-side lever trim on a panic device: retracts the same bolt(s) (tendon term added to the pad coupling)
    far_j = f"{leaf_name}_far_lever_hinge"
    if handle_joint and far_op and far_op != "none" and any(b.joint and b.joint.name == far_j for b in model.bodies):
        fo = H.OPERATORS[far_op]
        for td in model.tendons:
            if td.name.startswith(f"{leaf_name}_") and any(jn == handle_joint for jn, _ in td.sites):
                sc = next(abs(c) for jn, c in td.sites if jn == handle_joint) * max(opm.travel - opm.dead_travel, 1e-6) / max(fo.travel - fo.dead_travel, 1e-6)
                td.sites.append((far_j, -sc))
    # --- locks
    eqs = []
    if lk.kind in ("deadbolt_single", "deadbolt_double", "thumbturn_only", "mortise_deadbolt", "night_latch", "multipoint", "keypad_code") and lk.deadbolt_throw > 0:
        zdb = hz + 0.14 if lk.kind != "mortise_deadbolt" else hz + 0.06
        if opm.kind == "handleset":
            zdb = hz + 0.175          # above the thumb press at the top of the grip plate
        # thumbturn on the inside face (+1 = far side from robot means robot is outside)
        inside_face = 1.0 if spec["robot"]["robot_outside"] else -1.0
        tt_side = inside_face if lk.inside_release == "thumbturn" else None
        if lk.kind == "keypad_code" and not release and engaged:
            tt_side = None if spec["robot"]["robot_outside"] else inside_face
        if lk.kind == "keypad_code" and release and not spec["robot"]["robot_outside"]:
            tt_side = inside_face
        keyed = -inside_face if lk.outside_release in ("key", "code", "card") else None
        if lk.kind == "keypad_code":
            keyed = None                       # the key cylinder is part of the keypad unit
            if tt_side is not None:
                tt_side = -(keypad_face if keypad_face is not None else -1.0)   # thumbturn opposite the keypad
        if engaged and not release:
            tt_side = None  # no accessible release -> deadbolt fixed
        tt_so = 0.0
        if lk.kind == "night_latch":
            # rim night latch (Yale 77): surface case on the inside face carrying the snib / turn; keyed cylinder outside
            nm_ = C.mat_from_material(model, "brass_antique", "mat_night_latch")
            tt_so = 0.028
            leaf_body.geoms.append(C.box(f"{leaf_name}_night_latch_case", (x_edge - u * 0.060, inside_face * (t / 2 + tt_so / 2), zdb), (0.046, tt_so / 2, 0.042), nm_, 2500, True, True, FULL_SIMPLE, "lock", "Rim night latch case"))
        body, pk, eq = C.add_deadbolt(model, leaf_body, spec, u, v, x_edge, zdb, t, lk.deadbolt_throw, engaged, tt_side, lk.thumbturn_travel or 1.5708, lk.thumbturn_torque or 0.3, name=f"{leaf_name}_deadbolt", keyed_side=keyed, tt_standoff=tt_so)
        pockets += pk
        eqs += eq
        bm_ = C.mat_from_material(model, "brass", "mat_deadbolt")
        if not pair:
            C.add_strike_plate(world.geoms, f"{leaf_name}_deadbolt_strike", u * (Wo / 2), u, 0.0, zdb, 0.0095, 0.0155, bm_)
        if lk.kind == "multipoint":
            # multipoint lock: two more lock points (hook / shoot bolts) at +-0.5 m, driven with the main bolt
            drive = (f"{leaf_name}_deadbolt_slide", 1.0) if body.joint is not None else None
            for tag, zm in (("upper", min(zdb + 0.5, zb + Hh - 0.12)), ("lower", max(zdb - 0.55, zb + 0.15))):
                _, pk2, eq2 = C.add_deadbolt(model, leaf_body, spec, u, v, x_edge, zm, t, lk.deadbolt_throw, engaged, None, 1.5708, 0.3, name=f"{leaf_name}_multipoint_{tag}", keyed_side=None, couple_to=drive)
                pockets += pk2
                eqs += eq2
                if not pair:
                    C.add_strike_plate(world.geoms, f"{leaf_name}_multipoint_{tag}_strike", u * (Wo / 2), u, 0.0, zm, 0.0095, 0.0155, bm_)
            # the lock points sit on a full-height faceplate strip
            leaf_body.geoms.append(C.box(f"{leaf_name}_multipoint_strip", (x_edge - u * 0.0004, 0.0, (zb + zb + Hh) / 2), (0.0004, min(0.010, t / 2 - 0.002), Hh / 2 - 0.02), bm_, 8500, False, True, FULL_ONLY, "leaf", "Multipoint faceplate strip"))
    if lk.kind == "slide_bolt" and opm.kind != "slide_bolt_handle":
        # auxiliary barrel bolt (4 in brass barrel bolt) on the inside face above the handle: mounting plate, two
        # guide loops, rod with knob; the rod enters a keeper plate mortised over a pocket in the jamb
        sbm = H.OPERATORS["slide_bolt_barrel"]
        inside_face = 1.0 if spec["robot"]["robot_outside"] else -1.0
        mat = C.mat_from_material(model, sbm.material, f"mat_op_{sbm.material}")
        L, d, standoff, prot = 0.1, 0.012, 0.012, 0.030
        if op["frame"].get("casing") and t / 2 + standoff + d / 2 > t / 2 + C.LEAF_FACE_INSET:
            # the rod stands proud of the jamb's face, so it runs ACROSS the reveal onto a surface keeper: its tip
            # must stop inside the reveal, before the casing (which laps the jamb by 5 mm) - a 30 mm throw ran the
            # rod straight through the casing once the leaf was hung flush with the frame face
            prot = min(prot, max(0.012, jt_ - 0.005 - 0.003))
        zsb = hz + 0.25
        x_ab = x_edge
        if pair and inside_face == -v:
            # pairs: the astragal on the other leaf laps this face at the edge; mount the barrel 30 mm inboard
            x_ab, prot, L = x_edge - u * 0.03, prot + 0.03, 0.13
        sb, _ = C.add_barrel_bolt(model, leaf_body, f"{leaf_name}_aux_bolt", (x_ab, inside_face * t / 2, zsb), (u, 0, 0), (0, inside_face, 0), L, d, sbm.travel, engaged, mat, protrusion=prot, standoff=standoff, tiers=FULL_SIMPLE, role="lock", label="Barrel bolt (0 = engaged, + = withdrawn)", joint_name=f"{leaf_name}_aux_bolt_slide", grip_site=f"{leaf_name}_aux_bolt_grip", rod_semantic="lock")
        if engaged and not release:
            sb.joint.range = (0.0, 0.001)
            sb.joint.notes = "no inside access: bolt fixed"
        y_rod = inside_face * (t / 2 + standoff)
        pockets.append({"z": zsb, "h": d + 0.008, "w": d + 0.004, "depth": prot + 0.012, "ramp": False, "y": y_rod})
        if not pair:
            C.add_keeper_ring(world.geoms, f"{leaf_name}_aux_bolt_keeper", (u * (Wo / 2), y_rod, zsb), (-u, 0, 0), (0, 0, 1), (d + 0.004) / 2, (d + 0.008) / 2, mat)
    if lk.kind == "padlock" and opm.kind not in ("hasp", "slide_bolt_handle", "gate_latch_fork") and not pair:
        # padlock with any other operator: a hasp & staple above the handle on the outside face (padlock hanging in
        # the staple when locked; hasp flipped open otherwise)
        _add_door_hasp(model, world, leaf_body, spec, u, hx, x_edge, t, min(hz + 0.20, zb + Hh - 0.08), engaged and not release, leaf_name, Wo, v=v)
    if lk.kind in ("mag_lock", "delayed_egress") and engaged:
        model.equalities.append(Equality("weld", f"{leaf_name}_maglock", leaf_body.name, "world", (0, 0, 0, 0, 0), (0, 0, 0), ALL_TIERS, f"{lk.name} (env releases on REX / badge / timer)", active=True))
        mm = C.mat_from_material(model, "aluminum_dark", "mat_maglock")
        # magnet on the frame on the side the leaf closes AGAINST (-v); armature plate on the leaf face touches it
        world.geoms.append(C.box(f"{leaf_name}_maglock_body", (hx + x_edge - u * 0.30, -v * (t / 2 + 0.035), Ho - 0.065), (0.125, 0.025, 0.02), mm, 2000, True, True, FULL_SIMPLE, "lock", "Electromagnetic lock"))
        # a maglock hangs on its header bracket, not in mid air 45 mm under the soffit
        world.geoms.append(C.box(f"{leaf_name}_maglock_bracket", (hx + x_edge - u * 0.30, -v * (t / 2 + 0.035), Ho - 0.0225), (0.09, 0.025, 0.0225), mm, 7850, False, True, FULL_SIMPLE, "lock", "Maglock header bracket"))
        leaf_body.geoms.append(C.box(f"{leaf_name}_maglock_armature", (x_edge - u * 0.30, -v * (t / 2 + 0.005), Ho - 0.065), (0.09, 0.005, 0.02), mm, 7800, False, True, FULL_SIMPLE, "lock", "Maglock armature plate"))
        model.meta.setdefault("breakable_welds", []).append({"name": f"{leaf_name}_maglock", "holding_force_N": H.LATCHES["mag_lock_1200" if "1200" in lk.name else "mag_lock_600"].holding_force})
    if lk.kind == "chain" and engaged:
        # door chain: anchor plate on the inside face of the leaf, slotted track on the jamb's inside face, chain links
        # between them (links ride with the leaf; the leaf joint range models the slack)
        cm = C.mat_from_material(model, "brass", "mat_chain")
        f_c = 1.0 if spec["robot"]["robot_outside"] else -1.0
        z_c = hz + 0.30
        depth_c = op["wall_thickness"] if op["frame"]["kind"] != "aluminum_storefront" else max(0.114, op["wall_thickness"])
        y_track = float(model.meta.get("wall_y", 0.0)) + f_c * (depth_c / 2 + 0.005)
        leaf_body.geoms.append(C.box(f"{leaf_name}_chain_plate", (x_edge - u * 0.05, f_c * (t / 2 + 0.002), z_c), (0.012, 0.002, 0.020), cm, 8500, False, True, FULL_SIMPLE, "lock", "Chain anchor plate"))
        world.geoms.append(C.box(f"{leaf_name}_chain_track", (hx + x_edge + u * 0.035, y_track, z_c), (0.035, 0.005, 0.010), cm, 8500, False, True, FULL_SIMPLE, "lock", "Chain track (slotted)"))
        p0 = np.array([x_edge - u * 0.05, f_c * (t / 2 + 0.008), z_c])
        p1 = np.array([x_edge + u * 0.012, y_track, z_c - 0.012])
        n_l = 6
        for k in range(n_l):
            a_ = p0 + (p1 - p0) * (k + 0.5) / n_l
            leaf_body.geoms.append(Geom(f"{leaf_name}_chain_{k}", "capsule", (0.003, 0.006), tuple(a_), tuple(quat_z_to(p1 - p0)), cm, False, True, 8500.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, FULL_ONLY, "lock", "Chain link"))
        # the links are drawn rigid with the leaf (a real chain hangs slack): they may pass the track / jamb face
        model.meta.setdefault("clearance_allow", []).extend([[f"{leaf_name}_chain_*", f"{leaf_name}_chain_track", "chain end in its track"], [f"{leaf_name}_chain_*", "jamb_*", "slack chain vs jamb"], [f"{leaf_name}_chain_*", "stop_*", "slack chain vs stop"], [f"{leaf_name}_chain_*", "seal_*", "slack chain vs seal"], [f"{leaf_name}_chain_*", "casing_*", "slack chain vs casing"], [f"{leaf_name}_chain_*", "stud_*", "slack chain vs stud"]])
    if lk.kind == "swing_bar_guard" and engaged:
        gm = C.mat_from_material(model, "stainless", "mat_guard")
        world.geoms.append(C.box(f"{leaf_name}_guard_bar", (hx + x_edge + u * 0.02, 1.0 * (t / 2 + 0.03), hz + 0.30), (0.06, 0.006, 0.01), gm, 7900, False, True, FULL_ONLY, "lock", "Swing bar guard"))
        leaf_body.geoms.append(C.cyl(f"{leaf_name}_guard_stud", (x_edge - u * 0.05, 1.0 * (t / 2 + 0.015), hz + 0.30), 0.01, 0.015, gm, (0, 1, 0), 7900, False, True, FULL_ONLY, "lock", "Guard stud"))
    model.equalities += eqs
    # --- frame with pockets
    if not pair:
        C.add_frame(model, spec, v, world, with_stop=not both_ways and not hg.kind.startswith("pivot") and hg.kind != "gravity_pivot", strike_pockets=pockets, u=u, head_pockets=head_pockets)
    else:
        pair["pockets"] += [dict(p, leaf=leaf_name, u=u) for p in pockets]
        pair.setdefault("head_pockets", []).extend(head_pockets)
    # --- closer
    C.add_closer(model, world, leaf_body, spec, phys, u, v, hx, Hh, t, Wo, 0.019 if not pair else pair.get("jamb_t", 0.019), tier_full_arms=not pair)
    # --- extras (world-level extras only once for pairs)
    ex_spec = spec
    if pair and leaf_name != "leaf_a":
        ex_spec = {**spec, "extras": [e for e in spec["extras"] if e in ("kick_plate", "armor_plate", "bumper_rail", "push_pull_sign", "warning_placard")]}
    C.add_extras(model, world, leaf_body, ex_spec, u, v, x_leaf0, zb, W, Hh, t, Wo, Ho)
    C.add_pet_flap(model, leaf_body, spec, u, x_leaf0, zb, W, t)
    # opening stop hardware: a rubber bumper on a real mount (wall plate or floor riser), struck by the leaf at the limit
    stop = spec["kinematics"].get("stop")
    if stop in ("wall_bumper",) and not pair:
        C.add_bumper_stop(model, world, leaf_body, spec, u, v, hx, x_leaf0, W, t, Hh, zb)
    # --- sites for benchmark
    world.sites.append(Site("approach_point", (0, -1.5, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (0, 1.5, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (0, 0, Ho / 2), QUAT_ID, 0.02, "pass_plane"))
    leaf_body.sites.append(Site(f"{leaf_name}_edge_mid", (x_edge, 0, Hh / 2), QUAT_ID, 0.02, "leaf_edge"))
    # robot outside a panic door: it cannot reach the bar; the operator is the far-side trim (if any)
    if opm.kind in ("panic_touchbar", "panic_crossbar") and spec["robot"]["robot_outside"]:
        far = f"{leaf_name}_far_lever_hinge"
        handle_joint = far if any(b.joint and b.joint.name == far for b in model.bodies) else None
    if pair and leaf_name == "leaf_a":
        pair["op_a"] = handle_joint
    model.meta.update({"u": u, "v": v, "hinge_x": hx, "leaf_edge_x_local": x_edge, "handle_height": hz, "primary_joint": j.name, "operator_joint": handle_joint})
    return leaf_body


def build_swing_double(spec, phys, model: Model):
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    v = 1.0 if spec["robot"]["is_push"] else -1.0
    jt_ = C.frame_jamb_thickness(spec)
    depth_ = op["wall_thickness"] if op["frame"]["kind"] != "aluminum_storefront" else max(0.114, op["wall_thickness"])
    y_wall = -v * max(0.0, depth_ / 2 - t / 2 - 0.02) if not spec["kinematics"].get("double_egress") else 0.0
    world = C.add_floor_and_wall(model, spec, hole=(-Wo / 2 - jt_, Wo / 2 + jt_, 0.0, Ho + jt_ + C.STUD_POCKET), y_wall=y_wall)
    ctx = spec["context"]
    astragal = leaf.get("astragal", "none")
    mullion = astragal == "removable_mullion"
    inactive = leaf.get("inactive_leaf", {})
    both_active = inactive.get("active", False)
    double_egress = spec["kinematics"].get("double_egress", False)
    pair = {"world": world, "pockets": [], "jamb_t": 0.05}
    hg_ = H.HINGES[spec["hinge"]["model"]]
    inset_ = 0.006 if hg_.kind in ("pivot_center", "pivot_center_heavy") else C.GAP
    W_leaf = (Wo - 2 * inset_ - C.GAP) / 2 if not mullion else (Wo - 2 * inset_ - 0.05 - 2 * C.GAP) / 2
    # leaf A: hinge left (u=+1); leaf B: hinge right (u=-1)
    res = {}
    for name, u_, hx_, active in (("leaf_a", 1.0, -Wo / 2, True), ("leaf_b", -1.0, Wo / 2, both_active)):
        v_ = v if not (double_egress and name == "leaf_b") else -v
        sub = dict(spec)
        sub = {**spec, "leaf": {**leaf, "width": W_leaf}}
        if name == "leaf_b" and not mullion:
            pair["edge_pockets"] = [dict(p) for p in pair["pockets"] if p.get("leaf") == "leaf_a"]
            if active:
                sub = {**sub, "lock": {"model": "none", "engaged": False, "robot_side_release": True}}
        if not active:
            # inactive leaf: fixed (flush bolts / cane bolt), operator only visual pull
            sub = {**sub, "operator": {**spec["operator"], "model": "none"}, "latch": {"model": "none"}, "lock": {"model": "none", "engaged": False, "robot_side_release": True}, "closer": {"model": "none", "en_size": None, "spring_adjust": 1.0}, "extras": [e for e in spec["extras"] if e in ("kick_plate",)]}
        pair.update({"u": u_, "v": v_, "hx": hx_})
        sub_phys = phys
        lb = build_swing_single(sub, sub_phys, model, leaf_name=name, pair=pair)
        if not active:
            lb.joint.range = (0.0, 0.001)
            lb.joint.label = "Inactive leaf (flush bolts engaged)"
            lb.joint.notes = "inactive leaf held by flush bolts; set range to free it"
        res[name] = lb
    # frame: hinge jambs both sides, head; mullion or strike into inactive leaf
    fr = op["frame"]
    mat = C.mat_from_material(model, fr["material"], "mat_frame")
    dens = 300.0
    jamb_t = 0.05 if fr["kind"] == "hollow_metal_frame" else 0.019
    depth = op["wall_thickness"]
    for sgn, nm in ((-1, "l"), (1, "r")):
        world.geoms.append(C.box(f"jamb_{nm}", (sgn * (Wo / 2 + jamb_t / 2), y_wall, Ho / 2), (jamb_t / 2, depth / 2, Ho / 2), mat, dens, semantic="frame", label="Jamb"))
    C.add_head(world.geoms, "jamb_head", -(Wo / 2 + jamb_t), Wo / 2 + jamb_t, y_wall, depth, Ho, jamb_t, mat, dens, pair.get("head_pockets"))
    C.add_head(world.geoms, "head_stud", -(Wo / 2 + jamb_t), Wo / 2 + jamb_t, y_wall, depth, Ho + jamb_t, C.STUD_POCKET, "mat_wall", 500, [dict(p, depth=max(p["depth"] - jamb_t, 0.0)) for p in pair.get("head_pockets", []) if p["depth"] > jamb_t], label="Head stud")
    # stop (both leaves close against it)
    if fr.get("stop_depth", 0) > 0 and not double_egress and not hg_.kind.startswith("pivot") and leaf["panel_style"] != "glass_frameless":
        sd = fr["stop_depth"]
        ys = -v * (t / 2 + sd / 2 + 0.0005)
        for sgn, nm in ((-1, "l"), (1, "r")):
            world.geoms.append(C.box(f"stop_{nm}", (sgn * (Wo / 2 - 0.015), ys, Ho / 2), (0.015, sd / 2, Ho / 2), mat, dens, semantic="frame", label="Stop"))
        world.geoms.append(C.box("stop_head", (0, ys, Ho - 0.015), (Wo / 2, sd / 2, 0.015), mat, dens, semantic="frame", label="Stop (head)"))
    if mullion:
        # center mullion: one strike half-column per leaf (lipped / ramped pockets from pair["pockets"])
        mw = 0.05
        pk_a = [dict(p, y=p.get("y", 0.0)) for p in pair["pockets"] if p.get("leaf") == "leaf_a"]
        pk_b = [dict(p, y=p.get("y", 0.0)) for p in pair["pockets"] if p.get("leaf") == "leaf_b"]
        ya_, yb_ = y_wall - depth / 2, y_wall + depth / 2
        C._strike_column(world.geoms, "mullion_a", -mw / 2, 1.0, v, mw / 2, ya_, yb_, Ho, pk_a, mat, dens, jamb_seg_name="mullion_a_seg", z_bot=0.0)
        C._strike_column(world.geoms, "mullion_b", mw / 2, -1.0, -v if double_egress else v, mw / 2, ya_, yb_, Ho, pk_b, mat, dens, jamb_seg_name="mullion_b_seg", z_bot=0.0)
    else:
        lb = res["leaf_b"]
        if astragal in ("T_astragal_on_inactive", "overlapping_astragal"):
            am = C.mat_from_material(model, "aluminum", "mat_astragal")
            lb.geoms.append(C.box("astragal", (-(inset_ + W_leaf) - 0.004, -v * (t / 2 + 0.008), Hh / 2), (0.02, 0.008, Hh / 2 - 0.02), am, 2700, True, True, FULL_SIMPLE, "frame", "Astragal"))
    world.sites.append(Site("approach_point", (0, -1.5, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (0, 1.5, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (0, 0, Ho / 2), QUAT_ID, 0.02, "pass_plane"))
    model.meta.update({"pair": True, "primary_joint": "leaf_a_hinge", "secondary_joint": "leaf_b_hinge", "mullion": mullion, "operator_joint": pair.get("op_a")})
    return res


def build_dutch(spec, phys, model: Model):
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    split = leaf.get("dutch_split_height", Hh / 2)
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    u, v = _uv(spec)
    jt_ = C.frame_jamb_thickness(spec)
    world = C.add_floor_and_wall(model, spec, hole=C.frame_hole(spec, u, jt_))
    hx = u * (-Wo / 2)
    y_pin = v * (t / 2 + 0.007)
    model.meta["wall_y"] = -v * max(0.0, op["wall_thickness"] / 2 - t / 2 - 0.02)
    for g in world.geoms:
        if g.semantic == "wall":
            g.pos = (g.pos[0], model.meta["wall_y"], g.pos[2])
    bodies = []
    for name, z0, h_leaf in (("leaf_lower", C.BOTTOM_CLEAR, split - C.BOTTOM_CLEAR - 0.004), ("leaf_upper", split + 0.004, Hh - split - 0.004)):
        b = Body(name, None, (hx, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", name.replace("_", " ").title())
        j = hinge_joint(spec, phys, u, v, y_pin, name=f"{name}_hinge")
        j.pos = (u * C.GAP, y_pin, 0.0)
        j.frictionloss *= 0.6
        b.joint = j
        model.add_body(b)
        sub_leaf = {**leaf, "height": h_leaf}
        C.add_leaf_geoms(model, b, spec, sub_leaf, u, u * C.GAP, z0, None, name_prefix=name, Hh=h_leaf)
        bodies.append(b)
    lower, upper = bodies
    x_edge = u * (C.GAP + W)
    hz = spec["operator"]["height"]
    opm = H.OPERATORS[spec["operator"]["model"]]
    lt = H.LATCHES[spec["latch"]["model"]]
    pockets = []
    faces, _ = C.operator_faces(spec, v)
    lk, engaged, release = _lock_state(spec)
    handle_joint = None
    if opm.kind in ("lever", "knob"):
        hb = C.add_rotary_operator(model, lower, spec, phys, opm, u, v, x_edge - u * lt.backset, min(hz, split - 0.12), t, faces, None, name="lower_handle")
        handle_joint = hb.joint.name
    elif opm.kind == "handleset":
        for f in faces:
            if f < 0:
                C.add_pull(model, lower, opm, u, x_edge - u * 0.105, min(hz, split - 0.12), t, f, name="lower_handleset")
        hb = C.add_rotary_operator(model, lower, spec, phys, H.OPERATORS["knob_round"], u, v, x_edge - u * lt.backset, min(hz, split - 0.12), t, [1.0], None, name="lower_handle")
        handle_joint = hb.joint.name
    scale = -phys["latch"]["coupling"]["bolt_q = c0 + c1*op_q"][1] if phys["latch"]["coupling"] else 0.0
    res = C.add_spring_latch(model, lower, spec, phys, u, v, x_edge, min(hz, split - 0.12), t, lt, handle_joint, scale, name="lower_latch_bolt")
    pockets += res.pockets
    model.tendons += res.tendons
    # upper leaf: ball catch keeping it closed + joining bolt
    ub = C.add_spring_latch(model, upper, spec, phys, u, v, x_edge, split + (Hh - split) * 0.5, t, H.LATCHES["ball_catch"], None, 0.0, name="upper_catch", tiers=FULL_SIMPLE)
    for pk in ub.pockets:
        pk["ramp_both"] = True
    pockets += ub.pockets
    # joining bolt (Dutch door bolt): vertical barrel bolt on the upper leaf's inside face next to the split, its rod
    # dropping through a keeper loop on the lower leaf so the two leaves swing as one
    jm = C.mat_from_material(model, "brass", "mat_joinbolt")
    inside = 1.0 if spec["robot"]["robot_outside"] else -1.0
    eng = bool(spec["kinematics"].get("joining_bolt_engaged"))
    jb, _ = C.add_barrel_bolt(model, upper, "join_bolt", (x_edge - u * 0.12, inside * t / 2, split + 0.004), (0, 0, -1), (0, inside, 0), 0.10, 0.012, 0.05, eng, jm, protrusion=0.034, standoff=0.012, tiers=FULL_SIMPLE, role="lock", label="Joining bolt (0 = joined, + = lifted)", handle_at="rear", joint_name="join_bolt_slide", grip_site="join_bolt_grip", rod_semantic="lock")
    C.add_keeper_loop(lower.geoms, "join_keeper", (x_edge - u * 0.12, inside * t / 2, split - 0.03), (x_edge - u * 0.12, inside * (t / 2 + 0.012), split - 0.03), (0, 0, -1), (0, inside, 0), 0.006, jm, FULL_SIMPLE, base=0.026)
    C.add_frame(model, spec, v, world, True, strike_pockets=pockets, u=u)
    C.add_hinge_visuals(model, world, lower, spec, (0, y_pin), split - C.BOTTOM_CLEAR - 0.004, C.BOTTOM_CLEAR, v, u)
    C.add_hinge_visuals(model, world, upper, spec, (0, y_pin), Hh - split - 0.004, split + 0.004, v, u)
    world.sites.append(Site("approach_point", (0, -1.5, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (0, 1.5, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (0, 0, Ho / 2), QUAT_ID, 0.02, "pass_plane"))
    model.meta.update({"u": u, "v": v, "primary_joint": "leaf_lower_hinge", "secondary_joint": "leaf_upper_hinge", "operator_joint": handle_joint, "handle_height": hz})
    return bodies


def build_saloon(spec, phys, model: Model):
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    zb = leaf.get("bottom_clearance", 0.35)
    jt_ = C.frame_jamb_thickness(spec)
    world = C.add_floor_and_wall(model, spec, hole=(-Wo / 2 - jt_, Wo / 2 + jt_, 0.0, Ho + jt_))
    pair = spec["kinematics"].get("pair", True)
    C.add_frame(model, spec, 1.0, world, with_stop=False, strike_pockets=None, u=1.0)
    bodies = []
    leaves = (("leaf_a", 1.0, -Wo / 2), ("leaf_b", -1.0, Wo / 2)) if pair else (("leaf", 1.0, -Wo / 2),)
    for name, u, hx in leaves:
        b = Body(name, None, (hx, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Saloon leaf")
        j = hinge_joint(spec, phys, u, 1.0, 0.0, name=f"{name}_hinge", both_ways=True)
        j.pos = (u * (t / 2 + 0.006), 0.0, 0.0)   # centre-hung pivot at the leaf edge; corners sweep t/2, jamb gap t/2 + 6 mm
        j.stiffness = phys["closer"]["spring_stiffness_Nm_per_rad"]
        j.springref = 0.0
        j.damping = phys["closer"]["damping_closing"] * 0.5 + phys["hinge"]["air_damping_Nms_per_rad"]
        b.joint = j
        model.add_body(b)
        C.add_leaf_geoms(model, b, spec, leaf, u, u * (t / 2 + 0.006), zb, phys, name_prefix=name)
        C.add_hinge_visuals(model, world, b, spec, (u * (t / 2 + 0.006), 0), Hh, zb, 1.0, u)
        if spec["operator"]["model"] == "push_plate":
            for f in (-1.0, 1.0):
                C.add_pull(model, b, H.OPERATORS["push_plate"], u, u * (t / 2 + 0.006 + W - 0.12), zb + Hh * 0.6, t, f, name=f"{name}_push_plate")
        if "kick_plate" in spec["extras"]:
            for f in (-1.0, 1.0):
                C.add_kick_plate(model, b, u, u * (t / 2 + 0.006), zb, W, t, f, name=f"{name}_kick_{'p' if f > 0 else 'n'}")
        b.sites.append(Site(f"{name}_push_site", (u * (t / 2 + 0.006 + W * 0.75), -(t / 2), zb + Hh * 0.6), QUAT_ID, 0.015, "push"))
        bodies.append(b)
    world.sites.append(Site("approach_point", (0, -1.5, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (0, 1.5, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (0, 0, Ho / 2), QUAT_ID, 0.02, "pass_plane"))
    model.meta.update({"primary_joint": bodies[0].joint.name, "secondary_joint": bodies[1].joint.name if len(bodies) > 1 else None, "both_ways": True})
    return bodies


def _dog_shaft(d, mat, t: float, v: float, edge_dir: float):
    """The dog's shaft through the door and the crank out to its wedge.

    A dog is a lever on a shaft that passes through the door; the wedge sits on a crank on that shaft.  Without them
    the lever hung 9 mm off the door face and the wedge floated with nothing between it and the lever."""
    wy = t / 2 + 0.034
    # the shaft runs from the lever (always on the robot face) through the door to the wedge (on the sealing face):
    # on a door whose two faces are the other way round those are OPPOSITE sides, and a shaft on one side only would
    # leave the lever floating beside the wedge
    lo = min(-(t / 2 + 0.012), -v * (t / 2 + 0.046), -t / 2 + 0.004)
    hi = max(-(t / 2 + 0.012), -v * (t / 2 + 0.046), t / 2 - 0.004)
    d.geoms.append(C.cyl(f"{d.name}_shaft", (0, (lo + hi) / 2, 0), 0.008, (hi - lo) / 2, mat, (0, 1, 0), 7800, False, True, ALL_TIERS, "lock", "Dog shaft"))
    d.geoms.append(C.box(f"{d.name}_crank", (edge_dir * 0.026, -v * wy, 0), (0.026, 0.012, 0.018), mat, 7800, False, True, ALL_TIERS, "lock", "Dog crank"))


def build_ship(spec, phys, model: Model):
    """Watertight door: heavy leaf on raised coaming, N dog levers (or wheel) wedging against frame cleats."""
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    sill = op.get("sill_height", 0.3)
    u, v = _uv(spec)
    world = C.add_floor_and_wall(model, spec, hole=(-Wo / 2, Wo / 2, sill, sill + Ho))
    hx = u * (-Wo / 2)
    y_pin = v * (t / 2 + 0.02)
    lb = Body("leaf", None, (hx, 0, sill), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Watertight door leaf")
    lb.joint = hinge_joint(spec, phys, u, v, y_pin, name="leaf_hinge")
    lb.joint.pos = (-u * 0.03, y_pin, 0.0)
    model.add_body(lb)
    C.add_leaf_geoms(model, lb, spec, leaf, u, u * 0.004, 0.004, phys, name_prefix="leaf")
    # gasket (visual soft) around the leaf on the -v face; frame flange
    fm = C.mat_from_material(model, "steel_painted", "mat_frame")
    depth = 0.012
    flange = 0.06
    for nm, c, h in (("flange_l", (-Wo / 2 - flange / 2, -v * (t / 2 + 0.006), sill + Ho / 2), (flange / 2, 0.006, Ho / 2 + flange)), ("flange_r", (Wo / 2 + flange / 2, -v * (t / 2 + 0.006), sill + Ho / 2), (flange / 2, 0.006, Ho / 2 + flange)), ("flange_t", (0, -v * (t / 2 + 0.006), sill + Ho + flange / 2), (Wo / 2, 0.006, flange / 2)), ("flange_b", (0, -v * (t / 2 + 0.006), sill - flange / 2), (Wo / 2, 0.006, flange / 2))):
        world.geoms.append(C.box(nm, c, h, fm, 7850, True, True, ALL_TIERS, "frame", "Frame flange (door seats against it)", solref=(0.02, 1.0)))
    world.geoms.append(C.box("coaming", (0, 0, sill / 2), (Wo / 2 + 0.05, 0.03, sill / 2), fm, 7850, True, True, ALL_TIERS, "frame", "Coaming (raised sill)"))
    gm = C.mat_from_material(model, "rubber", "mat_gasket")
    for nm_, cx_, cz_, hx_, hz__ in (("l", u * 0.014, 0.004 + Hh / 2, 0.01, Hh / 2 - 0.01), ("r", u * (0.004 + W - 0.01), 0.004 + Hh / 2, 0.01, Hh / 2 - 0.01), ("b", u * (0.004 + W / 2), 0.014, W / 2 - 0.01, 0.01), ("t", u * (0.004 + W / 2), 0.004 + Hh - 0.01, W / 2 - 0.01, 0.01)):
        lb.geoms.append(C.box(f"gasket_{nm_}", (cx_, -v * (t / 2 + 0.004), cz_), (hx_, 0.004, hz__), gm, 1100, False, True, FULL_SIMPLE, "seal", "Knife-edge gasket"))
    # dogs
    n_dogs = spec["kinematics"].get("dogs", 0)
    opm = H.OPERATORS[spec["operator"]["model"]]
    mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
    dog_joints = []
    positions = []
    if n_dogs:
        per_side = max(1, n_dogs // 2)
        for k in range(per_side):
            z = 0.2 + (Hh - 0.4) * (k + 0.5) / per_side
            positions.append((u * (0.004 + W - 0.06), z, u))      # latch edge dogs
            positions.append((u * (0.004 + 0.06), z, -u))         # hinge edge dogs
        positions = positions[:n_dogs]
    for k, (xd, zd, edge_dir) in enumerate(positions):
        d = Body(f"dog_{k}", lb.name, (xd, 0, zd), QUAT_ID, None, [], [], ALL_TIERS, "lock", f"Dog {k + 1}")
        d.joint = Joint(f"dog_{k}_hinge", "hinge", (0, -edge_dir * u, 0), (0, 0, 0), (0.0, 1.5708), damping=0.5, frictionloss=1.5, role="lock", label=f"Dog {k + 1} (0 = dogged, + = released)")
        # lever on robot face (-1), wedge beyond edge
        key, mesh = MESH.lever_mesh(shape="dog", length=0.22, diameter=0.025, rose_diameter=0.06, standoff=0.05)
        d.geoms.append(C.mesh_geom(f"dog_{k}_lever", key, mesh, (0, -1.0 * (t / 2 + 0.009), 0), C.q_face(-1.0, edge_dir), mat, 7800, False, ALL_TIERS, "operator", "Dog lever"))
        d.geoms.append(Geom(f"dog_{k}_lever_col", "capsule", (0.0125, 0.10), (-edge_dir * 0.11, -1.0 * (t / 2 + 0.05), 0), tuple(quat_z_to((1, 0, 0))), mat, True, False, 7800, None, (0.7, 0.01, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", "Dog lever grip"))
        d.sites.append(Site(f"dog_{k}_grip", (-edge_dir * 0.18, -1.0 * (t / 2 + 0.05), 0), QUAT_ID, 0.012, "grip"))
        # wedge: box protruding beyond the leaf edge over the flange when dogged (pointing +edge_dir), lying on the -v face plane
        wy = t / 2 + 0.034
        d.geoms.append(C.box(f"dog_{k}_wedge", (edge_dir * 0.06, -v * wy, 0), (0.05, 0.012, 0.02), mat, 7800, True, True, ALL_TIERS, "lock", "Dog wedge"))
        _dog_shaft(d, mat, t, v, edge_dir)
        model.add_body(d)
        dog_joints.append(d.joint.name)
        # frame cleat: block on the +v side of the wedge so the door can't open while dogged
        cx = hx + xd + edge_dir * 0.08
        world.geoms.append(C.box(f"cleat_{k}", (cx + edge_dir * 0.0075, -v * (wy - 0.012 - 0.005 - 0.003), zd + sill), (0.0275, 0.005, 0.025), fm, 7850, True, True, ALL_TIERS, "lock", "Dog cleat"))
        world.geoms.append(C.box(f"cleat_{k}_base", (cx + edge_dir * 0.0225, -v * (wy + 0.036), zd + sill), (0.0425, 0.018, 0.025), fm, 7850, True, True, ALL_TIERS, "lock", "Cleat base (welded to its bridge)"))
        world.geoms.append(C.box(f"cleat_{k}_bridge", (cx + edge_dir * 0.043, -v * (wy + 0.008), zd + sill), (0.008, 0.03, 0.025), fm, 7850, True, True, ALL_TIERS, "lock", "Cleat bridge (welded to the cleat)"))
    if spec["kinematics"].get("wheel_dogging"):
        wm = H.OPERATORS["wheel_ship_hatch"]
        wb = C.add_rotary_operator(model, lb, spec, phys, wm, u, v, u * (0.004 + W / 2), Hh / 2, t, [-1.0, 1.0], None, name="wheel")
        # wheel drives 4 dogs (auto-created) if no explicit dogs
        if not positions:
            for k, (xd, zd, edge_dir) in enumerate([(u * (0.004 + W - 0.05), Hh * 0.25, u), (u * (0.004 + W - 0.05), Hh * 0.75, u), (u * 0.05, Hh * 0.25, -u), (u * 0.05, Hh * 0.75, -u)]):
                d = Body(f"dog_{k}", lb.name, (xd, 0, zd), QUAT_ID, None, [], [], ALL_TIERS, "lock", f"Dog {k + 1}")
                d.joint = Joint(f"dog_{k}_hinge", "hinge", (0, -edge_dir * u, 0), (0, 0, 0), (0.0, 1.5708), damping=0.5, frictionloss=0.5, role="lock", label=f"Dog {k + 1} (wheel-driven)", robot_interactive=False)
                wy = t / 2 + 0.034
                d.geoms.append(C.box(f"dog_{k}_wedge", (edge_dir * 0.06, -v * wy, 0), (0.05, 0.012, 0.02), mat, 7800, True, True, ALL_TIERS, "lock", "Dog wedge"))
                _dog_shaft(d, mat, t, v, edge_dir)
                model.add_body(d)
                cx = hx + xd + edge_dir * 0.08
                world.geoms.append(C.box(f"cleat_{k}", (cx + edge_dir * 0.0075, -v * (wy - 0.012 - 0.005 - 0.003), zd + sill), (0.0275, 0.005, 0.025), fm, 7850, True, True, ALL_TIERS, "lock", "Dog cleat"))
                world.geoms.append(C.box(f"cleat_{k}_base", (cx + edge_dir * 0.0225, -v * (wy + 0.036), zd + sill), (0.0425, 0.018, 0.025), fm, 7850, True, True, ALL_TIERS, "lock", "Cleat base (welded to its bridge)"))
                world.geoms.append(C.box(f"cleat_{k}_bridge", (cx + edge_dir * 0.043, -v * (wy + 0.008), zd + sill), (0.008, 0.03, 0.025), fm, 7850, True, True, ALL_TIERS, "lock", "Cleat bridge (welded to the cleat)"))
                model.equalities.append(Equality("joint", f"wheel_dog_{k}", d.joint.name, wb.joint.name, (0, 1.5708 / wm.travel, 0, 0, 0), tiers=ALL_TIERS, label="dog = wheel * (90deg / wheel travel)"))
        model.meta["operator_joint"] = wb.joint.name
    # no decorative rivet where a dog's shaft comes through the plate
    dogs_ = [b for b in model.bodies if b.name.startswith("dog_") and b.parent == lb.name]
    lb.geoms = [g for g in lb.geoms if not ("_rivet_" in g.name and any(abs(g.pos[0] - b.pos[0]) < 0.035 and abs(g.pos[2] - b.pos[2]) < 0.035 for b in dogs_))]
    C.add_hinge_visuals(model, world, lb, spec, (lb.joint.pos[0], y_pin), Hh, 0.004, v, u)
    if "warning_placard" in spec["extras"]:
        pm = C.mat_rgba(model, "mat_placard", (0.95, 0.75, 0.05, 1), 0.5)
        lb.geoms.append(C.box("warning_placard", (u * (0.004 + W / 2), -1.0 * (t / 2 + 0.001), Hh * 0.8), (0.11, 0.001, 0.08), pm, 1000, False, True, FULL_ONLY, "decor", "Warning placard"))
    world.sites.append(Site("approach_point", (0, -1.5, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (0, 1.5, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (0, 0, sill + Ho / 2), QUAT_ID, 0.02, "pass_plane"))
    model.meta.update({"u": u, "v": v, "primary_joint": "leaf_hinge", "dog_joints": dog_joints, "operator_joint": model.meta.get("operator_joint", dog_joints[0] if dog_joints else None), "sill_height": sill})
    return lb


def build_vault(spec, phys, model: Model):
    """Vault / blast door: massive leaf, handwheel driving N bolts into frame pockets."""
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    u, v = _uv(spec)
    jt_ = C.frame_jamb_thickness(spec)
    depth_v = op["wall_thickness"]
    y_wall_v = -v * max(0.0, depth_v / 2 - t / 2 - 0.02)     # leaf flush with the swing-side frame face
    world = C.add_floor_and_wall(model, spec, wall_height=max(Ho + 0.8, 2.9), hole=C.frame_hole(spec, u, jt_), y_wall=y_wall_v)
    hx = u * (-Wo / 2)
    y_pin = v * (t / 2 + 0.065)
    lb = Body("leaf", None, (hx, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Vault door leaf")
    j = hinge_joint(spec, phys, u, v, y_pin, name="leaf_hinge")
    j.pos = (u * 0.006, y_pin, 0.0)
    j.armature = 0.5
    lb.joint = j
    model.add_body(lb)
    zb = 0.05
    C.add_leaf_geoms(model, lb, spec, leaf, u, u * 0.006, zb, phys, name_prefix="leaf")
    x_edge = u * (0.006 + W)
    opm = H.OPERATORS[spec["operator"]["model"]]
    pockets = []
    if opm.kind == "wheel":
        wb = C.add_rotary_operator(model, lb, spec, phys, opm, u, v, u * (0.006 + W * 0.55), Hh * 0.5, t, [-1.0], None, name="wheel")
        wb.joint.frictionloss = max(wb.joint.frictionloss, 5.0)
        n_bolts = spec["kinematics"].get("bolts", 4)
        bm = C.mat_from_material(model, "stainless", "mat_vault_bolt")
        throw = H.LATCHES[spec["latch"]["model"]].throw + max(0.0, Wo - W - 0.006 - 0.004)   # + strike gap of the thick leaf
        for k in range(n_bolts):
            z = zb + Hh * (k + 0.5) / n_bolts
            b = Body(f"bolt_{k}", lb.name, (x_edge, 0, z), QUAT_ID, None, [], [], ALL_TIERS, "lock", f"Bolt {k + 1}")
            b.joint = Joint(f"bolt_{k}_slide", "slide", (-u, 0, 0), (0, 0, 0), (0.0, throw), damping=20.0, frictionloss=25.0, role="lock", label=f"Bolt {k + 1} (0 = thrown)", robot_interactive=False)
            r = 0.016
            b.geoms.append(Geom(f"bolt_{k}_geom", "capsule", (r, (throw + 0.08) / 2 - r), (u * (throw - 0.08) / 2, 0, 0), tuple(quat_z_to((u, 0, 0))), bm, True, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "lock", "Vault bolt"))
            model.add_body(b)
            model.equalities.append(Equality("joint", f"wheel_bolt_{k}", b.joint.name, wb.joint.name, (0, throw / opm.travel, 0, 0, 0), tiers=ALL_TIERS, label="bolt = wheel * throw/travel"))
            pockets.append({"z": z, "h": 2 * r + 0.01, "w": 2 * r + 0.004, "depth": throw + 0.006, "ramp": False})
        model.meta["operator_joint"] = wb.joint.name
    else:
        # lever bolts (blast door): each lever (hinge about the door normal) drives a sliding bolt into a jamb pocket
        mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
        bm = C.mat_from_material(model, "stainless", "mat_vault_bolt")
        throw = 0.05 + max(0.0, Wo - W - 0.006 - 0.004)
        for k, z in enumerate([zb + Hh * 0.25, zb + Hh * 0.75]):
            d = Body(f"dog_{k}", lb.name, (x_edge - u * 0.12, 0, z), QUAT_ID, None, [], [], ALL_TIERS, "lock", f"Lever bolt {k + 1}")
            d.joint = Joint(f"dog_{k}_hinge", "hinge", (0, -u, 0), (0, 0, 0), (0.0, 1.5708), damping=1.0, frictionloss=6.0, role="lock", label=f"Lever bolt {k + 1} (0 = engaged, + = released; detent friction 6 N*m, not back-drivable)")
            key, mesh = MESH.lever_mesh(shape="dog", length=0.30, diameter=0.03, rose_diameter=0.08, standoff=0.06)
            d.geoms.append(C.mesh_geom(f"dog_{k}_lever", key, mesh, (0, -1.0 * t / 2, 0), C.q_face(-1.0, u), mat, 7800, False, ALL_TIERS, "operator", "Lever"))
            d.geoms.append(Geom(f"dog_{k}_lever_col", "capsule", (0.015, 0.14), (-u * 0.15, -1.0 * (t / 2 + 0.06), 0), tuple(quat_z_to((1, 0, 0))), mat, True, False, 7800, None, (0.7, 0.01, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", "Lever grip"))
            d.sites.append(Site(f"dog_{k}_grip", (-u * 0.25, -1.0 * (t / 2 + 0.06), 0), QUAT_ID, 0.012, "grip"))
            model.add_body(d)
            b = Body(f"bolt_{k}", lb.name, (x_edge, 0, z), QUAT_ID, None, [], [], ALL_TIERS, "lock", f"Bolt {k + 1}")
            b.joint = Joint(f"bolt_{k}_slide", "slide", (-u, 0, 0), (0, 0, 0), (0.0, throw), damping=20.0, frictionloss=25.0, role="lock", label=f"Bolt {k + 1} (0 = thrown)", robot_interactive=False)
            r = 0.016
            b.geoms.append(Geom(f"bolt_{k}_geom", "capsule", (r, (throw + 0.10) / 2 - r), (u * (throw - 0.10) / 2, 0, 0), tuple(quat_z_to((u, 0, 0))), bm, True, True, 7850.0, None, (0.3, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "lock", "Vault bolt"))
            model.add_body(b)
            model.equalities.append(Equality("joint", f"lever_bolt_{k}", b.joint.name, d.joint.name, (0, throw / 1.5708, 0, 0, 0), tiers=ALL_TIERS, label="bolt = lever * throw/(pi/2)"))
            pockets.append({"z": z, "h": 2 * r + 0.01, "w": 2 * r + 0.004, "depth": throw + 0.006, "ramp": False})
        model.meta["operator_joint"] = "dog_0_hinge"
    if opm.kind == "lever" and False:
        pass
    C.add_frame(model, spec, v, world, with_stop=True, strike_pockets=pockets, u=u)
    # big hinges
    hm = C.mat_from_material(model, "steel", "mat_hinge")
    for k, z in enumerate([zb + 0.3, zb + Hh - 0.3] if spec["hinge"]["count"] == 2 else [zb + 0.3, zb + Hh / 2, zb + Hh - 0.3]):
        # crane hinge barrel on the pin, outside the wall face; arm on the leaf's swing face
        lb.geoms.append(C.cyl(f"hinge_{k}", (u * 0.006, y_pin, z), 0.04, 0.12, hm, (0, 0, 1), 7850, False, True, FULL_SIMPLE, "hinge", "Crane hinge"))
        lb.geoms.append(C.box(f"hinge_{k}_arm", (u * 0.116, v * (t / 2 + 0.04), z), (0.11, 0.04, 0.05), hm, 7850, False, True, FULL_SIMPLE, "hinge", "Hinge arm"))
        # the barrel has to hang on something: frame brackets above and below it, bolted to the jamb face
        xb_ = hx + u * 0.006
        vdp = 1.0 if y_pin > 0 else -1.0
        face_ = C.mount_face(world, xb_, z, 0.05, 0.24, vdp, default=-1e9)
        y1_ = abs(y_pin) + 0.02
        if -1e8 < face_ < y1_ - 0.004:
            for dz in (-1, 1):
                world.geoms.append(C.box(f"hinge_{k}_bracket_{'t' if dz > 0 else 'b'}", (xb_, vdp * (face_ + y1_) / 2, z + dz * 0.17),
                                         (0.05, (y1_ - face_) / 2, 0.05), hm, 7850, False, True, FULL_SIMPLE, "hinge", "Crane hinge frame bracket"))
    world.sites.append(Site("approach_point", (0, -1.5, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (0, 1.5, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (0, 0, Ho / 2), QUAT_ID, 0.02, "pass_plane"))
    model.meta.update({"u": u, "v": v, "primary_joint": "leaf_hinge", "handle_height": Hh * 0.5})
    return lb


def build_gate_or_fence(spec, phys, model: Model):
    """Outdoor swing gate: posts + fence panels instead of wall; ground clearance; reuses swing builder for the leaf."""
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    Hh = spec["leaf"]["height"]
    gc = op.get("ground_clearance", 0.05)
    lb = build_swing_single(spec, phys, model)
    world = model.body("world_env")
    # remove any frame geoms added by add_frame (gates have posts) -> we passed outdoor so add_floor_and_wall skipped walls; add_frame added jambs: convert to posts by keeping them (they act as posts). Add fence panels.
    fm = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
    ps = op["frame"].get("post_size", 0.1)
    for sgn in (-1, 1):
        # fence run 2 m each side
        fam_ctx = spec["context"]
        if fam_ctx in ("chain_link", "pool_safety") or spec["family"] == "gate_sliding":
            mm = C.mat_rgba(model, "mat_fence_mesh", (0.6, 0.62, 0.64, 0.5), 0.7, 0.8, True)
            world.geoms.append(C.box(f"fence_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + ps + 1.0), 0, gc + Hh / 2), (1.0, 0.002, Hh / 2), mm, 100, True, True, FULL_SIMPLE, "wall", "Fence"))
            world.geoms.append(C.cyl(f"fence_post_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + ps + 2.0), 0, (Hh + gc) / 2), 0.025, (Hh + gc) / 2, fm, (0, 0, 1), 7850, True, True, FULL_SIMPLE, "wall", "Fence post"))
        elif fam_ctx in ("wrought_iron",):
            for k in range(14):
                world.geoms.append(C.cyl(f"fence_bar_{'r' if sgn > 0 else 'l'}_{k}", (sgn * (Wo / 2 + ps + 0.1 + k * 0.14), 0, gc + Hh / 2), 0.008, Hh / 2, fm, (0, 0, 1), 7850, True, True, FULL_ONLY, "wall", "Fence bar"))
            world.geoms.append(C.box(f"fence_rail_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + ps + 1.0), 0, gc + Hh - 0.05), (1.0, 0.012, 0.012), fm, 7850, True, True, FULL_SIMPLE, "wall", "Fence rail"))
            world.geoms.append(C.box(f"fence_rail_b_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + ps + 1.0), 0, gc + 0.1), (1.0, 0.012, 0.012), fm, 7850, True, True, FULL_SIMPLE, "wall", "Fence rail"))
        else:
            wm = C.mat_from_material(model, "cedar", "mat_fence")
            n = 12
            for k in range(n):
                world.geoms.append(C.box(f"fence_picket_{'r' if sgn > 0 else 'l'}_{k}", (sgn * (Wo / 2 + ps + 0.08 + k * 0.16), 0, gc + Hh / 2), (0.04, 0.01, Hh / 2), wm, 400, True, True, FULL_ONLY, "wall", "Fence picket"))
            world.geoms.append(C.box(f"fence_rail_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + ps + 1.0), 0.015, gc + Hh - 0.15), (1.0, 0.015, 0.04), wm, 400, True, True, FULL_SIMPLE, "wall", "Fence rail"))
            world.geoms.append(C.box(f"fence_rail_b_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + ps + 1.0), 0.015, gc + 0.2), (1.0, 0.015, 0.04), wm, 400, True, True, FULL_SIMPLE, "wall", "Fence rail"))
    return lb


def build_stall(spec, phys, model: Model):
    """Toilet partition: pilasters + panel door with gravity hinge + slide latch."""
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo = op["width"]
    zb = leaf.get("bottom_clearance", 0.3)
    u, v = _uv(spec)
    world = C.add_floor_and_wall(model, spec, hole=(-Wo / 2 - 0.1, Wo / 2 + 0.1, 0.0, max(2.1, zb + Hh + 0.15)))
    # pilasters
    pm = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
    pw = 0.1
    top_z = max(1.92, zb + Hh + 0.05)
    for sgn in (-1, 1):
        world.geoms.append(C.box(f"pilaster_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + pw / 2), 0, 0.1 + (top_z - 0.1) / 2), (pw / 2, t / 2, (top_z - 0.1) / 2), pm, 1400, True, True, ALL_TIERS, "frame", "Pilaster"))
        world.geoms.append(C.box(f"pilaster_shoe_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + pw / 2), 0, 0.05), (pw / 2, t / 2 + 0.005, 0.05), C.mat_from_material(model, "stainless", "mat_shoe"), 7900, False, True, FULL_ONLY, "frame", "Pilaster shoe"))
    world.geoms.append(C.box("headrail", (0, 0, top_z + 0.02), (Wo / 2 + pw, 0.02, 0.02), C.mat_from_material(model, "aluminum", "mat_headrail"), 2700, True, True, FULL_SIMPLE, "frame", "Headrail"))
    # side partition panels
    for sgn in (-1, 1):
        world.geoms.append(C.box(f"partition_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + pw + 0.75), 0.75, 0.3 + 1.5 / 2 + 0.05), (0.006, 0.75, 0.75), pm, 1400, True, True, FULL_SIMPLE, "wall", "Partition panel"))
    hx = u * (-Wo / 2)
    lb = Body("leaf", None, (hx, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Stall door")
    riser = Body("leaf_riser", None, (hx, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "hinge", "Gravity hinge carrier")
    riser.joint = Joint("leaf_rise", "slide", (0, 0, 1), (0, 0, 0), (0.0, 0.03), damping=1.0, armature=0.5, role="mechanism", label="Gravity-hinge lift (coupled)", robot_interactive=False)
    model.add_body(riser)
    lb.parent = riser.name
    lb.pos = (0, 0, 0)
    j = hinge_joint(spec, phys, u, v, 0.0, name="leaf_hinge", tilt_deg=0.0)
    j.pos = (u * 0.05, 0.0, 0.0)   # gravity pivot ~44 mm inside the leaf edge: corner sweep radius 49 mm < 50 mm to the pilaster
    rise_per_rad = 0.010 / (math.pi / 2)
    if spec["kinematics"].get("rest_angle_deg"):
        j.initial = math.radians(spec["kinematics"]["rest_angle_deg"])
        # the coupling is relative to the joints' qpos0: a leaf resting open is already lifted by its gravity hinge,
        # otherwise the rise would have to go negative (below its 0 limit) when the leaf closes - a locked coupling
        riser.joint.initial = rise_per_rad * j.initial
    lb.joint = j
    model.add_body(lb)
    model.equalities.append(Equality("joint", "leaf_rise_couple", "leaf_rise", "leaf_hinge", (0.0, rise_per_rad, 0, 0, 0), tiers=ALL_TIERS, label="gravity hinge: 10 mm rise per 90 deg"))
    # heel gap: the leaf turns on a pivot 50 mm inside the pilaster face, so its heel corner sweeps
    # hypot(50 - gap, t/2) - a flat 6 mm gap left 0.8 mm of running clearance on 44 mm doors
    x_heel = max(0.006, C.pivot_heel_gap(0.05, t))
    C.add_leaf_geoms(model, lb, spec, leaf, u, u * x_heel, zb, phys, name_prefix="leaf")
    x_edge = u * (x_heel + W)
    hz = spec["operator"]["height"]
    opm = H.OPERATORS[spec["operator"]["model"]]
    lk, engaged, release = _lock_state(spec)
    mat = C.mat_from_material(model, "stainless", "mat_op_stainless")
    if opm.kind != "slide_bolt_handle":
        for f in (-1.0, 1.0):
            C.add_pull(model, lb, opm, u, x_edge - u * 0.08, hz - 0.15, t, f, name="pull")
    if True:
        inside = v   # slide latch on the face the door swings toward (opposite the stop strip on the pilaster)
        sb = Body("slide_latch", lb.name, (x_edge - u * 0.05, inside * (t / 2 + 0.008), hz), QUAT_ID, None, [], [], ALL_TIERS, "latch", "Slide latch")
        eng = engaged
        sb.joint = Joint("slide_latch_slide", "slide", (-u, 0, 0), (0, 0, 0), (0.0, 0.03), damping=0.5, frictionloss=0.5, role="lock", label="Slide latch (0 = latched, + = open)", initial=0.0 if eng else 0.03, modeled_at=0.0 if eng else 0.03)
        xo = 0 if eng else -u * 0.03
        sb.geoms.append(C.box("slide_latch_bar", (u * 0.03 + xo, 0, 0), (0.04, 0.004, 0.012), mat, 7900, True, True, ALL_TIERS, "latch", "Latch bar"))
        sb.geoms.append(Geom("slide_latch_knob", "capsule", (0.005, 0.012), (xo, inside * 0.012, 0), tuple(quat_z_to((0, inside, 0))), mat, True, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", "Latch knob"))
        sb.sites.append(Site("grip_latch", (xo, inside * 0.03, 0), QUAT_ID, 0.01, "grip"))
        model.add_body(sb)
        # housing on the door: back plate + guide bracket the flat bar slides through (partition slide latch)
        lb.geoms.append(C.box("slide_latch_plate", (x_edge - u * 0.045, inside * (t / 2 + 0.001), hz), (0.048, 0.001, 0.022), mat, 7900, False, True, FULL_SIMPLE, "latch", "Slide latch back plate"))
        C.add_guide_loop(lb.geoms, "slide_latch_guide", (x_edge - u * 0.05, inside * t / 2, hz), (u, 0, 0), (0, inside, 0), 0.026, 0.015, 0.015, mat, 0.003, 0.010, False, FULL_SIMPLE, "latch", "Slide latch guide")
        # keeper on pilaster
        yb_ = inside * (t / 2 + 0.008)
        for sy in (-1, 1):
            world.geoms.append(C.box(f"latch_keeper_{'p' if sy > 0 else 'n'}", (u * (Wo / 2 + 0.02), yb_ + sy * 0.0075, hz), (0.02, 0.0025, 0.02), mat, 7900, True, True, ALL_TIERS, "latch", "Keeper channel"))
        world.geoms.append(C.box("latch_keeper_base", (u * (Wo / 2 + 0.02), yb_, hz - 0.024), (0.02, 0.01, 0.003), mat, 7900, True, True, ALL_TIERS, "latch", "Keeper base"))
        model.meta["operator_joint"] = sb.joint.name
        if not (release or not engaged):
            sb.joint.range = (0.0, 0.001)
    # stop strip on the hinge-side pilaster (door closes against it)
    world.geoms.append(C.box("stop_strip", (u * (Wo / 2 - 0.01), -v * (t / 2 + 0.006), zb + Hh / 2), (0.01, 0.006, Hh / 2), pm, 1400, True, True, ALL_TIERS, "frame", "Stop strip"))
    if "coat_hook" in spec["extras"]:
        key, mesh = MESH.coat_hook_mesh()
        lb.geoms.append(C.mesh_geom("coat_hook", key, mesh, (u * (0.006 + W / 2), 1.0 * t / 2, zb + Hh - 0.15), C.q_face(1.0, u), mat, 7000, False, FULL_ONLY, "decor", "Coat hook"))
    C.add_hinge_visuals(model, world, lb, spec, (u * 0.02, 0.0), Hh, zb, v, u)
    world.sites.append(Site("approach_point", (0, -1.2, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (0, 1.0, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (0, 0, 1.0), QUAT_ID, 0.02, "pass_plane"))
    model.meta.update({"u": u, "v": v, "primary_joint": "leaf_hinge", "handle_height": hz})
    return lb

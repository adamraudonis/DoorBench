"""Sliding, folding, rotor, vertical-lift and horizontal-hinge families."""
from __future__ import annotations

import math

import numpy as np

from ..ir import (Body, Geom, Joint, Site, Equality, Tendon, Model, ALL_TIERS, FULL_ONLY, FULL_SIMPLE, QUAT_ID,
                  quat_from_axis_angle, quat_z_to)
from .. import materials as M
from .. import hardware as H
from . import common as C
from . import meshes as MESH


def _sites(world, Ho, ya=-1.5, yg=1.5):
    world.sites.append(Site("approach_point", (0, ya, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (0, yg, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (0, 0, Ho / 2), QUAT_ID, 0.02, "pass_plane"))


def slide_joint(spec, phys, axis, travel, name, initial=0.0, counterbalance=None, mass=None):
    rf = phys["roller"]
    j = Joint(name, "slide", tuple(axis), (0, 0, 0), (0.0, travel), damping=rf["viscous_damping_N_s_per_m"], frictionloss=rf["coulomb_force_N"], armature=0.02, role="primary", label="Slide (0 = closed, + = open)", initial=initial)
    if counterbalance and mass:
        cb = counterbalance
        # spring force ~ cb*m*g at closed, declining 30% over travel
        k = 0.3 * cb * mass * 9.81 / max(travel, 0.1)
        j.stiffness = k
        j.springref = cb * mass * 9.81 / k
        j.notes = f"counterbalance spring ~{cb:.0%} of weight"
    return j


def _add_hook(model, world, leaf_b, name, dir_, x_latch_edge, xc, yl, z, engaged, driver_joint, coeff):
    """Hook bolt on the leaf's latch edge: arm past the edge, vertical tip curling up behind a keeper bar on the jamb.
    Opening (+dir_) pulls the tip's inner face against the keeper.  Releasing rotates the hook down/outward."""
    hm = C.mat_from_material(model, "stainless", "mat_hook")
    hk = Body(f"{name}_hook", leaf_b.name, (x_latch_edge, 0, z), QUAT_ID, None, [], [], ALL_TIERS, "latch", "Hook bolt")
    hk.joint = Joint(f"{name}_hook_hinge", "hinge", (0, -dir_, 0), (0, 0, 0), (0.0, 1.0), damping=0.05, frictionloss=0.1, role="lock", label="Hook (0 = hooked, + = released)", initial=0.0 if engaged else 1.0, robot_interactive=False)
    # arm and tip at pivot height: the keeper load passes through the pivot (no back-driving torque; self-centring)
    hk.geoms.append(C.box(f"{name}_hook_arm", (-dir_ * 0.02, 0, -0.016), (0.022, 0.005, 0.006), hm, 7900, True, True, ALL_TIERS, "latch", "Hook arm"))
    hk.geoms.append(C.box(f"{name}_hook_tip", (-dir_ * 0.036, 0, 0.0), (0.006, 0.005, 0.018), hm, 7900, True, True, ALL_TIERS, "latch", "Hook tip"))
    model.add_body(hk)
    model.equalities.append(Equality("joint", f"{name}_hook_couple", hk.joint.name, driver_joint, (0, coeff, 0, 0, 0), tiers=ALL_TIERS, label="hook = driver * coeff"))
    world.geoms.append(C.box(f"{name}_hook_keeper", (xc + x_latch_edge - dir_ * 0.018, yl, z), (0.008, 0.015, 0.008), hm, 7900, True, True, ALL_TIERS, "latch", "Hook keeper bar"))
    if abs(yl) < 0.05:
        model.meta.setdefault("_jamb_pockets", []).append((z, 0.12, yl))
    return hk


# ---------------------------------------------------------------------------
# Sliding (single / bypass / patio / barn / pocket / auto / elevator / gate)
# ---------------------------------------------------------------------------
def build_sliding(spec, phys, model: Model):
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    wt = op["wall_thickness"]
    kin = spec["kinematics"]
    fam = spec["family"]
    ctx = spec.get("context", "")
    track = kin.get("track", "top_hung")
    n = leaf.get("count", 1)
    opens_left = spec["hinge"]["side"] == "left"
    s_open = -1.0 if opens_left else 1.0     # direction the leaf moves to open
    outdoor = bool(op.get("outdoor"))
    m_leaf = phys["mass"]["total_kg"]
    fixed_panel = bool(op.get("fixed_panel"))
    center = fam == "elevator" and kin.get("center_opening") or fam == "automatic_sliding" and kin.get("bi_parting")
    # y plane of the leaf(s)
    if track in ("surface_flat_track", "top_hung_industrial"):
        y_leaf = -(wt / 2 + 0.035)          # barn / industrial: hangs in front of the wall on the robot side
    elif fam in ("gate_sliding",):
        y_leaf = -(0.06)
    else:
        y_leaf = 0.0
    # wall + hole; for pockets create a cavity in the wall on the opening side
    wall_half = 2.5 if W < 2.0 else max(2.5, W * 1.2 + Wo / 2)
    jamb_t = 0.019 if op["frame"]["kind"].startswith(("wood", "kamoi")) else 0.045
    latch_side_ = -s_open if not (fam == "elevator" and kin.get("center_opening") or fam == "automatic_sliding" and kin.get("bi_parting")) else 0
    SLIDE_STUD = 0.06
    hole_x0 = -Wo / 2 - jamb_t - (SLIDE_STUD if latch_side_ < 0 else 0.0)
    hole_x1 = Wo / 2 + jamb_t + (SLIDE_STUD if latch_side_ > 0 else 0.0)
    world = C.add_floor_and_wall(model, spec, wall_half_width=wall_half, outdoor=outdoor, hole=(hole_x0, hole_x1, 0.0, Ho + jamb_t))
    mat_frame = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
    jamb_pockets = []   # (z, h) pockets in the latch-side jamb for hooks (built after the leaves)
    if track == "top_hung_pocket":
        # rebuild the wall on the pocket side as two skins
        side_x0 = s_open * (Wo / 2)
        pocket_len = W + 0.05
        # remove the wall segment on that side and re-add as skins beyond the pocket + solid further
        world.geoms = [g for g in world.geoms if g.name not in (("wall_left" if opens_left else "wall_right"),)]
        wm = "mat_wall"
        for sy in (-1, 1):
            world.geoms.append(C.box(f"pocket_skin_{'p' if sy > 0 else 'n'}", (side_x0 + s_open * pocket_len / 2, sy * (wt / 2 - 0.008), max(Ho + 0.6, 2.7) / 2), (pocket_len / 2, 0.008, max(Ho + 0.6, 2.7) / 2), wm, 800, True, True, ALL_TIERS, "wall", "Pocket wall skin"))
        far = side_x0 + s_open * pocket_len
        world.geoms.append(C.box("wall_pocket_far", ((far + s_open * wall_half) / 2, 0, max(Ho + 0.6, 2.7) / 2), (abs(s_open * wall_half - far) / 2, wt / 2, max(Ho + 0.6, 2.7) / 2), wm, 800, True, True, ALL_TIERS, "wall", "Wall"))
    # frame / jambs (latch-side jamb is built after the leaves so hook pockets can be cut into it)
    latch_side = -s_open if not center else 0
    if fam not in ("gate_sliding",):
        for sgn, nm in ((-1, "l"), (1, "r")):
            if track == "top_hung_pocket" and sgn == s_open:
                continue
            if sgn == latch_side:
                continue
            world.geoms.append(C.box(f"jamb_{nm}", (sgn * (Wo / 2 + jamb_t / 2), 0, Ho / 2), (jamb_t / 2, wt / 2, Ho / 2), mat_frame, 300, True, True, ALL_TIERS, "frame", "Jamb"))
        world.geoms.append(C.box("jamb_head", (0, 0, Ho + jamb_t / 2), (Wo / 2 + jamb_t, wt / 2, jamb_t / 2), mat_frame, 300, True, True, ALL_TIERS, "frame", "Head"))
    else:
        pm = mat_frame
        for sgn in (-1, 1):
            world.geoms.append(C.box(f"post_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + 0.06), 0.08, (Hh + 0.1) / 2 + 0.1), (0.06, 0.06, (Hh + 0.1) / 2 + 0.1), pm, 7850, True, True, ALL_TIERS, "frame", "Gate post (beside the leaf path)"))
    # track hardware
    tm = C.mat_from_material(model, "black_matte_metal" if track == "surface_flat_track" else "aluminum", "mat_track")
    if track == "surface_flat_track":
        L = 2 * W + 0.1
        world.geoms.append(C.box("flat_track", (0, y_leaf + 0.02, Ho + 0.12), (L / 2, 0.004, 0.02), tm, 7850, True, True, FULL_SIMPLE, "track", "Flat track"))
        for k in range(int(L / 0.6) + 1):
            world.geoms.append(C.cyl(f"track_standoff_{k}", (-L / 2 + k * 0.6, y_leaf + 0.03, Ho + 0.12), 0.012, 0.012, tm, (0, 1, 0), 7850, False, True, FULL_ONLY, "track", "Track standoff"))
    elif track in ("top_hung_bypass", "top_hung", "top_hung_industrial", "auto_header", "elevator_hanger_track", "sectional_vertical_lift"):
        world.geoms.append(C.box("track_header", (0, 0, Ho + jamb_t + 0.04), (Wo / 2 + W / 2 + 0.05, 0.04 if track != "auto_header" else 0.09, 0.04 if track != "auto_header" else 0.09), tm, 2700, True, True, FULL_SIMPLE, "track", "Track header"))
    elif track == "bottom_rolling":
        world.geoms.append(C.box("bottom_rail", (0, y_leaf, 0.012), (Wo / 2 + jamb_t, 0.02, 0.012), tm, 2700, True, True, ALL_TIERS, "track", "Bottom rail"))
    elif track == "wood_groove_bottom":
        wm2 = C.mat_from_material(model, "hinoki", "mat_shikii")
        world.geoms.append(C.box("shikii", (0, 0, 0.012), (Wo / 2 + 0.05, 0.05, 0.012), wm2, 410, True, True, ALL_TIERS, "track", "Shikii (bottom rail)"))
        world.geoms.append(C.box("kamoi", (0, 0, Ho + 0.03), (Wo / 2 + 0.05, 0.05, 0.03), wm2, 410, True, True, ALL_TIERS, "track", "Kamoi (head rail)"))
    elif track in ("cantilever", "bottom_rail"):
        world.geoms.append(C.box("gate_rail", (s_open * W / 2, y_leaf, 0.01), (W + 0.2, 0.03, 0.01), tm, 7850, True, True, ALL_TIERS, "track", "Rail"))
    # fixed panel (patio / auto sliding sidelites)
    if fixed_panel and fam in ("sliding_single", "automatic_sliding"):
        gm = C.mat_from_material(model, "glass_clear", "mat_fixed_glass")
        fm2 = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
        if fam == "sliding_single":
            xf = s_open * (Wo / 4)  # fixed panel occupies the half the leaf slides over
            if track == "wood_groove_bottom":
                gm = C.mat_from_material(model, "washi_paper", "mat_fixed_shoji")
            world.geoms.append(C.box("fixed_panel_glass", (xf, y_leaf + 0.04, Ho / 2), (Wo / 4 - 0.02, 0.003, Ho / 2 - 0.05), gm, 2500, True, True, ALL_TIERS, "glass", "Fixed panel"))
            world.geoms.append(C.box("fixed_panel_stile", (xf - s_open * (Wo / 4 - 0.03), y_leaf + t / 2 + 0.025, Ho / 2), (0.03, 0.012, Ho / 2), fm2, 1400, True, True, FULL_SIMPLE, "frame", "Fixed panel stile"))
        else:
            for sgn in (-1, 1):
                xf = sgn * (Wo / 4 + Wo / 8)
                world.geoms.append(C.box(f"sidelite_{'r' if sgn > 0 else 'l'}", (xf, 0.03, Ho / 2), (Wo / 8 - 0.02, 0.003, Ho / 2 - 0.05), gm, 2500, True, True, ALL_TIERS, "glass", "Sidelite"))
    # leaves
    bodies = []
    pockets_keeper = []
    opm = H.OPERATORS[spec["operator"]["model"]]
    lk = H.LOCKS[spec["lock"]["model"]]
    engaged = bool(spec["lock"].get("engaged"))
    release = bool(spec["lock"].get("robot_side_release"))
    lt = H.LATCHES[spec["latch"]["model"]]
    travel = kin["travel_m"]
    if center:
        # two leaves meeting at center, moving apart
        leaf_defs = [("leaf_a", -1.0, -W / 2, y_leaf), ("leaf_b", 1.0, W / 2, y_leaf)]
    elif fam == "sliding_bypass":
        leaf_defs = []
        for k in range(n):
            xc = -Wo / 2 + W / 2 + k * (W - 0.03)
            yk = y_leaf + ((k % 2) - 0.5) * (t + 0.008) * (1 if n > 1 else 0)
            leaf_defs.append((f"leaf_{k}", 1.0 if k % 2 == 0 else -1.0, xc, yk))
    elif fam == "sliding_single" and fixed_panel:
        # leaf covers the half opposite to the fixed panel; opens by sliding over the fixed panel
        leaf_defs = [("leaf", s_open, -s_open * (Wo / 4), y_leaf)]
    else:
        leaf_defs = [("leaf", s_open, 0.0, y_leaf)]
    for name, dir_, xc, yl in leaf_defs:
        b = Body(name, None, (xc, yl, 0.0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Sliding leaf")
        zb = 0.012 if track != "bottom_rolling" else 0.03
        if fam == "elevator":
            zb = 0.03
        if track == "wood_groove_bottom":
            zb = 0.027
        if fam in ("gate_sliding",):
            zb = op.get("ground_clearance", 0.08)
        if fam not in ("gate_sliding",) and track != "wood_groove_bottom" and zb + Hh > Ho - 0.004:
            zb = max(0.005, Ho - 0.004 - Hh)
        tr = travel if fam != "sliding_bypass" else W - 0.03
        j = slide_joint(spec, phys, (dir_, 0, 0), tr, f"{name}_slide")
        if fam == "sliding_bypass":
            j.range = (-(W - 0.03), W - 0.03) if 0 < leaf_defs.index((name, dir_, xc, yl)) < n - 1 else ((0.0, W - 0.03) if leaf_defs.index((name, dir_, xc, yl)) == 0 else (0.0, W - 0.03))
            j.axis = (1.0, 0, 0) if leaf_defs.index((name, dir_, xc, yl)) == 0 else (-1.0, 0, 0)
        b.joint = j
        model.add_body(b)
        C.add_leaf_geoms(model, b, spec, leaf, 1.0, -W / 2, zb, phys, name_prefix=name)
        eb_pockets = spec["latch"]["model"] == "electric_bolt" and not center and fam != "sliding_bypass"
        # hangers / rollers (visual)
        rm = C.mat_from_material(model, "steel_galvanized", "mat_roller")
        if track == "surface_flat_track":
            for k, xr in enumerate((-W / 2 + 0.12, W / 2 - 0.12)):
                b.geoms.append(C.cyl(f"{name}_hanger_wheel_{k}", (xr, 0.0 + 0.0, zb + Hh + 0.12), 0.05, 0.01, rm, (0, 1, 0), 7850, False, True, FULL_SIMPLE, "track", "Hanger wheel"))
                b.geoms.append(C.box(f"{name}_hanger_strap_{k}", (xr, -0.012, zb + Hh + 0.05), (0.02, 0.004, 0.07), rm, 7850, False, True, FULL_SIMPLE, "track", "Hanger strap"))
        elif track == "bottom_rolling":
            for k, xr in enumerate((-W / 2 + 0.1, W / 2 - 0.1)):
                b.geoms.append(C.cyl(f"{name}_roller_{k}", (xr, 0, zb - 0.012), 0.015, 0.008, rm, (0, 1, 0), 7850, False, True, FULL_ONLY, "track", "Roller"))
        bodies.append(b)
        # operator(s)
        hz = spec["operator"]["height"]
        x_lead = dir_ * (-W / 2 + 0.08)     # leading edge (edge toward the jamb it latches to) is opposite to the opening direction
        x_lead_edge = -dir_ * W / 2 * -1     # placeholder
        x_latch_edge = -dir_ * W / 2   # edge that meets the strike jamb / other leaf
        if eb_pockets:
            # electric DROP bolt (world-fixed solenoid above the leaf face) into a keeper bracket on the leaf: a bolt
            # perpendicular to the travel is the only kind that can hold a sliding leaf
            ebm = C.mat_from_material(model, "stainless", "mat_ebolt")
            f_eb = -1.0 if abs(yl) >= 0.05 else 1.0          # surface-hung leaves: keeper on the robot face (wall behind)
            x_b = xc + x_latch_edge + dir_ * 0.08
            y_b = yl + f_eb * (t / 2 + 0.014)
            zk = zb + Hh - 0.10
            eb = Body(f"{name}_electric_bolt", None, (x_b, y_b, zk + 0.035), QUAT_ID, None, [], [], ALL_TIERS, "lock", "Electric drop bolt")
            eb.joint = Joint(f"{name}_electric_bolt_slide", "slide", (0, 0, 1), (0, 0, 0), (0.0, 0.04), damping=5.0, frictionloss=2.0, role="lock", label="Electric drop bolt (0 = dropped into the keeper; lifts on access-control release)", robot_interactive=False, initial=0.0)
            eb.geoms.append(Geom(f"{name}_electric_bolt_geom", "capsule", (0.008, 0.042), (0, 0, 0), (1, 0, 0, 0), ebm, True, True, 7900.0, None, (0.4, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "lock", "Drop bolt"))
            model.add_body(eb)
            world.geoms.append(C.box(f"{name}_electric_bolt_housing", (x_b, y_b, zk + 0.13), (0.022, 0.018, 0.045), ebm, 7900, False, True, FULL_SIMPLE, "lock", "Solenoid housing"))
            for sx_ in (-1, 1):
                b.geoms.append(C.box(f"{name}_ebolt_keeper_{'p' if sx_ > 0 else 'n'}", (x_latch_edge + dir_ * 0.08 + sx_ * 0.018, f_eb * (t / 2 + 0.014), zk), (0.006, 0.014, 0.02), ebm, 7900, True, True, ALL_TIERS, "lock", "Keeper block"))
            b.geoms.append(C.box(f"{name}_ebolt_keeper_base", (x_latch_edge + dir_ * 0.08, f_eb * (t / 2 + 0.002), zk), (0.03, 0.002, 0.03), ebm, 7900, False, True, FULL_SIMPLE, "lock", "Keeper plate"))
            model.meta["env_release_joint"] = eb.joint.name
        if opm.kind in ("flush_pull", "pull", "push_plate"):
            faces = [-1.0, 1.0] if spec["operator"].get("sides", "both") == "both" and abs(yl) < 0.05 else [-1.0]
            for f in faces:
                C.add_pull(model, b, opm, dir_ * -1.0, x_latch_edge + dir_ * 0.09, hz, t, f, name=f"{name}_pull")
            if opm.id == "barn_privacy_hook":
                # teardrop latch on robot face at latch edge pivoting about y; hooks over a keeper on the jamb
                lm = C.mat_from_material(model, "black_matte_metal", "mat_teardrop")
                td = Body(f"{name}_teardrop", b.name, (x_latch_edge + dir_ * 0.06, -(t / 2 + 0.01), hz + 0.25), QUAT_ID, None, [], [], FULL_SIMPLE, "latch", "Teardrop latch")
                td.joint = Joint(f"{name}_teardrop_hinge", "hinge", (0, dir_, 0), (0, 0, 0), (0.0, 1.4), damping=0.02, frictionloss=0.02, role="operator", label="Teardrop latch (0 = dropped over the keeper, + = lifted)", initial=0.0)
                model.meta["operator_joint"] = td.joint.name
                td.geoms.append(C.box(f"{name}_teardrop_top", (-dir_ * 0.04, 0, -0.005), (0.04, 0.004, 0.006), lm, 7800, True, True, FULL_SIMPLE, "latch", "Teardrop bar"))
                td.geoms.append(C.box(f"{name}_teardrop_end", (-dir_ * 0.078, 0, -0.03), (0.005, 0.004, 0.03), lm, 7800, True, True, FULL_SIMPLE, "latch", "Teardrop end"))
                td.sites.append(Site(f"{name}_teardrop_grip", (-dir_ * 0.078, -0.01, -0.05), QUAT_ID, 0.01, "grip"))
                model.add_body(td)
                world.geoms.append(C.box(f"{name}_teardrop_keeper_post", (xc + x_latch_edge - dir_ * 0.005, yl - (t / 2 + 0.01), hz + 0.25 - 0.034), (0.005, 0.006, 0.018), lm, 7800, True, True, FULL_SIMPLE, "latch", "Keeper post"))
                world.geoms.append(C.box(f"{name}_teardrop_keeper_base", (xc + x_latch_edge - dir_ * 0.005, yl - (t / 2 + 0.024), hz + 0.25 - 0.03), (0.005, 0.008, 0.02), lm, 7800, False, True, FULL_ONLY, "latch", "Keeper base"))
        elif opm.kind in ("hook_lock_slider",):
            hb = C.add_rotary_operator(model, b, spec, phys, H.OPERATORS["lever_l_shape"], -dir_, 1.0, x_latch_edge + dir_ * 0.06, hz, t, [-1.0, 1.0], None, name=f"{name}_handle")
            hb.joint.label = "Slider handle / thumb latch (+ = unlock hook)"
            # the hook (latch) is always thrown when the leaf starts closed; the hook LOCK only freezes the handle
            _add_hook(model, world, b, name, dir_, x_latch_edge, xc, yl, hz, True, hb.joint.name, 1.0 / max(H.OPERATORS["hook_lock_slider"].travel, 0.5))
            if engaged and lk.kind == "hook_lock" and not release:
                hb.joint.range = (0.0, 0.05)
                hb.joint.notes = "hook lock engaged: handle blocked"
            model.meta["operator_joint"] = hb.joint.name
        if lk.kind == "hook_lock" and opm.kind != "hook_lock_slider" and fam in ("sliding_single",):
            # privacy hook lock: thumbturn on the inside face drives a hook into a jamb keeper
            hm = C.mat_from_material(model, "stainless", "mat_hook")
            inside = 1.0 if spec["robot"]["robot_outside"] else -1.0
            eng = engaged
            tt = Body(f"{name}_hook_thumbturn", b.name, (x_latch_edge + dir_ * 0.06, inside * t / 2, hz + 0.08), QUAT_ID, None, [], [], ALL_TIERS, "lock", "Hook thumbturn")
            tt.joint = Joint(f"{name}_hook_thumbturn_hinge", "hinge", (0, -inside, 0), (0, 0, 0), (0.0, 1.0), damping=0.05, frictionloss=0.25, role="lock", label="Thumbturn (0 = hooked, + = released)", initial=0.0 if eng else 1.0)
            key, mesh = MESH.thumbturn_mesh()
            tt.geoms.append(C.mesh_geom(f"{name}_hook_tt_mesh", key, mesh, (0, 0, 0), C.q_face(inside, -dir_), hm, 7100, True, ALL_TIERS, "lock", "Thumbturn"))
            tt.geoms.append(C.box(f"{name}_hook_tt_col", (0, inside * 0.02, 0), (0.006, 0.012, 0.016), hm, 7100, True, False, ALL_TIERS, "lock", "Thumbturn"))
            tt.sites.append(Site(f"{name}_hook_tt_grip", (0, inside * 0.03, 0), QUAT_ID, 0.01, "grip"))
            model.add_body(tt)
            _add_hook(model, world, b, name, dir_, x_latch_edge, xc, yl, hz + 0.08, eng, tt.joint.name, 1.0)
            if eng and not release:
                tt.joint.range = (0.0, 0.05)
        if opm.kind == "slide_bolt_handle":
            # drop bolt (cane bolt): vertical rod on the leaf face drops into a floor socket, blocking travel
            mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
            dia = opm.style_params.get("diameter", 0.02)
            eng = engaged
            xb = x_latch_edge + dir_ * 0.12
            L = zb + 0.30 - 0.026
            sb = Body(f"{name}_slide_bolt", b.name, (xb, -(t / 2 + dia), zb + 0.30), QUAT_ID, None, [], [], ALL_TIERS, "latch", "Drop bolt")
            sb.joint = Joint(f"{name}_slide_bolt_slide", "slide", (0, 0, 1), (0, 0, 0), (0.0, 0.08), damping=2.0, frictionloss=opm.spring_torque_preload, role="lock", label="Drop bolt (0 = in floor socket, + = lifted)", initial=0.0 if eng else 0.08, modeled_at=0.0 if eng else 0.08)
            zo = 0 if eng else 0.08
            sb.geoms.append(Geom(f"{name}_slide_bolt_rod", "capsule", (dia / 2, L / 2 - dia / 2), (0, 0, -L / 2 + zo), (1, 0, 0, 0), mat, True, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "latch", "Drop bolt rod"))
            sb.geoms.append(Geom(f"{name}_slide_bolt_knob", "capsule", (0.008, 0.03), (0, -0.03, 0.02 + zo), tuple(quat_z_to((0, -1, 0))), mat, True, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", "Bolt handle"))
            sb.sites.append(Site(f"{name}_grip_bolt", (0, -0.06, 0.02 + zo), QUAT_ID, 0.012, "grip"))
            model.add_body(sb)
            km = C.mat_from_material(model, "steel_galvanized", "mat_keeper")
            yk = yl - (t / 2 + dia)
            for sx_ in (-1, 1):
                world.geoms.append(C.box(f"{name}_socket_{'p' if sx_ > 0 else 'n'}", (xc + xb + sx_ * (dia / 2 + 0.006), yk, 0.04), (0.004, dia / 2 + 0.01, 0.02), km, 7800, True, True, ALL_TIERS, "lock", "Floor socket"))
            world.geoms.append(C.box(f"{name}_socket_b", (xc + xb, yk + (dia / 2 + 0.006), 0.04), (dia / 2 + 0.01, 0.004, 0.02), km, 7800, True, True, ALL_TIERS, "lock", "Floor socket"))
            world.geoms.append(C.box(f"{name}_socket_base", (xc + xb, yk, 0.01), (dia / 2 + 0.014, dia / 2 + 0.014, 0.01), km, 7800, False, True, FULL_SIMPLE, "lock", "Socket base"))
            model.meta["operator_joint"] = sb.joint.name
        elif opm.kind == "hasp":
            mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
            b.geoms.append(C.box(f"{name}_hasp", (x_latch_edge + dir_ * 0.04, -(t / 2 + 0.004), hz), (0.06, 0.003, 0.02), mat, 7800, True, True, ALL_TIERS, "lock", "Hasp"))
            if engaged:
                j.range = (0.0, 0.003)
        elif opm.kind == "none":
            pass
        # electric bolt / keyed lock -> lock joint
        if engaged and lk.kind in ("electric_strike", "keyed_cylinder", "padlock", "slide_bolt", "interlock") and opm.kind not in ("slide_bolt_handle",) and (not release or lk.kind in ("interlock", "electric_strike")):
            j.range = (0.0, 0.002)
            j.notes = f"{lk.name}: leaf locked (env releases on credential / call button)" if release else f"{lk.name}: leaf locked"
        b.sites.append(Site(f"{name}_edge_mid", (x_latch_edge, 0, zb + Hh / 2), QUAT_ID, 0.02, "leaf_edge"))
        if fam in ("automatic_sliding", "elevator"):
            act = kin.get("actuator", {})
            model.meta.setdefault("actuators", []).append({"name": f"{name}_drive", "joint": j.name, "kind": "position", "kp": 400.0, "kv": 60.0, "forcerange": (-act.get("max_force_N", 150), act.get("max_force_N", 150)), "ctrlrange": (0.0, tr)})
            j.frictionloss = max(j.frictionloss, 5.0)
    # latch-side jamb with hook pockets
    jamb_pockets += model.meta.pop("_jamb_pockets", [])
    if fam not in ("gate_sliding",) and latch_side != 0 and not (track == "top_hung_pocket" and latch_side == s_open):
        jamb_col = jamb_t + SLIDE_STUD
        xj = latch_side * (Wo / 2 + jamb_col / 2)
        segs, prev = [], 0.0
        for (zc, ph, yc) in sorted(jamb_pockets):
            segs.append((prev, zc - ph / 2))
            # pocket: keep thin walls at +-y beyond the leaf thickness, remove the core around the leaf plane
            for sy in (-1, 1):
                y0_, y1_ = sorted((yc + sy * (t / 2 + 0.006), sy * wt / 2))
                if y1_ - y0_ > 1e-4:
                    world.geoms.append(C.box(f"jamb_latch_pwall_{int(zc * 1000)}_{'p' if sy > 0 else 'n'}", (xj, (y0_ + y1_) / 2, zc), (jamb_col / 2, (y1_ - y0_) / 2, ph / 2), mat_frame, 300, True, True, ALL_TIERS, "frame", "Jamb pocket wall"))
            world.geoms.append(C.box(f"jamb_latch_pback_{int(zc * 1000)}", (latch_side * (Wo / 2 + jamb_col - 0.004), yc, zc), (0.004, t / 2 + 0.006, ph / 2), mat_frame, 300, True, True, ALL_TIERS, "frame", "Jamb pocket back"))
            prev = zc + ph / 2
        segs.append((prev, Ho))
        for k, (a, b) in enumerate(segs):
            if b - a > 1e-4:
                world.geoms.append(C.box(f"jamb_latch_{k}", (xj, 0, (a + b) / 2), (jamb_col / 2, wt / 2, (b - a) / 2), mat_frame, 300, True, True, ALL_TIERS, "frame", "Latch jamb"))
    # floor guide (barn)
    if "floor_guide" in spec.get("extras", []):
        for sy in (-1, 1):
            world.geoms.append(C.box(f"floor_guide_{'p' if sy > 0 else 'n'}", (s_open * (Wo / 2 + 0.05), y_leaf + sy * (t / 2 + 0.008), 0.02), (0.03, 0.004, 0.02), tm, 7850, True, True, FULL_SIMPLE, "track", "Floor guide"))
    if center:
        model.equalities.append(Equality("joint", "center_couple", bodies[1].joint.name, bodies[0].joint.name, (0, 1.0, 0, 0, 0), tiers=ALL_TIERS, label="leaves move symmetrically"))
    if fam == "elevator":
        # car doors behind (visual) and sill
        sm = C.mat_from_material(model, "stainless", "mat_sill")
        world.geoms.append(C.box("sill", (0, 0.0, 0.01), (Wo / 2 + 0.1, 0.06, 0.01), sm, 7900, True, True, ALL_TIERS, "frame", "Elevator sill"))
        world.geoms.append(C.box("car_floor", (0, 1.2, -0.01), (Wo / 2 + 0.6, 1.1, 0.01), sm, 7900, True, True, ALL_TIERS, "floor", "Car floor"))
        for sgn in (-1, 1):
            world.geoms.append(C.box(f"car_wall_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + 0.6), 1.2, Ho / 2 + 0.1), (0.02, 1.1, Ho / 2 + 0.1), sm, 7900, True, True, FULL_SIMPLE, "wall", "Car wall"))
        world.geoms.append(C.box("car_back", (0, 2.3, Ho / 2 + 0.1), (Wo / 2 + 0.6, 0.02, Ho / 2 + 0.1), sm, 7900, True, True, FULL_SIMPLE, "wall", "Car back wall"))
    C.add_extras(model, world, bodies[0], spec, 1.0, 1.0, -W / 2, 0.012, W, Hh, t, Wo, Ho)
    _sites(world, Ho)
    model.meta.update({"primary_joint": bodies[0].joint.name, "secondary_joint": bodies[1].joint.name if len(bodies) > 1 else None, "handle_height": spec["operator"]["height"], "opens_toward": "left" if opens_left else "right"})
    if "operator_joint" not in model.meta:
        model.meta["operator_joint"] = None
    return bodies


# ---------------------------------------------------------------------------
# Folding: bifold / accordion
# ---------------------------------------------------------------------------
def build_folding(spec, phys, model: Model):
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    n = leaf["count"]
    accordion = bool(spec["kinematics"].get("accordion"))
    world = C.add_floor_and_wall(model, spec)
    C.add_frame(model, spec, 1.0, world, with_stop=False, strike_pockets=None, u=1.0)
    tm = C.mat_from_material(model, "aluminum", "mat_track")
    world.geoms.append(C.box("fold_track", (0, 0, Ho - 0.015), (Wo / 2, 0.02, 0.015), tm, 2700, False, True, FULL_SIMPLE, "track", "Top track"))
    v = -1.0   # folds toward the robot
    rf = phys["roller"]
    bodies = []
    opm = H.OPERATORS[spec["operator"]["model"]]
    zb = 0.02
    groups = [(1.0, -Wo / 2)] if (not accordion and n == 2) or accordion else [(1.0, -Wo / 2), (-1.0, Wo / 2)]
    per_group = n // len(groups)
    for gi, (u, hx) in enumerate(groups):
        parent = None
        prev_name = None
        for k in range(per_group):
            name = f"panel_{gi}_{k}"
            if k == 0:
                b = Body(name, None, (hx, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Fold panel")
                j = Joint(f"{name}_hinge", "hinge", (0, 0, u * v), (u * 0.02, 0, 0), (0.0, math.radians(spec["kinematics"].get("max_open_deg", 90) if not accordion else 85)), damping=rf["viscous_damping_N_s_per_m"] * 0.2 + 0.2, frictionloss=rf["coulomb_force_N"] * 0.3 + 0.1, armature=0.005, role="primary", label="Pivot panel (0 = closed, + = folding open)")
            else:
                b = Body(name, prev_name, (u * (W - 0.0), 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Fold panel")
                j = Joint(f"{name}_hinge", "hinge", (0, 0, u * v), (0, 0, 0), (-math.pi, 0.0), damping=0.2, frictionloss=0.1, armature=0.002, role="secondary", label="Panel-to-panel hinge (driven)", robot_interactive=False)
            b.joint = j
            model.add_body(b)
            C.add_leaf_geoms(model, b, spec, leaf, u, u * 0.005, zb, phys if k == 0 else None, name_prefix=name)
            if k > 0:
                # alternating coupling: relative angle = -2 * previous relative? For equal panels on a track: q_k = -2*q_1 for k=1, then +2 q_1, -2 q_1...
                sign = -2.0 if k % 2 == 1 else 2.0
                model.equalities.append(Equality("joint", f"{name}_couple", j.name, f"panel_{gi}_0_hinge", (0, sign, 0, 0, 0), tiers=ALL_TIERS, label=f"q = {sign:+.0f} * q_pivot (track-guided fold)"))
                model.contact_excludes.append((name, prev_name))
                if k >= 2:
                    model.contact_excludes.append((name, f"panel_{gi}_{k - 2}"))
            if k == per_group - 1 or (accordion and k == per_group - 1):
                # knob/pull on the free panel's leading edge
                hz = spec["operator"]["height"]
                if opm.kind == "knob":
                    key, mesh = MESH.knob_mesh(shape="round", diameter=0.03, depth=0.03, rose_diameter=0.0)
                    mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
                    b.geoms.append(C.mesh_geom(f"{name}_knob", key, mesh, (u * (W - 0.05), -t / 2, hz), C.q_face(-1.0, u), mat, 3000, False, ALL_TIERS, "operator", "Bifold knob"))
                    b.geoms.append(C.sphere(f"{name}_knob_col", (u * (W - 0.05), -(t / 2 + 0.03), hz), 0.016, mat, 3000, True, ALL_TIERS, "operator", "Knob grip"))
                    b.sites.append(Site(f"{name}_grip", (u * (W - 0.05), -(t / 2 + 0.03), hz), QUAT_ID, 0.012, "grip"))
                else:
                    C.add_pull(model, b, opm, u, u * (W - 0.06), hz, t, -1.0, name=f"{name}_pull")
            # hinge visuals between panels
            bodies.append(b)
            prev_name = name
    if H.LATCHES[spec["latch"]["model"]].kind == "magnetic":
        model.meta.setdefault("notes", []).append("magnetic catch not simulated (holding force in spec.physics.latch)")
    _sites(world, Ho)
    model.meta.update({"primary_joint": "panel_0_0_hinge", "secondary_joint": "panel_1_0_hinge" if len(groups) > 1 else None, "operator_joint": None, "handle_height": spec["operator"]["height"]})
    return bodies


# ---------------------------------------------------------------------------
# Rotors: revolving door, turnstiles
# ---------------------------------------------------------------------------
def build_revolving(spec, phys, model: Model):
    leaf = spec["leaf"]
    Hh, t = leaf["height"], leaf["thickness"]
    op = spec["opening"]
    D = op["drum_diameter"]
    R = D / 2
    wings = leaf["count"]
    open_deg = op.get("drum_opening_deg", 100)
    world = C.add_floor_and_wall(model, spec, wall_half_width=max(3.0, R + 1.5), hole=(-R - 0.05, R + 0.05, 0.0, Hh + 0.1), wall_height=Hh + 0.6)
    fm = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
    gm = C.mat_from_material(model, "glass_clear", "mat_drum_glass")
    # drum: segments on ±x sides covering angles outside the openings (openings centered on ±y)
    nseg = 28
    half_open = math.radians(open_deg) / 2
    for i in range(nseg):
        a = 2 * math.pi * (i + 0.5) / nseg
        # opening if angle within half_open of +-90deg
        if abs(((a - math.pi / 2 + math.pi) % (2 * math.pi)) - math.pi) < half_open or abs(((a + math.pi / 2 + math.pi) % (2 * math.pi)) - math.pi) < half_open:
            continue
        seg_len = 2 * math.pi * (R + 0.02) / nseg
        q = quat_from_axis_angle([0, 0, 1], a + math.pi / 2)
        world.geoms.append(C.box(f"drum_{i}", ((R + 0.02) * math.cos(a), (R + 0.02) * math.sin(a), Hh / 2 + 0.05), (seg_len / 2 + 0.002, 0.006, Hh / 2 + 0.05), gm, 2500, True, True, ALL_TIERS, "glass", "Drum glass", quat=q))
    world.geoms.append(C.cyl("drum_canopy", (0, 0, Hh + 0.1 + 0.1), R + 0.08, 0.1, fm, (0, 0, 1), 300, True, True, ALL_TIERS, "frame", "Canopy"))
    world.geoms.append(C.cyl("drum_floor_ring", (0, 0, 0.003), R + 0.08, 0.003, C.mat_from_material(model, "stainless", "mat_floor_ring"), (0, 0, 1), 7900, False, True, FULL_ONLY, "frame", "Floor ring"))
    rotor = Body("rotor", None, (0, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Rotor")
    hf = phys["hinge"]
    rotor.joint = Joint("rotor_hinge", "hinge", (0, 0, 1), (0, 0, 0), None, damping=spec["kinematics"].get("speed_governor_damping", 30.0), frictionloss=hf["coulomb_torque_Nm"] + 2.0, armature=0.5, role="primary", label="Rotor (unbounded, + = CCW from above)")
    model.add_body(rotor)
    rotor.geoms.append(C.cyl("rotor_shaft", (0, 0, Hh / 2 + 0.05), 0.06, Hh / 2 + 0.05, fm, (0, 0, 1), 2700, True, True, ALL_TIERS, "leaf", "Center shaft"))
    for k in range(wings):
        a = 2 * math.pi * k / wings
        q = quat_from_axis_angle([0, 0, 1], a)
        Wl = R - 0.04
        rotor.geoms.append(C.box(f"wing_{k}_glass", (Wl / 2 * math.cos(a) + 0.02 * math.cos(a), Wl / 2 * math.sin(a) + 0.02 * math.sin(a), Hh / 2 + 0.05), (Wl / 2, t / 2, Hh / 2 - 0.05), gm, 2500, True, True, ALL_TIERS, "glass", f"Wing {k + 1} glass", quat=q, mass=phys["mass"]["total_kg"]))
        rotor.geoms.append(C.box(f"wing_{k}_stile", ((R - 0.02) * math.cos(a), (R - 0.02) * math.sin(a), Hh / 2 + 0.05), (0.02, 0.03, Hh / 2 + 0.05), fm, 2700, True, True, ALL_TIERS, "leaf", "Wing stile", quat=q))
        rotor.geoms.append(C.box(f"wing_{k}_rail_b", (Wl / 2 * math.cos(a), Wl / 2 * math.sin(a), 0.06), (Wl / 2, 0.03, 0.04), fm, 2700, True, True, FULL_SIMPLE, "leaf", "Bottom rail", quat=q))
        rotor.geoms.append(C.box(f"wing_{k}_rail_t", (Wl / 2 * math.cos(a), Wl / 2 * math.sin(a), Hh + 0.03), (Wl / 2, 0.03, 0.04), fm, 2700, True, True, FULL_SIMPLE, "leaf", "Top rail", quat=q))
        # push bar on each wing
        if spec["operator"]["model"] in ("pull_d", "push_plate"):
            pm = C.mat_from_material(model, "stainless", "mat_op_stainless")
            xb, yb = (R * 0.6) * math.cos(a), (R * 0.6) * math.sin(a)
            nx, ny = -math.sin(a), math.cos(a)
            rotor.geoms.append(Geom(f"wing_{k}_bar", "capsule", (0.012, 0.15), (xb + nx * (t / 2 + 0.05), yb + ny * (t / 2 + 0.05), 1.0), (1, 0, 0, 0), pm, True, True, 7900, None, (0.7, 0.01, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", "Push bar"))
            rotor.sites.append(Site(f"wing_{k}_push", (xb + nx * (t / 2 + 0.05), yb + ny * (t / 2 + 0.05), 1.0), QUAT_ID, 0.015, "push"))
    _sites(world, Hh, -R - 1.0, R + 1.0)
    model.meta.update({"primary_joint": "rotor_hinge", "operator_joint": None, "handle_height": 1.0, "drum_diameter": D})
    return rotor


def build_turnstile(spec, phys, model: Model, full_height=False):
    op = spec["opening"]
    Wo = op["width"]
    world = C.add_floor_and_wall(model, spec, outdoor=True)
    fm = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
    hf = phys["hinge"]
    if not full_height:
        # cabinet on -x side, rotor axis inclined 45 deg in the y-z plane, arms rotate; passage centered at x=+0.3
        cab_w, cab_d, cab_h = 0.28, 0.9, 0.98
        xc = -Wo / 2 - cab_w / 2 + 0.1
        world.geoms.append(C.box("cabinet", (xc, 0, cab_h / 2), (cab_w / 2, cab_d / 2, cab_h / 2), fm, 800, True, True, ALL_TIERS, "frame", "Turnstile cabinet"))
        world.geoms.append(C.box("cabinet_top", (xc, 0, cab_h + 0.01), (cab_w / 2 + 0.01, cab_d / 2 + 0.01, 0.01), fm, 7900, True, True, FULL_SIMPLE, "frame", "Cabinet top"))
        # opposite side guide rail
        world.geoms.append(C.box("guide_rail_post", (xc + cab_w / 2 + 0.27 + 0.62, 0, 0.5), (0.02, 0.45, 0.5), fm, 7900, True, True, ALL_TIERS, "frame", "Guide rail"))
        # rotor
        ax = np.array([0.0, math.sin(math.radians(45)), math.cos(math.radians(45))])
        boss = 0.27
        world.geoms.append(C.cyl("hub_boss", (xc + cab_w / 2 + boss / 2, 0.0, cab_h - 0.05), 0.04, boss / 2, fm, (1, 0, 0), 800, False, True, ALL_TIERS, "frame", "Hub boss"))
        rotor = Body("rotor", None, (xc + cab_w / 2 + boss + 0.02, 0.0, cab_h - 0.05), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Tripod rotor")
        rotor.joint = Joint("rotor_hinge", "hinge", tuple(ax), (0, 0, 0), None, damping=2.0, frictionloss=hf["coulomb_torque_Nm"] + 3.0, armature=0.05, role="primary", label="Tripod rotor (ratchets 120 deg; one-way enforced by env)", ratchet_one_way=bool(spec["kinematics"].get("one_way", True)))
        model.add_body(rotor)
        key, mesh = MESH.tripod_mesh(arm_len=0.5, r=0.019, hub_r=0.05)
        # mesh frame: arms in xy plane, hub along z -> rotate z to ax
        rotor.geoms.append(C.mesh_geom("tripod_mesh", key, mesh, (0, 0, 0), quat_z_to(ax), C.mat_from_material(model, "stainless", "mat_op_stainless"), 7900, False, ALL_TIERS, "operator", "Tripod arms"))
        # collision capsules for the 3 arms (in the plane perpendicular to ax)
        e1 = np.array([1.0, 0, 0])
        e2 = np.cross(ax, e1)
        for k in range(3):
            a = 2 * math.pi * k / 3
            d = math.cos(a) * e1 + math.sin(a) * e2
            rotor.geoms.append(Geom(f"arm_{k}_col", "capsule", (0.019, 0.25), tuple(d * 0.25), tuple(quat_z_to(d)), "mat_op_stainless", True, False, 7900, 2.0, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", f"Arm {k + 1}"))
        rotor.sites.append(Site("arm_push", tuple(e1 * 0.45), QUAT_ID, 0.015, "push"))
        world.sites.append(Site("approach_point", (xc + cab_w / 2 + 0.55, -1.2, 0), QUAT_ID, 0.05, "approach"))
        world.sites.append(Site("goal_point", (xc + cab_w / 2 + 0.55, 1.2, 0), QUAT_ID, 0.05, "goal"))
        world.sites.append(Site("door_plane_center", (xc + cab_w / 2 + 0.55, 0, 1.0), QUAT_ID, 0.02, "pass_plane"))
        model.meta.update({"primary_joint": "rotor_hinge", "operator_joint": None, "handle_height": 0.95, "ratchet_deg": 120, "one_way": True, "locked": bool(spec["kinematics"].get("locked_until_credential"))})
        if spec["kinematics"].get("locked_until_credential"):
            rotor.joint.range = (-0.05, 0.05)
            rotor.joint.notes = "locked until credential (env releases: set range None)"
        return rotor
    # full height
    wings = spec["kinematics"]["wings"]
    Rr = spec["leaf"]["width"]
    Hh = spec["leaf"]["height"]
    arms = spec["leaf"].get("arms_per_wing", 8)
    # cage: half-cylinder of bars on the +x side, rotor at x=0
    cage_R = Rr + 0.08
    for i in range(16):
        a = -math.pi / 2 + math.pi * (i + 0.5) / 16 + 0.0
        a = math.pi / 2 + math.pi * (i + 0.5) / 16   # bars from +y around +x?? we want the cage on the side opposite the passage entrance: bars around -x side
        a = math.pi * 0.5 + math.pi * (i + 0.5) / 16
        world.geoms.append(C.cyl(f"cage_bar_{i}", (cage_R * math.cos(a) + 0.0, cage_R * math.sin(a), Hh / 2 + 0.05), 0.02, Hh / 2 + 0.05, fm, (0, 0, 1), 7850, True, True, ALL_TIERS, "frame", "Cage bar"))
    world.geoms.append(C.cyl("cage_roof", (0, 0, Hh + 0.12), cage_R + 0.05, 0.02, fm, (0, 0, 1), 7850, True, True, ALL_TIERS, "frame", "Cage roof"))
    # side barriers guiding into the rotor
    for sgn in (-1, 1):
        world.geoms.append(C.box(f"barrier_{'p' if sgn > 0 else 'n'}", (cage_R + 0.4, sgn * (cage_R + 0.1), Hh / 2 + 0.05), (0.4, 0.02, Hh / 2 + 0.05), fm, 7850, True, True, FULL_SIMPLE, "wall", "Barrier"))
    rotor = Body("rotor", None, (0, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Full-height rotor")
    rotor.joint = Joint("rotor_hinge", "hinge", (0, 0, 1), (0, 0, 0), None, damping=6.0, frictionloss=hf["coulomb_torque_Nm"] + 4.0, armature=0.3, role="primary", label="Rotor (ratchets 360/wings deg; one-way enforced by env)", ratchet_one_way=bool(spec["kinematics"].get("one_way", True)))
    model.add_body(rotor)
    sm = C.mat_from_material(model, "stainless", "mat_op_stainless")
    rotor.geoms.append(C.cyl("rotor_column", (0, 0, Hh / 2 + 0.05), 0.06, Hh / 2 + 0.05, sm, (0, 0, 1), 7900, True, True, ALL_TIERS, "leaf", "Rotor column"))
    for k in range(wings):
        a = 2 * math.pi * k / wings
        d = np.array([math.cos(a), math.sin(a), 0])
        for m in range(arms):
            z = 0.15 + m * (Hh - 0.3) / (arms - 1)
            rotor.geoms.append(Geom(f"wing_{k}_arm_{m}", "capsule", (0.019, Rr / 2), tuple(d * Rr / 2 + np.array([0, 0, z])), tuple(quat_z_to(d)), sm, True, True, 7900, 0.9, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS if m % 2 == 0 else FULL_SIMPLE, "operator", "Arm"))
        rotor.sites.append(Site(f"wing_{k}_push", tuple(d * Rr * 0.8 + np.array([0, 0, 1.0])), QUAT_ID, 0.015, "push"))
    world.sites.append(Site("approach_point", (cage_R + 0.4, -1.5, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (cage_R + 0.4, 1.5, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (cage_R * 0.5, 0, 1.0), QUAT_ID, 0.02, "pass_plane"))
    model.meta.update({"primary_joint": "rotor_hinge", "operator_joint": None, "handle_height": 1.0, "ratchet_deg": 360 / wings, "one_way": bool(spec["kinematics"].get("one_way")), "locked": bool(spec["kinematics"].get("locked_until_credential"))})
    if spec["kinematics"].get("locked_until_credential"):
        rotor.joint.range = (-0.05, 0.05)
    return rotor


# ---------------------------------------------------------------------------
# Vertical lift: garage sectional, roll-up
# ---------------------------------------------------------------------------
def build_vertical(spec, phys, model: Model):
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    kin = spec["kinematics"]
    fam = spec["family"]
    rough = 0.12 if fam == "garage_sectional" else 0.0
    # vertical-lift approximation: the door rises straight up, so the wall above the opening is left open (high-lift bay)
    world = C.add_floor_and_wall(model, spec, wall_half_width=max(3.0, W / 2 + 1.0), wall_height=Ho + Hh + 0.3 if fam == "garage_sectional" else (Ho + 1.6 if fam == "rollup" else Ho + 1.2), hole=(-Wo / 2 - rough, Wo / 2 + rough, 0.0, Ho + Hh + 0.08) if rough else None)
    fm = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
    tm = C.mat_from_material(model, "steel_galvanized", "mat_track")
    if rough:
        # jamb fill beside the track / lock channel
        for sgn in (-1, 1):
            world.geoms.append(C.box(f"garage_jamb_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + rough - 0.04), 0, Ho / 2), (0.04, op["wall_thickness"] / 2, Ho / 2), fm, 400, True, True, ALL_TIERS, "frame", "Garage jamb"))
    m = phys["mass"]["total_kg"]
    cb = kin.get("counterbalance_fraction", 0.0)
    lb = Body("curtain" if fam == "rollup" else "door", None, (0, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Vertical lift door")
    j = slide_joint(spec, phys, (0, 0, 1), kin["travel_m"], f"{lb.name}_slide", counterbalance=cb, mass=m)
    j.label = "Vertical lift (0 = closed, + = raised)"
    lb.joint = j
    model.add_body(lb)
    y_leaf = 0.0 if fam != "rollup" else (op["wall_thickness"] / 2 + 0.03)
    if fam == "garage_sectional":
        ns = kin.get("n_sections", 4)
        sh = Hh / ns
        phys_k = {"mass": {"slab_kg": phys["mass"]["slab_kg"] / ns}}
        for k in range(ns):
            sub = {**leaf, "height": sh, "panel_style": leaf["panel_style"]}
            C.add_leaf_geoms(model, lb, spec, sub, 1.0, -W / 2, 0.01 + k * sh, phys_k, name_prefix=f"section_{k}", Hh=sh)
            if k > 0:
                # hinge line visual
                lb.geoms.append(C.box(f"section_hinge_line_{k}", (0, y_leaf + t / 2 + 0.001, 0.01 + k * sh), (W / 2 - 0.02, 0.001, 0.004), C.mat_rgba(model, "mat_hinge_line", (0.2, 0.2, 0.2, 1), 0.7), 1.0, False, True, FULL_ONLY, "leaf", "Section joint"))
                for xh in (-W / 2 + 0.1, 0.0, W / 2 - 0.1):
                    lb.geoms.append(C.box(f"section_hinge_{k}_{int((xh + W) * 100)}", (xh, y_leaf + t / 2 + 0.01, 0.01 + k * sh), (0.04, 0.01, 0.05), tm, 7850, False, True, FULL_ONLY, "hinge", "Section hinge"))
        # vertical tracks
        track_y = t / 2 + 0.06
        for sgn in (-1, 1):
            world.geoms.append(C.box(f"track_{'r' if sgn > 0 else 'l'}", (sgn * (W / 2 + 0.03), track_y, Ho / 2 + Hh / 2), (0.02, 0.02, Ho / 2 + Hh / 2), tm, 7850, True, True, FULL_SIMPLE, "track", "Vertical track"))
            for k in range(ns + 1):
                lb.geoms.append(C.cyl(f"roller_{'r' if sgn > 0 else 'l'}_{k}", (sgn * (W / 2 + 0.03), track_y, 0.01 + k * (Hh / ns) if k < ns else Hh - 0.05), 0.025, 0.01, tm, (1, 0, 0), 7850, False, True, FULL_ONLY, "track", "Track roller"))
                lb.geoms.append(C.box(f"roller_arm_{'r' if sgn > 0 else 'l'}_{k}", (sgn * (W / 2 - 0.02), track_y / 2 + t / 4, 0.01 + k * (Hh / ns) if k < ns else Hh - 0.05), (0.03, track_y / 2 - t / 4, 0.006), tm, 7850, False, True, FULL_ONLY, "track", "Roller hinge arm"))
        # torsion spring shaft
        world.geoms.append(C.cyl("torsion_shaft", (0, 0.12, Ho + 0.25), 0.013, W / 2 + 0.2, tm, (1, 0, 0), 7850, True, True, FULL_SIMPLE, "mechanism", "Torsion spring shaft"))
        world.geoms.append(C.cyl("torsion_spring", (0, 0.12, Ho + 0.25), 0.03, 0.35, C.mat_from_material(model, "steel", "mat_spring"), (1, 0, 0), 7850, False, True, FULL_ONLY, "mechanism", "Torsion spring"))
        if kin.get("opener", "none_manual") != "none_manual":
            world.geoms.append(C.box("opener_rail", (0, 1.5, Ho + 0.15), (0.03, 1.5, 0.03), tm, 7850, True, True, FULL_ONLY, "mechanism", "Opener rail"))
            world.geoms.append(C.box("opener_unit", (0, 3.0, Ho + 0.1), (0.15, 0.2, 0.1), C.mat_from_material(model, "black_matte_metal", "mat_opener"), 500, True, True, FULL_ONLY, "mechanism", "Opener unit"))
    elif fam == "rollup":
        C.add_leaf_geoms(model, lb, spec, leaf, 1.0, -W / 2, 0.01, phys, name_prefix="curtain", y_center=y_leaf)
        # slats visual lines
        ns = int(Hh / 0.075)
        for k in range(1, ns):
            lb.geoms.append(C.box(f"slat_line_{k}", (0, y_leaf - t / 2 - 0.001, 0.01 + k * Hh / ns), (W / 2, 0.001, 0.003), C.mat_rgba(model, "mat_slat_line", (0.25, 0.25, 0.25, 1), 0.7), 1.0, False, True, FULL_ONLY, "leaf", "Slat joint"))
        # guides + drum + hood
        for sgn in (-1, 1):
            if sgn < 0 and spec["lock"]["model"] == "garage_slide_lock":
                # left guide split around the slide-lock bar (z = 1.0): the guide segments are the keeper
                world.geoms.append(C.box("guide_l", (sgn * (W / 2 + 0.035), y_leaf, 0.985 / 2), (0.03, 0.025, 0.985 / 2), tm, 7850, True, True, ALL_TIERS, "track", "Curtain guide"))
                world.geoms.append(C.box("guide_l_upper", (sgn * (W / 2 + 0.035), y_leaf, (1.015 + Ho) / 2), (0.03, 0.025, (Ho - 1.015) / 2), tm, 7850, True, True, ALL_TIERS, "track", "Curtain guide"))
                continue
            world.geoms.append(C.box(f"guide_{'r' if sgn > 0 else 'l'}", (sgn * (W / 2 + 0.035), y_leaf, Ho / 2), (0.03, 0.025, Ho / 2), tm, 7850, True, True, ALL_TIERS, "track", "Curtain guide"))
        world.geoms.append(C.cyl("coil_drum", (0, y_leaf, Ho + 0.3), 0.25, W / 2 + 0.05, fm, (1, 0, 0), 300, False, True, FULL_SIMPLE, "mechanism", "Coil (visual)"))
        world.geoms.append(C.box("hood", (0, y_leaf, Ho + 0.3 + 0.28), (W / 2 + 0.12, 0.30, 0.02), tm, 7850, True, True, FULL_SIMPLE, "mechanism", "Hood"))
        world.geoms.append(C.box("hood_front", (0, y_leaf + 0.30, Ho + 0.32), (W / 2 + 0.12, 0.02, 0.28), tm, 7850, True, True, FULL_SIMPLE, "mechanism", "Hood front"))
        lb.geoms.append(C.box("bottom_bar", (0, y_leaf, 0.03), (W / 2, t / 2 + 0.015, 0.03), tm, 7850, True, True, ALL_TIERS, "leaf", "Bottom bar", mass=3.0 * W))
        if kin.get("opener") == "chain_hoist":
            world.geoms.append(C.cyl("hoist_chain", (W / 2 + 0.12, y_leaf, Ho / 2 + 0.3), 0.005, Ho / 2 + 0.3, C.mat_from_material(model, "steel", "mat_chain"), (0, 0, 1), 7850, True, True, FULL_ONLY, "mechanism", "Hoist chain"))
    # operator (lift handle / T-handle)
    opm = H.OPERATORS[spec["operator"]["model"]]
    hz = spec["operator"]["height"]
    if fam == "rollup" and opm.kind in ("pull", "ring_pull", "none"):
        C.add_pull(model, lb, H.OPERATORS["pull_lift_garage"], 1.0, 0.0, 0.16, t, -1.0, name="lift_handle")
        # add_pull assumes leaf centered at y=0; shift for the curtain plane offset
        for g in lb.geoms[-2:]:
            g.pos = (g.pos[0], g.pos[1] + y_leaf, g.pos[2])
        lb.sites[-1].pos = (lb.sites[-1].pos[0], lb.sites[-1].pos[1] + y_leaf, lb.sites[-1].pos[2])
    elif opm.kind == "pull":
        for f in (-1.0, 1.0):
            C.add_pull(model, lb, opm, 1.0, 0.0, hz, t, f, name="lift_handle")
    elif opm.kind == "t_handle":
        hb = C.add_rotary_operator(model, lb, spec, phys, opm, 1.0, 1.0, 0.0, hz, t, [-1.0], None, name="t_handle")
        model.meta["operator_joint"] = hb.joint.name
        # lock bars driven by T-handle rotation into the tracks
        bm = C.mat_from_material(model, "steel_galvanized", "mat_lockbar")
        for sgn in (-1, 1):
            bar = Body(f"lock_bar_{'r' if sgn > 0 else 'l'}", lb.name, (sgn * (W / 2 - 0.05), 0.03, hz), QUAT_ID, None, [], [], FULL_SIMPLE, "lock", "Lock bar")
            eng = spec["lock"].get("engaged") and spec["lock"]["model"] == "keyed_cylinder"
            bar.joint = Joint(f"lock_bar_{'r' if sgn > 0 else 'l'}_slide", "slide", (-sgn, 0, 0), (0, 0, 0), (0.0, 0.03), damping=2.0, frictionloss=1.0, role="lock", label="Lock bar", robot_interactive=False, initial=0.0)
            bar.geoms.append(Geom(f"lock_bar_{'r' if sgn > 0 else 'l'}_geom", "capsule", (0.006, 0.05), (sgn * 0.04, 0, 0), tuple(quat_z_to((1, 0, 0))), bm, False, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, FULL_SIMPLE, "lock", "Lock bar (visual; engaged state locks the door joint)"))
            model.add_body(bar)
            model.equalities.append(Equality("joint", f"lock_bar_{'r' if sgn > 0 else 'l'}_couple", bar.joint.name, hb.joint.name, (0, 0.03 / opm.travel, 0, 0, 0), tiers=FULL_SIMPLE, label="lock bar = T-handle * 0.03/travel"))
            if not eng:
                bar.joint.initial = 0.03
    if spec["lock"].get("engaged") and spec["lock"]["model"] in ("garage_slide_lock", "padlock", "keyed_cylinder") and (not spec["lock"].get("robot_side_release") or spec["lock"]["model"] == "keyed_cylinder"):
        j.range = (0.0, 0.003)
        j.notes = f"{spec['lock']['model']}: locked (T-handle lock bars engaged; env unlock frees the joint)"
        model.meta["locked"] = True
    # counterbalance from the actual body mass (sections + hardware)
    mtot = lb.inertial("full")[0]
    if cb and mtot > 0:
        k_ = 0.3 * cb * mtot * 9.81 / max(kin["travel_m"], 0.1)
        j.stiffness = k_
        j.springref = cb * mtot * 9.81 / k_
        j.notes = (j.notes + " " if j.notes else "") + f"counterbalance ~{cb:.0%} of {mtot:.0f} kg"
    if spec["lock"]["model"] == "garage_slide_lock":
        sm = C.mat_from_material(model, "steel_galvanized", "mat_slidelock")
        inside = 1.0
        y_sl = (y_leaf if fam == "rollup" else 0.0) + inside * (t / 2 + 0.01)
        sl = Body("garage_slide_lock", lb.name, (-W / 2 + 0.06, y_sl, 1.0), QUAT_ID, None, [], [], FULL_SIMPLE, "lock", "Slide lock")
        eng = bool(spec["lock"].get("engaged"))
        sl.joint = Joint("garage_slide_lock_slide", "slide", (1, 0, 0), (0, 0, 0), (0.0, 0.05), damping=1.0, frictionloss=1.0, role="lock", label="Slide lock (0 = in track, + = withdrawn)", initial=0.0 if eng else 0.05, modeled_at=0.0 if eng else 0.05)
        sl.geoms.append(Geom("garage_slide_lock_geom", "capsule", (0.006, 0.04), (-0.06 + (0 if eng else 0.05), 0, 0), tuple(quat_z_to((1, 0, 0))), sm, True, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, FULL_SIMPLE, "lock", "Slide lock bar"))
        sl.sites.append(Site("slide_lock_grip", (0, inside * 0.02, 0), QUAT_ID, 0.01, "grip"))
        sl.joint.modeled_at = sl.joint.initial
        model.add_body(sl)
        # keeper channel on the track: bar tip (x ~ -W/2-0.046 when engaged) between two blocks (roll-up: the split guide is the keeper)
        if fam != "rollup":
            for sz in (-1, 1):
                world.geoms.append(C.box(f"slide_lock_keeper_{'t' if sz > 0 else 'b'}", (-W / 2 - 0.045, y_sl, 1.0 + sz * 0.011), (0.015, 0.012, 0.004), tm, 7850, True, True, FULL_SIMPLE, "lock", "Slide lock keeper"))
            world.geoms.append(C.box("slide_lock_keeper_back", (-W / 2 - 0.062, y_sl, 1.0), (0.003, 0.012, 0.015), tm, 7850, True, True, FULL_SIMPLE, "lock", "Keeper back"))
    _sites(world, Ho, -2.0, 2.0)
    model.meta.update({"primary_joint": j.name, "handle_height": hz, "counterbalance_fraction": cb})
    if "operator_joint" not in model.meta:
        model.meta["operator_joint"] = None
    return lb


# ---------------------------------------------------------------------------
# Horizontal hinge axis: hatches, pet doors, strip curtains, tilt-up garage
# ---------------------------------------------------------------------------
def build_horizontal(spec, phys, model: Model):
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    fam = spec["family"]
    kin = spec["kinematics"]
    hf = phys["hinge"]
    cl = phys["closer"]
    if fam in ("hatch_floor", "hatch_ceiling"):
        ceiling = fam == "hatch_ceiling"
        elev = op.get("elevation", 0.0) if ceiling else 0.0
        if ceiling:
            world = C.add_floor_and_wall(model, spec, hole=None)
            world.geoms = [g for g in world.geoms if g.semantic == "floor"]
            cm = C.mat_rgba(model, "mat_ceiling", (0.92, 0.92, 0.90, 1), 0.9)
            x0, x1, y0, y1 = -Wo / 2, Wo / 2, -Ho / 2, Ho / 2
            for nm, c, h in (("ceil_a", ((x0 - 3) / 2, 0, elev - 0.05), ((x0 + 3) / 2, 3, 0.05)), ("ceil_b", ((x1 + 3) / 2, 0, elev - 0.05), ((3 - x1) / 2, 3, 0.05)), ("ceil_c", (0, (y0 - 3) / 2, elev - 0.05), ((x1 - x0) / 2, (y0 + 3) / 2, 0.05)), ("ceil_d", (0, (y1 + 3) / 2, elev - 0.05), ((x1 - x0) / 2, (3 - y1) / 2, 0.05))):
                world.geoms.append(C.box(nm, c, h, cm, 600, True, True, ALL_TIERS, "wall", "Ceiling"))
            zf = elev
        else:
            world = C.add_floor_and_wall(model, spec, hole=None, floor_hole=(-Wo / 2, Wo / 2, -Ho / 2, Ho / 2))
            world.geoms = [g for g in world.geoms if g.semantic == "floor"]
            # pit below
            pm = C.mat_rgba(model, "mat_pit", (0.3, 0.3, 0.3, 1), 0.9)
            world.geoms.append(C.box("pit_floor", (0, 0, -1.5), (Wo / 2 + 0.2, Ho / 2 + 0.2, 0.02), pm, 2400, True, True, FULL_SIMPLE, "floor", "Pit floor"))
            zf = 0.0
        fm = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
        curb = 0.05
        for nm, c, h in (("curb_l", (-Wo / 2 - curb / 2, 0, zf + 0.02), (curb / 2, Ho / 2 + curb, 0.02)), ("curb_r", (Wo / 2 + curb / 2, 0, zf + 0.02), (curb / 2, Ho / 2 + curb, 0.02)), ("curb_n", (0, -Ho / 2 - curb / 2, zf + 0.02), (Wo / 2, curb / 2, 0.02)), ("curb_f", (0, Ho / 2 + curb / 2, zf + 0.02), (Wo / 2, curb / 2, 0.02))):
            world.geoms.append(C.box(nm, c, h, fm, 7850, True, True, ALL_TIERS, "frame", "Hatch curb"))
        # leaf hinged at far (+y) edge; lies flat; positive q lifts near edge
        lb = Body("hatch", None, (0, Ho / 2, zf + 0.04), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Hatch leaf")
        mo = math.radians(kin.get("max_open_deg", 90))
        stiffness = cl.get("spring_stiffness_Nm_per_rad", 0.0)
        preload = cl.get("spring_preload_Nm", 0.0)
        j = Joint("hatch_hinge", "hinge", (-1.0, 0, 0), (0, 0, 0), (0.0, mo), damping=hf.get("air_damping_Nms_per_rad", 0.1) + cl.get("damping_opening", 0.0) + 0.5, frictionloss=hf["coulomb_torque_Nm"], stiffness=abs(stiffness) if stiffness else 0.0, springref=(-preload / stiffness) if stiffness else 0.0, armature=0.02, role="primary", label="Hatch (0 = closed, + = lifted)")
        lb.joint = j
        model.add_body(lb)
        lm = C.mat_from_finish(model, leaf["finish"], "mat_leaf")
        lb.geoms.append(C.box("hatch_slab", (0, -Ho / 2, 0), (W / 2, Ho / 2 - 0.004, t / 2), lm, 1.0, True, True, ALL_TIERS, "leaf", "Hatch slab", mass=phys["mass"]["slab_kg"]))
        if leaf["panel_style"] == "riveted_steel":
            rm = C.mat_from_material(model, "steel", "mat_rivet")
            for i in range(4):
                for jx in range(3):
                    lb.geoms.append(C.sphere(f"rivet_{i}_{jx}", (-W / 2 + 0.06 + jx * (W - 0.12) / 2, -0.06 - i * (Ho - 0.12) / 3, t / 2), 0.01, rm))
        opm = H.OPERATORS[spec["operator"]["model"]]
        hm = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
        if opm.kind == "ring_pull":
            ring = Body("ring", lb.name, (0, -Ho * 0.75, t / 2), QUAT_ID, None, [], [], ALL_TIERS, "operator", "Ring pull")
            ring.joint = Joint("ring_hinge", "hinge", (1, 0, 0), (0, 0, 0), (0.0, 1.5708), damping=0.01, role="operator", label="Ring pull (flip up)")
            ring.geoms.append(Geom("ring_geom", "capsule", (0.006, 0.035), (0, 0.04, 0.006), tuple(quat_z_to((1, 0, 0))), hm, True, True, 7800, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", "Ring"))
            ring.sites.append(Site("grip_ring", (0, 0.04, 0.006), QUAT_ID, 0.01, "grip"))
            model.add_body(ring)
            lb.geoms.append(C.box("ring_recess", (0, -Ho * 0.75, t / 2 + 0.001), (0.05, 0.05, 0.001), hm, 7800, False, True, FULL_ONLY, "operator", "Recess plate"))
            model.meta["operator_joint"] = "ring_hinge"
        elif opm.kind == "pull":
            C.add_pull(model, lb, opm, 1.0, 0.0, -Ho * 0.75, t, 1.0, name="hatch_pull")
            # the add_pull uses z for height; fix by moving geoms: (x, face*t/2, z) -> we want (x, -Ho*0.75, t/2)
            for g in lb.geoms[-2:]:
                g.pos = (g.pos[0], -Ho * 0.75, t / 2 + (g.pos[1]))
                g.quat = tuple(quat_from_axis_angle([1, 0, 0], -math.pi / 2)) if g.type == "mesh" else tuple(quat_from_axis_angle([1, 0, 0], math.pi / 2))
        if cl["kind"] == "gas_strut":
            sm = C.mat_from_material(model, "stainless", "mat_strut")
            lb.geoms.append(C.cyl("gas_strut", (W / 2 - 0.05, -Ho * 0.4, -0.15), 0.01, 0.2, sm, (0, 0.6, -0.8), 7900, False, True, FULL_ONLY, "closer", "Gas strut"))
        if spec["lock"]["model"] in ("padlock", "slide_bolt") and spec["lock"].get("engaged") and not spec["lock"].get("robot_side_release"):
            j.range = (0.0, 0.005)
        if spec["lock"]["model"] == "slide_bolt":
            bm = C.mat_from_material(model, "steel_galvanized", "mat_bolt")
            sb = Body("hatch_bolt", lb.name, (W / 2 - 0.1, -Ho * 0.5, t / 2 + 0.008), QUAT_ID, None, [], [], FULL_SIMPLE, "lock", "Hatch slide bolt")
            eng = bool(spec["lock"].get("engaged"))
            sb.joint = Joint("hatch_bolt_slide", "slide", (-1, 0, 0), (0, 0, 0), (0.0, 0.04), damping=1.0, frictionloss=1.0, role="lock", label="Slide bolt (0 = engaged)", initial=0.0 if eng else 0.04, modeled_at=0.0 if eng else 0.04)
            sb.geoms.append(Geom("hatch_bolt_geom", "capsule", (0.006, 0.06), (0.08 - (0 if eng else 0.04), 0, 0), tuple(quat_z_to((1, 0, 0))), bm, True, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, FULL_SIMPLE, "lock", "Bolt"))
            sb.sites.append(Site("hatch_bolt_grip", (0, 0, 0.02), QUAT_ID, 0.01, "grip"))
            model.add_body(sb)
            for sz in (-1, 1):
                world.geoms.append(C.box(f"hatch_bolt_keeper_{'t' if sz > 0 else 'b'}", (Wo / 2 + 0.03, 0.0, zf + 0.04 + t / 2 + 0.008 + sz * 0.011), (0.025, 0.014, 0.004), bm, 7800, True, True, FULL_SIMPLE, "lock", "Keeper"))
        if kin.get("stop") == "prop_arm":
            model.meta.setdefault("notes", []).append("prop arm holds hatch open (env can lock joint at max)")
        world.sites.append(Site("approach_point", (0, -1.5, zf if not ceiling else 0), QUAT_ID, 0.05, "approach"))
        world.sites.append(Site("goal_point", (0, 0, zf - 1.0 if not ceiling else elev + 0.5), QUAT_ID, 0.05, "goal"))
        world.sites.append(Site("door_plane_center", (0, 0, zf), QUAT_ID, 0.02, "pass_plane"))
        model.meta.update({"primary_joint": "hatch_hinge", "handle_height": zf, "horizontal": True})
        if "operator_joint" not in model.meta:
            model.meta["operator_joint"] = None
        return lb
    if fam == "pet_door":
        # host panel: wall or door slab piece with the pet opening at floor level
        wt = op["wall_thickness"]
        world = C.add_floor_and_wall(model, spec, wall_half_width=1.0, wall_height=2.1, hole=(-Wo / 2, Wo / 2, 0.05, 0.05 + Ho))
        fm = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
        for nm, c, h in (("pet_frame_l", (-Wo / 2 - 0.02, 0, 0.05 + Ho / 2), (0.02, wt / 2 + 0.006, Ho / 2 + 0.04)), ("pet_frame_r", (Wo / 2 + 0.02, 0, 0.05 + Ho / 2), (0.02, wt / 2 + 0.006, Ho / 2 + 0.04)), ("pet_frame_t", (0, 0, 0.05 + Ho + 0.02), (Wo / 2, wt / 2 + 0.006, 0.02))):
            world.geoms.append(C.box(nm, c, h, fm, 1200, True, True, ALL_TIERS, "frame", "Pet door frame"))
        # tunnel liner
        world.geoms.append(C.box("pet_sill", (0, 0, 0.05 - 0.006), (Wo / 2, wt / 2 + 0.006, 0.006), fm, 1200, True, True, ALL_TIERS, "frame", "Sill"))
        flap = Body("flap", None, (0, 0, 0.05 + Ho), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Pet flap")
        mo = math.radians(kin.get("max_open_deg", 90))
        flap.joint = Joint("flap_hinge", "hinge", (1, 0, 0), (0, 0, 0), (-mo, mo), damping=0.01 + hf.get("air_damping_Nms_per_rad", 0.0), frictionloss=hf["coulomb_torque_Nm"] + 0.005, armature=1e-4, role="primary", label="Flap (swings both ways)")
        model.add_body(flap)
        slab = M.SLABS[leaf["slab"]]
        gm = C.mat_from_material(model, slab.core_material, "mat_flap")
        flap.geoms.append(C.box("flap_geom", (0, 0, -Hh / 2), (W / 2 - 0.003, t / 2, Hh / 2 - 0.002), gm, 1.0, True, True, ALL_TIERS, "leaf", "Flap", mass=phys["mass"]["total_kg"]))
        if kin.get("magnet_force_N", 0) > 0:
            mm = C.mat_from_material(model, "steel", "mat_magnet")
            flap.geoms.append(C.box("flap_magnet", (0, 0, -Hh + 0.01), (W / 2 - 0.02, t / 2 + 0.001, 0.008), mm, 7850, False, True, FULL_ONLY, "latch", "Magnet strip"))
            world.geoms.append(C.box("sill_magnet", (0, -t / 2 - 0.003, 0.05 + 0.01), (W / 2 - 0.02, 0.002, 0.008), mm, 7850, False, True, FULL_ONLY, "latch", "Sill magnet"))
            model.meta.setdefault("notes", []).append(f"flap magnet {kin['magnet_force_N']} N not simulated natively (env applies detent torque near closed)")
        if spec["lock"]["model"] == "slide_bolt" and spec["lock"].get("engaged"):
            flap.joint.range = (-0.001, 0.001)
            flap.joint.notes = "locking panel slid in: flap blocked"
        world.sites.append(Site("approach_point", (0, -1.0, 0), QUAT_ID, 0.05, "approach"))
        world.sites.append(Site("goal_point", (0, 1.0, 0), QUAT_ID, 0.05, "goal"))
        world.sites.append(Site("door_plane_center", (0, 0, 0.05 + Ho / 2), QUAT_ID, 0.02, "pass_plane"))
        model.meta.update({"primary_joint": "flap_hinge", "operator_joint": None, "handle_height": 0.05 + Ho / 2, "both_ways": True})
        return flap
    if fam == "strip_curtain":
        world = C.add_floor_and_wall(model, spec, wall_half_width=max(2.5, Wo / 2 + 1.0), wall_height=Ho + 0.6)
        fm = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
        world.geoms.append(C.box("hanger_rail", (0, 0, Ho + 0.03), (Wo / 2 + 0.05, 0.03, 0.03), fm, 7900, True, True, ALL_TIERS, "track", "Hanger rail"))
        n = leaf["count"]
        sw = leaf["strip_width"]
        pitch = (Wo - sw) / max(n - 1, 1)
        gm = C.mat_from_material(model, "pvc_flexible", "mat_strip")
        strips = []
        for k in range(n):
            x = -Wo / 2 + sw / 2 + k * pitch
            y = ((k % 2) - 0.5) * (t + 0.004)
            s = Body(f"strip_{k}", None, (x, y, Ho), QUAT_ID, None, [], [], ALL_TIERS if k % 2 == 0 else FULL_SIMPLE, "leaf", f"Strip {k + 1}")
            s.joint = Joint(f"strip_{k}_hinge", "hinge", (1, 0, 0), (0, 0, 0), (-2.0, 2.0), damping=0.05, frictionloss=0.02, armature=1e-4, role="primary" if k == n // 2 else "secondary", label="Strip swings both ways")
            s.geoms.append(C.box(f"strip_{k}_geom", (0, 0, -Hh / 2), (sw / 2, t / 2, Hh / 2), gm, 1250, True, True, ALL_TIERS if k % 2 == 0 else FULL_SIMPLE, "leaf", "PVC strip", friction=(0.6, 0.005, 0.0001)))
            model.add_body(s)
            strips.append(s)
            if k > 0:
                model.contact_excludes.append((s.name, strips[k - 1].name))
        _sites(world, Ho)
        model.meta.update({"primary_joint": f"strip_{n // 2}_hinge", "operator_joint": None, "handle_height": 1.0, "both_ways": True, "n_strips": n})
        return strips
    if fam == "garage_tiltup":
        world = C.add_floor_and_wall(model, spec, wall_half_width=max(3.0, W / 2 + 1.0), wall_height=Ho + 1.2)
        fm = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
        zp = Hh * kin.get("pivot_height_frac", 0.5)
        lb = Body("door", None, (0, 0, zp), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Tilt-up door")
        m = phys["mass"]["total_kg"]
        cb = kin.get("counterbalance_fraction", 0.0)
        mo = math.radians(kin.get("max_open_deg", 88))
        # gravity torque about mid-height pivot is ~0 (balanced) for a centered pivot; springs add lift assist
        j = Joint("door_hinge", "hinge", (-1.0, 0, 0), (0, 0, 0), (0.0, mo), damping=8.0 + 0.1 * m, frictionloss=hf["coulomb_torque_Nm"] + 2.0, armature=0.5, role="primary", label="Tilt-up (0 = closed vertical, + = tilting open/up)")
        lb.joint = j
        model.add_body(lb)
        C.add_leaf_geoms(model, lb, spec, leaf, 1.0, -W / 2, -zp + 0.01, phys, name_prefix="door")
        for sgn in (-1, 1):
            world.geoms.append(C.box(f"jamb_{'r' if sgn > 0 else 'l'}", (sgn * (Wo / 2 + 0.03), 0, Ho / 2), (0.03, op["wall_thickness"] / 2, Ho / 2), fm, 400, True, True, ALL_TIERS, "frame", "Jamb"))
            # pivot arm hardware
            world.geoms.append(C.box(f"pivot_bracket_{'r' if sgn > 0 else 'l'}", (sgn * (W / 2 + 0.02), 0.06, zp), (0.02, 0.06, 0.03), C.mat_from_material(model, "steel_galvanized", "mat_track"), 7850, False, True, FULL_SIMPLE, "hinge", "Pivot bracket"))
        opm = H.OPERATORS[spec["operator"]["model"]]
        hz = spec["operator"]["height"]
        if opm.kind == "t_handle":
            hb = C.add_rotary_operator(model, lb, spec, phys, opm, 1.0, 1.0, 0.0, hz - zp, t, [-1.0], None, name="t_handle")
            model.meta["operator_joint"] = hb.joint.name
        else:
            C.add_pull(model, lb, opm, 1.0, 0.0, hz - zp, t, -1.0, name="lift_handle")
        if spec["lock"].get("engaged") and not spec["lock"].get("robot_side_release"):
            j.range = (0.0, 0.01)
        _sites(world, Ho, -2.0, 2.0)
        model.meta.update({"primary_joint": "door_hinge", "handle_height": hz, "counterbalance_fraction": cb})
        if "operator_joint" not in model.meta:
            model.meta["operator_joint"] = None
        return lb
    raise ValueError(fam)

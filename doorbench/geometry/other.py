"""Sliding, folding, rotor, vertical-lift and horizontal-hinge families."""
from __future__ import annotations

import math

import numpy as np

from ..ir import (Body, Geom, Joint, Site, Equality, Tendon, Model, ALL_TIERS, FULL_ONLY, FULL_SIMPLE, QUAT_ID,
                  quat_from_axis_angle, quat_z_to)
from .. import materials as M
from .. import hardware as H
from ..folding import (FOLD_TRACK_H, FOLD_TRACK_GAP, FOLD_FLOOR_GAP, FOLD_HINGE_GAP, FOLD_PIVOT_MAX_DEG, FOLD_PIVOT_IN, FOLD_JAMB_GAP, fold_jamb_gap,
                       fold_coupling, fold_hinge_range, fold_lead_gap, fold_meeting_gap, fold_groups)
from . import common as C
from . import meshes as MESH
from .sliding_tracks import add_tracks, add_barn_hangers, add_floor_guides, add_lane_floor_guides, add_enclosed_hangers, bypass_leaf_range
from .pocket_hardware import add_pocket_edge_pull, add_recessed_pull


def _sites(world, Ho, ya=-1.5, yg=1.5):
    world.sites.append(Site("approach_point", (0, ya, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (0, yg, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (0, 0, Ho / 2), QUAT_ID, 0.02, "pass_plane"))


def slide_joint(spec, phys, axis, travel, name, initial=0.0, counterbalance=None, mass=None):
    phys = phys.get("per_body_dynamics", {}).get(name.removesuffix("_slide"), phys)
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
    hk.geoms.append(C.box(f"{name}_hook_tip", (-dir_ * 0.036, 0, 0.0), (0.006, 0.005, 0.016), hm, 7900, True, True, ALL_TIERS, "latch", "Hook tip"))
    model.add_body(hk)
    model.equalities.append(Equality("joint", f"{name}_hook_couple", hk.joint.name, driver_joint, (0, coeff, 0, 0, 0), tiers=ALL_TIERS, label="hook = driver * coeff"))
    world.geoms.append(C.box(f"{name}_hook_keeper", (xc + x_latch_edge - dir_ * 0.018, yl, z), (0.008, 0.015, 0.006), hm, 7900, True, True, ALL_TIERS, "latch", "Hook keeper bar"))
    # faceplate on the stile edge around the hook (flush) and a keeper plate around the jamb pocket mouth
    leaf_b.geoms.append(C.box(f"{name}_hook_faceplate", (x_latch_edge + dir_ * 0.0006, 0.0, z), (0.0006, 0.011, 0.045), hm, 7900, False, True, FULL_SIMPLE, "leaf", "Hook faceplate"))
    if abs(yl) < 0.05:
        model.meta.setdefault("_jamb_pockets", []).append((z, 0.12, yl))
        model.meta.setdefault("_jamb_keeper_plates", []).append((xc + x_latch_edge - dir_ * 0.0 , yl, z, dir_, name))
    return hk


def _add_sliding_hasp(model, world, leaf_b, spec, name, dir_, x_latch_edge, xc, yl, t, z_h, Wo, latch_side, locked, material):
    """Hasp on a sliding leaf + staple on the member beside its latch edge: on sliding gates the posts stand behind
    the leaf, so the hasp sits on the back face and the staple is a bracket off the post's front face; sliding doors
    carry the hasp on the robot face with the staple standing off the latch jamb / wall face."""
    op, kin, fam = spec["opening"], spec["kinematics"], spec["family"]
    if kin.get("track") == "top_hung_pocket" or latch_side == 0:
        return
    wt = op["wall_thickness"]
    toward = (-dir_, 0.0, 0.0)
    hinge_x = x_latch_edge + dir_ * 0.06
    x_st = latch_side * (Wo / 2 + 0.03)
    if fam == "gate_sliding":
        # hasp on the robot face (it flips open into free space); the staple is a bracket welded to the post that
        # stands behind the leaf, reaching forward past the leaf's edge to the strap plane
        face, y_surf, plane_h = -1.0, 0.02, 0.006
    else:
        face = -1.0
        y_surf = -wt / 2
        plane_h = max(0.006, abs(y_surf) - abs(yl - t / 2) + 0.004) if abs(y_surf) > abs(yl - t / 2) else 0.006
    y_plane = yl + face * (t / 2 + plane_h)
    y_eye = y_plane + face * (0.003 + 0.008)
    strap_len = abs(x_st - (xc + hinge_x)) + 0.022
    mat = C.mat_from_material(model, material, f"mat_op_{material}")
    pm = C.mat_from_material(model, "brass", "mat_padlock")
    C.add_hasp_assembly(model, leaf_b, world, name, (hinge_x, face * t / 2, z_h), (0, face, 0), toward, strap_len, plane_h, (x_st, y_eye, z_h), (x_st, y_surf, z_h), (0, -1.0, 0), locked, mat, pm)


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
    if track in ("surface_flat_track", "top_hung_industrial") or (fam == "sliding_single" and track == "top_hung" and not fixed_panel):
        y_leaf = -(wt / 2 + t / 2 + 0.02)   # barn / industrial / wall-mounted top-hung: hangs in front of the wall on the robot side
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
        header_x0, header_x1 = sorted((-s_open*(Wo/2+jamb_t),far))
        world.geoms.append(C.box("pocket_structural_header", ((header_x0+header_x1)/2,0,Ho+0.04),
                                 ((header_x1-header_x0)/2,wt/2,0.04),mat_frame,500,True,True,ALL_TIERS,"frame","Continuous pocket track support header"))
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
    # fixed panel (patio / auto sliding sidelites)
    if fixed_panel and fam in ("sliding_single", "automatic_sliding"):
        gm = C.mat_from_material(model, "glass_clear", "mat_fixed_glass")
        fm2 = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
        if fam == "sliding_single" or (fam == "automatic_sliding" and not center):
            xf = s_open * (Wo / 4)  # fixed panel occupies the half the leaf slides over
            if track == "wood_groove_bottom":
                gm = C.mat_from_material(model, "washi_paper", "mat_fixed_shoji")
            world.geoms.append(C.box("fixed_panel_glass", (xf, y_leaf + 0.04, Ho / 2), (Wo / 4 - 0.02, 0.003, Ho / 2 - 0.05), gm, 2500, True, True, ALL_TIERS, "glass", "Fixed panel"))
            world.geoms.append(C.box("fixed_panel_stile", (xf - s_open * (Wo / 4 - 0.03), y_leaf + t / 2 + 0.025, Ho / 2), (0.03, 0.012, Ho / 2), fm2, 1400, True, True, FULL_SIMPLE, "frame", "Fixed panel stile"))
        else:
            for sgn in (-1, 1):
                xf = sgn * (Wo / 4 + Wo / 8)
                world.geoms.append(C.box(f"sidelite_{'r' if sgn > 0 else 'l'}", (xf, 0.042, Ho / 2), (Wo / 8 - 0.02, 0.003, Ho / 2 - 0.05), gm, 2500, True, True, ALL_TIERS, "glass", "Sidelite"))
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
        # The leaves share the opening on parallel tracks and overlap; both END leaves keep a real running clearance
        # to their jamb, closed AND at the end of their travel (they used to be authored flush on the jamb face -
        # 0.000 m - which MuJoCo at margin 0 ignores and PhysX resolves inside its contact offset).  The spec sizes
        # the opening with 10 mm of slack for exactly this; spend it as C.GAP per jamb and put the rest in the
        # leaf-to-leaf overlap.
        bypass_span = Wo - 2 * C.GAP                  # the run the leaves may occupy
        bypass_pitch = (bypass_span - W) / max(n - 1, 1)
        bypass_travel = bypass_span - W - BYPASS_END_STOP   # the track end stop, not the jamb, ends the stroke
        leaf_defs = []
        for k in range(n):
            xc = -Wo / 2 + C.GAP + W / 2 + k * bypass_pitch
            yk = y_leaf + (k - (n - 1) / 2) * (t + 0.05) * (1 if n > 1 else 0)   # one track per leaf
            leaf_defs.append((f"leaf_{k}", 1.0 if k == 0 else -1.0, xc, yk))
    elif fam in ("sliding_single", "automatic_sliding") and fixed_panel and not center:
        # leaf covers the half opposite to the fixed panel; opens by sliding over the fixed panel
        leaf_defs = [("leaf", s_open, -s_open * (Wo / 4), y_leaf)]
    else:
        leaf_defs = [("leaf", s_open, 0.0, y_leaf)]
    track_defs = add_tracks(model, world, spec, leaf_defs, y_leaf, jamb_t, tm)
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
        tr = travel if fam != "sliding_bypass" else bypass_travel
        j = slide_joint(spec, phys, (dir_, 0, 0), tr, f"{name}_slide")
        if fam == "sliding_bypass":
            # leaf 0 runs to the far jamb (+x), every other leaf runs back to the near jamb (-x); both are brought up
            # by the track end stop BYPASS_END_STOP short of the jamb's running clearance, so a hard push cannot
            # drive the leaf edge into the jamb through the joint limit's compliance
            k_ = leaf_defs.index((name, dir_, xc, yl))
            j.range = bypass_leaf_range(spec, k_)
            j.axis = (1.0, 0, 0) if k_ == 0 else (-1.0, 0, 0)
        b.joint = j
        model.add_body(b)
        C.add_leaf_geoms(model, b, spec, leaf, 1.0, -W / 2, zb, phys, name_prefix=name)
        if track_defs[name]["floor_guides_required"]:
            track_defs[name]["guide_leaf_geoms"] = [g.name for g in b.geoms if g.semantic in ("leaf", "glass")]
        if track_defs[name]["floor_guides_required"] and track != "surface_flat_track":
            add_lane_floor_guides(model, world, b, spec, zb, track_defs[name], tm)
        eb_pockets = spec["latch"]["model"] == "electric_bolt" and not center and fam != "sliding_bypass"
        # hangers / rollers (visual)
        rm = C.mat_from_material(model, "steel_galvanized", "mat_roller")
        if track == "surface_flat_track":
            add_barn_hangers(model, world, b, spec, zb, track_defs[name], rm, tm)
        elif track in ("top_hung_pocket", "top_hung_bypass"):
            add_enclosed_hangers(model, world, b, spec, zb, track_defs[name], rm, tm)
        elif track == "bottom_rolling":
            for k, xr in enumerate((-W / 2 + 0.1, W / 2 - 0.1)):
                # The rolling tread touches the rail top (24 mm); wheel is housed in the lower stile.
                wheel = C.cyl(f"{name}_roller_{k}", (xr, 0, 0.024 + 0.015), 0.015, 0.008, rm, (0, 1, 0), 7850, False, True, FULL_ONLY, "track", "Roller")
                b.geoms.append(wheel)
                track_defs[name]["rollers"].append(wheel.name)
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
            f_eb = -1.0 if (abs(yl) >= 0.05 or fixed_panel) else 1.0   # keeper on the robot face when a wall / fixed panel is behind
            x_b = xc + x_latch_edge + dir_ * 0.08
            y_b = yl + f_eb * (t / 2 + 0.014)
            zk = zb + Hh - 0.10
            eb = Body(f"{name}_electric_bolt", None, (x_b, y_b, zk + 0.035), QUAT_ID, None, [], [], ALL_TIERS, "lock", "Electric drop bolt")
            eb.joint = Joint(f"{name}_electric_bolt_slide", "slide", (0, 0, 1), (0, 0, 0), (0.0, 0.04), damping=5.0, frictionloss=2.0, role="lock", label="Electric drop bolt (0 = dropped into the keeper; lifts on access-control release)", robot_interactive=False, initial=0.0)
            eb.geoms.append(Geom(f"{name}_electric_bolt_geom", "capsule", (0.008, 0.042), (0, 0, 0), (1, 0, 0, 0), ebm, True, True, 7900.0, None, (0.4, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "lock", "Drop bolt"))
            model.add_body(eb)
            world.geoms.append(C.box(f"{name}_electric_bolt_housing", (x_b, y_b + f_eb * 0.004, zk + 0.13), (0.022, 0.014, 0.045), ebm, 7900, False, True, FULL_SIMPLE, "lock", "Solenoid housing"))
            for sx_ in (-1, 1):
                b.geoms.append(C.box(f"{name}_ebolt_keeper_{'p' if sx_ > 0 else 'n'}", (x_latch_edge + dir_ * 0.08 + sx_ * 0.018, f_eb * (t / 2 + 0.014), zk), (0.006, 0.014, 0.02), ebm, 7900, True, True, ALL_TIERS, "lock", "Keeper block"))
            b.geoms.append(C.box(f"{name}_ebolt_keeper_base", (x_latch_edge + dir_ * 0.08, f_eb * (t / 2 + 0.002), zk), (0.03, 0.002, 0.03), ebm, 7900, False, True, FULL_SIMPLE, "lock", "Keeper plate"))
            model.meta["env_release_joint"] = eb.joint.name
        if opm.kind in ("flush_pull", "pull", "ring_pull", "push_plate"):
            faces = [-1.0, 1.0] if spec["operator"].get("sides", "both") == "both" and abs(yl) < 0.05 and not fixed_panel else [-1.0]
            for f in faces:
                if opm.style_params.get("shape") in ("flush", "cup", "hikite"):
                    inset = 0.045 if leaf.get("panel_style") in ("louver_full", "louver_half") else 0.09
                    add_recessed_pull(model, b, opm, dir_, x_latch_edge + dir_ * inset, hz, t, f, name=f"{name}_pull")
                else:
                    C.add_pull(model, b, opm, dir_ * -1.0, x_latch_edge + dir_ * 0.09, hz, t, f, name=f"{name}_pull")
            if fixed_panel and spec["operator"].get("sides", "both") == "both" and opm.id != "pull_flush_recessed":
                add_recessed_pull(model, b, H.OPERATORS["pull_flush_recessed"], dir_, x_latch_edge + dir_ * 0.09, hz, t, 1.0, name=f"{name}_ext_pull")
            if opm.id == "barn_privacy_hook":
                # teardrop latch on robot face at latch edge pivoting about y; hooks over a keeper on the jamb
                lm = C.mat_from_material(model, "black_matte_metal", "mat_teardrop")
                td = Body(f"{name}_teardrop", b.name, (x_latch_edge + dir_ * 0.06, -(t / 2 + 0.01), hz + 0.45), QUAT_ID, None, [], [], FULL_SIMPLE, "latch", "Teardrop latch")
                td.joint = Joint(f"{name}_teardrop_hinge", "hinge", (0, dir_, 0), (0, 0, 0), (0.0, 1.4), damping=0.02, frictionloss=0.02, role="operator", label="Teardrop latch (0 = dropped over the keeper, + = lifted)", initial=0.0)
                model.meta["operator_joint"] = td.joint.name
                td.geoms.append(C.box(f"{name}_teardrop_top", (-dir_ * 0.04, 0, -0.005), (0.04, 0.004, 0.006), lm, 7800, True, True, FULL_SIMPLE, "latch", "Teardrop bar"))
                td.geoms.append(C.box(f"{name}_teardrop_end", (-dir_ * 0.078, 0, -0.03), (0.005, 0.004, 0.03), lm, 7800, True, True, FULL_SIMPLE, "latch", "Teardrop end"))
                td.sites.append(Site(f"{name}_teardrop_grip", (-dir_ * 0.078, -0.01, -0.05), QUAT_ID, 0.01, "grip"))
                model.add_body(td)
                world.geoms.append(C.box(f"{name}_teardrop_keeper_post", (xc + x_latch_edge - dir_ * 0.005, yl - (t / 2 + 0.01), hz + 0.45 - 0.034), (0.005, 0.006, 0.018), lm, 7800, True, True, FULL_SIMPLE, "latch", "Keeper post"))
                world.geoms.append(C.box(f"{name}_teardrop_keeper_base", (xc + x_latch_edge - dir_ * 0.005, yl - (t / 2 + 0.024), hz + 0.45 - 0.03), (0.005, 0.008, 0.02), lm, 7800, False, True, FULL_ONLY, "latch", "Keeper base"))
        elif opm.kind in ("hook_lock_slider",):
            # exterior face passes the fixed panel: flush pull there, lever handle on the robot face only
            hb = C.add_rotary_operator(model, b, spec, phys, H.OPERATORS["lever_l_shape"], -dir_, 1.0, x_latch_edge + dir_ * 0.06, hz, t, [-1.0], None, name=f"{name}_handle")
            # A thin-glass cup projects behind its prepared cutout. Keep it
            # below the inside hook lever's sweep, not through that mechanism.
            exterior_pull_z = hz - .16 if t < .028 else hz
            add_recessed_pull(model, b, H.OPERATORS["pull_flush_recessed"], dir_, x_latch_edge + dir_ * 0.09, exterior_pull_z, t, 1.0, name=f"{name}_ext_pull")
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
            if fixed_panel or abs(yl) >= 0.05:
                inside = -1.0      # the far face runs past the fixed panel / wall: thumbturn on the robot face
            eng = engaged
            tt = Body(f"{name}_hook_thumbturn", b.name, (x_latch_edge + dir_ * 0.06, inside * t / 2, hz + 0.22), QUAT_ID, None, [], [], ALL_TIERS, "lock", "Hook thumbturn")
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
            # drop bolt (cane bolt, e.g. 12 in cane bolt): vertical rod in guide loops on the leaf face, bent-over
            # handle at the top, drops into a floor socket that blocks the travel
            mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
            dia = opm.style_params.get("diameter", 0.02)
            xb = x_latch_edge + dir_ * 0.12
            L = zb + 0.30 - 0.026
            sb, info = C.add_barrel_bolt(model, b, f"{name}_slide_bolt", (xb, -t / 2, zb), (0, 0, -1), (0, -1, 0), L, dia, 0.08, engaged, mat, protrusion=zb - 0.026, standoff=dia, role="lock", label="Drop bolt (0 = in floor socket, + = lifted)", frictionloss=opm.spring_torque_preload, damping=2.0, handle_at="rear", handle_len=0.06, joint_name=f"{name}_slide_bolt_slide", grip_site=f"{name}_grip_bolt")
            km = C.mat_from_material(model, "steel_galvanized", "mat_keeper")
            yk = yl - (t / 2 + dia)
            for sx_ in (-1, 1):
                world.geoms.append(C.box(f"{name}_socket_{'p' if sx_ > 0 else 'n'}", (xc + xb + sx_ * (dia / 2 + 0.006), yk, 0.04), (0.004, dia / 2 + 0.01, 0.02), km, 7800, True, True, ALL_TIERS, "lock", "Floor socket"))
            world.geoms.append(C.box(f"{name}_socket_b", (xc + xb, yk + (dia / 2 + 0.006), 0.04), (dia / 2 + 0.01, 0.004, 0.02), km, 7800, True, True, ALL_TIERS, "lock", "Floor socket"))
            world.geoms.append(C.box(f"{name}_socket_base", (xc + xb, yk, 0.01), (dia / 2 + 0.014, dia / 2 + 0.014, 0.01), km, 7800, False, True, FULL_SIMPLE, "lock", "Socket base"))
            model.meta["operator_joint"] = sb.joint.name
        elif opm.kind == "hasp":
            # hasp on the leaf's back face, staple on a standoff from the post that stands behind the leaf's path
            _add_sliding_hasp(model, world, b, spec, name, dir_, x_latch_edge, xc, yl, t, hz, Wo, latch_side, engaged and lk.kind == "padlock", opm.material)
            if engaged and lk.kind == "padlock":
                j.range = (0.0, 0.003)
        elif opm.kind == "none":
            pass
        if lk.kind == "padlock" and opm.kind != "hasp" and fam in ("sliding_single", "gate_sliding") and not center:
            _add_sliding_hasp(model, world, b, spec, name, dir_, x_latch_edge, xc, yl, t, min(hz + 0.20, zb + Hh - 0.08), Wo, latch_side, engaged and not release, "steel_galvanized")
        if spec["lock"]["model"] == "electric_bolt" and not eb_pockets and fam in ("gate_sliding", "automatic_sliding") and not center:
            # electric drop bolt LOCK on a leaf whose latch is not the bolt: same world-fixed solenoid + keeper geometry
            ebm = C.mat_from_material(model, "stainless", "mat_ebolt")
            f_eb = -1.0 if (abs(yl) >= 0.05 or fixed_panel) else 1.0
            x_b = xc + x_latch_edge + dir_ * 0.08
            y_b = yl + f_eb * (t / 2 + 0.014)
            zk = zb + Hh - 0.10
            eb = Body(f"{name}_electric_bolt", None, (x_b, y_b, zk + 0.035), QUAT_ID, None, [], [], FULL_SIMPLE, "lock", "Electric drop bolt")
            eb.joint = Joint(f"{name}_electric_bolt_slide", "slide", (0, 0, 1), (0, 0, 0), (0.0, 0.04), damping=5.0, frictionloss=2.0, role="lock", label="Electric drop bolt (0 = dropped into the keeper; lifts on access-control release)", robot_interactive=False, initial=0.0 if engaged else 0.04)
            eb.geoms.append(Geom(f"{name}_electric_bolt_geom", "capsule", (0.008, 0.042), (0, 0, 0), (1, 0, 0, 0), ebm, True, True, 7900.0, None, (0.4, 0.005, 0.0001), None, None, False, None, None, 0.0, FULL_SIMPLE, "lock", "Drop bolt"))
            model.add_body(eb)
            world.geoms.append(C.box(f"{name}_electric_bolt_housing", (x_b, y_b + f_eb * 0.004, zk + 0.13), (0.022, 0.014, 0.045), ebm, 7900, False, True, FULL_SIMPLE, "lock", "Solenoid housing"))
            for sx_ in (-1, 1):
                b.geoms.append(C.box(f"{name}_ebolt_keeper_{'p' if sx_ > 0 else 'n'}", (x_latch_edge + dir_ * 0.08 + sx_ * 0.018, f_eb * (t / 2 + 0.014), zk), (0.006, 0.014, 0.02), ebm, 7900, True, True, FULL_SIMPLE, "lock", "Keeper block"))
            b.geoms.append(C.box(f"{name}_ebolt_keeper_base", (x_latch_edge + dir_ * 0.08, f_eb * (t / 2 + 0.002), zk), (0.03, 0.002, 0.03), ebm, 7900, False, True, FULL_SIMPLE, "lock", "Keeper plate"))
            model.meta["env_release_joint"] = eb.joint.name
        # electric bolt / keyed lock -> lock joint
        if engaged and lk.kind in ("electric_strike", "keyed_cylinder", "padlock", "slide_bolt", "interlock") and opm.kind not in ("slide_bolt_handle",) and (not release or lk.kind in ("interlock", "electric_strike")):
            j.range = (0.0, 0.002)
            j.notes = f"{lk.name}: leaf locked (env releases on credential / call button)" if release else f"{lk.name}: leaf locked"
        b.sites.append(Site(f"{name}_edge_mid", (x_latch_edge, 0, zb + Hh / 2), QUAT_ID, 0.02, "leaf_edge"))
        if track == "top_hung_pocket":
            # Keep the separate retracting hook lock's complete arc clear of
            # the edge-pull rocker, including the hook's released position.
            edge_pull_z = hz-0.12 if lk.kind == "hook_lock" else hz
            add_pocket_edge_pull(model, b, spec, dir_, x_latch_edge, edge_pull_z, tr)
            # A boxed relief in the closing jamb accepts the finger paddle
            # during return/closure; deployed hardware must not cut through
            # solid timber if the user keeps holding it while seating the leaf.
            jamb_pockets.append((edge_pull_z,0.14,yl))
        if track_defs[name]["floor_guides_required"]:
            track_defs[name]["guide_leaf_geoms"] = [g.name for g in b.geoms if g.collision and g.semantic in ("leaf", "glass")]
        model.meta.setdefault("sliding_leaf_controls", []).append({
            "body": name, "joint": j.name, "grip_sites": [s.name for s in b.sites if s.role == "grip" and "close_grip" not in s.name],
            "operator_grip_sites": [s.name for child in model.bodies if child.parent == b.name and child.name != f"{name}_edge_pull" for s in child.sites if s.role == "grip"],
            "open_direction": list(j.axis), "nominal_range": list(track_defs[name]["nominal_range"]),
            "robot_face": -1, "independent": fam == "sliding_bypass",
            "manual_contact_required": fam not in ("automatic_sliding", "elevator"),
            "access_note": "Rear-lane pulls may be occluded by foreground leaves; check reach before selecting a leaf." if fam == "sliding_bypass" else None,
        })
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
        km_ = C.mat_from_material(model, "stainless", "mat_hook")
        for (_, yc_, zc_, d_, nm_) in model.meta.pop("_jamb_keeper_plates", []):
            C.add_keeper_ring(world.geoms, f"{nm_}_hook_keeper_plate", (latch_side * (Wo / 2), yc_, zc_), (-latch_side, 0, 0), (0, 0, 1), t / 2 + 0.006, 0.06, km_, bar=0.006, thick=0.001, tiers=FULL_SIMPLE, semantic="latch", label="Hook keeper plate")
    else:
        model.meta.pop("_jamb_keeper_plates", None)
    if track == "surface_flat_track":
        add_floor_guides(model, world, spec, s_open, y_leaf, tm, track_defs["leaf"])
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
    if fam == "sliding_bypass":
        model.meta["interaction_mode"] = "independent_bypass"
        model.meta["preferred_manual_leaf"] = bodies[0].joint.name
    if fam == "elevator":
        from .elevator import rebuild_elevator
        rebuild_elevator(model, spec, bodies)
    return bodies


# ---------------------------------------------------------------------------
# Folding: bifold / accordion
# ---------------------------------------------------------------------------
def build_folding(spec, phys, model: Model):
    """Bifold / accordion (concertina) door.

    The real mechanism: panel 0 turns on a jamb pivot (top and bottom pins 35 mm in from its edge); every further
    panel hangs on a piano / butt hinge along the previous panel's edge, and every second hinge line rides in the
    top track on a glide.  A hinge lets two panels fold flat onto each other only when its axis lies on the pair of
    faces that come together, so the axis alternates between the two faces of the door (+y / -y) from hinge to
    hinge.  For equal panels the track makes the hinge angles q_k = -2 q0 (odd k) / +2 q0 (even k): the panels
    zigzag, the on-track hinge lines stay on the track line and the lead edge travels along the track towards the
    pivot jamb.  The couplings are joint equalities; the driven hinges get the half-turn range on the side the
    coupling drives them to ([-pi, 0] odd / [0, pi] even) so their limits never fight the coupling.  With the axes
    on the face planes the panels stack face to face at the fold without passing through each other; the pivot
    stop (85 deg = a 170 deg fold) is the stack limit.  The panels hang FOLD_TRACK_GAP below the track, which is
    mounted under the head jamb, and FOLD_FLOOR_GAP above the floor; panel edges stop FOLD_HINGE_GAP short of every
    hinge axis, so no panel touches a neighbour, the track or the frame anywhere in the travel.
    """
    leaf = spec["leaf"]
    W, Hh, t = leaf["width"], leaf["height"], leaf["thickness"]
    op = spec["opening"]
    Wo, Ho = op["width"], op["height"]
    n = leaf["count"]
    accordion = bool(spec["kinematics"].get("accordion"))
    zb = FOLD_FLOOR_GAP
    if zb + Hh + FOLD_TRACK_GAP > Ho - FOLD_TRACK_H + 1e-9:
        raise ValueError(f"{spec['id']}: {Hh:.3f} m folding panels do not clear the top track in a {Ho:.3f} m opening "
                         f"(need >= {zb + Hh + FOLD_TRACK_GAP + FOLD_TRACK_H:.3f} m)")
    world = C.add_floor_and_wall(model, spec)
    C.add_frame(model, spec, 1.0, world, with_stop=False, strike_pockets=None, u=1.0)
    tm = C.mat_from_material(model, "aluminum", "mat_track")
    if accordion:
        world.geoms.append(C.box("fold_track", (0, 0, Ho - FOLD_TRACK_H / 2), (Wo / 2, 0.02, FOLD_TRACK_H / 2), tm, 2700, False, True, FULL_SIMPLE, "track", "Top track"))
    else:
        from .folding_supports import add_bifold_track
        add_bifold_track(model, world, spec, tm)
    hg = H.HINGES[spec["hinge"]["model"]]
    km = C.mat_from_material(model, "steel_galvanized" if hg.bearing != "rusty" else "steel_rusty", "mat_hinge")
    v = -1.0   # folds toward the robot
    rf = phys["roller"]
    bodies = []
    opm = H.OPERATORS[spec["operator"]["model"]]
    q_max = math.radians(min(float(spec["kinematics"].get("max_open_deg") or FOLD_PIVOT_MAX_DEG), FOLD_PIVOT_MAX_DEG))
    n_groups = fold_groups(n, accordion)
    groups = [(1.0, -Wo / 2)] if n_groups == 1 else [(1.0, -Wo / 2), (-1.0, Wo / 2)]
    per_group = n // n_groups
    # Face-hinged zigzag: the lead edge first moves OUT along the track before it comes back (every panel link is
    # tilted by its thickness), by fold_lead_excursion(); the closed lead gap swallows that (folding.fold_lead_gap) or
    # the lead edge jams on the strike jamb a few degrees into the travel.  The spec sizes the opening for it.
    for gi, (u, hx) in enumerate(groups):
        prev_name = None
        # closed lead edge: the strike jamb (one stack) or the meeting line at the opening centre (two stacks, half a gap each)
        x_lead = u * (Wo / 2 - fold_lead_gap(per_group, W, t)) if n_groups == 1 else -u * fold_meeting_gap(per_group, W, t)
        for k in range(per_group):
            name = f"panel_{gi}_{k}"
            last = k == per_group - 1
            if k == 0:
                b = Body(name, None, (hx, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Fold panel")
                j = Joint(f"{name}_hinge", "hinge", (0, 0, u * v), (u * FOLD_PIVOT_IN, 0, 0), (0.0, q_max), damping=rf["viscous_damping_N_s_per_m"] * 0.2 + 0.2, frictionloss=rf["coulomb_force_N"] * 0.3 + 0.1, armature=0.005, role="primary", label="Pivot panel (0 = closed, + = folding open)")
                x_a = fold_jamb_gap(t)
            else:
                c = fold_coupling(k)
                side = 1.0 if v * c > 0 else -1.0      # the panel swings to +/-y: the hinge axis is on that face of both panels
                # child frame continues the parent's centre plane (slab centred at y = 0); the axis is on the closing face
                b = Body(name, prev_name, (u * W, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Fold panel")
                j = Joint(f"{name}_hinge", "hinge", (0, 0, u * v), (0, side * t / 2, 0), fold_hinge_range(k), damping=0.2, frictionloss=0.1, armature=0.002, role="secondary", label=f"Panel-to-panel hinge (track-driven, q = {c:+.0f} q_pivot)", robot_interactive=False)
                x_a = FOLD_HINGE_GAP
            b.joint = j
            model.add_body(b)
            if last:
                # the lead slab runs to the closed lead edge (the strike jamb / meeting line), no hinge there; the spec
                # sizes the opening so this is ~W (mm rounding of W and Wo aside)
                x_b = abs(x_lead - (hx + u * k * W))
                if not 0.8 * W <= x_b <= W + 0.02:
                    raise ValueError(f"{spec['id']}: lead panel would be {x_b:.3f} m for {W:.3f} m panels - the opening ({Wo:.3f} m) does not fit the stack")
            else:
                x_b = W - FOLD_HINGE_GAP
            C.add_leaf_geoms(model, b, spec, leaf, u, u * x_a, zb, phys if k == 0 else None, name_prefix=name, W=x_b - x_a)
            if k > 0:
                model.equalities.append(Equality("joint", f"{name}_couple", j.name, f"panel_{gi}_0_hinge", (0, c, 0, 0, 0), tiers=ALL_TIERS, label=f"q = {c:+.0f} * q_pivot (track-guided fold)"))
                # piano-hinge knuckle on the axis (in the FOLD_HINGE_GAP between the two panel edges)
                b.geoms.append(C.cyl(f"{name}_knuckle", (0, side * t / 2, zb + Hh / 2), 0.0045, Hh / 2 - 0.01, km, (0, 0, 1), 7850, False, True, FULL_SIMPLE, "hinge", "Piano hinge knuckle"))
            if last:
                # Johnson 100FD's installation drawing centres the knob on
                # each leading panel. A grip beside the track guide has almost
                # no outward-pull leverage in the closed configuration.
                grip_x = (x_a + x_b) / 2 if not accordion else x_b - 0.05
                hz = spec["operator"]["height"]
                if not accordion and "louver" in leaf["panel_style"]:
                    # A through-bolted knob needs solid wood, not floating
                    # attachment to a gap between the louver slats.
                    b.geoms.append(C.box(f"{name}_operator_rail", (u * grip_x, 0, hz),
                        ((x_b - x_a) / 2, t / 2, 0.035), b.geoms[0].material, 600,
                        True, True, ALL_TIERS, "leaf", "Solid crossrail carrying the bifold pull"))
                if opm.kind == "knob":
                    key, mesh = MESH.knob_mesh(shape="round", diameter=0.03, depth=0.03, rose_diameter=0.0)
                    mat = C.mat_from_material(model, opm.material, f"mat_op_{opm.material}")
                    b.geoms.append(C.mesh_geom(f"{name}_knob", key, mesh, (u * grip_x, -t / 2, hz), C.q_face(-1.0, u), mat, 3000, False, ALL_TIERS, "operator", "Bifold knob"))
                    b.geoms.append(C.sphere(f"{name}_knob_col", (u * grip_x, -(t / 2 + 0.03), hz), 0.016, mat, 3000, True, ALL_TIERS, "operator", "Knob grip"))
                    b.sites.append(Site(f"{name}_grip", (u * grip_x, -(t / 2 + 0.03), hz), QUAT_ID, 0.012, "grip"))
                elif opm.kind == "flush_pull":
                    add_recessed_pull(model, b, opm, u, u*(x_b-.06), hz, t, -1.0, name=f"{name}_pull")
                else:
                    C.add_pull(model, b, opm, u, u * (grip_x if not accordion else x_b - 0.06), hz, t, -1.0, name=f"{name}_pull")
            bodies.append(b)
            prev_name = name
    if H.LATCHES[spec["latch"]["model"]].kind == "magnetic":
        model.meta.setdefault("notes", []).append("magnetic catch not simulated (holding force in spec.physics.latch)")
    if not accordion:
        from .folding_supports import add_bifold_supports
        add_bifold_supports(model, world, spec, bodies, groups)
    _sites(world, Ho)
    model.meta.update({"primary_joint": "panel_0_0_hinge", "secondary_joint": "panel_1_0_hinge" if len(groups) > 1 else None, "operator_joint": None, "handle_height": spec["operator"]["height"]})
    return bodies


# ---------------------------------------------------------------------------
# Rotors: revolving door, turnstiles
# ---------------------------------------------------------------------------
REVOLVING_RUN_CLEAR = 0.015   # m; running clearance between the rotor top (rails / stiles / shaft) and the canopy ceiling
BYPASS_END_STOP = 0.020       # m; rubber track end stop that brings a bypass leaf up short of the jamb
HATCH_HINGE_OUT = 0.010       # m; hatch hinge barrel outboard of the curb's inner face (surface hinge on the curb)
ROLLUP_ASTRAGAL = 0.012       # m; rubber bottom seal under a roll-up curtain's steel bottom bar
ROTOR_RUN_CLEAR = 0.015       # m; running clearance at the ends of a full-height turnstile rotor column (roof / floor)
ROTOR_MIN_CLEAR = 0.010       # m; what the running-clearance gate demands of a rotor (real revolving / turnstile: 10-20 mm)


def build_revolving(spec, phys, model: Model):
    leaf = spec["leaf"]
    Hh, t = leaf["height"], leaf["thickness"]
    op = spec["opening"]
    D = op["drum_diameter"]
    R = D / 2
    wings = leaf["count"]
    open_deg = op.get("drum_opening_deg", 100)
    # Rotor envelope and enclosure heights.  The wings (glass + top/bottom rails + tip stiles) and the centre shaft all
    # end at z_top; the drum ceiling (canopy underside) sits REVOLVING_RUN_CLEAR above that, as on a real revolving
    # door where the wing top seals brush the ceiling with ~10-20 mm running clearance.  The enclosure fills the wall
    # opening (spec opening height = wing height + 0.3), so the wall header sits ON the canopy, never at wing height:
    # a header coplanar with the wing tops made a zero-gap box-box touch whose degenerate contact normal (orthogonal to
    # the only DOF) produced kN forces and jammed the rotor.
    z_bot = 0.02                       # rotor bottom above the floor ring (floor brush seal)
    z_top = Hh + 0.07                  # top of the top rails / stiles / shaft
    z_ceiling = z_top + REVOLVING_RUN_CLEAR   # canopy underside
    z_canopy_top = max(float(op["height"]), z_ceiling + 0.1)
    world = C.add_floor_and_wall(model, spec, wall_half_width=max(3.0, R + 1.5), hole=(-R - 0.05, R + 0.05, 0.0, z_canopy_top), wall_height=z_canopy_top + 0.3)
    fm = C.mat_from_material(model, op["frame"]["material"], "mat_frame")
    gm = C.mat_from_material(model, "glass_clear", "mat_drum_glass")
    # drum: segments on ±x sides covering angles outside the openings (openings centered on ±y); glass runs floor to ceiling
    nseg = 28
    half_open = math.radians(open_deg) / 2
    for i in range(nseg):
        a = 2 * math.pi * (i + 0.5) / nseg
        # opening if angle within half_open of +-90deg
        if abs(((a - math.pi / 2 + math.pi) % (2 * math.pi)) - math.pi) < half_open or abs(((a + math.pi / 2 + math.pi) % (2 * math.pi)) - math.pi) < half_open:
            continue
        seg_len = 2 * math.pi * (R + 0.02) / nseg
        q = quat_from_axis_angle([0, 0, 1], a + math.pi / 2)
        world.geoms.append(C.box(f"drum_{i}", ((R + 0.02) * math.cos(a), (R + 0.02) * math.sin(a), z_ceiling / 2), (seg_len / 2 + 0.002, 0.006, z_ceiling / 2), gm, 2500, True, True, ALL_TIERS, "glass", "Drum glass", quat=q))
    world.geoms.append(C.cyl("drum_canopy", (0, 0, (z_ceiling + z_canopy_top) / 2), R + 0.08, (z_canopy_top - z_ceiling) / 2, fm, (0, 0, 1), 300, True, True, ALL_TIERS, "frame", "Canopy"))
    # top bearing boss on the ceiling and floor pivot boss on the floor ring: the shaft runs between them with the running clearance
    boss_h = 0.006
    world.geoms.append(C.cyl("rotor_top_bearing", (0, 0, z_ceiling - boss_h / 2), 0.075, boss_h / 2, fm, (0, 0, 1), 2700, True, True, FULL_SIMPLE, "frame", "Top bearing housing"))
    world.geoms.append(C.cyl("drum_floor_ring", (0, 0, 0.003), R + 0.08, 0.003, C.mat_from_material(model, "stainless", "mat_floor_ring"), (0, 0, 1), 7900, False, True, FULL_ONLY, "frame", "Floor ring"))
    world.geoms.append(C.cyl("rotor_floor_pivot", (0, 0, 0.006 + boss_h / 2), 0.075, boss_h / 2, fm, (0, 0, 1), 2700, True, True, FULL_SIMPLE, "frame", "Floor pivot housing"))
    rotor = Body("rotor", None, (0, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Rotor")
    hf = phys["hinge"]
    rotor.joint = Joint("rotor_hinge", "hinge", (0, 0, 1), (0, 0, 0), None, damping=spec["kinematics"].get("speed_governor_damping", 30.0), frictionloss=hf["coulomb_torque_Nm"] + 2.0, armature=0.5, role="primary", label="Rotor (unbounded, + = CCW from above)")
    model.add_body(rotor)
    rotor.geoms.append(C.cyl("rotor_shaft", (0, 0, (z_bot + z_top) / 2), 0.06, (z_top - z_bot) / 2, fm, (0, 0, 1), 2700, True, True, ALL_TIERS, "leaf", "Center shaft"))
    for k in range(wings):
        a = 2 * math.pi * k / wings
        q = quat_from_axis_angle([0, 0, 1], a)
        Wl = R - 0.04
        rotor.geoms.append(C.box(f"wing_{k}_glass", (Wl / 2 * math.cos(a) + 0.02 * math.cos(a), Wl / 2 * math.sin(a) + 0.02 * math.sin(a), Hh / 2 + 0.05), (Wl / 2, t / 2, Hh / 2 - 0.05), gm, 2500, True, True, ALL_TIERS, "glass", f"Wing {k + 1} glass", quat=q, mass=phys["mass"]["total_kg"]))
        # tip stile: same z envelope as the rails (its outer face at R runs 14 mm inside the drum glass: brush-seal gap)
        rotor.geoms.append(C.box(f"wing_{k}_stile", ((R - 0.02) * math.cos(a), (R - 0.02) * math.sin(a), (z_bot + z_top) / 2), (0.02, 0.03, (z_top - z_bot) / 2), fm, 2700, True, True, ALL_TIERS, "leaf", "Wing stile", quat=q))
        rotor.geoms.append(C.box(f"wing_{k}_rail_b", (Wl / 2 * math.cos(a), Wl / 2 * math.sin(a), z_bot + 0.04), (Wl / 2, 0.03, 0.04), fm, 2700, True, True, FULL_SIMPLE, "leaf", "Bottom rail", quat=q))
        rotor.geoms.append(C.box(f"wing_{k}_rail_t", (Wl / 2 * math.cos(a), Wl / 2 * math.sin(a), z_top - 0.04), (Wl / 2, 0.03, 0.04), fm, 2700, True, True, FULL_SIMPLE, "leaf", "Top rail", quat=q))
        # A radial push bar is carried by the central shaft and outer stile.
        # Its two mounting legs create the actual load path; it never floats
        # beside the glass. Bare-glass variants get an explicit glass push face.
        radial = np.array([math.cos(a), math.sin(a), 0.])
        tangent = np.array([-math.sin(a), math.cos(a), 0.])
        contact = radial*(R*.6) + np.array([0., 0., 1.])
        surface_geom = f"wing_{k}_glass"
        if spec["operator"]["model"] in ("pull_d", "push_plate"):
            pm = C.mat_from_material(model, "stainless", "mat_op_stainless")
            offset = -(t/2+.05)
            r0, r1 = .04, R-.02
            center = radial*((r0+r1)/2) + tangent*offset + np.array([0.,0.,1.])
            surface_geom = f"wing_{k}_bar"
            rotor.geoms.append(C.cyl(surface_geom, tuple(center), .012, (r1-r0)/2,
                pm, tuple(radial), 7900, True, True, ALL_TIERS, "operator", "Radial push bar"))
            supports=[]
            for tag, rr, start, support in (("inner",r0,-.035,"rotor_shaft"),
                                            ("outer",r1,-.022,f"wing_{k}_stile")):
                nm=f"wing_{k}_bar_mount_{tag}"
                pos=radial*rr+tangent*((start+offset)/2)+np.array([0.,0.,1.])
                rotor.geoms.append(C.cyl(nm,tuple(pos),.009,abs(offset-start)/2,
                    pm,tuple(tangent),7900,True,True,ALL_TIERS,"operator","Push bar mounting leg"))
                supports.append([surface_geom,nm]);supports.append([nm,support])
            model.meta.setdefault("rotating_hardware",[]).append({"kind":"revolving_push_bar",
                "bar_geom":surface_geom,"attachments":supports})
            contact += tangent*(offset-.012)
        else:
            contact -= tangent*t/2
        site=f"wing_{k}_push"
        rotor.sites.append(Site(site,tuple(contact),tuple(quat_z_to(-tangent)),.012,"push"))
        model.meta.setdefault("rotating_contacts",[]).append({"site":site,"geom":surface_geom,
            "joint":"rotor_hinge","action":"push","positive_torque":True})
    _sites(world, Hh, -R - 1.0, R + 1.0)
    model.meta.update({"primary_joint": "rotor_hinge", "operator_joint": None, "handle_height": 1.0, "drum_diameter": D,
                       "running_clearance_min": ROTOR_MIN_CLEAR})
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
        # the arms sweep a tilted disc that passes INTO the housing (as on real tripods): the upper housing is a
        # U-shaped recess (two end blocks + back plate); the lower body is solid
        z_rec = cab_h - 0.05 - 0.33
        world.geoms.append(C.box("cabinet", (xc, 0, z_rec / 2), (cab_w / 2, cab_d / 2, z_rec / 2), fm, 800, True, True, ALL_TIERS, "frame", "Turnstile cabinet"))
        for sy in (-1, 1):
            world.geoms.append(C.box(f"cabinet_end_{'p' if sy > 0 else 'n'}", (xc, sy * (cab_d / 2 - 0.06), (z_rec + cab_h) / 2), (cab_w / 2, 0.06, (cab_h - z_rec) / 2), fm, 800, True, True, ALL_TIERS, "frame", "Housing end"))
        world.geoms.append(C.box("cabinet_back", (xc - cab_w / 2 + 0.01, 0, (z_rec + cab_h) / 2), (0.01, cab_d / 2, (cab_h - z_rec) / 2), fm, 800, True, True, ALL_TIERS, "frame", "Housing back"))
        for sy in (-1, 1):
            world.geoms.append(C.box(f"cabinet_top_{'p' if sy > 0 else 'n'}", (xc, sy * (cab_d / 2 - 0.06), cab_h + 0.01), (cab_w / 2 + 0.01, 0.07, 0.01), fm, 7900, True, True, FULL_SIMPLE, "frame", "Cabinet top"))
        # opposite side guide rail
        world.geoms.append(C.box("guide_rail_post", (xc + cab_w / 2 + 0.27 + 0.62, 0, 0.5), (0.02, 0.45, 0.5), fm, 7900, True, True, ALL_TIERS, "frame", "Guide rail"))
        # rotor
        ax = np.array([0.0, math.sin(math.radians(45)), math.cos(math.radians(45))])
        boss = 0.27
        rotor = Body("rotor", None, (xc + cab_w / 2 + boss + 0.02, 0.0, cab_h - 0.05), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Tripod rotor")
        rotor.geoms.append(C.cyl("hub_boss",tuple(-ax*.080),.018,.085,fm,tuple(ax),
                                7900,True,True,ALL_TIERS,"mechanism","Coaxial rotor journal"))
        # Fixed support stays behind the tilted rotor plane. The hollow housing
        # has a 0.5 mm radial assembly clearance; the ideal hinge represents
        # the internal bearing contact and axial retention.
        frame_q=quat_z_to(ax)
        frame_R=np.column_stack((np.array([1.,0.,0.]),np.cross(ax,[1.,0.,0.]),ax))
        bearing_names=[]
        for axis_id in (0,1):
            for side in (-1,1):
                local=np.array([0.,0.,-.1425]);local[axis_id]=side*.03175
                half=[.045,.045,.0475];half[axis_id]=.01325
                name=f"tripod_bearing_{axis_id}_{side}"
                # frame_q has its X axis unchanged for this YZ-plane axis.
                world.geoms.append(C.box(name,tuple(np.asarray(rotor.pos)+frame_R@local),tuple(half),
                    fm,7900,True,True,ALL_TIERS,"hinge","Open rotor bearing housing",quat=frame_q))
                bearing_names.append(name)
        back=np.asarray(rotor.pos)-ax*.196
        world.geoms.append(C.box("tripod_bearing_back",tuple(back),(.045,.045,.006),fm,7900,
            True,True,ALL_TIERS,"hinge","Bearing support backplate",quat=frame_q))
        x0=xc-cab_w/2+.012
        world.geoms.append(C.box("tripod_support_beam",((x0+back[0])/2,back[1],back[2]),
            ((back[0]-x0)/2,.020,.020),fm,7900,True,True,ALL_TIERS,"frame","Cabinet-anchored bearing cantilever"))
        model.meta.setdefault("rotating_hardware",[]).append({"kind":"tripod_journal",
            "shaft":"hub_boss","joint":"rotor_hinge","bearing_geoms":bearing_names,
            "attachments":[["tripod_support_beam","cabinet_back"],["tripod_support_beam","tripod_bearing_back"]]
                          +[[n,"tripod_bearing_back"] for n in bearing_names],
            "journal_clearance_m":.0005,
            "scope":"Coaxial geometry and fixed housing; ideal internal bearing/retention; separate native pawl and credential mechanism"})
        rotor.joint = Joint("rotor_hinge", "hinge", tuple(ax), (0, 0, 0), None, damping=2.0, frictionloss=hf["coulomb_torque_Nm"] + 3.0, armature=0.05, role="primary", label="Tripod rotor (ratchets 120 deg; one-way enforced by env)", ratchet_one_way=bool(spec["kinematics"].get("one_way", True)))
        model.add_body(rotor)
        sm=C.mat_from_material(model, "stainless", "mat_op_stainless")
        # Separate physical primitives retain the empty spaces between arms;
        # a convex collision hull of the old combined mesh filled those spaces.
        rotor.geoms.append(C.cyl("tripod_hub",(0,0,0),.05,.06,sm,tuple(ax),7900,True,True,ALL_TIERS,"operator","Tripod arm hub"))
        # Each arm overlaps its hub seat and carries the visible contact surface.
        e1 = np.array([1.0, 0, 0])
        e2 = np.cross(ax, e1)
        for k in range(3):
            a = 2 * math.pi * k / 3
            d = math.cos(a) * e1 + math.sin(a) * e2
            rotor.geoms.append(Geom(f"arm_{k}_col", "capsule", (0.019, 0.23), tuple(d * 0.27), tuple(quat_z_to(d)), sm, True, True, 7900, 2.0, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", f"Arm {k + 1}"))
        tangent=np.cross(ax,e1)
        rotor.sites.append(Site("arm_push",tuple(e1*.45-tangent*.019),tuple(quat_z_to(-tangent)),.015,"push"))
        model.meta.setdefault("rotating_contacts",[]).append({"site":"arm_push","geom":"arm_0_col",
            "joint":"rotor_hinge","action":"push","positive_torque":True})
        world.sites.append(Site("approach_point", (xc + cab_w / 2 + 0.55, -1.2, 0), QUAT_ID, 0.05, "approach"))
        world.sites.append(Site("goal_point", (xc + cab_w / 2 + 0.55, 1.2, 0), QUAT_ID, 0.05, "goal"))
        world.sites.append(Site("door_plane_center", (xc + cab_w / 2 + 0.55, 0, 1.0), QUAT_ID, 0.02, "pass_plane"))
        model.meta.update({"primary_joint": "rotor_hinge", "operator_joint": None, "handle_height": 0.95, "ratchet_deg": 120, "one_way": True,
                           "locked": bool(spec["kinematics"].get("locked_until_credential")), "running_clearance_min": ROTOR_MIN_CLEAR})
        from .turnstile_locks import add_turnstile_locks
        add_turnstile_locks(model,world,rotor,spec,full_height=False)
        if spec['kinematics'].get('drop_arm'):
            from .turnstile_drop import add_tripod_drop_arm
            add_tripod_drop_arm(model,world,rotor,spec)
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
    # The column runs between the floor and the cage roof (roof underside at Hh + 0.10) with a real running
    # clearance at both ends: a column authored flush on the roof and the floor is a zero-gap touch whose contact
    # normal is parallel to the rotor axis and orthogonal to its only DOF - MuJoCo at margin 0 never notices, PhysX
    # resolves it inside its contact offset and the rotor jams, drifts or explodes (7 of these doors did).
    z_bot, z_top = ROTOR_RUN_CLEAR, Hh + 0.10 - ROTOR_RUN_CLEAR
    rotor.geoms.append(C.cyl("rotor_column", (0, 0, (z_bot + z_top) / 2), 0.06, (z_top - z_bot) / 2, sm, (0, 0, 1), 7900, True, True, ALL_TIERS, "leaf", "Rotor column"))
    for k in range(wings):
        a = 2 * math.pi * k / wings
        d = np.array([math.cos(a), math.sin(a), 0])
        for m in range(arms):
            z = 0.15 + m * (Hh - 0.3) / (arms - 1)
            rotor.geoms.append(Geom(f"wing_{k}_arm_{m}", "capsule", (0.019, Rr / 2), tuple(d * Rr / 2 + np.array([0, 0, z])), tuple(quat_z_to(d)), sm, True, True, 7900, 0.9, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS if m % 2 == 0 else FULL_SIMPLE, "operator", "Arm"))
        # Choose an actual arm retained in every tier, not an empty point
        # between horizontal rungs. Surface normal faces the approaching hand.
        arm_index=min(range(0,arms,2),key=lambda index:abs(.15+index*(Hh-.3)/(arms-1)-1.0))
        contact_z=.15+arm_index*(Hh-.3)/(arms-1)
        tangent=np.cross([0.,0.,1.],d)
        rotor.sites.append(Site(f"wing_{k}_push",tuple(d*Rr*.8-tangent*.019+np.array([0.,0.,contact_z])),
                                tuple(quat_z_to(-tangent)),.015,"push"))
        model.meta.setdefault("rotating_contacts",[]).append({"site":f"wing_{k}_push",
            "geom":f"wing_{k}_arm_{arm_index}","joint":"rotor_hinge","action":"push","positive_torque":True})
    world.sites.append(Site("approach_point", (cage_R + 0.4, -1.5, 0), QUAT_ID, 0.05, "approach"))
    world.sites.append(Site("goal_point", (cage_R + 0.4, 1.5, 0), QUAT_ID, 0.05, "goal"))
    world.sites.append(Site("door_plane_center", (cage_R * 0.5, 0, 1.0), QUAT_ID, 0.02, "pass_plane"))
    model.meta.update({"primary_joint": "rotor_hinge", "operator_joint": None, "handle_height": 1.0, "ratchet_deg": 360 / wings, "one_way": bool(spec["kinematics"].get("one_way")),
                       "locked": bool(spec["kinematics"].get("locked_until_credential")), "running_clearance_min": ROTOR_MIN_CLEAR})
    from .turnstile_locks import add_turnstile_locks
    add_turnstile_locks(model,world,rotor,spec,full_height=True)
    return rotor


# ---------------------------------------------------------------------------
# Vertical lift: garage sectional, roll-up
# ---------------------------------------------------------------------------
def build_vertical(spec, phys, model: Model):
    if spec['family']=='garage_sectional':
        from .sectional import build_sectional
        return build_sectional(spec,phys,model)
    if spec['family']=='rollup':
        from .rollup import build_rollup
        return build_rollup(spec,phys,model)
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
    y_leaf = 0.0 if fam != "rollup" else (op["wall_thickness"] / 2 + 0.085)   # curtain stands off the wall so the inside lift handle clears the header
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
            # C-channel track: web outboard of the rollers + two flanges; rollers (r 25 mm) run in the cavity
            zt_, hz_ = Ho / 2 + Hh / 2, Ho / 2 + Hh / 2
            world.geoms.append(C.box(f"track_{'r' if sgn > 0 else 'l'}", (sgn * (W / 2 + 0.05), track_y, zt_), (0.004, 0.033, hz_), tm, 7850, True, True, FULL_SIMPLE, "track", "Vertical track web"))
            for sy in (-1, 1):
                world.geoms.append(C.box(f"track_{'r' if sgn > 0 else 'l'}_flange_{'p' if sy > 0 else 'n'}", (sgn * (W / 2 + 0.032), track_y + sy * 0.029, zt_), (0.022, 0.004, hz_), tm, 7850, True, True, FULL_SIMPLE, "track", "Track flange"))
            for k in range(ns + 1):
                lb.geoms.append(C.cyl(f"roller_{'r' if sgn > 0 else 'l'}_{k}", (sgn * (W / 2 + 0.03), track_y, max(0.04, 0.01 + k * (Hh / ns)) if k < ns else Hh - 0.05), 0.025, 0.01, tm, (1, 0, 0), 7850, False, True, FULL_ONLY, "track", "Track roller"))
                lb.geoms.append(C.box(f"roller_arm_{'r' if sgn > 0 else 'l'}_{k}", (sgn * (W / 2 - 0.02), track_y / 2 + t / 4, 0.01 + k * (Hh / ns) if k < ns else Hh - 0.05), (0.03, track_y / 2 - t / 4, 0.006), tm, 7850, False, True, FULL_ONLY, "track", "Roller hinge arm"))
        # torsion spring shaft
        world.geoms.append(C.cyl("torsion_shaft", (0, 0.17, Ho + 0.25), 0.013, W / 2 + 0.2, tm, (1, 0, 0), 7850, True, True, FULL_SIMPLE, "mechanism", "Torsion spring shaft"))
        world.geoms.append(C.cyl("torsion_spring", (0, 0.17, Ho + 0.25), 0.03, 0.35, C.mat_from_material(model, "steel", "mat_spring"), (1, 0, 0), 7850, False, True, FULL_ONLY, "mechanism", "Torsion spring"))
        if kin.get("opener", "none_manual") != "none_manual":
            world.geoms.append(C.box("opener_rail", (0, 1.75, Ho + 0.15), (0.03, 1.25, 0.03), tm, 7850, True, True, FULL_ONLY, "mechanism", "Opener rail"))
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
                # left guide split around the slide-lock bar (bottom-bar height): the guide segments are the keeper
                zsl_ = 0.30
                world.geoms.append(C.box("guide_l", (sgn * (W / 2 + 0.035), y_leaf, (zsl_ - 0.015) / 2), (0.03, 0.025, (zsl_ - 0.015) / 2), tm, 7850, True, True, ALL_TIERS, "track", "Curtain guide"))
                world.geoms.append(C.box("guide_l_upper", (sgn * (W / 2 + 0.035), y_leaf, (zsl_ + 0.015 + Ho) / 2), (0.03, 0.025, (Ho - zsl_ - 0.015) / 2), tm, 7850, True, True, ALL_TIERS, "track", "Curtain guide"))
                continue
            world.geoms.append(C.box(f"guide_{'r' if sgn > 0 else 'l'}", (sgn * (W / 2 + 0.035), y_leaf, Ho / 2), (0.03, 0.025, Ho / 2), tm, 7850, True, True, ALL_TIERS, "track", "Curtain guide"))
        # the curtain translates rigidly (no coiling is simulated): the coil sits in FRONT of the curtain plane, tangent to
        # it, inside an open-topped hood, so the raised curtain never intersects drum or hood (visual parts, no collision)
        world.geoms.append(C.cyl("coil_drum", (0, y_leaf + 0.278, Ho + 0.3), 0.25, W / 2 + 0.05, fm, (1, 0, 0), 300, False, True, FULL_SIMPLE, "mechanism", "Coil (visual)"))
        world.geoms.append(C.box("hood_front", (0, y_leaf + 0.54, Ho + 0.30), (W / 2 + 0.12, 0.02, 0.30), tm, 7850, False, True, FULL_SIMPLE, "mechanism", "Hood front"))
        for sx in (-1, 1):
            world.geoms.append(C.box(f"hood_end_{'r' if sx > 0 else 'l'}", (sx * (W / 2 + 0.12), y_leaf + 0.27, Ho + 0.30), (0.02, 0.29, 0.30), tm, 7850, False, True, FULL_SIMPLE, "mechanism", "Hood end plate"))
        # A real roll-up closes onto its rubber bottom astragal, not on bare steel: the bar is carried ROLLUP_ASTRAGAL
        # above the slab and the seal spans the gap.  (The steel bar used to end exactly on the floor - a 0.000 m
        # structural touch that MuJoCo at margin 0 ignores and PhysX resolves inside its contact offset.)
        lb.geoms.append(C.box("bottom_bar", (0, y_leaf, ROLLUP_ASTRAGAL + 0.03), (W / 2, t / 2 + 0.015, 0.03), tm, 7850, True, True, ALL_TIERS, "leaf", "Bottom bar", mass=3.0 * W))
        lb.geoms.append(C.box("bottom_astragal", (0, y_leaf, ROLLUP_ASTRAGAL / 2), (W / 2, t / 2 + 0.010, ROLLUP_ASTRAGAL / 2),
                              C.mat_rgba(model, "mat_astragal", (0.09, 0.09, 0.10, 1), 0.9), 1100, True, True, ALL_TIERS, "seal",
                              "Bottom astragal (rubber; the curtain seats on this, not on the steel bar)"))
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
            bar.joint = Joint(f"lock_bar_{'r' if sgn > 0 else 'l'}_slide", "slide", (-sgn, 0, 0), (0, 0, 0), (0.0, 0.03), damping=2.0, frictionloss=1.0, role="lock", label="Lock bar", robot_interactive=False, initial=0.0)
            bar.geoms.append(Geom(f"lock_bar_{'r' if sgn > 0 else 'l'}_geom", "capsule", (0.006, 0.05), (sgn * 0.04, 0, 0), tuple(quat_z_to((1, 0, 0))), bm, False, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, FULL_SIMPLE, "lock", "Lock bar (visual; engaged state locks the door joint)"))
            model.add_body(bar)
            # the bars rest extended in the track slots (spring) and retract 30 mm over the handle's travel; the coupling
            # is relative to qpos0, so an unlocked door must not start with the bars already retracted (it drove them
            # to 60 mm against a 30 mm limit and locked the handle against its own coupling)
            model.equalities.append(Equality("joint", f"lock_bar_{'r' if sgn > 0 else 'l'}_couple", bar.joint.name, hb.joint.name, (0, 0.03 / opm.travel, 0, 0, 0), tiers=FULL_SIMPLE, label="lock bar = T-handle * 0.03/travel"))
    if spec["lock"].get("engaged") and spec["lock"]["model"] in ("garage_slide_lock", "padlock", "keyed_cylinder") and (not spec["lock"].get("robot_side_release") or spec["lock"]["model"] == "keyed_cylinder"):
        j.range = (0.0, 0.003)
        j.notes = f"{spec['lock']['model']}: locked (T-handle lock bars engaged; env unlock frees the joint)"
        model.meta["locked"] = True
    # counterbalance from the actual body mass (sections + hardware)
    mtot = float(phys["mass"]["total_kg"])     # the leaf mass is reconciled to the spec after building; size the spring from it
    if cb and mtot > 0:
        k_ = 0.3 * cb * mtot * 9.81 / max(kin["travel_m"], 0.1)
        j.stiffness = k_
        j.springref = cb * mtot * 9.81 / k_
        j.notes = (j.notes + " " if j.notes else "") + f"counterbalance ~{cb:.0%} of {mtot:.0f} kg"
    if spec["lock"]["model"] == "garage_slide_lock":
        sm = C.mat_from_material(model, "steel_galvanized", "mat_slidelock")
        inside = 1.0
        y_sl = (y_leaf if fam == "rollup" else 0.0) + inside * (t / 2 + 0.01)
        z_sl = 1.0 if fam != "rollup" else 0.30         # roll-up: on the bottom bar, below the coil when raised
        if fam == "garage_sectional":
            sh_ = Hh / max(1, int(kin.get("n_sections", 4)))
            z_sl = 0.01 + (int((1.0 - 0.01) / sh_) + 0.5) * sh_      # mid-section, clear of the section hinges / roller arms
        # garage inside slide lock (e.g. National Hardware garage door side lock): plate + guides + spring bar with a
        # knob on the inside face; the bar shoots through a slot / keeper in the vertical track (roll-up: the split guide)
        eng = bool(spec["lock"].get("engaged"))
        y_face = (y_leaf if fam == "rollup" else 0.0) + inside * t / 2
        sl, _ = C.add_barrel_bolt(model, lb, "garage_slide_lock", (-W / 2, y_face, z_sl), (-1, 0, 0), (0, inside, 0), 0.14, 0.012, 0.05, eng, sm, protrusion=0.045, standoff=0.010, tiers=FULL_SIMPLE, role="lock", label="Slide lock (0 = in track, + = withdrawn)", joint_name="garage_slide_lock_slide", grip_site="slide_lock_grip", rod_semantic="lock")
        # keeper on the track: bar tip (x ~ -W/2-0.045 when engaged) captured by a U-loop off the track flange
        if fam != "rollup":
            C.add_keeper_loop(world.geoms, "slide_lock_keeper", (-W / 2 - 0.045, t / 2 + 0.027, z_sl), (-W / 2 - 0.045, y_sl, z_sl), (-1, 0, 0), (0, -1, 0), 0.006, tm, FULL_SIMPLE, base=0.03)
        else:
            # the curtain is modelled as a rigid translating sheet: raised, it (and its hardware) occupies the coil volume
            model.meta.setdefault("clearance_allow", []).append(["coil_drum", "garage_slide_lock*", "curtain coils into the drum (translation approximation)"])
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
        # Surface hinge screwed to the curb, not a pin on the lid's own edge: the barrel sits HATCH_HINGE_OUT
        # outboard of the curb's inner face, so the lid's heel corner swings up and away instead of raking across
        # the curb (a pin in the lid's mid-plane at the curb face put the corner 1.7 mm inside the curb).
        j = Joint("hatch_hinge", "hinge", (-1.0, 0, 0), (0, HATCH_HINGE_OUT, 0), (0.0, mo), damping=hf.get("air_damping_Nms_per_rad", 0.1) + 0.5, frictionloss=hf["coulomb_torque_Nm"], armature=0.02, role="primary", label="Hatch (0 = closed, + = lifted)")
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
        from .hatch_supports import add_hatch_pull, add_hatch_support
        add_hatch_pull(model, lb, spec, opm)
        add_hatch_support(model, world, lb, spec, HATCH_HINGE_OUT)
        if spec["lock"]["model"] in ("padlock", "slide_bolt") and spec["lock"].get("engaged") and not spec["lock"].get("robot_side_release"):
            j.range = (0.0, 0.005)
        if spec["lock"]["model"] == "slide_bolt":
            # barrel bolt on the operated face of the hatch shooting into a keeper loop on the curb
            bm = C.mat_from_material(model, "steel_galvanized", "mat_bolt")
            eng = bool(spec["lock"].get("engaged"))
            # the bolt sits on the leaf's top face and shoots into a keeper loop on the curb top (a ceiling hatch's
            # leaf lies inside its curb above the ceiling slab, so the bolt stays on the loft side)
            sb, _ = C.add_barrel_bolt(model, lb, "hatch_bolt", (W / 2, -Ho * 0.5, t / 2), (1, 0, 0), (0, 0, 1), 0.12, 0.012, 0.04, eng, bm, protrusion=0.036, standoff=0.010, tiers=FULL_SIMPLE, role="lock", label="Slide bolt (0 = engaged)", joint_name="hatch_bolt_slide", grip_site="hatch_bolt_grip", rod_semantic="lock")
            if eng and not spec["lock"].get("robot_side_release"):
                sb.joint.range = (0.0, 0.001)
            C.add_keeper_loop(world.geoms, "hatch_bolt_keeper", (Wo / 2 + 0.028, 0.0, zf + 0.04), (Wo / 2 + 0.028, 0.0, zf + 0.04 + t / 2 + 0.010), (1, 0, 0), (0, 0, 1), 0.006, bm, FULL_SIMPLE, base=0.03)
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
        # the flap pins hang below the frame's top rail by half the flap thickness plus a clearance: a flap swung past
        # ~45 deg lifts its top corners (t/2 sin(q) - 2 mm cos(q)) above the pin, so pins level with the rail underside
        # drove the corners ~1 mm into the rail / wall header (under the clearance gate's tolerance, 20+ kN of contact)
        flap = Body("flap", None, (0, 0, 0.05 + Ho - (t / 2 + 0.004)), QUAT_ID, None, [], [], ALL_TIERS, "leaf", "Pet flap")
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
            model.meta["locked"] = True   # like a credential-locked turnstile: QA / parity expect it to hold, not to swing
        world.sites.append(Site("approach_point", (0, -1.0, 0), QUAT_ID, 0.05, "approach"))
        world.sites.append(Site("goal_point", (0, 1.0, 0), QUAT_ID, 0.05, "goal"))
        world.sites.append(Site("door_plane_center", (0, 0, 0.05 + Ho / 2), QUAT_ID, 0.02, "pass_plane"))
        model.meta.update({"primary_joint": "flap_hinge", "operator_joint": None, "handle_height": 0.05 + Ho / 2, "both_ways": True})
        return flap
    if fam == "strip_curtain":
        from .strip_curtain import build_strip_curtain
        return build_strip_curtain(spec, phys, model)
    if fam == "garage_tiltup":
        from .garage_tiltup import build_tiltup
        return build_tiltup(spec, phys, model)
    raise ValueError(fam)

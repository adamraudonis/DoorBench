"""Shared geometry builders: walls, frames, leaves, hinges, strikes, latch
bolts, deadbolts, operators, closers, signage.

World frame: floor z=0, wall plane y=0, opening centered on x=0.  The robot
approaches from -y.  `u` is the sign of the leaf's local x direction (from the
hinge edge across the leaf), `v` is the swing direction sign (+1 = door opens
toward +y, i.e. the robot pushes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..ir import (Body, Geom, Joint, Site, Material, Equality, Tendon, Model, ALL_TIERS, FULL_ONLY, FULL_SIMPLE,
                  quat_from_axis_angle, quat_mul, quat_z_to, QUAT_ID, quat_rotate, mat_to_quat, quat_to_mat)
from .. import materials as M
from .. import hardware as H
from ..panels import glazing_layout, raised_panel_layout, louver_slats
from . import meshes as MESH

GAP = 0.003          # leaf-to-jamb clearance
BOTTOM_CLEAR = 0.012
HINGE_THROW_MAX = 0.050   # m; largest pin offset from the leaf face (a "swing clear" wide-throw butt hinge)
LEAF_FACE_INSET = 0.007   # m; the leaf's swing-side face sits this far inside the frame's swing-side face, so the
#                           hinge knuckle (one radius = 7 mm proud of the leaf) lands ON the frame face.  A door hung
#                           deeper than its knuckle cannot swing past ~90 deg without the frame's reveal arris raking
#                           across its face - the 20 mm inset used before did exactly that on 40 doors.


def pivot_heel_gap(pivot_in: float, t: float, clear: float = 0.003) -> float:
    """Gap a CENTRE-HUNG leaf's heel edge needs from the face it turns against (jamb reveal, pilaster, curb).

    The leaf turns on an axis ``pivot_in`` inside that face and in the leaf's own centre plane, so its heel CORNER -
    half a thickness off that plane - sweeps a circle of radius ``hypot(pivot_in - gap, t/2)``.  The corner therefore
    clears the face by ``pivot_in - radius``, which a flat 6 mm gap left at 0.8 mm on the toilet partitions and
    2.6 mm on the centre-pivot doors.  Solve the clearance instead:

        pivot_in - hypot(pivot_in - gap, t/2) >= clear  <=>  gap >= pivot_in - sqrt((pivot_in - clear)^2 - (t/2)^2)

    Returns 0.0 when the setback is too small for the thickness (pivot_in <= t/2 + clear): no heel gap can fix that,
    the pivot itself has to move out, and the running-clearance gate is what says so.

    (``folding.fold_jamb_gap`` is the same solve for bifold pivot panels; it lives there because the spec generator
    sizes folding openings from it and must not import the geometry package.)"""
    r2 = (pivot_in - clear) ** 2 - (t / 2.0) ** 2
    return pivot_in - math.sqrt(r2) + 0.0005 if r2 > 0.0 else 0.0


def hinge_throw(t: float, depth: float, y_wall: float, v: float, W: float, knuckle: float = 0.007, lead_gap: float = GAP) -> float:
    """Hinge pin offset from the leaf's mid-plane on the swing side (the hinge's *throw*).

    A butt hinge's knuckle stands one radius (7 mm) proud of the leaf face, and the leaf is hung so that face is
    LEAF_FACE_INSET behind the frame's swing-side face - i.e. the pin sits ON that face.  That is what lets the leaf
    swing past 90 deg: every point of the frame is then inside the pin's clearance circle (the jamb's reveal arris is
    GAP from the pin, the leaf face 7 mm) and nothing can rake across the leaf.  Where the frame face still ends up
    outside the pin (a leaf hung deeper than the knuckle in a thick frame), the throw is carried out to it, as a
    WIDE-THROW (swing-clear) butt hinge does - but never further than the leading edge can afford: a throw beyond the
    knuckle swings the LOCK edge hypot(W, throw + t/2) - W towards the strike jamb (and, on a pair, into the inactive
    leaf's meeting stile) before it comes back, and only the leading gap is there to swallow it.  Capped at
    HINGE_THROW_MAX, the widest catalogue hinge."""
    face = v * y_wall + depth / 2          # the jamb's swing-side face, as a distance on the swing side
    std = t / 2 + knuckle
    lead = math.sqrt(max((W + max(lead_gap - 0.0005, 0.0)) ** 2 - W ** 2, 0.0)) - t / 2   # what the leading gap allows
    return max(std, min(face, t / 2 + HINGE_THROW_MAX, lead))


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------
def mat_from_material(model: Model, mid: str, name: str | None = None) -> str:
    m = M.MATERIALS[mid]
    name = name or f"mat_{mid}"
    if name not in model.materials:
        model.add_material(Material(name, tuple(m.base_color), m.roughness, m.metallic, m.texture, m.transparent))
    return name


def mat_from_finish(model: Model, finish: dict, name: str = "mat_leaf") -> str:
    if name not in model.materials:
        rgba = tuple(finish.get("rgba", (0.8, 0.8, 0.8, 1)))
        model.add_material(Material(name, rgba, finish.get("roughness", 0.6), finish.get("metallic", 0.0), finish.get("texture"), rgba[3] < 0.99 if len(rgba) > 3 else False))
    return name


def mat_rgba(model: Model, name: str, rgba, roughness=0.6, metallic=0.0, transparent=False, texture=None) -> str:
    if name not in model.materials:
        model.add_material(Material(name, tuple(rgba), roughness, metallic, texture, transparent))
    return name


# ---------------------------------------------------------------------------
# geom helpers
# ---------------------------------------------------------------------------
def box(name, center, half, material, density=600.0, collision=True, visual=True, tiers=ALL_TIERS, semantic="structure", label="", friction=(0.6, 0.005, 0.0001), solref=None, quat=QUAT_ID, mass=None, margin=0.0):
    return Geom(name, "box", tuple(float(h) for h in half), tuple(float(c) for c in center), tuple(float(q) for q in quat), material, collision, visual, density, mass, friction, None, None, False, solref, None, margin, tiers, semantic, label)


def cyl(name, center, radius, half_len, material, axis=(0, 0, 1), density=7850.0, collision=True, visual=True, tiers=ALL_TIERS, semantic="structure", label="", mass=None):
    q = quat_z_to(axis)
    return Geom(name, "cylinder", (float(radius), float(half_len)), tuple(float(c) for c in center), tuple(float(x) for x in q), material, collision, visual, density, mass, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, tiers, semantic, label)


def sphere(name, center, radius, material, density=7850.0, collision=False, tiers=FULL_ONLY, semantic="decor", label=""):
    return Geom(name, "sphere", (float(radius),), tuple(float(c) for c in center), (1, 0, 0, 0), material, collision, True, density, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, tiers, semantic, label)


def mesh_geom(name, key, mesh, center, quat, material, density=7100.0, collision=False, tiers=FULL_ONLY, semantic="operator", label="", mass=None):
    return Geom(name, "mesh", (1, 1, 1), tuple(float(c) for c in center), tuple(float(q) for q in quat), material, collision, True, density, mass, (0.6, 0.005, 0.0001), mesh, key, True, None, None, 0.0, tiers, semantic, label)


def body_world_pos(model: Model, body: Body):
    """World position of a body (parent chain; body quats are identity in this IR)."""
    p = [float(body.pos[0]), float(body.pos[1]), float(body.pos[2])]
    parent, seen = body.parent, 0
    while parent and seen < 12:
        try:
            b = model.body(parent)
        except Exception:
            break
        p = [p[i] + float(b.pos[i]) for i in range(3)]
        parent, seen = b.parent, seen + 1
    return p


def q_face(v_dir_y: float, u: float = 1.0):
    """Quaternion taking mesh local +z (away from face) to world ±y (door normal) and
    mesh -x (reach direction) to the hinge-ward direction (-u x).
    Mesh frame: z away from face, x = across door (reach is -x), y = up.
    We need: z_local -> (0, v_dir_y, 0); y_local -> (0,0,1); x_local -> (u',0,0) with right-handedness.
    """
    zl = np.array([0.0, v_dir_y, 0.0])
    yl = np.array([0.0, 0.0, 1.0])
    xl = np.cross(yl, zl)  # right-handed
    R = np.column_stack([xl, yl, zl])
    from ..ir import mat_to_quat
    q = mat_to_quat(R)
    # if x maps to the wrong way for the requested reach direction, rotate 180 about z_local
    if np.sign(xl[0]) != np.sign(u) and abs(u) > 0:
        q = quat_mul(q, quat_from_axis_angle([0, 0, 1], math.pi))
    return q


def q_face_upright(v_dir_y: float):
    """Like q_face but never flips: mesh +y always maps to world +z (up) and +z to the face normal.  For meshes
    without a reach direction that are NOT symmetric top/bottom (handleset thumb press, card reader LED, keypads)."""
    zl = np.array([0.0, v_dir_y, 0.0])
    yl = np.array([0.0, 0.0, 1.0])
    xl = np.cross(yl, zl)
    return mat_to_quat(np.column_stack([xl, yl, zl]))


def q_axis_x_to(direction):
    """Quaternion mapping local +x to `direction` (keeping z up if possible)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    x = np.array([1.0, 0, 0])
    c = float(np.dot(x, d))
    if c > 1 - 1e-9:
        return QUAT_ID.copy()
    if c < -1 + 1e-9:
        return quat_from_axis_angle([0, 0, 1], math.pi)
    ax = np.cross(x, d)
    return quat_from_axis_angle(ax, math.acos(c))


# ---------------------------------------------------------------------------
# Environment: floor, wall with opening, frame
# ---------------------------------------------------------------------------
def add_floor_and_wall(model: Model, spec: dict, wall_half_width=2.5, wall_height=None, hole=None, floor_hole=None, outdoor=False, y_wall=0.0):
    """Static world geometry.  hole=(x0,x1,z0,z1) opening in the wall plane.  y_wall = wall centre plane offset
    (hinged doors hang near the swing-side face of the jamb, so the wall is offset from the leaf plane y=0)."""
    model.meta["wall_y"] = float(y_wall)
    op = spec["opening"]
    wt = op["wall_thickness"]
    Wo, Ho = op["width"], op["height"]
    wall_height = wall_height or max(Ho + 0.6, 2.7)
    world = Body("world_env", None, (0, 0, 0), QUAT_ID, None, [], [], ALL_TIERS, "wall", "Environment", static=True)
    model.add_body(world)
    floor_mat = mat_rgba(model, "mat_floor", (0.55, 0.53, 0.50, 1) if not outdoor else (0.35, 0.42, 0.25, 1), 0.85, 0.0, texture="concrete_floor_painted" if not outdoor else "aerial_wood_snips")
    wall_mat = mat_rgba(model, "mat_wall", (0.78, 0.76, 0.70, 1), 0.9, 0.0, texture="plastered_wall_04")
    if floor_hole is None:
        world.geoms.append(box("floor", (0, 0, -0.05), (6.0, 6.0, 0.05), floor_mat, 2400, True, True, ALL_TIERS, "floor", "Floor", friction=(0.8, 0.005, 0.0001)))
    else:
        x0, x1, y0, y1 = floor_hole
        world.geoms += [
            box("floor_a", ((x0 - 6) / 2, 0, -0.05), ((x0 + 6) / 2, 6.0, 0.05), floor_mat, 2400, semantic="floor"),
            box("floor_b", ((x1 + 6) / 2, 0, -0.05), ((6 - x1) / 2, 6.0, 0.05), floor_mat, 2400, semantic="floor"),
            box("floor_c", ((x0 + x1) / 2, (y0 - 6) / 2, -0.05), ((x1 - x0) / 2, (y0 + 6) / 2, 0.05), floor_mat, 2400, semantic="floor"),
            box("floor_d", ((x0 + x1) / 2, (y1 + 6) / 2, -0.05), ((x1 - x0) / 2, (6 - y1) / 2, 0.05), floor_mat, 2400, semantic="floor"),
        ]
    if outdoor or hole is None and spec["family"] in ("gate_swing", "gate_sliding", "baby_gate"):
        return world
    if hole is None:
        hole = (-Wo / 2, Wo / 2, 0.0, Ho)
    x0, x1, z0, z1 = hole
    # wall segments: left, right, header
    if x0 > -wall_half_width:
        world.geoms.append(box("wall_left", ((x0 - wall_half_width) / 2, y_wall, wall_height / 2), ((x0 + wall_half_width) / 2, wt / 2, wall_height / 2), wall_mat, 800, semantic="wall", label="Wall"))
    if x1 < wall_half_width:
        world.geoms.append(box("wall_right", ((x1 + wall_half_width) / 2, y_wall, wall_height / 2), ((wall_half_width - x1) / 2, wt / 2, wall_height / 2), wall_mat, 800, semantic="wall", label="Wall"))
    # A child/pet safety gate divides a full-height passage. Its opening height
    # describes the low gate, not a doorway lintel: keep the side walls but leave
    # the space above it open, including callers supplying an explicit frame hole.
    if z1 < wall_height and spec["family"] != "baby_gate":
        world.geoms.append(box("wall_header", ((x0 + x1) / 2, y_wall, (z1 + wall_height) / 2), ((x1 - x0) / 2, wt / 2, (wall_height - z1) / 2), wall_mat, 800, semantic="wall", label="Wall header"))
    return world


def frame_jamb_thickness(spec: dict) -> float:
    fr = spec["opening"]["frame"]
    k = fr["kind"]
    if k == "timber_frame_heavy":
        return 0.06
    if k == "vault_frame":
        return 0.12
    if k == "gate_posts":
        return float(fr.get("post_size", 0.1))
    if k == "pressure_frame":
        return 0.03
    if k.startswith(("wood", "timber", "kamoi")):
        return 0.019
    if k == "hollow_metal_frame":
        return 0.05
    return 0.045


STUD_POCKET = 0.03   # stud / shim space behind the strike jamb that bolts penetrate (rough opening)


def frame_hole(spec: dict, u: float, jamb_t: float):
    """Wall opening (x0, x1, z0, z1) around the frame: extra stud margin behind the strike jamb (+u side)."""
    Wo, Ho = spec["opening"]["width"], spec["opening"]["height"]
    x_h = -u * (Wo / 2 + jamb_t)
    x_s = u * (Wo / 2 + jamb_t + STUD_POCKET)
    return (min(x_h, x_s), max(x_h, x_s), 0.0, Ho + jamb_t + STUD_POCKET)


def _strike_column(geoms, prefix, sx, u, v, jamb_t, ya, yb, z_top, pockets, mat, dens, jamb_seg_name="jamb_strike", z_bot=0.0):
    """Vertical strike member (jamb or gate post) from z_bot..z_top with pockets cut into it.
    Each pocket: dict(z, h, w, depth, ramp, ramp_both, y).  The column spans x from sx to sx + u*jamb_t and y from ya..yb."""
    if not pockets:
        geoms.append(box(jamb_seg_name, (sx + u * jamb_t / 2, (ya + yb) / 2, (z_bot + z_top) / 2), (jamb_t / 2, (yb - ya) / 2, (z_top - z_bot) / 2), mat, dens, semantic="frame", label="Strike jamb"))
        return
    yw = (ya + yb) / 2
    depth = yb - ya
    zs = [z_bot]
    segs = []
    for pk in sorted(pockets, key=lambda d: d["z"]):
        zc, ph, pw, pd = pk["z"], pk["h"], pk["w"], pk["depth"]
        yc = float(pk.get("y", 0.0))
        z_lo, z_hi = zc - ph / 2, zc + ph / 2
        segs.append((zs[-1], z_lo))
        zs.append(z_hi)
        tag = f"{prefix}_{int(round(zc * 1000))}"
        y_edge_n = ya if v > 0 else yb          # non-swing side edge
        y_edge_p = yb if v > 0 else ya          # swing side edge
        yn0, yn1 = sorted((yc - v * pw / 2, y_edge_n))
        if yn1 - yn0 > 1e-4:
            geoms.append(box(f"{tag}_wall_n", (sx + u * jamb_t / 2, (yn0 + yn1) / 2, zc), (jamb_t / 2, (yn1 - yn0) / 2, ph / 2), mat, dens, semantic="frame", label="Strike pocket wall"))
        if pk.get("ramp") or pk.get("ramp_both"):
            # flat lip 16 mm past the pocket on the swing side, then a ramp; for a pocket centred in the member the
            # ramp runs out past the swing-side face (lip ~20 mm proud), for an off-centre pocket (surface-mounted
            # rim latch) it is short and the member is filled in beyond it
            y_m = yc + v * (pw / 2 + 0.016)
            yp0, yp1 = sorted((yc + v * pw / 2, y_m))
            if yp1 - yp0 > 1e-4:
                geoms.append(box(f"{tag}_wall_p", (sx + u * jamb_t / 2, (yp0 + yp1) / 2, zc), (jamb_t / 2, (yp1 - yp0) / 2, ph / 2), mat, dens, semantic="frame", label="Strike lip (flat)"))
            # the ramp always runs out ~20 mm past the member's swing-side face: a bolt tip descending onto the member
            # as the door closes must meet the ramp, never a flat face beyond it
            y_top = (y_edge_p + v * 0.02) if v * (y_edge_p + v * 0.02 - y_m) > 0.03 else (y_m + v * 0.03)
            rise = pk.get("ramp_rise", 0.014)
            run = abs(y_top - y_m)
            ang = math.atan2(rise, run)
            L = math.hypot(rise, run)
            nx, ny = math.cos(ang), -math.sin(ang)
            cx = sx + u * (rise / 2 + (jamb_t / 2) * nx)
            cy = (y_m + y_top) / 2 + v * (jamb_t / 2) * ny
            q = quat_from_axis_angle([0, 0, 1], -u * v * ang)
            geoms.append(box(f"{tag}_ramp", (cx, cy, zc), (jamb_t / 2, L / 2, ph / 2), mat, dens, semantic="frame", label="Strike lip (ramp)", quat=q, friction=(0.12, 0.005, 0.0001)))
            y_fill0 = y_top + v * ((jamb_t / 2) * math.sin(ang) + 0.001)
            if v * (y_edge_p - y_fill0) > 0.002:
                yf0, yf1 = sorted((y_fill0, y_edge_p))
                geoms.append(box(f"{tag}_wall_pp", (sx + u * jamb_t / 2, (yf0 + yf1) / 2, zc), (jamb_t / 2, (yf1 - yf0) / 2, ph / 2), mat, dens, semantic="frame", label="Strike jamb"))
            if pk.get("ramp_both"):
                yn_m = yc - v * (pw / 2 + 0.004)
                geoms.append(box(f"{tag}_ramp_n", (sx + u * (rise / 2 + (jamb_t / 2) * nx), yn_m - v * (run / 2 + (jamb_t / 2) * ny), zc), (jamb_t / 2, L / 2, ph / 2), mat, dens, semantic="frame", label="Catch ramp", quat=quat_from_axis_angle([0, 0, 1], u * v * ang), friction=(0.12, 0.005, 0.0001)))
        else:
            yp0, yp1 = sorted((yc + v * pw / 2, y_edge_p))
            wt_ = jamb_t
            if v * (y_edge_p - yc) < pw / 2 + 0.004:
                # pocket reaches the member's edge: add a keeper lip beyond the edge so the bolt is still captured
                yp0, yp1 = sorted((yc + v * pw / 2, yc + v * (pw / 2 + 0.006)))
                wt_ = min(jamb_t, max(0.03, pd + 0.002))     # deep enough to meet the pocket back
            geoms.append(box(f"{tag}_wall_p", (sx + u * wt_ / 2, (yp0 + yp1) / 2, zc), (wt_ / 2, (yp1 - yp0) / 2, ph / 2), mat, dens, semantic="frame", label="Strike pocket wall"))
        back_t = max(jamb_t - pd, 0.004)
        b_pos, b_half = (sx + u * (pd + back_t / 2), yc, zc), (back_t / 2, pw / 2, ph / 2)
        if not any(g.name.endswith("_back") and max(abs(a - b) for a, b in zip(g.pos, b_pos)) < 1e-6
                   and max(abs(a - b) for a, b in zip(g.size, b_half)) < 1e-6 for g in geoms):
            geoms.append(box(f"{tag}_back", b_pos, b_half, mat, dens, semantic="frame", label="Strike pocket back"))
    segs.append((zs[-1], z_top))
    for k, (a, b) in enumerate(segs):
        if b - a > 1e-4:
            geoms.append(box(f"{jamb_seg_name}_{k}", (sx + u * jamb_t / 2, yw, (a + b) / 2), (jamb_t / 2, depth / 2, (b - a) / 2), mat, dens, semantic="frame", label="Strike jamb"))


def add_head(geoms, name, x0, x1, yw, depth, z0, thick, mat, dens, head_pockets=None, label="Head jamb"):
    """Head member from x0..x1 at z0..z0+thick with optional pockets (dict x, hx, w, depth) for vertical rod latches."""
    hp = sorted(head_pockets or [], key=lambda p: p["x"])
    segs, prev = [], min(x0, x1)
    xe = max(x0, x1)
    for p in hp:
        segs.append((prev, p["x"] - p["hx"]))
        # pocket walls (+-y) and back (above the pocket depth); optional y offset (surface rods)
        yc = float(p.get("y", 0.0))
        for sy in (-1, 1):
            yw0, yw1 = sorted((yc + sy * p["w"] / 2, yw + sy * depth / 2))
            if yw1 - yw0 > 1e-4:
                geoms.append(box(f"{name}_pwall_{int(p['x'] * 1000)}_{'p' if sy > 0 else 'n'}", (p["x"], (yw0 + yw1) / 2, z0 + thick / 2), (p["hx"], (yw1 - yw0) / 2, thick / 2), mat, dens, semantic="frame", label="Head strike wall"))
        back = thick - p["depth"]
        if back > 0.003:
            geoms.append(box(f"{name}_pback_{int(p['x'] * 1000)}", (p["x"], yc, z0 + p["depth"] + back / 2), (p["hx"], p["w"] / 2, back / 2), mat, dens, semantic="frame", label="Head strike back"))
        prev = p["x"] + p["hx"]
    segs.append((prev, xe))
    for k, (a, b) in enumerate(segs):
        if b - a > 1e-4:
            geoms.append(box(f"{name}_{k}" if hp else name, ((a + b) / 2, yw, z0 + thick / 2), ((b - a) / 2, depth / 2, thick / 2), mat, dens, semantic="frame", label=label))


def add_threshold(model: Model, spec: dict, geoms: list, yw: float, Wo: float, jamb_t: float, depth: float, mat) -> None:
    """The sill the door closes over.  Split out of add_frame because build_swing_pair builds its own frame inline:
    a PAIR with a saddle threshold used to get no sill at all, which left the bottom shoot bolt's floor strike -
    placed on the saddle top by FLOOR_STRIKE_TOP - floating 13 mm above the floor on all 26 of them."""
    op = spec["opening"]
    thr = op.get("threshold", "none")
    if thr in ("saddle", "sill"):
        tm = mat_from_material(model, "aluminum", "mat_threshold")
        geoms.append(box("threshold", (0, yw, 0.0065), (Wo / 2 + jamb_t, depth / 2, 0.0065), tm, 2700, semantic="frame", label="Threshold saddle (13 mm)", friction=(0.5, 0.005, 0.0001)))
    elif thr == "ada_ramp":
        tm = mat_from_material(model, "aluminum", "mat_threshold")
        geoms.append(box("threshold", (0, yw, 0.004), (Wo / 2 + jamb_t, depth / 2 + 0.05, 0.004), tm, 2700, semantic="frame", label="Low-profile ADA threshold"))
    elif thr == "coaming":
        sh = op.get("sill_height", 0.3)
        geoms.append(box("coaming", (0, 0, sh / 2), (Wo / 2 + jamb_t, 0.01, sh / 2), mat, 7850, semantic="frame", label="Raised sill / coaming"))
    elif thr == "sill_step":
        geoms.append(box("sill_step", (0, 0, 0.02), (Wo / 2 + jamb_t, depth / 2, 0.02), mat, 7850, semantic="frame", label="Vault sill"))


def add_frame(model: Model, spec: dict, v: float, world: Body, with_stop=True, strike_pockets=None, u=1.0, head_pockets=None):
    """Door frame (jambs + head + stop + optional casing + threshold) as static geoms on `world`.
    strike_pockets: list of (z_center, pocket_h, pocket_w_y, pocket_depth_x) on the strike jamb (x = -u side... the
    strike jamb is at the *latch* side which is opposite the hinge: hinge at x = u*(-Wo/2) ... see build_swing).
    Returns dict with frame dims."""
    op = spec["opening"]
    fr = op["frame"]
    Wo, Ho, wt = op["width"], op["height"], op["wall_thickness"]
    t_leaf = spec["leaf"]["thickness"]
    jamb_t = frame_jamb_thickness(spec)
    mat = mat_from_material(model, fr["material"], "mat_frame")
    if fr["kind"] in ("gate_posts", "pressure_frame"):
        # posts only (no head, no stop except a small latch-post stop block); pockets become keeper blocks on the latch post
        ps = jamb_t
        gc = op.get("ground_clearance", 0.05)
        Hp = gc + spec["leaf"]["height"] + 0.05
        dens_p = M.MATERIALS[fr["material"]].density if M.MATERIALS[fr["material"]].family != "metal" else 800.0
        hx = u * (-Wo / 2)
        sx = u * (Wo / 2)
        xx = hx - u * ps / 2
        if M.MATERIALS[fr["material"]].family == "metal":
            world.geoms.append(cyl("post_hinge", (xx, 0, Hp / 2), ps / 2, Hp / 2, mat, (0, 0, 1), dens_p, True, True, ALL_TIERS, "frame", "Gate post"))
            world.geoms.append(sphere("post_hinge_cap", (xx, 0, Hp), ps / 2, mat, dens_p, False, FULL_ONLY, "frame", "Post cap"))
        else:
            world.geoms.append(box("post_hinge", (xx, 0, Hp / 2), (ps / 2, ps / 2, Hp / 2), mat, dens_p, semantic="frame", label="Gate post"))
            world.geoms.append(box("post_hinge_cap", (xx, 0, Hp + 0.02), (ps / 2 + 0.015, ps / 2 + 0.015, 0.02), mat, dens_p, False, True, FULL_ONLY, "frame", "Post cap"))
        # latch post: box column with pockets cut in (bolts, spring latches, latch bars)
        _strike_column(world.geoms, "post", sx, u, v, ps, -ps / 2, ps / 2, Hp, strike_pockets, mat, dens_p, jamb_seg_name="post_latch")
        world.geoms.append(box("post_latch_cap", (sx + u * ps / 2, 0, Hp + 0.02), (ps / 2 + 0.015, ps / 2 + 0.015, 0.02), mat, dens_p, False, True, FULL_ONLY, "frame", "Post cap"))
        # latch-side stop block (gate closes against it)
        if with_stop:
            t_leaf = spec["leaf"]["thickness"]
            brace_proud = 0.022 if spec["leaf"]["panel_style"] in ("plank_z_brace", "plank_x_brace", "board_batten") else 0.0
            world.geoms.append(box("gate_stop", (sx - u * 0.02, -v * (t_leaf / 2 + 0.012 + brace_proud), gc + min(0.25, spec["leaf"]["height"] * 0.3)), (0.02, 0.012, 0.06), mat, dens_p, True, True, ALL_TIERS, "frame", "Gate stop block"))
            model.meta.setdefault("_brace_pending", []).append({"geom": "gate_stop", "axes": ["x", "y"], "label": "Gate stop post bracket"})
        return {"jamb_t": ps, "depth": ps, "hx": hx, "sx": sx, "mat": mat}
    dens = M.MATERIALS[fr["material"]].density if M.MATERIALS[fr["material"]].family != "metal" else 300.0
    depth = wt if fr["kind"] != "aluminum_storefront" else max(0.114, wt)
    yw = float(model.meta.get("wall_y", 0.0))
    ya, yb = yw - depth / 2, yw + depth / 2          # jamb y-range (world)
    hx = u * (-Wo / 2)         # hinge jamb inner face x
    sx = u * (Wo / 2)          # strike jamb inner face x
    geoms = []
    geoms.append(box("jamb_hinge", (hx - u * jamb_t / 2, yw, Ho / 2), (jamb_t / 2, depth / 2, Ho / 2), mat, dens, semantic="frame", label="Hinge jamb"))
    jamb_vis_t = jamb_t
    if strike_pockets:
        jamb_t = jamb_t + STUD_POCKET     # jamb + stud column (bolts penetrate the stud through the strike box)
    _strike_column(geoms, "strike", sx, u, v, jamb_t, ya, yb, Ho, strike_pockets, mat, dens, jamb_seg_name="jamb_strike")
    if not strike_pockets:
        geoms.append(box("stud_strike", (sx + u * (jamb_t + STUD_POCKET / 2), yw, Ho / 2), (STUD_POCKET / 2, depth / 2, Ho / 2), "mat_wall" if "mat_wall" in model.materials else mat, 500, semantic="wall", label="Stud"))
    # head
    add_head(geoms, "jamb_head", -u * (Wo / 2 + jamb_vis_t), u * (Wo / 2 + jamb_vis_t + STUD_POCKET), yw, depth, Ho, jamb_vis_t, mat, dens, head_pockets)
    add_head(geoms, "head_stud", -u * (Wo / 2 + jamb_vis_t), u * (Wo / 2 + jamb_vis_t + STUD_POCKET), yw, depth, Ho + jamb_vis_t, STUD_POCKET, "mat_wall" if "mat_wall" in model.materials else mat, 500, [dict(p, depth=max(p["depth"] - jamb_vis_t, 0.0)) for p in (head_pockets or []) if p["depth"] > jamb_vis_t], label="Head stud")
    # stop: door closes against the stop on the non-swing side (y = -v side of the leaf)
    if with_stop and fr.get("stop_depth", 0) > 0 and spec["leaf"].get("panel_style") != "glass_frameless":
        # stop moulding: ~11 mm proud of the jamb face (laps the leaf edge by ~7 mm), 32 mm wide along the jamb depth
        sw = 0.011
        sd = 0.032
        ys = -v * (t_leaf / 2 + sd / 2 + 0.0005)
        seal = H.SEALS[spec["seal"]]
        soft = None   # hard stop; gasket compliance is modelled by the leaf joint's soft limit (a soft stop lets slammed leaves tunnel through)
        geoms.append(box("stop_hinge", (hx + u * sw / 2, ys, Ho / 2), (sw / 2, sd / 2, Ho / 2), mat, dens, semantic="frame", label="Stop (hinge side)", solref=soft))
        cuts = sorted([(p["z"] - p.get("stop_cut_half", p["h"] / 2 + 0.012), p["z"] + p.get("stop_cut_half", p["h"] / 2 + 0.012)) for p in (strike_pockets or []) if -v * p.get("y", 0.0) + p["w"] / 2 > t_leaf / 2 - 0.002])
        zs_, prev_ = [], 0.0
        for a_, b_ in cuts:
            zs_.append((prev_, a_))
            prev_ = b_
        zs_.append((prev_, Ho))
        for k_, (a_, b_) in enumerate(zs_):
            if b_ - a_ > 1e-4:
                geoms.append(box("stop_strike" if k_ == 0 else f"stop_strike_{k_}", (sx - u * sw / 2, ys, (a_ + b_) / 2), (sw / 2, sd / 2, (b_ - a_) / 2), mat, dens, semantic="frame", label="Stop (strike side)", solref=soft))
        geoms.append(box("stop_head", (0, ys, Ho - sw / 2), (Wo / 2, sd / 2, sw / 2), mat, dens, semantic="frame", label="Stop (head)", solref=soft))
        if seal["compression_m"] > 0:
            smat = mat_rgba(model, "mat_seal", (0.1, 0.1, 0.1, 1), 0.9)
            cm = seal["compression_m"]
            pen = 0.002   # initial interpenetration with the closed leaf (soft contact models the gasket)
            y_seal = -v * (t_leaf / 2 + cm / 2 - pen)
            # Seals are VISUAL ONLY: a compliant seal contact at the hinge line cannot be resolved by the solver
            # (the leaf cannot translate there) and produces spurious kN-scale forces.  Seal compliance is modelled by
            # the door joint's soft limit (limit_solref) and its closing resistance by the hinge friction term.
            y_seal = -v * (t_leaf / 2 + cm / 2)
            geoms.append(box("seal_hinge", (hx + u * 0.003, y_seal, Ho / 2), (0.004, cm / 2, Ho / 2), smat, 1100, False, True, FULL_SIMPLE, "seal", "Weatherstrip (visual)"))
            for k_, (a_, b_) in enumerate(zs_ if cuts else [(0.0, Ho)]):
                if b_ - a_ > 1e-4:
                    geoms.append(box("seal_strike" if k_ == 0 else f"seal_strike_{k_}", (sx - u * 0.003, y_seal, (a_ + b_) / 2), (0.004, cm / 2, (b_ - a_) / 2), smat, 1100, False, True, FULL_SIMPLE, "seal", "Weatherstrip (visual)"))
            geoms.append(box("seal_head", (0, y_seal, Ho - 0.012), (Wo / 2 - 0.02, cm / 2, 0.008), smat, 1100, False, True, FULL_SIMPLE, "seal", "Weatherstrip (visual)"))
    jamb_t = jamb_vis_t
    # casing (visual trim) both sides
    if fr.get("casing"):
        cw, ct = 0.07, 0.016
        for sgn in (-1, 1):
            yc = yw + sgn * (depth / 2 + ct / 2)
            geoms.append(box(f"casing_l_{'p' if sgn > 0 else 'n'}", (-Wo / 2 - jamb_t - cw / 2 + 0.005, yc, Ho / 2), (cw / 2, ct / 2, Ho / 2 + jamb_t / 2), mat, dens, False, True, FULL_ONLY, "decor", "Casing"))
            geoms.append(box(f"casing_r_{'p' if sgn > 0 else 'n'}", (Wo / 2 + jamb_t + cw / 2 - 0.005, yc, Ho / 2), (cw / 2, ct / 2, Ho / 2 + jamb_t / 2), mat, dens, False, True, FULL_ONLY, "decor", "Casing"))
            geoms.append(box(f"casing_h_{'p' if sgn > 0 else 'n'}", (0, yc, Ho + jamb_t + cw / 2 - 0.005), (Wo / 2 + jamb_t + cw - 0.005, ct / 2, cw / 2), mat, dens, False, True, FULL_ONLY, "decor", "Casing"))
    # threshold
    add_threshold(model, spec, geoms, yw, Wo, jamb_t, depth, mat)
    world.geoms += geoms
    return {"jamb_t": jamb_t, "depth": depth, "hx": hx, "sx": sx, "mat": mat}


# ---------------------------------------------------------------------------
# Leaf slab with panels, glazing, louvers, plates
# ---------------------------------------------------------------------------
def _hits_hole(hole, cx, cz, hw, hh):
    hx, hz, w, h = hole
    return abs(cx - hx) < hw + w / 2 and abs(cz - hz) < hh + h / 2


def _slab_boxes(leaf_body, name_prefix, xc, y_center, zc, W, t, Hh, lm, collision_tiers, slab_fric, mass, hole):
    """Leaf slab as one box, or as four boxes around a rectangular hole (pet flap) so nothing passes through solid geometry."""
    if not hole:
        leaf_body.geoms.append(box(f"{name_prefix}_slab", (xc, y_center, zc), (W / 2, t / 2, Hh / 2), lm, 1.0, True, True, collision_tiers, "leaf", "Leaf slab", slab_fric, mass=mass))
        return
    hx, hz, w, h = hole
    x_lo, x_hi, z_lo, z_hi = xc - W / 2, xc + W / 2, zc - Hh / 2, zc + Hh / 2
    parts = [
        ("slab_below", (xc, (z_lo + hz - h / 2) / 2), (W / 2, (hz - h / 2 - z_lo) / 2)),
        ("slab_above", (xc, (hz + h / 2 + z_hi) / 2), (W / 2, (z_hi - hz - h / 2) / 2)),
        ("slab_left", ((x_lo + hx - w / 2) / 2, hz), ((hx - w / 2 - x_lo) / 2, h / 2)),
        ("slab_right", ((hx + w / 2 + x_hi) / 2, hz), ((x_hi - hx - w / 2) / 2, h / 2)),
    ]
    area = W * Hh - w * h
    for nm, (px, pz), (hw, hh) in parts:
        if hw > 0.003 and hh > 0.003:
            leaf_body.geoms.append(box(f"{name_prefix}_{nm}", (px, y_center, pz), (hw, t / 2, hh), lm, 1.0, True, True, collision_tiers, "leaf", "Leaf slab", slab_fric, mass=(mass * (4 * hw * hh) / area) if mass else None))


def add_leaf_geoms(model: Model, leaf_body: Body, spec: dict, leaf: dict, u: float, x0: float, z0: float, phys: dict, name_prefix="leaf", collision_tiers=ALL_TIERS, y_center=0.0, W=None, Hh=None, edge_pockets=None, v_edge=1.0, hole=None):
    """Leaf slab geometry in the leaf body frame.  Leaf spans x from x0 to x0+u*W, z from z0 to z0+H, centered at y_center.
    Slab collision = one box (all tiers); panels/glazing/louvers are visual (+glass collision in full)."""
    W = W or leaf["width"]
    Hh = Hh or leaf["height"]
    t = leaf["thickness"]
    slab = M.SLABS[leaf["slab"]]
    fin = leaf["finish"]
    lm = mat_from_finish(model, fin, f"mat_{name_prefix}")
    face = M.slab_face_material(slab)
    mass = phys["mass"]["slab_kg"] if phys else None
    xc = x0 + u * W / 2
    zc = z0 + Hh / 2
    style = leaf.get("panel_style", "flush")
    glazing = leaf.get("glazing")
    is_frameless_glass = style == "glass_frameless" or slab.core_material in ("glass_clear", "mirror") and slab.monolithic
    is_mesh = style in ("mesh_panel", "pickets", "bar_grille", "ornamental_scroll", "grille_rollup", "strips")
    slab_fric = (face.friction_kinetic, 0.005, 0.0001)
    # main slab: for glazed styles we still use one collision box for the slab (glass included) for performance,
    # visuals get frame + glass panes.
    if is_mesh:
        # frame + infill: collision as a thin box (mesh treated as solid barrier), visual as bars/pickets
        gm = leaf_body.geoms
        fm = lm
        rail = 0.045 if style != "bar_grille" else 0.06
        gm.append(box(f"{name_prefix}_slab_col", (xc, y_center, zc), (W / 2, t / 2, Hh / 2), fm, 1.0, True, False, collision_tiers, "leaf", "Leaf (collision proxy)", slab_fric, mass=mass))
        # frame rails
        gm.append(box(f"{name_prefix}_rail_b", (xc, y_center, z0 + rail / 2), (W / 2, t / 2, rail / 2), fm, 1.0, False, True, ALL_TIERS, "leaf", "Bottom rail"))
        gm.append(box(f"{name_prefix}_rail_t", (xc, y_center, z0 + Hh - rail / 2), (W / 2, t / 2, rail / 2), fm, 1.0, False, True, ALL_TIERS, "leaf", "Top rail"))
        gm.append(box(f"{name_prefix}_stile_h", (x0 + u * rail / 2, y_center, zc), (rail / 2, t / 2, Hh / 2), fm, 1.0, False, True, ALL_TIERS, "leaf", "Hinge stile"))
        gm.append(box(f"{name_prefix}_stile_l", (x0 + u * (W - rail / 2), y_center, zc), (rail / 2, t / 2, Hh / 2), fm, 1.0, False, True, ALL_TIERS, "leaf", "Latch stile"))
        if style in ("pickets", "bar_grille", "ornamental_scroll"):
            pitch = 0.11 if style == "pickets" else 0.127
            n = max(2, int((W - 2 * rail) / pitch))
            pw = 0.04 if style == "pickets" else 0.019
            for i in range(n):
                px = x0 + u * (rail + (i + 0.5) * (W - 2 * rail) / n)
                if style == "pickets":
                    gm.append(box(f"{name_prefix}_picket_{i}", (px, y_center, zc + 0.02), (pw / 2, t * 0.35, Hh / 2 - rail * 0.3), fm, 1.0, False, True, FULL_SIMPLE, "leaf", "Picket"))
                else:
                    gm.append(cyl(f"{name_prefix}_bar_{i}", (px, y_center, zc), pw / 2, Hh / 2 - rail, fm, (0, 0, 1), 1.0, False, True, FULL_SIMPLE, "leaf", "Bar"))
            if style == "bar_grille":
                for k in (0.33, 0.66):
                    gm.append(box(f"{name_prefix}_crossbar_{int(k * 100)}", (xc, y_center, z0 + Hh * k), (W / 2 - rail, t * 0.4, 0.02), fm, 1.0, False, True, FULL_ONLY, "leaf", "Cross bar"))
        elif style == "mesh_panel":
            mm = mat_rgba(model, "mat_mesh_infill", (0.6, 0.62, 0.64, 0.55), 0.7, 0.8, True)
            gm.append(box(f"{name_prefix}_mesh", (xc, y_center, zc), (W / 2 - rail, 0.0015, Hh / 2 - rail), mm, 1.0, False, True, ALL_TIERS, "leaf", "Mesh infill"))
        elif style == "grille_rollup":
            n = max(3, int(Hh / 0.15))
            for i in range(n):
                gm.append(cyl(f"{name_prefix}_grille_{i}", (xc, y_center, z0 + (i + 0.5) * Hh / n), 0.006, W / 2 - rail, fm, (1, 0, 0), 1.0, False, True, FULL_SIMPLE, "leaf", "Grille rod"))
        return
    if is_frameless_glass:
        gmat = mat_from_material(model, slab.core_material if slab.monolithic else "glass_clear", "mat_glass_leaf")
        leaf_body.geoms.append(box(f"{name_prefix}_glass", (xc, y_center, zc), (W / 2, t / 2, Hh / 2), gmat, 1.0, True, True, collision_tiers, "glass", "Frameless glass leaf", (0.4, 0.005, 0.0001), mass=mass))
        # patch fittings top & bottom
        pm = mat_from_material(model, "stainless", "mat_patch")
        for zz, nm in (((z0 + 0.06, "bot"), (z0 + Hh - 0.06, "top")) if style == "glass_frameless" else ()):
            leaf_body.geoms.append(box(f"{name_prefix}_patch_{nm}", (x0 + u * 0.08, y_center, zz), (0.08, t / 2 + 0.008, 0.05), pm, 1.0, False, True, FULL_SIMPLE, "leaf", "Patch fitting"))
        return
    # latch-edge column with pockets (the leaf is a strike for another leaf's bolts)
    col_w = 0.0
    if edge_pockets:
        col_w = 0.06
        col_x = x0 + u * W    # latch edge; the column extends back into the leaf (-u)
        _strike_column(leaf_body.geoms, f"{name_prefix}_edge", col_x, -u, v_edge, col_w, y_center - t / 2, y_center + t / 2, z0 + Hh, [dict(p, y=p.get("y", 0.0), ramp_both=False) for p in edge_pockets], lm, 1.0, jamb_seg_name=f"{name_prefix}_edge_seg", z_bot=z0)
        for g in leaf_body.geoms:
            if g.name.startswith(f"{name_prefix}_edge"):
                g.semantic, g.collision, g.visual = "leaf", True, True
        W = W - col_w
        xc = x0 + u * W / 2
    # Regular slab
    if glazing and glazing.get("panel_style") in ("glass_full", "glass_half", "glass_15_lite", "glass_10_lite", "glass_6_lite", "glass_9_lite", "glass_1_lite_top", "glass_oval", "glass_fan", "steel_half_glass", "glass_vision", "steel_vision", "porthole", "sectional_long_windows", "glass_sidelite_style"):
        rects = glazing_layout(glazing["panel_style"], W, Hh)
        gmat = mat_from_material(model, glazing["material"], "mat_glass")
        gt = glazing.get("thickness", 0.006)
        # slab as one collision box; visual: frame pieces around the glass are approximated by drawing the slab
        # slightly thinner than glass so glass shows through? Simpler: draw slab box + glass panes protruding 0.5mm.
        _slab_boxes(leaf_body, name_prefix, xc, y_center, zc, W, t, Hh, lm, collision_tiers, slab_fric, mass, hole)
        for k, (rx, rz, rw, rh) in enumerate(rects):
            if rw < 0.02 or rh < 0.02:
                continue
            cx = x0 + u * (rx + rw / 2)
            cz = z0 + rz + rh / 2
            if hole and _hits_hole(hole, cx, cz, rw / 2, rh / 2):
                continue
            leaf_body.geoms.append(box(f"{name_prefix}_glass_{k}", (cx, y_center, cz), (rw / 2, t / 2 + 0.0008, rh / 2), gmat, 1.0, False, True, ALL_TIERS, "glass", "Glazing"))
            if glazing["panel_style"] in ("glass_15_lite", "glass_10_lite", "glass_6_lite", "glass_9_lite"):
                pass
        return
    # solid slab (+ raised panels, louvers)
    _slab_boxes(leaf_body, name_prefix, xc, y_center, zc, W, t, Hh, lm, collision_tiers, slab_fric, mass, hole)
    panels = raised_panel_layout(style, W, Hh)
    if panels:
        recess = 0.006
        dm = mat_rgba(model, f"mat_{name_prefix}_panel", tuple(min(1, c * 0.92) for c in fin["rgba"][:3]) + (1.0,), fin.get("roughness", 0.6), fin.get("metallic", 0.0), texture=fin.get("texture"))
        for k, (rx, rz, rw, rh) in enumerate(panels):
            if hole and _hits_hole(hole, x0 + u * (rx + rw / 2), z0 + rz + rh / 2, rw / 2, rh / 2):
                continue
            if rw / 2 - 0.02 < 0.006 or rh / 2 - 0.02 < 0.006:
                continue
            cx = x0 + u * (rx + rw / 2)
            cz = z0 + rz + rh / 2
            for sgn in (-1, 1):
                leaf_body.geoms.append(box(f"{name_prefix}_panel_{k}_{'p' if sgn > 0 else 'n'}", (cx, y_center + sgn * (t / 2 + recess / 2), cz), (rw / 2 - 0.02, recess / 2, rh / 2 - 0.02), dm, 1.0, False, True, FULL_SIMPLE, "leaf", "Raised panel"))
    region, n = louver_slats(style, W, Hh)
    if region and region[2] > 0.03 and region[3] > 0.05:
        rx, rz, rw, rh = region
        sm = mat_rgba(model, f"mat_{name_prefix}_louver", tuple(min(1, c * 0.85) for c in fin["rgba"][:3]) + (1.0,), 0.6)
        for i in range(n):
            cz = z0 + rz + (i + 0.5) * rh / n
            if hole and _hits_hole(hole, x0 + u * (rx + rw / 2), cz, rw / 2, rh / n / 2):
                continue
            g = box(f"{name_prefix}_louver_{i}", (x0 + u * (rx + rw / 2), y_center, cz), (rw / 2, 0.002, rh / n / 2 * 1.05), sm, 1.0, False, True, FULL_ONLY, "leaf", "Louver slat", quat=quat_from_axis_angle([1, 0, 0], math.radians(35)))
            leaf_body.geoms.append(g)
    if style in ("plank_z_brace", "plank_x_brace", "board_batten", "plank_vertical", "planks_diagonal", "arched_top"):
        bm = mat_rgba(model, f"mat_{name_prefix}_brace", tuple(min(1, c * 0.9) for c in fin["rgba"][:3]) + (1.0,), 0.8, texture=fin.get("texture"))
        for k in range(1, max(2, int(W / 0.14))):
            xk = x0 + u * (k * W / max(2, int(W / 0.14)))
            leaf_body.geoms.append(box(f"{name_prefix}_plank_gap_{k}", (xk, y_center, zc), (0.002, t / 2 + 0.001, Hh / 2 - 0.01), bm, 1.0, False, True, FULL_ONLY, "leaf", "Plank joint"))
        if style in ("plank_z_brace", "plank_x_brace", "board_batten"):
            bt = 0.02
            for zz in (z0 + 0.12, z0 + Hh - 0.12) + ((z0 + Hh / 2,) if style == "board_batten" else ()):
                leaf_body.geoms.append(box(f"{name_prefix}_batten_{int(zz * 100)}", (xc, y_center + (-1 if u > 0 else -1) * (t / 2 + bt / 2), zz), (W / 2 - 0.03, bt / 2, 0.045), bm, 1.0, False, True, FULL_SIMPLE, "leaf", "Batten"))
            if style in ("plank_z_brace", "plank_x_brace"):
                L = math.hypot(W - 0.1, Hh - 0.35)
                ang = math.atan2(Hh - 0.35, (W - 0.1))
                leaf_body.geoms.append(box(f"{name_prefix}_brace_z", (xc, y_center - (t / 2 + bt / 2), zc), ((L - 0.12) / 2, bt / 2, 0.045), bm, 1.0, False, True, FULL_SIMPLE, "leaf", "Diagonal brace", quat=quat_from_axis_angle([0, 1, 0], -u * ang)))
                if style == "plank_x_brace":
                    leaf_body.geoms.append(box(f"{name_prefix}_brace_x", (xc, y_center - (t / 2 + bt / 2), zc), ((L - 0.12) / 2, bt / 2, 0.045), bm, 1.0, False, True, FULL_SIMPLE, "leaf", "Diagonal brace", quat=quat_from_axis_angle([0, 1, 0], u * ang)))
    if style in ("riveted_steel",):
        rm = mat_from_material(model, "steel", "mat_rivet")
        n = 6
        for i in range(n):
            for j in range(2):
                leaf_body.geoms.append(sphere(f"{name_prefix}_rivet_{i}_{j}", (x0 + u * (0.05 + j * (W - 0.1)), y_center - (t / 2), z0 + 0.1 + i * (Hh - 0.2) / (n - 1)), 0.012, rm))
    if style == "lattice_shoji":
        km = mat_from_material(model, "hinoki", "mat_kumiko")
        nx, nz = max(2, int(W / 0.12)), max(4, int(Hh / 0.15))
        for i in range(1, nx):
            leaf_body.geoms.append(box(f"{name_prefix}_kumiko_v{i}", (x0 + u * (i * W / nx), y_center, zc), (0.006, t / 2 + 0.002, Hh / 2 - 0.05), km, 1.0, False, True, FULL_SIMPLE, "leaf", "Kumiko"))
        for j in range(1, nz):
            leaf_body.geoms.append(box(f"{name_prefix}_kumiko_h{j}", (xc, y_center, z0 + j * Hh / nz), (W / 2 - 0.04, t / 2 + 0.002, 0.006), km, 1.0, False, True, FULL_SIMPLE, "leaf", "Kumiko"))
    if style == "padded_diamond":
        bm = mat_rgba(model, "mat_button", (0.7, 0.6, 0.3, 1), 0.3, 1.0)
        for k, (rx, rz, rw, rh) in enumerate(raised_panel_layout(style, W, Hh)):
            leaf_body.geoms.append(sphere(f"{name_prefix}_tuft_{k}", (x0 + u * (rx + rw / 2), y_center - (t / 2 + 0.008), z0 + rz + rh / 2), 0.01, bm))


def add_kick_plate(model, body, u, x0, z0, W, t, y_side, name="kick_plate", height=0.25):
    km = mat_from_material(model, "stainless", "mat_kick")
    body.geoms.append(box(name, (x0 + u * W / 2, y_side * (t / 2 + 0.0008), z0 + height / 2 + 0.01), (W / 2 - 0.03, 0.0008, height / 2), km, 7900, False, True, FULL_SIMPLE, "decor", "Kick plate"))


# ---------------------------------------------------------------------------
# Hinges (visual knuckles)
# ---------------------------------------------------------------------------
def add_hinge_visuals(model: Model, world: Body, leaf_body: Body, spec: dict, hinge_pos_xy, Hh: float, z0: float, v: float, u: float):
    hg = H.HINGES[spec["hinge"]["model"]]
    n = spec["hinge"]["count"]
    if n <= 0 or hg.kind in ("rotor", "none", "flap_pin"):
        return
    hm = mat_from_material(model, "steel_galvanized" if hg.bearing != "rusty" else "steel_rusty", "mat_hinge")
    if hg.kind in ("pivot_offset", "pivot_center", "pivot_center_heavy", "gravity_pivot") or spec["kinematics"].get("both_ways"):
        # floor & top pivots, and the fittings they turn in: without the static half the leaf hangs on nothing and
        # walks away from every static part as it swings
        wp = body_world_pos(model, leaf_body)
        for zz, nm in ((z0 - 0.005, "bottom"), (z0 + Hh + 0.01, "top")):
            z_leaf = zz if nm == "bottom" else z0 + Hh + 0.004
            leaf_body.geoms.append(cyl(f"pivot_{nm}", (hinge_pos_xy[0], hinge_pos_xy[1], z_leaf), 0.014, 0.004, hm, (0, 0, 1), 7850, False, True, FULL_ONLY, "hinge", "Pivot"))
            xw_, yw_ = wp[0] + hinge_pos_xy[0], wp[1] + hinge_pos_xy[1]
            if nm == "bottom":
                # floor pivot box, sunk into the floor and reaching up to the leaf's pivot pin
                z_lo, z_hi = -0.01, z_leaf + 0.003
                world.geoms.append(box("pivot_bottom_socket", (xw_, yw_, (z_lo + z_hi) / 2), (0.035, 0.035, (z_hi - z_lo) / 2), hm, 7850, False, True, FULL_SIMPLE, "hinge", "Floor pivot socket"))
            else:
                # head pivot bracket, down from the structure above to the leaf's top pivot
                face = mount_face_z(world, xw_, yw_, 0.035, 0.035, z_leaf)
                z_lo_ = z_leaf + 0.004      # sits ON the pin's top face, not around it
                if face is not None and face - z_lo_ < 0.30 and face > z_lo_ + 0.002:
                    world.geoms.append(box("pivot_top_bracket", (xw_, yw_, (z_lo_ + face) / 2), (0.035, 0.035, (face - z_lo_) / 2), hm, 7850, False, True, FULL_SIMPLE, "hinge", "Head pivot bracket"))
        return
    if hg.kind == "continuous":
        leaf_body.geoms.append(cyl("hinge_continuous", (hinge_pos_xy[0], hinge_pos_xy[1], z0 + Hh / 2), hg.pin_radius * 1.6 + 0.004, Hh / 2 - 0.01, hm, (0, 0, 1), 2700, False, True, FULL_SIMPLE, "hinge", "Continuous hinge"))
        return
    if hg.kind == "strap":
        key, mesh = MESH.strap_hinge_mesh(length=hg.size[0], width=hg.size[1], thickness=0.006)
        zs = [z0 + 0.15, z0 + Hh - 0.15] if n == 2 else [z0 + 0.15, z0 + Hh / 2, z0 + Hh - 0.15]
        wp = body_world_pos(model, leaf_body)
        y_face = v * (spec["leaf"]["thickness"] / 2 + 0.003)
        y_pin_ = float(hinge_pos_xy[1])
        for k, zz in enumerate(zs):
            q = q_axis_x_to((u, 0, 0))
            leaf_body.geoms.append(mesh_geom(f"hinge_strap_{k}", key, mesh, (hinge_pos_xy[0], y_face, zz), q, hm, 7850, False, FULL_SIMPLE, "hinge", "Strap hinge"))
            # the strap's eye drops over a pintle screwed to the post: the eye is ON the swing axis (the strap
            # plate lies on the gate face, 50 mm away, and swings away from the post without it)
            leaf_body.geoms.append(cyl(f"hinge_strap_eye_{k}", (hinge_pos_xy[0], y_pin_, zz), 0.011, 0.012, hm, (0, 0, 1), 7850, False, True, FULL_SIMPLE, "hinge", "Strap eye"))
            leaf_body.geoms.append(box(f"hinge_strap_neck_{k}", (hinge_pos_xy[0], (y_pin_ + y_face) / 2, zz), (0.012, abs(y_pin_ - y_face) / 2 + 0.004, 0.006), hm, 7850, False, True, FULL_SIMPLE, "hinge", "Strap eye neck"))
            world.geoms.append(cyl(f"hinge_pintle_{k}", (wp[0] + hinge_pos_xy[0], wp[1] + y_pin_, wp[2] + zz - 0.026), 0.008, 0.02, hm, (0, 0, 1), 7850, False, True, FULL_SIMPLE, "hinge", "Pintle"))
            # the post it is screwed to may not exist yet (a gate builds its posts after the leaf): braced at the end
            model.meta.setdefault("_brace_pending", []).append({"geom": f"hinge_pintle_{k}", "axes": ["y", "x"], "label": "Pintle post strap"})
        return
    hh = hg.size[0]
    if n == 2:
        zs = [z0 + 0.25, z0 + Hh - 0.18]
    elif n == 3:
        zs = [z0 + 0.25, z0 + Hh / 2, z0 + Hh - 0.18]
    else:
        zs = [z0 + 0.25 + i * (Hh - 0.43) / (n - 1) for i in range(n)]
    leaf_w_ = min(hg.size[1] * 0.45, spec["leaf"]["thickness"] - 0.004)
    key, mesh = MESH.hinge_mesh(height=hh, radius=hg.pin_radius * 1.5, leaf_w=leaf_w_, leaf_t=0.003)
    from ..ir import mat_to_quat
    # hinge local frame -> world: +x -> +u (into the door), +y -> -v (across the thickness), z -> -u*v (right-handed)
    q = mat_to_quat(np.array([[u, 0.0, 0.0], [0.0, -v, 0.0], [0.0, 0.0, -u * v]]))
    # a hinge whose pin stands well off the leaf face is not mortised into anything - it is carried on welded lugs
    # (a watertight door's pin sits 20 mm proud of an 8 mm plate: without the lugs both knuckles float in mid air)
    t_leaf = float(spec["leaf"]["thickness"])
    stand = abs(float(hinge_pos_xy[1])) - t_leaf / 2
    vd = 1.0 if hinge_pos_xy[1] > 0 else -1.0
    for k, zz in enumerate(zs):
        leaf_body.geoms.append(mesh_geom(f"hinge_{k}", key, mesh, (hinge_pos_xy[0], hinge_pos_xy[1], zz), q, hm, 7850, False, FULL_SIMPLE, "hinge", f"Hinge {k + 1}"))
        # the frame-side plate is static (screwed to the jamb): it must not swing with the leaf
        keyj, meshj = MESH.hinge_jamb_mesh(height=hh, radius=hg.pin_radius * 1.5, leaf_w=leaf_w_, leaf_t=0.003)
        wp = body_world_pos(model, leaf_body)
        world.geoms.append(mesh_geom(f"hinge_{k}_jamb", keyj, meshj, (wp[0] + hinge_pos_xy[0], wp[1] + hinge_pos_xy[1], wp[2] + zz), q, hm, 7850, False, FULL_SIMPLE, "hinge", f"Hinge {k + 1} jamb plate"))
        z_jamb_lo = float(geom_local_aabb(world.geoms[-1])[0][2])
        if stand > 0.012:
            # gooseneck hinge: a foot welded flat on the leaf face, a strap rising from it to the pin, and a matching
            # strap on the bulkhead - offset in z like knuckles so the two never foul each other through the swing
            p_pin = u * float(hinge_pos_xy[0])
            p_edge = min((u * lo[0] if u > 0 else -hi[0]) for lo, hi in
                         (geom_local_aabb(g) for g in leaf_body.geoms if g.semantic in ("leaf", "glass")) ) if any(
                         g.semantic in ("leaf", "glass") for g in leaf_body.geoms) else p_pin + 0.03
            y_lo, y_mid, y_hi = t_leaf / 2, t_leaf / 2 + 0.35 * stand, abs(float(hinge_pos_xy[1])) + 0.006
            foot = (p_edge + 0.002, p_edge + 0.042)
            strap = (p_pin - 0.006, p_edge + 0.012)
            leaf_body.geoms.append(box(f"hinge_{k}_lug_foot", (u * (foot[0] + foot[1]) / 2, vd * (y_lo + y_mid + 0.002) / 2, zz + hh * 0.22),
                                       ((foot[1] - foot[0]) / 2, (y_mid + 0.002 - y_lo) / 2, hh * 0.2), hm, 7850, False, True, FULL_SIMPLE, "hinge", f"Hinge {k + 1} lug foot"))
            leaf_body.geoms.append(box(f"hinge_{k}_lug", (u * (strap[0] + strap[1]) / 2, vd * (y_mid + y_hi) / 2, zz + hh * 0.22),
                                       ((strap[1] - strap[0]) / 2, (y_hi - y_mid) / 2, hh * 0.2), hm, 7850, False, True, FULL_SIMPLE, "hinge", f"Hinge {k + 1} lug (leaf)"))
            face = max(mount_face(world, wp[0] + hinge_pos_xy[0], wp[2] + zz, 0.012, hh * 0.2, vd, default=t_leaf / 2, skip_semantics=("hinge",)),
                       t_leaf / 2 + 0.002)     # never down into the leaf's own hinge plate
            gap = abs(float(hinge_pos_xy[1])) - max(0.004, hg.pin_radius * 1.5 - 0.005) - face   # up to the knuckle's surface, not through the barrel
            if gap > 0.004:
                world.geoms.append(box(f"hinge_{k}_jamb_lug", (wp[0] + hinge_pos_xy[0], wp[1] + vd * (face + gap / 2), z_jamb_lo - 0.018),
                                       (0.010, gap / 2, 0.02), hm, 7850, False, True, FULL_SIMPLE, "hinge", f"Hinge {k + 1} lug (frame)"))


# ---------------------------------------------------------------------------
# Latch bolt with one-sided coupling (re-latches on slam), deadbolt, strikes
# ---------------------------------------------------------------------------
@dataclass
class LatchResult:
    bolt_body: Body | None
    pockets: list       # (z_center, pocket_h, pocket_w_y, pocket_depth_x)
    tendons: list


def add_spring_latch(model: Model, leaf_body: Body, spec: dict, phys: dict, u: float, v: float, x_edge: float, z: float, t: float, latch: H.LatchModel, handle_joint_name: str | None, coupling_scale: float, name="latch_bolt", tiers=ALL_TIERS, robot_face_sign=None, y: float = 0.0, faceplate: bool = True):
    """Spring latch bolt body on the leaf's latch edge.  Slide joint axis = -u (positive q retracts).
    One-sided coupling: bolt_q >= scale * handle_q via a limited fixed tendon (MJCF); mimic in URDF/USD.
    `y` offsets the bolt across the leaf thickness (surface-mounted rim latches live in their case on the face);
    `faceplate` adds the visual latch faceplate flush in the leaf edge (mortised latches)."""
    throw = latch.throw
    bw, bh = latch.bolt_size
    inside = min(0.03, max(0.010, (latch.backset or 0.06) - throw - 0.009))   # retracted bolt stops short of the spindle
    bm = mat_from_material(model, "brass" if latch.kind in ("tubular_latch", "deadlatch") else "stainless", "mat_bolt")
    if faceplate and abs(y) < 1e-6:
        leaf_body.geoms.append(box(f"{name}_faceplate", (x_edge - u * 0.0006, 0.0, z), (0.0006, min(bw / 2 + 0.006, t / 2 - 0.002), bh / 2 + 0.014), bm, 8500, False, True, FULL_SIMPLE, "leaf", "Latch faceplate (flush in the edge)"))
    body = Body(name, leaf_body.name, (x_edge, y, z), QUAT_ID, None, [], [], tiers, "latch", "Latch bolt")
    body.joint = Joint(f"{name}_slide", "slide", (-u, 0, 0), (0, 0, 0), (0.0, throw), damping=2.0, frictionloss=0.3,
                       stiffness=latch.spring_rate, springref=-latch.spring_preload / max(latch.spring_rate, 1e-6), armature=1e-4,
                       role="latch", label="Latch bolt (0 = extended, + = retracted)", robot_interactive=False)
    # capsule bolt (round tip rides over the strike lip on closing): axis along u, from x=-inside to x=+throw
    r = bw / 2
    half = (throw + inside) / 2 - r
    body.geoms.append(Geom(f"{name}_capsule", "capsule", (r, max(half, 0.002)), (u * (throw - inside) / 2, 0, 0), tuple(quat_z_to((u, 0, 0))), bm, True, True, 8500, None, (0.2, 0.005, 0.0001), None, None, False, None, None, 0.0, tiers, "latch", "Latch bolt"))
    model.add_body(body)
    tendons = []
    if handle_joint_name and coupling_scale > 0:
        tendons.append(Tendon(f"{name}_coupling", [(f"{name}_slide", 1.0), (handle_joint_name, -coupling_scale)], (0.0, 10.0), 0.0, 0.0, tiers, "bolt_q >= scale*handle_q (one-sided)"))
        tendons[-1].kind = "fixed"
    pocket = {"z": z, "h": bh + 0.008, "w": bw + 0.003, "depth": throw + 0.004, "ramp": True, "ramp_rise": throw - GAP + 0.004, "y": y}
    return LatchResult(body, [pocket], tendons)


def add_deadbolt(model: Model, leaf_body: Body, spec: dict, u: float, v: float, x_edge: float, z: float, t: float, throw: float, engaged: bool, thumbturn_side: float | None, thumbturn_travel: float, thumbturn_torque: float, name="deadbolt", tiers=FULL_SIMPLE, keyed_side: float | None = None, tt_standoff: float = 0.0, couple_to: tuple | None = None, faceplate: bool = True):
    """Deadbolt: slide body (axis -u, + = retracted); thumbturn body on the given face drives it (bilateral equality).
    tt_standoff: thumbturn / cylinder faces stand off the leaf face (on a rim case).  couple_to=(joint, ratio): the bolt
    is driven by another joint instead of its own thumbturn (multipoint lock points)."""
    bw, bh = 0.016, 0.025
    inside = 0.030
    bm = mat_from_material(model, "brass", "mat_deadbolt")
    if faceplate:
        leaf_body.geoms.append(box(f"{name}_faceplate", (x_edge - u * 0.0006, 0.0, z), (0.0006, min(bw / 2 + 0.006, t / 2 - 0.002), bh / 2 + 0.012), bm, 8500, False, True, FULL_SIMPLE, "leaf", "Deadbolt faceplate (flush in the edge)"))
    body = Body(name, leaf_body.name, (x_edge, 0.0, z), QUAT_ID, None, [], [], tiers, "lock", "Deadbolt")
    q0 = 0.0 if engaged else throw
    if thumbturn_side is not None or couple_to is not None:
        body.joint = Joint(f"{name}_slide", "slide", (-u, 0, 0), (0, 0, 0), (0.0, throw), damping=5.0, frictionloss=1.5, armature=1e-4, role="lock", label="Deadbolt (0 = thrown, + = retracted)", robot_interactive=False, initial=q0, modeled_at=q0)
    else:
        body.joint = None  # fixed in place (keyed both sides / no robot release)
    xoff = 0.0 if engaged else -u * throw
    body.geoms.append(box(f"{name}_box", (u * (throw - inside) / 2 + xoff, 0, 0), ((throw + inside) / 2, bw / 2, bh / 2), bm, 8500, True, True, tiers, "lock", "Deadbolt", friction=(0.3, 0.005, 0.0001)))
    model.add_body(body)
    eqs = []
    if couple_to is not None:
        eqs.append(Equality("joint", f"{name}_couple", f"{name}_slide", couple_to[0], (0.0, couple_to[1], 0, 0, 0), tiers=tiers, label="lock point follows the main bolt"))
    elif thumbturn_side is not None:
        tt = Body(f"{name}_thumbturn", leaf_body.name, (x_edge - u * 0.065, thumbturn_side * (t / 2 + tt_standoff), z), QUAT_ID, None, [], [], tiers, "lock", "Thumbturn")
        tt.joint = Joint(f"{name}_thumbturn_hinge", "hinge", (0, -thumbturn_side, 0), (0, 0, 0), (0.0, thumbturn_travel), damping=0.05, frictionloss=thumbturn_torque, armature=1e-5, role="lock", label="Thumbturn (rotate to retract deadbolt)", initial=0.0 if engaged else thumbturn_travel)
        key, mesh = MESH.thumbturn_mesh()
        tt.geoms.append(mesh_geom(f"{name}_thumbturn_mesh", key, mesh, (0, 0, 0), q_face(thumbturn_side, u), bm, 7100, True, tiers, "lock", "Thumbturn"))
        tt.geoms.append(box(f"{name}_thumbturn_col", (0, thumbturn_side * 0.02, 0), (0.006, 0.012, 0.016), bm, 7100, True, False, tiers, "lock", "Thumbturn collision"))
        model.add_body(tt)
        # The thumbturn's spindle passes THROUGH the lock case to drive the bolt: `{name}_thumbturn_mesh` and
        # `{name}_box` overlap by 2-5 mm at rest by construction (clearance.DEFAULT_ALLOW documents the same pair).
        # MuJoCo resolves that overlap softly and PhysX - since self-collision was enabled on the articulation -
        # resolves it rigidly, fighting the very equality below that makes the deadbolt work.  Excluding the pair
        # keeps both engines free of a contact that no real lock has (the case is bored for the spindle), and the
        # exporter mirrors the exclude into PhysxFilteredPairsAPI automatically.
        model.contact_excludes.append((body.name, tt.name))
        eqs.append(Equality("joint", f"{name}_couple", f"{name}_slide", f"{name}_thumbturn_hinge", (0.0, throw / thumbturn_travel, 0, 0, 0), tiers=tiers, label="deadbolt = throw/travel * thumbturn"))
    if keyed_side is not None:
        key, mesh = MESH.cylinder_face_mesh()
        leaf_body.geoms.append(mesh_geom(f"{name}_cylinder_face", key, mesh, (x_edge - u * 0.065, keyed_side * (t / 2 + (tt_standoff if keyed_side == thumbturn_side else 0.0)), z), q_face(keyed_side, u), bm, 7100, False, FULL_ONLY, "lock", "Key cylinder"))
    pocket = {"z": z, "h": bh + 0.006, "w": bw + 0.003, "depth": throw + 0.004, "ramp": False}
    return body, [pocket], eqs


def add_strike_plate(geoms, prefix, sx_w, u, y, z, half_w, half_h, material, tiers=FULL_SIMPLE):
    """Visual strike plate: a flat ring 1 mm proud of the strike jamb's reveal around a (non-ramped) pocket mouth."""
    add_keeper_ring(geoms, prefix, (sx_w, y, z), (-u, 0, 0), (0, 0, 1), half_w, half_h, material, bar=0.006, thick=0.001, tiers=tiers, semantic="lock", label="Strike plate")


# ---------------------------------------------------------------------------
# Surface-mounted bolt hardware: barrel / slide / drop bolts with mounting plate, guide loops, handle and keeper;
# hasp + staple + padlock; strike / face plates.  All built from convex primitives (boxes, capsules, spheres).
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def frame_quat(axis, normal):
    """Quaternion of the right-handed frame (x = axis, y = normal x axis, z = normal)."""
    a, n = _unit(axis), _unit(normal)
    lat = np.cross(n, a)
    return tuple(mat_to_quat(np.column_stack([a, lat, n])))


def obox(name, origin, axis, normal, s, l, n, hs, hl, hn, material, collision=False, tiers=FULL_SIMPLE, semantic="lock", label="", density=7800.0, friction=(0.6, 0.005, 0.0001)):
    """Box expressed in an (axis, lateral, normal) frame anchored at `origin`: centre = origin + axis*s + lateral*l +
    normal*n, half extents (hs, hl, hn) along (axis, lateral, normal).  lateral = normal x axis."""
    a, nn = _unit(axis), _unit(normal)
    lat = np.cross(nn, a)
    c = np.asarray(origin, float) + a * s + lat * l + nn * n
    return box(name, tuple(c), (hs, hl, hn), material, density, collision, True, tiers, semantic, label, friction, quat=frame_quat(a, nn))


def add_guide_loop(geoms, prefix, origin, axis, normal, s, half_lat, height, material, bar=0.004, bar_len=0.012, collision=False, tiers=FULL_SIMPLE, semantic="lock", label="Guide loop", base=0.0):
    """Rectangular U-loop standing on the mounting surface at origin + axis*s: two legs (lateral +-(half_lat + bar/2))
    rising `height` above the surface and a bridge across the top.  A rod/bar of half-width `half_lat` passes through
    along `axis`.  `base` > 0 adds a base plate of that length along the axis."""
    for sgn, tag in ((-1, "n"), (1, "p")):
        geoms.append(obox(f"{prefix}_leg_{tag}", origin, axis, normal, s, sgn * (half_lat + bar / 2), (height + bar) / 2, bar_len / 2, bar / 2, (height + bar) / 2, material, collision, tiers, semantic, label))
    geoms.append(obox(f"{prefix}_bridge", origin, axis, normal, s, 0.0, height + bar / 2, bar_len / 2, half_lat + bar, bar / 2, material, collision, tiers, semantic, label))
    if base > 0:
        geoms.append(obox(f"{prefix}_base", origin, axis, normal, s, 0.0, 0.001, base / 2, half_lat + bar + 0.006, 0.001, material, False, tiers, semantic, label + " base"))


def add_keeper_ring(geoms, prefix, center, normal, up, half_w, half_h, material, bar=0.006, thick=0.001, tiers=FULL_SIMPLE, semantic="lock", label="Keeper plate"):
    """Flat rectangular ring (plate with a hole) around a pocket mouth, 1 mm proud of the member face at `center`
    (normal = outward face normal, up = in-plane vertical).  Visual only: reads as a strike / keeper plate."""
    n, u_ = _unit(normal), _unit(up)
    side = np.cross(u_, n)
    for sgn in (-1, 1):
        geoms.append(obox(f"{prefix}_{'t' if sgn > 0 else 'b'}", center, side, n, 0.0, sgn * (half_h + bar / 2), thick / 2, half_w + bar, bar / 2, thick / 2, material, False, tiers, semantic, label))
        geoms.append(obox(f"{prefix}_{'r' if sgn > 0 else 'l'}", center, side, n, sgn * (half_w + bar / 2), 0.0, thick / 2, bar / 2, half_h, thick / 2, material, False, tiers, semantic, label))


def add_barrel_bolt(model: Model, parent: Body, name: str, ref, axis, normal, L: float, d: float, travel: float, engaged: bool, material: str, *, protrusion=None, standoff=None, tiers=ALL_TIERS, role="lock", label=None, frictionloss=1.0, damping=1.0, handle_at="front", handle_len=None, joint_name=None, grip_site=None, robot_interactive=True, rod_semantic="latch", body_semantic="lock", plate=True):
    """Surface-mounted barrel / slide / drop bolt on `parent`: mounting plate + 2-3 guide loops (fixed to the parent),
    a sliding rod with an L-handle (own body + slide joint, 0 = engaged, + = withdrawn).

    ref     : point (parent frame) ON THE MOUNTING SURFACE at the edge line the bolt crosses
    axis    : unit direction of engagement (the rod moves +axis to engage; its tip passes `ref` by `protrusion`)
    normal  : unit outward normal of the mounting surface; the rod axis lies at ref + normal * standoff
    Geometry is authored engaged; `initial` moves it to the withdrawn state when not engaged.
    Returns (bolt_body, info) where info has the rod axis point at the edge line (parent frame), radius, protrusion."""
    a, n = _unit(axis), _unit(normal)
    ref = np.asarray(ref, float)
    r = d / 2
    protrusion = min(0.045, travel - 0.004) if protrusion is None else protrusion   # withdrawn tip clears the edge by >= 4 mm
    standoff = max(d, 0.010) if standoff is None else standoff
    rod0 = ref + n * standoff
    bar, bar_len, gap = 0.004, 0.012, 0.003
    knob_r = min(0.008, max(0.005, 0.8 * r))
    hl = handle_len or (0.03 if d <= 0.014 else 0.05)
    g1 = -0.010 - bar_len / 2                    # guide next to the edge
    s_rear = protrusion - L                      # rear end of the rod when engaged
    if handle_at == "front":
        s_knob = g1 - bar_len / 2 - 0.008 - knob_r
        g2 = s_rear - travel - 0.008 + bar_len / 2
    else:                                        # handle at the rear end (cane / drop bolts: the rod is bent over)
        s_knob = s_rear + knob_r + 0.006
        g2 = s_knob - travel - knob_r - 0.010 - bar_len / 2
    guides = [g1, g2]
    mid = (g1 + g2) / 2
    sweep = (s_knob - travel - knob_r - 0.006, s_knob + knob_r + 0.006)
    if g1 - g2 > 0.22 and not (sweep[0] < mid + bar_len / 2 + 0.004 and sweep[1] > mid - bar_len / 2 - 0.004):
        guides.append(mid)
    lbl = label or "Slide bolt (0 = engaged, + = withdrawn)"
    body = Body(name, parent.name, tuple(rod0), QUAT_ID, None, [], [], tiers, body_semantic, lbl.split(" (")[0])
    body.joint = Joint(joint_name or f"{name}_slide", "slide", tuple(-a), (0, 0, 0), (0.0, travel), damping=damping, frictionloss=frictionloss, role=role, label=lbl, robot_interactive=robot_interactive, initial=0.0 if engaged else travel, modeled_at=0.0)
    body.geoms.append(Geom(f"{name}_rod", "capsule", (r, max(L / 2 - r, 0.004)), tuple(a * (protrusion - L / 2)), tuple(quat_z_to(a)), material, True, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, tiers, rod_semantic, "Bolt rod"))
    body.geoms.append(Geom(f"{name}_knob", "capsule", (knob_r, max(hl / 2 - knob_r, 0.004)), tuple(a * s_knob + n * (hl / 2)), tuple(quat_z_to(n)), material, True, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, tiers, "operator", "Bolt handle"))
    body.geoms.append(sphere(f"{name}_knob_end", tuple(a * s_knob + n * hl), knob_r * 1.35, material, 7850, False, FULL_ONLY, "operator", "Handle knob"))
    body.sites.append(Site(grip_site or f"{name}_grip", tuple(a * s_knob + n * (hl + 0.004)), QUAT_ID, 0.012, "grip"))
    model.add_body(body)
    if plate:
        plo, phi = g2 - bar_len / 2 - 0.004, -0.008
        parent.geoms.append(obox(f"{name}_plate", ref, a, n, (plo + phi) / 2, 0.0, 0.001, (phi - plo) / 2, r + gap + bar + 0.006, 0.001, material, False, FULL_SIMPLE, "lock", "Bolt mounting plate"))
    for k, s in enumerate(guides):
        add_guide_loop(parent.geoms, f"{name}_guide_{k}", ref, a, n, s, r + gap, standoff + r + gap, material, bar, bar_len, False, FULL_SIMPLE, "lock", "Bolt guide loop")
    return body, {"rod0": rod0, "axis": a, "normal": n, "r": r, "protrusion": protrusion, "standoff": standoff, "s_rear": s_rear, "s_knob": s_knob, "handle_len": hl}


def add_keeper_loop(geoms, prefix, surface_pt, rod_pt, axis, normal, r, material, tiers=ALL_TIERS, semantic="lock", label="Keeper loop", base=0.03, bar=0.005, bar_len=0.014, collision=True):
    """Surface keeper for a bolt rod: U-loop standing on `surface_pt` (member face, outward `normal`) straddling the
    rod whose axis passes through `rod_pt` along `axis` (3 mm clearance).  Blocks the rod in both lateral directions."""
    h = float(np.dot(np.asarray(rod_pt, float) - np.asarray(surface_pt, float), _unit(normal)))
    add_guide_loop(geoms, prefix, surface_pt, axis, normal, 0.0, r + 0.003, h + r + 0.003, material, bar, bar_len, collision, tiers, semantic, label, base=base)


def add_padlock(geoms, prefix, bar_pt, bar_dir, hang, material, tiers=ALL_TIERS, semantic="lock", label="Padlock"):
    """Padlock: shackle bar (through an eye at bar_pt, along bar_dir) with two legs down `hang` into a body block."""
    b, h = _unit(bar_dir), _unit(hang)
    bp = np.asarray(bar_pt, float)
    geoms.append(Geom(f"{prefix}_shackle", "capsule", (0.004, 0.016), tuple(bp), tuple(quat_z_to(b)), material, False, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, tiers, semantic, label + " shackle"))
    for sgn, tag in ((-1, "a"), (1, "b")):
        geoms.append(Geom(f"{prefix}_leg_{tag}", "capsule", (0.004, 0.012), tuple(bp + b * sgn * 0.016 + h * 0.012), tuple(quat_z_to(h)), material, False, True, 7850.0, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, tiers, semantic, label + " shackle"))
    c = bp + h * (0.024 + 0.018)
    # frame_quat(b, b x h): local x = shackle bar, local y = hang, local z = across the body
    geoms.append(box(f"{prefix}_body", tuple(c), (0.022, 0.018, 0.008), material, 7850, True, True, tiers, semantic, label, quat=frame_quat(b, np.cross(b, h))))


def add_hasp_assembly(model: Model, leaf_body: Body, world: Body, name: str, hinge_pt, normal, toward, strap_len: float, plane_h: float, eye_pt_w, staple_surf_w, staple_normal_w, locked: bool, material: str, lock_material: str, tiers=ALL_TIERS, hang=None):
    """Hinged hasp on the leaf (strap on a spacer/hinge plate, flips open about an axis in the leaf face) + U-staple on
    the frame / post (world) + padlock hanging in the staple eye when `locked`.

    hinge_pt / normal / toward : leaf frame; hinge line on the leaf surface, outward face normal, direction to the staple
    plane_h                    : height of the strap underside above the leaf surface (spacer block)
    eye_pt_w                   : world point of the staple eye (padlock shackle bar) - on the strap's outer surface + 8 mm
    staple_surf_w / normal_w   : world mounting point / outward normal of the staple base
    Unlocked hasps are modelled flipped back (open) so the leaf is free."""
    n, tw = _unit(normal), _unit(toward)
    lat = np.cross(n, tw)
    axis_j = np.cross(tw, n)
    w, tth = 0.035, 0.003
    hp = np.asarray(hinge_pt, float)
    ph = max(plane_h, 0.003)
    leaf_body.geoms.append(obox(f"{name}_hasp_plate", hp, tw, n, -0.020, 0.0, ph / 2, 0.024, w / 2 + 0.004, ph / 2, material, False, FULL_SIMPLE, "lock", "Hasp hinge plate / spacer"))
    hb = Body(f"{name}_hasp", leaf_body.name, tuple(hp + n * ph), QUAT_ID, None, [], [], tiers, "lock", "Hasp")
    hb.joint = Joint(f"{name}_hasp_hinge", "hinge", tuple(axis_j), (0, 0, 0), (0.0, 0.0015 if locked else 2.9), damping=0.02, frictionloss=0.05, role="lock", label="Hasp (0 = closed over the staple, + = flipped open)" + (" [padlocked]" if locked else ""), initial=0.0 if locked else 2.8, modeled_at=0.0)
    hb.geoms.append(obox(f"{name}_hasp_strap", (0, 0, 0), tw, n, strap_len / 2, 0.0, tth / 2, strap_len / 2, w / 2, tth / 2, material, True, tiers, "lock", "Hasp strap"))
    hb.geoms.append(cyl(f"{name}_hasp_knuckle", tuple(n * 0.004), 0.0045, w / 2 + 0.004, material, tuple(axis_j), 7850, False, True, FULL_ONLY, "lock", "Hasp hinge knuckle"))
    # The turned-up tab at the free end is what a hand lifts, and on a `hasp_padlock` door the hasp IS the
    # operator the spec declares - it used to be drawn entirely as `lock`, so the benchmark's grip sites, the
    # viewer's handle camera and the review's hardware close-up all missed it on 9 doors.
    hg_ = np.array([0.0, 0.0, -1.0]) if hang is None else _unit(hang)
    tab_side = -1.0 if float(np.dot(lat, hg_)) > 0 else 1.0     # on the side the padlock does NOT hang toward
    hb.geoms.append(obox(f"{name}_hasp_tab", tuple(tw * (strap_len - 0.030) + lat * (tab_side * (w / 2 + 0.006)) + n * 0.004),
                         tw, n, 0.0, 0.0, 0.0, 0.012, 0.006, 0.006, material, False, tiers, "operator",
                         "Hasp lifting tab"))
    hb.sites.append(Site(f"{name}_hasp_grip", tuple(tw * (strap_len - 0.02) + n * 0.01), QUAT_ID, 0.01, "grip"))
    model.add_body(hb)
    eye = np.asarray(eye_pt_w, float)
    surf = np.asarray(staple_surf_w, float)
    sn = _unit(staple_normal_w)
    tw_w = tw   # leaf frames are axis-aligned with the world in this IR (no leaf rotation at q = 0)
    h_eye = float(np.dot(eye - surf, sn))
    add_guide_loop(world.geoms, f"{name}_staple", surf, tw_w, sn, 0.0, 0.010, h_eye + 0.006, material, 0.005, 0.006, True, ALL_TIERS, "lock", "Staple", base=0.032)
    if locked:
        hg = np.array([0.0, 0.0, -1.0]) if hang is None else _unit(hang)
        add_padlock(world.geoms, f"{name}_padlock", eye, tw_w, hg, lock_material, ALL_TIERS, "lock", "Padlock")
    model.meta.setdefault("clearance_allow", []).extend([
        [f"{name}_hasp_strap", f"{name}_staple*", "staple passes through the strap slot"],
        [f"{name}_padlock*", f"{name}_staple*", "shackle through the staple eye"],
        [f"{name}_padlock*", f"{name}_hasp_strap", "padlock hangs over the strap"],
        [f"{name}_hasp_strap", f"{name}_hasp_plate", "strap over its hinge plate"],
        [f"{name}_hasp_knuckle", f"{name}_hasp_plate", "knuckle on its hinge plate"],
        [f"{name}_hasp_tab", f"{name}_staple*", "the lifting tab reaches past the staple"]])
    return hb


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
RETURN_LABEL = {"spring": "; spring return", "gravity": "; gravity return", "detent": "; stays where put", "none": ""}


def operator_faces(spec: dict, v: float):
    """Return list of face signs (in y) where the primary operator is present, and far-side operator id.
    Robot is at -y.  'push_side' = the face the door swings away from = -v side."""
    sides = spec["operator"].get("sides", "both")
    if sides == "both":
        return [-1.0, 1.0], None
    if sides == "push_side":
        return [-v], spec["operator"].get("far_side")
    if sides == "robot":
        return [-1.0], None
    if sides == "far":
        return [1.0], None
    return [], None


def add_paddle_operator(model: Model, leaf_body: Body, spec: dict, op: H.OperatorModel,
                        u: float, v: float, x_spindle: float, z: float, t: float,
                        faces: list, locked_backlash: float | None, name="handle", tiers=ALL_TIERS):
    """Vertical push/pull paddles on horizontal, face-mounted pivots.

    HL6-9000 installation instructions, p. 2, show fixed base plates, separate
    paddle pivots and cams.  This is a generic rocker representation, not an
    exact HL6 mechanism: opposite paddles use an explicit ideal 1:1 cam
    coupling.  The existing primary joint continues to drive the latch.
    See docs/research/paddle-mechanics.md for dimensions and limitations.
    """
    if not faces or any(f not in (-1.0, 1.0) for f in faces) or len(set(faces)) != len(faces):
        raise ValueError("Paddles require distinct door faces, each -1 or +1")
    mat = mat_from_material(model, op.material, f"mat_op_{op.material}")
    w, h = op.style_params.get("size", (0.10, 0.18))
    half_width, half_height = min(w / 2, 0.032), h * 0.35
    arm = op.grip_offset
    if arm <= half_height or not 0 < op.travel < math.pi / 4:
        raise ValueError("Paddle requires a grip below its pivot and a short throw")
    backset = H.LATCHES[spec["latch"]["model"]].backset or 0.06
    cx = x_spindle - u * max(0.012, half_width + 0.08 - backset)
    # Start tipped away from the leaf.  At full push travel the plate is
    # upright and remains 12 mm clear of the leaf; an axis-only change to the
    # former flat plate would drive its lower edge through the door.
    lean = op.travel
    pin_standoff = max(0.018, op.style_params.get("standoff", 0.045) - arm * math.sin(lean))
    pivot_z = z + arm * math.cos(lean)
    axis = (v, 0.0, 0.0)
    rng = (0.0, max(locked_backlash, 0.01)) if locked_backlash is not None else (0.0, op.travel)
    preload = max(op.spring_torque_preload, 1.5)  # retained native return-spring floor
    primary = None
    description = {"primary_joint": f"{name}_hinge", "faces": [],
                   "coupling": "ideal 1:1 paddle cam; internal cam contact and lost motion not modeled",
                   "source": "https://commercial.schlage.com/content/dam/allegion-us-2/web-documents-2/InstallInstructions/Glynn-Johnson_HL6-9000_Push_Pull_Mortise_Latch_Installation_Instructions_107563.pdf",
                   "source_page": 2, "rest_lean_rad": lean, "grip_moment_arm_m": arm}
    for i, f in enumerate(faces):
        tag = "p" if f > 0 else "n"
        body_name = name if i == 0 else f"{name}_paddle_follower_{tag}"
        pivot = (cx, f * (t / 2 + pin_standoff), pivot_z)
        body = Body(body_name, leaf_body.name, pivot, QUAT_ID, None, [], [], tiers, "operator", op.name)
        body.joint = Joint(f"{name}_hinge" if i == 0 else f"{body_name}_hinge", "hinge", axis,
                           range=rng, damping=0.02 if i == 0 else 0.0,
                           frictionloss=(0.02 + 0.02 * op.mass) if i == 0 else 0.0,
                           stiffness=op.spring_rate if i == 0 else 0.0,
                           springref=-preload / op.spring_rate if i == 0 and op.spring_rate > 0 else 0.0,
                           armature=2e-5, role="operator" if i == 0 else "mechanism",
                           robot_interactive=i == 0,
                           return_kind=op.return_kind if i == 0 else "", operator_model=op.id if i == 0 else "",
                           label=f"{op.name} (0 = rest, + = actuated{RETURN_LABEL.get(op.return_kind, '') if i == 0 else ''})",
                           notes="Face-mounted rocker; ideal cam drives latch" +
                           ("; locked: range limited to backlash" if locked_backlash is not None else ""))
        q = quat_from_axis_angle((1, 0, 0), f * lean)
        center = quat_rotate(q, (0, 0, -arm))
        geom_name = f"{name}_paddle_col_{tag}"
        body.geoms.append(box(geom_name, tuple(center), (half_width, 0.006, half_height), mat,
                              3000, True, True, tiers, "operator", "Paddle grip plate", quat=tuple(q)))
        neck_length = arm - half_height + 0.015
        body.geoms.append(box(f"{name}_paddle_neck_{tag}",
                              tuple(quat_rotate(q, (0, 0, -neck_length / 2))),
                              (0.012, 0.004, neck_length / 2), mat, 3000, True, True, tiers,
                              "operator", "Paddle neck", quat=tuple(q)))
        body.geoms.append(cyl(f"{name}_paddle_hub_{tag}", (0, 0, 0), 0.008, 0.016,
                              mat, (1, 0, 0), 3000, False, True, tiers, "mechanism", "Paddle pivot hub"))
        # A contact force INTO this face produces +arm N*m per newton.  It is
        # the outer face on the push side, and the finger-accessible inner
        # face on the pull side.  Site local +Z is the outward surface normal.
        site_name = f"{name}_grip_{tag}"
        site_q = quat_mul(q, quat_z_to((0, -v, 0)))
        body.sites.append(Site(site_name, tuple(quat_rotate(q, (0, -v * 0.006, -arm))),
                               tuple(site_q), 0.012, "push" if f == -v else "grip", tiers))
        # Door-fixed mounting plate, bearing ears and pivot pin.  No rocking
        # spindle passes through the slab and no mounting plate rides the joint.
        leaf_body.geoms.append(box(f"{name}_paddle_backplate_{tag}",
                                    (cx, f * (t / 2 + 0.003), pivot_z - h * 0.35),
                                    (half_width + 0.012, 0.003, h * 0.5), mat, 3000,
                                    True, True, tiers, "operator", "Fixed paddle backplate"))
        for side in (-1, 1):
            leaf_body.geoms.append(box(f"{name}_paddle_bearing_{tag}_{side}",
                                        (cx + side * (half_width + 0.006),
                                         f * (t / 2 + (pin_standoff + 0.006) / 2), pivot_z),
                                        (0.005, (pin_standoff - 0.006) / 2, 0.009), mat,
                                        3000, True, True, tiers, "mechanism", "Fixed paddle pivot support"))
        leaf_body.geoms.append(cyl(f"{name}_paddle_pin_{tag}", pivot, 0.004, half_width + 0.011,
                                   mat, (1, 0, 0), 7850, False, True, tiers, "mechanism", "Fixed paddle pivot pin"))
        model.add_body(body)
        if primary is None:
            primary = body
        else:
            model.equalities.append(Equality("joint", f"{name}_paddle_cam_{tag}", body.joint.name,
                                             primary.joint.name, (0, 1, 0, 0, 0), tiers=tiers,
                                             label="Ideal paddle cam: follower q = primary q"))
        description["faces"].append({"face": f, "body": body.name, "joint": body.joint.name,
                                      "site": site_name, "geom": geom_name,
                                      "action": "push" if f == -v else "pull",
                                      "contact_face": "outer" if f == -v else "inner"})
    # the rocker's hub and neck are one casting clamped on the door-fixed pivot pin: the pin is INSIDE them by design
    model.meta.setdefault("clearance_allow", []).extend([
        ["*_paddle_neck_*", "*_paddle_pin_*", "the rocker neck is part of the hub that is clamped on its pivot pin"],
        ["*_paddle_hub_*", "*_paddle_pin_*", "the pivot pin runs in the hub's bore"],
    ])
    model.meta.setdefault("paddle_mechanisms", []).append(description)
    return primary


def add_rotary_operator(model: Model, leaf_body: Body, spec: dict, phys: dict, op: H.OperatorModel, u: float, v: float, x_spindle: float, z: float, t: float, faces: list, locked_backlash: float | None, name="handle", tiers=ALL_TIERS, keypad_face: float = -1.0, cylinder_face: float | None = None, button_face: float | None = None, rim_case_face: float | None = None, spindle: bool = True):
    """Lever/knob/thumbturn-type operator rotating about the door normal.  One body through the door
    (spindle) carrying operator meshes on the requested faces.  Positive q = actuating (press down).
    cylinder_face / button_face: face carrying a key cylinder (keyed lever / knob) or a privacy turn button;
    rim_case_face: face carrying a surface-mounted rim lock case that the knob spindle passes through;
    spindle=False omits the spindle geom (a second trim body sharing the same bore, e.g. a declutched keypad lever)."""
    if op.kind == "paddle":
        return add_paddle_operator(model, leaf_body, spec, op, u, v, x_spindle, z, t,
                                   faces, locked_backlash, name, tiers)
    mat = mat_from_material(model, op.material, f"mat_op_{op.material}")
    body = Body(name, leaf_body.name, (x_spindle, 0.0, z), QUAT_ID, None, [], [], tiers, "operator", op.name)
    outside = 1.0 if not spec["robot"].get("robot_outside") else -1.0
    sp = op.style_params
    if rim_case_face is not None and sp.get("rim_box"):
        # surface rim lock case (Carpenter rim lock): box on the inside face around the spindle, up to the latch edge
        rw, rh, rd = sp["rim_box"]
        rm_ = mat_from_material(model, "cast_iron", "mat_rim_case")
        backset_r = H.LATCHES[spec["latch"]["model"]].backset or 0.06
        # the case ends 14 mm short of the leaf edge (clear of the frame stop moulding the door closes against)
        leaf_body.geoms.append(box(f"{name}_rim_case", (x_spindle + u * (backset_r - 0.014 - rw / 2), rim_case_face * (t / 2 + rd / 2), z), (rw / 2, rd / 2, rh / 2), rm_, 2500, True, True, FULL_SIMPLE, "lock", "Rim lock case"))
        model.meta.setdefault("clearance_allow", []).extend([[f"{name}_knob_*", f"{name}_rim_case", "knob spindle through the rim case"], [f"{name}_spindle", f"{name}_rim_case", "spindle through the case"]])
    travel = op.travel
    rng = (0.0, travel)
    if locked_backlash is not None:
        rng = (0.0, max(locked_backlash, 0.01))
    axis = (0.0, -u, 0.0)   # pressing lever (reaching -u) down = positive
    preload_ = op.spring_torque_preload
    body.joint = Joint(f"{name}_hinge", "hinge", axis, (0, 0, 0), rng, damping=0.02, frictionloss=0.02 + 0.02 * op.mass,
                       stiffness=op.spring_rate if op.return_kind == "spring" else 0.0,
                       springref=(-preload_ / op.spring_rate) if (op.return_kind == "spring" and op.spring_rate > 0) else 0.0, armature=2e-5,
                       role="operator", label=f"{op.name} (0 = rest, + = actuated{RETURN_LABEL.get(op.return_kind, '')})",
                       return_kind=op.return_kind, operator_model=op.id,
                       notes="locked: range limited to backlash" if locked_backlash is not None else "")
    # spindle through door - sized after the trim is built, so it actually reaches INTO the hub on each face
    # (a spindle that stops at the door faces leaves each lever floating as its own island)
    spindle_i = len(body.geoms)
    grip_sites = []
    trim_m = mat_from_material(model, "brass", "mat_trim")
    for f in faces:
        q = q_face(f, u)
        so_f = 0.0
        if rim_case_face is not None and sp.get("rim_box") and f == rim_case_face:
            so_f = sp["rim_box"][2]           # knob on the rim case's outer face
        if op.kind in ("lever", "keypad_lever", "card_lever", "dog"):
            key, mesh = MESH.lever_mesh(shape=sp.get("shape", "straight"), length=sp.get("length", 0.12), diameter=sp.get("diameter", 0.019), rose_diameter=sp.get("rose_diameter", 0.07), standoff=0.055, square=sp.get("square", False), ret=sp.get("return", False), escutcheon=None)
            body.geoms.append(mesh_geom(f"{name}_lever_{'p' if f > 0 else 'n'}", key, mesh, (0, f * t / 2, 0), q, mat, 3000, False, ALL_TIERS, "operator", "Lever"))
            if sp.get("escutcheon"):
                # the escutcheon plate is screwed to the leaf: it must not rotate with the lever
                eh, ew = sp["escutcheon"]
                leaf_body.geoms.append(box(f"{name}_escutcheon_{'p' if f > 0 else 'n'}", (x_spindle, f * (t / 2 + 0.0015), z), (ew / 2, 0.0015, eh / 2), mat, 3000, False, True, FULL_SIMPLE, "operator", "Escutcheon plate"))
            L, d = sp.get("length", 0.12), sp.get("diameter", 0.019)
            # collision capsule for the lever arm (all tiers), reach toward -u
            body.geoms.append(Geom(f"{name}_lever_col_{'p' if f > 0 else 'n'}", "capsule", (d / 2, L / 2 - d / 2), (-u * L / 2, f * (t / 2 + 0.055), 0), tuple(quat_z_to((-u, 0, 0))), mat, True, False, 3000, None, (0.7, 0.01, 0.0001), None, None, False, None, None, 0.0, ALL_TIERS, "operator", "Lever grip"))
            body.geoms.append(cyl(f"{name}_hub_col_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + 0.030), 0), 0.012, 0.022, mat, (0, 1, 0), 3000, True, False, ALL_TIERS, "operator", "Lever hub"))
            grip_sites.append(Site(f"{name}_grip_{'p' if f > 0 else 'n'}", (-u * L * 0.8, f * (t / 2 + 0.055), 0), QUAT_ID, 0.012, "grip"))
            # key cylinder in the hub end (keyed lever) / turn button (privacy lever) - protrude through the arm tube
            if cylinder_face is not None and f == cylinder_face:
                body.geoms.append(cyl(f"{name}_cylinder_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + 0.055 + d / 2 - 0.002), 0), 0.0085, 0.006, trim_m, (0, 1, 0), 7100, False, True, FULL_ONLY, "lock", "Key cylinder"))
                body.geoms.append(box(f"{name}_keyway_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + 0.055 + d / 2 + 0.0045), 0), (0.0012, 0.0008, 0.006), mat_rgba(model, "mat_keyway", (0.05, 0.05, 0.05, 1), 0.5), 7100, False, True, FULL_ONLY, "lock", "Keyway"))
            if button_face is not None and f == button_face:
                body.geoms.append(cyl(f"{name}_turn_button_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + 0.055 + d / 2 + 0.001), 0), 0.006, 0.005, trim_m, (0, 1, 0), 7100, False, True, FULL_ONLY, "lock", "Privacy turn button"))
        elif op.kind in ("knob", "keypad_deadbolt"):
            # A keypad deadbolt set (Schlage BE365) is a plain passage KNOB with the keypad on the rose above it.
            # `keypad_deadbolt` was not in this list, so 9 doors declared the set and drew no knob at all - the
            # spindle turned and there was nothing on either face to turn it by (docs/VISION_REVIEW.md class 12).
            key, mesh = MESH.knob_mesh(shape=sp.get("shape", "round"), diameter=sp.get("diameter", 0.054), depth=sp.get("depth", 0.06), rose_diameter=sp.get("rose_diameter", 0.064), childproof_cover=sp.get("childproof_cover", 0.0), privacy_button=bool(sp.get("privacy_button", False)) and f == outside * -1.0)
            body.geoms.append(mesh_geom(f"{name}_knob_{'p' if f > 0 else 'n'}", key, mesh, (0, f * (t / 2 + so_f), 0), q, mat, 3000, False, ALL_TIERS, "operator", "Knob"))
            D, dep = sp.get("diameter", 0.054), sp.get("depth", 0.06)
            body.geoms.append(sphere(f"{name}_knob_col_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + so_f + dep - D / 2 * 0.5), 0), D / 2 * 0.9, mat, 3000, True, ALL_TIERS, "operator", "Knob grip"))
            body.geoms[-1].friction = (0.9 if sp.get("childproof_cover", 0) == 0 else 0.15, 0.01, 0.0001)
            body.geoms.append(cyl(f"{name}_knob_neck_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + so_f + 0.012), 0), 0.011, 0.012, mat, (0, 1, 0), 3000, True, False, ALL_TIERS, "operator", "Knob neck"))
            grip_sites.append(Site(f"{name}_grip_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + so_f + dep - D / 2 * 0.5), 0), QUAT_ID, 0.012, "grip"))
            # key cylinder in the knob face (keyed knob); the privacy knob carries its own button in the mesh
            y_face_k = t / 2 + so_f + (dep - 0.75 * D) + 0.9 * D
            if cylinder_face is not None and f == cylinder_face and not sp.get("childproof_cover"):
                body.geoms.append(cyl(f"{name}_cylinder_{'p' if f > 0 else 'n'}", (0, f * (y_face_k + 0.0005), 0), 0.0085, 0.0015, trim_m, (0, 1, 0), 7100, False, True, FULL_ONLY, "lock", "Key cylinder"))
                body.geoms.append(box(f"{name}_keyway_{'p' if f > 0 else 'n'}", (0, f * (y_face_k + 0.0025), 0), (0.0012, 0.0008, 0.006), mat_rgba(model, "mat_keyway", (0.05, 0.05, 0.05, 1), 0.5), 7100, False, True, FULL_ONLY, "lock", "Keyway"))
            if button_face is not None and f == button_face and not sp.get("privacy_button") and not sp.get("childproof_cover"):
                body.geoms.append(cyl(f"{name}_turn_button_{'p' if f > 0 else 'n'}", (0, f * (y_face_k + 0.004), 0), 0.005, 0.005, trim_m, (0, 1, 0), 7100, False, True, FULL_ONLY, "lock", "Privacy button"))
        elif op.kind == "wheel":
            key, mesh = MESH.wheel_mesh(diameter=sp.get("diameter", 0.4), spokes=sp.get("spokes", 5), bar_diameter=sp.get("bar_diameter", 0.022), hub_len=0.08)
            body.geoms.append(mesh_geom(f"{name}_wheel_{'p' if f > 0 else 'n'}", key, mesh, (0, f * t / 2, 0), q, mat, 3000, False, ALL_TIERS, "operator", "Handwheel"))
            D = sp.get("diameter", 0.4)
            # collision: ring of small capsules
            for k in range(8):
                a = 2 * math.pi * k / 8
                body.geoms.append(sphere(f"{name}_wheel_col_{k}_{'p' if f > 0 else 'n'}", (D / 2 * math.cos(a), f * (t / 2 + 0.08), D / 2 * math.sin(a)), sp.get("bar_diameter", 0.022) / 2 + 0.004, mat, 3000, True, ALL_TIERS, "operator", "Wheel rim"))
            grip_sites.append(Site(f"{name}_grip_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + 0.08), D / 2), QUAT_ID, 0.012, "grip"))
        elif op.kind == "t_handle":
            # T-handle: bar centred on the spindle (both arms), keyed cylinder in the hub
            Lt = sp.get("length", 0.11)
            key, mesh = MESH.lever_mesh(shape="T", length=Lt, diameter=sp.get("diameter", 0.016), rose_diameter=0.05, standoff=0.04)
            body.geoms.append(mesh_geom(f"{name}_t_{'p' if f > 0 else 'n'}", key, mesh, (0, f * t / 2, 0), q, mat, 3000, False, ALL_TIERS, "operator", "T-handle"))
            body.geoms.append(box(f"{name}_t_col_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + 0.04), 0), (Lt / 2, 0.008, 0.008), mat, 3000, True, False, ALL_TIERS, "operator", "T grip"))
            body.geoms.append(cyl(f"{name}_cylinder_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + 0.04 + 0.008 + 0.002), 0), 0.0085, 0.004, trim_m, (0, 1, 0), 7100, False, True, FULL_ONLY, "lock", "Key cylinder"))
            grip_sites.append(Site(f"{name}_grip_{'p' if f > 0 else 'n'}", (-u * Lt * 0.35, f * (t / 2 + 0.04), 0), QUAT_ID, 0.012, "grip"))
        elif op.kind == "cremone":
            key, mesh = MESH.knob_mesh(shape="round", diameter=0.045, depth=0.05, rose_diameter=0.05)
            body.geoms.append(mesh_geom(f"{name}_cremone_knob_{'p' if f > 0 else 'n'}", key, mesh, (0, f * t / 2, 0), q, mat, 3000, False, ALL_TIERS, "operator", "Cremone knob"))
            body.geoms.append(sphere(f"{name}_cremone_col_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + 0.035), 0), 0.022, mat, 3000, True, ALL_TIERS, "operator", "Knob grip"))
            grip_sites.append(Site(f"{name}_grip_{'p' if f > 0 else 'n'}", (0, f * (t / 2 + 0.035), 0), QUAT_ID, 0.012, "grip"))
    # the spindle: long enough to reach 8 mm INTO the innermost part of the trim on every face it serves, so the
    # two faces' trim and the spindle are one connected part rather than three islands floating around the slab
    half_sp = max(t / 2 - 0.001, 0.004)
    for f in faces:
        inners = []
        for g in body.geoms[spindle_i:]:
            lo_, hi_ = geom_local_aabb(g)
            inner = lo_[1] if f > 0 else -hi_[1]
            if inner > 0:
                inners.append(inner)
        if inners:
            half_sp = max(half_sp, min(min(inners) + 0.008, 0.075))
    if spindle:
        body.geoms.insert(spindle_i, cyl(f"{name}_spindle", (0, 0, 0), 0.006, half_sp, mat, (0, 1, 0), 7850, False, True, FULL_ONLY, "mechanism", "Spindle"))
    body.sites += grip_sites
    if sp.get("reader"):
        # hotel RFID reader: part of the OUTSIDE trim, above the lever on the outside face
        rm = mat_from_material(model, "black_matte_metal", "mat_reader")
        key, mesh = MESH.card_reader_mesh(w=sp["escutcheon"][1] if sp.get("escutcheon") else 0.075, h=sp["escutcheon"][0] if sp.get("escutcheon") else 0.26)
        leaf_body.geoms.append(mesh_geom(f"{name}_reader", key, mesh, (x_spindle, outside * t / 2, z + 0.16), q_face_upright(outside), rm, 2000, False, FULL_SIMPLE, "lock", "Card reader"))
    model.add_body(body)
    return body


def keypad_button_layout(kp: H.KeypadModel):
    """[(label, dx, dz)] button centres relative to the keypad centre: dx to the RIGHT of somebody standing in
    front of the keypad (so the keys read left-to-right, top-to-bottom, like the real unit), dz up."""
    out = []
    n = len(kp.labels)
    if kp.layout == "2x5":                       # Schlage FE595 / BE365: 10 keys, five rows of two
        for i, lab in enumerate(kp.labels):
            r, c = divmod(i, 2)
            out.append((lab, (c - 0.5) * kp.pitch[0], ((n / 2 - 1) / 2 - r) * kp.pitch[1]))
    elif kp.layout == "3x4":                     # phone keypad
        for i, lab in enumerate(kp.labels):
            r, c = divmod(i, 3)
            out.append((lab, (c - 1) * kp.pitch[0], ((n / 3 - 1) / 2 - r) * kp.pitch[1]))
    else:                                        # column_5: Simplex 1000, one column, staggered left/right
        for i, lab in enumerate(kp.labels):
            out.append((lab, (kp.pitch[0] if i % 2 else -kp.pitch[0]), ((n - 1) / 2 - i) * kp.pitch[1]))
    return out


def add_keypad(model: Model, leaf_body: Body, spec: dict, u: float, x_spindle: float, z: float, t: float, face: float, kp: H.KeypadModel, name="keypad", z_min: float | None = None, z_max: float | None = None):
    """Keypad unit on `face`, above the handle: the housing plus one pressable body per button (full tier).

    `kp` (hardware.KEYPADS) fixes the layout, the stroke and the spring, so an electronic keypad (Schlage FE595 /
    BE365: 10 keys in five rows of two, 1.5 mm at 3 N) and a mechanical pushbutton lock (Kaba Simplex 1000: five
    stiff 4 mm buttons in a staggered column, + a key override cylinder) are both built from the same code.
    Returns {"z", "x", "pad", "buttons"} for the model's `keypad` meta block."""
    mechanical = kp.code_kind == "set"
    km = mat_from_material(model, "aluminum_dark" if mechanical else "black_matte_metal", "mat_keypad")
    backset_ = H.LATCHES[spec["latch"]["model"]].backset or 0.06
    kw, kh, kd = kp.pad
    zk = z + 0.12
    if z_min is not None:
        zk = max(zk, z_min + kh / 2 + 0.008)
    if z_max is not None:
        zk = min(zk, z_max - kh / 2)
    x_k = x_spindle - u * max(0.0, kw / 2 + 0.015 - backset_)
    if mechanical:
        zk += 0.02                                   # taller unit: keep its bottom clear of the lever rose
    key, mesh = MESH.keypad_body_mesh(w=kw, h=kh, keys=len(kp.labels), depth=kd)
    leaf_body.geoms.append(mesh_geom(f"{name}_body", key, mesh, (x_k, face * t / 2, zk), q_face_upright(face), km, 2000, True, FULL_SIMPLE, "lock", "Mechanical pushbutton lock body" if mechanical else "Keypad body"))
    model.meta.setdefault("clearance_allow", []).append([f"{name}_body", f"{name}_key_*", "buttons travel into the keypad body"])
    buttons = add_keypad_buttons(model, leaf_body, u, x_k, zk, t, face, kp, name=name)
    if mechanical:
        leaf_body.geoms.append(cyl(f"{name}_cylinder", (x_k, face * (t / 2 + kd + 0.002), zk - kh / 2 + 0.02), 0.009, 0.002, mat_from_material(model, "brass", "mat_trim"), (0, face, 0), 7100, False, True, FULL_ONLY, "lock", "Key override cylinder"))
    return {"z": zk, "x": x_k, "pad": [kw, kh, kd], "buttons": buttons}


def add_keypad_buttons(model: Model, leaf_body: Body, u: float, x_center: float, z_center: float, t: float, face: float, kp: H.KeypadModel, name="keypad"):
    """One pressable body per key: a slide joint into the door face with a return spring, sized from the keypad
    model (`preload_force` to break it away, `press_force` bottomed out over `travel`), and a `press` site on the
    button face for a fingertip.  Returns the button descriptors for the `keypad` meta block."""
    bm = mat_rgba(model, "mat_key", (0.85, 0.85, 0.82, 1), 0.5)
    k = (kp.press_force - kp.preload_force) / kp.travel            # N/m
    springref = -kp.preload_force / k                              # rest offset: F(0) = preload
    # Damped to 0.7 of critical for the DOF's reflected inertia (build.py's 0.1 kg armature floor on a lock slide):
    # a real keypad button is guided and dome-damped and does not bounce, and a bouncing button would read as two
    # presses.  tau = c/k is ~10 ms, so the button is back out well before the next digit.
    c_crit = 2.0 * math.sqrt(k * 0.1)
    kd = kp.pad[2]
    out = []
    for lab, dx, dz in keypad_button_layout(kp):
        safe = {"*": "star", "#": "hash"}.get(lab, lab)
        # somebody facing `face` has +x on their right when face = -1 and -x when face = +1
        px, pz = x_center - face * dx, z_center + dz
        b = Body(f"{name}_key_{safe}", leaf_body.name, (px, face * (t / 2 + kd), pz), QUAT_ID, None, [], [], FULL_ONLY, "lock", f"Button {lab}")
        b.joint = Joint(f"{name}_key_{safe}_slide", "slide", (0, -face, 0), (0, 0, 0), (0.0, kp.travel), damping=round(0.7 * c_crit, 3), frictionloss=0.05,
                        stiffness=k, springref=springref, armature=1e-6, role="lock",
                        label=f"Button {lab} (press {kp.press_force:.0f} N, {kp.travel * 1000:.1f} mm)")
        if kp.round_keys:
            b.geoms.append(cyl(f"{name}_key_{safe}_geom", (0, face * kp.proud / 2, 0), kp.key_size[0], kp.proud / 2, bm, (0, face, 0), 1200, True, True, FULL_ONLY, "lock", f"Button {lab}"))
        else:
            b.geoms.append(box(f"{name}_key_{safe}_geom", (0, face * kp.proud / 2, 0), (kp.key_size[0], kp.proud / 2, kp.key_size[1]), bm, 1200, True, True, FULL_ONLY, "lock", f"Button {lab}"))
        b.sites.append(Site(f"{name}_key_{safe}_press", (0, face * (kp.proud + 0.002), 0), QUAT_ID, 0.008, "press"))
        model.add_body(b)
        out.append({"label": lab, "body": b.name, "joint": b.joint.name, "site": f"{name}_key_{safe}_press",
                    "pos": [round(float(px), 5), round(float(face * (t / 2 + kd)), 5), round(float(pz), 5)]})
    return out


def keypad_meta_block(spec: dict, phys: dict, lk: H.LockModel, kp: H.KeypadModel, built: dict, face: float, engaged: bool,
                      clutch_joint: str | None, clutch_travel: float, bolt_joint: str | None, model: Model) -> dict:
    """``model.json -> meta.keypad``: everything a simulator, a QA gate, a policy or the viewer needs to work the
    keypad - where the buttons are, how hard and how far they press, what the code is (the dataset is open) and
    what the lock does when it is entered.

    release
      ``clutch``      the code frees the outside trim (`clutch_joint`), which then retracts the latch
      ``motor_bolt``  the code runs the motor that retracts the deadbolt (`bolt_joint`)
      ``none``        the lock is not thrown; the code is checked but there is nothing to release
    """
    code = spec["lock"].get("code")
    bolt = None
    if bolt_joint and any(b.joint is not None and b.joint.name == bolt_joint for b in model.bodies):
        bolt = bolt_joint
    release = "clutch" if (engaged and clutch_joint) else ("motor_bolt" if (engaged and bolt) else "none")
    out = {
        "lock_model": lk.id, "keypad_model": kp.id, "code": code, "code_kind": kp.code_kind, "engaged": bool(engaged),
        "face": float(face), "center": [round(float(built["x"]), 5), round(float(face * spec["leaf"]["thickness"] / 2), 5), round(float(built["z"]), 5)],
        "pad_size_m": [round(float(x), 4) for x in built["pad"]], "layout": kp.layout,
        "buttons": built["buttons"],
        "travel_m": kp.travel, "press_force_N": kp.press_force, "preload_force_N": kp.preload_force,
        "press_depth_frac": 0.6, "release_depth_frac": 0.3, "debounce_s": 0.02,
        "code_timeout_s": kp.timeout_s, "lockout_s": kp.lockout_s, "max_attempts": kp.max_attempts,
        "release": release, "clutch_joint": clutch_joint, "bolt_joint": bolt,
        "clutch_locked_rad": round(float(phys["lock"]["handle_backlash_locked_rad"]), 5) if clutch_joint else None,
        "clutch_open_rad": round(float(clutch_travel), 5) if clutch_joint else None,
        "bolt_throw_m": float(lk.deadbolt_throw) if bolt else None,
        "motor_force_N": 60.0 if bolt else None,
        "source": kp.source,
    }
    if kp.code_kind == "set":
        out["note"] = "mechanical combination chamber: press the buttons of the code in any order, then turn the outside lever; turning it on a wrong set clears the chamber and counts as a wrong attempt."
    else:
        out["note"] = f"electronic keypad: the digits in order; a partial entry clears after {kp.timeout_s:.0f} s, and {kp.max_attempts} wrong codes lock the keypad out for {kp.lockout_s:.0f} s."
    return out


def add_pull(model: Model, leaf_body: Body, op: H.OperatorModel, u: float, x: float, z: float, t: float, face: float, name="pull", tiers=ALL_TIERS):
    sp = op.style_params
    mat = mat_from_material(model, op.material, f"mat_op_{op.material}")
    shape = sp.get("shape", "d_pull")
    q = q_face(face, u)
    if shape in ("d_pull", "offset_bar", "ladder", "flat_bar", "lift_handle"):
        L = sp.get("length", 0.2)
        so = sp.get("standoff", 0.06)
        key, mesh = MESH.pull_mesh(shape=shape, length=L, diameter=sp.get("diameter", 0.019), standoff=so, width=sp.get("width", 0.032))
        zc = z if shape != "ladder" else max(L / 2 + 0.05, z)
        leaf_body.geoms.append(mesh_geom(f"{name}_{'p' if face > 0 else 'n'}", key, mesh, (x, face * t / 2, zc), q, mat, 3000, False, ALL_TIERS, "operator", op.name))
        if shape == "lift_handle":
            leaf_body.geoms.append(box(f"{name}_col_{'p' if face > 0 else 'n'}", (x, face * (t / 2 + so), zc), (L / 2, 0.006, sp.get("width", 0.03) / 2), mat, 3000, True, False, tiers, "operator", "Pull grip"))
        else:
            leaf_body.geoms.append(Geom(f"{name}_col_{'p' if face > 0 else 'n'}", "capsule", (sp.get("diameter", 0.019) / 2 if shape != "flat_bar" else 0.006, L / 2 - 0.02), (x, face * (t / 2 + so), zc), (1, 0, 0, 0), mat, True, False, 3000, None, (0.7, 0.01, 0.0001), None, None, False, None, None, 0.0, tiers, "operator", "Pull grip"))
        leaf_body.sites.append(Site(f"{name}_grip_{'p' if face > 0 else 'n'}", (x, face * (t / 2 + so), zc), QUAT_ID, 0.012, "grip"))
    elif shape == "ring":
        key, mesh = MESH.ring_mesh(ring_diameter=sp.get("ring_diameter", 0.12), bar_diameter=sp.get("bar_diameter", 0.014))
        leaf_body.geoms.append(mesh_geom(f"{name}_{'p' if face > 0 else 'n'}", key, mesh, (x, face * t / 2, z), q, mat, 3000, False, ALL_TIERS, "operator", "Ring pull"))
        leaf_body.geoms.append(Geom(f"{name}_col_{'p' if face > 0 else 'n'}", "capsule", (0.008, 0.05), (x, face * (t / 2 + 0.032), z - 0.06), tuple(quat_z_to((1, 0, 0))), mat, True, False, 3000, None, (0.7, 0.01, 0.0001), None, None, False, None, None, 0.0, tiers, "operator", "Ring grip"))
        leaf_body.sites.append(Site(f"{name}_grip_{'p' if face > 0 else 'n'}", (x, face * (t / 2 + 0.032), z - 0.06), QUAT_ID, 0.012, "grip"))
    elif shape in ("flush", "cup", "hikite"):
        w, h = sp.get("size", (0.05, 0.1)) if shape != "cup" else (sp.get("diameter", 0.055), sp.get("diameter", 0.055))
        dm = mat_from_material(model, op.material, f"mat_op_{op.material}")
        leaf_body.geoms.append(box(f"{name}_{'p' if face > 0 else 'n'}", (x, face * (t / 2 + 0.0005), z), (w / 2, 0.0005, h / 2), dm, 1000, False, True, ALL_TIERS, "operator", "Flush pull"))
        leaf_body.sites.append(Site(f"{name}_grip_{'p' if face > 0 else 'n'}", (x, face * (t / 2), z), QUAT_ID, 0.012, "grip"))
    elif shape == "plate":
        w, h = sp.get("size", (0.1, 0.4))
        leaf_body.geoms.append(box(f"{name}_{'p' if face > 0 else 'n'}", (x, face * (t / 2 + 0.0008), z), (w / 2, 0.0008, h / 2), mat, 7900, False, True, ALL_TIERS, "operator", "Push plate"))
        leaf_body.sites.append(Site(f"{name}_push_{'p' if face > 0 else 'n'}", (x, face * (t / 2), z), QUAT_ID, 0.012, "push"))
    elif shape == "handleset":
        key, mesh = MESH.handleset_mesh(grip_length=sp.get("grip_length", 0.24), plate=tuple(sp.get("plate", (0.3, 0.07))), thumb=False)
        leaf_body.geoms.append(mesh_geom(f"{name}_{'p' if face > 0 else 'n'}", key, mesh, (x, face * t / 2, z - 0.06), q_face_upright(face), mat, 3000, False, ALL_TIERS, "operator", "Handleset grip"))
        leaf_body.geoms.append(Geom(f"{name}_col_{'p' if face > 0 else 'n'}", "capsule", (0.011, 0.09), (x, face * (t / 2 + 0.06), z - 0.06), (1, 0, 0, 0), mat, True, False, 3000, None, (0.7, 0.01, 0.0001), None, None, False, None, None, 0.0, tiers, "operator", "Grip"))
        leaf_body.sites.append(Site(f"{name}_grip_{'p' if face > 0 else 'n'}", (x, face * (t / 2 + 0.06), z - 0.06), QUAT_ID, 0.012, "grip"))


def add_touchbar(model: Model, leaf_body: Body, spec: dict, op: H.OperatorModel, u: float, v: float, x_edge: float, x_hinge_edge: float, z: float, t: float, W: float, face: float, name="exit_device", tiers=ALL_TIERS, z_top: float | None = None, z_bot: float | None = None, case_end_gap: float = 0.03):
    """Exit device on the push face: fixed rail + moving pad (slide along door normal, + = pressed).
    case_end_gap: distance from the leaf edge to the end of the rim case (6 mm when the Pullman bolt lives in the
    case and the frame stop is cut for it; 30 mm otherwise / on pairs where the astragal covers the joint)."""
    sp = op.style_params
    mat = mat_from_material(model, op.material, f"mat_op_{op.material}")
    L = W * sp.get("bar_length_frac", 0.65)
    bh, bd = sp.get("bar_height", 0.05), sp.get("bar_depth", 0.065)
    xc = x_edge - u * (0.10 + L / 2)  # rail runs from near the latch edge toward the hinge
    if sp.get("shape") == "crossbar":
        # crossbar rotates about horizontal pivots at both ends (hinge along x)
        body = Body(name, leaf_body.name, (xc, face * (t / 2), z), QUAT_ID, None, [], [], tiers, "operator", op.name)
        body.joint = Joint(f"{name}_hinge", "hinge", (u, 0, 0), (0, 0, 0), (0.0, op.travel), damping=0.5, frictionloss=0.3, stiffness=op.spring_rate, springref=-op.spring_torque_preload / max(op.spring_rate, 1e-6), armature=1e-4, role="operator", label="Crossbar (+ = pushed in; spring return)", return_kind=op.return_kind, operator_model=op.id)
        key, mesh = MESH.crossbar_mesh(length=L, bar_diameter=sp.get("bar_diameter", 0.025), arm_length=sp.get("arm_length", 0.06))
        body.geoms.append(mesh_geom(f"{name}_mesh", key, mesh, (0, 0, 0), q_face(face, u), mat, 3000, False, ALL_TIERS, "operator", "Crossbar"))
        body.geoms.append(Geom(f"{name}_col", "capsule", (sp.get("bar_diameter", 0.025) / 2, L / 2), (0, face * sp.get("arm_length", 0.06), 0), tuple(quat_z_to((1, 0, 0))), mat, True, False, 3000, None, (0.7, 0.01, 0.0001), None, None, False, None, None, 0.0, tiers, "operator", "Crossbar grip"))
        body.sites.append(Site(f"{name}_push", (0, face * sp.get("arm_length", 0.06), 0), QUAT_ID, 0.015, "push"))
        model.add_body(body)
        # the crossbar's arms pivot in a latch case at the strike edge and an end case at the hinge edge (leaf-fixed)
        al = sp.get("arm_length", 0.06)
        bd_ = sp.get("bar_diameter", 0.025)
        x_lc = xc + u * (L / 2 + bd_ + 0.030)
        if abs(x_edge - x_lc) > 0.022 + 0.014:            # keep clear of the frame stop at the latch edge
            leaf_body.geoms.append(box(f"{name}_case", (x_lc, face * (t / 2 + al * 0.55), z), (0.022, al * 0.55, 0.045), mat, 3000, True, True, tiers, "latch", "Crossbar latch case"))
        x_ec = xc - u * (L / 2 + bd_ + 0.030)
        if abs(x_ec - x_hinge_edge) > 0.022 + 0.045:      # keep clear of the hinge-side stop
            leaf_body.geoms.append(box(f"{name}_end_case", (x_ec, face * (t / 2 + al * 0.5), z), (0.022, al * 0.5, 0.035), mat, 3000, True, True, tiers, "operator", "Crossbar end case"))
        return body, f"{name}_hinge"
    # rail fixed to leaf
    # housing = channel built from convex primitives (a channel-shaped mesh would collide as its convex hull)
    leaf_body.geoms.append(box(f"{name}_rail", (xc, face * (t / 2 + 0.006), z), (L / 2 + 0.02, 0.006, bh / 2), mat, 3000, True, True, tiers, "operator", "Exit device rail (back plate)"))
    for sx in (-1, 1):
        leaf_body.geoms.append(box(f"{name}_rail_end_{'p' if sx > 0 else 'n'}", (xc + sx * (L / 2 + 0.01), face * (t / 2 + bd / 2), z), (0.01, bd / 2, bh / 2), mat, 3000, True, True, tiers, "operator", "Rail end block"))
    if sp.get("rim_case", True):
        # rim device case: runs from the rail end toward the leaf edge; the Pullman latch lives inside it
        x_case0 = xc + u * (L / 2 + 0.02)
        x_case1 = x_edge - u * case_end_gap
        leaf_body.geoms.append(box(f"{name}_case", ((x_case0 + x_case1) / 2, face * (t / 2 + bd * 0.525), z), (abs(x_case1 - x_case0) / 2, bd * 0.525, bh * 0.75), mat, 3000, True, True, tiers, "latch", "Rim device case"))
    if sp.get("alarm"):
        am = mat_rgba(model, "mat_alarm", (0.75, 0.1, 0.1, 1), 0.5)
        leaf_body.geoms.append(box(f"{name}_alarm", (xc - u * (L / 2 + 0.045), face * (t / 2 + 0.02), z + 0.06), (0.03, 0.02, 0.025), am, 1500, False, True, FULL_SIMPLE, "lock", "Alarm / delayed-egress module"))
    if sp.get("vertical_rods"):
        rm = mat_from_material(model, op.material, f"mat_op_{op.material}")
        zt = (z_top if z_top is not None else spec["leaf"]["height"]) - 0.03
        zb_ = (z_bot if z_bot is not None else 0.0) + 0.03
        # the rods leave the device case (or the rail end block): they start ON it, not 20 mm above it
        has_case = bool(sp.get("rim_case", True))
        z_case = bh * (0.75 if has_case else 0.5)
        x_rod = (x_edge - u * 0.06) if has_case else (xc + u * (L / 2 + 0.01))
        a, b = z + z_case, zt
        if b - a > 0.02:
            leaf_body.geoms.append(cyl(f"{name}_rod_top", (x_rod, face * (t / 2 + 0.03), (a + b) / 2), 0.008, (b - a) / 2, rm, (0, 0, 1), 7850, False, True, FULL_SIMPLE, "latch", "Vertical rod (top)"))
        a2, b2 = zb_, z - z_case
        if b2 - a2 > 0.02:
            leaf_body.geoms.append(cyl(f"{name}_rod_bot", (x_rod, face * (t / 2 + 0.03), (a2 + b2) / 2), 0.008, (b2 - a2) / 2, rm, (0, 0, 1), 7850, False, True, FULL_SIMPLE, "latch", "Vertical rod (bottom)"))
    # pad body
    pad = Body(name, leaf_body.name, (xc, face * (t / 2), z), QUAT_ID, None, [], [], tiers, "operator", op.name)
    pad.joint = Joint(f"{name}_slide", "slide", (0, -face, 0), (0, 0, 0), (0.0, op.travel), damping=8.0, frictionloss=0.5, stiffness=op.spring_rate, springref=-op.spring_torque_preload / max(op.spring_rate, 1e-6), armature=1e-4, role="operator", label="Touch bar pad (+ = pressed; spring return)", return_kind=op.return_kind, operator_model=op.id)
    key, mesh = MESH.touchbar_pad_mesh(length=L, height=bh, depth=bd)
    pad.geoms.append(mesh_geom(f"{name}_pad", key, mesh, (0, 0, 0), q_face(face, u), mat, 3000, False, FULL_SIMPLE, "operator", "Touch pad"))
    pad.geoms.append(box(f"{name}_pad_col", (0, face * (bd - 0.015), 0), ((L - 0.006) / 2, 0.015, bh * 0.45), mat, 3000, True, False, tiers, "operator", "Touch pad", friction=(0.8, 0.01, 0.0001)))
    pad.sites.append(Site(f"{name}_push", (0, face * bd, 0), QUAT_ID, 0.015, "push"))
    model.add_body(pad)
    return pad, f"{name}_slide"


# ---------------------------------------------------------------------------
# Closer
# ---------------------------------------------------------------------------
def geom_local_aabb(g: Geom):
    """Axis-aligned bounding box (lo, hi) of a geom in its BODY frame, honouring its quaternion."""
    R = np.abs(quat_to_mat(np.asarray(g.quat, float)))
    if g.type == "box":
        h = np.asarray(g.size[:3], float)
    elif g.type in ("cylinder", "capsule"):
        r, hl = float(g.size[0]), float(g.size[1])
        h = np.array([r, r, hl + (r if g.type == "capsule" else 0.0)])
    elif g.type == "sphere":
        h = np.full(3, float(g.size[0]))
    elif g.type == "mesh" and g.mesh is not None:
        b = np.asarray(g.mesh.bounds, float)
        c = (b[0] + b[1]) / 2
        p = np.asarray(g.pos, float) + quat_to_mat(np.asarray(g.quat, float)) @ c
        h = R @ ((b[1] - b[0]) / 2)
        return p - h, p + h
    else:
        h = np.full(3, float(max(g.size)) if len(g.size) else 0.0)
    h = R @ h
    p = np.asarray(g.pos, float)
    return p - h, p + h


def face_proud(body: Body, v_face: float, half_t: float, x_lo: float, x_hi: float, z_lo: float, z_hi: float) -> float:
    """How far the parts of ``body`` stand proud of its slab face (local y = v_face * half_t) inside the local
    window (x_lo..x_hi, z_lo..z_hi).

    Battens, planks and applied mouldings put the real striking surface in front of the slab box; a stop authored
    against the slab face is then hit by the batten well before the leaf reaches its limit.  The window matters: a
    continuous hinge stands 17 mm proud of the same face, but it is at the other end of the leaf."""
    out = 0.0
    for g in body.geoms:
        lo, hi = geom_local_aabb(g)
        if hi[2] < z_lo or lo[2] > z_hi or hi[0] < x_lo or lo[0] > x_hi:
            continue
        y = hi[1] if v_face > 0 else -lo[1]
        out = max(out, float(y) - half_t)
    return max(0.0, out)


def mount_face(world: Body, x: float, z: float, hx: float, hz: float, v: float, default: float = 0.0,
               skip_semantics: tuple = ()) -> float:
    """Distance along +v from the door plane (y = 0) to the frontmost STATIC surface that covers the footprint
    (x +- hx, z +- hz).

    Anything bolted to the frame - a closer shoe, a stop, a keeper - has to land on the surface that is actually
    there at that spot, which is the head or jamb face on a plain frame but the casing / architrave face where trim
    stands proud of it, and neither is at +-wall_thickness/2 because the wall is offset from the door plane
    (meta["wall_y"]).  Only axis-aligned boxes are considered (every frame member is one)."""
    best = default
    for g in world.geoms:
        if g.semantic in skip_semantics:
            continue                               # the part's own family is not what carries it
        glo, ghi = geom_local_aabb(g)              # AABB: a gate post is a cylinder, a head a box
        px, py, pz = ((float(glo[k]) + float(ghi[k])) / 2 for k in range(3))
        sx, sy, sz = ((float(ghi[k]) - float(glo[k])) / 2 for k in range(3))
        if abs(px - x) > sx + hx or abs(pz - z) > sz + hz:
            continue
        best = max(best, v * py + sy)
    return best


# ---------------------------------------------------------------------------
# Opening stop hardware
# ---------------------------------------------------------------------------
STOP_TOUCH_GAP = 0.001     # m; the rubber tip is authored this far off the leaf face at the joint limit - the leaf
#                            arrives ON its stop (a millimetre is invisible and keeps a permanent contact out of the
#                            solver, which would otherwise sit in every settle / hold test)
STOP_WALL_REACH = 0.13     # m; the longest wall bumper made (the 4-4.5 in extended-projection type).  A leaf that stops
#                            further than this from the wall behind it cannot be held by a wall bumper at all - at
#                            90 deg the leaf stands perpendicular to its own wall and nothing on that wall can touch
#                            it - so the stop is built as the floor-mounted riser bumper that real doors use there.
STOP_TIP_R = 0.022         # m; rubber tip radius (44 mm dia, the common commercial bumper)
STOP_BASE_R = 0.022        # m; base plate radius; never larger than the tip, so the plate cannot reach past the
#                            leaf face plane and into the swept volume
STOP_TIP_Z = 0.075         # m; height of a floor riser bumper's centre (a 95 mm stop: 6 mm plate + post + tip)


def add_bumper_stop(model: Model, world: Body, leaf_body: Body, spec: dict, u: float, v: float, hx: float,
                    x0: float, W: float, t: float, Hh: float, zb: float = 0.0):
    """A rubber bumper stop that is actually mounted on something, and that the leaf actually strikes.

    The leaf's swing-side face at the maximum opening angle is the strike plane.  A ray from the strike point along
    that face's outward normal is cast at the static geometry: if it lands on a surface within ``STOP_WALL_REACH``
    the stop is the classic WALL bumper (base plate screwed flat to that surface, rubber tip projecting from it to
    the leaf).  If it does not - which is every door that stops at 90 deg, because there its leaf stands
    perpendicular to its own wall - the stop is a FLOOR riser bumper: base plate on the floor, post, rubber tip at
    bumper height.  Either way the tip's face ends ``STOP_TOUCH_GAP`` from the leaf at the joint limit and nothing
    hangs in the air.  The strike is recorded in ``meta["stops"]`` and re-verified by the attachment gate."""
    ang = math.radians(spec["kinematics"].get("max_open_deg") or 90)
    jp = leaf_body.joint.pos if leaf_body.joint is not None else (u * GAP, v * (t / 2 + LEAF_FACE_INSET), 0.0)
    wp = body_world_pos(model, leaf_body)
    r = W * 0.85                                   # along the leaf, near the leading edge (where a stop is fitted)
    phi = u * v * ang
    c_, s_ = math.cos(phi), math.sin(phi)
    nx, ny = -s_ * v, c_ * v                       # outward normal of that face at the maximum opening angle

    def strike(proud: float):
        """World (x, y) of the leaf's striking surface at the maximum opening angle, ``proud`` m in front of the slab."""
        rel = (x0 + u * r - jp[0], v * (t / 2 + proud) - jp[1])
        return (wp[0] + jp[0] + c_ * rel[0] - s_ * rel[1], wp[1] + jp[1] + s_ * rel[0] + c_ * rel[1])

    bm = mat_from_material(model, "rubber", "mat_bumper_stop")
    sm = mat_from_material(model, "chrome", "mat_stop_base")
    # --- can a wall bumper reach?  Only if the face normal runs into a static surface within a bumper's length.
    reach = None
    z_wall = max(STOP_TIP_Z, min(0.9 * Hh, 0.35))   # wall bumpers sit at door-rail height
    xs = x0 + u * r                                # local x of the strike point
    fx, fy = strike(face_proud(leaf_body, v, t / 2, xs - 0.06, xs + 0.06, z_wall - 0.03, z_wall + 0.03))
    if abs(ny) > 0.2:                              # a nearly parallel ray never lands on a wall in a useful place
        vd = 1.0 if ny > 0 else -1.0
        x_hit = fx + nx * (STOP_WALL_REACH * 0.6)
        d_front = mount_face(world, x_hit, z_wall, 0.03, 0.03, vd, default=-1e9)
        if d_front > -1e8:
            L = (vd * d_front - fy) / ny
            if STOP_TOUCH_GAP + 0.012 < L <= STOP_WALL_REACH:
                reach = L
    z_s = STOP_TIP_Z
    if reach is not None:
        # wall bumper: base plate flat on the wall, rubber tip projecting from it onto the leaf
        z_s = z_wall
        tip_h = (reach - STOP_TOUCH_GAP - 0.006) / 2
        world.geoms.append(cyl("door_stop_base", (fx + nx * (reach - 0.003), fy + ny * (reach - 0.003), z_s),
                               0.030, 0.003, sm, (nx, ny, 0), 7850, True, True, FULL_SIMPLE, "frame", "Wall bumper base plate"))
        world.geoms.append(cyl("door_stop_bumper", (fx + nx * (STOP_TOUCH_GAP + tip_h), fy + ny * (STOP_TOUCH_GAP + tip_h), z_s),
                               STOP_TIP_R, tip_h, bm, (nx, ny, 0), 1100, True, True, FULL_SIMPLE, "frame", "Wall bumper (rubber tip)"))
        mount = "wall"
    else:
        # floor riser bumper: base plate on the floor, post, rubber tip whose barrel the leaf face strikes
        # the riser is tall enough that its tip meets the leaf face even when the leaf is held off the ground (a gate)
        z_s = max(STOP_TIP_Z, zb + 0.06)
        fx, fy = strike(face_proud(leaf_body, v, t / 2, xs - 0.06, xs + 0.06, z_s - 0.02, z_s + 0.02))
        bx_, by_ = fx + nx * (STOP_TOUCH_GAP + STOP_TIP_R), fy + ny * (STOP_TOUCH_GAP + STOP_TIP_R)
        world.geoms.append(cyl("door_stop_base", (bx_, by_, 0.003), STOP_BASE_R, 0.003, sm, (0, 0, 1), 7850, True, True, FULL_SIMPLE, "frame", "Floor stop base plate"))
        world.geoms.append(cyl("door_stop_post", (bx_, by_, (0.006 + z_s - 0.020) / 2), 0.012, max(0.004, (z_s - 0.020 - 0.006) / 2), sm, (0, 0, 1), 7850, True, True, FULL_SIMPLE, "frame", "Floor stop riser"))
        world.geoms.append(cyl("door_stop_bumper", (bx_, by_, z_s), STOP_TIP_R, 0.020, bm, (0, 0, 1), 1100, True, True, FULL_SIMPLE, "frame", "Floor stop (rubber bumper)"))
        mount = "floor"
    model.meta.setdefault("stops", []).append({"geom": "door_stop_bumper", "joint": leaf_body.joint.name if leaf_body.joint else "",
                                               "mount": mount, "max_open_deg": spec["kinematics"].get("max_open_deg"),
                                               # the angle the stop is built for; the joint's own limit can be SHORTER
                                               # (an engaged door chain), and the stop is still correctly installed
                                               "q": round(ang, 5), "strike": [round(fx, 4), round(fy, 4), round(z_s, 4)]})
    model.meta.setdefault("notes", []).append(
        f"Opening stop: rubber bumper on a {mount} mount; the leaf face strikes its tip at the {spec['kinematics'].get('max_open_deg')} deg limit."
        + ("" if mount == "wall" else "  A wall bumper cannot reach a leaf that stops perpendicular to its own wall, so the floor riser real doors use there is modelled instead."))
    # The spec is a contract: it names the KIND of bumper stop, and this is the only place that decides the mount.
    # 149 doors used to be captioned `stop=wall_bumper` and build a floor riser (docs/VISION_REVIEW.md class 12);
    # a build-time assertion is what stops that coming back, because a spec that names the wrong kind now fails to
    # build rather than shipping a caption that contradicts the model.
    declared = spec["kinematics"].get("stop")
    if declared in ("wall_bumper", "floor_bumper"):
        want = "wall" if declared == "wall_bumper" else "floor"
        if mount != want:
            raise ValueError(f"{spec['id']}: spec declares stop '{declared}' ({want} mount) and the leaf's travel "
                             f"({spec['kinematics'].get('max_open_deg')} deg) only admits a {mount} mount")


def add_kick_down_holder(model: Model, leaf_body: Body, spec: dict, u: float, v: float, x0: float, z0: float,
                        W: float, t: float, name: str = "kickdown"):
    """Surface kick-down door holder, drawn retracted (the door is shipped closed).

    A kick-down holder is a cast housing on the leaf's LATCH stile carrying a pivoted arm with a rubber foot: kick
    the arm down and the foot bites the floor and holds the door open; kick it up and the door is free.  The latch
    stile is the part that leaves the frame first as the door opens, so a face-mounted holder there sweeps nothing.
    Both the `hold_open_kickdown` extra and ``kinematics["stop"] == "kick_down_holder"`` are this part."""
    if any(g.name.startswith(name + "_") for g in leaf_body.geoms):
        return                                     # the extra and the stop name the same holder; draw it once
    hm = mat_from_material(model, "brass", "mat_kickdown")
    xk = x0 + u * (W - 0.10)
    leaf_body.geoms.append(box(f"{name}_housing", (xk, -v * (t / 2 + 0.016), z0 + 0.16),
                               (0.022, 0.016, 0.045), hm, 8500, False, True, FULL_ONLY,
                               "decor", "Kick-down holder housing"))
    leaf_body.geoms.append(box(f"{name}_arm", (xk, -v * (t / 2 + 0.012), z0 + 0.095),
                               (0.012, 0.006, 0.045), hm, 8500, False, True, FULL_ONLY,
                               "decor", "Kick-down holder arm (retracted)"))
    leaf_body.geoms.append(cyl(f"{name}_pad", (xk, -v * (t / 2 + 0.012), z0 + 0.052), 0.014, 0.006,
                               mat_from_material(model, "pvc", "mat_kickdown_pad"), (0, 0, 1), 1200,
                               False, True, FULL_ONLY, "decor", "Kick-down rubber foot"))
    model.meta.setdefault("notes", []).append(
        "Hold-open: surface kick-down holder on the latch stile, drawn retracted (the door ships closed); kicking "
        "the arm down puts its rubber foot on the floor and holds the leaf where it stands.")


def add_prop_arm(model: Model, world: Body, lid: Body, spec: dict, x_a: float, y_pivot: float, y_dir: float,
                 L: float, z_face: float, face: float, socket_world: tuple, mat_id: str = "steel_galvanized"):
    """The folding prop arm that holds a hatch lid open, and the socket on the curb that it stands in.

    ``kinematics["stop"] == "prop_arm"`` used to draw nothing at all: the lid stood 90 deg open held by nothing
    (docs/VISION_REVIEW.md class 7).  The arm is a real body hinged on the lid, lying folded against the lid's
    outer face in the shipped (closed) pose and swinging out to stand in the curb socket as the lid rises; the
    socket is a jawed bracket on the curb whose lip the arm's foot drops behind.  Coordinates are in the LID's own
    frame; ``socket_world`` is the (x, y, z) of the socket on the curb."""
    am = mat_from_material(model, mat_id, "mat_prop")
    z_arm = z_face + face * 0.008
    arm = Body("prop_arm", lid.name, (x_a, y_pivot, z_arm), QUAT_ID, None, [], [], FULL_SIMPLE, "mechanism", "Hatch prop arm")
    arm.joint = Joint("prop_arm_hinge", "hinge", (1, 0, 0), (0, 0, 0), (0.0, 1.4), damping=0.4, frictionloss=0.8,
                      role="mechanism", return_kind="gravity", robot_interactive=True,
                      label="Prop arm (0 = folded on the lid, + = swung out to stand in the curb socket)")
    arm.geoms.append(box("prop_arm_bar", (0, y_dir * L / 2, 0), (0.010, L / 2, 0.005), am, 7850,
                         True, True, FULL_SIMPLE, "mechanism", "Prop arm bar"))
    arm.geoms.append(box("prop_arm_foot", (0, y_dir * (L - 0.010), face * 0.009), (0.014, 0.012, 0.009),
                         am, 7850, True, True, FULL_SIMPLE, "mechanism", "Prop arm foot"))
    arm.sites.append(Site("prop_arm_grip", (0, y_dir * (L - 0.04), face * 0.012), QUAT_ID, 0.012, "grip"))
    model.add_body(arm)
    lid.geoms.append(cyl("prop_arm_knuckle", (x_a, y_pivot, z_arm), 0.009, 0.016, am, (1, 0, 0), 7850, False, True,
                         FULL_SIMPLE, "mechanism", "Prop arm knuckle"))
    # the clip that holds the arm folded: a post off the lid face beside the bar, with a lip over it
    y_clip = y_pivot + y_dir * (L - 0.02)
    lid.geoms.append(box("prop_arm_clip_post", (x_a + 0.024, y_clip, z_face + face * 0.009),
                         (0.006, 0.010, 0.009), am, 7850, False, True, FULL_SIMPLE, "mechanism",
                         "Prop arm clip post"))
    lid.geoms.append(box("prop_arm_clip", (x_a + 0.010, y_clip, z_face + face * 0.0175),
                         (0.020, 0.010, 0.0025), am, 7850, False, True, FULL_SIMPLE, "mechanism",
                         "Prop arm retaining clip (holds the arm folded)"))
    # the socket the arm stands in, on the curb
    sx, sy, sz = socket_world
    for k_, dx_ in enumerate((-0.020, 0.020)):
        world.geoms.append(box(f"prop_arm_socket_{k_}", (sx + dx_, sy, sz + 0.014), (0.005, 0.018, 0.014), am, 7850,
                               True, True, FULL_SIMPLE, "frame", "Prop arm socket jaw"))
    world.geoms.append(box("prop_arm_socket_base", (sx, sy, sz + 0.004), (0.025, 0.018, 0.004), am, 7850, True, True,
                           FULL_SIMPLE, "frame", "Prop arm socket base"))
    model.meta.setdefault("clearance_allow", []).extend([
        ["prop_arm_bar", "prop_arm_knuckle", "the arm turns in its own knuckle"],
        ["prop_arm_bar", "prop_arm_clip*", "the folded arm is held by its clip"],
        ["prop_arm_foot", "prop_arm_clip*", "the folded arm's foot sits under its clip"],
    ])
    model.contact_excludes.append((arm.name, lid.name))
    model.meta.setdefault("notes", []).append(
        "Hold-open: folding prop arm hinged on the lid with a socket on the curb.  It is drawn FOLDED, which is "
        "where it lies on a shut hatch; swinging it out (prop_arm_hinge) stands it in the socket and it carries "
        "the lid.  The stand-and-catch itself is left to the environment, as the shipped pose is the closed one.")
    return arm


def add_hook_holdback(model: Model, world: Body, leaf_body: Body, spec: dict, u: float, v: float,
                      x0: float, W: float, t: float, Hh: float, zb: float = 0.0):
    """Hook-and-eye holdback: a deck stanchion carrying a hook, and a pad-eye on the leaf that it catches.

    Ten watertight doors were captioned ``stop=hook_holdback`` with no holdback modelled at all: at full open the
    leaf was held by nothing (docs/VISION_REVIEW.md class 7).  The stanchion stands where the leaf's own face
    arrives at its opening limit - the same strike computation the floor riser bumper uses - and the hook hangs
    from its head on a gravity pivot, so it drops over the leaf's pad-eye when the door is swung right open."""
    ang = math.radians(spec["kinematics"].get("max_open_deg") or 90)
    jp = leaf_body.joint.pos if leaf_body.joint is not None else (u * GAP, v * (t / 2 + LEAF_FACE_INSET), 0.0)
    wp = body_world_pos(model, leaf_body)
    r = W * 0.85
    phi = u * v * ang
    c_, s_ = math.cos(phi), math.sin(phi)
    nx, ny = -s_ * v, c_ * v
    # below the lowest dog lever (they start 0.2 m up the leaf): the leaf's face is plain slab down here, and the
    # stanchion is not swept by hardware standing proud of it
    z_h = max(0.30, wp[2] + 0.15)                    # world z of the stanchion head
    z_local = z_h - wp[2]                            # ... and the same height in the leaf's own frame
    xs = x0 + u * r
    proud = face_proud(leaf_body, v, t / 2, xs - 0.06, xs + 0.06, z_local - 0.05, z_local + 0.05)
    rel = (xs - jp[0], v * (t / 2 + proud) - jp[1])
    fx = wp[0] + jp[0] + c_ * rel[0] - s_ * rel[1]
    fy = wp[1] + jp[1] + s_ * rel[0] + c_ * rel[1]
    mm = mat_from_material(model, "steel_galvanized", "mat_holdback")
    d_hook = 0.060                                   # m from the leaf face to the hook's own axis at full open
    d_post = d_hook + 0.040                          # the stanchion stands clear behind it
    hx_, hy_ = fx + nx * d_hook, fy + ny * d_hook
    bx, by = fx + nx * d_post, fy + ny * d_post
    world.geoms.append(cyl("holdback_stop_base", (bx, by, 0.004), 0.045, 0.004, mm, (0, 0, 1), 7850, True, True,
                           FULL_SIMPLE, "frame", "Holdback stanchion base plate"))
    world.geoms.append(cyl("holdback_stop_post", (bx, by, (0.008 + z_h) / 2), 0.016, (z_h - 0.008) / 2, mm, (0, 0, 1),
                           7850, True, True, FULL_SIMPLE, "frame", "Holdback stanchion"))
    world.geoms.append(box("holdback_stop_head", ((bx + hx_) / 2, (by + hy_) / 2, z_h), (0.026, 0.010, 0.008), mm,
                           7850, True, True, FULL_SIMPLE, "frame", "Stanchion head (carries the hook pivot)",
                           quat=tuple(quat_from_axis_angle([0, 0, 1], math.atan2(hy_ - by, hx_ - bx)))))
    # the hook itself: hangs from the head on a gravity pivot, swinging in the plane of the leaf face
    hook = Body("holdback_hook", None, (hx_, hy_, z_h - 0.010), QUAT_ID, None, [], [], FULL_SIMPLE, "mechanism", "Holdback hook")
    hook.joint = Joint("holdback_hook_hinge", "hinge", (nx, ny, 0), (0, 0, 0), (-1.2, 1.2), damping=0.05,
                       frictionloss=0.05, role="mechanism", return_kind="gravity", robot_interactive=True,
                       label="Holdback hook (hangs on its pivot; drops over the leaf's pad-eye when the door is right open)")
    tx, ty = -nx, -ny                                   # toward the leaf face
    hook.geoms.append(cyl("holdback_hook_shank", (0, 0, -0.030), 0.005, 0.030, mm, (0, 0, 1),
                          7850, True, True, FULL_SIMPLE, "mechanism", "Hook shank"))
    hook.geoms.append(cyl("holdback_hook_bill", (tx * 0.014, ty * 0.014, -0.058), 0.005, 0.014, mm, (tx, ty, 0),
                          7850, True, True, FULL_SIMPLE, "mechanism", "Hook bill"))
    model.add_body(hook)
    model.meta.setdefault("clearance_allow", []).append(
        ["holdback_hook*", "holdback_stop_head", "the hook hangs on the head it pivots in"])
    # the pad-eye the hook drops into, on the leaf face at the same height: a plate, two lugs and the pin across them
    z_eye = z_local - 0.058                          # leaf-local: the pad-eye is a geom of the leaf
    leaf_body.geoms.append(box("holdback_eye_keeper_pad", (xs, -v * (t / 2 + 0.0025), z_eye), (0.030, 0.0025, 0.022),
                               mm, 7850, False, True, FULL_SIMPLE, "mechanism", "Holdback pad-eye plate"))
    for sx_ in (-1, 1):
        leaf_body.geoms.append(box(f"holdback_eye_keeper_lug_{'p' if sx_ > 0 else 'n'}",
                                   (xs + sx_ * 0.014, -v * (t / 2 + 0.012), z_eye), (0.005, 0.0075, 0.014),
                                   mm, 7850, False, True, FULL_SIMPLE, "mechanism", "Pad-eye lug"))
    leaf_body.geoms.append(cyl("holdback_eye_keeper_pin", (xs, -v * (t / 2 + 0.016), z_eye), 0.004, 0.014, mm,
                               (1, 0, 0), 7850, False, True, FULL_SIMPLE, "mechanism", "Pad-eye pin"))
    model.meta.setdefault("notes", []).append(
        "Hold-open: hook-and-eye holdback - a deck stanchion at the leaf's opening limit carrying a gravity-hung "
        "hook, and a pad-eye on the leaf face it drops over.")


def brace_pending(model: Model):
    """Bracket every part the builders parked in ``meta["_brace_pending"]``.

    Some hardware is placed before the member it is screwed to exists (a gate builds its posts after its leaf), so
    the builder records the part and this runs once the world is complete."""
    world = next((b for b in model.bodies if b.static), None)
    if world is None:
        return
    done = set()
    for e in model.meta.pop("_brace_pending", []):
        if e["geom"] in done:
            continue                    # a pair of leaves parks the same name twice; brace each geom once
        done.add(e["geom"])
        for i, g in enumerate([x for x in list(world.geoms) if x.name == e["geom"]]):
            d0 = 1.0 if float(g.pos[1]) > 0 else -1.0      # the structure is usually toward the wall plane
            for d in ([float(e["d"])] if e.get("d") else (d0, -d0)):
                if brace_to_structure(world, g, d, g.material, name=f"{g.name}_strap_{i}", semantic=g.semantic,
                                      label=e.get("label", "Mounting bracket"), tiers=FULL_SIMPLE, span=0.8,
                                      axes=tuple(e.get("axes", ("y", "x"))), pad=float(e.get("pad", 0.02)),
                                      reach=float(e.get("reach", 0.06))) is not None:
                    break


def mount_face_z(world: Body, x: float, y: float, hx: float, hy: float, z_from: float):
    """Z of the LOWEST static box face above ``z_from`` over the footprint (x +- hx, y +- hy), or None."""
    best = None
    for o in world.geoms:
        if o.type != "box" or abs(float(o.quat[0]) - 1.0) > 1e-9:
            continue
        px, py, pz = (float(c) for c in o.pos)
        sx, sy, sz = (float(q) for q in o.size[:3])
        if abs(px - x) > sx + hx or abs(py - y) > sy + hy:
            continue
        lo = pz - sz
        if lo > z_from + 0.002 and (best is None or lo < best):
            best = lo
    return best


def brace_to_structure(world: Body, g: Geom, d: float, mat, name: str | None = None, semantic: str = "frame",
                       label: str = "Mounting bracket", tiers=FULL_SIMPLE, span: float = 0.6, max_gap: float = 0.25,
                       axes: tuple = ("y", "x", "z"), pad: float = 0.02, reach: float = 0.06):
    """Bracket the static geom ``g`` back to the structure it stands off, in the ``d`` (+-1) sense of each axis tried.

    Surface-mounted hardware - a keeper on a barn-door jamb, an EXIT sign over the head, a drop-bolt housing on a
    gate post - is drawn where the door needs it, which is some way off the member it is really screwed to.  The
    nearest static surface behind it is found (over the part's own footprint) and the gap is filled with the
    bracket that would be there.  Axes are tried in order, so a part with a wall behind it gets a wall standoff and
    one with only a jamb beside it gets a jamb bracket.  Returns the bracket geom, or None."""
    lo, hi = geom_local_aabb(g)
    half = [float(hi[k] - lo[k]) / 2 for k in range(3)]
    ctr = [float(hi[k] + lo[k]) / 2 for k in range(3)]
    for ax in axes:
        a = {"x": 0, "y": 1, "z": 2}[ax]
        o1, o2 = [k for k in range(3) if k != a]
        s1, s2 = max(pad, half[o1] * span), max(pad, half[o2] * span)
        f_part = d * ctr[a] - half[a]                # the part's BACK face along +d (the bracket stops there)
        best, best_box = None, None
        for o in world.geoms:
            if o is g:
                continue
            olo, ohi = geom_local_aabb(o)          # AABB: a post is a cylinder, a keeper plate a box
            p_ = [(float(olo[k]) + float(ohi[k])) / 2 for k in range(3)]
            sz = [(float(ohi[k]) - float(olo[k])) / 2 for k in range(3)]
            if all(abs(p_[k] - ctr[k]) <= sz[k] + half[k] + 0.002 for k in range(3)):
                continue                           # already touching the part: its own island, not what carries it
            if abs(p_[o1] - ctr[o1]) > sz[o1] + s1 or abs(p_[o2] - ctr[o2]) > sz[o2] + s2:
                continue
            f = d * p_[a] + sz[a]
            if f <= f_part + 1e-9 and (best is None or f > best):
                best, best_box = f, (p_, sz)
        if best is None:
            continue
        gap = f_part - best
        if gap <= 0.002 or gap > max_gap:
            continue
        pos = list(ctr)
        pos[a] = d * (best + gap / 2)
        hsz = [max(0.005, half[k] * span) for k in range(3)]
        hsz[a] = gap / 2 + 0.0005
        # reach sideways where the support is offset from the part's own footprint (a gate stop that overhangs its
        # post): the bracket is an angle, not a plug
        for k in (o1, o2):
            plo, phi = pos[k] - hsz[k], pos[k] + hsz[k]
            slo, shi = best_box[0][k] - best_box[1][k], best_box[0][k] + best_box[1][k]
            if 0 < slo - phi <= reach:
                phi = slo + 0.004
            elif 0 < plo - shi <= reach:
                plo = shi - 0.004
            pos[k], hsz[k] = (plo + phi) / 2, (phi - plo) / 2
        br = box(name or f"{g.name}_bracket", tuple(pos), tuple(hsz), mat, 7850, False, True, tiers, semantic, label)
        world.geoms.append(br)
        return br
    return None


def add_closer(model: Model, world: Body, leaf_body: Body, spec: dict, phys: dict, u: float, v: float, x_hinge_axis: float, Hh: float, t: float, Wo: float, jamb_t: float, tier_full_arms=True):
    cl = H.CLOSERS[spec["closer"]["model"]]
    if cl.kind in ("none", "spring_hinge", "gate", "gas_strut", "pneumatic"):
        if cl.kind == "pneumatic":
            m = mat_from_material(model, "aluminum", "mat_closer")
            leaf_body.geoms.append(cyl("closer_pneumatic", (u * 0.25, v * (t / 2 + 0.03), Hh * 0.6), 0.015, 0.16, m, (1, 0, 0), 2700, False, True, FULL_SIMPLE, "closer", "Pneumatic closer"))
            for k_, dx_ in enumerate((-0.15, 0.15)):
                leaf_body.geoms.append(box(f"closer_pneumatic_bracket_{k_}", (u * (0.25 + dx_), v * (t / 2 + 0.016), Hh * 0.6), (0.008, 0.016, 0.018), m, 2700, False, True, FULL_SIMPLE, "closer", "Closer mounting bracket"))
        return
    m = mat_from_material(model, "aluminum_dark" if cl.kind != "floor_spring" else "stainless", "mat_closer")
    l, w, h = cl.body_size
    if cl.kind == "floor_spring":
        world.geoms.append(box("floor_spring_cover", (x_hinge_axis + u * 0.12, 0, 0.002), (0.17, 0.06, 0.002), m, 7900, False, True, FULL_SIMPLE, "closer", "Floor spring cover plate"))
        return
    if cl.kind == "concealed_overhead":
        world.geoms.append(box("closer_concealed_slot", (x_hinge_axis + u * 0.3, v * (t / 2 + 0.01), float(spec["opening"]["height"]) + 0.005), (0.25, 0.008, 0.004), m, 2700, False, True, FULL_ONLY, "closer", "Concealed closer track"))
        return
    if cl.kind in ("auto_operator_low_energy", "auto_operator_full"):
        # header-mounted operator box on the push face side of the frame
        world.geoms.append(box("auto_operator_header", (0, -v * (t / 2 + w / 2 + 0.01), Hh + jamb_t + h / 2 + 0.01), (min(l / 2, Wo / 2 + jamb_t), w / 2, h / 2), m, 1500, True, True, FULL_SIMPLE, "closer", "Automatic operator header"))
        # arm to the leaf (visual)
        leaf_body.geoms.append(box("auto_operator_arm", (u * 0.25, -v * (t / 2 + 0.03), Hh - 0.02), (0.22, 0.008, 0.008), m, 2700, False, True, FULL_ONLY, "closer", "Operator arm"))
        leaf_body.geoms.append(box("auto_operator_arm_shoe", (u * 0.44, -v * (t / 2 + 0.016), Hh - 0.02), (0.03, 0.016, 0.014), m, 2700, False, True, FULL_ONLY, "closer", "Operator arm door shoe"))
        return
    # surface closer: regular arm, body on pull face (+v side) near top, pinion at x_p from hinge
    x_p = u * 0.30
    y_body = v * (t / 2 + h / 2)
    zc = Hh - 0.06 - 0.02
    key, mesh = MESH.closer_body_mesh(l=l, w=w, h=h)
    # mesh frame: z away from face -> world v*y; length along x
    q = q_face(v, u)
    leaf_body.geoms.append(mesh_geom("closer_body", key, mesh, (x_p, v * t / 2, zc), q, m, 2000, False, FULL_SIMPLE, "closer", cl.name))
    leaf_body.geoms.append(box("closer_body_col", (x_p, v * (t / 2 + h / 2), zc), (l / 2, h / 2, w / 2), m, 2000, True, False, FULL_SIMPLE, "closer", "Closer body"))
    # arm kinematic loop (full tier)
    if not tier_full_arms:
        return
    wt_ = float(spec["opening"]["wall_thickness"])
    Ho_ = float(spec["opening"]["height"])
    # The frame head spans the whole wall depth, so its face on the closer side is where the shoe is screwed on.
    # The wall is offset from the leaf plane (meta["wall_y"]), so the face is NOT at +-wall_thickness/2 - reading it
    # off the origin instead put the shoe (and with it the pinion) up to 133 mm out in front of the frame.
    depth_ = wt_ if spec["opening"]["frame"]["kind"] != "aluminum_storefront" else max(0.114, wt_)
    pin_z = Ho_ + 0.03                                       # arm plane above the head: the leaf swings under it
    BRK_T = 0.012                                            # m; half thickness of the soffit shoe / mounting plate
    # the shoe is bolted to the surface that is really there above the opening (head face, or casing where trim
    # stands proud of it) - taking +-wall_thickness/2 off the origin put it up to 133 mm out in front of the frame
    face_v = max(t / 2 + 0.005, v * float(model.meta.get("wall_y", 0.0)) + depth_ / 2,
                 mount_face(world, x_hinge_axis + u * 0.10, pin_z, 0.030, 0.014, v, skip_semantics=("closer",)))
    y_face = v * face_v
    pin_y = v * max(t / 2 + h / 2, face_v + 0.035)           # pinion inside the closer body, clear of the head/casing face
    leaf_body.geoms.append(cyl("closer_pinion_shaft", (x_p, pin_y, (zc + pin_z) / 2), 0.008, (pin_z - zc) / 2, m, (0, 0, 1), 2700, False, True, FULL_ONLY, "closer", "Pinion shaft"))
    # shoe on the frame casing face above the opening, 10 cm from the hinge line (bolted flat to that face)
    # A cam-lift leaf raises both arms.  A fixed planar shoe would oppose that rise, so that variant uses a shoe
    # block sliding in a frame-mounted guide; its pivot then sits one plate thickness further out.
    rise_coupling = next((e for e in model.equalities if e.kind == "joint"
                          and e.b == leaf_body.joint.name and e.a.endswith("_rise")), None)
    x_b_rel = u * 0.10
    y_b = y_face + v * (0.018 if rise_coupling is not None else BRK_T)
    L1, L2 = 0.28, 0.26
    pfx = "" if leaf_body.name == "leaf" else leaf_body.name + "_"
    arm1 = Body(pfx + "closer_arm_main", leaf_body.name, (x_p, pin_y, pin_z), QUAT_ID, None, [], [], FULL_ONLY, "closer", "Closer main arm")
    arm1.joint = Joint(pfx + "closer_pinion", "hinge", (0, 0, 1), (0, 0, 0), None, damping=0.01, role="mechanism", label="Closer pinion", robot_interactive=False)
    # initial arm geometry at door closed: solve elbow position (2-link IK) in leaf frame
    px, py = x_p, pin_y
    bx, by = x_hinge_axis * 0 + x_b_rel, y_b   # bracket in leaf frame at q=0 (leaf frame coincides w/ world at hinge axis)
    d = math.hypot(bx - px, by - py)
    d = min(max(d, abs(L1 - L2) + 1e-3), L1 + L2 - 1e-3)
    a = math.atan2(by - py, bx - px)
    cosang = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)
    ang = math.acos(max(-1, min(1, cosang)))
    # choose elbow on the side away from the door (further out in +v y)
    e1 = (px + L1 * math.cos(a + ang), py + L1 * math.sin(a + ang))
    e2 = (px + L1 * math.cos(a - ang), py + L1 * math.sin(a - ang))
    ex, ey = e1 if v * e1[1] > v * e2[1] else e2
    th1 = math.atan2(ey - py, ex - px)
    arm1.quat = tuple(quat_from_axis_angle([0, 0, 1], th1))
    arm1.geoms.append(box("closer_arm_main_geom", (L1 / 2, 0, 0), (L1 / 2, 0.008, 0.005), m, 2700, False, True, FULL_ONLY, "closer", "Main arm"))
    model.add_body(arm1)
    arm2 = Body(pfx + "closer_arm_fore", arm1.name, (L1, 0, 0), QUAT_ID, None, [], [], FULL_ONLY, "closer", "Closer forearm")
    arm2.joint = Joint(pfx + "closer_elbow", "hinge", (0, 0, 1), (0, 0, 0), None, damping=0.01, role="mechanism", label="Closer elbow", robot_interactive=False)
    th2 = math.atan2(by - ey, bx - ex) - th1
    arm2.quat = tuple(quat_from_axis_angle([0, 0, 1], th2))
    arm2.geoms.append(box("closer_arm_fore_geom", ((L2 - 0.035) / 2, 0, 0), ((L2 - 0.035) / 2, 0.007, 0.004), m, 2700, False, True, FULL_ONLY, "closer", "Forearm"))
    model.add_body(arm2)
    # Its translation is passive: the connect constraint carries the shoe with the arm, without cancelling the
    # helical hinge's gravity load or inventing another closing spring.
    anchor_body = "world"
    if rise_coupling is not None:
        max_rise = max(0.0, rise_coupling.polycoeff[1] * leaf_body.joint.range[1])
        stroke = max_rise + 0.002  # 2 mm assembly/end-travel allowance beyond the cam rise
        anchor_body = pfx + "closer_shoe"
        shoe = Body(anchor_body, None, (x_hinge_axis + bx, by, pin_z), QUAT_ID, None, [], [], FULL_ONLY, "closer", "Vertically sliding closer shoe")
        shoe.joint = Joint(pfx + "closer_shoe_slide", "slide", (0, 0, 1), (0, 0, 0), (0.0, stroke),
                           damping=0.1, frictionloss=0.0, role="mechanism", robot_interactive=False,
                           label="Closer shoe lift (passive cam-lift accommodation)")
        shoe.geoms.append(box("closer_shoe_block", (0, 0, 0), (0.0195, 0.010, 0.008), m, 2700, False, True, FULL_ONLY, "closer", "Sliding shoe block"))
        # Continue the forearm to a visible pin; the ordinary fixed-shoe
        # arm drawing ends 35 mm short and would float beside this smaller shoe.
        arm2.geoms.append(box("closer_shoe_neck", (L2 - 0.0175, 0, 0), (0.0175, 0.003, 0.003), m, 2700, False, True, FULL_ONLY, "closer", "Forearm clevis neck"))
        arm2.geoms.append(cyl("closer_shoe_pivot", (L2, 0, 0), 0.005, 0.011, m, (0, 0, 1), 2700, False, True, FULL_ONLY, "closer", "Shoe pivot pin"))
        model.meta.setdefault("clearance_allow", []).extend([
            ["closer_shoe_pivot", "closer_shoe_block", "pivot pin occupies the sliding shoe's bore"],
            ["closer_shoe_neck", "closer_shoe_block", "forearm clevis enters the shoe around its pivot"],
        ])
        model.add_body(shoe)
        center_z = pin_z + stroke / 2
        # mounting plate flat on the frame face (its back face IS the frame face), shoe block riding on it
        world.geoms.append(box("closer_bracket", (x_hinge_axis + bx, y_face + v * 0.004, center_z),
                               (0.032, 0.004, stroke / 2 + 0.018), m, 2700, False, True, FULL_ONLY, "closer", "Slotted shoe mounting plate"))
        for side in (-1, 1):
            world.geoms.append(box(f"closer_shoe_guide_{side:+d}", (x_hinge_axis + bx + side * 0.026, y_face + v * 0.016, center_z),
                                   (0.006, 0.016, stroke / 2 + 0.018), m, 2700, False, True, FULL_ONLY, "closer", "Shoe guide (0.5 mm running clearance)"))
        model.meta.setdefault("notes", []).append("Cam-lift closer has a passive vertically sliding frame shoe; retaining lips/end caps are not modeled, and hydraulic torque remains the joint-level closer approximation.")
    else:
        # soffit shoe bolted flat to the frame face; the forearm's pivot pin drops into its bore
        world.geoms.append(box("closer_bracket", (x_hinge_axis + bx, y_face + v * BRK_T, pin_z), (0.030, BRK_T, 0.014), m, 2700, False, True, FULL_ONLY, "closer", "Closer soffit shoe"))
        arm2.geoms.append(box("closer_fore_neck", (L2 - 0.0175, 0, 0), (0.0175, 0.003, 0.003), m, 2700, False, True, FULL_ONLY, "closer", "Forearm clevis neck"))
        arm2.geoms.append(cyl("closer_fore_pivot", (L2, 0, 0), 0.005, 0.012, m, (0, 0, 1), 2700, False, True, FULL_ONLY, "closer", "Shoe pivot pin"))
        model.meta.setdefault("clearance_allow", []).extend([
            ["*closer_fore_pivot", "closer_bracket", "pivot pin occupies the soffit shoe's bore"],
            ["*closer_fore_neck", "closer_bracket", "forearm clevis enters the shoe around its pivot"],
            ["*closer_arm_fore_geom", "closer_bracket", "forearm reaches into the shoe it is pinned to"],
        ])
    model.equalities.append(Equality("connect", pfx + "closer_arm_connect", arm2.name, anchor_body, (0, 0, 0, 0, 0), (L2, 0, 0), FULL_ONLY, "Closer forearm pinned to frame shoe"))
    model.contact_excludes += [(arm1.name, leaf_body.name), (arm2.name, leaf_body.name), (arm1.name, arm2.name)]


# ---------------------------------------------------------------------------
# Extras / signage
# ---------------------------------------------------------------------------
def add_extras(model: Model, world: Body, leaf_body: Body, spec: dict, u: float, v: float, x0: float, z0: float, W: float, Hh: float, t: float, Wo: float, Ho: float):
    ex = set(spec.get("extras", []))
    if "kick_plate" in ex and not spec["leaf"].get("pet_flap"):
        for f in (-1, 1) if spec["family"] in ("swing_double", "hospital") else (-v,):
            add_kick_plate(model, leaf_body, u, x0, z0, W, t, f, name=f"kick_plate_{'p' if f > 0 else 'n'}")
    if "armor_plate" in ex:
        add_kick_plate(model, leaf_body, u, x0, z0, W, t, -v, name="armor_plate", height=0.9)
    if "bumper_rail" in ex:
        rm = mat_from_material(model, "pvc", "mat_bumper")
        for f in (-1, 1):
            leaf_body.geoms.append(box(f"bumper_rail_{'p' if f > 0 else 'n'}", (x0 + u * (W / 2 + 0.035), f * (t / 2 + 0.01), z0 + 0.65), (W / 2 - 0.085, 0.01, 0.04), rm, 1200, True, True, FULL_SIMPLE, "decor", "Bumper rail"))
    if "peephole" in ex or "door_viewer_camera" in ex:
        key, mesh = MESH.peephole_mesh()
        pm = mat_from_material(model, "brass", "mat_peephole")
        leaf_body.geoms.append(mesh_geom("peephole", key, mesh, (x0 + u * W / 2, -1.0 * t / 2, z0 + 1.5), q_face(-1.0, u), pm, 8500, False, FULL_ONLY, "decor", "Door viewer"))
    if "mail_slot" in ex:
        mm = mat_from_material(model, "brass", "mat_mailslot")
        pfx = "" if leaf_body.name == "leaf" else leaf_body.name + "_"
        flap = Body(pfx + "mail_slot_flap", leaf_body.name, (x0 + u * W / 2, -1.0 * (t / 2 + 0.0085), z0 + 0.95 + 0.035), QUAT_ID, None, [], [], FULL_ONLY, "decor", "Mail slot flap")
        flap.joint = Joint(pfx + "mail_slot_hinge", "hinge", (u, 0, 0), (0, 0, 0), (0.0, 1.2), damping=0.01, stiffness=0.3, springref=-0.3, role="decor", label="Mail slot flap", robot_interactive=False)
        flap.geoms.append(box("mail_slot_flap_geom", (0, 0, -0.035), (0.15, 0.002, 0.035), mm, 8500, True, True, FULL_ONLY, "decor", "Mail flap"))
        model.add_body(flap)
        leaf_body.geoms.append(box("mail_slot_frame", (x0 + u * W / 2, -1.0 * (t / 2 + 0.003), z0 + 0.95), (0.17, 0.003, 0.055), mm, 8500, False, True, FULL_ONLY, "decor", "Mail slot plate"))
    if "knocker" in ex and spec["leaf"].get("panel_style") not in ("plank_z_brace", "plank_x_brace", "board_batten"):
        key, mesh = MESH.knocker_mesh()
        km = mat_from_material(model, "brass_antique", "mat_knocker")
        pfx = "" if leaf_body.name == "leaf" else leaf_body.name + "_"
        kb = Body(pfx + "knocker", leaf_body.name, (x0 + u * W / 2, -1.0 * t / 2, z0 + 1.28), QUAT_ID, None, [], [], FULL_ONLY, "decor", "Door knocker")
        kb.joint = Joint(pfx + "knocker_hinge", "hinge", (u, 0, 0), (0, 0, 0.055), (0.0, 0.6), damping=0.005, role="decor", label="Knocker ring", robot_interactive=True)
        kb.geoms.append(mesh_geom("knocker_mesh", key, mesh, (0, 0, 0), q_face(-1.0, u), km, 8500, False, FULL_ONLY, "decor", "Knocker"))
        kb.geoms.append(Geom("knocker_col", "capsule", (0.007, 0.03), (0, -0.025, -0.01), tuple(quat_z_to((1, 0, 0))), km, True, False, 8500, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, FULL_ONLY, "decor", "Knocker ring"))
        model.add_body(kb)
    if "house_number" in ex:
        key, mesh = MESH.house_numbers_mesh(n=3)
        nm = mat_from_material(model, "black_matte_metal", "mat_numbers")
        leaf_body.geoms.append(mesh_geom("house_numbers", key, mesh, (x0 + u * W / 2, -1.0 * t / 2, z0 + Hh - 0.35), q_face(-1.0, u), nm, 1000, False, FULL_ONLY, "decor", "House numbers"))
    if "wreath" in ex:
        key, mesh = MESH.wreath_mesh(r=min(0.2, W * 0.25))
        wm = mat_rgba(model, "mat_wreath", (0.12, 0.30, 0.12, 1), 0.95)
        leaf_body.geoms.append(mesh_geom("wreath", key, mesh, (x0 + u * W / 2, -1.0 * t / 2, z0 + Hh * 0.78), q_face(-1.0, u), wm, 200, False, FULL_ONLY, "decor", "Wreath"))
        # the hook it hangs on (a wreath 7 mm off the door face with nothing behind it reads as floating)
        leaf_body.geoms.append(cyl("wreath_hook", (x0 + u * W / 2, -1.0 * (t / 2 + 0.007), z0 + Hh * 0.78 + min(0.2, W * 0.25) - 0.012),
                                   0.004, 0.008, mat_from_material(model, "brass", "mat_wreath_hook"), (0, 1, 0), 8500, False, True, FULL_ONLY, "decor", "Wreath hook"))
    # Three extras the taxonomy samples, physics.py charges hardware mass for, and no builder ever
    # drew: a door that says it has a louvre vent and has none is the "hardware the spec implies but
    # that is not visible" class the vision review exists to catch (docs/VISION_REVIEW.md).
    if "louver_vent" in ex and spec["leaf"].get("panel_style") not in ("louver_full", "louver_half") \
            and not spec["leaf"].get("pet_flap"):     # both want the bottom third of the leaf
        vm = mat_from_material(model, "aluminum", "mat_vent")
        vw, vh = min(0.30, W - 0.16), 0.25
        zc = z0 + 0.18 + vh / 2
        xc = x0 + u * W / 2
        for f in (-1.0, 1.0):
            for sgn in (-1, 1):                                   # vent frame stiles
                leaf_body.geoms.append(box(f"louver_vent_stile_{'p' if f > 0 else 'n'}_{'r' if sgn > 0 else 'l'}",
                                           (xc + sgn * (vw / 2 + 0.012), f * (t / 2 + 0.002), zc),
                                           (0.012, 0.002, vh / 2 + 0.012), vm, 2700, False, True,
                                           FULL_ONLY, "decor", "Louvre vent frame"))
            for sgn in (-1, 1):                                   # vent frame head / sill
                leaf_body.geoms.append(box(f"louver_vent_rail_{'p' if f > 0 else 'n'}_{'t' if sgn > 0 else 'b'}",
                                           (xc, f * (t / 2 + 0.002), zc + sgn * (vh / 2 + 0.012)),
                                           (vw / 2, 0.002, 0.012), vm, 2700, False, True,
                                           FULL_ONLY, "decor", "Louvre vent frame"))
        n_sl = 6
        for k in range(n_sl):                                      # slats, sloped like a real louvre
            zk = zc - vh / 2 + (k + 0.5) * vh / n_sl
            leaf_body.geoms.append(box(f"louver_vent_slat_{k}", (xc, 0.0, zk),
                                       (vw / 2, 0.0015, vh / n_sl * 0.36), vm, 2700, False, True,
                                       FULL_ONLY, "decor", "Louvre slat",
                                       quat=tuple(quat_from_axis_angle([1, 0, 0], math.radians(-30)))))
    if "weather_drip_cap" in ex:
        dm = mat_from_material(model, "aluminum", "mat_drip")
        # on the frame head above the opening, not on the leaf: a drip cap on a swinging leaf would
        # sweep the head casing.  Same standoff treatment as the EXIT sign.
        world.geoms.append(box("weather_drip_cap", (0, -v * 0.03, Ho + 0.045), (Wo / 2 + 0.05, 0.03, 0.008),
                               dm, 2700, False, True, FULL_ONLY, "decor", "Weather drip cap"))
        brace_to_structure(world, world.geoms[-1], -v, dm, name="weather_drip_cap_bracket",
                           semantic="decor", label="Drip cap fixing", tiers=FULL_ONLY, span=0.30)
    if "hold_open_kickdown" in ex or spec["kinematics"].get("stop") == "kick_down_holder":
        add_kick_down_holder(model, leaf_body, spec, u, v, x0, z0, W, t)
    if "coat_hook" in ex:
        key, mesh = MESH.coat_hook_mesh()
        hm = mat_from_material(model, "chrome", "mat_hook")
        leaf_body.geoms.append(mesh_geom("coat_hook", key, mesh, (x0 + u * W / 2, 1.0 * t / 2, z0 + Hh - 0.35), q_face(1.0, u), hm, 7000, False, FULL_ONLY, "decor", "Coat hook"))
    if "exit_sign" in ex:
        key, mesh = MESH.exit_sign_mesh()
        em = mat_rgba(model, "mat_exit_sign", (0.85, 0.1, 0.1, 1), 0.4)
        model.materials["mat_exit_sign"].emissive = (0.8, 0.05, 0.05)
        world.geoms.append(mesh_geom("exit_sign", key, mesh, (0, -v * 0.15, Ho + 0.30), QUAT_ID, em, 500, False, FULL_ONLY, "decor", "EXIT sign"))
        brace_to_structure(world, world.geoms[-1], -v, mat_from_material(model, "aluminum", "mat_sign_bracket"),
                           name="exit_sign_bracket", semantic="decor", label="EXIT sign back box", tiers=FULL_ONLY, span=0.35)
    if "push_pull_sign" in ex:
        # On a solid leaf the sign is an engraved plate; on a GLASS leaf (a storefront door, an automatic slider,
        # a revolving wing) it is a vinyl decal, which is a tenth of a millimetre of plastic and not two of steel.
        # The difference is 0.11 kg per face, and on a balanced rotor that is mass in the wrong place.
        glassy = (spec["leaf"].get("panel_style", "").startswith("glass")
                  or str(spec["leaf"].get("slab", "")).startswith(("glass", "storefront", "revolving")))
        sm = mat_from_material(model, "pvc" if glassy else "stainless", "mat_sign_decal" if glassy else "mat_sign")
        th, dens_s = (0.0002, 1200) if glassy else (0.001, 7900)
        leaf_body.geoms.append(box("sign_push", (x0 + u * W / 2, -v * (t / 2 + th), z0 + 1.35), (0.06, th, 0.03), sm, dens_s, False, True, FULL_ONLY, "decor", "PUSH sign"))
        leaf_body.geoms.append(box("sign_pull", (x0 + u * W / 2, v * (t / 2 + th), z0 + 1.35), (0.06, th, 0.03), sm, dens_s, False, True, FULL_ONLY, "decor", "PULL sign"))
    if "warning_placard" in ex:
        pm = mat_rgba(model, "mat_placard", (0.95, 0.75, 0.05, 1), 0.5)
        leaf_body.geoms.append(box("warning_placard", (x0 + u * W / 2, -1.0 * (t / 2 + 0.001), z0 + Hh * 0.75), (0.11, 0.001, 0.08), pm, 1000, False, True, FULL_ONLY, "decor", "Warning placard"))
    if "keypad_reader_wall" in ex or "rex_button" in ex or "wave_sensor" in ex or "call_button" in ex:
        wm = mat_from_material(model, "black_matte_metal", "mat_wallreader")
        xw = u * (Wo / 2 + 0.25)
        yw = float(model.meta.get("wall_y", 0.0))
        # keep the reader / button / sensor on the wall: a wide opening pushes this past the wall's own end
        x_wall_max = max([abs(float(g.pos[0])) + float(g.size[0]) for g in world.geoms if g.name.startswith("wall_") and g.type == "box"] or [abs(xw) + 0.2])
        xw = math.copysign(min(abs(xw), x_wall_max - 0.15), xw)
        if "keypad_reader_wall" in ex:
            world.geoms.append(box("wall_reader", (xw, yw - (spec["opening"]["wall_thickness"] / 2 + 0.015), 1.1), (0.04, 0.015, 0.06), wm, 1000, True, True, FULL_SIMPLE, "sensor", "Wall card/keypad reader"))
        if "rex_button" in ex:
            # the button sits ON its wall plate (it used to float 6 mm in front of it - and 15 cm above it when a
            # wave sensor pushed the button up the wall)
            z_rex = 1.25 if "wave_sensor" in ex else 1.1
            y_plate = yw + spec["opening"]["wall_thickness"] / 2
            rb = Body("rex_button", "world_env", (xw, y_plate + 0.006, z_rex), QUAT_ID, None, [], [], FULL_SIMPLE, "sensor", "Request-to-exit button")
            rb.joint = Joint("rex_button_slide", "slide", (0, -1, 0), (0, 0, 0), (0.0, 0.004), damping=1.0, stiffness=1500.0, springref=-0.004, role="lock", label="REX button (press to release maglock)")
            rb.geoms.append(cyl("rex_button_geom", (0, 0.004, 0), 0.02, 0.004, mat_rgba(model, "mat_rex", (0.1, 0.6, 0.2, 1), 0.4), (0, 1, 0), 1000, True, True, FULL_SIMPLE, "lock", "REX button"))
            model.add_body(rb)
            world.geoms.append(box("rex_plate", (xw, y_plate + 0.003, z_rex), (0.04, 0.003, 0.06), wm, 1000, False, True, FULL_SIMPLE, "sensor", "REX plate"))
        if "wave_sensor" in ex:
            # a sliding door sweeps the wall beside its opening: its motion sensor goes over the head, as on the
            # real thing, not at hand height where the leaf runs
            z_ws = (Ho + 0.07) if spec["family"] in ("automatic_sliding", "sliding_single", "sliding_bypass", "elevator") else 1.05
            x_ws = 0.0 if z_ws > 1.05 else xw
            for f in (-1, 1):
                world.geoms.append(box(f"wave_sensor_{'p' if f > 0 else 'n'}", (x_ws, yw + f * (spec["opening"]["wall_thickness"] / 2 + 0.012), z_ws), (0.05, 0.012, 0.05), wm, 1000, True, True, FULL_SIMPLE, "sensor", "Wave-to-open / push plate sensor"))
                brace_to_structure(world, world.geoms[-1], float(f), wm, name=f"wave_sensor_{'p' if f > 0 else 'n'}_pad",
                                   semantic="sensor", label="Sensor wall pad", tiers=FULL_SIMPLE, span=0.8, axes=("y",))
        if "call_button" in ex:
            y_cp = yw - spec["opening"]["wall_thickness"] / 2
            world.geoms.append(box("call_plate", (xw, y_cp - 0.003, 1.05), (0.03, 0.003, 0.05), wm, 1000, False, True, FULL_SIMPLE, "sensor", "Call button plate"))
            cb = Body("call_button", "world_env", (xw, y_cp - 0.006, 1.05), QUAT_ID, None, [], [], FULL_SIMPLE, "sensor", "Elevator call button")
            cb.joint = Joint("call_button_slide", "slide", (0, 1, 0), (0, 0, 0), (0.0, 0.003), damping=1.0, stiffness=1500.0, springref=-0.003, role="lock", label="Call button (press)")
            cb.geoms.append(cyl("call_button_geom", (0, -0.003, 0), 0.015, 0.003, mat_rgba(model, "mat_callbtn", (0.9, 0.9, 0.9, 1), 0.3, 0.8), (0, 1, 0), 1000, True, True, FULL_SIMPLE, "lock", "Call button"))
            model.add_body(cb)
    if "sidelite" in ex and spec["opening"].get("sidelite"):
        gm = mat_from_material(model, "glass_clear", "mat_sidelite")
        fm = mat_from_material(model, spec["opening"]["frame"]["material"], "mat_frame")
        xs = u * (Wo / 2 + 0.03 + 0.18)
        world.geoms.append(box("sidelite_glass", (xs, 0, Ho / 2), (0.15, 0.003, Ho / 2 - 0.05), gm, 2500, True, True, FULL_SIMPLE, "glass", "Sidelite"))
        world.geoms.append(box("sidelite_frame", (xs + u * 0.17, float(model.meta.get("wall_y", 0.0)), Ho / 2), (0.02, spec["opening"]["wall_thickness"] / 2, Ho / 2), fm, 600, True, True, FULL_SIMPLE, "frame", "Sidelite mullion"))
    if "transom_window" in ex and spec["opening"].get("transom"):
        gm = mat_from_material(model, "glass_clear", "mat_transom")
        world.geoms.append(box("transom_glass", (0, 0, Ho + 0.05 + 0.2), (Wo / 2, 0.003, 0.18), gm, 2500, True, True, FULL_SIMPLE, "glass", "Transom"))
    if "threshold_saddle" in ex and not any(g.name.startswith("threshold") for g in world.geoms):
        # A saddle is a real sill member.  22 sliding doors declared one and got nothing, because
        # add_threshold only fires on ``opening["threshold"]`` and a sliding spec sets that to "none".
        tm = mat_from_material(model, "aluminum", "mat_threshold")
        y_w = float(model.meta.get("wall_y", 0.0))
        depth = max(0.06, float(spec["opening"]["wall_thickness"]) / 2 + 0.02)
        world.geoms.append(box("threshold_saddle", (0, y_w, 0.006), (Wo / 2 + 0.02, depth, 0.006), tm, 2700,
                               True, True, FULL_SIMPLE, "frame", "Threshold saddle"))
    if "door_stop_floor" in ex or spec["kinematics"].get("stop") == "floor_dome":
        sm = mat_from_material(model, "chrome", "mat_stop")
        # dome on the swing side floor at the max-open sweep position of the leaf's far edge
        ang = math.radians(spec["kinematics"].get("max_open_deg", 90) or 90)
        r = W * 0.85
        jp = leaf_body.joint.pos if leaf_body.joint is not None else (u * GAP, v * (t / 2 + 0.007), 0.0)
        wp = body_world_pos(model, leaf_body)
        rel = (x0 + u * r - jp[0], v * t / 2 - jp[1])
        phi = u * v * ang
        c_, s_ = math.cos(phi), math.sin(phi)
        fx, fy = wp[0] + jp[0] + c_ * rel[0] - s_ * rel[1], wp[1] + jp[1] + s_ * rel[0] + c_ * rel[1]
        nx, ny = -s_ * v, c_ * v
        world.geoms.append(Geom("floor_stop_dome", "sphere", (0.025,), (fx + nx * 0.027, fy + ny * 0.027, 0.0), (1, 0, 0, 0), sm, True, True, 7000, None, (0.6, 0.005, 0.0001), None, None, False, None, None, 0.0, FULL_SIMPLE, "frame", "Floor dome stop"))


def add_pet_flap(model: Model, leaf_body: Body, spec: dict, u: float, x0: float, z0: float, W: float, t: float):
    pf = spec["leaf"].get("pet_flap")
    if not pf:
        return
    fw, fh = pf["width"], pf["height"]
    fm = mat_from_material(model, "pvc", "mat_petframe")
    slab = M.SLABS[pf["slab"]]
    gm = mat_from_material(model, slab.core_material, "mat_petflap")
    xc = x0 + u * W / 2
    zc = z0 + 0.05 + fh / 2
    leaf_body.geoms.append(box("pet_frame_top", (xc, 0, zc + fh / 2 + 0.02), (fw / 2 + 0.04, t / 2 + 0.006, 0.02), fm, 1200, False, True, FULL_SIMPLE, "decor", "Pet door frame"))
    for sgn in (-1, 1):
        leaf_body.geoms.append(box(f"pet_frame_{'l' if sgn < 0 else 'r'}", (xc + sgn * (fw / 2 + 0.02), 0, zc), (0.02, t / 2 + 0.006, fh / 2 + 0.04), fm, 1200, False, True, FULL_SIMPLE, "decor", "Pet door frame"))
    pfx = "" if leaf_body.name == "leaf" else leaf_body.name + "_"
    flap = Body(pfx + "pet_flap", leaf_body.name, (xc, 0, zc + fh / 2), QUAT_ID, None, [], [], FULL_SIMPLE, "leaf", "Pet flap")
    flap.joint = Joint(pfx + "pet_flap_hinge", "hinge", (u, 0, 0), (0, 0, 0), (-1.45, 1.45), damping=0.02, frictionloss=0.02, role="secondary", label="Pet flap (swings both ways)")
    ft = slab.typical_thickness[0]
    flap.geoms.append(box("pet_flap_geom", (0, 0, -fh / 2), (fw / 2 - 0.004, ft / 2, fh / 2 - 0.004), gm, M.MATERIALS[slab.core_material].density, True, True, FULL_SIMPLE, "leaf", "Pet flap"))
    model.add_body(flap)
    model.meta.setdefault("attachment_allow", []).append(
        ["*", f"{pfx}pet_flap*", "a pet flap swings in a hole: it hangs on its top hinge line and keeps a 4 mm running clearance from the frame all round"])
    model.meta.setdefault("notes", []).append("Pet flap: leaf slab collision box is NOT cut out (single box); the flap opening is visual + the flap is a separate articulated body.")

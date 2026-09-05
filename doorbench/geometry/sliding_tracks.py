"""Horizontal rail extents follow each leaf's complete, unlocked travel envelope."""
from __future__ import annotations

import math

from ..ir import FULL_ONLY, FULL_SIMPLE, ALL_TIERS
from . import common as C

CARRIER_MIN_GAP = 0.03   # m; below this there is no room for a hanger: the leaf runs directly under its header


def add_tracks(model, world, spec, leaf_defs, y_leaf, jamb_t, material):
    """Build a rail for each lane, retaining nominal travel when the lock limits q."""
    W = spec["leaf"]["width"]
    Ho = spec["opening"]["height"]
    track = spec["kinematics"]["track"]
    travel = spec["kinematics"]["travel_m"]
    definitions = {}
    for i, (name, direction, xc, yl) in enumerate(leaf_defs):
        lo, hi = (0.0, travel)
        if spec["family"] == "sliding_bypass":
            hi = W - 0.03
            direction = 1.0 if i == 0 else -1.0
            if 0 < i < len(leaf_defs) - 1:
                lo = -hi
        xs = (xc + direction * lo, xc + direction * hi)
        x0, x1 = min(xs) - W / 2 - 0.05, max(xs) + W / 2 + 0.05
        center, half = (x0 + x1) / 2, (x1 - x0) / 2
        prefix = "" if len(leaf_defs) == 1 else name + "_"
        if track == "surface_flat_track":
            rail_name, y, z, hy, hz = "flat_track", min(yl + 0.02, -spec["opening"]["wall_thickness"] / 2 - 0.02), Ho + 0.12, 0.004, 0.02
        elif track in ("bottom_rolling", "bottom_rail", "cantilever"):
            rail_name = "bottom_rail" if track == "bottom_rolling" else "gate_rail"
            y, z, hy, hz = yl, 0.012, 0.02, 0.012
        elif track == "wood_groove_bottom":
            rail_name, y, z, hy, hz = "shikii", yl, 0.012, 0.025, 0.012
        else:
            rail_name, y, z, hy, hz = "track_header", yl, Ho + jamb_t + 0.04, 0.025, 0.04
            if track == "auto_header":
                hy, hz, z = 0.09, 0.09, Ho + jamb_t + 0.09
        rail_name = prefix + rail_name
        tiers = ALL_TIERS if track in ("bottom_rolling", "bottom_rail", "cantilever", "wood_groove_bottom") else FULL_SIMPLE
        rail_material = C.mat_from_material(model, "hinoki", "mat_shikii") if track == "wood_groove_bottom" else material
        world.geoms.append(C.box(rail_name, (center, y, z), (half, hy, hz), rail_material, 7850 if track == "surface_flat_track" else 2700, True, True, tiers, "track", "Full-travel sliding rail"))
        if track == "wood_groove_bottom":
            world.geoms.append(C.box(prefix + "kamoi", (center, yl, Ho + 0.03), (half, 0.025, 0.03), rail_material, 410, True, True, ALL_TIERS, "track", "Kamoi (head rail)"))
        definition = {"joint": name + "_slide", "body": name, "rail": rail_name,
                      "nominal_range": [lo, hi], "leaf_width_m": W, "rollers": [], "end_stops": [],
                      "floor_guides_required": track == "surface_flat_track" or "floor_guide" in spec.get("extras", []),
                      "floor_guides": []}
        definitions[name] = definition
        if track == "surface_flat_track":
            # Every spacer terminates at the wall face; equal spacing includes the two rail ends.
            wall_face = -spec["opening"]["wall_thickness"] / 2
            rear = y + hy
            for k in range(math.ceil((x1 - x0 - 0.10) / 0.6) + 1):
                count = math.ceil((x1 - x0 - 0.10) / 0.6)
                x = x0 + 0.05 + (x1 - x0 - 0.10) * k / count
                world.geoms.append(C.cyl(f"track_standoff_{k}", (x, (rear + wall_face) / 2, z), 0.012, (wall_face - rear) / 2, material, (0, 1, 0), 7850, False, True, FULL_ONLY, "track", "Wall-to-track spacer"))
    model.meta["sliding_track_supports"] = list(definitions.values())
    return definitions


def add_barn_hangers(model, world, body, spec, zb, support, roller_material, track_material):
    """Treads bear on the top of the bar; bent straps join each axle to the door face."""
    W, Hh, t = (spec["leaf"][k] for k in ("width", "height", "thickness"))
    rail = next(g for g in world.geoms if g.name == support["rail"])
    radius, wheel_y = 0.05, rail.pos[1] - body.pos[1]
    wheel_z = rail.pos[2] + rail.size[2] + radius
    strap_y = wheel_y - 0.01 - 0.004
    face_y = -t / 2 - 0.004
    knee_z = zb + Hh + 0.025
    for k, xr in enumerate((-W / 2 + 0.12, W / 2 - 0.12)):
        wheel = C.cyl(f"{body.name}_hanger_wheel_{k}", (xr, wheel_y, wheel_z), radius, 0.01, roller_material, (0, 1, 0), 7850, False, True, FULL_SIMPLE, "track", "Rail-bearing hanger wheel")
        body.geoms.append(wheel)
        support["rollers"].append(wheel.name)
        for suffix, y, z0, z1 in (("mount", face_y, zb + Hh - 0.12, knee_z), ("strap", strap_y, knee_z, wheel_z + 0.012)):
            body.geoms.append(C.box(f"{body.name}_hanger_{suffix}_{k}", (xr, y, (z0 + z1) / 2), (0.018, 0.004, (z1 - z0) / 2), roller_material, 7850, False, True, FULL_SIMPLE, "track", "Hanger strap"))
        body.geoms.append(C.box(f"{body.name}_hanger_bend_{k}", (xr, (face_y + strap_y) / 2, knee_z), (0.018, abs(strap_y - face_y) / 2 + 0.004, 0.004), roller_material, 7850, False, True, FULL_SIMPLE, "track", "Strap offset above door"))
        body.geoms.append(C.cyl(f"{body.name}_hanger_axle_{k}", (xr, (strap_y + wheel_y) / 2, wheel_z), 0.006, (wheel_y - strap_y) / 2 + 0.014, roller_material, (0, 1, 0), 7850, False, True, FULL_ONLY, "track", "Hanger axle"))
        for dz in (-0.09, -0.045):
            body.geoms.append(C.cyl(f"{body.name}_hanger_bolt_{k}_{int(-dz*1000)}", (xr, face_y - 0.006, zb + Hh + dz), 0.007, 0.004, roller_material, (0, 1, 0), 7850, False, True, FULL_ONLY, "track", "Strap fixing bolt"))
    # Stops bolt to the bar, with a tall contact face tangent to the terminal wheel.
    lo, hi = support["nominal_range"]
    xs = [body.pos[0] + body.joint.axis[0] * q for q in (lo, hi)]
    for side, x in ((-1, min(xs) - W / 2 + 0.12 - radius), (1, max(xs) + W / 2 - 0.12 + radius)):
        stop_name = f"{body.name}_track_stop_{'l' if side < 0 else 'r'}"
        world.geoms.append(C.box(stop_name, (x + side * 0.012, rail.pos[1], wheel_z - 0.025), (0.012, 0.016, 0.027), track_material, 7850, True, True, FULL_SIMPLE, "track", "Bolted track-end bumper"))
        support["end_stops"].append(stop_name)


def add_floor_guides(model, world, spec, direction, y_leaf, material, support=None):
    """A floor-mounted fork overlaps the leaf at closed and fully-open positions."""
    W, t = spec["leaf"]["width"], spec["leaf"]["thickness"]
    x = direction * W / 2
    for side in (-1, 1):
        name = "floor_guide_p" if side > 0 else "floor_guide_n"
        y = y_leaf + side * (t / 2 + 0.005)
        world.geoms.append(C.box(name, (x, y, 0.009), (0.028, 0.004, 0.009), material, 7850, True, True, FULL_SIMPLE, "track", "Floor-mounted guide jaw (1 mm face clearance)"))
        world.geoms.append(C.box(name + "_foot", (x, y + side * 0.014, 0.002), (0.028, 0.018, 0.002), material, 7850, True, True, FULL_SIMPLE, "track", "Guide mounting foot"))
    if support is not None:
        support["floor_guides"].append({"jaws": ["floor_guide_n", "floor_guide_p"], "feet": ["floor_guide_n_foot", "floor_guide_p_foot"]})


def add_lane_floor_guides(model, world, body, spec, bottom_z, support, material):
    """Place enough guide stations to cover every position of one bypass lane.

    A bidirectional middle panel needs two stations: its complete sweep is wider
    than the panel, so a single fixed fork cannot remain engaged throughout.
    """
    W, t = spec["leaf"]["width"], spec["leaf"]["thickness"]
    centers = [body.pos[0] + body.joint.axis[0] * q for q in support["nominal_range"]]
    low, high = min(centers), max(centers)
    count = max(1, math.ceil((high - low) / (W - 0.02)))  # at least 10 mm panel overlap at the furthest center
    top = bottom_z + 0.008  # 8 mm of lateral restraint above the panel's lower edge
    for station in range(count):
        x = low + (station + 0.5) * (high - low) / count
        jaws, feet = [], []
        for side in (-1, 1):
            name = f"{body.name}_floor_guide_{station}_{'n' if side < 0 else 'p'}"
            y = body.pos[1] + side * (t / 2 + 0.005)
            world.geoms.append(C.box(name, (x, y, top / 2), (0.028, 0.004, top / 2), material, 7850, True, True, FULL_SIMPLE, "track", "Lane floor-guide jaw (1 mm face clearance)"))
            world.geoms.append(C.box(name + "_foot", (x, y + side * 0.014, 0.002), (0.028, 0.018, 0.002), material, 7850, True, True, FULL_SIMPLE, "track", "Floor-guide mounting foot"))
            jaws.append(name)
            feet.append(name + "_foot")
        support["floor_guides"].append({"jaws": jaws, "feet": feet})


def add_header_hangers(model, world, body, spec, zb, support, roller_material):
    """Roller carriers from the leaf's top edge up to whatever it hangs from.

    A top-hung leaf hangs from carriers running in the header above it.  The suspension used to be missing
    entirely: the leaf hung 50-240 mm below its rail with nothing between the two.  The carrier is built up to the
    LOWEST static surface above the leaf (its rail, or the frame head the rail sits in), and is skipped when that
    surface is less than CARRIER_MIN_GAP away - a leaf that fills its opening up to the head runs directly under it
    and is held by the opening, with no room for a visible carrier."""
    W, Hh, t = (spec["leaf"][k] for k in ("width", "height", "thickness"))
    rail = next((g for g in world.geoms if g.name == support["rail"]), None)
    if rail is None:
        return
    z_rail_lo = float(rail.pos[2]) - float(rail.size[2])
    z_top = zb + Hh
    x_body, y_body = float(body.pos[0]), float(body.pos[1])
    z_target = None
    for o in world.geoms:
        if o.type != "box" or abs(float(o.quat[0]) - 1.0) > 1e-9 or o.semantic in ("floor",):
            continue
        px, py, pz = (float(c) for c in o.pos)
        sx, sy, sz = (float(q) for q in o.size[:3])
        if abs(py - y_body) > sy + t / 2 + 0.01 or abs(px - x_body) > sx + W / 2:
            continue
        lo_z = pz - sz
        if lo_z > z_top + 0.004 and (z_target is None or lo_z < z_target):
            z_target = lo_z
    if z_target is None or z_target - z_top < CARRIER_MIN_GAP:
        return
    y_rel = float(rail.pos[1]) - y_body
    for k, xr in enumerate((-W / 2 + 0.12, W / 2 - 0.12)):
        # carrier plate bolted to the top edge (4 mm clear of the header), wheel bearing on the header
        z1 = z_target - 0.004
        body.geoms.append(C.box(f"{body.name}_carrier_{k}", (xr, y_rel, (z_top - 0.06 + z1) / 2),
                                (0.020, 0.005, (z1 - z_top + 0.06) / 2), roller_material, 7850, False, True,
                                FULL_SIMPLE, "track", "Roller carrier"))
        wheel = C.cyl(f"{body.name}_carrier_wheel_{k}", (xr, y_rel, z_target - 0.014), 0.014, 0.006, roller_material,
                      (0, 1, 0), 7850, False, True, FULL_SIMPLE, "track", "Carrier wheel")
        body.geoms.append(wheel)
        if abs(z_target - z_rail_lo) < 1e-6:
            support["rollers"].append(wheel.name)     # only a wheel that really bears on the RAIL is track support

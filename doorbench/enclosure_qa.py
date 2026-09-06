"""Deterministic ENCLOSURE gates: the door stays in its guides, and the hole in the wall is the door's opening.

Two defects that every other gate is blind to, both of them of the "a person would say that is obviously wrong"
kind, and both of them found by eye on the 2026-09 vision review (docs/VISION_REVIEW.md) with all deterministic
gates green:

``guided_travel``
    A moving assembly that LEAVES its own guides somewhere in its travel.  15 of 15 roll-up curtains rose as a
    rigid slab, left the top of their side guides at the opening head and ended up hanging 2.1-3.6 m in the air
    above the building with sky between the curtain and the wall.  Clearance could not see it (nothing was
    touching anything), attachment could not see it (the curtain still reached its guides at the shipped pose),
    and the sliding-track gate only covers HORIZONTAL rails.  This gate sweeps the primary joint over its whole
    range - with joint equalities resolved, so coupled parts (a coiling curtain's courses) follow - and requires
    every geom of the declared moving assembly to stay inside a declared guide envelope at every sample.

    The envelope cannot be invented: each of its zones names the static guide/track geoms that BACK it, and the
    zone's lateral faces and its top and bottom have to coincide with the extent of that real hardware
    (``BACK_TOL``).  Declaring a bigger box means drawing a bigger guide.

``wall_opening``
    A hole in the wall that is not the door's opening.  18 of 18 sectional garage doors cut the wall away over
    the door's whole lift envelope (Ho + Hh + 0.08) so the leaf could travel inside the wall plane, which left a
    2.0-2.5 m hole above the door, open to the sky, on every one of them.  This gate rasterises the wall plane,
    finds every part of it that no static geometry closes, and requires all of it to lie inside the DECLARED
    opening (plus the frame's own rough-opening margin ``OPENING_TOL``).

Both gates run in ``doorbench/qa.py`` inside ``signed_off``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .clearance import resolve_joint_equalities

TOL_ENVELOPE = 0.004     # m; slack on "inside the guides".  A part is authored to run 3-5 mm clear of its guide,
#                          so 4 mm of float on the envelope face is below the fit itself and four orders above the
#                          kinematic float; the defect this gate exists for is 2-3 m out.
BACK_TOL = 0.030         # m; how closely a declared envelope face has to coincide with the real guide that backs
#                          it.  30 mm is the depth of a guide's own flange - enough that a zone may be quoted from
#                          the outside face of a C-channel, far too little to declare an envelope no hardware makes.
OPENING_TOL = 0.150      # m; how much larger than the declared opening the hole in the wall may measure.  A rough
#                          opening is the finished one plus jamb lining and stud pocket (frame_hole: jamb_t +
#                          STUD_POCKET, up to 0.09 m a side); 0.15 m covers every frame in the dataset and is an
#                          order below the 2.0-2.5 m holes this gate exists to catch.
CELL = 0.02              # m; wall raster cell.  20 mm cells resolve a 60 mm gap and cost ~10^4 cells on a 6 x 5 m
#                          wall.
SAMPLES = 33             # sweep samples over the primary joint's range
WALL_GEOMS = ("wall_left", "wall_right", "wall_header")   # the panels add_floor_and_wall builds a wall plane from
SURFACE_TRIM = 0.05      # m; static geometry within this of the wall's faces still closes the wall (jamb lining,
#                          stop moulding, a bulkhead's coaming): it is what a person sees plugging the hole.
# Families whose wall opening is deliberately NOT the leaf's opening, each with the reason.  A model may declare
# meta["wall_opening"] = [x0, x1, z0, z1] only if it is one of these; everywhere else the declared door opening is
# what the wall has to show.
WALL_OPENING_EXEMPT = {
    "baby_gate": "a child/pet gate divides a full-height passage: the opening above the gate IS the passage",
    "stall": "a toilet partition stands in a full-height entrance: the door hangs 0.3 m off the floor with the "
             "statutory gap over it, and the headrail - not the wall - closes the top",
}


# ---------------------------------------------------------------------------
# guided travel
# ---------------------------------------------------------------------------
def _world_aabb(model, data, g: int) -> np.ndarray:
    """(lo, hi) world axis-aligned bounds of one geom, from MuJoCo's own local AABB."""
    R = data.geom_xmat[g].reshape(3, 3)
    c = data.geom_xpos[g] + R @ model.geom_aabb[g, :3]
    h = np.abs(R) @ model.geom_aabb[g, 3:]
    return np.stack([c - h, c + h])


def _contained(lo: np.ndarray, hi: np.ndarray, zones: List[dict], tol: float) -> Tuple[bool, str]:
    """Is the box inside the UNION of the zones (each a lateral box over a z band)?

    The zones of one envelope stack in z (guides up to the head, hood above them), so the box is split at every
    zone boundary and each slice has to fit one zone: that is exact for stacked zones and conservative for any
    other arrangement.
    """
    edges = sorted({float(lo[2]), float(hi[2])} | {float(z) for zn in zones for z in zn["z"]
                                                   if lo[2] - tol <= z <= hi[2] + tol})
    edges = [e for e in edges if lo[2] - tol <= e <= hi[2] + tol]
    if len(edges) < 2:
        edges = [float(lo[2]), float(hi[2])]
    for a, b in zip(edges, edges[1:]):
        if b - a < 1e-9:
            continue
        mid = (a + b) / 2
        ok = False
        for zn in zones:
            if not (zn["z"][0] - tol <= mid <= zn["z"][1] + tol):
                continue
            if (zn["z"][0] - tol <= a and b <= zn["z"][1] + tol
                    and zn["x"][0] - tol <= lo[0] and hi[0] <= zn["x"][1] + tol):
                ok = True
                break
        if not ok:
            return False, f"z {a:.3f}-{b:.3f} x {float(lo[0]):.3f}..{float(hi[0]):.3f}"
    return True, ""


def run_guided_travel_qa(model, metadata: dict, samples: int = SAMPLES, tol: float = TOL_ENVELOPE) -> dict:
    """Every part of the declared moving assembly stays inside its guides over the whole travel."""
    import mujoco

    decl = metadata.get("guided_travel")
    if not decl:
        return {"ok": True, "declared": False, "n_failures": 0, "failures": []}
    failures: List[dict] = []
    zones = decl.get("zones") or []
    y_min = float(decl.get("y_min", -1e9))
    static = np.asarray(model.body_weldid)[np.asarray(model.geom_bodyid)] == 0
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
    bname = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
    # ---- the envelope has to be made of real, static guide hardware
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for zn in zones:
        ids = [g for g in range(model.ngeom) if names[g] in set(zn.get("backed_by", []))]
        ids = [g for g in ids if static[g]]
        if not ids:
            failures.append({"check": "zone_unbacked", "zone": zn.get("label", ""), "reason": "no static geom backs this zone"})
            continue
        boxes = np.stack([_world_aabb(model, data, g) for g in ids])
        lo, hi = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
        if abs(float(lo[0]) - zn["x"][0]) > BACK_TOL:
            failures.append({"check": "zone_face", "zone": zn.get("label", ""), "face": "x0",
                             "declared": zn["x"][0], "hardware": float(lo[0])})
        if abs(float(hi[0]) - zn["x"][1]) > BACK_TOL:
            failures.append({"check": "zone_face", "zone": zn.get("label", ""), "face": "x1",
                             "declared": zn["x"][1], "hardware": float(hi[0])})
        if float(hi[2]) < zn["z"][1] - BACK_TOL:
            failures.append({"check": "zone_face", "zone": zn.get("label", ""), "face": "z1",
                             "declared": zn["z"][1], "hardware": float(hi[2])})
        if float(lo[2]) > zn["z"][0] + BACK_TOL:
            failures.append({"check": "zone_face", "zone": zn.get("label", ""), "face": "z0",
                             "declared": zn["z"][0], "hardware": float(lo[2])})
    # ---- the assembly: the declared bodies and everything hanging off them
    want = set(decl.get("bodies") or [])
    moving: List[int] = []
    for b in range(model.nbody):
        p, seen = b, 0
        while p > 0 and seen < 16:
            if bname[p] in want:
                moving.append(b)
                break
            p, seen = int(model.body_parentid[p]), seen + 1
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in set(moving)]
    if not geoms:
        failures.append({"check": "no_moving_geoms", "bodies": sorted(want)})
    try:
        joint = model.joint(decl["joint"]).id
    except KeyError:
        return {"ok": False, "declared": True, "n_failures": len(failures) + 1,
                "failures": failures + [{"check": "missing_joint", "joint": decl.get("joint")}]}
    # The NOMINAL unlocked travel is swept even when an engaged lock narrows the MJCF joint range to 3 mm - the
    # same rule doorbench/sliding_track_qa.py applies to a horizontal rail, and for the same reason: a door that
    # is locked today still has to be built to run its whole travel.
    lo_q, hi_q = (float(v) for v in (decl.get("range") or model.jnt_range[joint]))
    hull = ([min(zn["x"][0] for zn in zones), max(zn["x"][1] for zn in zones),
             min(zn["z"][0] for zn in zones), max(zn["z"][1] for zn in zones)] if zones else [0.0] * 4)
    worst = 0.0
    for q in np.linspace(lo_q, hi_q, samples):
        qpos = model.qpos0.copy()
        qpos[model.jnt_qposadr[joint]] = q
        for _ in range(2):                      # chains of couplings need a second pass (curtain courses)
            resolve_joint_equalities(model, qpos, mujoco)
            qpos[model.jnt_qposadr[joint]] = q
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for g in geoms:
            box = _world_aabb(model, data, g)
            ok, where = _contained(box[0], box[1], zones, tol)
            if not ok:
                # how far outside the guides' overall hull the part reaches (0 when it is between two zones)
                worst = max(worst, hull[0] - float(box[0][0]), float(box[1][0]) - hull[1],
                            float(box[1][2]) - hull[3], hull[2] - float(box[0][2]))
                failures.append({"check": "outside_guides", "geom": names[g], "q": float(q), "where": where})
            if float(box[0][1]) < y_min - tol:
                worst = max(worst, y_min - float(box[0][1]))
                failures.append({"check": "through_the_wall", "geom": names[g], "q": float(q),
                                 "y": float(box[0][1]), "y_min": y_min})
        if len(failures) > 40:
            break
    return {"ok": not failures, "declared": True, "n_samples": samples, "n_geoms": len(geoms),
            "n_zones": len(zones), "worst_excursion_m": float(worst), "n_failures": len(failures),
            "failures": failures[:12]}


# ---------------------------------------------------------------------------
# wall opening
# ---------------------------------------------------------------------------
def run_wall_opening_qa(model, metadata: dict, spec: dict, cell: float = CELL, tol: float = OPENING_TOL) -> dict:
    """The hole in the wall is the door's opening - not the door's whole lift envelope."""
    import mujoco

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
    ids = {n: g for g, n in enumerate(names) if n}
    panels = [ids[n] for n in WALL_GEOMS if n in ids]
    if len(panels) < 2:
        return {"ok": True, "declared": False, "reason": "no wall plane in this scene"}
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wall_y = float(metadata.get("wall_y", 0.0))
    wt = float(spec["opening"].get("wall_thickness", 0.2))
    Wo, Ho = float(spec["opening"]["width"]), float(spec["opening"]["height"])
    # the opening's height is measured from its own sill: a ship's door opens over a 0.3 m coaming, a hatch over
    # its curb, and the hole in the wall starts there
    base = float(spec["opening"].get("sill_height") or 0.0) + float(spec["opening"].get("elevation") or 0.0)
    want = [-Wo / 2, Wo / 2, base, base + Ho]
    declared = metadata.get("wall_opening")
    exempt = spec["family"] in WALL_OPENING_EXEMPT
    if declared and exempt:
        want = [float(v) for v in declared]
    boxes = np.stack([_world_aabb(model, data, g) for g in panels])
    x0, x1 = float(boxes[:, 0, 0].min()), float(boxes[:, 1, 0].max())
    z0, z1 = float(boxes[:, 0, 2].min()), float(boxes[:, 1, 2].max())
    static = np.asarray(model.body_weldid)[np.asarray(model.geom_bodyid)] == 0
    # everything static standing in (or on the face of) the wall plane closes the wall
    plug = []
    for g in range(model.ngeom):
        if not static[g] or int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        b = _world_aabb(model, data, g)
        if b[1][1] < wall_y - wt / 2 - SURFACE_TRIM or b[0][1] > wall_y + wt / 2 + SURFACE_TRIM:
            continue
        plug.append(b)
    nx, nz = max(1, int(round((x1 - x0) / cell))), max(1, int(round((z1 - z0) / cell)))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    zs = z0 + (np.arange(nz) + 0.5) * (z1 - z0) / nz
    covered = np.zeros((nx, nz), dtype=bool)
    for b in plug:
        i = (xs >= b[0][0]) & (xs <= b[1][0])
        k = (zs >= b[0][2]) & (zs <= b[1][2])
        if i.any() and k.any():
            covered[np.ix_(i, k)] = True
    open_cells = ~covered
    # every open cell must be inside the declared opening, plus the frame's own rough-opening margin
    inside = ((xs[:, None] >= want[0] - tol) & (xs[:, None] <= want[1] + tol)
              & (zs[None, :] >= want[2] - tol) & (zs[None, :] <= want[3] + tol))
    stray = open_cells & ~inside
    area = float(stray.sum()) * (x1 - x0) / nx * (z1 - z0) / nz
    worst = None
    if stray.any():
        i, k = np.unravel_index(int(np.argmax(stray * (zs[None, :] - z0 + 1.0))), stray.shape)
        worst = [float(xs[i]), float(zs[k])]      # the highest stray hole: the one open to the sky
    ox = np.flatnonzero(open_cells.any(axis=1))
    oz = np.flatnonzero(open_cells.any(axis=0))
    measured = ([float(xs[ox[0]] - cell / 2), float(xs[ox[-1]] + cell / 2),
                 float(zs[oz[0]] - cell / 2), float(zs[oz[-1]] + cell / 2)] if len(ox) and len(oz) else None)
    return {"ok": bool(area <= 4 * cell * cell), "declared": True, "wall_x": [x0, x1], "wall_z": [z0, z1],
            "expected_opening": want, "measured_opening": measured, "stray_open_area_m2": area,
            "worst_stray_point": worst, "exempt": bool(declared and exempt),
            "n_failures": int(bool(area > 4 * cell * cell)),
            "failures": ([{"check": "unintended_opening", "area_m2": area, "at": worst,
                           "measured": measured, "expected": want}] if area > 4 * cell * cell else [])}


def run_enclosure_qa(model, metadata: dict, spec: dict) -> Dict[str, dict]:
    """Both enclosure gates: {"guided_travel": ..., "wall_opening": ...}."""
    return {"guided_travel": run_guided_travel_qa(model, metadata),
            "wall_opening": run_wall_opening_qa(model, metadata, spec)}

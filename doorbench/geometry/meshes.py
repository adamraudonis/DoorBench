"""Procedural hardware mesh library (trimesh).

Every function returns a watertight-ish trimesh.Trimesh in a documented local
frame.  Meshes are cached & deduplicated by a content key so the exported
dataset shares one file per distinct hardware shape.

Local frame conventions (unless noted):
  * +z = spindle / mounting axis pointing *away* from the door face
  * origin at the spindle center on the door face plane
  * lever/knob "reach" direction is -x (toward the hinge side); callers rotate.
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from typing import Iterable

import numpy as np
import trimesh
from trimesh import creation, transformations as tf

_CACHE: dict[str, trimesh.Trimesh] = {}


def key_for(name: str, **params) -> str:
    blob = json.dumps({"fn": name, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()}}, sort_keys=True)
    return name + "_" + hashlib.sha1(blob.encode()).hexdigest()[:10]


def _cached(name, builder, **params):
    k = key_for(name, **params)
    if k not in _CACHE:
        m = builder(**params)
        m.merge_vertices()
        m.remove_unreferenced_vertices()
        m.metadata["key"] = k
        _CACHE[k] = m
    return k, _CACHE[k]


def get_mesh(key: str) -> trimesh.Trimesh:
    return _CACHE[key]


# ---------------------------------------------------------------------------
# primitives helpers
# ---------------------------------------------------------------------------
def _cyl(r, h, sections=24, transform=None):
    m = creation.cylinder(radius=r, height=h, sections=sections)
    if transform is not None:
        m.apply_transform(transform)
    return m


def _T(x=0, y=0, z=0):
    return tf.translation_matrix([x, y, z])


def _R(axis, ang, point=None):
    return tf.rotation_matrix(ang, axis, point)


def _union(meshes: Iterable[trimesh.Trimesh]) -> trimesh.Trimesh:
    ms = [m for m in meshes if m is not None and len(m.faces)]
    if len(ms) == 1:
        return ms[0]
    try:
        u = trimesh.boolean.union(ms, engine="manifold")
        if u is not None and len(u.faces):
            return u
    except Exception:
        pass
    return trimesh.util.concatenate(ms)


def tube_along_path(points, radius, sections=16, cap=True) -> trimesh.Trimesh:
    """Swept circular tube along a polyline (no boolean; straight segments + sphere elbows)."""
    pts = np.asarray(points, dtype=float)
    parts = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        c = creation.cylinder(radius=radius, height=L, sections=sections)
        # align z to d
        z = np.array([0, 0, 1.0])
        dn = d / L
        v = np.cross(z, dn)
        s = np.linalg.norm(v)
        if s < 1e-9:
            R = np.eye(4) if dn[2] > 0 else _R([1, 0, 0], math.pi)
        else:
            R = _R(v / s, math.atan2(s, float(np.dot(z, dn))))
        c.apply_transform(R)
        c.apply_translation((a + b) / 2)
        parts.append(c)
    if cap:
        for p in pts[1:-1]:
            sph = creation.icosphere(subdivisions=1, radius=radius)
            sph.apply_translation(p)
            parts.append(sph)
        for p in (pts[0], pts[-1]):
            sph = creation.icosphere(subdivisions=1, radius=radius)
            sph.apply_translation(p)
            parts.append(sph)
    return trimesh.util.concatenate(parts)


def revolve_profile(profile_rz, sections=32) -> trimesh.Trimesh:
    """Revolve an (r, z) polyline about the z axis."""
    prof = np.asarray(profile_rz, dtype=float)
    return creation.revolve(prof, sections=sections)


def rounded_box(extents, radius, sections=6):
    ex = np.asarray(extents, float)
    r = min(radius, ex.min() / 2 * 0.99)
    try:
        # build via minkowski-like approach: box shrunk + sphere sweep approximated by union of cylinders/spheres
        core = creation.box(extents=ex - 2 * r)
        parts = [core]
        for axis in range(3):
            e = ex.copy()
            e[axis] = ex[axis]
            other = [i for i in range(3) if i != axis]
            b = ex.copy()
            b[other[0]] -= 2 * r
            b[other[1]] -= 2 * r
            parts.append(creation.box(extents=b))
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    s = creation.icosphere(subdivisions=1, radius=r)
                    s.apply_translation([sx * (ex[0] / 2 - r), sy * (ex[1] / 2 - r), sz * (ex[2] / 2 - r)])
                    parts.append(s)
        # edges
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            for s0 in (-1, 1):
                for s1 in (-1, 1):
                    c = creation.cylinder(radius=r, height=ex[axis] - 2 * r, sections=sections * 2)
                    if axis == 0:
                        c.apply_transform(_R([0, 1, 0], math.pi / 2))
                    elif axis == 1:
                        c.apply_transform(_R([1, 0, 0], math.pi / 2))
                    p = np.zeros(3)
                    p[other[0]] = s0 * (ex[other[0]] / 2 - r)
                    p[other[1]] = s1 * (ex[other[1]] / 2 - r)
                    c.apply_translation(p)
                    parts.append(c)
        return trimesh.util.concatenate(parts)
    except Exception:
        return creation.box(extents=ex)


# ---------------------------------------------------------------------------
# Levers & knobs.  Frame: z = spindle axis (away from door face), lever reaches -x.
# ---------------------------------------------------------------------------
def _lever(shape="straight", length=0.125, diameter=0.019, rose_diameter=0.07, standoff=0.055, square=False, ret=False, escutcheon=None):
    parts = []
    r = diameter / 2
    if rose_diameter > 0:
        parts.append(_cyl(rose_diameter / 2, 0.010, 32, _T(0, 0, 0.005)))
    if escutcheon:
        h, w = escutcheon
        e = creation.box(extents=[w, h, 0.006])
        e.apply_translation([0, 0, 0.003])
        parts.append(e)
    # neck / spindle hub
    parts.append(_cyl(max(r * 1.3, 0.011), standoff - r, 24, _T(0, 0, (standoff - r) / 2 + 0.004)))
    zc = standoff  # lever arm center height above face
    if shape == "straight":
        path = [(0, 0, zc), (-length, 0, zc)]
        parts.append(tube_along_path(path, r, 16))
    elif shape == "return":
        path = [(0, 0, zc), (-length, 0, zc), (-length, 0, zc - (standoff - 0.02))]
        parts.append(tube_along_path(path, r, 16))
    elif shape == "wave":
        n = 8
        path = [(-length * i / n, 0, zc + 0.012 * math.sin(math.pi * i / n) - 0.006 * (i / n) ** 2) for i in range(n + 1)]
        parts.append(tube_along_path(path, r, 14))
    elif shape == "L":
        path = [(0, 0, zc), (-length, 0, zc)]
        if square:
            b = creation.box(extents=[length, diameter, diameter])
            b.apply_translation([-length / 2, 0, zc])
            parts.append(b)
            hub = creation.box(extents=[diameter * 1.3, diameter * 1.3, standoff])
            hub.apply_translation([0, 0, standoff / 2])
            parts.append(hub)
        else:
            parts.append(tube_along_path(path, r, 16))
    elif shape == "dog":
        path = [(0, 0, zc), (-length, 0, zc)]
        parts.append(tube_along_path(path, r, 12))
        knob = creation.icosphere(subdivisions=2, radius=r * 1.6)
        knob.apply_translation([-length, 0, zc])
        parts.append(knob)
    elif shape == "T":
        # T-handle: bar centred on the spindle
        parts.append(tube_along_path([(-length / 2, 0, zc), (length / 2, 0, zc)], r, 14))
    elif shape == "safeguard":
        # horizontal-axis handle: bar along -x with grip block
        path = [(0, 0, zc), (-length * 0.35, 0, zc), (-length, 0, zc + 0.03)]
        parts.append(tube_along_path(path, r * 1.2, 14))
        blk = creation.box(extents=[0.06, 0.09, 0.04])
        blk.apply_translation([0, 0, 0.02])
        parts.append(blk)
    return _union(parts) if False else trimesh.util.concatenate(parts)


def lever_mesh(**p):
    return _cached("lever", _lever, **p)


def _knob(shape="round", diameter=0.054, depth=0.060, rose_diameter=0.064, childproof_cover=0.0, privacy_button=False):
    parts = []
    if rose_diameter > 0:
        parts.append(_cyl(rose_diameter / 2, 0.008, 32, _T(0, 0, 0.004)))
    neck_h = depth - diameter * 0.75
    parts.append(_cyl(0.011, neck_h, 20, _T(0, 0, neck_h / 2 + 0.006)))
    R = diameter / 2
    zc = neck_h + R * 0.9
    if shape == "round":
        prof = [(0, zc - R * 0.95), (R * 0.55, zc - R * 0.9), (R * 0.92, zc - R * 0.5), (R, zc), (R * 0.9, zc + R * 0.5), (R * 0.5, zc + R * 0.85), (0, zc + R * 0.9)]
        parts.append(revolve_profile(prof, 32))
    elif shape == "egg":
        prof = [(0, zc - R), (R * 0.6, zc - R * 0.8), (R * 0.95, zc - R * 0.2), (R * 0.85, zc + R * 0.5), (R * 0.45, zc + R * 1.1), (0, zc + R * 1.2)]
        parts.append(revolve_profile(prof, 32))
    elif shape == "faceted":
        s = creation.icosphere(subdivisions=1, radius=R)
        s.apply_translation([0, 0, zc])
        parts.append(s)
    if privacy_button:
        parts.append(_cyl(0.005, 0.012, 12, _T(0, 0, zc + R * 0.9 + 0.006)))
    if childproof_cover > 0:
        cover = revolve_profile([(0.012, 0.0), (childproof_cover / 2, 0.0), (childproof_cover / 2, zc + R * 1.1), (childproof_cover / 2 - 0.02, zc + R * 1.35), (0, zc + R * 1.4)], 24)
        parts.append(cover)
    return trimesh.util.concatenate(parts)


def knob_mesh(**p):
    return _cached("knob", _knob, **p)


# ---------------------------------------------------------------------------
# Pulls (frame: z away from face; bar runs along local x for horizontal, or z? we use local y = up when mounted)
# Pull frame: origin at door face center of the pull; bar axis along local y (vertical when mounted), standoff along z.
# ---------------------------------------------------------------------------
def _pull(shape="d_pull", length=0.2, diameter=0.019, standoff=0.06, width=0.032):
    parts = []
    r = diameter / 2
    if shape in ("d_pull", "offset_bar", "ladder"):
        legs_y = [-length / 2 + r, length / 2 - r] if shape != "ladder" else list(np.linspace(-length / 2 + r, length / 2 - r, max(2, int(length / 0.45) + 1)))
        for ly in legs_y:
            parts.append(tube_along_path([(0, ly, 0), (0, ly, standoff)], r, 12))
        if shape == "offset_bar":
            parts.append(tube_along_path([(0, -length / 2, standoff), (0, length / 2, standoff)], r, 14))
        else:
            parts.append(tube_along_path([(0, -length / 2 + r, standoff), (0, length / 2 - r, standoff)], r, 14))
    elif shape == "flat_bar":
        b = creation.box(extents=[width, length, 0.008])
        b.apply_translation([0, 0, standoff])
        parts.append(b)
        for ly in (-length / 2 + 0.02, length / 2 - 0.02):
            l = creation.box(extents=[width, 0.02, standoff])
            l.apply_translation([0, ly, standoff / 2])
            parts.append(l)
    elif shape == "lift_handle":
        b = creation.box(extents=[length, width, 0.006])
        b.apply_translation([0, 0, standoff])
        parts.append(b)
        for lx in (-length / 2 + 0.015, length / 2 - 0.015):
            l = creation.box(extents=[0.02, width, standoff])
            l.apply_translation([lx, 0, standoff / 2])
            parts.append(l)
    return trimesh.util.concatenate(parts)


def pull_mesh(**p):
    return _cached("pull", _pull, **p)


def _ring(ring_diameter=0.12, bar_diameter=0.014, backplate=0.09):
    parts = []
    plate = _cyl(backplate / 2, 0.004, 24, _T(0, 0, 0.002))
    parts.append(plate)
    hub = _cyl(0.012, 0.03, 16, _T(0, 0, 0.017))
    parts.append(hub)
    ring = creation.torus(major_radius=ring_diameter / 2, minor_radius=bar_diameter / 2, major_sections=32, minor_sections=10)
    ring.apply_transform(_R([1, 0, 0], math.pi / 2))
    ring.apply_translation([0, -ring_diameter / 2 + 0.005, 0.032])
    parts.append(ring)
    return trimesh.util.concatenate(parts)


def ring_mesh(**p):
    return _cached("ring", _ring, **p)


def _spoked_wheel(diameter=0.4, spokes=5, bar_diameter=0.022, hub_len=0.08):
    parts = []
    rim = creation.torus(major_radius=diameter / 2, minor_radius=bar_diameter / 2, major_sections=40, minor_sections=10)
    rim.apply_translation([0, 0, hub_len])
    parts.append(rim)
    parts.append(_cyl(0.035, hub_len + 0.02, 20, _T(0, 0, (hub_len + 0.02) / 2)))
    for i in range(spokes):
        a = 2 * math.pi * i / spokes
        parts.append(tube_along_path([(0, 0, hub_len), (diameter / 2 * math.cos(a), diameter / 2 * math.sin(a), hub_len)], bar_diameter * 0.45, 10))
    return trimesh.util.concatenate(parts)


def wheel_mesh(**p):
    return _cached("wheel", _spoked_wheel, **p)


def _touchbar(length=0.6, height=0.05, depth=0.065, rim_case=True):
    """Exit device: rail (fixed) + pad (moving) are separate meshes; this returns the *pad*.
    Frame: x along the bar, y up, z away from door face."""
    pad = creation.box(extents=[length - 0.006, height * 0.9, 0.03])
    pad.apply_translation([0, 0, depth - 0.015])
    return pad


def touchbar_pad_mesh(**p):
    return _cached("touchbar_pad", _touchbar, **p)


def _touchbar_rail(length=0.6, height=0.05, depth=0.065, rim_case=True, case_len=0.07):
    """Exit-device housing as a CHANNEL: thin back plate + end blocks; the pad slides between the blocks."""
    parts = []
    back = creation.box(extents=[length + 0.04, height, 0.012])
    back.apply_translation([0, 0, 0.006])
    parts.append(back)
    for sx in (-1, 1):
        blk = creation.box(extents=[0.02, height, depth])
        blk.apply_translation([sx * (length / 2 + 0.01), 0, depth / 2])
        parts.append(blk)
    if rim_case:
        case = creation.box(extents=[case_len, height * 1.5, depth * 1.05])
        case.apply_translation([length / 2 + 0.02 + case_len / 2, 0, depth * 0.525])
        parts.append(case)
    end = creation.box(extents=[0.06, height * 1.3, depth * 0.9])
    end.apply_translation([-length / 2 - 0.05, 0, depth * 0.45])
    parts.append(end)
    return trimesh.util.concatenate(parts)


def touchbar_rail_mesh(**p):
    return _cached("touchbar_rail", _touchbar_rail, **p)


def _crossbar(length=0.7, bar_diameter=0.025, arm_length=0.06):
    parts = [tube_along_path([(-length / 2, 0, arm_length), (length / 2, 0, arm_length)], bar_diameter / 2, 16)]
    for lx in (-length / 2, length / 2):
        parts.append(tube_along_path([(lx, 0, 0.0), (lx, 0, arm_length)], bar_diameter / 2 * 0.8, 12))
    return trimesh.util.concatenate(parts)


def crossbar_mesh(**p):
    return _cached("crossbar", _crossbar, **p)


def _paddle(size=(0.10, 0.18), standoff=0.045, arm=False):
    w, h = size
    parts = []
    base = creation.box(extents=[w * 0.9, h, 0.006])
    base.apply_translation([0, 0, 0.003])
    parts.append(base)
    pad = creation.box(extents=[w, h * 0.7, 0.012])
    pad.apply_transform(_R([0, 1, 0], math.radians(15)))
    pad.apply_translation([0, 0, standoff])
    parts.append(pad)
    for ly in (-h * 0.3, h * 0.3):
        parts.append(tube_along_path([(0, ly, 0.006), (0, ly, standoff - 0.006)], 0.006, 10))
    return trimesh.util.concatenate(parts)


def paddle_mesh(**p):
    return _cached("paddle", _paddle, **p)


def _thumb_latch(handle_length=0.2, bar_length=0.18):
    """Suffolk latch grip side: backplate + bow handle + thumb press.  Frame: y up, z away."""
    parts = []
    plate = creation.box(extents=[0.035, handle_length + 0.06, 0.004])
    plate.apply_translation([0, 0, 0.002])
    parts.append(plate)
    bow = tube_along_path([(0, -handle_length / 2, 0.004), (0, -handle_length / 2 + 0.02, 0.035), (0, handle_length / 2 - 0.02, 0.035), (0, handle_length / 2, 0.004)], 0.007, 12)
    parts.append(bow)
    # (the thumb press is a separate articulated body, not part of this mesh)
    return trimesh.util.concatenate(parts)


def thumb_latch_mesh(**p):
    return _cached("thumb_latch", _thumb_latch, **p)


# Butt hinge local frame: z along the pin, +x toward the door interior, +y across the door thickness
# (from the pin, which sits 7 mm proud of the swing face, into the leaf).  Both plates lie in the door-edge /
# jamb-face plane (x ~ const), like a real mortised butt hinge; the door edge is at x = -3 mm (frame gap 3 mm).
def _hinge_knuckle(height=0.114, radius=0.0075, leaf_w=0.05, leaf_t=0.003, knuckles=5):
    """Door-side half: knuckle barrel + the plate mortised into the door edge."""
    parts = [_cyl(radius, height, 16)]
    l1 = creation.box(extents=[leaf_t, leaf_w, height * 0.92])
    l1.apply_translation([-0.003 + leaf_t / 2, 0.007 + leaf_w / 2, 0])
    parts.append(l1)
    return trimesh.util.concatenate(parts)


def _hinge_jamb_plate(height=0.114, radius=0.0075, leaf_w=0.05, leaf_t=0.003, knuckles=5):
    """Frame-side plate (static: mortised into the jamb face, 6 mm from the pin)."""
    l2 = creation.box(extents=[leaf_t, leaf_w, height * 0.92])
    l2.apply_translation([-0.006 - leaf_t / 2, 0.007 + leaf_w / 2, 0])
    return l2


def hinge_mesh(**p):
    return _cached("hinge", _hinge_knuckle, **p)


def hinge_jamb_mesh(**p):
    return _cached("hinge_jamb", _hinge_jamb_plate, **p)


def _strap_hinge(length=0.3, width=0.05, thickness=0.006):
    strap = creation.box(extents=[length, thickness, width])
    strap.apply_translation([length / 2, 0, 0])
    tip = creation.cylinder(radius=width / 2, height=thickness, sections=16)
    tip.apply_transform(_R([1, 0, 0], math.pi / 2))
    tip.apply_translation([length, 0, 0])
    barrel = _cyl(0.008, width * 1.2, 16)
    return trimesh.util.concatenate([strap, tip, barrel])


def strap_hinge_mesh(**p):
    return _cached("strap_hinge", _strap_hinge, **p)


def _keypad(w=0.07, h=0.15, keys=10, depth=0.02):
    parts = []
    body = creation.box(extents=[w, h, depth])
    body.apply_translation([0, 0, depth / 2])
    parts.append(body)
    if depth > 0.025:
        # mechanical pushbutton lock: rounded top cap
        cap = creation.box(extents=[w * 0.9, 0.012, depth * 0.6])
        cap.apply_translation([0, h / 2 + 0.006, depth * 0.3])
        parts.append(cap)
    return trimesh.util.concatenate(parts)


def keypad_body_mesh(**p):
    return _cached("keypad_body", _keypad, **p)


def _escutcheon(w=0.045, h=0.24, t=0.006):
    e = creation.box(extents=[w, h, t])
    e.apply_translation([0, 0, t / 2])
    return e


def escutcheon_mesh(**p):
    return _cached("escutcheon", _escutcheon, **p)


def _handleset_grip(grip_length=0.24, plate=(0.30, 0.07), thumb=False):
    """Handleset grip: back plate + bow grip.  The thumb press is a separate articulated body (thumb=False)."""
    parts = []
    pl = creation.box(extents=[plate[1], plate[0], 0.006])
    pl.apply_translation([0, 0, 0.003])
    parts.append(pl)
    grip = tube_along_path([(0, -grip_length / 2, 0.006), (0, -grip_length / 2 + 0.03, 0.06), (0, grip_length / 2 - 0.03, 0.06), (0, grip_length / 2, 0.006)], 0.011, 14)
    parts.append(grip)
    if thumb:
        th = creation.box(extents=[0.035, 0.04, 0.008])
        th.apply_translation([0, grip_length / 2 + 0.03, 0.02])
        parts.append(th)
    return trimesh.util.concatenate(parts)


def handleset_mesh(**p):
    return _cached("handleset", _handleset_grip, **p)


def _thumbturn(length=0.03, rose=0.045):
    parts = [_cyl(rose / 2, 0.006, 24, _T(0, 0, 0.003)), _cyl(0.01, 0.012, 16, _T(0, 0, 0.012))]
    tt = creation.box(extents=[0.008, length, 0.012])
    tt.apply_translation([0, 0, 0.024])
    parts.append(tt)
    return trimesh.util.concatenate(parts)


def thumbturn_mesh(**p):
    return _cached("thumbturn", _thumbturn, **p)


def _cylinder_face(rose=0.045):
    parts = [_cyl(rose / 2, 0.006, 24, _T(0, 0, 0.003)), _cyl(0.012, 0.012, 16, _T(0, 0, 0.012))]
    slot = creation.box(extents=[0.003, 0.014, 0.004])
    slot.apply_translation([0, 0, 0.019])
    parts.append(slot)
    return trimesh.util.concatenate(parts)


def cylinder_face_mesh(**p):
    return _cached("cyl_face", _cylinder_face, **p)


def _card_reader(w=0.075, h=0.26):
    parts = []
    body = creation.box(extents=[w, h, 0.02])
    body.apply_translation([0, 0, 0.01])
    parts.append(body)
    led = _cyl(0.004, 0.003, 12, _T(0, h / 2 - 0.03, 0.021))
    parts.append(led)
    return trimesh.util.concatenate(parts)


def card_reader_mesh(**p):
    return _cached("card_reader", _card_reader, **p)


def _magnalatch(height=0.2):
    parts = []
    body = creation.box(extents=[0.05, height, 0.04])
    body.apply_translation([0, 0, 0.02])
    parts.append(body)
    knob = _cyl(0.02, 0.03, 20, _T(0, height / 2 + 0.015, 0.02))
    parts.append(knob)
    return trimesh.util.concatenate(parts)


def magnalatch_mesh(**p):
    return _cached("magnalatch", _magnalatch, **p)


def _fork_latch(length=0.15):
    parts = []
    fork = creation.box(extents=[length, 0.02, 0.006])
    fork.apply_translation([length / 2, 0, 0])
    parts.append(fork)
    tine = creation.box(extents=[0.03, 0.02, 0.05])
    tine.apply_translation([length - 0.015, 0, -0.025])
    parts.append(tine)
    return trimesh.util.concatenate(parts)


def fork_latch_mesh(**p):
    return _cached("fork_latch", _fork_latch, **p)


def _cane_bolt(length=0.45, diameter=0.016):
    rod = tube_along_path([(0, 0, 0), (0, 0, -length)], diameter / 2, 12)
    handle = tube_along_path([(0, 0, 0), (0.06, 0, 0)], diameter / 2, 10)
    return trimesh.util.concatenate([rod, handle])


def cane_bolt_mesh(**p):
    return _cached("cane_bolt", _cane_bolt, **p)


def _door_closer_body(l=0.29, w=0.07, h=0.10):
    body = rounded_box([l, w, h], 0.012)
    body.apply_translation([0, 0, h / 2])
    return body


def closer_body_mesh(**p):
    return _cached("closer_body", _door_closer_body, **p)


def _exit_sign(w=0.33, h=0.2, d=0.05):
    box = creation.box(extents=[w, d, h])
    return box


def exit_sign_mesh(**p):
    return _cached("exit_sign", _exit_sign, **p)


def _knocker(w=0.09, h=0.14):
    parts = []
    plate = creation.box(extents=[w, h, 0.005])
    plate.apply_translation([0, 0, 0.0025])
    parts.append(plate)
    ring = creation.torus(major_radius=0.035, minor_radius=0.006, major_sections=28, minor_sections=8)
    ring.apply_translation([0, -0.01, 0.025])
    parts.append(ring)
    return trimesh.util.concatenate(parts)


def knocker_mesh(**p):
    return _cached("knocker", _knocker, **p)


def _peephole():
    return _cyl(0.008, 0.012, 16, _T(0, 0, 0.006))


def peephole_mesh(**p):
    return _cached("peephole", _peephole, **p)


def _tripod_rotor(arm_len=0.5, r=0.019, hub_r=0.05):
    parts = [_cyl(hub_r, 0.12, 24)]
    for i in range(3):
        a = 2 * math.pi * i / 3
        parts.append(tube_along_path([(0, 0, 0), (arm_len * math.cos(a), arm_len * math.sin(a), 0)], r, 12))
        cap = creation.icosphere(subdivisions=1, radius=r * 1.3)
        cap.apply_translation([arm_len * math.cos(a), arm_len * math.sin(a), 0])
        parts.append(cap)
    return trimesh.util.concatenate(parts)


def tripod_mesh(**p):
    return _cached("tripod", _tripod_rotor, **p)


def _dog_wedge(l=0.08, w=0.05, h=0.03):
    b = creation.box(extents=[l, w, h])
    b.apply_translation([l / 2, 0, 0])
    return b


def dog_wedge_mesh(**p):
    return _cached("dog_wedge", _dog_wedge, **p)


def _house_numbers(n=3):
    parts = []
    for i in range(n):
        b = creation.box(extents=[0.06, 0.10, 0.006])
        b.apply_translation([(i - (n - 1) / 2) * 0.08, 0, 0.003])
        parts.append(b)
    return trimesh.util.concatenate(parts)


def house_numbers_mesh(**p):
    return _cached("house_numbers", _house_numbers, **p)


def _wreath(r=0.2):
    t = creation.torus(major_radius=r, minor_radius=0.045, major_sections=32, minor_sections=10)
    t.apply_translation([0, 0, 0.05])
    return t


def wreath_mesh(**p):
    return _cached("wreath", _wreath, **p)


def _coat_hook():
    return tube_along_path([(0, 0, 0), (0, 0, 0.035), (0, 0.03, 0.05)], 0.006, 10)


def coat_hook_mesh(**p):
    return _cached("coat_hook", _coat_hook, **p)


def all_cached():
    return dict(_CACHE)

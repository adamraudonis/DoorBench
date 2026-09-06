"""URDF exporter.

Mapping notes (documented in README):
  * The static environment body becomes the root link `world_env` (frame + walls).
  * IR joint `pos` offsets are folded into the joint origin; child geometry is
    shifted accordingly (URDF child link frame == joint frame).
  * Joint springs (closers, latch springs) are not representable in URDF; they
    are emitted as `<doorbench:spring stiffness= springref=/>` extension tags
    under the joint plus in the sidecar `spec.json` -> physics block.
  * Polynomial joint couplings become `<mimic>` (bilateral).  One-sided tendon
    couplings (latch re-latching) are emitted as `<mimic>` too, with a
    `<doorbench:coupling one_sided="true"/>` note.
  * Mesh visuals reference the shared hardware library via relative paths.
"""
from __future__ import annotations

import math
import os
from xml.etree import ElementTree as ET
from xml.dom import minidom

import numpy as np

from ..ir import Model, Body, Geom, quat_to_rpy, quat_rotate, quat_mul, quat_conj, mat_to_quat


def _f(x, nd=6):
    if isinstance(x, (list, tuple, np.ndarray)):
        return " ".join(_f(v, nd) for v in x)
    v = float(x)
    if abs(v) < 1e-12:
        v = 0.0
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _origin(parent, pos, quat):
    r, p, y = quat_to_rpy(quat)
    ET.SubElement(parent, "origin", xyz=_f(pos), rpy=_f((r, p, y)))


def _geom_elems(link, g: Geom, shift, tier, mesh_dir_rel, materials_used):
    pos = np.asarray(g.pos) - shift
    for kind in (("visual",) if g.visual else ()) + (("collision",) if g.collision else ()):
        e = ET.SubElement(link, kind)
        e.set("name", f"{g.name}_{kind[:3]}")
        _origin(e, pos, g.quat)
        geo = ET.SubElement(e, "geometry")
        if g.type == "box":
            ET.SubElement(geo, "box", size=_f([2 * s for s in g.size]))
        elif g.type == "cylinder":
            ET.SubElement(geo, "cylinder", radius=_f(g.size[0]), length=_f(2 * g.size[1]))
        elif g.type == "capsule":
            # URDF has no capsule: cylinder for collision/visual (+ end spheres for visual)
            ET.SubElement(geo, "cylinder", radius=_f(g.size[0]), length=_f(2 * g.size[1]))
        elif g.type == "sphere":
            ET.SubElement(geo, "sphere", radius=_f(g.size[0]))
        elif g.type == "mesh":
            ET.SubElement(geo, "mesh", filename=f"{mesh_dir_rel}/{g.mesh_name}.obj")
        if kind == "visual":
            mat = ET.SubElement(e, "material", name=g.material)
            materials_used.add(g.material)


def build_urdf(model: Model, tier="full", mesh_dir_rel="../../hardware") -> ET.Element:
    model.validate_free_joints()
    bodies = model.bodies_in_tier(tier)
    names = {b.name for b in bodies}
    robot = ET.Element("robot", name=f"{model.name}_{tier}")
    robot.set("xmlns:doorbench", "https://github.com/adamraudonis/DoorBench/schema/urdf-ext")
    materials_used = set()
    # links
    joint_shift = {}
    for b in bodies:
        link = ET.SubElement(robot, "link", name=b.name)
        shift = np.array(b.joint.pos) if (b.joint is not None) else np.zeros(3)
        joint_shift[b.name] = shift
        m, com, I = b.inertial(tier)
        if not b.static:
            inert = ET.SubElement(link, "inertial")
            if m <= 1e-9:
                m, com, I = 0.01, np.zeros(3), np.eye(3) * 1e-6
            _origin(inert, np.asarray(com) - shift, (1, 0, 0, 0))
            ET.SubElement(inert, "mass", value=_f(m))
            # Small steel pins/eyes have valid inertia below 0.5e-9 kg m².
            # Decimal-place rounding erases it and creates a massless axis.
            # Preserve the physical tensor with round-trip significant digits.
            ET.SubElement(inert, "inertia", **{
                name: format(float(I[i, j]), ".17g")
                for name, i, j in (("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
                                   ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2))
            })
        for g in b.geoms:
            if tier in g.tiers:
                _geom_elems(link, g, shift, tier, mesh_dir_rel, materials_used)
    # root handling: bodies with parent None that are not static get attached to world_env by their joint
    root_static = [b for b in bodies if b.parent is None and b.static]
    root_name = root_static[0].name if root_static else None
    if root_name is None:
        root = ET.SubElement(robot, "link", name="world_env")
        root_name = "world_env"
    # joints
    for b in bodies:
        if b.parent is None and b.static:
            continue
        parent = b.parent if b.parent is not None else root_name
        # parent's frame in URDF is shifted by its joint pos; child origin = body.pos + child joint pos - parent shift (rotated)
        pshift = joint_shift.get(parent, np.zeros(3))
        cshift = joint_shift[b.name]
        origin_pos = np.asarray(b.pos) + quat_rotate(b.quat, cshift) - pshift
        if b.joint is None:
            j = ET.SubElement(robot, "joint", name=f"{b.name}_fixed", type="fixed")
            _origin(j, origin_pos, b.quat)
            ET.SubElement(j, "parent", link=parent)
            ET.SubElement(j, "child", link=b.name)
            continue
        jt = b.joint
        if jt.type == "free":
            j = ET.SubElement(robot, "joint", name=jt.name, type="floating")
            _origin(j, origin_pos, b.quat)
            ET.SubElement(j, "parent", link=parent)
            ET.SubElement(j, "child", link=b.name)
            ET.SubElement(j, "doorbench:free_pose", qpos_width="7", qvel_width="6",
                          note="Native world position plus WXYZ quaternion; scalar joint offsets and controls are unavailable.")
            continue
        typ = {"hinge": "revolute", "slide": "prismatic"}[jt.type]
        if jt.range is None:
            typ = "continuous" if jt.type == "hinge" else "prismatic"
        j = ET.SubElement(robot, "joint", name=jt.name, type=typ)
        _origin(j, origin_pos, b.quat)
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=b.name)
        ET.SubElement(j, "axis", xyz=_f(jt.axis))
        m0 = jt.modeled_at
        if abs(m0) > 1e-12:
            ET.SubElement(j, "doorbench:zero_offset", value=_f(m0), note="URDF q=0 is the spec initial state; DoorBench q = urdf_q + value")
        if jt.range is not None:
            lo, hi = jt.range[0] - m0, jt.range[1] - m0
            if hi - lo < 1e-9:
                hi = lo + 1e-6
            ET.SubElement(j, "limit", lower=_f(lo), upper=_f(hi), effort=_f(2000 if jt.type == "hinge" else 5000), velocity=_f(10.0))
        elif typ == "prismatic":
            ET.SubElement(j, "limit", lower="-10", upper="10", effort="5000", velocity="10")
        ET.SubElement(j, "dynamics", damping=_f(jt.damping), friction=_f(jt.frictionloss))
        if jt.stiffness:
            ET.SubElement(j, "doorbench:spring", stiffness=_f(jt.stiffness), springref=_f(jt.springref - m0), note="tau = -stiffness*(q - springref); not native URDF")
        if jt.damping_closing is not None and jt.damping_closing != jt.damping:
            ET.SubElement(j, "doorbench:closer", damping_closing=_f(jt.damping_closing), damping_opening=_f(jt.damping_opening or 0.0), backcheck_angle=_f(jt.backcheck_angle or 0.0), backcheck_damping=_f(jt.backcheck_damping or 0.0))
        if jt.ratchet_one_way:
            ET.SubElement(j, "doorbench:ratchet", one_way="true")
        if jt.notes:
            ET.SubElement(j, "doorbench:note", text=jt.notes)
        # mimic from equalities / tendons where this joint is the driven one
        m_of = {b2.joint.name: b2.joint.modeled_at for b2 in bodies if b2.joint is not None}
        for q in model.equalities:
            if q.kind == "joint" and q.a == jt.name and tier in q.tiers:
                c0, c1 = q.polycoeff[0], q.polycoeff[1]
                # q1 - m1 = c1*(q2 - m2) (MuJoCo semantics with ref) -> urdf: q1' = c1*q2' + (c0)
                ET.SubElement(j, "mimic", joint=q.b, multiplier=_f(c1), offset=_f(c0))
        for t in model.tendons:
            if tier in t.tiers and t.sites and t.sites[0][0] == jt.name and getattr(t, "kind", "fixed") == "fixed":
                # L = q_a + c*q_b >= 0 -> q_a >= -c q_b ; mimic as q_a = -c q_b
                other, c = t.sites[1]
                mm = ET.SubElement(j, "mimic", joint=other, multiplier=_f(-c), offset="0")
                ET.SubElement(j, "doorbench:coupling", one_sided="true", note="MJCF uses a one-sided tendon (bolt may retract further, e.g. riding over the strike lip); mimic is bilateral")
    # materials
    for name in sorted(materials_used):
        m = model.materials.get(name)
        if m is None:
            continue
        me = ET.SubElement(robot, "material", name=name)
        rgba = list(m.rgba)
        if m.transparent and rgba[3] > 0.6:
            rgba[3] = 0.45
        ET.SubElement(me, "color", rgba=_f(rgba, 3))
    # welds (maglock) as fixed joints? No: emit as extension
    for q in model.equalities:
        if q.kind == "weld" and tier in q.tiers and q.a in names:
            ET.SubElement(robot, "doorbench:weld", body1=q.a, body2=q.b or "world", label=q.label, active="true" if q.active else "false")
        if q.kind == "connect" and tier in q.tiers and q.a in names:
            ET.SubElement(robot, "doorbench:loop_closure", body1=q.a, body2=q.b or "world", anchor=_f(q.anchor), label=q.label)
    for spring in model.spatial_springs:
        if tier in spring.tiers:
            ET.SubElement(robot, "doorbench:spatial_spring", name=spring.name,
                site1=spring.sites[0], site2=spring.sites[1], stiffness=_f(spring.stiffness),
                springlength=_f(spring.springlength), damping=_f(spring.damping),
                native_support="false", note="URDF has no native spatial spring; simulation requires a force plugin and loop constraints.")
    for cable in model.spatial_cables:
        if tier in cable.tiers:
            ce=ET.SubElement(robot,"doorbench:spatial_cable",name=cable.name,max_length=_f(cable.max_length),
                native_support="false",note="URDF has no native routed cable or pulley-wrap limit; requires a cable constraint implementation.")
            for point in cable.path:
                ET.SubElement(ce,"doorbench:path_point",**point)
    return robot


def write_urdf(model: Model, out_dir: str, tiers=("full", "simple", "minimal"), mesh_dir_rel="../../hardware"):
    os.makedirs(out_dir, exist_ok=True)
    names = {"full": "door.urdf", "simple": "door_simple.urdf", "minimal": "door_minimal.urdf"}
    out = {}
    for tier in tiers:
        root = build_urdf(model, tier, mesh_dir_rel)
        raw = ET.tostring(root, encoding="unicode")
        txt = minidom.parseString(raw).toprettyxml(indent="  ")
        path = os.path.join(out_dir, names[tier])
        with open(path, "w") as f:
            f.write(txt)
        out[tier] = path
    return out

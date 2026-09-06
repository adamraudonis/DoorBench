"""MJCF (MuJoCo) exporter for the DoorBench IR.

Produces, per door and tier:
  door.xml           full fidelity (all mechanism bodies, mesh visuals, tendons, equality loops)
  door_simple.xml    leaf + primary operator + latch bolt, primitive geoms only
  door_minimal.xml   leaf only
  scene.xml          includes door.xml + floor/lights/camera, ready for `simulate`

Static bodies (frame, wall) are emitted as worldbody geoms so the leaf (a
world child) collides with them (MuJoCo exempts the world from parent/child
contact exclusion).
"""
from __future__ import annotations

import os
import math
from xml.etree import ElementTree as ET
from xml.dom import minidom

import numpy as np

from ..ir import Model, Body, Geom, Joint, quat_to_mat


def _f(x, nd=6):
    if isinstance(x, (list, tuple, np.ndarray)):
        return " ".join(_f(v, nd) for v in x)
    v = float(x)
    if abs(v) < 1e-12:
        v = 0.0
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _look_at(pos, target):
    f = target - pos
    f = f / (np.linalg.norm(f) + 1e-12)
    r = np.cross(f, np.array([0, 0, 1.0]))
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r = r / np.linalg.norm(r)
    up = np.cross(r, f)
    return r, up


def _pretty(elem) -> str:
    raw = ET.tostring(elem, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def _geom_xml(parent, g: Geom, tier: str, mesh_prefix: str, materials: dict, default_class=None):
    g.validate_contact_priority()
    e = ET.SubElement(parent, "geom")
    e.set("name", g.name)
    e.set("type", g.type)
    if g.contact_priority:
        e.set('priority', str(g.contact_priority))
    if g.type == "mesh":
        e.set("mesh", g.mesh_name)
    elif g.type == "box":
        e.set("size", _f(g.size))
    elif g.type in ("cylinder", "capsule"):
        e.set("size", _f(g.size[:2]))
    elif g.type == "sphere":
        e.set("size", _f(g.size[:1]))
    e.set("pos", _f(g.pos))
    if any(abs(a - b) > 1e-9 for a, b in zip(g.quat, (1, 0, 0, 0))):
        e.set("quat", _f(g.quat))
    if g.material in materials:
        e.set("material", g.material)
    if g.collision and g.visual:
        e.set("group", "0")
        e.set("contype", "1")
        e.set("conaffinity", "1")
    elif g.collision:
        e.set("group", "3")
        e.set("contype", "1")
        e.set("conaffinity", "1")
        e.set("rgba", "0.8 0.3 0.3 0.25")
    else:
        e.set("group", "1" if g.semantic not in ("decor",) else "2")
        e.set("contype", "0")
        e.set("conaffinity", "0")
    if g.collision:
        e.set("friction", _f(g.friction))
        if g.solref:
            e.set("solref", _f(g.solref))
        if g.solimp:
            e.set("solimp", _f(g.solimp))
        if g.margin:
            e.set("margin", _f(g.margin))
    # mass: we set explicit inertials on bodies; geoms carry density for reference only
    e.set("density", _f(g.density, 3))
    return e


def _joint_xml(parent, j: Joint):
    if j.type == "free":
        # freejoint deliberately avoids inherited scalar defaults. MuJoCo
        # derives all seven qpos0 values from the owning body's world pose.
        return ET.SubElement(parent, "freejoint", name=j.name)
    e = ET.SubElement(parent, "joint")
    e.set("name", j.name)
    e.set("type", j.type)
    e.set("axis", _f(j.axis))
    if any(abs(p) > 1e-12 for p in j.pos):
        e.set("pos", _f(j.pos))
    if j.range is not None:
        e.set("limited", "true")
        e.set("range", _f(j.range))
        if j.limit_solref:
            e.set("solreflimit", _f(j.limit_solref))
    else:
        e.set("limited", "false")
    if j.damping:
        e.set("damping", _f(j.damping))
    if j.frictionloss:
        e.set("frictionloss", _f(j.frictionloss))
    if j.stiffness:
        e.set("stiffness", _f(j.stiffness))
        e.set("springref", _f(j.springref))
    if j.armature:
        e.set("armature", _f(j.armature, 8))
    if abs(j.modeled_at) > 1e-12:
        e.set("ref", _f(j.modeled_at))
    return e


def _inertial_xml(parent, body: Body, tier: str):
    m, com, I = body.inertial(tier)
    if m <= 1e-9:
        # MuJoCo requires positive mass for bodies with joints
        m, com, I = 0.01, np.zeros(3), np.eye(3) * 1e-6
    e = ET.SubElement(parent, "inertial")
    e.set("pos", _f(com))
    e.set("mass", _f(m))
    # full inertia -> principal axes
    w, V = np.linalg.eigh(I)
    w = np.maximum(w, 1e-9)
    # ensure right-handed
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    # triangle inequality guard
    for _ in range(3):
        a, b, c = w
        if a + b < c:
            w[2] = a + b
        if a + c < b:
            w[1] = a + c
        if b + c < a:
            w[0] = b + c
    from ..ir import mat_to_quat
    q = mat_to_quat(V)
    e.set("quat", _f(q))
    e.set("diaginertia", _f(w, 9))


def build_mjcf(model: Model, tier: str = "full", mesh_dir_rel: str = "../../hardware", texture_dir_rel: str = "../../textures", include_env=True, timestep=0.002) -> ET.Element:
    model.validate_free_joints()
    fixed_names=model.meta.get('native_fixed_body_names',[])
    if not isinstance(fixed_names,list) or any(not isinstance(name,str) for name in fixed_names) or len(set(fixed_names))!=len(fixed_names):
        raise ValueError('native_fixed_body_names must be a list of distinct body names')
    known_bodies={body.name:body for body in model.bodies}
    for name in fixed_names:
        if name not in known_bodies or not known_bodies[name].static or known_bodies[name].joint is not None:
            raise ValueError(f'native_fixed_body_names requires an existing static body without a joint: {name}')
    fixed_names=set(fixed_names)
    native_bound=model.meta.get('native_timestep_s')
    if native_bound is not None:
        if isinstance(native_bound,bool) or not isinstance(native_bound,(int,float)) or not math.isfinite(native_bound) or native_bound<=0:
            raise ValueError('native_timestep_s must be a positive finite number')
        timestep=min(timestep,native_bound)
    integrator=model.meta.get('native_integrator','implicitfast')
    if not isinstance(integrator,str) or integrator not in ('implicitfast','implicit'):
        raise ValueError('native_integrator must be implicitfast or implicit')
    root = ET.Element("mujoco", model=f"{model.name}_{tier}")
    ET.SubElement(root, "compiler", angle="radian", meshdir=mesh_dir_rel, texturedir=texture_dir_rel, autolimits="true", inertiafromgeom="false")
    opt = ET.SubElement(root, "option", timestep=_f(timestep), integrator=integrator, cone="elliptic", impratio="10", noslip_iterations="3")
    ET.SubElement(opt, "flag", multiccd="enable")
    arena=model.meta.get('native_arena_memory_mib',16)
    if isinstance(arena,bool) or not isinstance(arena,(int,float)) or not math.isfinite(arena) or int(arena)!=arena or not 16<=arena<=128:
        raise ValueError('native_arena_memory_mib must be an integer from 16 through 128')
    ET.SubElement(root, "size", memory=f"{int(arena)}M")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", condim="4", solref="0.005 1", solimp="0.95 0.99 0.001")
    ET.SubElement(default, "joint", solreflimit="0.005 1", solimplimit="0.95 0.99 0.001")
    ET.SubElement(default, "tendon", solreflimit="0.005 1")
    ET.SubElement(default, "equality", solref="0.002 1")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7", specular="0.2 0.2 0.2")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "map", znear="0.01", zfar="60")
    asset = ET.SubElement(root, "asset")
    bodies = model.bodies_in_tier(tier)
    used_meshes = {}
    for b in bodies:
        for g in b.geoms:
            if tier in g.tiers and g.type == "mesh":
                used_meshes[g.mesh_name] = g
    for k in sorted(used_meshes):
        ET.SubElement(asset, "mesh", name=k, file=f"{k}.obj")
    # materials
    mats = {}
    for name, m in model.materials.items():
        rgba = list(m.rgba)
        if m.transparent and rgba[3] > 0.6:
            rgba[3] = 0.45
        me = ET.SubElement(asset, "material", name=name, rgba=_f(rgba, 3), specular=_f(0.2 + 0.6 * m.metallic, 3), shininess=_f(max(0.05, 1 - m.roughness), 3), reflectance=_f(0.4 * m.metallic if not m.transparent else 0.1, 3))
        mats[name] = me
    if include_env:
        ET.SubElement(asset, "texture", name="skybox", type="skybox", builtin="gradient", rgb1="0.6 0.7 0.85", rgb2="0.9 0.92 0.95", width="256", height="256")
    world = ET.SubElement(root, "worldbody")
    if include_env:
        ET.SubElement(world, "light", name="sun", pos="1.5 -2.5 4", dir="-0.3 0.5 -0.8", directional="true", diffuse="0.8 0.8 0.8", specular="0.3 0.3 0.3", castshadow="true")
        ET.SubElement(world, "light", name="fill", pos="-1.5 2.5 3", dir="0.3 -0.5 -0.7", directional="true", diffuse="0.35 0.35 0.4", castshadow="false")
        ext = float(model.meta.get("scene_extent", 2.4))
        tgt = np.array([float(model.meta.get("cam_target_x", 0.0)), float(model.meta.get("wall_y", 0.0)), float(model.meta.get("cam_target_z", 1.0))])
        u_ = float(model.meta.get("u", 1.0) or 1.0)
        face=float(model.meta.get('approach_face',-1))
        for name, pos, fov in (("robot_view", (tgt[0], tgt[1] + face * 1.25 * ext, tgt[2] + 0.35 * ext), 52), ("far_view", (tgt[0], tgt[1] - face * 1.25 * ext, tgt[2] + 0.35 * ext), 52), ("iso", (tgt[0] + 0.95 * ext * u_, tgt[1] + face * 1.05 * ext, tgt[2] + 0.55 * ext), 46)):
            r_, up_ = _look_at(np.array(pos), tgt)
            ET.SubElement(world, "camera", name=name, pos=_f(pos), xyaxes=_f(list(r_) + list(up_)), fovy=str(fov))
        if model.meta.get("handle_cam_pos") and model.meta.get("handle_cam_target"):
            hpos = np.array([float(v) for v in model.meta["handle_cam_pos"]])
            htgt = np.array([float(v) for v in model.meta["handle_cam_target"]])
        else:
            hpos = np.array([float(model.meta.get("handle_cam_x", 0.3)), float(model.meta.get("wall_y", 0.0)) - 0.9, float(model.meta.get("handle_height", 1.0)) + 0.15])
            htgt = np.array([float(model.meta.get("handle_cam_x", 0.3)), 0.0, float(model.meta.get("handle_height", 1.0))])
        r_, up_ = _look_at(hpos, htgt)
        ET.SubElement(world, "camera", name="detail_handle", pos=_f(hpos), xyaxes=_f(list(r_) + list(up_)), fovy="38")
    xml_bodies = {}
    site_elems = {}

    def emit_body(b: Body, parent_elem):
        if b.static and b.name not in fixed_names:
            # A compiler-only frame preserves the complete static transform
            # for geoms, sites AND articulated descendants. It adds no body
            # mass or parent collision exclusion to the compiled model.
            if any(abs(float(p))>1e-12 for p in b.pos) or any(abs(a-c)>1e-9 for a,c in zip(b.quat,(1,0,0,0))):
                parent_elem=ET.SubElement(parent_elem,'frame',pos=_f(b.pos),quat=_f(b.quat))
            # Emit geometry on the retained parent so world-exemption applies.
            for g in b.geoms:
                if tier in g.tiers:
                    _geom_xml(parent_elem, g, tier, mesh_dir_rel, mats)
            for s in b.sites:
                if tier in s.tiers:
                    se = ET.SubElement(parent_elem, "site", name=s.name, pos=_f(s.pos), quat=_f(s.quat), size=_f(s.size), rgba="0.2 0.9 0.2 0.5", group="4")
            xml_bodies[b.name] = parent_elem
            return parent_elem
        e = ET.SubElement(parent_elem, "body", name=b.name, pos=_f(b.pos))
        if any(abs(a - c) > 1e-9 for a, c in zip(b.quat, (1, 0, 0, 0))):
            e.set("quat", _f(b.quat))
        if not b.static:
            _inertial_xml(e, b, tier)
        if b.joint is not None:
            _joint_xml(e, b.joint)
        for g in b.geoms:
            if tier in g.tiers:
                _geom_xml(e, g, tier, mesh_dir_rel, mats)
        for s in b.sites:
            if tier in s.tiers:
                ET.SubElement(e, "site", name=s.name, pos=_f(s.pos), quat=_f(s.quat), size=_f(s.size), rgba="0.2 0.9 0.2 0.5", group="4")
        xml_bodies[b.name] = e
        return e

    def recurse(parent_name, parent_elem):
        for b in bodies:
            if b.parent == parent_name:
                e = emit_body(b, parent_elem)
                recurse(b.name, e)

    recurse(None, world)
    body_names = {b.name for b in bodies}
    joint_names = {b.joint.name for b in bodies if b.joint is not None}
    # tendons
    tend = [t for t in model.tendons if tier in t.tiers and all(j in joint_names for j, _ in t.sites)]
    springs = [s for s in model.spatial_springs if tier in s.tiers]
    cables = [c for c in model.spatial_cables if tier in c.tiers]
    if tend or springs or cables:
        te = ET.SubElement(root, "tendon")
        for t in tend:
            fe = ET.SubElement(te, "fixed", name=t.name, limited="true", range=_f(t.range))
            if t.stiffness:
                fe.set("stiffness", _f(t.stiffness))
            if t.damping:
                fe.set("damping", _f(t.damping))
            for j, c in t.sites:
                ET.SubElement(fe, "joint", joint=j, coef=_f(c))
        available_sites = {s.name for b in bodies for s in b.sites if tier in s.tiers}
        for spring in springs:
            if not set(spring.sites) <= available_sites:
                raise ValueError(f"Missing spatial spring endpoint: {spring.name}")
            se = ET.SubElement(te, "spatial", name=spring.name, limited="false",
                stiffness=_f(spring.stiffness), springlength=_f(spring.springlength),
                damping=_f(spring.damping), width=_f(spring.width))
            for name in spring.sites:
                ET.SubElement(se, "site", site=name)
        available_geoms = {g.name:g for b in bodies for g in b.geoms if tier in g.tiers}
        for cable in cables:
            try:
                cable.validate_path(available_sites,available_geoms)
            except AssertionError as exc:
                raise ValueError(f"Invalid spatial cable {cable.name}: {exc}") from exc
            ce = ET.SubElement(te,"spatial",name=cable.name,limited="true",range=_f((0.,cable.max_length)),
                width=_f(cable.width),solreflimit="0.002 1",solimplimit="0.99 0.999 0.001")
            for point in cable.path:
                ET.SubElement(ce,"site" if 'site' in point else "geom",**point)
    # equalities
    eqs = [q for q in model.equalities if tier in q.tiers]
    eq_elems = []
    for q in eqs:
        if q.kind == "joint" and q.a in joint_names and q.b in joint_names:
            eq_elems.append(("joint", q))
        elif q.kind == "connect" and q.a in body_names and (q.b in body_names or q.b == "world"):
            eq_elems.append(("connect", q))
        elif q.kind == "weld" and q.a in body_names and (q.b in body_names or q.b == "world"):
            eq_elems.append(("weld", q))
    if eq_elems:
        ee = ET.SubElement(root, "equality")
        for kind, q in eq_elems:
            solver = {}
            if q.solref is not None:
                solver['solref'] = _f(q.solref)
            if q.solimp is not None:
                solver['solimp'] = _f(q.solimp)
            if kind == "joint":
                ET.SubElement(ee, "joint", name=q.name, joint1=q.a, joint2=q.b, polycoef=_f(q.polycoeff), active="true" if q.active else "false", **solver)
            elif kind == "connect":
                b2 = q.b if q.b in body_names and (not model.body(q.b).static or q.b in fixed_names) else "world"
                ET.SubElement(ee, "connect", name=q.name, body1=q.a, body2=b2, anchor=_f(q.anchor), active="true" if q.active else "false", **solver)
            else:
                b2 = q.b if q.b in body_names and (not model.body(q.b).static or q.b in fixed_names) else "world"
                ET.SubElement(ee, "weld", name=q.name, body1=q.a, body2=b2, active="true" if q.active else "false", **solver)
    # contact excludes
    ex = [(a, b) for a, b in model.contact_excludes if a in body_names and b in body_names
          and (not model.body(a).static or a in fixed_names) and (not model.body(b).static or b in fixed_names)]
    pairs=model.meta.get('native_contact_pairs',[])
    if not isinstance(pairs,list):raise ValueError('native_contact_pairs must be a list')
    geoms={g.name:g for b in model.bodies if b.name in body_names for g in b.geoms if tier in g.tiers}
    encoded=[];seen=set()
    for pair in pairs:
        if not isinstance(pair,dict) or set(pair)-{'geom1','geom2','solref','solimp','friction'}:
            raise ValueError('Invalid native contact pair fields')
        names=(pair.get('geom1'),pair.get('geom2'))
        if any(not isinstance(n,str) or n not in geoms or not geoms[n].collision for n in names) or names[0]==names[1]:
            raise ValueError('Native contact pairs require two distinct collision geoms in every exported tier')
        key=tuple(sorted(names))
        if key in seen:raise ValueError('Duplicate native contact pair')
        seen.add(key);attrs={'geom1':names[0],'geom2':names[1]}
        for field,count in (('solref',2),('solimp',3),('friction',5)):
            if field not in pair:continue
            values=pair[field]
            if (not isinstance(values,(tuple,list)) or len(values)!=count or
                any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values)):
                raise ValueError(f'Invalid native contact pair {field}')
            if field=='solref' and not all(v>0 for v in values):raise ValueError('Contact pair solref requires positive time constant/damping')
            if field=='solimp' and not (0<values[0]<=values[1]<1 and values[2]>0):raise ValueError('Invalid contact pair impedance')
            if field=='friction' and any(v<0 for v in values):raise ValueError('Contact pair friction must be nonnegative')
            attrs[field]=_f(values)
        encoded.append(attrs)
    if ex or encoded:
        ce = ET.SubElement(root, "contact")
        for a, b in ex:
            ET.SubElement(ce, "exclude", body1=a, body2=b)
        for pair in encoded:
            # Explicit collision pairs restore only the authored stop/seat
            # contacts which native parent filtering would otherwise omit.
            ET.SubElement(ce,'pair',**pair)
    # actuators (position servos for automatic doors / general-purpose joint motors on primary joints)
    act = model.meta.get("actuators", [])
    act = [a for a in act if a["joint"] in joint_names]
    if act:
        ae = ET.SubElement(root, "actuator")
        for a in act:
            if a.get("kind") == "position":
                ET.SubElement(ae, "position", name=a["name"], joint=a["joint"], kp=_f(a.get("kp", 200)), kv=_f(a.get("kv", 20)), forcerange=_f(a.get("forcerange", (-150, 150))), ctrlrange=_f(a.get("ctrlrange", (0, 1))))
            else:
                ET.SubElement(ae, "motor", name=a["name"], joint=a["joint"], gear=_f(a.get("gear", 1)), ctrlrange=_f(a.get("ctrlrange", (-100, 100))), ctrllimited="true")
    # sensors: joint positions/velocities for primary + operator joints, touch on grip sites optional
    se = ET.SubElement(root, "sensor")
    for b in bodies:
        if b.joint is not None and b.joint.role in ("primary", "operator", "lock", "latch", "secondary"):
            ET.SubElement(se, "jointpos", name=f"{b.joint.name}_pos", joint=b.joint.name)
            ET.SubElement(se, "jointvel", name=f"{b.joint.name}_vel", joint=b.joint.name)
    return root


def write_mjcf(model: Model, out_dir: str, tiers=("full", "simple", "minimal"), mesh_dir_rel="../../hardware", texture_dir_rel="../../textures"):
    os.makedirs(out_dir, exist_ok=True)
    names = {"full": "door.xml", "simple": "door_simple.xml", "minimal": "door_minimal.xml"}
    out = {}
    for tier in tiers:
        root = build_mjcf(model, tier, mesh_dir_rel, texture_dir_rel)
        # remove empty keyframe placeholders
        for kf in root.findall("keyframe"):
            if len(kf) == 0:
                root.remove(kf)
        txt = _pretty(root)
        path = os.path.join(out_dir, names[tier])
        with open(path, "w") as f:
            f.write(txt)
        out[tier] = path
    # scene file
    scene = ET.Element("mujoco", model=f"{model.name}_scene")
    ET.SubElement(scene, "include", file="door.xml")
    ET.SubElement(scene, "statistic", center="0 0 1", extent="3")
    with open(os.path.join(out_dir, "scene.xml"), "w") as f:
        f.write(_pretty(scene))
    return out

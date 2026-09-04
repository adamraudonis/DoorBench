"""USD exporter (UsdPhysics + PhysX-style attributes for Isaac Sim / Isaac Lab).

Structure:
  /World/<door_id>              (Xform, articulation root on the moving subtree)
    /World/<door_id>/Env        static collision geometry (frame, walls, floor)
    /World/<door_id>/<body>     RigidBody Xforms with MassAPI, geometry children
    /World/<door_id>/Joints/<joint>  UsdPhysics Revolute/Prismatic joints with limits,
                                     drives (springs) and physxJoint:jointFriction.
  Couplings are emitted as PhysxMimicJointAPI-style attributes plus a JSON
  string attribute `doorbench:couplings` on the root.

Materials are UsdPreviewSurface (diffuseColor / roughness / metallic / opacity).
Shared hardware meshes are written once to <hardware_dir>/<key>.usdc and
referenced.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..ir import Model, Body, Geom, quat_to_mat, quat_mul, quat_conj, quat_rotate, mat_to_quat


def _gf(pxr):
    return pxr.Gf


def write_usd(model: Model, out_dir: str, hardware_dir: str, tier: str = "full", filename: str = "door.usda"):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Vt
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_path = Sdf.Path(f"/World/{model.name}")
    world = UsdGeom.Xform.Define(stage, "/World")
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(world.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    # materials
    mat_paths = {}
    looks = UsdGeom.Scope.Define(stage, root_path.AppendChild("Looks"))
    for name, m in model.materials.items():
        mp = looks.GetPath().AppendChild(name)
        mat = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, mp.AppendChild("Shader"))
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in m.rgba[:3]]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(m.roughness))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(m.metallic))
        opacity = float(m.rgba[3]) if len(m.rgba) > 3 else 1.0
        if m.transparent and opacity > 0.6:
            opacity = 0.45
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        if any(m.emissive):
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in m.emissive]))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        mat_paths[name] = mat
    bodies = model.bodies_in_tier(tier)
    body_paths = {}
    # world transforms at q=0
    wt = {}

    def world_tf(b: Body):
        if b.name in wt:
            return wt[b.name]
        if b.parent is None:
            pos, quat = np.asarray(b.pos, float), np.asarray(b.quat, float)
        else:
            ppos, pquat = world_tf(model.body(b.parent))
            pos = ppos + quat_rotate(pquat, b.pos)
            quat = quat_mul(pquat, np.asarray(b.quat))
        wt[b.name] = (pos, quat)
        return wt[b.name]

    def set_xform(xf, pos, quat):
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(p) for p in pos]))
        w, x, y, z = [float(c) for c in quat]
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    hw_written = {}

    def add_geom(parent_path, g: Geom, is_static):
        gp = parent_path.AppendChild(_safe(g.name))
        if g.type == "box":
            prim = UsdGeom.Cube.Define(stage, gp)
            prim.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*[float(p) for p in g.pos]))
            w, x, y, z = [float(c) for c in g.quat]
            xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            xf.AddScaleOp().Set(Gf.Vec3f(*[float(2 * s) for s in g.size]))
        elif g.type in ("cylinder", "capsule"):
            cls = UsdGeom.Cylinder if g.type == "cylinder" else UsdGeom.Capsule
            prim = cls.Define(stage, gp)
            prim.CreateRadiusAttr(float(g.size[0]))
            prim.CreateHeightAttr(float(2 * g.size[1]))
            prim.CreateAxisAttr("Z")
            xf = UsdGeom.Xformable(prim)
            set_xform(xf, g.pos, g.quat)
        elif g.type == "sphere":
            prim = UsdGeom.Sphere.Define(stage, gp)
            prim.CreateRadiusAttr(float(g.size[0]))
            xf = UsdGeom.Xformable(prim)
            set_xform(xf, g.pos, g.quat)
        elif g.type == "mesh":
            # reference shared mesh file
            mesh_path = os.path.join(hardware_dir, f"{g.mesh_name}.usdc")
            if g.mesh_name not in hw_written:
                if not os.path.exists(mesh_path):
                    _write_mesh_usd(g.mesh, mesh_path, g.mesh_name)
                hw_written[g.mesh_name] = True
            prim = UsdGeom.Xform.Define(stage, gp)
            xf = UsdGeom.Xformable(prim)
            set_xform(xf, g.pos, g.quat)
            rel = os.path.relpath(mesh_path, out_dir)
            prim.GetPrim().GetReferences().AddReference(rel, Sdf.Path(f"/{g.mesh_name}"))
        else:
            return None
        p = prim.GetPrim()
        if g.material in mat_paths:
            UsdShade.MaterialBindingAPI.Apply(p).Bind(mat_paths[g.material])
        if g.collision:
            UsdPhysics.CollisionAPI.Apply(p)
            if g.type == "mesh":
                mc = UsdPhysics.MeshCollisionAPI.Apply(p)
                mc.CreateApproximationAttr("convexHull")
            p.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.005)
            p.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
            # physics material (friction)
            pm_name = f"PhysMat_{g.material}"
            pmp = looks.GetPath().AppendChild(pm_name)
            if not stage.GetPrimAtPath(pmp):
                pm = UsdShade.Material.Define(stage, pmp)
                api = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
                api.CreateStaticFrictionAttr(float(g.friction[0]))
                api.CreateDynamicFrictionAttr(float(g.friction[0]) * 0.85)
                api.CreateRestitutionAttr(0.05)
            UsdShade.MaterialBindingAPI.Apply(p).Bind(UsdShade.Material(stage.GetPrimAtPath(pmp)), UsdShade.Tokens.weakerThanDescendants, "physics")
        if not g.visual:
            UsdGeom.Imageable(p).CreatePurposeAttr("guide")
        p.CreateAttribute("doorbench:semantic", Sdf.ValueTypeNames.String).Set(g.semantic)
        if g.part_label:
            p.CreateAttribute("doorbench:label", Sdf.ValueTypeNames.String).Set(g.part_label)
        return p

    # bodies
    for b in bodies:
        bp = root_path.AppendChild(_safe(b.name))
        xf = UsdGeom.Xform.Define(stage, bp)
        pos, quat = world_tf(b)
        set_xform(xf, pos, quat)
        body_paths[b.name] = bp
        prim = xf.GetPrim()
        prim.CreateAttribute("doorbench:semantic", Sdf.ValueTypeNames.String).Set(b.semantic)
        if not b.static:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            m, com, I = b.inertial(tier)
            if m <= 1e-9:
                m, com, I = 0.01, np.zeros(3), np.eye(3) * 1e-6
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(float(m))
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*[float(c) for c in com]))
            w_, V = np.linalg.eigh(I)
            if np.linalg.det(V) < 0:
                V[:, 0] *= -1
            q = mat_to_quat(V)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(max(x, 1e-9)) for x in w_]))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
            prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
            prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
        else:
            UsdPhysics.CollisionAPI.Apply(prim) if False else None
        for g in b.geoms:
            if tier in g.tiers:
                add_geom(bp, g, b.static)
        for s in b.sites:
            if tier in s.tiers:
                sp = bp.AppendChild(_safe(s.name))
                sx = UsdGeom.Xform.Define(stage, sp)
                set_xform(sx, s.pos, s.quat)
                sx.GetPrim().CreateAttribute("doorbench:site_role", Sdf.ValueTypeNames.String).Set(s.role)
    # articulation root on the first moving root body chain (all moving bodies hang off world)
    moving_roots = [b for b in bodies if not b.static and (b.parent is None or model.body(b.parent).static)]
    art = UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    root.GetPrim().CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    # joints
    jscope = UsdGeom.Scope.Define(stage, root_path.AppendChild("Joints"))
    couplings = []
    for b in bodies:
        if b.static:
            continue
        parent = b.parent
        parent_static = parent is None or model.body(parent).static
        jp = jscope.GetPath().AppendChild(_safe(b.joint.name if b.joint else f"{b.name}_fixed"))
        if b.joint is None:
            fj = UsdPhysics.FixedJoint.Define(stage, jp)
            if not parent_static:
                fj.CreateBody0Rel().SetTargets([body_paths[parent]])
            fj.CreateBody1Rel().SetTargets([body_paths[b.name]])
            # frames: joint at child origin
            fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            fj.CreateLocalRot1Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
            if not parent_static:
                fj.CreateLocalPos0Attr(Gf.Vec3f(*[float(p) for p in b.pos]))
                fj.CreateLocalRot0Attr(Gf.Quatf(float(b.quat[0]), Gf.Vec3f(*[float(c) for c in b.quat[1:]])))
            else:
                pos, quat = world_tf(b)
                fj.CreateLocalPos0Attr(Gf.Vec3f(*[float(p) for p in pos]))
                fj.CreateLocalRot0Attr(Gf.Quatf(float(quat[0]), Gf.Vec3f(*[float(c) for c in quat[1:]])))
            continue
        jt = b.joint
        cls = UsdPhysics.RevoluteJoint if jt.type == "hinge" else UsdPhysics.PrismaticJoint
        j = cls.Define(stage, jp)
        if not parent_static:
            j.CreateBody0Rel().SetTargets([body_paths[parent]])
        j.CreateBody1Rel().SetTargets([body_paths[b.name]])
        # joint frame: rotate local X onto the joint axis (UsdPhysics joints use axis token; we use X)
        axis = np.asarray(jt.axis, float)
        axis = axis / np.linalg.norm(axis)
        qa = _quat_x_to(axis)
        j.CreateAxisAttr("X")
        jpos = np.asarray(jt.pos, float)
        # child (body1) frame
        j.CreateLocalPos1Attr(Gf.Vec3f(*[float(p) for p in jpos]))
        j.CreateLocalRot1Attr(Gf.Quatf(float(qa[0]), Gf.Vec3f(*[float(c) for c in qa[1:]])))
        # parent (body0) frame
        if not parent_static:
            p0 = np.asarray(b.pos) + quat_rotate(b.quat, jpos)
            q0 = quat_mul(np.asarray(b.quat), qa)
        else:
            pos, quat = world_tf(b)
            p0 = pos + quat_rotate(quat, jpos)
            q0 = quat_mul(quat, qa)
        j.CreateLocalPos0Attr(Gf.Vec3f(*[float(p) for p in p0]))
        j.CreateLocalRot0Attr(Gf.Quatf(float(q0[0]), Gf.Vec3f(*[float(c) for c in q0[1:]])))
        if jt.range is not None:
            lo, hi = jt.range[0] - jt.modeled_at, jt.range[1] - jt.modeled_at
            if jt.type == "hinge":
                j.CreateLowerLimitAttr(float(math.degrees(lo)))
                j.CreateUpperLimitAttr(float(math.degrees(hi)))
            else:
                j.CreateLowerLimitAttr(float(lo))
                j.CreateUpperLimitAttr(float(hi))
        prim = j.GetPrim()
        prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(float(jt.frictionloss))
        prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(float(jt.armature))
        # drive: spring (stiffness) + damping.  Angular drive units: stiffness in N*m/deg? PhysX uses N*m/rad for angular when
        # authored via UsdPhysics? UsdPhysics DriveAPI angular stiffness is in (N*m)/deg... We author in per-degree for angular.
        drive_tok = "angular" if jt.type == "hinge" else "linear"
        drv = UsdPhysics.DriveAPI.Apply(prim, drive_tok)
        drv.CreateTypeAttr("force")
        if jt.type == "hinge":
            drv.CreateStiffnessAttr(float(jt.stiffness) * math.pi / 180.0)
            drv.CreateDampingAttr(float(jt.damping) * math.pi / 180.0)
            drv.CreateTargetPositionAttr(float(math.degrees(jt.springref - jt.modeled_at)) if jt.stiffness else 0.0)
        else:
            drv.CreateStiffnessAttr(float(jt.stiffness))
            drv.CreateDampingAttr(float(jt.damping))
            drv.CreateTargetPositionAttr(float(jt.springref - jt.modeled_at) if jt.stiffness else 0.0)
        drv.CreateMaxForceAttr(1e6)
        if not jt.stiffness:
            drv.CreateStiffnessAttr(0.0)
        prim.CreateAttribute("doorbench:role", Sdf.ValueTypeNames.String).Set(jt.role)
        prim.CreateAttribute("doorbench:label", Sdf.ValueTypeNames.String).Set(jt.label)
        prim.CreateAttribute("doorbench:initial", Sdf.ValueTypeNames.Float).Set(float(jt.initial))
        prim.CreateAttribute("doorbench:zero_offset", Sdf.ValueTypeNames.Float).Set(float(jt.modeled_at))
        if jt.damping_closing is not None:
            prim.CreateAttribute("doorbench:damping_closing", Sdf.ValueTypeNames.Float).Set(float(jt.damping_closing))
            prim.CreateAttribute("doorbench:damping_opening", Sdf.ValueTypeNames.Float).Set(float(jt.damping_opening or 0.0))
        if jt.ratchet_one_way:
            prim.CreateAttribute("doorbench:ratchet_one_way", Sdf.ValueTypeNames.Bool).Set(True)
        if jt.notes:
            prim.CreateAttribute("doorbench:notes", Sdf.ValueTypeNames.String).Set(jt.notes)
    # couplings (mimic) as metadata + PhysxMimicJointAPI-style attributes on the driven joint
    joint_paths = {b.joint.name: jscope.GetPath().AppendChild(_safe(b.joint.name)) for b in bodies if b.joint is not None}
    for q in model.equalities:
        if tier not in q.tiers:
            continue
        if q.kind == "joint" and q.a in joint_paths and q.b in joint_paths:
            couplings.append({"type": "polynomial", "driven": q.a, "driver": q.b, "coeff": list(q.polycoeff), "label": q.label})
            p = stage.GetPrimAtPath(joint_paths[q.a])
            p.CreateAttribute("physxMimicJoint:rotX:referenceJoint", Sdf.ValueTypeNames.String).Set(str(joint_paths[q.b]))
            p.CreateAttribute("physxMimicJoint:rotX:gearing", Sdf.ValueTypeNames.Float).Set(float(-q.polycoeff[1]))
            p.CreateAttribute("physxMimicJoint:rotX:offset", Sdf.ValueTypeNames.Float).Set(float(q.polycoeff[0]))
        elif q.kind == "connect":
            couplings.append({"type": "loop_closure_point", "body1": q.a, "body2": q.b, "anchor": list(q.anchor), "label": q.label})
        elif q.kind == "weld":
            couplings.append({"type": "weld", "body1": q.a, "body2": q.b, "label": q.label, "active": q.active})
    for t in model.tendons:
        if tier in t.tiers:
            couplings.append({"type": "one_sided_tendon", "terms": [list(x) for x in t.sites], "range": list(t.range), "label": t.label})
    root.GetPrim().CreateAttribute("doorbench:couplings", Sdf.ValueTypeNames.String).Set(json.dumps(couplings))
    root.GetPrim().CreateAttribute("doorbench:meta", Sdf.ValueTypeNames.String).Set(json.dumps({k: v for k, v in model.meta.items() if k not in ("notes",)}, default=str))
    stage.GetRootLayer().Save()
    return path


def _safe(name: str) -> str:
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if out and out[0].isdigit():
        out = "_" + out
    return out or "prim"


def _quat_x_to(direction):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    x = np.array([1.0, 0, 0])
    c = float(np.dot(x, d))
    if c > 1 - 1e-9:
        return np.array([1.0, 0, 0, 0])
    if c < -1 + 1e-9:
        return np.array([0.0, 0, 0, 1.0])
    ax = np.cross(x, d)
    ax = ax / np.linalg.norm(ax)
    ang = math.acos(c)
    return np.array([math.cos(ang / 2), *(ax * math.sin(ang / 2))])


def _write_mesh_usd(mesh, path, name):
    from pxr import Usd, UsdGeom, Vt, Gf, Sdf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    xf = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(xf.GetPrim())
    m = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    m.CreateSubdivisionSchemeAttr("none")
    ext = np.asarray(mesh.bounds, dtype=np.float32)
    m.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(ext))
    stage.GetRootLayer().Save()
    return path

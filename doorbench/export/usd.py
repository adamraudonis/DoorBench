"""USD exporter (UsdPhysics + PhysX schema attributes for Isaac Sim / Isaac Lab).

Two files are written per door:

``door.usda`` (full fidelity, every mechanism body)
  /<door_id>                      default prim (Xform); doorbench:door_id / doorbench:meta / doorbench:couplings
    /<door_id>/Looks              UsdPreviewSurface materials + physics materials (friction)
    /<door_id>/Env                static collision geometry (frame, walls, floor, strikes) + benchmark sites
    /<door_id>/Articulation       PhysicsArticulationRootAPI + PhysxArticulationAPI (fixed base)
      /base                       massless-ish fixed link, FixedJoint to the world (Joints/base_fixed)
      /<body>                     one RigidBody Xform per moving body (MassAPI: mass, COM, principal inertia)
      /Joints/<joint>             Revolute / Prismatic / Fixed joints: limits, drives (springs, closers),
                                  PhysxJointAxisAPI friction efforts (Coulomb torque / force), armature,
                                  PhysxMimicJointAPI couplings (thumbturn -> deadbolt, wheel -> bolts ...)
  /PhysicsScene                   outside the default prim: present when the file is opened standalone,
                                  not brought in when the file is referenced by Isaac Lab

``door_rl.usda`` (canonical articulation for vectorised RL, see ``write_usd_rl``)
  Every door has the SAME link and joint names and the same joint types, so Isaac Lab's
  ``MultiUsdFileCfg`` can spawn a different door in every environment of one ``Articulation``
  (PhysX articulation views require homogeneous articulations).

Conventions
  * Z up, metres, kilograms, seconds.  Revolute limits / drive targets in degrees (UsdPhysics), prismatic in metres.
  * USD joint value 0 is the spec's initial state (``bake_initial`` makes the authored geometry the initial
    configuration); ``doorbench:zero_offset`` on a joint is the MJCF ``ref`` offset (DoorBench q = usd_q + offset).
  * Positive joint values mean "opening" / "actuating" (as in the MJCF).
  * Drives are ``force`` drives with UsdPhysics units (angular stiffness in N*m/deg); Isaac Lab reads them back in
    N*m/rad.  Coulomb joint friction is exported as ``physxJointAxis:<axis>:staticFrictionEffort`` /
    ``dynamicFrictionEffort`` (PhysX >= 5.6 / Isaac Sim >= 5.0, torque or force units) and, for older PhysX, as the
    legacy unitless ``physxJoint:jointFriction`` coefficient (Coulomb torque divided by an estimate of the joint
    reaction force).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..ir import Model, Body, Geom, quat_to_mat, quat_mul, quat_conj, quat_rotate, mat_to_quat, quat_from_axis_angle

G_ACC = 9.81
# canonical structure of door_rl.usda
RL_LINKS = ("base", "carriage", "leaf", "operator_pivot", "operator", "latch", "carriage2", "leaf2")
RL_JOINTS = (
    # name, type, parent link, child link
    ("base_fixed", "fixed", None, "base"),
    ("door_slide", "prismatic", "base", "carriage"),
    ("door_hinge", "revolute", "carriage", "leaf"),
    ("operator_hinge", "revolute", "leaf", "operator_pivot"),
    ("operator_slide", "prismatic", "operator_pivot", "operator"),
    ("latch_slide", "prismatic", "leaf", "latch"),
    ("leaf2_slide", "prismatic", "base", "carriage2"),
    ("leaf2_hinge", "revolute", "carriage2", "leaf2"),
)
RL_DOF_JOINTS = tuple(j[0] for j in RL_JOINTS if j[1] != "fixed")
# locked ("dummy") joints: tiny symmetric range + stiff drive
LOCK_RANGE_M = 5e-4
LOCK_RANGE_DEG = 0.05
LOCK_STIFF_LIN = 2.0e4      # N/m
LOCK_STIFF_ANG = 2.0e3      # N*m/rad
LOCK_DAMP_LIN = 2.0e2
LOCK_DAMP_ANG = 2.0e1


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _safe(name: str) -> str:
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if out and out[0].isdigit():
        out = "_" + out
    return out or "prim"


def _quat_x_to(direction):
    """Quaternion (w,x,y,z) rotating local +X onto `direction`."""
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


def _axis_instance(jtype: str) -> str:
    """PhysX per-axis schema instance name for a joint whose axis token is X."""
    return "rotX" if jtype in ("hinge", "revolute") else "transX"


def _f(x):
    return float(x)


def _v3(Gf, v):
    return Gf.Vec3f(*[float(c) for c in v])


def _q(Gf, q):
    w, x, y, z = [float(c) for c in q]
    return Gf.Quatf(w, Gf.Vec3f(x, y, z))


def _set_xform(UsdGeom, Gf, prim, pos, quat, scale=None):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(p) for p in pos]))
    xf.AddOrientOp().Set(_q(Gf, quat))
    if scale is not None:
        xf.AddScaleOp().Set(_v3(Gf, scale))


def _compose(p_pos, p_quat, pos, quat):
    """Parent frame (p_pos, p_quat) composed with a child offset (pos, quat) -> world (pos, quat)."""
    return np.asarray(p_pos, float) + quat_rotate(p_quat, np.asarray(pos, float)), quat_mul(np.asarray(p_quat, float), np.asarray(quat, float))


def _relative(a_pos, a_quat, w_pos, w_quat):
    """Express a world frame (w_pos, w_quat) in frame A (a_pos, a_quat)."""
    qi = quat_conj(np.asarray(a_quat, float))
    return quat_rotate(qi, np.asarray(w_pos, float) - np.asarray(a_pos, float)), quat_mul(qi, np.asarray(w_quat, float))


def _joint_displacement(jt, dq):
    """(pos, quat) offset that moves a body by dq along/about its joint (in the body's own frame)."""
    axis = np.asarray(jt.axis, float)
    axis = axis / np.linalg.norm(axis)
    if jt.type == "slide":
        return axis * dq, np.array([1.0, 0, 0, 0])
    q = quat_from_axis_angle(axis, dq)
    jp = np.asarray(jt.pos, float)
    return jp - quat_rotate(q, jp), q


# ---------------------------------------------------------------------------
# stage-level writers shared by both exports
# ---------------------------------------------------------------------------
class _Writer:
    def __init__(self, model: Model, path: str, hardware_dir: str, out_dir: str):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Vt
        self.Usd, self.UsdGeom, self.UsdPhysics, self.UsdShade, self.Sdf, self.Gf, self.Vt = Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Vt
        self.model = model
        self.path = path
        self.hardware_dir = hardware_dir
        self.out_dir = out_dir
        self.stage = Usd.Stage.CreateNew(path)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.root_path = Sdf.Path(f"/{_safe(model.name)}")
        self.root = UsdGeom.Xform.Define(self.stage, self.root_path)
        self.stage.SetDefaultPrim(self.root.GetPrim())
        self.root.GetPrim().CreateAttribute("doorbench:door_id", Sdf.ValueTypeNames.String).Set(model.name)
        # physics scene outside the default prim (standalone opening only)
        scene = UsdPhysics.Scene.Define(self.stage, "/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(G_ACC)
        self.mat_paths = {}
        self.phys_mats = {}
        self.hw_written = {}
        self.looks = UsdGeom.Scope.Define(self.stage, self.root_path.AppendChild("Looks"))
        self._write_materials()

    # ---- materials ---------------------------------------------------------
    def _write_materials(self):
        UsdShade, Sdf, Gf = self.UsdShade, self.Sdf, self.Gf
        for name, m in self.model.materials.items():
            mp = self.looks.GetPath().AppendChild(_safe(name))
            mat = UsdShade.Material.Define(self.stage, mp)
            sh = UsdShade.Shader.Define(self.stage, mp.AppendChild("Shader"))
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
            self.mat_paths[name] = mat

    def _physics_material(self, g: Geom):
        UsdShade, UsdPhysics = self.UsdShade, self.UsdPhysics
        mu = float(g.friction[0]) if g.friction else 0.6
        key = f"PhysMat_{_safe(g.material)}_{mu:.3f}".replace(".", "_")
        if key not in self.phys_mats:
            pm = UsdShade.Material.Define(self.stage, self.looks.GetPath().AppendChild(key))
            api = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
            api.CreateStaticFrictionAttr(mu)
            api.CreateDynamicFrictionAttr(mu * 0.85)
            api.CreateRestitutionAttr(0.05)
            self.phys_mats[key] = pm
        return self.phys_mats[key]

    # ---- geometry ----------------------------------------------------------
    def add_geom(self, parent_path, g: Geom, pos=None, quat=None, name=None):
        """Write one geom prim under parent_path.  pos/quat override the geom's own local pose."""
        UsdGeom, UsdPhysics, UsdShade, Sdf, Gf = self.UsdGeom, self.UsdPhysics, self.UsdShade, self.Sdf, self.Gf
        pos = g.pos if pos is None else pos
        quat = g.quat if quat is None else quat
        gp = parent_path.AppendChild(_safe(name or g.name))
        if g.type == "box":
            prim = UsdGeom.Cube.Define(self.stage, gp)
            prim.CreateSizeAttr(1.0)
            _set_xform(UsdGeom, Gf, prim, pos, quat, scale=[2 * s for s in g.size])
        elif g.type in ("cylinder", "capsule"):
            cls = UsdGeom.Cylinder if g.type == "cylinder" else UsdGeom.Capsule
            prim = cls.Define(self.stage, gp)
            prim.CreateRadiusAttr(float(g.size[0]))
            prim.CreateHeightAttr(float(2 * g.size[1]))
            prim.CreateAxisAttr("Z")
            _set_xform(UsdGeom, Gf, prim, pos, quat)
        elif g.type == "sphere":
            prim = UsdGeom.Sphere.Define(self.stage, gp)
            prim.CreateRadiusAttr(float(g.size[0]))
            _set_xform(UsdGeom, Gf, prim, pos, quat)
        elif g.type == "mesh":
            mesh_path = os.path.join(self.hardware_dir, f"{g.mesh_name}.usdc")
            if g.mesh_name not in self.hw_written:
                if not os.path.exists(mesh_path):
                    _write_mesh_usd(g.mesh, mesh_path, g.mesh_name)
                self.hw_written[g.mesh_name] = True
            prim = UsdGeom.Xform.Define(self.stage, gp)
            _set_xform(UsdGeom, Gf, prim, pos, quat)
            rel = os.path.relpath(mesh_path, self.out_dir).replace(os.sep, "/")
            prim.GetPrim().GetReferences().AddReference(rel, Sdf.Path(f"/{g.mesh_name}"))
        else:
            return None
        p = prim.GetPrim()
        if g.material in self.mat_paths:
            UsdShade.MaterialBindingAPI.Apply(p).Bind(self.mat_paths[g.material])
        if g.collision:
            UsdPhysics.CollisionAPI.Apply(p)
            p.AddAppliedSchema("PhysxCollisionAPI")
            p.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.005)
            p.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
            if g.type == "mesh":
                # the referenced prim is an Xform holding a Mesh: MeshCollisionAPI on the referencing prim applies to
                # the mesh below it (Isaac Sim / PhysX descend into the subtree for collision shapes)
                mc = UsdPhysics.MeshCollisionAPI.Apply(p)
                mc.CreateApproximationAttr("convexHull")
                p.AddAppliedSchema("PhysxConvexHullCollisionAPI")
                p.CreateAttribute("physxConvexHullCollision:hullVertexLimit", Sdf.ValueTypeNames.Int).Set(64)
                p.CreateAttribute("physxConvexHullCollision:minThickness", Sdf.ValueTypeNames.Float).Set(0.001)
            pm = self._physics_material(g)
            UsdShade.MaterialBindingAPI.Apply(p).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
        if not g.visual:
            UsdGeom.Imageable(p).CreatePurposeAttr("guide")
        p.CreateAttribute("doorbench:semantic", Sdf.ValueTypeNames.String).Set(g.semantic)
        if g.part_label:
            p.CreateAttribute("doorbench:label", Sdf.ValueTypeNames.String).Set(g.part_label)
        p.CreateAttribute("doorbench:collision", Sdf.ValueTypeNames.Bool).Set(bool(g.collision))
        return p

    def add_site(self, parent_path, name, pos, quat, role):
        sp = parent_path.AppendChild(_safe(name))
        sx = self.UsdGeom.Xform.Define(self.stage, sp)
        _set_xform(self.UsdGeom, self.Gf, sx, pos, quat)
        sx.GetPrim().CreateAttribute("doorbench:site_role", self.Sdf.ValueTypeNames.String).Set(role)
        return sx

    # ---- rigid bodies ------------------------------------------------------
    def add_rigid_body(self, path, pos, quat, mass, com, I, semantic="", label=""):
        """RigidBody Xform with explicit mass properties (I is a 3x3 tensor about `com` in the body frame)."""
        UsdGeom, UsdPhysics, Sdf, Gf = self.UsdGeom, self.UsdPhysics, self.Sdf, self.Gf
        xf = UsdGeom.Xform.Define(self.stage, path)
        _set_xform(UsdGeom, Gf, xf, pos, quat)
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
        m = float(mass)
        I = np.asarray(I, float)
        if m <= 1e-9 or not np.all(np.isfinite(I)):
            m, com, I = 0.05, np.zeros(3), np.eye(3) * 1e-5
        w_, V = np.linalg.eigh(I)
        if np.linalg.det(V) < 0:
            V[:, 0] *= -1
        q = mat_to_quat(V)
        massapi = UsdPhysics.MassAPI.Apply(prim)
        massapi.CreateMassAttr(m)
        massapi.CreateCenterOfMassAttr(_v3(Gf, com))
        massapi.CreateDiagonalInertiaAttr(_v3(Gf, [max(float(x), 1e-9) for x in w_]))
        massapi.CreatePrincipalAxesAttr(_q(Gf, q))
        prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(5.0)
        prim.CreateAttribute("physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
        if semantic:
            prim.CreateAttribute("doorbench:semantic", Sdf.ValueTypeNames.String).Set(semantic)
        if label:
            prim.CreateAttribute("doorbench:label", Sdf.ValueTypeNames.String).Set(label)
        return prim

    def add_articulation_root(self, path):
        UsdGeom, UsdPhysics, Sdf = self.UsdGeom, self.UsdPhysics, self.Sdf
        art = UsdGeom.Xform.Define(self.stage, path)
        prim = art.GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        prim.AddAppliedSchema("PhysxArticulationAPI")
        prim.CreateAttribute("physxArticulation:articulationEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
        prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
        prim.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
        prim.CreateAttribute("physxArticulation:stabilizationThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
        return prim

    def add_base_link(self, art_path, joints_path):
        """Fixed base link + FixedJoint to the world (body0 unset = world frame)."""
        UsdPhysics, Gf = self.UsdPhysics, self.Gf
        base = self.add_rigid_body(art_path.AppendChild("base"), [0, 0, 0], [1, 0, 0, 0], 1.0, np.zeros(3), np.eye(3) * 0.01, semantic="base", label="Fixed base link")
        fj = UsdPhysics.FixedJoint.Define(self.stage, joints_path.AppendChild("base_fixed"))
        fj.CreateBody1Rel().SetTargets([base.GetPath()])
        fj.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        fj.CreateLocalRot0Attr(_q(Gf, [1, 0, 0, 0]))
        fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        fj.CreateLocalRot1Attr(_q(Gf, [1, 0, 0, 0]))
        fj.GetPrim().CreateAttribute("doorbench:role", self.Sdf.ValueTypeNames.String).Set("base")
        return base.GetPath()

    # ---- joints ------------------------------------------------------------
    def add_fixed_joint(self, path, body0_path, body1_path, pos0, rot0, pos1, rot1, label=""):
        UsdPhysics, Gf = self.UsdPhysics, self.Gf
        fj = UsdPhysics.FixedJoint.Define(self.stage, path)
        fj.CreateBody0Rel().SetTargets([body0_path])
        fj.CreateBody1Rel().SetTargets([body1_path])
        fj.CreateLocalPos0Attr(_v3(Gf, pos0))
        fj.CreateLocalRot0Attr(_q(Gf, rot0))
        fj.CreateLocalPos1Attr(_v3(Gf, pos1))
        fj.CreateLocalRot1Attr(_q(Gf, rot1))
        p = fj.GetPrim()
        p.CreateAttribute("doorbench:role", self.Sdf.ValueTypeNames.String).Set("fixed")
        if label:
            p.CreateAttribute("doorbench:label", self.Sdf.ValueTypeNames.String).Set(label)
        return p

    def add_dof_joint(self, path, jtype, body0_path, body1_path, pos0, rot0, pos1, rot1, lo, hi, *, stiffness=0.0, damping=0.0,
                      target=0.0, frictionloss=0.0, armature=0.0, reaction_force=None, max_force=1e6, extra=None):
        """Revolute (jtype 'hinge'/'revolute') or Prismatic joint with limits, drive and PhysX friction.

        All inputs in SI (rad, m, N*m/rad, N/m); converted to UsdPhysics units here.  The joint frame X axis is the
        joint axis (rot0/rot1 rotate X onto it).
        """
        UsdPhysics, Sdf, Gf = self.UsdPhysics, self.Sdf, self.Gf
        revolute = jtype in ("hinge", "revolute")
        cls = UsdPhysics.RevoluteJoint if revolute else UsdPhysics.PrismaticJoint
        j = cls.Define(self.stage, path)
        j.CreateBody0Rel().SetTargets([body0_path])
        j.CreateBody1Rel().SetTargets([body1_path])
        j.CreateAxisAttr("X")
        j.CreateLocalPos0Attr(_v3(Gf, pos0))
        j.CreateLocalRot0Attr(_q(Gf, rot0))
        j.CreateLocalPos1Attr(_v3(Gf, pos1))
        j.CreateLocalRot1Attr(_q(Gf, rot1))
        conv = (180.0 / math.pi) if revolute else 1.0
        j.CreateLowerLimitAttr(float(lo * conv))
        j.CreateUpperLimitAttr(float(hi * conv))
        prim = j.GetPrim()
        # drive: spring (stiffness) + damping in UsdPhysics units (per degree for angular drives)
        drv = UsdPhysics.DriveAPI.Apply(prim, "angular" if revolute else "linear")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(float(stiffness / conv))
        drv.CreateDampingAttr(float(damping / conv))
        drv.CreateTargetPositionAttr(float(target * conv))
        drv.CreateTargetVelocityAttr(0.0)
        drv.CreateMaxForceAttr(float(max_force))
        # PhysX joint attributes
        prim.AddAppliedSchema("PhysxJointAPI")
        prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(float(armature))
        # legacy (PhysX < 5.6) unitless coefficient: friction = coeff * |joint reaction force|
        coeff = 0.0
        if frictionloss > 0:
            coeff = float(frictionloss) / max(float(reaction_force or 0.0), 1.0)
        prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(float(min(coeff, 10.0)))
        inst = "rotX" if revolute else "transX"
        prim.AddAppliedSchema(f"PhysxJointAxisAPI:{inst}")
        prim.CreateAttribute(f"physxJointAxis:{inst}:staticFrictionEffort", Sdf.ValueTypeNames.Float).Set(float(frictionloss))
        prim.CreateAttribute(f"physxJointAxis:{inst}:dynamicFrictionEffort", Sdf.ValueTypeNames.Float).Set(float(frictionloss))
        prim.CreateAttribute(f"physxJointAxis:{inst}:viscousFrictionCoefficient", Sdf.ValueTypeNames.Float).Set(0.0)
        prim.CreateAttribute(f"physxJointAxis:{inst}:armature", Sdf.ValueTypeNames.Float).Set(float(armature))
        prim.CreateAttribute("doorbench:friction_effort", Sdf.ValueTypeNames.Float).Set(float(frictionloss))
        prim.CreateAttribute("doorbench:stiffness_si", Sdf.ValueTypeNames.Float).Set(float(stiffness))
        prim.CreateAttribute("doorbench:damping_si", Sdf.ValueTypeNames.Float).Set(float(damping))
        prim.CreateAttribute("doorbench:target_si", Sdf.ValueTypeNames.Float).Set(float(target))
        for k, v in (extra or {}).items():
            t = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}.get(type(v), Sdf.ValueTypeNames.String)
            prim.CreateAttribute(f"doorbench:{k}", t).Set(v if t != Sdf.ValueTypeNames.String else str(v))
        return prim

    def add_mimic(self, driven_path, driven_type, driver_path, driver_type, gearing, offset, label=""):
        """PhysxMimicJointAPI on the driven joint:  q_driven + gearing * q_driver + offset = 0  (PhysX units)."""
        Sdf = self.Sdf
        inst = _axis_instance(driven_type)
        p = self.stage.GetPrimAtPath(driven_path)
        p.AddAppliedSchema(f"PhysxMimicJointAPI:{inst}")
        p.CreateRelationship(f"physxMimicJoint:{inst}:referenceJoint").SetTargets([driver_path])
        p.CreateAttribute(f"physxMimicJoint:{inst}:referenceJointAxis", Sdf.ValueTypeNames.Token).Set(_axis_instance(driver_type))
        p.CreateAttribute(f"physxMimicJoint:{inst}:gearing", Sdf.ValueTypeNames.Float).Set(float(gearing))
        p.CreateAttribute(f"physxMimicJoint:{inst}:offset", Sdf.ValueTypeNames.Float).Set(float(offset))
        if label:
            p.CreateAttribute(f"doorbench:mimic_label", Sdf.ValueTypeNames.String).Set(label)

    def set_json(self, prim, name, obj):
        prim.CreateAttribute(name, self.Sdf.ValueTypeNames.String).Set(json.dumps(obj, default=_json_default))

    def save(self):
        self.stage.GetRootLayer().Save()
        return self.path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


# ---------------------------------------------------------------------------
# subtree helpers (world poses at q = 0 = initial state)
# ---------------------------------------------------------------------------
def _world_poses(model: Model, bodies):
    wt = {}

    def world_tf(b: Body):
        if b.name in wt:
            return wt[b.name]
        if b.parent is None:
            pos, quat = np.asarray(b.pos, float), np.asarray(b.quat, float)
        else:
            ppos, pquat = world_tf(model.body(b.parent))
            pos, quat = _compose(ppos, pquat, b.pos, b.quat)
        wt[b.name] = (pos, quat)
        return wt[b.name]

    for b in bodies:
        world_tf(b)
    return wt


def _subtree_mass(model: Model, root_name: str, tier: str, wt: dict, exclude=()):
    """Total mass and world COM of a body and its descendants (excluding named subtrees)."""
    total, acc = 0.0, np.zeros(3)
    stack = [root_name]
    while stack:
        n = stack.pop()
        if n in exclude:
            continue
        b = model.body(n)
        m, com, _ = b.inertial(tier)
        if m > 0:
            pos, quat = wt[n]
            acc += m * (pos + quat_rotate(quat, com))
            total += m
        stack += [c.name for c in model.bodies if c.parent == n]
    return total, (acc / total if total > 0 else np.zeros(3))


def _reaction_estimate(model: Model, b: Body, tier: str, wt: dict):
    """|gravity load| carried by the joint of body b (N): subtree weight (+ moment as a force at 1 m)."""
    m, com = _subtree_mass(model, b.name, tier, wt)
    pos, quat = wt[b.name]
    anchor = pos + quat_rotate(quat, np.asarray(b.joint.pos, float)) if b.joint is not None else pos
    r = com - anchor
    moment = m * G_ACC * float(np.hypot(r[0], r[1]))
    return m * G_ACC + moment


# ---------------------------------------------------------------------------
# full-fidelity export
# ---------------------------------------------------------------------------
def write_usd(model: Model, out_dir: str, hardware_dir: str, tier: str = "full", filename: str = "door.usda"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    W = _Writer(model, path, hardware_dir, out_dir)
    bodies = model.bodies_in_tier(tier)
    wt = _world_poses(model, bodies)
    env_path = W.root_path.AppendChild("Env")
    W.UsdGeom.Xform.Define(W.stage, env_path)
    art_path = W.root_path.AppendChild("Articulation")
    W.add_articulation_root(art_path)
    joints_path = art_path.AppendChild("Joints")
    W.UsdGeom.Scope.Define(W.stage, joints_path)
    base_path = W.add_base_link(art_path, joints_path)
    body_paths = {}
    # --- static bodies -> Env
    for b in bodies:
        if not b.static:
            continue
        bp = env_path.AppendChild(_safe(b.name))
        xf = W.UsdGeom.Xform.Define(W.stage, bp)
        pos, quat = wt[b.name]
        _set_xform(W.UsdGeom, W.Gf, xf, pos, quat)
        xf.GetPrim().CreateAttribute("doorbench:semantic", W.Sdf.ValueTypeNames.String).Set(b.semantic)
        for g in b.geoms:
            if tier in g.tiers:
                W.add_geom(bp, g)
        for s in b.sites:
            if tier in s.tiers:
                W.add_site(bp, s.name, s.pos, s.quat, s.role)
        body_paths[b.name] = bp
    # --- moving bodies -> Articulation
    for b in bodies:
        if b.static:
            continue
        bp = art_path.AppendChild(_safe(b.name))
        pos, quat = wt[b.name]
        m, com, I = b.inertial(tier)
        prim = W.add_rigid_body(bp, pos, quat, m, com, I, semantic=b.semantic, label=b.label)
        body_paths[b.name] = bp
        for g in b.geoms:
            if tier in g.tiers:
                W.add_geom(bp, g)
        for s in b.sites:
            if tier in s.tiers:
                W.add_site(bp, s.name, s.pos, s.quat, s.role)
    # --- joints
    joint_paths = {}
    joint_types = {}
    for b in bodies:
        if b.static:
            continue
        parent = b.parent
        parent_moving = parent is not None and not model.body(parent).static
        body0 = body_paths[parent] if parent_moving else base_path
        jname = _safe(b.joint.name if b.joint else f"{b.name}_fixed")
        jp = joints_path.AppendChild(jname)
        if b.joint is None:
            if parent_moving:
                p0, q0 = np.asarray(b.pos, float), np.asarray(b.quat, float)
            else:
                p0, q0 = wt[b.name]
            W.add_fixed_joint(jp, body0, body_paths[b.name], p0, q0, [0, 0, 0], [1, 0, 0, 0], label=f"{b.name} rigidly attached")
            continue
        jt = b.joint.for_tier(tier)
        axis = np.asarray(jt.axis, float)
        qa = _quat_x_to(axis)
        jpos = np.asarray(jt.pos, float)
        if parent_moving:
            p0, q0 = _compose(b.pos, b.quat, jpos, qa)
        else:
            pos, quat = wt[b.name]
            p0, q0 = _compose(pos, quat, jpos, qa)
        if jt.range is not None:
            lo, hi = jt.range[0] - jt.modeled_at, jt.range[1] - jt.modeled_at
        else:
            lo, hi = (-math.pi * 4, math.pi * 4) if jt.type == "hinge" else (-10.0, 10.0)
        extra = {"role": jt.role, "label": jt.label, "initial": float(jt.initial), "zero_offset": float(jt.modeled_at), "source_joint": jt.name,
                 "limited": jt.range is not None, "robot_interactive": bool(jt.robot_interactive)}
        if jt.damping_closing is not None:
            extra["damping_closing"] = float(jt.damping_closing)
            extra["damping_opening"] = float(jt.damping_opening or 0.0)
        if jt.backcheck_angle is not None:
            extra["backcheck_angle"] = float(jt.backcheck_angle)
            extra["backcheck_damping"] = float(jt.backcheck_damping or 0.0)
        if jt.ratchet_one_way:
            extra["ratchet_one_way"] = True
        if jt.notes:
            extra["notes"] = jt.notes
        W.add_dof_joint(jp, jt.type, body0, body_paths[b.name], p0, q0, jpos, qa, lo, hi,
                        stiffness=float(jt.stiffness), damping=float(jt.damping), target=(float(jt.springref - jt.modeled_at) if jt.stiffness else 0.0),
                        frictionloss=float(jt.frictionloss), armature=float(jt.armature), reaction_force=_reaction_estimate(model, b, tier, wt), extra=extra)
        joint_paths[jt.name] = jp
        joint_types[jt.name] = jt.type
    # --- couplings: mimic joints (bilateral polynomial equalities) + JSON for everything else
    couplings = []
    for q in model.equalities:
        if tier not in q.tiers:
            continue
        if q.kind == "joint" and q.a in joint_paths and q.b in joint_paths:
            c0, c1 = float(q.polycoeff[0]), float(q.polycoeff[1])
            # IR: q_a = c0 + c1 * q_b (USD coordinates)  ->  q_a + (-c1) * q_b + (-c0) = 0
            W.add_mimic(joint_paths[q.a], joint_types[q.a], joint_paths[q.b], joint_types[q.b], gearing=-c1, offset=-c0, label=q.label)
            couplings.append({"type": "mimic", "driven": q.a, "driver": q.b, "coeff": list(q.polycoeff), "gearing": -c1, "offset": -c0, "label": q.label,
                              "note": "PhysxMimicJointAPI on the driven joint; q_driven + gearing*q_driver + offset = 0 in rad / m"})
        elif q.kind == "connect" and q.a in body_paths:
            # loop closure: a spherical joint OUTSIDE the articulation tree (physxJoint:excludeFromArticulation) pins the
            # point `anchor` of body1 to body2 (the fixed base when body2 is the world / a static body); PhysX solves it
            # as a maximal-coordinate constraint between the two articulation links
            b2 = q.b if (q.b in body_paths and not model.body(q.b).static) else None
            path2 = body_paths[b2] if b2 else base_path
            pos1_w, quat1_w = wt[q.a]
            anchor_w = pos1_w + quat_rotate(quat1_w, np.asarray(q.anchor, float))
            if b2:
                pos2_w, quat2_w = wt[b2]
                lp2, _ = _relative(pos2_w, quat2_w, anchor_w, np.array([1.0, 0, 0, 0]))
            else:
                lp2 = anchor_w
            lp1, _ = _relative(pos1_w, quat1_w, anchor_w, np.array([1.0, 0, 0, 0]))
            jp_ = joints_path.AppendChild(_safe(q.name))
            sj = W.UsdPhysics.SphericalJoint.Define(W.stage, jp_)
            sj.CreateBody0Rel().SetTargets([body_paths[q.a]])
            sj.CreateBody1Rel().SetTargets([path2])
            sj.CreateLocalPos0Attr(_v3(W.Gf, lp1))
            sj.CreateLocalRot0Attr(_q(W.Gf, [1, 0, 0, 0]))
            sj.CreateLocalPos1Attr(_v3(W.Gf, lp2))
            sj.CreateLocalRot1Attr(_q(W.Gf, [1, 0, 0, 0]))
            sj.CreateExcludeFromArticulationAttr(True)
            pj_ = sj.GetPrim()
            pj_.AddAppliedSchema("PhysxJointAPI")
            pj_.CreateAttribute("physxJoint:enableProjection", W.Sdf.ValueTypeNames.Bool).Set(True)
            pj_.CreateAttribute("doorbench:role", W.Sdf.ValueTypeNames.String).Set("loop_closure")
            pj_.CreateAttribute("doorbench:label", W.Sdf.ValueTypeNames.String).Set(q.label)
            couplings.append({"type": "loop_closure_point", "body1": q.a, "body2": q.b, "anchor": list(q.anchor), "label": q.label, "usd_joint": str(jp_),
                              "note": "UsdPhysics SphericalJoint with physics:excludeFromArticulation = true (PhysX loop joint between two articulation links)"})
        elif q.kind == "weld":
            couplings.append({"type": "weld", "body1": q.a, "body2": q.b, "label": q.label, "active": q.active,
                              "note": "breakable weld (maglock): environment logic, see doorbench.benchmark.DoorEnv"})
    for t in model.tendons:
        if tier in t.tiers:
            couplings.append({"type": "one_sided_tendon", "terms": [list(x) for x in t.sites], "range": list(t.range), "label": t.label,
                              "note": "bolt_q >= scale*operator_q: drive the latch target from the operator in the environment (doorbench_isaaclab does this)"})
            # tag the latch joint with the coupling scale so environments can reproduce it
            try:
                latch_j, op_j = t.sites[0][0], t.sites[1][0]
                scale = -float(t.sites[1][1]) / float(t.sites[0][1])
                if latch_j in joint_paths:
                    p = W.stage.GetPrimAtPath(joint_paths[latch_j])
                    p.CreateAttribute("doorbench:latch_coupling_scale", W.Sdf.ValueTypeNames.Float).Set(scale)
                    p.CreateAttribute("doorbench:latch_coupling_joint", W.Sdf.ValueTypeNames.String).Set(op_j)
            except Exception:
                pass
    W.set_json(W.root.GetPrim(), "doorbench:couplings", couplings)
    meta = {k: v for k, v in model.meta.items() if k not in ("notes",)}
    meta["usd_layout"] = "v2: default prim = door root; Env (static) + Articulation (fixed base link `base`); loop closures = spherical joints excluded from the articulation"
    meta["joints"] = {name: str(p) for name, p in joint_paths.items()}
    meta["linkages"] = list(model.linkages)
    W.set_json(W.root.GetPrim(), "doorbench:meta", meta)
    return W.save()


# ---------------------------------------------------------------------------
# canonical RL export
# ---------------------------------------------------------------------------
def _subtree(model: Model, root_name: str, stop=()):
    """Names of root and all descendants (not descending into `stop` subtrees)."""
    out, stack = [], [root_name]
    while stack:
        n = stack.pop()
        if n in stop and n != root_name:
            continue
        out.append(n)
        stack += [c.name for c in model.bodies if c.parent == n]
    return out


def _welded_inertial(model: Model, names, frame_pos, frame_quat, wt, tier="full"):
    """Combine the inertials of several bodies (world poses wt) into one link frame."""
    total, acc = 0.0, np.zeros(3)
    items = []
    for n in names:
        b = model.body(n)
        m, com, I = b.inertial(tier)
        if m <= 0:
            continue
        pos, quat = wt[n]
        cw = pos + quat_rotate(quat, com)
        R = quat_to_mat(quat)
        Iw = R @ I @ R.T
        items.append((m, cw, Iw))
        acc += m * cw
        total += m
    if total <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com_w = acc / total
    Iw_tot = np.zeros((3, 3))
    for m, cw, Iw in items:
        d = cw - com_w
        Iw_tot += Iw + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    Rl = quat_to_mat(frame_quat)
    I_local = Rl.T @ Iw_tot @ Rl
    com_local = quat_rotate(quat_conj(frame_quat), com_w - frame_pos)
    return total, com_local, I_local


def _released_pose_shift(model: Model, b: Body):
    """Extra joint displacement to weld a latch/lock-like body in its released state."""
    jt = b.joint
    if jt is None or jt.range is None:
        return 0.0
    return float(jt.range[1] - jt.modeled_at)


def write_usd_rl(model: Model, out_dir: str, hardware_dir: str, filename: str = "door_rl.usda", spec: dict | None = None):
    """Canonical Isaac-Lab articulation: identical link/joint names & types for every door.

    Links   base -> carriage (door_slide) -> leaf (door_hinge) -> operator_pivot (operator_hinge) -> operator (operator_slide)
            leaf -> latch (latch_slide);   base -> carriage2 (leaf2_slide) -> leaf2 (leaf2_hinge)
    Slots   a door uses the joints it needs (a swing door: door_hinge; a slider: door_slide; a lever: operator_hinge;
            a touch bar: operator_slide; a spring latch: latch_slide; the second leaf of a pair / saloon / bypass /
            automatic slider: leaf2_*).  Unused joints are locked (range +-0.5 mm / +-0.05 deg, stiff drive).
    Welds   every other moving part in the primary leaf's subtree (deadbolts, thumbturns, closer arms, keypad keys,
            folded panels ...) is welded into its link at the initial state (locks engaged stay engaged; latch-like
            parts that could block the door are welded released).  Other world-attached moving bodies become static
            (release-type parts released) and further leaf-like panels beyond the second are omitted.
    Meta    ``doorbench:rl`` (JSON) on the default prim describes the slots, thresholds, grip points and sites.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    W = _Writer(model, path, hardware_dir, out_dir)
    tier = "full"
    bodies = model.bodies_in_tier(tier)
    names = {b.name for b in bodies}
    wt = _world_poses(model, bodies)
    meta = model.meta
    Sdf, Gf = W.Sdf, W.Gf
    notes = []

    def body_of_joint(jn):
        for b in bodies:
            if b.joint is not None and b.joint.name == jn:
                return b
        return None

    def is_moving_root(b):
        return (not b.static) and (b.parent is None or model.body(b.parent).static)

    def ancestors(n):
        out = []
        while n is not None:
            out.append(n)
            n = model.body(n).parent
        return out

    primary = body_of_joint(meta.get("primary_joint"))
    if primary is None:
        raise ValueError(f"{model.name}: no primary joint for the RL export")
    pj = primary.joint
    # moving ancestors of the primary (rising-hinge risers): welded into the base frame = ignored (locked at initial)
    prim_chain = [a for a in ancestors(primary.name)[1:] if not model.body(a).static]
    if prim_chain:
        notes.append(f"moving ancestors of the primary leaf locked at initial: {prim_chain}")
    # operator: the meta operator joint if its body descends from the primary leaf
    operator = body_of_joint(meta.get("operator_joint")) if meta.get("operator_joint") else None
    op_world_mounted = False
    if operator is not None and primary.name not in ancestors(operator.name):
        op_world_mounted = True
        notes.append(f"operator {operator.name} is not carried by the primary leaf: welded released, operator slot empty")
        operator = None
    # latch: a prismatic latch-role joint under the primary leaf (not under the operator)
    latch = None
    for b in bodies:
        if b.joint is not None and b.joint.role == "latch" and b.joint.type == "slide" and primary.name in ancestors(b.name) and (operator is None or operator.name not in ancestors(b.name)):
            latch = b
            break
    # secondary: heaviest other moving root with a leaf-like joint
    others = [b for b in bodies if is_moving_root(b) and b.name != primary.name and b.name not in prim_chain and b.name not in ancestors(primary.name)]
    secondary = None
    cands = [b for b in others if b.joint is not None and b.joint.role in ("primary", "secondary") and b.joint.type in ("hinge", "slide")]
    if cands:
        secondary = max(cands, key=lambda b: _subtree_mass(model, b.name, tier, wt)[0])
    # ---------------------------------------------------------------- link frames (world, initial state)
    leaf_pos, leaf_quat = wt[primary.name]
    axis_p = np.asarray(pj.axis, float)
    axis_p = axis_p / np.linalg.norm(axis_p)
    jpos_p = np.asarray(pj.pos, float)
    anchor_p = leaf_pos + quat_rotate(leaf_quat, jpos_p)           # world anchor of the primary joint
    # dummy axes: chosen so gravity does not load a locked joint (vertical hinge axes, horizontal slide axes)
    Z = np.array([0.0, 0.0, 1.0])
    X = np.array([1.0, 0.0, 0.0])

    slots = {}
    # --- door_slide (base -> carriage) and door_hinge (carriage -> leaf); both links share the leaf's frame
    carriage_pos, carriage_quat = leaf_pos.copy(), leaf_quat.copy()
    axis_p_w = quat_rotate(leaf_quat, axis_p)
    if pj.type == "slide":
        slots["door"] = "slide"
        slide_axis_w, hinge_axis_w = axis_p_w, Z
    else:
        slots["door"] = "hinge"
        slide_axis_w, hinge_axis_w = (X if abs(axis_p_w[2]) > 0.5 else Z), axis_p_w
    # --- operator chain
    if operator is not None:
        op_pos, op_quat = wt[operator.name]
        oj = operator.joint
        axis_o_w = quat_rotate(op_quat, np.asarray(oj.axis, float) / np.linalg.norm(oj.axis))
        anchor_o = op_pos + quat_rotate(op_quat, np.asarray(oj.pos, float))
        slots["operator"] = "slide" if oj.type == "slide" else "hinge"
        if oj.type == "slide":
            op_hinge_axis_w, op_slide_axis_w = Z, axis_o_w
        else:
            op_hinge_axis_w, op_slide_axis_w = axis_o_w, (X if abs(axis_o_w[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    else:
        slots["operator"] = "none"
        # empty operator links sit at the handle height on the leaf's free edge
        op_pos = leaf_pos + quat_rotate(leaf_quat, np.array([float(meta.get("leaf_edge_x_local", 0.0)) * 0.5, 0.0, float(meta.get("handle_height", 1.0))]))
        op_quat = leaf_quat.copy()
        anchor_o = op_pos
        op_hinge_axis_w, op_slide_axis_w = Z, X
    # --- latch
    if latch is not None:
        la_pos, la_quat = wt[latch.name]
        lj = latch.joint
        axis_l_w = quat_rotate(la_quat, np.asarray(lj.axis, float) / np.linalg.norm(lj.axis))
        anchor_l = la_pos + quat_rotate(la_quat, np.asarray(lj.pos, float))
        slots["latch"] = "slide"
    else:
        slots["latch"] = "none"
        la_pos, la_quat = op_pos.copy(), leaf_quat.copy()
        axis_l_w, anchor_l = X, la_pos
    # --- secondary chain
    if secondary is not None:
        s_pos, s_quat = wt[secondary.name]
        sj = secondary.joint
        axis_s_w = quat_rotate(s_quat, np.asarray(sj.axis, float) / np.linalg.norm(sj.axis))
        anchor_s = s_pos + quat_rotate(s_quat, np.asarray(sj.pos, float))
        slots["secondary"] = "slide" if sj.type == "slide" else "hinge"
        if sj.type == "slide":
            s_slide_axis_w, s_hinge_axis_w = axis_s_w, Z
        else:
            s_slide_axis_w, s_hinge_axis_w = (X if abs(axis_s_w[2]) > 0.5 else Z), axis_s_w
    else:
        slots["secondary"] = "none"
        s_pos, s_quat = np.array([0.0, 0.0, 0.05]), np.array([1.0, 0, 0, 0])
        anchor_s = s_pos
        s_slide_axis_w, s_hinge_axis_w = X, Z
    # ---------------------------------------------------------------- body -> link assignment
    stop = set()
    if operator is not None:
        stop.add(operator.name)
    if latch is not None:
        stop.add(latch.name)
    leaf_bodies = [n for n in _subtree(model, primary.name, stop=stop) if n in names]
    op_bodies = [n for n in _subtree(model, operator.name) if n in names] if operator is not None else []
    latch_bodies = [n for n in _subtree(model, latch.name) if n in names] if latch is not None else []
    sec_bodies = [n for n in _subtree(model, secondary.name) if n in names] if secondary is not None else []
    assigned = set(leaf_bodies) | set(op_bodies) | set(latch_bodies) | set(sec_bodies) | set(prim_chain)
    # world-attached moving bodies not in any link: static (released if they are latch/lock/operator parts) or omitted
    static_extra, omitted = [], []
    for b in others:
        if b.name in assigned:
            continue
        sub = [n for n in _subtree(model, b.name) if n in names]
        if b.semantic == "leaf" or (b.joint is not None and b.joint.role in ("primary", "secondary")):
            omitted += sub
            continue
        static_extra += sub
    if omitted:
        notes.append(f"leaf-like bodies omitted from the RL articulation (only two leaves are modelled): {omitted}")
    # welded pose shifts: parts welded released (latch-like) get their joint moved to the upper limit
    shift = {}
    for n in leaf_bodies + static_extra:
        b = model.body(n)
        if b.joint is None or n in (primary.name,):
            continue
        released = (b.joint.role == "latch") or (b.joint.role == "operator" and n in static_extra) or (b.joint.role == "lock" and n in static_extra and b.joint.type == "slide" and "pin" in n)
        if released:
            shift[n] = _released_pose_shift(model, b)
    # recompute world poses with the shifts applied (descendants follow)
    wt2 = {}

    def world_tf2(b: Body):
        if b.name in wt2:
            return wt2[b.name]
        if b.parent is None:
            pos, quat = np.asarray(b.pos, float), np.asarray(b.quat, float)
        else:
            ppos, pquat = world_tf2(model.body(b.parent))
            pos, quat = _compose(ppos, pquat, b.pos, b.quat)
        dq = shift.get(b.name, 0.0)
        if dq and b.joint is not None:
            dpos, dquat = _joint_displacement(b.joint, dq)
            pos, quat = _compose(pos, quat, dpos, dquat)
        wt2[b.name] = (pos, quat)
        return wt2[b.name]

    for b in bodies:
        world_tf2(b)
    # ---------------------------------------------------------------- write prims
    env_path = W.root_path.AppendChild("Env")
    W.UsdGeom.Xform.Define(W.stage, env_path)
    art_path = W.root_path.AppendChild("Articulation")
    W.add_articulation_root(art_path)
    joints_path = art_path.AppendChild("Joints")
    W.UsdGeom.Scope.Define(W.stage, joints_path)
    base_path = W.add_base_link(art_path, joints_path)
    sites = {"grip": [], "push": [], "leaf_edge": [], "approach": None, "goal": None, "pass_plane": None}
    # static environment (no floor: Isaac Lab provides the ground plane)
    for b in bodies:
        if not b.static:
            continue
        bp = env_path.AppendChild(_safe(b.name))
        xf = W.UsdGeom.Xform.Define(W.stage, bp)
        pos, quat = wt2[b.name]
        _set_xform(W.UsdGeom, Gf, xf, pos, quat)
        for g in b.geoms:
            if tier in g.tiers and g.semantic != "floor":
                W.add_geom(bp, g)
        for s in b.sites:
            if tier in s.tiers:
                W.add_site(bp, s.name, s.pos, s.quat, s.role)
                if s.role in ("approach", "goal", "pass_plane") and sites.get(s.role) is None:
                    sites[s.role] = [float(x) for x in (pos + quat_rotate(quat, s.pos))]
    for n in static_extra:
        b = model.body(n)
        bp = env_path.AppendChild(_safe(n))
        xf = W.UsdGeom.Xform.Define(W.stage, bp)
        pos, quat = wt2[n]
        _set_xform(W.UsdGeom, Gf, xf, pos, quat)
        xf.GetPrim().CreateAttribute("doorbench:welded_static", Sdf.ValueTypeNames.Bool).Set(True)
        for g in b.geoms:
            if tier in g.tiers:
                W.add_geom(bp, g)

    def write_link(link_name, frame_pos, frame_quat, body_names, semantic, min_mass=0.05, extra_mass_at=None):
        """Rigid link at (frame_pos, frame_quat) holding the geometry of body_names (welded at their poses)."""
        lp = art_path.AppendChild(link_name)
        m, com, I = _welded_inertial(model, body_names, frame_pos, frame_quat, wt2, tier)
        if m < min_mass:
            m = max(m, min_mass)
            I = I + np.eye(3) * (0.4 * min_mass * 0.03 ** 2)
        prim = W.add_rigid_body(lp, frame_pos, frame_quat, m, com, I, semantic=semantic, label=", ".join(body_names) if body_names else f"{link_name} (empty)")
        prim.CreateAttribute("doorbench:source_bodies", Sdf.ValueTypeNames.String).Set(json.dumps(body_names))
        for n in body_names:
            b = model.body(n)
            bpos, bquat = wt2[n]
            for g in b.geoms:
                if tier not in g.tiers:
                    continue
                gw_pos, gw_quat = _compose(bpos, bquat, g.pos, g.quat)
                lpos, lquat = _relative(frame_pos, frame_quat, gw_pos, gw_quat)
                W.add_geom(lp, g, pos=lpos, quat=lquat, name=f"{n}__{g.name}" if n != body_names[0] else g.name)
            for s in b.sites:
                if tier not in s.tiers:
                    continue
                sw_pos, sw_quat = _compose(bpos, bquat, s.pos, s.quat)
                lpos, lquat = _relative(frame_pos, frame_quat, sw_pos, sw_quat)
                W.add_site(lp, s.name, lpos, lquat, s.role)
                if s.role in ("grip", "push", "leaf_edge"):
                    sites[s.role].append({"name": s.name, "link": link_name, "pos": [float(x) for x in lpos]})
        return lp

    carriage_path = write_link("carriage", carriage_pos, carriage_quat, [], "carriage", min_mass=max(0.5, 0.05 * _subtree_mass(model, primary.name, tier, wt)[0]))
    leaf_path = write_link("leaf", leaf_pos, leaf_quat, leaf_bodies, "leaf")
    op_m = _subtree_mass(model, operator.name, tier, wt)[0] if operator is not None else 0.0
    pivot_path = write_link("operator_pivot", op_pos, op_quat, [], "operator_pivot", min_mass=max(0.05, 0.3 * op_m))
    operator_path = write_link("operator", op_pos, op_quat, op_bodies, "operator")
    latch_path = write_link("latch", la_pos, la_quat, latch_bodies, "latch")
    carriage2_path = write_link("carriage2", s_pos, s_quat, [], "carriage2", min_mass=max(0.5, 0.05 * _subtree_mass(model, secondary.name, tier, wt)[0]) if secondary is not None else 0.05)
    leaf2_path = write_link("leaf2", s_pos, s_quat, sec_bodies, "leaf2")
    link_paths = {"base": base_path, "carriage": carriage_path, "leaf": leaf_path, "operator_pivot": pivot_path, "operator": operator_path, "latch": latch_path,
                  "carriage2": carriage2_path, "leaf2": leaf2_path}
    link_frames = {"base": (np.zeros(3), np.array([1.0, 0, 0, 0])), "carriage": (carriage_pos, carriage_quat), "leaf": (leaf_pos, leaf_quat),
                   "operator_pivot": (op_pos, op_quat), "operator": (op_pos, op_quat), "latch": (la_pos, la_quat), "carriage2": (s_pos, s_quat), "leaf2": (s_pos, s_quat)}

    def joint_frames(parent, child, anchor_w, axis_w):
        """localPos/Rot in parent and child link frames for a joint frame at anchor_w with X along axis_w."""
        qa_w = _quat_x_to(axis_w)
        pp, pq = link_frames[parent]
        cp, cq = link_frames[child]
        pos0, rot0 = _relative(pp, pq, anchor_w, qa_w)
        pos1, rot1 = _relative(cp, cq, anchor_w, qa_w)
        return pos0, rot0, pos1, rot1

    joint_meta = {}

    def write_dof(name, jtype, parent, child, anchor_w, axis_w, src: Body | None, active: bool, reaction=None):
        pos0, rot0, pos1, rot1 = joint_frames(parent, child, anchor_w, axis_w)
        revolute = jtype == "revolute"
        if active and src is not None and src.joint is not None:
            jt = src.joint.for_tier("simple")       # the canonical RL articulation is the reduced (calibrated) model
            if jt.range is not None:
                lo, hi = jt.range[0] - jt.modeled_at, jt.range[1] - jt.modeled_at
            else:
                lo, hi = (-math.pi * 4, math.pi * 4) if revolute else (-10.0, 10.0)
            extra = {"role": jt.role, "label": jt.label, "source_joint": jt.name, "zero_offset": float(jt.modeled_at), "initial": float(jt.initial),
                     "limited": jt.range is not None, "locked": False, "slot_active": True}
            if jt.damping_closing is not None:
                extra["damping_closing"] = float(jt.damping_closing)
                extra["damping_opening"] = float(jt.damping_opening or 0.0)
            if jt.backcheck_angle is not None:
                extra["backcheck_angle"] = float(jt.backcheck_angle)
                extra["backcheck_damping"] = float(jt.backcheck_damping or 0.0)
            if jt.ratchet_one_way:
                extra["ratchet_one_way"] = True
            W.add_dof_joint(joints_path.AppendChild(name), jtype, link_paths[parent], link_paths[child], pos0, rot0, pos1, rot1, lo, hi,
                            stiffness=float(jt.stiffness), damping=float(jt.damping), target=(float(jt.springref - jt.modeled_at) if jt.stiffness else 0.0),
                            frictionloss=float(jt.frictionloss), armature=float(jt.armature), reaction_force=reaction, extra=extra)
            joint_meta[name] = {"active": True, "type": jtype, "source": jt.name, "role": jt.role, "range": [float(lo), float(hi)], "stiffness": float(jt.stiffness),
                                "damping": float(jt.damping), "target": (float(jt.springref - jt.modeled_at) if jt.stiffness else 0.0), "friction": float(jt.frictionloss),
                                "damping_closing": jt.damping_closing, "damping_opening": jt.damping_opening, "backcheck_angle": jt.backcheck_angle, "backcheck_damping": jt.backcheck_damping,
                                "ratchet_one_way": bool(jt.ratchet_one_way), "label": jt.label}
        else:
            lo, hi = (-math.radians(LOCK_RANGE_DEG), math.radians(LOCK_RANGE_DEG)) if revolute else (-LOCK_RANGE_M, LOCK_RANGE_M)
            W.add_dof_joint(joints_path.AppendChild(name), jtype, link_paths[parent], link_paths[child], pos0, rot0, pos1, rot1, lo, hi,
                            stiffness=LOCK_STIFF_ANG if revolute else LOCK_STIFF_LIN, damping=LOCK_DAMP_ANG if revolute else LOCK_DAMP_LIN, target=0.0,
                            frictionloss=0.0, armature=0.01 if revolute else 0.1, extra={"role": "locked", "label": f"{name} (unused slot, locked)", "locked": True, "slot_active": False})
            joint_meta[name] = {"active": False, "type": jtype, "range": [float(lo), float(hi)]}

    react_p = _reaction_estimate(model, primary, tier, wt)
    write_dof("door_slide", "prismatic", "base", "carriage", anchor_p, slide_axis_w, primary, pj.type == "slide", reaction=react_p)
    write_dof("door_hinge", "revolute", "carriage", "leaf", anchor_p, hinge_axis_w, primary, pj.type == "hinge", reaction=react_p)
    react_o = _reaction_estimate(model, operator, tier, wt) if operator is not None else None
    write_dof("operator_hinge", "revolute", "leaf", "operator_pivot", anchor_o, op_hinge_axis_w, operator, operator is not None and operator.joint.type == "hinge", reaction=react_o)
    write_dof("operator_slide", "prismatic", "operator_pivot", "operator", anchor_o, op_slide_axis_w, operator, operator is not None and operator.joint.type == "slide", reaction=react_o)
    write_dof("latch_slide", "prismatic", "leaf", "latch", anchor_l, axis_l_w, latch, latch is not None, reaction=_reaction_estimate(model, latch, tier, wt) if latch is not None else None)
    react_s = _reaction_estimate(model, secondary, tier, wt) if secondary is not None else None
    write_dof("leaf2_slide", "prismatic", "base", "carriage2", anchor_s, s_slide_axis_w, secondary, secondary is not None and secondary.joint.type == "slide", reaction=react_s)
    write_dof("leaf2_hinge", "revolute", "carriage2", "leaf2", anchor_s, s_hinge_axis_w, secondary, secondary is not None and secondary.joint.type == "hinge", reaction=react_s)
    # latch coupling (one-sided tendon in MJCF): scale such that bolt_q >= scale * operator_q
    latch_coupling = None
    if latch is not None and operator is not None:
        for t in model.tendons:
            js = [x[0] for x in t.sites]
            if latch.joint.name in js and operator.joint.name in js:
                wl = dict((x[0], float(x[1])) for x in t.sites)
                latch_coupling = {"scale": -wl[operator.joint.name] / wl[latch.joint.name], "operator_joint": "operator_hinge" if operator.joint.type == "hinge" else "operator_slide"}
                break
        if latch_coupling is None:
            for q in model.equalities:
                if q.kind == "joint" and q.a == latch.joint.name and q.b == operator.joint.name:
                    latch_coupling = {"scale": float(q.polycoeff[1]), "offset": float(q.polycoeff[0]), "operator_joint": "operator_hinge" if operator.joint.type == "hinge" else "operator_slide"}
    # secondary coupling (automatic sliding pairs, dutch joining bolts ...)
    secondary_coupling = None
    if secondary is not None:
        for q in model.equalities:
            if q.kind == "joint" and {q.a, q.b} == {secondary.joint.name, pj.name}:
                driven_is_secondary = q.a == secondary.joint.name
                secondary_coupling = {"driven": "secondary" if driven_is_secondary else "primary", "coeff": [float(c) for c in q.polycoeff[:2]], "label": q.label}
    # ---------------------------------------------------------------- meta
    spec = spec or {}
    kin = spec.get("kinematics", {})
    phys = spec.get("physics", {})
    opening = spec.get("opening", {})
    is_hinge = pj.type == "hinge"
    clear_travel = 1.9 if kin.get("type") == "slide_vertical" else 0.55
    if not is_hinge and pj.range is not None:
        clear_travel = min(clear_travel, 0.95 * float(pj.range[1] - pj.modeled_at))
    grip_pts = sites["grip"] or sites["push"] or sites["leaf_edge"]
    # prefer a grip on the operator link, then anything on the primary leaf (never the secondary leaf)
    grip_pts = sorted(grip_pts, key=lambda s: {"operator": 0, "leaf": 1}.get(s["link"], 2))
    grip_link = grip_pts[0]["link"] if grip_pts else "leaf"
    rl = {
        "door_id": model.name, "family": meta.get("family"), "task": meta.get("task"), "slots": slots,
        "links": list(RL_LINKS), "joints": joint_meta,
        "primary_joint": pj.name, "operator_joint": operator.joint.name if operator is not None else None, "latch_joint": latch.joint.name if latch is not None else None,
        "secondary_joint": secondary.joint.name if secondary is not None else None,
        "door_joint": "door_hinge" if is_hinge else "door_slide",
        "operator_slot_joint": None if operator is None else ("operator_hinge" if operator.joint.type == "hinge" else "operator_slide"),
        "secondary_slot_joint": None if secondary is None else ("leaf2_hinge" if secondary.joint.type == "hinge" else "leaf2_slide"),
        "latch_coupling": latch_coupling, "secondary_coupling": secondary_coupling,
        "open_threshold": math.radians(10) if is_hinge else 0.10, "clear_threshold": math.radians(60) if is_hinge else clear_travel,
        "closed_threshold": math.radians(3) if is_hinge else 0.03,
        "sites": {"approach": sites["approach"] or [0.0, -1.5, 0.0], "goal": sites["goal"] or [0.0, 1.5, 0.0], "pass_plane": sites["pass_plane"] or [0.0, 0.0, 1.0],
                  "grip": grip_pts, "grip_link": grip_link, "push": sites["push"], "leaf_edge": sites["leaf_edge"]},
        "opening": {"width": opening.get("width"), "height": opening.get("height")},
        "robot": spec.get("robot", {}), "handle_height": meta.get("handle_height"), "u": meta.get("u"), "v": meta.get("v"), "hinge_x": meta.get("hinge_x"),
        "lock": {"model": spec.get("lock", {}).get("model"), "engaged": bool(spec.get("lock", {}).get("engaged")), "robot_side_release": bool(spec.get("lock", {}).get("robot_side_release", True))},
        "closer": spec.get("closer", {}).get("model"), "operator": spec.get("operator", {}).get("model"), "latch": spec.get("latch", {}).get("model"),
        "damage": {k: phys.get("damage", {}).get(k) for k in ("leaf_dent_force_N", "glass_break_force_N", "operator_yield_torque_Nm", "slam_velocity_rad_s", "frame_impact_force_N")},
        "actuators": meta.get("actuators", []), "welded_static": static_extra, "omitted": omitted, "notes": notes,
        "mass_kg": float(_subtree_mass(model, primary.name, tier, wt)[0]),
    }
    W.set_json(W.root.GetPrim(), "doorbench:rl", rl)
    W.set_json(W.root.GetPrim(), "doorbench:meta", {k: v for k, v in meta.items() if k not in ("notes",)})
    return W.save()


def _write_mesh_usd(mesh, path, name):
    from pxr import Usd, UsdGeom, Vt
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

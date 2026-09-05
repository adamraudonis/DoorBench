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
    N*m/rad.  Coulomb joint friction is exported as ``physxJointAxis:angular|linear:staticFrictionEffort`` /
    ``dynamicFrictionEffort`` (PhysX >= 5.6 / Isaac Sim >= 5.0, torque or force units) - the instance name of the
    per-axis API is ``angular`` on a RevoluteJoint and ``linear`` on a PrismaticJoint (the same tokens as
    ``PhysicsDriveAPI``); ``rotX`` / ``transX`` are D6 tokens that the PhysX USD parser silently ignores on single-DoF
    joints (round 1 of the Isaac parity gate: friction read back 0.0 on every joint).  The legacy load-dependent
    ``physxJoint:jointFriction`` coefficient is authored as 0 so friction is not applied twice; the value the old
    formula would give is kept in ``doorbench:legacy_friction_coeff``.
  * MJCF position servos of automatic doors (``meta.actuators``: force = clip(kp (ctrl - q) - kv v, forcerange))
    are the same law as a PhysX PD drive.  When the servo joint carries no passive spring of its own the servo is
    folded into the joint drive (stiffness += kp, damping += kv, targetPosition = ctrl, maxForce = forcerange);
    ``doorbench:servo_in_drive`` marks such joints so runners / environments do not emulate the servo a second time.
    Joints with both a spring and a servo keep the spring in the drive (one drive per axis) and are emulated.
  * ``physxRigidBody:maxAngularVelocity`` is authored explicitly at 100 rad/s (5729.58 deg/s, the PhysX default,
    in the degrees-per-second unit of the schema): MuJoCo has no velocity cap and a 100 deg/s cap (as a value
    meant in rad/s would read) clamps every door leaf at 1.75 rad/s.
  * Rising (helical) hinges couple a vertical slide to the hinge (``rise = c1 * hinge``); the canonical RL file
    locks the riser, so ``doorbench:rl["rise_coupling"]`` carries the equivalent gravity closing torque
    ``-m g c1`` that the environment applies on the door joint (docs/ISAAC_LAB.md, parameter mapping).
  * Couplings.  ``PhysxMimicJointAPI`` is authored ONLY for rotational -> rotational equalities: PhysX articulation
    mimic joints support rotational axes only, so a mimic on a prismatic joint (or one referencing a prismatic axis)
    is parsed and silently dropped - thumbturn -> deadbolt, wheel -> bolts, cremone and the helical riser all fall in
    that class.  Those couplings are exported as first-class metadata instead (``doorbench:coupling_*`` on the driven
    and driver joint prims + ``doorbench:couplings``): gearing, offset, the driven DOF's effective inertia, its
    passive law (spring / damping / Coulomb friction) and the constant gravity bias, plus the reflected inertia
    ``c1^2 * I_driven`` the consumer adds to the driver's armature.  Both consumers
    (``scripts/isaaclab/isaac_parity.py``, ``doorbench_isaaclab.mdp.DoorMechanismAction``) apply the coupling
    bilaterally: the driven joint tracks ``q_a = c0 + c1 q_b`` kinematically AND the driver carries the reaction
    ``c1 * tau_a_ext`` (a pure kinematic write applies no reaction: the driver loses the coupled part's weight and
    friction and sags).  The helical hinge's documented closing torque ``-m g dz/dq`` is exactly this reaction.
  * Self-collision.  ``physxArticulation:enabledSelfCollisions`` is True (PhysX still skips joint-adjacent link
    pairs, which is MuJoCo's parent/child default) and every pair MuJoCo suppresses - same weld body, weld
    parent/child, ``contact_excludes`` - is authored as ``PhysxFilteredPairsAPI`` so both engines filter the same
    set.  Without it a latch that holds one moving link against another (swing pairs latched into the inactive
    leaf, gate / baby-gate lift pins, sliding drop bolts) passes straight through in PhysX.
  * Env-release locks (mag lock, delayed egress, electric bolt, interlock) are a MuJoCo ``<weld>`` leaf -> world.
    They are exported as a real breakable ``UsdPhysics.FixedJoint`` base -> leaf with
    ``physics:excludeFromArticulation = True`` (the articulation stays a tree; PhysX solves it as a loop joint),
    ``physics:breakForce / breakTorque = holding_force_N`` and ``physics:jointEnabled`` that the environment clears
    on REX / badge / timer, exactly as ``doorbench.benchmark.DoorEnv`` clears ``d.eq_active``.
    ``doorbench:env_release`` on the default prim names the joints.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ..ir import Model, Body, Geom, quat_to_mat, quat_mul, quat_conj, quat_rotate, mat_to_quat, quat_from_axis_angle

G_ACC = 9.81
DEG = 180.0 / math.pi
# PhysX rigid-body angular velocity cap (schema unit: degrees / second).  100 rad/s is the PhysX default; MuJoCo has
# no cap at all, so the cap must stay far above any door motion (the fastest leaf in the dataset, a pet flap under
# the QA push, reaches ~65 rad/s).  Isaac Lab's RigidBodyPropertiesCfg.max_angular_velocity uses the same unit.
MAX_ANGULAR_VELOCITY_DEG_S = 100.0 * DEG
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
# smoothing velocity of the Coulomb term in the exported coupling law (rad/s or m/s): the reaction a consumer applies
# on the driver is c1 * tau_driven, and tau_driven's friction bound needs a sign that is defined at v = 0
COUPLING_FRICTION_VEL_EPS = 1e-3


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
    """D6-style axis token for a joint whose axis is X (used by ``PhysxMimicJointAPI``, whose parser expects it)."""
    return "rotX" if jtype in ("hinge", "revolute") else "transX"


def _dof_instance(jtype: str) -> str:
    """Instance name of the per-DoF APIs (``PhysicsDriveAPI``, ``PhysxJointAxisAPI``, ``PhysxLimitAPI``, ``JointStateAPI``)
    on a single-DoF joint: ``angular`` on a RevoluteJoint, ``linear`` on a PrismaticJoint."""
    return "angular" if jtype in ("hinge", "revolute") else "linear"


def servo_for_joint(actuators, joint_name: str):
    """The MJCF position actuator (``meta.actuators`` entry) driving ``joint_name``, or None."""
    for a in actuators or []:
        if a.get("joint") == joint_name and a.get("kind", "position") == "position":
            return a
    return None


def servo_drive_params(jt, servo: dict | None):
    """How an MJCF position servo maps onto the single PhysX drive of its joint.

    MuJoCo: f = clip(kp (ctrl - q) - kv v, forcerange) plus the joint's own passive spring / damper, both integrated
    implicitly (implicitfast).  PhysX: f = clip(k (target - q) + d (v_target - v), maxForce), one drive per axis.
    * joint without a passive spring (automatic sliders, elevators): the drive IS the servo - k = kp, d = kv + damping,
      target = ctrl (MJCF coordinates), maxForce = forcerange.  The passive viscous term (2-8 N*s/m) is clipped
      together with the servo; at 0.5 m/s that is < 3 % of the 150 N saturation force.
    * joint with its own spring (automatic swing operators on a closer): folding both into one clipped drive would
      clip the closer spring with the servo's forcerange, so the drive keeps the spring and the servo stays a
      feed-forward emulation (``in_drive`` False).
    Returns None when nothing is folded, else {"kp", "kv", "force_limit", "ctrl", "stiffness", "damping", "target"} in SI
    (target in USD coordinates = ctrl - modeled_at).
    """
    if servo is None or float(jt.stiffness or 0.0) > 0.0:
        return None
    kp, kv = float(servo.get("kp", 0.0)), float(servo.get("kv", 0.0))
    fr = servo.get("forcerange", (-1e6, 1e6))
    lim = float(max(abs(float(fr[0])), abs(float(fr[1]))))
    ctrl = float(servo.get("ctrl", 0.0))
    return {"kp": kp, "kv": kv, "force_limit": lim, "ctrl": ctrl, "stiffness": kp, "damping": kv + float(jt.damping or 0.0),
            "target": ctrl - float(jt.modeled_at or 0.0), "name": servo.get("name")}


def rise_coupling_info(model: Model, primary, tier: str, wt: dict):
    """Rising / helical hinge: ``rise = c1 * hinge`` between a vertical slide carrying the leaf and the leaf's hinge.

    The coupling costs gravitational work m g dz per opening angle, i.e. a constant closing torque -m g c1 on the
    hinge (m = everything the riser carries).  Returned so consumers that cannot represent the screw joint (the
    canonical RL articulation locks the riser; PhysX drops translational mimic joints) apply the torque instead."""
    if primary is None or primary.joint is None or primary.joint.type != "hinge":
        return None
    chain = []
    n = primary.parent
    while n is not None:
        b = model.body(n)
        if b.static:
            break
        chain.append(b)
        n = b.parent
    for riser in chain:
        rj = riser.joint
        if rj is None or rj.type != "slide":
            continue
        for q in model.equalities:
            if q.kind == "joint" and q.a == rj.name and q.b == primary.joint.name and tier in q.tiers:
                c1 = float(q.polycoeff[1])
                pos, quat = wt[riser.name]
                axis_w = quat_rotate(quat, np.asarray(rj.axis, float) / np.linalg.norm(rj.axis))
                m, _ = _subtree_mass(model, riser.name, tier, wt)
                dz = float(axis_w[2]) * c1                    # vertical lift per radian of opening
                return {"rise_joint": rj.name, "hinge_joint": primary.joint.name, "coeff_m_per_rad": c1, "lift_m_per_rad": dz,
                        "carried_mass_kg": float(m), "gravity_torque_Nm": float(-m * G_ACC * dz), "label": q.label,
                        "note": "constant closing torque -m*g*dz/dq of the helical hinge; apply on the hinge when the rise joint is not simulated"}
    return None


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
        # degrees / second (schema unit): 100 rad/s, the PhysX default; MuJoCo has no cap
        prim.CreateAttribute("physxRigidBody:maxAngularVelocity", Sdf.ValueTypeNames.Float).Set(float(MAX_ANGULAR_VELOCITY_DEG_S))
        if semantic:
            prim.CreateAttribute("doorbench:semantic", Sdf.ValueTypeNames.String).Set(semantic)
        if label:
            prim.CreateAttribute("doorbench:label", Sdf.ValueTypeNames.String).Set(label)
        return prim

    def add_articulation_root(self, path):
        """Articulation root with self-collision ENABLED.

        PhysX still skips joint-adjacent link pairs, which is exactly MuJoCo's parent/child default; every further
        pair MuJoCo suppresses (same weld body, weld parent/child, ``contact_excludes``) is authored explicitly as
        ``PhysxFilteredPairsAPI`` (``add_filtered_pairs``).  With self-collision off, any latch that holds one moving
        link against another - swing pairs latched into the inactive leaf, gate / baby-gate lift pins, sliding drop
        bolts - passes straight through in PhysX while MuJoCo holds."""
        UsdGeom, UsdPhysics, Sdf = self.UsdGeom, self.UsdPhysics, self.Sdf
        art = UsdGeom.Xform.Define(self.stage, path)
        prim = art.GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        prim.AddAppliedSchema("PhysxArticulationAPI")
        prim.CreateAttribute("physxArticulation:articulationEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
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

    # ---- collision filtering -----------------------------------------------
    def add_filtered_pairs(self, pairs):
        """``PhysxFilteredPairsAPI`` for every (path_a, path_b) pair: PhysX must not collide these two prims.

        The API is single-apply, so all partners of one prim go into that prim's ``physxFilteredPairs:filteredPairs``
        relationship (PhysX reads the union of both directions; authoring one side is enough and keeps the file
        small).  ``pairs`` are Sdf paths of rigid-body prims."""
        by_prim = {}
        for a, b in pairs:
            key = str(a) if str(a) <= str(b) else str(b)
            other = str(b) if key == str(a) else str(a)
            by_prim.setdefault(key, []).append(other)
        for path, others in sorted(by_prim.items()):
            prim = self.stage.GetPrimAtPath(self.Sdf.Path(path))
            if not prim.IsValid():
                continue
            prim.AddAppliedSchema("PhysxFilteredPairsAPI")
            prim.CreateRelationship("physxFilteredPairs:filteredPairs").SetTargets([self.Sdf.Path(o) for o in sorted(set(others))])
        return sum(len(v) for v in by_prim.values())

    # ---- env-release lock (mag lock / delayed egress / electric bolt / interlock) ----
    def add_env_release_joint(self, path, body0_path, body1_path, pos0, rot0, pos1, rot1, holding_force, weld, enabled=True):
        """Breakable FixedJoint that reproduces the MJCF ``<weld>`` of an environment-released lock.

        ``physics:excludeFromArticulation = True`` keeps the articulation a tree (PhysX solves the joint as a
        maximal-coordinate loop joint between two articulation links).  ``physics:breakForce`` and
        ``physics:breakTorque`` are both the latch model's holding force: MuJoCo's own breakaway test compares
        every row of the weld constraint - three force rows and three torque rows - against the same number
        (``DoorEnv._lock_logic``), so the closest PhysX equivalent uses it on both channels.
        ``physics:jointEnabled`` is what the environment clears on REX / badge / timer / delayed-egress timeout,
        the counterpart of ``d.eq_active[eid] = 0``."""
        UsdPhysics, Sdf, Gf = self.UsdPhysics, self.Sdf, self.Gf
        fj = UsdPhysics.FixedJoint.Define(self.stage, path)
        fj.CreateBody0Rel().SetTargets([body0_path])
        fj.CreateBody1Rel().SetTargets([body1_path])
        fj.CreateLocalPos0Attr(_v3(Gf, pos0))
        fj.CreateLocalRot0Attr(_q(Gf, rot0))
        fj.CreateLocalPos1Attr(_v3(Gf, pos1))
        fj.CreateLocalRot1Attr(_q(Gf, rot1))
        fj.CreateExcludeFromArticulationAttr(True)
        fj.CreateJointEnabledAttr(bool(enabled))
        fj.CreateCollisionEnabledAttr(False)
        f = float(holding_force) if holding_force and math.isfinite(float(holding_force)) and float(holding_force) > 0 else float("inf")
        if math.isfinite(f):
            fj.CreateBreakForceAttr(f)
            fj.CreateBreakTorqueAttr(f)
        p = fj.GetPrim()
        p.CreateAttribute("doorbench:role", Sdf.ValueTypeNames.String).Set("env_release")
        p.CreateAttribute("doorbench:weld_name", Sdf.ValueTypeNames.String).Set(str(weld["name"]))
        p.CreateAttribute("doorbench:weld_body", Sdf.ValueTypeNames.String).Set(str(weld["body"]))
        p.CreateAttribute("doorbench:holding_force_N", Sdf.ValueTypeNames.Float).Set(float(f if math.isfinite(f) else 0.0))
        p.CreateAttribute("doorbench:label", Sdf.ValueTypeNames.String).Set(str(weld.get("label") or "environment-released lock"))
        return p

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
                      target=0.0, frictionloss=0.0, armature=0.0, reaction_force=None, max_force=1e6, extra=None, servo=None):
        """Revolute (jtype 'hinge'/'revolute') or Prismatic joint with limits, drive and PhysX friction.

        All inputs in SI (rad, m, N*m/rad, N/m); converted to UsdPhysics units here.  The joint frame X axis is the
        joint axis (rot0/rot1 rotate X onto it).

        MuJoCo -> PhysX mapping (docs/ISAAC_LAB.md):
          stiffness / springref  -> force drive stiffness / targetPosition (per degree on revolute joints)
          damping                -> drive damping (both implicit)
          frictionloss           -> PhysxJointAxisAPI:angular|linear static == dynamic friction effort (MuJoCo's
                                    frictionloss is one Coulomb bound for stick and slip); legacy coefficient 0
          armature               -> joint armature (added to the joint-space inertia in both engines)
          servo (``servo_drive_params``) -> the drive itself: stiffness / damping / target / maxForce
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
        conv = DEG if revolute else 1.0
        j.CreateLowerLimitAttr(float(lo * conv))
        j.CreateUpperLimitAttr(float(hi * conv))
        prim = j.GetPrim()
        inst = _dof_instance(jtype)
        if servo is not None:
            # MJCF position servo folded into the drive (see servo_drive_params): the joint has no spring of its own
            stiffness, damping, target, max_force = servo["stiffness"], servo["damping"], servo["target"], servo["force_limit"]
        # drive: spring (stiffness) + damping in UsdPhysics units (per degree for angular drives)
        drv = UsdPhysics.DriveAPI.Apply(prim, inst)
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(float(stiffness / conv))
        drv.CreateDampingAttr(float(damping / conv))
        drv.CreateTargetPositionAttr(float(target * conv))
        drv.CreateTargetVelocityAttr(0.0)
        drv.CreateMaxForceAttr(float(max_force))
        # PhysX joint attributes.  The per-axis API of PhysX >= 5.6 carries Coulomb friction as efforts (N*m / N,
        # load independent like MuJoCo's frictionloss) and the armature; its instance name on a single-DoF joint is
        # the drive's ("angular" / "linear").  The legacy PhysxJointAPI coefficient (friction = coeff * |joint
        # reaction force|) stays authored at 0 so PhysX never applies friction twice; ``physxJoint:armature`` is the
        # fallback for parsers without the axis API.
        prim.AddAppliedSchema("PhysxJointAPI")
        prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(float(armature))
        prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(0.0)
        legacy = 0.0
        if frictionloss > 0:
            legacy = float(min(float(frictionloss) / max(float(reaction_force or 0.0), 1.0), 10.0))
        prim.CreateAttribute("doorbench:legacy_friction_coeff", Sdf.ValueTypeNames.Float).Set(legacy)
        prim.AddAppliedSchema(f"PhysxJointAxisAPI:{inst}")
        prim.CreateAttribute(f"physxJointAxis:{inst}:staticFrictionEffort", Sdf.ValueTypeNames.Float).Set(float(frictionloss))
        prim.CreateAttribute(f"physxJointAxis:{inst}:dynamicFrictionEffort", Sdf.ValueTypeNames.Float).Set(float(frictionloss))
        prim.CreateAttribute(f"physxJointAxis:{inst}:viscousFrictionCoefficient", Sdf.ValueTypeNames.Float).Set(0.0)
        prim.CreateAttribute(f"physxJointAxis:{inst}:armature", Sdf.ValueTypeNames.Float).Set(float(armature))
        prim.CreateAttribute("doorbench:friction_effort", Sdf.ValueTypeNames.Float).Set(float(frictionloss))
        prim.CreateAttribute("doorbench:armature_si", Sdf.ValueTypeNames.Float).Set(float(armature))
        prim.CreateAttribute("doorbench:stiffness_si", Sdf.ValueTypeNames.Float).Set(float(stiffness))
        prim.CreateAttribute("doorbench:damping_si", Sdf.ValueTypeNames.Float).Set(float(damping))
        prim.CreateAttribute("doorbench:target_si", Sdf.ValueTypeNames.Float).Set(float(target))
        prim.CreateAttribute("doorbench:servo_in_drive", Sdf.ValueTypeNames.Bool).Set(servo is not None)
        if servo is not None:
            prim.CreateAttribute("doorbench:servo_kp_si", Sdf.ValueTypeNames.Float).Set(float(servo["kp"]))
            prim.CreateAttribute("doorbench:servo_kv_si", Sdf.ValueTypeNames.Float).Set(float(servo["kv"]))
            prim.CreateAttribute("doorbench:servo_force_limit", Sdf.ValueTypeNames.Float).Set(float(servo["force_limit"]))
            prim.CreateAttribute("doorbench:servo_ctrl", Sdf.ValueTypeNames.Float).Set(float(servo["ctrl"]))
            if servo.get("name"):
                prim.CreateAttribute("doorbench:servo_name", Sdf.ValueTypeNames.String).Set(str(servo["name"]))
        for k, v in (extra or {}).items():
            t = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int, float: Sdf.ValueTypeNames.Float}.get(type(v), Sdf.ValueTypeNames.String)
            prim.CreateAttribute(f"doorbench:{k}", t).Set(v if t != Sdf.ValueTypeNames.String else str(v))
        return prim

    def add_mimic(self, driven_path, driven_type, driver_path, driver_type, gearing, offset, label=""):
        """PhysxMimicJointAPI on the driven joint:  q_driven + gearing * q_driver + offset = 0  (PhysX units).

        Only valid when BOTH axes are rotational: PhysX articulation mimic joints support rotational axes only, and
        a mimic authored on a prismatic axis (or referencing one) is parsed without error and then ignored.  Callers
        must route the other couplings through ``add_coupling`` instead (``joint_couplings`` decides)."""
        assert driven_type in ("hinge", "revolute") and driver_type in ("hinge", "revolute"), \
            f"PhysxMimicJointAPI needs rotational axes on both sides (got {driven_type} <- {driver_type})"
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

    def add_coupling(self, c: dict, driven_path, driver_path, reflected_by_driver: dict):
        """Machine-readable coupling on the driven joint prim (+ the reflected inertia on the driver prim).

        Both consumers read these attributes and apply the coupling bilaterally: the driven joint tracks
        ``q_driven = c0 + c1 q_driver`` and the driver carries the reaction ``c1 * tau_driven_ext`` plus the
        reflected inertia ``c1^2 I_driven`` (as extra armature).  A pure kinematic write of the driven joint applies
        no reaction at all: the driver loses the coupled part's weight, spring and Coulomb friction and sags."""
        Sdf = self.Sdf
        p = self.stage.GetPrimAtPath(driven_path)
        p.CreateAttribute("doorbench:coupling_mode", Sdf.ValueTypeNames.String).Set(str(c["mode"]))
        p.CreateAttribute("doorbench:coupling_driver", Sdf.ValueTypeNames.String).Set(str(c["driver"]))
        p.CreateRelationship("doorbench:coupling_driver_joint").SetTargets([driver_path])
        p.CreateAttribute("doorbench:coupling_c0", Sdf.ValueTypeNames.Float).Set(float(c["coeff"][0]))
        p.CreateAttribute("doorbench:coupling_c1", Sdf.ValueTypeNames.Float).Set(float(c["coeff"][1]))
        p.CreateAttribute("doorbench:coupling_driven_inertia", Sdf.ValueTypeNames.Float).Set(float(c["driven_inertia"]))
        p.CreateAttribute("doorbench:coupling_reflected_inertia", Sdf.ValueTypeNames.Float).Set(float(c["reflected_inertia"]))
        p.CreateAttribute("doorbench:coupling_gravity_bias", Sdf.ValueTypeNames.Float).Set(float(c["driven_gravity_bias"]))
        p.CreateAttribute("doorbench:coupling_chain_order", Sdf.ValueTypeNames.Int).Set(int(c["chain_order"]))
        if c["mode"] == "emulated":
            reflected_by_driver[c["driver"]] = reflected_by_driver.get(c["driver"], 0.0) + float(c["reflected_inertia"])

    def set_reflected_inertia(self, driver_path, value):
        """``doorbench:coupling_reflected_armature`` on a driver joint: the inertia of everything it drives through
        couplings PhysX cannot represent.  Consumers add it to that joint's armature (the exporter does NOT fold it
        into ``physxJointAxis:*:armature``, which must keep matching the IR)."""
        p = self.stage.GetPrimAtPath(driver_path)
        p.CreateAttribute("doorbench:coupling_reflected_armature", self.Sdf.ValueTypeNames.Float).Set(float(value))

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


def _subtree_axis_inertia(model: Model, root_name: str, tier: str, wt: dict, axis_w, anchor_w, jtype: str):
    """Effective inertia of the DoF that moves ``root_name``'s subtree: mass along the axis (slide) or the
    subtree's inertia about the joint axis through ``anchor_w`` (hinge).  SI (kg or kg*m^2)."""
    n = np.asarray(axis_w, float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    total, I_ax = 0.0, 0.0
    for name in _subtree(model, root_name):
        b = model.body(name)
        if tier not in b.tiers or name not in wt:
            continue
        m, com, I = b.inertial(tier)
        if m <= 0:
            continue
        total += m
        if jtype != "slide":
            pos, quat = wt[name]
            R = quat_to_mat(quat)
            Iw = R @ np.asarray(I, float) @ R.T
            r = (pos + quat_rotate(quat, com)) - np.asarray(anchor_w, float)
            rp = r - float(np.dot(r, n)) * n                     # distance from the axis
            I_ax += float(n @ Iw @ n) + m * float(np.dot(rp, rp))
    return float(total if jtype == "slide" else I_ax)


def _gravity_bias(model: Model, root_name: str, tier: str, wt: dict, axis_w, anchor_w, jtype: str):
    """Generalised gravity force on that DoF at the authored pose (N on a slide, N*m on a hinge).

    Slide: -m g (axis . z).  Hinge: -(sum_i m_i g) . (n x r_i) = -g * n . (sum_i m_i (z x r_i)) with r_i the COM
    offset from the anchor.  This is what a consumer must reflect onto the driver of a coupling that PhysX drops;
    for the helical hinge (vertical riser slide, gearing c1) c1 * bias reproduces the documented -m g dz/dq."""
    n = np.asarray(axis_w, float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    g_vec = np.array([0.0, 0.0, -G_ACC])
    tau = 0.0
    for name in _subtree(model, root_name):
        b = model.body(name)
        if tier not in b.tiers or name not in wt:
            continue
        m, com, _ = b.inertial(tier)
        if m <= 0:
            continue
        pos, quat = wt[name]
        c = pos + quat_rotate(quat, com)
        if jtype == "slide":
            tau += m * float(np.dot(g_vec, n))
        else:
            tau += m * float(np.dot(np.cross(n, c - np.asarray(anchor_w, float)), g_vec))
    return float(tau)


def _weld_map(model: Model, bodies):
    """MuJoCo weld semantics: a body whose parent link carries no joint belongs to its parent's weld body.

    Returns ({body: weld root}, {body: weld root of the weld body's parent, or None for a world child}) over the
    moving bodies of the tier - the two quantities MuJoCo's contact filter uses (``weld1 == weld2`` -> skip;
    ``weld1 == parent2`` or ``weld2 == parent1`` -> skip unless one of them is the world)."""
    byname = {b.name: b for b in bodies}
    root = {}

    def wroot(n):
        if n in root:
            return root[n]
        b = byname[n]
        p = b.parent
        root[n] = n if (b.joint is not None or p is None or p not in byname or byname[p].static) else wroot(p)
        return root[n]

    for b in bodies:
        if not b.static:
            wroot(b.name)
    parent = {}
    for n in root:
        p = byname[root[n]].parent
        parent[n] = None if (p is None or p not in byname or byname[p].static) else root[p]
    return root, parent


def mujoco_filtered_pairs(model: Model, bodies, tier: str):
    """Sorted (body_a, body_b) pairs of colliding moving bodies whose contacts MuJoCo suppresses.

    MuJoCo skips a geom pair when both geoms belong to the same weld body, when the two weld bodies are in a
    parent/child relation (neither being the world), or when the body pair is listed in ``<contact><exclude>``.
    PhysX with ``enabledSelfCollisions`` skips only joint-adjacent links, so the rest must be authored as
    ``PhysxFilteredPairsAPI`` for the two engines to filter the same set."""
    moving = [b for b in bodies if not b.static and any(g.collision and tier in g.tiers for g in b.geoms)]
    root, wparent = _weld_map(model, bodies)
    excl = {tuple(sorted(x)) for x in model.contact_excludes}
    out = set()
    for i, a in enumerate(moving):
        for b in moving[i + 1:]:
            pair = tuple(sorted((a.name, b.name)))
            if root[a.name] == root[b.name] or wparent.get(a.name) == root[b.name] or wparent.get(b.name) == root[a.name] or pair in excl:
                out.add(pair)
    return sorted(out)


def env_release_welds(model: Model, tier: str):
    """Active ``weld`` equalities (mag lock / delayed egress / electric bolt / interlock leaf -> world) with the
    holding force the environment breaks them at (``meta.breakable_welds``)."""
    forces = {w["name"]: float(w.get("holding_force_N") or 0.0) for w in (model.meta.get("breakable_welds") or [])}
    out = []
    for q in model.equalities:
        if q.kind != "weld" or tier not in q.tiers:
            continue
        out.append({"name": q.name, "body": q.a, "other": q.b or "world", "label": q.label, "active": bool(q.active),
                    "holding_force_N": forces.get(q.name, 0.0)})
    return out


def joint_couplings(model: Model, tier: str, joint_types: dict, wt: dict, body_of_joint, servo_joints=()):
    """Bilateral polynomial joint equalities with everything a consumer needs to reproduce them.

    ``mode``: ``mimic`` when PhysX honours a ``PhysxMimicJointAPI`` (rotational driven axis AND rotational reference
    axis - the only case PhysX articulations support); ``servo`` when BOTH joints carry their own MJCF position
    servo in their PhysX drive (the two leaves of an automatic bi-parting slider / elevator: each leaf has its own
    operator in the MJCF too, so the drives already move them together and there is nothing to reflect);
    else ``emulated``.  For every coupling the entry carries the
    driven DOF's effective inertia, its passive law and its constant gravity bias, plus ``reflected_inertia``
    (``c1^2 * I_driven``, accumulated through coupling chains) that a consumer adds to the DRIVER's armature so the
    driver carries the coupled part's inertia the way MuJoCo's constraint does."""
    out = []
    for q in model.equalities:
        if q.kind != "joint" or tier not in q.tiers:
            continue
        if q.a not in joint_types or q.b not in joint_types:
            continue
        ta, tb = joint_types[q.a], joint_types[q.b]
        ba = body_of_joint(q.a)
        if ba is None or ba.joint is None:
            continue
        c0, c1 = float(q.polycoeff[0]), float(q.polycoeff[1])
        ja = ba.joint
        pos, quat = wt[ba.name]
        axis_w = quat_rotate(quat, np.asarray(ja.axis, float) / np.linalg.norm(ja.axis))
        anchor_w = pos + quat_rotate(quat, np.asarray(ja.pos, float))
        I_a = _subtree_axis_inertia(model, ba.name, tier, wt, axis_w, anchor_w, ja.type)
        bias = _gravity_bias(model, ba.name, tier, wt, axis_w, anchor_w, ja.type)
        rotational = ta in ("hinge", "revolute") and tb in ("hinge", "revolute")
        servoed = q.a in set(servo_joints) and q.b in set(servo_joints)
        mode = "mimic" if rotational else ("servo" if servoed else "emulated")
        out.append({
            "driven": q.a, "driver": q.b, "driven_type": ta, "driver_type": tb,
            "mode": mode,
            "coeff": [c0, c1], "gearing": -c1, "offset": -c0, "label": q.label,
            "driven_inertia": I_a, "reflected_inertia": c1 * c1 * I_a,
            "driven_gravity_bias": bias,
            "driven_stiffness": float(ja.stiffness), "driven_damping": float(ja.damping),
            "driven_target": float(ja.springref - ja.modeled_at) if ja.stiffness else 0.0,
            "driven_friction": float(ja.frictionloss),
            "driven_range": None if ja.range is None else [float(ja.range[0] - ja.modeled_at), float(ja.range[1] - ja.modeled_at)],
            "friction_vel_eps": COUPLING_FRICTION_VEL_EPS,
            "reason": (None if rotational else
                       ("both joints carry their own MJCF position servo in their PhysX drive: the drives move them together"
                        if servoed else
                        "PhysX articulation mimic joints support rotational axes only: a mimic on a prismatic axis is dropped")),
            "note": "q_driven = c0 + c1*q_driver; the driver carries the reaction c1 * tau_driven_ext and the reflected inertia c1^2 * I_driven",
        })
    # coupling chains (multipoint bolt <- deadbolt <- thumbturn): reflect the inertia of the whole driven chain onto
    # each driver, deepest first, so a driver that is itself driven passes its accumulated inertia on
    by_driven = {c["driven"]: c for c in out}
    order, seen = [], set()

    def visit(name, stack=()):
        if name in seen or name in stack or name not in by_driven:
            return
        c = by_driven[name]
        visit(c["driver"], stack + (name,))
        seen.add(name)
        order.append(name)

    for c in list(out):
        visit(c["driven"])
    eff = {c["driven"]: c["driven_inertia"] for c in out}
    for name in reversed(order):                 # deepest driven joint first
        c = by_driven[name]
        c["reflected_inertia"] = c["coeff"][1] ** 2 * eff[name]
        if c["driver"] in eff:
            eff[c["driver"]] += c["reflected_inertia"]
    depth = {name: i for i, name in enumerate(order)}
    for c in out:
        c["chain_order"] = int(depth.get(c["driven"], 0))
    return out


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
    servo_joints = {}
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
        jt = b.joint
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
        servo = servo_drive_params(jt, servo_for_joint(model.meta.get("actuators"), jt.name))
        W.add_dof_joint(jp, jt.type, body0, body_paths[b.name], p0, q0, jpos, qa, lo, hi,
                        stiffness=float(jt.stiffness), damping=float(jt.damping), target=(float(jt.springref - jt.modeled_at) if jt.stiffness else 0.0),
                        frictionloss=float(jt.frictionloss), armature=float(jt.armature), reaction_force=_reaction_estimate(model, b, tier, wt), extra=extra, servo=servo)
        if servo is not None:
            servo_joints[jt.name] = {k: servo[k] for k in ("kp", "kv", "force_limit", "ctrl")}
        joint_paths[jt.name] = jp
        joint_types[jt.name] = jt.type
    # --- couplings: PhysxMimicJointAPI where PhysX honours it (rotational -> rotational), explicit bilateral
    #     coupling metadata for the rest (hinge -> slide, slide -> slide: PhysX drops those mimics silently)
    def _body_of_joint(jn):
        for b in bodies:
            if b.joint is not None and b.joint.name == jn:
                return b
        return None

    couplings = []
    coupling_specs = joint_couplings(model, tier, joint_types, wt, _body_of_joint, servo_joints=set(servo_joints))
    reflected = {}
    for c in sorted(coupling_specs, key=lambda c: c["chain_order"]):
        dpath, rpath = joint_paths[c["driven"]], joint_paths[c["driver"]]
        if c["mode"] == "mimic":
            # IR: q_a = c0 + c1 * q_b (USD coordinates)  ->  q_a + (-c1) * q_b + (-c0) = 0
            W.add_mimic(dpath, joint_types[c["driven"]], rpath, joint_types[c["driver"]], gearing=c["gearing"], offset=c["offset"], label=c["label"])
        W.add_coupling(c, dpath, rpath, reflected)
        couplings.append(dict(c, type="mimic" if c["mode"] == "mimic" else "coupling_emulated", driven_path=str(dpath), driver_path=str(rpath),
                              note=("PhysxMimicJointAPI on the driven joint; q_driven + gearing*q_driver + offset = 0 in rad / m"
                                    if c["mode"] == "mimic" else
                                    "PhysX drops mimics on prismatic axes: emulate bilaterally (track q_driven = c0 + c1*q_driver, apply "
                                    "c1 * tau_driven_ext on the driver and add reflected_inertia to the driver's armature)")))
    for jn, val in reflected.items():
        W.set_reflected_inertia(joint_paths[jn], val)
    for q in model.equalities:
        if tier not in q.tiers:
            continue
        if q.kind == "connect":
            couplings.append({"type": "loop_closure_point", "body1": q.a, "body2": q.b, "anchor": list(q.anchor), "label": q.label,
                              "note": "not exported to USD (PhysX articulations are trees); closer arms are visual"})
    # --- environment-released locks: a real breakable FixedJoint outside the articulation tree
    env_release = []
    for w in env_release_welds(model, tier):
        if w["body"] not in body_paths:
            continue
        jp = joints_path.AppendChild(_safe(w["name"]))
        pos, quat = wt[w["body"]]
        prim = W.add_env_release_joint(jp, base_path, body_paths[w["body"]], pos, quat, [0, 0, 0], [1, 0, 0, 0],
                                       w["holding_force_N"], w, enabled=w["active"])
        env_release.append(dict(w, joint=str(jp), joint_name=prim.GetName(), body_prim=str(body_paths[w["body"]]), base_prim=str(base_path)))
        couplings.append({"type": "weld", "body1": w["body"], "body2": w["other"], "label": w["label"], "active": w["active"],
                          "joint": str(jp), "holding_force_N": w["holding_force_N"],
                          "note": "breakable FixedJoint (excludeFromArticulation); the environment clears physics:jointEnabled on release"})
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
    # --- collision filtering: PhysX skips joint-adjacent links, MuJoCo skips weld groups / weld parent-child /
    #     contact_excludes -> author the difference (and the adjacent ones, harmlessly) so both engines agree
    filtered = mujoco_filtered_pairs(model, bodies, tier)
    n_filtered = W.add_filtered_pairs([(body_paths[a], body_paths[b]) for a, b in filtered if a in body_paths and b in body_paths])
    W.set_json(W.root.GetPrim(), "doorbench:couplings", couplings)
    W.set_json(W.root.GetPrim(), "doorbench:env_release", env_release)
    W.set_json(W.root.GetPrim(), "doorbench:filtered_pairs", [list(x) for x in filtered])
    meta = {k: v for k, v in model.meta.items() if k not in ("notes",)}
    meta["usd_layout"] = "v2: default prim = door root; Env (static) + Articulation (fixed base link `base`)"
    meta["joints"] = {name: str(p) for name, p in joint_paths.items()}
    meta["self_collisions"] = True
    meta["filtered_pairs"] = [list(x) for x in filtered]
    meta["n_filtered_pairs"] = int(n_filtered)
    meta["env_release"] = env_release
    meta["couplings_emulated"] = [c for c in coupling_specs if c["mode"] == "emulated"]
    meta["couplings_servo"] = [c["driven"] for c in coupling_specs if c["mode"] == "servo"]
    meta["coupling_reflected_armature"] = reflected
    # physics mapping notes for runners (docs/ISAAC_LAB.md): servos folded into drives, rising-hinge gravity torque
    meta["servo_in_drive"] = servo_joints
    if meta.get("actuators"):
        meta["actuators"] = [dict(a, in_drive=a.get("joint") in servo_joints) for a in meta["actuators"]]
    primary = next((b for b in bodies if b.joint is not None and b.joint.name == model.meta.get("primary_joint")), None)
    meta["rise_coupling"] = rise_coupling_info(model, primary, tier, wt)
    meta["max_angular_velocity_deg_s"] = float(MAX_ANGULAR_VELOCITY_DEG_S)
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


def _engaged_in_ir(jt) -> bool:
    """True when the joint sits at its ENGAGED end at q0.

    Every latch / lock part in the IR is authored with ``range = (0, travel)`` and ``0 = engaged`` (thrown bolt,
    hooked hook, dropped pin, dogged dog); ``initial`` (== ``modeled_at`` after ``bake_initial``) is 0 when the spec
    says engaged and the travel end when it does not."""
    if jt is None or jt.range is None:
        return False
    lo, hi = float(jt.range[0]), float(jt.range[1])
    if hi - lo <= 1e-9:
        return False
    return abs(float(jt.modeled_at) - lo) <= 0.1 * (hi - lo)


def operator_driven_joints(model: Model, operator_joint: str | None, tier: str):
    """Joints the operator drives through bilateral equalities / tendons (transitive, driver -> driven).

    A lock part in that set retracts when the robot works the operator (cremone shoot bolts on the handle, hook
    bolts on the hook slider, dogs on a ship wheel), so the canonical RL articulation - which has no slot for it -
    must weld it RELEASED; a lock part outside the set (a thumbturn deadbolt, a second dog with its own lever) needs
    its own release the canonical file cannot offer and stays welded engaged."""
    if not operator_joint:
        return set()
    edges = {}
    for q in model.equalities:
        if q.kind == "joint" and tier in q.tiers and q.b:
            edges.setdefault(q.b, set()).add(q.a)
    for t in model.tendons:
        if tier not in t.tiers or len(t.sites) < 2:
            continue
        names = [x[0] for x in t.sites]
        for driver in names[1:]:
            edges.setdefault(driver, set()).add(names[0])
    out, stack = set(), [operator_joint]
    while stack:
        n = stack.pop()
        for m in edges.get(n, ()):
            if m not in out:
                out.add(m)
                stack.append(m)
    return out


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
    # ---------------------------------------------------------------- welded pose shifts (released vs engaged)
    # Every moving part that has no canonical slot is welded into its link.  Welding it at the INITIAL state locks a
    # door whose release part is a revolute hook, a cremone shoot bolt or a dog coupled to the operator: the protocol
    # expects those doors to open (the robot works the operator, MuJoCo's coupling retracts the part), so they are
    # welded RELEASED instead.  The decision is recorded per part in ``doorbench:rl`` (``welded`` / ``released_parts``
    # / ``welded_engaged``) so the parity protocol reads the ground truth rather than guessing from the spec.
    spec_lock = (spec or {}).get("lock", {}) if isinstance(spec, dict) else {}
    lock_engaged_spec = bool(spec_lock.get("engaged"))
    robot_can_release = bool(spec_lock.get("robot_side_release", True))
    op_driven = operator_driven_joints(model, meta.get("operator_joint"), tier)
    leaf_normal_w = quat_rotate(leaf_quat, np.array([0.0, 1.0, 0.0]))

    def _press_only(b):
        """A slide that moves along the leaf's normal is a BUTTON, not a bolt: it presses into the leaf face and can
        never reach the frame, so welding it in either state cannot hold or release the leaf (keypad keys, REX and
        call buttons, privacy buttons).  A bolt / rod / pin moves in the plane of the leaf, toward an edge."""
        jt = b.joint
        if jt is None or jt.type != "slide":
            return False
        pos, quat = wt[b.name]
        ax = quat_rotate(quat, np.asarray(jt.axis, float) / np.linalg.norm(jt.axis))
        return abs(float(np.dot(ax, leaf_normal_w))) > 0.9

    shift, weld_record = {}, []
    for n in leaf_bodies + static_extra:
        b = model.body(n)
        if b.joint is None or n in (primary.name,):
            continue
        role, jt = b.joint.role, b.joint
        engaged = _engaged_in_ir(jt)
        # can this part hold the leaf shut at all?  Only a bolt / hook / rod / pin that is in its engaged state; a
        # button (presses into the face), a sensor or a decoration never does.
        holding = (engaged and b.semantic not in ("sensor", "decor")
                   and (role in ("latch", "lock") or b.semantic in ("latch", "lock")) and not _press_only(b))
        if role == "latch":
            released, why = True, "spring latch hardware never blocks the canonical leaf"
        elif role == "operator" and n in static_extra:
            released, why = True, "world-mounted operator welded static in its released state"
        elif role == "lock" and lock_engaged_spec and not robot_can_release:
            # no robot-side release (keyed outside only, padlock, multipoint with no inside trim): the robot cannot
            # work the operator to retract this part, so the real door stays locked and so must the canonical one
            released, why = False, "engaged lock with no robot-side release: welded engaged (the door must stay locked)"
        elif b.semantic == "latch":
            released, why = True, "latch hardware never blocks the canonical leaf"
        elif role == "lock" and jt.name in op_driven:
            released, why = True, f"lock part driven by the operator ({meta.get('operator_joint')}): retracts when the robot works the operator"
        elif role == "lock" and (not lock_engaged_spec or not engaged):
            released, why = True, "lock not engaged in the spec / IR"
        elif role == "lock" and n in static_extra and jt.type == "slide" and "pin" in n:
            released, why = True, "world-mounted lift pin welded static in its released state"
        else:
            released, why = False, ("engaged lock with no canonical slot and no operator coupling: welded engaged"
                                    if holding else "part welded at its initial state")
        dq = _released_pose_shift(model, b) if released else 0.0
        if released:
            shift[n] = dq
        weld_record.append({"body": n, "joint": jt.name, "role": role, "semantic": b.semantic, "type": jt.type,
                            "released": bool(released), "shift": float(dq), "was_engaged": bool(engaged),
                            "holding": bool(holding), "press_only": bool(_press_only(b)),
                            "link": "static" if n in static_extra else "leaf", "reason": why})
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
    servo_slots = {}          # MJCF joint -> canonical joint whose drive carries the position servo

    def write_dof(name, jtype, parent, child, anchor_w, axis_w, src: Body | None, active: bool, reaction=None):
        pos0, rot0, pos1, rot1 = joint_frames(parent, child, anchor_w, axis_w)
        revolute = jtype == "revolute"
        if active and src is not None and src.joint is not None:
            jt = src.joint
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
            servo = servo_drive_params(jt, servo_for_joint(meta.get("actuators"), jt.name))
            W.add_dof_joint(joints_path.AppendChild(name), jtype, link_paths[parent], link_paths[child], pos0, rot0, pos1, rot1, lo, hi,
                            stiffness=float(jt.stiffness), damping=float(jt.damping), target=(float(jt.springref - jt.modeled_at) if jt.stiffness else 0.0),
                            frictionloss=float(jt.frictionloss), armature=float(jt.armature), reaction_force=reaction, extra=extra, servo=servo)
            joint_meta[name] = {"active": True, "type": jtype, "source": jt.name, "role": jt.role, "range": [float(lo), float(hi)], "stiffness": float(jt.stiffness),
                                "damping": float(jt.damping), "target": (float(jt.springref - jt.modeled_at) if jt.stiffness else 0.0), "friction": float(jt.frictionloss),
                                "armature": float(jt.armature),
                                "damping_closing": jt.damping_closing, "damping_opening": jt.damping_opening, "backcheck_angle": jt.backcheck_angle, "backcheck_damping": jt.backcheck_damping,
                                "ratchet_one_way": bool(jt.ratchet_one_way), "label": jt.label}
            if servo is not None:
                # the drive carries the servo: gains / target the environment will read back from PhysX
                joint_meta[name]["servo"] = {k: servo[k] for k in ("kp", "kv", "force_limit", "ctrl")}
                joint_meta[name]["drive"] = {"stiffness": servo["stiffness"], "damping": servo["damping"], "target": servo["target"], "max_force": servo["force_limit"]}
                servo_slots[jt.name] = name
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
    # ---------------------------------------------------------------- couplings between canonical slots
    # Almost every mechanism coupling is welded away in this file; what survives is a bilateral equality between two
    # ACTIVE slots (bi-parting sliders / elevators: leaf2 = c1 * leaf).  It is authored the same way as in
    # door.usda - PhysxMimicJointAPI when both canonical joints are revolute (PhysX honours rotational mimics only),
    # otherwise the doorbench:coupling_* emulation data that DoorMechanismAction applies with the reaction on the
    # driver.  The helical riser's coupling is the other survivor and keeps its own ``rise_coupling`` entry (the same
    # reaction, c1 * gravity_bias, precomputed because the riser has no slot at all).
    slot_of = {info["source"]: name for name, info in joint_meta.items() if info.get("active") and info.get("source")}
    slot_type = {name: ("hinge" if name.endswith("hinge") else "slide") for name in RL_DOF_JOINTS}

    def _rl_body_of_joint(jn):
        return body_of_joint(jn)

    rl_couplings = []
    reflected_rl = {}
    for c in sorted(joint_couplings(model, tier, {b.joint.name: b.joint.type for b in bodies if b.joint is not None}, wt, _rl_body_of_joint,
                                    servo_joints=set(servo_slots)), key=lambda c: c["chain_order"]):
        sa, sb = slot_of.get(c["driven"]), slot_of.get(c["driver"])
        if sa is None or sb is None:
            continue
        rot = slot_type[sa] == "hinge" and slot_type[sb] == "hinge"
        mode = "mimic" if rot else c["mode"]
        entry = dict(c, mode=mode, driven_slot=sa, driver_slot=sb)
        dpath, rpath = joints_path.AppendChild(sa), joints_path.AppendChild(sb)
        if rot:
            W.add_mimic(dpath, "hinge", rpath, "hinge", gearing=entry["gearing"], offset=entry["offset"], label=entry["label"])
        W.add_coupling(dict(entry, driver=sb), dpath, rpath, reflected_rl)
        rl_couplings.append(entry)
    for slot, val in reflected_rl.items():
        W.set_reflected_inertia(joints_path.AppendChild(slot), val)
    # ---------------------------------------------------------------- collision filtering (link level)
    # every IR body lives in exactly one canonical link; PhysX skips joint-adjacent links, so a link pair that is not
    # adjacent must be filtered whenever MuJoCo suppressed any of the body pairs it merges (leaf <-> operator: the
    # handle is a child of the leaf; leaf <-> leaf2 stays colliding, which is what latches a swing pair)
    link_of_body = {}
    for lname, blist in (("leaf", leaf_bodies), ("operator", op_bodies), ("latch", latch_bodies), ("leaf2", sec_bodies)):
        for n in blist:
            link_of_body[n] = lname
    RL_ADJACENT = {("base", "carriage"), ("carriage", "leaf"), ("leaf", "operator_pivot"), ("operator_pivot", "operator"),
                   ("leaf", "latch"), ("base", "carriage2"), ("carriage2", "leaf2")}
    RL_ADJACENT = {tuple(sorted(x)) for x in RL_ADJACENT}
    body_pairs = mujoco_filtered_pairs(model, bodies, tier)
    link_pairs = set()
    for a, b in body_pairs:
        la, lb = link_of_body.get(a), link_of_body.get(b)
        if la is None or lb is None or la == lb:
            continue
        pair = tuple(sorted((la, lb)))
        if pair not in RL_ADJACENT:
            link_pairs.add(pair)
    n_filtered = W.add_filtered_pairs([(link_paths[a], link_paths[b]) for a, b in sorted(link_pairs)])
    # ---------------------------------------------------------------- environment-released locks
    env_release = []
    for w in env_release_welds(model, tier):
        link = link_of_body.get(w["body"])
        if link is None:
            notes.append(f"env-release weld {w['name']} on {w['body']}, which is not part of the canonical articulation: not exported")
            continue
        jp = joints_path.AppendChild(_safe(w["name"]))
        fp, fq = link_frames[link]
        prim = W.add_env_release_joint(jp, base_path, link_paths[link], fp, fq, [0, 0, 0], [1, 0, 0, 0],
                                       w["holding_force_N"], w, enabled=w["active"])
        env_release.append(dict(w, joint=str(jp), joint_name=prim.GetName(), link=link, link_prim=str(link_paths[link]), base_prim=str(base_path)))
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
        # position servos: ``in_drive`` = the canonical joint's PhysX drive already IS the servo (kp / kv / forcerange),
        # the environment only moves its position target; otherwise it has to emulate the servo as feed-forward effort
        "actuators": [dict(a, slot=servo_slots.get(a.get("joint")), in_drive=a.get("joint") in servo_slots) for a in meta.get("actuators", [])],
        # rising / helical hinge: the riser is locked in this file -> apply the gravity closing torque on the door joint
        "rise_coupling": rise_coupling_info(model, primary, tier, wt),
        "max_angular_velocity_deg_s": float(MAX_ANGULAR_VELOCITY_DEG_S),
        "welded_static": static_extra, "omitted": omitted, "notes": notes,
        "mass_kg": float(_subtree_mass(model, primary.name, tier, wt)[0]),
        # ground truth for the parity protocol: which mechanism parts this file welded, and in which state.
        # ``released_parts`` are welded RELEASED (they cannot hold the leaf here even though they do in MuJoCo),
        # ``welded_engaged`` are welded ENGAGED (they hold the leaf shut and no canonical slot can release them).
        "welded": weld_record,
        "released_parts": [w for w in weld_record if w["released"]],
        "released_holding": [w for w in weld_record if w["released"] and w["holding"]],
        "welded_engaged": [w for w in weld_record if not w["released"] and w["holding"]],
        "operator_driven_joints": sorted(op_driven),
        "couplings": rl_couplings,
        "coupling_reflected_armature": reflected_rl,
        "self_collisions": True,
        "filtered_pairs": [list(x) for x in sorted(link_pairs)],
        "n_filtered_pairs": int(n_filtered),
        "env_release": env_release,
    }
    if rl["rise_coupling"] is not None:
        notes.append(f"rising hinge {rl['rise_coupling']['rise_joint']} locked: apply gravity closing torque {rl['rise_coupling']['gravity_torque_Nm']:.3f} N*m on {rl['door_joint']}")
    W.set_json(W.root.GetPrim(), "doorbench:rl", rl)
    W.set_json(W.root.GetPrim(), "doorbench:env_release", env_release)
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

#!/usr/bin/env python
"""Write the 6-DoF "gantry hand" agent used by DoorBench-Open-Hand-v0 (pxr only, runs anywhere).

A fixed-base articulation: three prismatic joints (x, y, z) then three revolute joints (yaw, pitch, roll) carrying a
palm (sphere) with a finger bar (an L-shape that can press levers, push plates and hook behind pull handles).  All
joints are driven by implicit PD drives (position targets); Isaac Lab's ``ImplicitActuatorCfg`` reads the gains
from the USD.  Output: isaaclab/doorbench_isaaclab/data/gantry_hand.usda

Links   base -> hand_x -> hand_y -> hand_z -> hand_yaw -> hand_pitch -> palm
Joints  hand_x, hand_y, hand_z (prismatic, m), hand_yaw, hand_pitch, hand_roll (revolute, deg in USD)
Frames  joint value 0 = palm at the base origin, finger pointing +y (toward the door when the base sits at the
        approach point); the environment places the base at the door's approach point.
"""
from __future__ import annotations

import argparse
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT = os.path.join(ROOT, "isaaclab", "doorbench_isaaclab", "data", "gantry_hand.usda")

HAND_JOINTS = (
    # name, type, axis (world at q=0), lower, upper, stiffness, damping, max_force, mass of the link it moves
    ("hand_x", "prismatic", (1, 0, 0), -1.5, 1.5, 4000.0, 400.0, 400.0, 0.5),
    ("hand_y", "prismatic", (0, 1, 0), -1.0, 3.5, 4000.0, 400.0, 400.0, 0.5),
    ("hand_z", "prismatic", (0, 0, 1), 0.05, 2.2, 4000.0, 400.0, 400.0, 0.5),
    ("hand_yaw", "revolute", (0, 0, 1), -math.pi, math.pi, 60.0, 6.0, 60.0, 0.3),
    ("hand_pitch", "revolute", (1, 0, 0), -math.pi / 2, math.pi / 2, 60.0, 6.0, 60.0, 0.3),
    ("hand_roll", "revolute", (0, 1, 0), -math.pi, math.pi, 30.0, 3.0, 30.0, 0.8),
)
PALM_RADIUS = 0.055
FINGER_LEN = 0.12
FINGER_R = 0.018


def _quat_x_to(d):
    import numpy as np
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    x = np.array([1.0, 0, 0])
    c = float(np.dot(x, d))
    if c > 1 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if c < -1 + 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    ax = np.cross(x, d)
    ax = ax / np.linalg.norm(ax)
    ang = math.acos(c)
    return (math.cos(ang / 2), *(ax * math.sin(ang / 2)))


def write_hand(path: str = DEFAULT_OUT, start_z: float = 1.0):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/gantry_hand")
    stage.SetDefaultPrim(root.GetPrim())
    rp = root.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(rp)
    rp.AddAppliedSchema("PhysxArticulationAPI")
    rp.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    rp.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
    rp.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    rp.CreateAttribute("physxArticulation:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    rp.CreateAttribute("doorbench:agent", Sdf.ValueTypeNames.String).Set("gantry_hand")
    # materials
    looks = UsdGeom.Scope.Define(stage, "/gantry_hand/Looks")
    mat = UsdShade.Material.Define(stage, "/gantry_hand/Looks/hand")
    sh = UsdShade.Shader.Define(stage, "/gantry_hand/Looks/hand/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.45, 0.1))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    pm = UsdShade.Material.Define(stage, "/gantry_hand/Looks/PhysMat_hand")
    pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    pma.CreateStaticFrictionAttr(1.0)
    pma.CreateDynamicFrictionAttr(0.9)
    pma.CreateRestitutionAttr(0.0)

    def link(name, mass, inertia=0.001):
        xf = UsdGeom.Xform.Define(stage, f"/gantry_hand/{name}")
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        xf.AddOrientOp().Set(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
        p = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p)
        p.AddAppliedSchema("PhysxRigidBodyAPI")
        p.CreateAttribute("physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
        p.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(3.0)
        m = UsdPhysics.MassAPI.Apply(p)
        m.CreateMassAttr(float(mass))
        m.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia, inertia, inertia))
        m.CreatePrincipalAxesAttr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
        return p

    joints = UsdGeom.Scope.Define(stage, "/gantry_hand/Joints")
    base = link("base", 5.0, 0.05)
    fj = UsdPhysics.FixedJoint.Define(stage, "/gantry_hand/Joints/base_fixed")
    fj.CreateBody1Rel().SetTargets([base.GetPath()])
    for attr, val in (("LocalPos0", Gf.Vec3f(0, 0, 0)), ("LocalPos1", Gf.Vec3f(0, 0, 0))):
        getattr(fj, f"Create{attr}Attr")(val)
    fj.CreateLocalRot0Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
    fj.CreateLocalRot1Attr(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
    parent = base
    link_names = ["hand_x", "hand_y", "hand_z", "hand_yaw", "hand_pitch", "palm"]
    for (jn, jt, axis, lo, hi, k, b, fmax, mass), ln in zip(HAND_JOINTS, link_names):
        child = link(ln, mass, 0.002 if ln == "palm" else 0.0005)
        rev = jt == "revolute"
        cls = UsdPhysics.RevoluteJoint if rev else UsdPhysics.PrismaticJoint
        j = cls.Define(stage, f"/gantry_hand/Joints/{jn}")
        j.CreateBody0Rel().SetTargets([parent.GetPath()])
        j.CreateBody1Rel().SetTargets([child.GetPath()])
        j.CreateAxisAttr("X")
        q = _quat_x_to(axis)
        j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
        j.CreateLocalRot1Attr(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
        conv = 180.0 / math.pi if rev else 1.0
        j.CreateLowerLimitAttr(float(lo * conv))
        j.CreateUpperLimitAttr(float(hi * conv))
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular" if rev else "linear")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(float(k / conv))
        drv.CreateDampingAttr(float(b / conv))
        drv.CreateTargetPositionAttr(0.0)
        drv.CreateMaxForceAttr(float(fmax))
        jp = j.GetPrim()
        jp.AddAppliedSchema("PhysxJointAPI")
        jp.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.05 if not rev else 0.005)
        jp.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(0.0)
        jp.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(3.0 if not rev else 720.0)
        parent = child
    # palm geometry: sphere + finger bar (+y) ; collision on both
    palm = stage.GetPrimAtPath("/gantry_hand/palm")
    sph = UsdGeom.Sphere.Define(stage, "/gantry_hand/palm/sphere")
    sph.CreateRadiusAttr(PALM_RADIUS)
    fin = UsdGeom.Capsule.Define(stage, "/gantry_hand/palm/finger")
    fin.CreateRadiusAttr(FINGER_R)
    fin.CreateHeightAttr(FINGER_LEN)
    fin.CreateAxisAttr("Y")
    fx = UsdGeom.Xformable(fin)
    fx.ClearXformOpOrder()
    fx.AddTranslateOp().Set(Gf.Vec3d(0, PALM_RADIUS + FINGER_LEN / 2 - 0.01, 0))
    tip = UsdGeom.Capsule.Define(stage, "/gantry_hand/palm/finger_tip")
    tip.CreateRadiusAttr(FINGER_R)
    tip.CreateHeightAttr(0.06)
    tip.CreateAxisAttr("Z")
    tx = UsdGeom.Xformable(tip)
    tx.ClearXformOpOrder()
    tx.AddTranslateOp().Set(Gf.Vec3d(0, PALM_RADIUS + FINGER_LEN - 0.01, -0.03))
    for g in (sph, fin, tip):
        p = g.GetPrim()
        UsdPhysics.CollisionAPI.Apply(p)
        p.AddAppliedSchema("PhysxCollisionAPI")
        p.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.005)
        p.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat)
        UsdShade.MaterialBindingAPI.Apply(p).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    # a marker for the hand "tip" (where the policy should bring the handle)
    site = UsdGeom.Xform.Define(stage, "/gantry_hand/palm/tip_site")
    sx = UsdGeom.Xformable(site)
    sx.ClearXformOpOrder()
    sx.AddTranslateOp().Set(Gf.Vec3d(0, PALM_RADIUS + FINGER_LEN * 0.6, 0))
    site.GetPrim().CreateAttribute("doorbench:site_role", Sdf.ValueTypeNames.String).Set("tip")
    rp.CreateAttribute("doorbench:joints", Sdf.ValueTypeNames.String).Set(",".join(j[0] for j in HAND_JOINTS))
    stage.GetRootLayer().Save()
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    print(write_hand(a.out))

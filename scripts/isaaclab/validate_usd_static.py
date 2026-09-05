#!/usr/bin/env python
"""Static (pxr-only, no Isaac Sim needed) validation of every door USD in the dataset.

Checks per file (door.usda = full articulation, door_rl.usda = canonical 7-DoF RL articulation):
  stage        default prim set, up axis Z, metersPerUnit 1, no NaN/Inf in any float attribute
  articulation exactly one PhysicsArticulationRootAPI prim, one fixed joint to the world (`base_fixed`),
               every rigid body reachable from `base` through joints (single tree, no cycles), body1 of every
               joint is a rigid body, body0 is a rigid body or the world (base_fixed only)
  bodies       every RigidBodyAPI prim has MassAPI with mass > 0 and positive finite principal inertia
  joints       Revolute/Prismatic joints: limits present, lower <= upper (strictly lower < upper unless the joint is a
               locked RL slot), local frames consistent (anchor and axis computed through body0 and body1 coincide),
               unit quaternions, drive (force type) present with finite stiffness/damping, PhysX joint-axis friction
               efforts present and >= 0 on the ``angular`` (revolute) / ``linear`` (prismatic) instance of
               PhysxJointAxisAPI - the only instance names the PhysX USD parser reads on single-DoF joints - and the
               legacy load-dependent ``physxJoint:jointFriction`` coefficient authored as 0 (no double friction);
               links carry ``physxRigidBody:maxAngularVelocity`` >= 1000 deg/s (a 100 deg/s cap clamps a leaf at 1.75 rad/s)
  vs model.json  full USD: joint names == model.json joint names; limits, spring stiffness, damping, spring target,
               armature and Coulomb friction match the IR (unit conversion checked); MJCF position servos folded into
               the drive (``doorbench:servo_in_drive``) add kp / kv to the gains and set maxForce = forcerange;
               rl USD: exactly the canonical joints/links
  collision    every collision geom of model.json (in tier) exists with PhysicsCollisionAPI; mesh colliders have
               MeshCollisionAPI convexHull; collision prims carry a physics material binding that resolves
  meshes       every reference resolves to an existing assets/hardware/*.usdc containing the referenced Mesh prim
  materials    material:binding targets exist
  doorbench:*  every string attribute holding JSON parses; RL meta has the slots / sites the environment needs

Usage
  python scripts/isaaclab/validate_usd_static.py [--assets assets] [--ids a,b] [--limit N] [--workers 8]
                                                 [--out assets/usd_validation.json]
Exit status 1 if any file fails.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

RL_JOINTS = ("door_slide", "door_hinge", "operator_hinge", "operator_slide", "latch_slide", "leaf2_slide", "leaf2_hinge")
RL_LINKS = ("base", "carriage", "leaf", "operator_pivot", "operator", "latch", "carriage2", "leaf2")


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _gfq(q):
    return np.array([q.GetReal(), *q.GetImaginary()], float)


def _applied_schemas(prim):
    """Applied API schema tokens, including multi-apply schemas this pxr build does not know (PhysX)."""
    out = set(prim.GetAppliedSchemas())
    md = prim.GetMetadata("apiSchemas")
    if md is not None:
        try:
            out |= set(md.GetAddedOrExplicitItems())
        except Exception:
            pass
    return out


def _finite(v):
    try:
        arr = np.asarray(v, float)
        return bool(np.all(np.isfinite(arr)))
    except Exception:
        return True


class Report:
    def __init__(self):
        self.errors, self.warnings, self.stats = [], [], {}

    def err(self, msg):
        if len(self.errors) < 60:
            self.errors.append(msg)

    def warn(self, msg):
        if len(self.warnings) < 60:
            self.warnings.append(msg)


def _world_xform(UsdGeom, prim):
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(0)
    R = np.array([[m[i][j] for i in range(3)] for j in range(3)], float)  # Gf matrices are row-vector convention
    t = np.array([m[3][0], m[3][1], m[3][2]], float)
    return t, R


def validate_stage(path: str, kind: str, model_json: dict | None, spec: dict | None) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
    R = Report()
    t0 = time.time()
    try:
        stage = Usd.Stage.Open(path)
    except Exception as e:
        R.err(f"cannot open: {e}")
        return {"ok": False, "errors": R.errors, "warnings": R.warnings, "stats": R.stats}
    if stage is None:
        R.err("cannot open stage")
        return {"ok": False, "errors": R.errors, "warnings": R.warnings, "stats": R.stats}
    # ---- stage metadata
    dp = stage.GetDefaultPrim()
    if not dp or not dp.IsValid():
        R.err("no default prim")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        R.err(f"up axis {UsdGeom.GetStageUpAxis(stage)} != Z")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        R.err(f"metersPerUnit {UsdGeom.GetStageMetersPerUnit(stage)} != 1")
    prims = list(stage.Traverse())
    R.stats["n_prims"] = len(prims)
    # ---- NaN scan + doorbench json attributes
    n_nan = 0
    json_attrs = {}
    for p in prims:
        for a in p.GetAttributes():
            if not a.HasAuthoredValue():
                continue  # schema fallbacks (e.g. physics:breakForce = inf) are not ours to validate
            try:
                v = a.Get()
            except Exception:
                continue
            if v is None:
                continue
            tn = a.GetTypeName()
            if tn in (Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Double):
                if not math.isfinite(float(v)):
                    n_nan += 1
                    R.err(f"non-finite {a.GetPath()}")
            elif "float" in str(tn).lower() or "double" in str(tn).lower() or "point" in str(tn).lower() or "vector" in str(tn).lower() or "quat" in str(tn).lower():
                try:
                    vals = _gfq(v) if "quat" in str(tn).lower() else np.asarray([list(x) if hasattr(x, "__len__") else x for x in (v if hasattr(v, "__len__") and not hasattr(v, "GetReal") else [v])], float)
                    if not np.all(np.isfinite(vals)):
                        n_nan += 1
                        R.err(f"non-finite {a.GetPath()}")
                except Exception:
                    pass
            if a.GetName().startswith("doorbench:") and tn == Sdf.ValueTypeNames.String and isinstance(v, str) and v[:1] in ("{", "["):
                try:
                    json_attrs[str(a.GetPath())] = json.loads(v)
                except Exception as e:
                    R.err(f"doorbench JSON attribute {a.GetPath()} does not parse: {e}")
    R.stats["n_nonfinite"] = n_nan
    # ---- articulation root(s)
    roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    R.stats["n_articulation_roots"] = len(roots)
    if len(roots) != 1:
        R.err(f"{len(roots)} articulation roots (expected 1)")
    root = roots[0] if roots else None
    if root is not None:
        if dp and not str(root.GetPath()).startswith(str(dp.GetPath())):
            R.err("articulation root is not under the default prim")
        if root.HasAPI(UsdPhysics.RigidBodyAPI):
            R.err("articulation root API on a rigid body (must be on the ancestor Xform for a fixed base)")
        for an in ("physxArticulation:enabledSelfCollisions", "physxArticulation:solverPositionIterationCount"):
            if not root.GetAttribute(an).IsValid():
                R.warn(f"missing {an} on the articulation root")
    # ---- rigid bodies
    bodies = {}
    for p in prims:
        if not p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = str(p.GetPath())
        bodies[name] = p
        if not p.HasAPI(UsdPhysics.MassAPI):
            R.err(f"{name}: RigidBodyAPI without MassAPI")
            continue
        m = p.GetAttribute("physics:mass").Get()
        if m is None or not (float(m) > 0):
            R.err(f"{name}: mass {m} <= 0")
        di = p.GetAttribute("physics:diagonalInertia").Get()
        if di is None or not all(float(x) > 0 and math.isfinite(float(x)) for x in di):
            R.err(f"{name}: diagonal inertia {di}")
        pa = p.GetAttribute("physics:principalAxes").Get()
        if pa is not None:
            qn = np.linalg.norm(_gfq(pa))
            if abs(qn - 1) > 1e-3:
                R.err(f"{name}: principal axes quaternion norm {qn}")
        # PhysX caps the angular velocity of every link (schema unit deg/s).  MuJoCo has no cap; anything below
        # 1000 deg/s (17 rad/s) would clamp door leaves in the protocol's free swings (3-5 rad/s reach the cap's
        # numerical neighbourhood at 100 deg/s; pet flaps reach 65 rad/s).
        mav = p.GetAttribute("physxRigidBody:maxAngularVelocity")
        if not (mav.IsValid() and mav.HasAuthoredValue()):
            R.err(f"{name}: physxRigidBody:maxAngularVelocity not authored")
        elif float(mav.Get()) < 1000.0:
            R.err(f"{name}: physxRigidBody:maxAngularVelocity {float(mav.Get()):.1f} deg/s caps the link at {math.radians(float(mav.Get())):.2f} rad/s")
        if root is not None and not str(p.GetPath()).startswith(str(root.GetPath())):
            R.err(f"{name}: rigid body outside the articulation root")
    R.stats["n_rigid_bodies"] = len(bodies)
    # world transforms of bodies (through xformOps)
    body_tf = {n: _world_xform(UsdGeom, p) for n, p in bodies.items()}
    # ---- joints
    joints = {}
    world_joints = []
    children_of = {n: [] for n in bodies}
    parent_of = {}
    dof_joint_names = set()
    for p in prims:
        is_rev, is_pri, is_fix = p.IsA(UsdPhysics.RevoluteJoint), p.IsA(UsdPhysics.PrismaticJoint), p.IsA(UsdPhysics.FixedJoint)
        if not (is_rev or is_pri or is_fix):
            if p.IsA(UsdPhysics.Joint):
                R.err(f"{p.GetPath()}: unsupported joint type {p.GetTypeName()}")
            continue
        jname = p.GetName()
        jp = UsdPhysics.Joint(p)
        b0 = [str(t) for t in jp.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in jp.GetBody1Rel().GetTargets()]
        if len(b1) != 1 or b1[0] not in bodies:
            R.err(f"{jname}: body1 {b1} is not a rigid body")
            continue
        if len(b0) == 0:
            world_joints.append(jname)
            if not is_fix:
                R.err(f"{jname}: non-fixed joint to the world (fixed base must use a fixed joint)")
        elif len(b0) != 1 or b0[0] not in bodies:
            R.err(f"{jname}: body0 {b0} is not a rigid body")
            continue
        joints[jname] = p
        if b1[0] in parent_of:
            R.err(f"{jname}: body {b1[0]} is the child of two joints ({parent_of[b1[0]]}, {jname})")
        parent_of[b1[0]] = jname if not b0 else b0[0]
        if b0:
            children_of[b0[0]].append(b1[0])
        # local frames
        lp0 = np.asarray(list(jp.GetLocalPos0Attr().Get() or (0, 0, 0)), float)
        lp1 = np.asarray(list(jp.GetLocalPos1Attr().Get() or (0, 0, 0)), float)
        lr0 = _gfq(jp.GetLocalRot0Attr().Get()) if jp.GetLocalRot0Attr().Get() is not None else np.array([1.0, 0, 0, 0])
        lr1 = _gfq(jp.GetLocalRot1Attr().Get()) if jp.GetLocalRot1Attr().Get() is not None else np.array([1.0, 0, 0, 0])
        for nm, q in (("localRot0", lr0), ("localRot1", lr1)):
            if abs(np.linalg.norm(q) - 1) > 1e-3:
                R.err(f"{jname}: {nm} not unit ({np.linalg.norm(q):.4f})")
        t1, R1 = body_tf[b1[0]]
        anchor1 = t1 + R1 @ lp1
        axis1 = R1 @ _quat_to_mat(lr1) @ np.array([1.0, 0, 0])
        if b0:
            t0_, R0 = body_tf[b0[0]]
            anchor0 = t0_ + R0 @ lp0
            axis0 = R0 @ _quat_to_mat(lr0) @ np.array([1.0, 0, 0])
        else:
            anchor0 = lp0
            axis0 = _quat_to_mat(lr0) @ np.array([1.0, 0, 0])
        if np.linalg.norm(anchor0 - anchor1) > 1e-4:
            R.err(f"{jname}: joint anchors disagree through body0/body1 by {np.linalg.norm(anchor0 - anchor1):.4g} m")
        if not is_fix and np.linalg.norm(axis0 - axis1) > 1e-4:
            R.err(f"{jname}: joint axes disagree through body0/body1 by {np.linalg.norm(axis0 - axis1):.4g}")
        if is_fix:
            continue
        dof_joint_names.add(jname)
        if p.GetAttribute("physics:axis").Get() != "X":
            R.err(f"{jname}: physics:axis {p.GetAttribute('physics:axis').Get()} != X")
        lo, hi = p.GetAttribute("physics:lowerLimit").Get(), p.GetAttribute("physics:upperLimit").Get()
        if lo is None or hi is None:
            R.err(f"{jname}: missing limits")
        else:
            locked = bool(p.GetAttribute("doorbench:locked").Get()) if p.GetAttribute("doorbench:locked").IsValid() else False
            limited = p.GetAttribute("doorbench:limited").Get() if p.GetAttribute("doorbench:limited").IsValid() else True
            if float(lo) > float(hi):
                R.err(f"{jname}: lower {lo} > upper {hi}")
            elif float(hi) - float(lo) <= 0 and not locked:
                R.err(f"{jname}: zero joint range")
            if limited is False and float(hi) - float(lo) < 100:
                R.warn(f"{jname}: unlimited in the IR but exported with range {lo}..{hi}")
        drive = "angular" if is_rev else "linear"
        st, dm = p.GetAttribute(f"drive:{drive}:physics:stiffness").Get(), p.GetAttribute(f"drive:{drive}:physics:damping").Get()
        if st is None or dm is None:
            R.err(f"{jname}: missing {drive} drive")
        elif float(st) < 0 or float(dm) < 0:
            R.err(f"{jname}: negative drive gains {st} {dm}")
        if p.GetAttribute(f"drive:{drive}:physics:type").Get() != "force":
            R.err(f"{jname}: drive type must be force")
        # per-axis PhysX API: instance name == the drive's ("angular" / "linear"); rotX / transX are D6 tokens that the
        # PhysX USD parser ignores on single-DoF joints (round-1 parity: friction read back 0 on every joint)
        inst = drive
        applied = _applied_schemas(p)
        if f"PhysxJointAxisAPI:{inst}" not in applied:
            R.err(f"{jname}: PhysxJointAxisAPI:{inst} not applied")
        for bad in ("PhysxJointAxisAPI:rotX", "PhysxJointAxisAPI:transX", "PhysxJointAxisAPI:rotY", "PhysxJointAxisAPI:rotZ", "PhysxJointAxisAPI:transY", "PhysxJointAxisAPI:transZ"):
            if bad in applied:
                R.err(f"{jname}: {bad} applied on a single-DoF joint (PhysX reads only the {inst} instance)")
        if "PhysxJointAPI" not in applied:
            R.warn(f"{jname}: PhysxJointAPI not applied")
        fe = p.GetAttribute(f"physxJointAxis:{inst}:staticFrictionEffort").Get()
        fd = p.GetAttribute(f"physxJointAxis:{inst}:dynamicFrictionEffort").Get()
        if fe is None or fd is None or float(fe) < 0 or float(fd) < 0 or float(fd) > float(fe) + 1e-9:
            R.err(f"{jname}: friction efforts static={fe} dynamic={fd}")
        arm = p.GetAttribute(f"physxJointAxis:{inst}:armature").Get()
        if arm is None or float(arm) < 0:
            R.err(f"{jname}: per-axis armature {arm}")
        jf = p.GetAttribute("physxJoint:jointFriction").Get()
        if jf is None or float(jf) < 0:
            R.err(f"{jname}: legacy jointFriction {jf}")
        elif float(jf) > 0:
            R.err(f"{jname}: legacy physxJoint:jointFriction {jf} must be 0 (the Coulomb efforts carry the friction; the coefficient would add load-dependent friction on top)")
        # MJCF position servo folded into the drive: gains / limit must be self-consistent
        if p.GetAttribute("doorbench:servo_in_drive").IsValid() and bool(p.GetAttribute("doorbench:servo_in_drive").Get()):
            conv = 180.0 / math.pi if is_rev else 1.0
            kp = p.GetAttribute("doorbench:servo_kp_si").Get()
            lim = p.GetAttribute("doorbench:servo_force_limit").Get()
            mf = p.GetAttribute(f"drive:{drive}:physics:maxForce").Get()
            if kp is None or lim is None or mf is None or abs(float(mf) - float(lim)) > 1e-6 * max(1.0, float(lim)):
                R.err(f"{jname}: servo in drive but maxForce {mf} != forcerange {lim}")
            if st is not None and kp is not None and float(st) * conv < float(kp) - 1e-6:
                R.err(f"{jname}: servo in drive but drive stiffness {float(st) * conv:.4g} < kp {kp}")
        # mimic joints
        for sch in applied:
            if sch.startswith("PhysxMimicJointAPI:"):
                minst = sch.split(":")[1]
                rel = p.GetRelationship(f"physxMimicJoint:{minst}:referenceJoint")
                tg = [str(t) for t in rel.GetTargets()] if rel else []
                if len(tg) != 1 or not stage.GetPrimAtPath(tg[0]).IsValid():
                    R.err(f"{jname}: mimic reference joint {tg} missing")
                for an in ("gearing", "offset", "referenceJointAxis"):
                    if p.GetAttribute(f"physxMimicJoint:{minst}:{an}").Get() is None:
                        R.err(f"{jname}: mimic attribute {an} missing")
    R.stats["n_joints"] = len(joints)
    R.stats["n_dof_joints"] = len(dof_joint_names)
    if len(world_joints) != 1:
        R.err(f"{len(world_joints)} joints to the world {world_joints} (expected exactly one fixed base joint)")
    # tree connectivity from base
    base_name = [n for n, j in parent_of.items() if j in world_joints]
    if base_name:
        seen, stack = set(), [base_name[0]]
        while stack:
            n = stack.pop()
            if n in seen:
                R.err("cycle in the joint tree")
                break
            seen.add(n)
            stack += children_of.get(n, [])
        missing = sorted(set(bodies) - seen)
        if missing:
            R.err(f"rigid bodies not connected to the base by joints: {[m.split('/')[-1] for m in missing][:10]}")
    for n in bodies:
        if n not in parent_of:
            R.err(f"{n.split('/')[-1]}: rigid body without a joint")
    # ---- collision + materials + references
    n_coll, n_mesh_ref = 0, 0
    ref_cache = {}
    for p in prims:
        # references
        for ref in p.GetMetadata("references").GetAddedOrExplicitItems() if p.HasAuthoredReferences() else []:
            n_mesh_ref += 1
            ap = ref.assetPath
            full = os.path.normpath(os.path.join(os.path.dirname(path), ap))
            if not os.path.exists(full):
                R.err(f"{p.GetPath()}: reference {ap} does not exist")
                continue
            key = (full, str(ref.primPath))
            if key not in ref_cache:
                try:
                    rs = Usd.Stage.Open(full)
                    tp = rs.GetPrimAtPath(ref.primPath) if ref.primPath else rs.GetDefaultPrim()
                    ok = tp.IsValid()
                    if ok:
                        meshes = [c for c in Usd.PrimRange(tp) if c.IsA(UsdGeom.Mesh)]
                        ok = bool(meshes) and all((UsdGeom.Mesh(mm).GetPointsAttr().Get() or []) for mm in meshes)
                        for mm in meshes:
                            pts = np.asarray(UsdGeom.Mesh(mm).GetPointsAttr().Get(), float)
                            if not np.all(np.isfinite(pts)):
                                ok = False
                    ref_cache[key] = ok
                except Exception:
                    ref_cache[key] = False
            if not ref_cache[key]:
                R.err(f"{p.GetPath()}: reference {ap}{ref.primPath} has no valid Mesh")
        if p.HasAPI(UsdPhysics.CollisionAPI):
            n_coll += 1
            if p.HasAuthoredReferences() or p.IsA(UsdGeom.Mesh):
                if not p.HasAPI(UsdPhysics.MeshCollisionAPI):
                    R.err(f"{p.GetPath()}: mesh collider without MeshCollisionAPI")
                elif p.GetAttribute("physics:approximation").Get() not in ("convexHull", "convexDecomposition", "sdf", "boundingCube", "boundingSphere"):
                    R.err(f"{p.GetPath()}: mesh approximation {p.GetAttribute('physics:approximation').Get()}")
            # physics material binding
            mb = UsdShade.MaterialBindingAPI(p)
            rel = p.GetRelationship("material:binding:physics")
            tg = [str(t) for t in rel.GetTargets()] if rel and rel.IsValid() else []
            if not tg:
                R.warn(f"{p.GetPath()}: collider without a physics material")
            else:
                mp = stage.GetPrimAtPath(tg[0])
                if not mp.IsValid() or not mp.HasAPI(UsdPhysics.MaterialAPI):
                    R.err(f"{p.GetPath()}: physics material {tg[0]} invalid")
            if root is not None and str(p.GetPath()).startswith(str(root.GetPath())):
                # collider inside the articulation must belong to a rigid body
                anc, okb = p.GetParent(), False
                while anc and anc.IsValid() and str(anc.GetPath()) != str(root.GetPath()):
                    if anc.HasAPI(UsdPhysics.RigidBodyAPI):
                        okb = True
                        break
                    anc = anc.GetParent()
                if not okb:
                    R.err(f"{p.GetPath()}: collider under the articulation root but not under a rigid body")
        rel = p.GetRelationship("material:binding")
        if rel and rel.IsValid():
            for t in rel.GetTargets():
                if not stage.GetPrimAtPath(t).IsValid():
                    R.err(f"{p.GetPath()}: material {t} missing")
    R.stats["n_colliders"] = n_coll
    R.stats["n_mesh_references"] = n_mesh_ref
    # ---- compare with model.json
    if model_json is not None:
        mj_joints = {b["joint"]["name"]: b["joint"] for b in model_json["bodies"] if b.get("joint")}
        if kind == "full":
            if set(mj_joints) != dof_joint_names:
                R.err(f"joint names differ from model.json: missing {sorted(set(mj_joints) - dof_joint_names)[:5]} extra {sorted(dof_joint_names - set(mj_joints))[:5]}")
            for jn, jt in mj_joints.items():
                p = joints.get(jn)
                if p is None:
                    continue
                rev = jt["type"] == "hinge"
                conv = 180.0 / math.pi if rev else 1.0
                if jt.get("range") is not None:
                    lo, hi = (jt["range"][0] - jt["modeled_at"]) * conv, (jt["range"][1] - jt["modeled_at"]) * conv
                    ulo, uhi = float(p.GetAttribute("physics:lowerLimit").Get()), float(p.GetAttribute("physics:upperLimit").Get())
                    if abs(lo - ulo) > 1e-3 or abs(hi - uhi) > 1e-3:
                        R.err(f"{jn}: limits {ulo:.4f}..{uhi:.4f} != IR {lo:.4f}..{hi:.4f}")
                drive = "angular" if rev else "linear"
                st = float(p.GetAttribute(f"drive:{drive}:physics:stiffness").Get()) * conv
                dm = float(p.GetAttribute(f"drive:{drive}:physics:damping").Get()) * conv
                # MJCF position servo (meta.actuators) folded into the drive: k = kp (+ spring), d = kv + damping,
                # target = ctrl, maxForce = forcerange; only for joints without a spring of their own
                servo = next((a for a in (model_json.get("meta", {}).get("actuators") or []) if a.get("joint") == jn and a.get("kind", "position") == "position"), None)
                in_drive = bool(p.GetAttribute("doorbench:servo_in_drive").Get()) if p.GetAttribute("doorbench:servo_in_drive").IsValid() else False
                if in_drive and servo is None:
                    R.err(f"{jn}: doorbench:servo_in_drive without an actuator in model.json")
                if servo is not None and not in_drive and not jt["stiffness"]:
                    R.err(f"{jn}: spring-less servo joint not folded into the drive")
                if servo is not None and in_drive and jt["stiffness"]:
                    R.err(f"{jn}: servo folded into a drive that also carries a spring (forcerange would clip the spring)")
                want_k = float(jt["stiffness"]) + (float(servo.get("kp", 0.0)) if in_drive else 0.0)
                want_d = float(jt["damping"]) + (float(servo.get("kv", 0.0)) if in_drive else 0.0)
                if abs(st - want_k) > 1e-3 * max(1.0, abs(want_k)):
                    R.err(f"{jn}: drive stiffness {st:.4g} != IR {want_k:.4g} (SI)")
                if abs(dm - want_d) > 1e-3 * max(1.0, abs(want_d)):
                    R.err(f"{jn}: drive damping {dm:.4g} != IR {want_d:.4g} (SI)")
                if jt["stiffness"] or in_drive:
                    tgt = float(p.GetAttribute(f"drive:{drive}:physics:targetPosition").Get()) / conv
                    want = (jt["springref"] - jt["modeled_at"]) if jt["stiffness"] else (float(servo.get("ctrl", 0.0)) - jt["modeled_at"])
                    if abs(tgt - want) > 1e-4 * max(1.0, abs(want)):
                        R.err(f"{jn}: drive target {tgt:.4g} != IR {want:.4g}")
                if in_drive:
                    mf = float(p.GetAttribute(f"drive:{drive}:physics:maxForce").Get())
                    fr = servo.get("forcerange", [-1e6, 1e6])
                    lim = max(abs(float(fr[0])), abs(float(fr[1])))
                    if abs(mf - lim) > 1e-6 * max(1.0, lim):
                        R.err(f"{jn}: servo drive maxForce {mf} != forcerange {lim}")
                inst = drive
                fe = float(p.GetAttribute(f"physxJointAxis:{inst}:staticFrictionEffort").Get())
                if abs(fe - float(jt["frictionloss"])) > 1e-5 * max(1.0, abs(jt["frictionloss"])):
                    R.err(f"{jn}: friction effort {fe} != IR {jt['frictionloss']}")
                for an in (f"physxJointAxis:{inst}:armature", "physxJoint:armature"):
                    arm = p.GetAttribute(an).Get()
                    if arm is None or abs(float(arm) - float(jt.get("armature") or 0.0)) > 1e-5 * max(1.0, abs(float(jt.get("armature") or 0.0))):
                        R.err(f"{jn}: {an} {arm} != IR armature {jt.get('armature')}")
                if jt["stiffness"] > 0 and st <= 0:
                    R.err(f"{jn}: spring in the IR but no drive stiffness")
            # collision geoms of moving + static bodies
            tier = model_json.get("tier", "full")
            want_coll = 0
            for b in model_json["bodies"]:
                for g in b["geoms"]:
                    if g.get("collision") and tier in g.get("tiers", [tier]):
                        want_coll += 1
            if n_coll < want_coll:
                R.err(f"{n_coll} colliders in USD < {want_coll} collision geoms in model.json")
            R.stats["n_collision_geoms_ir"] = want_coll
        else:
            if dof_joint_names != set(RL_JOINTS):
                R.err(f"RL joints {sorted(dof_joint_names)} != canonical {sorted(RL_JOINTS)}")
            links = {n.split("/")[-1] for n in bodies}
            if links != set(RL_LINKS):
                R.err(f"RL links {sorted(links)} != canonical {sorted(RL_LINKS)}")
            rl_meta = None
            for k, v in json_attrs.items():
                if k.endswith("doorbench:rl"):
                    rl_meta = v
            if rl_meta is None:
                R.err("doorbench:rl meta missing")
            else:
                for k in ("slots", "door_joint", "sites", "open_threshold", "clear_threshold", "closed_threshold", "joints"):
                    if k not in rl_meta:
                        R.err(f"doorbench:rl missing {k}")
                if rl_meta.get("slots", {}).get("door") not in ("hinge", "slide"):
                    R.err("doorbench:rl door slot invalid")
                dj = rl_meta.get("door_joint")
                if dj in joints:
                    lo, hi = float(joints[dj].GetAttribute("physics:lowerLimit").Get()), float(joints[dj].GetAttribute("physics:upperLimit").Get())
                    conv = 180.0 / math.pi if dj == "door_hinge" else 1.0
                    src = mj_joints.get(rl_meta.get("primary_joint"), {})
                    ir_range = ((src["range"][1] - src["range"][0]) * conv) if src.get("range") is not None else None
                    # the RL door joint must carry the IR's full travel (locked / jammed doors legitimately have only backlash)
                    if ir_range is not None and (hi - lo) < 0.99 * ir_range - 1e-6:
                        R.err(f"{dj} range {lo:.4f}..{hi:.4f} smaller than the IR range {ir_range:.4f}")
                    if ir_range is not None and ir_range < (1.0 if dj == "door_hinge" else 0.05):
                        R.warn(f"{dj}: door cannot open (IR range {ir_range:.4f}); lock {rl_meta.get('lock', {}).get('model')} engaged={rl_meta.get('lock', {}).get('engaged')}")
                if not rl_meta.get("sites", {}).get("grip"):
                    R.warn("no grip/push/edge site for the handle observation")
                # canonical joints described in the IR must map to model.json joints
                for cj, info in rl_meta.get("joints", {}).items():
                    if info.get("active") and info.get("source") not in mj_joints:
                        R.err(f"{cj}: source joint {info.get('source')} not in model.json")
                    if info.get("active") and cj in joints:
                        src = mj_joints.get(info.get("source"), {})
                        conv = 180.0 / math.pi if cj.endswith("hinge") else 1.0
                        fe = joints[cj].GetAttribute(f"physxJointAxis:{'angular' if cj.endswith('hinge') else 'linear'}:staticFrictionEffort").Get()
                        if fe is None or abs(float(fe) - float(src.get("frictionloss") or 0.0)) > 1e-5 * max(1.0, abs(float(src.get("frictionloss") or 0.0))):
                            R.err(f"{cj}: friction effort {fe} != IR {src.get('frictionloss')} ({info.get('source')})")
                        in_drive = joints[cj].GetAttribute("doorbench:servo_in_drive").Get() if joints[cj].GetAttribute("doorbench:servo_in_drive").IsValid() else False
                        if bool(in_drive) != bool(info.get("servo")):
                            R.err(f"{cj}: servo_in_drive {in_drive} but rl meta servo {info.get('servo')}")
                        if info.get("servo"):
                            st = float(joints[cj].GetAttribute(f"drive:{'angular' if cj.endswith('hinge') else 'linear'}:physics:stiffness").Get()) * conv
                            if abs(st - float(info["servo"]["kp"]) - float(src.get("stiffness") or 0.0)) > 1e-3 * max(1.0, float(info["servo"]["kp"])):
                                R.err(f"{cj}: servo drive stiffness {st:.4g} != kp {info['servo']['kp']}")
                # actuators: in_drive flags must match the joint prims
                slot_of = {info.get("source"): cj for cj, info in rl_meta.get("joints", {}).items() if info.get("active")}
                for a in rl_meta.get("actuators") or []:
                    slot = slot_of.get(a.get("joint"))
                    in_drive = bool(joints[slot].GetAttribute("doorbench:servo_in_drive").Get()) if (slot in joints and joints[slot].GetAttribute("doorbench:servo_in_drive").IsValid()) else False
                    if bool(a.get("in_drive")) != in_drive or (a.get("in_drive") and a.get("slot") != slot):
                        R.err(f"actuator {a.get('name')}: in_drive {a.get('in_drive')} / slot {a.get('slot')} inconsistent with the {slot} prim")
                # rising / helical hinge: the locked riser is replaced by a gravity closing torque on the door joint
                rc = rl_meta.get("rise_coupling")
                if rc is not None:
                    src = mj_joints.get(rc.get("rise_joint"))
                    if src is None or src.get("type") != "slide":
                        R.err(f"rise_coupling: rise joint {rc.get('rise_joint')} not a slide joint of model.json")
                    if not (math.isfinite(float(rc.get("gravity_torque_Nm", float('nan')))) and float(rc.get("carried_mass_kg", 0.0)) > 0 and float(rc.get("coeff_m_per_rad", 0.0)) > 0):
                        R.err(f"rise_coupling: implausible values {rc}")
                    elif abs(float(rc["gravity_torque_Nm"]) + float(rc["carried_mass_kg"]) * 9.81 * float(rc["lift_m_per_rad"])) > 1e-6:
                        R.err("rise_coupling: gravity torque != -m g dz/dq")
                    if rc.get("hinge_joint") != rl_meta.get("primary_joint") or rl_meta.get("door_joint") != "door_hinge":
                        R.err("rise_coupling: not on the primary hinge")
            R.stats["slots"] = rl_meta.get("slots") if rl_meta else None
    R.stats["time_s"] = round(time.time() - t0, 3)
    return {"ok": not R.errors, "errors": R.errors, "warnings": R.warnings, "stats": R.stats}


def validate_door(door_dir: str) -> dict:
    out = {"id": os.path.basename(door_dir)}
    mj = sp = None
    try:
        with open(os.path.join(door_dir, "model.json")) as f:
            mj = json.load(f)
        with open(os.path.join(door_dir, "spec.json")) as f:
            sp = json.load(f)
    except Exception as e:
        out["error"] = f"cannot read model/spec json: {e}"
    for kind, fn in (("full", "door.usda"), ("rl", "door_rl.usda")):
        p = os.path.join(door_dir, fn)
        if not os.path.exists(p):
            out[kind] = {"ok": False, "errors": [f"{fn} missing"], "warnings": [], "stats": {}}
            continue
        try:
            out[kind] = validate_stage(p, kind, mj, sp)
        except Exception as e:  # validator bug or badly broken file
            out[kind] = {"ok": False, "errors": [f"validator exception: {type(e).__name__}: {e}"], "warnings": [], "stats": {}}
    out["ok"] = all(out[k]["ok"] for k in ("full", "rl") if k in out) and "error" not in out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(ROOT, "assets"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    doors_root = os.path.join(a.assets, "doors")
    ids = sorted(d for d in os.listdir(doors_root) if os.path.isdir(os.path.join(doors_root, d)))
    if a.ids:
        want = set(a.ids.split(","))
        ids = [i for i in ids if i in want]
    if a.limit:
        ids = ids[: a.limit]
    dirs = [os.path.join(doors_root, i) for i in ids]
    t0 = time.time()
    if a.workers > 1 and len(dirs) > 8:
        with Pool(a.workers) as pool:
            rows = pool.map(validate_door, dirs, chunksize=4)
    else:
        rows = [validate_door(d) for d in dirs]
    n_ok = sum(1 for r in rows if r["ok"])
    err_hist = {}
    for r in rows:
        for kind in ("full", "rl"):
            for e in r.get(kind, {}).get("errors", []):
                key = f"{kind}: " + e.split(":")[-1].strip()[:70] if ":" in e else f"{kind}: {e[:70]}"
                err_hist.setdefault(key, []).append(r["id"])
    stats = {}
    for kind in ("full", "rl"):
        s = [r[kind]["stats"] for r in rows if kind in r and r[kind]["stats"]]
        stats[kind] = {
            "n_files": len(s), "n_ok": sum(1 for r in rows if kind in r and r[kind]["ok"]),
            "joints_total": int(sum(x.get("n_dof_joints", 0) for x in s)), "rigid_bodies_total": int(sum(x.get("n_rigid_bodies", 0) for x in s)),
            "colliders_total": int(sum(x.get("n_colliders", 0) for x in s)), "mesh_references_total": int(sum(x.get("n_mesh_references", 0) for x in s)),
            "warnings_total": int(sum(len(r[kind]["warnings"]) for r in rows if kind in r)),
        }
    if rows and "rl" in rows[0]:
        slot_hist = {}
        for r in rows:
            sl = r.get("rl", {}).get("stats", {}).get("slots")
            if sl:
                key = f"door={sl['door']} operator={sl['operator']} latch={sl['latch']} secondary={sl['secondary']}"
                slot_hist[key] = slot_hist.get(key, 0) + 1
        stats["rl"]["slot_histogram"] = dict(sorted(slot_hist.items(), key=lambda kv: -kv[1]))
    summary = {"n_doors": len(rows), "n_ok": n_ok, "n_failed": len(rows) - n_ok, "time_s": round(time.time() - t0, 1), "stats": stats,
               "error_histogram": {k: {"count": len(v), "examples": v[:5]} for k, v in sorted(err_hist.items(), key=lambda kv: -len(kv[1]))}}
    print(f"validated {len(rows)} doors in {summary['time_s']} s: {n_ok} ok, {len(rows) - n_ok} failed")
    for kind in ("full", "rl"):
        st = stats[kind]
        print(f"  {kind:4s}: {st['n_ok']}/{st['n_files']} ok, {st['joints_total']} joints, {st['rigid_bodies_total']} rigid bodies, {st['colliders_total']} colliders, {st['mesh_references_total']} mesh refs, {st['warnings_total']} warnings")
    if "slot_histogram" in stats.get("rl", {}):
        for k, v in list(stats["rl"]["slot_histogram"].items())[:12]:
            print(f"     {v:4d}  {k}")
    for k, v in list(summary["error_histogram"].items())[:25]:
        print(f"  ERR x{v['count']:4d}: {k}   e.g. {v['examples'][:3]}")
    if not a.quiet:
        for r in rows:
            if not r["ok"]:
                for kind in ("full", "rl"):
                    for e in r.get(kind, {}).get("errors", [])[:3]:
                        print(f"    {r['id']} [{kind}] {e}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "doors": rows}, f, indent=1)
        print(f"wrote {a.out}")
    sys.exit(0 if n_ok == len(rows) else 1)


if __name__ == "__main__":
    main()

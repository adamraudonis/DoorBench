"""MuJoCo -> PhysX parameter mapping of the USD exporter (doorbench/export/usd.py), checked on freshly exported
representative doors (no generated assets/ needed; ~10 s):

  * Coulomb friction: PhysxJointAxisAPI on the ``angular`` (revolute) / ``linear`` (prismatic) instance - the only
    instance names the PhysX USD parser reads on single-DoF joints - static == dynamic effort == MJCF frictionloss,
    legacy load-dependent ``physxJoint:jointFriction`` authored 0 (no double friction)
  * drives: stiffness / damping / target in UsdPhysics units (per degree on revolute joints), armature on both APIs
  * rigid bodies: ``physxRigidBody:maxAngularVelocity`` = 100 rad/s in deg/s (a 100 deg/s cap clamped every leaf at
    1.75 rad/s in parity round 1); the Isaac Lab cfg uses the same value
  * automatic doors: the MJCF position servo of a spring-less joint IS the PhysX drive (kp / kv / forcerange); a servo
    on a joint with its own spring stays a feed-forward emulation
  * rising / helical hinges: ``doorbench:rl["rise_coupling"]`` carries the gravity closing torque -m g dz/dq that
    replaces the locked riser of the canonical file
  * the static validator accepts the new files (and the Isaac Lab cfg text carries the deg/s cap)

Doors: bifold, pet_door, sliding_single, cold_storage (rising hinge), turnstile_fullheight, revolving,
automatic_sliding, automatic_swing, stall (gravity hinge).
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdPhysics  # noqa: E402

from doorbench.spec import generate_all  # noqa: E402
from doorbench.build import export_door  # noqa: E402
from doorbench.export import usd as XS  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "isaaclab"))
from validate_usd_static import validate_door  # noqa: E402

FAMILIES = ("bifold", "pet_door", "sliding_single", "cold_storage", "turnstile_fullheight", "revolving", "automatic_sliding", "automatic_swing", "stall")
DEG = 180.0 / math.pi


@pytest.fixture(scope="module")
def doors(tmp_path_factory):
    """{family: (spec, door_dir, model_json, full stage, rl stage)} for one door per family."""
    out = tmp_path_factory.mktemp("usd_mapping")
    specs = generate_all()
    picked = {}
    for s in specs:
        fam = s["family"]
        if fam in FAMILIES and fam not in picked:
            picked[fam] = s
    # a rising (cam-lift) cold-storage door and a bi-parting automatic slider make the coupling checks meaningful
    picked["cold_storage"] = next(s for s in specs if s["family"] == "cold_storage" and s["hinge"].get("axis_tilt_deg"))
    assert set(picked) == set(FAMILIES)
    res = {}
    for fam, s in picked.items():
        export_door(s, str(out / "doors"), str(out / "hardware"), formats=("usd", "json"))
        dd = str(out / "doors" / s["id"])
        with open(os.path.join(dd, "model.json")) as f:
            mj = json.load(f)
        with open(os.path.join(dd, "spec.json")) as f:
            spec = json.load(f)      # the exported spec carries the derived physics block
        res[fam] = (spec, dd, mj, Usd.Stage.Open(os.path.join(dd, "door.usda")), Usd.Stage.Open(os.path.join(dd, "door_rl.usda")))
    return res


def _dof_joints(stage):
    return {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)}


def _schemas(prim):
    md = prim.GetMetadata("apiSchemas")
    return set(md.GetAddedOrExplicitItems()) if md is not None else set()


def _ir_joints(mj):
    return {b["joint"]["name"]: b["joint"] for b in mj["bodies"] if b.get("joint")}


def _rl_meta(stage):
    return json.loads(stage.GetDefaultPrim().GetAttribute("doorbench:rl").Get())


def _meta(stage):
    return json.loads(stage.GetDefaultPrim().GetAttribute("doorbench:meta").Get())


# ---------------------------------------------------------------------------
# Coulomb friction / armature: per-axis API instance names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fam", FAMILIES)
def test_joint_axis_api_instance_and_friction(doors, fam):
    _, _, mj, full, rl = doors[fam]
    ir = _ir_joints(mj)
    for kind, stage in (("full", full), ("rl", rl)):
        joints = _dof_joints(stage)
        assert joints
        for name, p in joints.items():
            rev = p.IsA(UsdPhysics.RevoluteJoint)
            inst = "angular" if rev else "linear"
            sch = _schemas(p)
            assert f"PhysxJointAxisAPI:{inst}" in sch, (fam, kind, name, sch)
            assert not any(s.startswith("PhysxJointAxisAPI:") and s.split(":")[1] in ("rotX", "transX", "rotY", "rotZ", "transY", "transZ") for s in sch), (fam, kind, name, sch)
            assert "PhysxJointAPI" in sch
            # drive instance == axis-API instance
            assert p.GetAttribute(f"drive:{inst}:physics:stiffness").HasAuthoredValue(), (fam, kind, name)
            fe = p.GetAttribute(f"physxJointAxis:{inst}:staticFrictionEffort").Get()
            fd = p.GetAttribute(f"physxJointAxis:{inst}:dynamicFrictionEffort").Get()
            assert fe is not None and fd is not None and fe == fd and fe >= 0
            assert p.GetAttribute(f"physxJointAxis:{inst}:viscousFrictionCoefficient").Get() == 0.0
            # the legacy coefficient (friction = coeff * |joint force|) must not add load-dependent friction on top
            assert p.GetAttribute("physxJoint:jointFriction").Get() == 0.0, (fam, kind, name)
            assert p.GetAttribute("doorbench:friction_effort").Get() == pytest.approx(fe)
            arm_axis = p.GetAttribute(f"physxJointAxis:{inst}:armature").Get()
            arm_legacy = p.GetAttribute("physxJoint:armature").Get()
            assert arm_axis is not None and arm_axis == arm_legacy and arm_axis >= 0
            src = p.GetAttribute("doorbench:source_joint").Get() if p.GetAttribute("doorbench:source_joint").IsValid() else None
            if src in ir:
                j = ir[src]
                assert fe == pytest.approx(j["frictionloss"], rel=1e-5, abs=1e-6), (fam, kind, name)
                assert arm_axis == pytest.approx(j["armature"], rel=1e-5, abs=1e-7), (fam, kind, name)
                if j["frictionloss"] > 0:
                    assert p.GetAttribute("doorbench:legacy_friction_coeff").Get() > 0
        if kind == "full":
            assert set(joints) == set(ir)


def test_friction_values_are_the_physics_model(doors):
    """Spot values: hinge friction torque of the physics block, roller friction of the slider, rotor bearing torque."""
    for fam in ("cold_storage", "turnstile_fullheight", "revolving", "pet_door", "bifold", "sliding_single"):
        spec, _, mj, full, _ = doors[fam]
        ir = _ir_joints(mj)
        pj = mj["meta"]["primary_joint"]
        p = _dof_joints(full)[pj]
        inst = "angular" if ir[pj]["type"] == "hinge" else "linear"
        fe = p.GetAttribute(f"physxJointAxis:{inst}:staticFrictionEffort").Get()
        assert fe == pytest.approx(ir[pj]["frictionloss"], rel=1e-5)
        phys = spec["physics"]
        if fam in ("cold_storage", "pet_door", "turnstile_fullheight", "revolving"):
            # hinge / rotor: MJCF frictionloss = coulomb_torque + 0.5 * stick_torque (hinged.py hinge_joint) - exact for
            # hinged leaves; rotors / flaps add family terms, so only the order of magnitude is pinned
            assert fe >= 0.5 * phys["hinge"]["coulomb_torque_Nm"] - 1e-9
        if fam == "sliding_single":
            assert fe >= 0.5 * phys["roller"]["coulomb_force_N"] - 1e-9    # N (prismatic): roller Coulomb force
        if fam == "pet_door":
            assert fe < 1.0    # a flap, not a door: sub-N*m friction


# ---------------------------------------------------------------------------
# drives: units, spring targets, damping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fam", FAMILIES)
def test_drive_units_and_targets(doors, fam):
    _, _, mj, full, rl = doors[fam]
    ir = _ir_joints(mj)
    servo_of = {a["joint"]: a for a in mj["meta"].get("actuators", [])}
    for kind, stage in (("full", full), ("rl", rl)):
        for name, p in _dof_joints(stage).items():
            src = p.GetAttribute("doorbench:source_joint").Get() if p.GetAttribute("doorbench:source_joint").IsValid() else None
            if src not in ir:
                continue      # locked RL slots
            j = ir[src]
            rev = j["type"] == "hinge"
            inst = "angular" if rev else "linear"
            conv = DEG if rev else 1.0
            servo = servo_of.get(src) if not j["stiffness"] else None
            k = p.GetAttribute(f"drive:{inst}:physics:stiffness").Get() * conv
            d = p.GetAttribute(f"drive:{inst}:physics:damping").Get() * conv
            tgt = p.GetAttribute(f"drive:{inst}:physics:targetPosition").Get() / conv
            assert k == pytest.approx(j["stiffness"] + (servo["kp"] if servo else 0.0), rel=1e-5, abs=1e-7), (fam, kind, name)
            assert d == pytest.approx(j["damping"] + (servo["kv"] if servo else 0.0), rel=1e-5, abs=1e-7), (fam, kind, name)
            want_t = (j["springref"] - j["modeled_at"]) if j["stiffness"] else ((servo.get("ctrl", 0.0) - j["modeled_at"]) if servo else 0.0)
            assert tgt == pytest.approx(want_t, abs=1e-6), (fam, kind, name)
            assert p.GetAttribute(f"drive:{inst}:physics:type").Get() == "force"
            assert p.GetAttribute("doorbench:target_si").Get() == pytest.approx(want_t, abs=1e-6)
            assert p.GetAttribute("doorbench:stiffness_si").Get() == pytest.approx(k, rel=1e-5, abs=1e-6)
            # limits in degrees / metres, shifted by the MJCF ref
            if j["range"] is not None:
                lo, hi = p.GetAttribute("physics:lowerLimit").Get(), p.GetAttribute("physics:upperLimit").Get()
                assert lo == pytest.approx((j["range"][0] - j["modeled_at"]) * conv, abs=1e-4)
                assert hi == pytest.approx((j["range"][1] - j["modeled_at"]) * conv, abs=1e-4)


def test_cold_storage_handle_spring_in_degrees(doors):
    """2.5 N*m/rad return spring with -0.48 rad preload reference -> 0.04363 N*m/deg, target -27.5 deg."""
    _, _, mj, full, rl = doors["cold_storage"]
    ir = _ir_joints(mj)
    op = mj["meta"]["operator_joint"]
    j = ir[op]
    assert j["type"] == "hinge" and j["stiffness"] > 0
    for stage, name in ((full, op), (rl, "operator_hinge")):
        p = _dof_joints(stage)[name]
        assert p.GetAttribute("drive:angular:physics:stiffness").Get() == pytest.approx(j["stiffness"] / DEG, rel=1e-5)
        assert p.GetAttribute("drive:angular:physics:targetPosition").Get() == pytest.approx((j["springref"] - j["modeled_at"]) * DEG, rel=1e-5)
        assert p.GetAttribute("drive:angular:physics:damping").Get() == pytest.approx(j["damping"] / DEG, rel=1e-5)


# ---------------------------------------------------------------------------
# rigid bodies: velocity cap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fam", FAMILIES)
def test_max_angular_velocity_authored(doors, fam):
    _, _, _, full, rl = doors[fam]
    assert XS.MAX_ANGULAR_VELOCITY_DEG_S == pytest.approx(100.0 * DEG)       # 100 rad/s, the PhysX default
    for stage in (full, rl):
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert bodies
        for p in bodies:
            v = p.GetAttribute("physxRigidBody:maxAngularVelocity").Get()
            assert v is not None and v == pytest.approx(5729.578, abs=0.01), p.GetPath()
    assert _meta(full)["max_angular_velocity_deg_s"] == pytest.approx(100.0 * DEG)
    assert _rl_meta(rl)["max_angular_velocity_deg_s"] == pytest.approx(100.0 * DEG)


def test_isaaclab_cfg_velocity_cap_is_in_deg_s():
    """isaaclab/doorbench_isaaclab/assets.py (not importable here: needs Isaac Lab) must not cap the links below 1000 deg/s."""
    with open(os.path.join(ROOT, "isaaclab", "doorbench_isaaclab", "assets.py")) as f:
        src = f.read()
    tree = ast.parse(src)
    consts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    m = re.search(r"max_angular_velocity\s*=\s*([A-Za-z_][A-Za-z_0-9]*|[0-9.eE+-]+)", src)
    assert m, "DOOR_RIGID_PROPS.max_angular_velocity missing"
    val = consts.get(m.group(1)) if not re.match(r"^[0-9.eE+-]+$", m.group(1)) else float(m.group(1))
    assert val is not None and val >= 1000.0, f"max_angular_velocity {m.group(1)} = {val} deg/s clamps door leaves (round-1 bug: 100 deg/s = 1.75 rad/s)"
    assert val == pytest.approx(XS.MAX_ANGULAR_VELOCITY_DEG_S, rel=1e-6)


# ---------------------------------------------------------------------------
# automatic doors: position servo folded into the drive
# ---------------------------------------------------------------------------
def test_automatic_sliding_servo_is_the_drive(doors):
    _, _, mj, full, rl = doors["automatic_sliding"]
    ir = _ir_joints(mj)
    acts = {a["joint"]: a for a in mj["meta"]["actuators"]}
    assert acts, "automatic sliders carry MJCF position actuators"
    fj = _dof_joints(full)
    for jn, a in acts.items():
        p = fj[jn]
        assert bool(p.GetAttribute("doorbench:servo_in_drive").Get()) is True
        assert p.GetAttribute("drive:linear:physics:stiffness").Get() == pytest.approx(a["kp"])
        assert p.GetAttribute("drive:linear:physics:damping").Get() == pytest.approx(a["kv"] + ir[jn]["damping"])
        assert p.GetAttribute("drive:linear:physics:maxForce").Get() == pytest.approx(max(abs(a["forcerange"][0]), abs(a["forcerange"][1])))
        assert p.GetAttribute("drive:linear:physics:targetPosition").Get() == pytest.approx(0.0 - ir[jn]["modeled_at"])
        assert p.GetAttribute("doorbench:servo_kp_si").Get() == pytest.approx(a["kp"])
        assert p.GetAttribute("doorbench:servo_kv_si").Get() == pytest.approx(a["kv"])
        assert p.GetAttribute("doorbench:servo_force_limit").Get() == pytest.approx(150.0)
        assert ir[jn]["stiffness"] == 0.0
    meta = _meta(full)
    assert set(meta["servo_in_drive"]) == set(acts)
    assert all(a["in_drive"] for a in meta["actuators"])
    # other drives keep the unlimited default
    for jn, p in fj.items():
        if jn not in acts:
            assert p.GetAttribute(f"drive:{'angular' if p.IsA(UsdPhysics.RevoluteJoint) else 'linear'}:physics:maxForce").Get() == pytest.approx(1e6)
    # canonical file: door_slide (and leaf2_slide for the second leaf) carry the servo, meta says so
    rmeta = _rl_meta(rl)
    rj = _dof_joints(rl)
    slots = {a["joint"]: a["slot"] for a in rmeta["actuators"]}
    assert all(a["in_drive"] for a in rmeta["actuators"]) and slots[mj["meta"]["primary_joint"]] == "door_slide"
    for jn, slot in slots.items():
        p = rj[slot]
        assert bool(p.GetAttribute("doorbench:servo_in_drive").Get()) is True
        assert p.GetAttribute("drive:linear:physics:stiffness").Get() == pytest.approx(acts[jn]["kp"])
        assert p.GetAttribute("drive:linear:physics:maxForce").Get() == pytest.approx(150.0)
        assert rmeta["joints"][slot]["servo"]["kp"] == pytest.approx(acts[jn]["kp"])
        assert rmeta["joints"][slot]["drive"]["max_force"] == pytest.approx(150.0)
    for slot, p in rj.items():
        if slot not in slots.values():
            assert not bool(p.GetAttribute("doorbench:servo_in_drive").Get())


def test_automatic_swing_servo_stays_feed_forward(doors):
    """A servo on a joint with its own closer spring is not folded (one drive per axis; forcerange would clip the spring)."""
    _, _, mj, full, rl = doors["automatic_swing"]
    ir = _ir_joints(mj)
    pj = mj["meta"]["primary_joint"]
    acts = {a["joint"]: a for a in mj["meta"]["actuators"]}
    assert pj in acts and ir[pj]["stiffness"] > 0
    p = _dof_joints(full)[pj]
    assert bool(p.GetAttribute("doorbench:servo_in_drive").Get()) is False
    assert p.GetAttribute("drive:angular:physics:stiffness").Get() * DEG == pytest.approx(ir[pj]["stiffness"], rel=1e-5)
    assert p.GetAttribute("drive:angular:physics:maxForce").Get() == pytest.approx(1e6)
    assert _meta(full)["servo_in_drive"] == {}
    assert all(not a["in_drive"] for a in _meta(full)["actuators"])
    rmeta = _rl_meta(rl)
    assert all(not a["in_drive"] and a["slot"] is None for a in rmeta["actuators"])
    assert "servo" not in rmeta["joints"]["door_hinge"]


def test_servo_drive_params_rule():
    from doorbench.ir import Joint
    servo = {"name": "d", "kp": 400.0, "kv": 60.0, "forcerange": (-150, 150), "ctrlrange": (0, 1)}
    slide = Joint("s", "slide", (1, 0, 0), (0, 0, 0), (0.0, 1.0), damping=7.0, modeled_at=0.02)
    r = XS.servo_drive_params(slide, servo)
    assert r["stiffness"] == 400.0 and r["damping"] == 67.0 and r["force_limit"] == 150.0 and r["target"] == pytest.approx(-0.02)
    sprung = Joint("h", "hinge", (0, 0, 1), (0, 0, 0), (0.0, 1.5), stiffness=15.0, springref=-1.8)
    assert XS.servo_drive_params(sprung, servo) is None
    assert XS.servo_drive_params(slide, None) is None
    assert XS.servo_for_joint([servo | {"joint": "s"}], "s") is not None and XS.servo_for_joint([servo | {"joint": "s"}], "x") is None


# ---------------------------------------------------------------------------
# rising / helical hinge: gravity closing torque for the canonical file
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fam", ("cold_storage", "stall"))
def test_rise_coupling_gravity_torque(doors, fam):
    spec, _, mj, full, rl = doors[fam]
    ir = _ir_joints(mj)
    pj = mj["meta"]["primary_joint"]
    eq = next(e for e in mj["equalities"] if e["kind"] == "joint" and e["b"] == pj and ir[e["a"]]["type"] == "slide")
    c1 = eq["polycoeff"][1]
    rc = _rl_meta(rl)["rise_coupling"]
    assert rc is not None and rc["rise_joint"] == eq["a"] and rc["hinge_joint"] == pj
    assert rc["coeff_m_per_rad"] == pytest.approx(c1) and rc["lift_m_per_rad"] == pytest.approx(c1)   # vertical riser
    assert rc["gravity_torque_Nm"] == pytest.approx(-rc["carried_mass_kg"] * 9.81 * c1)
    assert rc["gravity_torque_Nm"] < 0                                   # opening lifts the leaf: closing torque
    m_spec = spec["physics"]["mass"]["total_kg"]
    assert rc["carried_mass_kg"] == pytest.approx(m_spec, rel=0.15)      # the riser carries the whole leaf + hardware
    assert _meta(full)["rise_coupling"] == rc
    assert any("gravity closing torque" in n for n in _rl_meta(rl)["notes"])
    # the riser is not a canonical slot: locked at initial in door_rl.usda, so the torque is the only way to keep the physics
    assert eq["a"] not in {info.get("source") for info in _rl_meta(rl)["joints"].values() if info.get("active")}
    # magnitude sanity: 12 mm / 90 deg on a 45 kg cold-room door ~ 3.4 N*m; 10 mm / 90 deg on a 15-20 kg stall door ~ 1 N*m
    assert 0.3 < abs(rc["gravity_torque_Nm"]) < 15.0


@pytest.mark.parametrize("fam", ("bifold", "pet_door", "sliding_single", "turnstile_fullheight", "revolving", "automatic_sliding"))
def test_no_rise_coupling_on_ordinary_doors(doors, fam):
    _, _, _, full, rl = doors[fam]
    assert _rl_meta(rl)["rise_coupling"] is None and _meta(full)["rise_coupling"] is None


# ---------------------------------------------------------------------------
# the static validator accepts the new files
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fam", FAMILIES)
def test_static_validator_accepts_export(doors, fam):
    _, dd, _, _, _ = doors[fam]
    r = validate_door(dd)
    assert r["full"]["ok"], r["full"]["errors"]
    assert r["rl"]["ok"], r["rl"]["errors"]


def test_static_validator_rejects_regressions(doors, tmp_path):
    """The checks that would have caught round 1: rotX instance, legacy coefficient, 100 deg/s cap."""
    from pxr import Sdf
    _, dd, _, _, _ = doors["sliding_single"]
    import shutil
    bad = tmp_path / os.path.basename(dd)
    shutil.copytree(dd, bad)
    st = Usd.Stage.Open(str(bad / "door.usda"))
    p = next(iter(_dof_joints(st).values()))
    inst = "angular" if p.IsA(UsdPhysics.RevoluteJoint) else "linear"
    p.GetAttribute("physxJoint:jointFriction").Set(0.5)
    p.AddAppliedSchema(f"PhysxJointAxisAPI:{'rotX' if inst == 'angular' else 'transX'}")
    body = next(q for q in st.Traverse() if q.HasAPI(UsdPhysics.RigidBodyAPI))
    body.GetAttribute("physxRigidBody:maxAngularVelocity").Set(100.0)
    st.GetRootLayer().Save()
    r = validate_door(str(bad))
    errs = " | ".join(r["full"]["errors"])
    assert not r["full"]["ok"]
    assert "jointFriction" in errs and ("rotX" in errs or "transX" in errs)
    assert "maxAngularVelocity" in errs

"""MuJoCo import + physics smoke tests over the generated dataset (task board G4).

Doors under test: one representative of every family in `assets/manifest.json` (the first signed-off door of each
family, manifest order) plus 20 seeded random doors.  For each door:

  * door.xml / door_simple.xml / door_minimal.xml / scene.xml load with `mujoco.MjModel.from_xml_path`
  * door.urdf loads with MuJoCo's URDF importer
  * 500 steps of free dynamics in every tier: no MuJoCo warnings, finite state
  * `doorbench.qa.run_qa`: a latched / locked door holds against a strong push, driving the operator (and any
    thumbturn / bolts / dogs) opens it when the lock has a robot-side release, locked doors stay shut, spring latches
    re-extend and re-latch, closers return
  * `DoorEnv` resets, steps 200 times with a programmatic hand torque and returns labels

plus the kinematic clearance gate (`doorbench.clearance.run_clearance`) on 10 doors.

Run:  pytest -q tests/test_mujoco_import.py        (~1 min; skipped when assets/ has not been generated)
Set DOORBENCH_ASSETS to test another dataset directory.
"""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from doorbench.qa import run_qa, door_flags  # noqa: E402
from doorbench.clearance import run_clearance  # noqa: E402
from doorbench.benchmark import DoorEnv, EpisodeLabels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.environ.get("DOORBENCH_ASSETS", os.path.join(ROOT, "assets"))
MANIFEST = os.path.join(ASSETS, "manifest.json")
N_RANDOM = 20
SEED = 20260904
TIERS = {"full": "door.xml", "simple": "door_simple.xml", "minimal": "door_minimal.xml"}


def _manifest():
    if not os.path.exists(MANIFEST):
        return None
    with open(MANIFEST) as f:
        return json.load(f)


def _select_doors():
    """One representative per family (first signed-off door in manifest order) + N_RANDOM seeded extra doors."""
    man = _manifest()
    if man is None:
        return [], []
    reps = {}
    for d in man["doors"]:
        if d["family"] not in reps and d.get("signed_off", True):
            reps[d["family"]] = d["id"]
    rest = [d["id"] for d in man["doors"] if d["id"] not in reps.values()]
    extra = random.Random(SEED).sample(rest, min(N_RANDOM, len(rest)))
    return list(reps.values()), extra


FAMILY_REPS, RANDOM_DOORS = _select_doors()
DOORS = FAMILY_REPS + RANDOM_DOORS
pytestmark = pytest.mark.skipif(not DOORS, reason=f"generated dataset not found at {ASSETS}")


def _door_dir(door_id):
    return os.path.join(ASSETS, "doors", door_id)


def _load_json(door_id, name):
    with open(os.path.join(_door_dir(door_id), name)) as f:
        return json.load(f)


def _warnings(d):
    return [mujoco.mjtWarning(i).name for i in range(mujoco.mjtWarning.mjNWARNING) if d.warning[i].number > 0]


@pytest.fixture(autouse=True)
def _no_global_passive_callback():
    """DoorEnv installs a process-global passive-force callback; make sure no stale one leaks between tests."""
    mujoco.set_mjcb_passive(None)
    yield
    mujoco.set_mjcb_passive(None)


def test_selection_covers_every_family():
    man = _manifest()
    assert set(man["families"]) == {man_door["family"] for man_door in man["doors"] if man_door["id"] in FAMILY_REPS}
    assert len(FAMILY_REPS) == len(man["families"]) == 30
    assert len(set(DOORS)) == len(DOORS) == 30 + N_RANDOM


@pytest.mark.parametrize("door_id", DOORS)
def test_mjcf_tiers_and_scene_load(door_id):
    meta = _load_json(door_id, "model.json")["meta"]
    for tier, xml in list(TIERS.items()) + [("scene", "scene.xml")]:
        m = mujoco.MjModel.from_xml_path(os.path.join(_door_dir(door_id), xml))
        assert m.nbody >= 2 and m.njnt >= 1, (door_id, tier)
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["primary_joint"]) >= 0, (door_id, tier, "primary joint missing")
        if tier in ("full", "scene"):
            for cam in ("robot_view", "iso"):
                assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0, (door_id, tier, cam)


@pytest.mark.parametrize("door_id", DOORS)
def test_urdf_loads_in_mujoco(door_id):
    meta = _load_json(door_id, "model.json")["meta"]
    m = mujoco.MjModel.from_xml_path(os.path.join(_door_dir(door_id), "door.urdf"))
    assert m.nbody >= 2 and m.njnt >= 1, door_id
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["primary_joint"]) >= 0, (door_id, "primary joint missing in URDF")


@pytest.mark.parametrize("door_id", DOORS)
def test_free_dynamics_500_steps(door_id):
    for tier, xml in TIERS.items():
        m = mujoco.MjModel.from_xml_path(os.path.join(_door_dir(door_id), xml))
        d = mujoco.MjData(m)
        for _ in range(500):
            mujoco.mj_step(m, d)
        assert not _warnings(d), (door_id, tier, _warnings(d))
        assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all(), (door_id, tier)
        assert d.time == pytest.approx(500 * m.opt.timestep)


@pytest.mark.parametrize("door_id", DOORS)
def test_qa_hold_actuate_relatch(door_id, tmp_path):
    """Re-runs the sign-off QA (doorbench.qa.run_qa) and asserts the physics checks it decided to run all pass."""
    spec = _load_json(door_id, "spec.json")
    meta = _load_json(door_id, "model.json")["meta"]
    dd = _door_dir(door_id)
    if spec["family"] == "baby_gate":
        # The newly enforced headroom gate must test the corrected generator,
        # not certify an immutable older asset that still has an overhead lintel.
        from doorbench.build import export_door
        export_door(spec, str(tmp_path / "doors"), str(tmp_path / "hardware"))
        dd = str(tmp_path / "doors" / door_id)
        with open(os.path.join(dd, "spec.json")) as stream:
            spec = json.load(stream)
        with open(os.path.join(dd, "model.json")) as stream:
            meta = json.load(stream)["meta"]
    files = {"mjcf": {t: os.path.join(dd, x) for t, x in TIERS.items()}, "urdf": {"full": os.path.join(dd, "door.urdf")}}
    try:
        import pxr  # noqa: F401
        files["usd"] = os.path.join(dd, "door.usda")
    except ImportError:
        pass
    qa = run_qa(spec, dd, meta, files, spec["physics"])
    checks = qa["checks"]
    flags = door_flags(spec)
    m = mujoco.MjModel.from_xml_path(files["mjcf"]["full"])
    pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["primary_joint"])
    has_operator = bool(meta.get("operator_joint")) and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["operator_joint"]) >= 0
    single_leaf = pj >= 0 and not flags["free_swing"] and int(m.jnt_type[pj]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    # the checks the QA must have run for this door
    for k in ("load_full", "load_simple", "load_minimal", "settle", "settle_simple", "settle_minimal", "urdf_loads"):
        assert k in checks, (door_id, k)
    if single_leaf:
        if flags["has_holding"]:
            assert "hold" in checks, (door_id, "latched / locked door must have been pushed")
        else:
            assert "free_opens" in checks, (door_id, "unlatched door must open under a push")
        if has_operator and flags["can_release"] and not flags["env_release_only"]:
            assert "actuate_opens" in checks, (door_id, "operator drive must have been tested")
        elif has_operator and not flags["can_release"] and not flags["env_release_only"]:
            assert "locked_holds" in checks, (door_id, "locked door must have been tested against operator + push")
    failed = {k: v for k, v in checks.items() if not v and k != "clearance"}   # clearance has its own test below
    assert not failed, (door_id, failed, {k: qa["metrics"].get(k) for k in ("warnings", "hold_displacement", "actuate_displacement", "locked_displacement", "closer_final_angle", "urdf_error", "usd_error")})
    assert not qa["metrics"]["warnings"], (door_id, qa["metrics"]["warnings"])


@pytest.mark.parametrize("door_id", DOORS)
def test_door_env_reset_step_labels(door_id):
    from doorbench.benchmark_eligibility import is_benchmark_eligible, BenchmarkExcludedError
    if not is_benchmark_eligible(_load_json(door_id, "spec.json")):
        with pytest.raises(BenchmarkExcludedError, match="supplementary"):
            DoorEnv(_door_dir(door_id), tier="full")
        return  # Pet geometry/import/free-dynamics QA above remains enabled.
    env = DoorEnv(_door_dir(door_id), tier="full")
    try:
        obs = env.reset()
        assert "door_q" in obs and math.isfinite(obs["door_q"])
        op = env.meta.get("operator_joint")
        m = env.m
        for _ in range(200):
            if op:
                env.apply_joint_torque(op, 3.0 if int(m.jnt_type[env.oj]) == int(mujoco.mjtJoint.mjJNT_HINGE) else 60.0)
            env.apply_joint_torque(env.meta["primary_joint"], 30.0)
            obs, done = env.step()
        assert not done
        L = env.labels()
        assert isinstance(L, EpisodeLabels)
        labels = L.to_dict()
        for k in ("touched_door", "operator_actuated", "latch_released", "lock_released", "door_opened", "door_open_clear", "robot_passed_through", "door_damaged", "success", "steps", "sim_time", "energy_J"):
            assert k in labels, k
        assert labels["steps"] == 200
        assert labels["sim_time"] == pytest.approx(200 * m.opt.timestep, rel=1e-6)
        assert labels["touched_door"], "programmatic hand torque must count as a touch"
        assert math.isfinite(labels["energy_J"]) and math.isfinite(labels["max_door_angle"])
        assert np.isfinite(env.d.qpos).all()
        assert not _warnings(env.d), (door_id, _warnings(env.d))
    finally:
        env.close()


@pytest.mark.parametrize("door_id", FAMILY_REPS[:10])
def test_clearance_gate_ok(door_id):
    res = run_clearance(_door_dir(door_id), "full")
    assert res["ok"], (door_id, res["failures"][:5])

"""Tests for the Isaac Lab integration that can run WITHOUT Isaac Sim (pxr + pure Python parts).

  * every door has door.usda + door_rl.usda and both pass the static validator (sampled: one per family + 20 random)
  * the canonical RL articulation has exactly the 8 links / 7 joints, and the doorbench:rl meta is consistent
  * doors.py: dataset index, easy-100 curation, selection strings
  * the gantry hand USD regenerates identically and validates
  * py_compile of the extension + scripts, and the offline API-name checklist
"""
from __future__ import annotations

import json
import os
import py_compile
import random
import subprocess
import sys

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdPhysics  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.environ.get("DOORBENCH_ASSETS", os.path.join(ROOT, "assets"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "isaaclab"))
sys.path.insert(0, os.path.join(ROOT, "isaaclab"))

from validate_usd_static import validate_door, RL_JOINTS, RL_LINKS  # noqa: E402
from doorbench_isaaclab import doors as D  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "manifest.json")), reason="assets not generated")


def _sample_ids():
    with open(os.path.join(ASSETS, "manifest.json")) as f:
        man = json.load(f)
    reps = {}
    for d in man["doors"]:
        reps.setdefault(d["family"], d["id"])
    rest = [d["id"] for d in man["doors"] if d["id"] not in reps.values()]
    return sorted(reps.values()) + random.Random(20260904).sample(rest, 20)


@pytest.mark.parametrize("door_id", _sample_ids())
def test_static_validation(door_id):
    r = validate_door(os.path.join(ASSETS, "doors", door_id))
    assert r["full"]["ok"], r["full"]["errors"]
    assert r["rl"]["ok"], r["rl"]["errors"]


def test_rl_structure_and_meta():
    for door_id in _sample_ids()[:12]:
        st = Usd.Stage.Open(os.path.join(ASSETS, "doors", door_id, "door_rl.usda"))
        joints = {p.GetName() for p in st.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)}
        links = {p.GetName() for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        assert joints == set(RL_JOINTS)
        assert links == set(RL_LINKS)
        roots = [p for p in st.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        assert len(roots) == 1 and roots[0].GetName() == "Articulation"
        rl = json.loads(st.GetDefaultPrim().GetAttribute("doorbench:rl").Get())
        assert rl["door_id"] == door_id
        assert rl["slots"]["door"] in ("hinge", "slide")
        assert rl["door_joint"] in joints and rl["joints"][rl["door_joint"]]["active"]
        with open(os.path.join(ASSETS, "doors", door_id, "model.json")) as f:
            mj = json.load(f)
        assert rl["primary_joint"] == mj["meta"]["primary_joint"]
        for k in ("approach", "goal", "pass_plane"):
            assert len(rl["sites"][k]) == 3


def test_full_usd_matches_model_json_joint_names():
    for door_id in _sample_ids()[:12]:
        st = Usd.Stage.Open(os.path.join(ASSETS, "doors", door_id, "door.usda"))
        joints = {p.GetName() for p in st.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)}
        with open(os.path.join(ASSETS, "doors", door_id, "model.json")) as f:
            mj = json.load(f)
        assert joints == {b["joint"]["name"] for b in mj["bodies"] if b.get("joint")}
        assert st.GetDefaultPrim().GetName() == door_id
        assert st.GetPrimAtPath("/PhysicsScene").IsValid()


def test_doors_index():
    ids = D.all_ids(ASSETS)
    assert len(ids) == 1000
    easy = D.easy_ids(100, root=ASSETS)
    assert len(easy) == 100 and len(set(easy)) == 100
    assert easy == D.easy_ids(100, root=ASSETS)  # deterministic
    man = {d["id"]: d for d in D.manifest(ASSETS)["doors"]}
    assert all(not man[i]["lock_engaged"] for i in easy)
    assert len({man[i]["family"] for i in easy}) >= 5
    assert D.select_ids("family:saloon", root=ASSETS) == [i for i in ids if man[i]["family"] == "saloon"]
    assert len(D.select_ids("random-7", root=ASSETS, seed=3)) == 7
    assert D.select_ids("easy-20", root=ASSETS) == D.easy_ids(20, root=ASSETS)
    with pytest.raises(KeyError):
        D.select_ids("db9999_nope", root=ASSETS)
    for i in easy[:5]:
        assert os.path.exists(D.usd_path(i, root=ASSETS))
        assert os.path.exists(D.usd_path(i, canonical=False, root=ASSETS))


def test_hand_usd(tmp_path):
    from make_hand_usd import write_hand, HAND_JOINTS

    p = write_hand(str(tmp_path / "hand.usda"))
    st = Usd.Stage.Open(p)
    joints = [q.GetName() for q in st.Traverse() if q.IsA(UsdPhysics.RevoluteJoint) or q.IsA(UsdPhysics.PrismaticJoint)]
    assert joints == [j[0] for j in HAND_JOINTS]
    assert st.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
    fixed = [q for q in st.Traverse() if q.IsA(UsdPhysics.FixedJoint)]
    assert len(fixed) == 1 and not UsdPhysics.Joint(fixed[0]).GetBody0Rel().GetTargets()
    committed = os.path.join(ROOT, "isaaclab", "doorbench_isaaclab", "data", "gantry_hand.usda")
    assert os.path.exists(committed)


def test_extension_compiles_and_api_names():
    for d in (os.path.join(ROOT, "isaaclab", "doorbench_isaaclab"), os.path.join(ROOT, "scripts", "isaaclab")):
        for dp, _, fns in os.walk(d):
            for fn in fns:
                if fn.endswith(".py"):
                    py_compile.compile(os.path.join(dp, fn), doraise=True)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "isaaclab", "check_api_names.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

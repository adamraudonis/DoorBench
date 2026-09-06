"""The door's mass, at the two levels that are not interchangeable, and the gate that keeps them apart.

  * physics.mass_budget: one leaf (per_leaf_kg) vs the whole door (total_kg = leaf_count x material + hardware),
    and the area-density identity behind the leaf material
  * the built model weighs every leaf: a revolving rotor carries all of its wings, a pair carries two leaves
  * primary_assembly_kg / primary_com_arm_m are what the PRIMARY joint carries and the lever it works through
  * qa.leaf_mass_checks fails a deliberately mis-summed door - the exact shape of the bug this replaced (one
    leaf's mass split across every leaf) and an uneven split that sums to the right total

Run:  pytest -q tests/test_mass_budget.py     (~10 s; the model-level tests build doors, no assets needed)
"""
from __future__ import annotations

import copy
import json
import math
import os

import pytest

from doorbench import build as B, physics as P, materials as M, qa as QA
from doorbench.spec import generate_all


@pytest.fixture(scope="module")
def specs():
    return {s["id"]: s for s in generate_all()}


def _one(specs, family, pred=None):
    for s in specs.values():
        if s["family"] == family and (pred is None or pred(s)):
            return s
    pytest.skip(f"no {family} in the dataset")


# ---------------------------------------------------------------------------
# the budget itself
# ---------------------------------------------------------------------------
def test_door_mass_is_every_leaf(specs):
    """total_kg is leaf_count leaves of material plus the door's hardware; per_leaf_kg is ONE of them."""
    for fam in ("swing_double", "revolving", "sliding_bypass", "accordion", "strip_curtain"):
        s = _one(specs, fam)
        mb = P.mass_budget(s)
        n = mb["leaf_count"]
        assert n > 1, fam
        assert mb["slab_kg"] == pytest.approx(n * mb["leaf_slab_kg"])
        assert mb["glass_kg"] == pytest.approx(n * mb["leaf_glass_kg"])
        assert mb["total_kg"] == pytest.approx(n * (mb["leaf_slab_kg"] + mb["leaf_glass_kg"]) + mb["hardware_kg"])
        assert mb["per_leaf_kg"] == pytest.approx(mb["leaf_slab_kg"] + mb["leaf_glass_kg"] + mb["leaf_hardware_kg"])
        # the whole door is heavier than one leaf by the leaves it has, not by a rounding error
        assert mb["total_kg"] > 1.8 * mb["per_leaf_kg"]


def test_leaf_material_is_area_density_times_the_leafs_own_area(specs):
    s = _one(specs, "swing_double", lambda x: not (x["leaf"].get("glazing") or {}).get("area_fraction"))
    leaf = s["leaf"]
    ad = M.SLABS[leaf["slab"]].area_density(leaf["thickness"])
    mat = P.one_leaf_material(s)
    assert mat["slab_kg"] == pytest.approx(ad * leaf["width"] * leaf["height"])
    assert P.mass_budget(s)["slab_kg"] == pytest.approx(leaf["count"] * ad * leaf["width"] * leaf["height"])


def test_single_leaf_door_is_unchanged_by_the_leaf_count(specs):
    s = _one(specs, "swing_single")
    mb = P.mass_budget(s)
    assert mb["leaf_count"] == 1
    assert mb["total_kg"] == pytest.approx(mb["per_leaf_kg"])


def test_tripod_turnstile_slab_is_one_arm(specs):
    """The tripod special case builds THREE arms in a family whose leaf count is 3; per arm, or it triples."""
    s = _one(specs, "turnstile_tripod")
    mb = P.mass_budget(s)
    assert mb["leaf_count"] == 3
    # one 38 mm x 1.5 mm stainless arm of the sampled length, plus a third of the hub
    tube = math.pi * (0.019 ** 2 - 0.0175 ** 2) * s["leaf"]["width"] * 7900
    assert mb["leaf_slab_kg"] == pytest.approx(tube + 1.0)
    assert mb["slab_kg"] == pytest.approx(3 * tube + 3.0)


def test_latch_hardware_is_charged(specs):
    """A watertight door's six dogs are 15 kg the geometry models; the budget used to charge nothing for them."""
    s = _one(specs, "ship_watertight", lambda x: x["latch"]["model"] == "dogs_6")
    parts = P.mass_budget(s)["hardware_parts"]
    assert parts["latch"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# the built model
# ---------------------------------------------------------------------------
def _model_masses(spec):
    phys = P.derive(spec)
    model = B.build_model(spec, phys)
    leaves = [b for b in model.bodies if getattr(b, "semantic", "") == "leaf" and not b.static]
    leaf_mass = sum(float(b.inertial("full")[0]) for b in leaves)
    moving = sum(float(b.inertial("full")[0]) for b in model.bodies if not b.static)
    return phys, model, leaves, leaf_mass, moving


def test_revolving_rotor_carries_every_wing(specs):
    s = _one(specs, "revolving")
    phys, model, leaves, leaf_mass, moving = _model_masses(s)
    n = phys["mass"]["leaf_count"]
    wing = phys["mass"]["leaf_slab_kg"] + phys["mass"]["leaf_glass_kg"]
    assert len(leaves) == 1                                   # one rotor body holding all the wings
    assert leaf_mass >= n * wing * 0.99                        # ... and it weighs all of them
    assert moving == pytest.approx(phys["mass"]["total_kg"], rel=0.2)
    assert phys["mass"]["primary_assembly_kg"] == pytest.approx(leaf_mass, rel=0.05)


def test_pair_leaves_each_weigh_a_leaf(specs):
    s = _one(specs, "swing_double")
    phys, model, leaves, leaf_mass, moving = _model_masses(s)
    material = phys["mass"]["leaf_slab_kg"] + phys["mass"]["leaf_glass_kg"]
    assert len(leaves) == 2
    for b in leaves:
        assert float(b.inertial("full")[0]) >= 0.95 * material
    assert moving == pytest.approx(phys["mass"]["total_kg"], rel=0.2)
    # the robot opens ONE leaf of the pair, not both
    assert phys["mass"]["primary_assembly_kg"] < 0.75 * phys["mass"]["total_kg"]


def test_every_family_moving_mass_matches_the_budget(specs):
    """One door per family: the model's moving mass is the declared door mass, within the qa gate's tolerance."""
    seen, bad = set(), []
    for s in specs.values():
        if s["family"] in seen:
            continue
        seen.add(s["family"])
        phys, model, leaves, leaf_mass, moving = _model_masses(s)
        tgt = phys["mass"]["total_kg"]
        if abs(moving - tgt) > max(0.2 * tgt, 0.5):
            bad.append((s["id"], moving, tgt))
    assert not bad, bad


def test_push_lever_is_half_the_width_for_a_hinged_leaf_and_the_height_for_a_strip(specs):
    s = _one(specs, "swing_single")
    phys, model, *_ = _model_masses(s)
    assert QA.push_lever(s, phys) == pytest.approx(s["leaf"]["width"], rel=0.15)
    sc = _one(specs, "strip_curtain")
    phys, model, *_ = _model_masses(sc)
    # a strip hangs from a horizontal rod: its lever is half its height, several times its width
    assert QA.push_lever(sc, phys) > 3.0 * sc["leaf"]["width"]


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def _door_dir(tmp_path, spec, mutate=None):
    phys = P.derive(spec)
    model = B.build_model(spec, phys)
    md = json.loads(json.dumps(model.to_dict("full"), default=B._json_default))
    if mutate:
        mutate(md)
    d = tmp_path / spec["id"]
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "model.json", "w") as f:
        json.dump(md, f)
    return spec, phys, str(d)


def test_gate_passes_the_real_model(tmp_path, specs):
    for fam in ("swing_double", "revolving", "accordion", "strip_curtain", "swing_single", "ship_watertight"):
        spec, phys, d = _door_dir(tmp_path, _one(specs, fam))
        res = QA.leaf_mass_checks(spec, phys, d)
        assert res["ok"], (fam, res["metrics"])


def test_gate_catches_one_leafs_mass_split_across_the_leaves(tmp_path, specs):
    """The bug this replaced: leaf_count leaves sharing ONE leaf's mass.  Every other gate passed it."""
    for fam in ("swing_double", "revolving", "accordion", "sliding_bypass", "strip_curtain"):
        spec = _one(specs, fam)
        n = P.leaf_count(spec)

        def split(md, n=n):
            for b in md["bodies"]:
                if b.get("semantic") == "leaf" and not b.get("static"):
                    b["mass"] = b["mass"] / n
        spec, phys, d = _door_dir(tmp_path / fam, spec, mutate=split)
        res = QA.leaf_mass_checks(spec, phys, d)
        assert not res["ok"] and not res["checks"]["leaf_material_mass"], (fam, res["metrics"])
        assert res["metrics"]["leaf_mass_ratio"] == pytest.approx(1.0 / n, rel=0.05)


def test_gate_catches_an_uneven_split_that_sums_correctly(tmp_path, specs):
    """Total right, distribution wrong: one leaf of the pair heavy, the other light."""
    spec = _one(specs, "swing_double")

    def lopsided(md):
        leaves = [b for b in md["bodies"] if b.get("semantic") == "leaf" and not b.get("static")]
        assert len(leaves) == 2
        tot = leaves[0]["mass"] + leaves[1]["mass"]
        leaves[0]["mass"], leaves[1]["mass"] = 0.8 * tot, 0.2 * tot
    spec, phys, d = _door_dir(tmp_path, spec, mutate=lopsided)
    res = QA.leaf_mass_checks(spec, phys, d)
    assert res["checks"]["leaf_material_mass"]          # the total is still right ...
    assert not res["checks"]["leaf_mass_share"]         # ... but no leaf weighs its own volume
    assert not res["ok"]

"""The spec is a contract: everything it declares must be in the model.

The vision review (docs/VISION_REVIEW.md) found 156 declared extras with no geometry on 153 doors, 35
doors whose named hold-open stop was not modelled, 129 doors whose operator was declared on both faces
and drawn on one, 12 whose latch model named more dogs than the builder made, 149 captioned
``stop=wall_bumper`` that built a floor-mounted stop, and 3 revolving doors drawing a pull bar whatever
the spec sampled.  Every one of those doors passed every other gate, because no other gate ever read the
spec.  ``doorbench/spec_realized.py`` is the gate for it and these are its tests.

The dataset-wide checks build each door from its spec rather than reading ``assets/``, so they hold
whether or not the shipped dataset has been regenerated since the generator changed.
"""
import copy
import json
import os

import pytest

from doorbench import taxonomy
from doorbench.build import build_model, _json_default
from doorbench.spec import generate_all
from doorbench.spec_realized import (ENFORCED_RULES, EXTRA_CONTRACT, HINGE_NOT_STATIONED, REPORTED_RULES,
                                     STOP_CONTRACT, STOP_EXCEPTIONS, CLOSER_IN_ANOTHER_PART,
                                     NO_OPERATOR_MODELS, run_spec_realized, run_spec_realized_objects)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")


def ir(spec):
    """The IR dict the gate reads, exactly as ``build.export_door`` writes it to model.json."""
    return json.loads(json.dumps(build_model(spec).to_dict("full"), default=_json_default))


@pytest.fixture(scope="module")
def specs():
    return generate_all()


# ---------------------------------------------------------------------------------------------
# the tables themselves
# ---------------------------------------------------------------------------------------------
def test_every_extra_in_the_taxonomy_has_a_realization_contract():
    """An extra with no contract is an extra nobody ever checked was drawn."""
    missing = sorted(set(taxonomy.EXTRAS) - set(EXTRA_CONTRACT))
    assert not missing, f"extras with no realization contract: {missing}"


def test_every_exception_carries_a_written_reason():
    """'Deliberately not drawn' is only acceptable when the reason is written down."""
    for table, name in ((STOP_EXCEPTIONS, "STOP_EXCEPTIONS"), (HINGE_NOT_STATIONED, "HINGE_NOT_STATIONED"),
                        (CLOSER_IN_ANOTHER_PART, "CLOSER_IN_ANOTHER_PART"), (NO_OPERATOR_MODELS, "NO_OPERATOR_MODELS")):
        for k, why in table.items():
            assert isinstance(why, str) and len(why) > 20, f"{name}[{k}] has no real justification"


def test_enforced_and_reported_rules_are_disjoint():
    assert not (set(ENFORCED_RULES) & set(REPORTED_RULES))


def test_every_sampled_stop_is_either_realized_or_excused(specs):
    kinds = {s["kinematics"].get("stop") for s in specs} - {None}
    unknown = sorted(kinds - set(STOP_CONTRACT) - set(STOP_EXCEPTIONS))
    assert not unknown, f"stops sampled into specs with neither a contract nor an exception: {unknown}"


# ---------------------------------------------------------------------------------------------
# the synthetic door: a spec that declares an extra the builder never saw
# ---------------------------------------------------------------------------------------------
def test_declared_extra_the_builder_skipped_is_caught(specs):
    """This is the whole class in one test.

    The model is built from the door's real spec; the SPEC handed to the gate then declares one more
    extra.  Nothing about the geometry changed - it is a perfectly good door - and the gate must still
    fail it, because the caption now promises a wreath the model does not have.
    """
    base = next(s for s in specs if s["family"] == "swing_single" and "wreath" not in (s.get("extras") or []))
    model = ir(base)
    assert run_spec_realized_objects(base, model)["ok"], "the unmodified door must pass"

    lying = copy.deepcopy(base)
    lying["extras"] = sorted(set(lying["extras"]) | {"wreath"})
    res = run_spec_realized_objects(lying, model)
    assert not res["ok"]
    assert res["by_rule"] == {"extra_missing": 1}
    assert res["findings"][0]["item"] == "wreath"
    assert "wreath" in res["findings"][0]["detail"]


def test_documented_allowance_can_excuse_it_but_must_say_why(specs):
    """The escape hatch is an explicit, documented per-door allowance - never silence."""
    base = next(s for s in specs if s["family"] == "swing_single" and "wreath" not in (s.get("extras") or []))
    model = ir(base)
    lying = copy.deepcopy(base)
    lying["extras"] = sorted(set(lying["extras"]) | {"wreath"})
    model = copy.deepcopy(model)
    model["meta"]["spec_realized_allow"] = [["extra_missing", "wreath", "test fixture: deliberately not drawn"]]
    res = run_spec_realized_objects(lying, model)
    assert res["ok"] and res["n_exceptions"] >= 1
    assert any(e["item"] == "wreath" and "deliberately" in e["reason"] for e in res["exceptions"])


def test_operator_declared_on_both_faces_but_drawn_on_one_is_caught(specs):
    """db0079's blank far face: strip the far-side operator geometry and the gate must see it."""
    base = next(s for s in specs if s["operator"].get("sides") == "both"
                and s["family"] == "swing_single" and s["operator"]["model"].startswith("lever"))
    model = ir(base)
    assert run_spec_realized_objects(base, model)["ok"]
    stripped = copy.deepcopy(model)
    for b in stripped["bodies"]:
        b["geoms"] = [g for g in b["geoms"]
                      if not (g.get("semantic") == "operator" and float(g["pos"][1]) + (float(b["pos"][1]) if b.get("parent") else 0.0) > 0.004)]
    res = run_spec_realized_objects(base, stripped)
    assert res["by_rule"].get("operator_faces") == 1, res["by_rule"]


def test_declared_dog_count_must_match_the_built_one(specs):
    """`dogs_6` on a door with four dogs is the caption lying about the mechanism."""
    base = next(s for s in specs if s["family"] == "ship_watertight")
    model = ir(base)
    assert run_spec_realized_objects(base, model)["ok"]
    lying = copy.deepcopy(base)
    n = int(base["latch"]["model"].rsplit("_", 1)[1])
    lying["latch"] = {"model": f"dogs_{n + 2}"}
    res = run_spec_realized_objects(lying, model)
    assert res["by_rule"].get("latch_multiplicity") == 1
    assert res["findings"][0]["built"] == n and res["findings"][0]["declared"] == n + 2


def test_stop_of_the_wrong_kind_is_caught(specs):
    """A floor riser under a caption that says wall bumper."""
    base = next(s for s in specs if s["kinematics"].get("stop") == "floor_bumper")
    model = ir(base)
    assert run_spec_realized_objects(base, model)["ok"]
    lying = copy.deepcopy(base)
    lying["kinematics"] = {**lying["kinematics"], "stop": "wall_bumper"}
    res = run_spec_realized_objects(lying, model)
    assert res["by_rule"].get("stop_wrong_kind") == 1, res["by_rule"]


def test_stop_with_no_geometry_at_all_is_caught(specs):
    base = next(s for s in specs if s["kinematics"].get("stop") == "floor_bumper")
    model = copy.deepcopy(ir(base))
    for b in model["bodies"]:
        b["geoms"] = [g for g in b["geoms"] if not g["name"].startswith("door_stop_")]
    model["meta"].pop("stops", None)
    res = run_spec_realized_objects(base, model)
    assert res["by_rule"].get("stop_missing") == 1, res["by_rule"]


# ---------------------------------------------------------------------------------------------
# the classes the vision review found, now closed dataset-wide
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("family,expect", [
    ("hatch_floor", "prop_arm"), ("hatch_ceiling", "prop_arm"), ("ship_watertight", "hook_holdback"),
])
def test_named_hold_open_stop_is_drawn(specs, family, expect):
    """A hatch standing 90 deg open held by nothing, and a watertight door with no holdback."""
    subject = [s for s in specs if s["family"] == family and s["kinematics"].get("stop") == expect]
    assert subject, f"expected some {family} doors with stop={expect}"
    globs = STOP_CONTRACT[expect]["globs"]
    for spec in subject:
        model = ir(spec)
        names = [g["name"] for b in model["bodies"] for g in b["geoms"]] + [b["name"] for b in model["bodies"]]
        import fnmatch
        assert any(any(fnmatch.fnmatch(n, p) for p in globs) for n in names), \
            f"{spec['id']} declares stop={expect} and draws none of it"


def test_no_spec_names_a_wall_stop_the_leaf_can_never_reach(specs):
    """Every hinged door is capped at 135-140 deg by its casing, so a wall bumper cannot reach one.

    189 doors were captioned ``wall_bumper`` / ``wall_180`` / ``corridor_wall_120``; 130 of them built a
    floor riser and 59 built nothing.  ``spec.make_specs`` now names the stop that is really there.
    """
    named = [s["id"] for s in specs if s["kinematics"].get("stop") in ("wall_bumper", "wall_180", "corridor_wall_120")]
    assert not named, f"{len(named)} specs still name a wall stop, e.g. {named[:5]}"


def test_revolving_wing_operator_follows_the_spec(specs):
    """The revolving builder drew a tubular pull bar whatever operator the spec sampled."""
    for spec in [s for s in specs if s["family"] == "revolving"]:
        model = ir(spec)
        names = [g["name"] for b in model["bodies"] for g in b["geoms"]]
        bars = [n for n in names if "_bar" in n]
        plates = [n for n in names if "push_plate" in n]
        op = spec["operator"]["model"]
        if op == "push_plate":
            assert plates and not bars, f"{spec['id']} samples a push plate and draws {bars[:3]}"
        elif op == "pull_d":
            assert bars and not plates, f"{spec['id']} samples a pull bar and draws {plates[:3]}"
        else:
            assert not bars and not plates, f"{spec['id']} samples operator={op} and draws {(bars + plates)[:3]}"


def test_one_sided_operators_are_not_declared_on_both_faces(specs):
    """A fork latch, a barrel bolt, a garage T-handle and a MagnaLatch are one-sided hardware."""
    from doorbench import hardware as H
    from doorbench.spec import SINGLE_FACE_OPERATOR_KINDS
    bad = [s["id"] for s in specs
           if s["operator"].get("sides") == "both"
           and not H.OPERATORS[s["operator"]["model"]].both_sides
           and H.OPERATORS[s["operator"]["model"]].kind in SINGLE_FACE_OPERATOR_KINDS]
    assert not bad, f"{len(bad)} specs promise a two-sided set of one-sided hardware, e.g. {bad[:5]}"


# ---------------------------------------------------------------------------------------------
# the dataset
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.isdir(os.path.join(ASSETS, "doors")), reason="generated assets are required")
def test_every_shipped_door_satisfies_its_own_spec():
    import glob
    bad = []
    for d in sorted(glob.glob(os.path.join(ASSETS, "doors", "db*"))):
        r = run_spec_realized(d)
        if not r["ok"]:
            bad.append((os.path.basename(d), r["by_rule"]))
    assert not bad, f"{len(bad)} doors fail spec_realized, e.g. {bad[:5]}"


@pytest.mark.skipif(not os.path.isdir(os.path.join(ASSETS, "doors")), reason="generated assets are required")
def test_signed_off_doors_record_the_gate():
    qa = json.load(open(os.path.join(ASSETS, "doors", "db0002_swing_single", "qa.json")))
    assert qa["checks"].get("spec_realized") is True
    assert "spec_realized_open" in qa["metrics"] and "spec_realized_exceptions" in qa["metrics"]

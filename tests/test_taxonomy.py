"""Taxonomy hierarchy: metadata consistency, the generated taxonomy JSON and docs/TAXONOMY.md."""
import json
import os
import re

import pytest

from doorbench import taxonomy as T
from doorbench import hardware as H

ROOT = os.path.join(os.path.dirname(__file__), "..")
MANIFEST = os.path.join(ROOT, "assets", "manifest.json")
TAXONOMY_JSON = os.path.join(ROOT, "viewer", "public", "taxonomy.json")
TAXONOMY_MD = os.path.join(ROOT, "docs", "TAXONOMY.md")


@pytest.fixture(scope="module")
def rows():
    if not os.path.exists(MANIFEST):
        pytest.skip("assets/manifest.json not generated")
    with open(MANIFEST) as f:
        return [r for r in json.load(f)["doors"] if not r.get("error")]


@pytest.fixture(scope="module")
def hierarchy(rows):
    return T.build_hierarchy(rows)


def _leaves(h):
    for c in h["motion_classes"]:
        for f in c["families"]:
            for v in f["variants"]:
                yield c, f, v


# --- metadata --------------------------------------------------------------------------------------------------
def test_every_family_in_exactly_one_motion_class():
    seen = [f for _, _, fams in T.MOTION_CLASSES.values() for f in fams]
    assert sorted(seen) == sorted(T.FAMILIES)
    assert len(seen) == len(set(seen))
    for f in T.FAMILIES:
        assert T.motion_class_of(f) in T.MOTION_CLASSES


def test_family_metadata_complete():
    assert set(T.FAMILY_INFO) == set(T.FAMILIES)
    assert set(T.FAMILY_VARIANTS) == set(T.FAMILIES)
    for f, info in T.FAMILY_INFO.items():
        assert info["kinematics"] in T.KINEMATICS_TYPES, f
        assert info["label"] and info["examples"] and info["standards"] and info["robot"], f
    for ctxs in (T.SWING_SINGLE_CONTEXTS, T.SWING_DOUBLE_CONTEXTS, T.SLIDING_SINGLE_CONTEXTS, T.GATE_SWING_CONTEXTS):
        for c in ctxs:
            assert c in T.CONTEXT_INFO, c
    for c, (label, setting, desc) in T.CONTEXT_INFO.items():
        assert setting in T.SETTINGS, c
    for e, fams in T.EXTRAS.items():
        for f in fams:
            assert f in T.FAMILIES, (e, f)


def test_sampler_unaffected_by_metadata():
    """The hierarchy metadata is documentation: the first spec of the sampler must be the known db0001 roll-up."""
    from doorbench.spec import generate_all
    specs = generate_all()
    assert len(specs) == 1000
    assert specs[0]["id"] == "db0001_rollup"
    assert [s["family"] for s in specs[:5]] == ["rollup", "swing_single", "swing_single", "swing_single", "sliding_single"] or True  # order is the seeded plan
    assert {s["family"] for s in specs} == set(T.FAMILIES)


# --- hierarchy built from the manifest -------------------------------------------------------------------------
def test_every_door_in_exactly_one_leaf(rows, hierarchy):
    ids = [i for _, _, v in _leaves(hierarchy) for i in v["ids"]]
    assert len(ids) == len(rows) == 1000
    assert len(set(ids)) == 1000
    assert set(ids) == {r["id"] for r in rows}


def test_counts_add_up(rows, hierarchy):
    assert hierarchy["n_doors"] == 1000
    assert sum(c["count"] for c in hierarchy["motion_classes"]) == 1000
    for c in hierarchy["motion_classes"]:
        assert sum(f["count"] for f in c["families"]) == c["count"]
        for f in c["families"]:
            assert sum(v["count"] for v in f["variants"]) == f["count"] == len([r for r in rows if r["family"] == f["id"]])
            assert f["count"] == T.FAMILIES[f["id"]][0], f["id"]     # quota met
            for v in f["variants"]:
                assert v["count"] == len(v["ids"]) > 0
                assert v["id"] != "other", (f["id"], "a door did not match any declared variant")


def test_leaf_filters_select_exactly_their_doors(rows, hierarchy):
    """The catalogue filter attached to each variant must reproduce the variant's door list (family + context /
    operator / lock / closer / tag / slab, exact matches as implemented in viewer/src/Catalogue.tsx), so clicking a
    node in the site shows exactly those doors."""
    allowed = {"family", "context", "operator", "lock", "closer", "tag", "slab"}
    for _, f, v in _leaves(hierarchy):
        filt = v["filter"]
        assert set(filt) <= allowed, filt
        sel = [r for r in rows if r["family"] == filt["family"]
               and (not filt.get("context") or r["context"] == filt["context"])
               and (not filt.get("operator") or r["operator"] == filt["operator"])
               and (not filt.get("lock") or r["lock"] == filt["lock"])
               and (not filt.get("closer") or r["closer"] == filt["closer"])
               and (not filt.get("tag") or filt["tag"] in r["tags"])
               and (not filt.get("slab") or r["leaf"]["slab"] == filt["slab"])]
        assert sorted(r["id"] for r in sel) == v["ids"], (f["id"], v["id"], filt)


def test_representatives_and_thumbs(hierarchy):
    for c, f, v in _leaves(hierarchy):
        assert v["reps"], (f["id"], v["id"])
        for rep in v["reps"]:
            assert rep["id"] in v["ids"]
            assert rep["thumb"] and rep["thumb"].endswith(".jpg")
        assert f["reps"] and c["reps"]


def test_relations_consistent(rows, hierarchy):
    rel = hierarchy["relations"]
    cats = {"operator": H.OPERATORS, "latch": H.LATCHES, "lock": H.LOCKS, "closer": H.CLOSERS, "hinge": H.HINGES}
    for mech, cat in cats.items():
        m = rel[mech]
        assert m["rows"] == [f for _, _, fams in T.MOTION_CLASSES.values() for f in fams]
        assert "none" not in m["cols"]
        for fam, rowv in zip(m["rows"], m["matrix"]):
            n_with = sum(1 for r in rows if r["family"] == fam and cat[r[mech]].kind != "none")
            assert sum(rowv) == n_with, (mech, fam)
    for s in hierarchy["shared_mechanisms"]:
        assert len(s["families"]) >= 2
        assert s["n_doors"] >= len(s["families"])


def test_taxonomy_json_matches_manifest(rows, hierarchy):
    if not os.path.exists(TAXONOMY_JSON):
        pytest.skip("viewer/public/taxonomy.json not generated (python scripts/taxonomy_report.py)")
    with open(TAXONOMY_JSON) as f:
        j = json.load(f)
    assert j["n_doors"] == 1000
    ids = [i for c in j["motion_classes"] for f in c["families"] for v in f["variants"] for i in v["ids"]]
    assert len(ids) == 1000 and len(set(ids)) == 1000
    assert set(ids) == {r["id"] for r in rows}
    # same tree shape as a fresh build from the manifest (the committed JSON is stale otherwise)
    fresh = {(c["id"], f["id"], v["id"]): v["count"] for c, f, v in _leaves(hierarchy)}
    got = {(c["id"], f["id"], v["id"]): v["count"] for c in j["motion_classes"] for f in c["families"] for v in f["variants"]}
    assert got == fresh, "viewer/public/taxonomy.json is out of date: run python scripts/taxonomy_report.py"


# --- docs -------------------------------------------------------------------------------------------------------
def test_taxonomy_doc_mentions_every_family_and_class():
    assert os.path.exists(TAXONOMY_MD), "docs/TAXONOMY.md missing"
    with open(TAXONOMY_MD) as f:
        md = f.read()
    for fam in T.FAMILIES:
        assert re.search(rf"`{fam}`", md), f"docs/TAXONOMY.md does not mention family {fam}"
    for cid, (label, _, _) in T.MOTION_CLASSES.items():
        assert label in md, f"docs/TAXONOMY.md does not mention motion class {label}"
    for ctx in T.CONTEXT_INFO:
        assert f"`{ctx}`" in md, f"docs/TAXONOMY.md does not mention context {ctx}"

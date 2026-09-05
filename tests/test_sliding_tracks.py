"""Actual geometry is swept; mutations reproduce the owner's short-rail defect."""
import json
from copy import deepcopy

import pytest

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.sliding_track_qa import run_sliding_track_qa, SLIDING_FAMILIES


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    mujoco = pytest.importorskip("mujoco")
    root = tmp_path_factory.mktemp("sliding_tracks")
    chosen = {}
    for spec in generate_all():
        if spec["family"] in SLIDING_FAMILIES:
            key = (spec["kinematics"]["track"], spec["leaf"]["count"], spec["hinge"]["side"])
            chosen.setdefault(key, spec)
        if spec["id"] == "db0079_sliding_single":
            chosen["reported"] = spec
        if spec["family"] == "sliding_bypass" and "floor_guide" in spec.get("extras", []):
            chosen["guide_" + spec["id"]] = spec
    results = {}
    for spec in chosen.values():
        result = export_door(spec, str(root / "doors"), str(root / "hardware"), formats=("mjcf", "json"))
        metadata = json.loads((root / "doors" / spec["id"] / "model.json").read_text())["meta"]
        model = mujoco.MjModel.from_xml_path(result["files"]["mjcf"]["full"])
        results[spec["id"]] = (model, metadata)
    return results


def test_track_coverage_and_existing_rollers_all_layouts(exported):
    assert len(exported) >= 20
    for name, (model, metadata) in exported.items():
        result = run_sliding_track_qa(model, metadata)
        assert result["ok"], (name, result)
        assert result["n_supports"] >= 1


def test_short_rail_fails_at_full_travel(exported):
    model, metadata = exported["db0079_sliding_single"]
    rail = model.geom("flat_track").id
    original = model.geom_size[rail, 0]
    try:
        model.geom_size[rail, 0] *= 0.55
        result = run_sliding_track_qa(model, metadata)
        assert not result["ok"]
        assert any(f["check"] == "rail_coverage" for f in result["failures"])
    finally:
        model.geom_size[rail, 0] = original


def test_suspended_wheel_fails_contact_gate(exported):
    model, metadata = exported["db0079_sliding_single"]
    wheel = model.geom("leaf_hanger_wheel_0").id
    original = model.geom_pos[wheel, 2]
    try:
        model.geom_pos[wheel, 2] += 0.020
        result = run_sliding_track_qa(model, metadata)
        assert not result["ok"]
        assert any(f["check"] == "wheel_contact" for f in result["failures"])
    finally:
        model.geom_pos[wheel, 2] = original


def test_lock_does_not_hide_short_unlocked_travel(exported):
    model, metadata = exported["db0079_sliding_single"]
    joint, rail = model.joint("leaf_slide").id, model.geom("flat_track").id
    original_range = model.jnt_range[joint].copy()
    original_size = model.geom_size[rail, 0]
    try:
        model.jnt_range[joint] = (0.0, 0.002)
        model.geom_size[rail, 0] *= 0.55
        assert not run_sliding_track_qa(model, metadata)["ok"]
    finally:
        model.jnt_range[joint] = original_range
        model.geom_size[rail, 0] = original_size


def test_missing_support_metadata_is_not_a_pass(exported):
    model, metadata = exported["db0079_sliding_single"]
    result = run_sliding_track_qa(model, {**metadata, "sliding_track_supports": []})
    assert not result["ok"]


def test_rail_only_scope_is_reported(exported):
    for model, metadata in exported.values():
        if metadata["family"] == "sliding_bypass":
            result = run_sliding_track_qa(model, metadata)
            assert result["rail_only_bodies"]
            assert result["wheels_checked"] == 0
            return
    pytest.fail("No bypass layout exported")


def test_stop_moved_off_wheel_plane_fails(exported):
    model, metadata = exported["db0079_sliding_single"]
    stop = model.geom("leaf_track_stop_r").id
    original = model.geom_pos[stop, 1]
    try:
        model.geom_pos[stop, 1] += 0.20
        result = run_sliding_track_qa(model, metadata)
        assert not result["ok"]
        assert any(f["check"] == "end_stop" for f in result["failures"])
    finally:
        model.geom_pos[stop, 1] = original


def test_requested_bypass_guides_cover_all_lanes(exported):
    covered = []
    for name, (model, metadata) in exported.items():
        if metadata["family"] != "sliding_bypass" or not any(s.get("floor_guides_required") for s in metadata["sliding_track_supports"]):
            continue
        covered.append(name)
        report = run_sliding_track_qa(model, metadata)
        assert report["ok"], (name, report)
        assert all(s["floor_guides"] for s in metadata["sliding_track_supports"])
        assert report["max_guide_gap_m"] <= 0.003
    assert len(covered) == 14
    _, metadata = exported["db0008_sliding_bypass"]
    middle = next(s for s in metadata["sliding_track_supports"] if s["body"] == "leaf_1")
    assert len(middle["floor_guides"]) == 2


def test_requested_guide_cannot_be_silently_removed(exported):
    model, original = exported["db0008_sliding_bypass"]
    metadata = deepcopy(original)
    metadata["sliding_track_supports"][0]["floor_guides"] = []
    report = run_sliding_track_qa(model, metadata)
    assert not report["ok"]
    assert any(f["check"] == "floor_guide_missing" for f in report["failures"])


def test_middle_leaf_losing_one_guide_station_fails(exported):
    model, metadata = exported["db0008_sliding_bypass"]
    middle = next(s for s in metadata["sliding_track_supports"] if s["body"] == "leaf_1")
    station = middle["floor_guides"][0]
    ids = [model.geom(name).id for name in station["jaws"] + station["feet"]]
    originals = model.geom_pos[ids].copy()
    try:
        model.geom_pos[ids, 0] += 10.0
        report = run_sliding_track_qa(model, metadata)
        assert not report["ok"]
        assert any(f["check"] == "floor_guide_engagement" and f["body"] == "leaf_1" for f in report["failures"])
    finally:
        model.geom_pos[ids] = originals


def test_floating_guide_foot_fails(exported):
    model, metadata = exported["db0020_sliding_bypass"]
    foot = model.geom(metadata["sliding_track_supports"][0]["floor_guides"][0]["feet"][0]).id
    original = model.geom_pos[foot, 2]
    try:
        model.geom_pos[foot, 2] += 0.020
        report = run_sliding_track_qa(model, metadata)
        assert not report["ok"]
        assert any(f["check"] == "floor_guide_mount" for f in report["failures"])
    finally:
        model.geom_pos[foot, 2] = original

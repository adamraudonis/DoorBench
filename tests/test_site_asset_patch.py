"""Corrections must preserve base provenance and never revive stale motion."""
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import site_asset_patch as patch
from test_site_assets import site  # Reuse the small, fully bound render fixture.


@pytest.fixture
def correction(site, tmp_path):
    base = site.assets.parent
    updated = tmp_path / "updated"
    for name in ("assets", "appearance"):
        shutil.copytree(base / name, updated / name)
    manifest = {"archive_sha256": "a" * 64,
                "dataset_manifest_sha256": patch.digest(base / "assets/manifest.json")}
    base_manifest = tmp_path / "base-release.json"
    base_manifest.write_text(json.dumps(manifest))
    model = updated / "assets/doors/door_one/model.json"
    value = json.loads(model.read_text())
    value["geometry_revision"] = "removed overhead wall"
    model.write_text(json.dumps(value))
    rendered = updated / "appearance/door_one/variant_000/render.json"
    value = json.loads(rendered.read_text())
    value["source_sha256"]["model.json"] = patch.digest(model)
    rendered.write_text(json.dumps(value))
    dataset = updated / "assets/manifest.json"
    value = json.loads(dataset.read_text())
    value["doors"][0]["reference_motion_available"] = False
    dataset.write_text(json.dumps(value))
    args = SimpleNamespace(base=base, updated=updated, base_manifest=base_manifest,
                           archive=tmp_path / "correction.tar.gz", manifest=tmp_path / "correction.json",
                           source_commit="b" * 40, release_url="https://example.test/release",
                           description="Fixture geometry correction")
    patch.pack(args)
    return args, manifest


def test_verified_correction_round_trip_retains_unmodified_files(correction):
    args, base_manifest = correction
    manifest = patch.read(args.manifest)
    before = (args.base / "assets/doors/door_one/door.xml").read_bytes()
    patch.apply_archive(args.archive, manifest, base_manifest, args.base)
    assert patch.digest(args.base / "assets/manifest.json") == manifest["dataset_manifest_sha256"]
    assert (args.base / "assets/doors/door_one/door.xml").read_bytes() == before
    assert patch.inventory(args.base).keys() == patch.inventory(args.updated).keys()
    for name in patch.inventory(args.updated):
        assert (args.base / name).read_bytes() == (args.updated / name).read_bytes()


def test_changed_base_rejected_before_any_output(correction):
    args, base_manifest = correction
    manifest = patch.read(args.manifest)
    original_manifest = (args.base / "assets/manifest.json").read_bytes()
    (args.base / "assets/doors/door_one/model.json").write_text("unexpected local change")
    with pytest.raises(ValueError, match="base file mismatch"):
        patch.apply_archive(args.archive, manifest, base_manifest, args.base)
    assert (args.base / "assets/manifest.json").read_bytes() == original_manifest


def test_corrupt_archive_rejected_before_any_output(correction):
    args, base_manifest = correction
    manifest = patch.read(args.manifest)
    before = (args.base / "assets/manifest.json").read_bytes()
    args.archive.write_bytes(args.archive.read_bytes()[:-1])
    with pytest.raises(ValueError, match="checksum or size"):
        patch.apply_archive(args.archive, manifest, base_manifest, args.base)
    assert (args.base / "assets/manifest.json").read_bytes() == before


@pytest.mark.parametrize("name", ["../outside", "assets/../../outside", "assets//data", "results/data", "assets\\data"])
def test_unsafe_manifest_paths_rejected(correction, name):
    args, base_manifest = correction
    manifest = patch.read(args.manifest)
    manifest["files"][0]["path"] = name
    with pytest.raises(ValueError, match="Unsafe correction path"):
        patch.apply_archive(args.archive, manifest, base_manifest, args.base)


def test_geometry_change_cannot_keep_old_motion_available(correction):
    args, _ = correction
    path = args.updated / "assets/manifest.json"
    value = patch.read(path)
    value["doors"][0]["reference_motion_available"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="disable historical motion"):
        patch.pack(args)

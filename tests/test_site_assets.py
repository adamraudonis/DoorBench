"""A release must reject unsafe archives before writing and verify render inputs."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


_spec = importlib.util.spec_from_file_location("site_assets_under_test", Path(__file__).parents[1] / "scripts/site_assets.py")
site_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(site_assets)


def _json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def site(tmp_path):
    assets, appearance = tmp_path / "assets", tmp_path / "appearance"
    door = assets / "doors" / "door_one"
    hardware = assets / "hardware"
    render = appearance / "door_one" / "variant_000"
    for directory in (door, hardware, render):
        directory.mkdir(parents=True)
    for name in site_assets.DOOR_FILES:
        (door / name).write_bytes(b"tiny generated file")
    _json(door / "spec.json", {"id": "door_one"})
    _json(door / "model.json", {"name": "door_one", "bodies": [{"name": "leaf", "geoms": [{"name": "grip", "type": "mesh", "mesh_name": "handle"}]}]})
    (door / "thumb_iso.jpg").write_bytes(b"thumbnail fixture")
    (hardware / "handle.obj").write_bytes(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (render / "rgb.png").write_bytes(b"render fixture")
    (render / "scene.blend").write_bytes(b"packed scene fixture")
    dataset = {"n_doors": 1, "doors": [{"id": "door_one"}]}
    _json(assets / "manifest.json", dataset)
    hashes = {name: site_assets.digest(door / name) for name in ("spec.json", "model.json", "door.xml")}
    metadata = {"door_id": "door_one", "variant": 0, "rendered": True, "quality": "preview", "recipe": {},
                "source_sha256": hashes, "mesh_sha256": {"handle": site_assets.digest(hardware / "handle.obj")},
                "artifact_sha256": {name: site_assets.digest(render / name) for name in ("rgb.png", "scene.blend")}}
    _json(render / "render.json", metadata)
    row = {"door_id": "door_one", "variant": 0, "quality": "preview", "recipe": {}, "source_sha256": dict(hashes),
           "metadata": "door_one/variant_000/render.json", "image": "door_one/variant_000/rgb.png",
           "blend": "door_one/variant_000/scene.blend"}
    index = {"schema_version": 1, "failed": [], "renders": [row]}
    _json(appearance / "index.json", index)
    return SimpleNamespace(assets=assets, appearance=appearance, door=door, hardware=hardware,
                           render=render, dataset=dataset, metadata=metadata, row=row, index=index)


def _archive(tmp_path, entries):
    path = tmp_path / "release.tar.gz"
    expanded = 0
    with tarfile.open(path, "w:gz") as archive:
        for name, payload, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(payload) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "../../outside"
            archive.addfile(member, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)
            expanded += member.size
    manifest = {"archive_bytes": path.stat().st_size, "archive_sha256": site_assets.digest(path),
                "expanded_bytes": expanded, "file_count": len(entries)}
    return path, manifest


def test_tiny_verified_site_pack_restore_round_trip(site, tmp_path):
    args = SimpleNamespace(assets=site.assets, appearance=site.appearance, archive=tmp_path / "release.tar.gz",
                           manifest=tmp_path / "release.json", release_url="https://example.test/release", source_commit="a" * 40)
    original_files, original_info = site_assets.verified_files(site.assets, site.appearance)
    site_assets.pack(args)
    out = tmp_path / "restored"
    manifest = site_assets.read(args.manifest)
    site_assets.unpack_verified(args.archive, manifest, out)
    restored_files, restored_info = site_assets.verified_files(out / "assets", out / "appearance")
    assert original_info == restored_info
    assert set(original_files) == set(restored_files)
    assert all(original_files[name].read_bytes() == restored_files[name].read_bytes() for name in original_files)


@pytest.mark.parametrize("damage", ["truncated", "same_size_corruption"])
def test_corrupt_archive_rejected_before_any_output(tmp_path, damage):
    archive, manifest = _archive(tmp_path, [("assets/data", b"good", tarfile.REGTYPE)])
    content = bytearray(archive.read_bytes())
    if damage == "truncated":
        content = content[:-1]
    else:
        content[len(content) // 2] ^= 1
    archive.write_bytes(content)
    out = tmp_path / "restored"
    with pytest.raises(ValueError, match="checksum or size"):
        site_assets.unpack_verified(archive, manifest, out)
    assert not out.exists()


@pytest.mark.parametrize("bad_name", ["../outside", "assets/../../outside", "assets\\outside", "other/data", "assets"])
def test_traversal_or_wrong_root_rejected_before_valid_member_is_written(tmp_path, bad_name):
    archive, manifest = _archive(tmp_path, [("appearance/good", b"good", tarfile.REGTYPE), (bad_name, b"bad", tarfile.REGTYPE)])
    out = tmp_path / "restored"
    with pytest.raises(ValueError, match="Unsafe release archive entry"):
        site_assets.unpack_verified(archive, manifest, out)
    assert not out.exists()
    assert not (tmp_path / "outside").exists()


def test_absolute_archive_path_rejected_before_write(tmp_path):
    absolute = tmp_path / "outside"
    archive, manifest = _archive(tmp_path, [(str(absolute), b"bad", tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="Unsafe release archive entry"):
        site_assets.unpack_verified(archive, manifest, tmp_path / "restored")
    assert not absolute.exists()
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_archive_links_and_special_files_rejected_before_write(tmp_path, kind):
    archive, manifest = _archive(tmp_path, [("assets/good", b"good", tarfile.REGTYPE), ("assets/unsafe", b"", kind)])
    out = tmp_path / "restored"
    with pytest.raises(ValueError, match="Unsafe release archive entry"):
        site_assets.unpack_verified(archive, manifest, out)
    assert not out.exists()


@pytest.mark.parametrize("alias", ["assets/data", "assets/./data", "assets//data"])
def test_duplicate_archive_targets_rejected_before_write(tmp_path, monkeypatch, alias):
    archive, manifest = _archive(tmp_path, [("assets/data", b"first", tarfile.REGTYPE), (alias, b"last", tarfile.REGTYPE)])
    # Archive-only fixture: bypass dataset semantics, never path validation.
    monkeypatch.setattr(site_assets, "verified_files", lambda *_: ({}, {"doors": 0, "renders": 0}))
    manifest.update(doors=0, renders=0)
    out = tmp_path / "restored"
    with pytest.raises(ValueError, match="Unsafe release archive entry"):
        site_assets.unpack_verified(archive, manifest, out)
    assert not out.exists()


def test_preexisting_output_symlink_cannot_escape_before_other_file_is_written(tmp_path):
    archive, manifest = _archive(tmp_path, [("appearance/good", b"good", tarfile.REGTYPE), ("assets/data", b"bad", tarfile.REGTYPE)])
    outside, out = tmp_path / "outside", tmp_path / "restored"
    outside.mkdir()
    out.mkdir()
    (out / "assets").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe output path"):
        site_assets.unpack_verified(archive, manifest, out)
    assert list(outside.iterdir()) == []
    assert not (out / "appearance").exists()


@pytest.mark.parametrize("field", ["file_count", "expanded_bytes"])
def test_archive_inventory_mismatch_rejected_before_output(tmp_path, field):
    archive, manifest = _archive(tmp_path, [("assets/data", b"good", tarfile.REGTYPE)])
    manifest[field] += 1
    out = tmp_path / "restored"
    with pytest.raises(ValueError, match="inventory mismatch"):
        site_assets.unpack_verified(archive, manifest, out)
    assert not out.exists()


@pytest.mark.parametrize("file_kind", ["source", "mesh", "image", "blend"])
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_missing_or_changed_render_inputs_and_outputs_rejected(site, file_kind, damage):
    path = {"source": site.door / "model.json", "mesh": site.hardware / "handle.obj",
            "image": site.render / "rgb.png", "blend": site.render / "scene.blend"}[file_kind]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"\n ")
    with pytest.raises((ValueError, FileNotFoundError), match="Missing|differs|checksum"):
        site_assets.verified_files(site.assets, site.appearance)


@pytest.mark.parametrize("field", ["source_sha256", "mesh_sha256"])
def test_missing_render_input_digest_inventory_rejected(site, field):
    site.metadata[field] = {}
    _json(site.render / "render.json", site.metadata)
    with pytest.raises(ValueError):
        site_assets.verified_files(site.assets, site.appearance)


def test_incomplete_source_digest_inventory_rejected(site):
    del site.metadata["source_sha256"]["door.xml"]
    _json(site.render / "render.json", site.metadata)
    with pytest.raises(ValueError):
        site_assets.verified_files(site.assets, site.appearance)


def test_duplicate_dataset_rows_cannot_hide_behind_unique_count(site):
    site.dataset["doors"].append(dict(site.dataset["doors"][0]))
    _json(site.assets / "manifest.json", site.dataset)
    with pytest.raises(ValueError, match="unique|duplicate"):
        site_assets.verified_files(site.assets, site.appearance)


@pytest.mark.parametrize("damage", ["no_default", "duplicate_default", "wrong_metadata_door", "not_rendered", "missing_image_field", "wrong_recipe", "wrong_quality"])
def test_default_coverage_and_index_metadata_contract_rejected(site, damage):
    if damage == "no_default":
        site.index["renders"] = []
    elif damage == "duplicate_default":
        site.index["renders"].append(dict(site.row))
    elif damage == "wrong_metadata_door":
        site.metadata["door_id"] = "another_door"
    elif damage == "not_rendered":
        site.metadata["rendered"] = False
    elif damage == "missing_image_field":
        site.row["image"] = None
    elif damage == "wrong_recipe":
        site.row["recipe"] = {"lighting": "another_lighting"}
    else:
        site.row["quality"] = "photo"
    _json(site.appearance / "index.json", site.index)
    _json(site.render / "render.json", site.metadata)
    with pytest.raises(ValueError):
        site_assets.verified_files(site.assets, site.appearance)

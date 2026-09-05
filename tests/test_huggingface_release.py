"""Deterministic release bytes, complete provenance and safe verified extraction."""
from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
import sys
import tarfile
from types import ModuleType, SimpleNamespace

import pytest

from scripts import huggingface_release as release


def _files(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (root / "data.json").write_bytes(b'{"door":"tiny"}\n')
    (root / "mesh.obj").write_bytes(b"v 0 0 0\n")
    files = release.component_files(root, "assets")
    return files, release.inventory_for(files, "assets")


def test_archives_are_reproducible_and_restore_exact_bytes(tmp_path):
    files, records = _files(tmp_path)
    first = release.write_archive(files, records, tmp_path / "first.tar.gz")
    second = release.write_archive(dict(reversed(list(files.items()))), records, tmp_path / "second.tar.gz")
    assert first["sha256"] == second["sha256"]
    with tarfile.open(tmp_path / "first.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            assert member.mtime == member.uid == member.gid == 0
            assert member.mode == 0o644
    out = tmp_path / "restored"
    release.extract_component(tmp_path / "first.tar.gz", first, records, out)
    assert all((out / name).read_bytes() == path.read_bytes() for name, path in files.items())


def test_source_mutation_during_preparation_does_not_publish_archive(tmp_path):
    files, records = _files(tmp_path)
    next(iter(files.values())).write_bytes(b"changed input")
    destination = tmp_path / "bad.tar.gz"
    with pytest.raises((ValueError, OSError), match="changed|unexpected end"):
        release.write_archive(files, records, destination)
    assert not destination.exists()
    assert not destination.with_suffix(".gz.partial").exists()


@pytest.mark.parametrize("path", ["../escape", "assets/../../escape", "/absolute", "assets\\escape", "assets/./data", "assets//data", "assets/\x00file"])
def test_unsafe_names_rejected(path):
    with pytest.raises(ValueError, match="Unsafe release path"):
        release.safe_name(path)


def _malformed_archive(tmp_path, entries):
    path = tmp_path / "malformed.tar.gz"
    records = {}
    with tarfile.open(path, "w:gz") as archive:
        for name, kind in entries:
            member = tarfile.TarInfo(name)
            member.type, member.size = kind, 1 if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "../../escape"
            archive.addfile(member, io.BytesIO(b"x") if kind == tarfile.REGTYPE else None)
            records[name] = {"bytes": member.size, "sha256": hashlib.sha256(b"x" if member.size else b"").hexdigest()}
    component = {"bytes": path.stat().st_size, "sha256": release.sha256(path),
                 "expanded_bytes": sum(r["bytes"] for r in records.values()), "file_count": len(entries)}
    return path, component, records


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_links_and_special_archive_members_rejected_before_any_write(tmp_path, kind):
    path, component, records = _malformed_archive(tmp_path, [("assets/good", tarfile.REGTYPE), ("assets/unsafe", kind)])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Unsafe or unexpected"):
        release.extract_component(path, component, records, out)
    assert not out.exists()


def test_file_directory_collision_rejected_before_any_write(tmp_path):
    path, component, records = _malformed_archive(tmp_path, [("assets/data", tarfile.REGTYPE), ("assets/data/child", tarfile.REGTYPE)])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="file/directory collision"):
        release.extract_component(path, component, records, out)
    assert not out.exists()


@pytest.mark.parametrize("damage", ["archive_bytes", "file_inventory", "file_digest"])
def test_corrupt_payload_or_inventory_rejected(tmp_path, damage):
    files, records = _files(tmp_path)
    path = tmp_path / "archive.tar.gz"
    component = release.write_archive(files, records, path)
    if damage == "archive_bytes":
        path.write_bytes(path.read_bytes() + b"wrong")
    elif damage == "file_inventory":
        records.pop(next(iter(records)))
    else:
        next(iter(records.values()))["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum|unexpected|inventory"):
        release.extract_component(path, component, records, tmp_path / "out")


def test_missing_usdc_reference_cannot_enter_full_assets(tmp_path, monkeypatch):
    assets, appearance = tmp_path / "assets", tmp_path / "appearance"
    assets.mkdir()
    appearance.mkdir()
    (assets / "door.usda").write_text('#usda 1.0\ndef Xform "door" (references = @hardware/missing.usdc@) {}')
    monkeypatch.setattr("scripts.site_assets.verified_files", lambda *_: ({}, {"doors": 1}))
    with pytest.raises(ValueError, match="Missing or external USD reference"):
        release._base_files(assets, appearance, 1)


@pytest.fixture
def motion(tmp_path):
    assets, motions = tmp_path / "assets", tmp_path / "motions"
    door = assets / "doors" / "tiny_door"
    door.mkdir(parents=True)
    (motions / "clips").mkdir(parents=True)
    (motions / "trajectories").mkdir()
    release.write_json(assets / "manifest.json", {"doors": [{"id": "tiny_door"}]})
    for name in ("spec.json", "model.json", "door.xml"):
        (door / name).write_bytes(b"source bytes")
    (motions / "clips/tiny_door.json").write_bytes(b'{"success":false}')
    (motions / "trajectories/tiny_door.npz").write_bytes(b"opaque native fixture")
    row = {"door_id": "tiny_door", "clip": "clips/tiny_door.json", "trajectory": "trajectories/tiny_door.npz",
           "clip_sha256": release.sha256(motions / "clips/tiny_door.json"),
           "trajectory_sha256": release.sha256(motions / "trajectories/tiny_door.npz"),
           "source_sha256": {name: release.sha256(door / name) for name in ("spec.json", "model.json", "door.xml")},
           "success": False, "outcome": "fail"}
    index = {"schema": "doorbench.reference-motion.v1", "manifest_sha256": release.sha256(assets / "manifest.json"), "clips": [row],
             "policy": "scripted_hand", "tier": "full", "seed": 0, "counts": {"success": 0, "fail": 1, "damaged": 0}}
    release.write_json(motions / "index.json", index)
    return assets, motions, index


def test_motion_collection_preserves_failure_and_includes_gzip_derivative(motion):
    assets, motions, index = motion
    (motions / "clips/tiny_door.json.gz").write_bytes(gzip.compress((motions / "clips/tiny_door.json").read_bytes(), mtime=0))
    index["clips"][0].update(web_clip="clips/tiny_door.json.gz", web_clip_sha256=release.sha256(motions / "clips/tiny_door.json.gz"))
    release.write_json(motions / "index.json", index)
    files, checked = release._motion_files(motions, {"tiny_door"}, assets)
    assert checked["clips"][0]["success"] is False
    assert "reference-motions/clips/tiny_door.json.gz" in files
    assert len(files) == 4


@pytest.mark.parametrize("damage", ["source", "native", "manifest", "missing_clip", "duplicate_id", "scope", "outcomes"])
def test_motion_mismatch_cannot_enter_release(motion, damage):
    assets, motions, index = motion
    if damage == "source":
        (assets / "doors/tiny_door/door.xml").write_bytes(b"new source")
    elif damage == "native":
        (motions / "trajectories/tiny_door.npz").write_bytes(b"damaged")
    elif damage == "manifest":
        index["manifest_sha256"] = "0" * 64
    elif damage == "missing_clip":
        (motions / "clips/tiny_door.json").unlink()
    elif damage == "duplicate_id":
        index["clips"].append(dict(index["clips"][0]))
    elif damage == "scope":
        index["policy"] = "unverified_policy"
    else:
        index["counts"]["success"] = 1
    release.write_json(motions / "index.json", index)
    with pytest.raises(ValueError):
        release._motion_files(motions, {"tiny_door"}, assets)


def test_compressed_clip_must_equal_original_even_with_matching_compressed_hash(motion):
    assets, motions, index = motion
    path = motions / "clips/tiny_door.json.gz"
    path.write_bytes(gzip.compress(b"wrong", mtime=0))
    index["clips"][0].update(web_clip="clips/tiny_door.json.gz", web_clip_sha256=release.sha256(path))
    release.write_json(motions / "index.json", index)
    with pytest.raises(ValueError, match="differs from original JSON"):
        release._motion_files(motions, {"tiny_door"}, assets)


def test_publication_requires_completed_motion_before_credentials_or_network(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "release_files", lambda _: ({"reference_motion": {"included": False}}, {}))
    with pytest.raises(ValueError, match="completed reference-motion"):
        release.publish(SimpleNamespace(folder=tmp_path, token_file=None))


@pytest.mark.parametrize("damage", [None, "private", "missing", "size", "lfs_hash", "git_hash"])
def test_publication_verifies_anonymous_remote_payload_hashes(tmp_path, damage):
    small, large = tmp_path / "README.md", tmp_path / "archive.tar.gz"
    small.write_bytes(b"dataset card")
    large.write_bytes(b"archive bytes")
    entries = [SimpleNamespace(path="README.md", size=small.stat().st_size, lfs=None,
                               blob_id=hashlib.sha1(b"blob 12\0dataset card").hexdigest()),
               SimpleNamespace(path="archive.tar.gz", size=large.stat().st_size, blob_id=None,
                               lfs=SimpleNamespace(sha256=release.sha256(large)))]
    if damage == "missing":
        entries.pop()
    elif damage == "size":
        entries[-1].size += 1
    elif damage == "lfs_hash":
        entries[-1].lfs.sha256 = "0" * 64
    elif damage == "git_hash":
        entries[0].blob_id = "0" * 40
    api = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(private=damage == "private", gated=False),
                          get_paths_info=lambda *a, **k: entries)
    if damage:
        with pytest.raises(ValueError):
            release.verify_public_files(api, "test/door", "f" * 40, {"README.md": small, "archive.tar.gz": large})
    else:
        release.verify_public_files(api, "test/door", "f" * 40, {"README.md": small, "archive.tar.gz": large})


def _download_fixture(tmp_path, monkeypatch):
    files, records = _files(tmp_path)
    hub = tmp_path / "hub"
    (hub / "archives").mkdir(parents=True)
    component = release.write_archive(files, records, hub / "archives/assets.tar.gz")
    release.write_json(hub / "inventory.json", {"schema_version": 1, "files": records})
    for name in ("LICENSE", "THIRD_PARTY.md"):
        (hub / name).write_bytes(b"license fixture")
    manifest = {"schema_version": 1, "release": "v-test", "repo_id": "test/door", "components": {"assets": component},
                "inventory_sha256": release.sha256(hub / "inventory.json"),
                "support_files": {n: {"sha256": release.sha256(hub / n), "bytes": (hub / n).stat().st_size}
                                  for n in ("LICENSE", "THIRD_PARTY.md")}}
    release.write_json(hub / "release.json", manifest)
    fake = ModuleType("huggingface_hub")
    fake.HfApi = lambda **kwargs: SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(sha="f" * 40))
    fake.hf_hub_download = lambda repo, filename, **kwargs: str(hub / filename)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return hub, SimpleNamespace(repo_id="test/door", revision="v-test", components="all", out=tmp_path / "result")


def test_verified_download_installs_only_completed_directory(tmp_path, monkeypatch):
    hub, args = _download_fixture(tmp_path, monkeypatch)
    release.download(args)
    assert (args.out / "assets/data.json").read_bytes() == b'{"door":"tiny"}\n'
    assert (args.out / "LICENSE").read_bytes() == b"license fixture"
    assert not list(tmp_path.glob(".doorbench-download-*"))


def test_subset_download_preserves_release_inventory_hash(tmp_path, monkeypatch):
    hub, args = _download_fixture(tmp_path, monkeypatch)
    inventory = release.read(hub / "inventory.json")
    inventory['files']['appearance/not-installed.png'] = {'component': 'appearance', 'bytes': 0, 'sha256': '0'*64}
    release.write_json(hub / 'inventory.json', inventory)
    manifest = release.read(hub / 'release.json')
    manifest['inventory_sha256'] = release.sha256(hub / 'inventory.json')
    manifest['components']['appearance'] = {'path': 'must-not-fetch.tar.gz'}
    release.write_json(hub / 'release.json', manifest)
    args.components = 'assets'
    release.download(args)
    assert release.sha256(args.out / 'inventory.json') == manifest['inventory_sha256']
    assert release.read(args.out / 'installed.json')['components'] == ['assets']
    assert not (args.out / 'appearance').exists()


@pytest.mark.parametrize("damage", ["archive", "license", "moving_branch"])
def test_failed_download_leaves_no_partial_destination(tmp_path, monkeypatch, damage):
    hub, args = _download_fixture(tmp_path, monkeypatch)
    if damage == "moving_branch":
        args.revision = "main"
    else:
        target = hub / ("archives/assets.tar.gz" if damage == "archive" else "LICENSE")
        target.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        release.download(args)
    assert not args.out.exists()
    assert not list(tmp_path.glob(".doorbench-download-*"))

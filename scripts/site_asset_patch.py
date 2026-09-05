#!/usr/bin/env python3
"""Publish small, verified corrections over an immutable generated site release.

The base release remains reproducible. Each changed file records both its old
and new checksum; corrections cannot silently apply to a different dataset.
Historical motion clips are retained with their original source bindings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.request

from site_assets import digest, read, verified_files

SCHEMA = "doorbench.site-correction.v1"
ROOTS = {"assets", "appearance"}
HASH = re.compile(r"[a-f0-9]{64}")


def safe_path(root: Path, name: str) -> Path:
    rel = PurePosixPath(name)
    if (not isinstance(name, str) or rel.is_absolute() or ".." in rel.parts
            or "\\" in name or name != rel.as_posix()
            or len(rel.parts) < 2 or rel.parts[0] not in ROOTS):
        raise ValueError(f"Unsafe correction path: {name!r}")
    target = root / name
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe correction output: {name}")
    return target


def validate_manifest(value: dict) -> dict:
    if value.get("schema") != SCHEMA or not value.get("files"):
        raise ValueError("Unsupported or empty correction manifest")
    for name in ("archive_sha256", "base_archive_sha256", "base_dataset_manifest_sha256",
                 "dataset_manifest_sha256", "appearance_index_sha256"):
        if not isinstance(value.get(name), str) or not HASH.fullmatch(value[name]):
            raise ValueError(f"Invalid {name}")
    if not re.fullmatch(r"[a-f0-9]{40}", value.get("source_commit", "")):
        raise ValueError("Correction requires a source commit")
    if type(value.get("archive_bytes")) is not int or value["archive_bytes"] <= 0:
        raise ValueError("Invalid archive byte size")
    seen = set()
    for row in value["files"]:
        name = row.get("path")
        if not isinstance(name, str):
            raise ValueError("Missing correction path")
        safe_path(Path("."), name)
        if name in seen:
            raise ValueError(f"Duplicate correction path: {name}")
        seen.add(name)
        if (not isinstance(row.get("sha256"), str) or not HASH.fullmatch(row["sha256"])
                or (row.get("before_sha256") is not None and
                    (not isinstance(row["before_sha256"], str) or not HASH.fullmatch(row["before_sha256"])))
                or type(row.get("bytes")) is not int or row["bytes"] < 0):
            raise ValueError(f"Invalid correction descriptor: {name}")
    return value


def inventory(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for parent in ROOTS
            for p in sorted((root / parent).rglob("*")) if p.is_file()}


def verify_current(root: Path, manifest: dict):
    # Appearance is checked against the corrected sources. Old motion is not
    # re-certified: the viewer disables clips for geometry-revised doors.
    _, info = verified_files(root / "assets", root / "appearance")
    for name in ("dataset_manifest_sha256", "appearance_index_sha256"):
        if info[name] != manifest[name]:
            raise ValueError(f"Corrected site differs from {name}")
    rows = read(root / "assets/manifest.json")["doors"]
    changed = {row["path"].split("/")[2] for row in manifest["files"]
               if row["path"].startswith("assets/doors/")
               and row["path"].endswith(("/spec.json", "/model.json", "/door.xml"))}
    for door in rows:
        if door["id"] in changed and door.get("reference_motion_available") is not False:
            raise ValueError(f"Revised source must disable historical motion: {door['id']}")
    return info


def pack(args):
    base_info = read(args.base_manifest)
    if digest(args.base / "assets/manifest.json") != base_info["dataset_manifest_sha256"]:
        raise ValueError("Base dataset differs from its release manifest")
    old, new = inventory(args.base), inventory(args.updated)
    if old.keys() - new.keys():
        raise ValueError("A correction cannot delete base files")
    files = []
    for name, path in sorted(new.items()):
        safe_path(args.updated, name)
        after = digest(path)
        before = digest(old[name]) if name in old else None
        if before != after:
            files.append({"path": name, "before_sha256": before,
                          "sha256": after, "bytes": path.stat().st_size})
    if not files:
        raise ValueError("No changed files to publish")
    metadata = {
        "schema": SCHEMA, "source_commit": args.source_commit,
        "description": args.description,
        "base_archive_sha256": base_info["archive_sha256"],
        "base_dataset_manifest_sha256": base_info["dataset_manifest_sha256"],
        "dataset_manifest_sha256": digest(args.updated / "assets/manifest.json"),
        "appearance_index_sha256": digest(args.updated / "appearance/index.json"),
        "files": files,
    }
    verify_current(args.updated, metadata)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz", compresslevel=1) as archive:
        for row in files:
            member = tarfile.TarInfo(row["path"])
            member.size, member.mode, member.mtime = row["bytes"], 0o644, 0
            with new[row["path"]].open("rb") as stream:
                archive.addfile(member, stream)
    metadata.update(download_url=args.release_url.rstrip("/") + "/" + args.archive.name,
                    archive_sha256=digest(args.archive), archive_bytes=args.archive.stat().st_size)
    validate_manifest(metadata)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metadata.items() if key != "files"}, indent=2))
    print(f"{len(files)} corrected files")


def apply_archive(archive_path: Path, manifest: dict, base_manifest: dict, out: Path):
    validate_manifest(manifest)
    if (base_manifest["archive_sha256"] != manifest["base_archive_sha256"]
            or base_manifest["dataset_manifest_sha256"] != manifest["base_dataset_manifest_sha256"]
            or digest(out / "assets/manifest.json") != manifest["base_dataset_manifest_sha256"]):
        raise ValueError("Correction requires its exact unmodified base release")
    if archive_path.stat().st_size != manifest["archive_bytes"] or digest(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Correction archive checksum or size mismatch")
    rows = {row["path"]: row for row in manifest["files"]}
    # Verify every old file before writing any correction.
    for name, row in rows.items():
        path = safe_path(out, name)
        if row["before_sha256"] is None:
            if path.exists():
                raise ValueError(f"Correction expected a new file: {name}")
        elif not path.is_file() or digest(path) != row["before_sha256"]:
            raise ValueError(f"Correction base file mismatch: {name}")
    with tempfile.TemporaryDirectory(prefix="doorbench-correction-") as temp:
        stage = Path(temp)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != len(rows) or {m.name for m in members} != set(rows):
                raise ValueError("Correction archive inventory mismatch")
            for member in members:
                row = rows[member.name]
                if not member.isfile() or member.size != row["bytes"]:
                    raise ValueError(f"Invalid correction member: {member.name}")
                path = safe_path(stage, member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if digest(path) != row["sha256"]:
                    raise ValueError(f"Correction payload mismatch: {member.name}")
        for name in rows:
            target = safe_path(out, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Copy to the destination filesystem before the atomic replacement.
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as pending:
                pending.write((stage / name).read_bytes())
                pending_name = pending.name
            os.replace(pending_name, target)
    info = verify_current(out, manifest)
    print(f"Applied {len(rows)} verified corrections; {info['doors']} downloadable doors")


def restore(args):
    manifest = read(args.manifest)
    base = read(args.base_manifest)
    if args.archive:
        apply_archive(args.archive, manifest, base, args.out)
        return
    url = manifest.get("download_url", "")
    if not url.startswith("https://github.com/adamraudonis/DoorBench/releases/download/"):
        raise ValueError("Expected a DoorBench GitHub release URL")
    with tempfile.TemporaryDirectory(prefix="doorbench-correction-download-") as temp:
        path = Path(temp) / "correction.tar.gz"
        with urllib.request.urlopen(url, timeout=60) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
        apply_archive(path, manifest, base, args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("pack")
    for name in ("base", "updated", "archive", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--base-manifest", type=Path, default=Path("deploy/site-assets.json"))
    p.add_argument("--release-url", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--description", required=True)
    p.set_defaults(func=pack)
    p = commands.add_parser("restore")
    p.add_argument("--manifest", type=Path, default=Path("deploy/collection-update.json"))
    p.add_argument("--base-manifest", type=Path, default=Path("deploy/site-assets.json"))
    p.add_argument("--archive", type=Path)
    p.add_argument("--out", type=Path, default=Path("site"))
    p.set_defaults(func=restore)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Package or restore the exact generated dataset and Blender renders used by Pages.

Generated media live in a versioned GitHub release, not in the source checkout.
Only the small deployment manifest is committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath


DOOR_FILES = {
    "model.json", "spec.json", "qa.json", "door.xml", "door_simple.xml",
    "door_minimal.xml", "scene.xml", "door.urdf", "door_simple.urdf",
    "door_minimal.urdf", "door.usda", "door_rl.usda",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def relative_file(root: Path, name: str) -> Path:
    rel = PurePosixPath(name)
    if not name or rel.is_absolute() or ".." in rel.parts or "\\" in name:
        raise ValueError(f"Unsafe relative asset path: {name!r}")
    p = root / name
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Missing or non-regular asset: {p}")
    return p


def verified_files(assets: Path, appearance: Path) -> tuple[dict[str, Path], dict]:
    dataset = read(assets / "manifest.json")
    index = read(appearance / "index.json")
    doors = {r["id"] for r in dataset["doors"]}
    if (not doors or len(dataset["doors"]) != len(doors) or len(doors) != dataset["n_doors"]
            or index.get("schema_version") != 1 or index.get("failed")):
        raise ValueError("Dataset IDs must be unique and render failures resolved")
    previews = [r for r in index["renders"] if r["variant"] == 0]
    if len(previews) != len(doors) or {r["door_id"] for r in previews} != doors:
        raise ValueError("Every dataset door must have exactly one default Blender preview")
    files = {"assets/manifest.json": assets / "manifest.json", "appearance/index.json": appearance / "index.json"}
    for door in sorted(doors):
        directory = assets / "doors" / door
        for name in sorted(DOOR_FILES):
            files[f"assets/doors/{door}/{name}"] = relative_file(directory, name)
        thumbs = sorted(directory.glob("thumb_*.jpg"))
        if not thumbs:
            raise ValueError(f"No simulation thumbnails for {door}")
        for p in thumbs:
            files[f"assets/doors/{door}/{p.name}"] = relative_file(directory, p.name)
    for p in sorted((assets / "hardware").glob("*.obj")):
        files[f"assets/hardware/{p.name}"] = relative_file(assets / "hardware", p.name)
    seen = set()
    for row in index["renders"]:
        key = (row["door_id"], row["variant"])
        if key in seen or key[0] not in doors:
            raise ValueError(f"Invalid or duplicate render: {key}")
        seen.add(key)
        meta_path = relative_file(appearance, row["metadata"])
        meta = read(meta_path)
        if not meta.get("rendered") or (meta["door_id"], meta["variant"]) != key:
            raise ValueError(f"Render metadata does not match index: {key}")
        if set(meta["source_sha256"]) != {"spec.json", "model.json", "door.xml"}:
            raise ValueError(f"Incomplete source checksums: {key}")
        model = read(assets / "doors" / key[0] / "model.json")
        meshes = {g["mesh_name"] for b in model["bodies"] for g in b["geoms"] if g["type"] == "mesh"}
        if set(meta["mesh_sha256"]) != meshes:
            raise ValueError(f"Incomplete hardware checksums: {key}")
        if row["recipe"] != meta["recipe"] or row["quality"] != meta["quality"]:
            raise ValueError(f"Appearance recipe differs from metadata: {key}")
        for name, expected in meta["source_sha256"].items():
            if digest(relative_file(assets / "doors" / key[0], name)) != expected:
                raise ValueError(f"Source differs from rendered image: {key[0]}/{name}")
        for name, expected in meta["mesh_sha256"].items():
            if digest(relative_file(assets / "hardware", name + ".obj")) != expected:
                raise ValueError(f"Hardware differs from rendered image: {name}")
        files[f"appearance/{row['metadata']}"] = meta_path
        for field, artifact in (("image", "rgb.png"), ("blend", "scene.blend")):
            if not row.get(field):
                if field == "image":
                    raise ValueError(f"Missing image for {key}")
                continue
            p = relative_file(appearance, row[field])
            if digest(p) != meta["artifact_sha256"][artifact]:
                raise ValueError(f"Artifact checksum mismatch: {p}")
            files[f"appearance/{row[field]}"] = p
    info = {
        "doors": len(doors), "renders": len(seen), "photo_renders": sum(r["quality"] == "photo" for r in index["renders"]),
        "dataset_manifest_sha256": digest(assets / "manifest.json"),
        "appearance_index_sha256": digest(appearance / "index.json"),
    }
    return files, info


def pack(args):
    files, info = verified_files(args.assets, args.appearance)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz", compresslevel=1) as archive:
        for name, path in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = path.stat().st_size
            member.mode = 0o644
            member.mtime = 0
            with path.open("rb") as f:
                archive.addfile(member, f)
    manifest = {
        "schema_version": 1,
        "download_url": args.release_url.rstrip("/") + "/" + args.archive.name,
        "archive_sha256": digest(args.archive), "archive_bytes": args.archive.stat().st_size,
        "expanded_bytes": sum(p.stat().st_size for p in files.values()), "file_count": len(files),
        "source_commit": args.source_commit, **info,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def unpack_verified(archive_path: Path, manifest: dict, out: Path):
    if archive_path.stat().st_size != manifest["archive_bytes"] or digest(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Release archive checksum or size mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        seen = set()
        for m in members:
            rel = PurePosixPath(m.name)
            if (not m.isfile() or rel.is_absolute() or ".." in rel.parts or "\\" in m.name
                    or m.name != rel.as_posix()
                    or len(rel.parts) < 2 or rel.parts[0] not in ("assets", "appearance")
                    or m.name in seen):
                raise ValueError(f"Unsafe release archive entry: {m.name}")
            seen.add(m.name)
        if len(members) != manifest["file_count"] or sum(m.size for m in members) != manifest["expanded_bytes"]:
            raise ValueError("Release archive inventory mismatch")
        for m in members:
            target = out / m.name
            if not target.resolve().is_relative_to(out.resolve()) or target.is_symlink():
                raise ValueError(f"Unsafe output path: {target}")
        for m in members:
            target = out / m.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(m) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    _, info = verified_files(out / "assets", out / "appearance")
    for key, value in info.items():
        if value != manifest[key]:
            raise ValueError(f"Restored site inventory mismatch: {key}")
    print(f"Verified {info['doors']} doors and {info['renders']} Blender images")


def restore(args):
    manifest = read(args.manifest)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported deployment manifest")
    if args.archive:
        unpack_verified(args.archive, manifest, args.out)
        return
    url = manifest["download_url"]
    if not url.startswith("https://github.com/adamraudonis/DoorBench/releases/download/"):
        raise ValueError("Expected a DoorBench GitHub release URL")
    with tempfile.TemporaryDirectory(prefix="doorbench-site-") as tmp:
        path = Path(tmp) / "assets.tar.gz"
        request = urllib.request.Request(url, headers={"User-Agent": "DoorBench-site-deploy/1.0"})
        with urllib.request.urlopen(request, timeout=120) as src, path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        unpack_verified(path, manifest, args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("pack")
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--appearance", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--release-url", required=True)
    p.add_argument("--source-commit", required=True)
    p.set_defaults(func=pack)
    p = commands.add_parser("restore")
    p.add_argument("--manifest", type=Path, default=Path("deploy/site-assets.json"))
    p.add_argument("--archive", type=Path, help="Use a local archive instead of downloading")
    p.add_argument("--out", type=Path, default=Path("site"))
    p.set_defaults(func=restore)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

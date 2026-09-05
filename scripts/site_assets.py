#!/usr/bin/env python3
"""Package or restore the exact generated dataset and Blender renders used by Pages.

Generated media live in a versioned GitHub release, not in the source checkout.
Only the small deployment manifest is committed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import re
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
REFERENCE_SCHEMA = "doorbench.reference-motion.v1"
REFERENCE_AVATAR_JOINTS = ["pelvis", "chest", "neck", "head", "shoulder_l", "elbow_l", "wrist_l",
                          "shoulder_r", "elbow_r", "wrist_r", "hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]
MAX_REFERENCE_CLIP_BYTES = 32 * 1024 * 1024


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


def _sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def _validate_reference_clip(clip: dict, row: dict, model: dict):
    door = row["door_id"]
    if (not isinstance(clip, dict) or clip.get("schema") != REFERENCE_SCHEMA or clip.get("door_id") != door
            or clip.get("source_sha256") != row["source_sha256"] or not _finite_tree(clip)):
        raise ValueError(f"Invalid reference clip identity, source or finite values: {door}")
    names = clip.get("joint_names")
    model_names = [b["joint"]["name"] for b in model["bodies"] if b.get("joint")]
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names) or set(names) != set(model_names)):
        raise ValueError(f"Reference joint names differ from model: {door}")
    avatar_names = clip.get("avatar_joint_names")
    if avatar_names != REFERENCE_AVATAR_JOINTS:
        raise ValueError(f"Invalid reference avatar joint order: {door}")
    bones = clip.get("avatar_bones")
    if (not isinstance(bones, list) or not bones
            or any(not isinstance(b, list) or len(b) != 2
                   or any(type(v) is not int or v < 0 or v >= len(avatar_names) for v in b) for b in bones)):
        raise ValueError(f"Invalid reference skeleton dimensions: {door}")
    times, duration = clip.get("times"), clip.get("duration")
    if (not isinstance(times, list) or len(times) < 2 or not all(_number(t) for t in times)
            or not _number(duration) or duration <= 0 or times[0] != 0
            or any(b <= a for a, b in zip(times, times[1:])) or abs(times[-1] - duration) > 0.0001):
        raise ValueError(f"Invalid reference timeline: {door}")
    n = len(times)
    for field, width in (("door_q", len(names)), ("avatar", len(avatar_names) * 3), ("targets", 3)):
        values = clip.get(field)
        if (not isinstance(values, list) or len(values) != n
                or any(not isinstance(v, list) or len(v) != width or not all(_number(q) for q in v) for v in values)):
            raise ValueError(f"Invalid reference {field} frame dimensions or finite values: {door}")
    for field in ("phases", "hand_active", "hand_error_m"):
        values = clip.get(field)
        if not isinstance(values, list) or len(values) != n:
            raise ValueError(f"Invalid reference {field} timeline dimensions: {door}")
    if (any(not isinstance(p, str) or not p for p in clip["phases"])
            or any(type(v) is not int or v not in (0, 1) for v in clip["hand_active"])
            or any(not _number(v) or v < 0 for v in clip["hand_error_m"])
            or not _number(clip.get("lead_in_s")) or not 0 <= clip["lead_in_s"] <= duration
            or not _number(clip.get("fps")) or not 1 <= clip["fps"] <= 60
            or clip.get("units") != "metres/radians/seconds" or clip.get("up_axis") != "Z"):
        raise ValueError(f"Invalid reference playback metadata: {door}")
    native = clip.get("native")
    if not isinstance(native, dict) or native.get("joint_names") != names or not _number(native.get("dt")) or native["dt"] <= 0:
        raise ValueError(f"Invalid reference native joint mapping: {door}")
    for field in ("qpos_addresses", "qvel_addresses"):
        values = native.get(field)
        if (not isinstance(values, list) or len(values) != len(names)
                or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(values)):
            raise ValueError(f"Invalid reference native {field}: {door}")
    outcome = clip.get("outcome")
    if (not isinstance(outcome, dict) or outcome.get("door_id") != door
            or type(outcome.get("success")) is not bool or outcome["success"] != row.get("success")
            or outcome.get("outcome") not in ("success", "fail", "damaged") or outcome["outcome"] != row.get("outcome")
            or outcome.get("scenario") != clip.get("scenario") or clip.get("scenario") != row.get("scenario")
            or row.get("frames") != n or not _number(row.get("duration")) or abs(row["duration"] - duration) > 1e-6):
        raise ValueError(f"Reference index metadata differs from clip: {door}")


def verified_reference_files(assets: Path, reference: Path) -> tuple[dict[str, Path], dict]:
    """Verify every compact web clip against the packaged sources; native NPZ files are not bundled or certified here."""
    dataset = read(assets / "manifest.json")
    index_path = relative_file(reference, "index.json")
    index = read(index_path)
    doors = [d["id"] for d in dataset["doors"]]
    rows = index.get("clips") if isinstance(index, dict) else None
    if (not isinstance(rows, list) or index.get("schema") != REFERENCE_SCHEMA
            or index.get("manifest_sha256") != digest(assets / "manifest.json")
            or not doors or len(set(doors)) != len(doors) or len(doors) != dataset["n_doors"]):
        raise ValueError("Invalid reference schema or dataset manifest checksum")
    ids = [r.get("door_id") if isinstance(r, dict) else None for r in rows]
    if (len(ids) != len(doors) or any(not isinstance(i, str) or re.fullmatch(r"[A-Za-z0-9_-]+", i) is None for i in ids)
            or len(set(ids)) != len(ids) or set(ids) != set(doors)):
        raise ValueError("Every dataset door must have exactly one reference clip")
    files = {"reference-motions/index.json": index_path}
    for row in rows:
        door = row["door_id"]
        source = row.get("source_sha256")
        if not isinstance(source, dict) or set(source) != {"spec.json", "model.json", "door.xml"}:
            raise ValueError(f"Incomplete reference source checksums: {door}")
        for name, expected in source.items():
            if not _sha256(expected) or digest(relative_file(assets / "doors" / door, name)) != expected:
                raise ValueError(f"Reference source checksum mismatch: {door}/{name}")
        clip_name = f"clips/{door}.json.gz"
        if row.get("web_clip") != clip_name or row.get("error"):
            raise ValueError(f"Invalid reference web clip path or recording error: {door}")
        path = relative_file(reference, clip_name)
        if not _sha256(row.get("web_clip_sha256")) or digest(path) != row["web_clip_sha256"]:
            raise ValueError(f"Reference web clip checksum mismatch: {door}")
        try:
            with gzip.open(path, "rb") as compressed:
                payload = compressed.read(MAX_REFERENCE_CLIP_BYTES + 1)
            if len(payload) > MAX_REFERENCE_CLIP_BYTES:
                raise ValueError("Reference clip exceeds decompressed size limit")
            clip = json.loads(payload)
        except (OSError, EOFError, ValueError) as exc:
            raise ValueError(f"Invalid compressed reference JSON: {door}: {exc}") from exc
        if not _sha256(row.get("clip_sha256")) or hashlib.sha256(payload).hexdigest() != row["clip_sha256"]:
            raise ValueError(f"Reference decoded clip checksum mismatch: {door}")
        _validate_reference_clip(clip, row, read(assets / "doors" / door / "model.json"))
        files[f"reference-motions/{clip_name}"] = path
    if dict(Counter(row["outcome"] for row in rows)) != index.get("counts"):
        raise ValueError("Reference outcome counts differ from clips")
    return files, {"reference_index_sha256": digest(index_path), "reference_count": len(rows)}


def verified_files(assets: Path, appearance: Path, reference: Path | None = None) -> tuple[dict[str, Path], dict]:
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
    for p in sorted(p for p in (assets / "hardware").iterdir() if p.suffix in (".obj", ".usdc")):
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
    if reference is not None:
        reference_files, reference_info = verified_reference_files(assets, reference)
        files.update(reference_files)
        info.update(reference_info)
    return files, info


def pack(args):
    files, info = verified_files(args.assets, args.appearance, getattr(args, "reference", None))
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
    has_reference = "reference_count" in manifest or "reference_index_sha256" in manifest
    if has_reference and (type(manifest.get("reference_count")) is not int or manifest["reference_count"] <= 0
                          or not _sha256(manifest.get("reference_index_sha256"))):
        raise ValueError("Incomplete reference deployment manifest")
    roots = ("assets", "appearance", "reference-motions") if has_reference else ("assets", "appearance")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        seen = set()
        for m in members:
            rel = PurePosixPath(m.name)
            if (not m.isfile() or rel.is_absolute() or ".." in rel.parts or "\\" in m.name
                    or m.name != rel.as_posix()
                    or len(rel.parts) < 2 or rel.parts[0] not in roots
                    or m.name in seen):
                raise ValueError(f"Unsafe release archive entry: {m.name}")
            if rel.parts[0] == "reference-motions" and not (m.name == "reference-motions/index.json"
                    or len(rel.parts) == 3 and rel.parts[1] == "clips" and rel.parts[2].endswith(".json.gz")):
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
    files, info = verified_files(out / "assets", out / "appearance", out / "reference-motions" if has_reference else None)
    if set(files) != seen:
        raise ValueError("Restored site file inventory mismatch")
    for key, value in info.items():
        if value != manifest[key]:
            raise ValueError(f"Restored site inventory mismatch: {key}")
    print(f"Verified {info['doors']} doors and {info['renders']} Blender images"
          + (f", {info['reference_count']} reference clips" if has_reference else ""))


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
    p.add_argument("--reference", type=Path, help="Optional reference-motions directory; bundle its index and compact .json.gz web clips")
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

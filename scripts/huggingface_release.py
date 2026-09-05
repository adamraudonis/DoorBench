#!/usr/bin/env python3
"""Prepare, publish and download version-pinned DoorBench dataset releases.

Preparation/download validation use stdlib; publication and Hub downloads import
huggingface_hub at runtime. Credentials are read privately from --token-file or
the standard HF_TOKEN environment variable and are never serialized.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

SCHEMA_VERSION = 1
COMPONENT_LICENSES = {"assets": "MIT", "appearance": "MIT AND CC0-1.0",
                      "textures": "CC0-1.0", "reference-motions": "MIT"}
SUPPORT_FILES = ("README.md", "LICENSE", "THIRD_PARTY.md", "DATASET_RELEASE.md", "download.py", "preview.jpg", "metadata/doors.jsonl")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def safe_name(name):
    if not isinstance(name, str):
        raise ValueError("Release paths must be strings")
    path = PurePosixPath(name)
    if (not name or any(ord(c) < 32 for c in name) or path.is_absolute() or ".." in path.parts or "\\" in name
            or name != path.as_posix() or not path.parts):
        raise ValueError(f"Unsafe release path: {name!r}")
    return name


def regular_file(root, name):
    root = Path(root).resolve()
    path = root / safe_name(name)
    if not path.is_file() or any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f"Missing or non-regular release file: {name}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Release file escapes source root: {name}")
    return path


def component_files(root, component):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Missing {component} directory: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks cannot enter a release: {path}")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if any(part.startswith(".") for part in PurePosixPath(name).parts):
                raise ValueError(f"Hidden files cannot enter a release: {name}")
            files[f"{component}/{name}"] = regular_file(root, name)
    return files


def inventory_for(files, component):
    return {safe_name(name): {"bytes": path.stat().st_size, "sha256": sha256(path),
                             "component": component, "license": COMPONENT_LICENSES[component]}
            for name, path in sorted(files.items())}


class _HashReader:
    def __init__(self, stream):
        self.stream, self.hash, self.count = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.hash.update(data)
        self.count += len(data)
        return data


def write_archive(files, records, destination):
    """Deterministic gzip/tar, rejecting source mutations during the read."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, path in sorted(files.items()):
                    record = records[name]
                    member = tarfile.TarInfo(safe_name(name))
                    member.size, member.mode, member.mtime = record["bytes"], 0o644, 0
                    with path.open("rb") as source:
                        reader = _HashReader(source)
                        archive.addfile(member, reader)
                        if (reader.count != record["bytes"] or reader.hash.hexdigest() != record["sha256"]
                                or source.read(1)):
                            raise ValueError(f"Release input changed during packing: {name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": "archives/" + destination.name, "sha256": sha256(destination),
            "bytes": destination.stat().st_size, "expanded_bytes": sum(r["bytes"] for r in records.values()),
            "file_count": len(records), "inventory_sha256": hashlib.sha256(canonical(records)).hexdigest()}


def _base_files(assets, appearance, expected_doors):
    # Reuse the strict source/render correspondence gate; include the complete
    # generated tree as well, including shared USDC meshes and validation reports.
    from scripts.site_assets import verified_files
    selected, info = verified_files(Path(assets), Path(appearance))
    if info["doors"] != expected_doors:
        raise ValueError(f"Expected {expected_doors} doors, found {info['doors']}")
    full = component_files(assets, "assets")
    allowed = {".json", ".xml", ".urdf", ".usda", ".usdc", ".obj", ".jpg", ".png"}
    if any(Path(name).suffix not in allowed for name in full):
        raise ValueError("Unexpected generated asset type; review it before release")
    # USD asset references must exist in the full tree, not only OBJ visuals.
    for name, path in full.items():
        if path.suffix == ".usda":
            for reference in re.findall(r"@([^@]+)@", path.read_text()):
                target = (path.parent / reference).resolve()
                if not target.is_relative_to(Path(assets).resolve()) or not target.is_file():
                    raise ValueError(f"Missing or external USD reference in {name}: {reference}")
    return {"assets": full, "appearance": {n: p for n, p in selected.items() if n.startswith("appearance/")}}, info


def _texture_files(manifest):
    from doorbench.appearance.textures import load_texture_library
    library = load_texture_library(manifest)
    root = Path(manifest).resolve().parent
    files = {"textures/manifest.json": root / "manifest.json"}
    for asset in library["assets"].values():
        if asset.get("license") != "CC0-1.0":
            raise ValueError("Only the documented CC0 texture library can be redistributed")
        for entry in asset["maps"].values():
            path = Path(entry["path"])
            if not path.is_relative_to(root):
                raise ValueError("Texture maps must remain under their licensed manifest directory")
            files["textures/" + path.relative_to(root).as_posix()] = path
    return files


def _motion_files(root, door_ids, assets):
    files = component_files(root, "reference-motions")
    index = read(Path(root) / "index.json")
    if index.get("schema") != "doorbench.reference-motion.v1":
        raise ValueError("Unsupported reference-motion schema")
    if (index.get("policy"), index.get("tier"), index.get("seed")) != ("scripted_hand", "full", 0):
        raise ValueError("Reference release requires the recorded full-tier scripted-hand seed-0 corpus")
    if index.get("manifest_sha256") != sha256(Path(assets) / "manifest.json"):
        raise ValueError("Reference motions belong to a different dataset manifest")
    rows = index.get("clips", [])
    if len(rows) != len(door_ids) or {r["door_id"] for r in rows} != door_ids:
        raise ValueError("Reference-motion index must contain every released door exactly once")
    for row in rows:
        door = row["door_id"]
        if (not isinstance(row.get("success"), bool) or row.get("outcome") not in ("success", "fail", "damaged")
                or row["success"] != (row["outcome"] == "success")):
            raise ValueError(f"Reference motion needs consistent explicit success/failure labels: {door}")
        if set(row.get("source_sha256", {})) != {"spec.json", "model.json", "door.xml"}:
            raise ValueError(f"Reference motion lacks source hashes: {door}")
        for name, expected in row["source_sha256"].items():
            if sha256(regular_file(Path(assets) / "doors" / door, name)) != expected:
                raise ValueError(f"Reference motion source changed: {door}/{name}")
        for field, extension, subdirectory in (("clip", ".json", "clips"), ("trajectory", ".npz", "trajectories")):
            if row.get(field) != f"{subdirectory}/{door}{extension}":
                raise ValueError(f"Reference motion has an unexpected {field} path: {door}")
            if sha256(regular_file(root, row[field])) != row.get(field + "_sha256"):
                raise ValueError(f"Reference motion checksum mismatch: {door}/{field}")
        if "web_clip" in row:
            if row["web_clip"] != f"clips/{door}.json.gz":
                raise ValueError(f"Unexpected compressed clip path: {door}")
            compressed = regular_file(root, row["web_clip"])
            if sha256(compressed) != row.get("web_clip_sha256"):
                raise ValueError(f"Compressed reference motion checksum mismatch: {door}")
            h, size = hashlib.sha256(), 0
            limit = regular_file(root, row["clip"]).stat().st_size
            with gzip.open(compressed, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
                    size += len(block)
                    if size > limit:
                        raise ValueError(f"Compressed clip differs from original JSON: {door}")
            if size != limit or h.hexdigest() != row["clip_sha256"]:
                raise ValueError(f"Compressed clip differs from original JSON: {door}")
    counts = {outcome: sum(row["outcome"] == outcome for row in rows) for outcome in ("success", "fail", "damaged")}
    if index.get("counts") != counts:
        raise ValueError("Reference-motion outcome totals differ from its per-door records")
    # Physical success and avatar accuracy remain explicit generator measurements;
    # archival validation never promotes a failed episode to a demonstration.
    clips = {Path(n).stem for n in files if n.startswith("reference-motions/clips/") and n.endswith(".json")}
    native = {Path(n).stem for n in files if n.startswith("reference-motions/trajectories/") and n.endswith(".npz")}
    if clips != door_ids or native != door_ids:
        raise ValueError("Reference motions need one JSON clip and one native NPZ for every released door")
    if any(Path(n).suffix not in (".json", ".npz", ".md", ".txt") and not n.endswith(".json.gz") for n in files):
        raise ValueError("Unexpected motion payload; review its format and license before publication")
    return files, index


def prepare(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.release):
        raise ValueError("Release tag must be one safe version name")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("Record the full 40-character generator source commit")
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.out).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    components, info = _base_files(args.assets, args.appearance, args.expected_doors)
    dataset = read(Path(args.assets) / "manifest.json")
    ids = {row["id"] for row in dataset["doors"]}
    if args.texture_manifest:
        components["textures"] = _texture_files(args.texture_manifest)
    motion = None
    if args.motions:
        components["reference-motions"], motion = _motion_files(args.motions, ids, args.assets)
    inventory, archives = {}, {}
    for component, files in components.items():
        records = inventory_for(files, component)
        signature = hashlib.sha256(canonical(records)).hexdigest()
        path = destination / "archives" / f"{component}-{signature[:16]}.tar.gz"
        sidecar = destination / ".prepared" / f"{component}-{signature}.json"
        if path.is_file() and sidecar.is_file() and sha256(path) == read(sidecar)["sha256"]:
            archives[component] = read(sidecar)
        else:
            archives[component] = write_archive(files, records, path)
            write_json(sidecar, archives[component])
        inventory.update(records)
        print(f"Prepared {component}: {len(files)} files, {archives[component]['bytes']} compressed bytes", flush=True)
    write_json(destination / "inventory.json", {"schema_version": 1, "files": inventory})
    release = {"schema_version": 1, "release": args.release, "repo_id": args.repo_id,
               "source_commit": args.source_commit, "source_url": f"https://github.com/adamraudonis/DoorBench/tree/{args.source_commit}",
               "inventory_sha256": sha256(destination / "inventory.json"), "components": archives,
               "licenses": COMPONENT_LICENSES, "summary": info,
               "reference_motion": {"included": motion is not None, "doors": len(ids) if motion is not None else 0,
                                    "counts": motion.get("counts", {}) if motion else {},
                                    "semantics": "Actual scripted-hand MuJoCo door states with a kinematic humanoid reference; not a controlled humanoid policy"}}
    write_json(destination / "release.json", release)
    metadata = destination / "metadata"
    metadata.mkdir(exist_ok=True)
    rows = [{k: row.get(k) for k in ("id", "family", "use_case", "context", "operator", "lock", "closer", "condition", "difficulty", "mass_kg", "signed_off")}
            for row in dataset["doors"]]
    (metadata / "doors.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    shutil.copyfile(root / "LICENSE", destination / "LICENSE")
    shutil.copyfile(root / "docs" / "review" / "blender" / "looks.jpg", destination / "preview.jpg")
    shutil.copyfile(root / "docs" / "DATASET_RELEASE.md", destination / "DATASET_RELEASE.md")
    shutil.copyfile(__file__, destination / "download.py")
    template = (root / "deploy" / "huggingface" / "README.md").read_text()
    replacements = {"REPO_ID": args.repo_id, "RELEASE": args.release, "SOURCE_COMMIT": args.source_commit,
                    "DOORS": str(info["doors"]), "RENDERS": str(info["renders"]), "PHOTO_RENDERS": str(info["photo_renders"]),
                    "MOTION_DOORS": str(len(ids) if motion is not None else 0)}
    for name, value in replacements.items():
        template = template.replace("{{" + name + "}}", value)
    (destination / "README.md").write_text(template)
    (destination / "THIRD_PARTY.md").write_text(
        "# Third-party appearance data\n\nDoorBench code, generated doors and procedural reference motion are MIT; see LICENSE.\n\n"
        "Poly Haven raster maps, including maps packed inside optional Blender scenes, are CC0-1.0. "
        "Source URLs, authors, provider checksums, DoorBench SHA256 hashes, map scale and calibrations are in "
        "textures/manifest.json and each appearance render record.\n\n"
        "License: https://polyhaven.com/license · https://creativecommons.org/publicdomain/zero/1.0/\n\n"
        "MuJoCo, Blender, G1 robot meshes and pretrained Unitree policy files are not redistributed in these archives. "
        "A kinematic reference avatar is not a recording of a real person.\n")
    release["support_files"] = {name: {"sha256": sha256(destination / name), "bytes": (destination / name).stat().st_size}
                                for name in SUPPORT_FILES}
    write_json(destination / "release.json", release)
    print(f"Prepared {args.repo_id} {args.release}; motion component included: {motion is not None}", flush=True)
    return release


def release_files(folder):
    folder = Path(folder)
    release = read(folder / "release.json")
    if set(release.get("support_files", {})) != set(SUPPORT_FILES):
        raise ValueError("Prepared release needs a complete support-file inventory")
    names = [*SUPPORT_FILES, "release.json", "inventory.json"]
    names.extend(value["path"] for value in release["components"].values())
    files = {safe_name(name): regular_file(folder, name) for name in names}
    if sha256(files["inventory.json"]) != release["inventory_sha256"]:
        raise ValueError("Prepared inventory changed")
    for name, record in release["support_files"].items():
        if files[name].stat().st_size != record["bytes"] or sha256(files[name]) != record["sha256"]:
            raise ValueError(f"Prepared support file changed: {name}")
    for component in release["components"].values():
        path = files[component["path"]]
        if path.stat().st_size != component["bytes"] or sha256(path) != component["sha256"]:
            raise ValueError(f"Prepared archive changed: {component['path']}")
    return release, files


def verify_public_files(api, repo_id, revision, files):
    info = api.repo_info(repo_id, repo_type="dataset", revision=revision)
    if info.private or info.gated:
        raise ValueError("Published dataset must be public and ungated")
    remote = {entry.path: entry for entry in api.get_paths_info(repo_id, list(files), repo_type="dataset", revision=revision)}
    for name, path in files.items():
        entry = remote.get(name)
        if entry is None or entry.size != path.stat().st_size:
            raise ValueError(f"Published file missing or wrong size: {name}")
        if entry.lfs:
            valid = entry.lfs.sha256 == sha256(path)
        else:
            payload = path.read_bytes()
            valid = entry.blob_id == hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        if not valid:
            raise ValueError(f"Published checksum mismatch: {name}")


def publish(args):
    release, files = release_files(args.folder)
    if not release["reference_motion"]["included"]:
        raise ValueError("Add the completed reference-motion component before publishing this release")
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
    token = Path(args.token_file).read_text().strip() if args.token_file else os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Supply --token-file or HF_TOKEN; credentials are never stored in the release")
    api = HfApi(token=token)
    repo_id = release["repo_id"]
    try:
        previous = api.repo_info(repo_id, repo_type="dataset", revision=release["release"])
    except (RepositoryNotFoundError, RevisionNotFoundError):
        previous = None
    if previous is not None:
        existing = read(api.hf_hub_download(repo_id, "release.json", repo_type="dataset", revision=previous.sha))
        if existing != release:
            raise ValueError("Release tag already identifies different bytes; choose a new version")
        api.update_repo_settings(repo_id, repo_type="dataset", private=False, gated=False)
        verify_public_files(HfApi(token=False), repo_id, previous.sha, files)
        print(f"Already published: https://huggingface.co/datasets/{repo_id}/tree/{release['release']}")
        return {"repo_id": repo_id, "commit": previous.sha, "release": release["release"]}
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_settings(repo_id, repo_type="dataset", private=False, gated=False)
    commit = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=args.folder,
                               allow_patterns=list(files), commit_message=f"Release DoorBench {release['release']}")
    public = HfApi(token=False)
    verify_public_files(public, repo_id, commit.oid, files)
    api.create_tag(repo_id, repo_type="dataset", tag=release["release"], revision=commit.oid,
                   tag_message=f"Immutable inventory: {release['inventory_sha256']}")
    if public.repo_info(repo_id, repo_type="dataset", revision=release["release"]).sha != commit.oid:
        raise ValueError("Public release tag did not resolve to the verified commit")
    result = {"repo_id": repo_id, "commit": commit.oid, "release": release["release"], "public": True,
              "url": f"https://huggingface.co/datasets/{repo_id}/tree/{release['release']}"}
    write_json(Path(args.folder) / "publication.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def extract_component(path, component, records, out):
    """Reject the entire unsafe archive inventory before extracting any member."""
    if path.stat().st_size != component["bytes"] or sha256(path) != component["sha256"]:
        raise ValueError("Downloaded archive checksum or size mismatch")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        seen = set()
        for member in members:
            name = safe_name(member.name)
            if not member.isfile() or name not in records or name in seen or member.size != records[name]["bytes"]:
                raise ValueError(f"Unsafe or unexpected archive member: {name}")
            seen.add(name)
        if seen != set(records) or len(members) != component["file_count"] or sum(m.size for m in members) != component["expanded_bytes"]:
            raise ValueError("Downloaded archive inventory mismatch")
        for name in seen:
            if any(parent.as_posix() in seen for parent in PurePosixPath(name).parents):
                raise ValueError(f"Archive file/directory collision: {name}")
            target = out / name
            if target.is_symlink() or not target.resolve().is_relative_to(out.resolve()):
                raise ValueError(f"Unsafe output path: {name}")
        for member in members:
            target = out / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with archive.extractfile(member) as source, target.open("wb") as destination:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(block)
                    h.update(block)
            if h.hexdigest() != records[member.name]["sha256"]:
                raise ValueError(f"Extracted file checksum mismatch: {member.name}")


def download(args):
    from huggingface_hub import HfApi, hf_hub_download
    revision = HfApi(token=False).repo_info(args.repo_id, repo_type="dataset", revision=args.revision).sha
    fetch = lambda name: Path(hf_hub_download(args.repo_id, safe_name(name), repo_type="dataset", revision=revision, token=False))
    release = read(fetch("release.json"))
    if release.get("schema_version") != 1 or release.get("repo_id") != args.repo_id:
        raise ValueError("Unsupported or mismatched release manifest")
    if args.revision != release["release"] and args.revision != revision:
        raise ValueError("Pin the dataset's release tag or full commit SHA, not a moving branch")
    inventory_path = fetch("inventory.json")
    if sha256(inventory_path) != release["inventory_sha256"]:
        raise ValueError("Release inventory checksum mismatch")
    records = read(inventory_path)["files"]
    wanted = list(release["components"]) if args.components == "all" else args.components.split(",")
    if not wanted or set(wanted) - set(release["components"]) or len(set(wanted)) != len(wanted):
        raise ValueError("Choose unique available components: " + ", ".join(release["components"]))
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError("Download destination already exists; choose a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".doorbench-download-", dir=out.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for name in wanted:
            component = release["components"][name]
            subset = {n: r for n, r in records.items() if r["component"] == name}
            if hashlib.sha256(canonical(subset)).hexdigest() != component["inventory_sha256"]:
                raise ValueError(f"Component inventory checksum mismatch: {name}")
            extract_component(fetch(component["path"]), component, subset, staging)
        write_json(staging / "release.json", release)
        shutil.copyfile(inventory_path, staging / "inventory.json")
        write_json(staging / "installed.json", {"schema_version": 1, "revision": revision,
                                               "components": wanted})
        for name in ("LICENSE", "THIRD_PARTY.md"):
            source = fetch(name)
            record = release["support_files"][name]
            if source.stat().st_size != record["bytes"] or sha256(source) != record["sha256"]:
                raise ValueError(f"Downloaded support file checksum mismatch: {name}")
            shutil.copyfile(source, staging / name)
        staging.replace(out)
    print(f"Verified {release['release']} ({', '.join(wanted)}) at {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--assets", type=Path, default=Path("assets"))
    p.add_argument("--appearance", type=Path, default=Path("out/appearance"))
    p.add_argument("--texture-manifest", type=Path, default=Path("out/appearance-textures/manifest.json"))
    p.add_argument("--motions", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--repo-id", default="adamraudonis/DoorBench")
    p.add_argument("--expected-doors", type=int, default=1000)
    p.set_defaults(func=prepare)
    p = commands.add_parser("publish")
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--token-file", type=Path)
    p.set_defaults(func=publish)
    p = commands.add_parser("download")
    p.add_argument("--repo-id", default="adamraudonis/DoorBench")
    p.add_argument("--revision", required=True, help="Pin a release tag or commit SHA")
    p.add_argument("--components", default="all")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=download)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # Preparation imports repository modules; the standalone download path does
    # not need the source checkout, MuJoCo, Blender, or a credential.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()

"""Portable identity of authored MJCF and the bytes of its referenced assets.

This proves source-design equality, not equality of machine-dependent compiled
convex hulls or of an imported live plant. Retain the strict compiled receipt
and regenerate geometry/physical qualification on each target runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

SCHEMA = "doorbench.robot-source-design.v1"
_FILE_ATTRIBUTES = {"file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"}
_ASSET_TAGS = {"mesh", "texture", "hfield", "skin"}
_DIRECTORY_ATTRIBUTES = {"assetdir", "meshdir", "texturedir"}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _parse(path):
    data = path.read_bytes()
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("External entities and document types are outside the MJCF identity contract")
    return ET.fromstring(data)


def _expanded_xml(path):
    """MJCF includes resolve relative to the main model, with unique files."""
    root = _parse(path)
    if root.tag != "mujoco":
        raise ValueError("Source-design identity currently requires an MJCF root")
    seen = {path}

    def expand(parent):
        for child in list(parent):
            if child.tag != "include":
                expand(child)
                continue
            if set(child.attrib) != {"file"} or len(child):
                raise ValueError("Malformed MJCF include")
            included = (path.parent / child.attrib["file"]).resolve(strict=True)
            if included in seen:
                raise ValueError("Repeated or cyclic MJCF include")
            seen.add(included)
            subtree = _parse(included)
            if not len(subtree):
                raise ValueError("An MJCF include must insert at least one element")
            expand(subtree)
            index = list(parent).index(child)
            parent.remove(child)
            for offset, element in enumerate(list(subtree)):
                parent.insert(index + offset, element)

    expand(root)
    return root


def robot_design_identity(path):
    """Hash exact authored settings and resolved file bytes without path names.

    Includes are expanded; default/class declarations and their order remain in
    the canonical tree, preserving inheritance and override semantics. Numeric
    attribute strings are never rounded. Attribute order, indentation, comments,
    include filenames and resolved asset locations do not affect identity.
    Numerically equivalent spellings may conservatively produce different hashes.

    Only the declared MJCF file asset types and built-in touch_grid plugin are
    accepted. An unsupported file provider must receive its own explicit audit;
    it is not silently omitted. Model compilation/physics is never performed.
    """
    path = Path(path).resolve(strict=True)
    root = _expanded_xml(path)
    # Use the actual pinned parser to validate default classes/includes and to
    # resolve compiler directory overrides. Do NOT serialize/compile the spec:
    # compiler-generated meshes and formatted numeric values are not the source.
    spec = mujoco.MjSpec.from_file(str(path))
    directories = {"mesh": spec.meshdir, "texture": spec.texturedir,
                   "hfield": spec.meshdir, "skin": spec.meshdir}
    compiler = {}
    for node in root.findall("compiler"):
        compiler.update(node.attrib)
    strip = compiler.get("strippath", "false") == "true"
    assets = []
    content_cache = {}

    for index, node in enumerate(root.iter()):
        if node.tag in ("attach", "model"):
            raise ValueError("Attached model assets require a separate recursive design contract")
        if node.tag == "plugin" and node.get("plugin") not in (None, "mujoco.sensor.touch_grid"):
            raise ValueError("Opaque plugins require a separate source-design contract")
        if node.tag == "config" and "file" in node.get("key", "").lower():
            raise ValueError("Plugin file providers require an explicit asset contract")
        for attribute in sorted(set(node.attrib) & _FILE_ATTRIBUTES):
            if node.tag not in _ASSET_TAGS:
                raise ValueError(f"Unsupported file provider: {node.tag}.{attribute}")
            filename = node.attrib[attribute]
            if not filename:
                continue
            file_path = Path(filename.replace("\\", "/"))
            if strip:
                file_path = Path(file_path.name)
            if not file_path.is_absolute():
                file_path = path.parent / directories[node.tag] / file_path
            file_path = file_path.resolve(strict=True)
            if not file_path.is_file():
                raise ValueError("Referenced asset must resolve to a regular file")
            if file_path not in content_cache:
                data = file_path.read_bytes()
                content_cache[file_path] = (hashlib.sha256(data).hexdigest(), len(data))
            digest, size = content_cache[file_path]
            # The decoder can depend on the suffix, even if bytes coincide.
            # Preserve inferred asset names when MJCF derives them from a file.
            assets.append(dict(element_index=index, tag=node.tag, attribute=attribute,
                sha256=digest, bytes=size, format_suffix=file_path.suffix.lower(),
                implicit_name=file_path.stem if not node.get("name") else None))
            node.set(attribute, "sha256:" + digest)
        if node.tag == "compiler":
            for attribute in _DIRECTORY_ATTRIBUTES:
                node.attrib.pop(attribute, None)

    def canonical(node):
        # Non-whitespace text is not silently dropped from the contract.
        return [node.tag, sorted(node.attrib.items()), (node.text or "").strip(),
                [canonical(child) for child in node], (node.tail or "").strip()]

    semantic_bytes = _encoded(canonical(root))
    payload = dict(schema=SCHEMA, mujoco_version=mujoco.__version__,
        canonicalization="ordered expanded MJCF; exact attribute strings; content-addressed asset paths; v1",
        canonical_xml_sha256=hashlib.sha256(semantic_bytes).hexdigest(),
        referenced_assets=assets,
        qualification="Source-design equality only; target compiled geometry and physical qualification required")
    return {**payload, "sha256": hashlib.sha256(_encoded(payload)).hexdigest()}


def verify_robot_design_identity(path, expected):
    if type(expected) is not dict or expected.get("schema") != SCHEMA:
        raise ValueError("Unsupported robot source-design identity schema")
    actual = robot_design_identity(path)
    if expected.get("mujoco_version") != actual["mujoco_version"]:
        raise ValueError("Source-design identity requires the pinned MJCF parser version")
    if expected != actual:
        raise ValueError("Authored robot settings or referenced asset bytes differ")
    return actual

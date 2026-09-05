"""Opt-in, local CC0 texture library from the Poly Haven public API.

Only the explicit ``fetch_library`` call uses the network. Raster maps are
cached below an output directory; no executable .blend files are downloaded.
API terms: unique application User-Agent and a visible provider credit during
fetch. Asset metadata dimensions are millimeters, converted to meters here.
Provider scales are starting points, not universally accurate measurements;
explicit visual calibrations retain the original dimensions and provenance.
These scans improve shading only: no simulation mass, geometry or friction is
changed. Unmapped finishes retain their procedural material.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.parse
import urllib.request

SCHEMA_VERSION = "1.0"
LICENSE_URL = "https://polyhaven.com/license"
CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
API_TERMS_URL = "https://github.com/Poly-Haven/Public-API/blob/master/ToS.md"
USER_AGENT = "DoorBench-Appearance/1.0 (local CC0 texture library)"
# Curated actual surfaces; avoid silently labeling slate as limestone or
# rectangular wall tile as square porcelain. Those presets stay procedural.
CURATED_ASSETS = (
    "walnut_veneer", "oak_veneer_01", "wood_floor",
    "white_plaster_02", "brick_wall_001", "concrete_floor",
    "stone_tiles_02", "long_white_tiles", "floor_tiles_06", "rough_wood",
)
PRESET_ASSETS = {
    "wood_source": "oak_veneer_01", "wood_oak": "oak_veneer_01",
    "wood_walnut": "walnut_veneer", "floor_oak": "wood_floor",
    # Continuous weathered timber: no baked board seams/nail rows to
    # contradict the source door's separate plank geometry.
    "wood_weathered": "rough_wood",
    "wall_white_plaster": "white_plaster_02", "wall_limewash": "white_plaster_02",
    "wall_red_brick": "brick_wall_001", "wall_subway_tile": "long_white_tiles",
    "wall_concrete": "concrete_floor", "floor_concrete": "concrete_floor",
    "floor_dark_concrete": "concrete_floor", "floor_slate": "stone_tiles_02",
    "floor_porcelain": "floor_tiles_06",
}
MAP_TYPES = {"diffuse": ("Diffuse", "jpg", "sRGB"),
             "normal": ("nor_gl", "png", "Non-Color"),
             "roughness": ("Rough", "jpg", "Non-Color")}
_HASH_CACHE = {}  # Only immutable paths/stat signatures/digests, never bpy refs.


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _safe_url(url, *, api=False):
    parsed = urllib.parse.urlparse(url)
    hosts = {"api.polyhaven.com"} if api else {"dl.polyhaven.org", "dl.polyhaven.com"}
    if parsed.scheme != "https" or parsed.hostname not in hosts or parsed.username or parsed.password:
        raise ValueError(f"Unexpected Poly Haven URL: {url!r}")
    if not api and Path(parsed.path).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Only JPG and PNG texture downloads are supported")
    return url


def _request(url, *, api=False):
    _safe_url(url, api=api)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        _safe_url(response.url, api=api)
        return response.read()


def _image_dimensions(data):
    """Read PNG/JPEG dimensions without requiring Pillow in the Blender Python."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return list(struct.unpack(">II", data[16:24]))
    if data.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 4 <= len(data):
            if data[offset] != 255:
                offset += 1
                continue
            while offset < len(data) and data[offset] == 255: offset += 1
            if offset >= len(data): break
            marker = data[offset]
            offset += 1
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7: continue
            length = int.from_bytes(data[offset:offset+2], "big")
            if length < 2: break
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                height, width = struct.unpack(">HH", data[offset+3:offset+7])
                return [width, height]
            offset += length
    raise ValueError("Texture is not a supported PNG or JPEG image")


def _file_digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    identity = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity not in _HASH_CACHE:
        # Remove stale versions for this file without retaining huge map data.
        for key in [k for k in _HASH_CACHE if k[0] == str(path)]: del _HASH_CACHE[key]
        _HASH_CACHE[identity] = _digest(path.read_bytes())
    return _HASH_CACHE[identity]


def fetch_library(out, resolution="2k"):
    """Fetch ten curated raster PBR sets; return ``out/manifest.json`` Path.

    Existing verified maps are reused. Downloads and manifest replacement are
    atomic, so a failed request leaves a prior complete library usable. Calls
    are sequential to keep demand on the public service modest.
    """
    if resolution not in {"1k", "2k", "4k"}:
        raise ValueError("resolution must be 1k, 2k or 4k")
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Fetching CC0 textures from Poly Haven ({LICENSE_URL})", flush=True)
    assets = {}
    for asset_id in CURATED_ASSETS:
        metadata_url = f"https://api.polyhaven.com/info/{asset_id}"
        files_url = f"https://api.polyhaven.com/files/{asset_id}"
        info = json.loads(_request(metadata_url, api=True))
        files = json.loads(_request(files_url, api=True))
        dimensions = info.get("dimensions")
        if not isinstance(dimensions, list) or len(dimensions) < 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in dimensions[:2]):
            raise ValueError(f"Missing real-world dimensions for {asset_id}")
        asset = dict(id=asset_id, name=info["name"], source_url=f"https://polyhaven.com/a/{asset_id}",
                     metadata_url=metadata_url, files_url=files_url, license="CC0-1.0", license_url=LICENSE_URL,
                     authors=info.get("authors", {}), dimensions_mm=dimensions[:2],
                     scale_m=[float(v)/1000 for v in dimensions[:2]],
                     rotation_deg=90 if asset_id in ("long_white_tiles",) else 0, normal_strength=.6 if "veneer" in asset_id else .8,
                     maps={})
        asset["provider_scale_m"] = list(asset["scale_m"])
        asset["scale_source"] = "Poly Haven metadata dimensions (millimeters converted to meters)"
        if asset_id == "brick_wall_001":
            # Visual calibration: the raster spans approximately five bricks
            # horizontally and fifteen courses vertically. This is our scale
            # choice, not a revised measurement supplied by Poly Haven.
            asset["scale_m"] = [1.125, 1.125]
            asset["scale_source"] = "DoorBench visual calibration of scan"
            asset["scale_calibration"] = {
                "method": "Approximately 5 x 225mm horizontal modules and 15 x 75mm courses; 215 x 65mm standard brick plus 10mm mortar joints",
                "reference_url": "https://www.wienerberger.co.uk/content/dam/wienerberger/united-kingdom/marketing/documents-magazines/technical/brick-technical-guidance-sheets/UK_MKT_DOC_Brickwork%20Dimension%20Tables.pdf",
                "uncertainty": "Visual module calibration, not a measured reconstruction of this particular wall"}
        for role, (api_key, extension, colorspace) in MAP_TYPES.items():
            try: item = files[api_key][resolution][extension]
            except KeyError as error: raise ValueError(f"Missing {resolution} {api_key}/{extension} map for {asset_id}") from error
            url = _safe_url(item["url"])
            destination = out / asset_id / resolution / f"{role}.{extension}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = destination.read_bytes() if destination.exists() else b""
            if hashlib.md5(data).hexdigest() != item["md5"]:
                data = _request(url)
                if hashlib.md5(data).hexdigest() != item["md5"]:
                    raise ValueError(f"Poly Haven checksum mismatch for {asset_id}/{role}")
                temporary = destination.with_suffix(destination.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(destination)
            pixel_dimensions = _image_dimensions(data)
            asset["maps"][role] = dict(path=str(destination.relative_to(out)), source_url=url,
                                       sha256=_digest(data), md5=item["md5"], bytes=len(data),
                                       dimensions_px=pixel_dimensions, colorspace=colorspace)
        assets[asset_id] = asset
        print(f"  Poly Haven: {asset_id} ({resolution})", flush=True)
    manifest = dict(schema_version=SCHEMA_VERSION, provider="Poly Haven", license="CC0-1.0",
                    license_url=LICENSE_URL, cc0_url=CC0_URL, api_terms_url=API_TERMS_URL,
                    resolution=resolution, assets=assets, preset_assets=dict(PRESET_ASSETS))
    manifest["library_sha256"] = _digest(_canonical(manifest))
    path = out / "manifest.json"
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    load_texture_library(path)  # Verify the exact public payload before returning.
    return path


def load_texture_library(manifest):
    """Validate manifest/maps and return portable JSON with absolute map paths.

    Accepts a manifest path or the already normalized dictionary embedded in a
    prepared render job. Every map is SHA256 checked; unchanged files use a
    size/mtime/ctime cache, so repeated render jobs do not reread the maps.
    """
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).expanduser().resolve()
        raw = path.read_bytes()
        result = json.loads(raw)
        claimed = result.pop("library_sha256", None)
        if claimed != _digest(_canonical(result)): raise ValueError("Texture manifest content hash mismatch")
        result["library_sha256"] = claimed
        result["manifest_sha256"] = _digest(raw)
        result["manifest_path"] = str(path)
        root = path.parent
    elif isinstance(manifest, dict):
        result = copy.deepcopy(manifest)
        claimed = result.pop("resolved_sha256", None)
        if claimed != _digest(_canonical(result)): raise ValueError("Resolved texture library content hash mismatch")
        root = Path(result.get("manifest_path", ".")).resolve().parent
    else:
        raise TypeError("texture library must be a manifest path or dictionary")
    if result.get("schema_version") != SCHEMA_VERSION or result.get("license") != "CC0-1.0":
        raise ValueError("Unsupported texture library schema/license")
    assets = result.get("assets")
    if not isinstance(assets, dict) or not assets: raise ValueError("Texture library has no assets")
    for asset_id, asset in assets.items():
        scale = asset.get("scale_m", [])
        if len(scale) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in scale):
            raise ValueError(f"Invalid meter scale for {asset_id}")
        for role, (_, _, colorspace) in MAP_TYPES.items():
            entry = asset.get("maps", {}).get(role)
            if not isinstance(entry, dict): raise ValueError(f"Missing {role} for {asset_id}")
            path = (root / entry["path"]).resolve()
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: raise ValueError("Texture map must be raster JPG/PNG")
            if not path.is_file(): raise ValueError(f"Texture map does not exist: {path}")
            if path.stat().st_size != entry["bytes"] or _file_digest(path) != entry["sha256"]:
                raise ValueError(f"Texture map SHA256 mismatch: {path}")
            if entry.get("colorspace") != colorspace: raise ValueError(f"Wrong colorspace for {asset_id}/{role}")
            entry["path"] = str(path)
    for preset_id, asset_id in result.get("preset_assets", {}).items():
        if asset_id not in assets: raise ValueError(f"Unknown texture asset {asset_id!r} for {preset_id}")
    result["resolved_sha256"] = _digest(_canonical(result))
    return result

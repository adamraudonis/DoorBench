"""Local texture provenance, cache integrity and opt-in rendering contract."""
import hashlib
import json
import struct

import pytest

from doorbench.appearance import textures


def fixture_manifest(tmp_path):
    data = b"\x89PNG\r\n\x1a\n" + b"\0"*8 + struct.pack(">II", 2, 2)
    maps = {}
    for role, (_, _, colorspace) in textures.MAP_TYPES.items():
        path = tmp_path / f"{role}.png"
        path.write_bytes(data)
        maps[role] = {"path": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                      "bytes": len(data), "colorspace": colorspace, "dimensions_px": [2, 2]}
    manifest = {"schema_version": "1.0", "license": "CC0-1.0", "assets": {
        "oak_veneer_01": {"id": "oak_veneer_01", "scale_m": [1.83, 1.83], "maps": maps}},
        "preset_assets": {"wood_oak": "oak_veneer_01"}}
    manifest["library_sha256"] = textures._digest(textures._canonical(manifest))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_manifest_normalizes_paths_and_survives_job_serialization(tmp_path):
    path = fixture_manifest(tmp_path)
    library = textures.load_texture_library(path)
    assert library["manifest_path"] == str(path)
    assert library["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert library == textures.load_texture_library(json.loads(json.dumps(library)))
    for entry in library["assets"]["oak_veneer_01"]["maps"].values():
        assert entry["path"].startswith(str(tmp_path))


def test_corrupt_texture_after_prepare_is_detected_even_same_size(tmp_path):
    library = textures.load_texture_library(fixture_manifest(tmp_path))
    path = tmp_path / "normal.png"
    path.write_bytes(path.read_bytes()[:-1] + b"X")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        textures.load_texture_library(library)


def test_manifest_tampering_detected(tmp_path):
    path = fixture_manifest(tmp_path)
    data = json.loads(path.read_text())
    data["assets"]["oak_veneer_01"]["scale_m"] = [10, 10]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="content hash mismatch"):
        textures.load_texture_library(path)


def test_job_library_tampering_detected(tmp_path):
    library = textures.load_texture_library(fixture_manifest(tmp_path))
    library["assets"]["oak_veneer_01"]["scale_m"] = [10, 10]
    with pytest.raises(ValueError, match="content hash mismatch"):
        textures.load_texture_library(library)


def test_unchanged_files_use_stat_hash_cache(tmp_path, monkeypatch):
    library = textures.load_texture_library(fixture_manifest(tmp_path))
    original = textures.Path.read_bytes
    def no_maps(path):
        if path.suffix == ".png": raise AssertionError("Unchanged map was reread")
        return original(path)
    monkeypatch.setattr(textures.Path, "read_bytes", no_maps)
    assert textures.load_texture_library(library) == library


@pytest.mark.parametrize("url", ["http://dl.polyhaven.org/foo.png", "https://example.com/a.png",
                                    "https://dl.polyhaven.org/foo.blend", "https://user@dl.polyhaven.org/foo.jpg"])
def test_fetch_rejects_non_raster_or_unknown_hosts(url):
    with pytest.raises(ValueError): textures._safe_url(url)


def test_curated_presets_and_scan_semantics_are_valid():
    from doorbench.appearance.catalog import surface_preset
    assert len(textures.CURATED_ASSETS) >= 8
    for preset, asset in textures.PRESET_ASSETS.items():
        assert surface_preset(preset) and asset in textures.CURATED_ASSETS
    assert "wall_limestone" not in textures.PRESET_ASSETS  # Slate is not limestone.
    assert textures.MAP_TYPES["normal"] == ("nor_gl", "png", "Non-Color")


def test_configure_is_explicit_and_no_network(tmp_path, monkeypatch):
    from doorbench.appearance.blender_materials import configure_texture_library, _texture_asset
    monkeypatch.setattr(textures, "_request", lambda *a, **k: pytest.fail("Unexpected network"))
    configure_texture_library(fixture_manifest(tmp_path))
    assert _texture_asset("wood_oak", {})["id"] == "oak_veneer_01"
    assert _texture_asset("glass_clear", {}) is None
    configure_texture_library(None)
    assert _texture_asset("wood_oak", {}) is None


def test_wood_rail_uses_part_anchor_and_long_axis(monkeypatch):
    from doorbench.appearance import blender_materials as materials
    result = {}
    def capture(name, preset, **kwargs):
        result.update(kwargs)
        return preset
    monkeypatch.setattr(materials, "build_material", capture)
    recipe = {"door_finish": "wood_oak", "seed": 1}
    body, part = object(), object()
    source = {"name": "mat_wood", "coordinate_object": body,
              "part_coordinate_object": part, "part_dimensions": [1.2, .04, .12]}
    geom = {"name": "leaf_barn_brace", "semantic": "leaf", "material": "mat_wood"}
    assert materials.material_for_geom(geom, source, {"leaf": {}}, recipe) == "wood_oak"
    assert result["source"]["coordinate_object"] is part
    assert result["source"]["grain_axis"] == "X"
    assert source["coordinate_object"] is body


@pytest.mark.parametrize("member,dimensions,expected", [
    ("leaf_batten_13", [1.4, .04, .1], "X"),
    ("leaf_batten_202", [.12, .04, 2.1], "Z"),
    ("hatch_slab", [1.2, .902, .032], "X"),
])
def test_batten_and_horizontal_hatch_grain(member, dimensions, expected, monkeypatch):
    from doorbench.appearance import blender_materials as materials
    captured = {}
    monkeypatch.setattr(materials, "build_material", lambda *a, **kwargs: captured.update(kwargs))
    anchor, part = object(), object()
    materials.material_for_geom({"name": member, "semantic": "leaf", "size": dimensions},
        {"name": "mat_wood", "coordinate_object": anchor, "part_coordinate_object": part,
         "part_dimensions": dimensions}, {"leaf": {}}, {"door_finish": "wood_oak"})
    assert captured["source"]["grain_axis"] == expected
    assert captured["source"]["coordinate_object"] is (anchor if member == "hatch_slab" else part)

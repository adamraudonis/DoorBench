"""The appearance layer must be deterministic and independent of physics/bpy."""
import copy
import json
import os
import subprocess
import sys

import pytest

from doorbench.appearance.catalog import (
    WALLS, FLOORS, DOOR_FINISHES, LIGHTING, resolve_recipe, preset_for_geom, surface_preset,
)


def spec(slab="solid_wood_oak", finish_kind="natural", color="oak_red"):
    return {"id": "db_test", "family": "swing_single", "context": "residential_interior",
            "leaf": {"slab": slab, "finish": {"kind": finish_kind, "color": color}},
            "opening": {"frame": {"material": "pine"}},
            "physics": {"mass": {"total_kg": 37.5}, "friction": .7}}


def test_catalog_surfaces_have_physical_scales_and_valid_colors():
    assert len(WALLS) >= 6 and len(FLOORS) >= 6 and len(DOOR_FINISHES) >= 8 and len(LIGHTING) >= 4
    keys = list(WALLS) + list(FLOORS) + list(DOOR_FINISHES)
    assert len(set(keys)) == len(keys)
    for key in keys:
        p = surface_preset(key)
        assert len(p["color"]) == len(p["scale_m"]) == 3
        assert all(0 <= c <= 1 for c in p["color"])
        assert all(v > 0 for v in p["scale_m"])
        assert 0 <= p["roughness"] <= 1 and p["bump_m"] >= 0
    json.dumps([WALLS, FLOORS, DOOR_FINISHES, LIGHTING])


def test_recipe_does_not_mutate_spec_or_physics():
    source = spec()
    before = copy.deepcopy(source)
    recipe = resolve_recipe(source, seed=41)
    assert source == before
    assert set(recipe) == {"schema_version", "seed", "wall", "floor", "door_finish", "lighting"}
    assert recipe == json.loads(json.dumps(recipe))
    assert recipe == resolve_recipe(source, seed=41)


def test_recipe_is_independent_of_python_hash_seed():
    command = "from doorbench.appearance.catalog import resolve_recipe; import json; print(json.dumps(resolve_recipe({'id':'stable-door','leaf':{'slab':'glass_frameless_12'}}, seed=123),sort_keys=True))"
    outputs = [subprocess.check_output([sys.executable, "-c", command], env={**os.environ, "PYTHONHASHSEED": h}, text=True)
               for h in ("11", "317")]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("slab,finish,color,expected", [
    ("solid_wood_oak", "natural", "oak_red", "wood_source"),
    ("barn_plank", "weathered", "reclaimed_barnwood", "wood_weathered"),
    ("hollow_core", "paint", "white", "paint_source"),
    ("hollow_metal_18ga", "powder_coat", "grey", "powdercoat_source"),
    ("stainless_hollow", "metal_bare", "stainless", "brushed_metal"),
    ("glass_frameless_12", "glass", "glass_clear", "glass_clear"),
    ("mirror_bypass", "glass", "mirror", "mirror"),
    ("shoji", "natural", "washi_paper", "paper_washi"),
    ("fusuma", "stain", "washi_paper", "paper_washi"),
    ("canvas_tent", "natural", "canvas", "fabric_canvas"),
    ("leather_padded", "natural", "leather", "leather"),
    ("screen_wood", "paint", "white", "paint_source"),
    ("screen_alu", "natural", "insect_screen", "brushed_metal"),
    ("strip_curtain", "natural", "pvc_flexible", "plastic_clear"),
])
def test_auto_finish_preserves_construction_family(slab, finish, color, expected):
    assert resolve_recipe(spec(slab, finish, color))["door_finish"] == expected


def test_slots_mix_independently_without_redrawing_other_slots():
    source = spec()
    base = resolve_recipe(source, 17)
    changed = resolve_recipe(source, 17, wall="wall_red_brick", door_finish="paint_sage")
    assert changed["wall"] == "wall_red_brick" and changed["door_finish"] == "paint_sage"
    assert changed["floor"] == base["floor"] and changed["lighting"] == base["lighting"]
    assert len({(r["wall"], r["floor"], r["lighting"]) for r in (resolve_recipe(source, seed) for seed in range(20))}) > 10


@pytest.mark.parametrize("slot", ["wall", "floor", "door_finish", "lighting"])
def test_bad_override_reports_slot_and_choices(slot):
    with pytest.raises(ValueError, match=f"Unknown {slot} preset"):
        resolve_recipe(spec(), **{slot: "nonexistent"})


def test_seed_validation_and_unknown_surface():
    for value in (False, 1.1, "12"):
        with pytest.raises(TypeError, match="seed"):
            resolve_recipe(spec(), seed=value)
    with pytest.raises(ValueError, match="Unknown surface"):
        surface_preset("not_a_preset")


@pytest.mark.parametrize("geom,source,expected", [
    ({"semantic": "glass", "name": "leaf_glass"}, {"name": "mat_glass"}, "glass_frosted"),
    ({"semantic": "hinge", "name": "hinge_knuckle"}, {"name": "mat_hinge", "metallic": 1}, "brushed_metal"),
    ({"semantic": "leaf", "name": "leaf_patch_top", "part_label": "Patch fitting"}, {"name": "mat_patch", "metallic": 1}, "brushed_metal"),
    ({"semantic": "lock", "name": "keyway"}, {"name": "mat_brass", "metallic": 1}, "brass_satin"),
    ({"semantic": "seal", "name": "weatherstrip"}, {"name": "mat_seal"}, "rubber_black"),
    ({"semantic": "frame", "name": "jamb"}, {"name": "mat_frame", "texture": "kitchen_wood"}, "wood_source"),
])
def test_door_override_protects_glass_fittings_and_seals(geom, source, expected):
    source_spec = spec()
    source_spec["leaf"]["glazing"] = {"material": "glass_frosted"}
    recipe = resolve_recipe(source_spec, door_finish="paint_charcoal")
    assert preset_for_geom(geom, source, source_spec, recipe) == expected


def test_mirror_is_not_recolored_by_an_opaque_door_override():
    source_spec = spec("mirror_bypass", "glass", "mirror")
    recipe = resolve_recipe(source_spec, door_finish="wood_oak")
    assert preset_for_geom({"semantic": "glass"}, {"name": "mat_glass_leaf"}, source_spec, recipe) == "mirror"


def test_shoji_lattice_and_paper_remain_distinct():
    source_spec = spec("shoji", "natural", "washi_paper")
    recipe = resolve_recipe(source_spec)
    assert preset_for_geom({"semantic": "leaf", "name": "leaf_kumiko_v1"}, {}, source_spec, recipe) == "wood_source"
    assert preset_for_geom({"semantic": "leaf", "name": "leaf_slab"}, {}, source_spec, recipe) == "paper_washi"


@pytest.mark.parametrize("slab,expected", [("screen_alu", "mesh_screen"), ("chain_link_gate", "mesh_chain_link"), ("expanded_metal_gate", "mesh_expanded")])
def test_mesh_infill_gets_cutout_but_rails_stay_solid(slab, expected):
    source_spec = spec(slab, "metal_bare", "steel")
    recipe = resolve_recipe(source_spec)
    assert preset_for_geom({"semantic": "leaf", "name": "leaf_mesh"}, {"name": "mat_mesh_infill"}, source_spec, recipe) == expected
    assert preset_for_geom({"semantic": "leaf", "name": "leaf_rail_b"}, {"name": "mat_leaf"}, source_spec, recipe) != expected


def test_blender_module_import_is_lazy():
    result = subprocess.check_output([sys.executable, "-c", "import sys; import doorbench.appearance.blender_materials; print('bpy' in sys.modules)"], text=True)
    assert result.strip() == "False"

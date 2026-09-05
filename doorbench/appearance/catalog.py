"""Deterministic *render-only* appearance recipes; no Blender dependency.

Colors are scene-linear RGB. ``scale_m`` contains characteristic procedural
feature sizes in meters, never object-normalized texture repetitions. These
are art-directed PBR starting points, not measured manufacturer BRDFs. Nothing
in this module changes a DoorBench material density, friction or geometry.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

SCHEMA_VERSION = "1.0"


def _rgb(hex_color: str) -> list[float]:
    values = [int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [round(v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4, 6) for v in values]


def _surface(kind, color, roughness, scale_m, bump_m, **kwargs):
    return dict(kind=kind, color=_rgb(color), roughness=roughness,
                scale_m=list(scale_m), bump_m=bump_m, **kwargs)


WALLS = {
    "wall_white_plaster": _surface("plaster", "EAE7E0", .84, (.012, .012, .012), .00035),
    "wall_limewash": _surface("limewash", "D3CEC2", .91, (.025, .025, .025), .0002,
                              cloud_scale_m=.55, variation=.055),
    "wall_sage_paint": _surface("paint", "9CA68F", .64, (.0015, .0015, .0015), .00007),
    "wall_concrete": _surface("concrete", "ADA9A0", .87, (.018, .018, .018), .0006,
                              cloud_scale_m=.48, variation=.16),
    "wall_red_brick": _surface("brick", "915A43", .84, (.22, .065, .1), .0025,
                               joint_m=.009, mortar_color=_rgb("B5ADA0")),
    "wall_subway_tile": _surface("tile", "E3E6DF", .22, (.20, .10, .1), .00065,
                                 joint_m=.0025, mortar_color=_rgb("A29E94"), stagger=.5),
    "wall_limestone": _surface("stone_tile", "C9BEAA", .72, (.6, .3, .1), .001,
                                joint_m=.003, mortar_color=_rgb("A99E8D"), stagger=.5),
}

FLOORS = {
    "floor_concrete": _surface("concrete", "8D8B84", .42, (.015, .015, .015), .00025,
                               cloud_scale_m=.7, variation=.15),
    "floor_terrazzo": _surface("terrazzo", "BCB8AC", .31, (.018, .018, .018), .00009,
                               chip_color=_rgb("6A726C"), chip_color_light=_rgb("E6DFCE")),
    "floor_limestone": _surface("stone_tile", "BCB09A", .5, (.6, .6, .1), .0006,
                                joint_m=.003, mortar_color=_rgb("8D8475"), stagger=0),
    "floor_porcelain": _surface("tile", "C3C8C5", .23, (.6, .6, .1), .0004,
                                joint_m=.002, mortar_color=_rgb("858A86"), stagger=0),
    "floor_oak": _surface("wood_floor", "B28A59", .4, (.008, .008, .35), .00015,
                          board_width_m=.18, board_length_m=1.35, joint_m=.001),
    "floor_slate": _surface("stone_tile", "515A5B", .63, (.4, .6, .1), .001,
                            joint_m=.003, mortar_color=_rgb("303637"), stagger=.5),
    "floor_dark_concrete": _surface("concrete", "626765", .58, (.025, .025, .025), .00035,
                                    cloud_scale_m=.8, variation=.12),
}

DOOR_FINISHES = {
    "wood_source": _surface("wood", "B18B61", .44, (.009, .009, .35), .00012, source_color=True),
    "wood_oak": _surface("wood", "B89463", .42, (.009, .009, .38), .00013),
    "wood_walnut": _surface("wood", "684A32", .36, (.006, .006, .42), .00009),
    "wood_mahogany": _surface("wood", "874B32", .33, (.007, .007, .5), .00008),
    "wood_weathered": _surface("wood", "82796A", .78, (.012, .012, .28), .0004),
    "paint_source": _surface("paint", "E4E2DC", .43, (.0012, .0012, .0012), .000045, source_color=True),
    "paint_porcelain": _surface("paint", "ECEBE4", .34, (.0012, .0012, .0012), .00004),
    "paint_charcoal": _surface("paint", "303739", .4, (.0012, .0012, .0012), .000045),
    "paint_sage": _surface("paint", "718576", .43, (.0012, .0012, .0012), .000045),
    "powdercoat_source": _surface("powdercoat", "7D8586", .49, (.00065, .00065, .00065), .000035, source_color=True),
    "brushed_metal": _surface("brushed_metal", "C1C5C7", .28, (.00012, .00012, .10), .000008, metallic=1.0, source_color=True),
    "brass_satin": _surface("brass", "C7A468", .26, (.00018, .00018, .08), .000006, metallic=1.0),
    "metal_rust": _surface("rust", "885336", .85, (.012, .012, .012), .00035, metallic=.18),
    "glass_clear": _surface("glass", "F1FAF7", .018, (.02, .02, .02), 0, ior=1.52, transmission=1.0),
    "glass_frosted": _surface("glass", "EDF3EF", .32, (.00015, .00015, .00015), .000001, ior=1.52, transmission=1.0),
    "glass_tinted": _surface("glass", "BACDC6", .025, (.02, .02, .02), 0, ior=1.52, transmission=1.0),
    "glass_wired": _surface("wired_glass", "E9F1E8", .055, (.02, .02, .02), 0, ior=1.52, transmission=1.0, wire_width_m=.00045),
    "mirror": _surface("mirror", "F2F3F3", .012, (.02, .02, .02), 0, metallic=1.0),
    "rubber_black": _surface("rubber", "242726", .8, (.0006, .0006, .0006), .000025),
    "paper_washi": _surface("paper", "E1D8BD", .86, (.00035, .00035, .025), .00006),
    "fabric_canvas": _surface("fabric", "C4BDA5", .87, (.0012, .0012, .0012), .00016, source_color=True),
    "leather": _surface("leather", "68432E", .52, (.0015, .0015, .0015), .0001, source_color=True),
    "plastic_source": _surface("plastic", "D4D7CF", .39, (.0008, .0008, .0008), .00002, source_color=True),
    "plastic_clear": _surface("glass", "E5F4EC", .09, (.006, .006, .006), .000001, ior=1.46, transmission=.93),
    "mesh_screen": _surface("mesh", "878D8B", .52, (.0015, .0015, .0015), 0, metallic=.8, wire_width_m=.00025),
    "mesh_chain_link": _surface("mesh", "ACAFAC", .45, (.05, .05, .05), 0, metallic=1.0, wire_width_m=.003, rotation_deg=45),
    "mesh_expanded": _surface("mesh", "969D99", .53, (.035, .015, .025), 0, metallic=.85, wire_width_m=.002, rotation_deg=45),
}

LIGHTING = {
    "daylight": dict(kind="daylight", world_color=_rgb("D8E5F5"), world_strength=.35,
                     key_energy_w=1800, key_color=_rgb("FFFDF8"), key_size_m=4.0, key_location_m=[-3, -4, 6],
                     fill_energy_w=650, fill_color=_rgb("E8F0FF"), fill_size_m=5.0, fill_location_m=[4, 1, 4],
                     sun_energy=1.1, sun_angle_rad=.12, sun_rotation_deg=[25, -25, -30]),
    "overcast": dict(kind="overcast", world_color=_rgb("DCE5E9"), world_strength=.55,
                     key_energy_w=1600, key_color=_rgb("EDF3F5"), key_size_m=7.0, key_location_m=[-2, -3, 6],
                     fill_energy_w=500, fill_color=_rgb("E1E7EB"), fill_size_m=6.0, fill_location_m=[4, 2, 5],
                     sun_energy=0, sun_angle_rad=.3, sun_rotation_deg=[0, 0, 0]),
    "warm_interior": dict(kind="interior", world_color=_rgb("D8CDBB"), world_strength=.16,
                          key_energy_w=950, key_color=_rgb("FFD9AE"), key_size_m=2.5, key_location_m=[-2, -2, 4],
                          fill_energy_w=500, fill_color=_rgb("E4EAF3"), fill_size_m=3.5, fill_location_m=[3, 1, 4],
                          sun_energy=0, sun_angle_rad=.1, sun_rotation_deg=[0, 0, 0]),
    "warehouse": dict(kind="industrial", world_color=_rgb("CBD7DF"), world_strength=.22,
                      key_energy_w=1800, key_color=_rgb("EBF4FA"), key_size_m=3.0, key_location_m=[-3, -2, 7],
                      fill_energy_w=800, fill_color=_rgb("F5E8D3"), fill_size_m=4.0, fill_location_m=[4, 3, 6],
                      sun_energy=0, sun_angle_rad=.1, sun_rotation_deg=[0, 0, 0]),
}


def stable_seed(*parts) -> int:
    """Portable seed unaffected by interpreter hash randomization or process order."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _auto_finish(spec):
    leaf = spec.get("leaf", {})
    slab = leaf.get("slab", "")
    finish = leaf.get("finish", {})
    color = str(finish.get("color", ""))
    if "mirror" in slab or color == "mirror": return "mirror"
    if slab in ("shoji", "fusuma") or "paper" in color: return "paper_washi"
    if "canvas" in slab: return "fabric_canvas"
    if "leather" in slab: return "leather"
    if slab in ("strip_curtain", "pet_flap_pvc", "pet_flap_acrylic", "polycarbonate_panel"):
        return "plastic_clear"
    if slab.startswith(("glass_frameless", "storefront", "patio_slider", "revolving_wing", "storm_alu")):
        return "glass_clear"
    if slab == "screen_wood": return "paint_source"
    if slab.startswith("screen") or "gate" in slab or slab in ("steel_bar_grille", "turnstile_arm"):
        return "powdercoat_source" if finish.get("kind") in ("paint", "powder_coat") else "brushed_metal"
    if finish.get("kind") == "powder_coat": return "powdercoat_source"
    if finish.get("kind") == "paint": return "paint_source"
    if finish.get("kind") == "metal_bare" or slab.startswith(("stainless", "steel", "hollow_metal")):
        return "brushed_metal"
    if finish.get("kind") == "glass": return "glass_clear"
    if slab.startswith(("solid_wood", "barn", "cedar", "bamboo", "louver_wood", "attic", "cellar", "garage_wood")) or finish.get("texture") and "wood" in str(finish["texture"]):
        return "wood_weathered" if finish.get("kind") == "weathered" else "wood_source"
    return "plastic_source" if slab.startswith(("upvc", "baby_gate")) else "paint_source"


def resolve_recipe(spec, seed=0, wall=None, floor=None, door_finish=None, lighting=None):
    """Resolve independently selectable presets without mutating the input spec.

    Defaults preserve source finish color and material meaning. Explicit door
    overrides are artistic choices; ``preset_for_geom`` still protects optical
    glass/mirror, seals and hardware from being painted like the door slab.
    """
    if not isinstance(spec, Mapping): raise TypeError("spec must be a mapping")
    if not isinstance(seed, int) or isinstance(seed, bool): raise TypeError("seed must be an integer")
    identity = spec.get("id") or {k: spec.get(k) for k in ("family", "context", "leaf")}
    selected = {}
    for slot, override, registry in (("wall", wall, WALLS), ("floor", floor, FLOORS),
                                     ("door_finish", door_finish, DOOR_FINISHES), ("lighting", lighting, LIGHTING)):
        if override is None or override == "auto":
            chosen = _auto_finish(spec) if slot == "door_finish" else sorted(registry)[stable_seed(SCHEMA_VERSION, identity, seed, slot) % len(registry)]
        elif isinstance(override, str) and override in registry: chosen = override
        else: raise ValueError(f"Unknown {slot} preset {override!r}; choose auto or one of {', '.join(sorted(registry))}")
        selected[slot] = chosen
    return dict(schema_version=SCHEMA_VERSION, seed=seed, **selected)


def surface_preset(preset_id):
    """Read a surface preset; callers must not mutate the shared catalogue."""
    for registry in (WALLS, FLOORS, DOOR_FINISHES):
        if preset_id in registry: return registry[preset_id]
    raise ValueError(f"Unknown surface preset {preset_id!r}")


def _source_finish(source, hint=""):
    text = " ".join(str(source.get(k, "")) for k in ("name", "texture")) + " " + hint
    text = text.lower()
    if "mirror" in text: return "mirror"
    if any(x in text for x in ("rubber", "gasket", "neoprene", "silicone", "seal")): return "rubber_black"
    if "brass" in text or "bronze" in text: return "brass_satin"
    if "rust" in text: return "metal_rust"
    if "glass" in text:
        return "glass_frosted" if "frost" in text else "glass_tinted" if "tint" in text else "glass_wired" if "wired" in text else "glass_clear"
    if any(x in text for x in ("paper", "washi")): return "paper_washi"
    if "leather" in text: return "leather"
    if "canvas" in text or "fabric" in text: return "fabric_canvas"
    if any(x in text for x in ("wood", "oak", "pine", "cedar", "walnut", "mahogany", "hinoki", "plywood", "teak", "maple", "bamboo", "fir")): return "wood_source"
    if source.get("metallic", 0) >= .5 or any(x in text for x in ("steel", "stainless", "aluminum", "metal")): return "brushed_metal"
    if source.get("transparent") or (source.get("rgba") or [1, 1, 1, 1])[3] < .8: return "plastic_clear"
    return "paint_source"


def preset_for_geom(geom, source_material, spec, recipe):
    """Pure selector for generated geom/material dictionaries.

    Perforated mesh is a render cutout on the existing surface, not newly
    simulated wire geometry. Glazing selection follows its own source, never
    the leaf's opaque paint override. Frame and fittings retain source meaning.
    """
    source = source_material or {}
    semantic = str(geom.get("semantic", "")).lower()
    name = str(geom.get("name", "")).lower()
    label = str(geom.get("part_label", geom.get("label", ""))).lower()
    leaf = spec.get("leaf", {})
    slab = str(leaf.get("slab", ""))
    material_name = str(source.get("name", geom.get("material", ""))).lower()
    if semantic == "wall": return recipe["wall"]
    if semantic == "floor": return recipe["floor"]
    if "mesh_infill" in material_name or name.endswith("_mesh"):
        return "mesh_chain_link" if "chain_link" in slab else "mesh_expanded" if "expanded" in slab else "mesh_screen"
    if semantic in ("seal", "gasket") or any(x in name for x in ("gasket", "seal_", "rubber")):
        return "rubber_black"
    if semantic == "glass" or "glass_leaf" in material_name:
        if "mirror" in slab or "mirror" in material_name: return "mirror"
        glazing = leaf.get("glazing") or {}
        return _source_finish(source, glazing.get("material", "glass_clear"))
    if semantic in ("operator", "hinge", "latch", "lock", "closer", "track", "mechanism"):
        return _source_finish(source)
    if semantic == "frame":
        return _source_finish(source, spec.get("opening", {}).get("frame", {}).get("material", ""))
    # Fittings frequently use semantic=leaf despite being a distinct metal.
    if any(x in name + " " + label for x in ("patch", "rivet", "kick plate", "kick_plate", "armor", "hinge", "bolt", "keeper", "rail_arm")):
        return _source_finish(source)
    if semantic == "leaf":
        # Shoji's kumiko is timber, its infill is paper; finish overrides must
        # not turn every frame strip into paper or glass.
        if "kumiko" in name or "kumiko" in label: return "wood_source"
        if source.get("metallic", 0) >= .5 and not material_name.startswith(("mat_leaf", "mat_curtain", "mat_door", "mat_panel")):
            return _source_finish(source)
        if _auto_finish(spec) in ("mirror", "glass_clear", "plastic_clear", "paper_washi", "fabric_canvas", "leather"):
            return _auto_finish(spec)
        return recipe["door_finish"]
    return _source_finish(source)

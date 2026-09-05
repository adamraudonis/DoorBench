"""Procedural Blender PBR materials for the optional appearance renderer.

Only rendering datablocks are touched. Mesh openings, bevels, correct glazing
thickness, lighting and sufficient samples still matter: shading cannot repair
an opaque slab behind glass or manufacture a missing mechanism. Procedural materials
are art-directed approximations; an explicit CC0 map library enables scanned
surfaces. Scans are appearance references, not manufacturer BRDF certification.

Coordinates are meters. Prefer ``source['coordinate_object']`` pointing to a
unit-scale body empty: textures then remain attached while the door articulates.
Without an anchor, world Position provides continuous, physically scaled wall
and floor textures. No bpy objects are held in a module-level cache.
"""
from __future__ import annotations

import hashlib
import json
import math

from .catalog import preset_for_geom, stable_seed, surface_preset
from .textures import load_texture_library

_TEXTURE_LIBRARY = None  # JSON only; never retain Blender datablock references.


def configure_texture_library(library=None):
    """Select an optional validated CC0 library for this render worker/job.

    Map SHA256 checks use stat-aware caching, so tampering after job preparation
    is detected while unchanged maps are not reread for every door. None resets
    the worker to its entirely procedural fallback.
    """
    global _TEXTURE_LIBRARY
    _TEXTURE_LIBRARY = load_texture_library(library) if library is not None else None
    return _TEXTURE_LIBRARY


def _texture_asset(preset_id, source):
    if _TEXTURE_LIBRARY is None: return None
    asset_id = _TEXTURE_LIBRARY.get("preset_assets", {}).get(preset_id)
    hint = " ".join(str(source.get(k, "")) for k in ("name", "texture")).lower()
    if preset_id == "wood_source" and "walnut" in hint:
        asset_id = "walnut_veneer"
    return _TEXTURE_LIBRARY["assets"].get(asset_id)


def _anchor_rotation(anchor):
    # Newly parented render objects may not have evaluated matrix_world yet.
    matrix = anchor.parent.matrix_world @ anchor.matrix_parent_inverse @ anchor.matrix_basis if anchor.parent is not None else anchor.matrix_world
    return matrix.to_quaternion()


def _rgba(color, factor=1.0):
    return tuple(max(0.0, min(1.0, float(c) * factor)) for c in color[:3]) + (1.0,)


class _Graph:
    def __init__(self, material, slot, source, seed):
        import bpy
        self.tree = material.node_tree
        self.nodes, self.links = self.tree.nodes, self.tree.links
        self.nodes.clear()
        self.slot = slot
        self.seed = seed
        self.grain_axis = source.get("grain_axis", "Z")
        self.source = source
        self._planar_frame = None
        self.i = 0
        self.geometry = self.node("ShaderNodeNewGeometry", "Surface in world meters")
        anchor = source.get("coordinate_object")
        if isinstance(anchor, str):
            obj = bpy.data.objects.get(anchor)
            if obj is None:
                raise ValueError(f"Appearance coordinate object does not exist: {anchor!r}")
            anchor = obj
        if anchor is not None:
            coord = self.node("ShaderNodeTexCoord", "Body-attached meter coordinates")
            coord.object = anchor
            self.coords = coord.outputs["Object"]
        else:
            self.coords = self.geometry.outputs["Position"]
        self.bsdf = self.node("ShaderNodeBsdfPrincipled", "Physically based surface")
        self.output = self.node("ShaderNodeOutputMaterial", "Surface output")
        self.links.new(self.bsdf.outputs["BSDF"], self.output.inputs["Surface"])

    def node(self, type_name, label):
        node = self.nodes.new(type_name)
        node.label = label
        node.location = ((self.i % 6) * 240 - 1100, -(self.i // 6) * 250)
        self.i += 1
        return node

    def math(self, operation, a, b=None, label=None):
        node = self.node("ShaderNodeMath", label or operation)
        node.operation = operation
        for index, value in enumerate((a, b)):
            if value is None: continue
            if hasattr(value, "is_output"): self.links.new(value, node.inputs[index])
            else: node.inputs[index].default_value = value
        return node.outputs[0]

    def mix(self, a, b, factor, mode="MIX", label="Color variation"):
        node = self.node("ShaderNodeMixRGB", label)
        node.blend_type = mode
        for index, value in enumerate((factor, a, b)):
            if hasattr(value, "is_output"): self.links.new(value, node.inputs[index])
            else: node.inputs[index].default_value = value
        return node.outputs[0]

    def scaled(self, scale_m, coords=None, offset=True):
        node = self.node("ShaderNodeVectorMath", "Meter scale: " + str(tuple(scale_m)))
        node.operation = "MULTIPLY"
        self.links.new(coords or self.coords, node.inputs[0])
        node.inputs[1].default_value = tuple(1 / max(float(v), 1e-7) for v in scale_m)
        if not offset: return node.outputs[0]
        shift = self.node("ShaderNodeVectorMath", "Deterministic texture offset")
        shift.operation = "ADD"
        self.links.new(node.outputs[0], shift.inputs[0])
        shift.inputs[1].default_value = tuple((stable_seed(self.seed, i) % 10000) / 113.0 for i in range(3))
        return shift.outputs[0]

    def noise(self, scale_m, detail=3, coords=None, label="Surface microstructure"):
        node = self.node("ShaderNodeTexNoise", label)
        node.noise_dimensions = "3D"
        self.links.new(self.scaled(scale_m, coords), node.inputs["Vector"])
        node.inputs["Scale"].default_value = 1
        node.inputs["Detail"].default_value = detail
        node.inputs["Roughness"].default_value = .7
        return node.outputs["Fac"]

    def ramp(self, factor, dark, light, low=.18, high=.82):
        node = self.node("ShaderNodeValToRGB", "Calibrated color range")
        node.color_ramp.elements[0].position = low
        node.color_ramp.elements[0].color = dark
        node.color_ramp.elements[1].position = high
        node.color_ramp.elements[1].color = light
        self.links.new(factor, node.inputs["Fac"])
        return node.outputs["Color"]

    def roughness(self, noise, roughness, spread=.07):
        node = self.node("ShaderNodeMapRange", "Microsurface roughness variation")
        node.inputs["From Min"].default_value = 0
        node.inputs["From Max"].default_value = 1
        node.inputs["To Min"].default_value = max(.008, roughness-spread)
        node.inputs["To Max"].default_value = min(.98, roughness+spread)
        self.links.new(noise, node.inputs["Value"])
        self.links.new(node.outputs["Result"], self.bsdf.inputs["Roughness"])

    def bump(self, height, distance, strength=.4):
        if distance <= 0: return
        node = self.node("ShaderNodeBump", f"Microrelief, {distance:g} meters")
        node.inputs["Distance"].default_value = distance
        node.inputs["Strength"].default_value = strength
        self.links.new(height, node.inputs["Height"])
        self.links.new(node.outputs["Normal"], self.bsdf.inputs["Normal"])

    def planar_frame(self):
        """Face-aware box projection and its live world tangent/bitangent.

        Dominant face normal is evaluated in the same anchor coordinates as
        the map, so XY hatches, XZ doors and YZ edges all receive full 2D maps.
        Object→world axes remain dynamic when a saved scene is articulated.
        Hard box seams occur at face boundaries; this does not synthesize a
        physically separate anatomical end-grain scan on cut timber edges.
        """
        if self._planar_frame is not None: return self._planar_frame
        import bpy
        from mathutils import Quaternion, Vector
        source = self.source
        anchor, part = source.get("coordinate_object"), source.get("part_coordinate_object")
        if isinstance(anchor, str): anchor = bpy.data.objects.get(anchor)
        if isinstance(part, str): part = bpy.data.objects.get(part)
        normal = self.geometry.outputs["Normal"]
        axes = [Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))]
        dynamic = anchor is not None and part is not None
        if dynamic:
            if anchor is part: relative = Quaternion()
            elif anchor is part.parent: relative = part.matrix_basis.to_quaternion().inverted()
            else: relative = _anchor_rotation(part).inverted() @ _anchor_rotation(anchor)
            axes = [relative @ axis for axis in axes]
            transform = self.node("ShaderNodeVectorTransform", "Live face normal in part coordinates")
            transform.vector_type = "NORMAL"
            transform.convert_from, transform.convert_to = "WORLD", "OBJECT"
            self.links.new(normal, transform.inputs["Vector"])
            normal = transform.outputs["Vector"]
        elif anchor is not None:
            axes = [_anchor_rotation(anchor) @ axis for axis in axes]
        face, world_axes = [], []
        for axis_name, vector in zip("XYZ", axes):
            dot = self.node("ShaderNodeVectorMath", f"Face normal along anchor {axis_name}")
            dot.operation = "DOT_PRODUCT"
            self.links.new(normal, dot.inputs[0])
            dot.inputs[1].default_value = tuple(vector)
            face.append(self.math("ABSOLUTE", dot.outputs["Value"]))
            if dynamic:
                transform = self.node("ShaderNodeVectorTransform", f"Live anchor {axis_name} axis to world")
                transform.vector_type = "VECTOR"
                transform.convert_from, transform.convert_to = "OBJECT", "WORLD"
                transform.inputs["Vector"].default_value = tuple(vector)
                world_axes.append(transform.outputs["Vector"])
            else:
                constant = self.node("ShaderNodeCombineXYZ", f"World anchor {axis_name} axis")
                for i in range(3): constant.inputs[i].default_value = vector[i]
                world_axes.append(constant.outputs[0])
        use_yz = self.math("MULTIPLY", self.math("GREATER_THAN", face[0], face[1]),
                           self.math("GREATER_THAN", face[0], face[2]), "Face projects to YZ")
        use_xz = self.math("MULTIPLY", self.math("SUBTRACT", 1, use_yz),
                           self.math("GREATER_THAN", face[1], face[2]), "Face projects to XZ")
        vertical = self.math("MAXIMUM", use_yz, use_xz)
        sep = self.node("ShaderNodeSeparateXYZ", "Anchor meter coordinates")
        self.links.new(self.coords, sep.inputs[0])
        planes = {}
        for label, order in (("XY", "XYZ"), ("XZ", "XZY"), ("YZ", "YZX")):
            node = self.node("ShaderNodeCombineXYZ", f"Full two-dimensional {label} face projection")
            for i, axis in enumerate(order): self.links.new(sep.outputs[axis], node.inputs[i])
            planes[label] = node.outputs[0]
        coords = self.mix(planes["XY"], planes["XZ"], use_xz, label="Horizontal or front scan plane")
        coords = self.mix(coords, planes["YZ"], use_yz, label="Side scan plane")
        tangent = self.mix(world_axes[0], world_axes[1], use_yz, label="Projected U axis in world")
        bitangent = self.mix(world_axes[1], world_axes[2], vertical, label="Projected V axis in world")
        self._planar_frame = (coords, tangent, bitangent)
        return self._planar_frame

    def plane(self):
        return self.planar_frame()[0]

    def brick(self, color, width, height, joint, mortar, stagger=.5):
        node = self.node("ShaderNodeTexBrick", "Masonry or board joints in meters")
        self.links.new(self.plane(), node.inputs["Vector"])
        node.inputs["Color1"].default_value = _rgba(color, .87)
        node.inputs["Color2"].default_value = _rgba(color, 1.10)
        node.inputs["Mortar"].default_value = _rgba(mortar)
        node.inputs["Scale"].default_value = 1.0
        node.inputs["Brick Width"].default_value = width
        node.inputs["Row Height"].default_value = height
        node.inputs["Mortar Size"].default_value = joint / 2
        node.inputs["Mortar Smooth"].default_value = joint / 5
        node.inputs["Bias"].default_value = 0
        node.offset = stagger
        node.offset_frequency = 2
        node.squash = 1
        node.squash_frequency = 2
        return node

    def grid_mask(self, period_m, wire_m, rotation=0):
        coords = self.plane()
        if rotation:
            mapping = self.node("ShaderNodeVectorRotate", "Diamond mesh orientation")
            mapping.rotation_type = "AXIS_ANGLE"
            mapping.inputs["Axis"].default_value = (0, 0, 1)
            mapping.inputs["Angle"].default_value = math.radians(rotation)
            self.links.new(coords, mapping.inputs["Vector"])
            coords = mapping.outputs["Vector"]
        sep = self.node("ShaderNodeSeparateXYZ", "Wire spacing coordinates")
        self.links.new(coords, sep.inputs[0])
        masks = []
        for axis, period in zip(("X", "Y"), period_m):
            # Periodic distance to a wire centerline. Width is in actual meters.
            phase = self.math("MULTIPLY", sep.outputs[axis], math.pi / period)
            wave = self.math("ABSOLUTE", self.math("SINE", phase))
            masks.append(self.math("LESS_THAN", wave, math.sin(math.pi * min(wire_m / period, .8) / 2)))
        return self.math("MAXIMUM", *masks, label="Opaque wire / transparent aperture")


def _wood(graph, preset, color):
    scale = preset["scale_m"]
    if graph.slot == "floor": scale = (.38, .007, .007)
    elif graph.grain_axis == "X": scale = (scale[2], scale[1], scale[0])
    grain = graph.noise(scale, 4, label="Longitudinal grain bundles")
    pores = graph.noise(tuple(v*.19 for v in scale), 2, label="Fine directional wood pores")
    combined = graph.mix(grain, pores, .24, label="Large and fine grain")
    colored = graph.ramp(combined, _rgba(color, .56), _rgba(color, 1.19))
    # Broader distorted bands break the uniform pinstripe appearance of
    # anisotropic noise alone. This approximates cut growth-ring figure; it
    # does not claim botanical reconstruction of a particular wood species.
    growth = graph.node("ShaderNodeTexWave", "Undulating cut growth-ring figure")
    growth.wave_type = "BANDS"
    growth.bands_direction = "Y" if graph.slot == "floor" else "Z" if graph.grain_axis == "X" else "X"
    growth_scale = (.8, .13, .13) if graph.slot == "floor" or graph.grain_axis == "X" else (.13, .13, .8)
    graph.links.new(graph.scaled(growth_scale), growth.inputs["Vector"])
    growth.inputs["Scale"].default_value = 1.6
    growth.inputs["Distortion"].default_value = 5
    growth.inputs["Detail"].default_value = 3
    growth.inputs["Detail Scale"].default_value = 1.2
    growth.inputs["Detail Roughness"].default_value = .65
    figure = graph.ramp(growth.outputs["Fac"], (.52, .52, .52, 1), (1.1, 1.1, 1.1, 1), .25, .65)
    colored = graph.mix(colored, figure, .42, "MULTIPLY", "Natural growth figure over fine fibers")
    if preset["kind"] == "wood_floor":
        boards = graph.brick([1, 1, 1], preset["board_length_m"], preset["board_width_m"],
                             preset["joint_m"], [.10, .065, .035], .47)
        colored = graph.mix(colored, boards.outputs["Color"], 1, "MULTIPLY", "Board-to-board finish variation")
        height = graph.math("ADD", graph.math("MULTIPLY", pores, .18), graph.math("MULTIPLY", boards.outputs["Fac"], -.82))
        graph.bump(height, .0006, .6)
    else: graph.bump(pores, preset["bump_m"], .36)
    graph.links.new(colored, graph.bsdf.inputs["Base Color"])
    graph.roughness(grain, preset["roughness"], .045)
    graph.bsdf.inputs["Coat Weight"].default_value = .13 if preset["roughness"] < .6 else .025
    graph.bsdf.inputs["Coat Roughness"].default_value = .32
    graph.bsdf.inputs["Anisotropic"].default_value = .15


def _mineral(graph, preset, color):
    kind = preset["kind"]
    scale = preset["scale_m"]
    micro = graph.noise(scale, 3, label="Mineral pores and trowel texture")
    cloud_scale = preset.get("cloud_scale_m", .35)
    cloud = graph.noise((cloud_scale,)*3, 3, label="Broad mineral color variation")
    spread = preset.get("variation", .10)
    colored = graph.ramp(cloud, _rgba(color, 1-spread), _rgba(color, 1+spread))
    height = micro
    if kind in ("brick", "tile", "stone_tile"):
        brick = graph.brick(color, scale[0], scale[1], preset["joint_m"], preset["mortar_color"], preset.get("stagger", .5))
        colored = graph.mix(brick.outputs["Color"], graph.ramp(micro, ( .84,.84,.84,1), (1.05,1.05,1.05,1)), .65, "MULTIPLY")
        height = graph.math("SUBTRACT", graph.math("MULTIPLY", micro, .10), brick.outputs["Fac"], "Recessed mortar, unbroken tile faces")
        graph.bump(height, preset["bump_m"], .7)
        if kind == "tile":
            graph.bsdf.inputs["Coat Weight"].default_value = .25
            graph.bsdf.inputs["Coat Roughness"].default_value = .16
    elif kind == "terrazzo":
        vor = graph.node("ShaderNodeTexVoronoi", "Embedded stone chip boundaries")
        vor.feature = "DISTANCE_TO_EDGE"
        graph.links.new(graph.scaled(scale), vor.inputs["Vector"])
        vor.inputs["Scale"].default_value = 1
        chip = graph.math("GREATER_THAN", vor.outputs["Distance"], .085)
        colors = graph.ramp(micro, _rgba(preset["chip_color"]), _rgba(preset["chip_color_light"]))
        colored = graph.mix(colored, colors, chip, label="Stone aggregate in polished binder")
        graph.bump(micro, preset["bump_m"], .2)
        graph.bsdf.inputs["Coat Weight"].default_value = .1
    else:
        graph.bump(height, preset["bump_m"], .45)
    graph.links.new(colored, graph.bsdf.inputs["Base Color"])
    graph.roughness(micro, preset["roughness"], .045 if kind in ("tile", "terrazzo") else .08)


def _optical(graph, preset, color):
    kind = preset["kind"]
    graph.bsdf.inputs["Base Color"].default_value = _rgba(color)
    graph.bsdf.inputs["Roughness"].default_value = preset["roughness"]
    graph.bsdf.inputs["IOR"].default_value = preset.get("ior", 1.52)
    graph.bsdf.inputs["Alpha"].default_value = 1.0
    graph.bsdf.inputs["Metallic"].default_value = 1 if kind == "mirror" else 0
    graph.bsdf.inputs["Transmission Weight"].default_value = 0 if kind == "mirror" else preset.get("transmission", 1)
    if kind == "wired_glass":
        # A shading representation of embedded wire; it has no wire depth or
        # altered fracture behavior. Explicit wire geometry is needed for macro shots.
        mask = graph.grid_mask(preset["scale_m"][:2], preset["wire_width_m"])
        glass_fraction = graph.math("SUBTRACT", 1, mask)
        graph.links.new(glass_fraction, graph.bsdf.inputs["Transmission Weight"])
        graph.links.new(mask, graph.bsdf.inputs["Metallic"])
        graph.links.new(graph.mix(_rgba(color), (.12, .13, .12, 1), mask), graph.bsdf.inputs["Base Color"])
    if preset["bump_m"]:
        micro = graph.noise(preset["scale_m"], 2, label="Microscopic etched finish")
        graph.bump(micro, preset["bump_m"], .15)


def _fabric(graph, preset, color):
    micro = graph.noise(preset["scale_m"], 2, label="Fibers or grain")
    colored = graph.ramp(micro, _rgba(color, .80), _rgba(color, 1.09))
    graph.links.new(colored, graph.bsdf.inputs["Base Color"])
    graph.bump(micro, preset["bump_m"], .4)
    graph.roughness(micro, preset["roughness"], .04)
    if preset["kind"] == "fabric":
        weave = graph.grid_mask(preset["scale_m"][:2], preset["scale_m"][0] * .42)
        graph.bump(graph.math("MULTIPLY", weave, micro), preset["bump_m"], .45)
        graph.bsdf.inputs["Sheen Weight"].default_value = .28
    elif preset["kind"] == "paper":
        translucent = graph.node("ShaderNodeBsdfTranslucent", "Light through paper fibers")
        translucent.inputs["Color"].default_value = _rgba(color)
        mix = graph.node("ShaderNodeMixShader", "Paper's diffuse transmission")
        mix.inputs[0].default_value = .18
        graph.links.new(graph.bsdf.outputs["BSDF"], mix.inputs[1])
        graph.links.new(translucent.outputs[0], mix.inputs[2])
        graph.links.new(mix.outputs[0], graph.output.inputs["Surface"])
    elif preset["kind"] == "leather":
        graph.bsdf.inputs["Coat Weight"].default_value = .07



def _scanned(graph, asset, preset, source):
    """Real diffuse/GL-normal/roughness maps on meter-scaled planar coordinates.

    Normal maps are transformed through the same projected axes as the color
    map, avoiding the unrelated primitive UV tangent basis. The projected frame
    transforms from part-local space to world dynamically, following all poses.
    Each face uses a nonsingular XY, XZ or YZ projection, including horizontal
    hatch leaves and thin slab edges.
    """
    import bpy
    coords = graph.scaled((*asset["scale_m"], 1), graph.plane())
    rotation = math.radians(asset.get("rotation_deg", 0) + (90 if graph.grain_axis == "X" and graph.slot != "floor" else 0))
    if rotation:
        rotate = graph.node("ShaderNodeVectorRotate", "Scan orientation in projected plane")
        rotate.rotation_type = "AXIS_ANGLE"
        rotate.inputs["Axis"].default_value = (0, 0, 1)
        rotate.inputs["Angle"].default_value = rotation
        graph.links.new(coords, rotate.inputs["Vector"])
        coords = rotate.outputs["Vector"]
    maps = {}
    for role, entry in asset["maps"].items():
        if role not in ("diffuse", "normal", "roughness"): continue
        img = bpy.data.images.load(entry["path"], check_existing=True)
        if img.get("doorbench_map_sha256") not in (None, entry["sha256"]): img.reload()
        img.colorspace_settings.name = entry["colorspace"]
        img["doorbench_map_sha256"] = entry["sha256"]
        node = graph.node("ShaderNodeTexImage", f"Poly Haven {asset['id']} / {role}")
        node.image = img
        node.interpolation = "Linear"
        node.extension = "REPEAT"
        graph.links.new(coords, node.inputs["Vector"])
        maps[role] = node.outputs["Color"]
    color = maps["diffuse"]
    # Preserve deliberately dark concrete/neutral limewash art direction. The
    # natural wood presets use the scan's actual albedo rather than painting
    # fake high-contrast grain onto a uniform base.
    if preset["kind"] == "plaster":
        color = graph.mix(color, _rgba(preset["color"]), .8, label="Clean plaster with restrained scan albedo variation")
    elif preset["kind"] == "limewash":
        color = graph.mix(color, _rgba(preset["color"], 1.2), .3, "MULTIPLY", "Subtle limewash tint")
    elif preset["kind"] == "concrete" and preset["color"][0] < .15:
        color = graph.mix(color, (.55, .58, .57, 1), 1, "MULTIPLY", "Dark concrete tone")
    graph.links.new(color, graph.bsdf.inputs["Base Color"])
    graph.links.new(maps["roughness"], graph.bsdf.inputs["Roughness"])
    if preset["kind"] in ("wood", "wood_floor"):
        graph.bsdf.inputs["Coat Weight"].default_value = .12
        graph.bsdf.inputs["Coat Roughness"].default_value = .3
    normal = graph.node("ShaderNodeVectorMath", "Decode OpenGL normal from 0..1 to -1..1")
    normal.operation = "MULTIPLY_ADD"
    graph.links.new(maps["normal"], normal.inputs[0])
    normal.inputs[1].default_value = (2, 2, 2)
    normal.inputs[2].default_value = (-1, -1, -1)
    sep = graph.node("ShaderNodeSeparateXYZ", "Projected scan normal components")
    graph.links.new(normal.outputs[0], sep.inputs[0])
    _, tangent_socket, bitangent_socket = graph.planar_frame()
    if rotation:
        # UV rotation changes the normal-map tangent frame by the inverse
        # angle. Keep this vector math in the shader for live articulation.
        def rotate_component(a, sa, b, sb, label):
            terms = []
            for vector, factor in ((a, sa), (b, sb)):
                node = graph.node("ShaderNodeVectorMath", label)
                node.operation = "SCALE"
                graph.links.new(vector, node.inputs[0])
                node.inputs["Scale"].default_value = factor
                terms.append(node.outputs[0])
            node = graph.node("ShaderNodeVectorMath", label)
            node.operation = "ADD"
            graph.links.new(terms[0], node.inputs[0])
            graph.links.new(terms[1], node.inputs[1])
            return node.outputs[0]
        tangent_socket, bitangent_socket = (
            rotate_component(tangent_socket, math.cos(rotation), bitangent_socket, -math.sin(rotation), "Rotated live scan U axis"),
            rotate_component(tangent_socket, math.sin(rotation), bitangent_socket, math.cos(rotation), "Rotated live scan V axis"))
    # Project tangent to the actual surface; this also handles slightly curved
    # and bevel faces without a normal discontinuity from world-space mixing.
    tnode = graph.node("ShaderNodeVectorMath", "Scan tangent perpendicular to face normal")
    tnode.operation = "CROSS_PRODUCT"
    graph.links.new(bitangent_socket, tnode.inputs[0])
    graph.links.new(graph.geometry.outputs["Normal"], tnode.inputs[1])
    tnorm = graph.node("ShaderNodeVectorMath", "Unit scan tangent")
    tnorm.operation = "NORMALIZE"
    graph.links.new(tnode.outputs[0], tnorm.inputs[0])
    desired = tangent_socket
    dot = graph.node("ShaderNodeVectorMath", "Keep tangent aligned with increasing U")
    dot.operation = "DOT_PRODUCT"
    graph.links.new(tnorm.outputs[0], dot.inputs[0])
    graph.links.new(desired, dot.inputs[1])
    sign = graph.math("SIGN", dot.outputs["Value"])
    components = []
    for axis, vector in (("X", tnorm.outputs[0]), ("Y", bitangent_socket), ("Z", graph.geometry.outputs["Normal"])):
        mul = graph.node("ShaderNodeVectorMath", f"World scan normal {axis}")
        mul.operation = "SCALE"
        if hasattr(vector, "is_output"): graph.links.new(vector, mul.inputs[0])
        else: mul.inputs[0].default_value = vector
        weight = sep.outputs[axis]
        if axis != "Z": weight = graph.math("MULTIPLY", weight, (.2 if preset["kind"] == "plaster" else asset.get("normal_strength", .6)))
        if axis == "X": weight = graph.math("MULTIPLY", weight, sign)
        graph.links.new(weight, mul.inputs["Scale"])
        components.append(mul.outputs[0])
    add = graph.node("ShaderNodeVectorMath", "Combine tangent-space normal in world frame")
    add.operation = "ADD"
    graph.links.new(components[0], add.inputs[0])
    graph.links.new(components[1], add.inputs[1])
    add2 = graph.node("ShaderNodeVectorMath", "Add geometric normal")
    add2.operation = "ADD"
    graph.links.new(add.outputs[0], add2.inputs[0])
    graph.links.new(components[2], add2.inputs[1])
    final = graph.node("ShaderNodeVectorMath", "Normalized PBR scan normal")
    final.operation = "NORMALIZE"
    graph.links.new(add2.outputs[0], final.inputs[0])
    graph.links.new(final.outputs[0], graph.bsdf.inputs["Normal"])


def build_material(name, preset_id, *, slot="door", source=None, seed=0):
    """Create/reuse a bpy Material with scene-linear, physically scaled nodes.

    ``source`` is a model.json material dictionary; optional coordinate_object
    is a Blender object or its name. Object identity is cached by name only.
    Unit-scale anchors and meter-sized geometry are required for stated scales.
    """
    import bpy
    preset = surface_preset(preset_id)
    source = dict(source or {})
    anchor = source.get("coordinate_object")
    serial_source = {k: source[k] for k in ("rgba", "roughness", "metallic", "emissive", "grain_axis") if k in source}
    serial_source["coordinate_object"] = anchor if isinstance(anchor, str) or anchor is None else anchor.name
    asset = _texture_asset(preset_id, source)
    part = source.get("part_coordinate_object")
    if isinstance(part, str): part = bpy.data.objects.get(part)
    if part is not None:
        serial_source["part_local_rotation"] = list(part.matrix_basis.to_quaternion())
        serial_source["anchor_is_part"] = anchor is part or anchor == part.name
    if anchor is not None:
        anchor_obj = bpy.data.objects.get(anchor) if isinstance(anchor, str) else anchor
        if anchor_obj is not None: serial_source["anchor_rotation"] = list(_anchor_rotation(anchor_obj))
    signature = json.dumps(["doorbench-pbr-v3", preset_id, preset, slot, seed, serial_source, asset], sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(signature.encode()).hexdigest()[:14]
    material_name = f"{str(name)[:40]}_{digest}"
    cached = bpy.data.materials.get(material_name)
    if cached is not None and cached.get("doorbench_appearance_signature") == signature:
        return cached
    material = bpy.data.materials.new(material_name)
    material.use_nodes = True
    try:
        graph = _Graph(material, slot, source, stable_seed(seed, preset_id))
        color = list(preset["color"])
        if preset.get("source_color") and source.get("rgba"):
            color = [float(v) for v in source["rgba"][:3]]
        kind = preset["kind"]
        graph.bsdf.inputs["Base Color"].default_value = _rgba(color)
        graph.bsdf.inputs["Roughness"].default_value = preset["roughness"]
        graph.bsdf.inputs["Metallic"].default_value = preset.get("metallic", 0)
        graph.bsdf.inputs["IOR"].default_value = preset.get("ior", 1.46)
        if asset is not None:
            _scanned(graph, asset, preset, source)
        elif kind in ("wood", "wood_floor"):
            _wood(graph, preset, color)
        elif kind in ("plaster", "limewash", "concrete", "brick", "tile", "stone_tile", "terrazzo"):
            _mineral(graph, preset, color)
        elif kind in ("glass", "wired_glass", "mirror"):
            _optical(graph, preset, color)
        elif kind in ("paper", "fabric", "leather"):
            _fabric(graph, preset, color)
        elif kind == "mesh":
            mask = graph.grid_mask(preset["scale_m"][:2], preset["wire_width_m"], preset.get("rotation_deg", 0))
            transparent = graph.node("ShaderNodeBsdfTransparent", "Actual visible screen apertures")
            mix = graph.node("ShaderNodeMixShader", "Wire surface over transparent openings")
            graph.links.new(mask, mix.inputs[0])
            graph.links.new(transparent.outputs[0], mix.inputs[1])
            graph.links.new(graph.bsdf.outputs[0], mix.inputs[2])
            graph.links.new(mix.outputs[0], graph.output.inputs["Surface"])
            # Cutout shading changes visibility only; the source collision
            # proxy remains an opaque barrier in the simulation export.
        else:
            micro = graph.noise(preset["scale_m"], 2)
            graph.bump(micro, preset["bump_m"], .32)
            graph.roughness(micro, preset["roughness"], .035)
            if kind in ("brushed_metal", "brass"):
                graph.bsdf.inputs["Anisotropic"].default_value = .65
                graph.links.new(graph.ramp(micro, _rgba(color, .91), _rgba(color, 1.03)), graph.bsdf.inputs["Base Color"])
            elif kind == "rust":
                graph.links.new(graph.ramp(micro, _rgba(color, .42), _rgba(color, 1.28)), graph.bsdf.inputs["Base Color"])
            elif kind in ("paint", "powdercoat", "plastic"):
                graph.bsdf.inputs["Metallic"].default_value = 0  # Opaque paint is dielectric even on steel.
                graph.bsdf.inputs["Coat Weight"].default_value = .1 if kind != "powdercoat" else .035
                graph.bsdf.inputs["Coat Roughness"].default_value = .3
        emission = source.get("emissive", [0, 0, 0])
        if any(float(v) > 0 for v in emission[:3]):
            graph.bsdf.inputs["Emission Color"].default_value = _rgba(emission)
            graph.bsdf.inputs["Emission Strength"].default_value = 1
        material.diffuse_color = _rgba(color)
        material.roughness = preset["roughness"]
        material.metallic = preset.get("metallic", 0)
        material["doorbench_appearance_signature"] = signature
        material["doorbench_appearance_preset"] = preset_id
        material["doorbench_appearance_slot"] = slot
        material["doorbench_scale_units"] = "meters"
        if asset is not None:
            material["doorbench_texture_asset"] = asset["id"]
            material["doorbench_texture_license"] = "CC0-1.0"
            material["doorbench_texture_scale_m"] = asset["scale_m"]
            material["doorbench_texture_provider_scale_m"] = asset.get("provider_scale_m", asset["scale_m"])
            material["doorbench_texture_scale_source"] = asset.get("scale_source", "Poly Haven metadata")
            if asset.get("scale_calibration"):
                material["doorbench_texture_scale_calibration"] = json.dumps(asset["scale_calibration"], sort_keys=True)
            material["doorbench_texture_library_sha256"] = _TEXTURE_LIBRARY["library_sha256"]
        return material
    except Exception:
        bpy.data.materials.remove(material)
        raise


def material_for_geom(geom, source_material, spec, recipe):
    """Return a bpy material selected by semantic role, with stable per-body reuse."""
    preset_id = preset_for_geom(geom, source_material, spec, recipe)
    semantic = geom.get("semantic", "")
    slot = semantic if semantic in ("wall", "floor") else "door"
    source = dict(source_material or {})
    dimensions = source.get("part_dimensions") or geom.get("size", [])
    if surface_preset(preset_id)["kind"] == "wood" and len(dimensions) >= 3 and dimensions[2] < min(dimensions[:2]):
        # A horizontal hatch's broad face is XY; run its grain along the long
        # horizontal dimension. The shared box projection also covers edges.
        source["grain_axis"] = "X" if dimensions[0] >= dimensions[1] else "Y"
    # A diagonal barn brace is a separate piece of lumber. Map in its own
    # local frame so its grain follows the cut member, not the overall leaf.
    member = str(geom.get("name", "")).lower()
    if surface_preset(preset_id)["kind"] in ("wood", "wood_floor") and any(token in member for token in ("brace", "batten", "rail", "stile", "kumiko", "mullion", "transom", "plank", "jamb", "casing", "head_stud", "stop_head")):
        if source.get("part_coordinate_object") is not None:
            source["coordinate_object"] = source["part_coordinate_object"]
            dimensions = source.get("part_dimensions") or geom.get("size", [1, 1, 1])
            if len(dimensions) < 3: dimensions = [1, 1, 1]
            source["grain_axis"] = "X" if dimensions[0] > dimensions[2] else "Z"
    return build_material(f"DoorBench_{slot}_{preset_id}", preset_id, slot=slot,
                          source=source, seed=recipe.get("seed", 0))

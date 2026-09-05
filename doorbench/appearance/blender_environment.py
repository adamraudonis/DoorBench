"""Visual-only architectural context, lighting, and cameras for Blender renders.

Imports are deliberately lazy: importing DoorBench never requires Blender.
Scene additions never change the source IR, joint state, or physics exports.
"""
from __future__ import annotations

import math


_LIGHTING = {
    "daylight": dict(world_color=(0.72, 0.82, 1.0), world_strength=0.35, key_energy_w=650,
                     key_color=(1.0, 0.91, 0.78), key_size_m=2.4, key_location_m=(-2.5, -3.0, 3.6),
                     fill_energy_w=140, fill_color=(0.76, 0.86, 1.0), fill_size_m=3.0),
    "overcast": dict(world_color=(0.78, 0.85, 1.0), world_strength=0.5, key_energy_w=420,
                     key_color=(0.92, 0.96, 1.0), key_size_m=4.0, key_location_m=(-2.0, -3.0, 3.6),
                     fill_energy_w=220, fill_color=(0.87, 0.93, 1.0), fill_size_m=4.0),
    "warm_interior": dict(world_color=(0.58, 0.66, 0.8), world_strength=0.12, key_energy_w=420,
                          key_color=(1.0, 0.71, 0.45), key_size_m=2.0, key_location_m=(-2.0, -2.5, 3.3),
                          fill_energy_w=100, fill_color=(0.8, 0.87, 1.0), fill_size_m=2.4),
    "warehouse": dict(world_color=(0.72, 0.8, 1.0), world_strength=0.22, key_energy_w=1600,
                      key_color=(0.94, 0.97, 1.0), key_size_m=4.0, key_location_m=(-1.5, -2.0, 5.0),
                      fill_energy_w=280, fill_color=(0.79, 0.87, 1.0), fill_size_m=3.0),
}


def _lighting(recipe):
    value = recipe.get("lighting", recipe.get("lighting_id", "daylight"))
    name = value if isinstance(value, str) else value.get("id", "daylight")
    values = dict(_LIGHTING.get(name, _LIGHTING["daylight"]))
    try:
        from .catalog import LIGHTING
        values.update(LIGHTING.get(name, {}))
    except ImportError:
        pass
    if isinstance(value, dict):
        values.update(value)
    return name, values


def _collection(name, clear=False):
    import bpy
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    elif clear:
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    return collection


def _tag(obj, kind="appearance_context"):
    obj["doorbench_visual_only"] = True
    obj["doorbench_semantic"] = kind
    obj["doorbench_physics_export"] = False
    return obj


def _aim(obj, target):
    from mathutils import Vector
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _area(collection, name, position, target, power, color, size, height=None):
    import bpy
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.color = float(power), color
    data.shape, data.size = "RECTANGLE", float(size)
    data.size_y = float(height if height is not None else size * 0.7)
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.location = position
    _aim(obj, target)
    return _tag(obj, "appearance_light")


def configure_scene(recipe, quality="preview", width=1024, height=768, seed=0):
    """Configure Cycles, AgX, denoising, and one of four lighting presets.

    ``recipe['render_device']`` accepts auto, CPU, or METAL. Auto uses an
    enumerated Metal GPU when available; unavailable backends fall back to CPU.
    The selected device is recorded on the scene for batch-render diagnostics.
    """
    import bpy
    if width <= 0 or height <= 0:
        raise ValueError("Render dimensions must be positive")
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    requested = str(recipe.get("render_device", "auto")).upper()
    selected, device_names = "CPU", []
    if requested in ("AUTO", "METAL"):
        try:
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = "METAL"
            preferences.get_devices()
            devices = [d for d in preferences.devices if d.type == "METAL"]
            if devices:
                for device in preferences.devices:
                    device.use = device.type == "METAL"
                selected = "GPU"
                device_names = [d.name for d in devices]
        except (AttributeError, KeyError, RuntimeError, TypeError):
            selected = "CPU"
    elif requested != "CPU":
        raise ValueError(f"Unknown Cycles render device {requested!r}")
    scene.cycles.device = selected
    photo = str(quality).lower() in ("photo", "photoreal", "final")
    scene.cycles.samples = 96 if photo else 16
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.018 if photo else 0.06
    scene.cycles.use_denoising = True
    scene.cycles.seed = int(seed) % 2147483647
    scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 12 if photo else 8
    scene.cycles.diffuse_bounces = 5
    scene.cycles.glossy_bounces = 6 if photo else 4
    scene.cycles.transmission_bounces = 10 if photo else 6
    scene.cycles.transparent_max_bounces = 16
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.render.resolution_x, scene.render.resolution_y = int(width), int(height)
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        scene.view_settings.look = "None"
    # The watt-valued architectural lights use a camera exposure, rather than
    # clipping their physical illumination into the display's white point.
    scene.view_settings.exposure = float(recipe.get("exposure", -3.5))
    scene.view_settings.gamma = 1.0
    name, lighting = _lighting(recipe)
    world = bpy.data.worlds.new("DoorBench architectural sky")
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (*lighting["world_color"], 1.0)
    background.inputs["Strength"].default_value = lighting["world_strength"]
    if name == "daylight":
        sky = nodes.new("ShaderNodeTexSky")
        sky_types = {item.identifier for item in sky.bl_rna.properties["sky_type"].enum_items}
        # Blender 5 renamed the Nishita options; keep sun setup independent of
        # that enum so a version change cannot silently restore the default sun.
        sky.sky_type = "MULTIPLE_SCATTERING" if "MULTIPLE_SCATTERING" in sky_types else "NISHITA"
        sky.sun_elevation = math.radians(32)
        sky.sun_rotation = math.radians(135)
        sky.sun_disc = False  # broad area lights supply controlled illumination
        links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    lights = _collection("DoorBench Appearance Lighting", clear=True)
    _area(lights, "DB key", lighting["key_location_m"], (0, 0, 1), lighting["key_energy_w"],
          lighting["key_color"], lighting["key_size_m"])
    _area(lights, "DB fill", lighting.get("fill_location_m", (2.5, -1.5, 2.6)), (0, 0, 1),
          lighting["fill_energy_w"], lighting["fill_color"], lighting["fill_size_m"])
    _area(lights, "DB rear room bounce", (1.5, 2.5, 2.6), (0, 2, 1),
          lighting["fill_energy_w"] * 0.45, lighting["fill_color"], lighting["fill_size_m"])
    scene["doorbench_render_device"] = "METAL" if selected == "GPU" else "CPU"
    scene["doorbench_render_devices"] = ", ".join(device_names) or "CPU"
    scene["doorbench_lighting"] = name
    return scene


def _semantic(obj):
    return obj.get("doorbench_semantic", obj.get("semantic", obj.get("db_semantic", "")))


def _context(obj):
    if _semantic(obj) in ("floor", "wall", "ceiling", "appearance_context", "appearance_light"):
        return True
    return obj.name in ("floor", "ground", "ceiling") or obj.name.startswith(("wall_left", "wall_right", "wall_top", "wall_pocket", "pocket_skin"))


def _corners(objects, assembly=True):
    from mathutils import Vector
    points = []
    for obj in objects.values():
        if obj.type not in ("MESH", "CURVE") or obj.hide_render or (assembly and _context(obj)):
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    return points


def _bounds(points, spec):
    from mathutils import Vector
    if not points:
        opening = spec.get("opening", {})
        w, h = opening.get("width", 1.0), opening.get("height", 2.1)
        return Vector((-w / 2, -0.1, 0)), Vector((w / 2, 0.1, h))
    return Vector(tuple(min(p[i] for p in points) for i in range(3))), Vector(tuple(max(p[i] for p in points) for i in range(3)))


def _material(name, preset, slot, seed, color):
    import bpy
    if preset:
        try:
            from .blender_materials import build_material
            return build_material(name, preset, slot=slot, seed=seed)
        except (ImportError, KeyError):
            pass
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = 0.72
    return material


def _box(collection, name, center, dimensions, material):
    import bpy
    from mathutils import Vector
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = Vector(dimensions)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)
    obj.data.materials.append(material)
    return _tag(obj)


def _fit_room_light(obj, target, minimum, maximum):
    """Keep the complete rectangular emitter inside its room, after aiming it."""
    from mathutils import Vector
    for _ in range(24):
        _aim(obj, target)
        rotation = obj.rotation_euler.to_matrix()
        offsets = [rotation @ Vector((x * obj.data.size / 2, y * obj.data.size_y / 2, 0))
                   for x in (-1, 1) for y in (-1, 1)]
        limits = [(minimum[i] - min(v[i] for v in offsets), maximum[i] - max(v[i] for v in offsets))
                  for i in range(3)]
        if any(a > b for a, b in limits):
            obj.data.size *= 0.85
            obj.data.size_y *= 0.85
            continue
        previous = obj.location.copy()
        for i, (a, b) in enumerate(limits):
            obj.location[i] = min(b, max(a, obj.location[i]))
        if (obj.location - previous).length < 1e-5:
            break
    _aim(obj, target)


def build_environment(model, spec, recipe, objects):
    """Add room/yard surfaces beyond the unchanged source-door workspace.

    Interior return walls, a distant window, and ceilings provide real reflected
    context for glass and mirrors. Horizontal hatches receive no new floor or
    ceiling plane, so neither opening can accidentally be covered by the stage.
    """
    import bpy
    from mathutils import Vector
    for body in model.get("bodies", []):
        for geom in body.get("geoms", []):
            if geom["name"] in objects:
                objects[geom["name"]]["doorbench_semantic"] = geom.get("semantic", "")
    bpy.context.view_layer.update()
    low, high = _bounds(_corners(objects), spec)
    center = (low + high) / 2
    span = high - low
    family = spec.get("family", "")
    horizontal = family in ("hatch_floor", "hatch_ceiling")
    outdoor = bool(spec.get("opening", {}).get("outdoor")) or family.startswith("gate_")
    lighting_name, lighting = _lighting(recipe)
    seed = int(recipe.get("seed", 0))
    env = _collection("DoorBench Appearance Environment", clear=True)
    wall = _material("DB context plaster", recipe.get("wall"), "wall", seed, (0.57, 0.55, 0.51))
    floor = _material("DB context floor", recipe.get("floor"), "floor", seed, (0.23, 0.22, 0.20))
    trim = _material("DB context trim", None, "wall", seed, (0.66, 0.64, 0.59))
    half_width = max(3.2, span.x * 0.95 + 1.0, abs(low.x) + 0.6, abs(high.x) + 0.6)
    depth = max(8.0, span.x * 3.5, span.y * 2.5 + 4.0)
    ceiling_z = max(3.15, high.z + 0.7, 5.0 if lighting_name == "warehouse" else 0)
    wall_y = float(model.get("meta", {}).get("wall_y", 0.0))
    if not horizontal:
        # A 1 mm lower continuation avoids z-fighting with the source floor.
        _box(env, "DB ground continuation", (center.x, wall_y, -0.016),
             (half_width * 2 + 4, depth * 2 + 4, 0.03), floor)
    if not outdoor:
        for side in (-1, 1):
            _box(env, f"DB return wall {side}", (center.x + side * half_width, wall_y, ceiling_z / 2),
                 (0.12, depth * 2, ceiling_z), wall)
            _box(env, f"DB return skirting {side}", (center.x + side * (half_width - 0.07), wall_y, 0.055),
                 (0.025, depth * 2, 0.11), trim)
        # The far wall has a real opening to the world sky, rather than a bright
        # card pasted into the hero door's opening. It is visual stage geometry.
        window_x = center.x - half_width * 0.38
        window_w, window_h = min(2.0, half_width * 0.5), min(1.5, ceiling_z - 1.2)
        sill = 0.9 if lighting_name != "warehouse" else ceiling_z - window_h - 0.35
        y = wall_y + depth
        left, right = center.x - half_width, center.x + half_width
        wx0, wx1 = window_x - window_w / 2, window_x + window_w / 2
        for name, x0, x1, z0, z1 in (
            ("left", left, wx0, 0, ceiling_z), ("right", wx1, right, 0, ceiling_z),
            ("below", wx0, wx1, 0, sill), ("above", wx0, wx1, sill + window_h, ceiling_z)):
            _box(env, "DB far wall " + name, ((x0 + x1) / 2, y, (z0 + z1) / 2), (x1 - x0, 0.12, z1 - z0), wall)
        for side in (-1, 1):
            _box(env, f"DB window jamb {side}", (window_x + side * window_w / 2, y - 0.065, sill + window_h / 2),
                 (0.06, 0.035, window_h + 0.08), trim)
        for z in (sill, sill + window_h):
            _box(env, f"DB window rail {z}", (window_x, y - 0.065, z), (window_w + 0.06, 0.035, 0.06), trim)
        _box(env, "DB front room wall", (center.x, wall_y - depth, ceiling_z / 2), (half_width * 2, 0.12, ceiling_z), wall)
        if not horizontal:
            _box(env, "DB room ceiling", (center.x, wall_y, ceiling_z + 0.04), (half_width * 2, depth * 2, 0.08), wall)
    # Adapt existing lights to actual assembly size; never put a room key above
    # its ceiling, which would black out an otherwise correctly imported model.
    scale = max(0.65, min(3.5, max(span.x / 1.4, span.z / 2.4)))
    fill_position = list(lighting.get("fill_location_m", (2.5, -1.5, 2.6)))
    # The front and rear rooms are separated by the source wall. Both require
    # their own illumination; putting the only fill behind that wall leaves the
    # hero surface dark while producing distracting bright aperture leaks.
    fill_position[1] = -max(1.5, abs(fill_position[1]))
    for name, position, size in (("DB key", lighting["key_location_m"], lighting["key_size_m"]),
                                 ("DB fill", fill_position, lighting["fill_size_m"]),
                                 ("DB rear room bounce", (1.5, 2.5, 2.6), lighting["fill_size_m"])):
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        obj.data.size = size * min(scale, 1.8)
        obj.data.size_y = min(obj.data.size * 0.7, max(1.0, ceiling_z * 0.45))
        z = position[2] * scale
        if not outdoor:
            z = min(z, ceiling_z - obj.data.size_y * 0.55 - 0.12)
        obj.location = (center.x + position[0] * min(scale, 1.6), wall_y + position[1] * min(scale, 1.6), z)
        obj.data.energy *= scale ** 1.25
        target = center.copy()
        if name == "DB rear room bounce":
            target.y = wall_y + min(4.0, depth * 0.6)
        if family == "hatch_ceiling":
            obj.location.z = max(0.5, low.z - 1.0)
            target.z = low.z
        _aim(obj, target)
        if not outdoor:
            rear = name == "DB rear room bounce"
            ymin = wall_y + 0.35 if rear and not horizontal else wall_y - depth + 0.15
            ymax = wall_y - 0.35 if not rear and not horizontal else wall_y + depth - 0.15
            _fit_room_light(obj, target,
                            (center.x - half_width + 0.15, ymin, 0.15),
                            (center.x + half_width - 0.15, ymax, ceiling_z - 0.15))
    bpy.context.scene["doorbench_environment_outdoor"] = outdoor
    bpy.context.scene["doorbench_environment_bounds"] = [center.x - half_width, center.x + half_width, wall_y - depth, wall_y + depth, ceiling_z]
    return {obj.name: obj for obj in env.objects}


def frame_camera(objects, spec, view="front", width=1024, height=768):
    """Fit assembly bounds, ignoring the large floor/walls, with straight verticals.

    Normal doors use a level architectural camera plus lens shift. Floor and
    ceiling hatches use appropriately directed oblique views of their opening.
    """
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = int(width), int(height)
    bpy.context.view_layer.update()
    points = _corners(objects)
    low, high = _bounds(points, spec)
    if not points:
        points = [Vector((x, y, z)) for x in (low.x, high.x) for y in (low.y, high.y) for z in (low.z, high.z)]
    center, span = (low + high) / 2, high - low
    camera = bpy.data.objects.get("DoorBench Camera")
    if camera is None:
        camera = bpy.data.objects.new("DoorBench Camera", bpy.data.cameras.new("DoorBench Camera"))
        scene.collection.objects.link(camera)
    camera.data.type, camera.data.lens, camera.data.sensor_fit = "PERSP", 48.0, "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.clip_start, camera.data.clip_end = 0.03, 250.0
    camera.data.dof.use_dof = False
    camera.data.shift_x = camera.data.shift_y = 0.0
    family = spec.get("family", "")
    horizontal = family in ("hatch_floor", "hatch_ceiling")
    quarter = view in ("iso", "three_quarter") or family in ("revolving", "turnstile_tripod", "turnstile_fullheight")
    angle = math.radians(23 if quarter else 8)
    back = view in ("back", "rear", "far", "reverse")
    direction = Vector((math.sin(angle), math.cos(angle) * (1 if back else -1), 0))
    aspect = width / height
    distance = max(0.65, span.x * 1.75, span.z * aspect * 1.65, span.y * 1.25)
    eye = max(low.z + 0.35, min(low.z + 1.55, low.z + max(span.z * 0.64, 0.4)))
    margin = 0.065
    for _ in range(35):
        if horizontal:
            up = -1.0 if family == "hatch_ceiling" else 1.0
            position = center + Vector((0.28, -0.55, 0.8 * up)).normalized() * distance
            if up < 0:
                position.z = max(0.65, position.z)
            camera.location = position
            _aim(camera, center)
        else:
            camera.location = center + direction * distance
            camera.location.z = eye
            _aim(camera, (center.x, center.y, eye))  # level lens keeps architectural verticals parallel
        camera.data.shift_x = camera.data.shift_y = 0.0
        bpy.context.view_layer.update()
        # Compute lens shifts from projection derivatives, independent of
        # Blender's sensor-fit/aspect normalization of shift_x and shift_y.
        for axis, attribute in ((0, "shift_x"), (1, "shift_y")):
            projected = [world_to_camera_view(scene, camera, point) for point in points]
            middle = (min(p[axis] for p in projected) + max(p[axis] for p in projected)) / 2
            base = world_to_camera_view(scene, camera, center)[axis]
            setattr(camera.data, attribute, 0.001)
            derivative = (world_to_camera_view(scene, camera, center)[axis] - base) / 0.001
            setattr(camera.data, attribute, (0.5 - middle) / derivative if abs(derivative) > 1e-8 else 0.0)
        projected = [world_to_camera_view(scene, camera, point) for point in points]
        if all(p.z > 0 and margin <= p.x <= 1 - margin and margin <= p.y <= 1 - margin for p in projected):
            break
        distance *= 1.055
    scene.camera = camera
    camera["doorbench_camera_convention"] = "+X right, +Y up, -Z forward; image pixels right/down"
    camera["doorbench_camera_view"] = view
    return camera

"""Small visual-only refinements for authored decorative wreath proxies.

The physics model remains untouched. Twigs and individual evergreen leaves fit
inside the existing wreath's annular envelope and share its rigid body pose.
"""
from __future__ import annotations

import hashlib
import math
import random


def _material(name, color, roughness, leaf=False):
    import bpy
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['IOR'].default_value = 1.42
    if leaf:
        shader.inputs['Coat Weight'].default_value = .12
        shader.inputs['Coat Roughness'].default_value = .35
        shader.inputs['Subsurface Weight'].default_value = .035
    return material


def _mesh_object(name, vertices, faces, parent, matrix, materials, assignments, source):
    import bpy
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent
    obj.matrix_basis = matrix.copy()
    for material in materials:
        mesh.materials.append(material)
    for face, assignment in zip(mesh.polygons, assignments):
        face.material_index = assignment
        face.use_smooth = True
    obj['doorbench_visual_only'] = True
    obj['doorbench_physics_export'] = False
    obj['doorbench_semantic'] = 'appearance_detail'
    obj['doorbench_source_geom'] = source.get('doorbench_geom', source.name)
    obj['doorbench_detail_kind'] = 'evergreen_wreath'
    return obj


def _tube(points, radius, vertices, faces, assignments):
    from mathutils import Vector
    first = len(vertices)
    sides = 6
    for i, point in enumerate(points):
        point = Vector(point)
        tangent = Vector(points[min(i + 1, len(points) - 1)]) - Vector(points[max(0, i - 1)])
        tangent.normalize()
        side = tangent.cross(Vector((0, 0, 1))).normalized()
        up = side.cross(tangent).normalized()
        for j in range(sides):
            angle = j * math.tau / sides
            vertices.append(tuple(point + radius * (side * math.cos(angle) + up * math.sin(angle))))
    for i in range(len(points) - 1):
        for j in range(sides):
            a = first + i * sides + j
            b = first + i * sides + (j + 1) % sides
            faces.append((a + sides, b + sides, b, a)); assignments.append(0)
    faces.extend([tuple(first + j for j in range(sides)), tuple(first + (len(points) - 1) * sides + j for j in reversed(range(sides)))])
    assignments.extend([0, 0])


def _leaf(base, direction, length, width, bend, curl, material, vertices, faces, assignments):
    from mathutils import Vector
    base, axis = Vector(base), Vector(direction).normalized()
    side = Vector((-axis.y, axis.x, 0)).normalized()
    top = [base]
    for t in (.16, .46, .77):
        middle = base + axis * (length * t) + side * (bend * math.sin(math.pi * t))
        middle.z += curl * t * t
        half = width * .5 * math.sin(math.pi * t) ** .85
        left, ridge, right = middle - side * half, middle.copy(), middle + side * half
        ridge.z += width * .12 * math.sin(math.pi * t)
        top.extend([left, ridge, right])
    tip = base + axis * length
    tip.z += curl
    top.append(tip)
    first, count = len(vertices), len(top)
    vertices.extend(tuple(p) for p in top)
    vertices.extend(tuple(p - Vector((0, 0, .00028))) for p in top)
    surface = [(0, 1, 2), (0, 2, 3)]
    for a in (1, 4):
        surface.extend([(a, a + 3, a + 4, a + 1), (a + 1, a + 4, a + 5, a + 2)])
    surface.extend([(7, 10, 8), (8, 10, 9)])
    for face in surface:
        faces.append(tuple(first + v for v in face)); assignments.append(material)
        faces.append(tuple(first + count + v for v in reversed(face))); assignments.append(material)
    outline = [0, 1, 4, 7, 10, 9, 6, 3]
    for a, b in zip(outline, outline[1:] + outline[:1]):
        faces.append((first + count + a, first + count + b, first + b, first + a)); assignments.append(material)


def build_details(objects, spec, seed=0):
    """Replace only named wreath render proxies; return tagged detail objects.

    Call after applying the authoritative body pose. Existing object transforms,
    simulation state, source meshes, and collision exports are unchanged.
    """
    from mathutils import Vector
    results = {}
    for name, proxy in objects.items():
        if proxy.type != 'MESH' or name != 'wreath':
            continue
        vertices = [v.co for v in proxy.data.vertices]
        radii = [math.hypot(v.x, v.y) for v in vertices]
        inner, outer = min(radii), max(radii)
        z0, z1 = min(v.z for v in vertices), max(v.z for v in vertices)
        if not 0 < inner < outer or z1 <= z0:
            raise ValueError('Wreath proxy must have a nonempty annular envelope')
        major, zcenter = (inner + outer) / 2, (z0 + z1) / 2
        factor = major / .2
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{spec.get('id', '')}:{seed}:{name}".encode()).digest()[:8], 'big'))
        twig_vertices, twig_faces, twig_materials = [], [], []
        leaf_vertices, leaf_faces, leaf_materials = [], [], []
        for strand in range(3):
            points = []
            for i in range(97):
                angle = math.tau * i / 96
                radius = major + .006 * factor * math.sin(angle * 7 + strand * 2)
                z = zcenter - .014 * factor + .004 * factor * math.cos(angle * 5 + strand * 2)
                points.append((radius * math.cos(angle), radius * math.sin(angle), z))
            _tube(points, .0015 * factor, twig_vertices, twig_faces, twig_materials)
        for branch in range(44):
            angle = math.tau * (branch + rng.uniform(-.24, .24)) / 44
            radial = Vector((math.cos(angle), math.sin(angle), 0))
            tangent = Vector((-radial.y, radial.x, 0))
            center = radial * (major + rng.uniform(-.008, .008) * factor)
            center.z = zcenter + rng.uniform(-.016, .010) * factor
            stem = [center + tangent * (factor * t) + radial * (.004 * factor * math.sin(t * 35)) for t in (-.026, -.012, .003, .023)]
            _tube(stem, .00085 * factor, twig_vertices, twig_faces, twig_materials)
            for pair, t in enumerate((-.022, -.010, .002, .014)):
                for sign in (-1, 1):
                    base = center + tangent * (t * factor) + radial * (rng.uniform(-.002, .002) * factor)
                    direction = tangent * rng.uniform(.45, .85) + radial * sign * rng.uniform(.65, 1.0)
                    direction.z = rng.uniform(-.23, .40)
                    _leaf(base, direction, rng.uniform(.026, .038) * factor, rng.uniform(.010, .016) * factor,
                          rng.uniform(-.0025, .0025) * factor, rng.uniform(-.005, .008) * factor,
                          rng.randrange(5), leaf_vertices, leaf_faces, leaf_materials)
        # The detail may alter silhouette within the authored decoration, never
        # enlarge the occupied annulus or push foliage through the door surface.
        def constrain(points):
            constrained = []
            for x, y, z in points:
                radius = math.hypot(x, y)
                clipped = max(inner + .0004, min(outer - .0004, radius))
                constrained.append((x * clipped / radius, y * clipped / radius, max(z0 + .0004, min(z1 - .0004, z))))
            return constrained
        twig_vertices, leaf_vertices = constrain(twig_vertices), constrain(leaf_vertices)
        bark = _material('DB wreath natural vine', (.065, .031, .011), .86)
        greens = [_material(f'DB wreath evergreen {i}', color, .37 + i * .035, leaf=True) for i, color in enumerate(
            [(.022, .078, .018), (.035, .12, .026), (.048, .15, .031), (.028, .094, .016), (.063, .17, .040)])]
        for suffix, points, faces, materials, assignments in (
            ('twigs', twig_vertices, twig_faces, [bark], twig_materials),
            ('leaves', leaf_vertices, leaf_faces, greens, leaf_materials)):
            obj = _mesh_object(f'DB detail {name} {suffix}', points, faces, proxy.parent, proxy.matrix_basis, materials, assignments, proxy)
            obj['doorbench_reference_envelope'] = [inner, outer, z0, z1]
            results[obj.name] = obj
        proxy.hide_render = True
        proxy['doorbench_render_replaced_by_detail'] = True
    return results

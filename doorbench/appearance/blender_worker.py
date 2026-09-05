"""Run inside Blender; consume versioned jobs prepared by appearance.pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from doorbench.appearance.blender_materials import material_for_geom, configure_texture_library
from doorbench.appearance.blender_environment import configure_scene, build_environment, frame_camera
from doorbench.appearance.blender_details import build_details


def reset_scene():
    """Clear worker-owned scene data while retaining reusable texture images.

    Operator deletion only unlinks objects from the scene. A material's Object
    texture-coordinate reference can retain its mesh object, producing an
    object -> mesh -> material -> object cycle that never reaches users == 0.
    Explicit datablock removal breaks those references before orphan cleanup.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.scene.world = None
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights, bpy.data.worlds):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def load_obj(path):
    """Read triangulated source hardware in its authored frame, not MuJoCo's COM frame."""
    verts, faces = [], []
    with path.open() as f:
        for line in f:
            words = line.split()
            if not words:
                continue
            if words[0] == 'v':
                verts.append(tuple(float(x) for x in words[1:4]))
            elif words[0] == 'f':
                indices = [int(x.split('/')[0]) for x in words[1:]]
                faces.append(tuple(i - 1 if i > 0 else len(verts) + i for i in indices))
    if not verts or not faces:
        raise ValueError(f'Empty source mesh: {path}')
    mesh = bpy.data.meshes.new(path.stem)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def primitive(geom, mesh_cache, hardware):
    kind, size, name = geom['type'], geom['size'], geom['name']
    if kind == 'mesh':
        key = geom['mesh_name']
        if key not in mesh_cache:
            mesh_cache[key] = load_obj(hardware / f'{key}.obj')
        obj = bpy.data.objects.new(name, mesh_cache[key])
        bpy.context.collection.objects.link(obj)
    elif kind == 'box':
        x, y, z = size
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(-x,-y,-z),(-x,-y,z),(-x,y,-z),(-x,y,z),(x,-y,-z),(x,-y,z),(x,y,-z),(x,y,z)], [],
                         [(2,6,4,0),(5,7,3,1),(4,5,1,0),(3,7,6,2),(1,3,2,0),(6,7,5,4)])
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
    elif kind == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=size[0], depth=2*size[1])
        obj = bpy.context.object
    elif kind == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=size[0])
        obj = bpy.context.object
    elif kind == 'capsule':
        # Revolve a hemisphere/cylinder profile: no intersecting end-cap spheres.
        r, h = size[:2]
        rings = []
        for i in range(9):
            a = -math.pi/2 + i*math.pi/16
            rings.append((r*math.cos(a), -h+r*math.sin(a)))
        for i in range(9):
            a = i*math.pi/16
            rings.append((r*math.cos(a), h+r*math.sin(a)))
        verts = [(rr*math.cos(j*math.tau/32), rr*math.sin(j*math.tau/32), z) for rr,z in rings for j in range(32)]
        faces = [(i*32+j, i*32+(j+1)%32, (i+1)*32+(j+1)%32, (i+1)*32+j) for i in range(len(rings)-1) for j in range(32)]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], faces); mesh.update()
        obj = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(obj)
    else:
        raise ValueError(f'Unsupported geometry: {kind}')
    obj.name = name
    if kind in ('sphere', 'capsule'):
        for face in obj.data.polygons:
            face.use_smooth = True
    elif kind == 'cylinder':
        for face in obj.data.polygons:
            face.use_smooth = len(face.vertices) == 4
    # Tiny edge rounding catches real highlights, without changing outer dimensions.
    if kind in ('box', 'cylinder') and geom.get('semantic') not in ('floor', 'wall', 'glass'):
        width = min(0.0007, min(size)*0.12)
        mod = obj.modifiers.new('Manufactured edge radius', 'BEVEL')
        mod.width, mod.segments = width, 3
        mod.affect = 'EDGES'
        mod.harden_normals = True
        norm = obj.modifiers.new('Face weighted normals', 'WEIGHTED_NORMAL')
        norm.keep_sharp = True
    return obj


def transform(pos, quat):
    return Matrix.Translation(Vector(pos)) @ Quaternion(quat).to_matrix().to_4x4()


def build_door(model, spec, state, recipe, hardware):
    """One source geom -> one tagged object; native body poses are authoritative."""
    objects, bodies, cache = {}, {}, {}
    for body in model['bodies']:
        pose = state['body_world'].get(body['name'])
        if pose is None:
            raise ValueError(f"State lacks source body {body['name']}; use a full-tier simulation snapshot")
        parent = bpy.data.objects.new('body::'+body['name'], None)
        bpy.context.collection.objects.link(parent)
        parent.matrix_world = transform(pose['pos'], pose['quat_wxyz'])
        parent['doorbench_body'] = body['name']
        bodies[body['name']] = parent
        for geom in body['geoms']:
            if not geom.get('visual', True):
                continue
            obj = primitive(geom, cache, hardware)
            obj.parent = parent
            obj.matrix_basis = transform(geom['pos'], geom['quat'])
            obj['doorbench_geom'] = geom['name']
            obj['doorbench_body'] = body['name']
            obj['semantic'] = geom.get('semantic', 'structure')
            obj['collision_source'] = geom.get('collision', False)
            obj['doorbench_visual_only'] = False
            material = material_for_geom(geom, {**model['materials'][geom['material']],
                                               'coordinate_object':parent, 'part_coordinate_object':obj,
                                               'part_dimensions':[2*s for s in geom['size']] if geom['type']=='box' else []}, spec, recipe)
            # Shared mesh vertices, per-object material slots: brass and steel knobs may share geometry.
            if not obj.data.materials:
                obj.data.materials.append(material)
            obj.material_slots[0].link = 'OBJECT'
            obj.material_slots[0].material = material
            objects[geom['name']] = obj
    bpy.context.view_layer.update()
    expected = sum(bool(g.get('visual', True)) for b in model['bodies'] for g in b['geoms'])
    if len(objects) != expected:
        raise ValueError(f'Duplicate/lost visual geometry: {len(objects)} != {expected}')
    return objects


def explicit_camera(camera):
    bpy.ops.object.camera_add()
    obj = bpy.context.object
    obj.name = 'Simulation camera'
    obj.matrix_world = transform(camera['pos'], camera['quat_wxyz'])
    width, height = camera['resolution']
    K = camera['intrinsics']
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    obj.data.type = 'PERSP'; obj.data.sensor_fit = 'HORIZONTAL'; obj.data.sensor_width = 36
    obj.data.lens = fx * 36 / width
    # Pixel aspect supports unequal focal lengths; principal point uses Blender shift.
    scene = bpy.context.scene
    scene.render.pixel_aspect_x = max(1, fy/fx); scene.render.pixel_aspect_y = max(1, fx/fy)
    obj.data.shift_x = (width/2-cx)/width
    obj.data.shift_y = (cy-height/2)*(fx/fy)/width
    obj.data.clip_start = .01; obj.data.clip_end = 1000
    scene.camera = obj
    return obj


def run_job(job):
    start = time.monotonic()
    print(f"BUILD {job['door_id']} variant {job['variant']}", flush=True)
    for name, expected in job['renderer_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Renderer changed after job preparation: {name}; prepare the batch again')
    source = Path(job['door_dir'])
    blobs = {name:(source/name).read_bytes() for name in job['source_sha256']}
    for name, blob in blobs.items():
        if hashlib.sha256(blob).hexdigest() != job['source_sha256'][name]:
            raise ValueError(f'Source changed after preparation: {source/name}')
    for name, expected in job['mesh_sha256'].items():
        path = Path(job['hardware_dir']) / f'{name}.obj'
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source mesh changed after preparation: {path}')
    model, spec = json.loads(blobs['model.json']), json.loads(blobs['spec.json'])
    reset_scene()
    configure_texture_library(job.get('texture_library'))
    camera_state = job['state'].get('camera')
    width, height = camera_state['resolution'] if camera_state else (job['width'], job['height'])
    configure_scene(job['recipe'], quality=job['quality'], width=width, height=height, seed=job['seed'])
    # Lay out the room and default camera in the fixed source reference pose.
    # A moving leaf must not move walls/lights or make the camera refit each frame.
    objects = build_door(model, spec, job['reference_state'], job['recipe'], Path(job['hardware_dir']))
    build_environment(model, spec, job['recipe'], objects)
    camera = explicit_camera(camera_state) if camera_state else frame_camera(objects, spec, view=job['view'], width=width, height=height)
    bpy.context.scene.camera = camera
    for obj in bpy.context.scene.objects:
        if obj.type == 'EMPTY' and obj.get('doorbench_body'):
            pose = job['state']['body_world'][obj['doorbench_body']]
            obj.matrix_world = transform(pose['pos'], pose['quat_wxyz'])
    details = build_details(objects, spec, job['seed'])
    bpy.context.view_layer.update()
    out = Path(job['out_dir']); out.mkdir(parents=True, exist_ok=True)
    rendered = False
    if not job['validate_only']:
        scene = bpy.context.scene
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.filepath = str(out/'rgb.png')
        print(f"RENDER {job['door_id']} {scene.get('doorbench_render_device')} {width}x{height}", flush=True)
        bpy.ops.render.render(write_still=True)
        rendered = True
    if job.get('save_blend'):
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    metadata = {'schema_version':1, 'door_id':spec['id'], 'variant':job['variant'], 'recipe':job['recipe'], 'source_sha256':job['source_sha256'],
                'mesh_sha256':job['mesh_sha256'], 'renderer_sha256':job['renderer_sha256'], 'job_sha256':job['job_sha256'], 'state_sha256':hashlib.sha256(json.dumps(job['state'],sort_keys=True).encode()).hexdigest(),
                'texture_library':job.get('texture_library'),
                'state_kind':job['state'].get('state_kind'), 'source_visual_geoms':len(objects),
                'appearance_detail_objects':len(details),
                'visual_only_context_objects':sum(bool(o.get('doorbench_visual_only')) for o in bpy.context.scene.objects),
                'blender_version':bpy.app.version_string, 'engine':bpy.context.scene.render.engine,
                'device':bpy.context.scene.get('doorbench_render_device'), 'samples':bpy.context.scene.cycles.samples,'quality':job['quality'],
                'resolution':[width,height], 'camera_world_matrix':[list(row) for row in camera.matrix_world],
                'rendered':rendered, 'image':'rgb.png' if rendered else None, 'blend':'scene.blend' if job.get('save_blend') else None,
                'time_s':round(time.monotonic()-start,3),
                'artifact_sha256':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
                                   (['rgb.png'] if rendered else [])+(['scene.blend'] if job.get('save_blend') else [])},
                'scope':'Appearance derivative of source geometry. Submillimeter bevels and tagged context are visual only; physics/geometry defects in the source are not certified by a successful build.'}
    (out/'render.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True)
    args = ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    config = json.loads(Path(args.jobs).read_text())
    results, failures = [], []
    for i, job in enumerate(config['jobs']):
        try:
            result = run_job(job); results.append(result)
            print(f"APPEARANCE {i+1}/{len(config['jobs'])} {job['door_id']} OK {result['time_s']}s", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failures.append({'door_id':job['door_id'], 'out_dir':job['out_dir'], 'error':str(exc)})
    Path(config['result_path']).write_text(json.dumps({'results':results,'failures':failures},indent=2)+'\n')
    if failures:
        raise RuntimeError(f'{len(failures)} appearance jobs failed')


if __name__ == '__main__':
    main()

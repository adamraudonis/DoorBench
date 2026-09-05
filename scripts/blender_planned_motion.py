"""Pack/render a planned door motion and its exact original collision rig.

Blender --background --python scripts/blender_planned_motion.py -- --job JOB
  --clip CLIP --trajectory NPZ --out scene.blend [--validation REPORT]
  [--render-time 3 --image frame.png]

All body poses are world transforms on actor_time, including retimed native
door poses. This exporter does not solve, repair, or certify the proposed motion.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.blender_reference_motion import _linear, sha256


def _names(value, label):
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value) or len(value) != len(set(value)):
        raise ValueError(f'{label} must contain unique nonempty names')
    return value


def _quaternions(value, label):
    if value.shape[-1:] != (4,) or not np.allclose(np.linalg.norm(value, axis=-1), 1., atol=2e-5, rtol=0):
        raise ValueError(f'{label} must contain unit WXYZ quaternions')


def _vector(value, length, label):
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{label} must be a finite vector of length {length}') from exc
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f'{label} must be a finite vector of length {length}')
    return array


def validation_label(clip, verification=None):
    """Never promote an input status string into a feasibility certificate."""
    if verification is not None and verification.get('bindings_verified') is True:
        return 'Sampled kinematic checks passed', 'Kinematic replay | Forces and balance not certified'
    qa = clip.get('qa', {})
    if not isinstance(qa, dict):
        qa = {}
    passed = qa.get('passed')
    if passed is False:
        return 'Constrained motion candidate', 'QA failed | Kinematic replay only'
    if passed is True and qa.get('independent') is True and clip.get('status') == 'accepted':
        return 'Constrained motion candidate', 'Declared QA passed | External validation report required'
    if passed is True:
        return 'Constrained motion candidate', 'Automated QA passed | Independent acceptance pending'
    return 'Constrained motion candidate', 'Unvalidated proposal | Kinematic replay only'


def load_validation_report(path, clip_path, trajectory_path, job):
    """Verify an external report without rewriting its immutable motion inputs.

    This verifies report content and checksum bindings, not a digital signature
    or validator re-execution. Its scope remains the report's sampled checks.
    """
    path = Path(path)
    report_hash = sha256(path)
    report = json.loads(path.read_text())
    clip = json.loads(Path(clip_path).read_text())
    if not isinstance(report, dict) or report.get('schema') != 'doorbench.planned-reference-validation.v1':
        raise ValueError('Unsupported independent validation report schema')
    if report.get('door_id') != clip.get('door_id') or report.get('door_id') != job['door_id']:
        raise ValueError('Validation report door identity mismatch')
    if report.get('clip_sha256') != sha256(clip_path) or report.get('trajectory_sha256') != sha256(trajectory_path):
        raise ValueError('Validation report clip/trajectory checksum mismatch')
    if report.get('source_sha256') != job['source_sha256'] or report.get('source_sha256') != clip.get('source_sha256'):
        raise ValueError('Validation report source checksum mismatch')
    for name, expected in job['source_sha256'].items():
        if sha256(Path(job['door_dir'])/name) != expected:
            raise ValueError(f'Validation source changed: {name}')
    completion = report.get('task_completion')
    if (report.get('accepted') is not True or report.get('kinematic_accepted') is not True or
            report.get('status') != 'accepted_kinematic' or report.get('failure_counts') != {}):
        raise ValueError('Validation report does not pass sampled kinematic checks')
    if (clip.get('complete_proposal') is not True or not isinstance(completion, dict) or
            completion.get('complete_proposal') is not True or completion.get('evidence_pass') is not True or
            completion.get('failure_counts') != {} or completion.get('source_success_declared') is not True):
        raise ValueError('Validation report does not establish complete task evidence')
    if type(report.get('frames')) is not int or report['frames'] != clip.get('frames'):
        raise ValueError('Validation report frame count mismatch')
    if sha256(path) != report_hash:
        raise ValueError('Validation report changed while being read')
    return {'bindings_verified': True, 'report_sha256': report_hash, 'report_path': str(path.resolve()),
            'clip_sha256': report['clip_sha256'], 'trajectory_sha256': report['trajectory_sha256'],
            'source_sha256': report['source_sha256'], 'status': report['status'], 'accepted': True,
            'kinematic_accepted': True, 'task_completion': completion, 'settings': report.get('settings'),
            'scope': report.get('scope'), 'runtime': report.get('runtime'),
            'verification_scope': 'Hash-bound report checks; no re-execution, signature, dynamics or personal visual approval.'}


def load_inputs(job_path, clip_path, trajectory_path):
    """Read and validate provenance and arrays without requiring bpy or MuJoCo."""
    from doorbench.appearance.pipeline import digest

    job = json.loads(Path(job_path).read_text())
    clip = json.loads(Path(clip_path).read_text())
    if job.get('job_sha256') != digest({k: v for k, v in job.items() if k != 'job_sha256'}):
        raise ValueError('Prepared appearance job checksum mismatch')
    if clip.get('schema') != 'doorbench.planned-reference.v1' or clip.get('door_id') != job['door_id']:
        raise ValueError('Planned clip schema or door_id mismatch')
    if clip.get('up_axis') != 'Z' or clip.get('units') != 'metres/radians/seconds':
        raise ValueError('Planned clip must use Z up and metres/radians/seconds')
    fps = clip.get('fps')
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps != int(fps) or not 1 <= fps <= 120:
        raise ValueError('Planned clip fps must be an integer from 1 to 120')
    if not isinstance(clip.get('qa', {}), dict) or not isinstance(clip.get('status', 'proposal'), str):
        raise ValueError('Clip qa must be an object and status must be text')
    if clip.get('trajectory_sha256') != sha256(trajectory_path):
        raise ValueError('Planned trajectory checksum mismatch; write/bind the NPZ before the clip')
    source_hashes = job.get('source_sha256', {})
    if set(source_hashes) != {'spec.json', 'model.json', 'door.xml'} or clip.get('source_sha256') != source_hashes:
        raise ValueError('Clip source hashes do not match the prepared job')
    source = Path(job['door_dir'])
    for name, expected in source_hashes.items():
        if sha256(source / name) != expected:
            raise ValueError(f'Source changed after preparation: {name}')
    for name, expected in job['renderer_sha256'].items():
        if Path(name).name != name or sha256(ROOT / 'doorbench/appearance' / name) != expected:
            raise ValueError(f'Appearance renderer changed: {name}')
    for name, expected in job['mesh_sha256'].items():
        if Path(name).name != name or sha256(Path(job['hardware_dir']) / (name + '.obj')) != expected:
            raise ValueError(f'Source hardware mesh changed: {name}')
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, array in arrays.items():
        if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
            raise ValueError(f'Trajectory {name} must contain finite numeric values')
    times = arrays.get('actor_time')
    if times is None or times.ndim != 1 or len(times) < 2 or abs(float(times[0])) > 1e-7 or np.any(np.diff(times) <= 0):
        raise ValueError('actor_time must start at zero and strictly increase')
    n = len(times)
    native_time = arrays.get('native_time')
    if native_time is None or native_time.shape != (n,) or np.any(native_time < 0) or np.any(np.diff(native_time) < 0):
        raise ValueError('native_time must be nonnegative, nondecreasing and aligned with actor_time')
    native_names = _names(clip.get('native', {}).get('body_names'), 'native.body_names')
    actor = clip.get('actor', {})
    actor_names = _names(actor.get('body_names'), 'actor.body_names')
    shapes = {
        'body_pos': (n, len(native_names), 3), 'body_quat': (n, len(native_names), 4),
        'actor_body_pos': (n, len(actor_names), 3), 'actor_body_quat': (n, len(actor_names), 4),
        'foot_pos': (n, 2, 3), 'foot_quat': (n, 2, 4),
    }
    for name, shape in shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f'{name} must have shape {shape}')
    for name in ('body_quat', 'actor_body_quat', 'foot_quat'):
        _quaternions(arrays[name], name)
    for name in ('qpos', 'actor_qpos'):
        value = arrays.get(name)
        if value is None or value.ndim != 2 or value.shape[0] != n or value.shape[1] < 1:
            raise ValueError(f'{name} must be a nonempty matrix aligned with actor_time')
    joints = arrays.get('actor_joints')
    if joints is None or joints.ndim != 3 or joints.shape[0] != n or joints.shape[1] < 1 or joints.shape[2] != 3:
        raise ValueError('actor_joints must have shape (N,J,3)')
    if 'foot_contact' in arrays and (arrays['foot_contact'].shape != (n, 2) or not np.isin(arrays['foot_contact'], [0, 1]).all()):
        raise ValueError('foot_contact must be a binary (N,2) matrix')
    for owner, name in ((clip.get('native', {}), 'qpos'), (actor, 'actor_qpos')):
        if 'nq' in owner and owner['nq'] != arrays[name].shape[1]:
            raise ValueError(f'{name} width does not match declared nq')
    if 'duration' in clip and (not isinstance(clip['duration'], (float, int)) or not math.isclose(clip['duration'], float(times[-1]), abs_tol=1e-5)):
        raise ValueError('Clip duration does not match actor_time')
    geometries = actor.get('geometries')
    if not isinstance(geometries, list) or not geometries:
        raise ValueError('actor.geometries must describe the actual rig primitives')
    geom_names = set()
    for geom in geometries:
        if not isinstance(geom, dict):
            raise ValueError('Every actor geometry must be an object')
        name, kind = geom.get('name'), geom.get('type')
        if not isinstance(name, str) or not name or name in geom_names:
            raise ValueError('Actor geometry names must be unique and nonempty')
        geom_names.add(name)
        if geom.get('body_name') not in actor_names:
            raise ValueError(f'Actor geometry {name} refers to an unknown body')
        sizes = {'box': 3, 'capsule': 2, 'cylinder': 2, 'sphere': 1}
        if kind not in sizes:
            raise ValueError(f'Unsupported actor geometry type: {kind}')
        size = np.asarray(geom.get('size'), dtype=float)
        needed = sizes[kind]
        if size.ndim != 1 or len(size) not in (needed, 3) or not np.isfinite(size).all():
            raise ValueError(f'Actor geometry {name} has invalid MuJoCo size')
        if size[0] <= 0 or (kind == 'box' and np.any(size[:3] <= 0)) or (kind in ('capsule', 'cylinder') and size[1] < 0):
            raise ValueError(f'Actor geometry {name} has nonpositive dimensions')
        _vector(geom.get('pos'), 3, f'{name}.pos')
        _quaternions(_vector(geom.get('quat_wxyz'), 4, f'{name}.quat_wxyz'), f'{name}.quat_wxyz')
    model, spec = [json.loads((source / name).read_text()) for name in ('model.json', 'spec.json')]
    if spec.get('id') != job['door_id']:
        raise ValueError('Source spec id does not match door_id')
    aliases = job['reference_state'].get('body_aliases', {})
    mapping = {}
    for body in model['bodies']:
        name = body['name']
        native = name if name in native_names else aliases.get(name)
        if native not in native_names:
            raise ValueError(f'No native body pose for source body {name}')
        mapping[name] = native_names.index(native)
    if set(native_names) - set(mapping) - set(aliases.values()) - {'world'}:
        raise ValueError('Native trajectory contains bodies without source geometry')
    return job, clip, arrays, model, spec, mapping


def _rotate(quat, vectors):
    """Broadcast WXYZ quaternion rotation, using the same body-frame convention."""
    q = np.asarray(quat, dtype=float)
    v = np.asarray(vectors, dtype=float)
    cross = 2.0 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * cross + np.cross(q[..., 1:], cross)


def geometry_world_corners(geom, positions, quaternions):
    """Conservative bounds of an exact primitive, including local geom rotation."""
    size = geom['size']
    if geom['type'] == 'box':
        extent = np.asarray(size[:3], float)
    elif geom['type'] == 'sphere':
        extent = np.repeat(size[0], 3)
    else:
        extent = np.array([size[0], size[0], size[1] + (size[0] if geom['type'] == 'capsule' else 0)])
    corners = np.array(list(itertools.product((-1., 1.), repeat=3))) * extent
    local = _rotate(geom['quat_wxyz'], corners) + np.asarray(geom['pos'])
    return _rotate(np.asarray(quaternions)[:, None, :], local[None, :, :]) + np.asarray(positions)[:, None, :]


def _animate(obj, times, positions, quaternions, fps):
    from mathutils import Quaternion
    obj.rotation_mode = 'QUATERNION'
    previous = None
    for sec, pos, raw in zip(times, positions, quaternions):
        quat = Quaternion(tuple(float(v) for v in raw))
        if previous is not None and quat.dot(previous) < 0:
            quat.negate()
        obj.location, obj.rotation_quaternion = pos, quat
        frame = 1 + float(sec) * fps
        obj.keyframe_insert('location', frame=frame)
        obj.keyframe_insert('rotation_quaternion', frame=frame)
        previous = quat.copy()
    _linear(obj)


def _status_card(scene, heading, detail, door_id):
    """Camera-only annotation; cannot cast shadows or masquerade as scene geometry."""
    import bpy
    from mathutils import Vector
    camera = scene.camera
    depth = max(.4, camera.data.clip_start * 4)
    frame = [v * depth / -v.z for v in camera.data.view_frame(scene=scene)]
    left, right = min(v.x for v in frame), max(v.x for v in frame)
    bottom, top = min(v.y for v in frame), max(v.y for v in frame)
    width, height = right-left, top-bottom

    def material(name, color):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = (*color, 1)
        # Scene exposure is inherited from the photograph setup. Counter it for
        # readable labels without changing light on the door or figure.
        emission.inputs['Strength'].default_value = 2.0 ** (-scene.view_settings.exposure)
        mat.node_tree.links.new(emission.outputs[0], output.inputs['Surface'])
        return mat

    def tag(obj):
        obj.parent = camera
        obj['doorbench_visual_only'] = True
        obj['doorbench_semantic'] = 'motion_status_annotation'
        for attr in ('visible_shadow', 'visible_diffuse', 'visible_glossy', 'visible_transmission', 'visible_volume_scatter'):
            if hasattr(obj, attr):
                setattr(obj, attr, False)

    background = material('Motion status background', (.025, .04, .035))
    foreground = material('Motion status text', (.92, .97, .93))
    mesh = bpy.data.meshes.new('Motion status card')
    x0, x1 = left+.025*width, right-.025*width
    y0, y1 = bottom+.022*height, bottom+.107*height
    mesh.from_pydata([(x0,y0,-depth),(x1,y0,-depth),(x1,y1,-depth),(x0,y1,-depth)], [], [(0,1,2,3)])
    panel = bpy.data.objects.new('Motion status card', mesh)
    scene.collection.objects.link(panel)
    tag(panel)
    panel.data.materials.append(background)
    for text, y, size in ((heading, y0+.050*height, .025*height),
                          (f'{door_id}  |  {detail}', y0+.020*height, .017*height)):
        curve = bpy.data.curves.new('Motion status text', 'FONT')
        curve.body, curve.size = text, size
        curve.align_x = 'LEFT'
        obj = bpy.data.objects.new('Motion status text', curve)
        scene.collection.objects.link(obj)
        tag(obj)
        obj.location = Vector((x0+.013*width, y, -depth+.0001))
        curve.materials.append(foreground)


def build(job, clip, arrays, model, spec, mapping, out, render_time=None, image=None, input_hashes=None, verification=None, *, save_scene=True):
    import bpy
    from mathutils import Quaternion, Vector
    from doorbench.appearance.blender_worker import reset_scene, build_door, primitive, explicit_camera, transform
    from doorbench.appearance.blender_materials import configure_texture_library
    from doorbench.appearance.blender_environment import configure_scene, build_environment, frame_camera, _context
    from doorbench.appearance.blender_details import build_details

    duration = float(arrays['actor_time'][-1])
    sec = duration / 2 if render_time is None else float(render_time)
    if not math.isfinite(sec) or not 0 <= sec <= duration:
        raise ValueError('render-time must lie within actor_time')
    reset_scene()
    configure_texture_library(job.get('texture_library'))
    camera_state = job['state'].get('camera')
    width, height = camera_state['resolution'] if camera_state else (job['width'], job['height'])
    scene = configure_scene(job['recipe'], quality=job['quality'], width=width, height=height, seed=job['seed'])
    objects = build_door(model, spec, job['reference_state'], job['recipe'], Path(job['hardware_dir']))
    build_environment(model, spec, job['recipe'], objects)
    build_details(objects, spec, job['seed'])
    fps, times = int(clip['fps']), arrays['actor_time']
    bodies = {obj['doorbench_body']: obj for obj in scene.objects if obj.type == 'EMPTY' and obj.get('doorbench_body')}
    for name, obj in bodies.items():
        index = mapping[name]
        _animate(obj, times, arrays['body_pos'][:, index], arrays['body_quat'][:, index], fps)

    material = bpy.data.materials.new('Original DoorBench matte teal rig')
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (.035, .19, .21, 1)
    shader.inputs['Roughness'].default_value = .62
    shader.inputs['Metallic'].default_value = 0.
    shader.inputs['Specular IOR Level'].default_value = .28
    actor_bodies = {}
    for index, name in enumerate(clip['actor']['body_names']):
        obj = bpy.data.objects.new('planned_actor_body::'+name, None)
        scene.collection.objects.link(obj)
        obj['doorbench_actor_body'] = name
        obj['doorbench_visual_only'] = True
        actor_bodies[name] = obj
        _animate(obj, times, arrays['actor_body_pos'][:, index], arrays['actor_body_quat'][:, index], fps)
    actor_objects = {}
    bounds = []
    for geom in clip['actor']['geometries']:
        size_count = {'box': 3, 'sphere': 1, 'capsule': 2, 'cylinder': 2}[geom['type']]
        obj = primitive({'name': 'planned_actor_geom::'+geom['name'], 'type': geom['type'], 'size': geom['size'][:size_count]}, {}, Path(job['hardware_dir']))
        # Exact rig collision surfaces take precedence over decorative edge bevels.
        for modifier in list(obj.modifiers):
            obj.modifiers.remove(modifier)
        obj.parent = actor_bodies[geom['body_name']]
        obj.matrix_basis = transform(geom['pos'], geom['quat_wxyz'])
        obj.data.materials.append(material)
        obj['doorbench_actor_geom'] = geom['name']
        obj['doorbench_actor_body'] = geom['body_name']
        obj['doorbench_visual_only'] = True
        obj['doorbench_semantic'] = 'planned_rig_collision_surface'
        obj['doorbench_primitive_spec'] = json.dumps(geom, sort_keys=True)
        actor_objects[geom['name']] = obj
        index = clip['actor']['body_names'].index(geom['body_name'])
        corners = geometry_world_corners(geom, arrays['actor_body_pos'][:, index], arrays['actor_body_quat'][:, index])
        bounds.extend([corners.min(axis=(0,1)), corners.max(axis=(0,1))])
    bpy.context.view_layer.update()
    for obj in objects.values():
        if _context(obj):
            continue
        local = np.array([tuple(obj.matrix_basis @ Vector(corner)) for corner in obj.bound_box])
        index = mapping[obj['doorbench_body']]
        world = _rotate(arrays['body_quat'][:, index, None, :], local[None, :, :]) + arrays['body_pos'][:, index, None, :]
        bounds.extend([world.min(axis=(0,1)), world.max(axis=(0,1))])
    low, high = np.min(bounds, axis=0), np.max(bounds, axis=0)
    framing_low = low.copy()
    # Reserve a caption band outside the actual trajectory to keep ankles visible.
    framing_low[2] -= max(.20, .15 * (high[2]-low[2]))
    mesh = bpy.data.meshes.new('Temporary planned motion bounds')
    mesh.from_pydata(list(itertools.product(*zip(framing_low, high))), [], [])
    proxy = bpy.data.objects.new('Temporary planned motion bounds', mesh)
    scene.collection.objects.link(proxy)
    bpy.context.view_layer.update()
    scene.camera = explicit_camera(camera_state) if camera_state else frame_camera({'trajectory': proxy}, spec, view=job['view'], width=width, height=height)
    bpy.data.objects.remove(proxy, do_unlink=True)
    bpy.data.meshes.remove(mesh)
    heading, detail = validation_label(clip, verification)
    _status_card(scene, heading, detail, clip['door_id'])
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, math.ceil(1 + duration * fps)
    frame = 1 + sec * fps
    scene.frame_set(math.floor(frame), subframe=frame-math.floor(frame))
    scene['doorbench_motion_scope'] = 'Constrained motion candidate with exact rig surfaces; kinematic replay, not a dynamics certification.'
    scene['doorbench_plan_status'] = clip.get('status', 'proposal')
    scene['doorbench_plan_qa'] = json.dumps(clip.get('qa', {}), sort_keys=True)
    if verification is not None:
        scene['doorbench_independent_validation'] = json.dumps(verification, sort_keys=True)
    metadata = {
        'schema': 'doorbench.blender-planned-motion.v1', 'door_id': clip['door_id'],
        'job_sha256': job['job_sha256'], 'source_sha256': job['source_sha256'],
        'renderer_sha256': job['renderer_sha256'], 'mesh_sha256': job['mesh_sha256'],
        'texture_library': job.get('texture_library'), **(input_hashes or {}),
        'script_sha256': sha256(__file__), 'replay_helper_sha256': sha256(ROOT/'scripts/blender_reference_motion.py'),
        'status': clip.get('status', 'proposal'), 'qa': clip.get('qa', {}),
        'independent_validation': verification,
        'display_title': heading, 'display_detail': detail,
        'native_body_mapping': mapping, 'actor_body_names': clip['actor']['body_names'],
        'actor_geometry_count': len(actor_objects), 'source_visual_geoms': len(objects),
        'actor_geometry_space': 'Primitive-local -> actor body world; no visual limb scaling or IK reconstruction',
        'timeline': 'All poses keyed directly on actor_time; native_time is provenance for source time-warp only',
        'native_time_range': [float(arrays['native_time'][0]), float(arrays['native_time'][-1])],
        'samples': len(times), 'fps': fps, 'duration': duration, 'render_time': sec,
        'camera': 'Explicit calibrated camera; actor coverage not guaranteed' if camera_state else 'Fixed complete trajectory bounds with status-caption margin',
        'trajectory_bounds': {'low': low.tolist(), 'high': high.tolist()},
        'limitations': clip.get('limitations', []) + ['Export does not certify physical balance, hand force, or continuous scene clearance.',
            'Original sampled poses are authoritative; Blender display interpolation is not the validator\'s geodesic interpolation.'],
        'interpolation': 'Linear position/quaternion components between samples; recorded sample poses are authoritative.',
        'blender_version': bpy.app.version_string,
    }
    provenance = bpy.data.texts.new('DoorBench planned motion provenance.json')
    provenance.write(json.dumps(metadata, indent=2))
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if image:
        image = Path(image).resolve()
        image.parent.mkdir(parents=True, exist_ok=True)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = str(image)
    if save_scene:
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
    if image:
        bpy.ops.render.render(write_still=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'clip', 'trajectory', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--render-time', type=float)
    parser.add_argument('--image')
    parser.add_argument('--validation', help='Optional accepted independent report bound to this exact clip and trajectory')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    out = Path(args.out).resolve()
    if out.suffix != '.blend':
        raise ValueError('--out must be a .blend file')
    sidecar = out.with_suffix('.metadata.json')
    inputs_paths = {Path(p).resolve() for p in (args.job, args.clip, args.trajectory)}
    if args.validation:
        inputs_paths.add(Path(args.validation).resolve())
    outputs = [out, sidecar] + ([Path(args.image).resolve()] if args.image else [])
    if inputs_paths.intersection(outputs) or len(outputs) != len(set(outputs)):
        raise ValueError('Output paths must be distinct and must not overwrite motion inputs')
    input_hashes = {'clip_sha256': sha256(args.clip), 'trajectory_sha256': sha256(args.trajectory)}
    inputs = load_inputs(args.job, args.clip, args.trajectory)
    if input_hashes != {'clip_sha256': sha256(args.clip), 'trajectory_sha256': sha256(args.trajectory)}:
        raise ValueError('Motion inputs changed during validation')
    verification = load_validation_report(args.validation, args.clip, args.trajectory, inputs[0]) if args.validation else None
    if verification is not None:
        input_hashes['validation_sha256'] = verification['report_sha256']
    metadata = build(*inputs, out, args.render_time, args.image, input_hashes, verification)
    current = {'clip_sha256': sha256(args.clip), 'trajectory_sha256': sha256(args.trajectory)}
    if args.validation:
        current['validation_sha256'] = sha256(args.validation)
    if current != input_hashes:
        raise ValueError('Motion or validation report changed during export; rendered output is not verified')
    metadata['blend_sha256'] = sha256(out)
    if args.image:
        metadata['image_sha256'] = sha256(args.image)
    sidecar.write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()

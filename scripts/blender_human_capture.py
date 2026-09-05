#!/usr/bin/env python3
"""Bake a licensed BVH capture onto a calibrated anatomical Blender rig.

Run with Blender --background --factory-startup --python-exit-code 1 --python
this_script.py -- --source capture.bvh --human human-preview.blend
--calibration tpose-calibration.json --out output [--video] [--glb].

The target keeps its own calibrated lengths and offsets. This is direct motion
transfer at the source clock, not an IK solver or physical interaction model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import struct
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from doorbench.human_reference.bvh import forward_kinematics, read_bvh

BASIS = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256')
    return digest.hexdigest()


def normalize_glb_alpha(path, *, expected_action=None):
    """Give exported human surfaces explicit glTF opacity semantics.

    MPFB's generic shader graphs mark even fully solid skin/clothes BLEND.
    That causes transparent sorting to show teeth through the face in WebGL.
    Keep the binary mesh/skin/texture/animation chunk byte-identical; only solid
    versus cutout material metadata changes. Blender materials remain untouched.
    """
    path = Path(path)
    data = path.read_bytes()
    magic, version, length = struct.unpack_from('<III', data)
    if (magic, version, length) != (0x46546C67, 2, len(data)):
        raise ValueError('Invalid binary glTF header')
    chunks, offset = [], 12
    while offset < len(data):
        size, kind = struct.unpack_from('<II', data, offset)
        payload = data[offset + 8:offset + 8 + size]
        if len(payload) != size:
            raise ValueError('Truncated glTF chunk')
        chunks.append((kind, payload))
        offset += 8 + size
    if chunks[0][0] != 0x4E4F534A:
        raise ValueError('glTF first chunk must be JSON')
    document = json.loads(chunks[0][1])
    if expected_action is not None:
        selected = [a for a in document.get('animations', []) if a.get('name') == expected_action]
        if len(selected) != 1:
            raise ValueError('Export must contain exactly one requested animation')
        document['animations'] = selected
    changes = []
    for material in document.get('materials', []):
        name = material.get('name', '')
        cutout = any(token in name.lower() for token in ('eyebrow', 'eyelash', 'short02'))
        old = material.get('alphaMode', 'OPAQUE')
        material['alphaMode'] = 'MASK' if cutout else 'OPAQUE'
        if cutout:
            material['alphaCutoff'] = .45
        else:
            material.pop('alphaCutoff', None)
        changes.append({'material': name, 'old': old, 'new': material['alphaMode']})
    payload = json.dumps(document, separators=(',', ':')).encode()
    payload += b' ' * (-len(payload) % 4)
    chunks[0] = (chunks[0][0], payload)
    body = b''.join(struct.pack('<II', len(value), kind) + value for kind, value in chunks)
    path.write_bytes(struct.pack('<III', magic, version, len(body) + 12) + body)
    return changes


def mapping(names):
    """Absolute source rotations; extra target segments never double angles."""
    result = {'root': 'Character1_Hips'}
    for target, source in [('spine05', 'Spine'), ('spine04', 'Spine'),
                           ('spine03', 'Spine1'), ('spine02', 'Spine1'),
                           ('spine01', 'Spine2'), ('neck01', 'Neck'),
                           ('neck02', 'Neck'), ('neck03', 'Neck'), ('head', 'Neck')]:
        result[target] = 'Character1_' + source
    for side, label in [('L', 'Left'), ('R', 'Right')]:
        result['pelvis.' + side] = 'Character1_Hips'
        for targets, source in [(['clavicle', 'shoulder01'], 'Shoulder'),
                                 (['upperarm01', 'upperarm02'], 'Arm'),
                                 (['lowerarm01', 'lowerarm02'], 'ForeArm'),
                                 (['wrist'], 'Hand'),
                                 (['upperleg01', 'upperleg02'], 'UpLeg'),
                                 (['lowerleg01', 'lowerleg02'], 'Leg'),
                                 (['foot'], 'Foot')]:
            for target in targets:
                result[target + '.' + side] = 'Character1_' + label + source
        for number, finger in [(1, 'Thumb'), (2, 'Index'), (3, 'Middle'),
                                (4, 'Ring'), (5, 'Pinky')]:
            # MPFB thumb includes a metacarpal-like proximal segment. For other
            # digits the unmeasured DIP shares the PIP's absolute rotation.
            source_indices = (1, 1, 2) if number == 1 else (1, 2, 2)
            for segment, source_index in enumerate(source_indices, 1):
                result[f'finger{number}-{segment}.{side}'] = (
                    f'Character1_{label}Hand{finger}{source_index}')
    unknown = set(result) - set(names)
    if unknown:
        raise ValueError(f'Mapping names absent from target: {sorted(unknown)}')
    return result


def construct(source, calibration, *, trim=2):
    if (calibration.get('unit'), calibration.get('up_axis'), calibration.get('forward_axis'),
            calibration.get('left_axis')) != ('m', 'Z', '+Y', '-X'):
        raise ValueError('Expected calibrated metre/Z-up/+Y-forward/-X-left target')
    capture = read_bvh(source)
    if calibration['source_bvh_sha256'] != sha(source):
        raise ValueError('Calibration is not bound to this capture')
    if trim != 2 or len(capture.values) <= trim:
        raise ValueError('Only the declared two leading calibration rows may be removed')
    if not np.array_equal(capture.values[0], capture.values[1]):
        raise ValueError('The two declared duplicate calibration rows differ')
    raw_pos, raw_rot = forward_kinematics(capture, length_scale=.01, basis=BASIS)
    source_indices = {bone.name: j for j, bone in enumerate(capture.joints)}
    target = calibration['bones']
    names = [b['name'] for b in target]
    indices = {name: j for j, name in enumerate(names)}
    parents = np.array([-1 if b['parent'] is None else indices[b['parent']] for b in target])
    if any(parent >= j for j, parent in enumerate(parents)):
        raise ValueError('Target bones must be ordered parent before child')
    cal = np.array([b['matrix_armature'] for b in target])
    raw_cal_rot, cal_pos = cal[:, :3, :3], cal[:, :3, 3]
    # Blender stores float32 matrices. Its deep facial hierarchy accumulates
    # ~1e-5 scale/shear roundoff, which must not compound as "rotation" at each
    # inherited child. Project each calibration basis to the nearest SO(3).
    u, singular, vh = np.linalg.svd(raw_cal_rot)
    cal_rot = u @ vh
    if np.min(np.linalg.det(cal_rot)) < .99999 or np.max(abs(singular - 1)) > 1e-4:
        raise ValueError('Calibration contains material scale/shear or reflection')
    rest = np.array([b['rest_matrix_local'] for b in target])
    source_map = mapping(names)
    count, bone_count = len(capture.values) - trim, len(target)
    rotations = np.empty((count, bone_count, 3, 3))
    positions = np.empty((count, bone_count, 3))
    offsets = np.zeros((bone_count, 3))
    root_j = indices['root']
    hips_j = source_indices['Character1_Hips']
    root_tail = np.asarray(target[root_j]['tail'])
    floor = float(calibration['calibration_floor_z_m'])
    # Fixed anthropometric adjustment makes target calibration pelvis height
    # agree with its own sole plane. Preserve every source horizontal sample
    # and every source vertical displacement thereafter.
    pelvis_offset = np.array([0., 0., root_tail[2] - floor - raw_pos[0, hips_j, 2]])
    pelvis = raw_pos[trim:, hips_j] + pelvis_offset
    for j, bone in enumerate(target):
        parent = parents[j]
        if bone['name'] in source_map:
            source_j = source_indices[source_map[bone['name']]]
            rotations[:, j] = raw_rot[trim:, source_j] @ cal_rot[j]
        elif parent >= 0:
            rotations[:, j] = rotations[:, parent] @ (cal_rot[parent].T @ cal_rot[j])
        else:
            raise ValueError('Unmapped root')
        if parent < 0:
            tail_offset = cal_rot[j].T @ (root_tail - cal_pos[j])
            positions[:, j] = pelvis - np.einsum('tij,j->ti', rotations[:, j], tail_offset)
        else:
            offsets[j] = cal_rot[parent].T @ (cal_pos[j] - cal_pos[parent])
            positions[:, j] = positions[:, parent] + np.einsum(
                'tij,j->ti', rotations[:, parent], offsets[j])
    matrices = np.tile(np.eye(4), (count, bone_count, 1, 1))
    matrices[:, :, :3, :3] = rotations
    matrices[:, :, :3, 3] = positions
    # Blender full-inheritance FK: P_child = P_parent R_parent^-1 R_child B_child.
    # Solve B explicitly to retain calibrated offsets and avoid copy-rotation
    # constraints, dependency graph cycles, or accidental parent-angle doubling.
    local_basis = np.empty_like(matrices)
    for j, parent in enumerate(parents):
        if parent < 0:
            local_basis[:, j] = np.linalg.inv(rest[j]) @ matrices[:, j]
        else:
            local_basis[:, j] = (np.linalg.inv(rest[j]) @ rest[parent]
                                 @ np.linalg.inv(matrices[:, parent]) @ matrices[:, j])
    lengths = np.array([math.dist(b['head'], b['tail']) for b in target])
    tails = positions + rotations[:, :, :, 1] * lengths[None, :, None]
    length_error = float(np.max(np.abs(np.linalg.norm(tails - positions, axis=-1) - lengths)))
    ortho_error = float(np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3))))
    determinant_error = float(np.max(np.abs(np.linalg.det(rotations) - 1)))
    if max(length_error, ortho_error, determinant_error) > 2e-5:
        raise ValueError('Constructed rigid transforms failed integrity checks')
    times = np.arange(count) * capture.frame_time
    arrays = dict(time=times, source_time=times + trim * capture.frame_time,
                  source_frame=np.arange(trim, len(capture.values)), bone_names=np.array(names),
                  parent_index=parents, bone_pos=positions, bone_tail=tails,
                  bone_rot=rotations, bone_matrix=matrices, bone_basis=local_basis,
                  calibration_parent_offsets=offsets, pelvis_pos=pelvis,
                  source_joint_names=np.array([b.name for b in capture.joints]),
                  source_joint_pos=raw_pos[trim:], source_joint_rot=raw_rot[trim:])
    details = dict(source_total_frames=len(capture.values), retained_frames=count,
                   source_frame_start=trim, source_clock_offset_s=trim * capture.frame_time,
                   source_frame_time_s=capture.frame_time, duration_s=float(times[-1]),
                   coordinate_basis=BASIS.tolist(), length_scale=.01,
                   pelvis_constant_offset_m=pelvis_offset.tolist(),
                   source_calibration_hips_m=raw_pos[0, hips_j].tolist(),
                   target_calibration_pelvis_m=root_tail.tolist(),
                   target_calibration_floor_z_m=floor,
                   calibration_rotation_projection_max_element_change=float(np.max(abs(cal_rot - raw_cal_rot))),
                   mapping=source_map, inherited_bones=sorted(set(names) - set(source_map)),
                   max_bone_length_error_m=length_error,
                   max_rotation_orthogonality_error=ortho_error,
                   max_rotation_determinant_error=determinant_error)
    return arrays, details


def bake(rig, arrays):
    import bpy
    from mathutils import Matrix
    count = len(arrays['time'])
    frames = np.arange(1, count + 1, dtype=np.float64)
    rig.animation_data_clear()
    for p in rig.pose.bones:
        if p.constraints:
            raise ValueError(f'Unexpected pose constraint on {p.name}')
        if not p.bone.use_inherit_rotation or p.bone.inherit_scale != 'FULL':
            raise ValueError(f'Unsupported inheritance mode on {p.name}')
        p.rotation_mode = 'QUATERNION'
        p.scale = (1, 1, 1)
    rig.animation_data_create()
    action = bpy.data.actions.new('CeTI_d02_o03.original_clock')
    slot = action.slots.new(id_type='OBJECT', name=rig.name)
    rig.animation_data.action = action
    rig.animation_data.action_slot = slot
    strip = action.layers.new('Source capture').strips.new(type='KEYFRAME')
    bag = strip.channelbags.new(slot)
    quats = np.empty((count, len(arrays['bone_names']), 4))
    for j, name in enumerate(arrays['bone_names']):
        matrices = arrays['bone_basis'][:, j]
        q = np.array([tuple(Matrix(m.tolist()).to_quaternion().normalized()) for m in matrices])
        for k in range(1, count):
            if np.dot(q[k - 1], q[k]) < 0:
                q[k] *= -1
        quats[:, j] = q
        for prop, values in [('location', matrices[:, :3, 3]), ('rotation_quaternion', q)]:
            path = rig.pose.bones[str(name)].path_from_id(prop)
            for component in range(values.shape[1]):
                curve = bag.fcurves.new(data_path=path, index=component)
                curve.keyframe_points.add(count)
                curve.keyframe_points.foreach_set('co', np.column_stack(
                    (frames, values[:, component])).astype(np.float32).ravel())
                for point in curve.keyframe_points:
                    point.interpolation = 'LINEAR'
                curve.update()
    arrays['bone_local_quat_wxyz'] = quats
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = count
    bpy.context.scene.render.fps = 100
    bpy.context.scene.render.fps_base = 100 * float(arrays['time'][1] - arrays['time'][0])
    return action


def validate_bake(rig, arrays):
    import bpy
    max_pos = max_rot = max_length = 0.
    # Check every baked source frame, not only sparse rendered phases.
    lengths = np.linalg.norm(arrays['bone_tail'][0] - arrays['bone_pos'][0], axis=1)
    for k in range(len(arrays['time'])):
        bpy.context.scene.frame_set(k + 1)
        for j, name in enumerate(arrays['bone_names']):
            p = rig.pose.bones[str(name)]
            actual = np.array(p.matrix)
            max_pos = max(max_pos, float(np.max(np.abs(actual[:3, 3] - arrays['bone_pos'][k, j]))))
            max_rot = max(max_rot, float(np.max(np.abs(actual[:3, :3] - arrays['bone_rot'][k, j]))))
            max_length = max(max_length, abs((p.tail - p.head).length - lengths[j]))
    if max_pos > 3e-5 or max_rot > 3e-5 or max_length > 3e-5:
        raise ValueError(f'Blender FK disagrees with constructed poses: {max_pos}, {max_rot}, {max_length}')
    return {'frames_checked': len(arrays['time']), 'bones_checked': len(arrays['bone_names']),
            'max_blender_fk_position_error_m': max_pos,
            'max_blender_fk_rotation_matrix_error': max_rot,
            'max_blender_bone_length_error_m': max_length}


def camera_at(scene, center, view):
    from mathutils import Vector
    offsets = {'front': (0, 5.7, 2.4), 'profile': (5.7, 0, 2.0), 'three_quarter': (3.6, 4.8, 2.6)}
    center = Vector((center[0], center[1], .91))
    scene.camera.location = Vector((center.x, center.y, 0)) + Vector(offsets[view])
    scene.camera.rotation_euler = (center - scene.camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera.data.type = 'ORTHO'
    scene.camera.data.ortho_scale = 2.22


def render_proofs(out, arrays, *, video, phases=None, views=None):
    import bpy
    scene = bpy.context.scene
    dt = float(arrays['time'][1] - arrays['time'][0])
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == 'METAL'
        scene.cycles.device = 'GPU'
    except (AttributeError, TypeError, RuntimeError):
        scene.cycles.device = 'CPU'
    scene.render.resolution_x = 720
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    if phases is None:
        phases = [0., 2.87, 5.3, 1.4, 3.8, float(arrays['time'][-1])]
    if views is None:
        views = ('three_quarter', 'front', 'profile')
    records = []
    for phase in phases:
        k = int(round(phase / dt))
        k = min(k, len(arrays['time']) - 1)
        scene.frame_set(k + 1)
        for view in views:
            camera_at(scene, arrays['pelvis_pos'][k], view)
            path = out / 'proof' / f'{k:04d}_{view}.png'
            path.parent.mkdir(exist_ok=True)
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            records.append({'path': str(path.relative_to(out)), 'source_frame': k + 2,
                            'source_time_s': float(arrays['source_time'][k]),
                            'motion_time_s': float(arrays['time'][k]), 'view': view,
                            'sha256': sha(path), 'bytes': path.stat().st_size})
    if video:
        folder = out / 'video_frames'
        folder.mkdir(exist_ok=True)
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_render_samples = 32
        scene.render.resolution_x = 768
        scene.render.resolution_y = 768
        preview_times = np.arange(math.floor(float(arrays['time'][-1]) * 30) + 1) / 30
        for i, t in enumerate(preview_times):
            f = 1 + t / dt
            scene.frame_set(math.floor(f), subframe=f - math.floor(f))
            center = [np.interp(t, arrays['time'], arrays['pelvis_pos'][:, axis]) for axis in range(3)]
            camera_at(scene, center, 'three_quarter')
            scene.render.filepath = str(folder / f'{i:04d}.png')
            bpy.ops.render.render(write_still=True)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-i',
                        str(folder / '%04d.png'), '-c:v', 'libx264', '-crf', '18',
                        '-frames:v', str(len(preview_times)),
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'normal-speed.mp4')], check=True)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--human', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--glb', action='store_true')
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args(argv)
    if args.video and args.no_render:
        parser.error('--video cannot be combined with --no-render')
    import bpy
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(args.calibration.read_text())
    rig_metadata_path = args.calibration.parent / 'rig.json'
    if sha(rig_metadata_path) != calibration['target_rig_json_sha256']:
        raise ValueError('Target rig metadata hash differs from calibration binding')
    arrays, details = construct(args.source, calibration)
    bpy.ops.wm.open_mainfile(filepath=str(args.human.resolve()))
    rig = bpy.data.objects['Human.rig']
    if not np.allclose(np.array(rig.matrix_world), np.eye(4), atol=1e-7, rtol=0):
        raise ValueError('Expected an identity-transform canonical rig')
    for bone in calibration['bones']:
        if not np.allclose(np.array(rig.data.bones[bone['name']].matrix_local),
                           bone['rest_matrix_local'], atol=1e-7, rtol=0):
            raise ValueError(f'Rest rig differs from calibration: {bone["name"]}')
    action = bake(rig, arrays)
    checks = validate_bake(rig, arrays)
    np.savez_compressed(args.out / 'poses.npz', **arrays)
    bpy.context.scene.frame_set(1)
    camera_at(bpy.context.scene, arrays['pelvis_pos'][0], 'three_quarter')
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / 'motion.blend'), compress=True)
    produced = [args.out / 'motion.blend', args.out / 'poses.npz']
    if args.glb:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.collections['Human'].objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.export_scene.gltf(filepath=str(args.out / 'animation.glb'), export_format='GLB',
                                  use_selection=True, export_animations=True, export_frame_range=True,
                                  export_frame_step=1, export_force_sampling=True,
                                  export_anim_slide_to_zero=True,
                                  export_animation_mode='ACTIVE_ACTIONS',
                                  export_nla_strips_merged_animation_name=action.name)
        normalize_glb_alpha(args.out / 'animation.glb', expected_action=action.name)
        produced.append(args.out / 'animation.glb')
    proofs = [] if args.no_render else render_proofs(args.out, arrays, video=args.video)
    if args.video:
        produced.append(args.out / 'normal-speed.mp4')
    attribution = dict(calibration['source_attribution'])
    attribution['modifications'] = (
        'Removed exactly the first two duplicate calibration rows (0.02s); converted centimetres/Y-up '
        'to metres/Z-up/+Y-forward; transferred source global rotations at original100Hz timing onto '
        'a fixed calibrated target anthropometry with explicit finger-segment approximations and '
        'a constant pelvis-height offset. No filtering, retiming, path warping or door fitting.')
    metadata = {'schema': 'doorbench.human-capture-transfer.v1',
                'title': 'Human capture transfer candidate — door not fitted',
                'status': 'source_capture_transfer_unvalidated_interaction',
                'source': {'path': str(args.source), 'sha256': sha(args.source),
                           'attribution': attribution},
                'calibration_attribution': calibration['source_attribution'],
                'rig_metadata_sha256': sha(rig_metadata_path),
                'calibration_sha256': sha(args.calibration), 'human_blend_sha256': sha(args.human),
                'script_sha256': sha(__file__), 'parser_sha256': sha(REPO / 'doorbench/human_reference/bvh.py'),
                'blender': bpy.app.version_string, 'action': action.name,
                **details, 'baked_fk_checks': checks, 'proofs': proofs,
                'preview': {'camera': 'pelvis-translation following; source heading remains unchanged',
                            'fps': 30 if args.video else None,
                            'frame_interpolation': 'linear local quaternion components normalized by Blender',
                            'clock': 'Original 100 Hz capture retained in .blend and NPZ; MP4 uses 30 Hz display sampling.'},
                'limitations': [
                    'Source motion is CC BY 4.0 IMU-based capture; target character is CC0. Not relabeled MIT.',
                    'Only two known leading calibration rows removed; no retiming, filtering, path warping, IK or old guide motion.',
                    'Fixed target anthropometry differs from recorded subject; source rhythm and global rotations are preserved.',
                    'Untracked face and toes inherit parent motion. Head inherits Neck because this source has no independent Head joint.',
                    'Target non-thumb middle/distal share source joint2; thumb first two segments share Thumb1. No independently tracked DIP is claimed.',
                    'No captured door channel, scene registration, measured contact, force, balance or physical feasibility claim.',
                    'Skin and clothes use authored deformation, not muscle or cloth dynamics.'],
                'artifacts': []}
    if args.glb:
        metadata['limitations'].append(
            'GLB is a browser preview: the exporter keeps at most4 skin influences per vertex and '
            'approximates enhanced Blender skin shaders. The packed .blend is the authoritative skinned render.')
    for file in produced:
        metadata['artifacts'].append({'path': file.name, 'sha256': sha(file), 'bytes': file.stat().st_size})
    (args.out / 'motion.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'completed': str(args.out), 'duration_s': details['duration_s'], 'checks': checks}), flush=True)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])

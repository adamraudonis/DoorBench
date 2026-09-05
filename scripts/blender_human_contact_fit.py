#!/usr/bin/env python3
"""Bake/check an explicitly adjusted human pose candidate without replacing raw.

Blender --background --factory-startup --python-exit-code 1 --python SCRIPT --
--raw out/human-reference/ceti-d02-o03 --poses ADJUSTED.npz --report REPORT.json
--out NEW_DIRECTORY [--glb] [--video]. Only report-bound, original-clock arrays
are accepted. The output remains a target contact-adjustment candidate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('doorbench_capture_baker', ROOT / 'scripts/blender_human_capture.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def sample_shoes(out, *, times, source_times, crossing_interval=(5., 5.6)):
    """Evaluate the same subdivision levels and modifiers as the final render."""
    import bpy
    obj = bpy.data.objects['Human.shoes01']
    state = [(m, m.show_viewport, getattr(m, 'levels', None)) for m in obj.modifiers]
    try:
        for modifier in obj.modifiers:
            modifier.show_viewport = modifier.show_render
            if modifier.type == 'SUBSURF':
                modifier.levels = modifier.render_levels
        bpy.context.view_layer.update()
        report = _sample_shoes_current_modifiers(
            out, times=times, source_times=source_times, crossing_interval=crossing_interval)
        report['surface_scope'] = 'Render-equivalent modifier visibility and subdivision levels explicitly evaluated.'
        report['subdivision_render_levels'] = [m.render_levels for m in obj.modifiers if m.type == 'SUBSURF']
        return report
    finally:
        for modifier, visible, levels in state:
            modifier.show_viewport = visible
            if levels is not None:
                modifier.levels = levels
        bpy.context.view_layer.update()


def _sample_shoes_current_modifiers(out, *, times, source_times, crossing_interval):
    """Actual evaluated vertices, with stable authored left/right membership.

    This measures visible shoe surfaces. It does not infer support forces or
    certify collisions. Saved triangle clouds allow an independent checker to
    test the source-clock crossing interval without repeating Blender setup.
    """
    import bpy
    out = Path(out)
    scene = bpy.context.scene
    obj = bpy.data.objects['Human.shoes01']
    names = {g.index: g.name for g in obj.vertex_groups}
    scene.frame_set(1)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        side = np.array([0 if sum(g.weight for g in v.groups if names[g.group].endswith('.L'))
                         > sum(g.weight for g in v.groups if names[g.group].endswith('.R'))
                         else 1 for v in mesh.vertices])
        if set(side) != {0, 1}:
            raise ValueError('Shoe mesh does not contain both authored foot groups')
        vertex_count = len(mesh.vertices)
        mesh.calc_loop_triangles()
        triangles = np.empty(len(mesh.loop_triangles) * 3, np.int32)
        mesh.loop_triangles.foreach_get('vertices', triangles)
        triangles = triangles.reshape(-1, 3)
        if not np.all(side[triangles] == side[triangles[:, 0]][:, None]):
            raise ValueError('A shoe triangle crosses left/right authored groups')
    finally:
        evaluated.to_mesh_clear()
    result = {k: [] for k in ['min_z', 'max_z', 'centroid', 'min_xyz', 'max_xyz']}
    clouds, cloud_frames = [], []
    for frame in range(1, len(times) + 1):
        scene.frame_set(frame)
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            if len(mesh.vertices) != vertex_count:
                raise ValueError('Shoe topology changed during animation')
            flat = np.empty(vertex_count * 3, dtype=np.float32)
            mesh.vertices.foreach_get('co', flat)
            matrix = np.array(obj.matrix_world)
            co = flat.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
            feet = [co[side == foot] for foot in (0, 1)]
            result['min_z'].append([f[:, 2].min() for f in feet])
            result['max_z'].append([f[:, 2].max() for f in feet])
            result['centroid'].append([f.mean(axis=0) for f in feet])
            result['min_xyz'].append([f.min(axis=0) for f in feet])
            result['max_xyz'].append([f.max(axis=0) for f in feet])
            if crossing_interval[0] - 1e-9 <= source_times[frame - 1] <= crossing_interval[1] + 1e-9:
                clouds.append(co)
                cloud_frames.append(frame - 1)
        finally:
            evaluated.to_mesh_clear()
    arrays = {k: np.asarray(v) for k, v in result.items()}
    np.savez_compressed(out / 'shoe-surface.npz', time=times, source_time=source_times,
                        foot_names=np.array(['left', 'right']), **arrays)
    selected = np.array(cloud_frames, dtype=int)
    np.savez_compressed(out / 'shoe-clouds-source5.0-5.6.npz', time=times[selected],
                        source_time=source_times[selected], vertices=np.asarray(clouds),
                        vertex_side=side, triangles=triangles)
    return {'vertices': vertex_count, 'frames_checked': len(times),
            'minimum_surface_z_m': arrays['min_z'].min(axis=0).tolist(),
            'maximum_surface_minimum_z_m': arrays['min_z'].max(axis=0).tolist(),
            'foot_side_assignment': 'Stable authored deform groups .L/.R; no current-world-X classification.',
            'coordinate_frame': 'metres/Z-up/+Y-forward',
            'crossing_cloud_source_interval_s': list(crossing_interval),
            'crossing_frames': len(selected), 'contact_claim': 'None; actual visible shoe surfaces only.'}


def check_inputs(raw, poses, report_path):
    raw, poses, report_path = Path(raw), Path(poses), Path(report_path)
    metadata = json.loads((raw / 'motion.json').read_text())
    report = json.loads(report_path.read_text())
    if capture.sha(raw / 'poses.npz') != report['raw_poses_sha256']:
        raise ValueError('Adjustment report raw-pose hash differs')
    if capture.sha(poses) != report['output_poses_sha256']:
        raise ValueError('Adjustment report output-pose hash differs')
    if report['source_bvh_sha256'] != metadata['source']['sha256']:
        raise ValueError('Adjustment source capture hash differs')
    if 'contact_fit_source_sha256' in report:
        if capture.sha(ROOT / 'doorbench/human_reference/contact_fit.py') != report['contact_fit_source_sha256']:
            raise ValueError('Adjustment generator source changed after proposal')
    raw_arrays = dict(np.load(raw / 'poses.npz', allow_pickle=False))
    arrays = dict(np.load(poses, allow_pickle=False))
    for key in ('time', 'source_time', 'source_frame', 'bone_names', 'parent_index',
                'source_joint_names', 'source_joint_pos', 'source_joint_rot'):
        if not np.array_equal(raw_arrays[key], arrays[key]):
            raise ValueError(f'Original clock/source/topology changed: {key}')
    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in 'fc'):
        raise ValueError('Nonfinite candidate pose')
    if report.get('unchanged_clock_pelvis_upperbody_source_arrays'):
        unchanged = [j for j, name in enumerate(arrays['bone_names']) if name not in report['affected_bones']]
        for key in ('bone_pos', 'bone_tail', 'bone_rot', 'bone_matrix'):
            if not np.array_equal(raw_arrays[key][:, unchanged], arrays[key][:, unchanged]):
                raise ValueError(f'Report claims unchanged upper body but differs: {key}')
        if not np.array_equal(raw_arrays['pelvis_pos'], arrays['pelvis_pos']):
            raise ValueError('Report claims unchanged pelvis but differs')
    raw_lengths = np.linalg.norm(raw_arrays['bone_tail'] - raw_arrays['bone_pos'], axis=-1)
    lengths = np.linalg.norm(arrays['bone_tail'] - arrays['bone_pos'], axis=-1)
    if np.max(abs(lengths - raw_lengths)) > 2e-5:
        raise ValueError('Candidate changed fixed target bone lengths')
    return arrays, metadata, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--poses', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--glb', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args(argv)
    if args.video and args.no_render:
        parser.error('--video cannot be combined with --no-render')
    import bpy
    out = args.out.resolve()
    if out == args.raw.resolve():
        raise ValueError('Candidate must use a separate output directory')
    out.mkdir(parents=True, exist_ok=True)
    arrays, base, report = check_inputs(args.raw, args.poses, args.report)
    raw_blend = args.raw / 'motion.blend'
    expected = next(x['sha256'] for x in base['artifacts'] if x['path'] == 'motion.blend')
    if capture.sha(raw_blend) != expected:
        raise ValueError('Raw scene hash changed')
    bpy.ops.wm.open_mainfile(filepath=str(raw_blend.resolve()))
    rig = bpy.data.objects['Human.rig']
    action = capture.bake(rig, arrays)
    action.name = 'CeTI_d02_o03.target_leg_contact_candidate'
    checks = capture.validate_bake(rig, arrays)
    bpy.context.scene.frame_set(1)
    capture.camera_at(bpy.context.scene, arrays['pelvis_pos'][0], 'three_quarter')
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'motion.blend'), compress=True)
    # Existing proposal arrays are immutable; copying is only needed for a
    # different requested output directory. Never rewrite normalized quats over
    # the report-bound NPZ while baking Blender channels.
    if args.poses.resolve() != out / 'poses.npz':
        (out / 'poses.npz').write_bytes(args.poses.read_bytes())
    surfaces = sample_shoes(out, times=arrays['time'], source_times=arrays['source_time'])
    produced = [out / 'motion.blend', out / 'poses.npz', out / 'shoe-surface.npz',
                out / 'shoe-clouds-source5.0-5.6.npz']
    if args.glb:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.collections['Human'].objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.export_scene.gltf(filepath=str(out / 'animation.glb'), export_format='GLB',
                                  use_selection=True, export_animations=True, export_frame_range=True,
                                  export_frame_step=1, export_force_sampling=True, export_anim_slide_to_zero=True,
                                  export_animation_mode='ACTIVE_ACTIONS',
                                  export_nla_strips_merged_animation_name=action.name)
        capture.normalize_glb_alpha(out / 'animation.glb', expected_action=action.name)
        produced.append(out / 'animation.glb')
    proofs = [] if args.no_render else capture.render_proofs(
        out, arrays, video=args.video, phases=[0., 2.87, 5.3])
    if args.video:
        produced.append(out / 'normal-speed.mp4')
    metadata = dict(base)
    metadata.update(title='Human capture with target leg contact adjustment — door not fitted',
                    status='target_leg_contact_adjustment_candidate', action=action.name,
                    raw_motion_metadata_sha256=capture.sha(args.raw / 'motion.json'),
                    raw_poses_sha256=report['raw_poses_sha256'],
                    adjustment_report_sha256=capture.sha(args.report),
                    adjustment_report=report, baked_fk_checks=checks, shoe_surface_checks=surfaces,
                    script_sha256=capture.sha(__file__), capture_baker_sha256=capture.sha(capture.__file__),
                    proofs=proofs, artifacts=[])
    metadata['source'] = dict(base['source'])
    metadata['source']['attribution'] = dict(base['source']['attribution'])
    metadata['raw_transfer_attribution'] = base['source']['attribution']
    metadata['source']['attribution']['modifications'] = (
        'Original100Hz clock, pelvis and upper-body transfer retained. Target leg positions/rotations '
        'receive explicitly authored contact adaptation using source ankleXY and smooth floor-clearance '
        'corrections. No added dynamic/contact-force or original-unmodified-retarget claim.')
    metadata['raw_transfer_limitations'] = base['limitations']
    metadata['limitations'] = [item for item in base['limitations'] if not item.startswith(
        ('Only two known', 'Fixed target anthropometry'))] + report['limitations'] + [
        'This is an adjusted target-leg candidate, separate from the raw capture transfer.',
        'Original clocks and upper-body capture are retained; target thigh/shin rotations and foot positions are adapted.',
        'Fresh deformed shoe checks are measurements, not a force/balance or complete-collision certificate.']
    metadata['preview'] = dict(base['preview'])
    metadata['preview']['video_available'] = args.video
    if not args.video:
        for key in ('fps', 'display_frames', 'last_sample_time_s', 'container_duration_s'):
            metadata['preview'].pop(key, None)
    metadata['glb_animation'] = {'name': action.name, 'start_s': 0.,
                                 'end_s': float(arrays['time'][-1]), 'up_axis': '+Y',
                                 'forward_axis': '-Z', 'export_anim_slide_to_zero': True}
    metadata.pop('raw_geometric_findings', None)
    metadata.pop('surface_diagnostics', None)
    metadata['surface_diagnostics'] = surfaces
    for file in produced:
        metadata['artifacts'].append({'path': file.name, 'sha256': capture.sha(file), 'bytes': file.stat().st_size})
    (out / 'motion.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'completed': str(out), 'checks': checks, 'surfaces': surfaces}), flush=True)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])

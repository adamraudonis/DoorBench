#!/usr/bin/env python
"""Explicit Blender integration checks; no ray rendering, GPU, or pytest dependency.

Run from the DoorBench Python environment:
  python scripts/validate_blender_appearance.py --assets assets --out out/appearance_validation.json

The launcher computes expected geometry poses from native MuJoCo transforms,
including mesh compiler alignment, and starts a separate Blender process. The
Blender process verifies every source visual geom, raw OBJ vertices, outward
box normals, and calibrated camera projection. Synthetic prescribed snapshots
exercise articulation without claiming physically successful opening.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_DOORS = 'db0012_swing_single,db0079_sliding_single,db0188_cold_storage'


def _prepare_cases(assets, door_ids):
    import mujoco
    import numpy as np
    from doorbench.appearance.catalog import resolve_recipe
    from doorbench.appearance.state import capture_initial_state, capture_mujoco_state
    from doorbench.ir import quat_to_mat

    cases = []
    for door_id in door_ids:
        source = assets / 'doors' / door_id
        model_json = json.loads((source / 'model.json').read_text())
        spec = json.loads((source / 'spec.json').read_text())
        model = mujoco.MjModel.from_xml_path(str(source / 'door.xml'))
        initial = capture_initial_state(source)
        for pose_name in ('initial', 'prescribed'):
            data = mujoco.MjData(model)
            data.qpos[:] = initial['qpos_vector']
            if pose_name == 'prescribed':
                # Intentional pose-only stress input; body transforms, not forces,
                # are under test. Native refs and nested bodies remain authoritative.
                for j in range(model.njnt):
                    if model.jnt_limited[j]:
                        low, high = model.jnt_range[j]
                        data.qpos[model.jnt_qposadr[j]] = low + .37 * (high - low)
                    else:
                        data.qpos[model.jnt_qposadr[j]] += .19
            mujoco.mj_forward(model, data)
            state = capture_mujoco_state(model, data, door_id=door_id)
            state['state_kind'] = 'kinematic_inspection'
            state['kinematic_inspection'] = True
            expected = {}
            for body in model_json['bodies']:
                for geom in body['geoms']:
                    if not geom.get('visual', True):
                        continue
                    gid = model.geom(geom['name']).id
                    rotation = data.geom_xmat[gid].reshape(3, 3)
                    position = data.geom_xpos[gid]
                    record = {'type': geom['type']}
                    if geom['type'] == 'mesh':
                        mesh_id = int(model.geom_dataid[gid])
                        align_rotation = quat_to_mat(model.mesh_quat[mesh_id])
                        rotation = rotation @ align_rotation.T
                        position = position - rotation @ model.mesh_pos[mesh_id]
                        scale = model.mesh_scale[mesh_id]
                        if not np.allclose(scale, 1):
                            raise ValueError('This validator requires already-scaled authored OBJ vertices')
                        vertices = []
                        with (assets / 'hardware' / (geom['mesh_name'] + '.obj')).open() as stream:
                            for line in stream:
                                fields = line.split()
                                if fields and fields[0] == 'v':
                                    vertices.append([float(v) for v in fields[1:4]])
                        sample_indices = sorted(set(int(v) for v in np.linspace(0, len(vertices) - 1, min(8, len(vertices)))))
                        record['mesh_vertex_count'] = len(vertices)
                        record['vertices_world'] = {str(i): (position + rotation @ vertices[i]).tolist() for i in sample_indices}
                    matrix = np.eye(4)
                    matrix[:3, :3], matrix[:3, 3] = rotation, position
                    record['world_matrix'] = matrix.tolist()
                    expected[geom['name']] = record
            cases.append({'door_id': door_id, 'pose': pose_name, 'model': model_json, 'spec': spec,
                          'state': state, 'recipe': resolve_recipe(spec, seed=19),
                          'hardware': str(assets / 'hardware'), 'expected': expected})
    return cases


def _inside_blender(config_path, result_path):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Quaternion, Vector
    from doorbench.appearance.blender_worker import build_door, explicit_camera, primitive, reset_scene

    cases = json.loads(config_path.read_text())['cases']
    results, failures = [], []
    total_geoms = 0
    for case in cases:
        try:
            reset_scene()
            objects = build_door(case['model'], case['spec'], case['state'], case['recipe'], Path(case['hardware']))
            if set(objects) != set(case['expected']):
                raise AssertionError('Source visual geom names differ from Blender objects')
            max_matrix, max_vertex = 0.0, 0.0
            for name, expected in case['expected'].items():
                obj = objects[name]
                error = max(abs(obj.matrix_world[i][j] - expected['world_matrix'][i][j]) for i in range(4) for j in range(4))
                max_matrix = max(max_matrix, error)
                if error > 2e-5:
                    raise AssertionError(f'{name}: source/native world transform error {error:g}')
                if expected['type'] == 'mesh':
                    if len(obj.data.vertices) != expected['mesh_vertex_count']:
                        raise AssertionError(f'{name}: original OBJ vertex count changed')
                    for index, position in expected['vertices_world'].items():
                        actual = obj.matrix_world @ obj.data.vertices[int(index)].co
                        error = max(abs(actual[i] - position[i]) for i in range(3))
                        max_vertex = max(max_vertex, error)
                        if error > 2e-5:
                            raise AssertionError(f'{name}: authored mesh/compiler frame error {error:g} m')
            total_geoms += len(objects)
            results.append({'door_id': case['door_id'], 'pose': case['pose'], 'geoms': len(objects),
                            'max_matrix_error': max_matrix, 'max_mesh_vertex_error_m': max_vertex})
        except Exception as error:
            failures.append({'door_id': case['door_id'], 'pose': case['pose'], 'error': str(error)})
    try:
        reset_scene()
        box = primitive({'name': 'outward_normals_fixture', 'type': 'box', 'size': [1, 2, 3], 'semantic': 'glass'}, {}, Path('.'))
        if not all(poly.normal.dot(poly.center) > 0 for poly in box.data.polygons):
            raise AssertionError('Box primitive has inward or degenerate face normals')
    except Exception as error:
        failures.append({'check': 'outward_box_normals', 'error': str(error)})
    camera_results = []
    for width, height, fx, fy, cx, cy in [(640, 480, 500, 500, 320, 240),
                                          (1200, 800, 1000, 1020, 576, 368),
                                          (960, 960, 700, 650, 530, 420),
                                          (480, 960, 400, 430, 190, 490)]:
        try:
            scene = bpy.context.scene
            scene.render.resolution_x, scene.render.resolution_y = width, height
            scene.render.resolution_percentage = 100
            rotation = Quaternion(Vector((1, 2, 3)).normalized(), .61)
            camera = explicit_camera({'pos': [1.3, -2.1, 1.7], 'quat_wxyz': list(rotation), 'resolution': [width, height],
                                      'intrinsics': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]})
            bpy.context.view_layer.update()
            worst = 0.0
            for point in [Vector((0, 0, -2)), Vector((.3, .4, -3)), Vector((-.5, .2, -1.7))]:
                projected = world_to_camera_view(scene, camera, camera.matrix_world @ point)
                pixel = (projected.x * width, (1 - projected.y) * height)
                expected = (fx * point.x / -point.z + cx, fy * -point.y / -point.z + cy)
                worst = max(worst, max(abs(a - b) for a, b in zip(pixel, expected)))
            if worst > .001:
                raise AssertionError(f'Calibrated projection error {worst:g} pixels')
            camera_results.append({'resolution': [width, height], 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                                   'max_error_pixels': worst})
        except Exception as error:
            failures.append({'check': 'camera_projection', 'resolution': [width, height], 'error': str(error)})
    report = {'ok': not failures, 'blender_version': bpy.app.version_string, 'case_count': len(cases),
              'source_geoms_checked': total_geoms, 'poses': results, 'cameras': camera_results, 'failures': failures,
              'scope': 'Native transforms, authored OBJ vertices, primitive normals, and calibrated pinhole projection. No ray rendering or visual realism certification.'}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('poses', 'cameras')}, indent=2))
    if failures:
        raise RuntimeError(f'{len(failures)} Blender integration failures; see {result_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    parser.add_argument('--doors', default=DEFAULT_DOORS)
    parser.add_argument('--out', type=Path, default=Path('out/appearance_validation.json'))
    parser.add_argument('--blender')
    parser.add_argument('--inside-blender', type=Path, help=argparse.SUPPRESS)
    arguments = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    if args.inside_blender:
        return _inside_blender(args.inside_blender, args.out)
    from doorbench.appearance.pipeline import find_blender
    cases = _prepare_cases(args.assets.resolve(), args.doors.split(','))
    with tempfile.TemporaryDirectory(prefix='doorbench-blender-validation-') as temp:
        config = Path(temp) / 'cases.json'
        config.write_text(json.dumps({'cases': cases}, allow_nan=False))
        command = [find_blender(args.blender), '--background', '--factory-startup', '--python-exit-code', '1',
                   '--python', str(Path(__file__).resolve()), '--', '--inside-blender', str(config), '--out', str(args.out.resolve())]
        completed = subprocess.run(command)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()

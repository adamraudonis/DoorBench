"""Optional headless Blender checks for appearance camera/stage regressions."""
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


def test_architectural_environment_preserves_apertures_and_frames_the_door(tmp_path):
    blender = shutil.which('blender')
    if not blender:
        bundled = Path('/Applications/Blender.app/Contents/MacOS/Blender')
        blender = str(bundled) if bundled.is_file() else None
    if not blender:
        pytest.skip('Blender is an optional appearance dependency')
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'check_environment.py'
    script.write_text(textwrap.dedent(f'''
        import copy
        import sys
        sys.path.insert(0, {str(root)!r})
        import bpy
        from mathutils import Vector
        from bpy_extras.object_utils import world_to_camera_view
        from doorbench.appearance.blender_environment import configure_scene, build_environment, frame_camera, _corners
        recipe = {{'lighting':'daylight','render_device':'CPU'}}
        for family in ('sliding_single', 'hatch_floor', 'hatch_ceiling'):
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            bpy.context.scene.render.pixel_aspect_y = 1.6
            scene = configure_scene(recipe, 'preview', 640, 360, 42)
            assert scene.render.pixel_aspect_x == scene.render.pixel_aspect_y == 1
            assert scene.render.engine == 'CYCLES' and scene.cycles.samples == 16
            assert scene.cycles.device == 'CPU' and scene.cycles.use_denoising
            assert scene.view_settings.view_transform == 'AgX'
            assert not scene.render.film_transparent
            assert next(n for n in scene.world.node_tree.nodes if n.type == 'TEX_SKY').sun_disc is False
            bpy.ops.mesh.primitive_cube_add(size=1)
            leaf = bpy.context.object
            leaf.name = 'test_leaf'
            leaf.dimensions = (3.2,.08,2.4) if family == 'sliding_single' else (1.2,1.1,.08)
            leaf.location.z = 1.2 if family == 'sliding_single' else 2.7 if family == 'hatch_ceiling' else .02
            bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
            model = {{'bodies':[{{'geoms':[{{'name':'test_leaf','semantic':'leaf'}}]}}]}}
            original = copy.deepcopy(model)
            objects = {{'test_leaf':leaf}}
            spec = {{'family':family}}
            environment = build_environment(model,spec,recipe,objects)
            assert model == original
            assert all(o.get('doorbench_visual_only') and o.get('doorbench_physics_export') is False for o in environment.values())
            if family.startswith('hatch_'):
                assert 'DB ground continuation' not in environment
                assert 'DB room ceiling' not in environment
            for width,height,view in ((640,360,'front'),(360,640,'reverse')):
                camera = frame_camera(objects,spec,view,width,height)
                points = [world_to_camera_view(scene,camera,p) for p in _corners(objects)]
                assert all(p.z>0 and .064<=p.x<=.936 and .064<=p.y<=.936 for p in points)
                if family == 'sliding_single':
                    forward = camera.matrix_world.to_quaternion() @ Vector((0,0,-1))
                    assert abs(forward.z)<1e-5
                    assert camera.location.y < 0 if view == 'front' else camera.location.y > 0
            bpy.context.view_layer.update()
            x0,x1,y0,y1,z1 = scene['doorbench_environment_bounds']
            for lamp in (o for o in scene.objects if o.type=='LIGHT'):
                corners = [lamp.matrix_world @ Vector((x*lamp.data.size/2,y*lamp.data.size_y/2,0)) for x in (-1,1) for y in (-1,1)]
                assert all(x0+.149<=p.x<=x1-.149 and y0+.149<=p.y<=y1-.149 and .149<=p.z<=z1-.149 for p in corners), (family,lamp.name)
        print('ENVIRONMENT_REGRESSIONS_PASS')
    '''))
    process = subprocess.run([blender, '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(script)],
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'ENVIRONMENT_REGRESSIONS_PASS' in process.stdout


def test_changing_door_pose_keeps_the_reference_room_lights_and_camera_fixed(tmp_path):
    blender = shutil.which('blender')
    if not blender:
        bundled = Path('/Applications/Blender.app/Contents/MacOS/Blender')
        blender = str(bundled) if bundled.is_file() else None
    if not blender:
        pytest.skip('Blender is an optional appearance dependency')
    pytest.importorskip('mujoco')
    import json
    from doorbench.appearance.pipeline import prepare_job
    from doorbench.appearance.state import export_state
    from doorbench.build import export_door
    from doorbench.spec import generate_all
    root = Path(__file__).resolve().parents[1]
    spec = next(s for s in generate_all() if s['id'] == 'db0002_swing_single')
    assets = tmp_path / 'assets'
    export_door(spec, str(assets / 'doors'), str(assets / 'hardware'), formats=('mjcf', 'json'))
    door = assets / 'doors' / spec['id']
    model = json.loads((door / 'model.json').read_text())
    moved = export_state(door, tmp_path / 'moved.json', qpos={model['meta']['primary_joint']: 1.0})
    jobs = [prepare_job(assets, spec['id'], tmp_path / 'initial', validate_only=True, device='CPU'),
            prepare_job(assets, spec['id'], tmp_path / 'moved', state=moved, validate_only=True, device='CPU')]
    assert jobs[0]['reference_state'] == jobs[1]['reference_state']
    config = tmp_path / 'jobs.json'
    config.write_text(json.dumps(jobs))
    script = tmp_path / 'check_reference.py'
    script.write_text(textwrap.dedent(f'''
        import json
        import sys
        from pathlib import Path
        sys.path.insert(0, {str(root)!r})
        import bpy
        from doorbench.appearance.blender_worker import run_job, transform
        jobs = json.loads(Path({str(config)!r}).read_text())
        scenes = []
        for job in jobs:
            run_job(job)
            scene = bpy.context.scene
            camera = scene.camera
            fixed = {{}}
            bodies = {{}}
            for obj in scene.objects:
                matrix = tuple(tuple(row) for row in obj.matrix_world)
                if obj.get('doorbench_visual_only'):
                    fixed[obj.name] = matrix
                    if obj.type == 'LIGHT':
                        fixed[obj.name + ' emission'] = (obj.data.energy, obj.data.size, obj.data.size_y, *obj.data.color)
                if obj.type == 'EMPTY' and obj.get('doorbench_body'):
                    name = obj['doorbench_body']
                    pose = job['state']['body_world'][name]
                    expected = transform(pose['pos'],pose['quat_wxyz'])
                    assert max(abs(obj.matrix_world[i][j]-expected[i][j]) for i in range(4) for j in range(4)) < 1e-6
                    bodies[name] = matrix
            scenes.append({{'fixed':fixed,'bodies':bodies,'camera':tuple(tuple(row) for row in camera.matrix_world),
                           'lens':(camera.data.lens,camera.data.shift_x,camera.data.shift_y)}})
        assert scenes[0]['fixed'] == scenes[1]['fixed'], 'Door motion changed room geometry or lighting'
        assert scenes[0]['camera'] == scenes[1]['camera'] and scenes[0]['lens'] == scenes[1]['lens'], 'Door motion refitted the camera'
        assert scenes[0]['bodies']['leaf'] != scenes[1]['bodies']['leaf'], 'The reference pose incorrectly replaced the actual articulated state'
        print('FIXED_REFERENCE_REGRESSION_PASS')
    '''))
    process = subprocess.run([blender, '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(script)],
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'FIXED_REFERENCE_REGRESSION_PASS' in process.stdout

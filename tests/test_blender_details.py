"""Render-only wreath foliage preserves the source proxy and rigid attachment."""
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


def test_wreath_details_are_deterministic_contained_and_follow_the_body(tmp_path):
    blender = shutil.which('blender')
    if not blender:
        path = Path('/Applications/Blender.app/Contents/MacOS/Blender')
        blender = str(path) if path.is_file() else None
    if not blender:
        pytest.skip('Blender is an optional appearance dependency')
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'check_wreath.py'
    script.write_text(textwrap.dedent(f'''
        import math
        import sys
        sys.path.insert(0, {str(root)!r})
        import bpy
        from mathutils import Vector
        from doorbench.appearance.blender_details import build_details
        signatures = []
        for _ in range(2):
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            parent = bpy.data.objects.new('leaf',None)
            bpy.context.scene.collection.objects.link(parent)
            bpy.ops.mesh.primitive_torus_add(major_radius=.2,minor_radius=.045,major_segments=32,minor_segments=10)
            proxy = bpy.context.object
            proxy.name = 'wreath'
            proxy.parent = parent
            proxy.location = (.3,-.02,1.5)
            proxy.rotation_euler.x = math.pi/2
            proxy['doorbench_geom'] = 'wreath'
            original = [tuple(v.co) for v in proxy.data.vertices]
            matrix = proxy.matrix_basis.copy()
            details = build_details({{'wreath':proxy}},{{'id':'fixture'}},seed=17)
            assert proxy.hide_render and proxy['doorbench_render_replaced_by_detail']
            assert [tuple(v.co) for v in proxy.data.vertices] == original
            assert proxy.matrix_basis == matrix
            assert len(details) == 2
            for obj in details.values():
                assert obj.parent == parent
                assert max(abs(obj.matrix_basis[i][j]-matrix[i][j]) for i in range(4) for j in range(4)) < 1e-6
                assert obj['doorbench_visual_only'] and not obj['doorbench_physics_export']
                assert obj['doorbench_source_geom'] == 'wreath'
                r0,r1,z0,z1 = obj['doorbench_reference_envelope']
                assert all(r0-1e-6<=math.hypot(v.co.x,v.co.y)<=r1+1e-6 and z0-1e-6<=v.co.z<=z1+1e-6 for v in obj.data.vertices)
            leaves = next(o for o in details.values() if o.name.endswith('leaves'))
            assert len(leaves.data.vertices) >= 7000
            signatures.append([tuple(v.co) for v in leaves.data.vertices])
            parent.rotation_euler.z = .9
            parent.location = (1,2,.3)
            bpy.context.view_layer.update()
            assert all(max(abs(o.matrix_world[i][j]-proxy.matrix_world[i][j]) for i in range(4) for j in range(4)) < 1e-6 for o in details.values())
            assert build_details({{'unrelated':proxy}},{{'id':'fixture'}}) == {{}}
        assert signatures[0] == signatures[1]
        print('WREATH_DETAILS_PASS')
    '''))
    process = subprocess.run([blender,'--background','--factory-startup','--python-exit-code','1','--python',str(script)],
                             capture_output=True,text=True,timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'WREATH_DETAILS_PASS' in process.stdout

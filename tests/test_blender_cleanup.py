"""Repeated worker scenes must release material/object reference cycles."""
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


def test_reset_breaks_object_material_cycles_and_preserves_image_cache(tmp_path):
    blender = shutil.which('blender')
    if not blender:
        path = Path('/Applications/Blender.app/Contents/MacOS/Blender')
        blender = str(path) if path.is_file() else None
    if not blender:
        pytest.skip('Blender is an optional appearance dependency')
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'check_cleanup.py'
    texture = tmp_path / 'cached_texture.png'
    script.write_text(textwrap.dedent(f'''
        import sys
        sys.path.insert(0, {str(root)!r})
        import bpy
        from doorbench.appearance.blender_worker import reset_scene
        reset_scene()
        generated = bpy.data.images.new('cache_fixture_source',width=4,height=4)
        generated.pixels = [.3,.4,.5,1] * 16
        generated.filepath_raw = {str(texture)!r}
        generated.file_format = 'PNG'
        generated.save()
        bpy.data.images.remove(generated)
        cached = bpy.data.images.load({str(texture)!r},check_existing=True)
        cached_pointer = cached.as_pointer()
        image_count = len(bpy.data.images)
        groups = ('objects','meshes','curves','materials','cameras','lights','worlds')
        for iteration in range(20):
            parent = bpy.data.objects.new('rigid_body',None)
            bpy.context.scene.collection.objects.link(parent)
            bpy.ops.mesh.primitive_cube_add()
            obj = bpy.context.object
            obj.parent = parent
            material = bpy.data.materials.new('part_local_wood')
            material.use_nodes = True
            # This exact cycle survived operator deletion plus users==0 cleanup:
            # object -> mesh -> material -> Texture Coordinate.object -> object.
            coordinate = material.node_tree.nodes.new('ShaderNodeTexCoord')
            coordinate.object = obj
            body_coordinate = material.node_tree.nodes.new('ShaderNodeTexCoord')
            body_coordinate.object = parent
            image_node = material.node_tree.nodes.new('ShaderNodeTexImage')
            image_node.image = cached
            obj.data.materials.append(material)
            bpy.context.scene.world = bpy.data.worlds.new('worker_world')
            assert len(bpy.data.objects) == 2 and len(bpy.data.meshes) == 1 and len(bpy.data.materials) == 1
            reset_scene()
            counts = {{name:len(getattr(bpy.data,name)) for name in groups}}
            assert all(count == 0 for count in counts.values()), (iteration,counts)
            assert bpy.data.images.load({str(texture)!r},check_existing=True).as_pointer() == cached_pointer
            assert len(bpy.data.images) == image_count and cached.users == 0
        print('CLEANUP_CYCLE_REGRESSION_PASS 20 cycles; 0 orphan scene datablocks; FILE image cache retained')
    '''))
    process = subprocess.run([blender,'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','--python',str(script)],
                             capture_output=True,text=True,timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'CLEANUP_CYCLE_REGRESSION_PASS' in process.stdout

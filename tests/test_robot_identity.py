import json
from pathlib import Path

import mujoco
import pytest

from doorbench.dexterous.robot_identity import compiled_robot_identity, robot_file_identity, verify_robot_identity

XML = '''<mujoco model="fixture"><option timestep=".002" gravity="0 0 -9.81"/>
<worldbody><body name="hand"><joint name="j1" range="0 1"/><geom type="box" size=".1 .2 .3" mass="1"/>
<body name="tip" pos="0 0 .3"><joint name="j2" range="0 1"/><geom type="sphere" size=".03" mass=".1"/></body></body></worldbody>
<tendon><fixed name="passive" limited="true" range="-1 0"><joint joint="j1" coef="1"/><joint joint="j2" coef="-1"/></fixed></tendon>
<actuator><motor name="drive" joint="j1" forcelimited="true" forcerange="-1 1"/></actuator></mujoco>'''


def test_relocated_xml_and_mesh_files_share_identity(tmp_path):
    mesh = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
    paths = []
    for folder in ('a', 'longer-other-cluster-directory'):
        directory=tmp_path/folder;directory.mkdir();(directory/'same.obj').write_text(mesh)
        path=directory/'robot.xml'
        path.write_text(XML.replace('<worldbody>',f'<asset><mesh name="mesh" file="{directory}/same.obj"/></asset><worldbody>').replace('type="sphere" size=".03"','type="mesh" mesh="mesh"'))
        paths.append(path)
    first=robot_file_identity(paths[0]);second=verify_robot_identity(paths[1],first)
    assert first['sha256']==second['sha256']


@pytest.mark.parametrize('replacement', [('mass="1"','mass="1.01"'),('size=".1 .2 .3"','size=".1 .2 .31"'),('forcerange="-1 1"','forcerange="-2 2"'),('range="-1 0"','range="-1 .1"'),('timestep=".002"','timestep=".004"'),('name="j2"','name="renamed"')])
def test_changed_plant_or_binding_is_rejected(tmp_path,replacement):
    path=tmp_path/'robot.xml';path.write_text(XML);expected=robot_file_identity(path)
    changed=XML.replace(*replacement)
    if replacement[0]=='name="j2"':changed=changed.replace('joint="j2"','joint="renamed"')
    path.write_text(changed)
    with pytest.raises(ValueError,match='identity differs'):verify_robot_identity(path,expected)


def test_changed_compiled_geometry_is_rejected(tmp_path):
    path=tmp_path/'robot.xml';path.write_text(XML);expected=robot_file_identity(path)
    model=mujoco.MjModel.from_xml_path(str(path));model.geom_size[0,0]+=.001
    assert compiled_robot_identity(model)['sha256']!=expected['sha256']


def test_model_version_cannot_be_silently_reinterpreted(tmp_path):
    path=tmp_path/'robot.xml';path.write_text(XML);expected=robot_file_identity(path)
    expected['mujoco_version']='other-version'
    with pytest.raises(ValueError,match='pinned MuJoCo'):verify_robot_identity(path,expected)


def test_touch_grid_source_config_is_included_and_opaque_entry_is_rejected(tmp_path):
    source=XML.replace('<worldbody>','<extension><plugin plugin="mujoco.sensor.touch_grid"/></extension><worldbody>').replace('<body name="hand">','<body name="hand"><site name="touch" size=".1"/>').replace('</mujoco>','<sensor><plugin plugin="mujoco.sensor.touch_grid" name="sensor" objtype="site" objname="touch"><config key="nchannel" value="3"/><config key="size" value="2 2"/><config key="fov" value="180 90"/><config key="gamma" value="0"/></plugin></sensor></mujoco>')
    path=tmp_path/'robot.xml';path.write_text(source);expected=robot_file_identity(path)
    model=mujoco.MjModel.from_xml_path(str(path))
    with pytest.raises(ValueError,match='Opaque'):compiled_robot_identity(model)
    path.write_text(source.replace('value="2 2"','value="3 2"'))
    with pytest.raises(ValueError,match='identity differs'):verify_robot_identity(path,expected)


def test_mesh_asset_change_fails_with_identical_xml_bytes(tmp_path):
    mesh=tmp_path/'mesh.obj'
    content='v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
    mesh.write_text(content)
    path=tmp_path/'robot.xml'
    path.write_text(XML.replace('<worldbody>',f'<asset><mesh name="mesh" file="{mesh}"/></asset><worldbody>').replace('type="sphere" size=".03"','type="mesh" mesh="mesh"'))
    xml=path.read_bytes();expected=robot_file_identity(path)
    mesh.write_text(content.replace('v 1 0 0','v 1.1 0 0'))
    assert path.read_bytes()==xml
    with pytest.raises(ValueError,match='identity differs'):verify_robot_identity(path,expected)

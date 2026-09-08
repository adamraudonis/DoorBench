"""Readiness must reject plausible-looking but incomplete assets/imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('asset_check',ROOT/'scripts/isaac/check_assets.py')
asset_check=importlib.util.module_from_spec(spec);spec.loader.exec_module(asset_check)


def test_requested_asset_cannot_be_satisfied_by_another_door(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(n_doors=1,doors=[dict(id='a')])) )
    with pytest.raises(ValueError,match='missing requested'):
        asset_check.check_assets(tmp_path,['b'])


def test_asset_files_without_signed_off_qa_are_not_ready(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(n_doors=1,doors=[dict(id='a')])) )
    door=tmp_path/'doors/a';door.mkdir(parents=True)
    for name in ('door.usda','door.xml','spec.json'):(door/name).write_text('')
    (door/'qa.json').write_text('{"signed_off": false}')
    with pytest.raises(ValueError,match='not passed QA'):asset_check.check_assets(tmp_path)
    (door/'qa.json').write_text('{"signed_off": true}')
    assert asset_check.check_assets(tmp_path)==['a']


def test_import_preparation_preserves_free_root_and_nested_joint_axes(tmp_path):
    mj=pytest.importorskip('mujoco')
    source=tmp_path/'robot.xml'
    source.write_text('''<mujoco><compiler angle="radian"/><default>
      <default class="outer"><joint axis="0 0 1" damping="2"/><default class="inner"><joint range="-.3 .9"/></default></default>
      </default><worldbody><body name="root"><joint name="free" type="free"/><geom type="box" size=".1 .1 .1" mass="1"/>
      <body name="one" pos="0 0 .3"><joint name="a" class="inner"/><geom type="box" size=".04 .04 .1" mass=".3"/>
      <body name="two" pos="0 0 .2"><joint name="b" class="inner" axis="1 0 0"/><geom type="box" size=".03 .03 .1" mass=".2"/>
      </body></body></body></worldbody><tendon><fixed name="motor"><joint joint="a" coef=".5"/><joint joint="b" coef="1"/></fixed></tendon>
      <actuator><position name="coupled" tendon="motor" kp="3" forcerange="-2 2" ctrlrange="-1 1"/></actuator></mujoco>''')
    output=tmp_path/'import.xml'
    subprocess.run([sys.executable,str(ROOT/'scripts/dexterous/prepare_isaac_import.py'),'--robot',str(source),'--output',str(output)],check=True)
    r=ET.parse(output);joints={j.get('name'):j for j in r.findall('./worldbody//joint')}
    assert joints['free'].get('type')=='free'
    assert joints['a'].get('type')==joints['b'].get('type')=='hinge'
    assert joints['a'].get('axis')=='0 0 1' and joints['b'].get('axis')=='1 0 0'
    assert joints['a'].get('damping')=='2.0'
    # Explicit motor transmission preserves virtual work and force saturation.
    contract=json.loads(output.with_suffix('.motors.json').read_text())
    assert contract['joint_names']==['a','b']
    motor=contract['actuators'][0];assert motor['terms']=={'a':.5,'b':1.}
    velocity=np.array([.7,-.1]);matrix=np.array([.5,1.]);force=1.5
    assert np.dot(matrix*force,velocity)==pytest.approx(force*np.dot(matrix,velocity))
    native=mj.MjModel.from_xml_path(str(source));converted=mj.MjModel.from_xml_path(str(output))
    np.testing.assert_array_equal(native.jnt_type,converted.jnt_type)
    np.testing.assert_allclose(native.jnt_axis,converted.jnt_axis)
    np.testing.assert_allclose(native.body_mass,converted.body_mass)

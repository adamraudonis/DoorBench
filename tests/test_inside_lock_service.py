"""Physical inside controls persist when the robot has no outside release."""
import json
import mujoco
import numpy as np
import pytest
from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.qa import _prepare_closer_service
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.runner import torque_limits
from doorbench.benchmark.interactions import ContactSites


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    root=tmp_path_factory.mktemp('inside-service');rows=[]
    for spec in generate_all():
        if spec['index'] not in (72,83,102,312,551,919):continue
        summary=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('json','mjcf'))
        path=root/'doors'/spec['id'];ir=json.loads((path/'model.json').read_text())
        rows.append((spec,path,ir,summary['files']['mjcf']))
    assert len(rows)==6
    return rows


@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_native_inside_service_preserves_outside_restrictions_and_model(fixtures,tier):
    for spec,path,ir,files in fixtures:
        m=mujoco.MjModel.from_xml_path(files[tier]);d=mujoco.MjData(m)
        initial=m.qpos0.copy();limits=m.jnt_range.copy()
        result=_prepare_closer_service(m,d,spec,ir['meta'],str(path))
        assert result['ok'],(spec['id'],tier,result)
        assert not np.any(d.warning.number)
        np.testing.assert_array_equal(m.qpos0,initial)
        np.testing.assert_array_equal(m.jnt_range,limits)
        e=DoorEnv(str(path),tier=tier);e.reset(randomize=False)
        try:
            allowed=torque_limits(e,str(path));contacts=ContactSites(e)
            for body in ir['bodies']:
                j=body.get('joint') or {}
                if j.get('role')!='lock' or j.get('robot_interactive',True):continue
                if not any(x in j['name'] for x in ('thumbturn_hinge','aux_bolt_slide')):continue
                assert m.jnt_range[m.joint(j['name']).id,1]>.015
                assert allowed.get(j['name'],0.)==0.
                assert contacts.select(j['name']) is None
        finally:e.close()


def test_fixed_inside_input_cannot_be_treated_as_successful_service(fixtures):
    spec,path,ir,files=next(r for r in fixtures if r[0]['index']==102)
    m=mujoco.MjModel.from_xml_path(files['full']);d=mujoco.MjData(m)
    joint=m.joint('leaf_deadbolt_thumbturn_hinge').id;m.jnt_range[joint,1]=.001
    result=_prepare_closer_service(m,d,spec,ir['meta'],str(path))
    assert not result['ok']
    assert result['released_joint_positions']['leaf_deadbolt_slide']<.001

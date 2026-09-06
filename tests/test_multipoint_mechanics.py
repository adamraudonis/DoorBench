"""Lift lever compatibility and actual contact transmission counterexamples."""
import json
import mujoco
import numpy as np
import pytest
from doorbench.spec import generate_all
from doorbench.build import export_door
from doorbench.multipoint_qa import run_multipoint_qa


@pytest.fixture(scope='module')
def locks(tmp_path_factory):
    root=tmp_path_factory.mktemp('multipoint-locks');rows=[]
    for s in generate_all():
        if s['lock']['model']!='multipoint':continue
        ex=export_door(s,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        meta=json.load(open(root/'doors'/s['id']/'model.json'))['meta']
        rows.append((s,ex['files']['mjcf'],meta))
    assert len(rows)==7
    return rows


@pytest.mark.parametrize('tier',['full','simple','minimal'])
def test_all7_lift_key_depress_cycles_are_contact_driven(locks,tier):
    for spec,paths,meta in locks:
        assert spec['operator']['model']=='lever_euro_backplate'
        m=mujoco.MjModel.from_xml_path(paths[tier]);before=m.jnt_range.copy()
        report=run_multipoint_qa(m,meta)
        assert report['ok'],(spec['id'],tier,report['failures'])
        np.testing.assert_array_equal(m.jnt_range,before)
        for row in meta['multipoint_locks']:
            followers={m.joint(row['drivebar_joint']).id,*[m.joint(a['joint']).id for a in row['auxiliary']]}
            assert not any(int(m.eq_type[e])==int(mujoco.mjtEq.mjEQ_JOINT) and
                           (int(m.eq_obj1id[e]) in followers or int(m.eq_obj2id[e]) in followers) for e in range(m.neq))


@pytest.mark.parametrize('missing',['lever_pin','key_interlock'])
def test_missing_actual_load_path_fails_native_cycle(locks,missing):
    _,paths,meta=locks[0];m=mujoco.MjModel.from_xml_path(paths['full']);r=meta['multipoint_locks'][0]
    names=[r['lever_pin_geom']] if missing=='lever_pin' else r['key_window_geoms']
    for name in names:
        g=m.geom(name).id;m.geom_contype[g]=0;m.geom_conaffinity[g]=0
    result=run_multipoint_qa(m,meta)
    assert not result['ok']
    if missing=='key_interlock':assert any('arrest' in f['reason'] for f in result['failures'])

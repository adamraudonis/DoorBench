"""A thin panel needs supported knob clearance around its edge cartridge."""
from copy import deepcopy
from functools import lru_cache
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from doorbench.build import build_model
from doorbench.export.mjcf import build_mjcf
from doorbench.spec import generate_all
from doorbench.lock_stock_qa import run_lock_stock_qa


@lru_cache(None)
def source(index):
    return build_model(next(s for s in generate_all() if s['index']==index))


def native(ir,tier):
    assets={g.mesh_name+'.obj':g.mesh.export(file_type='obj',include_normals=False,include_texture=False).encode()
            for b in ir.bodies for g in b.geoms if g.type=='mesh'}
    return mujoco.MjModel.from_xml_string(ET.tostring(build_mjcf(ir,tier=tier,mesh_dir_rel=''),encoding='unicode'),assets)


@pytest.mark.parametrize('index',(232,425))
@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_bored_spacer_and_spindle_keep_knob_sweep_clear(index,tier):
    ir=source(index);m=native(ir,tier);d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
    report=run_lock_stock_qa(m,ir.meta,tier=tier)
    assert report['ok'],report
    rows=[v for r in ir.meta['lock_stock'] for v in r.get('operator_standoffs',[])]
    assert len(rows)==2
    for r in rows:
        assert .008<r['standoff_m']<.009
        shaft=m.geom(r['shaft_geom']).id
        gaps=[mujoco.mj_geomDistance(m,d,shaft,m.geom(n).id,.002,None) for n in r['spacer_geoms']]
        assert .0005<min(gaps)<.001
    assert sum(x['part']=='operator_standoff' for x in report['measurements'])==2


@pytest.mark.parametrize('defect',('old_knob_position','filled_spacer'))
def test_direct_stock_gate_rejects_old_overlap_and_filled_bore(defect):
    ir=source(232);m=native(ir,'full')
    mount=next(v for r in ir.meta['lock_stock'] for v in r.get('operator_standoffs',[]))
    if defect=='old_knob_position':
        for name in mount['moving_geoms']:
            m.geom_pos[m.geom(name).id,1]-=mount['face']*mount['standoff_m']
    else:
        d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
        fixed=m.geom(mount['spacer_geoms'][0]).id;shaft=m.geom(mount['shaft_geom']).id
        body=m.geom_bodyid[fixed]
        m.geom_pos[fixed]=d.xmat[body].reshape(3,3).T@(d.geom_xpos[shaft]-d.xpos[body])
        m.geom_size[fixed]=[.009,.009,.009]
    result=run_lock_stock_qa(m,ir.meta)
    assert not result['ok']
    assert any(f.get('part')=='operator_standoff' for f in result['failures'])

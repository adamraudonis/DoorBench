"""Physical Dutch keeper load transfer, service access and causal failures."""
import json

import mujoco
import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.paired_mechanics_qa import run_dutch_join_qa
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.interactions import ContactSites


IDS={95,118,204,333,391,423,460,626,700,847,906,974}


@pytest.fixture(scope='module')
def doors(tmp_path_factory):
    root=tmp_path_factory.mktemp('dutch-joining');rows=[]
    for spec in generate_all():
        if spec['family']!='dutch':continue
        summary=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        path=root/'doors'/spec['id'];source=json.loads((path/'model.json').read_text())
        rows.append((spec,path,source,summary['files']['mjcf']))
    assert {s['index'] for s,_,_,_ in rows}==IDS
    return rows


@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_every_dutch_keeper_carries_and_releases_both_native_load_directions(doors,tier):
    for spec,path,source,files in doors:
        m=mujoco.MjModel.from_xml_path(files[tier]);meta=source['meta'];r=meta['dutch_joining_bolt']
        assert r['face']==(1 if spec['robot']['robot_outside'] else -1)
        assert r['engaged_initial']==spec['kinematics']['joining_bolt_engaged']
        # The original upper catch must exist even in the minimal tier.
        assert m.geom('upper_catch_capsule').id>=0
        result=run_dutch_join_qa(m,meta)
        (path/('joining-proof-'+tier+'.json')).write_text(json.dumps(result,indent=2))
        assert result['ok'],(spec['id'],tier,result)
        assert sum(p['phase']=='joined_lower_open' for p in result['phases'])==2
        assert all(p['keeper_contacts'] for p in result['phases'] if p['phase'] in ('joined_upper_load','joined_lower_open'))
        assert result['max_grip_force_N']<=20. and result['max_leaf_fixture_torque_Nm']<=40.
        env=DoorEnv(str(path),tier=tier);env.reset(randomize=False)
        try:assert (ContactSites(env).select(r['joint']) is not None)==r['accessible_from_robot']
        finally:env.close()


def test_slab_material_controls_dutch_mass_distribution_in_every_tier(doors):
    for spec,_,_,files in doors:
        for tier in ('full','simple','minimal'):
            m=mujoco.MjModel.from_xml_path(files[tier]);d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
            for name in ('leaf_lower','leaf_upper'):
                b=m.body(name).id;j=m.joint(name+'_hinge').id;axis=m.jnt_axis[j]
                R=d.ximat[b].reshape(3,3);I=R@np.diag(m.body_inertia[b])@R.T
                zrot=float(axis@I@axis);expected=float(m.body_mass[b])*spec['leaf']['width']**2/12
                # Broad material-derived bound: hardware and prepared cavities
                # may move the COM, but cannot replace a slab with a hinge mass.
                assert .65*expected<zrot<1.4*expected,(spec['id'],tier,name,zrot,expected)


@pytest.mark.parametrize('defect',('guide_filled','keeper_removed','restricted_leaf','off_surface'))
def test_bolt_gate_rejects_missing_load_path_or_fabricated_operation(doors,defect):
    spec,_,source,files=next(row for row in doors if row[0]['index']==118)
    meta=source['meta'];r=meta['dutch_joining_bolt'];m=mujoco.MjModel.from_xml_path(files['full'])
    if defect=='keeper_removed':
        for name in r['keeper_geoms']:m.geom_pos[m.geom(name).id,0]+=5.
    elif defect=='guide_filled':
        g=m.geom(r['guide_geoms'][0]).id;body=m.geom_bodyid[g]
        d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
        rod=m.geom(r['rod_geom']).id
        m.geom_pos[g]=d.xmat[body].reshape(3,3).T@(d.geom_xpos[rod]-d.xpos[body]);m.geom_size[g]=[.02,.02,.02]
    elif defect=='restricted_leaf':m.jnt_range[m.joint(r['lower_joint']).id,1]=.001
    else:m.site_pos[m.site(r['site']).id,0]+=.03
    result=run_dutch_join_qa(m,meta)
    assert not result['ok'],(spec['id'],defect)
    if defect=='keeper_removed':assert 'transfer upper-leaf load' in result['failures'][0]

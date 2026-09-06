"""Contact-driven drop and manual reset, including missing-part controls."""
import copy
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.turnstile_drop import compile_turnstile_drop,apply_turnstile_drop
from doorbench.turnstile_drop_qa import run_turnstile_drop_qa,run_turnstile_drop_mount_qa

DROP_IDS=('db0202_turnstile_tripod','db0272_turnstile_tripod','db0344_turnstile_tripod',
          'db0393_turnstile_tripod','db0516_turnstile_tripod','db0946_turnstile_tripod')

@pytest.fixture(scope='module')
def drops(tmp_path_factory):
    root=tmp_path_factory.mktemp('native-drop');rows={}
    for spec in generate_all():
        if not spec['kinematics'].get('drop_arm'):continue
        ex=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        path=Path(ex['files']['mjcf']['full']);meta=json.loads(path.with_name('model.json').read_text())['meta']
        rows[spec['id']]=(spec,path,meta)
    assert tuple(rows)==DROP_IDS
    return rows


@pytest.mark.parametrize('door_id',DROP_IDS)
@pytest.mark.parametrize('filename',('door.xml','door_simple.xml','door_minimal.xml'))
def test_all6_drop_sources_all3_indices_two_cycles_every_tier(drops,door_id,filename):
    spec,path,meta=drops[door_id];path=path.with_name(filename)
    m=mujoco.MjModel.from_xml_path(str(path));result=run_turnstile_drop_qa(m,meta)
    result['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    result['metadata_sha256']=hashlib.sha256(path.with_name('model.json').read_bytes()).hexdigest()
    path.with_name(filename+'.drop-proof.json').write_text(json.dumps(result,indent=2))
    assert result['ok'],(door_id,filename,result['failures'])
    assert len(result['probes'])==27
    assert {r['index'] for r in result['probes']}=={0,1,2}


def test_missing_toe_or_release_nose_cannot_pass_physical_cycle(drops):
    _,path,meta=drops['db0272_turnstile_tripod'];row=meta['turnstile_drop_arm']
    for name,expected in [(row['arms'][0]['toe_geom'],'powered_catch_did_not_hold_load'),
                          (row['release_nose_geom'],'gravity_drop_not_at_physical_stop')]:
        m=mujoco.MjModel.from_xml_path(str(path));g=m.geom(name).id
        # A missing load toe includes the attached web because its upper edge
        # otherwise remains a real secondary contact against the catch.
        names=[name]+(['turnstile_drop_web_0'] if name==row['arms'][0]['toe_geom'] else ['turnstile_drop_release_stem'])
        for geom in names:
            g=m.geom(geom).id;m.geom_contype[g]=0;m.geom_conaffinity[g]=0
        result=run_turnstile_drop_qa(m,meta,indices=(0,),cycles=1)
        assert not result['ok']
        assert any(f.get('code')==expected for f in result['failures'])


def test_explicit_parent_child_stop_pair_is_required(drops):
    _,path,meta=drops['db0272_turnstile_tripod'];s=mujoco.MjSpec.from_file(str(path))
    # Delete only the three explicitly authored stop pairs. Global parent
    # filtering remains on, so a nearby drawing cannot be mistaken for a stop.
    for pair in list(s.pairs):s.delete(pair)
    result=run_turnstile_drop_qa(s.compile(),meta,indices=(0,),cycles=1)
    assert not result['ok']
    assert any(f.get('code')=='missing_drop_load_path' for f in result['failures'])


def test_supports_are_connected_and_missing_bracket_fails(drops):
    for _,path,meta in drops.values():
        m=mujoco.MjModel.from_xml_path(str(path));assert run_turnstile_drop_mount_qa(m,meta)['ok']
    m=mujoco.MjModel.from_xml_path(str(path));g=m.geom('turnstile_drop_cross_beam').id;m.geom_pos[g,2]+=.030
    r=run_turnstile_drop_mount_qa(m,meta);assert not r['ok']
    assert any('turnstile_drop_cross_beam' in f['detached_parts'] for f in r['failures'])


def test_actual_tube_budget_surface_grip_and_all_tier_stop_binding(drops):
    from doorbench.physics import _unit_leaf_mass,leaf_mass
    for spec,path,meta in drops.values():
        row=meta['turnstile_drop_arm'];expected=3*math.pi*(.019**2-.0175**2)*(spec['leaf']['width']-.065)*7900+3.
        assert _unit_leaf_mass(spec)['slab_kg']==pytest.approx(expected)
        budget=leaf_mass(spec);assert budget['hardware_parts']['hinges_half']==0
        assert budget['per_body'][0]['catalogue_rotary_bearing_replaced_kg']==pytest.approx(2.5)
        for filename in ('door.xml','door_simple.xml','door_minimal.xml'):
            m=mujoco.MjModel.from_xml_path(str(path.with_name(filename)));d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
            assert m.body_mass[m.body(row['journal_body']).id]==pytest.approx(math.pi*.018**2*.270*7900,abs=1e-6)
            pairs={frozenset((m.geom(int(a)).name,m.geom(int(b)).name)) for a,b in zip(m.pair_geom1,m.pair_geom2)}
            for k,arm in enumerate(row['arms']):
                assert frozenset(arm['fold_stop_geoms']) in pairs
                assert m.body_mass[m.body(arm['arm_body']).id]==pytest.approx(arm['tube_mass_kg'],abs=1e-6)
                sid=m.site(arm['reset_site']).id;g=m.geom(f'arm_{k}_col').id
                local=d.geom_xmat[g].reshape(3,3).T@(d.site_xpos[sid]-d.geom_xpos[g])
                assert abs(np.linalg.norm(local[:2])-m.geom_size[g,0])<1e-6
                assert abs(local[2])<m.geom_size[g,1]
                aj=m.joint(arm['arm_joint']).id;normal=d.site_xmat[sid].reshape(3,3)[:,2]
                moment=np.dot(np.cross(d.site_xpos[sid]-d.xanchor[aj],-normal),d.xaxis[aj])
                assert moment<-.30  # inward force on underside raises the arm
                assert m.jnt_range[aj,1]>math.pi/2+.10


def test_power_force_preserves_state_and_strictly_binds_native_plunger(drops):
    _,path,meta=drops['db0272_turnstile_tripod'];m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    rules=compile_turnstile_drop(m,meta);saved={n:getattr(d,n).copy() for n in ('qpos','qvel','qacc','ctrl','qfrc_applied')}
    native={n:getattr(m,n).copy() for n in ('jnt_range','jnt_limited','dof_damping','geom_contype')}
    apply_turnstile_drop(m,d,rules,True)
    assert np.flatnonzero(d.qfrc_passive).tolist()==[rules[0].dof]
    assert d.qfrc_passive[rules[0].dof]==pytest.approx(-100.)
    before=d.qfrc_passive.copy();apply_turnstile_drop(m,d,rules,False);np.testing.assert_array_equal(before,d.qfrc_passive)
    for n,value in saved.items():np.testing.assert_array_equal(getattr(d,n),value)
    for n,value in native.items():np.testing.assert_array_equal(getattr(m,n),value)
    bad=copy.deepcopy(meta);bad['turnstile_drop_arm']['release_stroke_m']=.05
    with pytest.raises(ValueError):compile_turnstile_drop(m,bad)
    m.jnt_axis[m.joint(rules[0].name).id]=[1,0,0]
    with pytest.raises(ValueError):compile_turnstile_drop(m,meta)


def test_catalogue_matches_actual_credential_mechanism_without_inventing_access():
    from collections import Counter
    from doorbench import hardware as H
    from doorbench.physics import leaf_mass
    rows=[s for s in generate_all() if s['family'] in ('turnstile_tripod','turnstile_fullheight')]
    assert len(rows)==20
    assert sum(s['lock']['engaged'] for s in rows)==13
    assert sum(s['kinematics']['locked_until_credential'] for s in rows)==13
    assert Counter((s['task'],s['lock']['engaged']) for s in rows)=={
        ('push_through',True):12,('push_through',False):4,
        ('traverse_open',True):1,('traverse_open',False):3}
    for spec in rows:
        assert spec['lock']['model']=='turnstile_index_bolt'
        assert spec['lock']['robot_side_release'] is False
        assert spec['lock']['engaged']==spec['kinematics']['locked_until_credential']
        assert leaf_mass(spec)['hardware_parts']['lock']==0
    lock=H.LOCKS['turnstile_index_bolt']
    assert lock.kind=='credential_index_bolt' and lock.mass==0
    assert lock.deadbolt_throw==.022 and lock.inside_release=='card'

"""Material/count/native-inertia regressions, independent of reconciliation."""
import copy
import math

import numpy as np
import pytest

from doorbench.build import build_model, export_door
from doorbench import materials as M
from doorbench.mass_reconciliation import reconcile_moving_mass
from doorbench.physics import derive, roller_friction
from doorbench.spec import generate_all


@pytest.fixture(scope='module')
def specs():return {s['id']:s for s in generate_all()}


def test_all1000_reported_assembly_equals_all_moving_body_mass(specs):
    for spec in specs.values():
        physics=derive(spec);model=build_model(spec,physics)
        actual=sum(b.inertial('full')[0] for b in model.bodies if not b.static)
        assert actual==pytest.approx(physics['mass']['total_kg'],rel=1e-10),spec['id']
        assert len(physics['mass']['per_body'])==len([b for b in model.bodies if b.semantic=='leaf' and not b.static])
        assert all(r['slab_kg']+r['glass_kg']<=r['total_kg'] for r in physics['mass']['per_body'])


def test_db8_three_glass_leaves_each_retain_physical_mass_and_independent_friction(specs):
    spec=specs['db0008_sliding_bypass'];p=derive(spec);model=build_model(spec,p)
    glass=spec['leaf']['width']*spec['leaf']['height']*.006*2500
    assert p['mass']['total_kg']>=3*glass
    for row in p['mass']['per_body']:
        body=model.body(row['body'])
        assert body.mass_override>=glass
        expected=roller_friction(spec,row['total_kg'])['coulomb_force_N']
        assert body.joint.frictionloss==pytest.approx(expected)
        assert body.joint.frictionloss<roller_friction(spec,p['mass']['total_kg'])['coulomb_force_N']/2.9


def test_duplicate_leaf_geometry_does_not_change_panel_mass_allocation(specs):
    spec=specs['db0008_sliding_bypass'];p=derive(spec);model=build_model(spec,p)
    before={b.name:b.mass_override for b in model.bodies if b.semantic=='leaf'}
    duplicate=copy.deepcopy(model.body('leaf_0').geoms[0]);duplicate.name='audit_duplicate'
    duplicate.size=tuple(10*x for x in duplicate.size)
    model.body('leaf_0').geoms.append(duplicate)
    reconcile_moving_mass(model,p)
    assert before=={b.name:b.mass_override for b in model.bodies if b.semantic=='leaf'}


def test_geometry_backed_mechanism_preserves_density_and_adds_to_budget(specs):
    from doorbench.ir import Body, Joint, QUAT_ID
    from doorbench.geometry import common as C
    spec=specs['db0008_sliding_bypass'];p=derive(spec);model=build_model(spec,p)
    before=p['mass']['total_kg'];leaf_before=model.body('leaf_0').mass_override
    body=Body('audit_steel_arm','leaf_0',(0,0,0),QUAT_ID)
    body.joint=Joint('audit_arm_hinge','hinge')
    body.geoms.append(C.box('audit_solid_steel',(0,0,0),(.01,.02,.30),'default',7850))
    model.add_body(body);mass=8*.01*.02*.30*7850
    model.meta['mechanism_mass_bodies']=[body.name]
    reconcile_moving_mass(model,p)
    assert body.inertial('full')[0]==pytest.approx(mass)
    assert p['mass']['total_kg']==pytest.approx(before+mass)
    assert model.body('leaf_0').mass_override==pytest.approx(leaf_before)
    assert p['mass']['hardware_parts']['geometry_backed_mechanisms']==pytest.approx(mass)
    # Reserializing/reconciling must not double-add the same physical material.
    reconcile_moving_mass(model,p)
    assert p['mass']['total_kg']==pytest.approx(before+mass)
    assert body.inertial('full')[0]==pytest.approx(mass)


def test_geometry_backed_body_list_rejects_unknown_or_static_names(specs):
    model=build_model(specs['db0008_sliding_bypass']);p=derive(specs['db0008_sliding_bypass'])
    for names in (['missing_arm'],['leaf_0'],['world_env']):
        model.meta['mechanism_mass_bodies']=names
        with pytest.raises(ValueError,match='moving non-leaf'):
            reconcile_moving_mass(model,p)


@pytest.mark.parametrize('door', ['db0008_sliding_bypass','db0010_swing_double','db0023_sliding_single','db0095_dutch','db0037_strip_curtain','db0187_turnstile_fullheight'])
def test_exported_native_assembly_matches_serialized_budget(specs,tmp_path,door):
    mujoco=pytest.importorskip('mujoco')
    import json
    export_door(specs[door],str(tmp_path/'doors'),str(tmp_path/'hardware'),formats=('mjcf','json'))
    folder=tmp_path/'doors'/door
    source=json.loads((folder/'spec.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(folder/'door.xml'))
    # MJCF uses finite decimal serialization; bound accumulated roundoff well
    # below a gram, independently of the broader public QA mass tolerance.
    assert sum(model.body_mass[1:])==pytest.approx(source['physics']['mass']['total_kg'],rel=1e-7,abs=2e-6*model.nbody)


def test_all_framed_glass_has_real_ply_thickness_air_gap_and_hollow_frame(specs):
    checked=0
    for spec in specs.values():
        if spec['leaf']['slab'] not in M.FRAMED_GLASS_SLABS:continue
        model=build_model(spec);profiles=model.meta['framed_glass_constructions']
        assert profiles
        for p in profiles:
            body=model.body(p['body']);prefix=p['prefix']
            glass=[g for g in body.geoms if g.name.startswith(prefix+'_glass_ply_') and g.visual]
            assert len(glass)==len(p['glass_plies_m'])
            np.testing.assert_allclose([2*g.size[1] for g in glass],p['glass_plies_m'],atol=1e-12)
            if len(glass)==2:
                gap=glass[1].pos[1]-glass[1].size[1]-(glass[0].pos[1]+glass[0].size[1])
                assert gap==pytest.approx(p['sealed_gap_m'])
            assert all(2*g.size[1]<.013 for g in glass)
            actual=sum(g.volume()*g.density for g in body.geoms if g.visual and g.semantic in ('leaf','glass','seal') and g.density>100)
            assert actual<=p['total_kg']*1.01  # mortises can remove real material
        checked+=1
    assert checked==54


def test_frame_depth_does_not_turn_into_glass_mass():
    for slab in M.FRAMED_GLASS_SLABS:
        a=M.framed_glass_profile(slab,.914,2.134,.045)
        b=M.framed_glass_profile(slab,.914,2.134,.06)
        assert a['glass_kg']==b['glass_kg']
        assert b['frame_kg']>a['frame_kg']
        with pytest.raises(ValueError,match='do not fit'):
            M.framed_glass_profile(slab,.914,2.134,.01)


def test_compiled_insulating_glass_gap_contains_no_hidden_slab(specs,tmp_path):
    mujoco=pytest.importorskip('mujoco')
    export_door(specs['db0023_sliding_single'],str(tmp_path/'doors'),str(tmp_path/'hardware'),formats=('mjcf','json'))
    model=mujoco.MjModel.from_xml_path(str(tmp_path/'doors/db0023_sliding_single/door.xml'))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    ids=[model.geom(f'leaf_glass_ply_{i}').id for i in (0,1)]
    centre=(data.geom_xpos[ids[0]]+data.geom_xpos[ids[1]])/2
    for geom in range(model.ngeom):
        if not model.geom_contype[geom] or model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_BOX:continue
        local=data.geom_xmat[geom].reshape(3,3).T@(centre-data.geom_xpos[geom])
        assert max(abs(local)-model.geom_size[geom])>=0,model.geom(geom).name
    normal=data.geom_xmat[ids[0]].reshape(3,3)[:,1].copy()
    hit=np.array([-1],dtype=np.int32)
    distance=mujoco.mj_ray(model,data,centre,normal,None,True,-1,hit)
    assert hit[0]==ids[1]
    assert distance==pytest.approx(.011/2,abs=2e-6)

"""Native journal/stock regressions for conventional rotary assemblies."""
from copy import deepcopy
import xml.etree.ElementTree as ET
import mujoco,numpy as np,pytest
from doorbench.spec import generate_all
from doorbench.build import build_model
from doorbench.export.mjcf import build_mjcf
from doorbench.geometry import common as C
from doorbench.ir import ALL_TIERS
from doorbench.rotary_shaft_qa import run_rotary_shaft_mount_qa


@pytest.fixture(scope='module')
def corpus():
    models={}
    for s in generate_all():
        ir=build_model(s)
        if ir.meta.get('rotary_shafts'):models[s['id']]=(s,ir)
    assert len(models)==287
    return models


def native(ir,tier='full'):
    assets={g.mesh_name+'.obj':g.mesh.export(file_type='obj',include_normals=False,include_texture=False).encode()
        for b in ir.bodies for g in b.geoms if g.type=='mesh'}
    return mujoco.MjModel.from_xml_string(ET.tostring(build_mjcf(ir,tier=tier,mesh_dir_rel=''),encoding='unicode'),assets)


@pytest.mark.parametrize('family',('swing_single','swing_double','automatic_swing','dutch','gate_swing','cold_storage','sliding_single','garage_tiltup','pivot'))
def test_all_native_journals_stock_and_material_inertia(corpus,family):
    for door,(spec,ir) in corpus.items():
        if spec['family']!=family:continue
        full=native(ir)
        assert full.body_mass.sum()==pytest.approx(ir.meta['moving_assembly_mass_kg'],abs=.001),door
        for tier in ('full','simple','minimal'):
            m=full if tier=='full' else native(ir,tier)
            report=run_rotary_shaft_mount_qa(m,ir.meta)
            assert report['ok'],(door,tier,report['failures'])
            for row in ir.meta['rotary_shafts']:
                for name in (row['body'],row['support_body']):
                    assert m.body(name).mass==pytest.approx(full.body(name).mass,abs=1e-9),(door,tier,name)
                    assert m.body(name).inertia==pytest.approx(full.body(name).inertia,abs=1e-9),(door,tier,name)
                assert m.geom(row['shaft_geom']).contype[0]
                assert m.dof_armature[m.joint(row['joint']).dofadr[0]]==0.
        for row in ir.meta['rotary_shaft_accounting']:
            assert row['removed_material_kg']['leaf']>=0 and row['removed_material_kg']['glass']>=0
            assert row['removed_operator_allowance_kg']+row['retained_operator_allowance_kg']==pytest.approx(row['original_row']['hardware_parts']['operator'])


@pytest.mark.parametrize('defect',('filled_stock_bore','filled_journal','floating_journals'))
def test_real_compiled_defects_cannot_hide_behind_parent_filtering(corpus,defect):
    ir=deepcopy(corpus['db0002_swing_single'][1]);row=ir.meta['rotary_shafts'][0]
    support=ir.body(row['support_body']);body=ir.body(row['body']);leaf=ir.body(row['leaf'])
    mat=support.geoms[0].material
    if defect=='floating_journals':support.pos=(support.pos[0],support.pos[1]+.20,support.pos[2])
    elif defect=='filled_stock_bore':
        g=C.box('negative_filled_bore',body.pos,(.009,.01,.009),mat,0.,True,True,ALL_TIERS,'leaf','Restored uncut stock')
        leaf.geoms.append(g);row['leaf_stock_geoms'].append(g.name)
    else:
        g=C.box('negative_filled_journal',support.geoms[0].pos,(.009,.004,.009),mat,0.,True,True,ALL_TIERS,'mechanism','Filled shaft journal')
        support.geoms.append(g);row['support_geoms'].append(g.name)
    report=run_rotary_shaft_mount_qa(native(ir),ir.meta)
    assert not report['ok']


def test_rotated_brace_pieces_preserve_material_volume():
    from doorbench.ir import Body,quat_from_axis_angle
    from doorbench.geometry.rotary_shafts import _bore_stock
    leaf=Body('test_leaf',None);g=C.box('diagonal',(.0,.0,.0),(.5,.01,.05),'wood',500.,True,True,ALL_TIERS,'leaf')
    g.quat=tuple(quat_from_axis_angle((0,1,0),.6));leaf.geoms.append(g)
    before=g.volume();result=_bore_stock(leaf,(-.007,-.1,-.007),(.007,.1,.007),'shaft_bore')
    # The square bore lies wholly within this rotated board and crosses its
    # unchanged20 mm thickness, giving an independent exact removed volume.
    removed=.014**2*.02
    assert result['removed_geometry_volume_m3']==pytest.approx(removed,abs=1e-12)
    assert sum(v.volume() for v in leaf.geoms)==pytest.approx(before-removed,abs=1e-12)
    assert all(v.mesh.is_watertight and v.mesh.is_convex for v in leaf.geoms)


def test_single_face_stubs_do_not_enter_independent_opposite_hardware(corpus):
    rows=[(door,ir,row) for door,(_,ir) in corpus.items() for row in ir.meta['rotary_shafts'] if len(row['faces'])==1]
    assert len(rows)==29
    assert sum('allocation estimate' in a['allocation_scope'] for _,ir,_ in rows for a in ir.meta['rotary_shaft_accounting'])==10
    for door,ir,row in rows:
        body=ir.body(row['body']);shaft=next(g for g in body.geoms if g.name==row['shaft_geom'])
        assert abs(shaft.pos[1])-shaft.size[1]==pytest.approx(0.,abs=1e-12),door
        absent='n' if row['faces'][0]>0 else 'p'
        assert not any(g.name==row['body']+'_shaft_collar_'+absent for g in body.geoms),door
    # Recompile the old unwanted through-face projection. A direct check
    # against only the new journal body would miss the original handleset.
    ir=deepcopy(corpus['db0114_swing_single'][1]);row=ir.meta['rotary_shafts'][0]
    body=ir.body(row['body']);shaft=next(g for g in body.geoms if g.name==row['shaft_geom'])
    low=-.034;high=shaft.pos[1]+shaft.size[1]
    shaft.pos=(0.,(low+high)/2,0.);shaft.size=(shaft.size[0],(high-low)/2)
    report=run_rotary_shaft_mount_qa(native(ir),ir.meta)
    assert any(f['check']=='moving_trim_vs_fixed_parent_stock' for f in report['failures'])

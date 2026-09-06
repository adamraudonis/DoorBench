"""Compiled material, contact and load-path checks for flexible PVC curtains."""
import copy
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from doorbench.build import build_model, export_door
from doorbench.export.mjcf import build_mjcf
from doorbench.physics import derive
from doorbench.spec import generate_all
from doorbench.strip_mechanics_qa import run_strip_mechanics_qa


SPECS = [s for s in generate_all() if s['family'] == 'strip_curtain']


@pytest.fixture(scope='module')
def exports(tmp_path_factory):
    root = tmp_path_factory.mktemp('strips')
    for spec in SPECS:
        export_door(spec, str(root/'doors'), str(root/'hardware'), formats=('mjcf', 'json'))
    return root


@pytest.mark.parametrize('spec', SPECS, ids=lambda s: s['id'])
def test_all_strips_real_material_inertia_contacts_and_access(exports, spec):
    folder = exports/'doors'/spec['id']
    desc = json.loads((folder/'model.json').read_text())
    meta = desc['meta']; curtain = meta['strip_curtain']
    count = spec['leaf']['count']; width = spec['leaf']['strip_width']
    height = spec['leaf']['height']; thickness = spec['leaf']['thickness']
    for tier in ('full', 'simple', 'minimal'):
        m = mujoco.MjModel.from_xml_path(str(folder/('door.xml' if tier == 'full' else f'door_{tier}.xml')))
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        expected=set()
        for c in curtain['controls']:
            a,b=sorted((m.body(c['fixed_tab_body']).id,m.body(c['segment_bodies'][0]).id))
            expected.add((a<<16)+b)
        assert set(map(int,m.exclude_signature)) == expected
        assert len(curtain['controls']) == count
        assert curtain['fixed_pvc_mass_kg'] == pytest.approx(count*width*thickness*.028*1250)
        volume = 0.
        for control in curtain['controls']:
            bodies = [m.body(name).id for name in control['segment_bodies']]
            tab=m.geom(control['fixed_tab_geom']).id;first=m.geom(control['segment_bodies'][0]+'_pvc').id
            assert d.geom_xpos[tab,2]-m.geom_size[tab,2] == pytest.approx(d.geom_xpos[first,2]+m.geom_size[first,2],abs=1e-8)
            assert control['cut_stock_length_m'] == pytest.approx(height+.028)
            for clamp in control['clamp_geoms']:
                assert mujoco.mj_geomDistance(m,d,tab,m.geom(clamp).id,.01,None) == pytest.approx(0,abs=1e-8)
            for k, body in enumerate(bodies):
                assert m.body_parentid[body] == (bodies[k-1] if k else m.body(control['fixed_tab_body']).id)
                joint = int(m.body_jntadr[body]); dof = int(m.jnt_dofadr[joint])
                geom = m.geom(control['segment_bodies'][k]+'_pvc').id
                assert m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
                assert m.geom_contype[geom] and m.geom_conaffinity[geom]
                assert m.geom_margin[geom] == 0  # no artificial preload at touching lateral edges
                assert m.dof_armature[dof] == 0
                assert m.body_mass[body] > 0 and np.all(m.body_inertia[body] > 0)
                assert 2*m.geom_size[geom, 0] == pytest.approx(width, abs=1e-6)
                assert 2*m.geom_size[geom, 1] == pytest.approx(thickness, abs=1e-6)
                volume += 8*float(np.prod(m.geom_size[geom]))
            # Gravity and joint inertia are from the whole downstream strip,
            # while material density is not multiplied by decorative surfaces.
            native_mass = sum(float(m.body_mass[b]) for b in bodies)
            assert native_mass == pytest.approx(width*height*thickness*1250, rel=1e-5)
        assert volume == pytest.approx(count*width*height*thickness, rel=1e-5)
        # Check two same-layer rectangles directly, not a semantic exception.
        for a in curtain['controls']:
            for b in curtain['controls']:
                if a['strip'] >= b['strip'] or a['layer'] != b['layer']: continue
                ga=m.geom(a['segment_bodies'][0]+'_pvc').id
                gb=m.geom(b['segment_bodies'][0]+'_pvc').id
                assert abs(d.geom_xpos[ga,0]-d.geom_xpos[gb,0]) >= width-1e-6
        # Each approached side has an actual exposed PVC face at contact height.
        for layer, sign, site_index in ((0, 1, 0), (1, -1, 1)):
            selected = min((c for c in curtain['controls'] if c['layer']==layer),
                           key=lambda c:abs(c['strip']-count//2))
            site=m.site(selected['push_sites'][site_index]).id
            origin=d.site_xpos[site]+np.array([0., -sign*.08, 0.])
            hit=np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(m,d,origin,np.array([0.,sign,0.]),None,True,-1,hit)
            assert distance == pytest.approx(.08,abs=1e-6)
            assert m.geom_bodyid[hit[0]] == m.site_bodyid[site]


def test_strip_dynamics_mass_is_whole_carried_strip():
    spec=SPECS[0]; phys=derive(spec)
    rows=phys['mass']['per_body']
    first=rows[0]
    expected=sum(row['total_kg'] for row in rows if row['strip_index']==first['strip_index'])
    assert phys['mass']['dynamics_mass_kg'] == pytest.approx(expected)
    assert phys['mass']['hardware_kg'] == 0  # rail and clamps are world-fixed
    assert expected > 4*first['total_kg']


def test_geom_impedance_and_native_bounds_survive_serialization():
    model=build_model(SPECS[0]); geom=next(g for b in model.bodies for g in b.geoms if g.name.endswith('_pvc'))
    geom.solimp=(.91,.97,.0003)
    assert geom.to_dict()['solimp'] == [.91,.97,.0003]
    root=build_mjcf(model,timestep=.00001)
    compiled=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    assert compiled.opt.timestep == .00001
    assert compiled.geom_solimp[compiled.geom(geom.name).id,:3] == pytest.approx(geom.solimp)
    assert root.find('size').attrib['memory'] == f"{model.meta['native_arena_memory_mib']}M"
    model.meta.pop('native_timestep_s'); model.meta.pop('native_arena_memory_mib')
    root=build_mjcf(model)
    assert root.find('option').attrib['timestep'] == '0.002'
    assert root.find('size').attrib['memory'] == '16M'


@pytest.mark.parametrize('key,value', [
    ('native_timestep_s', x) for x in (True, 0, -1, float('nan'), float('inf'), '0.001')
] + [
    ('native_arena_memory_mib', x) for x in (True, 15, 129, 32.5, float('nan'), float('inf'), '64')
])
def test_invalid_native_solver_bounds_rejected(key, value):
    model=build_model(copy.deepcopy(SPECS[0])); model.meta[key]=value
    with pytest.raises(ValueError, match=key): build_mjcf(model)


@pytest.mark.parametrize('value', ['strip_0_clamp_tab', ['missing'], ['strip_0'],
                                  ['strip_0_clamp_tab','strip_0_clamp_tab']])
def test_fixed_material_wrapper_contract_rejects_invalid_names(value):
    model=build_model(SPECS[0]);model.meta['native_fixed_body_names']=value
    with pytest.raises(ValueError,match='native_fixed_body_names'):build_mjcf(model)


@pytest.mark.parametrize('defect,expected', [
    ('collision', 'disabled_strip_contact'),
    ('stiffness', 'wrong_material_bending_stiffness'),
    ('preload', 'preloaded_strip_contact_margin'),
    ('site', 'occluded_or_off_surface_push_site'),
    ('timestep', 'native_timestep_exceeds_authored_bound'),
])
def test_native_gate_rejects_missing_contacts_and_fabricated_response(exports, defect, expected):
    folder=exports/'doors'/SPECS[0]['id']
    m=mujoco.MjModel.from_xml_path(str(folder/'door.xml'))
    meta=json.loads((folder/'model.json').read_text())['meta']
    control=meta['strip_curtain']['controls'][meta['n_strips']//2]
    geom=m.geom(control['segment_bodies'][1]+'_pvc').id
    if defect=='collision': m.geom_contype[geom]=m.geom_conaffinity[geom]=0
    if defect=='stiffness': m.jnt_stiffness[m.joint(control['root_joint']).id]*=100
    if defect=='preload': m.geom_margin[geom]=.0003
    if defect=='site': m.site_pos[m.site(control['push_sites'][0]).id,1]-=.02
    if defect=='timestep': m.opt.timestep=.002
    result=run_strip_mechanics_qa(m,meta)
    assert not result['ok']
    assert expected in [f['check'] for f in result['failures']]
    assert 'phases' not in result['measurements']  # reject before running a misleading cycle


def test_native_gate_rejects_floating_clamping_tab(exports):
    folder=exports/'doors'/SPECS[0]['id'];m=mujoco.MjModel.from_xml_path(str(folder/'door.xml'))
    meta=json.loads((folder/'model.json').read_text())['meta']
    tab=m.geom(meta['strip_curtain']['controls'][0]['fixed_tab_geom']).id
    m.geom_pos[tab,2]+=.003
    result=run_strip_mechanics_qa(m,meta)
    assert not result['ok']
    assert any(f['check']=='unbonded_clamping_tab' for f in result['failures'])


def test_bond_exclusion_retains_neighbor_and_world_contacts():
    # The fixed tab shares world weld ID, so prove the explicit body exclusion
    # does not accidentally suppress all fixed geometry against this strip.
    xml='''<mujoco><worldbody>
      <geom name="wall" type="box" pos="0 .035 -.12" size=".1 .02 .02"/>
      <body name="tab"><geom name="fixed_tab" type="box" pos="0 0 .025" size=".1 .005 .025"/>
        <body name="strip"><joint type="hinge" axis="1 0 0"/>
          <geom name="pvc" type="box" pos="0 0 -.1" size=".1 .005 .1"/>
        </body></body>
      <body name="neighbor"><geom name="neighbor_tab" type="box" pos=".1 0 .025" size=".1 .005 .025"/></body>
      </worldbody><contact><exclude body1="tab" body2="strip"/></contact></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);d.qpos[0]=.2;mujoco.mj_forward(m,d)
    pairs={frozenset((m.geom(c.geom1).name,m.geom(c.geom2).name)) for c in d.contact}
    assert frozenset(('fixed_tab','pvc')) not in pairs
    assert frozenset(('neighbor_tab','pvc')) in pairs
    assert frozenset(('wall','pvc')) in pairs


def test_real_surface_load_opens_and_releases_without_prescribed_flexures(exports):
    folder=exports/'doors'/SPECS[0]['id']
    m=mujoco.MjModel.from_xml_path(str(folder/'door.xml'))
    meta=json.loads((folder/'model.json').read_text())['meta']
    result=run_strip_mechanics_qa(m,meta,repetitions=1)
    assert result['ok'], result
    assert len(result['measurements']['phases']) == 5
    for phase in result['measurements']['phases']:
        if phase['phase'].startswith('release_'):
            assert phase['energy_change_J'] < 0
        elif phase['phase'] != 'settle':
            assert phase['peak_displacement_m'] > .10


def test_real_obstacle_blocks_surface_load_instead_of_passing_open_gate():
    from doorbench.geometry import common as C
    model=build_model(SPECS[0])
    # The block is behind the sheet, so the initial front-face ray remains
    # accessible. Only the native force response discovers the blocked path.
    model.body('world_env').geoms.append(C.box('test_obstruction',(0,.055,1.),
        (.05,.02,.20),'mat_wall',800,semantic='wall'))
    m=mujoco.MjModel.from_xml_string(ET.tostring(build_mjcf(model),encoding='unicode'))
    result=run_strip_mechanics_qa(m,model.meta,repetitions=1)
    assert not result['ok'], result
    checks=[f['check'] for f in result['failures']]
    assert 'surface_load_did_not_open_material' in checks or 'thin_sheet_penetration' in checks
    assert 'occluded_or_off_surface_push_site' not in checks

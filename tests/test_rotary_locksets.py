"""Source-bound independent egress and exterior catch regressions."""
import copy,json
from pathlib import Path
import mujoco,numpy as np,pytest
from doorbench.spec import generate_all
from doorbench import hardware as H
from doorbench.build import export_door,build_model
from doorbench.geometry.rotary_lockset import applicable
from doorbench.rotary_lockset import compile_rotary_catches,apply_rotary_catches
from doorbench.rotary_lockset_qa import run_rotary_lockset_qa

SPECS=[s for s in generate_all() if applicable(s,H.OPERATORS[s['operator']['model']],[-1.,1.])]

@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    root=tmp_path_factory.mktemp('independent-locksets')
    for s in SPECS:export_door(s,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
    return root


def load(root,spec,tier='full'):
    p=root/'doors'/spec['id'];m=mujoco.MjModel.from_xml_path(str(p/('door.xml' if tier=='full' else f'door_{tier}.xml')))
    return m,json.loads((p/'model.json').read_text()),json.loads((p/'spec.json').read_text())


def test_scope_and_keypad_entry_side_normalization():
    assert len(SPECS)==97
    spec=next(s for s in SPECS if s['id']=='db0264_swing_single')
    assert spec['operator']['sides']=='both' and spec['operator']['far_side'] is None
    large=[s['id'] for s in SPECS if s['operator']['model']=='knob_round_large']
    assert large==['db0462_swing_single','db0592_swing_single','db0815_swing_single']
    a,b=H.OPERATORS['knob_round'],H.OPERATORS['knob_round_large']
    for key in ('travel','dead_travel','spring_torque_preload','spring_rate','operable_force_limit','mass'):
        assert getattr(a,key)==getattr(b,key)


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_actual_independent_shafts_stock_preparation_and_all_tier_installation(fixtures,spec):
    full,full_model,full_spec=load(fixtures,spec)
    assert full.body_mass.sum()==pytest.approx(full_spec['physics']['mass']['total_kg'],abs=.001)
    row=full_model['meta']['rotary_locksets'][0]
    required_bodies={full.body(int(full.jnt_bodyid[full.joint(row[k]).id])).name for k in ('inside_joint','outside_joint','catch_joint')}
    required_bodies.add('leaf_handle_chassis')
    for tier in ('full','simple','minimal'):
        m,model,s=load(fixtures,spec,tier);meta=model['meta'];r=meta['rotary_locksets'][0]
        inside,outside=(m.joint(r[k]).id for k in ('inside_joint','outside_joint'))
        assert inside!=outside and m.jnt_bodyid[inside]!=m.jnt_bodyid[outside]
        assert m.jnt_range[inside,1]==pytest.approx(r['operator_travel_rad'],abs=1e-6)
        assert m.jnt_range[outside,1]==pytest.approx(r['operator_travel_rad'],abs=1e-6)
        assert meta['inside_egress_inputs']==[r['inside_joint']]
        assert r['released_by_default']==(not spec['lock']['engaged'])
        assert compile_rotary_catches(m,meta)[0].released==r['released_by_default']
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        for joint,sites in r['input_sites'].items():
            assert len(sites)==(2 if r['input_model']=='opposed_surface_pair' else 1)
            j=m.joint(joint).id
            for name in sites:assert m.site(name).bodyid[0]==m.jnt_bodyid[j]
            if len(sites)==2:
                centers=np.array([d.site_xpos[m.site(name).id] for name in sites])
                tangent=np.cross(d.xaxis[j],centers-d.xanchor[j])
                assert tangent.sum(axis=0)==pytest.approx(np.zeros(3),abs=1e-9)
                geom=m.geom(r['input_surface_geoms'][joint]).id
                if r['input_surface_shape']=='rounded_profile':
                    assert np.linalg.norm(tangent[0])==pytest.approx(.0361,abs=1e-9)
                    for name,point in zip(sites,centers):
                        normal=d.site_xmat[m.site(name).id].reshape(3,3)[:,2]
                        assert mujoco.mj_rayMesh(m,d,geom,point+normal*.005,-normal)==pytest.approx(.005,abs=1e-6)
                else:
                    radius=m.geom_size[geom,0]
                    assert np.linalg.norm(tangent[0])==pytest.approx(.8*radius,abs=1e-9)
                    for name,point in zip(sites,centers):
                        radial=point-d.geom_xpos[geom]
                        assert np.linalg.norm(radial)==pytest.approx(radius,abs=1e-9)
                        normal=d.site_xmat[m.site(name).id].reshape(3,3)[:,2]
                        assert normal@radial/radius==pytest.approx(1.,abs=1e-6)
        stock=[g['name'] for b in model['bodies'] if b['name']==r['leaf'] for g in b['geoms'] if g['semantic'] in ('leaf','glass') and g['name'] in [m.geom(i).name for i in range(m.ngeom)]]
        def gap(a,b):return mujoco.mj_geomDistance(m,d,m.geom(a).id,m.geom(b).id,.2,None)
        assert min(gap(r['support_geom'],g) for g in stock)<=1e-6
        for bearing in r['bearing_geoms']:
            assert min(gap(bearing,g) for g in stock)<=1e-6
        assert min(gap(r['support_geom'],g) for g in r['guide_geoms'])<=1e-6
        assert min(gap(r['catch_geom'],g) for g in r['guide_geoms'])>=.00025
        for stop in r['native_stop_geoms']:
            assert min(gap(stop,g) for g in [r['support_geom'],*r['guide_geoms'],*r['native_stop_geoms']] if g!=stop)<=1e-6
        for shaft in r['shaft_geoms']:
            assert all(mujoco.mj_geomDistance(m,d,m.geom(shaft).id,m.geom(g).id,.05,None)>=-1e-6 for g in stock)
        preparations=r['escutcheon_preparations']
        assert len(preparations)==(2 if H.OPERATORS[spec['operator']['model']].style_params.get('escutcheon') else 0)
        for preparation in preparations:
            assert preparation['original_mass_kg']-preparation['remaining_mass_kg']==pytest.approx(
                preparation['density_kg_m3']*preparation['cut']['removed_geometry_volume_m3'],abs=1e-10)
            for plate in preparation['geoms']:
                assert m.geom(plate).contype[0] and min(gap(plate,g) for g in stock)<=1e-6
                assert min(gap(plate,g) for g in r['shaft_geoms']+r['bearing_geoms'])>=.0009
        assert mujoco.mj_geomDistance(m,d,*[m.geom(n).id for n in r['shaft_geoms']],.05,None)==pytest.approx(.001,abs=1e-6)
        # Existing optional keypad keys/pet flaps/closer risers can be absent
        # in reduced tiers; the installed rotary mechanism cannot change.
        for name in required_bodies:
            assert m.body(name).mass==pytest.approx(full.body(name).mass,abs=1e-9)
            assert m.body(name).inertia==pytest.approx(full.body(name).inertia,abs=1e-9)
        assert s['physics']['mass']['hardware_parts']['operator']==0
        assert meta['rotary_material_accounting']['removed_stock_kg']>0


@pytest.mark.parametrize('spec',[s for s in SPECS if s['id'] in ('db0111_swing_single','db0166_swing_single','db0264_swing_single')],ids=lambda s:s['id'])
@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_native_independent_inputs_release_return_and_removed_pin(fixtures,spec,tier):
    m,model,_=load(fixtures,spec,tier)
    xml=fixtures/'doors'/spec['id']/('door.xml' if tier=='full' else f'door_{tier}.xml')
    r=run_rotary_lockset_qa(m,model['meta'],source_xml=xml)
    assert r['ok'],r['failures']
    assert r['fixture']['kind']=='recompiled_rigid_leaf_bench'
    phases=('outside_locked','unload','inside_locked','inside_return','catch_release',
            'outside_released','outside_return','both_inputs','both_return','catch_reengage')
    assert [p['phase'] for p in r['probes']]==['unpowered_settle',
        *[f'{cycle}:{phase}' for cycle in range(2) for phase in phases],'removed_locking_pin_negative']
    row=model['meta']['rotary_locksets'][0]
    assert max(p['max_input_force_N'] for p in r['probes'])<=row['operator_force_cap_N']+1e-9
    if row['input_model']=='opposed_surface_pair':
        assert max(p['max_total_absolute_surface_force_per_input_N'] for p in r['probes'])<=2*row['operator_force_cap_N']+1e-9
        assert max(p['max_resultant_force_per_input_N'] for p in r['probes'])<1e-9


def test_credential_availability_does_not_change_installed_lock_state():
    original=next(s for s in SPECS if s['id']=='db0166_swing_single')
    a=build_model(original);s=copy.deepcopy(original);s['lock']['robot_side_release']=False;b=build_model(s)
    for model in (a,b):
        r=model.meta['rotary_locksets'][0];catch=next(x.joint for x in model.bodies if x.joint and x.joint.name==r['catch_joint'])
        assert not r['released_by_default'] and catch.initial==0
        assert model.body('leaf_handle').joint.range[1]==.87


@pytest.mark.parametrize('door',('db0061_swing_single','db0160_swing_single','db0294_swing_single'))
def test_restored_solid_escutcheon_is_a_real_shaft_intersection(fixtures,door):
    spec=next(s for s in SPECS if s['id']==door)
    native,model,_=load(fixtures,spec)
    row=model['meta']['rotary_locksets'][0]
    preparation=row['escutcheon_preparations'][0]
    geoms=[native.geom(n).id for n in preparation['geoms']]
    lo=np.min(native.geom_pos[geoms]-native.geom_size[geoms],axis=0)
    hi=np.max(native.geom_pos[geoms]+native.geom_size[geoms],axis=0)
    # Recompile the former solid plate. Do not mutate compiled bounds and
    # accidentally rely on stale MuJoCo broadphase acceleration structures.
    source=mujoco.MjSpec.from_file(str(fixtures/'doors'/door/'door.xml'))
    parent=source.body(native.body(int(native.geom_bodyid[geoms[0]])).name)
    parent.add_geom(name='negative_solid_plate',type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(hi-lo)/2,pos=(hi+lo)/2,density=preparation['density_kg_m3'])
    damaged=source.compile();data=mujoco.MjData(damaged);mujoco.mj_forward(damaged,data)
    assert min(mujoco.mj_geomDistance(damaged,data,damaged.geom('negative_solid_plate').id,
        damaged.geom(n).id,.05,None) for n in row['shaft_geoms'])<-.005


def test_reversed_approach_selects_same_physical_inside_when_robot_moves_inside():
    s=copy.deepcopy(next(s for s in SPECS if s['id']=='db0166_swing_single'))
    original=build_model(s).meta['rotary_locksets'][0]
    s['robot']['approach_side']='+y';s['robot']['robot_outside']=False
    model=build_model(s);r=model.meta['rotary_locksets'][0]
    assert r['outside_face']==original['outside_face']==-1
    assert r['inside_joint']=='leaf_handle_hinge'
    assert next(b.joint for b in model.bodies if b.joint and b.joint.name==r['outside_joint']).robot_interactive is False


def test_reversed_approach_native_controls_and_bench_binding(tmp_path):
    s=copy.deepcopy(next(s for s in SPECS if s['id']=='db0166_swing_single'))
    s['robot']['approach_side']='+y';s['robot']['robot_outside']=False
    export_door(s,str(tmp_path/'doors'),str(tmp_path/'hardware'),formats=('mjcf','json'))
    m,model,_=load(tmp_path,s);xml=tmp_path/'doors'/s['id']/'door.xml'
    proof=run_rotary_lockset_qa(m,model['meta'],source_xml=xml)
    assert proof['ok'],proof['failures']
    damaged=copy.copy(m);damaged.geom_pos[0,0]+=.01
    with pytest.raises(ValueError,match='source/native mismatch'):
        run_rotary_lockset_qa(damaged,model['meta'],source_xml=xml)


def test_catch_input_never_changes_pose_ranges_or_permissions(fixtures):
    m,model,_=load(fixtures,next(s for s in SPECS if s['id']=='db0166_swing_single'));d=mujoco.MjData(m)
    rules=compile_rotary_catches(m,model['meta']);before=(d.qpos.copy(),m.jnt_range.copy(),d.qfrc_passive.copy())
    apply_rotary_catches(m,d,rules,True)
    assert np.array_equal(d.qpos,before[0]) and np.array_equal(m.jnt_range,before[1])
    delta=d.qfrc_passive-before[2];assert np.count_nonzero(delta)==1 and delta[rules[0].dof]==8.
    damaged=copy.deepcopy(model['meta']);damaged['rotary_locksets'][0]['released_threshold_m']=.1
    with pytest.raises(ValueError):compile_rotary_catches(m,damaged)

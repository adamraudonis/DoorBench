"""Physical guard retention/release and source-construction regressions."""
from copy import deepcopy
from functools import lru_cache
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from doorbench.build import build_model
from doorbench.export.mjcf import build_mjcf
from doorbench.spec import generate_all
from doorbench.security_mechanics_qa import run_security_mechanics_qa

IDS=('db0117_swing_single','db0159_automatic_swing','db0481_swing_single',
     'db0727_swing_single','db0781_swing_single','db0953_swing_single')
SPECS={s['id']:s for s in generate_all() if s['id'] in IDS}


@lru_cache(None)
def fixture(door,tier='full',isolated=False):
    spec=deepcopy(SPECS[door])
    if isolated:spec['closer']['model']='none'
    model=build_model(spec)
    assets={g.mesh_name+'.obj':g.mesh.export(file_type='obj',include_normals=False,include_texture=False).encode()
            for b in model.bodies for g in b.geoms if g.type=='mesh' and tier in g.tiers}
    xml=ET.tostring(build_mjcf(model,tier=tier,mesh_dir_rel=''),encoding='unicode')
    return xml,assets,deepcopy(model.meta)


def native(door,tier='full',isolated=False):
    xml,assets,meta=fixture(door,tier,isolated)
    return mujoco.MjModel.from_xml_string(xml,assets),deepcopy(meta)


def test_security_swing_normalization_does_not_change_access_choices():
    assert set(SPECS)==set(IDS)
    changed={s['id'] for s in SPECS.values() if 'security_guard_inward_swing' in s.get('tags',[])}
    assert changed=={'db0159_automatic_swing','db0481_swing_single','db0781_swing_single'}
    for s in SPECS.values():assert s['robot']['is_push']==s['robot']['robot_outside']


@pytest.mark.parametrize('door',('db0117_swing_single','db0481_swing_single','db0781_swing_single','db0953_swing_single'))
def test_fixed_anchor_bracket_has_actual_pin_bore_despite_parent_filtering(door):
    m,meta=native(door);d=mujoco.MjData(m)
    pin=m.geom('leaf_chain_anchor_pin').id
    brackets=[m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith('leaf_chain_frame_bracket')]
    assert len(brackets)==4
    j=m.joint('leaf_chain_yaw').id
    for yaw in (-2.,-.5,0.,1.,2.5):
        d.qpos[m.jnt_qposadr[j]]=yaw;mujoco.mj_kinematics(m,d)
        assert min(mujoco.mj_geomDistance(m,d,pin,g,.02,None) for g in brackets)>=.0005-1e-7
    # Refill the hole with the exact original bracket envelope. The explicit
    # distance catches the steel overlap even though this pin is visual-only.
    lo=np.min([d.geom_xpos[g]-m.geom_size[g] for g in brackets],axis=0)
    hi=np.max([d.geom_xpos[g]+m.geom_size[g] for g in brackets],axis=0)
    g=brackets[0];m.geom_pos[g]=(lo+hi)/2;m.geom_size[g]=(hi-lo)/2
    mujoco.mj_kinematics(m,d)
    assert mujoco.mj_geomDistance(m,d,pin,g,.02,None)<-.003


@pytest.mark.parametrize('door',('db0117_swing_single','db0481_swing_single','db0781_swing_single','db0953_swing_single'))
def test_adjacent_chain_wire_interfaces_are_open_and_explicit_native_contacts(door):
    m,meta=native(door);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    pairs=meta['security_guards'][0]['adjacent_wire_contact_pairs']
    assert len(pairs)==128
    compiled={frozenset((int(a),int(b))) for a,b in zip(m.pair_geom1,m.pair_geom2)}
    for pair in pairs:
        a,b=(m.geom(pair[k]).id for k in ('geom1','geom2'))
        assert frozenset((a,b)) in compiled
        assert m.geom_contype[a] and m.geom_contype[b]
        assert mujoco.mj_geomDistance(m,d,a,b,.03,None)>-.000001
    # The prior3mm-wide wire centres give only a3mm opening for3mm wire;
    # the initial folded middle pair then intersects by2.5mm. Restoring that
    # solid envelope must fail the same direct geometric measurement.
    from doorbench.ir import quat_z_to
    segment=.148/8
    for i in (3,4):
        across=np.array([1.,0.,0.]) if i%2==0 else np.array([0.,1.,0.])
        points=[-across*.003+[0,0,.0015],across*.003+[0,0,.0015],
                across*.003+[0,0,segment-.0015],-across*.003+[0,0,segment-.0015]]
        for k in range(4):
            a,b=points[k],points[(k+1)%4];g=m.geom(f'leaf_chain_link_{i}_wire_{k}').id
            m.geom_pos[g]=(a+b)/2;m.geom_quat[g]=quat_z_to(b-a)
            m.geom_size[g,:2]=(.0015,np.linalg.norm(b-a)/2)
    mujoco.mj_kinematics(m,d)
    if meta['security_guards'][0]['engaged_initial']:
        assert mujoco.mj_geomDistance(m,d,m.geom('leaf_chain_link_3_wire_2').id,
                                     m.geom('leaf_chain_link_4_wire_0').id,.02,None)<-.001


@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_chain_anchor_mounts_on_actual_collidable_casing_face(tier):
    m,meta=native('db0781_swing_single',tier);d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
    casing=m.geom('casing_r_p').id;plate=m.geom('leaf_chain_frame_plate').id
    assert m.geom_contype[casing] and m.geom_conaffinity[casing]
    assert abs(d.geom_xpos[plate,0]-d.geom_xpos[casing,0])+m.geom_size[plate,0]<=m.geom_size[casing,0]+1e-8
    assert d.geom_xpos[plate,1]-m.geom_size[plate,1]==pytest.approx(
        d.geom_xpos[casing,1]+m.geom_size[casing,1],abs=1e-8)
    assert m.geom_size[casing,0]>m.geom_size[plate,0]
    guard=meta['security_guards'][0]
    leaf=m.body(guard['leaf_body']).id
    anchor=np.asarray(guard['frame_anchor_world'])
    for offset in (0.,.025):
        head=np.asarray(guard['keyhole_center_leaf'])+np.asarray(guard['release_sequence'][2]['direction'])*offset
        terminal=head-np.asarray(guard['head_center_local'])+d.xpos[leaf]
        assert np.linalg.norm(terminal-anchor)<guard['chain_length_m']-.001
    # Reinstating the old hidden mounting position buries the anchor plate.
    m.geom_pos[plate,0]=.4815;m.geom_pos[plate,1]=.031
    mujoco.mj_kinematics(m,d)
    assert mujoco.mj_geomDistance(m,d,casing,plate,.1,None)<-.001


def test_chain_service_gate_rejects_removed_adjacent_contact_override():
    m,meta=native('db0953_swing_single')
    row=meta['security_guards'][0]['adjacent_wire_contact_pairs'][0]
    intended=frozenset(m.geom(row[k]).id for k in ('geom1','geom2'))
    index=next(i for i,(a,b) in enumerate(zip(m.pair_geom1,m.pair_geom2))
               if frozenset((int(a),int(b)))==intended)
    # The geoms keep normal collision flags; only the override of MuJoCo's
    # parent/child filter is removed. A contype-only inventory misses this.
    m.pair_geom2[index]=m.geom('floor').id
    result=run_security_mechanics_qa(m,meta)
    assert not result['ok']
    assert 'Missing adjacent wire contact override' in result['failures'][0]


@pytest.mark.parametrize('corruption',('truncate','duplicate'))
def test_chain_service_gate_requires_complete_distinct_adjacent_inventory(corruption):
    m,meta=native('db0953_swing_single');row=meta['security_guards'][0]
    pairs=row['adjacent_wire_contact_pairs']
    row['adjacent_wire_contact_pairs']=pairs[:1] if corruption=='truncate' else [pairs[0]]*128
    result=run_security_mechanics_qa(m,meta)
    assert not result['ok']
    assert 'Incomplete or duplicated adjacent wire contact inventory' in result['failures'][0]


@pytest.mark.parametrize('corruption',('priority','soft_material','truncated_inventory'))
def test_chain_service_gate_rejects_weakened_steel_contact_material(corruption):
    m,meta=native('db0481_swing_single');g=m.geom('leaf_chain_link_6_wire_2').id
    if corruption=='priority':m.geom_priority[g]=0
    elif corruption=='soft_material':m.geom_solref[g,0]=.005
    else:meta['security_guards'][0]['contact_solver_scope']['priority_geoms'].pop()
    result=run_security_mechanics_qa(m,meta)
    assert not result['ok']
    assert 'steel chain contact material' in result['failures'][0]


def test_standalone_service_fails_warning_callback_without_data_counter(monkeypatch):
    from doorbench import security_mechanics_qa as Q
    m,meta=native('db0481_swing_single');data=mujoco.MjData(m)
    previous=mujoco.get_mju_user_warning();forwarded=[]
    def prior(message):forwarded.append(message)
    def exercise(*args):
        callback=mujoco.get_mju_user_warning()
        assert callback is not prior
        callback('Linesearch objective is not convex')
        assert not np.any(data.warning.number)
        return {'failures':[]}
    monkeypatch.setattr(Q,'_exercise',exercise)
    mujoco.set_mju_user_warning(prior)
    try:
        result=Q.run_security_mechanics_qa(m,meta)
        assert not result['ok']
        assert result['native_warning_messages']==['Linesearch objective is not convex']
        assert any('Linesearch objective is not convex' in x for x in result['failures'])
        assert forwarded==['Linesearch objective is not convex']
        assert mujoco.get_mju_user_warning() is prior
    finally:mujoco.set_mju_user_warning(previous)


def test_actual_neck_lip_clearance_catches_tilt_with_unchanged_head_height():
    from doorbench.security_mechanics_qa import _chain_neck_slot_clearance
    from doorbench.ir import quat_from_axis_angle
    m,meta=native('db0481_swing_single');d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
    r=meta['security_guards'][0];head=m.geom(r['head_geom']).id;neck=m.geom(r['neck_geom']).id
    head_before=d.geom_xpos[head].copy()
    good=_chain_neck_slot_clearance(m,d,r)
    assert good['spans_lip'] and good['minimum_clearance_m']==pytest.approx(.0007,abs=1e-7)
    # The head-height check alone still passes, while a tilted cylindrical
    # neck intersects the actual retaining lip near one face.
    m.geom_quat[neck]=quat_from_axis_angle((1,0,0),-math.pi/2+.4)
    mujoco.mj_kinematics(m,d)
    np.testing.assert_array_equal(d.geom_xpos[head],head_before)
    bad=_chain_neck_slot_clearance(m,d,r)
    assert bad['spans_lip'] and bad['minimum_clearance_m']<.0001


def test_neck_must_span_both_lip_faces_before_slot_engagement():
    from doorbench.security_mechanics_qa import _chain_neck_slot_clearance
    m,meta=native('db0481_swing_single');d=mujoco.MjData(m);r=meta['security_guards'][0]
    m.geom_pos[m.geom(r['neck_geom']).id,1]+=.030
    mujoco.mj_kinematics(m,d)
    clearance=_chain_neck_slot_clearance(m,d,r)
    assert clearance['minimum_clearance_m']>.0001
    assert not clearance['spans_lip']


@pytest.mark.parametrize('door',IDS)
@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_all_security_models_have_native_hardware_and_full_leaf_range(door,tier):
    m,meta=native(door,tier);d=mujoco.MjData(m);mujoco.mj_forward(m,d);r=meta['security_guards'][0]
    primary=m.joint(meta['primary_joint']).id
    assert m.jnt_range[primary,1]>.75
    assert m.opt.timestep<=.0001
    assert r['accessible_from_robot']==(not SPECS[door]['robot']['robot_outside'])
    assert len(r['keeper_geoms'])>=3
    assert np.isfinite(d.xpos).all()
    assert min((c.dist for c in d.contact),default=0)>-1e-5
    for name in r['guard_joints']:
        j=m.joint(name).id;b=m.jnt_bodyid[j]
        assert m.dof_armature[m.jnt_dofadr[j]]==0
        assert m.body_mass[b]>0 and min(m.body_inertia[b])>0
    for name in r['keeper_geoms']+[r['head_geom']]:
        assert m.geom_contype[m.geom(name).id]!=0
    if r['kind']=='chain':
        assert m.opt.timestep<=.000025
        assert m.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICIT
        assert len(r['chain_bodies'])==12
        assert m.geom_priority[m.geom(r['head_geom']).id]==1
        steel={m.geom(i).name for i in range(m.ngeom) if m.geom(i).name.startswith('leaf_chain')}
        assert {m.geom(i).name for i in range(m.ngeom) if m.geom_priority[i]}==steel
        assert set(r['contact_solver_scope']['priority_geoms'])==steel
        assert r['slot_width_m']<r['head_diameter_m']<r['keyhole_width_m']
        assert meta['site_wrench_limits_Nm'][r['release_site']]==.02
        assert r['release_sequence'][1]['head_target_leaf']==r['keyhole_center_leaf']
        assert r['handoff_tolerances_m']==pytest.approx({'keyhole_lateral':.0029,'slot_vertical':.0006,'seated_position':.003})
        # The former overlapping sphere proxy is now a genuinely hollow grip.
        eye=m.geom('leaf_chain_tip_eye').id
        walls=[m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith('leaf_chain_finger_grip_')]
        assert len(walls)==6
        assert min(mujoco.mj_geomDistance(m,d,eye,g,.02,None) for g in walls)>.0004
    else:
        j=m.joint(r['guard_joint']).id
        assert m.jnt_range[j,1]==pytest.approx(math.pi,abs=1e-6)
        sid=m.site(r['release_site']).id;g=m.geom('leaf_guard_finger_end').id
        assert np.linalg.norm(d.site_xpos[sid]-d.geom_xpos[g])==pytest.approx(m.geom_size[g,0],abs=1e-8)


@pytest.mark.parametrize('door',('db0953_swing_single','db0727_swing_single'))
def test_guard_native_release_and_reengagement_twice(door):
    # Isolate the security mechanism from a separate closer; full-source
    # catalogue runs additionally retain each authored closer/effort profile.
    m,meta=native(door,isolated=True)
    result=run_security_mechanics_qa(m,meta)
    assert result['ok'],result
    phases=result['measurements'][0]['phases']
    assert sum(p['phase']=='released_open' and p['end_angle_rad']>.4 for p in phases)==2
    assert sum(p['phase']=='reengaged_load' and .005<p['max_angle_rad']<.2 for p in phases)==2


@pytest.mark.parametrize('door',('db0953_swing_single','db0727_swing_single'))
def test_native_gate_rejects_disabled_guard_contact(door):
    m,meta=native(door);head=m.geom(meta['security_guards'][0]['head_geom']).id
    m.geom_contype[head]=m.geom_conaffinity[head]=0
    r=run_security_mechanics_qa(m,meta)
    assert not r['ok'] and 'contact disabled' in r['failures'][0]


def test_native_gate_rejects_a_reintroduced_leaf_range_shortcut():
    m,meta=native('db0953_swing_single');j=m.joint(meta['primary_joint']).id;m.jnt_range[j,1]=.1
    r=run_security_mechanics_qa(m,meta)
    assert not r['ok'] and 'artificial security range' in r['failures'][0]


def test_native_gate_rejects_relaxed_keyhole_handoff_metadata():
    m,meta=native('db0953_swing_single')
    meta['security_guards'][0]['handoff_tolerances_m']['keyhole_lateral']=.006
    r=run_security_mechanics_qa(m,meta)
    assert not r['ok'] and 'physical slot/keyhole clearances' in r['failures'][0]


@pytest.mark.parametrize('setting',('integrator','timestep'))
def test_chain_proof_rejects_weakened_native_resolution(setting):
    m,meta=native('db0953_swing_single')
    if setting=='integrator':m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    else:m.opt.timestep=.00005
    r=run_security_mechanics_qa(m,meta)
    assert not r['ok']
    assert 'implicit integration' in r['failures'][0] or 'maximum timestep' in r['failures'][0]


def test_removing_the_bar_end_removes_actual_native_retention():
    m,meta=native('db0727_swing_single',isolated=True)
    for name in ('leaf_guard_closed_end','leaf_guard_finger_end'):
        g=m.geom(name).id;m.geom_contype[g]=m.geom_conaffinity[g]=0
    result=run_security_mechanics_qa(m,meta)
    assert not result['ok']
    assert 'Engaged guard does not allow-and-limit' in result['failures'][0]


def test_removing_chain_slot_lips_removes_actual_native_retention():
    m,meta=native('db0953_swing_single',isolated=True)
    row=meta['security_guards'][0]
    lips=[name for name in row['keeper_geoms'] if '_keeper_lip_' in name]
    assert len(lips)>=4
    for name in lips:
        g=m.geom(name).id;m.geom_contype[g]=m.geom_conaffinity[g]=0
    # Remove the inventory declaration too: this negative exercises native
    # load-path failure rather than merely the disabled-collider preflight.
    row['keeper_geoms']=[name for name in row['keeper_geoms'] if name not in lips]
    result=run_security_mechanics_qa(m,meta)
    assert not result['ok']
    assert 'Engaged guard does not allow-and-limit' in result['failures'][0]


def test_a_physically_closed_keyhole_fails_the_native_cycle():
    xml,assets,meta=fixture('db0953_swing_single',isolated=True)
    tree=ET.fromstring(xml)
    # Keep the declared metadata intact and close only the actual enlarged
    # opening to the 5mm slot width. An 8mm head cannot withdraw through it.
    # Recompile the altered source: changing MjModel.geom_size/pos in place
    # leaves geom_aabb/rbound and the collision hierarchy stale.
    for side in (-1,1):
        node=tree.find(f".//geom[@name='leaf_chain_keeper_lip_keyhole_{side}']")
        size=list(map(float,node.get('size').split()));pos=list(map(float,node.get('pos').split()))
        size[2]+=.00225;pos[2]-=side*.00225
        node.set('size',' '.join(map(str,size)));node.set('pos',' '.join(map(str,pos)))
    m=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'),assets)
    for side in (-1,1):
        g=m.geom(f'leaf_chain_keeper_lip_keyhole_{side}').id
        assert m.geom_aabb[g,5]==pytest.approx(m.geom_size[g,2])
    result=run_security_mechanics_qa(m,meta)
    assert not result['ok'],result
    assert any('Released guard still blocks' in s or 'penetration' in s or
               'missed keyhole' in s or 'did not reach retaining slot' in s
               for s in result['failures']),result


def test_passive_linkage_inertia_metadata_rejects_a_missing_joint(monkeypatch):
    from doorbench.geometry import security_guards as G
    original=G.add_chain_guard
    def broken(model,*args,**kwargs):
        result=original(model,*args,**kwargs)
        model.meta['physical_inertia_joints'].append('missing_chain_joint')
        return result
    monkeypatch.setattr(G,'add_chain_guard',broken)
    with pytest.raises(ValueError,match='distinct existing joints'):build_model(deepcopy(SPECS['db0953_swing_single']))


def test_passive_linkage_inertia_requires_actual_geometry(monkeypatch):
    from doorbench.geometry import security_guards as G
    original=G.add_chain_guard
    def broken(model,*args,**kwargs):
        result=original(model,*args,**kwargs)
        b=model.body('leaf_chain_tip_swivel');b.geoms=[];b.extra_mass=1.
        return result
    monkeypatch.setattr(G,'add_chain_guard',broken)
    with pytest.raises(ValueError,match='positive geometric mass'):build_model(deepcopy(SPECS['db0953_swing_single']))


def test_guard_builder_rejects_wrong_hardware_side():
    s=deepcopy(SPECS['db0953_swing_single']);s['robot']['is_push']=False
    with pytest.raises(ValueError,match='inward swing side'):build_model(s)


def test_terminal_swivel_has_real_clearance_through_all_three_axes():
    m,meta=native('db0953_swing_single');d=mujoco.MjData(m)
    axes=[m.joint(n).id for n in ('leaf_chain_tip_yaw','leaf_chain_tip_roll_hinge','leaf_chain_tip_pitch')]
    ball=m.geom('leaf_chain_tip_eye').id
    ring=[m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith('leaf_chain_tip_roll_ring_')]
    walls=[m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith('leaf_chain_finger_grip_')]
    # These direct distances remain authoritative even where native
    # parent-child contact filtering would hide an intersecting bearing.
    from itertools import product
    for angles in product((-.9,0.,.9),repeat=3):
        d.qpos[:]=m.qpos0
        for j,q in zip(axes,angles):d.qpos[m.jnt_qposadr[j]]=q
        mujoco.mj_kinematics(m,d)
        assert min(mujoco.mj_geomDistance(m,d,ball,g,.01,None) for g in ring)>.00014
        assert min(mujoco.mj_geomDistance(m,d,a,b,.01,None) for a in ring for b in walls)>.00007


def test_service_wrapper_owns_private_fields_and_preserves_caller_callback(monkeypatch):
    import doorbench.security_mechanics_qa as qa
    import doorbench.closer_pinion as pinion
    import doorbench.closer_track_hold as track
    m,meta=native('db0159_automatic_swing');before=qa._model_sha256(m);saved=deepcopy(meta)
    previous=mujoco.get_mjcb_passive();caller=[];own=[]
    def old_callback(native,data):caller.append(native)
    monkeypatch.setattr(pinion,'compile_pinion_closers',lambda native,metadata:('pinion',))
    monkeypatch.setattr(track,'compile_track_holds',lambda native,metadata:('track',))
    monkeypatch.setattr(pinion,'apply_pinion_closers',lambda native,data,rules:own.append(('pinion',native,rules)))
    monkeypatch.setattr(track,'apply_track_holds',lambda native,data,rules:own.append(('track',native,rules)))
    def inspect(private,metadata,**kwargs):
        assert private is not m and metadata is not meta
        assert not np.shares_memory(private.actuator_gainprm,m.actuator_gainprm)
        assert not np.any(private.actuator_gainprm) and not np.any(private.actuator_biasprm)
        assert np.any(m.actuator_gainprm)
        mujoco.mj_forward(private,mujoco.MjData(private))
        assert not caller and [r[0] for r in own]==['pinion','track']
        mujoco.mj_forward(m,mujoco.MjData(m))
        assert caller==[m] and len(own)==2
        private.geom_size[0]*=1.1;metadata['private_only']=True
        return {'ok':True,'applicable':True,'failures':[]}
    monkeypatch.setattr(qa,'run_security_mechanics_qa',inspect)
    try:
        mujoco.set_mjcb_passive(old_callback)
        result=qa.run_security_service_qa(m,meta)
        assert mujoco.get_mjcb_passive() is old_callback
    finally:mujoco.set_mjcb_passive(previous)
    assert result['service_fixture']['source_model_unchanged']
    assert result['service_fixture']['source_model_mjb_sha256_before']==before==qa._model_sha256(m)
    assert meta==saved


def test_service_wrapper_restores_callback_after_failure(monkeypatch):
    import doorbench.security_mechanics_qa as qa
    m,meta=native('db0159_automatic_swing');before=qa._model_sha256(m)
    previous=mujoco.get_mjcb_passive()
    def old_callback(native,data):pass
    def broken(private,metadata,**kwargs):
        private.dof_damping[:]=99.
        raise RuntimeError('deliberate fixture failure')
    monkeypatch.setattr(qa,'run_security_mechanics_qa',broken)
    try:
        mujoco.set_mjcb_passive(old_callback)
        with pytest.raises(RuntimeError,match='deliberate fixture failure'):
            qa.run_security_service_qa(m,meta)
        assert mujoco.get_mjcb_passive() is old_callback
        assert qa._model_sha256(m)==before
    finally:mujoco.set_mjcb_passive(previous)

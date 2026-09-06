"""Physical turnstile contact and production integration controls."""
import copy,json
from pathlib import Path
import mujoco,numpy as np,pytest
from doorbench.spec import generate_all
from doorbench.build import export_door
from doorbench.turnstile_locks import compile_turnstile_locks,apply_turnstile_locks
from doorbench.turnstile_lock_qa import run_turnstile_lock_qa,run_turnstile_mount_qa

@pytest.fixture(scope='module')
def turnstiles(tmp_path_factory):
    root=tmp_path_factory.mktemp('native-turnstiles');rows=[]
    for spec in generate_all():
        if spec['family'] not in ('turnstile_tripod','turnstile_fullheight'):continue
        ex=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'));path=Path(ex['files']['mjcf']['full']);meta=json.loads(path.with_name('model.json').read_text())['meta']
        assert meta['turnstile_locks']['schema']=='doorbench.turnstile-lock.v1'
        rows.append((spec,path,meta))
    assert len(rows)==20
    return rows


def test_all20_turnstiles_actual_arrest_release_and_direction_in_every_tier(turnstiles):
    receipts=[]
    for spec,path,meta in turnstiles:
        for name in ('door.xml','door_simple.xml','door_minimal.xml'):
            m=mujoco.MjModel.from_xml_path(str(path.with_name(name)));r=run_turnstile_lock_qa(m,meta)
            receipts.append({'door_id':spec['id'],'file':str(path.with_name(name)),'report':r})
            (path.parents[2]/'native-proof.json').write_text(json.dumps(receipts,indent=2))
    assert all(row['report']['ok'] for row in receipts),[(r['door_id'],r['file'],r['report']['failures']) for r in receipts if not r['report']['ok']]


def test_missing_pawl_or_bolt_is_not_certified(turnstiles):
    _,path,meta=next(r for r in turnstiles if r[0]['id']=='db0272_turnstile_tripod')
    for part in ('pawl','bolt'):
        m=mujoco.MjModel.from_xml_path(str(path));row=meta['turnstile_locks']
        if part=='pawl':names=[row['pawl_tip_geom']]
        else:names=[row['bolt_geom']]
        for name in names:
            g=m.geom(name).id;m.geom_contype[g]=0;m.geom_conaffinity[g]=0
        r=run_turnstile_lock_qa(m,meta)
        assert not r['ok'],part
        assert any('physical_arrest_failed' in f or 'missing_reverse_load_path' in f or 'missing_index_load_contact' in f for f in r['failures'])


def test_coil_only_pushes_actual_armature_and_preserves_native_state(turnstiles):
    _,path,meta=turnstiles[0];m=mujoco.MjModel.from_xml_path(str(path));rules=compile_turnstile_locks(m,meta)
    d=mujoco.MjData(m);initial={name:getattr(d,name).copy() for name in ('qpos','qvel','qacc','qfrc_applied','ctrl')};limits=m.jnt_limited.copy();ranges=m.jnt_range.copy()
    apply_turnstile_locks(m,d,rules,True)
    nonzero=np.flatnonzero(d.qfrc_passive);assert nonzero.tolist()==[rules[0].dof]
    assert 0<d.qfrc_passive[rules[0].dof]<rules[0].force
    for name,value in initial.items():assert np.array_equal(getattr(d,name),value)
    assert np.array_equal(m.jnt_limited,limits);assert np.array_equal(m.jnt_range,ranges)
    before=d.qfrc_passive.copy();apply_turnstile_locks(m,d,rules,False);assert np.array_equal(before,d.qfrc_passive)
    bad=copy.deepcopy(meta);bad['turnstile_locks']['stroke_m']=.03
    with pytest.raises(ValueError):compile_turnstile_locks(m,bad)


def test_disconnected_coil_end_cap_cannot_pass_mount_gate(turnstiles):
    from doorbench.ir import quat_to_mat
    for _,path,meta in turnstiles:
        m=mujoco.MjModel.from_xml_path(str(path));assert run_turnstile_mount_qa(m,meta)['ok']
        # The drop variants rotate the complete coil assembly. Move the cap
        # outward along its authored axis, not into it along a fixed world X.
        outward=quat_to_mat(meta['turnstile_locks']['mechanism_frame_quat'])[:,0]
        m.geom_pos[m.geom('turnstile_coil_end').id]+=.010*outward
        report=run_turnstile_mount_qa(m,meta)
        assert not report['ok']
        assert 'turnstile_coil_end' in report['failures'][0]['detached_supports']


def test_fixed_housing_excluded_and_physical_parts_keep_mass_and_world_frames(turnstiles):
    from doorbench.ir import quat_to_mat
    for spec,path,meta in turnstiles:
        ir=json.loads(path.with_name('model.json').read_text());bodies={b['name']:b for b in ir['bodies']}
        frame=bodies['turnstile_mechanism_frame'];assert frame['static']
        expected={name:mass for row in meta['mass_reconciliation']['panels'] for name,mass in row['geometry_backed_bodies_kg'].items()}
        assert 'turnstile_mechanism_frame' not in expected
        for filename in ('door.xml','door_simple.xml','door_minimal.xml'):
            m=mujoco.MjModel.from_xml_path(str(path.with_name(filename)));d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
            for name in ('turnstile_ratchet_wheel','turnstile_credential_bolt')+ (('turnstile_reverse_pawl',) if meta['turnstile_locks']['one_way'] else ()):
                assert m.body_mass[m.body(name).id]==pytest.approx(expected[name],abs=1e-6)
            rotation=quat_to_mat(meta['turnstile_locks']['mechanism_frame_quat']);origin=np.asarray(meta['turnstile_locks']['mechanism_frame_pos'])
            for name,local in [('turnstile_credential_bolt',(.081,0,.025))]+([('turnstile_reverse_pawl',(-.060,-.085,0))] if meta['turnstile_locks']['one_way'] else []):
                assert np.linalg.norm(d.xpos[m.body(name).id]-(origin+rotation@local))<2e-6


def test_contact_inspection_seats_actual_pawl_without_changing_inputs(turnstiles):
    from doorbench.turnstile_contact_preview import TurnstileContactPreview
    for fixture in ('db0187_turnstile_fullheight','db0272_turnstile_tripod'):
        _,path,meta=next(r for r in turnstiles if r[0]['id']==fixture)
        m=mujoco.MjModel.from_xml_path(str(path));preview=TurnstileContactPreview(m,meta)
        before=m.body_pos.copy();q=m.qpos0.copy();q[m.joint(meta['turnstile_locks']['bolt_joint']).qposadr[0]]=.022
        pawl=m.joint(meta['turnstile_locks']['pawl_joint']).qposadr[0]
        for angle in np.linspace(0,2*np.pi/36,9):
            q[m.joint('rotor_hinge').qposadr[0]]=angle;original=q.copy();report=preview.resolve(q)
            assert report['ok'],(fixture,angle,report)
            assert report['snapshot_max_geometry_error_m']<2e-6
            assert np.array_equal(original,q);assert np.array_equal(before,m.body_pos)
            expected=q.copy();expected[pawl]=report['qpos'][pawl]
            assert np.array_equal(expected,report['qpos'])


def test_default_native_coil_position_is_verified_before_unlocked_sweep(turnstiles):
    from doorbench.turnstile_contact_preview import TurnstileContactPreview
    for spec,path,meta in turnstiles:
        m=mujoco.MjModel.from_xml_path(str(path));q=m.qpos0.copy();original=q.copy()
        result=TurnstileContactPreview(m,meta).default_qpos(q)
        assert result['ok'],(spec['id'],result)
        stroke=result['qpos'][m.joint(meta['turnstile_locks']['bolt_joint']).qposadr[0]]
        assert stroke >= .0215 if meta['turnstile_locks']['powered_by_default'] else abs(stroke)<.0001
        assert np.array_equal(q,original)

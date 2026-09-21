"""Synthetic CPU evidence only: these fixtures never run or qualify PhysX."""
import copy
import gzip
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from doorbench.dexterous import isaac_standing_continuation_audit as audit
from doorbench.dexterous.continuation_record_stream import iter_continuation_records
from doorbench.dexterous.isaac_opening_measurements import contact_force_pairs, panel_surface_loads
from doorbench.dexterous.isaac_standing_continuation_measurements import pack_standing_continuation
from doorbench.dexterous.npz_record_stream import iter_npz_records
from doorbench.dexterous.qualified_isaac_grasp import digest
from doorbench.dexterous.standing_body_record import PLANNER_BODIES, POSE_CONVENTION


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def gzwrite(path, records):
    with gzip.open(path, 'wt') as stream: json.dump(records, stream, allow_nan=False)


def capture_inputs():
    """Normal slot 0 and independent friction slot 0 intentionally coexist."""
    paths = ['/World/H1/'+name for name in audit.BODY_NAMES[:4]]
    scene = sorted(paths+['/World/floor', '/World/Door/Articulation/leaf', '/World/Door/Articulation/leaf_handle'])
    filters = [scene[:] for _ in paths]; shape = (4, len(scene)); capacity = 16384
    force = np.full((capacity, 1), np.nan); points = np.full((capacity, 3), np.nan)
    normals = points.copy(); gaps = force.copy(); counts = np.zeros(shape, int); starts = counts.copy()
    matrix = np.zeros((*shape, 3))
    for i, (other, load, normal, gap) in enumerate([
            ('/World/floor', 200., [0, 0, 1], -.0001), ('/World/floor', 201., [0, 0, 1], -.0001),
            ('/World/Door/Articulation/leaf', 4., [0, -1, 0], -.0001),
            ('/World/Door/Articulation/leaf_handle', 0., [0, -1, 0], .002)]):
        j = scene.index(other); counts[i, j] = 1; starts[i, j] = i
        force[i] = load; points[i] = [i, 0, 0]; normals[i] = normal; gaps[i] = gap
        matrix[i, j] = load*np.array(normal)
    fc = np.zeros(shape, int); fs = fc.copy(); fc[2, scene.index('/World/Door/Articulation/leaf')] = 1
    vectors = np.full((capacity, 3), np.nan); fpoints = vectors.copy()
    vectors[0] = [.5, 0., 0.]; fpoints[0] = [2., 0., 0.]
    return dict(time_s=.002, pose_time_s=.002, sensor_paths=paths, filter_paths=filters,
        normal_matrix=matrix, normal_buffers=(force, points, normals, gaps, counts, starts),
        friction_buffers=(vectors, fpoints, fc, fs), capacity=capacity, physics_qualified=True,
        body_poses={name:np.array([float(i), 0., 0., 1., 0., 0., 0.], np.float32)
                    for i, name in enumerate(audit.BODY_NAMES)})


def capture_contract(inputs):
    return dict(schema='doorbench.isaac-standing-continuation-capture.v1',
        sensor_paths=inputs['sensor_paths'], filter_paths=inputs['filter_paths'], capacity=16384,
        body_pose_order=list(audit.BODY_NAMES), dt_s=.002,
        clock='completed PhysX interval, global episode time',
        normal_and_friction_slots_are_independent=True, authorized_stages=0)


def fixture_scene(path, inputs):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateNew(str(path))
    for body_path in inputs['filter_paths'][0]:
        if body_path == '/World/floor':
            prim = UsdGeom.Cube.Define(stage, body_path).GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
        else:
            body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            prim = UsdGeom.Cube.Define(stage, body_path+'/collision').GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
    stage.GetRootLayer().Save()


@pytest.fixture
def source(tmp_path):
    trial = tmp_path/'synthetic source'/'trial'; trial.mkdir(parents=True)
    inputs = capture_inputs(); contract = capture_contract(inputs)
    fixture_scene(trial/'scene.usda', inputs)
    names = ['j'+str(i) for i in range(69)]
    robot = tmp_path/'robot.xml'; robot.write_text('Synthetic binding only; no physical model')
    motors = dict(source_xml_sha256=digest(robot), passive={name:dict(damping=.1 if i<61 else 0.,
        friction=.03 if i<61 else 0., armature=.001, stiffness=0., springref=0.) for i,name in enumerate(names)},
        actuators=[dict(name='motor'+str(i), terms={names[i]:1.}, force_range=[-10.,10.]) for i in range(61)])
    original_motors = tmp_path/'motors.json'; write(original_motors, motors)
    robot_usd = tmp_path/'robot.usda'; robot_usd.write_text('#usda 1.0\n')
    door_usd = tmp_path/'door.usda'; door_usd.write_text('#usda 1.0\n')
    provenance = dict(files={str(p):digest(p) for p in (robot, original_motors, robot_usd, door_usd)})
    for name in audit.CAPTURE_SOURCES:
        original = tmp_path/name; original.write_text('# Synthetic archived source fixture\n')
        (trial/('source-'+name)).write_bytes(original.read_bytes())
        provenance['files'][str(original)] = digest(original)
    config = dict(args=dict(acquisition=True, operate_after_acquisition=True,
        acquisition_stance_profile='landed-foot-v1', standing_withdrawal_route='synthetic-not-a-route',
        joint_passive_profile='legacy-tanh-v1', motors=str(original_motors), native_robot=str(robot),
        robot_usd=str(robot_usd), door_usd=str(door_usd)), dt=.002,
        runtime_pose_writes=0, direct_door_commands=False, robot_joint_names=names,
        door_joint_names=['leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide'],
        root_state_convention='actor-origin pose and world actor-origin linear/angular velocity',
        standing_continuation_body_names=list(audit.BODY_NAMES),
        standing_planner_body_names=list(PLANNER_BODIES), standing_planner_body_pose_convention=POSE_CONVENTION,
        standing_leaf_pose_convention=POSE_CONVENTION)
    readback = dict(api='ArticulationView.get_dof_actuation_forces', semantics=audit.SUBMISSION_SEMANTICS,
        root_controller_field='root_link_state_w', legacy_diagnostic_field='legacy_root_state_w',
        initial_interval_s=[0.,0.], initial_motor_input=[0.]*61, root_com_offset_in_actor_m=[0.,0.,0.],
        archive_fields=dict(actual_joint_effort='Legacy field name: backend submitted generalized input',
            actual_motor_forces='Legacy field name: motor-equivalent reconstruction of submitted input'))
    arrays = {name:np.zeros((3,*shape), dtype=np.float32) for name,shape in audit.SHAPES.items()}
    arrays['time_s'] = np.arange(1,4)*.002
    arrays['motor_forces'] = np.zeros((3,61), dtype=float)
    arrays['actual_motor_forces'] = np.zeros((3,61), dtype=float)
    arrays['actual_foot_loads'] = np.tile([200.,201.], (3,1))
    arrays['root'][:,3] = arrays['legacy_root_state_w'][:,3] = 1.
    arrays['standing_body_poses'][:,:,3] = 1.
    arrays['joint_velocity'][:,0] = np.array([.00111123,.00234561,.003781], np.float32)
    arrays['pre_step_joint_velocity'][:,0] = np.r_[.000731, arrays['joint_velocity'][:-1,0]]
    arrays['motor_forces'][:,0] = [.1,.2,.3]
    matrix = np.eye(61,69); damp=np.r_[np.full(61,.1), np.zeros(8)]; fric=np.r_[np.full(61,.03),np.zeros(8)]
    records = []; surfaces=[]
    for i in range(3):
        velocity = arrays['pre_step_joint_velocity'][i]
        arrays['actual_joint_effort'][i] = matrix.T@arrays['motor_forces'][i]-damp*velocity-fric*np.tanh(velocity/.001)
        v64 = velocity.astype(float)
        arrays['actual_motor_forces'][i] = matrix@(arrays['actual_joint_effort'][i].astype(float)+damp*v64+fric*np.tanh(v64/.001))
        inputs['time_s'] = inputs['pose_time_s'] = (i+1)*.002
        row = pack_standing_continuation(**inputs); records.append(row)
        poses=np.asarray(list(row['body_poses'].values())); arrays['continuation_body_poses'][i]=poses
        for ni,oi in ((0,0),(1,1),(2,4),(3,3),(5,5)): arrays['standing_body_poses'][i,oi]=poses[ni]
        arrays['standing_leaf_pose'][i]=poses[4]
        pairs=contact_force_pairs(inputs['normal_matrix'], inputs['friction_buffers'][0],
            *inputs['friction_buffers'][2:],capacity=16384)
        surfaces.append(dict(time_s=(i+1)*.002,leaf_pose=poses[4].tolist(), surface=panel_surface_loads(
            inputs['sensor_paths'], inputs['filter_paths'], pairs, poses[4])))
    report = dict(passed=False, physics_dt_s=.002, duration_s=.006, standing_continuation_capture=dict(
        observations=3, body_pose_order=list(audit.BODY_NAMES), archive='standing-continuation-steps.json.gz',
        maximum_motor_input_reconstruction_residual_Nm=0., authorized_stages=0))
    for name,value in [('configuration',config),('provenance',provenance),('operation-report',report),
            ('motor-contract',motors),('motor-readback-contract',readback),
            ('joint-passive-profile',dict(profile='legacy-tanh-v1')),('standing-continuation-contract',contract)]:
        write(trial/(name+'.json'),value)
    result=dict(trial=trial,arrays=arrays,records=records,surfaces=surfaces,contract=contract,
        config=config,provenance=provenance,report=report,motors=motors,readback=readback)
    save_streams(result)
    return result


def save_streams(source):
    trial=source['trial']; np.savez_compressed(trial/'acquisition-physics.npz',**source['arrays'])
    gzwrite(trial/'standing-continuation-steps.json.gz',source['records'])
    gzwrite(trial/'standing-transfer-steps.json.gz',source['surfaces'])


def test_fresh_full_accounting_does_not_invent_task_success(source):
    result=audit.audit_standing_continuation(source['trial'])
    assert result['passed'] and result['accounting_passed'] and result['physical_intervals']==3
    assert not result['source_report_passed'] and not result['physical_task_qualification']
    assert result['authorized_stages']==0 and not result['first_pre_step_velocity_independently_crosschecked']
    assert result['normal_patches_checked']==12 and result['friction_patches_checked']==3
    assert result['endpoint']['hand_forces_world_N']['/World/H1/lh_palm']==[.5,-4.,0.]
    assert result['endpoint']['body_poses']['leaf'][0]==4.
    assert result['maxima']['submitted_generalized_input_error_Nm']<1e-4
    assert result['maxima']['archived_motor_reconstruction_error_Nm']==0.


def test_admission_requires_exact_fresh_receipt_and_separate_epoch_binding(source):
    result=audit.audit_standing_continuation(source['trial']); path=source['trial']/'audit.json';write(path,result)
    admitted=audit.admit_standing_continuation_observations(source['trial'],path,.006,result['source_physics_sha256'])
    assert admitted['requires_separate_physical_source_qualification'] and admitted['authorized_stages']==0
    assert admitted['input_sha256'][str(path)]==digest(path)
    for epoch,sha in ((.004,result['source_physics_sha256']),(.006,'a'*64),(True,result['source_physics_sha256'])):
        with pytest.raises(ValueError):audit.admit_standing_continuation_observations(source['trial'],path,epoch,sha)
    result['endpoint']['foot_loads_N'][0]+=1.;write(path,result)
    with pytest.raises(ValueError):audit.admit_standing_continuation_observations(source['trial'],path,.006,result['source_physics_sha256'])


@pytest.mark.parametrize('mutation',['foot','hand','evidence','normal_matrix','normal_vector','normal_negative','normal_slots',
    'normal_duplicate','friction_slots','pose_time','interval','pose_archive','leaf_archive','body_archive','pre_velocity',
    'motor_reconstruction','clock','extra_record','missing_record','surface','surface_epoch','raw_missing','authority'])
def test_corrupt_streams_cannot_pass(source,mutation):
    row=source['records'][1]; arrays=source['arrays']
    if mutation=='foot':row['foot_loads_N'][0]+=1
    elif mutation=='hand':row['hand_forces_world_N']['/World/H1/lh_palm'][0]+=1
    elif mutation=='evidence':row['evidence']['physics_qualified']=False
    elif mutation=='normal_matrix':row['raw']['normal_force_pairs'][0][-1]+=.0011
    elif mutation=='normal_vector':row['raw']['normal']['normal_world'][0]=[0,0,2]
    elif mutation=='normal_negative':row['raw']['normal']['force_N'][0]=[-1.]
    elif mutation=='normal_slots':row['raw']['normal']['slots'][0]=7
    elif mutation=='normal_duplicate':row['raw']['normal']['pairs'][1][2]=0
    elif mutation=='friction_slots':row['raw']['friction']['slots'][0]=3
    elif mutation=='pose_time':row['pose_time_s']-=.002
    elif mutation=='interval':row['contact_interval_s'][0]-=.002
    elif mutation=='pose_archive':row['body_poses']['lh_palm'][0]+=.001
    elif mutation=='leaf_archive':arrays['standing_leaf_pose'][1,0]+=.001
    elif mutation=='body_archive':arrays['standing_body_poses'][1,4,0]+=.001
    elif mutation=='pre_velocity':arrays['pre_step_joint_velocity'][1,0]+=.001
    elif mutation=='motor_reconstruction':arrays['actual_motor_forces'][1,0]+=.001
    elif mutation=='clock':arrays['time_s'][1]=.003
    elif mutation=='extra_record':source['records'].append(copy.deepcopy(row))
    elif mutation=='missing_record':source['records'].pop()
    elif mutation=='surface':source['surfaces'][1]['surface']['palm_normal_load_N']+=1
    elif mutation=='surface_epoch':source['surfaces'][1]['time_s']-=.002
    elif mutation=='raw_missing':row['raw']['normal'].pop('point_world')
    else:row['authorized_stages']=1
    save_streams(source)
    with pytest.raises(ValueError):audit.audit_standing_continuation(source['trial'])


@pytest.mark.parametrize('mutation',['sensor_omission','filter_omission','duplicate_filter','capture_source','asset','source_motor',
    'body_order','missing_field','short_array','checkpoint_only','passive_mismatch','bad_semantics','reported_count'])
def test_incomplete_source_contract_rejected(source,mutation):
    trial=source['trial']
    if mutation.startswith(('sensor','filter','duplicate')):
        c=source['contract']
        if mutation=='sensor_omission':c['sensor_paths'].pop();c['filter_paths'].pop()
        elif mutation=='filter_omission':c['filter_paths'][0].pop()
        else:c['filter_paths'][0][0]=c['filter_paths'][0][1]
        write(trial/'standing-continuation-contract.json',c)
    elif mutation=='capture_source':(trial/'source-isaac_standing_continuation_measurements.py').write_text('# changed')
    elif mutation=='asset':Path(source['config']['args']['native_robot']).write_text('changed')
    elif mutation=='source_motor':source['motors']['actuators'][0]['force_range']=[-20.,20.];write(trial/'motor-contract.json',source['motors'])
    elif mutation=='body_order':source['config']['standing_continuation_body_names'].reverse();write(trial/'configuration.json',source['config'])
    elif mutation=='missing_field':source['arrays'].pop('pre_step_joint_velocity');save_streams(source)
    elif mutation=='short_array':source['arrays']['actual_foot_loads']=source['arrays']['actual_foot_loads'][:2];save_streams(source)
    elif mutation=='checkpoint_only':(trial/'standing-continuation-steps.json.gz').unlink()
    elif mutation=='passive_mismatch':write(trial/'joint-passive-profile.json',dict(profile='backend-dry-v2'))
    elif mutation=='bad_semantics':source['readback']['semantics']='delivered torque';write(trial/'motor-readback-contract.json',source['readback'])
    else:source['report']['standing_continuation_capture']['observations']=2;write(trial/'operation-report.json',source['report'])
    with pytest.raises((ValueError,FileNotFoundError)):audit.audit_standing_continuation(trial)


def test_correctly_captured_failed_submission_stays_failed(source):
    source['arrays']['actual_joint_effort'][1,68]=.001
    # Reconstruct the producer's explicit failed-input fallback at this interval.
    v=source['arrays']['pre_step_joint_velocity'][1];damp=np.r_[np.full(61,.1),np.zeros(8)];fric=np.r_[np.full(61,.03),np.zeros(8)]
    source['arrays']['actual_motor_forces'][1]=np.eye(61,69)@(source['arrays']['actual_joint_effort'][1]+damp*v+fric*np.tanh(v/.001))
    for row in source['records'][1:]:row['evidence']['physics_qualified']=False
    save_streams(source); result=audit.audit_standing_continuation(source['trial'])
    assert result['accounting_passed'] and not result['passed']
    assert result['invalid_submission_intervals']==1 and result['first_invalid_submission_time_s']==.004
    assert not result['endpoint']['evidence']['physics_qualified']


def test_audit_hashes_inputs_before_and_after_streaming(source,monkeypatch):
    original=audit.iter_npz_records
    def changed(*args,**kwargs):
        yield from original(*args,**kwargs)
        with (source['trial']/'configuration.json').open('a') as stream:stream.write('\n')
    monkeypatch.setattr(audit,'iter_npz_records',changed)
    with pytest.raises(ValueError,match='changed during'):audit.audit_standing_continuation(source['trial'])


def test_historical_python_is_captured_not_current_code(source):
    original=next(Path(p) for p in source['provenance']['files'] if p.endswith('isaac_opening.py'))
    original.write_text('# changed current checkout is not historic evidence')
    assert audit.audit_standing_continuation(source['trial'])['passed']


@pytest.mark.parametrize('bad',['overlap','truncated','duplicate_pair','float_index','missing_value','bad_shape','bad_order'])
def test_sparse_decoder_rejects_corrupt_inventories(bad):
    row=pack_standing_continuation(**capture_inputs());raw=row['raw']['normal']
    if bad=='overlap':raw['pairs'][1][2]=0
    elif bad=='truncated':raw['pairs'][0][3]=16384
    elif bad=='duplicate_pair':raw['pairs'][1][:2]=raw['pairs'][0][:2]
    elif bad=='float_index':raw['pairs'][0][0]=0.
    elif bad=='missing_value':raw['force_N'].pop()
    elif bad=='bad_shape':raw['shape']=[4,1]
    else:raw['pairs'].reverse()
    with pytest.raises(ValueError):audit.decode_sparse_buffer(raw,(4,7),16384,{'force_N':1,'point_world':3,'normal_world':3,'distance_m':1})


@pytest.mark.parametrize('bad',['duplicate','nested_duplicate','nan','overflow','truncated','trailing','nonobject','too_large','comma'])
def test_strict_json_stream_rejects_corruption(bad):
    data={'duplicate':'[{"a":1,"a":2}]','nested_duplicate':'[{"a":{"b":1,"b":2}}]',
        'nan':'[{"a":NaN}]','overflow':'[{"a":1e309}]','truncated':'[{"a":1}',
        'trailing':'[{}] {}','nonobject':'[1]','too_large':'[{"a":"abcdefghijk"}]','comma':'[{},]'}[bad]
    with pytest.raises(ValueError):list(iter_continuation_records(io.StringIO(data),chunk_size=2,max_record_chars=16 if bad=='too_large' else 100))


def test_strict_stream_handles_chunk_boundaries():
    assert list(iter_continuation_records(io.StringIO(' \n[{"a":1}, {"b":[2,3]}] \n'),chunk_size=1))==[{'a':1},{'b':[2,3]}]
    assert list(iter_continuation_records(io.StringIO('[]'),chunk_size=1))==[]


@pytest.mark.parametrize('bad',['missing','nonfinite','object','fortran','short','duplicate','trailing'])
def test_bounded_npz_decoder_rejects_bad_member(tmp_path,bad):
    path=tmp_path/'rows.npz';a=np.arange(12,dtype=np.float32).reshape(3,4)
    if bad=='missing':np.savez(path,b=a)
    elif bad=='nonfinite':a[0,0]=np.nan;np.savez(path,a=a)
    elif bad=='object':np.savez(path,a=a.astype(object))
    elif bad=='fortran':np.savez(path,a=np.asfortranarray(a))
    elif bad=='short':np.savez(path,a=a[:2])
    else:
        raw=io.BytesIO();np.save(raw,a);data=raw.getvalue()
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('a.npy',data+(b'x' if bad=='trailing' else b''))
            if bad=='duplicate':
                with pytest.warns(UserWarning):archive.writestr('a.npy',data)
    with pytest.raises(ValueError):list(iter_npz_records(path,{'a':(4,)},expected_rows=3,block_rows=1))


def test_npz_reader_retains_actual_dtype_and_detaches_rows(tmp_path):
    path=tmp_path/'rows.npz';np.savez_compressed(path,a=np.arange(12,dtype=np.float32).reshape(3,4),t=np.arange(3,dtype=float))
    rows=list(iter_npz_records(path,{'a':(4,),'t':()},expected_rows=3,block_rows=2))
    assert rows[0]['a'].dtype==np.float32 and rows[0]['t'].dtype==np.float64
    rows[0]['a'][0]=999.;assert rows[1]['a'][0]==4.


def test_original_normal_matrix_tolerance_is_preserved():
    inputs=capture_inputs();inputs['normal_matrix'][0,inputs['filter_paths'][0].index('/World/floor'),2]+=.0008
    row=pack_standing_continuation(**inputs)
    result=audit.decode_continuation_row(row,capture_contract(inputs),time_s=.002,submission_valid=True)
    assert .00079<result['maximum_normal_matrix_error_N']<.00081


@pytest.mark.parametrize('field',['force','pose','time','foot'])
def test_numeric_strings_do_not_become_actual_readbacks(field):
    inputs=capture_inputs();row=pack_standing_continuation(**inputs)
    if field=='force':row['raw']['normal']['force_N'][0][0]='200'
    elif field=='pose':row['body_poses']['leaf'][0]='4'
    elif field=='time':row['time_s']='0.002'
    else:row['foot_loads_N'][0]='200'
    with pytest.raises(ValueError):audit.decode_continuation_row(row,capture_contract(inputs),time_s=.002,submission_valid=True)


def test_motor_model_uses_actual_recorded_joint_order_and_full_rank(source):
    names=source['config']['robot_joint_names'][::-1]
    matrix,inverse,caps,declaration=audit._motor_model(source['motors'],names,dict(profile='legacy-tanh-v1'),source['readback'])
    assert matrix[0,68]==1. and matrix[0,0]==0. and np.array_equal(matrix[:,::-1],np.eye(61,69))
    assert np.allclose(inverse@matrix.T,np.eye(61),atol=1e-12,rtol=0)
    source['motors']['actuators'][1]['terms']=source['motors']['actuators'][0]['terms'].copy()
    with pytest.raises(ValueError,match='rank'):audit._motor_model(source['motors'],names,dict(profile='legacy-tanh-v1'),source['readback'])


def test_cli_receipt_freshly_admits_and_never_overwrites(source):
    from scripts.dexterous.audit_isaac_standing_continuation import main
    path=source['trial']/'new-audit.json'
    assert main(['--trial',str(source['trial']),'--output',str(path)])==0
    result=json.loads(path.read_text())
    assert audit.admit_standing_continuation_observations(source['trial'],path,.006,result['source_physics_sha256'])['passed']
    before=path.read_bytes()
    with pytest.raises(ValueError):main(['--trial',str(source['trial']),'--output',str(path)])
    assert path.read_bytes()==before


def test_cli_preserves_malformed_evidence_and_failed_receipt(source):
    from scripts.dexterous.audit_isaac_standing_continuation import main
    broken=source['trial']/'standing-continuation-steps.json.gz';broken.write_bytes(b'corrupt gzip')
    output=source['trial']/'failed-audit.json'
    assert main(['--trial',str(source['trial']),'--output',str(output)])==1
    result=json.loads(output.read_text())
    assert not result['passed'] and not result['accounting_passed'] and result['authorized_stages']==0
    assert broken.read_bytes()==b'corrupt gzip'

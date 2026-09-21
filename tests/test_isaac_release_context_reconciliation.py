"""Synthetic reconciliation tests; no source admission or physics is fabricated.

The four changed values below reproduce the saved Kit/CPU comparison exactly.
The fixture's surrounding source is explicitly synthetic. A separate optional
local test consumes the complete frozen actual receipts read-only.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from doorbench.dexterous.isaac_release_planning import IsaacReleasePlanningContext


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def context(value):
    return IsaacReleasePlanningContext(encode(value))


def reconcile(candidate, fresh):
    from doorbench.dexterous.isaac_release_context_reconciliation import reconcile_isaac_release_context
    return reconcile_isaac_release_context(candidate, fresh)


def update_measured_hash(admission):
    body=admission['coordinate_admission']['measured_body_admission']
    body['measured_sha256']=sha(body['measured'])


def candidate_for(admission):
    return dict(schema='doorbench.isaac-profiled-release-candidate.v1', source_engine='isaac-physx',
        source_admission=copy.deepcopy(admission), source_context_sha256=sha(admission),
        initial_qpos=copy.deepcopy(admission['initial_qpos']),
        initial_time_s=admission['measured_rest']['terminal_time_s'],
        grasp_profile=admission['grasp_profile'], input_sha256=copy.deepcopy(admission['input_sha256']))


def make_model(scalars_before_root=3):
    # The free-root angular address is model-derived, deliberately not [3:6].
    scalar=''.join('<body><joint type="slide"/><geom size=".01"/></body>' for _ in range(scalars_before_root))
    return mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+scalar+
        '<body name="robot"><freejoint/><geom size=".1"/><body><joint/><geom size=".01"/></body></body>'+
        '</worldbody></mujoco>')


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    model=make_model();free=int(np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0])
    angular=list(range(int(model.jnt_dofadr[free])+3,int(model.jnt_dofadr[free])+6))
    original=tmp_path/'synthetic-source-evidence.txt';original.write_text('Explicit synthetic receipt fixture; not a physics episode.')
    inputs={str(original):hashlib.sha256(original.read_bytes()).hexdigest()}
    measured=dict(qpos=model.qpos0.tolist(),qvel=np.zeros(model.nv).tolist(),time_s=42.548,
        geometry_time_s=42.548,body_names=['synthetic-body'],body_poses_xyz_wxyz=[[0.,0.,0.,1.,0.,0.,0.]])
    measured['qvel'][angular[-1]]=0.014784089070371823
    body=dict(schema='doorbench.destination-return-kinematics.v1',profile='isaac-float32-2um-2urad-v1',
        passed=True,position_limit_m=2e-6,rotation_limit_rad=2e-6,quaternion_normalization_limit=2e-6,
        maximum_position_error_m=6.060591796429764e-7,maximum_rotation_error_rad=1.60749524522936e-6,
        maximum_root_quaternion_normalization_change=6.516244654974912e-9,
        measured=measured,measured_sha256=sha(measured),simulation_steps=0,active_plant_writes=0,
        reconstructed_positions_m=[[0.,0.,0.]],reconstructed_rotations=[np.eye(3).tolist()])
    state=dict(passed=True,state_sha256='a'*64,actual_state_sha256='a'*64,complete_robot_joint_count=69,
        measured_time_s=42.548,tolerance=1e-5,maximum_errors={'joint_position':0.,'joint_velocity':0.})
    contacts=[dict(digit=d,body='robot/rh_'+d+'distal',body_position_m=[0.,0.,0.],position=[0.,0.,0.],
        normal_force_N=.5,pad_qualified=True,inward_radial_normal_alignment=.99,axial_clearance_m=.01)
        for d in ['ff','mf','rf','lf','th']]
    contacts[3]['inward_radial_normal_alignment']=.9999996645999522
    admission=dict(schema='doorbench.isaac-release-source.v1',source_engine='isaac-physx',source_run=str(tmp_path),
        grasp_profile='volar-phalange-v1',contact_audit_name='independent-contact-audit.json',
        source_qualification=dict(passed=True,state_sha256='a'*64,time_s=42.548,input_sha256=inputs.copy(),
            local_direct_summary=dict(passed=True,all_loaded_handle_patches_qualified=True,
                independent_audit_passed=True,isaac_runtime_passed=True)),
        coordinate_admission=dict(passed=True,state_sha256='a'*64,coordinate_admission=dict(state_admission=state,
            free_roots=1,scalar_coordinates=model.nv-6,simulation_steps=0,active_plant_writes=0),measured_body_admission=body),
        material_frame_admission=dict(passed=True,maximum_position_error_m=8.09414560984358e-7,
            maximum_body_position_error_m=8.003232804672213e-7,
            maximum_body_rotation_error_rad=1.236538092666255e-6,position_limit_m=2e-6,rotation_limit_rad=2e-6),
        measured_rest=dict(terminal_time_s=42.548,window_samples=251,minimum_palm_load_N=2.6411051363475058,
            maximum_abs_operator_rad=1.463378168642393e-10,maximum_abs_latch_m=0.,endpoint_contacts=contacts),
        initial_qpos=model.qpos0.tolist(),robot_path='synthetic-original.xml',door_xml_path='synthetic-door.xml',
        door_usd_path='synthetic-door.usda',input_sha256=inputs.copy(),physics_steps=0,source_sample_playback=0,authorized_stages=0)
    fresh=copy.deepcopy(admission)
    fresh['coordinate_admission']['measured_body_admission']['measured']['qvel'][angular[-1]]=0.014784089070371821
    update_measured_hash(fresh)
    fresh['material_frame_admission']['maximum_body_rotation_error_rad']=1.2365380926630737e-6
    fresh['measured_rest']['endpoint_contacts'][3]['inward_radial_normal_alignment']=.999999664599952
    scenes=[]
    def scene(self):
        self.verify_inputs();scenes.append(self.sha256)
        return SimpleNamespace(m=model,d=mujoco.MjData(model))
    monkeypatch.setattr(IsaacReleasePlanningContext,'scene',scene)
    return dict(old=admission,fresh=fresh,candidate=candidate_for(admission),model=model,
        angular=angular,original=original,scenes=scenes)


def test_exact_four_recorded_differences_keep_archived_identity_and_inputs(evidence):
    e=evidence;before=copy.deepcopy(e['candidate']);fresh=context(e['fresh'])
    bound,receipt=reconcile(e['candidate'],fresh)
    assert isinstance(bound,IsaacReleasePlanningContext)
    assert bound.sha256==sha(e['old']) and bound.admission==e['old']
    assert fresh.admission==e['fresh'] and e['candidate']==before
    assert bound.sha256!=fresh.sha256
    assert receipt['passed'] is True and receipt['authorized_stages']==0
    assert receipt['difference_count']==4
    assert receipt['archived_context_sha256']==bound.sha256
    assert receipt['fresh_context_sha256']==fresh.sha256
    assert receipt['transformed_root_angular_velocity_indices']==[6,7,8]
    assert receipt['diagnostic_absolute_tolerance']==1e-15
    assert e['scenes']
    # Mutating returned detached copies cannot rewrite either canonical context.
    value=bound.admission;value['initial_qpos'][0]=999
    assert bound.admission==e['old']
    json.dumps(receipt,allow_nan=False)


def test_identical_receipt_remains_exact_without_requiring_difference(evidence):
    bound,receipt=reconcile(evidence['candidate'],context(evidence['old']))
    assert bound.sha256==sha(evidence['old']) and receipt['passed']


@pytest.mark.parametrize('change',[
    'candidate_sha','candidate_qpos','qpos','state_sha','physics_archive_hash','raw_body_pose',
    'raw_qpos','root_translation_rate','scalar_rate','contact_force','contact_point','contact_body',
    'contact_label','axial_clearance','reconstructed_rotation','position_error','grasp_profile',
    'source_path','missing_field','extra_field','bool_as_int','float_as_int',
])
def test_every_other_source_field_type_hash_label_and_coordinate_is_exact(evidence,change):
    e=evidence;fresh=e['fresh'];b=fresh['coordinate_admission']['measured_body_admission'];c=e['candidate']
    if change=='candidate_sha':c['source_context_sha256']='b'*64
    elif change=='candidate_qpos':c['initial_qpos'][0]=np.nextafter(c['initial_qpos'][0],1.).item()
    elif change=='qpos':fresh['initial_qpos'][0]=np.nextafter(fresh['initial_qpos'][0],1.).item()
    elif change=='state_sha':fresh['source_qualification']['state_sha256']='b'*64
    elif change=='physics_archive_hash':fresh['input_sha256'][str(e['original'])]='b'*64
    elif change=='raw_body_pose':b['measured']['body_poses_xyz_wxyz'][0][0]=1e-18;update_measured_hash(fresh)
    elif change=='raw_qpos':b['measured']['qpos'][0]=1e-18;update_measured_hash(fresh)
    elif change=='root_translation_rate':b['measured']['qvel'][e['angular'][0]-1]=1e-18;update_measured_hash(fresh)
    elif change=='scalar_rate':b['measured']['qvel'][-1]=1e-18;update_measured_hash(fresh)
    elif change=='contact_force':fresh['measured_rest']['endpoint_contacts'][0]['normal_force_N']=np.nextafter(.5,1.).item()
    elif change=='contact_point':fresh['measured_rest']['endpoint_contacts'][0]['body_position_m'][0]=1e-18
    elif change=='contact_body':fresh['measured_rest']['endpoint_contacts'][0]['body']='robot/other'
    elif change=='contact_label':fresh['measured_rest']['endpoint_contacts'][0]['pad_qualified']=False
    elif change=='axial_clearance':fresh['measured_rest']['endpoint_contacts'][0]['axial_clearance_m']+=1e-16
    elif change=='reconstructed_rotation':b['reconstructed_rotations'][0][0][0]=np.nextafter(1.,0.).item()
    elif change=='position_error':b['maximum_position_error_m']+=1e-18
    elif change=='grasp_profile':fresh['grasp_profile']='distal-pad-v1'
    elif change=='source_path':fresh['source_run']+='other'
    elif change=='missing_field':del fresh['contact_audit_name']
    elif change=='extra_field':fresh['invented_receipt_field']=True
    elif change=='bool_as_int':fresh['authorized_stages']=False
    else:fresh['physics_steps']=0.
    with pytest.raises(ValueError):reconcile(c,context(fresh))


@pytest.mark.parametrize('side',['old','fresh'])
def test_dependent_digest_must_be_valid_for_each_full_measured_payload(evidence,side):
    e=evidence
    admission=e[side];admission['coordinate_admission']['measured_body_admission']['measured_sha256']='c'*64
    if side=='old':e['candidate']=candidate_for(admission)
    with pytest.raises(ValueError):reconcile(e['candidate'],context(e['fresh']))


@pytest.mark.parametrize('kind',['qvel','material','alignment'])
@pytest.mark.parametrize('difference',[1e-14,1e-12,float('inf'),float('nan')])
def test_whitelisted_paths_still_require_finite_absolute_roundoff_only(evidence,kind,difference):
    e=evidence;fresh=e['fresh'];old=e['old']
    if kind=='qvel':
        fresh['coordinate_admission']['measured_body_admission']['measured']['qvel'][e['angular'][-1]]=old['coordinate_admission']['measured_body_admission']['measured']['qvel'][e['angular'][-1]]+difference
        if np.isfinite(difference):update_measured_hash(fresh)
    elif kind=='material':fresh['material_frame_admission']['maximum_body_rotation_error_rad']=old['material_frame_admission']['maximum_body_rotation_error_rad']+difference
    else:fresh['measured_rest']['endpoint_contacts'][3]['inward_radial_normal_alignment']=old['measured_rest']['endpoint_contacts'][3]['inward_radial_normal_alignment']+difference
    # The context itself rejects nonfinite canonical JSON; it cannot be carried
    # through reconciliation as a valid freshly admitted finite context.
    with pytest.raises(ValueError):reconcile(e['candidate'],context(fresh))


@pytest.mark.parametrize('change',['physical','coordinate','state','body','material','position_threshold',
    'rotation_threshold','normalization_threshold','body_error','material_error','source_steps','stage_authority','rest_window'])
def test_fresh_original_checks_are_required_even_when_both_receipts_agree(evidence,change):
    a=evidence['fresh'];body=a['coordinate_admission']['measured_body_admission']
    if change=='physical':a['source_qualification']['passed']=False
    elif change=='coordinate':a['coordinate_admission']['passed']=False
    elif change=='state':a['coordinate_admission']['coordinate_admission']['state_admission']['passed']=False
    elif change=='body':body['passed']=False
    elif change=='material':a['material_frame_admission']['passed']=False
    elif change=='position_threshold':body['position_limit_m']=3e-6
    elif change=='rotation_threshold':body['rotation_limit_rad']=3e-6
    elif change=='normalization_threshold':body['quaternion_normalization_limit']=3e-6
    elif change=='body_error':body['maximum_rotation_error_rad']=2.1e-6
    elif change=='material_error':a['material_frame_admission']['maximum_body_rotation_error_rad']=2.1e-6
    elif change=='source_steps':a['physics_steps']=1
    elif change=='stage_authority':a['authorized_stages']=1
    else:a['measured_rest']['window_samples']=250
    with pytest.raises(ValueError):reconcile(candidate_for(a),context(a))


def test_changed_actual_input_bytes_fail_before_binding(evidence):
    evidence['original'].write_bytes(b'Changed after fresh receipt construction')
    with pytest.raises(ValueError):reconcile(evidence['candidate'],context(evidence['fresh']))


def test_requires_real_immutable_planning_context_type(evidence):
    with pytest.raises(ValueError):reconcile(evidence['candidate'],SimpleNamespace(admission=evidence['fresh']))


@pytest.mark.parametrize('scalars_before',[0,5])
def test_angular_whitelist_comes_from_original_model_dof_address(evidence,monkeypatch,scalars_before):
    model=make_model(scalars_before);free=int(np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0])
    angular=list(range(int(model.jnt_dofadr[free])+3,int(model.jnt_dofadr[free])+6))
    for admission in (evidence['old'],evidence['fresh']):
        admission['initial_qpos']=model.qpos0.tolist()
        measured=admission['coordinate_admission']['measured_body_admission']['measured']
        measured['qpos']=model.qpos0.tolist();measured['qvel']=np.zeros(model.nv).tolist()
        update_measured_hash(admission)
    fresh=evidence['fresh'];fresh['coordinate_admission']['measured_body_admission']['measured']['qvel'][angular[-1]]=1e-16
    update_measured_hash(fresh)
    monkeypatch.setattr(IsaacReleasePlanningContext,'scene',lambda self:SimpleNamespace(m=model))
    _,receipt=reconcile(candidate_for(evidence['old']),context(fresh))
    assert receipt['transformed_root_angular_velocity_indices']==angular
    assert receipt['difference_count']==4
    # A scalar after the free joint stays exact even if numerically adjacent.
    fresh['coordinate_admission']['measured_body_admission']['measured']['qvel'][angular[-1]+1]=1e-18
    update_measured_hash(fresh)
    with pytest.raises(ValueError):reconcile(candidate_for(evidence['old']),context(fresh))


@pytest.mark.parametrize('difference,accepted',[(1e-15,True),(np.nextafter(1e-15,np.inf).item(),False)])
def test_exact_absolute_roundoff_boundary_without_relative_tolerance(evidence,difference,accepted):
    fresh=copy.deepcopy(evidence['old'])
    fresh['coordinate_admission']['measured_body_admission']['measured']['qvel'][evidence['angular'][0]]=difference
    update_measured_hash(fresh)
    if accepted:
        _,receipt=reconcile(evidence['candidate'],context(fresh));assert receipt['difference_count']==2
    else:
        with pytest.raises(ValueError):reconcile(evidence['candidate'],context(fresh))


def test_input_changes_during_private_model_inspection_fail_closed(evidence,monkeypatch):
    def changed(self):
        evidence['original'].write_bytes(b'Changed during reconciliation')
        return SimpleNamespace(m=evidence['model'])
    monkeypatch.setattr(IsaacReleasePlanningContext,'scene',changed)
    with pytest.raises(ValueError):reconcile(evidence['candidate'],context(evidence['fresh']))


def test_actual_frozen_four_path_kit_comparison_read_only_when_available():
    comparison=Path(__file__).resolve().parents[2]/'isaac-release-context-kit-diff002.json'
    if not comparison.exists():pytest.skip('Local actual comparison archive is not bundled in portable tests')
    record=json.loads(comparison.read_text())
    candidates=[Path(p) for p in record['input_sha256'] if Path(p).name=='candidate.json']
    assert len(candidates)==1 and record['fresh_original_source_admission_passed']
    assert record['difference_count']==4 and record['initial_qpos_equal']
    path=candidates[0];candidate=json.loads(path.read_text());fresh_path=Path(record['fresh_admission_path'])
    fresh=context(json.loads(fresh_path.read_text()))
    originals={p:p.read_bytes() for p in (comparison,path,fresh_path)}
    assert hashlib.sha256(originals[path]).hexdigest()==record['input_sha256'][str(path)]
    bound,receipt=reconcile(candidate,fresh)
    assert bound.sha256==record['candidate_context_sha256']
    assert fresh.sha256==record['fresh_context_sha256']
    assert receipt['difference_count']==4 and receipt['transformed_root_angular_velocity_indices']==[6,7,8]
    assert receipt['authorized_stages']==0 and receipt['physics_steps']==0
    assert all(p.read_bytes()==value for p,value in originals.items())

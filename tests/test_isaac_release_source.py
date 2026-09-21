"""Synthetic CPU admission fixtures; no simulator stepping or physical claim."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import doorbench.dexterous.isaac_release_source as release
from doorbench.dexterous.isaac_pad_audit import shadow_physx_pad_grasp


DIGITS=('ff','mf','rf','lf','th')
DOORS=['leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide']


def window():
    times=1.+np.arange(251)*.002
    doors=np.tile([.09,0.,0.],(251,1))
    contacts=[];transforms={};pairs={}
    for i,digit in enumerate(DIGITS):
        body='/World/H1/pelvis/rh_'+digit+'distal'
        thumb=digit=='th';point=np.array([0.,-.007 if thumb else .007,(i-2)*.012])
        normal=np.array([0.,-1. if thumb else 1.,0.]);quat=np.array([0.,0.,1.,0.]) if thumb else np.array([0.,0.,0.,1.])
        local=np.array([0.,-.007,.01]);origin=point-Rotation.from_quat(quat).apply(local)
        contacts.append(dict(body=body,position=point.tolist(),normal=normal.tolist(),normal_force_N=1.))
        transforms[body]=np.r_[origin,quat].tolist();pairs[body]=normal.tolist()
    lever=dict(center=[0.,0.,0.],axis=[0.,0.,1.],half_length=.053,radius=.007)
    grasp=shadow_physx_pad_grasp(contacts,transforms,lever['center'],lever['axis'],half_length=.053,radius=.007,profile='volar-phalange-v1')
    assert grasp['valid_pad_grasp']
    pads=[];surfaces=[]
    for t in times:
        raw=dict(schema='doorbench.shadow-raw-pad-evidence.v1',interval_start_s=t-.002,interval_end_s=t,geometry_time_s=t,
            clock='physx-interval-end',scope='complete-handle-body',contacts=contacts,body_transforms_xyzw=transforms,
            handle_pair_forces_world_N=pairs,lever=lever,contact_capacity=100,active_contact_count=5,normal_pair_force_consistency_error_N=0.)
        pads.append(copy.deepcopy(dict(grasp,sim_time_s=t,physics_dt_s=.002,raw_evidence=raw,contact_capacity=100,active_contact_count=5)))
        surfaces.append(dict(time_s=t,leaf_pose=[0.,0.,0.,1.,0.,0.,0.],surface=dict(palm_normal_load_N=3.,total_normal_load_N=3.,body_panel_forces_world_N={'lh_palm':[0.,-3.,0.]})))
    return times,doors,pads,surfaces


def validate(values,**kwargs):
    times,doors,pads,surfaces=values
    return release.validate_rest_window(times,doors,DOORS,pads,surfaces,terminal=kwargs.get('terminal',1.5),profile=kwargs.get('profile','volar-phalange-v1'))


def test_actual_window_reconstructs_contacts_and_palm_force_without_mutation():
    values=window();before=copy.deepcopy(values[2:]);result=validate(values)
    assert result['window_samples']==251 and result['minimum_palm_load_N']==3.
    assert {c['digit'] for c in result['endpoint_contacts']}==set(DIGITS)
    result['endpoint_contacts'][0]['position'][0]=9.
    assert values[2:]==before


@pytest.mark.parametrize('terminal',[float('nan'),float('inf'),True,.49,1.502])
def test_invalid_terminal_is_rejected(terminal):
    with pytest.raises(ValueError):validate(window(),terminal=terminal)


@pytest.mark.parametrize('kind',['times_count','times_shape','times_nan','gap','duplicate','door_nan','leaf','operator','latch','pad_count','surface_count'])
def test_complete_same_epoch_original_mechanism_rest_required(kind):
    times,doors,pads,surfaces=window()
    if kind=='times_count':times=times[:-1]
    if kind=='times_shape':times=times.reshape(-1,1)
    if kind=='times_nan':times[50]=float('nan')
    if kind=='gap':times[50]+=.0001
    if kind=='duplicate':times[50]=times[49]
    if kind=='door_nan':doors[50,0]=float('nan')
    if kind=='leaf':doors[50,0]=.101
    if kind=='operator':doors[50,1]=.051
    if kind=='latch':doors[50,2]=.0011
    if kind=='pad_count':pads=pads[:-1]
    if kind=='surface_count':surfaces=surfaces[:-1]
    with pytest.raises(ValueError):validate((times,doors,pads,surfaces))


@pytest.mark.parametrize('kind',['pad_nan','surface_nan','raw_nan','wrong_profile','wrong_hand','wrong_dt','reported_false','raw_epoch','capacity','digit_load','patch_flag','normal','pair_force','palm_force','palm_nan','palm_below','finger_substitute'])
def test_reported_labels_never_substitute_for_actual_raw_contact_or_palm(kind):
    values=window();pad=values[2][125];surface=values[3][125];raw=pad['raw_evidence']
    if kind=='pad_nan':pad['sim_time_s']=float('nan')
    if kind=='surface_nan':surface['time_s']=float('nan')
    if kind=='raw_nan':raw['geometry_time_s']=float('nan')
    if kind=='wrong_profile':pad['grasp_profile']='distal-pad-v1'
    if kind=='wrong_hand':pad['hand']='lh'
    if kind=='wrong_dt':pad['physics_dt_s']=.01
    if kind=='reported_false':pad['valid_pad_grasp']=False
    if kind=='raw_epoch':raw['interval_start_s']-=.002
    if kind=='capacity':raw['active_contact_count']=100;pad['active_contact_count']=100
    if kind=='digit_load':pad['qualified_pad_forces_N']['ff']=9.
    if kind=='patch_flag':pad['contacts'][0]['pad_qualified']=False
    if kind=='normal':raw['contacts'][0]['normal']=[0.,-1.,0.]
    if kind=='pair_force':raw['handle_pair_forces_world_N'][raw['contacts'][0]['body']]=[0.,2.,0.]
    if kind=='palm_force':surface['surface']['body_panel_forces_world_N']['lh_palm']=[0.,-1.,0.]
    if kind=='palm_nan':surface['surface']['palm_normal_load_N']=float('nan')
    if kind=='palm_below':surface['surface']['body_panel_forces_world_N']['lh_palm']=[0.,-1.99,0.];surface['surface']['palm_normal_load_N']=1.99
    if kind=='finger_substitute':surface['surface']['body_panel_forces_world_N']={'lh_lfmiddle':[0.,-3.,0.]}
    with pytest.raises(ValueError):validate(values)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path,monkeypatch):
    """Mock prior qualification/kinematic plumbing, keep real raw rest admission."""
    values=window();run=tmp_path/'explicit synthetic source';trial=run/'trial';trial.mkdir(parents=True)
    robot=tmp_path/'robot.xml';door=tmp_path/'door.xml';usd=tmp_path/'door.usda'
    for p in (robot,door,usd):p.write_text('synthetic fixture asset '+p.name)
    declaration=tmp_path/'prospective.json';write(declaration,dict(profile='volar-phalange-v1',robot_xml_sha256=release.digest(robot)))
    (trial/'grasp-profile-definition.json').write_bytes(declaration.read_bytes())
    route=tmp_path/'route.json';write(route,dict(schema='doorbench.standing-transfer.v1',geometric_screen_passed=True,robot_xml_sha256=release.digest(robot),door_xml_sha256=release.digest(door)))
    captured=trial/'standing-transfer-route.json';captured.write_bytes(route.read_bytes())
    configuration=dict(args=dict(grasp_profile='volar-phalange-v1',grasp_profile_definition=str(declaration),standing_transfer_route=str(route)),door_joint_names=DOORS)
    report=dict(grasp_profile='volar-phalange-v1',standing_transfer=dict(route=str(route)))
    # Match real standalone archives: door XML is bound by the captured route,
    # not listed directly in the producer's top-level provenance.
    provenance=dict(files={str(p):release.digest(p) for p in (robot,usd,declaration,route)})
    write(trial/'configuration.json',configuration);write(trial/'operation-report.json',report);write(trial/'provenance.json',provenance)
    np.savez(trial/'acquisition-physics.npz',time_s=values[0],door=values[1])
    for name,rows in [('acquisition-pad-steps.json.gz',values[2]),('standing-transfer-steps.json.gz',values[3])]:
        with gzip.open(trial/name,'wt') as f:json.dump(rows,f)
    paths=list(trial.iterdir())+[robot,usd,declaration,route]
    hashes={str(p):release.digest(p) for p in paths}
    qualification=dict(passed=True,state_sha256='a'*64,time_s=1.5,input_sha256=hashes)
    extracted=dict(binding=dict(robot_source_sha256=release.digest(robot),door_source_sha256=release.digest(usd)))
    import scripts.dexterous.plan_local_isaac_transfer as loader
    monkeypatch.setattr(loader,'load_local_source',lambda *_:(copy.deepcopy(extracted),{},copy.deepcopy(qualification)))
    import doorbench.dexterous.isaac_prefix_witness as prefix
    class Witness:
        def __init__(self,*args,**kwargs):self.kwargs=kwargs
        def receipt(self):return dict(input_sha256=copy.deepcopy(hashes),source_qualification=copy.deepcopy(qualification),stage_entry_authorized=False,intervals_verified=0)
        def observe(self,*_):pytest.fail('Adapter must never replay source samples')
        def require_stage_entry(self,*_):pytest.fail('Adapter must never authorize a stage')
    monkeypatch.setattr(prefix,'LiveIsaacPrefixWitness',Witness)
    raw=values[2][-1]['raw_evidence'];poses=list(raw['body_transforms_xyzw'].values());bodies={body.rsplit('/',1)[-1]:i for i,body in enumerate(raw['body_transforms_xyzw'])}
    m=SimpleNamespace(body=lambda name:SimpleNamespace(id=bodies[name.removeprefix('robot/')]))
    d=SimpleNamespace(qpos=np.zeros(7),xpos=np.array([p[:3] for p in poses]),xmat=np.array([Rotation.from_quat(p[3:]).as_matrix().reshape(9) for p in poses]))
    monkeypatch.setattr(release,'LandedLeftScene',lambda *_:SimpleNamespace(m=m,d=d))
    monkeypatch.setattr(release.mujoco,'mj_kinematics',lambda *_:None)
    monkeypatch.setattr(release.mujoco,'mj_step',lambda *_:pytest.fail('No physical stepping permitted'))
    monkeypatch.setattr(release,'admit_destination_planner',lambda *a,**k:(SimpleNamespace(qpos=np.zeros(7)),dict(passed=True)))
    return dict(run=run,trial=trial,robot=robot,door=door,usd=usd,declaration=declaration,route=route,captured=captured,configuration=configuration,report=report,provenance=provenance,hashes=hashes,qualification=qualification,extracted=extracted,data=d)


def admit(source):
    return release.admit_isaac_release_source(source['run'],robot=source['robot'],door_xml=source['door'],door_usd=source['usd'])


def test_admission_returns_original_frame_material_points_with_zero_stages(source):
    original={p:release.digest(p) for p in source['hashes']}
    result=admit(source)
    assert result['authorized_stages']==result['physics_steps']==result['source_sample_playback']==0
    assert result['source_engine']=='isaac-physx' and result['material_frame_admission']['passed']
    assert result['planning_asset_binding']['direct_xml_provenance'] is False
    assert result['source_qualification']==source['qualification']
    assert all(c['body'].startswith('robot/rh_') and c['isaac_body_path'].startswith('/World/H1/') for c in result['measured_rest']['endpoint_contacts'])
    assert {p:release.digest(p) for p in original}==original
    assert 'native_manifest' not in result and 'release_passed' not in result


@pytest.mark.parametrize('kind',['missing_transfer','wrong_profile','wrong_robot','wrong_usd','changed_declaration','changed_capture','changed_route','wrong_xml','body_translation','body_rotation'])
def test_invalid_source_identity_or_material_mapping_rejected(source,kind):
    if kind=='missing_transfer':source['report'].pop('standing_transfer');write(source['trial']/'operation-report.json',source['report'])
    if kind=='wrong_profile':source['report']['grasp_profile']='distal-pad-v1';write(source['trial']/'operation-report.json',source['report'])
    if kind=='wrong_robot':source['robot'].write_text('wrong robot')
    if kind=='wrong_usd':source['usd'].write_text('wrong USD')
    if kind=='changed_declaration':source['declaration'].write_text('{}')
    if kind=='changed_capture':source['captured'].write_text('{}')
    if kind=='changed_route':source['route'].write_text('{}')
    if kind=='wrong_xml':source['door'].write_text('other XML')
    if kind=='body_translation':source['data'].xpos[0,0]+=3e-6
    if kind=='body_rotation':source['data'].xmat[0]=Rotation.from_rotvec([0.,0.,3e-6]).as_matrix().reshape(9)
    with pytest.raises(ValueError):admit(source)


def test_changed_original_hash_is_never_overwritten_by_later_tracking(source):
    expected=dict(source['hashes']);source['route'].write_text('changed')
    with pytest.raises(ValueError,match='changed'):release._track(expected,source['route'])
    assert expected==source['hashes']


def test_source_mutation_after_planning_admission_is_detected(source,monkeypatch):
    def destination(*a,**k):
        source['route'].write_text('changed during unstepped mapping')
        return SimpleNamespace(qpos=np.zeros(7)),dict(passed=True)
    monkeypatch.setattr(release,'admit_destination_planner',destination)
    with pytest.raises(ValueError,match='changed'):admit(source)


def test_prior_source_qualification_failure_propagates_before_any_geometry(source,monkeypatch):
    import scripts.dexterous.plan_local_isaac_transfer as loader
    def reject(*_):raise ValueError('Complete passing physical operation report required')
    monkeypatch.setattr(loader,'load_local_source',reject)
    monkeypatch.setattr(release,'LandedLeftScene',lambda *_:pytest.fail('Failed source reached geometry'))
    with pytest.raises(ValueError,match='passing physical'):admit(source)

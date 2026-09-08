import copy
import numpy as np
import pytest
from test_isaac_pad_audit import fixture
from doorbench.dexterous.sensor_acquisition_evidence import evaluate_pad_evidence,RAW_SCHEMA


def raw():
    contacts,poses=fixture()
    return dict(schema=RAW_SCHEMA,interval_start_s=.998,interval_end_s=1.,geometry_time_s=1.,clock='physx-interval-end',scope='complete-handle-body',
        contacts=[dict(body=c['body'],position=c['position'].tolist(),normal=c['normal'].tolist(),normal_force_N=c['normal_force_N']) for c in contacts],
        body_transforms_xyzw={n:p.tolist() for n,p in poses.items()},handle_pair_forces_world_N={c['body']:(c['normal']*c['normal_force_N']).tolist() for c in contacts},
        lever=dict(center=[0,0,0],axis=[1,0,0],half_length=.053,radius=.007),contact_capacity=8,active_contact_count=5,normal_pair_force_consistency_error_N=0.)


def test_reduction_uses_raw_force_geometry_not_an_actor_label():
    e=raw();r=evaluate_pad_evidence(e,time_s=1.);assert r['valid_pad_grasp'] and r['invalid_loaded_patches']==0
    e['clock']='native-interval-start';e['scope']='native-lever-collider';e['geometry_time_s']=.998
    e['contact_capacity']=e['handle_pair_forces_world_N']=e['normal_pair_force_consistency_error_N']=None
    assert evaluate_pad_evidence(e,time_s=1.)['valid_pad_grasp']


@pytest.mark.parametrize('bad',['epoch','mixed_engine','pose','normal','pair','label','capacity','count','missing_body','negative','future'])
def test_incomplete_or_mixed_actual_contact_data_fails_closed(bad):
    e=raw();path=e['contacts'][0]['body']
    if bad=='epoch':e['geometry_time_s']=.998
    if bad=='mixed_engine':e['clock']='native-interval-start'
    if bad=='pose':e['body_transforms_xyzw'][path][-1]=2.
    if bad=='normal':e['contacts'][0]['normal']=[0,2,0]
    if bad=='pair':e['handle_pair_forces_world_N'][path]=[0,0,0]
    if bad=='label':e['contacts'][0]['pad_qualified']=True
    if bad=='capacity':e['contact_capacity']=5
    if bad=='count':e['active_contact_count']=4
    if bad=='missing_body':e['body_transforms_xyzw'].pop(path)
    if bad=='negative':e['contacts'][0]['normal_force_N']=-1
    if bad=='future':e['interval_end_s']=1.002
    with pytest.raises(ValueError):evaluate_pad_evidence(e,time_s=1.)


def test_dorsal_patch_and_missing_thumb_reject_grasp_without_erasing_loads():
    e=raw();path=e['contacts'][-1]['body'];e['body_transforms_xyzw'][path][1]-=.016
    r=evaluate_pad_evidence(e,time_s=1.);assert not r['valid_pad_grasp'] and r['invalid_loaded_patches']==1 and r['digit_forces_N']['th']==1
    e=raw();path=e['contacts'][-1]['body'];e['contacts'][-1]['normal_force_N']=0.;e['handle_pair_forces_world_N'][path]=[0,0,0]
    r=evaluate_pad_evidence(e,time_s=1.);assert not r['valid_pad_grasp'] and r['qualified_pad_forces_N']['th']==0


def rounded_raw():
    # Multiple float32 patches exercise multiplication AND ordered axis-sum.
    e=raw();first=e['contacts'][0];path=first['body'];patches=[]
    for force in np.array([2.7,.123,.5],np.float32):
        c=copy.deepcopy(first);c['normal_force_N']=float(force)
        c['normal']=np.array([.6,-.8,0],np.float32).tolist();patches.append(c)
    e['contacts']=patches+e['contacts'][1:];e['active_contact_count']=7;e['contact_capacity']=16
    f=np.array([c['normal_force_N'] for c in patches],np.float32)
    n=np.array([c['normal'] for c in patches],np.float32)
    e['handle_pair_forces_world_N'][path]=(f[:,None]*n).sum(axis=0).tolist()
    return e


def test_float32_receipt_roundtrip_keeps_independent_physical_threshold():
    e=rounded_raw();r=evaluate_pad_evidence(e,time_s=1.)
    assert r['raw_pair_force_reconstruction_error_N']>1e-8
    assert r['raw_pair_force_reconstruction_error_N']<1e-3
    assert r['raw_source_pair_force_reconstruction_error_N']==0.
    e['normal_pair_force_consistency_error_N']=5e-7
    with pytest.raises(ValueError,match='reproduce'):evaluate_pad_evidence(e,time_s=1.)


def test_precision_fix_does_not_admit_bad_matrix_or_requantized_source():
    e=rounded_raw();path=e['contacts'][0]['body']
    e['handle_pair_forces_world_N'][path][0]+=float(np.float32(.00390625))
    with pytest.raises(ValueError):evaluate_pad_evidence(e,time_s=1.)
    e=rounded_raw();e['contacts'][0]['normal_force_N']+=1e-10
    with pytest.raises(ValueError,match='exact float32'):evaluate_pad_evidence(e,time_s=1.)


def test_nonzero_float32_norm_is_reproduced_not_assumed_zero():
    e=rounded_raw();path=e['contacts'][0]['body']
    pair=np.asarray(e['handle_pair_forces_world_N'][path],np.float32)
    original=pair.copy();pair[0]=np.nextafter(pair[0],np.float32(np.inf))
    e['handle_pair_forces_world_N'][path]=pair.tolist()
    declared=float(np.linalg.norm(original-pair));assert declared>1e-8
    e['normal_pair_force_consistency_error_N']=declared
    assert evaluate_pad_evidence(e,time_s=1.)['raw_source_pair_force_reconstruction_error_N']==declared


def test_missing_digit_still_fails_with_correct_float32_receipt():
    e=rounded_raw();thumb=e['contacts'][-1]
    thumb['normal_force_N']=0.;e['handle_pair_forces_world_N'][thumb['body']]=[0.,0.,0.]
    r=evaluate_pad_evidence(e,time_s=1.)
    assert not r['valid_pad_grasp'] and r['qualified_pad_forces_N']['th']==0.

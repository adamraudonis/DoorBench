"""Synthetic copied measurements only; no robot or physical qualification."""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doorbench.dexterous.isaac_transfer_rest_stop import TransferRestStop,SCHEMA


START=.002
FIRST_WINDOW_TICK=4001


def sample(t,*,start=START):
    contacts=[dict(digit=d,normal_force_N=.5,pad_qualified=True) for d in ('ff','mf','rf','lf','th')]
    raw=dict(schema='doorbench.shadow-raw-pad-evidence.v1',interval_start_s=t-.002,
        interval_end_s=t,geometry_time_s=t,clock='physx-interval-end',scope='complete-handle-body',
        contacts=[{'synthetic_patch':True} for _ in contacts],contact_capacity=64,active_contact_count=5,
        normal_pair_force_consistency_error_N=0.)
    pad=dict(sim_time_s=t,physics_dt_s=.002,valid_pad_grasp=True,contacts=contacts,
        non_digit_handle_force_N=0.,grasp_profile='volar-phalange-v1',hand='rh',raw_evidence=raw,
        contact_capacity=64,active_contact_count=5,normal_pair_force_consistency_error_N=0.)
    surface=dict(time_s=t,leaf_pose=[0,0,0,1,0,0,0],stance_status='solved',started_s=start,
        surface=dict(palm_normal_load_N=2.,body_panel_forces_world_N={'lh_palm':[0,-2.,0]}))
    return pad,surface,dict(leaf=.09,operator=.0,latch=.0)


@pytest.fixture(scope='module')
def observed_prefix():
    detector=TransferRestStop(START)
    for tick in range(1,FIRST_WINDOW_TICK):
        assert not detector.observe(tick*.002,*sample(tick*.002))
    return detector


@pytest.fixture
def detector(observed_prefix):return copy.deepcopy(observed_prefix)


def feed(detector,start,end):
    results=[]
    for tick in range(start,end+1):results.append(detector.observe(tick*.002,*sample(tick*.002)))
    return results


def test_first_terminal_requires_251_inclusive_samples_after_route_reach(detector):
    assert detector.receipt()['window_samples']==0
    assert not any(feed(detector,FIRST_WINDOW_TICK,FIRST_WINDOW_TICK+249))
    assert detector.receipt()['window_samples']==250
    assert detector.observe((FIRST_WINDOW_TICK+250)*.002,*sample((FIRST_WINDOW_TICK+250)*.002))
    r=detector.receipt()
    assert r['schema']==SCHEMA and r['triggered'] and r['failure'] is None
    assert r['terminal_time_s']==pytest.approx(START+8.5)
    assert r['window_start_s']==pytest.approx(START+8.)
    assert r['window_samples']==251 and r['required_window_samples']==251
    assert r['maximum_window_samples']==251 and r['maximum_window_duration_s']==.5
    assert all(r['checks'].values()) and not r['failed_checks'] and not r['physical_qualification']


def test_first_receipt_frozen_and_returned_copies_cannot_mutate_it(detector):
    feed(detector,FIRST_WINDOW_TICK,FIRST_WINDOW_TICK+250);original=detector.receipt()
    assert detector.observe(float('nan'),None,None,None)
    assert detector.observe(999.,*sample(999.))
    changed=detector.receipt();changed['checks'].clear();changed['terminal_time_s']=999.
    assert detector.receipt()==original


@pytest.mark.parametrize('bad',['grasp','patch','non_digit','palm','palm_accounting','stance','leaf_low','leaf_high',
    'operator','latch','wrong_start','not_started','truncated','full_buffer','pair_force','empty','negative_patch'])
def test_each_ordinary_failed_gate_resets_the_whole_joint_window(detector,bad):
    feed(detector,FIRST_WINDOW_TICK,FIRST_WINDOW_TICK+248)
    tick=FIRST_WINDOW_TICK+249;t=tick*.002;pad,surface,angles=sample(t)
    if bad=='grasp':pad['valid_pad_grasp']=False
    if bad=='patch':pad['contacts'][0]['pad_qualified']=False
    if bad=='non_digit':pad['non_digit_handle_force_N']=.00001
    if bad=='palm':surface['surface'].update(palm_normal_load_N=1.999,body_panel_forces_world_N={'lh_palm':[0,-1.999,0]})
    if bad=='palm_accounting':surface['surface']['palm_normal_load_N']=3.
    if bad=='stance':surface['stance_status']='maximum iterations reached'
    if bad=='leaf_low':angles['leaf']=.074999
    if bad=='leaf_high':angles['leaf']=.100001
    if bad=='operator':angles['operator']=.050001
    if bad=='latch':angles['latch']=.001001
    if bad=='wrong_start':surface['started_s']=START+.002
    if bad=='not_started':surface['started_s']=None
    if bad=='truncated':pad['raw_evidence']['truncated']=True
    if bad=='full_buffer':
        pad['contact_capacity']=5;pad['raw_evidence']['contact_capacity']=5
    if bad=='pair_force':
        pad['normal_pair_force_consistency_error_N']=.001001
        pad['raw_evidence']['normal_pair_force_consistency_error_N']=.001001
    if bad=='empty':pad['contacts']=[]
    if bad=='negative_patch':pad['contacts'][0]['normal_force_N']=-1.
    assert not detector.observe(t,pad,surface,angles)
    r=detector.receipt()
    assert r['failure'] is None and r['window_samples']==0 and r['maximum_window_samples']==249
    assert r['window_start_s'] is None and r['terminal_time_s'] is None and r['failed_checks']
    assert not any(feed(detector,tick+1,tick+250))
    assert detector.observe((tick+251)*.002,*sample((tick+251)*.002))
    assert detector.receipt()['terminal_time_s']==(tick+251)*.002


@pytest.mark.parametrize('bad',['skip','repeat','pad_clock','raw_start','raw_end','geometry_clock','surface_clock','dt',
    'nan_palm','nan_force','nan_angles','missing_raw','malformed_flag','profile_change'])
def test_corrupt_or_discontinuous_evidence_fails_stickily(detector,bad):
    tick=FIRST_WINDOW_TICK;t=tick*.002;pad,surface,angles=sample(t)
    if bad=='skip':t+=.002
    if bad=='repeat':t-=.002
    if bad=='pad_clock':pad['sim_time_s']-=.002
    if bad=='raw_start':pad['raw_evidence']['interval_start_s']-=.002
    if bad=='raw_end':pad['raw_evidence']['interval_end_s']-=.002
    if bad=='geometry_clock':pad['raw_evidence']['geometry_time_s']-=.002
    if bad=='surface_clock':surface['time_s']-=.002
    if bad=='dt':pad['physics_dt_s']=.004
    if bad=='nan_palm':surface['surface']['body_panel_forces_world_N']['lh_palm'][1]=float('nan')
    if bad=='nan_force':pad['contacts'][0]['normal_force_N']=float('nan')
    if bad=='nan_angles':angles['leaf']=float('nan')
    if bad=='missing_raw':pad.pop('raw_evidence')
    if bad=='malformed_flag':pad['valid_pad_grasp']='true'
    if bad=='profile_change':pad['grasp_profile']='distal-pad-v1'
    with pytest.raises(ValueError):detector.observe(t,pad,surface,angles)
    with pytest.raises(ValueError):detector.observe(tick*.002,*sample(tick*.002))
    r=detector.receipt()
    assert r['failure'] and not r['triggered'] and r['observed_samples']==tick-1


def test_actual_leaf_rotation_defines_palm_normal_and_zero_force_patch_is_unloaded(detector):
    t=FIRST_WINDOW_TICK*.002;pad,surface,angles=sample(t)
    rotation=Rotation.from_euler('z',.3);q=rotation.as_quat()
    surface['leaf_pose']=[0,0,0,*q[[3,0,1,2]]]
    surface['surface']['body_panel_forces_world_N']['lh_palm']=(-2.1*rotation.as_matrix()[:,1]).tolist()
    surface['surface']['palm_normal_load_N']=2.1
    surface['stance_status']='solved inaccurate'
    pad['contacts'].append(dict(digit='ff',normal_force_N=0.,pad_qualified=False))
    pad['raw_evidence']['contacts'].append({'synthetic_zero_load':True})
    pad['active_contact_count']=6;pad['raw_evidence']['active_contact_count']=6
    before=copy.deepcopy((pad,surface,angles))
    assert not detector.observe(t,pad,surface,angles)
    assert detector.receipt()['window_samples']==1 and all(detector.receipt()['checks'].values())
    assert (pad,surface,angles)==before


def test_palm_vector_with_wrong_normal_sign_does_not_supply_support(detector):
    t=FIRST_WINDOW_TICK*.002;pad,surface,angles=sample(t)
    surface['surface']['body_panel_forces_world_N']['lh_palm']=[0,2.,0]
    surface['surface']['palm_normal_load_N']=0.
    assert not detector.observe(t,pad,surface,angles)
    assert detector.receipt()['failed_checks']==['left_palm_support']


@pytest.mark.parametrize('leaf,operator,latch',[(.075,.05,.001),(.10,-.05,-.001)])
def test_original_mechanism_limits_are_inclusive(detector,leaf,operator,latch):
    t=FIRST_WINDOW_TICK*.002;pad,surface,_=sample(t)
    detector.observe(t,pad,surface,dict(leaf=leaf,operator=operator,latch=latch))
    assert detector.receipt()['window_samples']==1


@pytest.mark.parametrize('start,dt',[(0.,.002),(-1.,.002),(.003,.002),(.002,.004),(float('nan'),.002),(True,.002)])
def test_no_alternative_clock_or_unaligned_start(start,dt):
    with pytest.raises(ValueError):TransferRestStop(start,dt)


def test_original_28_second_start_earliest_window_math():
    r=TransferRestStop(28.).receipt()
    assert r['minimum_terminal_time_s']==36.5 and r['start_seconds']==28.
    assert r['window_samples']==0 and r['maximum_window_samples']==0

"""Offline study evidence must not mix clocks, zero buffers or outcome labels."""
import copy

import numpy as np
import pytest
import mujoco

from scripts.dexterous.build_jev_contact_adjustment_study import (
    DIGITS, assert_payload_boundary, material_point_sensitivity, patch_geometry, reduce_row,
)


def actual_row():
    contacts=[]
    for digit,angle in zip(DIGITS,[0.,.1,-.1,.2,2.22]):
        point=[np.cos(angle)*.007,np.sin(angle)*.007,0.]
        contacts.append(dict(body='/World/H1/rh_'+digit+'distal',position=point,
            normal=[np.cos(angle),np.sin(angle),0.],normal_force_N=1.))
    # Explicit zero-force contact-buffer entry is not a loaded patch.
    contacts.append(dict(contacts[0],normal_force_N=0.))
    transforms={c['body']:[0.,0.,0.,0.,0.,0.,1.] for c in contacts}
    return dict(sim_time_s=.002,physics_dt_s=.002,hand='rh',grasp_profile='volar-phalange-v1',
        digit_forces_N={d:1. for d in DIGITS},valid_pad_grasp=False,
        raw_evidence=dict(clock='physx-interval-end',scope='complete-handle-body',
            interval_start_s=0.,interval_end_s=.002,geometry_time_s=.002,
            contacts=contacts,body_transforms_xyzw=transforms,
            lever=dict(center=[0.,0.,0.],axis=[0.,0.,1.],half_length=.053,radius=.007)))


def test_actual_mean_can_hide_a_loaded_individual_pair_failure():
    reduced=reduce_row(actual_row(),1)
    assert reduced['thumb_vs_mean_finger_centroid_radial_dot']<-.5
    assert reduced['minimum_finger_centroid_dot_to_mean']>.5
    assert max(reduced['thumb_finger_radial_dot'].values())>-.5
    assert reduced['digits']['ff']['normal_load_N']==1.
    assert reduced['digits']['ff']['positive_patch_count']==1
    assert reduced['digits']['ff']['zero_force_buffer_patch_count']==1


@pytest.mark.parametrize('change',['geometry_epoch','interval','index','load','profile','recorded_score'])
def test_changed_clock_force_profile_or_score_is_rejected(change):
    row=actual_row();index=1
    if change=='geometry_epoch':row['raw_evidence']['geometry_time_s']=0.
    if change=='interval':row['raw_evidence']['interval_start_s']=-.002
    if change=='index':index=2
    if change=='load':row['digit_forces_N']['mf']=4.
    if change=='profile':row['grasp_profile']='distal-pad-v1'
    if change=='recorded_score':row['maximum_thumb_finger_dot']=1.
    with pytest.raises(ValueError):reduce_row(row,index)


def test_reset_buffer_is_explicitly_not_a_solved_interval():
    row=reduce_row(dict(sim_time_s=0.),0)
    assert 'digits' not in row
    assert 'excluded from every case' in row['source_note']


def test_actual_body_transform_and_normal_sign_have_no_verdicts():
    contact=dict(body='/World/H1/rh_ffmiddle',position=[1.,2.007,3.],normal=[0.,1.,0.],normal_force_N=2.)
    result=patch_geometry(contact,[1.,2.,3.,0.,0.,0.,1.],
        dict(center=[1.,2.,3.],axis=[0.,0.,1.],half_length=.053))
    np.testing.assert_allclose(result['position_body_m'],[0.,.007,0.],atol=1e-12)
    np.testing.assert_allclose(result['outward_normal_body'],[0.,-1.,0.])
    assert result['inward_radial_normal_alignment']==1.
    assert result['lever_side_axial_clearance_m']==.053
    assert not {'passed','pad_qualified','expected'}&result.keys()


@pytest.mark.parametrize('leak',['future','passed','source_run','source_valid_pad_grasp','expected'])
def test_nested_evaluator_information_is_not_inference_input(leak):
    payload=dict(state=dict(history=[dict(relative_time_s=-.2),dict(relative_time_s=0.)]))
    assert_payload_boundary(payload)
    changed=copy.deepcopy(payload);changed['state']['history'][0][leak]=False
    with pytest.raises(ValueError):assert_payload_boundary(changed)


@pytest.mark.parametrize('bad_time',[.002,-.202])
def test_future_or_wrong_history_window_is_rejected(bad_time):
    payload=dict(state=dict(history=[dict(relative_time_s=bad_time),dict(relative_time_s=0.)]))
    with pytest.raises(ValueError):assert_payload_boundary(payload)


def simple_measured_fk_scene():
    children=[];names=[]
    for i in range(69):
        name='joint_'+str(i);names.append(name)
        body='rh_'+DIGITS[i]+'distal' if i<5 else 'unused_'+str(i)
        children.append(f'<body name="{body}" pos="{i*.001} 0 0"><joint name="{name}" axis="0 0 1"/><geom type="sphere" size=".001" mass=".01"/></body>')
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="pelvis"><freejoint name="free_base"/><geom type="sphere" size=".01" mass="1"/>'+''.join(children)+'</body></worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
    transforms={};contacts=[]
    for i,digit in enumerate(DIGITS):
        short='rh_'+digit+'distal';body=model.body(short).id
        transforms[short]=np.r_[data.xpos[body],data.xquat[body][[1,2,3,0]]].tolist()
        contacts.append(dict(body=short,position=(data.xpos[body]+[.007,.002,0.]).tolist(),normal=[1.,0.,0.],normal_force_N=1.))
    configuration=dict(root_state_convention='actor-origin pose and world actor-origin linear/angular velocity',robot_joint_names=names)
    physics=dict(root=np.array([np.r_[data.qpos[:7],np.zeros(6)]]),joints=np.zeros((1,69)))
    now=dict(raw_loaded_contacts=contacts,loaded_body_transforms_xyzw=transforms,
        lever=dict(center=[0.,0.,0.],axis=[0.,0.,1.],radius=.007,half_length=.053))
    return model,configuration,physics,now


def test_material_probe_is_zero_step_and_preserves_other_material_points(monkeypatch):
    model,cfg,physics,now=simple_measured_fk_scene()
    def forbidden(*args,**kwargs):raise AssertionError('No physics stepping in archived study')
    monkeypatch.setattr(mujoco,'mj_step',forbidden)
    before=physics['joints'].copy()
    result=material_point_sensitivity(model,cfg,physics,1,now,{'candidate':dict(joint='joint_0',reference_delta_rad=.005)})
    np.testing.assert_array_equal(physics['joints'],before)
    candidate=result['candidates']['candidate']
    assert np.linalg.norm(candidate['frozen_patch_centroid_shift_world_m']['ff'])>0.
    for digit in DIGITS[1:]:np.testing.assert_array_equal(candidate['frozen_patch_centroid_shift_world_m'][digit],[0.,0.,0.])
    assert result['material_point_reconstruction_max_error_m']<1e-12
    assert 'not to an archived motor reference' in result['scope']


def test_foreign_actual_body_pose_is_rejected_before_static_probe():
    model,cfg,physics,now=simple_measured_fk_scene()
    now['loaded_body_transforms_xyzw']['rh_ffdistal'][0]+=.001
    with pytest.raises(ValueError,match='reproduce actual same-epoch'):
        material_point_sensitivity(model,cfg,physics,1,now,{'candidate':dict(joint='joint_0',reference_delta_rad=.005)})

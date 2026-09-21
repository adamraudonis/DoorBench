"""Unstepped toy geometry and synthetic binding tests; no physical source claim."""
import copy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from doorbench.dexterous.isaac_release_geometry_audit import (
    audit_geometry, validate_candidate_binding, ORIGINAL_LIMITS,
)
from doorbench.dexterous.qualified_isaac_grasp import digest


def geometry_fixture(*, extra_handle_touch=False, lever_touch=False):
    """A small unstepped model tests the auditor, not an H1 task rollout."""
    palm_collision = '1' if lever_touch else '0'
    lever_position = '-1.9 0 0' if lever_touch else '0 0 0'
    hub = '<geom name="hub" type="sphere" pos="-1.85 0 .01" size=".015"/>' if extra_handle_touch else ''
    xml = f'''<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body name="leaf" pos="2 0 1"><joint name="leaf_hinge" range="0 1"/>
        <geom type="sphere" size=".01" contype="0" conaffinity="0"/>
        <body name="leaf_handle"><geom name="leaf_handle_lever_col_n" type="capsule"
          pos="{lever_position}" size=".01 .04"/>{hub}</body></body>
      <body name="robot/torso_link" pos="0 0 1"><freejoint name="robot/free_base"/>
        <geom type="sphere" size=".03" contype="0" conaffinity="0"/>
        <body name="robot/left_ankle_link" pos="-.1 0 -.95"><geom size=".01" contype="0" conaffinity="0"/></body>
        <body name="robot/right_ankle_link" pos=".1 0 -.95"><geom size=".01" contype="0" conaffinity="0"/></body>
        <body name="robot/lh_palm" pos="-.2 0 0"><geom size=".01" contype="0" conaffinity="0"/>
          <site name="robot/lh_palm_touch"/></body>
        <body name="robot/arm" pos="-.3 0 0"><joint name="robot/right_shoulder_pitch" range="-1 1"/>
          <geom size=".01" contype="0" conaffinity="0"/></body>
        <body name="robot/rh_palm" pos=".1 0 0"><geom size=".01" contype="{palm_collision}" conaffinity="{palm_collision}"/>
          <site name="robot/rh_palm_touch"/>
          <body name="robot/rh_ffdistal" pos=".05 0 0"><joint name="robot/rh_FFJ1" range="0 1"/>
            <geom type="capsule" fromto="0 0 .002 0 0 .02" size=".005"/></body></body>
      </body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model,data)
    scene = SimpleNamespace(m=model,d=data,root=int(model.joint('robot/free_base').qposadr[0]))
    q = data.qpos.copy()
    hand = model.site('robot/rh_palm_touch').id
    def row(time):
        return dict(time_s=time,qpos=q.tolist(),palm_position=data.site_xpos[hand].tolist(),
            palm_rotation=data.site_xmat[hand].reshape(3,3).tolist(),
            joints={'right_shoulder_pitch':float(q[model.joint('robot/right_shoulder_pitch').qposadr[0]])},
            finger_joints={'rh_FFJ1':float(q[model.joint('robot/rh_FFJ1').qposadr[0]])})
    candidate=dict(grasp_profile='volar-phalange-v1',trials=[dict(rows=[row(0.),row(8.)])])
    return scene,q,candidate


def test_unstepped_toy_path_is_sampled_2001_times_without_force_success_claim():
    scene,q,candidate=geometry_fixture()
    before=copy.deepcopy(candidate)
    result=audit_geometry(scene,candidate,q,duration_s=16.)
    assert result['passed'] and result['samples']==2001
    assert result['physics_steps']==0 and result['active_state_writes']==0
    assert not result['physical_contact_qualification'] and not result['delivered_motor_force_checked']
    assert not result['motor_force_feasibility_inferred']
    assert result['original_thresholds']==ORIGINAL_LIMITS
    assert candidate==before


@pytest.mark.parametrize('change', ['position','orientation','joint_limit','speed','hub','anatomy'])
def test_original_geometric_failure_gates_cannot_be_hidden_by_audit_claim(change):
    scene,q,candidate=geometry_fixture(extra_handle_touch=change=='hub',lever_touch=change=='anatomy')
    model=scene.m
    row=candidate['trials'][0]['rows'][-1]
    duration=16.
    if change=='position': row['qpos'][scene.root]+=.01
    if change=='orientation': row['qpos'][scene.root+3:scene.root+7]=[np.cos(.02),0,0,np.sin(.02)]
    if change in ('joint_limit','speed'):
        value=.5 if change=='speed' else 1.2
        row['qpos'][model.joint('robot/rh_FFJ1').qposadr[0]]=value
        row['finger_joints']['rh_FFJ1']=value
        if change=='speed': duration=.1
    candidate['passed']=True
    candidate['old_native_audit']={'passed':True,'samples':2001}
    result=audit_geometry(scene,candidate,q,duration_s=duration)
    expected={'position':'foot_and_palm_position','orientation':'foot_and_palm_orientation',
        'joint_limit':'finite_joint_and_tendon_limits','speed':'joint_reference_speed',
        'hub':'whole_handle_geometry','anatomy':'selected_right_anatomy'}[change]
    assert not result['passed'] and not result['checks'][expected]


@pytest.mark.parametrize('change', ['first','clock','quaternion','rotation','named_target','missing_joint','door','nan'])
def test_incomplete_or_reinterpreted_path_fails_before_geometry(change):
    scene,q,candidate=geometry_fixture()
    rows=candidate['trials'][0]['rows']
    if change=='first': rows[0]['qpos'][scene.root]=np.nextafter(q[scene.root],1.)
    if change=='clock': rows[1]['time_s']=0.
    if change=='quaternion': rows[1]['qpos'][scene.root+3]=1.1
    if change=='rotation': rows[1]['palm_rotation'][0][0]=2.
    if change=='named_target': rows[1]['finger_joints']['rh_FFJ1']=.1
    if change=='missing_joint': rows[1]['joints']={}
    if change=='door': rows[1]['qpos'][scene.m.joint('leaf_hinge').qposadr[0]]=.1
    if change=='nan': rows[1]['qpos'][scene.root]=float('nan')
    with pytest.raises(ValueError): audit_geometry(scene,candidate,q,duration_s=16.)


@pytest.mark.parametrize('duration', [0.,-1.,float('inf'),float('nan'),True])
def test_no_invalid_duration_can_skip_reference_speed_check(duration):
    scene,q,candidate=geometry_fixture()
    with pytest.raises(ValueError): audit_geometry(scene,candidate,q,duration_s=duration)


@pytest.fixture
def bound(tmp_path):
    archive=tmp_path/'acquisition-physics.npz'
    archive.write_bytes(b'synthetic binding fixture only')
    hashes={str(archive):digest(archive)}
    admission=dict(grasp_profile='volar-phalange-v1',input_sha256=hashes,synthetic=True)
    context=SimpleNamespace(admission=admission,sha256='a'*64,qpos=np.array([1.,2.]),
        terminal_time_s=42.,state_archive_path=archive,verify_inputs=lambda:None)
    candidate=dict(schema='doorbench.isaac-profiled-release-candidate.v1',source_engine='isaac-physx',
        physics_steps=0,source_sample_playback=0,grasp_profile='volar-phalange-v1',
        source_context_sha256=context.sha256,source_admission=copy.deepcopy(admission),
        initial_time_s=42.,initial_qpos=[1.,2.],input_sha256=hashes.copy())
    return candidate,context,archive


def test_actual_physics_archive_binding_is_required(bound):
    candidate,context,_=bound
    validate_candidate_binding(candidate,context)
    assert 'trajectory.npz' not in str(candidate)


@pytest.mark.parametrize('change', ['old_schema','native_engine','qualification','context','epoch','state','hash','file'])
def test_old_native_labels_cannot_supply_missing_actual_isaac_evidence(bound,change):
    candidate,context,archive=bound
    candidate['old_native_audit']={'passed':True}
    if change=='old_schema': candidate['schema']='old-standing-audit'
    if change=='native_engine': candidate['source_engine']='native-mujoco'
    if change=='qualification': candidate['source_admission']['synthetic']=False
    if change=='context': candidate['source_context_sha256']='b'*64
    if change=='epoch': candidate['initial_time_s']=41.998
    if change=='state': candidate['initial_qpos'][0]=np.nextafter(1.,2.)
    if change=='hash': candidate['input_sha256'].clear()
    if change=='file': archive.write_bytes(b'changed')
    with pytest.raises(ValueError): validate_candidate_binding(candidate,context)

"""Synthetic admission and unstepped toy geometry; no physical source claim."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from doorbench.dexterous import isaac_coupled_release_geometry as module
from doorbench.dexterous.coupled_release_geometry import CoupledReleaseGeometry
from doorbench.dexterous.qualified_isaac_grasp import digest
from test_withdrawal_source_context import isaac,load_isaac


def write(path,value):
    path.write_text(json.dumps(value))


@pytest.fixture
def bound(isaac,tmp_path):
    data=load_isaac(isaac).data
    config=tmp_path/'withdrawal.json';write(config,isaac['config'])
    plan=dict(schema=module.SCHEMA,source_engine='isaac-physx',
        source_admission=copy.deepcopy(data['source_admission']),
        source_context_sha256=data['source_context_sha256'],source_state_sha256=data['source_state_sha256'],
        initial_episode_time_s=data['start_time_s'],initial_qpos=data['initial_qpos'],duration_s=data['duration_s'],
        grasp_profile=data['source_admission']['grasp_profile'],motor_contract_sha256=data['motor_contract_sha256'],
        physics_steps=0,source_sample_playback=0,active_state_writes=0,authorized_stages=0,
        physical_admission=False,runtime_route_exported=False,source_config=str(config),
        right_hand_candidate=data['screen_path'],right_hand_dense_audit=data['audit_path'],
        source_physics_archive=data['source_archive_path'],input_sha256={**data['input_sha256'],str(config):digest(config)})
    return plan,data,config


def test_new_binding_accepts_only_the_actual_revalidated_detached_context(bound):
    plan,data,config=bound
    hashes=module.validate_envelope_binding(plan,data,config)
    assert hashes[str(config)]==digest(config)
    assert hashes[data['source_archive_path']]==digest(data['source_archive_path'])
    assert plan['authorized_stages']==0 and not plan['runtime_route_exported']


@pytest.mark.parametrize('change',['schema','engine','qualification','context','state','epoch','qpos','duration',
    'profile','motor','physics','playback','plant_write','authorization','physical','runtime',
    'config_path','candidate_path','audit_path','archive_path','missing_hash','changed_file'])
def test_cannot_relabel_a_native_envelope_or_replace_actual_evidence(bound,change):
    plan,data,config=bound
    replacements=dict(schema=('schema','doorbench.coupled-release-envelope.v1'),engine=('source_engine','native-mujoco'),
        context=('source_context_sha256','c'*64),state=('source_state_sha256','c'*64),epoch=('initial_episode_time_s',41.998),
        duration=('duration_s',15.),profile=('grasp_profile','distal-pad-v1'),motor=('motor_contract_sha256','c'*64),
        physics=('physics_steps',1),playback=('source_sample_playback',1),plant_write=('active_state_writes',1),
        authorization=('authorized_stages',1),physical=('physical_admission',True),runtime=('runtime_route_exported',True),
        config_path=('source_config','another.json'),candidate_path=('right_hand_candidate','another.json'),
        audit_path=('right_hand_dense_audit','another.json'),archive_path=('source_physics_archive','trajectory.npz'))
    if change in replacements:
        key,value=replacements[change];plan[key]=value
    if change=='qualification':plan['source_admission']['source_qualification']['passed']=False
    if change=='qpos':plan['initial_qpos']=list(data['initial_qpos']);plan['initial_qpos'][0]=np.nextafter(plan['initial_qpos'][0],1.)
    if change=='missing_hash':plan['input_sha256'].pop(data['source_archive_path'])
    if change=='changed_file':Path(data['source_archive_path']).write_bytes(b'changed synthetic archive')
    with pytest.raises(ValueError):module.validate_envelope_binding(plan,data,config)


def map_fixture():
    initial=np.r_[.091,0.,0.,np.linspace(-.1,.1,len(module.JOINT_NAMES))]
    qa=np.arange(3,len(initial))
    times=np.linspace(0,16,81);angles=np.sort(np.unique(np.r_[np.linspace(.08,.4,17),initial[0]]))
    coordinates=np.tile(np.r_[np.zeros(6),initial[qa]],(len(times),len(angles),1))
    plan=dict(joint_names=list(module.JOINT_NAMES),elapsed_s=times.tolist(),leaf_rad=angles.tolist(),
        coordinates=coordinates.tolist(),duration_s=16.,operator_reference_rad=0.,operator_envelope_rad=[-.01,.01],
        follow_handle_through_route_seconds=3.5,admitted_leaf_upper_nodes=[[0,.10],[16,.4]])
    return plan,initial,qa


def test_raw_map_source_point_is_checked_before_exact_evaluation_can_mask_it():
    plan,initial,qa=map_fixture()
    times,angles,values=module.validate_map(plan,initial,qa,0,1,2)
    np.testing.assert_array_equal(values[0,list(angles).index(initial[0])],np.r_[np.zeros(6),initial[qa]])
    assert len(times)==81 and len(angles)==18


@pytest.mark.parametrize('change',['joint_order','coarse_time','coarse_angle','unordered','nan','shape','duration',
    'source_node','source_coordinate','operator','operator_reference','latch','follow_nan','follow_end','domain_start','domain_end'])
def test_map_cannot_hide_missing_source_or_expand_measured_domain(change):
    plan,initial,qa=map_fixture()
    if change=='joint_order':plan['joint_names'].reverse()
    if change=='coarse_time':plan['elapsed_s']=plan['elapsed_s'][::2];plan['coordinates']=plan['coordinates'][::2]
    if change=='coarse_angle':plan['leaf_rad']=plan['leaf_rad'][::2];plan['coordinates']=[row[::2] for row in plan['coordinates']]
    if change=='unordered':plan['elapsed_s'][1]=0.
    if change=='nan':plan['coordinates'][20][4][0]=float('nan')
    if change=='shape':plan['coordinates'][0].pop()
    if change=='duration':plan['duration_s']=float('inf')
    if change=='source_node':initial[0]=np.nextafter(initial[0],1.)
    if change=='source_coordinate':plan['coordinates'][0][plan['leaf_rad'].index(initial[0])][0]=1e-15
    if change=='operator':plan['operator_envelope_rad']=[-.051,.01]
    if change=='operator_reference':plan['operator_reference_rad']=.001
    if change=='latch':initial[2]=.00101
    if change=='follow_nan':plan['follow_handle_through_route_seconds']=float('nan')
    if change=='follow_end':plan['follow_handle_through_route_seconds']=8.
    if change=='domain_start':plan['admitted_leaf_upper_nodes'][0][1]=.09
    if change=='domain_end':plan['admitted_leaf_upper_nodes'][-1][1]=.400001
    with pytest.raises(ValueError):module.validate_map(plan,initial,qa,0,1,2)


def toy_scene():
    """Named scalar toy checks the adapter's mechanics, never task performance."""
    bodies=[]
    for name in module.JOINT_NAMES:
        body='robot/torso_link' if name=='torso' else 'robot/'+name+'_body'
        hand='lh' if name=='lh_WRJ1' else 'rh' if name=='rh_WRJ1' else None
        hand_x='-.2' if hand=='lh' else '.2'
        palm=(f'<body name="robot/{hand}_palm" pos="{hand_x} 0 0"><geom size=".01"/><site name="robot/{hand}_palm_touch"/></body>' if hand else '')
        bodies.append(f'<body name="{body}"><joint name="robot/{name}" range="-2 2"/><geom size=".001" contype="0" conaffinity="0"/>{palm}</body>')
    xml='''<mujoco><compiler angle="radian"/><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
      <body name="leaf" pos="2 0 1"><joint name="leaf_hinge" range="0 1"/><geom size=".02"/>
        <body name="leaf_handle" pos=".1 0 0"><joint name="leaf_handle_hinge" range="-.1 1"/>
          <geom name="leaf_handle_lever_col_n" type="capsule" size=".01 .04"/>
          <body name="bolt"><joint name="leaf_latch_bolt_slide" type="slide" range="-.01 .01"/><geom size=".005"/></body>
        </body></body><body name="robot/base" pos="0 0 1"><freejoint name="robot/free_base"/><geom size=".01" contype="0" conaffinity="0"/>
      <body name="robot/left_ankle_link" pos="-.1 0 -.98"><geom size=".01" contype="0" conaffinity="0"/></body>
      <body name="robot/right_ankle_link" pos=".1 0 -.98"><geom size=".01" contype="0" conaffinity="0"/></body>
      '''+''.join(bodies)+'''</body></worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    d.qpos[m.joint('leaf_hinge').qposadr[0]]=.091
    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    return SimpleNamespace(m=m,d=d,root=int(m.joint('robot/free_base').qposadr[0]))


def test_adapter_uses_unchanged_evaluator_and_never_native_environment(monkeypatch,bound,tmp_path):
    plan,data,config=bound;scene=toy_scene();m,d=scene.m,scene.d
    initial=d.qpos.copy();qa=m.jnt_qposadr[[m.joint('robot/'+n).id for n in module.JOINT_NAMES]]
    mp,_,_=map_fixture();mp['coordinates']=np.tile(np.r_[np.zeros(6),initial[qa]],(81,18,1)).tolist()
    plan.update(mp);plan['initial_qpos']=initial.tolist();data['initial_qpos']=initial.tolist()
    rh=m.site('robot/rh_palm_touch').id
    names=module.JOINT_NAMES
    row=dict(qpos=initial.tolist(),palm_position=d.site_xpos[rh].tolist(),palm_rotation=d.site_xmat[rh].reshape(3,3).tolist(),
        joints={n:float(initial[m.joint('robot/'+n).qposadr[0]]) for n in names},finger_joints={'rh_FFJ1':0.})
    # Real _path_arrays is independently tested; this toy has no finger chain.
    data['screen']=dict(trials=[dict(rows=[dict(time_s=0.,**row),dict(time_s=8.,**row)])])
    monkeypatch.setattr(module,'_path_arrays',lambda *args:(np.array([0.,.5,8.5]),np.tile(initial,(3,1)),
        np.tile(d.site_xpos[rh],(3,1)),np.tile(d.site_xmat[rh].reshape(3,3),(3,1,1))))
    calls=[]
    def load(config,motors,*,measured_rest):
        calls.append(measured_rest);return SimpleNamespace(data=copy.deepcopy(data))
    monkeypatch.setattr(module,'load_withdrawal_source_context',load)
    monkeypatch.setattr(module,'LandedLeftScene',lambda *args:scene)
    monkeypatch.setattr(CoupledReleaseGeometry,'__init__',lambda *args:pytest.fail('Native constructor invoked'))
    path=tmp_path/'envelope.json';write(path,plan)
    model=module.IsaacCoupledReleaseGeometry(path,source_config=config)
    assert calls==[True]
    for name in ('evaluate','coordinates','upper_angle','_correct_arm'):
        assert getattr(module.IsaacCoupledReleaseGeometry,name) is getattr(CoupledReleaseGeometry,name)
    result=model.evaluate(0.,initial[model.lq],initial[model.oq],initial[model.bq])
    np.testing.assert_array_equal(result['qpos'],initial)
    assert not hasattr(model,'sim') and model.scene is scene
    np.testing.assert_array_equal(model.c['initial_feet_positions'],d.xpos[[m.body('robot/'+s+'_ankle_link').id for s in ('left','right')]])
    model.close()
    path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):model.verify_inputs()


def test_failed_actual_source_admission_propagates_before_any_geometry(monkeypatch,bound,tmp_path):
    plan,_,config=bound;path=tmp_path/'envelope.json';write(path,plan)
    def fail(*args,**kwargs):raise ValueError('Actual PhysX source qualification failed')
    monkeypatch.setattr(module,'load_withdrawal_source_context',fail)
    monkeypatch.setattr(module,'LandedLeftScene',lambda *args:pytest.fail('Geometry constructed before source qualification'))
    with pytest.raises(ValueError,match='source qualification failed'):
        module.IsaacCoupledReleaseGeometry(path,source_config=config)

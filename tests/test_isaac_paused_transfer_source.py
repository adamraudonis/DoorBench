"""Synthetic identity scaffold tests; no successful physical source is invented."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from doorbench.dexterous.isaac_paused_transfer_source import (
    SCHEMA, SOURCE_KIND, inspect_paused_isaac_transfer_snapshot,
    validate_live_pause_anchor,
)
from doorbench.dexterous.isaac_attained_state import RECORDED_ROOT_CONVENTION
from doorbench.dexterous.standing_body_record import PLANNER_BODIES, POSE_CONVENTION


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def snapshot(tmp_path):
    folder = tmp_path/'pause'; folder.mkdir()
    assets = tmp_path/'assets'; assets.mkdir()
    robot = assets/'robot.xml'; robot.write_text('synthetic geometry identity, not a model')
    door = assets/'door.usd'; door.write_text('synthetic USD identity')
    definition = assets/'volar.json'
    definition.write_text(json.dumps(dict(profile='volar-phalange-v1', robot_xml_sha256=sha(robot))))
    original = assets/'isaac_opening.py'; original.write_text('# synthetic captured producer\n')
    (folder/'source-isaac_opening.py').write_bytes(original.read_bytes())
    joints = ['j'+str(i) for i in range(69)]
    motor_names = ['m'+str(i) for i in range(61)]
    motors = dict(source_xml_sha256=sha(robot), joint_names=joints,
        actuators=[dict(name=n,force_range=[-2.,2.]) for n in motor_names])
    configuration = dict(dt=.002,runtime_pose_writes=0,direct_door_commands=False,
        robot_joint_names=joints,door_joint_names=['leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide'],
        root_state_convention=RECORDED_ROOT_CONVENTION,
        standing_planner_body_names=list(PLANNER_BODIES),standing_planner_body_pose_convention=POSE_CONVENTION,
        args=dict(native_robot=str(robot),door_usd=str(door),grasp_profile_definition=str(definition),
            acquisition=True,operate_after_acquisition=True,acquisition_stance_profile='landed-foot-v1',
            grasp_profile='volar-phalange-v1',standing_transfer_route='synthetic-unadmitted-route'))
    documents = dict(configuration=configuration,motor_contract=motors,
        provenance=dict(files={str(p):sha(p) for p in (robot,door,definition,original)}),
        pad_steps=[],transfer_steps=[],rest_stop=dict(triggered=False),
        phase_report=dict(schema='synthetic-phase-declaration',passed=False),
        mechanical_audit={},reset_state={},transfer_route={},source_prefix_witness={},observer_state={},
        grasp_profile_definition=json.loads(definition.read_text()))
    files = {}
    for name, value in documents.items():
        path=folder/(name+'.json'); path.write_text(json.dumps(value))
        files[name]=dict(path=str(path),sha256=sha(path))
    count=251
    root=np.zeros((count,13),np.float32);root[:,3]=1
    bodies=np.zeros((count,6,7),np.float32);bodies[:,:,3]=1
    physics=dict(time_s=np.arange(1,count+1)*.002,root=root,
        joints=np.zeros((count,69),np.float32),joint_velocity=np.zeros((count,69),np.float32),
        motor_forces=np.zeros((count,61),np.float64),door=np.zeros((count,3),np.float32),
        door_velocity=np.zeros((count,3),np.float32),standing_body_poses=bodies,torso_tilt_deg=np.zeros(count))
    archive=folder/'core.npz';np.savez(archive,**physics)
    files['physics']=dict(path=str(archive),sha256=sha(archive))
    command=physics['motor_forces'][-1]
    value=dict(schema=SCHEMA,source_kind=SOURCE_KIND,source_engine='isaac-physx',
        live_pause=dict(pause_token='a'*64,episode_id='synthetic-process-1',controller_identity='b'*64,
            step_index=count,epoch_s=count*.002,physics_dt_s=.002,
            measurement_fingerprint='c'*64,controller_fingerprint='d'*64),
        closed_prefix=True,episode_complete=False,files=files,
        terminal_command=dict(command_time_s=(count-1)*.002,post_step_time_s=count*.002,
            motor_names=motor_names,dtype=command.dtype.str,shape=[61],
            bytes_sha256=hashlib.sha256(command.tobytes()).hexdigest(),values=command.tolist()))
    path=folder/'snapshot.json'
    def save(): path.write_text(json.dumps(value))
    def write_role(name, document):
        p=Path(value['files'][name]['path']);p.write_text(json.dumps(document));value['files'][name]['sha256']=sha(p);save()
    def write_physics():
        np.savez(archive,**physics);value['files']['physics']['sha256']=sha(archive);save()
    save()
    return dict(path=path,value=value,physics=physics,save=save,write_role=write_role,
        write_physics=write_physics,configuration=configuration,folder=folder)


def test_core_inspection_is_never_phase_qualification(snapshot):
    result=inspect_paused_isaac_transfer_snapshot(snapshot['path'])
    receipt=result.inspection
    assert receipt['core_intervals']==251 and receipt['core_identity_checked']
    assert receipt['terminal_command']['command_time_s']==.5
    assert receipt['live_pause']['epoch_s']==.502
    for name in ('phase_qualified','raw_contacts_audited','rest_window_audited',
                 'material_frames_admitted','geometry_admitted','live_process_state_verified','resume_authorized'):
        assert receipt[name] is False
    assert receipt['authorized_stages']==receipt['physical_steps']==receipt['active_state_writes']==0
    # Empty contacts/failed phase declaration intentionally do not become proof.
    assert not hasattr(result,'qpos') and not hasattr(result,'scene')
    receipt['live_pause']['epoch_s']=123
    assert result.terminal_time_s==.502
    result.verify_inputs()


@pytest.mark.parametrize('key,value',[
    ('schema','doorbench.completed-run.v1'),('source_kind','native-mujoco'),
    ('source_engine','native-mujoco'),('closed_prefix',False),('closed_prefix',1),
    ('episode_complete',True),('episode_complete',0),('passed',True),
])
def test_wrong_source_scope_fails(snapshot,key,value):
    snapshot['value'][key]=value;snapshot['save']()
    with pytest.raises(ValueError):inspect_paused_isaac_transfer_snapshot(snapshot['path'])


@pytest.mark.parametrize('key,value',[
    ('step_index',True),('step_index',250),('epoch_s',.504),('epoch_s',float('nan')),
    ('physics_dt_s',.004),('episode_id',' '),('pause_token','A'*64),
    ('measurement_fingerprint','bad'),('controller_identity',123),('unknown',True),
])
def test_anchor_is_strict(snapshot,key,value):
    anchor=dict(snapshot['value']['live_pause']);anchor[key]=value
    with pytest.raises(ValueError):validate_live_pause_anchor(anchor)


@pytest.mark.parametrize('change',['missing','samefile','outside','wronghash','fake_completed'])
def test_evidence_inventory(snapshot,tmp_path,change):
    files=snapshot['value']['files']
    if change=='missing':del files['pad_steps']
    elif change=='samefile':files['pad_steps']=files['transfer_steps'].copy()
    elif change=='outside':
        p=tmp_path/'outside.json';p.write_text('{}');files['pad_steps']=dict(path=str(p),sha256=sha(p))
    elif change=='wronghash':files['pad_steps']['sha256']='0'*64
    else:
        p=snapshot['folder']/'operation-report.json';p.write_text('{}');files['phase_report']=dict(path=str(p),sha256=sha(p))
    snapshot['save']()
    with pytest.raises(ValueError):inspect_paused_isaac_transfer_snapshot(snapshot['path'])


@pytest.mark.parametrize('change',['clock','clockdtype','missingfield','nan','cap','short','shape','quat'])
def test_actual_complete_core_required(snapshot,change):
    p=snapshot['physics']
    if change=='clock':p['time_s'][20]+=.002
    elif change=='clockdtype':p['time_s']=p['time_s'].astype(np.float32)
    elif change=='missingfield':del p['joint_velocity']
    elif change=='nan':p['root'][0,0]=np.nan
    elif change=='cap':p['motor_forces'][4,3]=2.00002
    elif change=='short':p['joints']=p['joints'][:-1]
    elif change=='shape':p['standing_body_poses']=p['standing_body_poses'][:,:5]
    else:p['root'][-1,3]=.9
    snapshot['write_physics']()
    with pytest.raises(ValueError):inspect_paused_isaac_transfer_snapshot(snapshot['path'])


@pytest.mark.parametrize('change',['time','epoch','order','dtype','shape','digest','value','bool'])
def test_exact_prior_command_required(snapshot,change):
    c=snapshot['value']['terminal_command']
    if change=='time':c['command_time_s']+=.002
    elif change=='epoch':c['post_step_time_s']-=.002
    elif change=='order':c['motor_names'].reverse()
    elif change=='dtype':c['dtype']='<f4'
    elif change=='shape':c['shape']=[60]
    elif change=='digest':c['bytes_sha256']='0'*64
    elif change=='value':c['values'][0]=1e-20
    else:c['values'][0]=False
    snapshot['save']()
    with pytest.raises(ValueError):inspect_paused_isaac_transfer_snapshot(snapshot['path'])


def test_captured_consumer_and_after_read_input_changes_fail(snapshot):
    result=inspect_paused_isaac_transfer_snapshot(snapshot['path'])
    p=snapshot['folder']/'source-isaac_opening.py';p.write_text('# changed\n')
    with pytest.raises(ValueError):result.verify_inputs()
    with pytest.raises(ValueError):inspect_paused_isaac_transfer_snapshot(snapshot['path'])


def test_duplicate_json_key_and_overflow_rejected(snapshot):
    p=snapshot['path'];text=p.read_text()
    p.write_text(text[:-1]+',"closed_prefix":true}')
    with pytest.raises(ValueError,match='Duplicate'):inspect_paused_isaac_transfer_snapshot(p)
    p.write_text(text.replace('0.502','1e309'))
    with pytest.raises(ValueError,match='Finite'):inspect_paused_isaac_transfer_snapshot(p)


def test_source_motor_and_original_profile_binding(snapshot):
    config=snapshot['configuration'];config['args']['grasp_profile']='distal-pad-v1'
    snapshot['write_role']('configuration',config)
    with pytest.raises(ValueError):inspect_paused_isaac_transfer_snapshot(snapshot['path'])

"""Synthetic admission seams and private coordinate arithmetic; no rollout proof."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doorbench.dexterous import isaac_panel_bridge_seed as module
from doorbench.dexterous import isaac_panel_planning as planner
from doorbench.dexterous import isaac_panel_source as source_module
from doorbench.dexterous.qualified_isaac_grasp import digest


def write(path, value):path.write_text(json.dumps(value,allow_nan=False))


def synthetic_scene():
    fingers = [side+'_'+digit+'J'+str(i) for side in ('lh','rh')
        for digit,count in (('FF',4),('MF',4),('RF',4),('LF',5),('TH',5)) for i in range(count,0,-1)]
    # Deliberately unlike panel, motor, and archive orders. Names, never indices,
    # must determine which previous nominal values survive the overlay.
    names = list(reversed(planner.JOINT_NAMES+fingers))
    body = ''.join(f'<body><joint name="robot/{n}" range="-3 3"/><geom size=".001"/></body>' for n in names)
    xml = ('<mujoco><compiler angle="radian"/><worldbody><body name="leaf" pos="2 0 1">'
        '<joint name="leaf_hinge" range="0 1.75"/><geom size=".01"/>'
        '<body><joint name="leaf_handle_hinge" range="0 1"/><geom size=".01"/></body>'
        '<body><joint name="leaf_latch_bolt_slide" type="slide" range="0 .01"/><geom size=".01"/></body>'
        '</body><body name="robot/base" pos="0 0 1"><freejoint name="robot/free_base"/><geom size=".01"/>'
        '<site name="robot/lh_palm_touch" pos="-.2 .1 .3"/><site name="robot/rh_palm_touch" pos=".2 .1 .3"/>'
        +body+'</body></worldbody></mujoco>')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    rq=int(m.joint('robot/free_base').qposadr[0])
    d.qpos[rq+3:rq+7]=Rotation.from_rotvec([.2,-.12,.4]).as_quat()[[3,0,1,2]]
    d.qpos[m.joint('leaf_hinge').qposadr[0]]=.15
    for i,n in enumerate(names):d.qpos[m.joint('robot/'+n).qposadr[0]]=.001*(i+1)
    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    return SimpleNamespace(m=m,d=d,rq=rq,names=names)


@pytest.fixture
def source(tmp_path,monkeypatch):
    """Physical and observational admission functions are explicit test seams.

    Real context construction and all candidate-integrity/chart/goal code run;
    no synthetic passing label is used as an actual-source experiment.
    """
    run=tmp_path/'synthetic-run';trial=run/'trial';trial.mkdir(parents=True)
    scene=synthetic_scene();q=scene.d.qpos.copy();v=scene.d.qvel.copy();names=scene.names
    runtime=dict(schema='doorbench.isaac-standing-withdrawal-runtime.v1',capture_returned_motor_command=True,
        inherit_transfer_support=True,left_arm_only=True,left_full_orientation=True,coupled_motion_projection='fixed-poses-v1')
    write(trial/'standing-withdrawal-runtime.json',runtime)
    poses=[]
    for i in range(10):poses.append([2.,0.,1.,*Rotation.from_rotvec([0.,0.,.06+.009*(i+1)]).as_quat()[[3,0,1,2]]])
    archive=trial/'acquisition-physics.npz'
    np.savez(archive,time_s=np.arange(1,11)*.002,standing_leaf_pose=np.array(poses))
    hashes={str(p):digest(p) for p in (archive,trial/'standing-withdrawal-runtime.json')}
    contract=dict(joint_names=list(reversed(names)))
    oldpose=[.001,-.002,.999,*Rotation.from_rotvec([.19,-.1,.39]).as_quat()[[3,0,1,2]]]
    previous=np.r_[[.002,-.003,.004,.013,-.012,.011],[q[scene.m.joint('robot/'+n).qposadr[0]]+.1 for n in names]]
    velocity=np.linspace(-.001,.001,75)
    rows=[]
    source_binding=dict(runtime_path=str(trial/'standing-withdrawal-runtime.json'),
        runtime_sha256=digest(trial/'standing-withdrawal-runtime.json'),motor_contract_sha256='a'*64,source_state_sha256='b'*64)
    localr=Rotation.from_rotvec([.1,.2,-.3]).as_matrix().tolist()
    left=dict(names=['torso',*['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')],'lh_WRJ2','lh_WRJ1'],
        last_update_s=.012,progress=1.,panel_palm_rotation=localr,normal_offset_m=.007,
        support_target_N=6.,hybrid_normal_target_N=5.999999720839327,filtered_palm_load_N=5.91)
    for i in range(3):
        t=(7+i)*.002
        rows.append(dict(command_time_s=t,post_step_time_s=(8+i)*.002,
            coupled=dict(joint_names=names.copy(),value=(previous+(i-2)*.002*velocity).tolist(),velocity=velocity.tolist(),
                root_coordinate_origin_xyz_wxyz=oldpose.copy(),time_s=t,coordinate_convention=module.CONVENTION),
            left=copy.deepcopy(left),right=dict(goal_position_world=[.4,.2,1.2],goal_rotation_world=np.eye(3).tolist(),
                palm={'offset':[.01,.02]},arm={'previous_target':[.4]}),
            stance=dict(root_bias=[.003,.002,-.001],rotation_bias=np.eye(3).tolist(),joint_bias=[.01],qp_warm_start=[.4]),
            support=dict(maximum_target_N=6.,previous_s=t),hand=dict(grip_preload_scale=0.,targets=[.42],preload=[0.]),
            withdrawal=dict(started_s=.004,release_started_s=.006),returned_motor_command=(np.arange(61)*.01).tolist()))
    motor_names=['synthetic-motor-'+str(i) for i in range(61)]
    handoff=dict(command_time_s=.020-.002,interval_end_s=.020,command_Nm=copy.deepcopy(rows[-1]['returned_motor_command']),
        motor_names=motor_names.copy(),motor_contract_sha256='a'*64,semantics='Submitted input, not delivered torque')
    admitted=dict(schema=planner.SOURCE_SCHEMA,source_engine='isaac-physx',source_run=str(run),source_time_s=.020,
        initial_qpos=q.tolist(),initial_qvel=v.tolist(),source_qualification={'passed':True,'state_sha256':'c'*64},
        coordinate_admission={'passed':True},input_sha256=hashes.copy(),authorized_stages=0,
        robot_path=str(tmp_path/'synthetic-robot.xml'),door_xml_path=str(tmp_path/'synthetic-door.xml'),door_usd_path=str(tmp_path/'synthetic-door.usd'),
        motor_contract=contract,motor_handoff=handoff,extracted_state={'binding':{'sha256':'c'*64}},
        actual_body_poses_xyz_wxyz={'leaf':poses[-1]})
    reference=dict(passed=True,reference_tail_accounting_passed=True,source_report_passed=True,source_run=str(run),source_trial=str(trial),
        source_terminal_time_s=.020,source_physics_sha256=digest(archive),input_sha256=hashes.copy(),tail=rows,source_binding=source_binding,
        observation_admission={'physical_intervals':10},static_controller_data=dict(motor_names=motor_names,left_path_names=left['names'],
            left_path_at_construction=[{'position':[.3,.04,.5]}],path_scope='Static authored left path; only its last nominal is replaced by the coupled reference. That current nominal is copied in every accepted row.'),
        diagnostics={'coordinate_joint_names':names,'synthetic_only':True},attained_tracking_contract={'hand':{'stiffness_scale':5.}},
        terminal_nominal_delta={'reference_command_epoch_s':.018,'actual_state_epoch_s':.020},
        support_profile_accounting={'profile':'release-unload-restore-v1','tail_active_targets_N':[left['hybrid_normal_target_N']]*3})
    calls=[]
    def admit(*args,**kwargs):calls.append(('physical',args,kwargs));return copy.deepcopy(admitted)
    def tail(*args,**kwargs):calls.append(('reference',args,kwargs));return copy.deepcopy(reference)
    monkeypatch.setattr(source_module,'admit_isaac_panel_source',admit)
    monkeypatch.setattr(planner,'LandedLeftScene',lambda *a:synthetic_scene())
    monkeypatch.setattr(module,'admit_standing_reference_tail',tail)
    monkeypatch.setattr(mujoco,'mj_step',lambda *a:pytest.fail('Seed helper stepped physics'))
    context=planner.admit_isaac_panel_context(run,robot=admitted['robot_path'],door_xml=admitted['door_xml_path'],door_usd=admitted['door_usd_path'])
    def fixed(m,d,base,options,progress=None):
        rows=[];lq=int(m.joint('leaf_hinge').qposadr[0])
        for angle in np.linspace(base[lq],options.target_aperture_rad,options.nodes):
            knot=base.copy();knot[lq]=angle
            rows.append(dict(leaf_angle_rad=float(angle),qpos=knot.tolist(),root_delta=[0.]*6,
                joint_targets={n:float(base[m.joint('robot/'+n).qposadr[0]]) for n in planner.JOINT_NAMES}))
        return rows,list(planner.JOINT_NAMES)
    monkeypatch.setattr(planner,'_solve_panel',fixed)
    candidate=planner.generate_panel_candidate(context,{'nodes':21})
    candidate_path=tmp_path/'synthetic-panel-candidate.json';write(candidate_path,candidate)
    calls.clear()
    return SimpleNamespace(run=run,trial=trial,admitted=admitted,reference=reference,candidate=candidate,candidate_path=candidate_path,
        archive=archive,scene=scene,calls=calls,poses=poses,runtime=runtime,context=context)


def prepare(s):
    return module.prepare_isaac_panel_bridge_seed(s.run,continuation_audit=s.trial/'synthetic-observation-audit.json',
        panel_candidate=s.candidate_path,robot=s.admitted['robot_path'],door_xml=s.admitted['door_xml_path'],door_usd=s.admitted['door_usd_path'])


def test_fresh_admissions_and_all_previous_nominals_preserved(source):
    before=copy.deepcopy((source.admitted,source.reference,source.candidate));result=prepare(source)
    assert [c[0] for c in source.calls]==['physical','reference']
    assert source.calls[-1][2]==dict(expected_epoch_s=.02,expected_physics_sha256=digest(source.archive))
    previous=source.reference['tail'][-1]['coupled']
    assert result['previous_reference']['value']==previous['value']
    assert result['previous_reference']['velocity']==previous['velocity']
    assert result['previous_reference']['command_time_s']==source.reference['tail'][-1]['command_time_s']
    assert result['measured_terminal']['time_s']==.020
    assert result['measured_terminal']['normalized_qpos']==source.admitted['initial_qpos']
    names=previous['joint_names'];mapped=result['panel_first_target_in_old_chart']
    for n in result['preserved_nominal_joint_names']:
        i=names.index(n);assert mapped[6+i]==previous['value'][6+i]
        assert mapped[6+i]!=source.context.qpos[source.scene.m.joint('robot/'+n).qposadr[0]]
    assert len(result['preserved_nominal_joint_names'])==44
    for n in planner.JOINT_NAMES:assert mapped[6+names.index(n)]==source.candidate['rows'][0]['joint_targets'][n]
    assert result['support']['source_target_N']==6.
    assert result['support']['active_target_N']==5.999999720839327
    assert result['predecessor_controller']['stance']==source.reference['tail'][-1]['stance']
    assert result['predecessor_controller']['hand']['grip_preload_scale']==0.
    assert result['reference_source_binding']['source_state_sha256']!=result['source_qualification']['state_sha256']
    assert (source.admitted,source.reference,source.candidate)==before
    for field in ('authorized_stages','physics_steps','source_sample_playback','active_state_writes'):assert result[field]==0
    for field in ('runtime_route_exported','controller_state_restoration_supported','bridge_feasibility_qualified','physical_task_qualification'):assert result[field] is False
    assert result['input_sha256'][str(Path(module.__file__).resolve())]==digest(module.__file__)
    for name in module.HELPER_NAMES:
        path=Path(module.__file__).resolve().with_name(name)
        assert result['input_sha256'][str(path)]==digest(path)
    json.dumps(result,allow_nan=False)


def test_noncommuting_root_chart_conversion_and_position_only_output(source):
    seed=prepare(source);before=copy.deepcopy(seed);x=np.r_[[.02,-.01,.003,.03,-.04,.02],np.linspace(.1,.3,25)]
    mapped=module.panel_coordinates_in_withdrawal_chart(seed,x)
    oldp,oldr=module._pose(seed['previous_reference']['root_coordinate_origin_xyz_wxyz'])
    newp,newr=module._pose(seed['panel_root_coordinate_origin_xyz_wxyz'])
    np.testing.assert_allclose(oldp+mapped[:3],newp+x[:3],rtol=0,atol=1e-15)
    np.testing.assert_allclose(Rotation.from_rotvec(mapped[3:6]).as_matrix()@oldr,
        Rotation.from_rotvec(x[3:6]).as_matrix()@newr,rtol=0,atol=1e-14)
    # Rotations do not commute: simple coordinate subtraction is wrong.
    assert np.linalg.norm(mapped[3:6]-(x[3:6]+Rotation.from_matrix(newr@oldr.T).as_rotvec()))>1e-5
    assert seed==before
    mapped[:]=99.;assert seed==before


def test_held_goal_uses_earlier_ik_pose_and_offset_once(source):
    result=prepare(source);held=result['held_left_goal'];left=source.reference['tail'][-1]['left']
    assert held['last_update_s']==.012 and .012<source.reference['tail'][0]['command_time_s']
    assert held['leaf_pose_xyz_wxyz']==source.poses[5]
    p,r=module._pose(source.poses[5]);local=np.array([.3,.04,.5]);offset=.007
    expected=p+r@local+r[:,1]*offset
    np.testing.assert_allclose(held['position_world'],expected,atol=1e-15)
    p2,r2=module._pose(source.poses[-1]);inverse=result['left_path_at_terminal_preserving_held_world_goal']
    np.testing.assert_allclose(p2+r2@inverse['position']+r2[:,1]*offset,expected,atol=1e-15)
    np.testing.assert_allclose(r2@inverse['rotation'],r@left['panel_palm_rotation'],atol=1e-15)
    assert np.linalg.norm(result['discrepancies']['current_material_goal_minus_held_left_goal_m'])>.001
    assert np.linalg.norm(p2+r2@inverse['position']+2*r2[:,1]*offset-expected)>.0069


@pytest.mark.parametrize('kind',['failed_source','failed_reference','failed_accounting','failed_report','source_run','source_trial',
    'source_epoch','source_archive','names','duplicate_names','convention','reference_time','post_time','handoff_epoch',
    'handoff_command','handoff_motor_order','handoff_contract','incomplete_left','orientation_changes','future_ik','stale_ik',
    'unaligned_ik','changing_held_offset','static_path_semantics','source_hash_conflict','candidate_source','hidden_finger'])
def test_bad_binding_or_unsupported_goal_reconstruction_fails_closed(source,kind):
    a=source.admitted;r=source.reference;last=r['tail'][-1]
    if kind=='failed_source':a['source_qualification']['passed']=False
    elif kind=='failed_reference':r['passed']=False
    elif kind=='failed_accounting':r['reference_tail_accounting_passed']=False
    elif kind=='failed_report':r['source_report_passed']=False
    elif kind=='source_run':r['source_run']=str(source.run/'other')
    elif kind=='source_trial':r['source_trial']=str(source.run/'other')
    elif kind=='source_epoch':r['source_terminal_time_s']+=.002
    elif kind=='source_archive':r['source_physics_sha256']='f'*64
    elif kind=='names':last['coupled']['joint_names'][0]='not_original'
    elif kind=='duplicate_names':last['coupled']['joint_names'][0]=last['coupled']['joint_names'][1]
    elif kind=='convention':last['coupled']['coordinate_convention']='body angular velocity'
    elif kind=='reference_time':last['coupled']['time_s']+=.002
    elif kind=='post_time':last['post_step_time_s']+=.002
    elif kind=='handoff_epoch':a['motor_handoff']['command_time_s']+=.002
    elif kind=='handoff_command':a['motor_handoff']['command_Nm'][3]+=.001
    elif kind=='handoff_motor_order':a['motor_handoff']['motor_names']=list(reversed(a['motor_handoff']['motor_names']))
    elif kind=='handoff_contract':a['motor_handoff']['motor_contract_sha256']='d'*64
    elif kind=='incomplete_left':last['left']['progress']=.999999
    elif kind=='orientation_changes':r['tail'][0]['left']['panel_palm_rotation']=np.eye(3).tolist()
    elif kind=='future_ik':last['left']['last_update_s']=.020
    elif kind=='stale_ik':last['left']['last_update_s']=.006
    elif kind=='unaligned_ik':last['left']['last_update_s']=.013
    elif kind=='changing_held_offset':r['tail'][0]['left']['normal_offset_m']+=.0001
    elif kind=='static_path_semantics':r['static_controller_data']['path_scope']='Arbitrary moving path'
    elif kind=='source_hash_conflict':r['input_sha256'][str(source.archive)]='f'*64
    elif kind=='candidate_source':source.candidate['source_time_s']+=.002;write(source.candidate_path,source.candidate)
    elif kind=='hidden_finger':source.candidate['rows'][1]['qpos'][-1]+=.001;write(source.candidate_path,source.candidate)
    # Keep candidate context current to isolate downstream cross-binding tests.
    if kind.startswith('handoff'):
        context=planner.admit_isaac_panel_context(source.run,robot=a['robot_path'],door_xml=a['door_xml_path'],door_usd=a['door_usd_path'])
        source.candidate['source_admission']=context.admission;source.candidate['source_context_sha256']=context.sha256
        write(source.candidate_path,source.candidate)
    with pytest.raises((ValueError,KeyError)):prepare(source)


@pytest.mark.parametrize('key,value',[('panel_plan_path','unexpected'),('left_arm_only',False),('left_full_orientation',False),
    ('coupled_motion_projection','other'),('capture_returned_motor_command',False)])
def test_arbitrary_runtime_cannot_use_fixed_goal_assumption(source,key,value):
    source.runtime[key]=value;path=source.trial/'standing-withdrawal-runtime.json';write(path,source.runtime)
    h=digest(path)
    for obj in (source.admitted,source.reference,source.candidate):obj['input_sha256'][str(path)]=h
    context=planner.admit_isaac_panel_context(source.run,robot=source.admitted['robot_path'],door_xml=source.admitted['door_xml_path'],door_usd=source.admitted['door_usd_path'])
    source.candidate.update(source_admission=context.admission,source_context_sha256=context.sha256)
    write(source.candidate_path,source.candidate)
    with pytest.raises(ValueError,match='minimal runtime'):prepare(source)


@pytest.mark.parametrize('kind',['nan_vector','missing_coordinate','invalid_quaternion','invalid_rotation','bool_offset'])
def test_pure_coordinate_helpers_reject_corrupt_inputs(source,kind):
    seed=prepare(source);x=np.zeros(31)
    if kind=='nan_vector':x[4]=np.nan
    elif kind=='missing_coordinate':x=x[:-1]
    elif kind=='invalid_quaternion':seed['panel_root_coordinate_origin_xyz_wxyz'][3:]=[0.,0.,0.,0.]
    if kind in ('nan_vector','missing_coordinate','invalid_quaternion'):
        with pytest.raises(ValueError):module.panel_coordinates_in_withdrawal_chart(seed,x)
    else:
        with pytest.raises(ValueError):module.left_path_goal_before_offset(source.poses[-1],[0.,0.,0.],
            np.zeros((3,3)) if kind=='invalid_rotation' else np.eye(3),True if kind=='bool_offset' else .007)


def test_current_inputs_rechecked_after_read_and_result_is_detached(source,monkeypatch):
    original=module._held_left_goal
    def mutate(*args):
        value=original(*args);source.archive.write_bytes(b'changed after admission');return value
    monkeypatch.setattr(module,'_held_left_goal',mutate)
    with pytest.raises(ValueError,match='changed'):prepare(source)


def test_default_source_target_remains_exact_and_nested_outputs_are_copies(source):
    for row in source.reference['tail']:row['left']['hybrid_normal_target_N']=6.
    source.reference.pop('support_profile_accounting')
    seed=prepare(source)
    assert seed['support']==dict(source_target_N=6.,active_target_N=6.,accounting=None)
    seed['previous_reference']['velocity'][0]=999.
    seed['previous_three_references'][-1]['coupled']['value'][0]=999.
    seed['predecessor_controller']['hand']['targets'][0]=999.
    assert source.reference['tail'][-1]['coupled']['velocity'][0]!=999.
    assert source.reference['tail'][-1]['coupled']['value'][0]!=999.
    assert source.reference['tail'][-1]['hand']['targets'][0]!=999.


@pytest.mark.parametrize('kind',['missing_member','truncated_length','misaligned_epoch'])
def test_saved_ik_leaf_stream_is_independently_checked(source,kind):
    arrays=dict(time_s=np.arange(1,11)*.002,standing_leaf_pose=np.array(source.poses))
    if kind=='missing_member':arrays.pop('standing_leaf_pose')
    elif kind=='truncated_length':arrays['standing_leaf_pose']=arrays['standing_leaf_pose'][:-1]
    else:arrays['time_s'][5]+=.001
    np.savez(source.archive,**arrays)
    # Synthetic upstream seam reports these bytes admitted; this deliberately
    # tests the new bounded reader's own independent field/clock checks.
    h=digest(source.archive)
    for obj in (source.admitted,source.reference,source.candidate):obj['input_sha256'][str(source.archive)]=h
    source.reference['source_physics_sha256']=h
    context=planner.admit_isaac_panel_context(source.run,robot=source.admitted['robot_path'],door_xml=source.admitted['door_xml_path'],door_usd=source.admitted['door_usd_path'])
    source.candidate.update(source_admission=context.admission,source_context_sha256=context.sha256)
    write(source.candidate_path,source.candidate)
    with pytest.raises(ValueError):prepare(source)


def test_native_or_failed_admission_never_reaches_candidate_loading(source,monkeypatch):
    def reject(*a,**kw):raise ValueError('Original physical withdrawal failed')
    monkeypatch.setattr(source_module,'admit_isaac_panel_source',reject)
    source.candidate_path.unlink()
    with pytest.raises(ValueError,match='physical withdrawal failed'):prepare(source)

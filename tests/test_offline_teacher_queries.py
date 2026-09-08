import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.offline_teacher_queries import load_query_evidence,correction_quality,POSE_CONVENTION,FORCE_SEMANTICS,CLOCK_CONVENTION
from doorbench.dexterous.correction_demonstrations import CorrectionDemonstration
from doorbench.dexterous.sensor_actor import ActorDimensions
from doorbench.dexterous.sensor_contract import SENSOR_KEYS
from doorbench.dexterous.stance import validate_stance_solver_settings


def make_source(root,*,extra_query=False):
    root.mkdir();put=lambda name,value:(root/name).write_text(json.dumps(value))
    (root/'sensors').mkdir();(root/'teacher-query-evidence').mkdir()
    n=4;times=np.arange(1,n+1)*.002;d=ActorDimensions(tactile=12);names=[f'j{i}' for i in range(69)]
    layout=dict(interface_version='doorbench.sensors.v2',joint_order=names,action_order=[f'a{i}' for i in range(61)],tactile_dimension=12,robot_xml_sha256='fixture')
    motors=dict(source_xml_sha256='fixture',joint_names=names,actuators=[dict(name=n,force_range=[-10.,10.]) for n in layout['action_order']])
    put('motor-contract.json',motors);put('sensors/layout.json',layout);put('configuration.json',dict(robot_joint_names=names))
    put('provenance.json',{'files':{'/fixture/reference.json':hashlib.sha256(b'{}').hexdigest()}})
    keys=('joint_stops','documented_loopbacks','self_collision','environment_collision','working_hand_collision','plant_parameters_unchanged','finite','motor_delivery_matches_command','native_motor_caps')
    put('report.json',dict(passed=False,checks={k:True for k in keys},closed_loop_evaluated=True,teacher_fallback=False))
    put('mechanical-audit.json',dict(passed=True))
    pose=np.array([0.,0.,.87,1.,0.,0.,0.],np.float32);roots=np.tile(np.r_[pose,np.zeros(6,np.float32)],(n+1,1))
    # The last two actual pre-action states violate the unchanged upright gate.
    q=Rotation.from_euler('y',20,degrees=True).as_quat();roots[2:,3:7]=q[[3,0,1,2]]
    joints=np.zeros((n+1,69),np.float32)
    put('acquisition-reset.json',dict(root=roots[0].tolist(),joints=joints[0].tolist(),door={'leaf':0.}))
    np.savez_compressed(root/'acquisition-physics.npz',time_s=times,root=roots[1:],joints=joints[1:],motor_forces=np.zeros((n,61)),door=np.zeros((n,3)))
    cold={key:np.zeros(shape,dtype=np.uint8 if key.startswith('rgb_') else np.float32) for key,shape in d.shapes.items()}
    cold.update(previous_action=np.zeros(61,np.float32),sensor_time_s=np.full(7,-1.),sensor_valid=np.zeros(7,bool))
    np.savez_compressed(root/'sensors/actor-initial-decision.npz',**cold,time_s=np.asarray(0.),motor_forces=np.zeros(61))
    numeric={key:np.repeat(v[None],n,axis=0) for key,v in cold.items() if not key.startswith('rgb_')}
    numeric['sensor_valid'][:]=True;numeric['sensor_time_s'][:]=times[:,None];numeric['sensor_time_s'][:2,-2:]=times[0];numeric['sensor_time_s'][2:,-2:]=times[2]
    np.savez_compressed(root/'sensors/actor-sensors.npz',time_s=times,**numeric)
    images=np.zeros((2,128,128,3),np.uint8);np.savez_compressed(root/'sensors/actor-rgb.npz',time_s=times[[0,2]],rgb_left=images,rgb_right=images)
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    put('sensors/report.json',dict(control_source='sensor_actor',capture_complete=False,numeric_file_sha256={k:sha(root/'sensors'/k) for k in ('actor-sensors.npz','actor-rgb.npz')}))
    m=n+int(extra_query);qt=np.arange(m)*.002
    arrays=dict(time_s=qt,root_state=roots[:m],joint_position=joints[:m],joint_velocity=joints[:m],
        handle_pose=np.tile(pose,(m,1)),leaf_pose=np.tile(pose,(m,1)),door_position=np.zeros((m,3)),right_hand_forces_world=np.zeros((m,1,3)))
    np.savez_compressed(root/'teacher-query-evidence/pre-action-measurements.npz',**arrays)
    put('teacher-query-evidence/contract.json',dict(schema='doorbench.teacher-query-evidence.v1',actor_input=False,contains_teacher_labels=False,
        physics_dt_s=.002,joint_order=names,hand_body_order=['rh_palm'],pose_convention=POSE_CONVENTION,
        force_semantics=FORCE_SEMANTICS,clock=CLOCK_CONVENTION,door_position_order=['operator','leaf','latch']))
    put('teacher-query-evidence/report.json',dict(samples=m,executed_steps=n,contains_teacher_labels=False,actor_input=False,
        file_sha256={k:sha(root/'teacher-query-evidence'/k) for k in ('contract.json','pre-action-measurements.npz')}))
    return arrays


def test_exact_preaction_alignment_excludes_unexecuted_last_query(tmp_path):
    root=tmp_path/'actor';make_source(root,extra_query=True)
    query,contract,physical,meta=load_query_evidence(root)
    assert meta['unexecuted_query_samples']==1 and len(query['time_s'])==4
    np.testing.assert_allclose(query['time_s']+.002,physical['time_s'],atol=1e-12)
    assert meta['source_actor_task_passed'] is False


def test_corrupted_query_file_is_rejected(tmp_path):
    root=tmp_path/'actor';make_source(root)
    p=root/'teacher-query-evidence/pre-action-measurements.npz';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash'):load_query_evidence(root)


@pytest.mark.parametrize('defect',['action_order','joint_order','pose_convention','force_semantics','clock','door_position_order'])
def test_reject_ambiguous_or_different_frames_and_orders(tmp_path,defect):
    root=tmp_path/'actor';make_source(root)
    name='sensors/layout.json' if defect.endswith('_order') and defect!='door_position_order' else 'teacher-query-evidence/contract.json'
    path=root/name;data=json.loads(path.read_text())
    if defect in ('action_order','joint_order'):data[defect][0]=data[defect][1]
    else:data[defect]='changed'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='contract|order'):load_query_evidence(root)


@pytest.mark.parametrize('defect',['tilted','low','inaccurate_solver','cap','failed_source'])
def test_correction_eligibility_preserves_physical_and_numerical_bounds(defect):
    root=np.array([0.,0.,.87,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]);force=np.zeros(61);caps=np.tile([-10.,10.],(61,1));status='solved';source=True
    if defect=='tilted':q=Rotation.from_euler('y',13,degrees=True).as_quat();root[3:7]=q[[3,0,1,2]]
    elif defect=='low':root[2]=.69
    elif defect=='inaccurate_solver':status='solved inaccurate'
    elif defect=='cap':force[0]=11.
    else:source=False
    assert not correction_quality(root,force,caps,solver_status=status,source_physical=source)['candidate_label_valid']


@pytest.mark.parametrize('settings',[{'eps_abs':1e-2},{'eps_rel':1e-2},{'max_iter':0},{'max_iter':1.5},{'rho':0.},{'rho':float('inf')}])
def test_solver_tuning_cannot_relax_residual_tolerances_or_use_invalid_values(settings):
    with pytest.raises(ValueError):validate_stance_solver_settings(settings)


def test_solver_tuning_is_explicit_and_preserves_default_settings():
    assert validate_stance_solver_settings(None)=={}
    assert validate_stance_solver_settings({'max_iter':100000,'rho':.001})=={'max_iter':100000,'rho':.001}


def test_correction_labels_are_expert_forces_at_same_decision_not_student_actions(tmp_path):
    root=tmp_path/'actor';make_source(root);queries,contract,physical,meta=load_query_evidence(root)
    output=tmp_path/'corrections';output.mkdir();rows=[];valid=[];caps=np.tile([-10.,10.],(61,1))
    for i,t in enumerate(queries['time_s']):
        quality=correction_quality(queries['root_state'][i],np.full(61,2.),caps,solver_status='solved',source_physical=True)
        valid.append(quality['candidate_label_valid']);rows.append(dict(time_s=float(t),solver_status='solved',teacher_exception=None,**quality))
    (output/'queries.json').write_text(json.dumps(rows));valid=np.array(valid,bool)
    np.savez_compressed(output/'counterfactual-teacher-actions.npz',time_s=queries['time_s'],motor_force=np.full((4,61),2.),
        normalized_force=np.full((4,61),.2),label_valid=valid,contiguous_valid_prefix=np.logical_and.accumulate(valid))
    for name in ('reference.json','manifest.json','source.tar.gz'):(output/name).write_text('{}')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(schema='doorbench.counterfactual-acquisition-teacher.v1',teacher_actions_delivered_to_physics=False,
        student_actions_used_as_expert_labels=False,maximum_teacher_internal_sim_time_s=0.,physical_recovery_evaluated=False,
        robot_sha256='fixture',reference_sha256=sha(output/'reference.json'),
        source=meta,teacher_force_calls=4,queries=4,candidate_valid_labels=2,contiguous_valid_prefix_labels=2,
        files_sha256={n:sha(output/n) for n in ('reference.json','counterfactual-teacher-actions.npz','queries.json','manifest.json','source.tar.gz')})
    (output/'report.json').write_text(json.dumps(report))
    data=CorrectionDemonstration(output)
    assert len(data)==2 and data.times.tolist()==[0.,.002]
    inputs,labels=data.sequence(0,2);np.testing.assert_allclose(labels,.2)
    assert not data.numeric['previous_action'].any() and 'root_state' not in inputs
    assert data.metadata['source_actor_task_passed'] is False
    with pytest.raises(ValueError,match='prefix'):data.sequence(1,2)

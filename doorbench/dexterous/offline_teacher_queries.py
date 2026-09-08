"""Audit real pre-action student measurements before offline expert queries.

Partial task failures can contain a complete executed prefix. They are never
teacher demonstrations. This module validates factual state/clock identity and
leaves expert label feasibility and physical success as separate questions.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .motor_contract_identity import motor_contract_fingerprint

POSE_CONVENTION = 'xyz+wxyz; root13 includes world linear and angular velocity'
FORCE_SEMANTICS = ('Actual right-hand body normal-contact force vectors summed over handle and leaf filters, '
    'matching AcquisitionTeacher Isaac adapter; tangential forces and other counterparts are not included')
CLOCK_CONVENTION = 'Measured before each executed actor decision, including reset t=0; no future state or interpolated impulses'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_query_evidence(run):
    run=Path(run);read=lambda name:json.loads((run/name).read_text())
    report=read('teacher-query-evidence/report.json');contract=read('teacher-query-evidence/contract.json')
    sensor=read('sensors/report.json');task=read('report.json');motors=read('motor-contract.json');layout=read('sensors/layout.json')
    if (contract.get('schema')!='doorbench.teacher-query-evidence.v1' or contract.get('actor_input') is not False or
            contract.get('contains_teacher_labels') is not False or report.get('contains_teacher_labels') is not False or
            report.get('actor_input') is not False or sensor.get('control_source')!='sensor_actor' or
            task.get('closed_loop_evaluated') is not True or task.get('teacher_fallback') is not False):
        raise ValueError('Expected separate unlabelled measurements of an actual sensor-only actor')
    if (contract.get('pose_convention') != POSE_CONVENTION or contract.get('force_semantics') != FORCE_SEMANTICS or
            contract.get('clock') != CLOCK_CONVENTION or contract.get('door_position_order') != ['operator','leaf','latch']):
        raise ValueError('Unsupported measured frame, force, clock or door order contract')
    if (layout['action_order'] != [a['name'] for a in motors['actuators']] or len(set(layout['action_order'])) != 61 or
            len(layout['joint_order']) != 69 or len(set(layout['joint_order'])) != 69 or
            set(layout['joint_order']) != set(contract['joint_order'])):
        raise ValueError('Ambiguous or mismatched sensor motor/joint order')
    for name in ('contract.json','pre-action-measurements.npz'):
        if report.get('file_sha256',{}).get(name)!=sha(run/'teacher-query-evidence'/name):
            raise ValueError('Teacher-query source hash mismatch')
    for name in ('actor-sensors.npz','actor-rgb.npz'):
        if sensor.get('numeric_file_sha256',{}).get(name)!=sha(run/'sensors'/name):
            raise ValueError('Actual actor observation archive hash mismatch')
    with np.load(run/'teacher-query-evidence/pre-action-measurements.npz',allow_pickle=False) as z:queries={k:z[k].copy() for k in z.files}
    with np.load(run/'acquisition-physics.npz',allow_pickle=False) as z:physical={k:z[k].copy() for k in z.files}
    with np.load(run/'sensors/actor-sensors.npz',allow_pickle=False) as z:observed={k:z[k].copy() for k in z.files}
    expected={'time_s','root_state','joint_position','joint_velocity','handle_pose','leaf_pose','door_position','right_hand_forces_world'}
    if set(queries)!=expected:raise ValueError('Unknown or missing teacher-query arrays')
    count=report['executed_steps'];dt=contract['physics_dt_s'];cfg=read('configuration.json')
    if (not isinstance(count,int) or count<=0 or dt!=.002 or len(queries['time_s'])!=report['samples'] or
            len(queries['time_s'])<count or len(physical['time_s'])!=count or len(observed['time_s'])!=count):
        raise ValueError('Actual executed-prefix counts disagree')
    # An attempted but unexecuted last query is retained in its original archive,
    # but is excluded from all potential labels.
    queries={k:v[:count] for k,v in queries.items()}
    times=np.arange(count)*dt
    if (not np.array_equal(queries['time_s'],times) or
            not np.allclose(times+dt,physical['time_s'],rtol=0,atol=1e-12) or
            not np.array_equal(physical['time_s'],observed['time_s'])):
        raise ValueError('Pre-action query must precede exactly its matching executed physical step')
    if (contract['joint_order']!=cfg['robot_joint_names'] or len(contract['joint_order'])!=69 or
            len(set(contract['joint_order']))!=69 or len(set(contract['hand_body_order']))!=len(contract['hand_body_order'])):
        raise ValueError('Ambiguous measured joint/body order')
    n=count;shapes=dict(time_s=(n,),root_state=(n,13),joint_position=(n,69),joint_velocity=(n,69),
        handle_pose=(n,7),leaf_pose=(n,7),door_position=(n,3),right_hand_forces_world=(n,len(contract['hand_body_order']),3))
    if any(queries[k].shape!=shape or not np.isfinite(queries[k]).all() for k,shape in shapes.items()):
        raise ValueError('Nonfinite or malformed actual teacher-query measurements')
    for key in ('root_state','handle_pose','leaf_pose'):
        if not np.allclose(np.linalg.norm(queries[key][:,3:7],axis=1),1.,atol=1e-5,rtol=0):raise ValueError('Invalid actual pose quaternion')
    reset=read('acquisition-reset.json');order=[layout['joint_order'].index(name) for name in contract['joint_order']]
    if (not np.array_equal(queries['root_state'][0],np.asarray(reset['root'],np.float32)) or
            not np.array_equal(queries['joint_position'][0],np.asarray(reset['joints'],np.float32)) or
            not np.array_equal(queries['root_state'][1:],physical['root'][:-1]) or
            not np.array_equal(queries['joint_position'][1:],physical['joints'][:-1]) or
            not np.array_equal(queries['joint_position'][1:],observed['joint_position'][:-1,order]) or
            not np.array_equal(queries['joint_velocity'][1:],observed['joint_velocity'][:-1,order])):
        raise ValueError('Measured query state differs from actual preceding state or encoder record')
    if layout['robot_xml_sha256']!=motors['source_xml_sha256']:raise ValueError('Sensor and actual mechanics model identity differ')
    files=('teacher-query-evidence/contract.json','teacher-query-evidence/report.json','teacher-query-evidence/pre-action-measurements.npz',
        'report.json','configuration.json','mechanical-audit.json','acquisition-reset.json','acquisition-physics.npz','provenance.json','motor-contract.json',
        'sensors/layout.json','sensors/report.json','sensors/actor-sensors.npz','sensors/actor-rgb.npz','sensors/actor-initial-decision.npz')
    metadata=dict(source_run=str(run.resolve()),executed_steps=count,unexecuted_query_samples=report['samples']-count,
        source_actor_task_passed=task['passed'],motor_contract_sha256=motor_contract_fingerprint(motors),
        files_sha256={name:sha(run/name) for name in files},
        limitation='Exact recorded student-state prefix; no expert labels or corrected physical trajectory yet')
    return queries,contract,physical,metadata


def correction_quality(root_state,force,caps,*,solver_status,source_physical,teacher_exception=None):
    """Conservative candidate-label eligibility, separate from recovery success."""
    from scipy.spatial.transform import Rotation
    root=np.asarray(root_state,float);force=np.asarray(force,float);caps=np.asarray(caps,float)
    if root.shape!=(13,) or not np.isfinite(root).all():raise ValueError('Invalid actual root state')
    rotation=Rotation.from_quat([*root[4:7],root[3]]).as_matrix()
    tilt=float(np.degrees(np.arccos(np.clip(rotation[2,2],-1,1))))
    physical=bool(source_physical and tilt<12. and root[2]>.7)
    finite_caps=bool(force.shape==(61,) and caps.shape==(61,2) and np.isfinite(force).all() and
        np.all(force>=caps[:,0]-1e-8) and np.all(force<=caps[:,1]+1e-8))
    return dict(physical_state_valid=physical,torso_tilt_deg=tilt,finite_original_caps=finite_caps,
        candidate_label_valid=bool(teacher_exception is None and physical and solver_status=='solved' and finite_caps))

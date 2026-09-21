"""Detached accounting and finite-difference analysis of three actual references.

No source task is qualified here. References are prior command inputs, not
measured states or a bridge proved feasible. No reference/controller is evaluated.
"""
import copy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .isaac_standing_continuation_audit import (
    admit_standing_continuation_observations, _json, _array, _same, SHAPES, BODY_NAMES)
from .npz_record_stream import iter_npz_records
from .qualified_isaac_grasp import digest

SCHEMA='doorbench.isaac-standing-reference-admission.v1'
DT=.002


def _clock(value,label):
    if type(value) not in (int,float) or not np.isfinite(value):
        raise ValueError('Finite recorded '+label+' required')
    return float(value)


def _names(value,expected,label):
    if type(value) is not list or any(type(v) is not str for v in value) or len(value)!=len(expected) or set(value)!=set(expected):
        raise ValueError('Complete unambiguous '+label+' required')
    return value


def _rotation(value,label):
    matrix=_array(value,(3,3),label)
    if not np.allclose(matrix.T@matrix,np.eye(3),atol=1e-8,rtol=0) or abs(np.linalg.det(matrix)-1.)>1e-8:
        raise ValueError('Recorded proper '+label+' required')
    return matrix


def _pose_rotation(pose):
    pose=_array(pose,(7,),'recorded body pose')
    if not np.isclose(np.linalg.norm(pose[3:]),1.,atol=1e-5):
        raise ValueError('Original normalized body pose required')
    return Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()


def _exact_json_values(actual,expected,label):
    a=_array(actual,np.asarray(expected).shape,label)
    if not np.array_equal(a,np.asarray(expected)):
        raise ValueError('Captured command inputs differ from actual archive: '+label)


def _indices(value,count,label):
    if (type(value) is not list or any(type(v) is not int for v in value)
            or len(set(value))!=len(value) or not value or min(value)<0 or max(value)>=count):
        raise ValueError('Exact captured '+label+' inventory required')
    return np.asarray(value,int)


def _optional_state(value,width,label,*,with_time=False,now=None):
    if type(value) is not dict or set(value)!={'available','value'} or type(value['available']) is not bool:
        raise ValueError('Explicit availability of '+label+' required')
    if not value['available']:
        if value['value'] is not None:raise ValueError('Absent '+label+' cannot contain invented data')
        return
    if with_time:
        pair=value['value']
        if type(pair) is not list or len(pair)!=2 or _clock(pair[0],label+' epoch')>now:
            raise ValueError('Actual prior '+label+' required')
        _array(pair[1],(width,),label)
    else:_array(value['value'],(width,),label)


def _consistency(row,static,tracking,motors,joint_names):
    """Check algebraic relationships without running FK, QP or a controller."""
    coupled=row['coupled'];names=_names(coupled['joint_names'],joint_names,'coupled joint order')
    values=_array(coupled['value'],(75,),'accepted coupled values')
    velocity=_array(coupled['velocity'],(75,),'accepted coupled velocity')
    origin=_array(coupled['root_coordinate_origin_xyz_wxyz'],(7,),'coupled origin')
    root=origin[:3]+values[:3]
    rotation=(Rotation.from_rotvec(values[3:6])*Rotation.from_matrix(_pose_rotation(origin))).as_matrix()
    nominal=dict(zip(names,values[6:]))
    postnames=_names(row['right']['post_feedback_joint_names'],joint_names,'post-feedback joint order')
    post=_array(row['right']['post_feedback_joint_targets'],(69,),'post-feedback targets')
    postmap=dict(zip(postnames,post));expected=dict(nominal)
    palm=row['right']['palm'];pn=palm['names']
    if type(pn) is not list or len(set(pn))!=len(pn) or not set(pn)<=set(names):
        raise ValueError('Explicit original palm-correction joint subset required')
    for n,v in zip(pn,_array(palm['offset'],(len(pn),),'palm offsets')):expected[n]+=v
    _same(post,[expected[n] for n in postnames],'nominal plus recorded palm correction',atol=1e-10)
    stance=row['stance'];sn=stance['joint_names']
    if type(sn) is not list or len(set(sn))!=len(sn) or not set(sn)<=set(names):
        raise ValueError('Explicit stance target subset required')
    _same(stance['target_root'],root+_array(stance['root_bias'],(3,),'stance root bias'),'biased stance root',atol=1e-10)
    _same(_rotation(stance['target_rotation'],'stance orientation'),
        _rotation(stance['rotation_bias'],'stance rotation bias')@rotation,'biased stance orientation',atol=1e-10)
    _same(stance['joint_target'],np.array([nominal[n] for n in sn])+_array(stance['joint_bias'],(len(sn),),'stance bias'),
        'biased stance joints',atol=1e-10)
    left=row['left'];ln=left['names']
    if ln!=static['left_path_names'] or len(set(ln))!=len(ln) or not set(ln)<=set(names):
        raise ValueError('Unchanged named left nominal path required')
    _same(left['nominal_joint_target'],[nominal[n] for n in ln],'left coupled nominal',atol=1e-10)
    if left['solve_names']!=ln[1:]:raise ValueError('Original isolated seven-joint LH solve order required')
    _array(left['target'],(len(ln)-1,),'left IK target')
    _array(left['previous'],(len(ln)-1,),'left IK seed')
    _array(left['initial_position_delta'],(3,),'initial LH material offset')
    _array(left['surface_velocity_world'],(3,),'copied surface velocity')
    _array(left['normal_contact_point_local'],(3,),'copied material point')
    _rotation(left['panel_palm_rotation'],'LH panel-local orientation')
    for key in ('started_s','last_update_s','progress','tracking_error_m','waist_delta','normal_offset_m',
                'support_target_N','filtered_palm_load_N','hybrid_normal_target_N','hybrid_blend'):
        _clock(left[key],'LH '+key)
    if left['loaded_since_s'] is not None:_clock(left['loaded_since_s'],'LH loaded-since epoch')
    _optional_state(left['target_velocity'],len(ln)-1,'LH target velocity')
    _optional_state(row['support']['previous_ik_target'],len(ln)-1,'support previous IK target',with_time=True,now=row['command_time_s'])
    if stance['qp_warm_start'] is not None:
        warm=np.asarray(stance['qp_warm_start'])
        if warm.ndim!=1 or not len(warm) or not np.isfinite(warm).all():raise ValueError('Finite actual primal QP warm start required')
    if stance['solver_metadata'] is not None and type(stance['solver_metadata']) is not dict:
        raise ValueError('Explicit copied solver metadata required')
    arm=row['right']['arm'];an=arm['names']
    if an!=tracking['arm']['names'] or not set(an)<=set(names):raise ValueError('Unchanged actual arm tracking names required')
    _same(arm['previous_target'],[postmap[n] for n in an],'actual arm target',atol=1e-10)
    _array(arm['reference_velocity'],(len(an),),'held arm target velocity')
    _array(arm['preload'],(len(an),),'actual arm preload')
    if not 0<_clock(palm['correction_gain_s_inv'],'actual palm correction gain')<=6:
        raise ValueError('Original bounded palm correction gain required')
    t=row['command_time_s']
    if (arm['last_call_time_s']!=t or palm['last_time_s']!=t or coupled['time_s']!=t
            or _clock(arm['previous_target_time_s'],'arm target epoch')>t
            or _clock(left['last_update_s'],'left IK epoch')>t):
        raise ValueError('Controller reference clocks must belong to this prior command')
    indices=_indices(row['hand']['motor_indices'],61,'hand motor')
    if row['hand']['motor_indices']!=tracking['hand']['motor_indices']:
        raise ValueError('Unchanged actual enhanced hand motor inventory required')
    matrix=np.zeros((61,69));index={n:i for i,n in enumerate(postnames)}
    for i,motor in enumerate(motors['actuators']):
        for n,coefficient in motor['terms'].items():matrix[i,index[n]]=coefficient
    _same(row['hand']['targets'],(matrix@post)[indices],'hand transmission targets',atol=1e-10)
    scale=_clock(row['hand']['grip_preload_scale'],'grip preload scale')
    if not 0<=scale<=1:raise ValueError('Original bounded grip preload scale required')
    _same(row['hand']['preload'],_array(row['hand']['original_preload'],(len(indices),),'original hand preload')*scale,
        'released hand preload',atol=1e-10)
    if row['hand']['finger_reference_velocity'] is not None or row['hand']['previous_finger_target'] is not None:
        raise ValueError('This minimal Isaac runtime has no separately enabled finger velocity route')
    withdrawal=row['withdrawal'];_array(withdrawal['handoff_offset'],(61,),'actual motor handoff offset')
    if not 0<_clock(withdrawal['handoff_seconds'],'motor handoff duration')<=2:
        raise ValueError('Original bounded motor handoff duration required')
    _clock(withdrawal['handoff_elapsed_s'],'motor handoff epoch')
    if withdrawal['release_started_s'] is not None:_clock(withdrawal['release_started_s'],'release epoch')
    support=row['support'];leaf=row['observed_input']['leaf_pose_xyz_wxyz'];leaf_rotation=_pose_rotation(leaf)
    _same(support['previous_leaf_position'],leaf[:3],'actual support leaf position',atol=0)
    _same(_rotation(support['previous_leaf_rotation'],'support leaf rotation'),leaf_rotation,'actual support leaf rotation',atol=1e-10)
    _same(left['normal_world'],leaf_rotation[:,1],'actual support panel normal',atol=1e-10)
    if (support['previous_s']!=t or support['previous_leaf_time_s']!=t
            or support['live_feedback_identity_retained'] is not True
            or left['support_target_N']!=left['hybrid_normal_target_N']
            or left['support_target_N']!=support['maximum_target_N']
            or not 2<float(left['support_target_N'])<=8):
        raise ValueError('Unchanged same-epoch source-bound support required')
    _array(row['right']['goal_position_world'],(3,),'requested RH touch-site goal')
    _rotation(row['right']['goal_rotation_world'],'requested RH touch-site orientation')
    _array(row['private_generalized_bias'],(75,),'private model bias')
    return dict(names=names,position=values,velocity=velocity,root=root,rotation=rotation,
        postnames=postnames,post=post,stance_root=np.asarray(stance['target_root'],float),
        right_goal=np.asarray(row['right']['goal_position_world'],float))


def admit_standing_reference_tail(trial, continuation_audit, expected_epoch_s, expected_physics_sha256):
    """Fresh observational admission; no actual released-source task assumed.

    The caller must separately qualify the physical source before using these
    diagnostics to plan a bridge. This function never exports a runtime route.
    """
    observed=admit_standing_continuation_observations(trial,continuation_audit,
        expected_epoch_s=expected_epoch_s,expected_physics_sha256=expected_physics_sha256)
    trial=Path(observed['source_trial']);hashes=dict(observed['input_sha256'])
    names=observed['robot_joint_names'];end=observed['source_terminal_time_s'];count=observed['physical_intervals']
    paths={key:trial/name for key,name in dict(tail='standing-continuation-reference-tail.json',
        admission='standing-withdrawal-admission.json',runtime='standing-withdrawal-runtime.json').items()}
    for path in paths.values():hashes[str(path)]=digest(path)
    captured=trial/'source-isaac_standing_reference_capture.py'
    if hashes.get(str(captured))!=digest(captured):raise ValueError('Original reference-capture helper must be in immutable provenance')
    for name in ('isaac_standing_reference_audit.py',):
        path=Path(__file__).with_name(name).resolve();hashes[str(path)]=digest(path)
    tail=_json(paths['tail']);configuration=_json(trial/'configuration.json');motors=_json(trial/'motor-contract.json')
    context=_json(paths['admission'])['source_context']
    runtime=_json(paths['runtime'])
    required=dict(capture_returned_motor_command=True,inherit_transfer_support=True,left_arm_only=True,
        left_full_orientation=True,coupled_motion_projection='fixed-poses-v1')
    if (runtime.get('schema')!='doorbench.isaac-standing-withdrawal-runtime.v1'
            or any(type(runtime.get(k)) is not type(v) or runtime[k]!=v for k,v in required.items())
            or any(k in runtime for k in ('finger_velocity_feedforward','panel_plan_path','hybrid_support','thumb_pad_feedback','finger_pad_feedback'))):
        raise ValueError('Original minimal Isaac withdrawal runtime declaration required')
    runtime_path=Path(configuration['args']['standing_withdrawal_route']).resolve()
    binding={k:context[k] for k in ('runtime_path','runtime_sha256','motor_contract_sha256','source_state_sha256')}
    if (configuration.get('standing_continuation_reference_capture')!='accepted-command-tail-v1'
            or Path(binding['runtime_path']).resolve()!=runtime_path
            or binding['runtime_sha256']!=hashes.get(str(runtime_path))
            or digest(paths['runtime'])!=binding['runtime_sha256']
            or binding['motor_contract_sha256']!=observed['motor_contract_sha256']
            or tail.get('source_binding')!=binding):
        raise ValueError('Exact captured withdrawal runtime, contract and predecessor source required')
    if (tail.get('schema')!='doorbench.isaac-standing-continuation-reference-tail.v1'
            or type(tail.get('authorized_stages')) is not int or tail['authorized_stages']!=0
            or tail.get('physical_task_qualification') is not False
            or tail.get('controller_state_restoration_supported') is not False
            or tail.get('failure') is not None or tail.get('pending_unaccepted_command') is not None
            or tail.get('tail_samples')!=3 or not isinstance(tail.get('tail'),list) or len(tail['tail'])!=3):
        raise ValueError('Complete final three accepted-only references required')
    start=_clock(context['start_time_s'],'withdrawal source epoch')
    completed=tail.get('completed_reference_intervals')
    if (type(completed) is not int or completed<3 or abs((end-start)/DT-completed)>1e-7
            or type(tail.get('tail_samples')) is not int):
        raise ValueError('Complete accepted reference count must reach the actual source endpoint')
    static=tail['static_controller_data'];tracking=tail['attained_tracking_contract']
    if static['motor_names']!=[m['name'] for m in motors['actuators']]:raise ValueError('Original named motor order required')
    _same(static['motor_caps'],[m['force_range'] for m in motors['actuators']],'original motor caps',atol=0)
    if not isinstance(static['left_path_at_construction'],list) or not static['left_path_at_construction']:
        raise ValueError('Actual static predecessor path capture required')
    _array(tracking['arm']['kp'],(len(tracking['arm']['names']),),'actual enhanced arm gains')
    _array(tracking['arm']['kd'],(len(tracking['arm']['names']),),'actual enhanced arm damping')
    hi=_indices(tracking['hand']['motor_indices'],61,'enhanced hand motor')
    _array(tracking['hand']['kp'],(len(hi),),'actual enhanced hand gains')
    _array(tracking['hand']['kd'],(len(hi),),'actual enhanced hand damping')
    if not 1<=_clock(tracking['hand']['stiffness_scale'],'enhanced hand scale')<=10:
        raise ValueError('Original bounded attained-hand stiffness scale required')
    need=set();rows=tail['tail']
    for i,row in enumerate(rows):
        t=(count-3+i)*DT;post=(count-2+i)*DT
        if (t<=0 or abs(_clock(row['command_time_s'],'command epoch')-t)>1e-10
                or abs(_clock(row['post_step_time_s'],'physical epoch')-post)>1e-10
                or row['source_binding']!=binding or row['withdrawal']['started_s']!=start
                or type(row.get('authorized_stages')) is not int or row['authorized_stages']!=0
                or row['observed_input']['time_s']!=row['command_time_s']):
            raise ValueError('Three actual consecutive prior commands ending at the source epoch required')
        _same(row['completed_interval_s'],[t,post],'completed reference interval',atol=1e-10)
        need.update((round(t/DT)-1,round(post/DT)-1))
    samples={}
    for i,sample in enumerate(iter_npz_records(trial/'acquisition-physics.npz',SHAPES,expected_rows=count)):
        if i in need:samples[i]=sample
    if len(samples)!=len(need):raise ValueError('Every commanded and observed source interval required')
    diagnosed=[]
    roles={'leaf':'leaf_hinge','operator':'leaf_handle_hinge','latch':'leaf_latch_bolt_slide'}
    for row in rows:
        before=samples[round(row['command_time_s']/DT)-1];after=samples[round(row['post_step_time_s']/DT)-1]
        actual=row['observed_input'];order=_names(actual['joint_names'],names,'actual command input joint order')
        index=[names.index(n) for n in order]
        _exact_json_values(actual['root13_actor_origin'],before['root'],'root')
        _exact_json_values(actual['joint_position'],before['joints'][index],'joints')
        _exact_json_values(actual['joint_velocity'],before['joint_velocity'][index],'joint velocity')
        _exact_json_values(actual['joint_velocity'],after['pre_step_joint_velocity'][index],'next interval pre-step velocity')
        _exact_json_values(actual['leaf_pose_xyz_wxyz'],before['continuation_body_poses'][BODY_NAMES.index('leaf')],'leaf pose')
        _exact_json_values(actual['handle_pose_xyz_wxyz'],before['continuation_body_poses'][BODY_NAMES.index('leaf_handle')],'handle pose')
        if set(actual['angles'])!=set(roles):raise ValueError('Actual complete measured mechanism coordinates required')
        for role,name in roles.items():
            if name not in observed['door_joint_names']:raise ValueError('Original named mechanism coordinates required')
            _exact_json_values(actual['angles'][role],before['door'][observed['door_joint_names'].index(name)],role)
        dtype=np.dtype(row['returned_motor_dtype'])
        if dtype.kind!='f' or dtype.itemsize not in (4,8):raise ValueError('Actual floating returned motor dtype required')
        force=_array(row['returned_motor_command'],(61,),'returned motor command').astype(dtype)
        if dtype!=after['motor_forces'].dtype or force.tobytes()!=after['motor_forces'].tobytes():
            raise ValueError('Returned command must exactly equal the actual submitted motor vector')
        diagnosed.append(_consistency(row,static,tracking,motors,names))
    first=diagnosed[0]
    if any(d['names']!=first['names'] or d['postnames']!=first['postnames']
            or rows[i]['coupled']['root_coordinate_origin_xyz_wxyz']!=rows[0]['coupled']['root_coordinate_origin_xyz_wxyz'] for i,d in enumerate(diagnosed)):
        raise ValueError('Reference coordinate origins and orders cannot change inside the tail')
    times=np.array([r['command_time_s'] for r in rows]);dt=np.diff(times)
    values=np.array([d['position'] for d in diagnosed]);velocities=np.array([d['velocity'] for d in diagnosed])
    finite_velocity=np.diff(values,axis=0)/dt[:,None]
    _same(velocities[1:],finite_velocity,'stored accepted reference velocity versus observed value differences',atol=1e-10)
    acceleration=np.diff(velocities,axis=0)/dt[:,None]
    world_angular=np.array([Rotation.from_matrix(diagnosed[i+1]['rotation']@diagnosed[i]['rotation'].T).as_rotvec()/dt[i] for i in range(2)])
    limits=dict(root_translation_speed_m_s=.02,root_rotvec_speed_rad_s=.03,joint_speed_rad_s=1.2,joint_acceleration_rad_s2=3.)
    maxima=dict(root_translation_speed_m_s=float(np.max(np.linalg.norm(velocities[:,:3],axis=1))),
        root_rotvec_speed_rad_s=float(np.max(np.linalg.norm(velocities[:,3:6],axis=1))),
        joint_speed_rad_s=float(np.max(abs(velocities[:,6:]))),joint_acceleration_rad_s2=float(np.max(abs(acceleration[:,6:]))))
    within=bool(maxima['root_translation_speed_m_s']<=.02+1e-12
        and maxima['root_rotvec_speed_rad_s']<=.03+1e-12 and maxima['joint_speed_rad_s']<=1.2+1e-12
        and np.all(abs(np.diff(velocities[:,6:],axis=0))<=3.*dt[:,None]+1e-12))
    terminal=samples[count-1];latest=diagnosed[-1]
    target_order=[names.index(n) for n in latest['names']]
    actual_rotation=_pose_rotation(terminal['root'][:7])
    terminal_delta=dict(reference_command_epoch_s=float(times[-1]),actual_state_epoch_s=end,
        nominal_root_minus_actual_m=(latest['root']-terminal['root'][:3]).tolist(),
        nominal_root_rotation_error_rad=float(Rotation.from_matrix(latest['rotation']@actual_rotation.T).magnitude()),
        nominal_joint_names=latest['names'],nominal_joint_minus_actual_rad=(latest['position'][6:]-terminal['joints'][target_order]).tolist(),
        scope='Prior accepted nominal at T-0.002 compared with actual T; these differences are not a newly evaluated target or a proved feasible bridge')
    for path,expected in hashes.items():
        if digest(path)!=expected:raise ValueError('Source/reference evidence changed during detached admission: '+path)
    return dict(schema=SCHEMA,passed=within,reference_tail_accounting_passed=True,
        source_run=observed['source_run'],source_trial=str(trial),source_terminal_time_s=end,
        source_physics_sha256=observed['source_physics_sha256'],source_report_passed=observed['source_report_passed'],
        source_binding=binding,reference_tail_sha256=hashes[str(paths['tail'])],
        observation_admission=observed,tail=copy.deepcopy(rows),static_controller_data=copy.deepcopy(static),
        attained_tracking_contract=copy.deepcopy(tracking),input_sha256=hashes,
        diagnostics=dict(command_times_s=times.tolist(),coordinate_joint_names=latest['names'],
            accepted_nominal_values=values.tolist(),stored_coordinate_velocity=velocities.tolist(),
            finite_difference_coordinate_velocity=finite_velocity.tolist(),finite_difference_coordinate_acceleration=acceleration.tolist(),
            root_positions_world=[d['root'].tolist() for d in diagnosed],root_rotations_world=[d['rotation'].tolist() for d in diagnosed],
            root_angular_velocity_world_interval_average=world_angular.tolist(),
            root_angular_acceleration_world_between_interval_averages=((world_angular[1]-world_angular[0])/float(np.mean(dt))).tolist(),
            stance_root_velocity_interval_average=(np.diff(np.array([d['stance_root'] for d in diagnosed]),axis=0)/dt[:,None]).tolist(),
            right_goal_velocity_interval_average=(np.diff(np.array([d['right_goal'] for d in diagnosed]),axis=0)/dt[:,None]).tolist(),
            post_feedback_joint_names=latest['postnames'],
            post_feedback_joint_velocity_interval_average=(np.diff(np.array([d['post'] for d in diagnosed]),axis=0)/dt[:,None]).tolist(),
            original_coupled_motion_limits=limits,tail_maximum=maxima,within_original_coupled_motion_limits=within,
            first_stored_velocity_before_tail_independently_verified=False,
            derivative_scope='Two backward interval differences only; rotvec coordinate rate differs from world angular interval rate. No instantaneous derivative or unrecorded acceleration is asserted.',
            right_goal_scope='Requested palm touch-site goal; actual continuation rh_palm is a body origin, so no site/body tracking error is fabricated.'),
        terminal_nominal_delta=terminal_delta,requires_separate_physical_source_qualification=True,
        controller_state_restoration_supported=False,bridge_feasibility_qualified=False,
        physical_task_qualification=False,authorized_stages=0,
        scope='Detached exact source/reference accounting and three-sample numerical diagnostics only; no reference evaluation, controller restart, runtime route, or stage authority')

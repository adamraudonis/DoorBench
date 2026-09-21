"""Detached chart/goal inputs for a future continuous Isaac panel bridge.

This does not construct a bridge, evaluate a live reference, restore controller
state, or grant stage authority. All three existing source admissions run fresh.
"""
import copy
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .isaac_panel_planning import JOINT_NAMES, admit_isaac_panel_context
from .isaac_panel_geometry_audit import validate_panel_candidate
from .isaac_standing_reference_audit import admit_standing_reference_tail
from .isaac_standing_continuation_audit import _json, _array
from .npz_record_stream import iter_npz_records
from .qualified_isaac_grasp import digest

SCHEMA = 'doorbench.isaac-panel-bridge-seed.v1'
DT = .002
CONVENTION = 'World root displacement/relative rotation-vector followed by named scalar joint targets'
HELPER_NAMES = ('isaac_panel_bridge_seed.py','isaac_panel_planning.py','isaac_panel_geometry_audit.py',
    'isaac_standing_reference_audit.py','isaac_standing_continuation_audit.py',
    'npz_record_stream.py','qualified_isaac_grasp.py')


def _rotation(value):
    r = _array(value, (3, 3), 'proper rotation')
    if not np.allclose(r.T@r, np.eye(3), rtol=0, atol=1e-8) or abs(np.linalg.det(r)-1)>1e-8:
        raise ValueError('Proper finite rotation required')
    return r


def _pose(value):
    pose = _array(value, (7,), 'XYZ/WXYZ pose')
    # Same recorded-pose convention/tolerance as reference accounting; the
    # source coordinate admission independently enforces its tighter contract.
    if abs(np.linalg.norm(pose[3:])-1)>1e-5:
        raise ValueError('Original normalized XYZ/WXYZ pose required')
    return pose[:3], Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()


def _merge_hashes(*groups):
    hashes = {}
    for group in groups:
        if not isinstance(group, dict) or not group:
            raise ValueError('Bound input hashes required')
        for name, value in group.items():
            path = Path(name)
            if not path.is_absolute():raise ValueError('Absolute source input required')
            key = str(path.resolve())
            if key in hashes and hashes[key]!=value:raise ValueError('Conflicting source input identity')
            if digest(path)!=value:raise ValueError('Bridge seed source input changed: '+name)
            hashes[key] = value
    return hashes


def panel_coordinates_in_withdrawal_chart(seed, coordinates31):
    """Map positions only; retain all 44 previous non-panel NOMINAL joints.

    The panel rotation vector is relative to measured T; the returned rotation
    vector is relative to the withdrawal origin. No derivative is transformed
    or invented, and this unadmitted vector is never installed in a controller.
    """
    if seed.get('schema')!=SCHEMA:raise ValueError('Explicit detached bridge seed required')
    previous = seed['previous_reference']
    names = previous['joint_names']
    if (type(names) is not list or len(names)!=69 or len(set(names))!=69
            or not set(JOINT_NAMES)<=set(names) or seed['panel_joint_names']!=JOINT_NAMES):
        raise ValueError('Exact original 69-joint order and 25 panel names required')
    value = _array(previous['value'], (75,), 'previous nominal').copy()
    x = _array(coordinates31, (31,), 'panel coordinates')
    oldp, oldr = _pose(previous['root_coordinate_origin_xyz_wxyz'])
    panelp, panelr = _pose(seed['panel_root_coordinate_origin_xyz_wxyz'])
    value[:3] = panelp+x[:3]-oldp
    worldr = Rotation.from_rotvec(x[3:6]).as_matrix()@panelr
    value[3:6] = Rotation.from_matrix(worldr@oldr.T).as_rotvec()
    for n, q in zip(JOINT_NAMES, x[6:]):value[6+names.index(n)] = q
    return value


def left_path_goal_before_offset(leaf_pose, desired_touch_position_world,
                                 desired_touch_rotation_world, normal_offset_m):
    """Re-express a world goal for a consumer that adds leaf-normal offset once."""
    if type(normal_offset_m) not in (int,float) or not np.isfinite(normal_offset_m):
        raise ValueError('Finite recorded normal offset required')
    p, r = _pose(leaf_pose)
    local = r.T@(_array(desired_touch_position_world, (3,), 'world touch goal')-p)
    local[1] -= normal_offset_m
    return dict(position=local.tolist(), rotation=(_rotation(r).T@
        _rotation(desired_touch_rotation_world)).tolist(), normal_offset_m=normal_offset_m,
        semantics='Consumer adds its existing leaf-local +Y normal offset exactly once')


def _held_left_goal(trial, reference, hashes):
    """Known completed minimal mode only; recover the last 100 Hz IK leaf pose."""
    runtime_path = trial/'standing-withdrawal-runtime.json'
    if hashes.get(str(runtime_path))!=digest(runtime_path):
        raise ValueError('Original minimal runtime must be source-bound')
    runtime = _json(runtime_path)
    required = dict(capture_returned_motor_command=True, inherit_transfer_support=True,
        left_arm_only=True, left_full_orientation=True, coupled_motion_projection='fixed-poses-v1')
    allowed = {'schema','scope',*required,'source_config_path','source_config_sha256',
        'coupled_envelope_path','coupled_envelope_sha256','coupled_audit_path','coupled_audit_sha256',
        'withdrawal_palm_load_profile'}
    if (runtime.get('schema')!='doorbench.isaac-standing-withdrawal-runtime.v1'
            or set(runtime)-allowed or any(type(runtime.get(k)) is not type(v) or runtime[k]!=v for k,v in required.items())):
        raise ValueError('Held LH goal requires the original minimal runtime modes')
    rows = reference['tail']; last = rows[-1]; left = last['left']
    static = reference['static_controller_data']
    if static.get('path_scope')!=('Static authored left path; only its last nominal is replaced by the coupled reference. '
        'That current nominal is copied in every accepted row.'):
        raise ValueError('Fixed captured LH material path semantics required')
    local = _array(static['left_path_at_construction'][-1]['position'], (3,), 'final static LH material position')
    rotation = _rotation(left['panel_palm_rotation'])
    epoch = left['last_update_s']
    if (type(epoch) not in (int,float) or not np.isfinite(epoch) or epoch<=0
            or abs(epoch/DT-round(epoch/DT))>1e-7 or epoch>last['command_time_s']
            or last['command_time_s']-epoch>.01+1e-8):
        raise ValueError('Actual held 100 Hz LH goal epoch required')
    for row in rows:
        previous = row['left']
        if (previous['progress']!=1. or previous['panel_palm_rotation']!=left['panel_palm_rotation']
                or previous['names']!=static['left_path_names']
                or previous['last_update_s']>row['command_time_s']):
            raise ValueError('Completed fixed-path LH progress and constant local orientation required')
        if previous['last_update_s']==epoch and previous['normal_offset_m']!=left['normal_offset_m']:
            raise ValueError('Held offset cannot change outside the captured IK update')
    # Consume to EOF so required member length/trailing-data checks still run.
    wanted = round(epoch/DT)-1; pose = None
    for i, sample in enumerate(iter_npz_records(trial/'acquisition-physics.npz',
            {'time_s':(), 'standing_leaf_pose':(7,)},
            expected_rows=reference['observation_admission']['physical_intervals'])):
        if abs(float(sample['time_s'])-(i+1)*DT)>1e-8:raise ValueError('Original contiguous leaf epochs required')
        if i==wanted:pose = sample['standing_leaf_pose']
    if pose is None:raise ValueError('Saved leaf pose at held LH IK epoch required')
    p, r = _pose(pose); offset = left['normal_offset_m']
    if type(offset) not in (int,float) or not np.isfinite(offset):raise ValueError('Finite held LH offset required')
    return dict(last_update_s=epoch, leaf_pose_xyz_wxyz=pose.tolist(),
        path_position_before_offset_local=local.tolist(), normal_offset_m=offset,
        position_world=(p+r@local+r[:,1]*offset).tolist(), rotation_world=(r@rotation).tolist(),
        scope='Reconstructed goal consumed at the recorded last IK update in completed fixed-path minimal mode; not a target evaluated at T')


def prepare_isaac_panel_bridge_seed(source, *, continuation_audit, panel_candidate,
                                   robot, door_xml, door_usd):
    """Freshly bind an actual released source and its prior accepted references.

    Candidate validation establishes only source/coordinate integrity. The
    mapped vector deliberately retains previous finger nominals, so even an
    eventual candidate geometry proof cannot qualify this different vector.
    """
    context = admit_isaac_panel_context(source, robot=robot, door_xml=door_xml, door_usd=door_usd)
    admitted = context.admission; scene = context.scene(); m, d = scene.m, scene.d
    candidate_path = Path(panel_candidate).resolve(); candidate_hash = digest(candidate_path)
    candidate = _json(candidate_path); validate_panel_candidate(candidate, context, scene)
    source = Path(admitted['source_run']).resolve(); trial = source/'trial'; terminal = admitted['source_time_s']
    archive = trial/'acquisition-physics.npz'
    expected_hash = admitted['input_sha256'].get(str(archive))
    if expected_hash is None:raise ValueError('Actual endpoint physics archive must be bound')
    reference = admit_standing_reference_tail(trial, continuation_audit,
        expected_epoch_s=terminal, expected_physics_sha256=expected_hash)
    if (reference.get('passed') is not True or reference.get('reference_tail_accounting_passed') is not True
            or reference.get('source_report_passed') is not True
            or Path(reference['source_run']).resolve()!=source or Path(reference['source_trial']).resolve()!=trial
            or reference['source_terminal_time_s']!=terminal or reference['source_physics_sha256']!=expected_hash):
        raise ValueError('Fresh successful source and exact admitted reference endpoint required')
    helpers = [Path(__file__).resolve().with_name(name) for name in HELPER_NAMES]
    hashes = _merge_hashes(admitted['input_sha256'], candidate['input_sha256'], reference['input_sha256'],
        {str(candidate_path):candidate_hash, **{str(path):digest(path) for path in helpers}})
    rows = reference['tail']; last = rows[-1]; coupled = last['coupled']; names = coupled['joint_names']
    contract = admitted['motor_contract']; handoff = admitted['motor_handoff']
    if (type(names) is not list or len(names)!=69 or len(set(names))!=69
            or set(names)!=set(contract['joint_names']) or not set(JOINT_NAMES)<=set(names)
            or coupled['coordinate_convention']!=CONVENTION or coupled['time_s']!=last['command_time_s']
            or abs(last['command_time_s']-(terminal-DT))>1e-10 or last['post_step_time_s']!=terminal):
        raise ValueError('Exact 69 named previous nominal coordinates at T-0.002 required')
    if (abs(handoff['command_time_s']-last['command_time_s'])>1e-10 or handoff['interval_end_s']!=terminal
            or handoff['motor_contract_sha256']!=reference['source_binding']['motor_contract_sha256']
            or handoff['motor_names']!=reference['static_controller_data']['motor_names']
            or handoff['command_Nm']!=last['returned_motor_command']):
        raise ValueError('Previous actual submitted motor handoff must match the admitted tail')
    for name in names:
        if m.joint('robot/'+name).type[0]!=mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError('Original scalar named robot joints required')
    held = _held_left_goal(trial, reference, hashes)
    rq = int(m.joint('robot/free_base').qposadr[0]); initial = context.qpos
    previous = dict(command_time_s=last['command_time_s'], paired_post_step_time_s=terminal,
        joint_names=copy.deepcopy(names), value=copy.deepcopy(coupled['value']),
        velocity=copy.deepcopy(coupled['velocity']),
        root_coordinate_origin_xyz_wxyz=copy.deepcopy(coupled['root_coordinate_origin_xyz_wxyz']),
        coordinate_convention=CONVENTION,
        derivative_scope='Exact stored old-chart coordinate velocity at T-0.002; rotational entries are rotvec rates, not world or body angular velocity')
    _array(previous['value'], (75,), 'previous nominal');_array(previous['velocity'], (75,), 'previous nominal velocity')
    preserved = [n for n in names if n not in JOINT_NAMES]
    if len(preserved)!=44:raise ValueError('Exactly 44 previous non-panel nominal joints required')
    seed = dict(schema=SCHEMA, source_run=str(source), source_time_s=terminal,
        source_context_sha256=context.sha256, source_qualification=copy.deepcopy(admitted['source_qualification']),
        source_physics_sha256=expected_hash, candidate_path=str(candidate_path), candidate_sha256=candidate_hash,
        input_sha256=hashes, reference_source_binding=copy.deepcopy(reference['source_binding']),
        previous_reference=previous, previous_three_references=copy.deepcopy(rows),
        reference_diagnostics=copy.deepcopy(reference['diagnostics']),
        panel_joint_names=list(JOINT_NAMES), panel_root_coordinate_origin_xyz_wxyz=initial[rq:rq+7].tolist(),
        preserved_nominal_joint_names=preserved,
        preserved_nominal_joint_values=[previous['value'][6+names.index(n)] for n in preserved],
        measured_terminal=dict(time_s=terminal, normalized_qpos=initial.tolist(), normalized_qvel=context.qvel.tolist(),
            extracted_state=copy.deepcopy(admitted['extracted_state']),
            body_poses_xyz_wxyz=copy.deepcopy(admitted['actual_body_poses_xyz_wxyz'])),
        preceding_submitted_motor_handoff=copy.deepcopy(handoff),
        predecessor_controller=dict(stance=copy.deepcopy(last['stance']),right=copy.deepcopy(last['right']),
            left=copy.deepcopy(last['left']),support=copy.deepcopy(last['support']),hand=copy.deepcopy(last['hand']),
            withdrawal=copy.deepcopy(last['withdrawal']),attained_tracking_contract=copy.deepcopy(reference['attained_tracking_contract'])),
        support=dict(source_target_N=last['left']['support_target_N'],active_target_N=last['left']['hybrid_normal_target_N'],
            accounting=copy.deepcopy(reference.get('support_profile_accounting'))),
        held_left_goal=held, authorized_stages=0, physics_steps=0, source_sample_playback=0, active_state_writes=0,
        runtime_route_exported=False, controller_state_restoration_supported=False,
        bridge_feasibility_qualified=False, physical_task_qualification=False,
        scope='Detached previous-nominal and source-goal inputs only; no settling stage, bridge, live controller, panel success or traversal authority')
    first = candidate['rows'][0]
    panel_x = first['root_delta']+[first['joint_targets'][n] for n in JOINT_NAMES]
    mapped = panel_coordinates_in_withdrawal_chart(seed, panel_x)
    lh = m.site('robot/lh_palm_touch').id; rh = m.site('robot/rh_palm_touch').id
    leaf_pose = admitted['actual_body_poses_xyz_wxyz']['leaf']; lp, lr = _pose(leaf_pose)
    local = np.asarray(held['path_position_before_offset_local']); offset = held['normal_offset_m']
    seed['left_path_at_terminal_preserving_held_world_goal'] = left_path_goal_before_offset(
        leaf_pose,held['position_world'],held['rotation_world'],offset)
    seed['panel_first_target_in_old_chart'] = mapped.tolist()
    seed['discrepancies'] = dict(previous_nominal_vs_measured_terminal=copy.deepcopy(reference['terminal_nominal_delta']),
        panel_first_minus_previous_nominal=(mapped-np.asarray(previous['value'])).tolist(),
        held_left_goal_minus_admitted_terminal_touch_site_m=(np.asarray(held['position_world'])-d.site_xpos[lh]).tolist(),
        prior_right_goal_minus_admitted_terminal_touch_site_m=(np.asarray(last['right']['goal_position_world'])-d.site_xpos[rh]).tolist(),
        current_material_goal_minus_held_left_goal_m=(lp+lr@local+lr[:,1]*offset-np.asarray(held['position_world'])).tolist(),
        touch_site_scope='Unstepped normalized source FK sites, separately admitted against actual bodies; not inferred contact or measured site tracking',
        mapped_vector_scope='25 panel positions plus 44 previous nominal joints; this differs from the static candidate and requires its own bridge geometry/rate checks')
    context.verify_inputs();_merge_hashes(hashes)
    return seed

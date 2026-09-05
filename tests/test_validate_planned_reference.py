"""Independent validator fixtures: actual original rig and analytically simple world."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import mujoco
import pytest
from doorbench.reference.rig import rig_xml
from scripts.validate_planned_reference import validate, LANDMARKS, normalized_rig_fingerprint, RIG_FINGERPRINT, task_completion


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path, offsets=None, wall=False):
    directory = tmp_path/'assets/doors/fixture'; directory.mkdir(parents=True)
    extra = '<geom name="wall" type="box" pos="0 -1 1.58" size=".1 .1 .1"/>' if wall else ''
    xml = ('<mujoco><compiler angle="radian"/><worldbody><geom name="floor" type="box" '
           'pos="0 -1 -.05" size="5 5 .05"/>'+extra+
           '<body name="leaf" pos="4 0 1"><joint name="hinge" type="hinge" range="-1 1"/>'
           '<geom name="leaf_geom" type="box" size=".2 .02 .3"/></body></worldbody></mujoco>')
    (directory/'door.xml').write_text(xml)
    (directory/'spec.json').write_text(json.dumps({'id': 'fixture', 'benchmark': {'primary_scenario': 'locked_recognize',
        'scenarios': [{'name': 'locked_recognize'}]}}))
    (directory/'model.json').write_text(json.dumps({'bodies': [{'geoms': [{'name': 'floor', 'semantic': 'floor'},
        {'name': 'wall', 'semantic': 'wall'}, {'name': 'leaf_geom', 'semantic': 'leaf'}]}]}))
    actor_xml = rig_xml(); actor = mujoco.MjModel.from_xml_string(actor_xml)
    native = mujoco.MjModel.from_xml_string(xml)
    spec = mujoco.MjSpec.from_file(str(directory/'door.xml'))
    spec.attach(mujoco.MjSpec.from_string(actor_xml), frame=spec.worldbody.add_frame(name='attach'), prefix='', suffix='')
    model = spec.compile(); data = mujoco.MjData(model)
    home = actor.qpos0.copy(); alpha = math.acos((.94-.06-.055)/.86)
    for side in ['l', 'r']:
        for name, value in [('hip_'+side+'_pitch', alpha), ('knee_'+side, -2*alpha), ('ankle_'+side+'_pitch', alpha),
                            ('shoulder_'+side+'_roll', .16 if side == 'l' else -.16), ('elbow_'+side, .15)]:
            home[int(actor.joint('actor_'+name).qposadr[0])] = value
    offsets = np.zeros((3, 3)) if offsets is None else np.asarray(offsets)
    aq = np.repeat(home[None], len(offsets), axis=0); aq[:, :3] += offsets
    names = [actor.joint(i).name for i in range(actor.njnt)]
    body_names = [actor.body(i).name for i in range(1, actor.nbody)]
    native_names = [native.body(i).name for i in range(native.nbody)]
    geom_rows = [{'name': actor.geom(i).name, 'body_name': actor.body(int(actor.geom_bodyid[i])).name,
                 'type': mujoco.mjtGeom(actor.geom_type[i]).name.removeprefix('mjGEOM_').lower(),
                 'size': actor.geom_size[i].tolist(), 'pos': actor.geom_pos[i].tolist(), 'quat_wxyz': actor.geom_quat[i].tolist()}
                 for i in range(actor.ngeom)]
    arrays = {k: [] for k in ['body_pos', 'body_quat', 'actor_body_pos', 'actor_body_quat', 'actor_joints', 'foot_pos', 'foot_quat']}
    for state in aq:
        data.qpos[:] = model.qpos0
        for i, name in enumerate(names):
            start = int(actor.jnt_qposadr[i]); end = int(actor.jnt_qposadr[i+1]) if i+1 < actor.njnt else actor.nq
            target = int(model.joint(name).qposadr[0]); data.qpos[target:target+end-start] = state[start:end]
        mujoco.mj_kinematics(model, data)
        arrays['body_pos'].append(data.xpos[[int(model.body(n).id) for n in native_names]].copy())
        arrays['body_quat'].append(data.xquat[[int(model.body(n).id) for n in native_names]].copy())
        arrays['actor_body_pos'].append(data.xpos[[int(model.body(n).id) for n in body_names]].copy())
        arrays['actor_body_quat'].append(data.xquat[[int(model.body(n).id) for n in body_names]].copy())
        arrays['actor_joints'].append(data.site_xpos[[int(model.site('actor_site_'+n).id) for n in LANDMARKS]].copy())
        sites = [int(model.site('actor_site_ankle_'+s).id) for s in ['l', 'r']]
        arrays['foot_pos'].append(data.site_xpos[sites].copy())
        quats = np.zeros((2, 4))
        for k, site in enumerate(sites): mujoco.mju_mat2Quat(quats[k], data.site_xmat[site])
        arrays['foot_quat'].append(quats.copy())
    arrays = {k: np.asarray(v) for k, v in arrays.items()}; n = len(aq)
    arrays.update(actor_time=np.arange(n)*.1, native_time=np.zeros(n), qpos=np.zeros((n, native.nq)), actor_qpos=aq,
                  foot_contact=np.ones((n, 2), bool), hand_contact=np.zeros((n, 2), bool), hand_target=np.zeros((n, 2, 3)))
    trajectory = tmp_path/'trajectory.npz'; np.savez_compressed(trajectory, **arrays)
    clip = {'schema': 'doorbench.planned-reference.v1', 'door_id': 'fixture', 'status': 'unvalidated',
            'units': 'metres/radians/seconds', 'up_axis': 'Z', 'frames': n, 'duration': float(arrays['actor_time'][-1]),
            'complete_proposal': True, 'trajectory_sha256': digest(trajectory),
            'source_sha256': {p: digest(directory/p) for p in ['door.xml', 'model.json', 'spec.json']},
            'actor': {'body_names': body_names, 'joint_names': names, 'qpos_addresses': actor.jnt_qposadr.tolist(),
                      'geometries': geom_rows, 'mjcf_xml': actor_xml, 'landmark_names': LANDMARKS},
            'native': {'body_names': native_names, 'joint_names': [native.joint(i).name for i in range(native.njnt)],
                       'qpos_addresses': native.jnt_qposadr.tolist()}, 'contact_exclusions': [[] for _ in range(n)]}
    clip['proposal'] = {'door_id': 'fixture', 'scenario': 'locked_recognize', 'source_sha256': clip['source_sha256'].copy(),
                        'source_outcome': {'door_id': 'fixture', 'scenario': 'locked_recognize', 'success': True, 'outcome': 'success'},
                        'traversal': 'not_requested'}
    clip_path = tmp_path/'clip.json'; clip_path.write_text(json.dumps(clip))
    return clip_path, trajectory, tmp_path/'assets', clip, arrays


def rewrite(clip_path, trajectory, clip, arrays=None):
    if arrays is not None: np.savez_compressed(trajectory, **arrays)
    clip['trajectory_sha256'] = digest(trajectory); clip_path.write_text(json.dumps(clip))


def test_independent_rig_fingerprint_ignores_only_global_start_pose():
    assert normalized_rig_fingerprint(rig_xml((2, -3, .94), .5)) == RIG_FINGERPRINT
    assert normalized_rig_fingerprint(rig_xml().replace('size="0.1"', 'size="0.05"')) != RIG_FINGERPRINT


def test_stationary_original_rig_passes_and_source_files_are_untouched(tmp_path):
    cp, trajectory, assets, clip, _ = fixture(tmp_path)
    hashes = {cp: digest(cp), trajectory: digest(trajectory)}
    result = validate(cp, trajectory, assets)
    assert result['accepted'], result
    assert result['actor_geometry_count'] == 16
    assert result['collision_samples'] > result['frames']
    assert {p: digest(p) for p in hashes} == hashes


def test_metadata_joint_order_uses_names_and_explicit_addresses(tmp_path):
    cp, trajectory, assets, clip, _ = fixture(tmp_path)
    clip['actor']['joint_names'].reverse(); clip['actor']['qpos_addresses'].reverse()
    rewrite(cp, trajectory, clip)
    assert validate(cp, trajectory, assets)['accepted']


@pytest.mark.parametrize('field', ['actor_joints', 'actor_body_pos', 'body_pos', 'foot_pos'])
def test_forged_cached_fk_cannot_pass(tmp_path, field):
    cp, trajectory, assets, clip, arrays = fixture(tmp_path)
    arrays[field][1, 0, 0] += .01; rewrite(cp, trajectory, clip, arrays)
    result = validate(cp, trajectory, assets)
    assert not result['accepted'] and result['failure_counts']['fk_'+field] > 0


def test_declared_stance_sliding_fails_even_when_saved_fk_is_consistent(tmp_path):
    cp, trajectory, assets, _, _ = fixture(tmp_path, offsets=[[0, 0, 0], [.004, 0, 0], [.004, 0, 0]])
    result = validate(cp, trajectory, assets)
    assert result['failure_counts']['stance_position_drift_m'] > 0
    assert result['maxima']['stance_position_drift_m'] == pytest.approx(.004)
    assert result['maxima']['fk_actor_joints'] < 1e-10


def test_actual_head_wall_collision_is_rejected(tmp_path):
    cp, trajectory, assets, _, _ = fixture(tmp_path, wall=True)
    result = validate(cp, trajectory, assets)
    assert result['failure_counts']['noncontact_clearance'] > 0
    assert result['minimum_noncontact_distance_m'] < -.1


@pytest.mark.parametrize('edit', ['shrunk_rig', 'missing_geom', 'bad_qpos_mapping', 'inactive_hand_mask', 'floor_mask', 'backward_time'])
def test_invalid_contract_is_rejected(tmp_path, edit):
    cp, trajectory, assets, clip, arrays = fixture(tmp_path)
    if edit == 'shrunk_rig': clip['actor']['mjcf_xml'] = clip['actor']['mjcf_xml'].replace('size="0.1"', 'size="0.05"')
    elif edit == 'missing_geom': clip['actor']['geometries'].pop()
    elif edit == 'bad_qpos_mapping': clip['actor']['qpos_addresses'][1] = 0
    elif edit == 'inactive_hand_mask': clip['contact_exclusions'][0] = [['actor_geom_hand_r', 'leaf_geom']]
    elif edit == 'floor_mask':
        arrays['hand_contact'][0, 1] = True; clip['contact_exclusions'][0] = [['actor_geom_hand_r', 'floor']]
    elif edit == 'backward_time': arrays['native_time'][1] = -.1
    rewrite(cp, trajectory, clip, arrays)
    assert validate(cp, trajectory, assets)['status'] == 'invalid_input'


def test_incomplete_proposal_cannot_be_accepted_as_a_complete_motion(tmp_path):
    cp, trajectory, assets, clip, _ = fixture(tmp_path)
    clip['complete_proposal'] = False; rewrite(cp, trajectory, clip)
    assert validate(cp, trajectory, assets)['failure_counts']['incomplete_proposal'] == 1


def test_fake_hand_contact_in_free_space_cannot_pass(tmp_path):
    cp, trajectory, assets, clip, arrays = fixture(tmp_path)
    arrays['hand_contact'][:, 1] = True
    arrays['hand_target'][:, 1] = arrays['actor_joints'][:, 9]
    clip['contact_exclusions'] = [[['actor_geom_hand_r', 'leaf_geom']] for _ in arrays['actor_time']]
    rewrite(cp, trajectory, clip, arrays)
    result = validate(cp, trajectory, assets)
    assert result['maxima']['active_hand_target_error_m'] < 1e-10
    assert result['failure_counts']['hand_contact_surface_gap_m'] > 0


def test_velocity_and_acceleration_use_actual_joint_state_not_cached_metrics(tmp_path):
    cp, trajectory, assets, _, _ = fixture(tmp_path, offsets=[[0, 0, 0], [.1, 0, 0], [0, 0, 0]])
    result = validate(cp, trajectory, assets)
    assert result['maxima']['fk_actor_joints'] == 0
    assert result['maxima']['actor_velocity_limit_ratio'] == pytest.approx(1.25)
    assert result['maxima']['actor_acceleration_limit_ratio'] == pytest.approx(20/3)
    assert result['failure_counts']['actor_velocity_limit_ratio'] == 2


def test_actual_joint_range_violation_is_rejected(tmp_path):
    cp, trajectory, assets, clip, arrays = fixture(tmp_path)
    j = clip['actor']['joint_names'].index('actor_elbow_r')
    arrays['actor_qpos'][1, clip['actor']['qpos_addresses'][j]] = 2.75
    rewrite(cp, trajectory, clip, arrays)
    result = validate(cp, trajectory, assets)
    assert result['maxima']['joint_limit_violation_rad'] == pytest.approx(.1)


@pytest.mark.parametrize('offset, failure', [([0, 0, -.005], 'foot_ground_penetration'), ([6, 0, 0], 'unsupported_stance_sole')])
def test_floor_intersection_and_missing_support_are_different_failures(tmp_path, offset, failure):
    cp, trajectory, assets, _, _ = fixture(tmp_path, offsets=[offset]*3)
    result = validate(cp, trajectory, assets)
    assert result['failure_counts'][failure] > 0
    assert result['maxima']['fk_actor_joints'] == 0


def test_trajectory_checksum_cannot_be_bypassed_by_valid_array_shapes(tmp_path):
    cp, trajectory, assets, _, arrays = fixture(tmp_path)
    arrays['native_time'][1] = .1; arrays['native_time'][2] = .2
    np.savez_compressed(trajectory, **arrays)
    result = validate(cp, trajectory, assets)
    assert result['status'] == 'invalid_input'
    assert 'checksum' in str(result['failure_examples'])


def test_native_door_sweep_is_checked_between_clear_endpoints(tmp_path):
    cp, trajectory, assets, clip, arrays = fixture(tmp_path)
    directory = assets/'doors/fixture'
    (directory/'door.xml').write_text('<mujoco><worldbody><geom name="floor" type="box" '
        'pos="0 -1 -.05" size="5 5 .05"/><body name="leaf" pos="-.4 -1 1.58">'
        '<joint name="hinge" type="slide" axis="1 0 0" range="0 .8"/>'
        '<geom name="leaf_geom" type="box" size=".02 .04 .05"/></body></worldbody></mujoco>')
    native = mujoco.MjModel.from_xml_path(str(directory/'door.xml')); data = mujoco.MjData(native)
    arrays['qpos'][:, 0] = [0, .8, 0]
    for i, q in enumerate(arrays['qpos']):
        data.qpos[:] = q; mujoco.mj_kinematics(native, data)
        arrays['body_pos'][i] = data.xpos; arrays['body_quat'][i] = data.xquat
    clip['source_sha256']['door.xml'] = digest(directory/'door.xml')
    clip['proposal']['source_sha256'] = clip['source_sha256'].copy()
    rewrite(cp, trajectory, clip, arrays)
    result = validate(cp, trajectory, assets)
    assert result['maxima']['fk_body_pos'] == 0
    assert result['failure_counts']['noncontact_clearance'] > 0
    assert all(isinstance(e['frame'], str) for e in result['failure_examples']['noncontact_clearance'])
    assert result['collision_samples'] > 100


def completion_fixture():
    clip = {'door_id': 'fixture', 'complete_proposal': True, 'source_sha256': {'spec.json': 'verified'},
            'proposal': {'door_id': 'fixture', 'scenario': 'open_and_traverse', 'traversal': 'proposed',
                         'source_sha256': {'spec.json': 'verified'}, 'source_outcome': {'door_id': 'fixture',
                         'scenario': 'open_and_traverse', 'success': True, 'outcome': 'success'}}}
    spec = {'benchmark': {'primary_scenario': 'open_and_traverse', 'scenarios': [{'name': 'open_and_traverse',
        'pass_plane': {'center': [0, 0, 1], 'normal': [0, 1, 0], 'traverse_direction': [0, 1, 0], 'width': 1., 'height': 2.},
        'goal': {'center': [0, 1.5, 0], 'radius': .5}}]}}
    return clip, spec


def test_task_completion_needs_actual_inside_crossing_and_final_goal():
    clip, spec = completion_fixture(); t = np.array([0., 1., 2.])
    inside = task_completion(clip, spec, np.array([[0, -1, .94], [0, .1, .94], [0, 1.5, .94]]), t)
    around = task_completion(clip, spec, np.array([[2, -1, .94], [2, .1, .94], [0, 1.5, .94]]), t)
    short = task_completion(clip, spec, np.array([[0, -1, .94], [0, .1, .94], [0, .2, .94]]), t)
    assert inside['evidence_pass'] and inside['inside_forward_crossings'] == 1
    assert around['failure_counts']['task_outside_aperture_crossing'] == 1
    assert short['failure_counts']['task_goal_region_not_reached'] == 1


@pytest.mark.parametrize('field', ['source_failed', 'route_unresolved', 'partial'])
def test_finished_motion_array_is_not_automatically_a_completed_task(field):
    clip, spec = completion_fixture()
    if field == 'source_failed': clip['proposal']['source_outcome'].update(success=False, outcome='fail')
    elif field == 'route_unresolved': clip['proposal']['traversal'] = 'unresolved'
    else: clip['complete_proposal'] = False
    report = task_completion(clip, spec, np.array([[0, -1, .94], [0, 1.5, .94]]), np.array([0., 1.]))
    assert not report['evidence_pass']

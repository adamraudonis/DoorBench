"""Synthetic CPU fixtures only: these tests do not run or qualify a simulator."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from doorbench.dexterous.isaac_attained_state import RECORDED_ROOT_CONVENTION
from doorbench.dexterous.isaac_prefix_witness import (
    LiveIsaacPrefixWitness, PrefixDivergenceError, PREFIX_FIELDS,
)
from doorbench.dexterous.qualified_isaac_grasp import AUDITED_FILES, TRANSFER_CHECKS, digest
from doorbench.dexterous.standing_body_record import PLANNER_BODIES, POSE_CONVENTION
from scripts.dexterous.plan_local_isaac_transfer import load_local_source


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path):
    """Tiny explicitly synthetic files exercising the real qualification loader."""
    run = tmp_path / 'synthetic source'
    trial = run / 'trial'
    trial.mkdir(parents=True)
    producer = tmp_path / 'old worktree' / 'isaac_opening.py'
    producer.parent.mkdir()
    producer.write_text('# synthetic CPU fixture producer\n')
    captured = trial / ('source-' + producer.name)
    captured.write_bytes(producer.read_bytes())
    door = tmp_path / 'door.usda'
    door.write_text('synthetic CPU fixture asset')
    names = ['joint_' + str(i) for i in range(69)]
    configuration = dict(
        args=dict(door_usd=str(door), acquisition=True, operate_after_acquisition=True,
                  acquisition_stance_profile='landed-foot-v1'),
        dt=.002, runtime_pose_writes=0, direct_door_commands=False,
        robot_joint_names=names, door_joint_names=['leaf_hinge', 'handle_joint', 'latch'],
        root_state_convention=RECORDED_ROOT_CONVENTION,
        standing_planner_body_names=list(PLANNER_BODIES),
        standing_planner_body_pose_convention=POSE_CONVENTION)
    motors = dict(joint_names=names, source_xml_sha256='a' * 64,
                  actuators=[dict(name='motor_' + str(i), terms={names[i]: 1.},
                                  force_range=[-10., 10.]) for i in range(61)])
    arrays = dict(time_s=np.arange(1, 4) * .002, root=np.zeros((3, 13), np.float32),
                  joints=np.zeros((3, 69), np.float32),
                  joint_velocity=np.zeros((3, 69), np.float32),
                  motor_forces=np.zeros((3, 61), np.float64),
                  door=np.zeros((3, 3), np.float32),
                  door_velocity=np.zeros((3, 3), np.float32),
                  standing_body_poses=np.zeros((3, 6, 7), np.float32))
    arrays['root'][:, 3] = 1.
    arrays['standing_body_poses'][:, :, 3] = 1.
    report = dict(passed=True, checks=dict(synthetic_cpu_fixture_only=True),
                  duration_s=.006, physics_dt_s=.002)
    provenance = dict(files={str(producer): digest(producer), str(door): digest(door)},
                      fixture='Synthetic test only; never physical evidence')
    write(trial / 'configuration.json', configuration)
    write(trial / 'motor-contract.json', motors)
    write(trial / 'provenance.json', provenance)
    write(trial / 'operation-report.json', report)
    np.savez(trial / 'acquisition-physics.npz', **arrays)
    (trial / 'acquisition-pad-steps.json.gz').write_bytes(b'synthetic test fixture, not raw contacts')
    audit = dict(accounting_passed=True, task_passed=True,
                 independent_raw_contact_audit_complete=True,
                 checks=dict(synthetic_cpu_fixture_only=True), invalid_loaded_patches=0,
                 physical_intervals=3, raw_intervals=3,
                 final_half_second_failed_samples=[],
                 input_sha256={name: digest(trial / name) for name in AUDITED_FILES})
    write(run / 'independent-contact-audit.json', audit)
    extracted, _, _ = load_local_source(run, run / 'independent-contact-audit.json')
    return dict(run=run, trial=trial, producer=producer, captured=captured, door=door,
                configuration=configuration, motors=motors, arrays=arrays,
                report=report, audit=audit, provenance=provenance,
                state_sha256=extracted['binding']['sha256'])


def witness(source, **overrides):
    kwargs = dict(expected_source_state_sha256=source['state_sha256'], stage_start_s=.006,
                  runtime_configuration=copy.deepcopy(source['configuration']),
                  runtime_motor_contract=copy.deepcopy(source['motors']))
    kwargs.update(overrides)
    return LiveIsaacPrefixWitness(source['run'], **kwargs)


def sample(source, i):
    return {key: value[i].copy() for key, value in source['arrays'].items()}


def complete(source, value):
    for i in range(3):
        assert value.observe(sample(source, i)) is (i == 2)


def rebind_audit(source):
    source['audit']['input_sha256'] = {
        name: digest(source['trial'] / name) for name in AUDITED_FILES}
    write(source['run'] / 'independent-contact-audit.json', source['audit'])


def test_requires_every_interval_then_authorizes_once_without_mutating_records(source):
    value = witness(source)
    before = {key: item.copy() for key, item in source['arrays'].items()}
    assert value.receipt()['passed'] is False
    complete(source, value)
    assert value.complete and value.receipt()['passed'] is False
    result = value.require_stage_entry(.006)
    assert result['passed'] and result['intervals_verified'] == 3
    assert result['fields'] == list(PREFIX_FIELDS)
    assert result['measured_motor_torque_checked'] is False
    assert 'commanded' in result['motor_forces_semantics']
    assert result['source_qualification']['passed']
    assert result['source_provenance'] == source['provenance']
    for key in before:
        np.testing.assert_array_equal(before[key], source['arrays'][key])
    with pytest.raises(PrefixDivergenceError):
        value.require_stage_entry(.006)
    assert value.failed and not value.receipt()['passed']


@pytest.mark.parametrize('field', PREFIX_FIELDS)
def test_one_ulp_in_any_physical_or_command_field_is_sticky_failure(source, field):
    value = witness(source)
    row = sample(source, 0)
    changed = np.array(row[field], copy=True)
    changed.flat[0] = np.nextafter(changed.flat[0], np.array(np.inf, dtype=changed.dtype))
    row[field] = changed
    with pytest.raises(PrefixDivergenceError) as error:
        value.observe(row)
    assert error.value.receipt['failure']['field'] == field
    assert value.intervals_verified == 0 and value.failed
    with pytest.raises(PrefixDivergenceError):
        value.observe(sample(source, 0))
    with pytest.raises(PrefixDivergenceError):
        value.require_stage_entry(.006)


@pytest.mark.parametrize('kind', ['missing', 'shape', 'dtype', 'nan', 'negative_zero', 'object'])
def test_partial_or_reinterpreted_record_cannot_pass(source, kind):
    value = witness(source)
    row = sample(source, 0)
    if kind == 'missing': row.pop('joints')
    if kind == 'shape': row['joints'] = row['joints'].reshape(3, 23)
    if kind == 'dtype': row['joints'] = row['joints'].astype(np.float64)
    if kind == 'nan': row['joints'][0] = np.nan
    if kind == 'negative_zero': row['joints'][0] = -0.
    if kind == 'object': row['joints'] = object()
    with pytest.raises(PrefixDivergenceError): value.observe(row)
    assert value.failed and value.intervals_verified == 0


@pytest.mark.parametrize('indices', [(1,), (0, 0), (0, 2)])
def test_skip_reorder_or_duplicate_interval_rejected(source, indices):
    value = witness(source)
    with pytest.raises(PrefixDivergenceError):
        for index in indices: value.observe(sample(source, index))
    assert value.failed


@pytest.mark.parametrize('count,time', [(0, 0.), (2, .006), (3, .004), (3, .008), (3, float('nan'))])
def test_entry_cannot_skip_pending_prefix_or_use_stale_epoch(source, count, time):
    value = witness(source)
    for i in range(count): value.observe(sample(source, i))
    with pytest.raises(PrefixDivergenceError): value.require_stage_entry(time)
    assert value.failed and not value.receipt()['stage_entry_authorized']


def test_post_prefix_observation_is_explicitly_out_of_scope(source):
    value = witness(source)
    complete(source, value)
    with pytest.raises(PrefixDivergenceError, match='extra observation'):
        value.observe(sample(source, 2))


def test_changed_current_producer_is_allowed_only_with_unchanged_historical_capture(source):
    old = source['captured'].read_bytes()
    source['producer'].write_text('# new continuation code, honestly captured by the next run\n')
    value = witness(source)
    complete(source, value)
    receipt = value.require_stage_entry(.006)
    assert receipt['passed']
    assert source['captured'].read_bytes() == old
    assert str(source['producer']) not in receipt['input_sha256']
    assert receipt['historical_source_copies'][str(source['producer'])]['sha256'] == digest(source['captured'])
    source['captured'].write_bytes(source['producer'].read_bytes())
    with pytest.raises(ValueError, match='hash mismatch'): witness(source)


@pytest.mark.parametrize('target', ['captured', 'door', 'archive', 'qualification'])
def test_source_mutation_between_admission_and_entry_latches_failure(source, target):
    value = witness(source)
    complete(source, value)
    path = dict(captured=source['captured'], door=source['door'],
                archive=source['trial'] / 'acquisition-physics.npz',
                qualification=source['run'] / 'independent-contact-audit.json')[target]
    with path.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(PrefixDivergenceError, match='changed before stage entry'):
        value.require_stage_entry(.006)


def test_receipt_mutation_cannot_erase_hash_binding_or_qualification(source):
    value = witness(source)
    receipt = value.receipt()
    receipt['input_sha256'].clear()
    receipt['source_qualification']['passed'] = False
    receipt['source_provenance']['files'].clear()
    complete(source, value)
    result = value.require_stage_entry(.006)
    assert result['passed'] and result['input_sha256']
    assert result['source_qualification']['passed'] and result['source_provenance']['files']


@pytest.mark.parametrize('field', ['robot_joint_names', 'door_joint_names', 'standing_planner_body_names',
                                 'root_state_convention', 'standing_planner_body_pose_convention'])
def test_same_numbers_with_changed_coordinate_semantics_rejected(source, field):
    configuration = copy.deepcopy(source['configuration'])
    item = configuration[field]
    configuration[field] = list(reversed(item)) if isinstance(item, list) else item + ' changed'
    with pytest.raises(ValueError, match='coordinate order or convention'):
        witness(source, runtime_configuration=configuration)


@pytest.mark.parametrize('kind', ['order', 'cap', 'terms'])
def test_commanded_motor_contract_cannot_be_reinterpreted(source, kind):
    motors = copy.deepcopy(source['motors'])
    if kind == 'order': motors['actuators'].reverse()
    if kind == 'cap': motors['actuators'][0]['force_range'][1] += 1.
    if kind == 'terms': motors['actuators'][0]['terms']['joint_0'] = 2.
    with pytest.raises(ValueError, match='motor contract'):
        witness(source, runtime_motor_contract=motors)


@pytest.mark.parametrize('change', ['physical', 'contact', 'hold', 'intervals', 'input_hash'])
def test_original_physical_and_contact_source_gates_are_not_relaxed(source, change):
    if change == 'physical':
        source['report']['passed'] = False
        write(source['trial'] / 'operation-report.json', source['report'])
        rebind_audit(source)
    if change == 'contact': source['audit']['invalid_loaded_patches'] = 1
    if change == 'hold': source['audit']['final_half_second_failed_samples'] = [2]
    if change == 'intervals': source['audit']['raw_intervals'] = 2
    if change == 'input_hash': source['audit']['input_sha256']['configuration.json'] = 'b' * 64
    write(source['run'] / 'independent-contact-audit.json', source['audit'])
    with pytest.raises(ValueError): witness(source)


def test_transfer_source_still_requires_bound_independent_transfer_qualification(source):
    source['report']['standing_transfer'] = dict(synthetic_cpu_fixture_only=True)
    source['report']['checks'].update(dict.fromkeys(TRANSFER_CHECKS, True))
    report_path = source['trial'] / 'operation-report.json'
    stream_path = source['trial'] / 'standing-transfer-steps.json.gz'
    write(report_path, source['report'])
    stream_path.write_bytes(b'synthetic transfer stream')
    rebind_audit(source)
    audit = dict(passed=True, producer_matches=True, checks=dict.fromkeys(TRANSFER_CHECKS, True),
                 input_sha256={str(path): digest(path) for path in (report_path, stream_path)})
    write(source['run'] / 'isaac-transfer-audit.json', audit)
    assert not witness(source).failed
    audit['checks']['final_left_palm_support'] = False
    write(source['run'] / 'isaac-transfer-audit.json', audit)
    with pytest.raises(ValueError, match='transfer checks'): witness(source)


@pytest.mark.parametrize('kind', ['missing', 'nonfinite', 'clock', 'width'])
def test_incomplete_or_corrupt_source_archive_rejected(source, kind):
    arrays = copy.deepcopy(source['arrays'])
    if kind == 'missing': arrays.pop('motor_forces')
    if kind == 'nonfinite': arrays['motor_forces'][0, 0] = np.nan
    if kind == 'clock': arrays['time_s'][0] = .001
    if kind == 'width': arrays['motor_forces'] = np.zeros((3, 60))
    np.savez(source['trial'] / 'acquisition-physics.npz', **arrays)
    with pytest.raises(ValueError): witness(source)


def test_stage_endpoint_binding_is_not_just_a_passing_source_flag(source):
    with pytest.raises(ValueError, match='exact qualified source endpoint'):
        witness(source, expected_source_state_sha256='b' * 64)
    with pytest.raises(ValueError, match='exact qualified source endpoint'):
        witness(source, stage_start_s=.004)


def test_offline_prefix_audit_field_contract_remains_identical(source, tmp_path):
    # Use the existing independent audit, rather than duplicate its field list.
    path = Path(__file__).parents[1] / 'scripts/isaac/run_local_operation.py'
    spec = importlib.util.spec_from_file_location('prefix_launcher_test', path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    later = tmp_path / 'synthetic new trial'
    later.mkdir()
    np.savez(later / 'acquisition-physics.npz', **source['arrays'])
    audit = launcher.audit_transfer_prefix(source['run'], later, .006)
    assert audit['passed'] and tuple(audit['fields']) == PREFIX_FIELDS

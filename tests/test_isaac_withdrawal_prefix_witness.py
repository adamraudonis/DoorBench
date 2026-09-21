"""Synthetic CPU-only prefix records; no simulator qualification is invented."""
import copy
import gzip
import json

import numpy as np
import pytest

from test_isaac_prefix_witness import source, write, rebind_audit, sample
from doorbench.dexterous.isaac_prefix_witness import PREFIX_FIELDS, PrefixDivergenceError
from doorbench.dexterous.isaac_withdrawal_prefix_witness import (
    LiveIsaacWithdrawalPrefixWitness, LEAF_POSE_CONVENTION, SCHEMA)
from doorbench.dexterous.qualified_isaac_grasp import TRANSFER_CHECKS, digest


@pytest.fixture
def transfer_source(source):
    source['report']['standing_transfer'] = dict(synthetic_cpu_fixture_only=True)
    source['report']['checks'].update(dict.fromkeys(TRANSFER_CHECKS, True))
    report = source['trial'] / 'operation-report.json'
    stream = source['trial'] / 'standing-transfer-steps.json.gz'
    write(report, source['report'])
    leaf = np.array([[.1, .2, .3, 1., 0., 0., 0.]]*3, np.float32)
    source['leaf'] = leaf
    source['leaf_rows'] = [dict(time_s=(i+1)*.002, leaf_pose=p.tolist()) for i, p in enumerate(leaf)]
    with gzip.open(stream, 'wt') as output:
        json.dump(source['leaf_rows'], output)
    source['transfer_audit'] = dict(passed=True, producer_matches=True,
        checks=dict.fromkeys(TRANSFER_CHECKS, True),
        input_sha256={str(p): digest(p) for p in (report, stream)})
    write(source['run'] / 'isaac-transfer-audit.json', source['transfer_audit'])
    rebind_audit(source)
    return source


def witness(s):
    config = copy.deepcopy(s['configuration'])
    config['standing_leaf_pose_convention'] = LEAF_POSE_CONVENTION
    return LiveIsaacWithdrawalPrefixWitness(s['run'], expected_source_state_sha256=s['state_sha256'],
        stage_start_s=.006, runtime_configuration=config, runtime_motor_contract=s['motors'])


def observe(w, s, i, **kwargs):
    return w.observe(sample(s, i), **dict(leaf_pose=s['leaf'][i], leaf_time_s=(i+1)*.002, **kwargs))


def test_combined_permission_requires_full_secondary_and_original_core(transfer_source):
    s = transfer_source
    w = witness(s)
    for i in range(3):
        assert observe(w, s, i) is (i == 2)
    assert not w.receipt()['passed']
    receipt = w.require_stage_entry(.006)
    assert receipt['schema'] == SCHEMA and receipt['passed']
    assert receipt['core']['fields'] == list(PREFIX_FIELDS)
    assert receipt['core']['passed'] and receipt['leaf_pose']['passed']
    assert receipt['leaf_pose']['intervals_verified'] == 3
    assert not receipt['leaf_pose']['historical_tensor_dtype_compared']
    assert str(s['trial'] / 'standing-transfer-steps.json.gz') in receipt['input_sha256']
    receipt['leaf_pose']['input_sha256'].clear()
    assert w.receipt()['leaf_pose']['input_sha256']
    with pytest.raises(PrefixDivergenceError):
        w.require_stage_entry(.006)
    assert w.failed and not w.receipt()['passed']


@pytest.mark.parametrize('kind', ['bit', 'nonfinite', 'shape', 'stale', 'core_bit', 'core_dtype', 'missing_core'])
def test_leaf_or_core_failure_is_sticky(transfer_source, kind):
    s = transfer_source
    w = witness(s)
    row = sample(s, 0)
    pose = s['leaf'][0].astype(np.float64)
    epoch = .002
    if kind == 'bit': pose[0] = np.nextafter(pose[0], np.inf)
    if kind == 'nonfinite': pose[0] = np.nan
    if kind == 'shape': pose = pose[:6]
    if kind == 'stale': epoch = 0.
    if kind == 'core_bit': row['joints'][0] = np.nextafter(row['joints'][0], np.float32(np.inf))
    if kind == 'core_dtype': row['joints'] = row['joints'].astype(np.float64)
    if kind == 'missing_core': row.pop('motor_forces')
    with pytest.raises(PrefixDivergenceError):
        w.observe(row, leaf_pose=pose, leaf_time_s=epoch)
    assert w.failed and w.receipt()['leaf_pose']['intervals_verified'] == 0
    with pytest.raises(PrefixDivergenceError): observe(w, s, 0)
    with pytest.raises(PrefixDivergenceError): w.require_stage_entry(.006)


@pytest.mark.parametrize('indices', [(1,), (0, 0), (0, 2)])
def test_skipped_or_repeated_interval_fails(transfer_source, indices):
    w = witness(transfer_source)
    with pytest.raises(PrefixDivergenceError):
        for i in indices: observe(w, transfer_source, i)


@pytest.mark.parametrize('count,epoch', [(0, .006), (2, .006), (3, .004), (3, float('nan'))])
def test_premature_or_wrong_epoch_entry_fails(transfer_source, count, epoch):
    w = witness(transfer_source)
    for i in range(count): observe(w, transfer_source, i)
    with pytest.raises(PrefixDivergenceError): w.require_stage_entry(epoch)


@pytest.mark.parametrize('name', ['standing-transfer-steps.json.gz', 'isaac-transfer-audit.json'])
def test_changed_secondary_source_before_entry_fails(transfer_source, name):
    s = transfer_source
    w = witness(s)
    for i in range(3): observe(w, s, i)
    path = (s['run'] if name == 'isaac-transfer-audit.json' else s['trial']) / name
    with path.open('ab') as output: output.write(b'changed')
    with pytest.raises(PrefixDivergenceError): w.require_stage_entry(.006)


@pytest.mark.parametrize('kind', ['short', 'duplicate', 'nonfinite', 'missing_pose'])
def test_even_rebound_secondary_stream_must_have_complete_clock_and_pose(transfer_source, kind):
    s = transfer_source
    rows = s['leaf_rows']
    if kind == 'short': rows.pop()
    if kind == 'duplicate': rows[1]['time_s'] = .002
    if kind == 'nonfinite': rows[1]['leaf_pose'][0] = float('nan')
    if kind == 'missing_pose': rows[1].pop('leaf_pose')
    path = s['trial'] / 'standing-transfer-steps.json.gz'
    with gzip.open(path, 'wt') as output: json.dump(rows, output)
    s['transfer_audit']['input_sha256'][str(path)] = digest(path)
    write(s['run'] / 'isaac-transfer-audit.json', s['transfer_audit'])
    with pytest.raises((ValueError, TypeError)): witness(s)

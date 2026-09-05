"""Pet assets stay available while all benchmark entry paths exclude them."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from doorbench.benchmark_eligibility import (
    BenchmarkExcludedError, POLICY_VERSION, benchmark_eligibility,
    collection_counts, is_benchmark_eligible,
)
from doorbench.benchmark import runner
from doorbench.benchmark.scenarios import assign_scenarios, build_benchmark, benchmark_summary, make_scenario
from doorbench.spec import generate_all

PET = {'id': 'db0037_pet_door', 'family': 'pet_door', 'benchmark': {
    'scenarios': ['open_and_traverse'], 'primary': 'open_and_traverse'},
    'benchmark_eligibility': {'eligible': True}}
STANDARD = {'id': 'db0002_swing_single', 'family': 'swing_single', 'extras': ['pet_flap'],
            'benchmark': {'scenarios': ['open_and_traverse'], 'primary': 'open_and_traverse'}}


def test_family_scope_and_original_asset_inventory():
    specs = generate_all()
    counts = collection_counts(specs)
    assert counts == dict(n_assets_total=1000, n_doors_eligible=985, n_doors_supplementary=15, eligibility_policy=POLICY_VERSION)
    inserts = [s for s in specs if 'pet_flap' in s.get('extras', [])]
    assert inserts and all(is_benchmark_eligible(s) for s in inserts)
    assert not is_benchmark_eligible(PET), 'stale eligibility metadata must not override family policy'
    assert is_benchmark_eligible(STANDARD)


def test_pet_scenario_metadata_is_empty_and_explains_why():
    # No model/physics lookups occur for excluded scenario assignment.
    assert assign_scenarios(PET) == []
    block = build_benchmark(PET, {}, {})
    assert block['scenarios'] == [] and block['primary_scenario'] is None
    assert block['suites'] == {'core': [], 'human': []}
    info = benchmark_summary(block)
    assert info['primary'] is None and info['time_budget_s'] is None
    assert info['benchmark_eligibility'] == benchmark_eligibility(PET)
    with pytest.raises(BenchmarkExcludedError):
        make_scenario('open_and_traverse', PET, {}, {})


@pytest.mark.parametrize('suite', ['core', 'human', 'all'])
@pytest.mark.parametrize('only', [None, ['primary'], ['open_and_traverse']])
def test_old_pet_manifest_cannot_create_jobs(suite, only):
    assert runner.scenarios_for(PET, suite, only) == []
    assert runner.door_scenarios(PET, suite) == []
    assert not runner.make_jobs([PET], '/does-not-exist', suite, only, [0], 'full', 'random')


def test_all_selectors_exclude_and_explicit_pet_requests_fail(tmp_path):
    manifest = {'doors': [PET, STANDARD]}
    for selector in ['all', 'first:20', 'every:1', 'sample:20', 'scenario:open_and_traverse']:
        assert runner.select_doors(manifest, selector) == [STANDARD]
    for selector in [PET['id'], 'ids:' + PET['id'], 'family:pet_door', STANDARD['id'] + ',' + PET['id']]:
        with pytest.raises(BenchmarkExcludedError):
            runner.select_doors(manifest, selector)
    ids = tmp_path/'ids.txt'; ids.write_text(PET['id'] + '\n')
    with pytest.raises(BenchmarkExcludedError):
        runner.select_doors(manifest, '@' + str(ids))


def test_direct_env_episode_recorder_and_planner_fail_before_native_or_output(tmp_path, monkeypatch):
    from doorbench.benchmark.env import DoorEnv
    from doorbench.reference.record import record_one
    from doorbench.reference.guidance import make_guide
    from doorbench.reference.solve import solve_door
    directory = tmp_path/'doors'/PET['id']; directory.mkdir(parents=True)
    (directory/'spec.json').write_text(json.dumps(PET))
    def forbidden(*args, **kwargs):
        pytest.fail('excluded request reached native model/policy work')
    monkeypatch.setattr(DoorEnv, '_build', forbidden)
    monkeypatch.setattr(runner, '_policy_for', forbidden)
    with pytest.raises(BenchmarkExcludedError):
        DoorEnv(str(directory))
    # Actual spec defeats an incorrectly labelled manifest job.
    job = runner.Job({**STANDARD, 'id': PET['id']}, str(directory), 'open_and_traverse', 0, 'full', 'random')
    with pytest.raises(BenchmarkExcludedError):
        runner.run_episode(job)
    for call in [lambda: record_one((PET, tmp_path, tmp_path/'out', 20)),
                 lambda: make_guide(directory, tmp_path/'missing-recordings'),
                 lambda: solve_door(directory, tmp_path/'missing-recordings', tmp_path/'out')]:
        with pytest.raises(BenchmarkExcludedError):
            call()
    assert not (tmp_path/'out').exists()


def _episode(door, success, *, outcome=None, seed=0):
    return {'door_id': door['id'], 'family': door['family'], 'scenario': 'open_and_traverse',
            'suite': 'core', 'seed': seed, 'success': success, 'outcome': outcome or ('success' if success else 'fail'),
            'damage': False, 'difficulty': 2, 'episode_return': 12 if success else -2,
            'time_to_pass': 2 if success else None, 'wall_s': 3}


def _result(episodes):
    return {'run': {'n_doors': len({e['door_id'] for e in episodes}), 'seeds': [0], 'date': '2026-09-05',
                    'simulator': 'mujoco', 'tier': 'full', 'suite': 'core', 'wall_time_s': 99,
                    'scenarios': [{'name': 'open_and_traverse', 'suite': 'core'}]},
            'benchmark': {'n_doors_total': 2}, 'policy': {'name': 'random'},
            'aggregate': runner.aggregate(episodes, {d['id']: d for d in [PET, STANDARD]}), 'episodes': episodes}


def test_historical_index_recomputes_subset_and_keeps_raw_provenance(tmp_path):
    from scripts.build_results_index import summarize
    manifest = {'doors': [PET, STANDARD]}
    doc = _result([_episode(PET, False), _episode(STANDARD, True)])
    path = tmp_path/'random.json'; path.write_text(json.dumps(doc))
    raw = path.read_bytes()
    result = summarize(str(path), 1, 0, manifest)
    subset = result['historical_subset']
    assert subset['applied'] and subset['source_sha256'] == hashlib.sha256(raw).hexdigest()
    assert subset['original_n_doors'] == 2 and subset['excluded_door_ids'] == [PET['id']]
    assert subset['excluded_n_episodes'] == 1 and subset['retained_n_doors'] == 1
    assert result['n_doors'] == result['n_doors_total'] == 1
    core = result['suites']['core']
    assert core['doors_solved'] == 1 and core['success_rate'] == 1 and core['complete']
    assert 'pet_door' not in core['by_family'] and PET['id'] not in core['doors']
    assert result['wall_time_s'] == 99 and 'original full run' in subset['note']
    assert path.read_bytes() == raw


def test_index_completeness_requires_every_eligible_scenario_and_seed(tmp_path):
    from scripts.build_results_index import summarize
    door = copy.deepcopy(STANDARD); door['benchmark']['scenarios'].append('open_then_close')
    path = tmp_path/'random.json'; path.write_text(json.dumps(_result([_episode(STANDARD, True)])))
    result = summarize(str(path), 1, 0, {'doors': [door]})
    assert not result['suites']['core']['complete'] and not result['leaderboard']


def test_current_results_reject_pet_but_historical_raw_stays_readable():
    from scripts.validate_result import semantic_errors
    manifest = {'doors': [PET, STANDARD]}
    doc = _result([_episode(PET, False), _episode(STANDARD, True)])
    assert not semantic_errors(doc, manifest, False, 'random.json')
    doc['benchmark'].update(eligibility_policy=POLICY_VERSION, n_doors_total=1)
    assert any('supplementary pet door' in e for e in semantic_errors(doc, manifest, False, 'random.json'))
    doc['benchmark'].pop('eligibility_policy')
    assert any('supplementary pet door' in e for e in semantic_errors(doc, manifest, True, 'random.json'))


def test_submission_coverage_uses_eligible_denominator():
    from scripts.validate_result import semantic_errors
    doc = _result([_episode(STANDARD, True, seed=s) for s in [0, 1, 2]])
    doc['run'].update(seeds=[0, 1, 2], simulator_version='3.6')
    doc['benchmark'].update(commit='abc', n_doors_total=1, eligibility_policy=POLICY_VERSION)
    assert not semantic_errors(doc, {'doors': [PET, STANDARD]}, True, 'random.json')


def test_isaac_selectors_and_config_guard_preserve_raw_asset_qa(tmp_path):
    import importlib.util
    source = Path(__file__).resolve().parents[1]/'isaaclab/doorbench_isaaclab/doors.py'
    spec = importlib.util.spec_from_file_location('isolated_isaac_doors', source)
    D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
    (tmp_path/'manifest.json').write_text(json.dumps({'doors': [PET, STANDARD]}))
    for row in [PET, STANDARD]:
        directory = tmp_path/'doors'/row['id']; directory.mkdir(parents=True)
        (directory/'spec.json').write_text(json.dumps(row))
        (directory/'door_rl.usda').write_text('#usda 1.0')
    assets = str(tmp_path)
    assert D.select_ids('all', root=assets) == [STANDARD['id']]
    assert D.select_ids('random-100', root=assets) == [STANDARD['id']]
    assert D.select_ids('all', root=assets, benchmark_only=False) == [PET['id'], STANDARD['id']]
    assert D.usd_path(PET['id'], root=assets).endswith('door_rl.usda'), 'download/physical QA lookup stays available'
    for selector in [PET['id'], 'family:pet_door']:
        with pytest.raises(BenchmarkExcludedError):
            D.select_ids(selector, root=assets)
    listing = tmp_path/'ids.txt'; listing.write_text(PET['id'])
    with pytest.raises(BenchmarkExcludedError):
        D.select_ids('@' + str(listing), root=assets)
    with pytest.raises(BenchmarkExcludedError):
        D.require_eligible_ids([PET['id']], root=assets)
    D.require_eligible_ids([STANDARD['id']], root=assets)
    # Family-spoofed manifest cannot bypass actual-source config validation.
    (tmp_path/'doors'/STANDARD['id']/'spec.json').write_text(json.dumps(PET))
    with pytest.raises(BenchmarkExcludedError):
        D.require_eligible_ids([STANDARD['id']], root=assets)


def test_pet_asset_export_retains_geometry_without_benchmark_scenarios(tmp_path):
    from doorbench.build import export_door
    spec = next(s for s in generate_all() if s['family'] == 'pet_door')
    summary = export_door(spec, str(tmp_path/'doors'), str(tmp_path/'hardware'), formats=('json', 'mjcf'))
    directory = tmp_path/'doors'/spec['id']
    saved = json.loads((directory/'spec.json').read_text())
    assert saved['family'] == 'pet_door' and saved['benchmark']['scenarios'] == []
    assert summary['n_bodies'] > 0 and (directory/'door.xml').stat().st_size > 100
    assert json.loads((directory/'model.json').read_text())['bodies']

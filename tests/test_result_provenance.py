import copy
import json
from pathlib import Path
import subprocess

import pytest

from doorbench import result_provenance as provenance
from doorbench.result_aggregation import aggregate
from scripts import validate_result as validator


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE).decode().strip()


@pytest.fixture
def history(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; repo.mkdir()
    git(repo, 'init', '--quiet')
    door = dict(id='db0002_swing_single', family='swing_single', lock_engaged=False,
                benchmark=dict(primary='open_and_traverse', scenarios=['open_and_traverse']))
    original = dict(name='DoorBench', version='0.1.0', generated='2026-09-04T11:00:16', doors=[door])
    (repo/'assets').mkdir(); (repo/'assets/manifest.json').write_text(json.dumps(original))
    git(repo, 'add', 'assets/manifest.json')
    git(repo, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '--quiet', '-m', 'original assets')
    commit = git(repo, 'rev-parse', 'HEAD')
    eps = [dict(door_id=door['id'], family=door['family'], scenario='open_and_traverse', suite='core',
        seed=seed, success=True, outcome='success', sim_time=2., steps=100, wall_s=.1, events=[], labels={}) for seed in range(3)]
    doc = dict(schema_version='1.1', benchmark=dict(name='DoorBench', n_doors_total=1,
        commit=commit, dirty=False, dataset_version='0.1.0', dataset_generated=original['generated']),
        policy=dict(name='random', embodiment='hand_base'), run=dict(date='2026-09-04', simulator='mujoco',
        simulator_version='3.12', tier='full', suite='core', scenarios=[dict(name='open_and_traverse', suite='core')],
        seeds=[0,1,2], n_doors=1, scenario_filter='all'), episodes=eps, aggregate=aggregate(eps, {door['id']:door}))
    (repo/'results').mkdir()
    files = []
    for name in sorted(provenance.ARCHIVED_BASELINES):
        path = repo/'results'/name; path.write_text(json.dumps(doc)); files.append(path)
    registry_path = repo/'registry.json'
    registry_path.write_text(json.dumps(provenance.freeze_historical_results(files, repo=repo)))
    current = copy.deepcopy(original)
    current['generated'] = '2026-09-08T11:00:16'
    current['doors'][0]['benchmark'] = dict(primary='locked_recognize', scenarios=['locked_recognize'])
    monkeypatch.setattr(validator, 'historical_context', lambda path, document:
        provenance.historical_context(path, document, repo=repo, registry_path=registry_path))
    return dict(repo=repo, path=repo/'results/random.json', doc=doc, original=original,
                current=current, registry=registry_path, files=files, commit=commit)


def context(history):
    return provenance.historical_context(history['path'], history['doc'], repo=history['repo'], registry_path=history['registry'])


def test_real_source_revision_replaces_current_scenario_check_without_skipping_semantics(history):
    source, info = context(history)
    assert source == history['original'] and info['source_commit'] == history['commit']
    assert validator.semantic_errors(history['doc'], history['current'], False, str(history['path']))
    schema = json.loads(Path(validator.SCHEMA).read_text())
    assert validator.validate_file(str(history['path']), schema, history['current'], False) == []
    # Even a freshly registered file with wrong counts must fail semantic audit.
    bad = copy.deepcopy(history['doc']); bad['aggregate']['core']['n_success'] = 2
    history['path'].write_text(json.dumps(bad))
    history['registry'].write_text(json.dumps(provenance.freeze_historical_results(history['files'], repo=history['repo'])))
    assert any('n_success' in e for e in validator.validate_file(str(history['path']), schema, history['current'], False))


@pytest.mark.parametrize('change', ['missing_registry', 'invalid_registry', 'invalid_entry', 'missing_entry', 'missing_manifest', 'wrong_manifest_hash',
                                  'wrong_blob', 'wrong_result_hash', 'missing_source', 'wrong_metadata'])
def test_historical_proof_fails_closed(history, change):
    registry = json.loads(history['registry'].read_text())
    manifest = registry['manifests'][history['commit']]
    if change == 'missing_registry': history['registry'].unlink()
    elif change == 'invalid_registry': registry = []
    elif change == 'invalid_entry': registry['results']['random.json'] = []
    elif change == 'missing_entry': del registry['results']['random.json']
    elif change == 'missing_manifest': del registry['manifests'][history['commit']]
    elif change == 'wrong_manifest_hash': manifest['sha256'] = '0'*64
    elif change == 'wrong_blob': manifest['git_blob_oid'] = '0'*40
    elif change == 'wrong_result_hash': registry['results']['random.json']['sha256'] = '0'*64
    elif change == 'missing_source':
        registry['results']['random.json']['source_commit'] = '1'*40
        history['doc']['benchmark']['commit'] = '1'*40
    elif change == 'wrong_metadata': history['doc']['benchmark']['dataset_generated'] = 'tomorrow'
    if change != 'missing_registry': history['registry'].write_text(json.dumps(registry))
    with pytest.raises(ValueError): context(history)


def test_original_result_bytes_are_immutable(history):
    with history['path'].open('a') as stream: stream.write('\n')
    with pytest.raises(ValueError, match='bytes changed'): context(history)


def test_renaming_old_file_does_not_implicitly_register_a_historical_submission(history):
    renamed = history['repo']/'results/team_random.json'
    renamed.write_bytes(history['path'].read_bytes())
    with pytest.raises(ValueError, match='explicit result-byte registration'):
        provenance.historical_context(renamed, history['doc'], repo=history['repo'], registry_path=history['registry'])


def test_submission_always_uses_current_manifest_and_current_eligibility(history):
    selected, info = validator.validation_manifest(history['path'], history['doc'], history['current'], submission=True)
    assert selected == history['current'] and info['mode'] == 'current-submission'
    schema = json.loads(Path(validator.SCHEMA).read_text())
    assert any('does not list' in error for error in validator.validate_file(str(history['path']), schema, history['current'], True))
    with pytest.raises(ValueError, match='Current asset manifest'):
        validator.validation_manifest(history['path'], history['doc'], None, submission=True)
    # A new result without an eligibility declaration still cannot evaluate pets.
    new = copy.deepcopy(history['doc']); new['benchmark']['commit'] = 'f'*40
    for episode in new['episodes']: episode['family'] = 'pet_door'
    new_path = history['repo']/'results/team_random.json'; new_path.write_text(json.dumps(new))
    assert any('supplementary pet door' in e for e in validator.validate_file(str(new_path), schema, history['original'], False))


def test_historical_index_coverage_and_lock_groups_use_the_run_manifest(history):
    from scripts.build_results_index import summarize
    source, info = context(history)
    result = summarize(str(history['path']), 1, 0, history['current'], source_manifest=source, validation_context=info)
    assert result['suites']['core']['complete']
    assert result['validation_manifest']['source_commit'] == history['commit']
    assert result['suites']['core']['by_lock_state']['unlocked']['n_doors'] == 1
    assert not summarize(str(history['path']), 1, 0, history['current'])['suites']['core']['complete']


@pytest.mark.parametrize('change', ['dirty', 'unknown_cleanliness', 'short_revision', 'metadata', 'inventory'])
def test_freezing_requires_actual_clean_recorded_source(history, change):
    doc = copy.deepcopy(history['doc'])
    if change == 'dirty': doc['benchmark']['dirty'] = True
    elif change == 'unknown_cleanliness': doc['benchmark'].pop('dirty')
    elif change == 'short_revision': doc['benchmark']['commit'] = history['commit'][:8]
    elif change == 'metadata': doc['benchmark']['dataset_generated'] = '2026-09-08'
    elif change == 'inventory': doc['benchmark']['n_doors_total'] = 1000
    history['path'].write_text(json.dumps(doc))
    with pytest.raises(ValueError): provenance.freeze_historical_results([history['path']], repo=history['repo'])


def test_pr_top_level_glob_excludes_receipts_from_policy_submission_validation(history):
    nested = history['repo']/'results/provenance/receipt.json'; nested.parent.mkdir(); nested.write_text('{}')
    git(history['repo'], 'add', 'results')
    changed = git(history['repo'], 'diff', '--cached', '--name-only', '--', ':(top,glob)results/*.json').splitlines()
    assert 'results/random.json' in changed
    assert 'results/provenance/receipt.json' not in changed

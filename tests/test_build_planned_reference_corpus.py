"""Corpus provenance, independent acceptance, restart and input-preservation gates."""
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

from scripts import build_planned_reference_corpus as corpus


@pytest.fixture
def scene(tmp_path):
    assets = tmp_path/'assets'; recordings = tmp_path/'recordings'; out = tmp_path/'candidates'
    root = tmp_path/'generator'
    for name in ('doorbench/reference/solve.py', 'doorbench/reference/gait.py',
                 'scripts/validate_planned_reference.py', 'scripts/build_planned_reference_corpus.py', 'pyproject.toml'):
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# source\n')
    rows = []
    for door_id in ('one', 'two'):
        directory = assets/'doors'/door_id; directory.mkdir(parents=True)
        (directory/'door.xml').write_text('<mujoco><compiler meshdir="../../hardware"/>'
                                         '<asset><mesh name="handle" file="handle.obj"/></asset></mujoco>')
        (directory/'model.json').write_text('{}'); (directory/'spec.json').write_text('{}')
        sources = {name: corpus.sha(directory/name) for name in corpus.SOURCE_FILES}
        outcome = {'door_id': door_id, 'scenario': 'open_and_traverse', 'success': True, 'outcome': 'success', 'error': None}
        corpus.atomic_json(recordings/'clips'/f'{door_id}.json', {
            'schema': 'doorbench.reference-motion.v1', 'door_id': door_id, 'scenario': 'open_and_traverse',
            'source_sha256': sources, 'outcome': outcome})
        trajectory = recordings/'trajectories'/f'{door_id}.npz'; trajectory.parent.mkdir(exist_ok=True)
        trajectory.write_bytes(b'original native recording')
        rows.append({'id': door_id, 'family': 'swing_single'})
    (assets/'hardware').mkdir(); (assets/'hardware/handle.obj').write_text('v 0 0 0\n')
    corpus.atomic_json(assets/'manifest.json', {'doors': rows})
    scene = {'assets': assets, 'recordings': recordings, 'out': out, 'generator_root': root}
    refresh_native_index(scene)
    return scene


def refresh_native_index(scene):
    recordings = scene['recordings']; clips = []
    for path in sorted((recordings/'clips').glob('*.json')):
        clip = corpus.read_json(path); door_id = clip['door_id']
        clips.append({'door_id': door_id, 'clip': f'clips/{door_id}.json', 'trajectory': f'trajectories/{door_id}.npz',
                      'source_sha256': clip['source_sha256'], 'scenario': clip['scenario'],
                      'success': clip['outcome']['success'], 'outcome': clip['outcome']['outcome'],
                      'clip_sha256': corpus.sha(path), 'trajectory_sha256': corpus.sha(recordings/'trajectories'/f'{door_id}.npz')})
    corpus.atomic_json(recordings/'index.json', {'schema': 'doorbench.reference-motion.v1',
                       'manifest_sha256': corpus.sha(scene['assets']/'manifest.json'), 'clips': clips})


def plan(scene, **kwargs):
    return corpus.prepare_plan(**scene, **kwargs)


def fake_solver(door_dir, recordings, out, *, fps, max_frames, gait_profile):
    directory = Path(out)/Path(door_dir).name; directory.mkdir(exist_ok=True, parents=True)
    source = corpus.read_json(Path(recordings)/'clips'/f'{directory.name}.json')
    (directory/'trajectory.npz').write_bytes(b'new original actor poses')
    clip = {'schema': 'doorbench.planned-reference.v1', 'door_id': directory.name, 'frames': 3, 'duration': .1,
            'complete_proposal': max_frames is None, 'source_sha256': source['source_sha256'],
            'proposal': {'source_outcome': source['outcome']}}
    corpus.atomic_json(directory/'clip.json', clip)
    corpus.atomic_json(directory/'solver-diagnostics.json', {'claim': 'not trusted'})
    print('authored candidate')
    return clip


def fake_validator(clip_path, trajectory_path, assets):
    clip = corpus.read_json(clip_path)
    return {'schema': 'doorbench.planned-reference-validation.v1', 'door_id': clip['door_id'],
            'status': 'accepted_kinematic', 'accepted': True, 'kinematic_accepted': True,
            'failure_counts': {}, 'task_completion': {'complete_proposal': True, 'evidence_pass': True},
            'clip_sha256': corpus.sha(clip_path), 'trajectory_sha256': corpus.sha(trajectory_path),
            'source_sha256': clip['source_sha256']}


def run_first(scene, **kwargs):
    job = plan(scene, doors='one', **kwargs)['rows'][0]['job']
    return corpus.run_job(job, solver=fake_solver, validator=fake_validator)


def test_inventory_all_subset_shared_resources_and_runtime_identity(scene):
    full = plan(scene); subset = plan(scene, doors='two')
    assert full['selected_ids'] == ['one', 'two'] and subset['selected_ids'] == ['two']
    assert [r['action'] for r in full['rows']] == ['run', 'run']
    job = full['rows'][0]['job']
    assert 'hardware/handle.obj' in job['provenance']['native_resources_sha256']
    assert job['provenance']['generator_sha256'] == full['generator']['sha256']
    assert full['generator']['runtime']['packages']['numpy']
    assert not scene['out'].exists(), 'inventory must not create attempts'


@pytest.mark.parametrize('doors', ['', 'missing', 'one,one', '../assets'])
def test_invalid_selection_rejected(scene, doors):
    with pytest.raises(ValueError):
        plan(scene, doors=doors)


@pytest.mark.parametrize('kwargs', [{'fps': 9}, {'fps': True}, {'max_frames': 1}, {'gait_profile': 'unknown'},
                                    {'timeout_s': 0}, {'timeout_s': float('inf')}, {'timeout_s': True}])
def test_invalid_options_rejected(scene, kwargs):
    with pytest.raises(ValueError):
        plan(scene, **kwargs)


def test_missing_recording_is_explicit_unresolved_inventory(scene):
    (scene['recordings']/'trajectories/one.npz').unlink()
    row = plan(scene)['rows'][0]
    assert row['action'] == 'blocked' and row['status'] == 'unresolved'
    assert 'FileNotFoundError' in row['error']


def test_changed_door_source_cannot_reuse_native_recording(scene):
    (scene['assets']/'doors/one/spec.json').write_text('{"changed":true}')
    row = plan(scene)['rows'][0]
    assert row['action'] == 'blocked' and 'different source bytes' in row['error']


def test_mesh_mutation_and_generator_mutation_are_detected(scene):
    job = plan(scene)['rows'][0]['job']
    (scene['assets']/'hardware/handle.obj').write_text('v 1 0 0\n')
    with pytest.raises(ValueError, match='Input changed'):
        corpus.verify_job(job)
    job = plan(scene)['rows'][0]['job']
    (scene['generator_root']/'doorbench/reference/gait.py').write_text('# new gait\n')
    with pytest.raises(ValueError, match='Generator/runtime changed'):
        corpus.verify_job(job)


@pytest.mark.parametrize('which', ['assets', 'recordings'])
def test_output_may_not_overlap_frozen_inputs(scene, which):
    for out in (scene[which], scene[which]/'new', scene[which].parent):
        with pytest.raises(ValueError, match='overlaps immutable'):
            plan({**scene, 'out': out})


def test_external_xml_resource_rejected(scene, tmp_path):
    external = tmp_path/'external.obj'; external.write_text('v 0 0 0')
    xml = scene['assets']/'doors/one/door.xml'
    xml.write_text(f'<mujoco><asset><mesh file="{external}"/></asset></mujoco>')
    with pytest.raises(ValueError, match='escapes assets'):
        corpus.xml_dependencies(xml, scene['assets'])


def test_independent_complete_acceptance_resume_and_frozen_inputs_unchanged(scene):
    inputs = {str(p): corpus.sha(p) for base in (scene['assets'], scene['recordings']) for p in base.rglob('*') if p.is_file()}
    result = run_first(scene)
    assert result['status'] == 'accepted_kinematic'
    assert result['new_completion']['complete_proposal'] is True
    assert result['source_outcome']['outcome'] == 'success'
    assert corpus.read_json(scene['out']/'one/clip.json').get('status') is None, 'acceptance report must not mutate its hashed clip'
    resumed = plan(scene, doors='one')['rows'][0]
    assert resumed['action'] == 'resume' and resumed['result']['run_id'] == result['run_id']
    for path, digest in inputs.items():
        assert corpus.sha(path) == digest


@pytest.mark.parametrize('mutation', [
    {'accepted': False}, {'accepted': 'true'}, {'kinematic_accepted': False},
    {'task_completion': {'complete_proposal': True, 'evidence_pass': False}},
    {'task_completion': {'evidence_pass': True}}, {'clip_sha256': 'wrong'},
    {'source_sha256': {}}, {'failure_counts': {'collision': 1}}, {'status': 'invalid_input'}, {'schema': 'unknown'},
])
def test_permissive_or_unbound_validator_never_promotes_to_accepted(scene, mutation):
    job = plan(scene, doors='one')['rows'][0]['job']
    def validate(*args):
        return {**fake_validator(*args), **mutation}
    result = corpus.run_job(job, solver=fake_solver, validator=validate)
    assert result['status'] == 'rejected'


def test_truncated_proposal_is_rejected_even_if_validator_claims_completion(scene):
    result = run_first(scene, max_frames=2)
    assert result['status'] == 'rejected'
    assert result['new_completion']['complete_proposal'] is False


def test_source_failure_is_explicit_unresolved_without_attempting_new_motion(scene):
    path = scene['recordings']/'clips/one.json'; source = corpus.read_json(path)
    source['outcome'].update(success=False, outcome='timeout')
    corpus.atomic_json(path, source)
    refresh_native_index(scene)
    result = run_first(scene)
    assert result['status'] == 'unresolved'
    assert result['reason_code'] == 'native_source_unsuccessful'
    assert result['source_outcome']['outcome'] == 'timeout'
    assert result['new_completion']['task_evidence_pass'] is False
    assert result['new_completion']['source_success_declared'] is False
    assert not (scene['out']/'one/clip.json').exists()


def test_exception_has_current_attempt_failure_trace_and_log(scene):
    def broken(*args, **kwargs):
        print('attempt reached solver')
        raise RuntimeError('no scene route')
    job = plan(scene, doors='one')['rows'][0]['job']
    result = corpus.run_job(job, solver=broken, validator=fake_validator)
    directory = scene['out']/'one'
    assert result['status'] == 'unresolved'
    assert result['error']['type'] == 'RuntimeError'
    assert 'no scene route' in result['error']['traceback']
    assert corpus.read_json(directory/'failure.json')['run_id'] == result['run_id']
    assert 'attempt reached solver' in (directory/'attempt.log').read_text()
    assert plan(scene, doors='one')['rows'][0]['action'] == 'resume', 'retry unresolved attempts must be explicit'


def test_changed_artifacts_or_options_block_resume_without_overwrite(scene):
    result = run_first(scene); directory = scene['out']/'one'
    assert plan(scene, doors='one', fps=30)['rows'][0]['action'] == 'blocked'
    (directory/'trajectory.npz').write_bytes(b'changed')
    row = plan(scene, doors='one')['rows'][0]
    assert row['action'] == 'blocked' and 'artifact changed' in row['error']
    assert corpus.read_json(directory/'result.json')['run_id'] == result['run_id']


def test_force_archives_old_failure_without_contaminating_new_attempt(scene):
    directory = scene['out']/'one'; directory.mkdir(parents=True)
    corpus.atomic_json(directory/'failure.json', {'error': 'previous AI failure'})
    assert plan(scene, doors='one')['rows'][0]['action'] == 'blocked'
    result = run_first(scene, force=True)
    assert result['status'] == 'accepted_kinematic'
    assert not (directory/'failure.json').exists()
    assert corpus.read_json(scene['out']/result['previous_attempt']/'failure.json')['error'] == 'previous AI failure'
    (directory/'failure.json').write_text('{}')
    row = plan(scene, doors='one')['rows'][0]
    assert row['action'] == 'blocked' and 'stale failure' in row['error']


def test_generator_change_during_solve_is_unresolved_not_accepted(scene):
    job = plan(scene, doors='one')['rows'][0]['job']
    def changing(*args, **kwargs):
        result = fake_solver(*args, **kwargs)
        (scene['generator_root']/'doorbench/reference/solve.py').write_text('# modified during solve')
        return result
    result = corpus.run_job(job, solver=changing, validator=fake_validator)
    assert result['status'] == 'unresolved' and 'changed since preparation' in result['error']['message']


def test_snapshot_counts_scope_separates_native_and_new_results(scene):
    run_first(scene)
    prepared = plan(scene)
    report = corpus.write_snapshot(prepared)
    index = corpus.read_json(scene['out']/'index.json')
    assert report['snapshot_id'] == index['snapshot_id']
    assert report['status_counts'] == {'accepted_kinematic': 1, 'rejected': 0, 'unresolved': 1}
    assert report['source_outcome_counts'] == {'success': 2}
    assert report['accepted_ids'] == ['one']
    assert report['personal_visual_review'] == 'not performed by this runner'
    assert not list(scene['out'].glob('*.tmp'))


def test_plan_only_writes_inventory_without_invoking_solver(scene, monkeypatch):
    real_prepare = corpus.prepare_plan
    monkeypatch.setattr(corpus, 'prepare_plan', lambda *a, **kw: real_prepare(*a, **kw, generator_root=scene['generator_root']))
    monkeypatch.setattr(corpus, 'run_plan', lambda *a: pytest.fail('plan-only must not execute workers'))
    args = ['--assets', str(scene['assets']), '--recordings', str(scene['recordings']), '--out', str(scene['out']), '--plan-only']
    assert corpus.main(args) == 0
    assert corpus.read_json(scene['out']/'plan.json')['selected_ids'] == ['one', 'two']
    assert not (scene['out']/'one').exists() and not (scene['out']/'index.json').exists()


def test_advisory_lock_prevents_concurrent_corpus_writers(scene):
    with corpus.corpus_lock(scene['out']):
        with pytest.raises(ValueError, match='Another corpus runner'):
            with corpus.corpus_lock(scene['out']):
                pytest.fail('second writer acquired lock')


def test_parallel_process_workers_capture_changed_inputs_and_snapshot_every_result(scene):
    prepared = plan(scene)
    # Child processes must reject the changed shared hardware before importing
    # the real solver. This exercises process isolation and durable aggregation.
    (scene['assets']/'hardware/handle.obj').write_text('v 2 0 0')
    report = corpus.run_plan(prepared, workers=2)
    assert report['status_counts'] == {'unresolved': 2, 'rejected': 0, 'accepted_kinematic': 0}
    assert report['action_counts'] == {'blocked': 2}, 'final refresh must flag the now-stale input identity'
    for door_id in ['one', 'two']:
        failure = corpus.read_json(scene['out']/door_id/'failure.json')
        assert failure['error']['type'] == 'ValueError'
        assert 'Input changed' in failure['error']['message']


def test_guide_preflight_keeps_unresolved_task_distinct_from_kinematic_rejection(scene):
    job = plan(scene, doors='one')['rows'][0]['job']
    assert job['provenance']['options']['fps'] == 60
    assert job['provenance']['options']['gait_profile'] == 'smooth'
    def guide(*args, **kwargs):
        assert kwargs == {'fps': 60, 'gait_profile': 'smooth'}
        return SimpleNamespace(metadata={'traversal': 'unresolved', 'traversal_reason': 'no aperture route'})
    with pytest.raises(corpus.TaskUnresolved, match='no aperture route') as exc:
        corpus.preflight_guide(job, guide)
    assert exc.value.code == 'guide_traversal_unresolved'
    good = lambda *a, **kw: SimpleNamespace(metadata={'traversal': 'proposed'})
    assert 'not passed' in corpus.preflight_guide(job, good)['scope']


def test_source_change_after_preparation_invalidates_resumed_acceptance(scene):
    original = run_first(scene)
    prepared = plan(scene, doors='one')
    assert prepared['rows'][0]['action'] == 'resume'
    (scene['assets']/'hardware/handle.obj').write_text('v 3 0 0')
    report = corpus.run_plan(prepared, workers=1)
    assert report['status_counts']['accepted_kinematic'] == 0
    assert report['status_counts']['unresolved'] == 1
    row = corpus.read_json(scene['out']/'index.json')['doors'][0]
    assert row['action'] == 'blocked' and 'Input changed' in row['error']
    assert corpus.read_json(scene['out']/'one/result.json')['run_id'] == original['run_id']


@pytest.mark.parametrize('artifact', ['clips/one.json', 'trajectories/one.npz'])
def test_native_replacement_cannot_create_new_valid_identity_against_frozen_index(scene, artifact):
    path = scene['recordings']/artifact
    path.write_bytes(path.read_bytes()+b'\n')
    row = plan(scene, doors='one')['rows'][0]
    assert row['action'] == 'blocked' and 'frozen recording index' in row['error']


def fake_worker_command(tmp_path, behavior):
    """Use real OS processes/signals, but never import or crash the engine."""
    script = tmp_path/'fake-worker.py'
    script.write_text(f'''
import os, signal, sys, time, importlib.util, faulthandler
sys.path.insert(0, {str(corpus.ROOT)!r})
from scripts import build_planned_reference_corpus as corpus
spec=importlib.util.spec_from_file_location('corpus_test_helpers', {str(Path(__file__).resolve())!r})
helpers=importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
job=corpus.read_json(sys.argv[1])
faulthandler.enable(all_threads=True)
corpus.stage_marker(job,'fake_worker_start')
if job['door_id']=='one':
    {behavior}
corpus.run_job(job,solver=helpers.fake_solver,validator=helpers.fake_validator,publish=False)
corpus.stage_marker(job,'fake_worker_end')
''')
    return lambda path: [sys.executable, '-u', str(script), str(path)]


@pytest.mark.parametrize('behavior,kind,returncode', [
    ('os._exit(23)', 'nonzero_exit', 23),
    ('os.kill(os.getpid(), signal.SIGKILL)', 'signal', -9),
])
def test_native_exit_is_isolated_other_doors_complete_and_resume(scene, tmp_path, behavior, kind, returncode):
    prepared = plan(scene, timeout_s=10)
    command = fake_worker_command(tmp_path, behavior)
    launch = lambda job, **kw: corpus.isolated_job(job, command_factory=command, **kw)
    report = corpus.run_plan(prepared, workers=2, launcher=launch)
    assert report['status_counts'] == {'unresolved': 1, 'rejected': 0, 'accepted_kinematic': 1}
    assert report['execution_failure_counts'] == {kind: 1}
    failed = corpus.read_json(scene['out']/'one/result.json')
    good = corpus.read_json(scene['out']/'two/result.json')
    assert failed['reason_code'] == 'execution_failure'
    assert failed['execution']['failure_kind'] == kind and failed['execution']['returncode'] == returncode
    assert failed['new_completion'] is None, 'process failure is not physical infeasibility'
    assert failed['execution']['pid'] != good['execution']['pid']
    log = (scene['out']/'one/execution.log').read_text()
    assert 'fake_worker_start' in log and str(failed['execution']['pid']) in log
    assert 'fake_worker_end' in (scene['out']/'two/execution.log').read_text()
    before = (scene['out']/'two/result.json').read_bytes()
    resumed = plan(scene, timeout_s=10)
    assert [row['action'] for row in resumed['rows']] == ['resume', 'resume']
    corpus.run_plan(resumed, workers=2, launcher=lambda *a, **kw: pytest.fail('valid resumes must not spawn'))
    assert (scene['out']/'two/result.json').read_bytes() == before


def test_timeout_kills_only_one_job_and_later_queued_door_still_completes(scene, tmp_path):
    prepared = plan(scene, timeout_s=1)
    command = fake_worker_command(tmp_path, 'time.sleep(30)')
    launch = lambda job, **kw: corpus.isolated_job(job, command_factory=command, **kw)
    report = corpus.run_plan(prepared, workers=1, launcher=launch)
    assert report['status_counts'] == {'unresolved': 1, 'rejected': 0, 'accepted_kinematic': 1}
    assert report['execution_failure_counts'] == {'timeout': 1}
    failed = corpus.read_json(scene['out']/'one/result.json')
    assert failed['execution']['runtime_s'] < 3
    assert failed['execution']['returncode'] == -9


def test_cancellation_is_explicit_and_does_not_start_a_subprocess(scene):
    job = plan(scene, doors='one')['rows'][0]['job']; cancel = threading.Event(); cancel.set()
    result = corpus.isolated_job(job, cancel=cancel)
    assert result['status'] == 'unresolved' and result['reason_code'] == 'execution_failure'
    assert result['execution']['failure_kind'] == 'cancelled'
    assert result['execution']['pid'] is None


def test_isolation_and_timeout_are_resume_provenance(scene):
    prepared = plan(scene)
    assert prepared['execution'] == {'isolation_mode': 'fresh_python_process_per_door', 'timeout_s': 600.}
    assert prepared['generator']['runtime']['isolation_mode'] == 'fresh_python_process_per_door'
    run_first(scene)
    assert plan(scene, doors='one', timeout_s=601)['rows'][0]['action'] == 'blocked'


def test_snapshot_rewrites_are_throttled_during_fast_results(scene, monkeypatch):
    prepared = plan(scene)
    writes=[]; original=corpus.write_snapshot
    def snapshot(p):
        writes.append(True)
        return original(p)
    monkeypatch.setattr(corpus, 'write_snapshot', snapshot)
    launch = lambda job, **kw: corpus.run_job(job, solver=fake_solver, validator=fake_validator)
    corpus.run_plan(prepared, workers=1, launcher=launch, snapshot_interval_s=100)
    assert len(writes) == 2, 'initial and final snapshots suffice; each result is separately durable'


def test_parent_interrupt_cancels_queue_and_writes_final_snapshot(scene, monkeypatch):
    prepared = plan(scene)
    def interrupted(_):
        raise KeyboardInterrupt
    def launch(job, *, cancel):
        assert cancel.wait(2), 'parent did not propagate interruption to active supervisors'
        return corpus.isolated_job(job, cancel=cancel)
    monkeypatch.setattr(corpus, 'as_completed', interrupted)
    with pytest.raises(KeyboardInterrupt):
        corpus.run_plan(prepared, workers=1, launcher=launch)
    report = corpus.read_json(scene['out']/'report.json')
    assert report['status_counts']['accepted_kinematic'] == 0
    assert report['status_counts']['unresolved'] == 2
    assert 'run' not in report['action_counts']

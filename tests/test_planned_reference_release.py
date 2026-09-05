"""Immutable experimental packaging: accepted-only bytes and honest status scope."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

import pytest

from scripts import planned_reference_release as release
from scripts import build_planned_reference_corpus as runner
from scripts.export_planned_reference_web import export_corpus
from tests.test_build_planned_reference_corpus import scene, plan, fake_solver, fake_validator
from tests.test_export_planned_reference_web import corpus


def write(path, value):
    release.common.write_json(path, value)


def load(path):
    return release.common.read(path)


def digest(path):
    return release.common.sha256(path)


def test_native_dependency_reads_published_inventory_schema_and_verifies_pin(tmp_path, monkeypatch):
    write(tmp_path/'inventory.json', {'schema_version': 1, 'files': {'reference-motions/index.json': {'sha256': 'a'*64}}})
    write(tmp_path/'release.json', {'release': release.BASE_TAG, 'repo_id': release.BASE_REPO,
        'inventory_sha256': digest(tmp_path/'inventory.json'), 'summary': {'dataset_manifest_sha256': 'b'*64},
        'components': {name: {'path': f'archives/{name}.tar.gz', 'sha256': 'c'*64} for name in ['assets', 'reference-motions']}})
    monkeypatch.setattr(release, 'BASE_RELEASE_SHA', digest(tmp_path/'release.json'))
    native = release.native_dependency(tmp_path/'release.json')
    assert native['recording_index_sha256'] == 'a'*64
    assert '/resolve/'+release.BASE_COMMIT+'/' in native['components']['assets']['url']
    (tmp_path/'release.json').write_bytes((tmp_path/'release.json').read_bytes()+b' ')
    with pytest.raises(ValueError, match='pinned'): release.native_dependency(tmp_path/'release.json')


@pytest.fixture
def complete(scene):
    for row in plan(scene)['rows']:
        runner.run_job(row['job'], solver=fake_solver, validator=fake_validator)
    runner.write_snapshot(plan(scene))
    (scene['out']/'.corpus.lock').touch()
    native = {'manifest_sha256': digest(scene['assets']/'manifest.json'),
              'recording_index_sha256': digest(scene['recordings']/'index.json')}
    return scene, native


def inspect(fixture):
    scene, native = fixture
    return release.inspect_corpus(scene['out'], scene['assets'], scene['recordings'], native,
                                  expected_doors=2, generator_root=scene['generator_root'])


def test_complete_inventory_revalidates_every_result_and_native_binding(complete):
    checked = inspect(complete)
    assert checked['counts'] == {'accepted_kinematic': 2}
    assert checked['accepted_bytes'] > 0
    scene, native = complete
    native['recording_index_sha256'] = '0'*64
    with pytest.raises(ValueError, match='recording index'):
        inspect(complete)


@pytest.mark.parametrize('change', ['pending', 'snapshot', 'generator', 'artifact', 'duplicate', 'count'])
def test_incomplete_stale_or_forged_corpus_never_packages(complete, change):
    scene, _ = complete
    path = scene['out']/'index.json'; data = load(path)
    if change == 'pending': data['doors'][0]['action'] = 'run'
    elif change == 'snapshot': data['snapshot_id'] = 'different'
    elif change == 'generator': (scene['generator_root']/'doorbench/reference/solve.py').write_text('changed')
    elif change == 'artifact': (scene['out']/'one/trajectory.npz').write_bytes(b'changed')
    elif change == 'duplicate': data['doors'][1] = data['doors'][0]
    else:
        report = load(scene['out']/'report.json'); report['status_counts']['accepted_kinematic'] = 99
        write(scene['out']/'report.json', report)
    write(path, data)
    with pytest.raises(ValueError): inspect(complete)


def test_running_corpus_lock_blocks_packaging_without_mutating_sources(complete):
    scene, _ = complete; path = scene['out']/'.corpus.lock'
    with path.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='still running'):
            with release.idle_corpus(scene['out']): pass
    with release.idle_corpus(scene['out']): assert path.read_bytes() == b''


@pytest.mark.parametrize('tag', ['main', 'v2026.09.05', 'planned-', 'planned-../escape', 'planned-a/b'])
def test_release_namespace_cannot_replace_native_or_escape(tag):
    with pytest.raises(ValueError): release.prefix(tag)


@pytest.fixture
def bundle(corpus, tmp_path):
    root, _, assets = corpus; out = tmp_path/'bundle'; out.mkdir()
    web = export_corpus(root, out/'web', assets)
    # Real independently validated original-rig fixture plus 999 nonplayable rows.
    web['doors'][1]['reason'] = 'No accepted complete motion was produced.'
    for index in range(998):
        web['doors'].append({'door_id': f'waiting{index:04}', 'family': 'swing_single', 'status': 'unresolved',
                             'reason': 'Source success unavailable.', 'reason_code': 'source_failure', 'clip': None, 'audits': {}})
    web['counts'] = {'accepted_kinematic': 1, 'unresolved': 999}
    release.project_rejection_reports(web, out/'web')
    result = load(root/'fixture/result.json')
    audit = {'results': {'fixture': result, 'rejected': {'status': 'rejected'}}}
    archives, downloads = release.archive_accepted(audit, root, out, 1024**2)
    shutil.rmtree(out/'rig-sources')
    for row in web['doors']: row['research_download'] = downloads.get(row['door_id'])
    write(out/'web/index.json', web)
    (out/'status.jsonl').write_bytes(b''.join(release.common.canonical(row)+b'\n' for row in web['doors']))
    native = {'commit': release.BASE_COMMIT, 'release_sha256': release.BASE_RELEASE_SHA, 'manifest_sha256': web['manifest_sha256']}
    write(out/'native-dependency.json', native)
    for name in ['README.md', 'LICENSE', 'LIMITATIONS.md', 'download.py', 'archive_helpers.py']:
        (out/name).write_text('public support file\n')
    manifest = {'schema': release.SCHEMA, 'experimental': True, 'release': 'planned-test-v1', 'repo_id': release.BASE_REPO,
                'path_in_repo': release.prefix('planned-test-v1'), 'complete_corpus': True, 'doors': 1000,
                'counts': web['counts'], 'accepted_scenarios':release.accepted_scenario_counts(web['doors']),'corpus_index_sha256': web['corpus_index_sha256'],
                'generator': {'sha256': web['generator_sha256']}, 'native_dependency': native,
                'archives': archives, 'browser_compatibility': release.browser_compatibility(web, out/'web')}
    write(out/'release.json', manifest); refresh(out)
    return out, root, assets


def refresh(folder):
    manifest = load(folder/'release.json')
    manifest['files'] = {p.relative_to(folder).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size, 'license': 'MIT'}
                         for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ['release.json', 'publication.json']}
    manifest['files_sha256'] = hashlib.sha256(release.common.canonical(manifest['files'])).hexdigest()
    write(folder/'release.json', manifest)


def test_research_archive_exact_original_bytes_and_accepted_only(bundle):
    out, root, _ = bundle; manifest, files = release.release_files(out)
    assert manifest['counts'] == {'accepted_kinematic': 1, 'unresolved': 999}
    assert manifest['accepted_scenarios']=={'locked_recognize':1}
    archive = next(iter(manifest['archives'].values()))
    with tarfile.open(out/archive['path'], 'r:gz') as stream:
        assert set(stream.getnames()) == {f'accepted/fixture/{name}' for name in ['clip.json', 'trajectory.npz', 'validation.json', 'actor.xml']}
        for name in ['clip.json', 'trajectory.npz', 'validation.json']:
            assert stream.extractfile('accepted/fixture/'+name).read() == (root/'fixture'/name).read_bytes()
        assert stream.extractfile('accepted/fixture/actor.xml').read() == load(root/'fixture/clip.json')['actor']['mjcf_xml'].encode()
    assert all('execution.log' not in name and '/assets/' not in name for name in files)


def test_archive_refuses_changed_artifacts_after_export(corpus, tmp_path):
    root, _, _ = corpus; result = load(root/'fixture/result.json')
    (root/'fixture/trajectory.npz').write_bytes(b'changed')
    out = tmp_path/'archive'; out.mkdir()
    with pytest.raises(ValueError, match='changed before archive'):
        release.archive_accepted({'results': {'fixture': result}}, root, out, 1024**2)


@pytest.mark.parametrize('name', ['README.md', 'web/index.json', 'research-inventory.json'])
def test_prepared_support_and_index_changes_block_publication(bundle, name):
    out, _, _ = bundle; (out/name).write_bytes((out/name).read_bytes()+b' ')
    with pytest.raises(ValueError, match='file changed'): release.release_files(out)


@pytest.mark.parametrize('change', ['missing_row', 'nonaccepted_clip', 'download_redirect', 'extra_member', 'wrong_rig', 'false_validation', 'duplicate_archive', 'native_manifest', 'source_scenario','scenario_counts'])
def test_semantic_violations_fail_even_with_refreshed_file_checksums(bundle, change):
    out, _, _ = bundle; manifest = load(out/'release.json'); web = load(out/'web/index.json')
    archive_name = next(iter(manifest['archives'])); inventory = load(out/'research-inventory.json')
    row = web['doors'][0]
    if change == 'missing_row': web['doors'].pop()
    elif change == 'scenario_counts':manifest['accepted_scenarios']={'open_and_traverse':1}
    elif change == 'source_scenario': row['source_scenario']='open_and_traverse'
    elif change == 'native_manifest': web['manifest_sha256'] = '0'*64
    elif change == 'nonaccepted_clip': web['doors'][1]['clip'] = row['clip']
    elif change == 'download_redirect': row['research_download']['member_prefix'] = 'accepted/other/'
    elif change == 'extra_member': inventory[archive_name]['accepted/rejected/trajectory.npz'] = {'bytes': 1, 'sha256': '0'*64}
    elif change == 'wrong_rig': inventory[archive_name]['accepted/fixture/actor.xml']['sha256'] = '0'*64
    elif change == 'duplicate_archive': manifest['archives']['another.tar.gz'] = copy.deepcopy(manifest['archives'][archive_name])
    else:
        path = out/'web'/row['audits']['validation.json']['path']; validation = load(path); validation['accepted'] = False; write(path, validation)
        row['audits']['validation.json'].update(bytes=path.stat().st_size, sha256=digest(path))
        inventory[archive_name]['accepted/fixture/validation.json'].update(bytes=path.stat().st_size, sha256=digest(path))
    manifest['archives'][archive_name]['inventory_sha256'] = hashlib.sha256(release.common.canonical(inventory[archive_name])).hexdigest()
    manifest['archives'][archive_name]['expanded_bytes'] = sum(item['bytes'] for item in inventory[archive_name].values())
    write(out/'release.json', manifest); write(out/'research-inventory.json', inventory); write(out/'web/index.json', web)
    (out/'status.jsonl').write_bytes(b''.join(release.common.canonical(row)+b'\n' for row in web['doors']))
    refresh(out)
    with pytest.raises(ValueError): release.release_files(out)


@pytest.mark.parametrize('limit', list(release.BROWSER_LIMITS))
def test_browser_bounds_report_exact_door_without_truncation(corpus, limit):
    root, out, assets = corpus; web = export_corpus(root, out, assets); before = digest(out/web['doors'][0]['clip']['path'])
    limits = release.BROWSER_LIMITS.copy(); limits[limit] = 1
    result = release.browser_compatibility(web, out, limits=limits)
    assert not result['compatible'] and result['violations'][0]['door_id'] == 'fixture'
    assert limit in result['violations'][0]['limits_exceeded']
    assert digest(out/web['doors'][0]['clip']['path']) == before


def test_rejected_report_projection_retains_hash_and_removes_paths(corpus):
    root, out, assets = corpus; web = export_corpus(root, out, assets)
    from scripts.export_planned_reference_web import audit, encoded
    raw = encoded({'message': 'Cannot read /Users/private/run/file.xml', 'traceback': 'PRIVATE', 'pid': 123})
    web['doors'][1]['audits']['validation.json'] = audit(out, 'waiting', 'validation', raw)
    release.project_rejection_reports(web, out)
    projected = load(out/web['doors'][1]['audits']['validation.json']['path'])
    assert projected['original_validation_sha256'] == hashlib.sha256(raw).hexdigest()
    assert projected['report'] == {'message': 'Cannot read [local path]'}
    assert web['doors'][0]['reason'].startswith('Independent sampled')


def mock_hub(monkeypatch, folder, *, published=True):
    import huggingface_hub
    import httpx
    from huggingface_hub.errors import RevisionNotFoundError
    calls = []; commit = 'e'*40
    class API:
        def __init__(self, token=None): pass
        def repo_info(self, repo, **kwargs):
            if kwargs.get('revision') == 'planned-test-v1' and not published and not any(c[0] == 'tag' for c in calls):
                raise RevisionNotFoundError('No such revision', response=httpx.Response(404, request=httpx.Request('GET', 'https://example.test')))
            return SimpleNamespace(sha=commit, private=False)
        def upload_folder(self, **kwargs): calls.append(('upload', kwargs)); return SimpleNamespace(oid=commit)
        def create_tag(self, *args, **kwargs): calls.append(('tag', kwargs))
    def fetch(repo, name, **kwargs):
        assert kwargs['revision'] == commit and kwargs['token'] is False
        relative = name.removeprefix(release.prefix('planned-test-v1')+'/'); calls.append(('fetch', relative))
        return str(folder/relative)
    monkeypatch.setattr(huggingface_hub, 'HfApi', API); monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fetch)
    monkeypatch.setattr(release, 'verify_remote', lambda *args: calls.append(('verified', list(args[-1]))))
    return calls, commit


def test_download_verifies_safe_archive_and_retains_full_inventory(bundle, monkeypatch, tmp_path):
    folder, root, _ = bundle; calls, commit = mock_hub(monkeypatch, folder)
    out = tmp_path/'installed'
    result = release.download(SimpleNamespace(repo_id=release.BASE_REPO, release='planned-test-v1', revision='planned-test-v1', archives='all', out=out))
    assert result['revision'] == commit
    assert (out/'accepted/fixture/trajectory.npz').read_bytes() == (root/'fixture/trajectory.npz').read_bytes()
    assert (out/'research-inventory.json').read_bytes() == (folder/'research-inventory.json').read_bytes()
    assert load(out/'installed.json')['revision'] == commit
    assert not any(name.startswith('web/clips') for kind, name in calls if kind == 'fetch')


def test_failed_download_leaves_no_partially_installed_directory(bundle, monkeypatch, tmp_path):
    folder, _, _ = bundle; mock_hub(monkeypatch, folder)
    archive = next(iter(load(folder/'release.json')['archives'].values())); (folder/archive['path']).write_bytes(b'corrupt')
    out = tmp_path/'installed'
    with pytest.raises(ValueError, match='checksum'):
        release.download(SimpleNamespace(repo_id=release.BASE_REPO, release='planned-test-v1', revision='planned-test-v1', archives='all', out=out))
    assert not out.exists() and not list(tmp_path.glob('.planned-download-*'))


def test_download_rejects_web_manifest_detached_from_native_dependency(bundle, monkeypatch, tmp_path):
    folder, _, _ = bundle; mock_hub(monkeypatch, folder)
    web = load(folder/'web/index.json'); web['manifest_sha256'] = '0'*64; write(folder/'web/index.json', web); refresh(folder)
    out = tmp_path/'installed'
    with pytest.raises(ValueError, match='native manifest binding'):
        release.download(SimpleNamespace(repo_id=release.BASE_REPO, release='planned-test-v1', revision='planned-test-v1', archives='all', out=out))
    assert not out.exists()


def test_publish_is_nested_allowlisted_and_receipt_is_commit_pinned(bundle, monkeypatch):
    folder, _, _ = bundle; calls, commit = mock_hub(monkeypatch, folder, published=False)
    monkeypatch.setenv('HF_TOKEN', 'private-test-token')
    (folder/'private-not-in-manifest.txt').write_text('must not upload')
    result = release.publish(SimpleNamespace(folder=folder, token_file=None, dry_run=False))
    upload = next(value for kind, value in calls if kind == 'upload')
    assert upload['path_in_repo'] == release.prefix('planned-test-v1')
    assert 'private-not-in-manifest.txt' not in upload['allow_patterns'] and 'publication.json' not in upload['allow_patterns']
    assert result['web_index_url'].split('/resolve/')[1].startswith(commit+'/')
    assert result['counts'] == {'accepted_kinematic': 1, 'unresolved': 999}
    assert result['web_index_sha256'] == digest(folder/'web/index.json')
    assert result['native_manifest_sha256'] == load(folder/'web/index.json')['manifest_sha256']
    assert 'private-test-token' not in (folder/'publication.json').read_text()
    assert [kind for kind, _ in calls][-2:] == ['verified', 'tag']


def test_existing_identical_tag_is_reverified_without_upload(bundle, monkeypatch):
    folder, _, _ = bundle; calls, _ = mock_hub(monkeypatch, folder)
    monkeypatch.setenv('HF_TOKEN', 'private-test-token')
    release.publish(SimpleNamespace(folder=folder, token_file=None, dry_run=False))
    assert not any(kind in ('upload', 'tag') for kind, _ in calls)
    assert any(kind == 'verified' for kind, _ in calls)


def test_existing_tag_different_manifest_cannot_be_overwritten(bundle, monkeypatch, tmp_path):
    folder, _, _ = bundle; calls, _ = mock_hub(monkeypatch, folder)
    monkeypatch.setenv('HF_TOKEN', 'private-test-token')
    different = load(folder/'release.json'); different['corpus_index_sha256'] = '0'*64
    remote = tmp_path/'remote-release.json'; write(remote, different)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *args, **kwargs: str(remote))
    with pytest.raises(ValueError, match='different bytes'):
        release.publish(SimpleNamespace(folder=folder, token_file=None, dry_run=False))
    assert not any(kind in ('upload', 'tag') for kind, _ in calls)


def test_publish_dry_run_never_loads_credentials_or_contacts_hub(bundle, monkeypatch):
    folder, _, _ = bundle
    result = release.publish(SimpleNamespace(folder=folder, token_file='does-not-exist', dry_run=True))
    assert result['published'] is False


@pytest.mark.parametrize('when',['upload','remote_verification','manifest_reformat'])
def test_staging_mutation_during_publication_never_creates_tag_or_receipt(bundle,monkeypatch,when):
    import huggingface_hub
    folder,_,_=bundle;calls,_=mock_hub(monkeypatch,folder,published=False)
    monkeypatch.setenv('HF_TOKEN','private-test-token')
    if when=='upload':
        original=huggingface_hub.HfApi.upload_folder
        def changing_upload(self,**kwargs):
            result=original(self,**kwargs);(folder/'README.md').write_text('changed during upload');return result
        monkeypatch.setattr(huggingface_hub.HfApi,'upload_folder',changing_upload)
    else:
        def changing_verification(*args):
            calls.append(('verified',[]))
            if when=='manifest_reformat':(folder/'release.json').write_bytes((folder/'release.json').read_bytes()+b' ')
            else:(folder/'README.md').write_text('changed during remote verification')
        monkeypatch.setattr(release,'verify_remote',changing_verification)
    with pytest.raises(ValueError,match='changed'):
        release.publish(SimpleNamespace(folder=folder,token_file=None,dry_run=False))
    assert not any(kind=='tag' for kind,_ in calls)
    assert not (folder/'publication.json').exists()


def test_prepare_dry_run_waits_for_active_corpus(complete, monkeypatch, tmp_path):
    scene, native = complete; monkeypatch.setattr(release, 'native_dependency', lambda _: native)
    args = SimpleNamespace(release='planned-test-v1', repo_id=release.BASE_REPO, corpus=scene['out'], assets=scene['assets'],
                           recordings=scene['recordings'], native_release='unused', out=tmp_path/'release', dry_run=True)
    with (scene['out']/'.corpus.lock').open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = release.prepare(args)
    assert result['ready'] is False and 'still running' in result['reason']
    assert not args.out.exists() and (tmp_path/'release.plan.json').exists()


@pytest.mark.parametrize('dry_run', [False, True])
def test_local_prepare_runs_export_archive_and_browser_checks_without_publication(corpus, monkeypatch, tmp_path, dry_run):
    root, _, assets = corpus; (root/'.corpus.lock').touch(); write(root/'report.json', {'snapshot_id': 'test'})
    actual_result = load(root/'fixture/result.json')
    checked = {'index': load(root/'index.json'), 'index_sha256': digest(root/'index.json'), 'report_sha256': digest(root/'report.json'),
               'results': {'fixture': actual_result, **{f'waiting{i:04}': {'status': 'unresolved'} for i in range(999)}},
               'counts': {'accepted_kinematic': 1, 'unresolved': 999}, 'accepted_bytes': 100000}
    native = {'commit': release.BASE_COMMIT, 'release_sha256': release.BASE_RELEASE_SHA,
              'manifest_sha256': load(root/'index.json')['manifest_sha256'],
              'release_url': 'https://example.test/pinned/release.json'}
    monkeypatch.setattr(release, 'native_dependency', lambda _: native)
    monkeypatch.setattr(release, 'inspect_corpus', lambda *args: checked)
    monkeypatch.setattr(release, 'verify_commit', lambda *args: {})
    import scripts.export_planned_reference_web as exporter
    def export(*args):
        web = export_corpus(*args)
        web['doors'].extend({'door_id': f'waiting{i:04}', 'family': 'swing_single', 'status': 'unresolved',
                             'reason_code': 'source_failure', 'clip': None, 'audits': {}} for i in range(998))
        web['counts'] = checked['counts']; write(Path(args[1])/'index.json', web); return web
    monkeypatch.setattr(exporter, 'export_corpus', export)
    recordings = tmp_path/'recordings'; recordings.mkdir()
    args = SimpleNamespace(release='planned-test-v1', repo_id=release.BASE_REPO, corpus=root, assets=assets, recordings=recordings,
                           native_release='unused', out=tmp_path/'prepared', source_commit='f'*40, shard_mib=1, dry_run=dry_run)
    before = {p: digest(p) for p in root.rglob('*') if p.is_file()}
    result = release.prepare(args)
    assert result['browser_compatibility']['compatible']
    assert result['counts'] == checked['counts'] and result.get('ready', True)
    assert {p: digest(p) for p in before} == before
    if dry_run:
        assert not args.out.exists() and (tmp_path/'prepared.plan.json').exists()
    else:
        release.release_files(args.out)
        assert result['accepted_scenarios']=={'locked_recognize':1}
        readme=(args.out/'README.md').read_text()
        assert '**0 traversal references**' in readme and '**1 locked-door checks**' in readme
        assert not (args.out/'publication.json').exists()
        assert not (args.out/'rig-sources').exists() and not list(tmp_path.glob('.planned-release-*'))


def test_standalone_download_cli_imports_without_repository(tmp_path):
    shutil.copyfile(release.__file__, tmp_path/'download.py')
    shutil.copyfile(release.common.__file__, tmp_path/'archive_helpers.py')
    import subprocess
    result = subprocess.run([sys.executable, str(tmp_path/'download.py'), 'download', '--help'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def hub_error(status, retry_after=None):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError
    response = httpx.Response(status, headers={} if retry_after is None else {'Retry-After': retry_after},
                              request=httpx.Request('GET', 'https://huggingface.co/api/datasets/example/paths-info'))
    return HfHubHTTPError('Mock metadata response', response=response)


def test_remote_verification_retries_only_failed_twenty_path_batch(tmp_path, monkeypatch):
    # Exercise the real Git-blob verifier after a transient paths-info failure.
    files = {}
    for i in range(43):
        path = tmp_path/f'{i:03}.txt'; path.write_text(str(i)); files[path.name] = path
    calls = []; sleeps = []; failure = hub_error(429, '7')
    class API:
        def repo_info(self, *args, **kwargs):
            assert kwargs['revision'] == 'c'*40
            return SimpleNamespace(private=False, gated=False)
        def get_paths_info(self, repo, names, **kwargs):
            calls.append(names)
            if len(calls) == 2: raise failure
            return [SimpleNamespace(path=name, size=files[name].stat().st_size, lfs=None,
                    blob_id=hashlib.sha1(f'blob {files[name].stat().st_size}\0'.encode()+files[name].read_bytes()).hexdigest())
                    for name in names]
    monkeypatch.setattr(release.time, 'sleep', sleeps.append)
    release.verify_remote(API(), 'owner/repo', 'c'*40, files)
    assert [len(x) for x in calls] == [20, 20, 20, 3]
    assert calls[1] == calls[2] and calls[0] != calls[1]
    assert sleeps == [7]


@pytest.mark.parametrize('status', [400, 401, 403, 404, 409, 500, 502, 504])
def test_remote_verification_permanent_errors_propagate_without_sleep(monkeypatch, status):
    failure = hub_error(status, '3'); calls = []; sleeps = []
    def fail(*args): calls.append(args); raise failure
    monkeypatch.setattr(release.common, 'verify_public_files', fail)
    monkeypatch.setattr(release.time, 'sleep', sleeps.append)
    with pytest.raises(type(failure)) as caught:
        release.verify_remote(None, 'repo', 'commit', {'one': Path('unused')})
    assert caught.value is failure and len(calls) == 1 and sleeps == []


def test_remote_checksum_failure_is_never_retried(monkeypatch):
    sleeps = []; failure = ValueError('Published checksum mismatch')
    monkeypatch.setattr(release.common, 'verify_public_files', lambda *args: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(release.time, 'sleep', sleeps.append)
    with pytest.raises(ValueError, match='checksum mismatch'):
        release.verify_remote(None, 'repo', 'commit', {'one': Path('unused')})
    assert sleeps == []


@pytest.mark.parametrize('header', [None, 'invalid', '-1', 'NaN', 'Infinity'])
def test_remote_503_uses_bounded_exponential_backoff(monkeypatch, header):
    calls = []; sleeps = []; failure = hub_error(503, header)
    def fail(*args): calls.append(args); raise failure
    monkeypatch.setattr(release.common, 'verify_public_files', fail)
    monkeypatch.setattr(release.time, 'sleep', sleeps.append)
    with pytest.raises(type(failure)) as caught:
        release.verify_remote(None, 'repo', 'commit', {'one': Path('unused')})
    assert caught.value is failure and len(calls) == 6
    assert sleeps == [2, 4, 8, 16, 32]


def test_remote_retry_after_http_date_and_excessive_wait(monkeypatch):
    from datetime import datetime, timezone, timedelta
    from email.utils import format_datetime
    now = datetime.now(timezone.utc)
    header = format_datetime(now+timedelta(seconds=30), usegmt=True)
    calls = []; sleeps = []
    def verify(*args):
        calls.append(args)
        if len(calls) == 1: raise hub_error(429, header)
    monkeypatch.setattr(release.common, 'verify_public_files', verify)
    monkeypatch.setattr(release.time, 'sleep', sleeps.append)
    release.verify_remote(None, 'repo', 'commit', {'one': Path('unused')})
    assert len(sleeps) == 1 and 28 <= sleeps[0] <= 30
    failure = hub_error(429, '99999999999999999')
    monkeypatch.setattr(release.common, 'verify_public_files', lambda *args: (_ for _ in ()).throw(failure))
    with pytest.raises(type(failure)) as caught:
        release.verify_remote(None, 'repo', 'commit', {'one': Path('unused')})
    assert caught.value is failure and len(sleeps) == 1


@pytest.mark.parametrize('budget', ['sleep', 'retries'])
def test_remote_retry_budget_spans_all_batches(monkeypatch, budget):
    calls = []; sleeps = []; attempts = {}
    failure = hub_error(429, '60' if budget == 'sleep' else None)
    def verify(api, repo, commit, batch):
        first = next(iter(batch)); attempts[first] = attempts.get(first, 0)+1; calls.append(first)
        if attempts[first] <= (1 if budget == 'sleep' else 4): raise failure
    monkeypatch.setattr(release.common, 'verify_public_files', verify)
    monkeypatch.setattr(release.time, 'sleep', sleeps.append)
    with pytest.raises(type(failure)) as caught:
        release.verify_remote(None, 'repo', 'commit', {f'{i:03}': Path('unused') for i in range(100)})
    assert caught.value is failure
    if budget == 'sleep': assert sleeps == [60, 60, 60] and len(calls) == 7
    else: assert sleeps == [2, 4, 8, 16]*3 and len(calls) == 16

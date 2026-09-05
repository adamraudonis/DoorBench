"""Resume never publishes failed jobs or trusts stale/corrupted render artifacts."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from doorbench.appearance.pipeline import prepare_job, run_jobs
from doorbench.appearance.state import capture_initial_state
from doorbench.build import export_door
from doorbench.spec import generate_all


def _fixture(tmp_path, *, job_hash='current'):
    out = tmp_path / 'appearance'
    dest = out / 'door/variant_000'
    dest.mkdir(parents=True)
    job = {'door_id': 'door', 'job_sha256': job_hash, 'out_dir': str(dest), 'variant': 0,
           'validate_only': False, 'save_blend': False, 'recipe': {}, 'quality': 'preview', 'source_sha256': {}}
    metadata = {'door_id': 'door', 'job_sha256': job_hash, 'rendered': True,
                'artifact_sha256': {'rgb.png': hashlib.sha256(b'original image').hexdigest()}}
    (dest / 'rgb.png').write_bytes(b'original image')
    (dest / 'render.json').write_text(json.dumps(metadata))
    return out, dest, job, metadata


def _mock_worker(monkeypatch, out, *, fail=False, crash=False):
    calls = []
    monkeypatch.setattr('doorbench.appearance.pipeline.find_blender', lambda _: '/mock/blender')

    def run(command, **kwargs):
        config = json.loads((out / 'jobs.json').read_text())
        calls.append(config['jobs'])
        if crash:
            return SimpleNamespace(returncode=-11)
        results, failures = [], []
        for job in config['jobs']:
            if fail:
                failures.append({'door_id': job['door_id'], 'error': 'intentional worker failure'})
                continue
            dest = Path(job['out_dir'])
            image = b'new successfully rendered image'
            (dest / 'rgb.png').write_bytes(image)
            metadata = {'door_id': job['door_id'], 'job_sha256': job['job_sha256'], 'rendered': True,
                        'artifact_sha256': {'rgb.png': hashlib.sha256(image).hexdigest()}}
            (dest / 'render.json').write_text(json.dumps(metadata))
            results.append(metadata)
        Path(config['result_path']).write_text(json.dumps({'results': results, 'failures': failures}))
        return SimpleNamespace(returncode=1 if fail else 0)

    monkeypatch.setattr('doorbench.appearance.pipeline.subprocess.run', run)
    return calls


def test_valid_resume_verifies_artifact_and_skips_worker(tmp_path, monkeypatch):
    out, _, job, _ = _fixture(tmp_path)
    calls = _mock_worker(monkeypatch, out)
    index = run_jobs([job], out, resume=True)
    assert index['completed'] == index['rendered'] == 1
    assert calls == []


@pytest.mark.parametrize('damage', ['stale_job', 'truncated_metadata', 'missing_image', 'changed_image'])
def test_invalid_cache_is_a_miss_and_rerenders(tmp_path, monkeypatch, damage):
    out, dest, job, metadata = _fixture(tmp_path)
    if damage == 'stale_job':
        metadata['job_sha256'] = 'previous_source_or_recipe'
        (dest / 'render.json').write_text(json.dumps(metadata))
    elif damage == 'truncated_metadata':
        (dest / 'render.json').write_text('{"job_sha256":')
    elif damage == 'missing_image':
        (dest / 'rgb.png').unlink()
    else:
        (dest / 'rgb.png').write_bytes(b'truncated or different PNG')
    calls = _mock_worker(monkeypatch, out)
    index = run_jobs([job], out, resume=True)
    assert len(calls) == 1 and len(calls[0]) == 1
    assert index['completed'] == index['rendered'] == 1
    assert (dest / 'rgb.png').read_bytes() == b'new successfully rendered image'


@pytest.mark.parametrize('crash', [False, True])
def test_failure_replaces_previous_index_and_does_not_publish_stale_image(tmp_path, monkeypatch, crash):
    out, dest, job, _ = _fixture(tmp_path)
    old_image = (dest / 'rgb.png').read_bytes()
    (out / 'index.json').write_text(json.dumps({'completed': 1, 'renders': [{'image': 'old/rgb.png'}]}))
    job['job_sha256'] = 'new_recipe_that_fails'
    _mock_worker(monkeypatch, out, fail=True, crash=crash)
    with pytest.raises(RuntimeError, match='Incomplete appearance batch'):
        run_jobs([job], out, resume=True)
    index = json.loads((out / 'index.json').read_text())
    assert index['completed'] == index['rendered'] == 0
    assert index['renders'] == []
    assert index['failed']
    # The old file may remain for recovery, but must not be advertised as new output.
    assert (dest / 'rgb.png').read_bytes() == old_image


def test_stale_offline_snapshot_rejected_before_job_preparation(tmp_path):
    assets = tmp_path / 'assets'
    spec = next(s for s in generate_all() if s['id'] == 'db0079_sliding_single')
    export_door(spec, str(assets / 'doors'), str(assets / 'hardware'), formats=('mjcf', 'json'))
    source = assets / 'doors' / spec['id']
    snapshot = capture_initial_state(source)
    (source / 'model.json').write_text((source / 'model.json').read_text() + '\n')
    with pytest.raises(ValueError, match='Stale simulation snapshot'):
        prepare_job(assets, spec['id'], tmp_path / 'out', state=snapshot)


def test_incremental_variant_keeps_other_verified_renders_but_removes_failed_slot(tmp_path, monkeypatch):
    out, _, original, _ = _fixture(tmp_path)
    run_jobs([original], out, resume=True)
    variant = {**original, 'variant': 1, 'job_sha256': 'second-variant',
               'out_dir': str(out / 'door/variant_001')}
    Path(variant['out_dir']).mkdir()
    _mock_worker(monkeypatch, out)
    index = run_jobs([variant], out)
    assert index['completed'] == 2 and index['batch_completed'] == 1
    assert [r['variant'] for r in index['renders']] == [0, 1]
    variant['job_sha256'] = 'changed-variant-that-fails'
    _mock_worker(monkeypatch, out, fail=True)
    with pytest.raises(RuntimeError):
        run_jobs([variant], out)
    index = json.loads((out / 'index.json').read_text())
    assert index['completed'] == 1 and index['batch_completed'] == 0
    assert index['renders'][0]['variant'] == 0

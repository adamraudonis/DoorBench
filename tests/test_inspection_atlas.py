"""An atlas cannot silently mix stale assets or hide a forced/failed render pose."""
import hashlib
import json

import pytest
from PIL import Image

from scripts.inspection_atlas import main, pose_label, validate_render_record


@pytest.fixture
def provenance(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    record = {'door_id': 'test_door'}
    for kind, filename in [('model', 'model.json'), ('spec', 'spec.json'), ('xml', 'door.xml')]:
        content = f'original {kind}'.encode()
        (source / filename).write_bytes(content)
        record[f'source_{kind}_sha256'] = hashlib.sha256(content).hexdigest()
    return source, record


@pytest.mark.parametrize('filename', ['model.json', 'spec.json', 'door.xml'])
def test_stale_render_rejected_for_every_recorded_source(provenance, filename):
    source, record = provenance
    (source / filename).write_text('changed since this image was rendered')
    with pytest.raises(ValueError, match=r'stale render'):
        validate_render_record(source, record, 'test_door')


def test_legacy_model_only_provenance_is_explicit(provenance):
    source, record = provenance
    legacy = {k: v for k, v in record.items() if k in ('door_id', 'source_model_sha256')}
    assert validate_render_record(source, legacy, 'test_door') == {
        'source_model_sha256': legacy['source_model_sha256']}
    with pytest.raises(ValueError, match='no model provenance'):
        validate_render_record(source, {'door_id': 'test_door'}, 'test_door')


def test_forced_and_unsolved_flags_are_not_dropped():
    lines, info = pose_label('open', 'open / prescribed pose', {
        'forced_locked_pose': True, 'loop_residual_m': 0.013332})
    assert 'FORCED LOCKED POSE' in lines
    assert any('UNSOLVED LOOP' in line and '13.33 mm' in line for line in lines)
    assert info['forced_locked_pose'] and info['loop_unsolved']
    assert 'solved pose' not in info['label']
    _, boundary = pose_label('open', 'open / prescribed pose', {
        'forced_locked_pose': False, 'loop_residual_m': 0.001})
    assert not boundary['loop_unsolved']


def test_atlas_index_retains_render_flags_and_provenance(tmp_path, monkeypatch):
    assets, renders, out = (tmp_path / name for name in ('assets', 'renders', 'atlas'))
    source, dest = assets / 'doors/test_door', renders / 'test_door'
    source.mkdir(parents=True)
    dest.mkdir(parents=True)
    (source / 'model.json').write_text('{}')
    model_hash = hashlib.sha256((source / 'model.json').read_bytes()).hexdigest()
    views = ['front', 'reverse', 'hardware', 'open']
    record = {'door_id': 'test_door', 'source_model_sha256': model_hash, 'views': [
        {'view': view, 'forced_locked_pose': view == 'open',
         'loop_residual_m': 0.012 if view == 'open' else 0.0} for view in views]}
    (dest / 'render.json').write_text(json.dumps(record))
    for view in views:
        Image.new('RGB', (24, 18)).save(dest / f'{view}.jpg')
    (assets / 'manifest.json').write_text(json.dumps({'generated': 'test', 'doors': [
        {'id': 'test_door', 'family': 'test', 'index': 1, 'leaf': {'slab': 'test'},
         'operator': 'none', 'closer': 'none', 'latch': 'none', 'lock': 'none', 'mass_kg': 1}]}))
    monkeypatch.setattr('sys.argv', ['inspection_atlas', '--assets', str(assets),
                                   '--renders', str(renders), '--out', str(out)])
    main()
    entry = json.loads((out / 'index.json').read_text())['pages'][0]['doors'][0]
    image = next(v for v in entry['images'] if v['view'] == 'open')
    assert image['forced_locked_pose'] and image['loop_unsolved']
    assert 'FORCED LOCKED POSE' in image['label'] and 'UNSOLVED LOOP' in image['label']
    assert entry['verified_render_provenance']['source_model_sha256'] == model_hash

import hashlib
import json

import pytest

from doorbench.dexterous.controller_input_snapshot import snapshot_controller_inputs


def test_documents_survive_original_changes_and_cyclic_references(tmp_path):
    config=tmp_path/'control.json';plan=tmp_path/'plan.json'
    config.write_text(json.dumps(dict(plan_path=str(plan),source_run=str(tmp_path/'external-episode'))))
    plan.write_text(json.dumps(dict(audit_path=str(config),positions=[1,2,3])))
    original=plan.read_bytes()
    output=tmp_path/'snapshot'
    result=snapshot_controller_inputs([config,config],output)
    assert len(result['documents'])==2
    assert result['source_episodes'][str(tmp_path/'external-episode')]['available_locally'] is False
    record=result['documents'][str(plan)]
    plan.write_text('changed after capture')
    assert (output/record['snapshot']).read_bytes()==original
    assert record['sha256']==hashlib.sha256(original).hexdigest()
    assert json.loads((output/'manifest.json').read_text())==result
    with pytest.raises(FileExistsError):snapshot_controller_inputs([config],output)


def test_source_episode_trajectory_is_hashed_but_not_duplicated(tmp_path):
    episode=tmp_path/'source';episode.mkdir()
    (episode/'trajectory.npz').write_bytes(b'stand-in trajectory bytes')
    (episode/'report.json').write_text('{"passed": true}')
    control=tmp_path/'control.json';control.write_text(json.dumps(dict(configuration=dict(source_run=str(episode)))))
    output=tmp_path/'snapshot';result=snapshot_controller_inputs([control],output)
    assert result['source_episodes'][str(episode)]['trajectory_sha256']==hashlib.sha256(b'stand-in trajectory bytes').hexdigest()
    assert len(result['documents'])==2
    assert not list(output.glob('*.npz'))


def test_missing_explicit_document_is_not_silently_omitted(tmp_path):
    with pytest.raises(FileNotFoundError):
        snapshot_controller_inputs([tmp_path/'missing.json'],tmp_path/'snapshot')

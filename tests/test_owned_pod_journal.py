"""Journal selection must isolate resource actions and their guards."""
import importlib.util
from pathlib import Path


def load():
    path=Path(__file__).parents[1]/'scripts/dexterous/pod.py'
    spec=importlib.util.spec_from_file_location('isolated_owned_pod',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_explicit_journal_is_used_by_api_without_modifying_original(tmp_path,monkeypatch):
    original=tmp_path/'old.json';original.write_text('{"id":"stopped-old"}')
    replacement=tmp_path/'new.json'
    monkeypatch.setenv('DOORBENCH_POD_STATE',str(replacement))
    module=load()
    assert module.STATE==replacement and module.api().STATE==replacement
    assert original.read_text()=='{"id":"stopped-old"}' and not replacement.exists()


def test_default_journal_preserves_existing_workflow(monkeypatch):
    monkeypatch.delenv('DOORBENCH_POD_STATE',raising=False)
    module=load()
    assert module.STATE==Path.home()/'.runpod/doorbench_dexterous_pod.json'

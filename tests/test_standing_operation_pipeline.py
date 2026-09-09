"""A rejected prerequisite must leave evidence and never start a simulator."""
import importlib.util,json,time
from pathlib import Path


def test_wrong_mechanics_stops_before_commands(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('standing_pipeline',Path(__file__).parents[1]/'scripts/isaac/run_standing_operation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ready=tmp_path/'ready.json';ready.write_text(json.dumps({'ready':True,'mechanics_profile':'upstream-v1'}))
    out=tmp_path/'trial'
    monkeypatch.setattr(module.sys,'argv',['pipeline','--source',str(tmp_path),'--ready',str(ready),'--reference',str(tmp_path/'reference.json'),'--output',str(out),'--work',str(tmp_path),'--deadline-unix',str(time.time()+3600)])
    monkeypatch.setattr(module.subprocess,'Popen',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('Physics must not run')))
    assert module.main()==1
    result=json.loads((out/'coordinator-result.json').read_text())
    assert not result['passed'] and 'mechanics readiness' in result['error']
    assert not (out/'commands.json').exists()

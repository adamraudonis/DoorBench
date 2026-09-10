"""A rejected prerequisite must leave evidence and never start a simulator."""
import importlib.util,json,time
import pytest
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


@pytest.mark.parametrize('offset', [('nan','0','0'),('.011','0','0'),('.008','.008','0')])
def test_invalid_recenter_rejects_before_creating_run(tmp_path,monkeypatch,offset):
    spec=importlib.util.spec_from_file_location('standing_pipeline',Path(__file__).parents[1]/'scripts/isaac/run_standing_operation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=tmp_path/'trial'
    argv=['pipeline','--source',str(tmp_path),'--ready',str(tmp_path/'ready.json'),'--reference',str(tmp_path/'reference.json'),'--output',str(out),'--work',str(tmp_path),'--deadline-unix',str(time.time()+3600),'--operation-grasp-offset-in-handle-m',*offset]
    monkeypatch.setattr(module.sys,'argv',argv)
    with pytest.raises(ValueError,match='recentring|recentering'):module.main()
    assert not out.exists()


@pytest.mark.parametrize('flag,value', [('--operation-index-proximal-offset-rad','nan'),
    ('--operation-index-proximal-offset-rad','.101'),('--operation-index-tendon-offset-rad','-.121')])
def test_invalid_index_offset_rejects_before_run(tmp_path,monkeypatch,flag,value):
    spec=importlib.util.spec_from_file_location('standing_pipeline',Path(__file__).parents[1]/'scripts/isaac/run_standing_operation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=tmp_path/'trial'
    monkeypatch.setattr(module.sys,'argv',['pipeline','--source',str(tmp_path),'--ready',str(tmp_path/'ready.json'),'--reference',str(tmp_path/'reference.json'),'--output',str(out),'--work',str(tmp_path),'--deadline-unix',str(time.time()+3600),flag,value])
    with pytest.raises(ValueError,match='index posture'):module.main()
    assert not out.exists()


def test_index_offsets_reach_both_backends_and_isaac_teacher():
    import ast
    from types import SimpleNamespace
    root=Path(__file__).parents[1]
    tree=ast.parse((root/'scripts/isaac/run_standing_operation.py').read_text())
    values={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('native_index_options','isaac_index_options'):
            values[node.targets[0].id]=eval(compile(ast.Expression(node.value),'<options>','eval'),{'a':SimpleNamespace(operation_index_proximal_offset_rad=-.02,operation_index_tendon_offset_rad=.06)})
    assert values['native_index_options']==['--index-proximal-offset-rad','-0.02','--index-tendon-offset-rad','0.06']
    assert values['isaac_index_options']==['--operation-index-proximal-offset-rad','-0.02','--operation-index-tendon-offset-rad','0.06']
    consumed={n.value.id for n in ast.walk(tree) if isinstance(n,ast.Starred) and isinstance(n.value,ast.Name)}
    assert set(values)<=consumed
    tree=ast.parse((root/'scripts/dexterous/isaac_opening.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='DoorOperationTeacher']
    assert len(calls)==1
    kwargs={k.arg:ast.unparse(k.value) for k in calls[0].keywords}
    assert kwargs['index_proximal_offset_rad']=='a.operation_index_proximal_offset_rad'
    assert kwargs['index_tendon_offset_rad']=='a.operation_index_tendon_offset_rad'

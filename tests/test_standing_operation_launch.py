import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


def module():
    spec=importlib.util.spec_from_file_location('standing_launch',Path(__file__).parents[1]/'scripts/isaac/launch.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_dispatch_is_bounded_detached_and_collects_task_evidence(tmp_path,monkeypatch):
    m=module();calls=[];monkeypatch.setattr(m.time,'time',lambda:100.)
    monkeypatch.setattr(m,'run',lambda argv,**kw:calls.append((argv,kw)))
    monkeypatch.setattr(m.subprocess,'run',lambda argv,**kw:(calls.append((argv,kw)) or SimpleNamespace(stdout='{"pid":123}')))
    receipt=dict(mechanics_profile='shadow-loopback-v2',remote='/workspace/a space',ready_receipt='/workspace/a space/ready.json',host='root@example',port=22,key='/tmp/key')
    r=m.dispatch_standing_operation(receipt,tmp_path,tmp_path/'runs.json','demo','/workspace',4000.)
    script=calls[0][1]['input']
    assert 'nohup ' in script and '2>&1 < /dev/null &' in script
    assert 'if test ! -e ' in script and 'door55-standing-acquisition-v1.json' in script
    assert "'/workspace/a space/scripts/isaac/run_standing_operation.py'" in script
    assert '--deadline-unix 3940.0' in script
    collector=calls[1][0]
    assert 'coordinator-result.json' in collector and '--detach' in collector
    assert r['deadline_unix']==3940. and r['evidence_collector']=={'pid':123}


def test_insufficient_budget_does_not_dispatch(tmp_path,monkeypatch):
    m=module();monkeypatch.setattr(m.time,'time',lambda:100.)
    monkeypatch.setattr(m,'run',lambda *a,**k:pytest.fail('Must not launch'))
    with pytest.raises(ValueError,match='40 minutes'):
        m.dispatch_standing_operation({'mechanics_profile':'shadow-loopback-v2'},tmp_path,tmp_path/'registry','test','/workspace',500.)

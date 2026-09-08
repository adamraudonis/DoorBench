"""Refuse wrong allocation, unbounded deadlines and changed process identities."""
import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts/dexterous'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('renew_owned_guard',SCRIPTS/'renew_owned_guard.py')
renew=importlib.util.module_from_spec(spec);spec.loader.exec_module(renew)


def state():return dict(id='owned',active=True,created=1000.,deadline=10000.)


def test_three_hour_window_stays_within_total_allocation_bound():
    assert renew.validate_owned(state(),'owned',9000.,3.)==19800.


@pytest.mark.parametrize('change,pod,now,hours',[
    ({},'other',9000.,3.),({'active':False},'owned',9000.,3.),
    ({},'owned',9900.,3.),({},'owned',9000.,3.1),
    ({'created':-20000.},'owned',9000.,3.),({},'owned',9000.,.1),
    ({'deadline':float('nan')},'owned',9000.,3.),
])
def test_unsafe_renewals_are_refused(change,pod,now,hours):
    value=state();value.update(change)
    with pytest.raises(ValueError):renew.validate_owned(value,pod,now,hours)


@pytest.mark.parametrize('field,value',[
    ('pid',68192),('argv',['python3','isaac_opening.py']),('start','reused PID start'),('script_sha256','changed script')])
def test_every_process_identity_field_must_match_before_signal(field,value):
    old=dict(pid=165,argv=['python3','/workspace/dex-remote-guard.py'],start='474031370',script_sha256='verified')
    assert renew.same_process(dict(old),old)
    changed=dict(old);changed[field]=value
    assert not renew.same_process(changed,old)
    assert not renew.same_process(None,old)


def test_guard_scripts_compile_without_running_or_loading_credentials():
    compile(renew.GUARD,'guard.py','exec');compile(renew.REMOTE_OPS,'remote_ops.py','exec')


@pytest.mark.parametrize('journal_id',['owned','new-allocation'])
def test_deadline_guard_deletes_only_captured_id_and_preserves_new_journal(tmp_path,monkeypatch,journal_id):
    import json,urllib.request
    journal=tmp_path/'journal.json';journal.write_text(json.dumps(dict(id=journal_id,active=True,deadline=0.,costPerHr=1.25)))
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(dict(id='owned',deadline=0.,key='dummy-test-value',journal=str(journal))))
    urls=[]
    class Response:
        def close(self):pass
    def request(value,timeout):urls.append((value.full_url,value.method));return Response()
    monkeypatch.setattr(urllib.request,'urlopen',request);monkeypatch.setattr(sys,'argv',['guard.py',str(cfg)])
    exec(renew.GUARD,{})
    assert urls==[('https://rest.runpod.io/v1/pods/owned','DELETE')]
    updated=json.loads(journal.read_text())
    assert updated['id']==journal_id and updated['costPerHr']==1.25
    assert updated['active']==(journal_id!='owned')

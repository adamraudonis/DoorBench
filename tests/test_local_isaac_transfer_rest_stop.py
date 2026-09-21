import copy
from types import SimpleNamespace

import pytest

from test_local_operation_launcher import module,transfer_fixture,write


def test_opt_in_stop_preserves_the_controller_recipe(tmp_path,monkeypatch):
    launcher,args,argv,hashes,_=transfer_fixture(tmp_path,monkeypatch)
    base,_=launcher.prepare_transfer(args,argv,hashes)
    args.standing_transfer_stop_on_rest=True
    with pytest.raises(ValueError,match='live prefix witness'):
        launcher.prepare_transfer(args,argv,hashes)
    # This fixture intentionally has no historical code archive. Keep its existing
    # admission fake narrow; live witness behavior is covered by witness tests.
    from doorbench.dexterous import isaac_prefix_witness
    monkeypatch.setattr(isaac_prefix_witness,'historical_source_hashes',lambda source:{})
    args.standing_transfer_live_prefix_witness=True
    actual,_=launcher.prepare_transfer(args,argv,hashes)
    assert actual==base+['--standing-transfer-prefix-source',str(args.standing_transfer_source),
        '--standing-transfer-stop-on-rest']


def receipt():
    return dict(schema='doorbench.isaac-transfer-rest-stop-run.v1',maximum_seconds=44.,mode='terminate',
        terminated_on_qualified_rest=True,continued_to_withdrawal=False,
        detector=dict(triggered=True,terminal_time_s=42.548))


def test_declared_stop_accepts_only_the_recorded_actual_terminal(tmp_path,monkeypatch):
    launcher=module(monkeypatch,tmp_path)
    args=SimpleNamespace(seconds=44.,standing_transfer_stop_on_rest=True)
    record=receipt();write(tmp_path/'standing-transfer-rest-stop.json',record)
    report=dict(duration_s=42.548,standing_transfer_rest_stop=record)
    assert launcher.runtime_duration_passed(report,args,tmp_path)
    args.standing_transfer_stop_on_rest=False
    assert not launcher.runtime_duration_passed(report,args,tmp_path)
    assert launcher.runtime_duration_passed(dict(duration_s=44.),args,tmp_path)
    args.standing_transfer_stop_on_rest=True;args.standing_withdrawal_route='route'
    assert not launcher.runtime_duration_passed(report,args,tmp_path)


@pytest.mark.parametrize('key,value',[
    ('maximum_seconds',45.),('mode','prefix-only'),('terminated_on_qualified_rest',False),
    ('continued_to_withdrawal',True),('schema','invented'),
])
def test_misdeclared_stop_cannot_bypass_duration(tmp_path,monkeypatch,key,value):
    launcher=module(monkeypatch,tmp_path)
    record=receipt();record[key]=value
    write(tmp_path/'standing-transfer-rest-stop.json',record)
    report=dict(duration_s=42.548,standing_transfer_rest_stop=record)
    assert not launcher.runtime_duration_passed(report,
        SimpleNamespace(seconds=44.,standing_transfer_stop_on_rest=True),tmp_path)


def test_report_cannot_replace_the_recorded_stop(tmp_path,monkeypatch):
    launcher=module(monkeypatch,tmp_path)
    record=receipt();write(tmp_path/'standing-transfer-rest-stop.json',record)
    altered=copy.deepcopy(record);altered['detector']['terminal_time_s']=42.550
    assert not launcher.runtime_duration_passed(dict(duration_s=42.550,standing_transfer_rest_stop=altered),
        SimpleNamespace(seconds=44.,standing_transfer_stop_on_rest=True),tmp_path)

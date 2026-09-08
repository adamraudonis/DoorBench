import copy
import gzip
import json
import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('balance_audit',Path(__file__).parents[1]/'scripts/dexterous/audit_isaac_sensor_balance.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    layout=dict(sensor_paths=['/robot/left_ankle_link','/robot/right_ankle_link','/robot/rh_palm'],
        filter_paths=[['/World/floor','/World/wall']]*3,capacity=20)
    def c(i,j,slot,load=100.,gap=-.001):return dict(sensor=i,filter=j,slot=slot,position=[0.,0.,0.],normal=[0.,0.,1.],force_N=load,distance_m=gap)
    return layout,dict(contacts=[c(0,0,0),c(1,0,1),c(0,1,2,1000.),c(2,1,3,0.,.001)])


def test_only_actual_floor_contact_supports_the_robot():
    layout,row=fixture();feet,hands=m.recompute_contacts(layout,row)
    np.testing.assert_array_equal(feet,[100.,100.]);assert hands==0
    row['contacts'][-1]['force_N']=.1
    assert m.recompute_contacts(layout,row)[1]==1


@pytest.mark.parametrize('change',['duplicate','invalid_normal','negative_force','capacity','missing_foot'])
def test_bad_raw_contacts_cannot_produce_support_evidence(change):
    layout,row=fixture()
    if change=='duplicate':row['contacts'].append(copy.deepcopy(row['contacts'][0]))
    if change=='invalid_normal':row['contacts'][0]['normal']=[0.,0.,2.]
    if change=='negative_force':row['contacts'][0]['force_N']=-1.
    if change=='capacity':row['contacts'][0]['slot']=20
    if change=='missing_foot':layout['sensor_paths'][0]='/robot/elbow'
    with pytest.raises(ValueError):m.recompute_contacts(layout,row)


def test_exception_prefix_cannot_become_a_qualified_report(tmp_path,monkeypatch):
    run=tmp_path/'failed';run.mkdir();(run/'sensors').mkdir()
    with gzip.open(run/'balance-steps.json.gz','wt') as f:json.dump([],f)
    with gzip.open(run/'balance-contacts.jsonl.gz','wt') as f:f.write('')
    layout,_=fixture();(run/'balance-contact-layout.json').write_text(json.dumps(layout))
    (run/'balance-report.json').write_text(json.dumps(dict(passed=False,error='controller rejected packet',physical_evidence_complete=False)))
    for name in ('sensor-balance-calibration.json','configuration.json','provenance.json','motor-contract.json','sensors/layout.json'):
        (run/name).write_text('{}')
    output=tmp_path/'independent.json'
    monkeypatch.setattr('sys.argv',['audit','--run',str(run),'--output',str(output)])
    m.main();r=json.loads(output.read_text())
    assert not r['verification_passed'] and not r['actual_stationary_trial_passed']
    assert r['errors']==['No completed supported balance qualification report']

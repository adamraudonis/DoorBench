import copy
import hashlib
import json
import numpy as np
import pytest
from doorbench.dexterous.passive_evidence import audit_joint_passive_evidence, PROPERTY_ORDER


def archive(tmp_path):
    names = ['joint'+str(i) for i in range(69)]
    contract = dict(joint_names=names, passive={n:dict(friction=.01, damping=.05+i/100,
        armature=.0002, stiffness=0., springref=0.) for i,n in enumerate(names)})
    # Backend order deliberately differs from the original contract order.
    names = names[::-1]
    expected = np.array([[.01,.01,contract['passive'][n]['damping']] for n in names],np.float32).astype(float).tolist()
    armature = np.full(69,.0002,np.float32).astype(float).tolist()
    start = dict(profile='backend-dry-v2',joint_names=names,property_order=PROPERTY_ORDER,
        backend_friction_properties=expected,native_armature_readback_kg_m2=armature,
        explicit_damping=[0.]*69,explicit_friction=[0.]*69)
    receipt = dict(schema='doorbench.passive-property-invariant.v1',profile='backend-dry-v2',passed=True,
        joint_names=names,property_order=PROPERTY_ORDER,expected_float32_properties=expected,
        expected_float32_armature_kg_m2=armature,maximum_absolute_property_difference=[0.,0.,0.],
        maximum_armature_difference_kg_m2=0.,physics_dt_s=.002,attempted_intervals=3,
        checked_intervals=3,valid_intervals=3,last_checked_interval_end_s=.006,first_failure=None)
    config=dict(args=dict(joint_passive_profile='backend-dry-v2',motors='/original/motors.json'),robot_joint_names=names,dt=.002)
    values={'motor-contract.json':contract,'joint-passive-profile.json':start,
        'joint-passive-invariants.json':receipt,'configuration.json':config}
    for name,value in values.items():(tmp_path/name).write_text(json.dumps(value))
    source=tmp_path/'source-isaac_joint_passive.py';source.write_text('# frozen source fixture\n')
    provenance=dict(files={'/original/motors.json':hashlib.sha256((tmp_path/'motor-contract.json').read_bytes()).hexdigest(),
        '/original/doorbench/dexterous/isaac_joint_passive.py':hashlib.sha256(source.read_bytes()).hexdigest()})
    (tmp_path/'provenance.json').write_text(json.dumps(provenance))
    return [{'time_s':.002},{'time_s':.004},{'time_s':.006}],dict(joint_passive_profile='backend-dry-v2',expected_duration_s=.006)


def change(path,func):
    d=json.loads(path.read_text());func(d);path.write_text(json.dumps(d))


def test_reconstructs_named_float32_terms_and_full_intervals(tmp_path):
    rows,report=archive(tmp_path);r=audit_joint_passive_evidence(tmp_path,rows,report)
    assert r['verification_passed'] and r['persistence_complete']
    assert len(r['source_sha256'])==6


@pytest.mark.parametrize('mutation', ['coeff','armature','double','joint_order','property_order','expected',
    'count','clock','last','failure','nonfinite','float32_rounding','source','contract','missing','label','boolean_count'])
def test_profile_label_cannot_replace_original_coefficients_or_complete_persistence(tmp_path,mutation):
    rows,report=archive(tmp_path);start=tmp_path/'joint-passive-profile.json';guard=tmp_path/'joint-passive-invariants.json'
    if mutation=='coeff':change(start,lambda d:d['backend_friction_properties'][0].__setitem__(2,.04))
    if mutation=='armature':change(start,lambda d:d['native_armature_readback_kg_m2'].__setitem__(0,0.))
    if mutation=='double':change(start,lambda d:d['explicit_damping'].__setitem__(0,.05))
    if mutation=='joint_order':change(start,lambda d:d['joint_names'].reverse())
    if mutation=='property_order':change(guard,lambda d:d['property_order'].reverse())
    if mutation=='expected':change(guard,lambda d:d['expected_float32_properties'][0].__setitem__(0,0.))
    if mutation=='count':change(guard,lambda d:d.__setitem__('valid_intervals',2))
    if mutation=='clock':rows[1]['time_s']=.005
    if mutation=='last':change(guard,lambda d:d.__setitem__('last_checked_interval_end_s',.004))
    if mutation=='failure':change(guard,lambda d:d.__setitem__('first_failure',{'interval_index':3}))
    if mutation=='nonfinite':change(guard,lambda d:d.__setitem__('maximum_armature_difference_kg_m2',float('nan')))
    if mutation=='float32_rounding':change(start,lambda d:d['backend_friction_properties'][0].__setitem__(0,.01))
    if mutation=='source':(tmp_path/'source-isaac_joint_passive.py').write_text('changed')
    if mutation=='contract':change(tmp_path/'motor-contract.json',lambda d:d['passive']['joint0'].__setitem__('damping',.04))
    if mutation=='missing':guard.unlink()
    if mutation=='label':report['joint_passive_profile']='legacy-tanh-v1'
    if mutation=='boolean_count':change(guard,lambda d:d.__setitem__('valid_intervals',True))
    r=audit_joint_passive_evidence(tmp_path,rows,report)
    assert not r['verification_passed'] and r['errors']


def test_original_unversioned_archive_is_supported_without_new_persistence_claim(tmp_path):
    (tmp_path/'configuration.json').write_text('{}')
    r=audit_joint_passive_evidence(tmp_path,[],{})
    assert r['verification_passed'] and r['profile']=='legacy-unversioned'
    assert r['persistence_complete'] is None


def test_failed_prefix_remains_failed_even_with_a_successful_prefix_guard(tmp_path):
    rows,report=archive(tmp_path)
    change(tmp_path/'joint-passive-invariants.json',lambda d:d.update(attempted_intervals=2,checked_intervals=2,
        valid_intervals=2,last_checked_interval_end_s=.004))
    r=audit_joint_passive_evidence(tmp_path,rows[:2],report)
    assert not r['verification_passed'] and r['persistence_complete'] is False

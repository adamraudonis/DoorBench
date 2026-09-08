import gzip
import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from doorbench.dexterous import legacy_teacher_provenance as provenance


def fixture(tmp_path,monkeypatch):
    def put(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    for name in provenance.FILES:put(name,{})
    contents={name:('fixture '+name).encode() for name in provenance.SOURCE_HASHES}
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    with tarfile.open(tmp_path/'launch-source/source.tar.gz','w:gz') as tar:
        for name,data in contents.items():
            member=tarfile.TarInfo(name);member.size=len(data);tar.addfile(member,io.BytesIO(data))
            (tmp_path/('source-'+name.rsplit('/',1)[-1])).write_bytes(data)
    package=provenance.sha(tmp_path/'launch-source/source.tar.gz')
    monkeypatch.setattr(provenance,'APPROVED_SOURCE_ARCHIVE',package)
    monkeypatch.setattr(provenance,'SOURCE_HASHES',hashes)
    put('launch-source/manifest.json',dict(source_archive_sha256=package,captured_at_unix=1,source_hashes=hashes))
    put('provenance.json',dict(captured_before_steps_unix=2,files={'/workspace/'+name:sha for name,sha in hashes.items()}))
    put('acquisition-report.json',dict(passed=True,checks={'complete_physics_steps':True,'sustained_pad_grasp':True},runtime_robot_pose_writes=0,direct_door_commands=False,physics_dt_s=.002,duration_s=1.))
    put('operation-report.json',dict(passed=False,operation_reference=dict(operation_start_s=.8)))
    put('mechanical-audit.json',dict(passed=True))
    put('configuration.json',dict(runtime_pose_writes=0,direct_door_commands=False,args=dict(acquisition=True,operate_after_acquisition=True)))
    put('trace.json',[dict(joint_torque_command=[1.,2.],joint_torque_sent=[1.,2.])])
    times=np.arange(1,501)*.002
    for name in ('acquisition-physics.npz','sensors/actor-sensors.npz'):
        np.savez_compressed(tmp_path/name,time_s=times)
    rows=[dict(sim_time_s=float(t),valid_pad_grasp=True) for t in np.arange(501)*.002]
    with gzip.open(tmp_path/'acquisition-pad-steps.json.gz','wt') as f:json.dump(rows,f)
    return put


def test_receipt_requires_original_source_and_continuous_preoperation_grip(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);receipt=provenance.audit_legacy_acquisition_prefix(tmp_path)
    assert receipt['sample_end_time_s']==.8 and receipt['supervised_examples']==399
    assert receipt['excluded_first_operation_action_time_s']==.802
    assert receipt['qualified_hold_physics_samples']==251
    path=tmp_path/'separate-receipt.json';path.write_text(json.dumps(receipt))
    assert provenance.validate_legacy_teacher_receipt(tmp_path,path,qualification='acquisition-report.json')==receipt
    with pytest.raises(ValueError,match='never failed operation'):
        provenance.validate_legacy_teacher_receipt(tmp_path,path,qualification='operation-report.json')
    receipt['sample_end_time_s']=1.;path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='differs'):
        provenance.validate_legacy_teacher_receipt(tmp_path,path,qualification='acquisition-report.json')


@pytest.mark.parametrize('defect',['source','actor','force','clock','grip','assistance','failed_acquisition'])
def test_legacy_receipt_rejects_incomplete_or_untrusted_evidence(tmp_path,monkeypatch,defect):
    put=fixture(tmp_path,monkeypatch)
    if defect=='source':(tmp_path/'launch-source/source.tar.gz').write_bytes(b'unknown code')
    elif defect=='actor':put('sensors/report.json',dict(control_source='sensor_actor'))
    elif defect=='force':put('trace.json',[dict(joint_torque_command=[1.],joint_torque_sent=[2.])])
    elif defect=='clock':np.savez_compressed(tmp_path/'sensors/actor-sensors.npz',time_s=np.arange(1,500)*.002)
    elif defect=='grip':
        with gzip.open(tmp_path/'acquisition-pad-steps.json.gz','rt') as f:rows=json.load(f)
        rows[399]['valid_pad_grasp']=False
        with gzip.open(tmp_path/'acquisition-pad-steps.json.gz','wt') as f:json.dump(rows,f)
    elif defect=='assistance':put('configuration.json',dict(runtime_pose_writes=1,direct_door_commands=False,args=dict(acquisition=True,operate_after_acquisition=True)))
    else:
        report=json.loads((tmp_path/'acquisition-report.json').read_text());report['passed']=False;put('acquisition-report.json',report)
    with pytest.raises(ValueError):provenance.audit_legacy_acquisition_prefix(tmp_path)

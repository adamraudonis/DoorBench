import gzip
import json

import numpy as np
import pytest

from doorbench.dexterous.release_source_admission import admit_release_source, sha


def write(path, value):
    path.write_text(json.dumps(value));return path


@pytest.fixture
def source(tmp_path):
    trial=tmp_path/'trial';trial.mkdir();raw=trial/'raw-transitions';raw.mkdir()
    robot=tmp_path/'robot.xml';robot.write_text('original robot')
    door=tmp_path/'door';door.mkdir();(door/'door.xml').write_text('original door')
    write(trial/'manifest.json',dict(configuration=dict(robot=str(robot),door=str(door)),
        working_directory=str(tmp_path),inputs=dict(robot=dict(sha256=sha(robot)),door={'door.xml':sha(door/'door.xml')})))
    np.savez(trial/'trajectory.npz',terminal_time_s=1.,terminal_qpos=[1.,2.])
    np.savez(raw/'last.npz',interval_end_s=[1.],qpos_after=[[1.,2.]])
    write(raw/'manifest.json',dict(schema='doorbench.native-transitions.v1',complete=True,
        chunks=[dict(file='last.npz',sha256=sha(raw/'last.npz'))]))
    write(trial/'report.json',dict(passed=True,checks={'final_left_palm_support':True}))
    rows=[dict(sim_time_s=.5+i*.002,pad_grasp=dict(valid_pad_grasp=True,contacts=[]),
        left_surface=dict(palm_normal_load_N=3.),handle_angle_rad=0.,bolt_slide_m=0.,door_q=.08) for i in range(251)]
    with gzip.open(trial/'physics-steps.json.gz','wt') as stream:json.dump(rows,stream)
    rebind(trial)
    return trial


def rebind(source):
    names=['manifest.json','report.json','raw-transitions/manifest.json','physics-steps.json.gz']
    for report in ['independent-contact-audit.json','independent-whole-handle-audit.json','independent-transfer-audit.json']:
        write(source/report,dict(passed=True,input_sha256={n:sha(source/n) for n in names}))


def admit(source, **kwargs):
    return admit_release_source(source,contact_audit_name='independent-contact-audit.json',measured_rest=True,**kwargs)


def test_admission_binds_actual_endpoint_without_return_report(source):
    result=admit(source)
    assert result['measured_rest']['window_samples']==251
    assert result['measured_rest']['minimum_palm_load_N']==3.
    assert not (source/'standing-return.json').exists()
    assert str(source/'raw-transitions/last.npz') in result['input_sha256']


def test_altered_actual_chunk_cannot_reuse_passing_audit(source):
    np.savez(source/'raw-transitions/last.npz',interval_end_s=[1.],qpos_after=[[8.,9.]])
    with pytest.raises(ValueError,match='terminal transition chunk changed'):admit(source)


def test_trajectory_cannot_substitute_another_pose(source):
    np.savez(source/'trajectory.npz',terminal_time_s=1.,terminal_qpos=[8.,9.])
    with pytest.raises(ValueError,match='actual terminal transition'):admit(source)


def test_finger_only_or_broken_recording_fails_even_if_outer_report_says_pass(source):
    with gzip.open(source/'physics-steps.json.gz','rt') as stream:rows=json.load(stream)
    rows[80]['left_surface']=dict(palm_normal_load_N=0.,total_normal_load_N=6.)
    with gzip.open(source/'physics-steps.json.gz','wt') as stream:json.dump(rows,stream)
    rebind(source)
    with pytest.raises(ValueError,match='sustained opposed grip, palm support'):admit(source)


def test_profile_selection_must_equal_prospective_source(source):
    with pytest.raises(ValueError,match='prospectively declared source profile'):admit(source,profile='volar-phalange-v1')


def test_source_audit_cannot_reference_changed_or_unbound_evidence(source):
    report=source/'independent-contact-audit.json';data=json.loads(report.read_text())
    data['input_sha256'].pop('report.json');write(report,data)
    with pytest.raises(ValueError,match='actual episode archive'):admit(source)

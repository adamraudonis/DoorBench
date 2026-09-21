import copy
import gzip
import json

import numpy as np
import pytest

from doorbench.dexterous.isaac_transfer_rest_audit import audit_transfer_rest_stop,validate_transfer_rest_evidence
from doorbench.dexterous.isaac_transfer_rest_stop import TransferRestStop
from test_isaac_transfer_rest_stop import START,sample


def write(path,value):path.write_text(json.dumps(value))


@pytest.fixture
def trial(tmp_path):
    detector=TransferRestStop(START)
    pads=[dict(sim_time_s=0.)];surfaces=[];times=[];doors=[]
    for tick in range(1,4252):
        t=tick*.002;pad,surface,angles=sample(t)
        pads.append(pad);surfaces.append(surface);times.append(t);doors.append([.09,0.,0.])
        detector.observe(t,pad,surface,angles)
    assert detector.receipt()['triggered']
    record=dict(schema='doorbench.isaac-transfer-rest-stop-run.v1',maximum_seconds=10.,mode='terminate',
        terminated_on_qualified_rest=True,continued_to_withdrawal=False,detector=detector.receipt())
    write(tmp_path/'standing-transfer-rest-stop.json',record)
    report=dict(duration_s=times[-1],physics_dt_s=.002,standing_transfer=dict(started_s=START),
        standing_transfer_rest_stop=record)
    write(tmp_path/'operation-report.json',report)
    write(tmp_path/'configuration.json',dict(args=dict(standing_transfer_stop_on_rest=True,
        standing_transfer_start_seconds=START,seconds=10.),
        door_joint_names=['leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide']))
    with gzip.open(tmp_path/'acquisition-pad-steps.json.gz','wt') as stream:json.dump(pads,stream)
    with gzip.open(tmp_path/'standing-transfer-steps.json.gz','wt') as stream:json.dump(surfaces,stream)
    np.savez(tmp_path/'acquisition-physics.npz',time_s=np.array(times),door=np.array(doors))
    return tmp_path,report


def test_first_hold_reconstructed_from_complete_measured_stream(trial):
    path,report=trial
    audit=audit_transfer_rest_stop(path)
    assert audit['passed'] and audit['producer_matches'] and audit['intervals_reconstructed']==4251
    validate_transfer_rest_evidence(report,dict(rest_stop=audit),path)
    corrupted=copy.deepcopy(audit);corrupted['checks'].pop('actual_stage_outcome')
    with pytest.raises(ValueError,match='Independently reconstructed'):
        validate_transfer_rest_evidence(report,dict(rest_stop=corrupted),path)
    for mode in ('missing_detector','changed_terminal','zero_intervals'):
        corrupted=copy.deepcopy(audit)
        if mode=='missing_detector':corrupted.pop('detector')
        elif mode=='changed_terminal':corrupted['detector']['terminal_time_s']+=.002
        else:corrupted['intervals_reconstructed']=0
        with pytest.raises(ValueError,match='Independently reconstructed'):
            validate_transfer_rest_evidence(report,dict(rest_stop=corrupted),path)
    with gzip.open(path/'standing-transfer-steps.json.gz','at') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='differs from the actual source'):
        validate_transfer_rest_evidence(report,dict(rest_stop=audit),path)


def test_selected_later_endpoint_is_not_first_stop(trial):
    path,report=trial
    report['duration_s']+=.002
    report['standing_transfer_rest_stop']['detector']['terminal_time_s']+=.002
    write(path/'operation-report.json',report);write(path/'standing-transfer-rest-stop.json',report['standing_transfer_rest_stop'])
    with np.load(path/'acquisition-physics.npz') as values:
        times=np.r_[values['time_s'],report['duration_s']];doors=np.vstack([values['door'],[.09,0,0]])
    np.savez(path/'acquisition-physics.npz',time_s=times,door=doors)
    # Even before inspecting any later record, the first actual hold disagrees.
    audit=audit_transfer_rest_stop(path)
    assert not audit['passed'] and not audit['producer_matches']


def test_missing_sample_cannot_be_bridged(trial):
    path,_=trial
    with gzip.open(path/'standing-transfer-steps.json.gz','rt') as stream:rows=json.load(stream)
    rows.pop(4005)
    with gzip.open(path/'standing-transfer-steps.json.gz','wt') as stream:json.dump(rows,stream)
    with pytest.raises(ValueError):audit_transfer_rest_stop(path)


def test_changed_physical_rest_poses_invalidate_receipt(trial):
    path,_=trial
    with np.load(path/'acquisition-physics.npz') as values:
        times=values['time_s'].copy();doors=values['door'].copy()
    doors[-10,0]=.11
    np.savez(path/'acquisition-physics.npz',time_s=times,door=doors)
    audit=audit_transfer_rest_stop(path)
    assert not audit['passed'] and not audit['checks']['first_joint_rest_triggered']


def test_replay_cannot_authorize_a_different_stage_start(trial):
    path,report=trial
    report['standing_transfer']['started_s']=START+.002
    write(path/'operation-report.json',report)
    audit=audit_transfer_rest_stop(path)
    assert not audit['passed'] and not audit['checks']['declared_transfer_entry']

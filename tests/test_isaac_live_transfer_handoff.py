"""Synthetic transfer observations; no model or physical source qualification."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous.isaac_live_transfer_handoff import LiveTransferHandoffObserver


ANGLES = dict(leaf=.09, operator=.001, latch=.0001)


def fixture():
    calls = []
    acquisition = SimpleNamespace(caps=np.tile([-2., 2.], (61, 1)))
    returned = np.linspace(-1., 1., 61, dtype=np.float64); info = {'synthetic': True}
    def force(t, *args, **kwargs):calls.append((t, args, kwargs));return returned, info
    transfer = SimpleNamespace(acquisition=acquisition, operation=object(),
        started=0., force=force)
    return LiveTransferHandoffObserver(transfer), calls, returned, info


def step(observer, index, *, good=True):
    t = index*.002; command, info = observer.force(t, 'root', flag='unchanged')
    observer.observe_completed_interval(command_time_s=t, post_step_time_s=(index+1)*.002,
        returned_command=command, grasp_qualified=good, left_palm_load=3.,
        angles=ANGLES, submission_valid=True)
    return command, info


def test_real_return_objects_bytes_and_single_delegation_are_unchanged():
    observer, calls, returned, info = fixture()
    prior = returned.tobytes(); force, actual_info = step(observer, 0)
    assert force is returned and actual_info is info and returned.tobytes() == prior
    assert calls == [(0., ('root',), {'flag': 'unchanged'})]
    assert observer.motor_capture.previous_time == 0.
    assert observer.accepted_epoch == .002 and observer.pending is None
    receipt = observer.receipt(); returned[:] = 0.
    assert receipt['terminal_command']['value'] != returned.tolist()


def test_late_entry_reuses_genuine_251_observations_without_counting_T_twice():
    observer, calls, returned, info = fixture()
    for index in range(251):step(observer, index)
    t = 251*.002; prior = copy.deepcopy(observer.receipt())
    assert observer.require_ready(t, transfer=observer.transfer)['rest']['measured_rest_ready']
    assert observer.observe_rest(t, grasp_qualified=True, left_palm_load=3., angles=ANGLES)
    assert observer.receipt() == prior and len(calls) == 251
    np.testing.assert_array_equal(observer.motor_capture.capture(t), returned)
    assert prior['terminal_command']['command_time_s'] == 250*.002
    assert prior['authorized_stages'] == 0


@pytest.mark.parametrize('bad', ['submission', 'time', 'post', 'bytes', 'dtype', 'grasp_type'])
def test_corrupt_or_unaccepted_interval_cannot_populate_handoff(bad):
    observer, calls, returned, info = fixture(); command, _ = observer.force(0.)
    args = dict(command_time_s=0., post_step_time_s=.002, returned_command=command.copy(),
        grasp_qualified=True, left_palm_load=3., angles=ANGLES, submission_valid=True)
    if bad == 'submission':args['submission_valid'] = False
    elif bad == 'time':args['command_time_s'] = .002
    elif bad == 'post':args['post_step_time_s'] = .004
    elif bad == 'bytes':args['returned_command'][0] += .001
    elif bad == 'dtype':args['returned_command'] = command.astype(np.float32)
    elif bad == 'grasp_type':args['grasp_qualified'] = 1
    with pytest.raises(ValueError):observer.observe_completed_interval(**args)
    assert observer.accepted_intervals == 0 and observer.motor_capture.previous_time is None
    assert observer.pending is not None and observer.failure is not None
    with pytest.raises(ValueError):observer.force(.002)
    assert len(calls) == 1


def test_missing_completion_cannot_call_predecessor_again():
    observer, calls, _, _ = fixture();observer.force(0.)
    with pytest.raises(ValueError, match='unaccepted'):observer.force(.002)
    assert len(calls) == 1


def test_failed_grasp_resets_rest_without_rejecting_genuine_completed_command():
    observer, calls, _, _ = fixture()
    for index in range(251):step(observer, index)
    step(observer, 251, good=False)
    assert observer.accepted_intervals == 252 and observer.failure is None
    assert not observer.rest.ready and observer.rest.since is None
    assert observer.motor_capture.previous_time == 251*.002


@pytest.mark.parametrize('bad', ['new_object', 'shifted_epoch', 'changed_rest'])
def test_wrong_late_entry_is_rejected(bad):
    observer, calls, _, _ = fixture()
    for index in range(251):step(observer, index)
    with pytest.raises(ValueError):
        if bad == 'new_object':observer.require_ready(.502, transfer=copy.copy(observer.transfer))
        elif bad == 'shifted_epoch':observer.require_ready(.504)
        else:observer.observe_rest(.502, grasp_qualified=True, left_palm_load=3.1, angles=ANGLES)
    assert len(calls) == 251

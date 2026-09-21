"""Observe a live transfer for later construction at a genuine planning pause.

The wrapper returns the original delegated command unchanged. Rest and command
history are accepted only after the caller confirms its matching physical
interval. Nothing here starts withdrawal, restores state or authorizes a pause.
"""
import copy
import hashlib

import numpy as np

from .resting_transfer import RestingSupportWindow
from .withdrawal_motor_capture import WithdrawalMotorCapture


class LiveTransferHandoffObserver:
    requires_measured_rest = True

    def __init__(self, transfer):
        self.transfer = transfer
        self.acquisition = transfer.acquisition
        self.operation = transfer.operation
        self.rest = RestingSupportWindow()
        self.motor_capture = WithdrawalMotorCapture(self.acquisition.caps)
        self.pending = None
        self.accepted_epoch = None
        self.accepted_rest = None
        self.accepted_command = None
        self.accepted_intervals = 0
        self.failure = None

    @property
    def started(self):return self.transfer.started

    @property
    def return_started(self):return None

    def _fail(self, error):
        self.failure = self.failure or str(error)
        raise ValueError('Live transfer handoff observation failed: '+self.failure)

    def force(self, t, *args, **kwargs):
        if self.failure is not None:self._fail(self.failure)
        if self.pending is not None:self._fail('Previous delegated interval is still unaccepted')
        if (type(t) not in (int, float) or not np.isfinite(t)
                or (self.accepted_epoch is not None and t != self.accepted_epoch)):
            self._fail('Next command must use the previous accepted physical epoch')
        force, info = self.transfer.force(t, *args, **kwargs)
        command = np.asarray(force)
        if (command.dtype.kind != 'f' or command.shape != (len(self.motor_capture.caps),)
                or not np.isfinite(command).all()
                or np.any(command < self.motor_capture.caps[:, 0])
                or np.any(command > self.motor_capture.caps[:, 1])):
            self._fail('Original complete finite capped delegated command required')
        self.pending = (float(t), command.copy())
        return force, info

    def observe_completed_interval(self, *, command_time_s, post_step_time_s,
            returned_command, grasp_qualified, left_palm_load, angles,
            submission_valid):
        if self.failure is not None:self._fail(self.failure)
        try:
            if self.pending is None or submission_valid is not True:
                raise ValueError('One matching accepted physical interval required')
            t, command = self.pending
            observed = np.asarray(returned_command)
            if (type(command_time_s) not in (int, float)
                    or type(post_step_time_s) not in (int, float)
                    or not np.isfinite([command_time_s, post_step_time_s]).all()
                    or command_time_s != t or abs(post_step_time_s-t-.002) > 1e-10
                    or observed.dtype != command.dtype or observed.shape != command.shape
                    or observed.tobytes() != command.tobytes()):
                raise ValueError('Original command bytes and t/T epochs must match')
            if type(grasp_qualified) is not bool:
                raise ValueError('Actual boolean selected grasp observation required')
            rest = dict(grasp_qualified=grasp_qualified, left_palm_load=left_palm_load,
                angles=copy.deepcopy(angles))
            self.rest.observe(post_step_time_s, transfer_started=self.started,
                grasp_qualified=grasp_qualified, palm_load=left_palm_load, angles=angles)
            self.motor_capture.observe(t, command)
            self.accepted_epoch = float(post_step_time_s)
            self.accepted_rest = rest
            self.accepted_command = command.copy()
            self.accepted_intervals += 1
            self.pending = None
        except Exception as error:self._fail(error)

    def observe_rest(self, t, *, grasp_qualified, left_palm_load, angles):
        """The late controller may read T again; it cannot add another sample."""
        self.require_ready(t)
        observed = dict(grasp_qualified=grasp_qualified, left_palm_load=left_palm_load,
            angles=angles)
        if observed != self.accepted_rest:self._fail('Entry rest observation changed after the accepted interval')
        return self.rest.ready

    def require_ready(self, t, *, transfer=None):
        if (self.failure is not None or self.pending is not None
                or (transfer is not None and transfer is not self.transfer)
                or type(t) not in (int, float) or not np.isfinite(t)
                or t != self.accepted_epoch or not self.rest.ready
                or self.rest.previous != t):
            self._fail('Genuine unchanged ready rest/command observers required at pause T')
        self.motor_capture.capture(t)
        return self.receipt()

    def receipt(self):
        command = self.accepted_command
        return dict(schema='doorbench.live-isaac-transfer-handoff-observation.v1',
            accepted_epoch_s=self.accepted_epoch, accepted_intervals=self.accepted_intervals,
            pending_unaccepted_command=self.pending is not None, failure=self.failure,
            rest=self.rest.diagnostic(), accepted_rest=copy.deepcopy(self.accepted_rest),
            terminal_command=(None if command is None else dict(
                command_time_s=self.motor_capture.previous_time,
                post_step_time_s=self.accepted_epoch, dtype=command.dtype.str,
                shape=list(command.shape), value=command.tolist(),
                sha256=hashlib.sha256(command.tobytes()).hexdigest())),
            authorized_stages=0, controller_state_restoration_supported=False)

    def snapshot(self):
        """Copy the explicit observer-state file contract; no new observation."""
        capture = self.motor_capture
        return dict(schema='doorbench.paused-transfer-observers.v1',
            rest_window=dict(previous=self.rest.previous, since=self.rest.since,
                verified_at=self.rest.verified_at, ready=self.rest.ready),
            motor_capture=dict(previous_time=capture.previous_time,
                previous_command=(None if capture.previous_command is None else capture.previous_command.tolist()),
                step_seconds=capture.step_seconds,caps=capture.caps.tolist()))

    def retained_objects(self):
        """Strong identity inventory for the producer's same-process pause."""
        return dict(handoff_observer=self,transfer=self.transfer,rest=self.rest,
            motor_capture=self.motor_capture,acquisition=self.acquisition,
            operation=self.operation,left=self.transfer.left,
            support_feedback=self.transfer.support_feedback)

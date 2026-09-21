"""Capture the actual preceding delegated command for withdrawal handoff.

A nested operation controller's cache may precede transfer/arm overrides.
This observer copies the command returned by the outer controller. It never
changes a source command or reconstructs it from velocity/gravity estimates.
"""
import numpy as np


class WithdrawalMotorCapture:
    def __init__(self, caps, *, step_seconds=.002):
        self.caps = np.asarray(caps, dtype=float).copy()
        if (self.caps.ndim != 2 or self.caps.shape[1] != 2 or
                not np.isfinite(self.caps).all() or np.any(self.caps[:, 0] >= self.caps[:, 1]) or
                not np.isfinite(step_seconds) or not 0 < step_seconds <= .05):
            raise ValueError('Original finite motor limits and controller interval required')
        self.step_seconds = float(step_seconds)
        self.previous_time = None
        self.previous_command = None

    def observe(self, t, command):
        command = np.asarray(command, dtype=float)
        if (not np.isfinite(t) or command.shape != (len(self.caps),) or
                not np.isfinite(command).all() or np.any(command < self.caps[:, 0]) or
                np.any(command > self.caps[:, 1])):
            raise ValueError('Actual delegated command must respect the original motor limits')
        if self.previous_time is not None and not 0 < t-self.previous_time <= self.step_seconds+1e-8:
            raise ValueError('Consecutive monotonic delegated commands required')
        self.previous_time = float(t)
        self.previous_command = command.copy()

    def capture(self, t):
        if (self.previous_time is None or not np.isfinite(t) or
                abs(t-self.previous_time-self.step_seconds) > 1e-8):
            raise ValueError('Withdrawal needs the immediately preceding actual delegated command')
        return self.previous_command.copy()

    def diagnostic(self, t, stale_cache):
        actual = self.capture(t)
        cache = np.asarray(stale_cache, dtype=float)
        if cache.shape != actual.shape or not np.isfinite(cache).all():
            raise ValueError('Finite complete controller cache required for comparison')
        return dict(withdrawal_motor_capture='delegated-command-v1',
                    captured_command_time_s=self.previous_time,
                    maximum_cache_disagreement_Nm=float(np.max(np.abs(actual-cache))),
                    captured_command_Nm=actual.tolist(),
                    scope='Actual outer-controller command; source forces and motor limits unchanged')

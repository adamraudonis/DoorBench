"""Measured resting support bridge; this does not execute or claim a return."""
import math


class RestingSupportWindow:
    """Require consecutive 500 Hz observations of the original resting gates."""
    def __init__(self):
        self.previous = None
        self.since = None
        self.verified_at = None
        self.ready = False

    def observe(self, t, *, transfer_started, grasp_qualified, palm_load, angles):
        values = (t, palm_load, angles['operator'], angles['latch'], angles['leaf'])
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) or palm_load < 0:
            raise ValueError('Finite measured palm load, time and mechanism state required')
        if self.previous is not None and t < self.previous-1e-9:
            raise ValueError('Measured-rest clock cannot move backwards')
        if self.previous is not None and t-self.previous > .002+1e-8:
            self.since = None
        good = (transfer_started is not None and grasp_qualified and palm_load >= 2.
                and abs(angles['operator']) <= .05 and abs(angles['latch']) <= .001
                and .075 <= angles['leaf'] <= .10)
        if not good:
            self.since = None
        elif self.since is None:
            self.since = float(t)
        self.previous = float(t)
        self.ready = bool(self.since is not None and t-self.since >= .5-1e-8)
        if self.ready and self.verified_at is None:
            self.verified_at = float(t)
        return self.ready

    def diagnostic(self):
        return dict(measured_rest_ready=self.ready, measured_rest_since_s=self.since,
                    measured_rest_verified_s=self.verified_at,
                    measured_rest_last_observation_s=self.previous,
                    measured_rest_support_surface='left_palm_only',
                    measured_rest_scope='Observed spring-rest state; no lever-return milestone')


class RestingTransferBridge:
    """Delegate unchanged capped motors while measuring actual resting support."""
    requires_measured_rest = True

    def __init__(self, transfer):
        self.transfer = transfer
        self.acquisition = transfer.acquisition
        self.operation = transfer.operation
        self.rest = RestingSupportWindow()

    @property
    def started(self):
        return self.transfer.started

    @property
    def return_started(self):
        return None

    def observe_rest(self, t, *, grasp_qualified, left_palm_load, angles):
        return self.rest.observe(t, transfer_started=self.started,
            grasp_qualified=grasp_qualified, palm_load=left_palm_load, angles=angles)

    def force(self, t, root, joints, velocities, handle_pose, leaf_pose, angles,
              hand_loads, *, grasp_qualified, left_panel_load, left_palm_load):
        self.observe_rest(t, grasp_qualified=grasp_qualified,
                          left_palm_load=left_palm_load, angles=angles)
        force, info = self.transfer.force(t, root, joints, velocities, handle_pose,
            leaf_pose, angles, hand_loads, grasp_qualified=grasp_qualified,
            left_panel_load=left_panel_load, left_palm_load=left_palm_load)
        return force, {**info, **self.rest.diagnostic()}

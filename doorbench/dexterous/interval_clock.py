"""Validate recorded interval clocks against repeated simulator timesteps."""
import math


def validate_step_epochs(raw, endpoint, expected_start, dt=.002):
    """Return the next expected epoch without taking time from the recording.

    Repeated addition matches the native simulator's clock. Multiplying the
    step index by dt drifts relative to it by over 1ns after about 362 seconds.
    Keep the original 1ns epoch tolerance; missing/reordered frames still fail.
    """
    if not math.isfinite(expected_start) or expected_start < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Finite nonnegative epoch and positive timestep required')
    end=expected_start+dt
    for value,wanted in ((raw['interval_start_s'],expected_start),
                         (raw['geometry_time_s'],expected_start),
                         (raw['interval_end_s'],end),
                         (endpoint['contact_geometry_time_s'],expected_start),
                         (endpoint['measurement_pose_time_s'],end)):
        if not math.isfinite(value) or abs(value-wanted)>1e-9:
            raise ValueError('Contact or endpoint epoch mismatch')
    return end

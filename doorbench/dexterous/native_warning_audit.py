"""Actual-data MuJoCo warning counters, including contact/constraint capacity."""
import mujoco
import numpy as np


def warning_counts(data):
    return np.array([data.warning[i].number for i in range(int(mujoco.mjtWarning.mjNWARNING))], dtype=np.int64)


def warning_interval(before, after):
    before, after = np.asarray(before), np.asarray(after)
    expected = (int(mujoco.mjtWarning.mjNWARNING),)
    if (before.shape != expected or after.shape != expected or
            before.dtype.kind not in 'iu' or after.dtype.kind not in 'iu' or
            np.any(before < 0) or np.any(after < before)):
        raise ValueError('Complete monotonically increasing actual warning counters required')
    delta = after-before
    return dict(passed=bool(np.all(delta == 0)), before=before.tolist(), after=after.tolist(),
                deltas=delta.tolist())

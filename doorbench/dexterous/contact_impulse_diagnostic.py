"""Time-resolved contact measurements, with no task pass/fail classification.

Inputs are nonnegative normal loads from complete, consecutive physical
intervals. Never substitute a contact count or a recomputed forward solve.
"""
from __future__ import annotations

import numpy as np


def _loads(forces, dt):
    force = np.asarray(forces, dtype=np.float64)
    if force.ndim != 1 or not force.size or not np.isfinite(force).all():
        raise ValueError("Require a nonempty finite one-dimensional load stream")
    if (force < 0).any() or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Require nonnegative loads and a positive finite timestep")
    return force


def longest_gap(forces, dt, threshold):
    """Duration of the longest contiguous interval below the declared load."""
    force = _loads(forces, dt)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Require a positive finite threshold")
    longest = current = 0
    for value in force:
        current = current + 1 if value < threshold else 0
        longest = max(longest, current)
    return float(longest * dt)


def impulse_summary(forces, dt):
    """Measure exact interval integrals at 10 and 20 ms; do not score a run.

    Windows begin at every physics interval boundary. Fixed steps must resolve
    both windows exactly. A zero-load sample means <1e-9 N for both its count
    and contiguous duration. The extra boundary point of a pointwise state
    report is not an additional physical force interval.
    """
    force = _loads(forces, dt)
    out = dict(samples=len(force), zero_load_samples=int(np.sum(force < 1e-9)),
               below_2N_samples=int(np.sum(force < 2.)), mean_N=float(force.mean()),
               longest_zero_load_gap_s=longest_gap(force, dt, 1e-9),
               longest_below_2N_gap_s=longest_gap(force, dt, 2.))
    for seconds in (.01, .02):
        n = round(seconds / dt)
        if n < 1 or abs(n * dt - seconds) > 1e-10 or len(force) < n:
            raise ValueError("Timestep and sample count must resolve each impulse window")
        impulse = np.convolve(force, np.ones(n) * dt, mode="valid")
        minimum = float(impulse.min())
        out[f"rolling_{round(seconds * 1000)}ms"] = dict(
            minimum_impulse_Ns=minimum, minimum_average_force_N=minimum / seconds,
            windows_below_2N_average=int(np.sum(impulse < 2. * seconds)),
            windows=len(impulse))
    return out

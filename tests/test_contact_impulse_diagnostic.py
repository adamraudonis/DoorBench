"""Distinguish brief rigid contact pulses from long unsupported intervals."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "doorbench/dexterous/contact_impulse_diagnostic.py"
spec = importlib.util.spec_from_file_location("contact_impulse_diagnostic", path)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_pulsed_support_and_equal_mean_coasting_are_distinguishable():
    pulses = diagnostic.impulse_summary(np.tile([0., 10.], 125), .002)
    coast = diagnostic.impulse_summary(np.r_[np.zeros(125), np.full(125, 10.)], .002)
    assert pulses["mean_N"] == coast["mean_N"] == 5.
    assert pulses["longest_zero_load_gap_s"] == .002
    assert coast["longest_zero_load_gap_s"] == .25
    assert pulses["rolling_10ms"]["minimum_average_force_N"] == pytest.approx(4.)
    assert coast["rolling_20ms"]["minimum_impulse_Ns"] == 0.


def test_same_piecewise_constant_load_is_invariant_to_subdivision():
    loads = np.r_[np.full(23, 4.), np.zeros(2), np.full(225, 4.)]
    reports = [diagnostic.impulse_summary(np.repeat(loads, factor), .002 / factor)
               for factor in (1, 2, 4)]
    for r in reports:
        assert r["longest_below_2N_gap_s"] == .004
        assert r["mean_N"] == pytest.approx(3.968)
        assert r["rolling_10ms"]["minimum_impulse_Ns"] == pytest.approx(.024)
        assert r["rolling_20ms"]["minimum_impulse_Ns"] == pytest.approx(.064)


def test_one_large_impulse_cannot_hide_a_long_zero_load_gap():
    r = diagnostic.impulse_summary(np.r_[1000., np.zeros(249)], .002)
    assert r["mean_N"] == 4.
    assert r["longest_zero_load_gap_s"] == .498
    assert r["rolling_10ms"]["minimum_impulse_Ns"] == 0.


def test_interval_count_excludes_extra_state_boundary():
    r = diagnostic.impulse_summary(np.full(250, 2.), .002)
    assert r["rolling_10ms"]["windows"] == 246
    assert r["rolling_20ms"]["windows"] == 241
    assert r["longest_below_2N_gap_s"] == 0.


@pytest.mark.parametrize("loads,dt", [([], .002), ([[1]], .002),
    ([np.nan] * 10, .002), ([np.inf] * 10, .002), ([-1.] * 10, .002),
    ([1.] * 10, 0.), ([1.] * 10, float("nan")), ([1.] * 10, .003),
    ([1.] * 9, .002)])
def test_invalid_or_unresolved_evidence_rejected(loads, dt):
    with pytest.raises(ValueError):
        diagnostic.impulse_summary(loads, dt)

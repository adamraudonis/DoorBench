import pytest

from scripts.dexterous.summarize_grasp_migration import summarize


def row(t, body, force, z, qualified):
    return {'sim_time_s': t, 'valid_pad_grasp': qualified, 'contacts': [{
        'body': '/World/H1/' + body, 'normal_force_N': force,
        'body_position_m': [0., 0., z], 'pad_qualified': qualified}]}


def test_local_positions_and_first_loaded_failure():
    result = summarize(iter([row(0., 'distal', 1., .01, True),
        row(.1, 'distal', 3., .02, True), row(.2, 'middle', .01, .5, False),
        row(.3, 'middle', 2., .04, False)]))
    assert result['first_unqualified_loaded_patch']['time_s'] == .3
    bodies = result['bins'][0]['bodies']
    assert bodies['distal']['force_weighted_body_position_m'][2] == pytest.approx(.0175)
    assert bodies['middle']['force_weighted_body_position_m'][2] == .04
    assert result['recorded_samples'] == 4


def test_repeated_epoch_rejected():
    with pytest.raises(ValueError, match='epochs'):
        summarize([row(1., 'distal', 1., 0., True)] * 2)


def test_nonfinite_force_rejected():
    with pytest.raises(ValueError, match='force'):
        summarize([row(1., 'distal', float('nan'), 0., True)])

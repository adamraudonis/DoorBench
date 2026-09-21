import numpy as np
import pytest

from doorbench.dexterous.motor_handoff import MotorHandoff
from doorbench.dexterous.withdrawal_motor_capture import WithdrawalMotorCapture


def test_outer_command_overrides_stale_nested_cache_without_changing_source():
    caps = np.array([[-100., 100.], [-20., 20.], [-1., 1.]])
    actual = np.array([3.582824106, -11.85317067, .12635821])
    original = actual.copy()
    stale = np.array([-46.68718327, .63075212, .12635821])
    observer = WithdrawalMotorCapture(caps)
    observer.observe(49.998, actual)
    assert np.array_equal(actual, original)
    captured = observer.capture(50.)
    blend = MotorHandoff(captured, stale, caps, 1.)
    np.testing.assert_allclose(blend.force(stale, 0.), original, rtol=0., atol=1e-12)
    assert observer.diagnostic(50., stale)['maximum_cache_disagreement_Nm'] == pytest.approx(50.270007376)


def test_capture_is_a_copy_and_includes_all_final_motor_overrides():
    caps = [[-100., 100.], [-100., 100.]]
    observer = WithdrawalMotorCapture(caps)
    nested = np.array([1., 2.])
    returned = nested + [10., 20.]
    observer.observe(1., returned)
    returned[:] = 0.
    captured = observer.capture(1.002)
    assert captured.tolist() == [11., 22.]
    captured[:] = 0.
    assert observer.capture(1.002).tolist() == [11., 22.]


@pytest.mark.parametrize('delta', [0., -.002, .001, .004, float('nan')])
def test_stale_future_and_missing_step_commands_fail_closed(delta):
    observer = WithdrawalMotorCapture([[-1., 1.]])
    observer.observe(50., [.2])
    with pytest.raises(ValueError):
        observer.capture(50.+delta)


def test_missing_capture_and_invalid_force_are_rejected():
    observer = WithdrawalMotorCapture([[-1., 1.]])
    with pytest.raises(ValueError):
        observer.capture(50.)
    for command in ([1.001], [float('nan')], [.2, .3]):
        with pytest.raises(ValueError):
            observer.observe(50., command)


def test_observation_requires_a_continuous_source_clock():
    observer = WithdrawalMotorCapture([[-1., 1.]])
    observer.observe(0., [.2])
    observer.observe(.002, [.3])
    assert observer.capture(.004).tolist() == [.3]
    with pytest.raises(ValueError):
        observer.observe(.006, [.4])


@pytest.mark.parametrize('enabled', [False, True])
def test_withdrawal_prefix_returns_delegated_command_and_info_unchanged(enabled):
    from types import SimpleNamespace
    from doorbench.dexterous.standing_withdrawal import StandingWithdrawalTeacher
    command = np.array([3., -2.])
    info = {'source_phase': 'transfer'}
    teacher = StandingWithdrawalTeacher.__new__(StandingWithdrawalTeacher)
    teacher.acquisition = SimpleNamespace()
    teacher.returned = SimpleNamespace(force=lambda *args, **kwargs: (command, info))
    teacher.palm_only_support = True
    teacher.measured_rest = False
    teacher.started_withdrawal = None
    teacher.start_time = 50.
    teacher.qualified_since = None
    teacher.motor_capture = WithdrawalMotorCapture([[-10., 10.], [-10., 10.]]) if enabled else None
    for t in (49.996, 49.998):
        returned, metadata = teacher.force(t, None, {}, {}, None, None, {}, {},
            grasp_qualified=True, left_panel_load=4., left_palm_load=4.)
        assert returned is command
        assert metadata is info
        assert np.array_equal(command, [3., -2.])
    if enabled:
        assert np.array_equal(teacher.motor_capture.capture(50.), command)

"""Opt-in smooth finger stiffness experiment over the frozen tactile controller.

Only copied policy feedback coefficients change. Native actuator gains, passive
mechanics, transmissions, force caps, touch gates and joint goals are unchanged.
"""
import numpy as np
from .sensor_distal_touch_control import SensorDistalTouchController


PROTOCOL = {
    'schema': 'doorbench.sensor-distal-touch-impedance.v1',
    'acquisition_unchanged_until_s': 19.,
    'initial_finger_position_gain_multiplier': 4.,
    'final_finger_position_gain_multiplier': 16.,
    'ramp_duration_s': 2.,
    'ramp_shape': 'quintic-smoothstep',
    'finger_velocity_damping_Nm_s_per_rad': .05,
    'finger_target_velocity_damping': True,
    'touch_protocol': 'local_distal_touch_preload_v1',
    'duration_s': 36.,
}


def validate_impedance_protocol(protocol):
    if type(protocol) is not dict or set(protocol) != set(PROTOCOL):
        raise ValueError('Exact declared stiffness comparison protocol required')
    for name, expected in PROTOCOL.items():
        value = protocol[name]
        if isinstance(expected, float):
            if type(value) not in (int, float) or not np.isfinite(value) or value != expected:
                raise ValueError('Unsupported stiffness comparison parameter: ' + name)
        elif type(value) is not type(expected) or value != expected:
            raise ValueError('Unsupported stiffness comparison parameter: ' + name)
    return dict(protocol)


def finger_gain_multiplier(now_s):
    if isinstance(now_s, (bool, np.bool_)) or not np.isfinite(now_s) or now_s < 0:
        raise ValueError('Finite nonnegative numeric clock required')
    u = float(np.clip((now_s - 19.) / 2., 0., 1.))
    return 4. + 12. * u**3 * (10. + u * (-15. + 6. * u))


class SensorDistalTouchImpedanceController(SensorDistalTouchController):
    def __init__(self, arm_controller, operation_schedule, sensor_layout, protocol, motor_contract):
        self.impedance_protocol = validate_impedance_protocol(protocol)
        arm = arm_controller
        if (arm.finger_impedance_multiplier != 4. or arm.finger_velocity_damping != .05
                or not arm.finger_target_velocity_damping):
            raise ValueError('Stiffness ramp requires the frozen fourfold acquisition controller')
        b = arm.balance
        self._finger_mask = arm.finger_motors.copy()
        self._initial_kp = b.kp.copy()
        self._initial_position_bias = b.bias[:, 1].copy()
        # The base's robot-only calculator has normalized affine motors. Read
        # the original contract already admitted against XML by the base, not
        # those normalized calculator coefficients.
        original = motor_contract['actuators']
        if ([r['name'] for r in original] != list(arm.action_names)
                or not np.array_equal(b.caps, [r['force_range'] for r in original])):
            raise ValueError('Original motor contract order or caps differ')
        native_kp = np.array([r['kp'] for r in original])
        native_position_bias = np.array([r['bias'][1] for r in original])
        if (not np.array_equal(b.kp[self._finger_mask], 4. * native_kp[self._finger_mask])
                or not np.array_equal(b.bias[self._finger_mask, 1], 4. * native_position_bias[self._finger_mask])):
            raise ValueError('Finger feedback must start at exactly four times the authored coefficients')
        super().__init__(arm, operation_schedule, sensor_layout)

    def reset_episode(self):
        super().reset_episode()
        self.arm.balance.kp[:] = self._initial_kp
        self.arm.balance.bias[:, 1] = self._initial_position_bias

    def force(self, packet, *, now_s):
        if self.failed_reason is not None:
            raise RuntimeError('Touch controller requires reset_episode: ' + self.failed_reason)
        try:
            gain = finger_gain_multiplier(now_s)
            # Assignment from immutable initial arrays avoids cumulative gain
            # multiplication. No extra force is added after the capped adapter.
            b = self.arm.balance
            b.kp[self._finger_mask] = self._initial_kp[self._finger_mask] * (gain / 4.)
            b.bias[self._finger_mask, 1] = self._initial_position_bias[self._finger_mask] * (gain / 4.)
            force, info = super().force(packet, now_s=now_s)
            self.info = dict(info, high_level_controller='local_distal_touch_impedance_v1',
                effective_finger_position_gain_multiplier=gain,
                finger_gain_ramp_start_s=19., finger_gain_ramp_duration_s=2.,
                coefficient_scope='copied policy position feedback only; original native mechanics and force caps')
            return force, dict(self.info)
        except Exception as exc:
            self.failed_reason = type(exc).__name__ + ': ' + str(exc)
            raise

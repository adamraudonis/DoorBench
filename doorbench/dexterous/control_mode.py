"""Fail closed when a sensor actor is combined with an oracle controller."""


def validate_sensor_actor_mode(options):
    if not getattr(options,'sensor_policy_checkpoint',None):
        return
    if not getattr(options,'sensor_layout',None):
        raise ValueError('Sensor policy requires the actual sensor calibration')
    forbidden={'native_robot':None,'acquisition':False,'operate_after_acquisition':False,
        'full_sequence_reset':None,'panel_push':False,'mechanism_test':False,
        'upright_gain':0.,'grip_force':0.,'finger_curl':0.,'torso_damping':0.,
        'stance_qp':False,'press_feedforward':False,'grip_reset_targets':False,
        'arm_impedance':1.,'grip_impedance':1.,'grip_rotation_fraction':1.,
        'operator_compliance_gain':0.,'acquisition_middle_finger_force':None,
        'acquisition_index_finger_force':None,'time_scale':1.}
    enabled=[name for name,default in forbidden.items() if getattr(options,name,default)!=default]
    if enabled:
        raise ValueError('Sensor-only execution cannot enable teacher/extra feedback controls: '+', '.join(enabled))

    if getattr(options,'reset_from_acquisition_path',False) is not True or not getattr(options,'sensor_reset_preflight',None):
        raise ValueError('Sensor actor requires --reset-from-acquisition-path and --sensor-reset-preflight for a bound contact-free reset')

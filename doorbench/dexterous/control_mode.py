"""Fail closed when a sensor actor is combined with an oracle controller."""


def validate_sensor_balance_protocol(options):
    """Keep separately qualified balance experiments and their durations distinct."""
    calibration=getattr(options,'sensor_balance_calibration',None)
    robot=getattr(options,'sensor_balance_robot',None)
    arms=getattr(options,'sensor_arm_schedule',None)
    reach=getattr(options,'sensor_reach_protocol',None)
    route=getattr(options,'sensor_reach_route',None)
    acquisition=getattr(options,'sensor_acquisition_protocol',None)
    acquisition_route=getattr(options,'sensor_acquisition_route',None)
    if bool(calibration)!=bool(robot):
        raise ValueError('Sensor balance requires both frozen calibration and robot-only XML')
    if bool(reach)!=bool(route):
        raise ValueError('Coordinated reach requires both its frozen protocol and joint-only route')
    if bool(acquisition)!=bool(acquisition_route):
        raise ValueError('Acquisition requires both its frozen protocol and joint-only route')
    if acquisition and (not calibration or reach or arms):
        raise ValueError('Acquisition requires sensor balance and its own contact-enabled experiment')
    if arms and not calibration:
        raise ValueError('Scripted arm balance requires the frozen sensor balance calibration')
    if reach and (not calibration or arms):
        raise ValueError('Coordinated reach requires sensor balance and a separate experiment from the scripted-arm protocol')
    if calibration:
        duration=19. if acquisition else 11. if reach else 6. if arms else 5.
        if getattr(options,'sensor_policy_checkpoint',None) or getattr(options,'seconds',None)!=duration:
            raise ValueError('Sensor balance requires its separate frozen protocol: 5s stationary, 6s scripted arms, 11s contact-free reach or 19s acquisition')


def validate_sensor_actor_mode(options):
    if not (getattr(options,'sensor_policy_checkpoint',None) or getattr(options,'sensor_balance_calibration',None)):
        return
    if not getattr(options,'sensor_layout',None):
        raise ValueError('Sensor policy requires the actual sensor calibration')
    forbidden={'native_robot':None,'acquisition':False,'operate_after_acquisition':False,
        'full_sequence_reset':None,'full_opening':False,'left_palm_targets':None,'right_release_screen':None,'bimanual_runtime_screen':None,'panel_push':False,'mechanism_test':False,
        'upright_gain':0.,'grip_force':0.,'finger_curl':0.,'torso_damping':0.,
        'stance_qp':False,'press_feedforward':False,'grip_reset_targets':False,
        'arm_impedance':1.,'grip_impedance':1.,'grip_rotation_fraction':1.,
        'operator_compliance_gain':0.,'acquisition_middle_finger_force':None,
        'acquisition_index_finger_force':None,'time_scale':1.,
        'follow_leaf_during_transfer':False,'panel_profile':None,'palm_load_target':None,
        'left_planning_profile':None,'whole_body_return_path':None,'whole_body_ungrip_path':None,
        'transfer_load_target':4.}
    enabled=[name for name,default in forbidden.items() if getattr(options,name,default)!=default]
    if enabled:
        raise ValueError('Sensor-only execution cannot enable teacher/extra feedback controls: '+', '.join(enabled))

    if getattr(options,'reset_from_acquisition_path',False) is not True or not getattr(options,'sensor_reset_preflight',None):
        raise ValueError('Sensor actor requires --reset-from-acquisition-path and --sensor-reset-preflight for a bound contact-free reset')

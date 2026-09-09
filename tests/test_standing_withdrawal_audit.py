from doorbench.dexterous.standing_withdrawal_audit import withdrawal_checks


def rows():
    return [dict(sim_time_s=i/10,handle_angle_rad=0.,bolt_slide_m=0.,
        pad_grasp=dict(valid_pad_grasp=i<=10,contacts=[]),
        left_surface=dict(palm_normal_load_N=3.),right_environment_clearance_m=.05 if i>=15 else 0.) for i in range(21)]


def audit(data):
    return withdrawal_checks({'sustained_pad_grasp':False,'native_motor_limits':True},data,
        dt=.1,duration=2.,started=.6,release_started=1.,completed=True)


def test_intentional_release_requires_prior_opposition_and_final_clearance():
    result=audit(rows())
    assert 'sustained_pad_grasp' not in result and all(result.values())
    data=rows();data[8]['pad_grasp']['valid_pad_grasp']=False
    assert not audit(data)['opposed_grip_before_intentional_release']
    data=rows();data[-2]['right_environment_clearance_m']=.03
    assert not audit(data)['final_hand_clear_of_environment']


def test_release_does_not_exempt_wrong_loaded_surfaces_or_failed_motor_checks():
    data=rows();data[12]['pad_grasp']['contacts']=[{'pad_qualified':False}]
    assert not audit(data)['no_invalid_loaded_right_surfaces']
    result=withdrawal_checks({'native_motor_limits':False},rows(),dt=.1,duration=2.,started=.6,release_started=None,completed=False)
    assert not result['native_motor_limits'] and not result['withdrawal_route_completed']
    assert not result['opposed_grip_before_intentional_release']

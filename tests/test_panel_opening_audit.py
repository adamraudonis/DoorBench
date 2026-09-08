from doorbench.dexterous.panel_opening_audit import audit_panel_opening


def rows(angles, *, contact=True):
    return [dict(time_s=i*.02, door={'leaf_hinge':a}, teacher={'panel_phase':'push'},
        hand_forces_panel_N={'palm':[0,8 if contact else 0,0]}) for i,a in enumerate(angles)]


def test_post_release_coasting_does_not_count_as_panel_opening():
    result=audit_panel_opening(rows([.4,.7,1.2],contact=False),{'passed':True},{'passed':True})
    assert not result['passed']
    assert not result['checks']['usable_aperture_under_contact']


def test_touching_an_already_open_door_does_not_count_as_opening():
    result=audit_panel_opening(rows([.9,1.1,1.3]),{'passed':True},{'passed':True})
    assert not result['passed']
    assert not result['checks']['contact_before_usable_aperture']


def test_loaded_opening_retains_the_original_and_mechanical_gates():
    data=rows([.35,.55,.75,1.2])
    assert audit_panel_opening(data,{'passed':True},{'passed':True})['passed']
    assert not audit_panel_opening(data,{'passed':False},{'passed':True})['passed']
    assert not audit_panel_opening(data,{'passed':True},{'passed':False})['passed']


def test_brief_brush_near_threshold_does_not_count_as_substantial_push():
    result=audit_panel_opening(rows([.69,.70,.71]),{'passed':True},{'passed':True})
    assert not result['passed']
    assert not result['checks']['substantial_loaded_panel_travel']


def test_loaded_closing_motion_does_not_count_as_opening():
    result=audit_panel_opening(rows([.69,.5,.2]),{'passed':True},{'passed':True})
    assert not result['passed']
    assert result['loaded_panel_travel_rad']==0.

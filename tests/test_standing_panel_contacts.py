from scripts.dexterous.audit_standing_panel_contacts import allowed_contact


def test_only_planned_external_support_is_allowed_in_either_order():
    assert allowed_contact(['robot/lh_palm','leaf'],['palm','slab'])
    assert allowed_contact(['leaf','robot/lh_ffdistal'],['slab','finger'])
    assert allowed_contact(['world','robot/left_ankle_link'],['floor','foot'])
    assert not allowed_contact(['leaf','robot/left_elbow_link'],['slab','elbow'])
    assert not allowed_contact(['robot/lh_palm','world'],['palm','jamb'])
    assert not allowed_contact(['robot/rh_palm','leaf'],['palm','slab'])

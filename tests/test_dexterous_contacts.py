from doorbench.dexterous.contact_audit import opposition


def contacts(thumb=-1):
    return [{'digit':d,'normal_force_N':1.,'position':[i*.01,1 if d!='th' else thumb,0]}
            for i,d in enumerate(('ff','mf','rf','lf','th'))]


def test_loaded_opposition_and_same_side_rejection():
    assert opposition(contacts(),[0,0,0],[1,0,0])['opposed']
    assert not opposition(contacts(1),[0,0,0],[1,0,0])['opposed']


def test_nearby_unloaded_and_missing_fingers_do_not_count():
    rows=contacts();rows[-1]['normal_force_N']=0
    assert not opposition(rows,[0,0,0],[1,0,0])['opposed']
    assert not opposition(contacts()[1:],[0,0,0],[1,0,0])['opposed']


def test_one_finger_on_thumb_side_fails():
    rows=contacts();rows[0]['position']=[0,-1,0]
    assert not opposition(rows,[0,0,0],[1,0,0])['opposed']


def test_native_capsule_without_contacts_is_not_a_grasp():
    import mujoco
    from doorbench.dexterous.contact_audit import lever_contacts
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="lever" type="capsule" size=".007 .053"/></worldbody></mujoco>')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    report=lever_contacts(m,d,'lever')
    assert not report['opposed'] and report['contacts']==[]

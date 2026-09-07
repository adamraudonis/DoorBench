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

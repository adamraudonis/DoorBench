import pytest
from doorbench.dexterous.isaac_acquisition_contacts import acquisition_hand_contact_counts


def count(body,other,force=1.,distance=-.0001):
    return acquisition_hand_contact_counts(['/World/H1/pelvis/'+body],[[other]],
        [dict(sensor=0,filter=0,force_N=force,distance_m=distance)])


def test_only_working_digits_against_the_exact_handle_are_admitted():
    handle='/World/Door/Articulation/leaf_handle'
    assert count('rh_ffdistal',handle)==dict(hand_contact_count=1,unintended_hand_contact_count=0)
    for body,other in [('rh_palm',handle),('lh_thdistal',handle),('rh_ffdistal','/World/Door/Articulation/leaf'),
                       ('rh_ffdistal','/World/Other/leaf_handle')]:
        assert count(body,other)==dict(hand_contact_count=1,unintended_hand_contact_count=1)
    assert count('right_ankle_link',handle)['hand_contact_count']==0


def test_unloaded_separated_contact_buffers_are_not_actual_touch():
    assert count('rh_palm','/World/Door/Articulation/leaf_handle',0.,.001)['hand_contact_count']==0
    assert count('rh_palm','/World/Door/Articulation/leaf_handle',0.,0.)['unintended_hand_contact_count']==1


@pytest.mark.parametrize('force',[float('nan'),float('inf'),-1.])
def test_invalid_forces_fail_closed(force):
    with pytest.raises(ValueError):count('rh_ffdistal','/World/Door/Articulation/leaf_handle',force)

import numpy as np
import pytest
from doorbench.dexterous.palm_normal_admittance import PalmNormalAdmittance


def test_contact_loss_cannot_wind_up_or_cross_the_screened_offset():
    controller=PalmNormalAdmittance();previous=0.;maximum_speed=0.;maximum_acceleration=0.
    for t in np.arange(0,20,.002):
        offset,info=controller.update(float(t),0.)
        assert -1e-12<=offset<=.002+1e-12
        assert offset>=previous-1e-12;previous=offset
        maximum_speed=max(maximum_speed,abs(info['normal_offset_velocity_m_s']))
        maximum_acceleration=max(maximum_acceleration,abs(info['normal_offset_acceleration_m_s2']))
    assert offset==pytest.approx(.002,abs=1e-12)
    assert maximum_speed<=.000250000001 and maximum_acceleration<=.000500000001
    for t in np.arange(20,40,.002):offset,info=controller.update(float(t),8.)
    assert offset==pytest.approx(0.,abs=1e-12)
    assert info['normal_offset_velocity_m_s']==pytest.approx(0.,abs=1e-12)


def test_measured_load_not_commanded_force_drives_slow_target():
    controller=PalmNormalAdmittance()
    for t in np.arange(0,2,.002):offset,info=controller.update(float(t),4.)
    assert offset==0. and info['filtered_palm_load_N']==4.
    offset,info=controller.update(2.,.65)
    assert 0<=offset<1e-8
    assert .65<info['filtered_palm_load_N']<4.


@pytest.mark.parametrize('time,load',[(.004,4.),(.002,np.nan),(.002,-1.)])
def test_invalid_tactile_packet_is_rejected(time,load):
    controller=PalmNormalAdmittance();controller.update(0.,4.)
    with pytest.raises(ValueError):controller.update(time,load)

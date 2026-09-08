import copy
import numpy as np
import pytest
mujoco=pytest.importorskip('mujoco')
from doorbench.dexterous.isaac_tendons import native_passive_tendons,validate_contract


def model(extra=''):
    return mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody><body name="middle"><joint name="rh_FFJ2" range="0 1.5708"/><geom size=".01" mass=".02"/><body name="distal" pos="0 0 .05"><joint name="rh_FFJ1" range="0 1.5708"/><geom size=".01" mass=".02"/></body></body></worldbody><tendon><fixed name="rh_FFJ0"><joint joint="rh_FFJ1" coef="1"/><joint joint="rh_FFJ2" coef="1"/></fixed>'''+extra+'''</tendon></mujoco>''')


LIMIT='''<fixed name="rh_FF_loopback" limited="true" range="-2 0"><joint joint="rh_FFJ1" coef="1"/><joint joint="rh_FFJ2" coef="-1"/></fixed>'''


def test_motor_transmission_is_not_an_unrequested_spring():
    assert native_passive_tendons(model())==[]


def test_passive_difference_is_retained_separately_without_actuators():
    m=model(LIMIT);before=m.actuator_trnid.copy();contract=native_passive_tendons(m)
    assert len(contract)==1 and contract[0]['terms']=={'rh_FFJ1':1.,'rh_FFJ2':-1.}
    assert contract[0]['root_joint']=='rh_FFJ2' and contract[0]['range_rad']==[-2.,0.]
    assert contract[0]['spring_stiffness']==contract[0]['damping']==0
    assert np.array_equal(before,m.actuator_trnid)
    assert validate_contract(contract)==contract


@pytest.mark.parametrize('edit',[
    lambda s:s.replace('rh_FF_loopback','unknown'),
    lambda s:s.replace('coef="-1"','coef="1"'),
    lambda s:s.replace('range="-2 0"','range="-2 .1"'),
    lambda s:s.replace('limited="true"','limited="true" stiffness="1"'),
])
def test_unknown_or_wrong_passive_mechanics_never_silently_drop(edit):
    with pytest.raises(ValueError):native_passive_tendons(model(edit(LIMIT)))


def test_contract_rejects_bilateral_spring_or_wrong_sign():
    original=native_passive_tendons(model(LIMIT))
    for key,value in [('spring_stiffness',1.),('damping',1.),('range_rad',[-2.,.1]),('physx_limit_stiffness',np.inf),('root_joint','rh_FFJ1')]:
        modified=copy.deepcopy(original);modified[0][key]=value
        with pytest.raises(ValueError):validate_contract(modified)
    with pytest.raises(ValueError):validate_contract(original+original)

from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.digit_pressure_feedback import DigitPressureFeedback,pressure_correction


def test_load_sign_tangential_invariance_and_bounded_response():
    inward=np.array([0.,1.,0.])
    assert pressure_correction(3.,[0.,-3.,0.],inward)==(0.,3.)
    assert pressure_correction(3.,[100.,-3.,-20.],inward)==(0.,3.)
    assert pressure_correction(3.,[0.,-100.,0.],inward)[0]==-1.
    assert pressure_correction(6.,[0.,0.,0.],inward)[0]==1.
    with pytest.raises(ValueError):pressure_correction(3.,[0.,0.,0.],[0.,2.,0.])


def test_original_finger_motor_only_and_no_physical_state_or_force_writes():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="lever" type="sphere" pos=".1 .025 0" size=".01"/>
      <body name="rh_ffdistal"><joint name="finger" axis="0 0 1"/>
        <geom type="sphere" pos=".1 0 0" size=".01"/></body>
      <body name="arm" pos="0 0 1"><joint name="arm_joint"/><geom size=".01"/></body>
      </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    t=SimpleNamespace(m=m,d=d,lever=m.geom('lever').id,digit_geoms={'ff':[1]},
        digit_forces={'ff':3.},fingers=np.array([0]),finger_inverse=np.array([[1.,0.]]),
        va=np.array([0,1]),caps=np.array([[-.02,.02],[-1.,1.]]))
    before=[a.copy() for a in (d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied)]
    c=DigitPressureFeedback(t)
    balanced,_=c.force([0.,.4],1.,{'robot/rh_ffdistal':[0.,-3.,0.]})
    unloaded,info=c.force([0.,.4],1.,{})
    np.testing.assert_allclose(balanced,[0.,.4]);np.testing.assert_allclose(unloaded,[.02,.4])
    assert info['digit_pressure_feedback']['ff']['measured_normal_N']==0
    for a,b in zip((d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied),before):np.testing.assert_array_equal(a,b)

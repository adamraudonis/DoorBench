"""Critical sensor-to-native state conventions for the portable body teacher."""
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.approach_teacher import ApproachBodyTeacher


def state_reader():
    reader=ApproachBodyTeacher.__new__(ApproachBodyTeacher)
    reader.m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/><body><joint name="hinge"/><geom type="sphere" size=".05" pos="0 0 .2"/></body></body></worldbody></mujoco>')
    reader.d=mujoco.MjData(reader.m);reader.names=['hinge'];reader.qa=np.array([7]);reader.va=np.array([6])
    return reader


def test_world_angular_velocity_is_converted_without_stepping(monkeypatch):
    reader=state_reader()
    def forbidden(*args,**kwargs):raise AssertionError('The analytic mirror must not step')
    monkeypatch.setattr(mujoco,'mj_step',forbidden)
    root=np.array([1.,2.,3.,2**-.5,0.,0.,2**-.5,4.,5.,6.,1.,0.,0.])
    reader._state(.042,root,{'hinge':.2},{'hinge':.3})
    np.testing.assert_allclose(reader.d.qpos[:7],root[:7])
    np.testing.assert_allclose(reader.d.qvel[:3],[4.,5.,6.])
    np.testing.assert_allclose(reader.d.qvel[3:6],[0.,-1.,0.],atol=1e-12)
    assert reader.d.qpos[7]==.2 and reader.d.qvel[6]==.3 and reader.d.time==.042


def test_rejects_missing_or_nonfinite_measurement():
    reader=state_reader()
    with pytest.raises(ValueError,match='Invalid measured'):reader._state(0.,np.zeros(12),{'hinge':0.},{'hinge':0.})
    with pytest.raises(ValueError,match='Invalid measured'):reader._state(0.,np.zeros(13),{'hinge':np.nan},{'hinge':0.})


def test_rejects_skipped_physics_tick_before_control():
    reader=ApproachBodyTeacher.__new__(ApproachBodyTeacher);reader.last_t=.002
    with pytest.raises(ValueError,match='2 ms clock'):reader.force(.006,None,None,None,None)

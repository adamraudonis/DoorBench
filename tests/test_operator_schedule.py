"""Generic component-load schedules use native seconds, not an assumed step."""
import mujoco
import pytest
from doorbench.qa import drive_operators

@pytest.mark.parametrize('dt',(.001,.0005,.00025))
def test_operator_drive_applies_same_duration_at_finer_native_steps(dt):
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep="{dt}" gravity="0 0 0"/>
      <worldbody><body><joint name="leaf" type="slide" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>
      <body><joint name="handle" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m)
    q=drive_operators(m,d,m.joint('leaf').id,[m.joint('handle').id],[],-1,1.,False)
    assert d.time==pytest.approx(3.2,abs=1e-10)
    # One newton accelerates one kilogram for 2.6 seconds after the 0.6 s delay.
    assert q==pytest.approx(.5*2.6**2,abs=2*dt)

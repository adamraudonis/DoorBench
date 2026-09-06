"""Physical test durations must survive a smaller mechanism integration step."""
import mujoco
import pytest
from doorbench.qa import push_primary


@pytest.mark.parametrize('timestep',[.002,.00025,.0001])
@pytest.mark.parametrize('held',[False,True])
def test_native_push_has_same_one_second_minimum_at_finer_timesteps(timestep,held):
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep="{timestep}" gravity="0 0 0"/>
      <worldbody><body><joint name="leaf" type="hinge" axis="0 0 1"/>
      <geom type="box" size=".1 .02 .2" mass="5"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m)
    value=push_primary(m,d,0,.01,held,.1)
    assert d.time==pytest.approx(1.,abs=timestep)
    assert value>.1

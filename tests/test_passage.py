"""Counterexamples to angle/travel-only humanoid passage labels."""
import math

import mujoco
import numpy as np
import pytest

from doorbench.benchmark.passage import Passage


@pytest.mark.parametrize('offset',[-.85,-.247,.247,.85])
def test_offset_curtain_blocks_until_it_clears_a_standing_traveller(offset):
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body pos="0 {offset} 1.2"><joint type="slide" axis="0 0 1"/>
        <geom type="box" size="1.2 .02 1.2" mass="60"/>
      </body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    p=Passage(m,{'family':'rollup','opening':{'width':2.4}}, {})
    assert p.intervals(d)==[]
    d.qpos[0]=1.75;mujoco.mj_forward(m,d)
    assert p.intervals(d)==[]
    d.qpos[0]=1.85;mujoco.mj_forward(m,d)
    assert p.intervals(d)


def test_bypass_opening_is_offset_and_centre_stays_blocked():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="rear_leaf" type="box" pos=".6 .08 1" size=".6 .015 1"/>
      <body name="front_leaf" pos="-.6 -.03 1"><joint type="slide" axis="1 0 0"/>
        <geom type="box" size=".6 .015 1" mass="37"/>
      </body></worldbody></mujoco>''')
    d=mujoco.MjData(m)
    p=Passage(m,{'family':'sliding_bypass','opening':{'width':2.4}}, {})
    mujoco.mj_forward(m,d);assert p.intervals(d)==[]
    d.qpos[0]=1.2;mujoco.mj_forward(m,d)
    clear=p.intervals(d)
    np.testing.assert_allclose(clear,[[-.945,-.255]],atol=1e-6)
    assert not any(lo<=0<=hi for lo,hi in clear)
    # A positive opening cannot survive closing at the same timestamp.
    d.qpos[0]=0;mujoco.mj_forward(m,d);assert p.intervals(d)==[]


def test_waist_height_tiltup_is_not_clear_at_ninety_degrees():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body pos="0 0 1"><joint name="lift" type="slide" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
        <body><joint name="tilt" axis="1 0 0"/>
          <geom type="box" size="1.2 .02 1" mass="60"/>
        </body></body></worldbody></mujoco>''')
    d=mujoco.MjData(m)
    p=Passage(m,{'family':'garage_tiltup','opening':{'width':2.4}}, {})
    d.qpos[1]=math.pi/2;mujoco.mj_forward(m,d)
    assert p.intervals(d)==[]
    d.qpos[0]=1.1;mujoco.mj_forward(m,d)
    assert p.intervals(d)


def test_embodiment_does_not_block_its_own_passage_but_fixed_obstacles_do():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="human" mocap="true" pos="0 0 .9">
        <geom type="capsule" size=".22 .65"/>
      </body>
      <geom name="fixed_obstruction" type="box" pos="-.38 0 1" size=".12 .03 1"/>
      </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    spec={'family':'swing_single','opening':{'width':1.}}
    assert Passage(m,spec,{}).intervals(d)==[]
    clear=Passage(m,spec,{},exclude_bodies={'human'}).intervals(d)
    assert clear and clear[0][0]>-.01  # fixed post still consumes left opening

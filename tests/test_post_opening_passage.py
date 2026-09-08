"""Actual-geometry passage guards and whole-body finish bounds."""
from types import SimpleNamespace
import mujoco,numpy as np
import pytest
from doorbench.dexterous.passage import RobotBounds,screen_translation


def scene(angle=1.6,receiving_only=False):
    xml='''<mujoco><compiler angle="radian"/><worldbody>
    <geom name="floor" type="plane" size="4 4 .1"/>
    <geom name="wall_l" type="box" pos="-.75 0 1" size=".2 .1 1"/>
    <geom name="wall_r" type="box" pos=".75 0 1" size=".2 .1 1"/>
    <body name="leaf" pos="-.45 0 0"><joint name="leaf_hinge" axis="0 0 1"/><geom type="box" name="panel" pos=".45 0 1" size=".45 .02 1"/></body>
    <body name="robot/pelvis" pos="0 -.8 1"><freejoint name="robot/root"/><geom name="robot/box" type="box" size=".2 .2 .8"/></body>
    </worldbody></mujoco>'''
    if receiving_only:xml=xml.replace('name="wall_r" type=', 'name="wall_r" contype="0" conaffinity="1" type=')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);d.qpos[m.jnt_qposadr[m.joint('leaf_hinge').id]]=angle;mujoco.mj_forward(m,d)
    return SimpleNamespace(m=m,d=d,root_qadr=int(m.jnt_qposadr[m.joint('robot/root').id]),qadr=np.array([],int))


def test_true_shape_clearance_passes_without_moving_active_state():
    s=scene();before=s.d.qpos.copy();result=screen_translation(s,[0.,.8],samples=31)
    assert result['passed'] and result['minimum_gap_m']>=.003
    assert np.array_equal(s.d.qpos,before) and s.d.time==0


def test_colliding_corridor_is_rejected():
    result=screen_translation(scene(),[.6,.8],samples=31)
    assert not result['passed'] and result['failures']


def test_insufficient_actual_aperture_fails_even_if_pose_is_clear():
    result=screen_translation(scene(angle=1.2),[0.,-.7],samples=31)
    assert not result['passed'] and result['measured_leaf_rad']==1.2


def test_finish_uses_trailing_collision_extent_not_just_pelvis():
    s=scene();bounds=RobotBounds(s.m);s.d.qpos[s.root_qadr+1]=.3;mujoco.mj_forward(s.m,s.d)
    lo,hi=bounds(s.d)
    assert s.d.qpos[s.root_qadr+1]>.2 and lo[1]<.2
    s.d.qpos[s.root_qadr+1]=.6;mujoco.mj_forward(s.m,s.d)
    assert bounds(s.d)[0][1]>.2


def test_receiving_only_wall_is_not_omitted():
    s=scene(receiving_only=True)
    assert s.m.geom_contype[s.m.geom('wall_r').id]==0
    result=screen_translation(s,[.6,.8],samples=31)
    assert not result['passed'] and result['failures'][0]['environment_geom']=='wall_r'


def test_measured_template_root_orientation_is_part_of_clearance_screen():
    s=scene();template=s.d.qpos.copy()
    template[s.root_qadr+3:s.root_qadr+7]=[np.cos(np.pi/8),0,np.sin(np.pi/8),0]
    assert screen_translation(s,[0,.8],samples=31)['passed']
    result=screen_translation(s,[0,.8],samples=31,templates=[template])
    assert not result['passed'] and result['failures'][0]['template']==1


def test_malformed_template_is_rejected():
    with pytest.raises(ValueError,match='Malformed'):
        screen_translation(scene(),[0,.8],samples=31,templates=[np.zeros(2)])

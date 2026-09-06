"""Closing bevels must cam into a real strike and re-extend, with no pose drive."""
import copy
import math

import mujoco
import numpy as np
import pytest

from doorbench import hardware as H
from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.benchmark.env import DoorEnv
from doorbench.geometry.closer_mounts import resolve_closer_configuration
from doorbench.initial_configuration import resolve_joint_followers


@pytest.fixture(scope='module')
def latch_doors(tmp_path_factory):
    root=tmp_path_factory.mktemp('closing-bevels');rows={}
    for source in generate_all():
        if source['index'] not in (256,280,310,501,508,835):continue
        for condition in ([source['condition'],'normal'] if source['index'] in (280,508) else [source['condition']]):
            spec=copy.deepcopy(source);spec['condition']=condition
            target=root/f"{spec['index']}-{condition}"
            export_door(spec,str(target/'doors'),str(target/'hardware'),formats=('mjcf','json'))
            rows[spec['index'],condition]=target/'doors'/spec['id']
    return rows


def native_return(path,tier='full',old_round_nose=False):
    env=DoorEnv(str(path),tier=tier);env.reset(randomize=False)
    m,d=env.m,env.d;primary=env.pj
    if old_round_nose:
        # Restore the old top-rod collision shape only. Keep the current real
        # mass, spring, strike and closer so this isolates the missing bevel.
        geom=m.geom('leaf_top_latch_capsule').id
        bolt=H.LATCHES['rim_exit'];radius=bolt.bolt_size[0]/2;inside=.05
        m.geom_type[geom]=mujoco.mjtGeom.mjGEOM_CAPSULE
        m.geom_size[geom]=[radius,(bolt.throw+inside)/2-radius,0.]
        m.geom_pos[geom]=[0.,0.,(bolt.throw-inside)/2]
        m.geom_quat[geom]=[1.,0.,0.,0.]
    d.qpos[m.jnt_qposadr[primary]]=math.pi/3
    resolve_joint_followers(m,d.qpos,[env.meta['primary_joint']])
    resolve_closer_configuration(m,d.qpos,env.meta)
    env._with_passive(lambda:mujoco.mj_forward(m,d))
    for _ in range(round(15./m.opt.timestep)):
        env._with_passive(lambda:mujoco.mj_step(m,d))
        assert not np.any(d.warning.number)
    angle=float(d.qpos[m.jnt_qposadr[primary]])
    bolts={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)
           if m.joint(j).name.endswith(('latch_bolt_slide','top_latch_slide'))}
    env.close()
    return angle,bolts


@pytest.mark.parametrize('tier',['full','simple','minimal'])
def test_healthy_top_rim_and_deadlatch_close_and_reextend(latch_doors,tier):
    for (index,condition),path in latch_doors.items():
        if index in (280,508) and condition!='normal':continue
        angle,bolts=native_return(path,tier)
        assert abs(angle)<math.radians(.5),(index,tier,angle,bolts)
        assert bolts and all(abs(q)<.001 for q in bolts.values()),(index,tier,bolts)


def test_round_top_rod_restores_real_arrest(latch_doors):
    path=next(path for (index,_),path in latch_doors.items() if index==256)
    angle,bolts=native_return(path,old_round_nose=True)
    assert angle>math.radians(1.5),(angle,bolts)


@pytest.mark.parametrize('index',[508])
def test_rusted_hardware_can_fail_to_self_latch_without_changing_force_limits(latch_doors,index):
    path=next(path for (i,c),path in latch_doors.items() if i==index and c!='normal')
    angle,bolts=native_return(path)
    assert angle>math.radians(.5)
    assert any(q>.002 for q in bolts.values())
    angle,bolts=native_return(latch_doors[index,'normal'])
    assert abs(angle)<math.radians(.5)
    assert all(abs(q)<.001 for q in bolts.values())


def test_flush_lip_removes_rim_arrest_even_with_original_rust(latch_doors):
    path=next(path for (index,condition),path in latch_doors.items() if index==280 and condition!='normal')
    angle,bolts=native_return(path)
    assert abs(angle)<math.radians(.5)
    assert all(abs(q)<.001 for q in bolts.values())

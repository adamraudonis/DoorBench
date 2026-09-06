"""A retracted latch must let the real slab pass its flush strike lip."""
import mujoco
import numpy as np
import pytest
from doorbench.spec import generate_all
from doorbench.build import export_door
from doorbench.benchmark.env import DoorEnv


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    root=tmp_path_factory.mktemp('pair-strike')
    spec=next(s for s in generate_all() if s['index']==846)
    export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
    return root/'doors'/spec['id']


def open_pair(path,tier='full',proud=False):
    e=DoorEnv(str(path),tier=tier);e.reset(randomize=False);m,d=e.m,e.d
    if proud:
        # Restore only the obsolete1.5mm intrusion into the aperture.
        g=m.geom('leaf_b_edge_965_lip_plate').id;m.geom_pos[g,0]-=.0015
    a,v=int(m.jnt_qposadr[e.pj]),int(m.jnt_dofadr[e.pj]);operator=int(m.jnt_dofadr[e.oj])
    peak=0.
    for _ in range(round(4./m.opt.timestep)):
        d.qfrc_applied[operator]=2.
        if d.time>1:d.qfrc_applied[v]=np.clip(100*(1.-d.qpos[a])-20*d.qvel[v],-70,70)
        e.step();peak=max(peak,float(d.qpos[a]))
        assert not np.any(d.warning.number)
    return e,peak


@pytest.mark.parametrize('tier',['full','simple','minimal'])
def test_real_lip_allows_opening_and_relatch_in_every_tier(pair,tier):
    e,peak=open_pair(pair,tier);assert peak>.8
    m,d=e.m,e.d;a=m.jnt_qposadr[e.pj];v=m.jnt_dofadr[e.pj]
    for _ in range(round(6./m.opt.timestep)):
        d.qfrc_applied[v]=np.clip(-100*d.qpos[a]-20*d.qvel[v],-40,40);e.step()
    assert abs(d.qpos[a])<.01
    b=m.jnt_qposadr[e.bj];assert abs(d.qpos[b])<.001
    assert not np.any(d.warning.number);e.close()


def test_old_proud_lip_arrests_retracted_door(pair):
    e,peak=open_pair(pair,proud=True)
    assert peak<.10
    assert e.d.qpos[e.m.jnt_qposadr[e.bj]]>.011
    e.close()

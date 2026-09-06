"""A credential supplies a real coil; bolt travel establishes release."""
import mujoco
import numpy as np
import pytest
from doorbench.benchmark.env import DoorEnv
from doorbench.build import export_door
from doorbench.spec import generate_all


@pytest.fixture(scope='module')
def door(tmp_path_factory):
    root=tmp_path_factory.mktemp('turnstile-power')
    s=next(s for s in generate_all() if s['family']=='turnstile_tripod' and s['kinematics'].get('locked_until_credential'))
    # This service fixture explicitly supplies a valid credential. Published
    # locked-recognize variants keep their unavailable-credential setting.
    s['lock']['robot_side_release']=True
    export_door(s,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
    return root/'doors'/s['id']


def test_badge_withdraws_actual_bolt_then_power_loss_arrests_next_sector(door):
    env=DoorEnv(str(door));env.reset(randomize=False);m,d=env.m,env.d
    row=env.meta['turnstile_locks'];j=env._jid(row['bolt_joint']);a=m.jnt_qposadr[j]
    primary=m.jnt_qposadr[env.pj];v=m.jnt_dofadr[env.pj]
    state={k:getattr(d,k).copy() for k in ('qpos','qvel','eq_active','ctrl')}
    ranges=m.jnt_range.copy();limited=m.jnt_limited.copy()
    env.badge()
    assert not env.tracker.L.lock_released
    for k,value in state.items():np.testing.assert_array_equal(getattr(d,k),value)
    for _ in range(round(.5/m.opt.timestep)):env.step()
    assert d.qpos[a]>row['stroke_m']-.0005 and env.tracker.L.lock_released
    for _ in range(round(.6/m.opt.timestep)):
        d.qfrc_applied[v]=20.-10*d.qvel[v];env.step()
    assert d.qpos[primary]>.15
    start=float(d.qpos[primary]);env.set_turnstile_power(False)
    for _ in range(round(4./m.opt.timestep)):
        d.qfrc_applied[v]=20.-10*d.qvel[v];env.step()
    assert 0<d.qpos[primary]-start<row['sector_angle_rad']+.06
    # Side load can hold the bolt partly withdrawn while its nose carries
    # the actual sector reaction. Full seating is not necessary for arrest.
    assert abs(d.qvel[v])<.01
    assert any(row['bolt_geom'] in (m.geom(c.geom1).name,m.geom(c.geom2).name)
               and any(g in row['index_geoms'] for g in (m.geom(c.geom1).name,m.geom(c.geom2).name))
               for c in d.contact)
    np.testing.assert_array_equal(m.jnt_range,ranges);np.testing.assert_array_equal(m.jnt_limited,limited)
    assert not np.any(d.warning.number)
    env.badge();env.reset(randomize=False)
    assert env.turnstile_power is None and not env.tracker.L.lock_released
    for _ in range(round(.5/m.opt.timestep)):env.step()
    assert d.qpos[a]<.0005
    env.close()


def test_unavailable_credential_cannot_energize_coil(door):
    env=DoorEnv(str(door));env.reset(randomize=False)
    env.spec['lock']['robot_side_release']=False
    before={k:getattr(env.d,k).copy() for k in ('qpos','qvel','eq_active','ctrl')}
    assert env.badge() is False
    assert env.turnstile_power is None and not env.unlocked_by_env
    for k,value in before.items():np.testing.assert_array_equal(getattr(env.d,k),value)
    for _ in range(round(.5/env.m.opt.timestep)):env.step()
    row=env.meta['turnstile_locks'];j=env._jid(row['bolt_joint'])
    assert env.d.qpos[env.m.jnt_qposadr[j]]<.0005
    assert not env.tracker.L.lock_released
    env.close()


def test_whole_device_power_loss_drops_only_indexed_arm_and_restoration_cannot_lift_it(door):
    env=DoorEnv(str(door));env.reset(randomize=False);m,d=env.m,env.d
    row=env.meta['turnstile_drop_arm']
    arms=[int(m.jnt_qposadr[env._jid(a['arm_joint'])]) for a in row['arms']]
    for _ in range(round(.6/m.opt.timestep)):env.step()
    assert max(abs(float(d.qpos[a])) for a in arms)<.015
    before={k:getattr(d,k).copy() for k in ('qpos','qvel','eq_active','ctrl')}
    ranges=m.jnt_range.copy();limited=m.jnt_limited.copy()
    env.set_turnstile_supply(False)
    assert env.turnstile_supply is False
    assert env.turnstile_power is None and env.turnstile_drop_power is None
    for k,value in before.items():np.testing.assert_array_equal(getattr(d,k),value)
    for _ in range(round(2.5/m.opt.timestep)):env.step()
    assert 1.55<float(d.qpos[arms[0]])<1.59
    assert max(abs(float(d.qpos[a])) for a in arms[1:])<.015
    env.set_turnstile_supply(True)
    for _ in range(round(.8/m.opt.timestep)):env.step()
    assert float(d.qpos[arms[0]])>1.55
    bolt=env.meta['turnstile_locks']['bolt_joint']
    # The falling arm shifts the rotor into the finite index-slot flank,
    # which can side-load a partly seated bolt. Test retained engagement and
    # actual rotor arrest, not an assumed unloaded zero-stroke position.
    primary=m.jnt_qposadr[env.pj];v=m.jnt_dofadr[env.pj];start=float(d.qpos[primary])
    for _ in range(round(2./m.opt.timestep)):
        d.qfrc_applied[v]=20.-10*d.qvel[v];env.step()
    assert abs(float(d.qpos[primary])-start)<.04
    assert d.qpos[m.jnt_qposadr[env._jid(bolt)]]<env.meta['turnstile_locks']['stroke_m']-.001
    assert not env.tracker.L.lock_released
    np.testing.assert_array_equal(m.jnt_range,ranges);np.testing.assert_array_equal(m.jnt_limited,limited)
    assert not np.any(d.warning.number)
    env.reset(randomize=False)
    assert env.turnstile_power is None and env.turnstile_drop_power is None
    assert env.turnstile_supply is None
    for _ in range(round(.6/m.opt.timestep)):env.step()
    assert max(abs(float(d.qpos[a])) for a in arms)<.015
    env.close()

"""External closer power remains a force input, with process-safe callbacks."""
import mujoco
import numpy as np
import pytest

from doorbench.benchmark.env import DoorEnv
from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.geometry.closer_mounts import resolve_closer_configuration


@pytest.fixture(scope='module')
def door(tmp_path_factory):
    root=tmp_path_factory.mktemp('closer-power')
    spec=next(s for s in generate_all() if s['index']==24)
    export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
    return root/'doors'/spec['id']


@pytest.mark.parametrize('release',['power','button'])
def test_native_hold_releases_from_external_power_or_real_button(door,release):
    env=DoorEnv(str(door));env.reset(randomize=False)
    row=env.meta['closer_track_holds'][0]
    leaf=env.m.jnt_qposadr[env._jid(row['leaf_joint'])]
    button=env.m.jnt_dofadr[env._jid(row['button_joint'])]
    env.d.qpos[leaf]=row['nominal_hold_angle_rad']
    resolve_closer_configuration(env.m,env.d.qpos,env.meta)
    env._with_passive(lambda:mujoco.mj_forward(env.m,env.d))
    for _ in range(round(3/env.m.opt.timestep)):env.step()
    assert abs(env.d.qpos[leaf]-np.pi/2)<np.deg2rad(.5)
    before={name:getattr(env.d,name).copy() for name in ('qpos','qvel','eq_active','ctrl')}
    ranges=env.m.jnt_range.copy();damping=env.m.dof_damping.copy()
    if release=='power':env.set_closer_power(False)
    for name,value in before.items():np.testing.assert_array_equal(getattr(env.d,name),value)
    np.testing.assert_array_equal(env.m.jnt_range,ranges)
    np.testing.assert_array_equal(env.m.dof_damping,damping)
    for _ in range(round(7/env.m.opt.timestep)):
        if release=='button':env.d.qfrc_applied[button]=8.
        env.step()
    assert abs(env.d.qpos[leaf])<np.deg2rad(1)
    assert not np.any(env.d.warning.number)
    env.close()


def test_power_inputs_are_copied_and_invalid_addresses_rejected(door):
    env=DoorEnv(str(door));name=env.meta['closer_track_holds'][0]['plunger_joint']
    power={name:False};env.set_closer_power(power);power[name]=True
    assert env.closer_power=={name:False}
    for invalid in (1,'false',{'unknown':False},{name:1}):
        with pytest.raises(ValueError):env.set_closer_power(invalid)
    env.set_closer_power(None);assert env.closer_power is None


def test_callbacks_restore_on_error_and_do_not_cross_same_shape_environments(door):
    first=DoorEnv(str(door));second=DoorEnv(str(door))
    first.reset(randomize=False);second.reset(randomize=False)
    assert first.m.nq==second.m.nq and first.m.nv==second.m.nv
    second.set_closer_power(False)
    def forward(env):mujoco.mj_forward(env.m,env.d);return env.d.qfrc_passive.copy()
    expected=second._with_passive(lambda:forward(second))
    nested=first._with_passive(lambda:second._with_passive(lambda:forward(second)))
    np.testing.assert_array_equal(nested,expected)
    expected=first._with_passive(lambda:forward(first))
    nested=first._with_passive(lambda:first._with_passive(lambda:forward(first)))
    np.testing.assert_array_equal(nested,expected)
    original=mujoco.get_mjcb_passive();calls=[]
    def existing(m,d):calls.append(m)
    try:
        mujoco.set_mjcb_passive(existing)
        def failing():forward(first);raise RuntimeError('deliberate callback failure')
        with pytest.raises(RuntimeError):first._with_passive(failing)
        assert calls and mujoco.get_mjcb_passive() is existing
        first.close();assert mujoco.get_mjcb_passive() is existing
    finally:mujoco.set_mjcb_passive(original)

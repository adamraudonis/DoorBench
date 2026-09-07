"""Native integration gates. Assets must be prepared explicitly; never downloaded by pytest."""
import json
import os
from pathlib import Path
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')
from doorbench.dexterous.environment import DexterousDoorEnv, OBSERVATION_KEYS

@pytest.fixture(scope='module')
def env():
    root=Path(os.environ.get('DOORBENCH_DEXTEROUS_OUTPUT','out/dexterous'))
    robot=root/'robot/h1-shadow.xml';door=root/'assets/doors/db0055_swing_single'
    if not robot.exists() or not (door/'door.xml').exists():
        pytest.skip('Run scripts/dexterous/setup_robot.py and generate the documented development door')
    e=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    yield e
    e.close()

def test_full_articulation_is_preserved(env):
    m=env.m
    assert len(env.actuators)==61
    assert len(env.joints)==69
    assert len(env.tactile_indices)==448*3
    assert m.nmocap==0
    assert np.all(m.actuator_forcelimited[env.actuators])
    assert not np.any(m.body_gravcomp[[i for i in range(m.nbody) if m.body(i).name.startswith('robot/')]])
    mass=sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith('robot/'))
    assert mass==pytest.approx(env.audit['mass_kg'],abs=1e-8)

def test_actor_has_no_door_or_global_pose_fields(env):
    obs=env.reset(seed=7,images=False)
    assert set(obs)==OBSERVATION_KEYS-{'rgb_left','rgb_right'}
    assert obs['joint_position'].shape==(69,)
    for j in env.joints:
        assert env.m.joint(j).name.startswith('robot/')
    before={k:v.copy() for k,v in obs.items()}
    old=env.plant.spec.get('id')
    env.plant.spec['id']='hidden-metadata-counterfactual'
    try:
        after=env.observe(images=False)
        for k in before:np.testing.assert_array_equal(before[k],after[k])
    finally:env.plant.spec['id']=old

def test_observation_arrays_do_not_alias_native_state(env):
    obs=env.reset(images=False)
    before=env.d.qpos.copy()
    obs['joint_position'][:]=42
    np.testing.assert_array_equal(before,env.d.qpos)

def test_reject_invalid_actions(env):
    for action in (np.zeros(60),np.full(61,np.nan),np.full(61,np.inf)):
        with pytest.raises(ValueError):env.denormalize(action)
    np.testing.assert_allclose(env.denormalize(np.full(61,10.)),env.high)

def test_nominal_start_faces_door(env):
    env.reset(randomize=False,images=False)
    forward=env.d.xmat[env.pelvis].reshape(3,3)[:,0]
    assert forward[1]>.99
    assert env.d.qpos[env.root_qadr+2]==pytest.approx(.98)

def test_robot_action_cannot_directly_actuate_door(env):
    env.reset(randomize=False,images=False)
    door_actuators=[i for i in range(env.m.nu) if i not in env.actuators]
    before=env.d.ctrl[door_actuators].copy()
    env.step(env.previous_action,images=False)
    np.testing.assert_array_equal(before,env.d.ctrl[door_actuators])
    assert not np.any(env.d.xfrc_applied)
    assert not np.any(env.d.qfrc_applied)
    assert abs(env.plant._door_q())<1e-4

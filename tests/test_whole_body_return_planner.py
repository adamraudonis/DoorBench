import copy
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.whole_body_return_planner import copy_attained_return_state


def fixture():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="leaf_handle"><geom size=".02"/></body>
    <body name="robot/torso_link" pos="0 0 1"><freejoint/><geom size=".1"/>
      <site name="robot/rh_palm_touch" pos="0 .2 0"/>
      <site name="robot/lh_palm_touch" pos="0 -.2 0"/>
      <body name="robot/left_ankle_link" pos="0 .1 -.8"><geom size=".03"/></body>
      <body name="robot/right_ankle_link" pos="0 -.1 -.8"><geom size=".03"/></body>
    </body></worldbody></mujoco>''')
    d = mujoco.MjData(m); mujoco.mj_kinematics(m,d)
    return m,d,dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),time_s=1.,geometry_time_s=1.,
                   body_ids=np.arange(m.nbody),body_positions_world_m=d.xpos.copy(),
                   body_rotations_world=d.xmat.copy().reshape(-1,3,3))


def test_private_planning_state_does_not_step_or_alias_active_data(monkeypatch):
    m,active,state=fixture();q=active.qpos.copy();v=active.qvel.copy()
    def forbidden(*a,**k): raise AssertionError('Planning must not advance physics or solve contacts')
    monkeypatch.setattr(mujoco,'mj_step',forbidden);monkeypatch.setattr(mujoco,'mj_forward',forbidden)
    d,identity=copy_attained_return_state(m,**state)
    assert d is not active and len(identity)==64
    d.qpos[0]=2.;state['qpos'][1]=3.;state['body_positions_world_m'][:]=4.
    np.testing.assert_array_equal(active.qpos,q);np.testing.assert_array_equal(active.qvel,v)
    assert d.qpos[1]==0. and active.time==0.


@pytest.mark.parametrize('field', ['qpos','qvel','body_positions_world_m','body_rotations_world'])
def test_nonfinite_actual_state_rejected(field):
    m,_,state=fixture();state[field].flat[0]=np.nan
    with pytest.raises(ValueError,match='finite'):copy_attained_return_state(m,**state)


def test_one_step_stale_pose_and_kinematic_disagreement_rejected():
    m,_,state=fixture();state['geometry_time_s']-=.002
    with pytest.raises(ValueError,match='same-epoch'):copy_attained_return_state(m,**state)
    state['geometry_time_s']=state['time_s'];state['body_positions_world_m'][-1,0]+=.001
    with pytest.raises(ValueError,match='disagree'):copy_attained_return_state(m,**state)


def test_observations_cannot_omit_or_duplicate_required_body():
    m,_,state=fixture();state['body_ids'][-1]=state['body_ids'][-2]
    with pytest.raises(ValueError,match='unique'):copy_attained_return_state(m,**state)
    m,_,state=fixture()
    for field in ('body_ids','body_positions_world_m','body_rotations_world'):state[field]=state[field][:-1]
    with pytest.raises(ValueError,match='observations'):copy_attained_return_state(m,**state)


def test_identity_tracks_actual_velocity_and_rotation_is_verified():
    m,_,state=fixture();_,a=copy_attained_return_state(m,**state)
    changed=copy.deepcopy(state);changed['qvel'][0]=.01;_,b=copy_attained_return_state(m,**changed)
    assert a!=b
    state['body_rotations_world'][-1,0,0]=.999
    with pytest.raises(ValueError,match='disagree'):copy_attained_return_state(m,**state)

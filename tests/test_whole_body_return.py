from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from doorbench.dexterous.whole_body_return import WholeBodyLeverReturn, apply_stance_goal


def test_body_goal_is_continuous_and_rotation_is_proper():
    c=WholeBodyLeverReturn.__new__(WholeBodyLeverReturn)
    c.started=10.;c.return_seconds=4.;c.progress=np.array([0.,1.])
    q=Rotation.from_rotvec([.02,.04,-.01]).as_quat()
    c.roots=np.array([[0,0,.87,1,0,0,0],[.008,-.014,.885,q[3],*q[:3]]])
    c.body_names=['torso'];c.joint_targets=np.array([[-.14],[-.27]])
    a=c.body_goal(10.);b=c.body_goal(12.);z=c.body_goal(14.)
    np.testing.assert_array_equal(a['position'],c.roots[0,:3])
    np.testing.assert_allclose(b['position'],c.roots[:,:3].mean(0))
    np.testing.assert_allclose(z['rotation'],Rotation.from_quat(q).as_matrix())
    np.testing.assert_allclose(b['rotation'].T@b['rotation'],np.eye(3),atol=1e-14)
    assert b['joints']['torso']==-.20500000000000002


def test_stance_targets_do_not_modify_foot_references_or_physical_state():
    physical=np.arange(20.)
    m=SimpleNamespace(joint=lambda j:SimpleNamespace(name=['left_knee','right_knee'][j]))
    foot_positions=np.ones((2,3));foot_rotations=np.tile(np.eye(3),(2,1,1))
    stance=SimpleNamespace(sim=SimpleNamespace(m=m,d=SimpleNamespace(qpos=physical)),
        joints=[0,1],target_root=np.zeros(3),target_rotation=np.eye(3),joint_target=np.zeros(2),
        foot_positions=foot_positions,foot_rotations=foot_rotations)
    body=SimpleNamespace(stance=stance,stage='low stance hold',height=.87)
    goal=dict(position=np.array([.008,-.014,.885]),rotation=Rotation.from_rotvec([.02,.04,-.01]).as_matrix(),
              joints={'left_knee':1.2,'right_knee':1.3})
    apply_stance_goal(body,goal)
    np.testing.assert_array_equal(physical,np.arange(20.))
    assert stance.foot_positions is foot_positions and stance.foot_rotations is foot_rotations
    np.testing.assert_array_equal(foot_positions,np.ones((2,3)))
    np.testing.assert_array_equal(stance.target_root,goal['position'])
    np.testing.assert_array_equal(stance.joint_target,[1.2,1.3])

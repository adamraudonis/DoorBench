from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.whole_body_ungrip import WholeBodyMeasuredUngrip, interpolate_rotation


def fixture():
    c=WholeBodyMeasuredUngrip.__new__(WholeBodyMeasuredUngrip)
    c.handoff_posture_targets=False
    c.observed_time=6.;c.started=0.;c.ungrip_started=6.;c.command_bridge_seconds=.5;c.ungrip_delay=6.
    c.ungrip_times=np.array([0.,.5,4.5,11.08]);c.ungrip_body_names=['torso']
    c.ungrip_roots=np.tile([0,0,.9,1,0,0,0.],(4,1));c.ungrip_roots[-1,1]=-.045
    c.ungrip_joints=np.array([[0],[0],[.1],[.2]])
    c.ungrip_fingers=np.array([[.4],[.4],[.35],[0.]])
    c.ungrip_positions=np.array([[0,-.10,0],[0,-.10,0],[0,-.10,0],[0,-.26,0]])
    c.ungrip_rotations=np.tile(np.eye(3),(4,1,1));c.ungrip_finger_indices=[1]
    c.teacher=SimpleNamespace(names=['torso','rh_FFJ3'],positions=np.zeros((1,3)),rotations=np.eye(3)[None],path=np.zeros((1,2)),digit_forces={})
    c.bridge_position=np.array([0,-.101,0]);c.bridge_rotation=np.eye(3);c.bridge_fingers=np.array([.42])
    c.bridge_body=dict(position=np.array([.001,0,.901]),rotation=np.eye(3),joints={'torso':.01})
    c.goal_frame="attained-resting-world";c.ungrip_handle_pose=np.array([0,0,0,1,0,0,0]);c.torso_index=0;c.frozen=None;c.last_time=6.;c.ready_for_panel=False
    c.release_path_start=4.5;c.digit_forces={'ff':2.,'th':6.}
    c._current=lambda t:(np.array([0,0,0,1,0,0,0]),None,dict(operator=0.,leaf=.1),None)
    return c


def test_command_bridge_preserves_first_desired_commands_and_then_attains_path():
    c=fixture();c.update(6.)
    np.testing.assert_array_equal(c.teacher.positions[-1],c.bridge_position)
    np.testing.assert_array_equal(c.teacher.path[-1],[.01,.42])
    np.testing.assert_array_equal(c.body_goal(6.)['position'],c.bridge_body['position'])
    c.update(6.5)
    np.testing.assert_array_equal(c.teacher.positions[-1],c.ungrip_positions[0])
    np.testing.assert_array_equal(c.teacher.path[-1],[0.,.4])


def test_panel_readiness_waits_for_whole_route_even_after_grip_unloads():
    c=fixture();c.update(12.)
    assert c.teacher.digit_forces=={'ff':0.,'th':0.}
    assert c.ready_for_panel is False
    c.update(17.579999)
    assert c.ready_for_panel is False
    c.update(17.582)
    assert c.ready_for_panel is True and c.info['release_fraction']==1.


def test_quaternion_sign_equivalence_does_not_create_root_spin():
    c=fixture();c.ungrip_roots[1,3]=-1.
    np.testing.assert_allclose(c._sample(.25)['rotation'],np.eye(3),atol=1e-15)
    target=Rotation.from_rotvec([.1,-.05,.02]).as_matrix()
    mid=interpolate_rotation(np.eye(3),target,.5)
    np.testing.assert_allclose(mid.T@mid,np.eye(3),atol=1e-15)
    np.testing.assert_allclose(mid@mid,target,atol=1e-15)


def test_observation_is_owned_finite_and_monotonic():
    c=fixture();c.state_time=None
    root=np.r_[0,0,.9,1,np.zeros(9)];joints={'torso':0.,'rh_FFJ3':.4}
    c.observe_state(6.,root,joints);root[0]=999;joints['torso']=999
    assert c.current_state[0][0]==0 and c.current_state[1]['torso']==0
    with pytest.raises(ValueError):c.observe_state(5.,root,joints)
    root[0]=np.nan
    with pytest.raises(ValueError):c.observe_state(7.,root,joints)


def test_exact_attained_state_guard_rejects_other_scene_before_target_change():
    c=fixture();c.ungrip_started=None;c.state_time=6.
    c.current_state=(np.r_[0,0,.9,1,np.zeros(9)],{'torso':0.,'rh_FFJ3':.4})
    c.ungrip_plan={'initial_root':[0,.01,.9,1,0,0,0]}
    old=c.teacher.path.copy()
    with pytest.raises(ValueError,match='physically attained root'):c._begin_ungrip(6.)
    np.testing.assert_array_equal(c.teacher.path,old)
    c.state_time=5.998
    with pytest.raises(ValueError,match='coherent current'):c._begin_ungrip(6.)


def test_attained_resting_goal_does_not_follow_an_uncontrolled_leaf_drift():
    c=fixture();c._current=lambda t:(np.array([.1,.2,.3,1,0,0,0]),None,dict(operator=0.,leaf=.5),None)
    c.update(11.)
    np.testing.assert_array_equal(c.teacher.positions[-1],c.ungrip_positions[2])
    assert c.info['goal_frame']=='attained-resting-world'


def test_declared_endpoint_cannot_weaken_margin_or_accept_nonfinite_evidence():
    from doorbench.dexterous.whole_body_ungrip import validate_endpoint_clearance
    good=dict(required_clearance_m=.04,minimum_all_rh_environment_clearance_m=.044,
              passed=True,hand_shapes=308,environment_shapes=64)
    validate_endpoint_clearance(good)
    for change in ({'minimum_all_rh_environment_clearance_m':.039},
                   {'required_clearance_m':.02},{'passed':False},
                   {'minimum_all_rh_environment_clearance_m':float('nan')},
                   {'hand_shapes':0}):
        with pytest.raises(ValueError):validate_endpoint_clearance({**good,**change})


def test_cleared_release_preserves_panel_posture_between_slower_ik_updates():
    c=fixture();c.handoff_posture_targets=True
    c.update(17.582)
    assert c.ready_for_panel is True
    c.frozen=(c.teacher.positions[-1].copy(),c.teacher.rotations[-1].copy())
    # Panel IK updates at 100 Hz; release is still called at 500 Hz.
    for block in range(3):
        desired=np.array([-.3-.1*block,.12+.01*block])
        c.teacher.path[-1]=desired
        for tick in range(5):
            c.update(17.584+.01*block+.002*tick)
            np.testing.assert_array_equal(c.teacher.path[-1],desired)
            assert c.info['posture_target_owner']=='panel'
    np.testing.assert_array_equal(c.teacher.positions[-1],c.frozen[0])


def test_posture_handoff_does_not_start_before_cleared_complete_route():
    c=fixture();c.handoff_posture_targets=True
    c.teacher.path[-1]=[999,999];c.update(12.)
    assert np.max(c.teacher.path[-1])<1
    c.frozen=(np.zeros(3),np.eye(3))
    with pytest.raises(ValueError,match='complete cleared release'):c.update(12.002)


def test_legacy_posture_overwrite_remains_reproducible_only_without_opt_in():
    c=fixture();c.update(17.582)
    c.frozen=(c.teacher.positions[-1].copy(),c.teacher.rotations[-1].copy())
    c.teacher.path[-1]=[-.8,.12];c.update(17.584)
    np.testing.assert_array_equal(c.teacher.path[-1],[.2,0.])

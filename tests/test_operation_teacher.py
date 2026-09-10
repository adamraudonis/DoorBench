from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doorbench.dexterous.operation_teacher import DoorOperationTeacher, pose_components, reproject_grasp


GEOMETRY = dict(operator_origin=np.array([.01, .02, 0.]), operator_axis=np.array([0., -1., 0.]),
                leaf_origin=np.array([.03, 0., 0.]), leaf_axis=np.array([0., 0., 1.]))
POSE = np.array([.7, .1, 1., 1., 0., 0., 0.])


class Acquisition:
    def __init__(self):
        self.positions = np.array([[.8, .1, 1.]])
        self.rotations = np.eye(3)[None]
        self.position_integral = np.ones(3)
        self.rotation_integral = np.ones(3)
        self.palm = 0
        self.d = SimpleNamespace(site_xpos=self.positions.copy(), site_xmat=np.eye(3).reshape(1, 9))
        self.fraction = 1.

    def force(self, *args):
        return np.arange(6.), dict(path_fraction=self.fraction)


def tick(wrapper, t, valid=True, **angles):
    return wrapper.force(t, None, {}, {}, POSE, POSE,
                         dict(operator=angles.get('operator', 0.), leaf=angles.get('leaf', 0.),
                              latch=angles.get('latch', 0.)), {}, grasp_qualified=valid)


def test_qualification_resets_on_contact_loss_and_observation_gap():
    acq = Acquisition()
    wrapper = DoorOperationTeacher(acq, GEOMETRY)
    acq.fraction = .9
    for t in np.arange(0., .7, .01):
        tick(wrapper, t)
    assert wrapper.started is None
    acq.fraction = 1.
    for t in np.arange(.7, 1., .01):
        tick(wrapper, t)
    tick(wrapper, 1., False)
    for t in np.arange(1.01, 1.4, .01):
        tick(wrapper, t)
    tick(wrapper, 1.5)  # Missing observations cannot qualify a continuous hold.
    for t in np.arange(1.51, 2., .01):
        tick(wrapper, t)
    assert wrapper.started is None
    force, info = tick(wrapper, 2.01)
    assert wrapper.started == pytest.approx(2.01)
    np.testing.assert_array_equal(force, np.arange(6.))
    assert info['phase'] == 'lever_operation'
    np.testing.assert_array_equal(acq.position_integral, 0.)


def test_opening_requires_actual_release_and_keeps_goal_continuous():
    wrapper = DoorOperationTeacher(Acquisition(), GEOMETRY, press_seconds=1.)
    for t in np.arange(0., .51, .01):
        tick(wrapper, t)
    assert wrapper.started == pytest.approx(.5)
    tick(wrapper, 1.51, operator=.85, latch=.005, leaf=.006)
    assert wrapper.open_started is None
    tick(wrapper, 1.52, operator=.5, latch=.012, leaf=.006)
    assert wrapper.open_started is None
    tick(wrapper, 1.53, operator=.85, latch=.012, leaf=.006)
    assert wrapper.open_started == pytest.approx(1.53)
    assert wrapper.info['goal_leaf_rad'] == 0.  # Do not jump to soft leaf deflection.
    tick(wrapper, 1.54, valid=False, operator=.85, latch=.012, leaf=.006)
    assert wrapper.info['phase'] == 'partial_opening'  # Loads remain audited separately.
    assert 0. < wrapper.info['goal_leaf_rad'] < 1e-6


def test_measured_release_can_start_before_press_timer_without_a_goal_jump():
    wrappers=[DoorOperationTeacher(Acquisition(),GEOMETRY,press_seconds=5.,wait_for_press_completion=value) for value in (True,False)]
    for wrapper in wrappers:
        for t in np.arange(0.,.51,.01):tick(wrapper,t)
        tick(wrapper,1.,operator=.85,latch=.005)
        assert wrapper.open_started is None
        tick(wrapper,1.01,operator=.85,latch=.012)
    assert wrappers[0].open_started is None
    assert wrappers[1].open_started==pytest.approx(1.01)
    assert wrappers[1].info['goal_leaf_rad']==0.


def transformed_pose(pose, rotation, translation):
    p, r = pose_components(pose)
    q = Rotation.from_matrix(rotation @ r).as_quat()
    return np.r_[translation + rotation @ p, q[3], q[:3]]


def test_reprojection_preserves_bound_pose_and_is_world_frame_equivariant():
    relative = np.array([.1, -.03, .02])
    rr = Rotation.from_rotvec([.1, .2, .3]).as_matrix()
    angles = dict(operator=.3, leaf=.2)
    p, r = reproject_grasp(POSE, POSE, angles, angles, relative, rr, GEOMETRY)
    np.testing.assert_allclose(p, POSE[:3] + relative, atol=1e-12)
    np.testing.assert_allclose(r, rr, atol=1e-12)
    goals = dict(operator=.8, leaf=.6)
    p, r = reproject_grasp(POSE, POSE, angles, goals, relative, rr, GEOMETRY)
    world_r = Rotation.from_rotvec([.4, -.7, .2]).as_matrix()
    world_t = np.array([-2., 1., .2])
    moved = transformed_pose(POSE, world_r, world_t)
    p2, r2 = reproject_grasp(moved, moved, angles, goals, relative, rr, GEOMETRY)
    np.testing.assert_allclose(p2, world_t + world_r @ p, atol=1e-12)
    np.testing.assert_allclose(r2, world_r @ r, atol=1e-12)


def test_measured_interface_rejects_invalid_state():
    with pytest.raises(ValueError):
        pose_components([0.] * 7)
    with pytest.raises(ValueError):
        DoorOperationTeacher(Acquisition(), {**GEOMETRY, 'leaf_axis': [0., 0., 2.]})
    wrapper = DoorOperationTeacher(Acquisition(), GEOMETRY)
    with pytest.raises(ValueError):
        tick(wrapper, 0., valid=None)
    tick(wrapper, 1.)
    with pytest.raises(ValueError):
        tick(wrapper, .9)


def test_handle_frame_recenter_is_smooth_bounded_and_never_a_pose_write():
    teacher=Acquisition()
    wrapper=DoorOperationTeacher(teacher,GEOMETRY,grasp_offset_in_handle_m=[.004,0,0])
    for t in np.arange(0.,.51,.01):tick(wrapper,t)
    original=teacher.d.site_xpos.copy()
    tick(wrapper,.501)
    assert np.linalg.norm(wrapper.info['grasp_offset_in_handle_m'])<1e-9
    tick(wrapper,1.)
    np.testing.assert_allclose(wrapper.info['grasp_offset_in_handle_m'],[.002,0,0],atol=1e-12)
    tick(wrapper,1.5)
    np.testing.assert_allclose(wrapper.info['grasp_offset_in_handle_m'],[.004,0,0])
    np.testing.assert_array_equal(teacher.d.site_xpos,original)
    for bad in ([.011,0,0],[float('nan'),0,0],[0,0]):
        with pytest.raises(ValueError,match='Grasp offset'):
            DoorOperationTeacher(Acquisition(),GEOMETRY,grasp_offset_in_handle_m=bad)


def test_compliance_is_bounded_freezes_on_release_and_preserves_actual_target():
    wrapper=DoorOperationTeacher(Acquisition(),GEOMETRY,press_seconds=.1,
        wait_for_press_completion=False,operator_compliance_gain=10.,operator_compliance_limit=.06)
    for t in np.arange(0.,.81,.01):tick(wrapper,t)
    assert wrapper.operator_compliance==pytest.approx(.06)
    assert wrapper.info['goal_handle_rad']==pytest.approx(.87)
    tick(wrapper,.81,operator=.82,latch=.012)
    frozen=wrapper.operator_compliance
    for t in np.arange(.82,1.5,.01):tick(wrapper,t,operator=.75,latch=.012)
    assert wrapper.operator_compliance==frozen
    assert wrapper.info['goal_handle_rad']==pytest.approx(.87)
    original=DoorOperationTeacher(Acquisition(),GEOMETRY)
    for t in np.arange(0.,1.,.01):tick(original,t)
    assert original.operator_compliance==0.
    with pytest.raises(ValueError):
        DoorOperationTeacher(Acquisition(),GEOMETRY,operator_compliance_gain=float('nan'))


def test_index_reference_offset_is_bounded_ramped_and_does_not_change_acquisition():
    acq=Acquisition();acq.names=['rh_FFJ3'];acq.path=np.array([[1.1]])
    wrapper=DoorOperationTeacher(acq,GEOMETRY,index_proximal_offset_rad=-.025)
    for t in np.arange(0.,.51,.01):tick(wrapper,t)
    assert acq.path[0,0]==pytest.approx(1.1)
    tick(wrapper,.51)
    assert 1.09999<acq.path[0,0]<=1.1
    tick(wrapper,1.5)
    assert acq.path[0,0]==pytest.approx(1.075)
    for invalid in (float('nan'),.1001,-.1001):
        with pytest.raises(ValueError):DoorOperationTeacher(acq,GEOMETRY,index_proximal_offset_rad=invalid)


def test_index_tendon_reference_preserves_joint_difference_and_changes_sum():
    acq=Acquisition();acq.names=['rh_FFJ1','rh_FFJ2'];acq.path=np.array([[.43,.431]])
    wrapper=DoorOperationTeacher(acq,GEOMETRY,index_tendon_offset_rad=.09)
    for t in np.arange(0.,.51,.01):tick(wrapper,t)
    np.testing.assert_array_equal(acq.path,[[.43,.431]])
    tick(wrapper,1.5)
    assert acq.path[0].sum()==pytest.approx(.951)
    assert acq.path[0,1]-acq.path[0,0]==pytest.approx(.001)
    for invalid in (float('nan'),.1201,-.1201):
        with pytest.raises(ValueError):DoorOperationTeacher(acq,GEOMETRY,index_tendon_offset_rad=invalid)


def test_operator_hold_waits_for_actual_release_and_valid_grasp():
    wrapper=DoorOperationTeacher(Acquisition(),GEOMETRY,press_seconds=1.,attained_hold_stage='operator')
    seen=[]
    class Capture:
        def force(self,t,force,joints,velocities,*,eligible):
            seen.append(eligible)
            return force,{}
    wrapper.attained_hold=Capture()
    for t in np.arange(0.,.51,.01):tick(wrapper,t)
    tick(wrapper,1.51,operator=.85,latch=.005,leaf=.006)
    assert seen[-1] is False
    tick(wrapper,1.52,operator=.85,latch=.012,leaf=.006)
    assert seen[-1] is True
    tick(wrapper,1.53,valid=False,operator=.85,latch=.012,leaf=.006)
    assert seen[-1] is False


def test_opened_leaf_capture_does_not_require_holding_a_released_latch_down():
    observed={}
    class Capture:
        def __init__(self,stage):self.stage=stage
        def force(self,t,force,joints,velocities,*,eligible):
            observed[self.stage]=eligible
            return force,{}
    for stage in ('opening','aperture'):
        wrapper=DoorOperationTeacher(Acquisition(),GEOMETRY,press_seconds=1.,attained_hold_stage=stage)
        wrapper.attained_hold=Capture(stage)
        for t in np.arange(0.,.51,.01):tick(wrapper,t)
        tick(wrapper,1.52,operator=.85,latch=.012,leaf=.006)
        tick(wrapper,4.53,operator=.7,latch=.01,leaf=.08)
    assert observed=={'opening':False,'aperture':True}


@pytest.mark.parametrize('limit',[float('nan'),float('inf'),0.,.001,.031])
def test_measured_leaf_lead_rejects_invalid_bounds(limit):
    with pytest.raises(ValueError,match='Measured leaf lead'):
        DoorOperationTeacher(Acquisition(),GEOMETRY,leaf_lead_limit_rad=limit)


def test_measured_leaf_lead_waits_for_door_and_resumes_with_motion():
    wrapper=DoorOperationTeacher(Acquisition(),GEOMETRY,press_seconds=1.,leaf_lead_limit_rad=.012)
    for t in np.arange(0.,.51,.01):tick(wrapper,t)
    tick(wrapper,1.51,operator=.85,latch=.012)
    _,info=tick(wrapper,5.,operator=.85,latch=.012,leaf=.006)
    assert info['requested_leaf_goal_rad']==pytest.approx(.08)
    assert info['goal_leaf_rad']==pytest.approx(.018)
    _,info=tick(wrapper,5.002,operator=.85,latch=.012,leaf=.03)
    assert info['goal_leaf_rad']==pytest.approx(.042)
    _,info=tick(wrapper,5.004,operator=.85,latch=.012,leaf=.079)
    assert info['goal_leaf_rad']==pytest.approx(.08)
    assert info['actual_leaf_rad']==pytest.approx(.079)


def test_default_leaf_reference_preserves_original_opening_command():
    wrapper=DoorOperationTeacher(Acquisition(),GEOMETRY,press_seconds=1.)
    for t in np.arange(0.,.51,.01):tick(wrapper,t)
    tick(wrapper,1.51,operator=.85,latch=.012)
    _,info=tick(wrapper,5.,operator=.85,latch=.012,leaf=.006)
    assert info['leaf_lead_limit_rad'] is None
    assert info['goal_leaf_rad']==pytest.approx(.08)


def test_operator_follow_requires_measured_clearance_and_smoothly_releases_press():
    wrapper=DoorOperationTeacher(Acquisition(),GEOMETRY,press_seconds=1.,operator_follow_after_leaf_rad=.02)
    for t in np.arange(0.,.51,.01):tick(wrapper,t)
    tick(wrapper,1.51,operator=.85,latch=.012)
    _,info=tick(wrapper,2.,operator=.82,latch=.012,leaf=.019)
    assert info['operator_follow_started_s'] is None
    _,info=tick(wrapper,2.01,operator=.82,latch=.012,leaf=.021)
    assert info['commanded_operator_reference_rad']==pytest.approx(.87)
    _,info=tick(wrapper,2.51,operator=.7,latch=.01,leaf=.03)
    assert info['commanded_operator_reference_rad']==pytest.approx((.87+.7)/2)
    _,info=tick(wrapper,3.01,operator=.4,latch=.006,leaf=.04)
    assert info['commanded_operator_reference_rad']==pytest.approx(.4)
    assert info['operator_follow_fraction']==pytest.approx(1.)
    assert info['goal_leaf_rad']>0


@pytest.mark.parametrize('threshold',[float('nan'),float('inf'),.01,.051])
def test_operator_follow_rejects_invalid_clearance(threshold):
    with pytest.raises(ValueError,match='Operator follow'):
        DoorOperationTeacher(Acquisition(),GEOMETRY,operator_follow_after_leaf_rad=threshold)


def test_prospective_lower_opening_trigger_still_requires_bolt_clearance():
    wrappers=[DoorOperationTeacher(Acquisition(),GEOMETRY,press_seconds=1.,release_operator_threshold=threshold) for threshold in (.80,.75)]
    for wrapper in wrappers:
        for t in np.arange(0.,.51,.01):tick(wrapper,t)
        tick(wrapper,1.51,operator=.76,latch=.010)
        assert wrapper.open_started is None
        tick(wrapper,1.52,operator=.76,latch=.0112)
        assert wrapper.operator_target==.87
        assert wrapper.release_bolt_threshold==.011
    assert wrappers[0].open_started is None
    assert wrappers[1].open_started==pytest.approx(1.52)
    assert wrappers[1].info['goal_leaf_rad']==0.


def test_attained_hold_cannot_overwrite_live_hub_clearance_force():
    wrapper = DoorOperationTeacher(Acquisition(), GEOMETRY)
    for t in np.arange(0., .51, .01):
        tick(wrapper, t)

    class CapturedHold:
        def force(self, t, forces, joints, velocities, *, eligible):
            result = forces.copy()
            result[0] = 2.0  # Captured posture replaces this finger command.
            return result, {'attained_hold_started_s': .5}

    class HubFeedback:
        def force(self, forces, blend):
            result = forces.copy()
            result[0] -= .3
            return result, {'hub_avoidance_force_N': .3}

    wrapper.attained_hold = CapturedHold()
    wrapper.hub_avoidance = HubFeedback()
    forces, info = tick(wrapper, 1.6, operator=.8, latch=.012)
    assert forces[0] == pytest.approx(1.7)
    np.testing.assert_array_equal(forces[1:], np.arange(6.)[1:])
    assert info['hub_avoidance_force_N'] == .3


@pytest.mark.parametrize('clearance', [.003, .0081, float('nan'), float('inf')])
def test_hub_activation_retains_bounded_declared_clearance(clearance):
    with pytest.raises(ValueError, match='Hub clearance'):
        DoorOperationTeacher(Acquisition(), GEOMETRY, hub_clearance_m=clearance)

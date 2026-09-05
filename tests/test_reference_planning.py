from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import box
from shapely.prepared import prep
from doorbench.reference.planning import SceneNavigator, NoRoute, heading, smoothstep


def navigator(obstacle):
    nav=object.__new__(SceneNavigator)
    nav.radius=.20;nav.resolution=.08;nav.floor_bounds=(-3,-3,3,3)
    nav.obstacles=obstacle;nav.blocked=obstacle.buffer(.20);nav.prepared=prep(nav.blocked)
    return nav


def test_route_goes_around_obstacle_without_corner_cutting():
    nav=navigator(box(-.3,-.4,.3,.4))
    route=nav.route([-1,0],[1,0])
    assert len(route)>2
    np.testing.assert_allclose(route[0],[-1,0]);np.testing.assert_allclose(route[-1],[1,0])
    assert all(nav.segment_clear(a,b) for a,b in zip(route,route[1:]))


def test_enclosed_goal_is_a_search_failure_not_a_fake_route():
    nav=navigator(box(-.3,-4,.3,4))
    with pytest.raises(NoRoute):nav.route([-1,0],[1,0])


def test_heading_and_quintic_endpoints():
    assert heading([0,1])==0
    assert heading([-1,0])==pytest.approx(np.pi/2)
    assert smoothstep(0)==0 and smoothstep(1)==1
    assert smoothstep(.0001)<1e-10


def test_real_door_stance_is_on_clear_near_side():
    root=Path(__file__).resolve().parents[1]
    nav=SceneNavigator(root/'assets/doors/db0002_swing_single')
    target=nav.site('leaf_handle_grip_n')
    stance=nav.stance(target,[0,-1.5])
    assert stance.xy[1]<0 and nav.clear(stance.xy)
    route=nav.route([0,-1.5],stance.xy)
    assert all(nav.segment_clear(a,b) for a,b in zip(route,route[1:]))


def test_ordinary_handle_prefers_upright_reachable_stance():
    nav=navigator(box(-2,0,2,.1))
    target=np.array([0.,-.02,1.])
    stance=nav.stance(target,[0,-1.5])
    right=np.array([np.cos(stance.yaw),np.sin(stance.yaw)])
    sign=1 if stance.hand=='right_hand' else -1
    shoulder=np.r_[stance.xy+right*sign*.18,stance.pelvis_height+.41]
    assert stance.pelvis_height==.94
    assert .20<np.linalg.norm(target-shoulder)<.545
    assert stance.clearance>=.30


def test_unreachable_high_hardware_is_unresolved_not_fake_reach():
    nav=navigator(box(-2,0,2,.1))
    with pytest.raises(NoRoute,match='reachable stance'):
        nav.stance([0,-.02,2.6],[0,-1.5])


def test_native_path_clock_removes_recorded_pause_burst():
    from doorbench.reference.guidance import native_progress
    source={'time':np.linspace(0,10,1001),'qpos':np.zeros((1001,1)),
            'target':np.zeros((1001,3))}
    source['qpos'][990:,0]=np.linspace(0,.5,11)
    source['target'][:,0]=source['qpos'][:,0]*.4
    arc,duration=native_progress(source,0,1000)
    t=np.linspace(0,duration,10001)
    nt=np.interp(smoothstep(t/duration)*arc[-1],arc,source['time'])
    q=np.interp(nt,source['time'],source['qpos'][:,0])
    assert np.max(np.diff(q)/np.diff(t))<=.45*1.0001
    assert q[0]==0 and q[-1]==pytest.approx(.5)
    assert np.all(np.diff(nt)>=0)


def test_gait_can_turn_comfortably_with_bounded_root_transfer():
    from doorbench.reference.gait import plan_walk
    g=plan_walk([0,0],0,[[0,0],[.8,0]],waypoint_yaws=[np.pi,np.pi/2],
                fps=120,step_length=.42,stance_width=.21,blend_turns=True,
                max_step_yaw_deg=45,pelvis_acceleration_m_s2=1.5)
    acceleration=np.diff(g['pelvis_xyz'][:,:2],n=2,axis=0)*120**2
    assert np.max(np.linalg.norm(acceleration,axis=1))<=1.5*1.002
    assert g['style_metrics']['max_step_yaw_deg']<=45.00001
    assert g['foot_contact'].any(axis=1).all()


def test_body_guidance_filter_preserves_authored_contacts_and_native_motion():
    from doorbench.reference.guidance import MotionGuide,smooth_body_guidance,rest_hands
    n=61;t=np.arange(n)/60;pelvis=np.c_[.03*np.sign(np.sin(t*20)),t*.2,np.full(n,.94)]
    feet=np.zeros((n,2,3));feet[:,0,0]=-.105;feet[:,1,0]=.105;feet[:,:,2]=.055
    yaw=np.zeros(n);hands=rest_hands(pelvis,yaw,feet);hands[:,1]=[.2,0,1.]
    weights=np.zeros((n,2));weights[:,1]=1
    g=MotionGuide(t,t.copy(),np.zeros((n,1)),pelvis,yaw,feet,np.tile([1,0,0,0],(n,2,1)),
                  np.ones((n,2),bool),hands,weights.astype(bool),weights,['operate']*n,{})
    before={k:getattr(g,k).copy() for k in ['time','native_time','native_qpos','foot_pos','foot_quat','foot_contact','hand_contact']}
    active=g.hands[:,1].copy();original=g.pelvis.copy()
    smooth_body_guidance(g,60)
    for k,v in before.items():np.testing.assert_array_equal(getattr(g,k),v)
    np.testing.assert_array_equal(g.hands[:,1],active)
    assert np.max(np.abs(np.diff(g.pelvis[:,0],n=2)))<np.max(np.abs(np.diff(original[:,0],n=2)))

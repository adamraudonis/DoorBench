"""Physical invariants and rejection behavior for constrained reference IK."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
mujoco=pytest.importorskip('mujoco')
pytest.importorskip('mink')
from doorbench.reference.ik import DoorHumanoidIK,_PositionTask,_ClearanceLimit
from doorbench.reference.rig import combine_with_door


@pytest.fixture
def door(tmp_path):
    (tmp_path/'door.xml').write_text('''<mujoco model="tiny"><compiler angle="radian"/><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body name="leaf" pos="-.5 0 1"><joint name="hinge" axis="0 0 1" range="0 1.5"/>
      <geom name="leaf_geom" type="box" pos=".5 0 0" size=".5 .04 1" mass="20"/>
      <geom name="handle" type="sphere" pos=".8 -.09 0" size=".025" mass=".1"/>
      </body></worldbody></mujoco>''')
    (tmp_path/'model.json').write_text(json.dumps({'bodies':[{'name':'world','geoms':[{'name':'floor','semantic':'floor'}]},
                              {'name':'leaf','geoms':[{'name':'leaf_geom','semantic':'leaf'},{'name':'handle','semantic':'handle'}]}]}))
    return tmp_path


def stance(solver):return {name:{**pose,'contact':True} for name,pose in solver.foot_poses().items()}


def test_rig_metric_links_anatomy_and_ankle_frame(door):
    s=DoorHumanoidIK(door);p=s.joint_positions()
    assert p.shape==(16,3)
    assert p[4,0]<p[7,0] and p[10,0]<p[13,0]
    for a,b,length in [(4,5,.30),(5,6,.28),(7,8,.30),(8,9,.28),(10,11,.43),(11,12,.43),(13,14,.43),(14,15,.43)]:
        assert np.linalg.norm(p[a]-p[b])==pytest.approx(length,abs=1e-9)
    for pose in s.foot_poses().values():
        assert pose['pos'][2]==pytest.approx(.055,abs=1e-9)
        assert abs(pose['quat_wxyz'][0])==pytest.approx(1.)
    assert s.diagnostics()['min_noncontact_distance_m']>0
    assert s.diagnostics()['min_foot_ground_distance_m']==pytest.approx(0.,abs=1e-9)
    assert len(s.collision_geometries())>10
    json.dumps(s.collision_geometries(),allow_nan=False)


def test_private_attachment_preserves_native_pose_and_warm_start_cannot_move_door(door):
    before=hashlib.sha256((door/'door.xml').read_bytes()).hexdigest()
    s=DoorHumanoidIK(door,native_qpos=np.array([.2]))
    old=s.qpos.copy();s.set_door_state(np.array([.5]))
    r=s.solve(stance(s),.05,previous_q=old)
    assert r.native_qpos.tolist()==[.5]
    assert r.qpos[s.native_qpos_indices].tolist()==[.5]
    native=mujoco.MjModel.from_xml_path(str(door/'door.xml'));data=mujoco.MjData(native);data.qpos[0]=.5;mujoco.mj_kinematics(native,data)
    actual=s._fresh_data(r.qpos)
    assert np.allclose(actual.geom('leaf_geom').xpos,data.geom('leaf_geom').xpos,atol=1e-12)
    assert hashlib.sha256((door/'door.xml').read_bytes()).hexdigest()==before
    assert r.actor_qpos.shape==(len(s.actor_qpos_indices),)


def test_reachable_hand_with_six_dimensional_stance_and_total_step_velocity(door):
    s=DoorHumanoidIK(door);targets=stance(s);targets['right_hand']={'pos':[.28,-.7,1.]}
    for _ in range(8):
        r=s.solve(targets,.05)
        assert r.diagnostics['max_velocity_limit_ratio']<=1.0001
        assert r.diagnostics['joint_limit_violation_rad']<=1e-7
    assert r.success
    assert r.diagnostics['target_residuals']['right_hand']['position_m']<.01
    for name in ('left_foot','right_foot'):
        residual=r.diagnostics['target_residuals'][name]
        assert residual['position_m']<.001
        assert residual['orientation_rad']<np.deg2rad(.5)
    assert r.diagnostics['min_noncontact_distance_m']>=s.clearance-1e-5


def test_unreachable_goal_is_not_labeled_success_and_budget_not_repeated_per_iteration(door):
    s=DoorHumanoidIK(door,max_iterations=20);targets=stance(s);targets['right_hand']={'pos':[5,5,5]}
    r=s.solve(targets,.01)
    assert not r.success
    assert r.diagnostics['target_residuals']['right_hand']['position_m']>1.
    assert r.diagnostics['max_velocity_limit_ratio']<=1.0001


def test_scene_blocks_hand_target_inside_door(door):
    s=DoorHumanoidIK(door,root_pos=(0,-.7,.94));targets=stance(s);targets['right_hand']={'pos':[.25,0,1.]}
    for _ in range(12):r=s.solve(targets,.05)
    assert not r.success
    assert r.diagnostics['min_noncontact_distance_m']>=s.clearance-1e-5
    assert r.diagnostics['target_residuals']['right_hand']['position_m']>.02


def test_contact_exclusion_is_only_named_hand_and_named_geom(door):
    s=DoorHumanoidIK(door);t=stance(s);t['right_hand']={'pos':s.joint_positions()[9].tolist(),'grip_geoms':['handle']}
    r=s.solve(t,.05)
    assert r.diagnostics['allowed_grip_pairs']==[['actor_geom_hand_r','handle']]
    left=int(s.model.geom('actor_geom_hand_l').id);handle=int(s.model.geom('handle').id);leaf=int(s.model.geom('leaf_geom').id)
    assert (left,handle) in s._pairs
    assert (s._hand_geom_ids['right_hand'],leaf) in s._pairs


def test_actor_com_does_not_include_heavy_native_door(door):
    a=DoorHumanoidIK(door).diagnostics()['com']
    path=door/'door.xml';path.write_text(path.read_text().replace('mass="20"','mass="2000"'))
    b=DoorHumanoidIK(door).diagnostics()['com']
    assert np.allclose(a,b,atol=1e-12)


@pytest.mark.parametrize('floor_type',['plane','box'])
def test_exact_ground_contact_allows_lateral_reposition_and_swing(door,floor_type):
    if floor_type=='box':
        xml=door/'door.xml'
        xml.write_text(xml.read_text().replace('type="plane" size="5 5 .1"','type="box" size="5 5 .1" pos="0 0 -.1"'))
    s=DoorHumanoidIK(door);targets=stance(s)
    targets['left_foot']['pos'][0]-=.005;targets['right_foot']['pos'][0]+=.005
    r=s.solve(targets,.5)
    assert r.success # Coincident closest witnesses formerly produced a false X normal.
    targets=stance(s);targets['right_foot']['contact']=False
    targets['right_foot']['pos'][0]+=.02;targets['right_foot']['pos'][2]+=.03
    for _ in range(5):
        r=s.solve(targets,.05)
        assert r.diagnostics['min_foot_ground_distance_m']>=-1e-5
    assert r.success
    assert r.foot_poses['right_foot']['pos'][2]>.08
    # Ground contact permission does not permit a requested foot below the floor.
    targets['right_foot']['pos'][2]=.02
    for _ in range(5):r=s.solve(targets,.05)
    assert not r.success
    assert r.diagnostics['min_foot_ground_distance_m']>=-1e-5


def test_moving_obstacle_restoration_improves_each_pair_but_never_accepts_collision(door):
    initial=DoorHumanoidIK(door);hand=initial.joint_positions()[9]
    center=hand.copy();center[0]-=.055
    xml=door/'door.xml'
    xml.write_text(xml.read_text().replace('</worldbody>',
        f'<body name="moving_obstacle" pos="{center[0]+3} {center[1]} {center[2]}">'
        '<joint name="obstacle_slide" type="slide" axis="1 0 0"/>'
        '<geom name="obstacle_geom" type="sphere" size=".03" mass="1"/></body></worldbody>'))
    s=DoorHumanoidIK(door,max_iterations=1)
    assert s.diagnostics()['min_noncontact_distance_m']>=s.clearance
    s.set_door_state([0,-3]) # Native geometry has now entered the previous actor pose.
    targets=stance(s);target=hand.copy();target[0]+=.1
    targets['right_hand']={'pos':target}
    before=s._collision_violations(s.qpos)
    assert before
    for _ in range(10):
        r=s.solve(targets,.001);after=s._collision_violations(r.qpos)
        assert r.diagnostics['max_velocity_limit_ratio']<=1.0001
        assert set(after)<=set(before)
        assert all(value<=before[pair]+1e-10 for pair,value in after.items())
        assert sum(after.values())<sum(before.values())
        if after:
            assert not r.success and not r.diagnostics['kinematically_feasible']
        else:break
        before=after
    assert not after and r.diagnostics['kinematically_feasible']


def test_position_only_task_native_jacobian_and_no_artificial_wrist_objective(door):
    s=DoorHumanoidIK(door);point=s.joint_positions()[9]+[.04,.02,.03]
    tasks,_,_=s._targets({'right_hand':{'pos':point},'left_elbow':{'pos':s.joint_positions()[5]}})
    task=next(t for t in tasks if isinstance(t,_PositionTask))
    jac=task.compute_jacobian(s.configuration);start=s.qpos.copy();eps=1e-7
    numeric=np.zeros_like(jac)
    for i in range(s.model.nv):
        tangent=np.zeros(s.model.nv);tangent[i]=1
        plus=start.copy();minus=start.copy()
        mujoco.mj_integratePos(s.model,plus,tangent,eps);mujoco.mj_integratePos(s.model,minus,tangent,-eps)
        s.configuration.update(plus);a=task.compute_error(s.configuration)
        s.configuration.update(minus);b=task.compute_error(s.configuration)
        numeric[:,i]=(a-b)/(2*eps)
    s.configuration.update(start)
    assert np.allclose(jac,numeric,atol=2e-8)
    wrist_dofs=[int(s.model.joint('actor_wrist_r_'+axis).dofadr[0]) for axis in ['pitch','roll','yaw']]
    assert np.allclose(jac[:,wrist_dofs],0,atol=1e-12)
    initial_error=task.compute_error(s.configuration).copy()
    for axis in ['pitch','roll','yaw']:start[int(s.model.joint('actor_wrist_r_'+axis).qposadr[0])]=.3
    s.configuration.update(start)
    assert np.allclose(task.compute_error(s.configuration),initial_error,atol=1e-12)


def test_collision_bound_uses_displacement_units_and_restores_nominal_gap():
    import mink
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="other_slide" type="slide" axis="1 0 0"/>
      <geom name="fixed" type="sphere" size=".1"/></body>
      <body pos=".22 0 0"><joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="moving" type="sphere" size=".1" mass="1"/></body>
      </worldbody></mujoco>''')
    configuration=mink.Configuration(model)
    limit=_ClearanceLimit(model,[(['fixed'],['moving'])],gain=.85,
        minimum_distance_from_collisions=.003,collision_detection_distance=.06)
    a=limit.compute_qp_inequalities(configuration,1/60)
    b=limit.compute_qp_inequalities(configuration,.5)
    assert a.h[0]==pytest.approx(.85*(.02-.003),abs=1e-12)
    assert np.array_equal(a.G,b.G) and np.array_equal(a.h,b.h)
    # Start 10 microns below the nominal clearance. Zero displacement must no
    # longer satisfy the row: it asks for separation, within the recovery cap.
    configuration.update(np.array([0.,-.01701]))
    row=limit.compute_qp_inequalities(configuration,1/60)
    assert row.h[0]<0
    assert row.h[0]==pytest.approx(-.85*.00001,abs=1e-12)


def test_cached_broadphase_matches_exhaustive_native_distance_queries(door):
    s=DoorHumanoidIK(door)
    for angle,gripping in [(0.,False),(.5,True),(.9,False)]:
        s.set_door_state([angle])
        target={'right_hand':{'pos':s.joint_positions()[9].tolist(),
                              'grip_geoms':['handle'] if gripping else []}}
        s._targets(target)
        data=s._fresh_data()
        for pairs in (s._pairs,s._ground_pairs):
            exact=min(mujoco.mj_geomDistance(s.model,data,a,b,1e6,None) for a,b in pairs)
            assert s._minimum(data,pairs)[0]==pytest.approx(exact,abs=1e-11)


@pytest.mark.parametrize('target,match',[
    ({'missing':{'pos':[0,0,1]}},'Unknown target'),
    ({'left_foot':{'pos':[0,0,.055],'contact':True}},'explicit 6D'),
    ({'right_hand':{'pos':[0,0,1],'quat_wxyz':[0,0,0,0]}},'unit WXYZ'),
    ({'right_hand':{'pos':[0,0,1],'grip_geoms':['missing']}},'Unknown grip'),
    ({'right_hand':{'pos':[0,0,1],'grip_geoms':['floor']}},'non-floor'),
    ({'pelvis':{'pos':[0,0,1],'grip_geoms':['handle']}},'hand target'),
])
def test_reject_malformed_contacts_and_targets(door,target,match):
    with pytest.raises(ValueError,match=match):DoorHumanoidIK(door).solve(target,.05)


def test_invalid_state_rejected(door):
    s=DoorHumanoidIK(door)
    with pytest.raises(ValueError,match='native_qpos'):s.set_door_state([float('nan')])
    with pytest.raises(ValueError,match='previous_q'):s.solve({},.05,previous_q=[0])
    with pytest.raises(ValueError,match='dt'):s.solve({},0.)

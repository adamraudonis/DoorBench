"""Independent derivative checks for pose-preserving local time parameterization."""
import json
import numpy as np
import pytest

mujoco=pytest.importorskip('mujoco')
from doorbench.reference.retime import retime_trajectory


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="native_leaf"><joint name="door_hinge"/><geom size=".1" mass="1"/></body>
      <body name="actor_pelvis" pos="0 0 1"><freejoint name="actor_root"/><geom size=".1" mass="1"/>
        <body name="actor_arm" pos=".2 0 0"><joint name="actor_elbow"/><geom size=".1" mass="1"/></body>
      </body></worldbody></mujoco>''')


def states(model,n):return np.repeat(model.qpos0[None],n,axis=0)


def independent_ratios(model,q,time):
    """Recompute from returned time, using fresh native differences and world axes."""
    root=model.joint('actor_root');address=int(root.qposadr[0]);dof=int(root.dofadr[0])
    indices=np.arange(dof,model.nv);dt=np.diff(time);v=np.zeros((len(dt),model.nv))
    for i,step in enumerate(dt):mujoco.mj_differentiatePos(model,v[i],step,q[i],q[i+1])
    vmax=np.array([.8]*3+[1.5]*3+[2.5]);amax=np.array([3.]*3+[8.]*3+[15.])
    vr=np.abs(v[:,indices])/vmax
    rotation=np.empty(9)
    for i in range(len(v)):
        mujoco.mju_quat2Mat(rotation,q[i,address+3:address+7])
        v[i,dof+3:dof+6]=rotation.reshape(3,3)@v[i,dof+3:dof+6]
    ar=np.abs(np.diff(v[:,indices],axis=0))/((dt[:-1]+dt[1:])*.5)[:,None]/amax
    return vr,ar


def test_valid_motion_exact_passthrough_and_native_motion_is_not_actor_limited(model):
    time=np.linspace(2.,4.,101);q=states(model,len(time));address=int(model.joint('actor_root').qposadr[0])
    q[:,address]=.1*(time-time[0]);q[:,0]=np.arange(len(q))%2*100 # Deliberately fast native door.
    native=np.linspace(0,1,len(q),dtype=np.float32);q_copy=q.copy();time_copy=time.copy();native_copy=native.copy()
    result=retime_trajectory(model,q,time,native_time=native)
    assert result.success and result.metrics['changed_intervals']==0
    np.testing.assert_array_equal(result.time,time)
    np.testing.assert_array_equal(result.native_time,native)
    np.testing.assert_array_equal(q,q_copy);np.testing.assert_array_equal(time,time_copy);np.testing.assert_array_equal(native,native_copy)
    assert result.native_time is not native and result.native_time.dtype==native.dtype
    json.dumps(result.metrics,allow_nan=False)


def test_isolated_fast_segment_stretches_locally_with_all_derivatives_rechecked(model):
    time=np.linspace(0.,10.,1001);q=states(model,len(time));q[500,-1]=.12
    original=q.copy();native=np.maximum(0,time-1);result=retime_trajectory(model,q,time,native_time=native)
    assert result.success
    vr,ar=independent_ratios(model,q,result.time)
    assert vr.max()<=1.+1e-8 and ar.max()<=1.+1e-8
    assert result.metrics['before']['max_acceleration_ratio']>100
    assert result.metrics['duration_scale']<1.3 # Global scaling would require >10x.
    assert np.allclose(result.interval_scale[:100],1,atol=1e-12)
    assert np.allclose(result.interval_scale[-100:],1,atol=1e-12)
    assert np.max(np.maximum(result.interval_scale[1:]/result.interval_scale[:-1],
                             result.interval_scale[:-1]/result.interval_scale[1:]))<=1.15+1e-10
    np.testing.assert_array_equal(q,original);np.testing.assert_array_equal(result.native_time,native)
    again=retime_trajectory(model,q,time,native_time=native)
    np.testing.assert_array_equal(result.time,again.time)
    assert result.metrics==again.metrics


def test_acceleration_uses_world_rotation_basis_for_free_root(model):
    time=np.arange(80)/120;q=states(model,len(time));root=model.joint('actor_root');address=int(root.qposadr[0]);dof=int(root.dofadr[0])
    # Noncommuting root rotations produce changing local tangent axes. The world
    # transport must match the independent checker, including nonuniform dt.
    tangent=np.zeros(model.nv)
    for i in range(1,len(q)):
        tangent[dof+3:dof+6]=[.8*np.sin(i*.21),1.1*np.cos(i*.13),.7]
        q[i]=q[i-1];mujoco.mj_integratePos(model,q[i],tangent,.035)
    result=retime_trajectory(model,q,time,max_interval_scale=30)
    assert result.success
    vr,ar=independent_ratios(model,q,result.time)
    assert result.metrics['after']['max_velocity_ratio']==pytest.approx(vr.max(),abs=1e-11)
    assert result.metrics['after']['max_acceleration_ratio']==pytest.approx(ar.max(),abs=1e-11)
    assert vr.max()<=1.+1e-8 and ar.max()<=1.+1e-8


def test_too_small_stretch_cap_returns_explicit_failure(model):
    time=np.arange(4)*.01;q=states(model,len(time));q[1,-1]=1.
    result=retime_trajectory(model,q,time,max_interval_scale=1.1)
    assert not result.success and result.metrics['failure']
    assert result.metrics['after']['max_velocity_ratio']>1
    assert result.interval_scale.max()<=1.1
    assert np.diff(result.time).min()>0


def test_iteration_budget_failure_is_not_reported_as_feasible(model):
    q=states(model,5);q[2,-1]=.1
    result=retime_trajectory(model,q,np.arange(5)*.02,max_iterations=1)
    assert not result.success
    assert result.metrics['after']['max_acceleration_ratio']>1


def test_independent_saved_validator_rechecks_derivatives_and_retains_geometry_failures(tmp_path):
    from tests.test_validate_planned_reference import fixture,rewrite
    from scripts.validate_planned_reference import validate
    from doorbench.reference.rig import combine_with_door
    clip_path,path,assets,clip,arrays=fixture(tmp_path,offsets=[[0,0,0],[.005,0,0],[0,0,0]])
    arrays['actor_time']=np.array([0.,.001,.002]);clip['duration']=.002
    rewrite(clip_path,path,clip,arrays)
    before=validate(clip_path,path,assets)
    assert before['failure_counts']['actor_acceleration_limit_ratio']
    rig=combine_with_door(assets/'doors/fixture');q=states(rig.model,3)
    q[:,rig.actor_qpos_indices]=arrays['actor_qpos'];q[:,rig.native_qpos_indices]=arrays['qpos']
    result=retime_trajectory(rig.model,q,arrays['actor_time'],max_interval_scale=100)
    assert result.success
    arrays['actor_time']=result.time;clip['duration']=float(result.time[-1]);rewrite(clip_path,path,clip,arrays)
    after=validate(clip_path,path,assets)
    assert 'actor_acceleration_limit_ratio' not in after['failure_counts']
    assert 'actor_velocity_limit_ratio' not in after['failure_counts']
    assert not after['accepted'] # The unchanged sliding stance remains invalid.
    assert after['failure_counts']['stance_position_drift_m']


def test_two_pose_trajectory_velocity_limit_and_nonzero_time_origin(model):
    q=states(model,2);root_address=int(model.joint('actor_root').qposadr[0]);q[1,root_address]=.16
    result=retime_trajectory(model,q,[7.,7.1])
    assert result.success
    assert result.time[0]==7. and result.time[1]==pytest.approx(7.2)
    assert result.metrics['after']['max_acceleration_ratio']==0
    assert result.metrics['after']['worst_acceleration'] is None


@pytest.mark.parametrize('kwargs,match',[
    ({'max_interval_scale':.9},'max_interval_scale'),
    ({'max_iterations':0},'max_iterations'),
    ({'neighbor_ratio':1},'neighbor_ratio'),
    ({'root_joint':'missing'},'Unknown free root'),
    ({'root_joint':'actor_elbow'},'free joint'),
    ({'actor_dof_indices':[1,1]},'unique'),
    ({'actor_dof_indices':[1.5]},'integer'),
    ({'velocity_limits':[1]},'velocity_limits'),
    ({'acceleration_limits':[-1]*7},'acceleration_limits'),
    ({'native_time':[1,0,2]},'native_time'),
    ({'native_time':np.array([1,0,2],dtype=np.uint64)},'native_time'),
])
def test_reject_invalid_options(model,kwargs,match):
    with pytest.raises(ValueError,match=match):retime_trajectory(model,states(model,3),[0,.1,.2],**kwargs)


def test_reject_malformed_pose_or_time_without_normalizing_input(model):
    q=states(model,3)
    with pytest.raises(ValueError,match='time'):retime_trajectory(model,q,[0,.1,.1])
    with pytest.raises(ValueError,match='shape'):retime_trajectory(model,q[:,:-1],[0,.1,.2])
    address=int(model.joint('actor_root').qposadr[0]);q[1,address+3:address+7]=0
    with pytest.raises(ValueError,match='unit quaternions'):retime_trajectory(model,q,[0,.1,.2])
    assert np.all(q[1,address+3:address+7]==0)

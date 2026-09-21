"""Pure synthetic controller snapshots; no plant stepping or physical claims."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as S

import numpy as np
import pytest

from doorbench.dexterous.isaac_standing_reference_capture import StandingContinuationReferenceTail

ROOT=Path(__file__).resolve().parents[1]


def fixture(t=50.):
    names=['j'+str(i) for i in range(69)];force=np.linspace(-.3,.3,61)
    stance=S(target_root=np.array([.1,.2,.3]),target_rotation=np.eye(3),joint_target=np.array([.01,.02]),
        last=np.arange(10,dtype=float),last_solver_metadata={'status':'solved'})
    left=S(names=names[:8],path=[dict(position=np.array([.2,-.04,1.2]),normal=np.array([0.,-1.,0.]),
        nominal=np.arange(8,dtype=float)/100,phase='left_reach') for _ in range(2)],
        started=32.,last_update=t-.004,progress=np.float64(1.),tracking=.001,loaded_since=40.,
        target=np.arange(7,dtype=float)/100,previous=np.arange(7,dtype=float)/100,solve_names=names[1:8],
        initial_position_delta=np.array([.001,.002,.003]),waist_delta=np.float64(.01),panel_palm_rotation=np.eye(3),
        offset=.004,support_load_target=6.,filtered_palm_load=5.8,hybrid_normal_target=6.,hybrid_blend=1.,
        normal=np.array([0.,1.,0.]),surface_velocity_world=np.array([.01,0.,0.]),normal_contact_point_local=np.array([0.,0.,.01]))
    feedback=S(left=left,previous=(t,np.array([1.,2.,3.]),np.eye(3)),started=32.,maximum_target_N=6.)
    inherited=S(feedback=feedback,transfer=S(support_feedback=feedback),left=left,previous=t,started=44.,target_N=6.)
    c=S(isaac_runtime=object(),source_context=S(data=dict(runtime_path='synthetic.json',runtime_sha256='a'*64,
        motor_contract_sha256='b'*64,source_state_sha256='c'*64)),
        coupled=S(names=names,motion=S(value=np.r_[np.zeros(6),np.arange(69)/1000],velocity=np.arange(75)/10000,time=t),
            geometry=S(initial=np.r_[np.zeros(3),1.,np.zeros(3),np.zeros(69)],rq=0),accepted=10),
        acquisition=S(names=names,caps=np.tile([-2.,2.],(61,1)),last_force=force.copy(),stance=stance,
            d=S(qfrc_bias=np.arange(75)/10),m=S(nv=75)),
        started_withdrawal=t,release_started=48.,left=left,inherited_support=inherited,
        stance_names=names[:2],stance_root_bias=np.array([.001,0.,0.]),stance_rotation_bias=np.eye(3),stance_joint_bias=np.array([.002,.003]),
        arm=S(names=names[10:18],previous_target=np.arange(8)/100,previous_time=t-.01,last_call_time=t,
            reference_velocity=np.arange(8)/10000,preload=np.arange(8)/10,act=np.arange(8),va=np.arange(10,18),kp=np.ones(8)*10,kd=np.ones(8)),
        palm=S(names=names[10:17],offset=np.arange(7)/10000,last_time=t,correction_gain_s_inv=3.),
        hand=S(indices=np.arange(18),targets=np.arange(18)/100,preload=np.zeros(18),scale=5.,kp=np.ones(18)*5,kd=np.ones(18)*2),preload=np.ones(18),
        previous_finger_target=None,handoff=S(offset=np.arange(61)/100,seconds=1.,last_elapsed=6.))
    kwargs=dict(post_feedback_targets=dict(zip(names,np.arange(69)/1000+.001)),
        right_goal_position=np.array([.4,.5,.6]),right_goal_rotation=np.eye(3),returned_command=force,
        finger_reference_velocity=None,grip_preload_scale=0.,measured_root=np.r_[np.zeros(3),1.,np.zeros(9)],
        measured_joints=dict(zip(names,np.arange(69)/100)),measured_velocities=dict(zip(names,np.arange(69)/1000)),
        measured_handle_pose=np.r_[np.ones(3),1.,np.zeros(3)],measured_leaf_pose=np.r_[np.zeros(3),1.,np.zeros(3)],
        measured_angles=dict(leaf=.2,operator=.001,latch=0.))
    return c,kwargs


def advance(c,t):
    c.coupled.motion.time=t;c.coupled.motion.value[6]+=.001;c.coupled.accepted+=1
    c.inherited_support.previous=t
    c.inherited_support.feedback.previous=(t,np.array([1.,2.,3.]),np.eye(3))
    c.arm.last_call_time=t;c.palm.last_time=t


def test_complete_copied_state_preserves_epochs_and_all_three_tail_rows():
    c,args=fixture();observer=StandingContinuationReferenceTail(c,motor_names=['motor'+str(i) for i in range(61)])
    for i in range(5):
        t=50.+i*.002
        if i:advance(c,t)
        before=copy.deepcopy(c.coupled.motion.value);command=args['returned_command'].copy()
        observer.capture_command(c,t,**args)
        assert observer.completed==i and observer.pending['command_time_s']==t
        observer.observe_completed_interval(command_time_s=t,post_step_time_s=t+.002,returned_command=args['returned_command'])
        np.testing.assert_array_equal(c.coupled.motion.value,before)
        np.testing.assert_array_equal(args['returned_command'],command)
    receipt=observer.receipt()
    assert receipt['completed_reference_intervals']==5 and receipt['tail_samples']==3
    assert receipt['tail'][0]['command_time_s']==50.004
    row=receipt['tail'][-1]
    assert row['observed_input']['time_s']==row['command_time_s']==50.008
    assert row['post_step_time_s']==50.008+.002 and row['completed_interval_s']==[50.008,50.008+.002]
    assert row['right']['goal_position_world']==[.4,.5,.6]
    assert row['right']['post_feedback_joint_targets'][0]==.001
    assert row['left']['target_velocity']==dict(available=False,value=None)
    assert row['support']['previous_ik_target']==dict(available=False,value=None)
    assert not row['coupled']['acceleration_state_available'] and row['authorized_stages']==0
    assert not receipt['physical_task_qualification'] and not receipt['controller_state_restoration_supported']
    json.dumps(receipt,allow_nan=False)


def test_aliases_and_later_mutation_cannot_change_accepted_snapshot():
    c,args=fixture();observer=StandingContinuationReferenceTail(c,motor_names=['motor'+str(i) for i in range(61)]);observer.capture_command(c,50.,**args)
    observer.observe_completed_interval(command_time_s=50.,post_step_time_s=50.002,returned_command=args['returned_command'])
    expected=observer.receipt()
    c.coupled.motion.value[:]=999.;c.left.path[-1]['nominal'][:]=999.;args['right_goal_position'][:]=999.
    args['post_feedback_targets']['j0']=999.;c.arm.reference_velocity[:]=999.;c.inherited_support.feedback.previous[1][:]=999.
    args['returned_command'][:]=999.
    assert observer.receipt()==expected
    expected['tail'][0]['right']['goal_position_world'][0]=888.
    assert observer.receipt()['tail'][0]['right']['goal_position_world'][0]==.4


@pytest.mark.parametrize('kind',['native','missing_coupled','missing_binding'])
def test_constructor_is_explicit_isaac_only(kind):
    c,_=fixture()
    if kind=='native':c.isaac_runtime=None
    elif kind=='missing_coupled':c.coupled=None
    else:c.source_context.data.pop('source_state_sha256')
    with pytest.raises((ValueError,KeyError)):StandingContinuationReferenceTail(c,motor_names=['motor'+str(i) for i in range(61)])


@pytest.mark.parametrize('kind',['stale_motion','stale_support','wrong_support_object','stale_arm','missing_nominal','missing_filter',
    'bad_command','incomplete_targets','incomplete_input','nan','missing_velocity','pending','early'])
def test_missing_or_stale_state_never_invents_a_snapshot(kind):
    c,args=fixture();observer=StandingContinuationReferenceTail(c,motor_names=['motor'+str(i) for i in range(61)])
    if kind=='stale_motion':c.coupled.motion.time-=.002
    elif kind=='stale_support':c.inherited_support.previous-=.002
    elif kind=='wrong_support_object':c.inherited_support.transfer.support_feedback=object()
    elif kind=='stale_arm':c.arm.last_call_time-=.002
    elif kind=='missing_nominal':c.left.path=[]
    elif kind=='missing_filter':del c.left.filtered_palm_load
    elif kind=='bad_command':args['returned_command'][0]=3.
    elif kind=='incomplete_targets':args['post_feedback_targets'].pop('j0')
    elif kind=='incomplete_input':args['measured_velocities'].pop('j0')
    elif kind=='nan':c.palm.offset[0]=np.nan
    elif kind=='missing_velocity':del c.coupled.motion.velocity
    elif kind=='pending':observer.capture_command(c,50.,**args)
    else:c.started_withdrawal=None
    with pytest.raises((ValueError,AttributeError)):observer.capture_command(c,50.,**args)
    assert observer.completed==0 and observer.failure


@pytest.mark.parametrize('kind',['missing','early','late','different_force','dtype','signed_zero'])
def test_only_the_exact_completed_command_commits(kind):
    c,args=fixture();args['returned_command'][0]=0.;c.acquisition.last_force[0]=0.
    observer=StandingContinuationReferenceTail(c,motor_names=['motor'+str(i) for i in range(61)])
    if kind!='missing':observer.capture_command(c,50.,**args)
    end=50.002;force=args['returned_command'].copy()
    if kind=='early':end=50.
    elif kind=='late':end=50.004
    elif kind=='different_force':force[1]+=1e-12
    elif kind=='dtype':force=force.astype(np.float32)
    elif kind=='signed_zero':force[0]=-0.
    with pytest.raises(ValueError):observer.observe_completed_interval(command_time_s=50.,post_step_time_s=end,returned_command=force)
    assert observer.completed==0 and observer.failure


def test_pending_unaccepted_command_is_not_labeled_an_accepted_interval():
    c,args=fixture();observer=StandingContinuationReferenceTail(c,motor_names=['motor'+str(i) for i in range(61)]);observer.capture_command(c,50.,**args)
    receipt=observer.receipt()
    assert receipt['tail']==[] and receipt['completed_reference_intervals']==0
    assert receipt['pending_unaccepted_command']['command_time_s']==50.
    assert 'post_step_time_s' not in receipt['pending_unaccepted_command']


def test_optional_created_velocity_state_is_preserved_without_reset():
    c,args=fixture();c.left.target_velocity=np.arange(7)/100;c.inherited_support.feedback.previous_target=(49.996,np.arange(7)/10)
    observer=StandingContinuationReferenceTail(c,motor_names=['motor'+str(i) for i in range(61)]);observer.capture_command(c,50.,**args)
    row=observer.pending
    assert row['left']['target_velocity']==dict(available=True,value=c.left.target_velocity.tolist())
    assert row['support']['previous_ik_target']==dict(available=True,value=[49.996,(np.arange(7)/10).tolist()])


def test_static_path_is_copied_once_and_actual_enhanced_gains_are_explicit():
    c,args=fixture();observer=StandingContinuationReferenceTail(c,motor_names=['motor'+str(i) for i in range(61)])
    class NoIteration(list):
        def __iter__(self):raise AssertionError('Per-command full static path traversal forbidden')
    c.left.path=NoIteration(c.left.path)
    c.left.path[-1]['nominal'][0]=.123
    observer.capture_command(c,50.,**args)
    result=observer.receipt()
    assert result['static_controller_data']['left_path_at_construction'][-1]['nominal'][0]==0.
    assert result['pending_unaccepted_command']['left']['nominal_joint_target'][0]==.123
    assert 'path' not in result['pending_unaccepted_command']['left']
    assert result['attained_tracking_contract']['hand']['stiffness_scale']==5.
    assert result['attained_tracking_contract']['arm']['kp']==[10.]*8
    assert result['pending_unaccepted_command']['private_generalized_bias']==(np.arange(75)/10).tolist()


def _execute(nodes,scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<synthetic capture wiring>','exec'),scope)


def test_force_hook_is_opt_in_and_copies_the_computed_locals_only():
    tree=ast.parse((ROOT/'doorbench/dexterous/standing_withdrawal.py').read_text())
    hook=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'continuation_reference_capture' in ast.unparse(n.test))
    calls=[];observer=S(capture_command=lambda *args,**kwargs:calls.append((args,kwargs)))
    scope=dict(self=S(isaac_runtime=None,continuation_reference_capture=observer))
    # No local reference objects are required or copied in the native path.
    _execute([hook],scope);assert calls==[]
    scope['self']=S(isaac_runtime=object());_execute([hook],scope);assert calls==[]
    c,args=fixture();scope.update(self=S(isaac_runtime=object(),continuation_reference_capture=observer),t=50.,
        targets=args['post_feedback_targets'],goal=args['right_goal_position'],rotation=args['right_goal_rotation'],
        force=args['returned_command'],reference_velocity=None,scale=0.,root=args['measured_root'],joints=args['measured_joints'],
        velocities=args['measured_velocities'],handle_pose=args['measured_handle_pose'],leaf_pose=args['measured_leaf_pose'],angles=args['measured_angles'])
    _execute([hook],scope)
    assert len(calls)==1 and calls[0][0][1]==50.
    assert calls[0][1]['returned_command'] is scope['force'] and calls[0][1]['right_goal_position'] is scope['goal']


def test_producer_commit_is_poststep_and_absent_before_withdrawal():
    tree=ast.parse((ROOT/'scripts/dexterous/isaac_opening.py').read_text())
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(v,ast.Call)
        and isinstance(v.func,ast.Attribute) and v.func.attr=='observe_completed_interval' for v in ast.walk(n))
        and 'started_withdrawal' in ast.unparse(n.test))
    calls=[];scope=dict(standing_controller=S(started_withdrawal=None),standing_reference_tail=S(observe_completed_interval=lambda **kw:calls.append(kw)))
    _execute([block],scope);assert calls==[]
    scope.update(standing_controller=S(started_withdrawal=50.),step=25000,dt=.002,forces=np.zeros(61))
    _execute([block],scope)
    assert calls[0]['command_time_s']==50. and calls[0]['post_step_time_s']==50.002
    assert calls[0]['returned_command'] is scope['forces']


def test_failed_backend_submission_guard_precedes_accepting_reference_tail():
    tree=ast.parse((ROOT/'scripts/dexterous/isaac_opening.py').read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and any(isinstance(x,ast.If)
        and 'delivery_failure' in ast.unparse(x.test) and any(isinstance(v,ast.Raise) for v in x.body) for x in n.body))
    guard_index=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.If) and 'delivery_failure' in ast.unparse(n.test)
        and any(isinstance(v,ast.Raise) for v in n.body))
    commit=loop.body[guard_index+1]
    assert 'observe_completed_interval' in ast.unparse(commit)
    calls=[];scope=dict(delivery_failure='original failed motor input',standing_controller=S(started_withdrawal=50.),
        standing_reference_tail=S(observe_completed_interval=lambda **kw:calls.append(kw)),step=25000,dt=.002,forces=np.zeros(61))
    with pytest.raises(RuntimeError,match='original failed motor input'):_execute([loop.body[guard_index],commit],scope)
    assert not calls


def test_capture_helper_is_in_both_explicit_withdrawal_registries():
    for path in ('scripts/dexterous/isaac_opening.py','scripts/isaac/run_local_operation.py'):
        tree=ast.parse((ROOT/path).read_text())
        matches=[n for n in ast.walk(tree) if isinstance(n,ast.If)
            and any(isinstance(v,ast.Constant) and v.value=='isaac_standing_reference_capture.py' for v in ast.walk(n))]
        assert any('withdrawal' in ast.unparse(n.test) for n in matches)


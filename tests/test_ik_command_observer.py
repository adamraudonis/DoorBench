"""Synthetic data-only observation tests; no controller or simulation runs."""
import copy
import json

import numpy as np
import pytest

from doorbench.dexterous.ik_command_observer import SelectedWindowIkObserver, window_finite_differences


def make(windows=((0.,.004),), **kwargs):
    defaults=dict(windows=windows,joint_names=['torso','left_elbow','left_wrist'],
        motor_names=['torso','left_elbow','left_wrist'],motor_caps=[[-200.,200.],[-18.,18.],[-5.,5.]],
        component_names=['acquisition','left_palm'],
        source_binding=dict(episode_id='synthetic-not-physical',input_sha256={'fixture':'a'*64}))
    defaults.update(kwargs)
    return SelectedWindowIkObserver(**defaults)


def component(indices=(0,),names=('torso',),refresh=0.,target=(.2,),damping=(20.,),bias_velocity=(-6.,),ff=(0.,)):
    n=len(indices)
    return dict(ik_joint_names=list(names),held_ik_position=np.asarray(target),last_refresh_time_s=refresh,
        servo_motor_indices=list(indices),servo_target=np.asarray(target),servo_reference_velocity=None if not any(ff) else np.zeros(n),
        measured_position=np.zeros(n),measured_velocity=np.ones(n),kp=np.full(n,300.),
        affine_bias=np.column_stack((np.zeros(n),np.full(n,-300.),bias_velocity)),
        impedance_gain=np.full(n,9.),extra_damping=np.asarray(damping),
        reference_velocity_coefficient=np.asarray(ff),model_bias_contribution=np.zeros(n),
        other_feedforward=np.zeros(n),servo_unclipped=np.full(n,1.),
        goal_position_world=np.array([.1,.2,.3]),goal_orientation_kind='rotation-matrix',goal_orientation_world=np.eye(3))


def sample(t=0., **kwargs):
    final=np.array([1.,2.,3.])
    defaults=dict(command_time_s=t,measured_time_s=t,phase='synthetic acquisition',
        root13_actor_origin=np.r_[np.zeros(3),1.,np.zeros(9)],joint_position=np.zeros(3),joint_velocity=np.ones(3),
        components=dict(acquisition=component(),left_palm=None),
        pipeline_stages=[('acquisition_return',final.copy()),('outer_return',final.copy())],
        final_motor_command=final)
    defaults.update(kwargs)
    return defaults


def test_complete_vectors_and_distinct_coefficients_are_copied_without_pd_recalculation():
    observer=make()
    for t in [0.,.002,.004]:
        data=sample(t)
        data['components']['left_palm']=component((1,2),('left_elbow','left_wrist'),target=(1.2,.1),
            damping=(10.,.8),bias_velocity=(-2.,-.2),ff=(10.,.8))
        assert observer.capture(**data)
    receipt=observer.receipt(include_rows=True)
    assert receipt['selected_windows_complete'] and receipt['captured_records']==3
    assert receipt['rows'][0]['components']['left_palm']['held_ik_position']==[1.2,.1]
    # Supplied output deliberately need not equal fixture PD algebra. Capture
    # is a copier; a future separate source audit must check decomposition.
    assert receipt['rows'][0]['components']['acquisition']['servo_unclipped']==[1.]
    diagnostics=window_finite_differences(receipt)['rows']
    assert diagnostics[0]['measured_velocity_coefficient']==[-26.]
    assert diagnostics[0]['reference_velocity_coefficient']==[0.]
    assert diagnostics[1]['measured_velocity_coefficient']==[-12.,-1.]
    assert diagnostics[1]['reference_velocity_coefficient']==[10.,.8]
    assert not receipt['motor_delivery_measured'] and not receipt['physical_intervals_cross_bound']
    assert receipt['authorized_stages']==0 and not receipt['physical_qualification']


def test_no_argument_aliases_and_command_dtype_bytes_retained():
    observer=make(((0.,0.),));args=sample();original=copy.deepcopy(args)
    assert observer.capture(**args)
    np.testing.assert_array_equal(args['final_motor_command'],original['final_motor_command'])
    args['components']['acquisition']['held_ik_position'][0]=999.
    args['final_motor_command'][0]=999.
    args['pipeline_stages'][0][1][0]=999.
    receipt=observer.receipt(include_rows=True)
    assert receipt['rows'][0]['components']['acquisition']['held_ik_position']==[.2]
    assert receipt['rows'][0]['final_motor_command']['value']==[1.,2.,3.]
    assert receipt['rows'][0]['final_motor_command']['dtype']==np.dtype('float64').str
    receipt['rows'][0]['components']['acquisition']['held_ik_position'][0]=999.
    assert observer.receipt(include_rows=True)['rows'][0]['components']['acquisition']['held_ik_position']==[.2]


def test_selected_window_storage_bounded_and_no_difference_across_gap():
    observer=make(((.004,.006),(.02,.022)),max_records=4)
    for tick in range(12):
        args=sample(tick*.002)
        args['components']['acquisition']['held_ik_position'][:]=tick
        args['components']['acquisition']['last_refresh_time_s']=tick*.002
        selected=observer.capture(**args)
        assert selected==(tick in (2,3,10,11))
    receipt=observer.receipt(include_rows=True)
    assert receipt['captured_records']==4 and receipt['selected_windows_complete']
    d=window_finite_differences(receipt)['rows']
    assert [v['consecutive_sample_pair'] for v in d]==[False,True,False,True]
    assert d[1]['held_ik_step']==[1.] and d[1]['held_ik_interval_velocity']==[500.]
    assert d[2]['held_ik_interval_velocity'] is None


def test_inactive_component_and_coordinate_inventory_change_break_derivative():
    observer=make()
    first=sample();observer.capture(**first)
    second=sample(.002);second['components']['acquisition']=None;observer.capture(**second)
    third=sample(.004);third['components']['acquisition']=component((1,),('left_elbow',),refresh=.004)
    observer.capture(**third)
    d=window_finite_differences(observer.receipt(include_rows=True))['rows']
    assert len(d)==2 and not any(r['consecutive_sample_pair'] for r in d)


def test_actual_axis_only_lh_constraint_does_not_invent_a_full_orientation():
    observer=make(((0.,0.),));args=sample();c=args['components']['acquisition']
    c['goal_orientation_kind']='axis-z';c['goal_orientation_world']=np.array([0.,1.,0.])
    observer.capture(**args)
    row=observer.receipt(include_rows=True)['rows'][0]['components']['acquisition']
    assert row['goal_orientation_kind']=='axis-z' and row['goal_orientation_world']==[0.,1.,0.]


@pytest.mark.parametrize('targets', [(-1e308, 1e308), (0., 1e306)])
def test_finite_extreme_held_targets_reject_overflowing_subtraction_or_division(targets):
    observer=make(((0.,.002),))
    for t,target in zip((0.,.002),targets):
        args=sample(t)
        args['components']['acquisition']['held_ik_position']=np.array([target])
        observer.capture(**args)
    receipt=observer.receipt(include_rows=True)
    # Both supplied vectors are finite and the observations remain unchanged;
    # neither subtraction overflow nor 2 ms division may yield an inf report.
    before=json.dumps(receipt,sort_keys=True,allow_nan=False)
    with pytest.raises(ValueError,match='Nonfinite derived held IK difference or interval velocity'):
        window_finite_differences(receipt)
    assert json.dumps(receipt,sort_keys=True,allow_nan=False)==before
    assert observer.receipt()['selected_windows_complete']


def test_finite_extreme_pre_pd_terms_reject_overflowing_derived_damping_coefficient():
    observer=make(((0.,0.),));args=sample()
    c=args['components']['acquisition']
    c['affine_bias'][0,2]=-1e308;c['extra_damping'][0]=1e308
    observer.capture(**args)
    receipt=observer.receipt(include_rows=True)
    with pytest.raises(ValueError,match='Nonfinite derived measured velocity coefficient'):
        window_finite_differences(receipt)
    assert observer.receipt()['selected_windows_complete']


@pytest.mark.parametrize('kind',['missing_component','extra_component','partial_ik','unknown_ik',
    'duplicate_ik','incomplete_motor','duplicate_motor','future_refresh','nonfinite','bad_measured_epoch',
    'missing_terms','extra_terms','missing_reference_velocity','final_caps','final_diff','final_dtype','final_negative_zero'])
def test_malformed_or_unaccounted_record_fails_before_storage_and_stays_failed(kind):
    observer=make();a=sample();c=a['components']['acquisition']
    if kind=='missing_component':del a['components']['left_palm']
    elif kind=='extra_component':a['components']['invented']=None
    elif kind=='partial_ik':c['held_ik_position']=[]
    elif kind=='unknown_ik':c['ik_joint_names']=['not_a_robot_joint']
    elif kind=='duplicate_ik':c['ik_joint_names']=['torso','torso']
    elif kind=='incomplete_motor':c['servo_motor_indices']=[3]
    elif kind=='duplicate_motor':c['servo_motor_indices']=[0,0]
    elif kind=='future_refresh':c['last_refresh_time_s']=.002
    elif kind=='nonfinite':c['measured_velocity'][0]=np.nan
    elif kind=='bad_measured_epoch':a['measured_time_s']=.002
    elif kind=='missing_terms':del c['kp']
    elif kind=='extra_terms':c['made_up']=True
    elif kind=='missing_reference_velocity':c['reference_velocity_coefficient']=[26.]
    elif kind=='final_caps':a['final_motor_command'][1]=19.;a['pipeline_stages'][-1][1][1]=19.
    elif kind=='final_diff':a['final_motor_command'][0]=np.nextafter(1.,2.)
    elif kind=='final_dtype':a['final_motor_command']=a['final_motor_command'].astype(np.float32)
    else:a['final_motor_command'][0]=-0.;a['pipeline_stages'][-1][1][0]=0.
    with pytest.raises(ValueError):observer.capture(**a)
    assert observer.receipt()['captured_records']==0
    assert not observer.receipt()['selected_windows_complete']
    with pytest.raises(ValueError,match='after failure'):observer.capture(**sample())


@pytest.mark.parametrize('next_time',[0.,-.002,.003,.004,float('nan')])
def test_repeated_skipped_nonfinite_or_off_grid_selected_clock_rejects(next_time):
    observer=make();observer.capture(**sample())
    with pytest.raises(ValueError):observer.capture(**sample(next_time))
    assert observer.receipt()['captured_records']==1


def test_skip_window_via_unselected_call_does_not_silently_complete():
    observer=make(((.01,.012),))
    assert observer.capture(**sample()) is False
    with pytest.raises(ValueError,match='omitted'):observer.capture(**sample(.014))


def test_export_is_exclusive_and_incomplete_receipt_stays_incomplete(tmp_path):
    observer=make();observer.capture(**sample());path=tmp_path/'observation.json'
    observer.export(path);document=json.loads(path.read_text())
    assert not document['selected_windows_complete'] and document['captured_records']==1
    assert not document['rows'][0]['completed_interval_observed']
    assert document['rows'][0]['intended_interval_s']==[0.,.002]
    with pytest.raises(FileExistsError):observer.export(path)


@pytest.mark.parametrize('options',[dict(windows=[]),dict(windows=[(.003,.004)]),
    dict(windows=[(.004,.002)]),dict(windows=[(0.,.002),(.002,.004)]),dict(max_records=2),
    dict(max_records=10001),dict(joint_names=['torso','torso']),dict(motor_caps=[[0.,1.]]),
    dict(source_binding={}),dict(component_names=[])])
def test_invalid_static_boundaries_fail_before_collection(options):
    with pytest.raises(ValueError):make(**options)

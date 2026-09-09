import numpy as np
import pytest
from doorbench.dexterous.tactile_grasp_reflex import TactileGraspReflex
from doorbench.dexterous.sensor_contract import SENSOR_KEYS


def fixture(profile='four-finger-preload-v1'):
    sensors=[dict(body_name='rh_'+d+'distal',width=4,height=2,dimension=24) for d in ('ff','mf','rf','lf')]
    layout=dict(channel_order=['z','x','y'],sensors=sensors,tactile_dimension=96)
    nominal={f'rh_{d}J{j}':.2 for d in ('FF','MF','RF','LF') for j in ('1','2')}
    nominal.update(rh_THJ2=.1,torso=.3)
    c=TactileGraspReflex(layout,{n:(0.,1.) for n in nominal},profile=profile)
    packet=dict(tactile=np.zeros(96),sensor_time_s=np.zeros(len(SENSOR_KEYS)),sensor_valid=np.ones(len(SENSOR_KEYS),bool))
    return c,packet,nominal


def test_no_premature_closure_then_bounded_finger_only_preload():
    c,p,q=fixture()
    for i in range(9500):
        t=i*.002;p['sensor_time_s'][:]=t
        r=c.apply(p,t,q)
        if t<17:assert r==q
    assert r['rh_THJ2']==q['rh_THJ2'] and r['torso']==q['torso']
    for d in ('FF','MF','RF','LF'):
        assert 0<r[f'rh_{d}J1']-q[f'rh_{d}J1']<=.04000001
        assert r[f'rh_{d}J1']==r[f'rh_{d}J2']


def test_loaded_digit_does_not_receive_missing_digit_preload():
    c,p,q=fixture();p['tactile'][:8]=.1
    for i in range(9500):
        t=i*.002;p['sensor_time_s'][:]=t;r=c.apply(p,t,q)
    assert r['rh_FFJ1']==q['rh_FFJ1']
    assert r['rh_MFJ1']>q['rh_MFJ1']


def test_missing_or_stale_touch_cannot_trigger_closing():
    c,p,q=fixture();p['sensor_valid'][:]=False
    for i in range(9500):r=c.apply(p,i*.002,q)
    assert r==q
    p['sensor_valid'][:]=True
    with pytest.raises(ValueError,match='stale'):c.apply(p,19.,q)
    c.reset();p['sensor_time_s'][:]=.1
    with pytest.raises(ValueError,match='future'):c.apply(p,0.,q)


@pytest.mark.parametrize('profile,target',[('four-finger-preload-v2',1.5),('four-finger-preload-v3',.8)])
def test_higher_force_target_keeps_the_original_motion_bounds(profile,target):
    c,p,q=fixture(profile);p['tactile'][:]=.05
    for i in range(9500):
        t=i*.002;p['sensor_time_s'][:]=t;c.apply(p,t,q)
    assert c.info['target_normal_load_N']==target
    assert all(0<v<=.08 for v in c.info['coupled_motor_preload_rad'].values())
    assert c.info['maximum_coupled_slew_radps']==.08
    assert not c.info['thumb_target_changed']


@pytest.mark.parametrize('version',[1,2,3,4])
def test_checked_in_pressure_schedule_passes_runtime_admission(version):
    import json
    from pathlib import Path
    from doorbench.dexterous.sensor_acquisition_runtime import validate_parameters
    path=Path(__file__).resolve().parents[1]/f'configs/dexterous/sensor-acquisition-pressure-v{version}.json'
    validate_parameters(json.loads(path.read_text()))


def test_v4_exposes_more_position_authority_without_more_speed_or_changed_thumb():
    c,p,q=fixture('four-finger-preload-v4');previous={d:0. for d in c.offsets}
    for i in range(9500):
        t=i*.002;p['sensor_time_s'][:]=t;r=c.apply(p,t,q)
        assert all(abs(c.offsets[d]-previous[d])<=.08*.002+1e-12 for d in c.offsets)
        previous=dict(c.offsets)
    assert all(v==pytest.approx(.12) for v in c.offsets.values())
    assert r['rh_THJ2']==q['rh_THJ2'] and r['torso']==q['torso']
    assert c.info['target_normal_load_N']==.8


@pytest.mark.parametrize('version',[3,4])
def test_fixed_qp_schedule_is_explicit_and_admitted(version):
    import json
    from pathlib import Path
    from doorbench.dexterous.sensor_acquisition_runtime import validate_parameters
    path=Path(__file__).resolve().parents[1]/f'configs/dexterous/sensor-acquisition-pressure-v{version}-fixed-qp.json'
    values=json.loads(path.read_text());validate_parameters(values)
    assert values['balance_solver_profile']=='fixed-rho-interval25-v1'
    values['balance_solver_profile']='unrecorded'
    with pytest.raises(ValueError,match='balance_solver_profile'):validate_parameters(values)

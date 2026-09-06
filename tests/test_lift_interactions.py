"""Native action and task counterexamples for articulated lifts and locking stays."""
import json
import numpy as np
import pytest

from doorbench.spec import generate_all
from doorbench.build import export_door
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.runner import AutoDoorSensor,Job,run_episode
from doorbench.reference.record import Recorder


@pytest.fixture(scope='module')
def lifts(tmp_path_factory):
    root=tmp_path_factory.mktemp('lift-actions');rows={}
    for spec in generate_all():
        if spec['index'] not in (148,175,780,806,360,718):continue
        export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('json','mjcf'))
        rows[spec['index']]=(root/'doors'/spec['id'],{'id':spec['id'],'family':spec['family']})
    return rows


def run(fixture,scenario):
    path,row=fixture;rec=Recorder(10)
    result=run_episode(Job(row,str(path),scenario,0,'full','scripted_hand',randomize=False,time_budget_s=32),observer=rec)
    assert result.get('error') is None,result
    return result,rec


@pytest.mark.parametrize('index',[148,780])
def test_manual_sectional_force_is_at_bottom_grip_and_retains_passive_loads(lifts,index):
    result,rec=run(lifts[index],'open_and_traverse');assert result['success'],result
    forces=[v for f in rec.frames for v in f['site_forces'].values()]
    assert forces and max(np.linalg.norm(v) for v in forces)<=120.+1e-8
    assert {f['target_site'] for f in rec.frames if f['active']}=={'lift_handle_grip'}
    names=dict(zip(rec.info['joint_names'],rec.info['qpos_addresses']))
    # The door continues rearward overhead after top-roller height saturates.
    assert rec.frames[-1]['qpos'][names['door_rear_slide']]>1.5
    result,_=run(lifts[index],'close_only');assert result['success'],result


def test_failed_counterbalance_has_no_hidden_force_boost(lifts):
    result,rec=run(lifts[806],'open_and_traverse')
    assert not result['success'] and not result['labels']['door_open_clear']
    assert max(np.linalg.norm(v) for f in rec.frames for v in f['site_forces'].values())<=120.+1e-8


def test_rollup_uses_actual_grip_dynamics_and_native_open_initialization(lifts):
    from doorbench.benchmark.lift_state import lift_state
    path,_=lifts[718];env=DoorEnv(str(path));env.reset(randomize=False)
    state=lift_state(env.m,env.d,env.meta,True)
    assert 0<state['grip_effective_mass_kg']<state['carried_mass_kg']/2
    assert state['grip_speed_m_s']==0.
    env.close()
    for scenario in ('open_and_traverse','close_only'):
        result,rec=run(lifts[718],scenario);assert result['success'],result
        forces=[v for f in rec.frames for v in f['site_forces'].values()]
        assert forces and max(np.linalg.norm(v) for v in forces)<=120.+1e-8
        if scenario=='close_only':
            assert rec.env.initialization_evidence['ok']


def test_powered_trolley_needs_physical_button_for_open_and_close(lifts):
    path,_=lifts[175];env=DoorEnv(str(path));env.reset(randomize=False);sensor=AutoDoorSensor(env)
    for _ in range(500):
        sensor.step([0,-.3],float(env.d.time));env.step()
    assert not np.any(env.d.ctrl)
    assert abs(env._door_q())<.03
    env.close()
    for scenario in ('open_and_traverse','close_only'):
        result,rec=run(lifts[175],scenario);assert result['success'],result
        assert {f['target_site'] for f in rec.frames if f['active']}=={'activation_button_n_push'}
        assert max(abs(float(v)) for f in rec.frames for v in f['ctrl'])<=600.+1e-8


@pytest.mark.parametrize('tier',['full','simple','minimal'])
def test_activation_input_is_available_in_every_physics_tier(lifts,tier):
    env=DoorEnv(str(lifts[175][0]),tier=tier);env.reset(randomize=False)
    assert env._jid('activation_button_n_slide')>=0
    assert env.m.site('activation_button_n_push').id>=0
    sensor=AutoDoorSensor(env)
    for _ in range(100):sensor.step([0,-.3],float(env.d.time));env.step()
    assert not np.any(env.d.ctrl)
    env.close()


def test_native_recording_does_not_fit_a_human_and_preserves_every_sample(lifts,tmp_path,monkeypatch):
    from doorbench.reference import record as recording
    from doorbench.reference.native_validation import validate_native
    path,row=lifts[148];root=path.parent.parent
    manifest={'doors':[row]};(root/'manifest.json').write_text(json.dumps(manifest))
    def reject_fit(*args,**kwargs):raise AssertionError('Native recording must not generate a human')
    monkeypatch.setattr(recording,'fit_motion',reject_fit)
    result=recording.record_one((row,str(root),str(tmp_path),10,True))
    assert 'error' not in result,result
    (tmp_path/'index.json').write_text(json.dumps({'schema':recording.NATIVE_SCHEMA,'clips':[result],
        'manifest_sha256':recording.digest(root/'manifest.json')}))
    report=validate_native(tmp_path,root)
    assert report['doors']==1 and report['checks'][0]['frames_checked']==result['frames']
    clip=json.loads((tmp_path/result['clip']).read_text())
    assert 'avatar' not in clip
    assert any(c.get('force_N') for frame in clip['oracle_contacts'] for c in frame)
    # A source mismatch must fail even when the recording is numerically sound.
    index=json.loads((tmp_path/'index.json').read_text());index['manifest_sha256']='0'*64
    (tmp_path/'index.json').write_text(json.dumps(index))
    with pytest.raises(AssertionError):validate_native(tmp_path,root)


def test_hatch_opens_into_its_stay_then_requires_pin_release_to_close(lifts):
    path,_=lifts[360];spec=json.loads((path/'spec.json').read_text())
    assert [s['name'] for s in spec['benchmark']['scenarios']]==['open_only','close_only']
    assert spec['benchmark']['scenarios'][0]['expected_transit_terms']['pass_s']==0
    result,rec=run(lifts[360],'open_only');assert result['success'],result
    addr=dict(zip(rec.info['joint_names'],rec.info['qpos_addresses']))
    assert rec.frames[-1]['qpos'][addr['hatch_stay_release']]<.002
    assert not result['labels']['robot_passed_through']
    result,rec=run(lifts[360],'close_only');assert result['success'],result
    release=[f for f in rec.frames if f['active'] and f['target_site']=='stay_release_grip']
    assert release
    first_lowered=next(f for f in rec.frames if f['qpos'][addr['hatch_hinge']]<1.3)
    assert release[0]['time']<first_lowered['time']

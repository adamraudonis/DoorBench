"""Inside egress, actual grasp forces and credential/contact ordering."""
import json
import numpy as np
import pytest

from doorbench.build import build_model, export_door
from doorbench.spec import generate_all
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.scenarios import assign_scenarios, has_free_inside_trim
from doorbench.benchmark.runner import Job, build_door_info, torque_limits
from doorbench.benchmark.baselines.scripted_hand import ScriptedHandPolicy
from doorbench.benchmark.rotary_control import surface_action
from doorbench.benchmark.site_forces import SiteForces


@pytest.fixture(scope='module')
def doors(tmp_path_factory):
    root=tmp_path_factory.mktemp('rotary-runtime');out={}
    for spec in generate_all():
        if spec['index'] not in (73,86,111,230,264,861):continue
        export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('json','mjcf'))
        out[spec['index']]=root/'doors'/spec['id']
    return out


def test_every_independent_inside_trim_keeps_exterior_lock_but_allows_egress():
    checked=[]
    for spec in generate_all():
        if spec['family'] not in ('swing_single','automatic_swing','pivot'):continue
        model=build_model(spec).to_dict()
        if not has_free_inside_trim(spec,model) or not spec['lock']['engaged']:continue
        old=json.dumps(spec,sort_keys=True)
        names=assign_scenarios(spec,model)
        assert names[0] in ('open_only','open_and_traverse')
        assert 'locked_recognize' not in names and 'unlock_and_traverse' not in names
        assert json.dumps(spec,sort_keys=True)==old
        assert not any(r['released_by_default'] for r in model['meta']['rotary_locksets'])
        checked.append(spec['index'])
    assert len(checked)==27


@pytest.mark.parametrize('index',(73,111,264,861))
def test_inside_controller_uses_grips_without_unlocking_exterior(doors,index):
    path=doors[index];env=DoorEnv(str(path));env.reset(randomize=False)
    try:
        m,d=env.m,env.d;row=env.meta['rotary_locksets'][0]
        assert env.benchmark['approach_access']=='free_inside_trim'
        sc=env.scenario(env.benchmark['primary_scenario'])
        job=Job({'id':env.spec['id'],'family':env.spec['family']},str(path),sc['name'],0,'full','scripted_hand')
        limits=torque_limits(env,str(path))
        info=build_door_info(env,job,sc,10.,.004,limits,env.start_pose)
        policy=ScriptedHandPolicy();policy.reset(info,env=env)
        assert policy.rotary_free_egress and not policy.badge_needed
        assert not policy.code_keys and not policy.locks and not policy.buttons
        assert row['inside_joint'] in policy.ops and row['outside_joint'] not in policy.ops
        forces=SiteForces(env,limits);peak=0.;latch=0.
        original_ranges=m.jnt_range.copy()
        for tick in range(round(.75/m.opt.timestep)):
            if tick%round(.004/m.opt.timestep)==0:
                action=surface_action(env,[row],{'torques':{row['inside_joint']:4.}})
                assert row['inside_joint'] not in action['torques']
                values=np.array(list(action['site_forces'].values()))
                assert np.max(np.linalg.norm(values,axis=1))<=row['operator_force_cap_N']+1e-8
                if row['input_model']=='opposed_surface_pair':
                    assert len(values)==2 and np.linalg.norm(values.sum(axis=0))<1e-8
            d.qfrc_applied[:]=forces.generalized(d,action['site_forces'])
            env.step()
            peak=max(peak,abs(float(d.qpos[m.joint(row['outside_joint']).qposadr[0]])))
            latch=max(latch,float(d.qpos[m.joint(row['latch_joint']).qposadr[0]]))
        assert latch>=m.jnt_range[m.joint(row['latch_joint']).id,1]-.0005
        assert peak<.01 and not env.rotary_release_requested and not env.tracker.L.lock_released
        assert np.array_equal(m.jnt_range,original_ranges)
        assert not np.any(d.warning.number)
    finally:env.close()


def test_unavailable_badge_and_wrong_code_do_not_release_catch(doors):
    for index in (86,230):
        env=DoorEnv(str(doors[index]));env.reset(randomize=False)
        try:
            row=env.meta['rotary_locksets'][0];m,d=env.m,env.d
            assert not env.badge()  # keypad is not a card; card is unavailable
            if index==86:
                wrong='0' if env.spec['lock']['code']!='000000' else '1'
                key='leaf_keypad_key_'+wrong+'_slide'
                for press in range(6):
                    for tick in range(round(.12/m.opt.timestep)):
                        if tick*m.opt.timestep<.06:env.apply_joint_torque(key,10.)
                        env.step()
            else:
                for _ in range(round(.2/m.opt.timestep)):env.step()
            assert not env.rotary_release_requested and not env.tracker.L.credential_accepted
            assert float(d.qpos[m.joint(row['catch_joint']).qposadr[0]])<.0005
        finally:env.close()


def test_valid_code_requests_motion_before_actual_catch_release(doors):
    env=DoorEnv(str(doors[86]));env.reset(randomize=False)
    try:
        row=env.meta['rotary_locksets'][0];m,d=env.m,env.d
        original=m.jnt_range.copy();accepted_before_release=False;released=False
        for digit in env.spec['lock']['code']:
            for tick in range(round(.16/m.opt.timestep)):
                if tick*m.opt.timestep<.08:env.apply_joint_torque('leaf_keypad_key_'+digit+'_slide',10.)
                env.step()
                if env.tracker.L.credential_accepted and not env.tracker.L.lock_released:
                    accepted_before_release=True
                if env.tracker.L.lock_released:
                    assert float(d.qpos[m.joint(row['catch_joint']).qposadr[0]])>=row['released_threshold_m']
                    released=True
        for _ in range(round(.4/m.opt.timestep)):env.step()
        assert accepted_before_release and env.rotary_release_requested
        assert released or env.tracker.L.lock_released
        assert np.array_equal(m.jnt_range,original)
        assert not np.any(d.warning.number)
    finally:env.close()

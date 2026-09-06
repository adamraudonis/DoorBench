"""A real native instability must never become a successful episode."""
import pytest
import numpy as np
from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.benchmark import runner
from doorbench.benchmark.policy import Policy


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp('runner-failure')
    spec=next(s for s in generate_all() if s['index']==2)
    export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
    return root/'doors'/spec['id'],{'id':spec['id'],'family':spec['family']}


@pytest.mark.parametrize('failure',('native','mechanism','callback'))
def test_invalid_dynamics_and_failed_mechanism_stop_with_explicit_failed_results(fixture,monkeypatch,failure):
    class Invalid(Policy):
        def reset(self,info,env=None):self.env=env
        def act(self,obs):
            if failure=='callback':
                # Exercise the native callback boundary without a MjData
                # counter. This mirrors counter-free linesearch warnings.
                self.env.mj.get_mju_user_warning()('Linesearch objective is not convex')
                assert not np.any(self.env.d.warning.number)
                return {}
            if failure=='native':
                # Real native warning/reset, not a mocked warning counter.
                self.env.d.qvel[:]=1e20
                return {}
            return {'mechanism_failure':'positive_pin_failed_to_retain_curtain'}
    monkeypatch.setattr(runner,'_policy_for',lambda _:Invalid())
    path,row=fixture
    result=runner.run_episode(runner.Job(row,str(path),'open_and_traverse',0,'full','scripted_hand',
        randomize=False,time_budget_s=1.))
    assert not result.get('error'),result
    expected='native' if failure=='callback' else failure
    assert not result['success'] and result['outcome']==expected+'_failure'
    assert result['outcome'] in runner.OUTCOMES
    import json
    from pathlib import Path
    schema=json.loads((Path(__file__).parents[1]/'results/schema.json').read_text())
    assert result['outcome'] in schema['properties']['episodes']['items']['properties']['outcome']['enum']
    assert result['steps']==1
    if failure=='native':assert result['native_failure']['warnings']
    elif failure=='callback':assert result['native_failure']['messages']==['Linesearch objective is not convex']
    else:assert result['mechanism_failure']=='positive_pin_failed_to_retain_curtain'


def test_recorder_preserves_actual_native_input_even_without_a_policy_command(fixture):
    from doorbench.benchmark.env import DoorEnv
    from doorbench.reference.record import Recorder
    path,_=fixture;env=DoorEnv(str(path));env.reset(randomize=False)
    rec=Recorder(1000);base=np.array([0.,-1.5,.5])
    rec('reset',env,base,{'torque_limits':runner.torque_limits(env,str(path))})
    assert not np.any(rec.frames[0]['tau'])
    force=np.zeros(env.m.nv);force[env.m.jnt_dofadr[env.pj]]=1.23456789
    env.d.qfrc_applied[:]=force;env.step();rec('final',env,base,{})
    # The input may originate outside the policy action dictionary. Its
    # exact applied vector must survive the environment's post-step clear.
    assert not np.any(env.d.qfrc_applied)
    np.testing.assert_array_equal(rec.frames[-1]['tau'],force)
    env.reset(randomize=False)
    assert not np.any(env.last_applied_qfrc)
    env.close()

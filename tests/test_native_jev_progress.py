"""Native Jev integration contracts; no physics or live model calls."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from doorbench.dexterous.jev_advisor import AstraPlan
from doorbench.dexterous.jev_progress_advisor import ProgressAdvice
from scripts.dexterous.probe_acquisition_operation import native_progress_snapshot,NativeJevProgressGate


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/dexterous/probe_acquisition_operation.py'


def measured_row():
    return dict(sim_time_s=12.,finite=True,root_height_m=1.,torso_tilt_deg=1.,numerical_warnings=0,
        native_motor_limits=True,max_joint_limit_violation_rad=0.,max_shadow_loopback_violation_rad=0.,
        max_nonfoot_penetration_m=0.,external_wrench_max=0.,applied_generalized_force_max=0.,
        pad_grasp=dict(valid_pad_grasp=True,digit_forces_N=dict(ff=1.,mf=.4,rf=.5,lf=.6,th=3.)))


def measurements(row=None):
    return dict(episode_id='native-test',sample_id=6000,phase='lever_operation',
        physics_row=measured_row() if row is None else row,stance_status='solved',
        simulation_time_s=12.,physics_dt=.002)


def test_snapshot_uses_actual_digit_order_and_does_not_mutate_physics_evidence():
    row=measured_row();original=json.dumps(row,sort_keys=True)
    sample=native_progress_snapshot(now_s=100.,**measurements(row))
    assert sample.normal_loads_N==(1.,.4,.5,.6,3.)
    assert sample.grip_stable and sample.balance_ready and sample.local_continue_allowed
    assert sample.contact_source=='privileged_digit_handle_contact'
    assert json.dumps(row,sort_keys=True)==original
    row['pad_grasp']['digit_forces_N']['mf']=0.
    assert sample.normal_loads_N[1]==.4  # Worker input owns a copy.


@pytest.mark.parametrize('changes',[
    {'sim_time_s':11.998},{'native_motor_limits':False},{'max_joint_limit_violation_rad':.021},
    {'max_shadow_loopback_violation_rad':.021},{'max_nonfoot_penetration_m':.004},
    {'numerical_warnings':1},{'external_wrench_max':1.},{'root_height_m':.6},
])
def test_measured_physical_failures_remove_local_progress_permission(changes):
    row={**measured_row(),**changes}
    sample=native_progress_snapshot(now_s=100.,**measurements(row))
    assert not sample.local_continue_allowed


def test_missing_stance_is_unknown_and_missing_load_is_not_invented():
    args=measurements();args['stance_status']=None
    sample=native_progress_snapshot(now_s=100.,**args)
    assert sample.balance_ready is None and not sample.local_continue_allowed
    del args['physics_row']['pad_grasp']['digit_forces_N']['th']
    with pytest.raises(KeyError):native_progress_snapshot(now_s=100.,**args)


def test_gate_submits_without_waiting_and_records_exact_consumed_permission(tmp_path):
    class Advisor:
        def __init__(self):self.samples=[];self.advice=None;self.closed=False
        def poll(self,sample,plan):return self.advice
        def submit(self,sample,plan):self.samples.append(sample);return True
        def close(self):self.closed=True
    now=[100.];advisor=Advisor()
    plan=AstraPlan('astra-native-test','lever_operation','Continue the existing bounded press',contact_threshold_N=.2)
    gate=NativeJevProgressGate(plan,advisor,tmp_path,clock=lambda:now[0])
    info=dict(press_progress_s=.2,opening_progress_s=0.,progress_rate=0.)
    try:
        allow,context=gate.choose(**measurements())
        assert not allow and context['reason']=='awaiting_advice'
        assert len(advisor.samples)==1
        gate.record_submission(context,info)
        first=advisor.samples[0]
        advisor.advice=ProgressAdvice(first.episode_id,first.sample_id,plan.plan_id,first.phase,
            first.capture_monotonic_s,100.1,100.45,accepted=True,action='continue_press',
            reason='accepted_progress_advice',proposed_action='continue_press',latency_ms=100.,simulation_time_s=12.)
        now[0]=100.1
        args=measurements();args['sample_id']=6001;args['simulation_time_s']=12.002;args['physics_row']['sim_time_s']=12.002
        allow,context=gate.choose(**args)
        assert allow and context['advice_sample_id']==6000
        assert context['advice_age_simulation_s']==pytest.approx(.002)
        assert len(advisor.samples)==1  # Wall cadence, no extra call at this physics tick.
        gate.record_submission(context,{**info,'progress_rate':.01})
        advisor.advice=replace(advisor.advice,accepted=False,action='pause_press',reason='local_guard_removed_progress_permission')
        now[0]=100.3
        allow,context=gate.choose(**args)
        assert not allow
        assert len(advisor.samples)==2
        gate.record_submission(context,info)
    finally:gate.close()
    assert advisor.closed
    summary=json.loads((tmp_path/'jev-progress-summary.json').read_text())
    assert summary['continue_intervals']==1 and summary['pause_intervals']==2
    assert summary['model_replies_consumed']==1 and summary['requests_submitted']==2
    rows=[json.loads(row) for row in (tmp_path/'jev-progress.jsonl').read_text().splitlines()]
    submitted=[row for row in rows if row['event']=='controller_submission']
    assert [row['allow_progress'] for row in submitted]==[False,True,False]


def cli(tmp_path,extra=()):
    args=[sys.executable,str(SCRIPT)]
    for name in ('robot','door','reference','motors','output'):args+=['--'+name,str(tmp_path/name)]
    return args+['--validate-arguments-only',*extra]


def test_explicit_live_mode_preflight_needs_no_api_key_assets_or_physics(tmp_path):
    env={**os.environ,'PYTHONPATH':str(ROOT)};env.pop('TYPESAFE_API_KEY',None)
    args=cli(tmp_path,['--portable-wrapper','--record-transitions','--jev-progress-plan',str(tmp_path/'missing-plan'),
                       '--jev-sample-period','.3'])
    result=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==dict(arguments_valid=True,physics_started=False)
    assert not (tmp_path/'output').exists()
    for invalid in ([a for a in args if a!='--record-transitions'],
                    [a for a in args if a!='--portable-wrapper'],
                    args+['--standing-transfer-path','missing'],args+['--jev-sample-period','0']):
        rejected=subprocess.run(invalid,cwd=ROOT,env=env,capture_output=True,text=True)
        assert rejected.returncode==2


def test_default_path_does_not_import_jev_or_construct_a_transport(tmp_path):
    argv=cli(tmp_path)[1:]
    code=('import runpy,sys;sys.argv='+repr(argv)+';runpy.run_path(sys.argv[0],run_name="__main__");'
          'assert not any(n.startswith("doorbench.dexterous.jev_") for n in sys.modules)')
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,
        env={**os.environ,'PYTHONPATH':str(ROOT)},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not (tmp_path/'output').exists()


def test_native_gate_rejects_isaac_advisory_policy_before_opening_a_log(tmp_path):
    plan=AstraPlan('native-nondefault-rejected','lever_operation','Fixture only',
        progress_policy='local_guard_with_jev_advice')
    with pytest.raises(ValueError,match='Native Jev progress supports only require_jev'):
        NativeJevProgressGate(plan,None,tmp_path)
    assert not (tmp_path/'jev-progress.jsonl').exists()


def test_native_run_rejects_nondefault_policy_before_credentials_assets_or_physics(tmp_path):
    plan=AstraPlan('native-nondefault-rejected','lever_operation','Fixture only',
        progress_policy='local_guard_with_jev_advice')
    plan_path=tmp_path/'advisory-plan.json'
    plan_path.write_text(json.dumps(asdict(plan)))
    args=cli(tmp_path,['--portable-wrapper','--record-transitions','--jev-progress-plan',str(plan_path)])
    args.remove('--validate-arguments-only')
    env={**os.environ,'PYTHONPATH':str(ROOT)};env.pop('TYPESAFE_API_KEY',None)
    result=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Native Jev progress supports only require_jev' in result.stderr
    assert 'API key' not in result.stderr and not (tmp_path/'output').exists()

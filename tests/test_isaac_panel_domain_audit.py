"""Synthetic actual-source seams and detached geometry only; no robot rollout."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from test_isaac_panel_planning import actual_fixture, candidate_without_solves
from doorbench.dexterous import isaac_panel_domain_audit as domain
from doorbench.dexterous import isaac_panel_geometry_audit as static
from doorbench.dexterous import isaac_panel_planning as planner


def spec(start=.091, end=.09100001):
    return dict(schema=domain.SAMPLING_SCHEMA, reference_aperture_rad=[start,end], reference_samples=2001,
        leaf_lag_rad=[0.,0.],leaf_lag_samples=1,operator_rad=[0.,0.],operator_samples=1,
        latch_m=[0.,0.],latch_samples=1,maximum_samples=250000)


def grid(options=None, *, source=None, knots=None):
    return domain.sampling_grid(options or spec(0.1,0.5),initial_aperture=.1,final_aperture=.5,
        knot_progress=knots if knots is not None else [0.,.123456789,.7,1.],
        source_angles=source or dict(leaf=.1,operator=0.,latch=0.))


def test_grid_retains_endpoints_zero_source_and_exact_knots():
    options=spec(.1,.5)
    options.update(leaf_lag_rad=[-.01,.02],leaf_lag_samples=2,
        operator_rad=[-.005,.01],operator_samples=2,latch_m=[-.001,.001],latch_samples=2)
    actual=grid(options,source=dict(leaf=.1,operator=.003,latch=-.0002))
    assert actual['reference_aperture_rad'][0]==.1 and actual['reference_aperture_rad'][-1]==.5
    assert .123456789 in actual['progress']
    np.testing.assert_array_equal(actual['leaf_lag_rad'],[-.01,0.,.02])
    np.testing.assert_array_equal(actual['operator_rad'],[-.005,.003,.01])
    np.testing.assert_array_equal(actual['latch_m'],[-.001,-.0002,.001])
    assert actual['total_samples']==1+len(actual['progress'])*27


def test_subsegment_does_not_extend_domain_to_source():
    options=spec(.2,.3)
    actual=grid(options)
    assert actual['reference_aperture_rad'][0]==.2 and actual['reference_aperture_rad'][-1]==.3
    assert actual['source_point_samples']==1
    assert not 0. in actual['progress'] and not 1. in actual['progress']


@pytest.mark.parametrize('options',[
    {'reference_samples':2000},{'reference_samples':True},{'reference_samples':20002},
    {'reference_aperture_rad':[.099,.5]},{'reference_aperture_rad':[.1,.501]},
    {'reference_aperture_rad':[.3,.3]},{'reference_aperture_rad':[float('nan'),.5]},
    {'leaf_lag_rad':[float('inf'),.01]},{'leaf_lag_rad':[.02,-.02]},
    {'operator_rad':[.001,.01]},{'latch_m':[.001,.002]},
    {'maximum_samples':250001},{'maximum_samples':1},{'maximum_samples':True},
    {'leaf_lag_samples':True},{'leaf_lag_samples':2},{'leaf_lag_rad':[-.01,.01]},
    {'ignored_domain':True}, {'schema':'native-domain'},
])
def test_bad_or_ignored_domains_and_budget_reject(options):
    value=spec(.1,.5);value.update(options)
    with pytest.raises(ValueError):grid(value)


def test_mechanism_budget_includes_inserted_nodes():
    value=spec(.1,.5);value.update(leaf_lag_rad=[-.01,.01],leaf_lag_samples=2,
        operator_rad=[-.1,.1],operator_samples=2,latch_m=[-.001,.001],latch_samples=2,
        maximum_samples=2001*8+1)
    with pytest.raises(ValueError,match='denominator'):grid(value)


def test_sub_ulp_grid_duplicates_are_rejected_not_removed_from_denominator():
    options=spec(.1,float(np.nextafter(.1,.5)))
    with pytest.raises(ValueError,match='distinct floating-point'):grid(options)
    options=spec(.1,.5)
    options.update(operator_rad=[1.,float(np.nextafter(1.,2.))],operator_samples=3)
    with pytest.raises(ValueError,match='distinct floating-point'):
        grid(options,source=dict(leaf=.1,operator=1.,latch=0.))


@pytest.fixture
def artifacts(actual_fixture, monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch,target=.09100001)
    candidate_path=actual_fixture.tmp/'candidate.json';candidate_path.write_text(json.dumps(candidate))
    receipt,plan,_=static.audit_panel_candidate(candidate_path)
    assert receipt['passed'], 'Tiny synthetic frozen-reference aperture must pass original static geometry'
    plan_path=actual_fixture.tmp/'plan.json';audit_path=actual_fixture.tmp/'audit.json'
    plan_path.write_text(json.dumps(plan));audit_path.write_text(json.dumps(receipt))
    return SimpleNamespace(fixture=actual_fixture,candidate=candidate,receipt=receipt,plan=plan,
        candidate_path=candidate_path,plan_path=plan_path,audit_path=audit_path,
        paths=(candidate_path,plan_path,audit_path))


def test_full_entry_re_admits_source_and_samples_geometry_without_physics(artifacts,monkeypatch):
    for name in ('mj_step','mj_step1','mj_step2'):
        monkeypatch.setattr(mujoco,name,lambda *args:pytest.fail('Domain audit stepped physics'))
    count=len(artifacts.fixture.calls);progress=[]
    result=domain.audit_panel_domain(*artifacts.paths,spec(),progress=progress.append)
    assert len(artifacts.fixture.calls)==count+1
    assert result['passed'] and result['sampled_points_passed']
    assert result['attempted_samples']==result['grid']['total_samples']>=2002
    assert result['failed_samples']==result['rejected_samples']==0
    assert result['source_point']['measured_angles']==dict(leaf=.091,operator=0.,latch=0.)
    assert result['limits']==domain.ORIGINAL_LIMITS and result['input_hashes_unchanged']
    assert result['authorized_stages']==result['physics_steps']==result['active_state_writes']==0
    assert not result['continuous_domain_proved'] and not result['physical_admission']
    assert not result['runtime_route_exported'] and not result['geometric_admission']
    assert progress[-1]['attempted_samples']==result['attempted_samples']


@pytest.mark.parametrize('change',['candidate_hash','coordinates','source_epoch','limits','rate','failed',
    'embedded_receipt','missing_hash','source_context','stage_authority','duration','names','clearance','maxima'])
def test_static_plan_and_original_proof_corruption_reject(artifacts,change):
    plan=copy.deepcopy(artifacts.plan);receipt=copy.deepcopy(artifacts.receipt)
    if change=='candidate_hash':plan['candidate_sha256']='wrong'
    if change=='coordinates':plan['coordinates'][1][0]+=.001
    if change=='source_epoch':plan['initial_episode_time_s']-=.002
    if change=='limits':receipt['limits']['foot_position_m']=.001
    if change=='rate':receipt['reference_phase_envelope']['maximum_joint_speed_rad_s']=.5
    if change=='failed':receipt['passed']=False
    if change=='embedded_receipt':plan['screen_receipt']['duration_s']=41.
    if change=='missing_hash':receipt['input_sha256'].pop(next(iter(receipt['input_sha256'])))
    if change=='source_context':plan['source_context_sha256']='wrong'
    if change=='stage_authority':plan['authorized_stages']=1
    if change=='duration':plan['duration_s']=41.
    if change=='names':plan['joint_names'].reverse()
    if change=='clearance':receipt['minimum_all_rh_scene_clearance_capped_m']=.03999
    if change=='maxima':receipt['maxima']['foot_position_m']=.001
    if change!='embedded_receipt':plan['screen_receipt']=receipt
    artifacts.plan_path.write_text(json.dumps(plan));artifacts.audit_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):domain.audit_panel_domain(*artifacts.paths,spec())


def test_every_rejected_point_is_counted_with_full_failure_denominator(artifacts,monkeypatch):
    original=domain.IsaacPanelGeometryProbe.evaluate;seen=[]
    def evaluate(self,x,*,reference_aperture_rad,measured_angles):
        seen.append((reference_aperture_rad,dict(measured_angles)))
        if len(seen)>1:raise ValueError('Synthetic invalid mechanism input')
        return original(self,x,reference_aperture_rad=reference_aperture_rad,measured_angles=measured_angles)
    monkeypatch.setattr(domain.IsaacPanelGeometryProbe,'evaluate',evaluate)
    value=spec();value.update(leaf_lag_rad=[-.02,.02],leaf_lag_samples=2)
    result=domain.audit_panel_domain(*artifacts.paths,value)
    assert not result['passed'] and result['source_point']['passed']
    assert len(seen)==result['attempted_samples']==result['grid']['total_samples']
    assert len(result['failures'])==result['rejected_samples']==result['attempted_samples']-1
    lags={round(ref-measured['leaf'],10) for ref,measured in seen[1:]}
    assert lags=={-.02,0.,.02}
    assert result['failures'][-1]['sample_index']==result['attempted_samples']-1


def test_nonfinite_point_result_fails_instead_of_disappearing(artifacts,monkeypatch):
    original=domain.IsaacPanelGeometryProbe.evaluate
    def evaluate(self,*args,**kwargs):
        point=original(self,*args,**kwargs);point['values']['foot_position_m']=float('nan');return point
    monkeypatch.setattr(domain.IsaacPanelGeometryProbe,'evaluate',evaluate)
    result=domain.audit_panel_domain(*artifacts.paths,spec())
    assert not result['passed'] and result['rejected_samples']==result['attempted_samples']
    assert len(result['failures'])==result['attempted_samples']
    json.dumps(result,allow_nan=False)


def test_changed_inputs_during_sampling_never_get_pass_receipt(artifacts):
    def mutate(progress):
        if progress['attempted_samples']==1:artifacts.fixture.file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        domain.audit_panel_domain(*artifacts.paths,spec(),progress=mutate)


def test_measured_source_operator_and_latch_are_not_zeroed(actual_fixture,monkeypatch):
    scene=actual_fixture.context.scene()
    for name,value in [('leaf_handle_hinge',-.00000004),('leaf_latch_bolt_slide',.0000002)]:
        actual_fixture.admitted['initial_qpos'][scene.m.joint(name).qposadr[0]]=value
    actual_fixture.context=planner.admit_isaac_panel_context('synthetic',robot='r',door_xml='d',door_usd='u')
    candidate=candidate_without_solves(actual_fixture,monkeypatch,target=.09100001)
    p=actual_fixture.tmp/'source-coordinate-candidate.json';p.write_text(json.dumps(candidate))
    receipt,plan,_=static.audit_panel_candidate(p)
    pp=p.with_name('source-coordinate-plan.json');ap=p.with_name('source-coordinate-audit.json')
    pp.write_text(json.dumps(plan));ap.write_text(json.dumps(receipt))
    value=spec();value.update(operator_rad=[-.00000004,-.00000004],latch_m=[.0000002,.0000002])
    result=domain.audit_panel_domain(p,pp,ap,value)
    assert result['source_point']['measured_angles']['operator']==-.00000004
    assert result['source_point']['measured_angles']['latch']==.0000002
    assert result['grid']['operator_rad']==[-.00000004] and result['grid']['latch_m']==[.0000002]


def test_failed_fresh_actual_source_rejects_before_sampling(artifacts,monkeypatch):
    def reject(*args,**kwargs):raise ValueError('Actual release is not qualified')
    monkeypatch.setattr(domain,'admit_isaac_panel_context',reject)
    with pytest.raises(ValueError,match='not qualified'):
        domain.audit_panel_domain(*artifacts.paths,spec())


def test_cli_preserves_failed_receipt_and_existing_output(tmp_path,monkeypatch):
    from scripts.dexterous import audit_local_isaac_panel_domain as cli
    sampling=tmp_path/'sampling.json';sampling.write_text(json.dumps(spec()))
    output=tmp_path/'failed.json'
    def inspect(*args,**kwargs):
        return dict(passed=False,attempted_samples=2002,failed_samples=2002,rejected_samples=2002,
            authorized_stages=0,input_sha256={})
    monkeypatch.setattr(cli,'audit_panel_domain',inspect)
    args=['--candidate','synthetic-candidate','--plan','synthetic-plan','--static-audit','synthetic-audit',
        '--sampling',str(sampling),'--output',str(output)]
    assert cli.main(args)==1
    result=json.loads(output.read_text());assert result['failed_samples']==2002
    assert str(sampling.resolve()) in result['input_sha256']
    before=output.read_bytes()
    with pytest.raises(SystemExit):cli.main(args)
    assert output.read_bytes()==before


def test_cli_changed_sampling_never_emits_receipt(tmp_path,monkeypatch):
    from scripts.dexterous import audit_local_isaac_panel_domain as cli
    sampling=tmp_path/'sampling.json';sampling.write_text(json.dumps(spec()));output=tmp_path/'absent.json'
    def inspect(*args,**kwargs):sampling.write_text('{}');return {}
    monkeypatch.setattr(cli,'audit_panel_domain',inspect)
    with pytest.raises(ValueError,match='Sampling document changed'):
        cli.main(['--candidate','c','--plan','p','--static-audit','a','--sampling',str(sampling),'--output',str(output)])
    assert not output.exists()

"""Synthetic source fixtures and unstepped solver tests; never robot rollouts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous import isaac_coupled_release_planning as planner
from doorbench.dexterous import isaac_coupled_release_geometry as geometry
from doorbench.dexterous.qualified_isaac_grasp import digest
from test_isaac_coupled_release_geometry import bound,toy_scene
from test_withdrawal_source_context import isaac,load_isaac


def test_only_numeric_preferences_survive_an_old_native_document():
    old=dict(initial_qpos=[999],trajectory='old-native.npz',source_run='old native source',
        source_admission={'passed':True},coordinates=[999],configuration=dict(
            max_nfev=45,follow_handle_through_route_seconds=3.7,source_candidate='native.json',
            admitted_leaf_upper_nodes=[[0,999],[16,999]]))
    before=copy.deepcopy(old);result=planner.numeric_coupled_preferences(old)
    assert set(result)==set(planner.DEFAULT_PREFERENCES)
    assert result['max_nfev']==45 and result['follow_handle_through_route_seconds']==3.7
    assert old==before and 'coordinates' not in result and 'source_candidate' not in result


@pytest.mark.parametrize('key,value',[
    ('time_nodes',80),('time_nodes',True),('time_nodes',402),('max_nfev',0),('max_nfev',float('inf')),
    ('follow_handle_through_route_seconds',8.),('operator_margin_rad',.0001),
    ('operator_margin_rad',float('nan')),('initial_aperture_headroom_rad',-.001),
    ('final_aperture_upper_rad',.401)])
def test_invalid_numerical_preferences_cannot_change_original_domains(key,value):
    with pytest.raises(ValueError):planner.numeric_coupled_preferences({key:value})


@pytest.fixture
def source_model(bound,monkeypatch):
    _,data,config=bound;scene=toy_scene();m,d=scene.m,scene.d
    initial=d.qpos.copy();rh=m.site('robot/rh_palm_touch').id
    data['initial_qpos']=initial.tolist()
    calls=[]
    def admit(path):
        calls.append(Path(path));return SimpleNamespace(data=copy.deepcopy(data)),{str(config):digest(config)}
    monkeypatch.setattr(planner,'load_isaac_coupled_source',admit)
    monkeypatch.setattr(planner,'LandedLeftScene',lambda *args:scene)
    # This minimal named toy has no finger chain. Actual candidate row/named
    # finger validation is covered independently by the dense-audit tests.
    monkeypatch.setattr(geometry,'_path_arrays',lambda *args:(np.array([0.,.5,8.5]),np.tile(initial,(3,1)),
        np.tile(d.site_xpos[rh].copy(),(3,1)),np.tile(d.site_xmat[rh].reshape(3,3).copy(),(3,1,1))))
    return dict(data=data,config=config,scene=scene,calls=calls,initial=initial)


def test_seed_is_derived_from_new_source_without_exporting_an_unsolved_map(source_model):
    source=source_model;before=set(source['config'].parent.iterdir())
    model=planner._seed_model(source['config'],planner.numeric_coupled_preferences({}))
    assert source['calls']==[source['config']]
    np.testing.assert_array_equal(model.initial,source['initial'])
    assert model.path is None and model.plan['source_engine']=='isaac-physx'
    assert model.plan['right_hand_candidate']==source['data']['screen_path']
    assert model.plan['right_hand_dense_audit']==source['data']['audit_path']
    assert model.plan['source_physics_archive']==source['data']['source_archive_path']
    assert model.plan['admitted_leaf_upper_nodes'][0][1]==pytest.approx(.106)
    assert set(source['config'].parent.iterdir())==before
    assert model.plan['geometric_admission'] is False and model.plan['authorized_stages']==0


@pytest.mark.parametrize('time,angle',[(0.,.091),(4.,.12),(8.,.2),(16.,.4)])
def test_solver_goals_match_unchanged_evaluator_measured_frame_arithmetic(source_model,time,angle):
    model=planner._seed_model(source_model['config'],planner.numeric_coupled_preferences({}))
    _,rp,rr,lp,lr,preference=planner._reference_targets(model,time,angle)
    observed=model.evaluate(time,angle,model.initial[model.oq],model.initial[model.bq],correct=False)
    np.testing.assert_array_equal(rp,observed['right_palm_position'])
    np.testing.assert_array_equal(rr,observed['right_palm_rotation'])
    np.testing.assert_array_equal(lp,observed['left_palm_position'])
    np.testing.assert_array_equal(lr,observed['left_palm_rotation'])
    assert preference.shape==(31,) and np.isfinite(preference).all()


def test_full_generator_solves_all_nodes_but_convergence_never_promotes_route(source_model,monkeypatch):
    calls=[];progress=[]
    def fit(fun,seed,*,bounds,**kwargs):
        residual=fun(seed)
        assert residual.shape==(120,) and np.isfinite(residual).all()
        np.testing.assert_array_equal(bounds[0][:6],[-.03,-.03,-.03,-.04,-.04,-.12])
        np.testing.assert_array_equal(bounds[1][:6],[.03,.03,.012,.04,.04,.12])
        assert kwargs['max_nfev']==180
        calls.append(seed.copy())
        return SimpleNamespace(x=seed.copy(),success=False,nfev=1)
    monkeypatch.setattr(planner,'least_squares',fit)
    output=planner.generate_isaac_coupled_envelope(source_model['config'],progress=progress.append)
    assert output['solver_grid_points']==81*18 and len(calls)==81*18-1
    assert output['unconverged_solves']==len(calls)
    assert output['schema']==geometry.SCHEMA and output['initial_episode_time_s']==42.
    assert output['source_sample_playback']==output['physics_steps']==output['authorized_stages']==0
    assert not output['runtime_route_exported'] and not output['physical_admission'] and not output['geometric_admission']
    source_column=output['leaf_rad'].index(.091)
    np.testing.assert_array_equal(output['coordinates'][0][source_column],np.zeros(31))
    assert output['input_sha256'][str(Path(planner.__file__).resolve())]==digest(planner.__file__)
    assert progress[-1]['column']==18


def test_small_private_numeric_solve_runs_without_a_physics_step(source_model,monkeypatch):
    model=planner._seed_model(source_model['config'],planner.numeric_coupled_preferences({}))
    # Unit-test the numerical solver at six points. This reduced workspace is
    # never serialized; generate() still enforces all original map nodes.
    model.elapsed=np.array([0.,16.]);model.angles=np.array([.08,.091,.4])
    def forbidden(*args):pytest.fail('A physics step was called')
    import mujoco
    monkeypatch.setattr(mujoco,'mj_step',forbidden)
    coordinates,residuals,converged,evaluations=planner._solve_map(model,
        planner.numeric_coupled_preferences({'max_nfev':2}))
    assert coordinates.shape==(2,3,31) and np.isfinite(coordinates).all()
    assert residuals.shape==(2,3) and np.isfinite(residuals).all()
    assert converged[0,1] and evaluations[0,1]==0
    np.testing.assert_array_equal(coordinates[0,1],np.zeros(31))
    assert np.all(evaluations<=2)


def test_nonfinite_solve_never_emits_envelope(source_model,monkeypatch):
    monkeypatch.setattr(planner,'least_squares',lambda *args,**kwargs:
        SimpleNamespace(x=np.full(31,np.nan),success=True,nfev=1))
    with pytest.raises(ValueError,match='Nonfinite'):
        planner.generate_isaac_coupled_envelope(source_model['config'])


def test_failed_actual_source_is_rejected_before_model_or_solver(monkeypatch,tmp_path):
    def reject(*args):raise ValueError('Actual source physical qualification failed')
    monkeypatch.setattr(planner,'load_isaac_coupled_source',reject)
    monkeypatch.setattr(planner,'LandedLeftScene',lambda *args:pytest.fail('Scene created for failed source'))
    monkeypatch.setattr(planner,'least_squares',lambda *args:pytest.fail('Solver ran for failed source'))
    with pytest.raises(ValueError,match='physical qualification failed'):
        planner.generate_isaac_coupled_envelope(tmp_path/'source.json')


def test_configuration_builder_uses_actual_endpoint_and_full_new_audit(isaac):
    source=isaac
    config=planner.make_isaac_withdrawal_source_config(source=source['run'],candidate=source['screen_path'],
        dense_audit=source['audit_path'],robot=source['config']['robot_path'],
        door_xml=source['config']['door_xml_path'],door_usd=source['config']['door_usd_path'])
    assert config['source_engine']=='isaac-physx' and config['start_time_s']==42.
    assert config['duration_s']==16. and config['source_state_sha256']=='a'*64
    assert config['screen_sha256']==digest(source['screen_path']) and config['audit_sha256']==digest(source['audit_path'])
    assert config['authorized_stages']==0 and not config['runtime_route_exported']
    assert len(source['calls'])==2


def test_configuration_cannot_be_built_from_failed_dense_audit(isaac):
    document=json.loads(isaac['audit_path'].read_text());document['passed']=False
    isaac['audit_path'].write_text(json.dumps(document))
    with pytest.raises(ValueError,match='dense geometry audit'):
        planner.make_isaac_withdrawal_source_config(source=isaac['run'],candidate=isaac['screen_path'],
            dense_audit=isaac['audit_path'],robot=isaac['config']['robot_path'],
            door_xml=isaac['config']['door_xml_path'],door_usd=isaac['config']['door_usd_path'])


def test_prepare_cli_failure_retains_evidence_and_never_writes_source_config(tmp_path,monkeypatch):
    from scripts.dexterous import prepare_local_isaac_withdrawal_source as cli
    def reject(**kwargs):raise ValueError('Complete passing physical operation report required')
    monkeypatch.setattr(cli,'make_isaac_withdrawal_source_config',reject)
    args=SimpleNamespace(output=tmp_path/'source.json',isaac_source=tmp_path/'failed-source',
        candidate=tmp_path/'candidate.json',dense_audit=tmp_path/'audit.json',robot='robot.xml',
        door_xml='door.xml',door_usd='door.usda',grasp_profile='volar-phalange-v1')
    with pytest.raises(ValueError):cli.run(args)
    assert not args.output.exists()
    failure=json.loads((tmp_path/'source-failure.json').read_text())
    assert not failure['passed'] and failure['authorized_stages']==failure['physics_steps']==0
    with pytest.raises(FileExistsError):cli.run(args)


def test_map_cli_failure_retains_capture_but_cannot_export_envelope(tmp_path,monkeypatch):
    from scripts.dexterous import plan_local_isaac_coupled_release as cli
    def reject(*args,**kwargs):raise ValueError('Actual source was not qualified')
    monkeypatch.setattr(cli,'generate_isaac_coupled_envelope',reject)
    args=SimpleNamespace(output=tmp_path/'map',source_config=tmp_path/'failed-source.json',preferences=None)
    with pytest.raises(ValueError):cli.run(args)
    assert not (args.output/'envelope.json').exists()
    assert list(args.output.glob('source-*.py'))
    receipt=json.loads((args.output/'failure.json').read_text())
    assert not receipt['passed'] and receipt['physics_steps']==receipt['authorized_stages']==0
    with pytest.raises(FileExistsError):cli.run(args)

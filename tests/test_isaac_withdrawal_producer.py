"""Exercise the producer's opt-in wiring with measured synthetic CPU records."""
import ast
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_isaac_prefix_witness import source
from test_isaac_withdrawal_prefix_witness import transfer_source, witness
from test_isaac_transfer_prefix_wiring import (setup, ENTRY, RECORD, FINAL, execute,
                                               names, Tensor, MAIN, SOURCE)
from test_isaac_standing_transfer_cli import command, ROOT
from doorbench.dexterous.isaac_prefix_witness import PrefixDivergenceError
from doorbench.dexterous.isaac_evidence_cleanup import EvidenceCleanup

LEAF_RECORD = next(n for n in RECORD.body if isinstance(n, ast.If)
                   and 'leaf_sample' in names(n) and 'withdrawal_prefix' in names(n))
LEAF_CAPTURE = next(n for n in RECORD.body if isinstance(n,ast.If)
    and ast.unparse(n.test)=='record_standing_continuation' and 'leaf_sample' in names(n))


def make_scope(s, tmp_path, monkeypatch):
    _, scope, record_core, _ = setup(s, tmp_path, monkeypatch)
    combined = witness(s)
    events = []
    controller = SimpleNamespace(start_time=.006)
    controller.authorize_source_prefix = lambda receipt: events.append(('permission', receipt))
    def force(*args, **kwargs):
        events.append(('force', scope['step'], scope['withdrawal_prefix_authorized']))
        return np.zeros(61), {}
    controller.force = force
    scope.update(standing_controller=controller, withdrawal_prefix=combined,
                 transfer_prefix=None, withdrawal_prefix_authorized=False,record_standing_continuation=True,
                 pending_withdrawal_steps=None)
    scope['acquisition_states']['standing_leaf_pose'] = []
    def record(i):
        scope['record_standing_continuation']=False  # Tiny fixture's core omits separate continuation tensors.
        record_core(i)
        poses = scope['door'].data.body_state_w.value.copy()
        poses[0, 0] = s['leaf'][i]
        scope['door'].data.body_state_w = Tensor(poses)
        scope['record_standing_continuation']=True
        execute([LEAF_CAPTURE,LEAF_RECORD], scope)
    return scope, combined, events, record


def test_exact_leaf_and_core_prefix_authorize_before_first_release_force(transfer_source, tmp_path, monkeypatch):
    scope, value, events, record = make_scope(transfer_source, tmp_path, monkeypatch)
    original = scope['standing_transfer']
    for i in range(3):
        scope['step'] = i
        execute([ENTRY], scope)
        assert events[-1] == ('force', i, False)
        record(i)
    assert value.complete and not value.receipt()['passed']
    scope['step'] = 3
    execute([ENTRY], scope)
    assert events[-2][0] == 'permission' and events[-2][1]['passed']
    assert events[-1] == ('force', 3, True)
    assert scope['standing_transfer'] is original
    assert len(scope['acquisition_states']['standing_body_poses']) == 3
    assert np.asarray(scope['acquisition_states']['standing_body_poses']).shape == (3, 6, 7)
    assert np.asarray(scope['acquisition_states']['standing_leaf_pose']).shape == (3, 7)
    receipt = json.loads((tmp_path/'live-withdrawal-prefix-witness.json').read_text())
    assert receipt['core']['passed'] and receipt['leaf_pose']['passed']


def test_missing_last_leaf_blocks_motor_submission_and_exports_separate_failure(transfer_source, tmp_path, monkeypatch):
    scope, value, events, record = make_scope(transfer_source, tmp_path, monkeypatch)
    record(0); record(1)
    scope['step'] = 3
    with pytest.raises(PrefixDivergenceError): execute([ENTRY], scope)
    assert events == [] and not scope['withdrawal_prefix_authorized']
    execute(FINAL, scope)
    receipt = json.loads((tmp_path/'live-withdrawal-prefix-witness.json').read_text())
    assert receipt['leaf_pose']['intervals_verified'] == 2 and receipt['failure']
    assert not (tmp_path/'live-prefix-witness.json').exists()


def test_controller_rejection_never_submits_a_stage_motor_vector(transfer_source, tmp_path, monkeypatch):
    scope, _, events, record = make_scope(transfer_source, tmp_path, monkeypatch)
    for i in range(3): record(i)
    def reject(receipt): raise ValueError('Synthetic controller source mismatch')
    scope['standing_controller'].authorize_source_prefix = reject
    scope['step'] = 3
    with pytest.raises(ValueError, match='source mismatch'): execute([ENTRY], scope)
    assert events == [] and not scope['withdrawal_prefix_authorized']


def test_withdrawal_cli_requires_exact_standalone_prerequisites():
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    valid = ['--standing-withdrawal-route','prospective.json','--standing-transfer-prefix-source','source',
             '--native-door','door.xml','--seconds','70']
    result = subprocess.run(command(valid), cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['physics_started'] is False
    for extra in (['--sensor-layout','sensors.json'], ['--time-scale','2'], ['--panel-push'],
                  ['--whole-body-return-path','native.json'], ['--whole-body-ungrip-path','native.json']):
        result = subprocess.run(command(valid+extra), cwd=ROOT, env=env, capture_output=True, text=True)
        assert result.returncode != 0
        assert 'Standing withdrawal requires' in result.stderr
    for omit in ('--native-door', '--standing-transfer-prefix-source'):
        args = list(valid); index = args.index(omit); del args[index:index+2]
        result = subprocess.run(command(args), cwd=ROOT, env=env, capture_output=True, text=True)
        assert result.returncode != 0


def test_exception_path_exports_partial_release_stream_and_failed_report(tmp_path):
    handler = next(h for n in ast.walk(MAIN) if isinstance(n, ast.Try) for h in n.handlers
                   if isinstance(h.name, str) and h.name == 'run_error')
    branch = next(n for n in handler.body if isinstance(n, ast.If) and 'failed_withdrawal' in names(n))
    exports = []
    scope = dict(json=json, out=tmp_path, run_error=ValueError('Synthetic domain rejection'), dt=.002,
        withdrawal_steps=SimpleNamespace(exported_path=None,export=lambda path: exports.append(path.name)),
        standing_transfer=None,transfer_rest_stop=None,
        acquisition_states={'time_s':[.002,.004]},
        pad_steps=[dict(sim_time_s=.002,valid_pad_grasp=True),dict(sim_time_s=.004,valid_pad_grasp=True)],
        a=SimpleNamespace(standing_withdrawal_route='prospective.json'),
        standing_controller=SimpleNamespace(started_withdrawal=None,release_started=None,
            source_admission={'synthetic':True},info={'failed':True}))
    scope['evidence_cleanup']=EvidenceCleanup(scope['run_error'])
    execute([branch], scope)
    assert exports == ['standing-withdrawal-steps.json.gz']
    for name in ('standing-withdrawal-report.json','operation-report.json','report.json'):
        result = json.loads((tmp_path/name).read_text())
        assert result['passed'] is False and result['duration_s'] == .004
        assert result['physical_evidence_complete'] is False
        assert result['observed_final_pad_grasp_hold'] is False
        assert result['checks']=={'controller_or_backend_succeeded':False}
    scope['acquisition_states']['time_s']=[1.]
    scope['pad_steps']=[dict(sim_time_s=i*.002,valid_pad_grasp=True) for i in range(250,501)]
    execute([branch],scope)
    assert json.loads((tmp_path/'operation-report.json').read_text())['observed_final_pad_grasp_hold'] is True
    scope['pad_steps'][100]['valid_pad_grasp']=False
    execute([branch],scope)
    assert json.loads((tmp_path/'operation-report.json').read_text())['observed_final_pad_grasp_hold'] is False


@pytest.mark.parametrize('seconds', [59.998,60.,60.002])
def test_producer_rejects_unadmitted_short_or_extended_route_clock(seconds):
    branch=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If)
        and isinstance(n.test,ast.Compare) and 'standing_controller' in names(n.test)
        and any(isinstance(x,ast.Raise) and 'exact admitted route duration' in ast.unparse(x) for x in n.body))
    scope=dict(a=SimpleNamespace(seconds=seconds),standing_controller=SimpleNamespace(start_time=44.,duration=16.))
    if seconds==60.:execute([branch],scope)
    else:
        with pytest.raises(ValueError,match='exact admitted route duration'):execute([branch],scope)


def test_poststep_geometry_uses_current_cached_state_only_after_release_epoch():
    block = next(n for n in ast.walk(MAIN) if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name) and n.test.id == 'standing_transfer'
        and 'withdrawal_measurement' in names(n))
    leaf = np.array([.1,.2,.3,1.,0.,0.,0.], np.float32)
    handle = np.array([.4,.5,.6,1.,0.,0.,0.], np.float32)
    hand = np.array([.7,.8,.9,1.,0.,0.,0.], np.float32)
    root = np.arange(13, dtype=np.float32)
    joints = np.array([.1,.2], np.float32)
    calls = []
    def measure(**kwargs):
        calls.append(kwargs)
        return dict(right_environment_clearance_m=.045, time_s=kwargs['time_s'])
    surface = dict(total_normal_load_N=3.,palm_normal_load_N=3.)
    contact = SimpleNamespace(
        get_contact_force_matrix=lambda **kwargs:Tensor(np.zeros((1,1,3))),
        get_friction_data=lambda dt:tuple(Tensor(x) for x in (
            np.zeros((16384,3)),np.zeros((16384,3)),np.zeros((1,1),int),np.zeros((1,1),int))))
    scope = dict(np=np, step=2, dt=.002, standing_transfer=SimpleNamespace(started=.004),
        standing_controller=SimpleNamespace(start_time=.006),
        withdrawal_geometry=SimpleNamespace(required_robot_body_names=('rh_palm',),
            required_door_body_names=('leaf','leaf_handle'),read=measure),
        robot=SimpleNamespace(body_names=['rh_palm'],data=SimpleNamespace(body_state_w=Tensor(hand[None,None]))),
        door=SimpleNamespace(body_names=['leaf','leaf_handle'],data=SimpleNamespace(
            body_state_w=Tensor(np.stack([leaf,handle])[None]),joint_pos=np.array([[.09,.01,.0001]]))),
        dnames=['leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide'],rnames=['a','b'],
        audit_contacts=contact,audit_paths=[],audit_filters=[],
        contact_force_pairs=lambda *args,**kwargs:np.zeros((1,1,3)),panel_surface_loads=lambda *args:surface,
        transfer_steps=[],withdrawal_steps=[],teacher_info={'phase':'standing_withdrawal','stance_status':'solved'},
        transfer_rest_stop=None,standing_continuation_steps=None,
        acquisition_states={'root':[root],'joints':[joints]},
        pad_steps=[dict(sim_time_s=.006,valid_pad_grasp=True,contacts=[])])
    scope['withdrawal_recorder']=scope['withdrawal_steps']
    execute([block], scope)
    assert calls == [] and scope['withdrawal_steps'][-1]['geometry'] is None
    scope['step'] = 3
    scope['pad_steps'][-1]['sim_time_s'] = .008
    execute([block], scope)
    assert len(calls) == 1 and calls[0]['time_s'] == calls[0]['pose_time_s'] == .008
    np.testing.assert_array_equal(calls[0]['root'],root)
    np.testing.assert_array_equal(calls[0]['leaf_pose'],leaf)
    np.testing.assert_array_equal(calls[0]['handle_pose'],handle)
    np.testing.assert_array_equal(calls[0]['body_poses']['rh_palm'],hand)
    assert calls[0]['joints'] == dict(zip(['a','b'],joints))
    assert scope['withdrawal_steps'][-1]['geometry']['time_s'] == .008
    assert scope['withdrawal_steps'][-1]['leaf_pose'] == leaf.astype(float).tolist()


def test_new_runtime_and_proof_inputs_are_captured_separately(tmp_path):
    start = next(i for i,n in enumerate(MAIN.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets))
    end = next(i for i in range(start,len(MAIN.body)) if isinstance(MAIN.body[i],ast.For)
        and isinstance(MAIN.body[i].target,ast.Name) and MAIN.body[i].target.id=='source')+1
    class Args:
        def __getattr__(self,name): return None
    a = Args()
    a.acquisition=True; a.operate_after_acquisition=True
    a.standing_transfer_prefix_source=str(tmp_path/'original-source')
    for name in ('robot_usd','door_usd','motors','reference','standing_transfer_route','native_door'):
        path=tmp_path/name;path.write_text('{}');setattr(a,name,str(path))
    context = dict(input_sha256={'historical-proof.py':'a'*64})
    for key in ('runtime_path','source_config_path','coupled_envelope_path','coupled_audit_path'):
        path=tmp_path/(key+'.json');path.write_text('{}');context[key]=str(path)
    a.standing_withdrawal_route=context['runtime_path']
    out=tmp_path/'new-run';out.mkdir()
    scope=dict(Path=Path,__file__=str(SOURCE),a=a,out=out,record_standing_body_poses=True,physics_audit_enabled=True,
        sequence=None,full_opening=None,continuous=None,sensor_actor=None,jev_plan=None,
        hashlib=hashlib,json=json,time=time,
        standing_controller=SimpleNamespace(source_context=SimpleNamespace(data=context),source_admission={}))
    execute(MAIN.body[start:end],scope)
    provenance=json.loads((out/'provenance.json').read_text())
    assert provenance['withdrawal_admission_input_sha256']==context['input_sha256']
    assert 'historical-proof.py' not in provenance['files']
    assert a.native_door in provenance['files'] and context['runtime_path'] in provenance['files']
    for name in ('isaac_withdrawal_prefix_witness.py','isaac_withdrawal_measurements.py','isaac_withdrawal_evaluation.py',
                 'isaac_withdrawal_runtime.py','isaac_withdrawal_support.py','isaac_coupled_release_reference.py'):
        current=ROOT/'doorbench/dexterous'/name
        assert provenance['files'][str(current.resolve())]==hashlib.sha256(current.read_bytes()).hexdigest()
        assert (out/('source-'+name)).read_bytes()==current.read_bytes()
    for name in ('runtime','source','envelope','envelope-audit','admission'):
        assert (out/('standing-withdrawal-'+name+'.json')).is_file()

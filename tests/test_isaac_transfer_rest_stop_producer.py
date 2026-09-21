"""CPU execution of prospective endpoint selection in the real producer AST."""
import ast
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from doorbench.dexterous.isaac_transfer_rest_stop import TransferRestStop
from test_isaac_transfer_rest_stop import sample as rest_sample, observed_prefix, detector

from test_isaac_prefix_witness import source
from test_isaac_withdrawal_prefix_witness import transfer_source
from test_isaac_withdrawal_producer import make_scope
from test_isaac_transfer_prefix_wiring import MAIN, ENTRY, SOURCE, names, execute
from test_isaac_standing_transfer_cli import command, ROOT

SAVE=next(n for n in ast.walk(MAIN) if isinstance(n,ast.FunctionDef) and n.name=='save_transfer_rest_stop')
STOP=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If)
    and 'transfer_rest_triggered' in names(n.test) and 'evaluation_seconds' in names(n))


def stop_scope(tmp_path,*,withdrawal=False,triggered=True):
    scope=dict(json=json,out=tmp_path,step=18499,dt=.002,evaluation_seconds=48.,
        a=SimpleNamespace(seconds=48.,standing_withdrawal_route='runtime.json' if withdrawal else None),
        transfer_rest_triggered=triggered,transfer_rest_terminated=False,transfer_rest_continued=False,
        transfer_rest_stop=SimpleNamespace(receipt=lambda:dict(triggered=triggered,terminal_time_s=37. if triggered else None)))
    execute([SAVE],scope)
    return scope


@pytest.mark.parametrize('withdrawal,triggered',[(False,False),(False,True),(True,True)])
def test_only_explicit_standalone_trigger_stops_and_changes_scoring_epoch(tmp_path,withdrawal,triggered):
    scope=stop_scope(tmp_path,withdrawal=withdrawal,triggered=triggered)
    # Execute the real break statement inside a one-iteration loop, preserving
    # whether any later work was reached. Never run physics or replay commands.
    loop=ast.For(target=ast.Name(id='_once',ctx=ast.Store()),iter=ast.List(elts=[ast.Constant(0)],ctx=ast.Load()),
        body=[STOP,ast.Assign(targets=[ast.Name(id='continued_loop',ctx=ast.Store())],value=ast.Constant(True))],orelse=[])
    scope['continued_loop']=False
    execute([ast.fix_missing_locations(loop)],scope)
    terminated=triggered and not withdrawal
    assert scope['transfer_rest_terminated'] is terminated
    assert scope['continued_loop'] is (not terminated)
    assert scope['evaluation_seconds']==(37. if terminated else 48.)
    assert scope['a'].seconds==48.
    if terminated:
        receipt=json.loads((tmp_path/'standing-transfer-rest-stop.json').read_text())
        assert receipt['maximum_seconds']==48. and receipt['detector']['terminal_time_s']==37.
        assert receipt['mode']=='terminate' and receipt['terminated_on_qualified_rest']
        assert not receipt['continued_to_withdrawal']


def test_prefix_only_receipt_records_continuation_without_claiming_termination(tmp_path):
    scope=stop_scope(tmp_path,withdrawal=True)
    scope['transfer_rest_continued']=True
    receipt=scope['save_transfer_rest_stop']()
    assert receipt['mode']=='prefix-only' and receipt['continued_to_withdrawal']
    assert not receipt['terminated_on_qualified_rest']
    assert receipt['maximum_seconds']==scope['a'].seconds==48.


@pytest.mark.parametrize('terminal',[None,.004,.006,.008])
def test_withdrawal_requires_same_first_endpoint_before_any_motor_submission(transfer_source,tmp_path,monkeypatch,terminal):
    scope,witness,events,record=make_scope(transfer_source,tmp_path,monkeypatch)
    for i in range(3):record(i)
    saves=[]
    scope.update(transfer_rest_stop=SimpleNamespace(receipt=lambda:dict(triggered=terminal is not None,terminal_time_s=terminal)),
        transfer_rest_continued=False,save_transfer_rest_stop=lambda:saves.append(True),step=3)
    if terminal==.006:
        execute([ENTRY],scope)
        assert events[-1]==('force',3,True) and scope['transfer_rest_continued']
        assert saves==[True]
    else:
        with pytest.raises(ValueError,match='same first qualified transfer rest epoch'):execute([ENTRY],scope)
        assert events==[] and not witness.receipt()['stage_entry_authorized'] and saves==[]


def test_actual_endpoint_drives_final_pad_window_and_counts_only_after_explicit_stop():
    block=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If)
        and isinstance(n.test,ast.Name) and n.test.id=='physics_audit_enabled'
        and any(isinstance(x,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='tail' for v in x.targets) for x in n.body))
    last=next(i for i,n in enumerate(block.body) if isinstance(n,ast.Expr)
        and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
        and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='checks')
    pads=[dict(sim_time_s=i*.002,valid_pad_grasp=True,active_contact_count=0) for i in range(501)]
    states=dict(root=np.tile([0,0,1,1,0,0,0,0,0,0,0,0,0],(500,1)),motor_forces=np.zeros((500,61)),
        time_s=np.arange(1,501)*.002,torso_tilt_deg=np.zeros(500))
    for end,complete,count in ((1.,True,251),(2.,False,0)):
        scope=dict(np=np,evaluation_seconds=end,a=SimpleNamespace(seconds=2.),dt=.002,pad_steps=pads,
            acquisition_states=states,rows=[{}],mechanical_audit={'checks':{'joint_stops':True}},
            acquisition_reset={'door':{'leaf_hinge':0.,'leaf_handle_hinge':0.}},
            max_motor_delivery_error=0.,force_ranges=np.tile([-10.,10.],(61,1)))
        execute(block.body[:last+1],scope)
        assert scope['checks']['complete_physics_steps'] is complete
        assert scope['checks']['sustained_pad_grasp'] is complete
        assert len(scope['tail'])==count and scope['a'].seconds==2.


def test_deadline_without_trigger_remains_failed_in_final_operation_report(tmp_path):
    branch=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If)
        and 'qualified_transfer_rest_endpoint' in ast.unparse(n)
        and isinstance(n.test,ast.Compare) and 'transfer_rest_stop' in names(n.test))
    for terminated in (False,True):
        scope=stop_scope(tmp_path,triggered=terminated)
        scope.update(transfer_rest_terminated=terminated,operation_checks={'original_physics':True},operation_report={})
        execute([branch],scope)
        assert scope['operation_report']['passed'] is terminated
        assert scope['operation_checks']['qualified_transfer_rest_endpoint'] is terminated
        assert scope['operation_report']['maximum_seconds']==48.


def test_completion_recording_uses_untrimmed_selected_actual_endpoint(tmp_path):
    assignment=next(n for n in ast.walk(MAIN) if isinstance(n,ast.Assign)
        and any(isinstance(v,ast.Name) and v.id=='completed_recording' for v in n.targets))
    for end,complete in ((1.,True),(2.,False)):
        scope=dict(evaluation_seconds=end,acquisition_states={'time_s':np.arange(1,501)*.002},dt=.002,
            continuous=None,full_opening=None,full_aperture_crossed=False,out=tmp_path)
        execute([assignment],scope)
        assert scope['completed_recording'] is complete
        assert len(scope['acquisition_states']['time_s'])==500


def test_rest_stop_cli_is_explicit_and_requires_transfer():
    env=dict(os.environ,PYTHONPATH=str(ROOT))
    argv=command(['--standing-transfer-stop-on-rest'])
    result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    index=argv.index('--standing-transfer-route');del argv[index:index+2]
    result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'rest stop requires an explicit transfer route' in result.stderr


def test_actual_poststep_inputs_trigger_only_after_251_joint_valid_samples(detector,tmp_path):
    branch=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If)
        and 'transfer_rest_stop' in names(n.test) and 'transfer_rest_triggered' in names(n.test)
        and any(isinstance(x,ast.Assign) for x in n.body))
    scope=stop_scope(tmp_path,triggered=False)
    scope.update(transfer_rest_stop=detector,dnames=['leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide'])
    for tick in range(4001,4252):
        t=tick*.002;pad,surface,angles=rest_sample(t)
        scope.update(step=tick-1,pad_steps=[pad],transfer_steps=[surface],
            door=SimpleNamespace(data=SimpleNamespace(joint_pos=np.array([[angles['leaf'],angles['operator'],angles['latch']]]))))
        execute([branch],scope)
        assert scope['transfer_rest_triggered'] is (tick==4251)
    receipt=json.loads((tmp_path/'standing-transfer-rest-stop.json').read_text())
    assert receipt['detector']['terminal_time_s']==8.502
    assert receipt['detector']['observed_samples']==4251 and receipt['detector']['window_samples']==251
    # The detector freezes first-trigger evidence while the loop later decides
    # whether to stop or continue toward explicitly admitted withdrawal.
    assert receipt['terminated_on_qualified_rest'] is False


def test_exception_report_retains_actual_pad_observation_and_failed_endpoint(tmp_path):
    from doorbench.dexterous.isaac_evidence_cleanup import EvidenceCleanup
    branch=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If) and 'failed_transfer' in names(n)
        and 'withdrawal_steps' in names(n.test))
    scope=stop_scope(tmp_path,triggered=False)
    scope.update(withdrawal_steps=None,run_error=ValueError('Synthetic nonfinite evidence'),
        acquisition_states={'time_s':[1.]},
        pad_steps=[dict(sim_time_s=i*.002,valid_pad_grasp=True) for i in range(250,501)],
        standing_transfer=SimpleNamespace(started=.002,info={}))
    scope['a'].standing_transfer_route='route.json'
    scope['evidence_cleanup']=EvidenceCleanup(scope['run_error'])
    execute([branch],scope)
    report=json.loads((tmp_path/'operation-report.json').read_text())
    assert not report['passed'] and report['duration_s']==1. and report['maximum_seconds']==48.
    assert report['checks']['sustained_pad_grasp'] is True
    assert not report['checks']['qualified_transfer_rest_endpoint']
    assert report['standing_transfer_rest_stop']['detector']['triggered'] is False

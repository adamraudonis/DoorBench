"""Exercise the actual end-of-interval probe without importing or stepping Kit."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE=Path(__file__).resolve().parents[1]/'scripts/dexterous/isaac_opening.py'
TREE=ast.parse(SOURCE.read_text())
MAIN=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='main')
PROBE=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If)
    and 'pause_probe_counter' in ast.unparse(n.test) and 'step + 1' in ast.unparse(n.test))


def exercise(tmp_path, *, changed=False, read_error=False, counter=True):
    events=[];reads=[]
    def capture(**kw):
        events.append('read');reads.append(kw)
        if read_error:raise ValueError('backend read failed')
        return dict(state=2 if changed and len(reads)==2 else 1)
    clock=iter([10.,12.1])
    scope=dict(step=999,dt=.002,a=SimpleNamespace(pause_readback_probe_at_seconds=2.),
        pause_probe_counter=SimpleNamespace(receipt=lambda:dict(count=1000)) if counter else None,
        out=tmp_path,json=json,teacher=object(),operation=object(),standing_transfer=None,
        standing_controller=None,sim=object(),time_origin=0.,robot=object(),door=object(),
        audit_contacts=object(),invariant_getters={},acquisition_states={'time_s':[0]*1000},
        pad_steps=[0]*1001,transfer_steps=None,capture_native_pause_anchor=capture,
        time=SimpleNamespace(monotonic=lambda:next(clock),sleep=lambda seconds:events.append(('sleep',seconds))))
    execute=lambda:exec(compile(ast.Module(body=[PROBE],type_ignores=[]),str(SOURCE),'exec'),scope)
    return execute,events,reads


def test_actual_probe_reads_sleeps_reads_once_without_control_or_step(tmp_path):
    run,events,reads=exercise(tmp_path)
    run()
    assert events==['read',('sleep',2.),'read']
    assert all(r['step_index']==1000 and r['epoch_s']==2. for r in reads)
    assert reads[0]['controllers']==reads[1]['controllers']
    receipt=json.loads((tmp_path/'pause-readback-probe.json').read_text())
    assert receipt['passed'] and receipt['exact_anchor_equal']
    assert receipt['authorized_stages']==0
    assert json.loads((tmp_path/'pause-probe-before.json').read_text())==json.loads((tmp_path/'pause-probe-after.json').read_text())


def test_changed_state_stops_and_preserves_both_observations(tmp_path):
    run,events,_=exercise(tmp_path,changed=True)
    with pytest.raises(RuntimeError,match='changed actual state'):
        run()
    receipt=json.loads((tmp_path/'pause-readback-probe.json').read_text())
    assert not receipt['passed'] and not receipt['exact_anchor_equal']
    assert receipt['error_type']=='RuntimeError'
    assert events==['read',('sleep',2.),'read']
    assert (tmp_path/'pause-probe-after.json').exists()


def test_failed_initial_read_stops_without_sleep_or_resume(tmp_path):
    run,events,_=exercise(tmp_path,read_error=True)
    with pytest.raises(ValueError,match='backend read'):
        run()
    receipt=json.loads((tmp_path/'pause-readback-probe.json').read_text())
    assert not receipt['passed'] and receipt['error_type']=='ValueError'
    assert events==['read']


def test_absent_probe_performs_no_read_or_wait(tmp_path):
    run,events,_=exercise(tmp_path,counter=False)
    run()
    assert events==[] and list(tmp_path.iterdir())==[]


def test_probe_is_after_complete_interval_and_before_rest_exit():
    loop=next(n for n in ast.walk(MAIN) if isinstance(n,ast.For) and PROBE in n.body)
    i=loop.body.index(PROBE)
    before=ast.unparse(ast.Module(body=loop.body[:i],type_ignores=[]))
    after=ast.unparse(ast.Module(body=loop.body[i+1:],type_ignores=[]))
    assert 'standing_reference_tail.observe_completed_interval' in before
    assert 'wall_timing.finish' in before
    assert 'Nonfinite robot state' in before
    assert 'transfer_rest_terminated = True' in after
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr in ('update','step','force','render','write_data_to_sim')
        and ast.unparse(n.func.value)!='probe_receipt' for n in ast.walk(PROBE))

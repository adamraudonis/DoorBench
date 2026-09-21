"""Exercise the actual producer finalizer without importing Kit or stepping physics."""
import ast
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from doorbench.dexterous.bounded_evidence import BoundedEvidence
from doorbench.dexterous.isaac_evidence_cleanup import EvidenceCleanup, export_once


SOURCE = Path(__file__).resolve().parents[1]/'scripts/dexterous/isaac_opening.py'
TREE = ast.parse(SOURCE.read_text())
MAIN = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


RUN = next(n for n in ast.walk(MAIN) if isinstance(n, ast.Try)
           and any(h.name == 'run_error' for h in n.handlers))
HANDLER = next(h for h in RUN.handlers if h.name == 'run_error')
WITHDRAWAL_FAILURE = next(n for n in HANDLER.body if isinstance(n, ast.If)
                          and 'failed_withdrawal' in names(n))


def execute(nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)


def evidence(tmp_path, name, count=311):
    writer = BoundedEvidence(tmp_path/name, chunk_size=250)
    rows = [dict(sim_time_s=(i+1)*.002, value=[i, i/3]) for i in range(count)]
    for row in rows:
        writer.append(row)
    return writer, rows


def final_scope(tmp_path, withdrawal, continuation):
    return dict(out=tmp_path, json=json, transfer_rest_stop=None,
                transfer_prefix=None, withdrawal_prefix=None,
                withdrawal_steps=withdrawal, standing_continuation_steps=continuation,
                standing_reference_tail=SimpleNamespace(receipt=lambda:dict(records=[{'time_s':.622}])) ,
                jev_gate=None, rows=[], acquisition_states={},pad_steps=None,physics_audit_enabled=False,
                transfer_steps=None,writer=None,hand_writer=None)


def decoded(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def test_original_controller_error_survives_already_exported_withdrawal_and_all_pending_tail(tmp_path):
    withdrawal, withdrawal_rows = evidence(tmp_path, 'withdrawal')
    continuation, continuation_rows = evidence(tmp_path, 'continuation')
    assert len(continuation.chunks) == 1 and len(continuation.pending) == 61
    path = tmp_path/'standing-withdrawal-steps.json.gz'
    original = ValueError('Measured operator/latch outside screened coupled envelope')
    scope = final_scope(tmp_path, withdrawal, continuation)
    with pytest.raises(ValueError) as caught:
        try:
            raise original
        except BaseException as run_error:
            scope['evidence_cleanup'] = cleanup = EvidenceCleanup(run_error)
            assert cleanup.export('failed_withdrawal_export', withdrawal, path) == 'exported'
            saved = path.read_bytes()
            raise
        finally:
            execute(RUN.finalbody, scope)
    assert caught.value is original
    assert path.read_bytes() == saved
    assert decoded(path) == withdrawal_rows
    assert decoded(tmp_path/'standing-continuation-steps.json.gz') == continuation_rows
    assert not continuation.pending and continuation.exported_path is not None
    assert json.loads((tmp_path/'standing-continuation-reference-tail.json').read_text())['records']
    receipt = json.loads((tmp_path/'evidence-cleanup.json').read_text())
    assert receipt['primary_error']['message'] == str(original)
    assert receipt['cleanup_actions_succeeded'] and not receipt['physical_qualification']
    assert all(action['succeeded'] for action in receipt['actions'])
    with pytest.raises(ValueError, match='Fresh evidence'):
        withdrawal.export(path)
    with pytest.raises(ValueError, match='finalized'):
        continuation.append({})


@pytest.mark.parametrize('prior_export', [False, True])
def test_unrelated_destination_never_overwritten_and_other_exports_still_attempted(tmp_path, prior_export):
    withdrawal, rows = evidence(tmp_path, 'withdrawal')
    if prior_export:
        withdrawal.export(tmp_path/'different.json.gz')
    path = tmp_path/'standing-withdrawal-steps.json.gz'
    path.write_bytes(b'unrelated immutable evidence')
    continuation, expected = evidence(tmp_path, 'continuation')
    original = RuntimeError('Original controller stopped')
    scope = final_scope(tmp_path, withdrawal, continuation)
    with pytest.raises(RuntimeError) as caught:
        try:
            raise original
        finally:
            execute(RUN.finalbody, scope)
    assert caught.value is original
    assert path.read_bytes() == b'unrelated immutable evidence'
    assert list(withdrawal) == rows
    assert decoded(tmp_path/'standing-continuation-steps.json.gz') == expected
    assert (tmp_path/'standing-continuation-reference-tail.json').is_file()
    receipt = json.loads((tmp_path/'evidence-cleanup.json').read_text())
    assert not receipt['cleanup_actions_succeeded']
    assert [(a['action'], a['succeeded']) for a in receipt['actions']] == [
        ('final_withdrawal_export', False), ('final_continuation_export', True), ('final_reference_tail', True),
        ('core_trace', True)]
    assert 'Evidence cleanup failures' in original.__notes__[0]


def test_failed_export_verification_retains_chunks_and_pending_file_without_losing_other_evidence(tmp_path, monkeypatch):
    withdrawal, rows = evidence(tmp_path, 'withdrawal')
    continuation, expected = evidence(tmp_path, 'continuation')
    original_open = gzip.open
    def corrupt_verification(path, mode='rb', *args, **kwargs):
        if Path(path).name == 'standing-withdrawal-steps.json.gz.pending' and mode == 'rb':
            raise OSError('Injected verification read failure')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(gzip, 'open', corrupt_verification)
    scope = final_scope(tmp_path, withdrawal, continuation)
    with pytest.raises(OSError, match='verification read failure'):
        execute(RUN.finalbody, scope)
    assert withdrawal.exported_path is None
    assert len(withdrawal.chunks) == 2 and list(withdrawal) == rows
    assert (tmp_path/'standing-withdrawal-steps.json.gz.pending').is_file()
    assert not (tmp_path/'standing-withdrawal-steps.json.gz').exists()
    assert decoded(tmp_path/'standing-continuation-steps.json.gz') == expected
    assert (tmp_path/'standing-continuation-reference-tail.json').is_file()
    receipt = json.loads((tmp_path/'evidence-cleanup.json').read_text())
    assert receipt['primary_error'] is None and not receipt['cleanup_actions_succeeded']


def test_secondary_exception_handler_error_does_not_replace_original_failure(tmp_path):
    continuation, rows = evidence(tmp_path, 'continuation')
    scope = final_scope(tmp_path, None, continuation)
    original = ValueError('Original controller rejection')
    secondary = OSError('Failed intermediate report write')
    with pytest.raises(ValueError) as caught:
        try:
            raise original
        except ValueError as error:
            scope['evidence_cleanup'] = EvidenceCleanup(error)
            raise secondary
        finally:
            execute(RUN.finalbody, scope)
    assert caught.value is original
    assert decoded(tmp_path/'standing-continuation-steps.json.gz') == rows
    receipt = json.loads((tmp_path/'evidence-cleanup.json').read_text())
    assert receipt['actions'][-1]['action'] == 'exception_handler'
    assert receipt['actions'][-1]['error']['message'] == str(secondary)
    assert str(secondary) in original.__notes__[0]


@pytest.mark.parametrize('filename', ['standing-continuation-reference-tail.json', 'evidence-cleanup.json'])
def test_existing_cleanup_document_is_preserved_and_error_propagates_after_exports(tmp_path, filename):
    continuation, expected = evidence(tmp_path, 'continuation')
    existing = tmp_path/filename
    existing.write_bytes(b'previous evidence')
    original = ValueError('Original controller rejection')
    with pytest.raises(ValueError) as caught:
        try:
            raise original
        finally:
            execute(RUN.finalbody, final_scope(tmp_path, None, continuation))
    assert caught.value is original and original.__notes__
    assert existing.read_bytes() == b'previous evidence'
    assert decoded(tmp_path/'standing-continuation-steps.json.gz') == expected


def test_missing_prior_export_is_not_treated_as_success(tmp_path):
    writer, _ = evidence(tmp_path, 'chunks')
    target = tmp_path/'final.json.gz'
    writer.export(target)
    target.rename(tmp_path/'moved.json.gz')
    with pytest.raises(ValueError, match='different or missing'):
        export_once(writer, target)
    assert not target.exists()


def test_failed_withdrawal_retains_predecessor_diagnostics_without_promoting_physical_result(tmp_path, monkeypatch):
    import doorbench.dexterous.standing_transfer_evaluation as evaluation
    withdrawal, _ = evidence(tmp_path, 'withdrawal')
    transfer, _ = evidence(tmp_path, 'transfer')
    transfer.export(tmp_path/'standing-transfer-steps.json.gz')
    calls = []
    def checks(records, **kwargs):
        calls.append((len(list(records)), kwargs))
        return dict(final_left_palm_support=True, sustained_pad_grasp=True)
    monkeypatch.setattr(evaluation, 'standing_transfer_checks', checks)
    stop_receipt = dict(triggered=True, terminal_time_s=.5)
    original = ValueError('Actual envelope rejection')
    scope = dict(json=json, out=tmp_path, run_error=original, dt=.002,
        evidence_cleanup=EvidenceCleanup(original), withdrawal_steps=withdrawal,
        acquisition_states={'time_s':[.622]}, transfer_steps=transfer,
        pad_steps=[dict(sim_time_s=(i+1)*.002,valid_pad_grasp=True) for i in range(311)],
        a=SimpleNamespace(standing_withdrawal_route='release.json',standing_transfer_route='transfer.json'),
        standing_transfer=SimpleNamespace(started=.1,info={'stance_status':'solved'}),
        transfer_rest_stop=object(), save_transfer_rest_stop=lambda:stop_receipt,
        standing_controller=SimpleNamespace(started_withdrawal=.5,release_started=None,
            source_admission={'synthetic':True},info={'failed':True}))
    execute([WITHDRAWAL_FAILURE], scope)
    assert calls == [(311, dict(seconds=.622,dt=.002,started_s=.1))]
    for name in ('operation-report.json', 'report.json', 'standing-withdrawal-report.json'):
        report = json.loads((tmp_path/name).read_text())
        assert report['passed'] is False and report['physical_evidence_complete'] is False
        assert report['checks']['controller_or_backend_succeeded'] is False
        assert report['checks']['final_left_palm_support'] is True
        assert report['standing_transfer']['started_s'] == .1
        assert report['standing_transfer_rest_stop'] == stop_receipt
        assert report['observed_final_pad_grasp_hold'] is True


def test_failed_pad_summary_read_cannot_hide_original_error_or_core_and_remaining_evidence(tmp_path):
    class FailsOnlyAfterExport(BoundedEvidence):
        def __iter__(self):
            if self.exported_path is not None:
                raise OSError('Injected failed pad summary read')
            yield from super().__iter__()
    pad = FailsOnlyAfterExport(tmp_path/'pad', chunk_size=250)
    pad_rows = [dict(sim_time_s=(i+1)*.002,valid_pad_grasp=True) for i in range(311)]
    for row in pad_rows:
        pad.append(row)
    withdrawal, withdrawal_rows = evidence(tmp_path, 'withdrawal')
    continuation, continuation_rows = evidence(tmp_path, 'continuation')
    transfer, transfer_rows = evidence(tmp_path, 'transfer')
    closed = []
    original = ValueError('Actual controller domain rejection')
    trace = [dict(time_s=.622,actual=True)]
    scope = final_scope(tmp_path, withdrawal, continuation)
    scope.update(np=np,rows=trace,physics_audit_enabled=True,pad_steps=pad,
        acquisition_states={'time_s':[.002,.622],'joints':[[1.,2.],[3.,4.]]},
        transfer_steps=transfer,run_error=original,dt=.002,
        writer=SimpleNamespace(close=lambda:closed.append('body')),
        hand_writer=SimpleNamespace(close=lambda:closed.append('hand')))
    with pytest.raises(ValueError) as caught:
        try:
            raise original
        except ValueError:
            execute(HANDLER.body, scope)
        finally:
            execute(RUN.finalbody, scope)
    assert caught.value is original
    assert 'Injected failed pad summary read' in original.__notes__[0]
    assert json.loads((tmp_path/'trace.json').read_text()) == trace
    with np.load(tmp_path/'acquisition-physics.npz') as arrays:
        np.testing.assert_array_equal(arrays['joints'],[[1.,2.],[3.,4.]])
    assert decoded(tmp_path/'acquisition-pad-steps.json.gz') == pad_rows
    assert decoded(tmp_path/'standing-withdrawal-steps.json.gz') == withdrawal_rows
    assert decoded(tmp_path/'standing-transfer-steps.json.gz') == transfer_rows
    assert closed == ['body','hand']
    assert decoded(tmp_path/'standing-continuation-steps.json.gz') == continuation_rows
    assert (tmp_path/'standing-continuation-reference-tail.json').is_file()
    receipt = json.loads((tmp_path/'evidence-cleanup.json').read_text())
    assert receipt['primary_error']['message'] == str(original)
    assert receipt['actions'][-1]['action'] == 'exception_handler'


@pytest.mark.parametrize('failure', ['continuation','reference_tail','cleanup_receipt'])
def test_normal_loop_cleanup_failure_still_preserves_core_records(tmp_path, failure):
    pad, pad_rows = evidence(tmp_path, 'pad')
    continuation, continuation_rows = evidence(tmp_path, 'continuation')
    transfer, transfer_rows = evidence(tmp_path, 'transfer')
    closed = []
    files = {'continuation':'standing-continuation-steps.json.gz',
             'reference_tail':'standing-continuation-reference-tail.json',
             'cleanup_receipt':'evidence-cleanup.json'}
    target = tmp_path/files[failure]
    target.write_bytes(b'preexisting unrelated file')
    trace = [dict(time_s=.622)]
    scope = final_scope(tmp_path, None, continuation)
    scope.update(rows=trace,acquisition_states={'time_s':[.002,.622],'joints':[[1.,2.],[3.,4.]]},
                 pad_steps=pad,physics_audit_enabled=True,transfer_steps=transfer,
                 writer=SimpleNamespace(close=lambda:closed.append('body')),
                 hand_writer=SimpleNamespace(close=lambda:closed.append('hand')))
    with pytest.raises((ValueError,FileExistsError)):
        execute(RUN.finalbody,scope)
    assert target.read_bytes() == b'preexisting unrelated file'
    assert json.loads((tmp_path/'trace.json').read_text()) == trace
    with np.load(tmp_path/'acquisition-physics.npz') as arrays:
        np.testing.assert_array_equal(arrays['joints'],[[1.,2.],[3.,4.]])
    assert decoded(tmp_path/'acquisition-pad-steps.json.gz') == pad_rows
    assert decoded(tmp_path/'standing-transfer-steps.json.gz') == transfer_rows
    assert closed == ['body','hand']
    if failure == 'continuation':
        assert list(continuation) == continuation_rows and continuation.exported_path is None
    else:
        assert decoded(tmp_path/'standing-continuation-steps.json.gz') == continuation_rows


def test_core_preservation_is_attempted_only_once_without_retrying_partial_failures(tmp_path, monkeypatch):
    from doorbench.dexterous.isaac_evidence_cleanup import preserve_core_evidence
    pad, pad_rows = evidence(tmp_path, 'pad')
    calls = []
    def fail_npz(*args, **kwargs):
        calls.append('npz')
        raise OSError('Injected partial physics archive failure')
    monkeypatch.setattr(np,'savez_compressed',fail_npz)
    cleanup = EvidenceCleanup()
    args = (cleanup,tmp_path,[dict(time_s=.622)],{'time_s':[.622]},pad,True)
    preserve_core_evidence(*args)
    before = (tmp_path/'trace.json').stat().st_mtime_ns
    preserve_core_evidence(*args)
    assert calls == ['npz']
    assert len(cleanup.errors) == 1
    assert decoded(tmp_path/'acquisition-pad-steps.json.gz') == pad_rows
    assert (tmp_path/'trace.json').stat().st_mtime_ns == before


def test_one_video_close_failure_does_not_skip_other_video_or_core_evidence(tmp_path):
    closed = []
    def first():
        closed.append('body')
        raise OSError('Body video close failure')
    scope = final_scope(tmp_path,None,None)
    scope.update(writer=SimpleNamespace(close=first),hand_writer=SimpleNamespace(close=lambda:closed.append('hand')))
    with pytest.raises(OSError,match='video close failure'):
        execute(RUN.finalbody,scope)
    assert closed == ['body','hand'] and (tmp_path/'trace.json').is_file()
    receipt = json.loads((tmp_path/'evidence-cleanup.json').read_text())
    assert [(a['action'],a['succeeded']) for a in receipt['actions']] == [
        ('final_video_close',False),('final_hand_video_close',True),('core_trace',True)]

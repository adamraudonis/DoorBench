from types import SimpleNamespace
import pytest
from doorbench.dexterous import storage_budget as budget


def test_admission_reserves_export_space_and_host_headroom(tmp_path, monkeypatch):
    free=[budget.RESERVE_BYTES + 100*budget.EVIDENCE_BYTES_PER_SECOND]
    monkeypatch.setattr(budget.shutil,'disk_usage',lambda _:SimpleNamespace(free=free[0]))
    assert budget.check_storage(tmp_path/'new'/'run',seconds_remaining=100)['free_bytes']==free[0]
    with pytest.raises(OSError,match='host reserve'):
        budget.check_storage(tmp_path,seconds_remaining=101)
    free[0]=budget.RESERVE_BYTES-1
    with pytest.raises(OSError): budget.check_storage(tmp_path)
    with pytest.raises(ValueError): budget.check_storage(tmp_path,seconds_remaining=float('nan'))


def test_retention_budget_counts_hardlinks_once_and_reserves_next_run(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(budget,'RETAINED_LIMIT_BYTES',100)
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    (a/'record').write_bytes(b'x'*60);os.link(a/'record',b/'same-record')
    assert budget.check_retained_budget([a,b],incoming_bytes=40)['retained_bytes']==60
    with pytest.raises(OSError,match='budget exceeded'):
        budget.check_retained_budget([a,b],incoming_bytes=41)


def test_retention_combines_worktree_and_saved_runs_without_double_count(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, 'RETAINED_LIMIT_BYTES', 100)
    worktree = tmp_path / 'out'
    archives = tmp_path / 'DoorBench-runs'
    target = archives / 'today' / 'trial'
    worktree.mkdir()
    target.mkdir(parents=True)
    (worktree / 'native').write_bytes(b'x' * 60)
    (target / 'isaac').write_bytes(b'x' * 30)
    roots = [target.parent, worktree, archives]
    assert budget.check_retained_budget(roots, incoming_bytes=10)['retained_bytes'] == 90
    with pytest.raises(OSError, match='budget exceeded'):
        budget.check_retained_budget(roots, incoming_bytes=11)

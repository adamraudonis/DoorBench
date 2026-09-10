"""A continuation must be rejected before allocating a scene or output files."""
import sys
import pytest
from scripts.dexterous import probe_post_opening as driver


def arguments(tmp_path,seconds='34'):
    args=['probe_post_opening.py']
    for name in ('robot','door','motors','initial-trajectory','body-reset','checkpoint'):
        args += ['--'+name,str(tmp_path/('missing-'+name))]
    return args+['--output',str(tmp_path/'run'),'--seconds',seconds]


@pytest.mark.parametrize('seconds',['nan','inf','0','-1'])
def test_unbounded_duration_rejected_before_files_or_scene(tmp_path,monkeypatch,seconds):
    monkeypatch.setattr(sys,'argv',arguments(tmp_path,seconds))
    with pytest.raises(ValueError,match='Finite positive'):driver.main()
    assert not (tmp_path/'run').exists()


def test_disk_reserve_failure_precedes_model_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',arguments(tmp_path))
    def reject(*args,**kwargs):raise OSError('Insufficient evidence storage')
    monkeypatch.setattr(driver,'check_storage',reject)
    with pytest.raises(OSError,match='Insufficient evidence storage'):driver.main()
    assert not (tmp_path/'run').exists()

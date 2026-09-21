import pytest

from test_local_operation_launcher import transfer_fixture
from doorbench.dexterous.transfer_preload import transfer_preload


def test_preload_choice_changes_only_the_explicit_post_prefix_stage(tmp_path,monkeypatch):
    launcher,args,argv,hashes,_=transfer_fixture(tmp_path,monkeypatch)
    base,_=launcher.prepare_transfer(args,argv,hashes)
    args.standing_transfer_preload_profile='maintain'
    assert launcher.prepare_transfer(args,argv,hashes)[0]==base
    args.standing_transfer_preload_profile='balanced-4n'
    assert launcher.prepare_transfer(args,argv,hashes)[0]==base+['--standing-transfer-preload-profile','balanced-4n']
    args.standing_transfer_route=None
    with pytest.raises(ValueError,match='preload'):
        launcher.prepare_transfer(args,argv,hashes)


def test_unknown_profile_cannot_reach_physics(tmp_path,monkeypatch):
    launcher,args,argv,hashes,_=transfer_fixture(tmp_path,monkeypatch)
    args.standing_transfer_preload_profile='unbounded'
    with pytest.raises(ValueError,match='preload'):launcher.prepare_transfer(args,argv,hashes)


def test_existing_balanced_profile_starts_at_exact_attained_loads_and_blends_smoothly():
    initial=dict(ff=3.,mf=2.,rf=2.,lf=2.,th=6.)
    assert transfer_preload(initial,0.,'balanced-4n')==initial
    assert transfer_preload(initial,1.,'balanced-4n')==dict(ff=4.,mf=4.,rf=4.,lf=4.,th=8.)
    epsilon=1e-4
    assert max(abs(transfer_preload(initial,epsilon,'balanced-4n')[k]-v) for k,v in initial.items())<1e-9

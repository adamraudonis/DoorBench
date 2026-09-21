import gzip
import json
from types import SimpleNamespace
import numpy as np
import pytest

from test_local_operation_launcher import module,write


def fixture(tmp_path,monkeypatch):
    from doorbench.dexterous import isaac_withdrawal_runtime,isaac_prefix_witness
    launcher=module(monkeypatch,tmp_path)
    source=tmp_path/'transfer';source.mkdir()
    route=tmp_path/'runtime.json';write(route,{'schema':'synthetic-runtime'})
    door=tmp_path/'door.xml';door.write_text('synthetic authored door')
    argv=['isaac-python','producer.py','--output','new','--seconds','60.0',
        '--standing-transfer-route','actual-transfer.json','--standing-transfer-prefix-source','thumb-source']
    prior=argv.copy();prior[3]='old';prior[5]='44.0'
    write(source/'launch.json',dict(argv=prior))
    context=dict(source_run=str(source),door_xml_path=str(door))
    admission=SimpleNamespace(source_context=SimpleNamespace(data=context),start_time=44.,duration=16.,
        input_sha256={str(route):launcher.sha(route),str(door):launcher.sha(door)})
    monkeypatch.setattr(isaac_withdrawal_runtime,'admit_isaac_withdrawal_runtime',lambda p:admission)
    monkeypatch.setattr(isaac_prefix_witness,'historical_source_hashes',lambda p:{})
    args=SimpleNamespace(standing_withdrawal_route=route,jev_progress_plan=None,seconds=60.)
    return launcher,args,argv,admission


def test_new_stage_adds_only_bound_geometry_and_route_after_exact_recipe(tmp_path,monkeypatch):
    launcher,args,argv,admission=fixture(tmp_path,monkeypatch)
    changed,hashes=launcher.prepare_withdrawal(args,argv,{})
    assert changed[:len(argv)]==argv
    assert changed[len(argv):]==['--native-door',admission.source_context.data['door_xml_path'],
        '--standing-withdrawal-route',str(args.standing_withdrawal_route)]
    assert hashes[str(args.standing_withdrawal_route)]==launcher.sha(args.standing_withdrawal_route)


@pytest.mark.parametrize('change',['recipe','missing_prefix','jev','short','long'])
def test_withdrawal_rejects_any_changed_or_unbounded_prefix(tmp_path,monkeypatch,change):
    launcher,args,argv,_=fixture(tmp_path,monkeypatch)
    if change=='recipe':argv[0]='another-python'
    elif change=='missing_prefix':argv=argv[:-2]
    elif change=='jev':args.jev_progress_plan='live.json'
    elif change=='short':args.seconds=59.998
    else:args.seconds=60.002
    with pytest.raises(ValueError):launcher.prepare_withdrawal(args,argv,{})


def test_old_mode_does_not_attempt_new_admission(tmp_path,monkeypatch):
    launcher,args,argv,_=fixture(tmp_path,monkeypatch)
    args.standing_withdrawal_route=None
    assert launcher.prepare_withdrawal(args,argv,{'old':'hash'})==(argv,{'old':'hash'})


def test_offline_full_prefix_includes_canonical_historical_leaf(tmp_path,monkeypatch):
    launcher=module(monkeypatch,tmp_path)
    source=tmp_path/'source/trial';source.mkdir(parents=True)
    trial=tmp_path/'new';trial.mkdir()
    poses=np.array([[0,0,0,1,0,0,0],[0,.0001,0,1,0,0,0]],np.float32)
    archive=dict(time_s=np.array([.002,.004]),root=np.zeros((2,13)),joints=np.zeros((2,69)),
        joint_velocity=np.zeros((2,69)),motor_forces=np.zeros((2,61)),door=np.zeros((2,3)),
        door_velocity=np.zeros((2,3)),standing_body_poses=np.zeros((2,6,7)))
    np.savez(source/'acquisition-physics.npz',**archive)
    np.savez(trial/'acquisition-physics.npz',**archive,standing_leaf_pose=poses)
    with gzip.open(source/'standing-transfer-steps.json.gz','wt') as f:
        json.dump([dict(time_s=(i+1)*.002,leaf_pose=p.tolist()) for i,p in enumerate(poses)],f)
    assert launcher.audit_withdrawal_prefix(source.parent,trial,.004)['passed']
    poses[1,1]+=.001
    np.savez(trial/'acquisition-physics.npz',**archive,standing_leaf_pose=poses)
    result=launcher.audit_withdrawal_prefix(source.parent,trial,.004)
    assert not result['passed'] and not result['leaf_pose_prefix_passed']
    assert all(result['fields'].values())

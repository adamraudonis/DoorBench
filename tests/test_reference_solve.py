"""Tiny actual-MuJoCo solve/export integration, including the optional clock."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco');pytest.importorskip('mink')
from doorbench.reference import solve as module
from doorbench.reference.ik import DoorHumanoidIK
from doorbench.reference.retime import RetimeResult
from tests.test_reference_ik import door


def test_solve_export_preserves_proposal_source_and_pose_correspondence(door,tmp_path,monkeypatch):
    (door/"spec.json").write_text(json.dumps({"id": door.name, "family": "swing_single"}))
    initial=DoorHumanoidIK(door);poses=initial.foot_poses();landmarks=initial.joint_positions()
    n=3;time=np.array([0.,.1,.2]);native_time=np.array([0.,.02,.04])
    source_hashes={name:hashlib.sha256((door/name).read_bytes()).hexdigest() for name in ['door.xml','model.json','spec.json']}
    guide=SimpleNamespace(time=time,native_time=native_time,native_qpos=np.array([[0.],[.01],[.02]]),
        pelvis=np.tile([0.,-1.,.94],(n,1)),yaw=np.zeros(n),
        foot_pos=np.tile([poses['left_foot']['pos'],poses['right_foot']['pos']],(n,1,1)),
        foot_quat=np.tile([poses['left_foot']['quat_wxyz'],poses['right_foot']['quat_wxyz']],(n,1,1)),
        foot_contact=np.ones((n,2),bool),hand_contact=np.zeros((n,2),bool),
        hand_weight=np.zeros((n,2)),hands=np.tile(landmarks[[6,9]],(n,1,1)),
        phases=['hold']*n,metadata={'source_sha256':source_hashes,'scope':'test native recording'})
    monkeypatch.setattr(module,'make_guide',lambda *args,**kwargs:guide)
    recordings=tmp_path/'recordings';(recordings/'trajectories').mkdir(parents=True)
    np.savez(recordings/'trajectories'/f'{door.name}.npz',time=native_time,target=guide.hands[:,1])
    observed={};new_time=np.array([0.,.15,.4])
    def clock(model,qpos,original_time,**kwargs):
        observed.update(model=model,qpos=qpos.copy(),time=original_time.copy(),native_time=kwargs['native_time'].copy())
        return RetimeResult(new_time.copy(),kwargs['native_time'].copy(),np.diff(new_time)/np.diff(original_time),
                            {'schema':'doorbench.retime.v1','test_clock':True},True)
    monkeypatch.setattr(module,'retime_trajectory',clock)
    destination=tmp_path/'solved';metadata=module.solve_door(door,recordings,destination)
    with np.load(destination/door.name/'trajectory.npz',allow_pickle=False) as result:
        np.testing.assert_array_equal(result['actor_time'],new_time)
        np.testing.assert_array_equal(result['proposal_time'],time)
        np.testing.assert_array_equal(result['native_time'],native_time)
        np.testing.assert_array_equal(result['qpos'],guide.native_qpos)
        np.testing.assert_array_equal(result['hand_contact'],guide.hand_contact)
        np.testing.assert_array_equal(result['foot_contact'],guide.foot_contact)
        np.testing.assert_array_equal(result['foot_target_pos'],guide.foot_pos)
        native_address=int(observed['model'].joint('hinge').qposadr[0])
        np.testing.assert_array_equal(observed['qpos'][:,native_address],guide.native_qpos[:,0])
    assert metadata['duration']==.4 and metadata['fps']==60
    assert metadata['retiming']=={'success':True,'schema':'doorbench.retime.v1','test_clock':True}
    assert metadata['source_sha256']==source_hashes and metadata['proposal']==guide.metadata
    assert metadata['phases']==guide.phases and metadata['status']=='unvalidated'
    assert metadata['trajectory_sha256']==hashlib.sha256((destination/door.name/'trajectory.npz').read_bytes()).hexdigest()
    assert json.loads((destination/door.name/'clip.json').read_text())==metadata


@pytest.mark.parametrize('frames',[0,1,-1,True,2.5])
def test_bad_frame_budget_rejected_before_reading_sources(tmp_path,frames):
    with pytest.raises(ValueError,match='max_frames'):
        module.solve_door(tmp_path/'missing',tmp_path/'recordings',tmp_path/'out',max_frames=frames)


def test_cli_rejects_one_frame_budget_before_opening_missing_manifest(monkeypatch):
    monkeypatch.setattr('sys.argv',['solve','--assets','missing-assets','--doors','missing','--max-frames','1'])
    with pytest.raises(SystemExit) as error:module.main()
    assert error.value.code==2

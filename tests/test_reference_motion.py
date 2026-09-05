import os
import shutil
import json
from pathlib import Path
import numpy as np
import pytest
from doorbench.reference.humanoid import two_bone, fit_motion
from doorbench.reference.record import Recorder, record_one
from doorbench.benchmark.runner import load_manifest, Job, run_episode

@pytest.mark.parametrize('target',[[.2,.3,.1],[0,4,0],[0,0,0],[-2,-1,1]])
def test_fixed_limb_lengths_and_truthful_reach(target):
    root=np.zeros(3); target=np.array(target,float)
    elbow,end,error=two_bone(root,target,.30,.28,[1,0,-1])
    assert np.linalg.norm(elbow-root)==pytest.approx(.30)
    assert np.linalg.norm(end-elbow)==pytest.approx(.28)
    assert error==pytest.approx(np.linalg.norm(end-target))
    assert np.isfinite(elbow).all()


@pytest.mark.parametrize("door_id",["db0002_swing_single","db0004_bifold","db0031_saloon","db0066_revolving","db0202_turnstile_tripod"])
def test_recorder_does_not_change_benchmark_outcome(door_id):
    root=Path(__file__).resolve().parents[1]
    door=next(d for d in load_manifest(str(root/'assets'))['doors'] if d['id']==door_id)
    job=Job(door,str(root/'assets/doors'/door['id']),'open_and_traverse',0,'full','scripted_hand',randomize=False)
    plain=run_episode(job); recorder=Recorder(20); observed=run_episode(job,observer=recorder)
    for key in ['outcome','success','steps','sim_time','events','labels','criteria','door_q_end']:
        assert observed[key]==plain[key]
    assert recorder.frames[0]['time']==0
    assert recorder.frames[-1]['time']==pytest.approx(observed['sim_time'],abs=.001)
    assert all(b['time']>a['time'] for a,b in zip(recorder.frames,recorder.frames[1:]))


def test_native_and_web_export_agree(tmp_path):
    root=Path(__file__).resolve().parents[1]; door=load_manifest(str(root/'assets'))['doors'][1]
    row=record_one((door,str(root/'assets'),str(tmp_path),10))
    assert 'error' not in row
    c=json.loads((tmp_path/row['clip']).read_text())
    with np.load(tmp_path/row['trajectory'],allow_pickle=False) as data:
        addr=c['native']['qpos_addresses']; lead=round(c['lead_in_s']*c['fps'])
        np.testing.assert_allclose(np.array(c['door_q'])[lead:],data['qpos'][:,addr],atol=.000006)
        np.testing.assert_allclose(np.array(c['avatar']).reshape(-1,16,3),data['actor_joints'],atol=.00006)
        assert np.isfinite(data['body_quat']).all()
        assert data['time'][-1]==pytest.approx(c['duration']-c['lead_in_s'],abs=.001)
    assert c['outcome']['outcome']==row['outcome']


def test_recorded_world_poses_match_current_coordinates():
    import mujoco
    root=Path(__file__).resolve().parents[1]
    door=next(d for d in load_manifest(str(root/'assets'))['doors'] if d['id']=='db0079_sliding_single')
    recorder=Recorder(10)
    result=run_episode(Job(door,str(root/'assets/doors'/door['id']),'open_and_traverse',0,'full','scripted_hand',randomize=False),observer=recorder)
    assert result['error'] is None
    m=mujoco.MjModel.from_xml_path(str(root/'assets/doors'/door['id']/'door.xml')); d=mujoco.MjData(m)
    for sample in recorder.frames:
        d.qpos[:]=sample['qpos'];mujoco.mj_kinematics(m,d)
        np.testing.assert_allclose(sample['body_pos'],d.xpos,atol=1e-12)
        np.testing.assert_allclose(sample['body_quat'],d.xquat,atol=1e-12)


def test_anatomical_left_is_left_when_facing_positive_y():
    times=np.array([0.,.05]); bases=np.tile([0.,-1.,.5],(2,1)); targets=np.tile([0.,0.,1.],(2,1))
    poses,_,_,_=fit_motion(times,bases,targets,np.zeros(2,bool),bases[0])
    assert poses[0,4,0]<poses[0,7,0]  # left shoulder is negative X
    assert poses[0,10,0]<poses[0,13,0]  # left hip is negative X


def test_recorder_is_strictly_read_only_at_fallback_family_torque_samples(tmp_path):
    """A fallback-strength family previously ran live mj_forward inside the observer.

    Equal outcome summaries alone did not detect its qacc mutation. Guard the
    actual integration, controls, applied forces and derived state at every call.
    """
    root = Path(__file__).resolve().parents[1]
    door = next(d for d in load_manifest(str(root / "assets"))["doors"] if d["id"] == "db0004_bifold")
    # The fallback-strength path needs a door without a calibrated qa_push.  Since the free-swing families are pushed
    # by the sign-off QA too (the jam gate), every shipped qa.json carries one: strip it from a scratch copy of the door
    # (the MJCF resolves its meshes through ../../hardware, so the copy keeps the doors/<id> layout and links hardware).
    scratch = tmp_path / "assets"
    (scratch / "doors").mkdir(parents=True)
    shutil.copytree(root / "assets/doors" / door["id"], scratch / "doors" / door["id"])
    os.symlink(root / "assets/hardware", scratch / "hardware")
    directory = scratch / "doors" / door["id"]
    qa = json.loads((directory / "qa.json").read_text())
    qa["metrics"].pop("qa_push", None)
    (directory / "qa.json").write_text(json.dumps(qa))
    assert not json.loads((directory / "qa.json").read_text())["metrics"].get("qa_push")
    recorder = Recorder(20)
    guarded_torque_samples = []
    fields = ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied", "xpos", "xquat")

    def guarded(event, env, base, action):
        before = {name: getattr(env.d, name).copy() for name in fields}
        time_before, count_before = float(env.d.time), len(recorder.frames)
        recorder(event, env, base, action)
        assert float(env.d.time) == time_before
        for name in fields:
            np.testing.assert_array_equal(getattr(env.d, name), before[name], err_msg=f"observer changed {name} at {event} t={time_before}")
        if event == "step" and len(recorder.frames) > count_before and any(action.get("torques", {}).values()):
            guarded_torque_samples.append(time_before)
            assert np.any(np.abs(recorder.frames[-1]["tau"]) > 0), "fixture must record a real nonzero policy effort"

    outcome = run_episode(Job(door, str(directory), "open_and_traverse", 0, "full", "scripted_hand", randomize=False), observer=guarded)
    assert outcome["error"] is None, outcome["error"]
    assert guarded_torque_samples and guarded_torque_samples[0] >= .5

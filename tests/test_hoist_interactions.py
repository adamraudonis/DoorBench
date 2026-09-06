"""The baseline must use the real chain and cannot walk through its curtain."""
import json
import numpy as np

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.runner import Job,run_episode
from doorbench.reference.record import Recorder,write_native_recording,write_json,digest,NATIVE_SCHEMA
from doorbench.reference.native_validation import validate_native


def test_chain_input_and_free_body_recording_keep_a_partially_open_curtain_blocked(tmp_path):
    spec=next(s for s in generate_all() if s['index']==419)
    assets=tmp_path/'assets';out=tmp_path/'reference-motions'
    summary=export_door(spec,str(assets/'doors'),str(assets/'hardware'),formats=('json','mjcf'))
    path=assets/'doors'/spec['id'];row={'id':spec['id'],'family':spec['family'],'benchmark':summary['benchmark']}
    write_json(assets/'manifest.json',{'doors':[row]})
    env=DoorEnv(str(path));env.reset(randomize=False)
    assert env.start_pose['xy'][1]>0
    assert env.tracker.passage.intervals(env.d)==[]
    assert not env._door_clear_now()
    assert env.spec['benchmark']['primary_scenario']=='open_and_traverse'
    assert {s['name'] for s in env.spec['benchmark']['scenarios']}=={'open_and_traverse','close_only'}
    for scenario in env.spec['benchmark']['scenarios']:
        assert len(scenario['handle_targets'])==3
        assert scenario['handle_targets'][-1]=='hoist_keeper_grip'
        assert all(s.startswith('hoist_chain_grip_') for s in scenario['handle_targets'][:2])
        if scenario['name']!='close_only':assert scenario['expected_transit_terms']['open_s']>9
        assert scenario['start']['center'][1]>0
        assert scenario['pass_plane']['traverse_direction']==[0.,-1.,0.]
        if scenario['goal']:assert scenario['goal']['center'][1]<0
    from doorbench.benchmark.interactions import ContactSites
    primary=env.meta['primary_joint']
    assert ContactSites(env).select(primary) is not None
    env.spec['robot']['approach_side']='-y'
    assert ContactSites(env).select(primary) is None
    env.spec['robot']['approach_side']='+y'
    env.close()
    rec=Recorder(5)
    episode=run_episode(Job(row,str(path),'open_and_traverse',0,'full','scripted_hand',
        randomize=False,time_budget_s=8,wall_timeout_s=300),observer=rec)
    assert not episode.get('error'),episode
    assert not episode['success'] and not episode['labels']['robot_passed_through']
    assert not episode['labels']['door_open_clear']
    assert all(f['base'][1]>0 for f in rec.frames)
    assert .01<episode['door_q_end']<1.8
    result=write_native_recording(row,path,out,rec,episode)
    write_json(out/'index.json',{'schema':NATIVE_SCHEMA,'clips':[result],'manifest_sha256':digest(assets/'manifest.json')})
    checked=validate_native(out,assets)
    assert checked['checks'][0]['frames_checked']==result['frames']
    clip=json.loads((out/result['clip']).read_text())
    contacts=[c for frame in clip['oracle_contacts'] for c in frame if np.linalg.norm(c.get('force_N',[0,0,0]))>1e-9]
    assert contacts and all(c['site'].startswith('hoist_chain_grip_') or c['site']=='hoist_keeper_grip' for c in contacts)
    assert any(c['site']=='hoist_keeper_grip' for c in contacts)
    assert max(np.linalg.norm(c['force_N']) for c in contacts)<=120.+1e-8
    assert len({c['site'] for c in contacts})>1
    native=clip['native'];free=native['joint_types'].index('mjJNT_FREE')
    assert native['qpos_widths'][free]==7 and native['qvel_widths'][free]==6
    assert len(native['poses'][0])==7*len(native['body_names'])
    assert 'avatar' not in clip

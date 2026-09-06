"""Native regressions for reported wrong-side / wrong-hardware interactions."""
import json
from pathlib import Path

import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.runner import AutoDoorSensor, Job, run_episode
from doorbench.reference.record import Recorder


@pytest.fixture(scope='module')
def doors(tmp_path_factory):
    root = tmp_path_factory.mktemp('mechanical-contacts')
    specs = {s['index']: s for s in generate_all()}
    rows = {}
    for i in (5, 8, 9, 11, 12, 14, 16, 18):
        s = specs[i]
        summary = export_door(s, str(root/'doors'), str(root/'hardware'), formats=('mjcf','json'))
        rows[i] = (root/'doors'/s['id'], {'id':s['id'],'family':s['family'],'benchmark':summary['benchmark']})
    return rows


def record(fixture, scenario='open_and_traverse'):
    path, row = fixture
    recorder = Recorder(20)
    result = run_episode(Job(row, str(path), scenario, 0, 'full', 'scripted_hand', randomize=False), observer=recorder)
    assert result.get('error') is None
    return result, recorder


@pytest.mark.parametrize('index,site', [(12,'leaf_pull_grip_n'), (16,'leaf_far_pull_grip_n')])
def test_unlocked_door_uses_real_approach_pull_not_lock(doors, index, site):
    result, rec = record(doors[index])
    assert result['success']
    contacts = [f for f in rec.frames if f['active']]
    assert contacts and {f['target_site'] for f in contacts} == {site}
    assert {f['target_joint'] for f in contacts} == {'leaf_hinge'}
    # No hidden generalized torque on the already withdrawn bolt/turn.
    for i, name in enumerate(rec.info['joint_names']):
        if 'bolt' in name or 'thumbturn' in name:
            address = rec.info['qvel_addresses'][i]
            assert max(abs(f['tau'][address]) for f in rec.frames) == 0


def test_labels_account_for_applied_work_before_efforts_are_cleared(doors):
    env=DoorEnv(str(doors[16][0]));env.reset(randomize=False)
    dof=int(env.m.jnt_dofadr[env.pj]);env.d.qvel[dof]=.2;env.d.qfrc_applied[dof]=3.
    env.step()
    assert env.tracker.L.energy_J>0
    assert not np.any(env.d.qfrc_applied)
    env.close()


def test_breakaway_friction_is_not_labeled_an_unreleasable_lock():
    from doorbench.benchmark.scenarios import assign_scenarios
    from doorbench.build import build_model
    from doorbench.physics import derive
    examples=[s for s in generate_all() if s['lock']['model']=='jam_stuck']
    assert len(examples)==12
    for spec in examples:
        assert not spec['lock']['engaged']
        assert 'locked_recognize' not in assign_scenarios(spec)
        phys=derive(spec);model=build_model(spec,phys)
        leaf=next(b for b in model.bodies if b.joint and b.joint.name==model.meta['primary_joint'])
        assert leaf.joint.range[1]>.5
        assert leaf.joint.frictionloss>=2*phys['hinge']['stick_torque_Nm']


def test_wall_button_requires_press_then_bar_retracts_before_swing(doors):
    path, _ = doors[11]
    env = DoorEnv(str(path)); env.reset(randomize=False)
    sensor = AutoDoorSensor(env)
    # Standing in front of a knowing-act button must not impersonate pressing it.
    for _ in range(300):
        sensor.step([0,-.3], float(env.d.time)); env.step()
    assert abs(float(env.d.qpos[env.m.jnt_qposadr[env.pj]])) < .01
    assert not np.any(env.d.ctrl)
    env.close()
    result, rec = record(doors[11]); assert result['success']
    addr = dict(zip(rec.info['joint_names'],rec.info['qpos_addresses']))
    bar = [f['qpos'][addr['leaf_exit_device_slide']] for f in rec.frames]
    assert max(bar) > .0128
    first_open = next(f for f in rec.frames if f['qpos'][addr['leaf_hinge']] > .05)
    assert first_open['qpos'][addr['leaf_latch_bolt_slide']] > .8*.016
    assert {f['target_site'] for f in rec.frames if f['active']} == {'activation_button_n_push'}
    source = json.loads((path/'model.json').read_text())
    assert not any(g['name'] == 'wall_reader' for b in source['bodies'] for g in b.get('geoms', []))


def test_bypass_selects_one_panel_and_its_actual_cup(doors):
    result, rec = record(doors[8])
    assert result['success'], result
    meta=json.loads((doors[8][0]/'model.json').read_text())['meta']
    controls=meta['sliding_leaf_controls']
    addr=dict(zip(rec.info['joint_names'],rec.info['qpos_addresses']))
    assert max(f['qpos'][addr[controls[0]['joint']]] for f in rec.frames)>.5
    for c in controls[1:]:
        assert max(abs(f['qpos'][addr[c['joint']]]) for f in rec.frames)<1e-4
    active=[f for f in rec.frames if f['active']]
    assert active
    assert {f['target_site'] for f in active} <= set(controls[0]['grip_sites'])


def test_bifold_opens_banks_sequentially_for_closet_access(doors):
    result, rec = record(doors[9], 'open_only')
    assert result['success'], result
    meta=json.loads((doors[9][0]/'model.json').read_text())['meta']
    banks=meta['folding_banks']
    assert len(banks)==2
    addr=dict(zip(rec.info['joint_names'],rec.info['qpos_addresses']))
    first=addr[banks[0]['pivot_joint']]; second=addr[banks[1]['pivot_joint']]
    second_starts=next(f for f in rec.frames if abs(f['qpos'][second])>.05)
    assert abs(second_starts['qpos'][first]) > .90*abs(banks[0]['open_q'])
    # These folded banks leave <0.5 m between their knobs: a standing
    # traversal must fail honestly instead of inheriting an angle-only pass.
    assert not rec.env.tracker.passage.intervals(rec.env.d)
    assert not result['labels']['robot_passed_through']
    s=json.loads((doors[9][0]/'spec.json').read_text())
    assert s['benchmark']['primary_scenario']=='open_only'
    assert all('traverse' not in c['name'] for c in s['benchmark']['scenarios'])


def test_pocket_opening_hands_off_before_cup_enters_wall(doors):
    result, rec = record(doors[18]); assert result['success'], result
    meta=json.loads((doors[18][0]/'model.json').read_text())['meta']
    p=meta['pocket_edge_pull']
    active=[f for f in rec.frames if f['active']]
    assert active
    addr=dict(zip(rec.info['joint_names'],rec.info['qpos_addresses']))
    cups=set(meta['sliding_leaf_controls'][0]['grip_sites'])
    face=[f for f in active if f['target_site'] in cups]
    edge=[f for f in active if f['target_site']==p['final_push_site']]
    assert face and edge
    assert max(f['qpos'][addr[p['leaf_joint']]] for f in face)<p['face_cup_occlusion_q']
    assert face[0]['time']<edge[0]['time']
    assert all(f['site_forces'] for f in edge)
    assert max(f['qpos'][addr[p['leaf_joint']]] for f in edge)>p['recessed_leaf_q']-.02
    assert max(abs(f['qpos'][addr[p['joint']]]) for f in rec.frames)<.05


def test_fully_recessed_pocket_requires_press_grasp_extract_then_face_pull(doors):
    result, rec=record(doors[18], 'close_only')
    assert result['success'],result
    meta=json.loads((doors[18][0]/'model.json').read_text())['meta'];p=meta['pocket_edge_pull']
    addr=dict(zip(rec.info['joint_names'],rec.info['qpos_addresses']))
    leaf=addr[p['leaf_joint']];edge=addr[p['joint']]
    assert abs(rec.frames[0]['qpos'][leaf]-p['recessed_leaf_q'])<1e-6
    press=[f for f in rec.frames if f['active'] and f['target_site']==p['press_site']]
    extract=[f for f in rec.frames if f['active'] and f['target_site']==p['grip_site']]
    face=[f for f in rec.frames if f['active'] and f['target_site'] in meta['sliding_leaf_controls'][0]['grip_sites']]
    assert press and extract and face
    assert press[0]['time']<extract[0]['time']<face[0]['time']
    assert extract[0]['qpos'][edge] >= p['minimum_grasp_q']-.02
    assert p['recessed_leaf_q']-face[0]['qpos'][leaf]>=p['face_grip_after_extract_m']
    assert all(f['site_forces'] for f in press+extract)


def test_stale_effort_sidecar_cannot_change_strength_or_reset_live_pose(doors, tmp_path):
    from doorbench.benchmark.runner import qa_push_for
    path,_=doors[18]
    env=DoorEnv(str(path));env.reset(scenario='close_only',randomize=False)
    before=env.d.qpos.copy()
    expected=qa_push_for(str(tmp_path),env)
    (tmp_path/'qa.json').write_text(json.dumps({'metrics':{'qa_push':1e9},'source_sha256':{'door.xml':'stale'}}))
    assert qa_push_for(str(tmp_path),env)==expected<4001
    np.testing.assert_array_equal(env.d.qpos,before)
    assert env.d.time==0
    env.close()


def test_site_force_cannot_use_far_face_or_exceed_joint_limits(doors):
    from doorbench.benchmark.site_forces import SiteForces
    from doorbench.benchmark.runner import torque_limits
    env=DoorEnv(str(doors[18][0]));env.reset(randomize=False)
    limits=torque_limits(env,str(doors[18][0]));forces=SiteForces(env,limits)
    far=next(s for s in env.meta['sliding_leaf_controls'][0]['grip_sites'] if s.endswith('_p'))
    with pytest.raises(ValueError,match='approach-side'):
        forces.generalized(env.d,{far:[1e6,0,0]})
    p=env.meta['pocket_edge_pull']
    tau=forces.generalized(env.d,{p['press_site']:[1e6,0,0],p['grip_site']:[1e6,0,0]})
    assert np.isfinite(tau).all()
    assert all(abs(tau[dof])<=limit+1e-8 for dof,limit in forces.limits.items())
    env.close()


def test_generalized_effort_and_badge_cannot_bypass_a_far_side_manual_lock(doors):
    from doorbench.benchmark.runner import torque_limits
    for index,joint in ((12,'leaf_aux_bolt_slide'),(16,'leaf_deadbolt_thumbturn_hinge')):
        path,_=doors[index];env=DoorEnv(str(path));env.reset(randomize=False)
        assert torque_limits(env,str(path))[joint]==0
        before={k:getattr(env.d,k).copy() for k in ('qpos','qvel','ctrl','eq_active')}
        ranges=env.m.jnt_range.copy()
        assert env.badge() is False and not env.unlocked_by_env
        for k,v in before.items():np.testing.assert_array_equal(getattr(env.d,k),v)
        np.testing.assert_array_equal(env.m.jnt_range,ranges)
        assert not env.tracker.L.lock_released
        env.close()


def test_inward_storing_garages_do_not_claim_access_to_rear_only_locks():
    from doorbench.benchmark.scenarios import assign_scenarios
    specs=[s for s in generate_all() if s['family'] in ('garage_tiltup','garage_sectional','rollup')]
    assert len(specs)==40
    for s in specs:
        inside=s['family']=='rollup' and s['kinematics'].get('opener')=='chain_hoist'
        assert s['robot']['robot_outside']==(not inside)
        assert s['robot']['approach_side']==('+y' if inside else '-y')
        if s['lock']['model'] in ('garage_slide_lock','keyed_cylinder','padlock'):
            assert not s['lock']['robot_side_release']
            if s['lock']['engaged']:assert assign_scenarios(s)[0]=='locked_recognize'


def test_grasp_wrench_needs_authored_permission_and_records_effective_load(doors):
    from doorbench.benchmark.site_forces import SiteForces
    from doorbench.benchmark.runner import torque_limits
    env=DoorEnv(str(doors[16][0]));env.reset(randomize=False)
    site='leaf_far_pull_grip_n';limits=torque_limits(env,str(doors[16][0]))
    forces=SiteForces(env,limits)
    with pytest.raises(ValueError,match='grasp-wrench'):
        forces.generalized(env.d,{}, {site:[0,0,.02]})
    env.meta['site_wrench_limits_Nm']={site:.02};forces=SiteForces(env,limits)
    tau,resolved=forces.resolve(env.d,{site:[1e6,1e6,0]}, {site:[0,0,1e6]})
    force=np.array(resolved[site]['force_N']);torque=np.array(resolved[site]['torque_Nm'])
    assert np.linalg.norm(force)<=120.+1e-8 and np.linalg.norm(torque)<=.02+1e-12
    expected=np.zeros(env.m.nv);sid=env.m.site(site).id
    env.mj.mj_applyFT(env.m,env.d,force,torque,env.d.site_xpos[sid],env.m.site_bodyid[sid],expected)
    np.testing.assert_allclose(tau,expected,atol=1e-10)
    env.close()


def test_native_initialization_cost_does_not_consume_episode_timeout(doors,monkeypatch):
    from doorbench.benchmark import runner
    original=DoorEnv.reset;clock=runner.time.time;offset=[0.]
    def reset(self,*args,**kwargs):
        value=original(self,*args,**kwargs);offset[0]+=10000.;return value
    monkeypatch.setattr(DoorEnv,'reset',reset)
    monkeypatch.setattr(runner.time,'time',lambda:clock()+offset[0])
    result,_=record(doors[16])
    assert result['success'] and result['steps']>0
    assert result['initialization_wall_s']>=10000.

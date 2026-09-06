"""Actual wall-switch stock, contact strokes, side access and relay ordering."""
import json
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.wall_switch_qa import run_wall_switch_qa
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.interactions import ContactSites
from doorbench.benchmark.runner import AutoDoorSensor, torque_limits


@pytest.fixture(scope='module')
def doors(tmp_path_factory):
    root=tmp_path_factory.mktemp('wall-switches');rows=[]
    for spec in generate_all():
        if not set(spec['extras']).intersection(('rex_button','call_button')):continue
        summary=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        path=root/'doors'/spec['id']
        meta=json.loads((path/'model.json').read_text())['meta']
        rows.append((spec,path,meta,summary['files']['mjcf']))
    assert len(rows)==26
    return rows


def test_all_switches_have_prepared_stock_and_two_native_cycles_in_every_tier(doors):
    for spec,path,meta,files in doors:
        for tier in ('full','simple','minimal'):
            m=mujoco.MjModel.from_xml_path(files[tier])
            result=run_wall_switch_qa(m,meta)
            assert result['ok'],(spec['id'],tier,result)
            assert len(result['measurements'])==1
            env=DoorEnv(str(path),tier=tier);env.reset(randomize=False)
            row=meta['wall_switches'][0]
            accessible=row['kind']=='call_button' or not spec['robot']['robot_outside']
            assert row['accessible_from_robot']==accessible
            assert (ContactSites(env).select(row['joint']) is not None)==accessible
            assert (torque_limits(env,str(path))[row['joint']]>0)==accessible
            env.close()


def test_refilled_stem_guide_detached_plate_and_off_surface_site_fail(doors):
    _,_,meta,files=doors[0];r=meta['wall_switches'][0]
    for defect in ('filled_guide','detached_plate','off_surface'):
        m=mujoco.MjModel.from_xml_path(files['full']);d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
        if defect=='filled_guide':
            plate=m.geom(r['plate_geoms'][0]).id;stem=m.geom(r['stem_geom']).id;b=m.geom_bodyid[plate]
            m.geom_pos[plate]=d.xmat[b].reshape(3,3).T@(d.geom_xpos[stem]-d.xpos[b])
            m.geom_size[plate]=[.01,.006,.01]
        elif defect=='detached_plate':m.geom_pos[m.geom(r['plate_geoms'][0]).id,1]+=.03
        else:m.site_pos[m.site(r['site']).id,2]+=.1
        assert not run_wall_switch_qa(m,meta)['ok'],defect


@pytest.mark.parametrize('kind',('call_button','rex_button'))
def test_only_physical_button_stroke_releases_waiting_system(doors,kind):
    spec,path,meta,_=next(row for row in doors if row[0]['lock']['engaged'] and
        row[2]['wall_switches'][0]['kind']==kind and row[2]['wall_switches'][0]['accessible_from_robot'])
    env=DoorEnv(str(path));env.reset(randomize=False);m,d=env.m,env.d
    sensor=AutoDoorSensor(env);r=meta['wall_switches'][0];j=env._jid(r['joint']);q=m.jnt_qposadr[j]
    for _ in range(round(.6/m.opt.timestep)):
        sensor.step([0,-.3],float(d.time));env.step()
    assert not env.tracker.L.lock_released
    assert abs(d.qpos[m.jnt_qposadr[env.pj]])<.01
    assert not np.any(d.ctrl)
    crossed=False
    for _ in range(round(.3/m.opt.timestep)):
        before=float(d.qpos[q]);released=env.tracker.L.lock_released
        env.apply_site_force(r['site'],[0,-r['face']*20,0])
        sensor.step([0,-.3],float(d.time));env.step()
        if max(before,float(d.qpos[q]))>.8*r['travel_m']:crossed=True
        if kind!='call_button' and env.tracker.L.lock_released and not released:
            assert max(before,float(d.qpos[q]))>.8*r['travel_m'];crossed=True
    assert crossed
    if kind=='call_button':
        for _ in range(round(3./m.opt.timestep)):
            sensor.step([0,-.3],float(d.time));env.step()
            if d.qpos[m.jnt_qposadr[env.pj]]>.01:
                assert all(d.qpos[m.jnt_qposadr[env._jid(row['hook_joint'])]]>=row['released_angle_rad']
                           for row in meta['elevator_interlocks']['leaves'])
        assert d.qpos[m.jnt_qposadr[env.pj]]>.05
    assert env.tracker.L.lock_released
    assert not np.any(d.warning.number)
    env.close()

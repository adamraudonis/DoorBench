"""Supported activation caps, native strokes and physical activation ordering."""
import json

import mujoco
import numpy as np
import pytest

from doorbench.build import build_model, export_door
from doorbench.spec import generate_all
from doorbench.wall_switch_qa import run_wall_switch_qa
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.interactions import ContactSites
from doorbench.benchmark.runner import AutoDoorSensor


IDS={11,136,153,175,225,433,474,629,801,829,939}


@pytest.fixture(scope='module')
def doors(tmp_path_factory):
    root=tmp_path_factory.mktemp('activation-pads');rows={}
    for spec in generate_all():
        if spec['family'] not in ('automatic_swing','automatic_sliding','garage_sectional'):continue
        model=build_model(spec)
        if not model.meta.get('automatic_activation',{}).get('buttons'):continue
        summary=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        path=root/'doors'/spec['id'];source=json.loads((path/'model.json').read_text())
        rows[spec['index']]=(spec,path,source,summary['files']['mjcf'])
    assert set(rows)==IDS
    return rows


def test_every_activation_pad_has_supported_stock_and_native_cycles_in_every_tier(doors):
    for spec,path,source,files in doors.values():
        meta=source['meta'];buttons=meta['automatic_activation']['buttons']
        switches=[r for r in meta['wall_switches'] if r['kind'].startswith('activation_button_')]
        assert len(buttons)==len(switches)==2
        for r in switches:
            tag='n' if r['face']<0 else 'p';name='activation_button_'+tag
            assert r['joint']==name+'_slide' and r['site']==name+'_push'
            assert next(b for b in buttons if b['joint']==r['joint'])=={
                'joint':r['joint'],'site':r['site'],'face':r['face'],'threshold_m':.002}
            body=next(b for b in source['bodies'] if b['name']==r['body'])
            assert body['joint']['role']=='operator'
            assert body['joint']['robot_interactive']==(r['face']<0)
            assert r['body'] in meta['mechanism_mass_bodies']
        for tier in ('full','simple','minimal'):
            m=mujoco.MjModel.from_xml_path(files[tier]);result=run_wall_switch_qa(m,meta)
            (path/('activation-proof-'+tier+'.json')).write_text(json.dumps(result,indent=2))
            assert result['ok'],(spec['id'],tier,result)
            assert m.opt.timestep<=.0005
            env=DoorEnv(str(path),tier=tier);env.reset(randomize=False)
            try:
                sites=ContactSites(env)
                for r in switches:
                    assert (sites.select(r['joint']) is not None)==(r['face']<0)
                    stem=m.geom(r['stem_geom']).id
                    assert m.geom_size[stem,0]==pytest.approx(.004,abs=1e-8)
                    assert m.geom_size[stem,1]==pytest.approx(.0045,abs=1e-8)
            finally:env.close()


@pytest.mark.parametrize('index',(11,153,175))
def test_actual_site_force_precedes_drive_and_native_opening(doors,index):
    _,path,source,_=doors[index];env=DoorEnv(str(path));env.reset(randomize=False)
    try:
        m,d=env.m,env.d;sensor=AutoDoorSensor(env)
        r=next(b for b in source['meta']['automatic_activation']['buttons'] if b['face']<0)
        address=int(m.jnt_qposadr[env._jid(r['joint'])]);primary=int(m.jnt_qposadr[env.pj])
        for _ in range(round(.2/m.opt.timestep)):
            sensor.step([0,-.3],float(d.time));env.step()
        assert not np.any(d.ctrl)
        assert abs(d.qpos[primary])<.01
        crossed=False;commanded=False;max_primary=abs(float(d.qpos[primary]))
        for tick in range(round(1.2/m.opt.timestep)):
            if tick*m.opt.timestep<.3:
                env.apply_site_force(r['site'],[0,-r['face']*20.,0])
            crossed=crossed or d.qpos[address]>=r['threshold_m']
            sensor.step([0,-.3],float(d.time))
            if np.any(d.ctrl):
                assert crossed,'Drive command before measured physical button threshold'
                commanded=True
            env.step();max_primary=max(max_primary,abs(float(d.qpos[primary])))
            if index==11 and abs(d.qpos[primary])>.05:
                latch=int(m.jnt_qposadr[env._jid('leaf_latch_bolt_slide')])
                assert d.qpos[latch]>.8*.016
        assert crossed and commanded and max_primary>.05
        assert not np.any(d.warning.number)
    finally:env.close()


def test_activation_gate_rejects_refilled_guide_and_detached_plate(doors):
    _,_,source,files=doors[153];meta=source['meta'];r=meta['wall_switches'][0]
    for defect in ('guide_filled','plate_detached'):
        m=mujoco.MjModel.from_xml_path(files['full']);d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
        g=m.geom(r['plate_geoms'][0]).id
        if defect=='guide_filled':
            stem=m.geom(r['stem_geom']).id;b=m.geom_bodyid[g]
            m.geom_pos[g]=d.xmat[b].reshape(3,3).T@(d.geom_xpos[stem]-d.xpos[b])
            m.geom_size[g]=[.01,.006,.01]
        else:m.geom_pos[g,1]+=.03
        assert not run_wall_switch_qa(m,meta)['ok'],defect


def test_scripted_control_rate_really_depresses_every_available_wall_button(doors):
    from doorbench.benchmark.runner import Job,run_episode
    from doorbench.reference.record import Recorder
    for spec,path,source,_ in doors.values():
        if spec['lock']['engaged'] and not spec['lock'].get('robot_side_release'):continue
        row={'id':spec['id'],'family':spec['family']};rec=Recorder(60)
        result=run_episode(Job(row,str(path),'open_and_traverse',0,'full','scripted_hand',
            randomize=False,time_budget_s=2.,wall_timeout_s=120.),observer=rec)
        assert not result.get('error') and result['outcome']!='native_failure',result
        q=rec.info['qpos_addresses'][rec.info['joint_names'].index('activation_button_n_slide')]
        pressed=[f for f in rec.frames if f['qpos'][q]>=.002]
        if (spec['kinematics'].get('actuator') or {}).get('powered') is False:
            # Manual fallback operates the installed leaf hardware. A dead
            # electrical wall station is not another door handle.
            assert not pressed
            assert not any(c['site']=='activation_button_n_push' for f in rec.frames for c in f['contact_sites'])
            assert not any(np.any(f['ctrl']) for f in rec.frames)
            continue
        assert pressed,(spec['id'],max(f['qpos'][q] for f in rec.frames))
        driven=[f for f in rec.frames if np.any(f['ctrl'])]
        assert driven and driven[0]['time']>=pressed[0]['time']-1/60

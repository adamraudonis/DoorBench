"""Closed keeper release must use real floor support, not preload a seated pin."""
import json
import mujoco
import numpy as np
import pytest
from test_hoist_keeper import fixtures, SPECS
from doorbench.rollup_hoist import compile_hoist,hoist_control
from doorbench.hoist_keeper import (compile_keeper,begin_keeper_transition,
    keeper_transition_action,keeper_pin_load,_closed_floor_support)
from doorbench.native_warnings import capture_native_warnings


@pytest.fixture(autouse=True)
def no_uncounted_native_messages():
    with capture_native_warnings() as messages:
        yield
    assert not messages,messages


def load(fixtures,door='db0419_rollup'):
    ir,_,path=fixtures[door]
    m=mujoco.MjModel.from_xml_path(str(path/'door.xml'));d=mujoco.MjData(m)
    mujoco.mj_forward(m,d)
    return m,d,compile_hoist(m,ir.meta),compile_keeper(m,ir.meta),path


def apply(m,d,forces):
    d.qfrc_applied[:]=0
    for name,force in forces.items():
        assert np.linalg.norm(force)<=120.+1e-9
        sid=m.site(name).id
        mujoco.mj_applyFT(m,d,np.array(force),np.zeros(3),d.site_xpos[sid],m.site_bodyid[sid],d.qfrc_applied)
    mujoco.mj_step(m,d)


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_original_closed_release_uses_floor_then_actual_unloading_if_needed(fixtures,spec):
    m,d,h,k,path=load(fixtures,spec['id'])
    s=begin_keeper_transition(m,d,h,k,mode='release')
    assert s['phase']=='settle_on_floor' and s['initial_floor_reaction_N']>100.
    peak=depth=load_peak=chain_peak=0.;initial=float(d.site_xpos[h.bottom_site,2])
    for _ in range(round(4./m.opt.timestep)):
        mujoco.mj_forward(m,d)
        a=keeper_transition_action(m,d,h,k,s);s=a['next_state']
        assert not a['failed'],a['reason']
        peak=max(peak,float(np.linalg.norm(a['site_forces'].get(k.grip_name,[0,0,0]))))
        chain_peak=max(chain_peak,max((np.linalg.norm(f) for n,f in a['site_forces'].items() if n!=k.grip_name),default=0.))
        if s['release_support']=='measured_floor':assert set(a['site_forces'])<={k.grip_name}
        load_peak=max(load_peak,keeper_pin_load(m,d,k))
        if a['done']:break
        apply(m,d,a['site_forces'])
        depth=max(depth,max((-float(c.dist) for c in d.contact),default=0.))
    assert s['done']
    if spec['index'] in (313,636):
        assert s['reason']=='actual_pin_withdrawn_after_chain_unload'
        assert s['floor_handoff_pin_load_N']>=5. and s['unloaded_pin_load_N']<5.
        assert 0.<chain_peak<=120.
    else:
        assert s['reason']=='actual_pin_withdrawn_under_measured_floor_support'
        assert chain_peak==0. and load_peak<5.
    assert d.qpos[k.qpos]>.078 and abs(d.site_xpos[h.bottom_site,2]-initial)<.003
    assert peak<20. and depth<.001
    assert not any(w.number for w in d.warning)
    (path/'floor-release.json').write_text(json.dumps({'door_id':spec['id'],'ok':True,
        'elapsed_native_s':float(d.time),'peak_keeper_force_N':float(peak),
        'peak_chain_force_N':float(chain_peak),'transition':s,
        'peak_pin_chain_load_N':float(load_peak),'max_penetration_m':depth,
        'initial_floor_reaction_N':s['initial_floor_reaction_N'],
        'final_bottom_z_m':float(d.site_xpos[h.bottom_site,2]),'keeper_q_m':float(d.qpos[k.qpos]),
        'scope':'Original-source closed-state mechanical release: measured floor support, plus actual chain unloading if weak-source slack loads the pin; no human task claim.'},indent=2)+'\n')


def test_closed_height_without_actual_floor_cannot_bypass_chain_unload(fixtures):
    m,d,h,k,_=load(fixtures)
    floor=m.geom('floor').id;m.geom_contype[floor]=m.geom_conaffinity[floor]=0
    mujoco.mj_forward(m,d)
    assert abs(d.site_xpos[h.bottom_site,2]-h.closed_z)<.001
    assert _closed_floor_support(m,d,h)==(False,0.)
    s=begin_keeper_transition(m,d,h,k,mode='release')
    assert s['phase']=='unload'
    a=keeper_transition_action(m,d,h,k,s)
    assert k.grip_name not in a['site_forces']


def test_loss_of_floor_support_aborts_with_no_hidden_hands(fixtures):
    m,d,h,k,_=load(fixtures)
    s=begin_keeper_transition(m,d,h,k,mode='release')
    assert s['phase']=='settle_on_floor'
    floor=m.geom('floor').id;m.geom_contype[floor]=m.geom_conaffinity[floor]=0
    mujoco.mj_forward(m,d)
    a=keeper_transition_action(m,d,h,k,s)
    assert a['failed'] and not a['done'] and a['site_forces']=={}
    assert a['reason']=='measured_floor_support_lost_during_keeper_withdrawal'


def test_actual_loaded_pin_cannot_use_floor_release_branch(fixtures):
    m,d,h,k,_=load(fixtures)
    peak=0.
    for _ in range(round(.8/m.opt.timestep)):
        mujoco.mj_forward(m,d)
        a=hoist_control(m,d,h,elapsed_s=float(d.time))
        apply(m,d,{a['site']:[0.,0.,-100.*min(1.,float(d.time)/.5)]})
        peak=max(peak,keeper_pin_load(m,d,k))
    mujoco.mj_forward(m,d)
    assert peak>50. and keeper_pin_load(m,d,k)>5.
    s=begin_keeper_transition(m,d,h,k,mode='release')
    assert s['phase']=='unload'
    # Even a stale floor-release state must recheck the actual pin load.
    s.update(phase='withdraw_from_floor')
    a=keeper_transition_action(m,d,h,k,s)
    assert not a['done'] and a['phase']=='unload'
    assert a['next_state']['handoff_keeper_q_m']==pytest.approx(max(0.,d.qpos[k.qpos]))
    assert set(a['site_forces'])=={k.grip_name}  # Hold; do not continue withdrawal.

"""Native open initialization must return a held state, not a continuing hand."""
import copy
import hashlib
import json
import mujoco
import numpy as np
import pytest
from test_rollup_hoist import fixtures
from doorbench.rollup_hoist import prepare_hoist_open,compile_hoist
from doorbench.hoist_keeper import compile_keeper
from doorbench.native_warnings import capture_native_warnings


def test_initializer_without_physical_keeper_cannot_return_unheld_open_state(fixtures):
    ir,_,path=fixtures['db0419_rollup']
    m=mujoco.MjModel.from_xml_path(str(path/'door.xml'))
    meta=copy.deepcopy(ir.meta);del meta['rollup_hoist']['keeper']
    # This source still has a keeper dynamic root, so unsupported assembly
    # rejection is equally valid; neither path may invent held coordinates.
    result=prepare_hoist_open(m,meta,time_limit_s=.001)
    assert not result['ok'] and 'qpos' not in result and result['elapsed_native_s']==0.
    assert result['reason'] in ('physical_keeper_required_for_hands_free_open_state',
        'additional_dynamic_bodies_require_door_only_initialization')


def test_counter_free_native_warning_is_retained_and_cannot_produce_state(fixtures,monkeypatch):
    ir,_,path=fixtures['db0419_rollup'];m=mujoco.MjModel.from_xml_path(str(path/'door.xml'))
    step=mujoco.mj_step;previous=mujoco.get_mju_user_warning()
    def warned_step(model,data):
        step(model,data)
        mujoco.get_mju_user_warning()('Linesearch objective is not convex')
    monkeypatch.setattr(mujoco,'mj_step',warned_step)
    result=prepare_hoist_open(m,ir.meta,time_limit_s=.0015)
    assert not result['ok'] and result['reason']=='native_solver_message'
    assert result['native_warning_messages']==['Linesearch objective is not convex']
    assert not any(result['warnings']) and 'qpos' not in result and 'qvel' not in result
    assert result['native_warning_events'][0]['time_s']==pytest.approx(m.opt.timestep)
    assert mujoco.get_mju_user_warning() is previous
    cached=prepare_hoist_open(m,ir.meta,time_limit_s=.0015)
    assert cached['cache_hit'] and not cached['ok'] and 'qpos' not in cached


def test_full_implicit_half_step_initializer_actual_release_open_and_two_seconds_without_hands(fixtures):
    ir,_,path=fixtures['db0419_rollup']
    m=mujoco.MjModel.from_xml_path(str(path/'door.xml'))
    # Explicit numerical convergence fixture. Production integrator promotion
    # requires its separate source proof; mass/contact/force parameters stay.
    m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICIT
    m.opt.timestep*=.5
    q0=m.qpos0.copy();mass=m.body_mass.copy();friction=m.dof_frictionloss.copy()
    result=prepare_hoist_open(m,ir.meta,time_limit_s=90.)
    (path/'keeper-open-initialization.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['ok'],{k:v for k,v in result.items() if k!='trace'}
    assert result['reason']=='native_open_state_reached_hands_free_with_keeper_seated'
    assert result['hands_free_hold_s']>=2. and result['peak_force_N']<=120.
    assert result['native_warning_messages']==[]
    assert result['hands_free_support_kind'] in ('positive_keeper','fixed_upper_stops_with_seated_keeper')
    assert [r['phase'] for r in result['transitions']]==['release','full_open_dwell','engage']
    final=result['transitions'][-1]['state']
    assert final['reason']=='positive_pin_two_second_hands_free_hold'
    hold=[r for r in result['trace'] if r['time_s']>final['phase_start_s']]
    assert len(hold)>=19 and all(r['site_forces']=={} for r in hold)
    assert result['max_penetration_m']<.001 and result['max_loop_residual_m']<.001
    assert result['max_gear_residual_rad']<.005 and not any(result['warnings'])
    assert np.array_equal(m.qpos0,q0) and np.array_equal(m.body_mass,mass) and np.array_equal(m.dof_frictionloss,friction)
    h=compile_hoist(m,ir.meta);k=compile_keeper(m,ir.meta)
    d=mujoco.MjData(m);d.qpos[:]=result['qpos'];d.qvel[:]=result['qvel'];mujoco.mj_forward(m,d)
    assert d.site_xpos[h.bottom_site,2]>=h.open_z-.04 and d.qpos[k.qpos]<.002
    start=float(d.site_xpos[h.bottom_site,2])
    # Independent continuation has no controller, actuator or applied force.
    with capture_native_warnings() as messages:
        for _ in range(round(.5/m.opt.timestep)):mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
    assert not messages,messages
    assert abs(d.site_xpos[h.bottom_site,2]-start)<.005
    assert not any(w.number for w in d.warning)
    cached=prepare_hoist_open(m,ir.meta,time_limit_s=90.)
    assert cached['cache_hit'] and cached['qpos']==result['qpos'] and cached['qvel']==result['qvel']

"""Real solver buffers are copied before any endpoint recomputation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('doorbench.dexterous.native_transition_audit',Path(__file__).resolve().parents[1]/'doorbench/dexterous/native_transition_audit.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


def fixture():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002" gravity="0 0 -9.81"/>
    <worldbody><geom type="plane" size="2 2 .1"/>
    <body name="robot/lh_palm" pos="0 0 .099"><freejoint/>
    <geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    return SimpleNamespace(m=m,d=d)


def test_copied_actual_contact_solution_survives_forward_recomputation():
    sim=fixture();m,d=sim.m,sim.d
    d.qvel[2]=-1.;mujoco.mj_forward(m,d)
    pre=d.xpos.copy();mujoco.mj_step(m,d)
    np.testing.assert_array_equal(d.xpos,pre)
    loads,raw=module._contact_solution(sim)
    archived=json.dumps(raw,sort_keys=True);force=loads['lh_palm'].copy()
    assert force[2]>0
    q=d.qpos.copy();v=d.qvel.copy();actual=d.actuator_force.copy()
    mujoco.mj_kinematics(m,d)
    np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v)
    np.testing.assert_array_equal(d.actuator_force,actual)
    assert not np.array_equal(d.xpos,pre)
    # A later solve is intentionally counterfactual and cannot mutate archives.
    d.qvel[2]=1.;mujoco.mj_forward(m,d)
    new_loads,_=module._contact_solution(sim)
    assert not np.allclose(new_loads['lh_palm'],force)
    assert json.dumps(raw,sort_keys=True)==archived
    np.testing.assert_array_equal(loads['lh_palm'],force)


def test_contact_world_sign_and_geometry_use_the_same_force_epoch():
    sim=fixture();m,d=sim.m,sim.d;mujoco.mj_step(m,d)
    loads,raw=module._contact_solution(sim)
    assert loads['lh_palm'][2]>0
    np.testing.assert_allclose(loads['lh_palm'][:2],0,atol=1e-12)
    for contact in raw['contacts']:
        normal=np.array(contact['frame_world'])[0]
        sign=1 if m.body(contact['body'][1]).name=='robot/lh_palm' else -1
        assert sign*normal[2]>0
    ix=raw['body_ids'].index(m.body('robot/lh_palm').id)
    assert raw['body_positions_world_m'][ix][2]==pytest.approx(.099)
    assert d.qpos[2]!=pytest.approx(.099,abs=1e-8)


def test_missing_or_duplicate_step_snapshot_is_rejected():
    obj=module.NativeTransitionRecorder.__new__(module.NativeTransitionRecorder)
    obj.pending=None
    with pytest.raises(ValueError,match='Missing'):obj.after_step()
    obj.pending={}
    with pytest.raises(ValueError,match='Unconsumed'):obj.before_step()


def test_rk4_cannot_claim_a_single_preintegration_contact_epoch():
    sim=fixture();sim.m.opt.integrator=mujoco.mjtIntegrator.mjINT_RK4
    with pytest.raises(ValueError,match='RK4'):
        module.NativeTransitionRecorder(sim,'unused',handle_joint='unused')

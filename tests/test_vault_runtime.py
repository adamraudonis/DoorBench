"""Runtime completion must follow actual boltwork, including both inputs."""
import mujoco
import numpy as np
import pytest
from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.vault_control import VaultControl
from doorbench.geometry.vault_hardware import resolve_vault_configuration


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    root=tmp_path_factory.mktemp('vault-runtime');paths=[]
    for s in generate_all():
        if s['index'] not in (124,179):continue
        export_door(s,str(root/'doors'),str(root/'hardware'),formats=('json','mjcf'))
        paths.append(root/'doors'/s['id'])
    return paths


@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_open_fixture_has_withdrawn_bolts_and_consistent_native_loops(fixtures,tier):
    for path in fixtures:
        e=DoorEnv(str(path),tier=tier)
        try:
            m=e.m;initial=m.qpos0.copy();limits=m.jnt_range.copy()
            e.reset('close_only',randomize=False)
            assert e.initialization_evidence['kind']=='prescribed_open_fixture'
            assert not e.initialization_evidence['native_release_history']
            assert VaultControl(e).released()
            equal=e.d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            # XML poses are serialized to micrometre precision.
            assert max(abs(e.d.efc_pos[equal]),default=0)<1e-6
            assert min((c.dist for c in e.d.contact),default=0)>-.0005
            np.testing.assert_array_equal(m.qpos0,initial)
            np.testing.assert_array_equal(m.jnt_range,limits)
            np.testing.assert_array_equal(e.d.qvel,np.zeros(m.nv))
        finally:e.close()


def test_one_lever_cannot_report_all_bolts_released(fixtures):
    tested=False
    for path in fixtures:
        e=DoorEnv(str(path));e.reset(randomize=False)
        try:
            rows=e.meta['vault_boltwork']['groups']
            if len(rows)!=2:continue
            tested=True;control=VaultControl(e)
            # Prescribed poses isolate the completion predicate. They are not
            # claimed as a native actuation trajectory; service tests cover it.
            r=rows[0];e.d.qpos[e.m.jnt_qposadr[e.m.joint(r['operator_joint']).id]]=r['operator_nominal_range'][1]
            resolve_vault_configuration(e.m,e.d.qpos,e.meta);mujoco.mj_forward(e.m,e.d)
            e.tracker.step(e.d)
            assert not control.released() and not e.tracker.L.lock_released
            action=control.act(goal=1.)
            assert action['contact_joint']==rows[1]['operator_joint']
            r=rows[1];e.d.qpos[e.m.jnt_qposadr[e.m.joint(r['operator_joint']).id]]=r['operator_nominal_range'][1]
            resolve_vault_configuration(e.m,e.d.qpos,e.meta);mujoco.mj_forward(e.m,e.d)
            e.tracker.step(e.d)
            assert control.released() and e.tracker.L.lock_released
        finally:e.close()
    assert tested

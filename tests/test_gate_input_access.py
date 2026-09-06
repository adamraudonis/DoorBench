"""Real radial knobs, screened acquisition paths, and native bounded pulls."""
import json
import numpy as np
import pytest
from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.interactions import ContactSites
from doorbench.benchmark.runner import torque_limits
from doorbench.benchmark.site_forces import SiteForces


@pytest.fixture(scope='module')
def gates(tmp_path_factory):
    root=tmp_path_factory.mktemp('gate-access');rows=[]
    for s in generate_all():
        if s['operator']['model'] not in ('gate_latch_magnetic','baby_gate_latch'):continue
        export_door(s,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        rows.append(root/'doors'/s['id'])
    assert len(rows)==16
    return rows


@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_all_raised_release_inputs_have_a_surface_path_and_native_pull(gates,tier):
    routes=set()
    for path in gates:
        e=DoorEnv(str(path),tier=tier);e.reset(randomize=False);m,d=e.m,e.d
        try:
            r=e.meta['gate_hardware'][0];contacts=ContactSites(e);sid=contacts.select(r['operator_joint'])
            assert sid is not None,path.name
            name=m.site(sid).name;route=contacts.access_paths[name];routes.add(route['mode'])
            assert name in r['release_face_sites'].values()
            geom=m.geom(r['knob_geom']).id
            local=d.geom_xmat[geom].reshape(3,3).T@(d.site_xpos[sid]-d.geom_xpos[geom])
            assert np.linalg.norm(local[:2])==pytest.approx(m.geom_size[geom,0],abs=2e-6)
            if route['mode']=='over_top':
                points=route['waypoints_world_m'];assert len(points)==4
                assert points[0][1]<0 and points[0][2]>e.spec['leaf']['height']+.1
            limits=torque_limits(e,str(path));forces=SiteForces(e,limits)
            for _ in range(round(.4/m.opt.timestep)):
                d.qfrc_applied[:]=forces.generalized(d,{name:[0.,0.,22.2]});e.step()
            assert d.qpos[m.jnt_qposadr[e._jid(r['operator_joint'])]]>=r['release_travel_m']-.001
            assert not np.any(d.warning.number)
        finally:e.close()
    assert routes=={'direct','over_top'}


def test_stock_blocking_both_paths_cannot_be_bypassed_by_site_name(gates):
    path=next(p for p in gates if p.name=='db0176_baby_gate')
    e=DoorEnv(str(path));e.reset(randomize=False)
    try:
        assert ContactSites(e).select(e.meta['operator_joint']) is not None
        g=e.m.geom('post_latch').id
        e.m.geom_pos[g]=[0.,0.,e.spec['operator']['height']+.2]
        e.m.geom_size[g]=[2.,.2,.5]
        e.mj.mj_kinematics(e.m,e.d)
        assert ContactSites(e).select(e.meta['operator_joint']) is None
    finally:e.close()

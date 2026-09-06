"""Load-path and release regressions for the actual ten marine door variants."""
import mujoco
import numpy as np
import pytest

from doorbench.build import build_model, export_door
from doorbench.clearance import Clearance
from doorbench.spec import generate_all

SPECS=[s for s in generate_all() if s['family']=='ship_watertight']


@pytest.fixture(scope='module')
def exports(tmp_path_factory):
    root=tmp_path_factory.mktemp('marine-dog-mounts')
    for s in SPECS:
        export_door(s,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
    return root


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_every_dog_has_a_real_bored_mount_and_connected_rear_cleat(exports,spec):
    ir=build_model(spec);leaf=ir.body('leaf');dogs=ir.meta['marine_dog_mounts']
    assert len(dogs)==(spec['kinematics']['dogs'] or 4)
    for row in dogs:
        body=ir.body(row['body']);x,_,z=body.pos
        shaft=next(g for g in body.geoms if g.name==row['spindle'])
        assert shaft.collision and shaft.size[0]==.007
        # A spindle may not occupy uncut leaf stock, even though the native
        # parent filter would hide that modeling error.
        for g in leaf.geoms:
            if g.semantic in ('leaf','glass') and g.type=='box':
                assert not (abs(g.pos[0]-x)<g.size[0]+.007-1e-10 and
                            abs(g.pos[2]-z)<g.size[2]+.007-1e-10)
        rings=[g for g in ir.body(row['mount_body']).geoms if g.name.startswith(row['bearing_prefix'])]
        assert len(rings)==24 and all(g.collision for g in rings)
        assert all(np.min(np.linalg.norm(np.asarray(g.mesh.vertices)[:,[0,2]],axis=1))>.007 for g in rings)
        assert any(g.name==row['body']+'_dog_web' and g.collision for g in body.geoms)
        assert row['body'] in ir.meta['mechanism_mass_bodies']
    for tier in ('full','simple','minimal'):
        m=mujoco.MjModel.from_xml_path(str(exports/'doors'/spec['id']/('door.xml' if tier=='full' else f'door_{tier}.xml')))
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        for k in range(len(dogs)):
            # This was a 20 mm gap: a rear restraint disconnected from the
            # flange cannot carry the wedge reaction simply by being static.
            base=m.geom(f'cleat_{k}_base').id;bridge=m.geom(f'cleat_{k}_bridge').id
            assert mujoco.mj_geomDistance(m,d,base,bridge,.1,None)<=1e-6
        assert max((-c.dist for c in d.contact),default=0)<.001


@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_bored_mounts_and_dogs_clear_the_complete_released_leaf_sweep(exports,spec):
    gate=Clearance(str(exports/'doors'/spec['id']))
    assert gate.run(40)['ok']
    assert gate.run_running(40)['ok']


def test_heavy_two_bolt_operators_are_normalized_after_seeded_generation():
    matches=[s for s in generate_all() if 'independent_vault_lever_bolts' in s.get('tags',[])]
    assert {s['id'] for s in matches}=={'db0124_vault','db0288_blast','db0530_vault',
                                    'db0623_blast','db0672_blast','db0772_blast','db0960_blast'}
    assert all(s['operator']['model']=='vault_lever' and
               s['latch']['model']=='vault_bolts_2' and
               s['lock']['model']=='vault_lever_boltwork' for s in matches)


def test_original_floating_rear_base_regression():
    # Old half-width20 mm at x=.08 cannot touch the bridge at x=.125±.005.
    gap=(.125-.005)-(.08+.02)
    assert gap==pytest.approx(.020)
    assert (.125-.005)-(.095+.035)<0


def test_component_success_cannot_hide_other_required_unmodeled_mechanisms():
    for spec in generate_all():
        if spec['family'] not in ('ship_watertight','vault','blast'):continue
        model=build_model(spec)
        flags={row['component'] for row in model.meta.get('mechanical_incomplete',[])}
        if spec['family']=='ship_watertight':assert 'hook_holdback' in flags
        else:
            assert not {'vault_bolt_transmission','vault_crane_hinge_mount'}&flags
            assert model.meta['vault_boltwork']['groups'] and model.meta['vault_crane_journals']

"""Physical capture, structural support and repeated release of ten ship doors."""
import json

import mujoco
import numpy as np
import pytest

from doorbench.build import build_model, write_hardware_meshes
from doorbench.export.mjcf import write_mjcf
from doorbench.geometry.ship_holdback import add_ship_holdback, first_ship_holdback_stop_angle
from doorbench.mass_reconciliation import reconcile_moving_mass
from doorbench.physics import derive
from doorbench.ship_holdback_qa import run_ship_holdback_qa
from doorbench.spec import generate_all

SPECS = [s for s in generate_all() if s['family'] == 'ship_watertight']


@pytest.fixture(scope='module')
def exports(tmp_path_factory):
    root = tmp_path_factory.mktemp('ship-holdback')
    for spec in SPECS:
        physical = derive(spec); model = build_model(spec, physical)
        if 'ship_holdback' not in model.meta:
            add_ship_holdback(model, spec)
            reconcile_moving_mass(model, physical)
        model.validate()
        folder = root / 'doors' / spec['id']; folder.mkdir(parents=True)
        write_hardware_meshes(model, str(root / 'hardware'))
        write_mjcf(model, str(folder), mesh_dir_rel='../../hardware')
        (folder / 'model.json').write_text(json.dumps(model.to_dict(), indent=2) + '\n')
        (folder / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    return root


def load(exports, spec, tier='full'):
    folder = exports / 'doors' / spec['id']
    model = mujoco.MjModel.from_xml_path(str(folder / ('door.xml' if tier == 'full' else f'door_{tier}.xml')))
    return model, json.loads((folder / 'model.json').read_text())['meta'], folder


@pytest.mark.parametrize('spec', SPECS, ids=lambda s: s['id'])
def test_real_pivot_bore_anchorage_grip_and_stock_mass(exports, spec):
    model, meta, folder = load(exports, spec); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hb = meta['ship_holdback']; station = model.body(hb['station_body']).id
    hook = model.body(hb['hook_body']).id
    assert model.body_parentid[hook] == station and model.body_jntnum[station] == 0
    assert model.body_mass[hook] > .2
    assert model.dof_armature[model.jnt_dofadr[model.joint(hb['hook_joint']).id]] == 0.
    assert hb['hook_body'] in meta['mechanism_mass_bodies']
    assert 'ship_holdback_striker' in meta['mechanism_mass_bodies']
    striker = model.geom(hb['striker_geom']).id
    # Rounded ends avoid the opposing duplicate cylinder/box contact normals
    # reproduced during the second native closing cycle on DB0911. Preserve
    # the actual bar stock envelope and geometry-derived inertia.
    assert model.geom_type[striker] == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert model.geom_size[striker, 0] == pytest.approx(.006)
    assert 2 * model.geom_size[striker, :2].sum() == pytest.approx(.090)
    shaft = model.geom('ship_holdback_pivot_pin').id
    cheeks = [g for g in range(model.ngeom) if model.geom(g).name.startswith('ship_holdback_cheek_')]
    assert len(cheeks) == 6
    for geom in cheeks:
        # Static/static native filtering cannot hide an unmachined shaft hole.
        assert mujoco.mj_geomDistance(model, data, shaft, geom, .02, None) > .0005
    washers = [g for g in range(model.ngeom) if model.geom(g).name.startswith('ship_holdback_thrust_washer_')]
    eyes = [g for g in range(model.ngeom) if model.geom(g).name.startswith('ship_holdback_hook_eye_')]
    assert len(washers) == 24 and len(eyes) == 12
    for geom in washers:
        assert mujoco.mj_geomDistance(model, data, shaft, geom, .02, None) > .0003
        assert min(mujoco.mj_geomDistance(model, data, geom, eye, .02, None) for eye in eyes) > .0009
    handle = model.geom('ship_holdback_release_handle').id
    site = model.site(hb['release_site']).id
    local = data.geom_xmat[handle].reshape(3, 3).T @ (data.site_xpos[site] - data.geom_xpos[handle])
    # MJCF rounds these irrational surface coordinates to micrometre-scale
    # decimal precision (observed radial error 0.157 micrometres).
    assert np.linalg.norm(local[:2]) == pytest.approx(model.geom_size[handle, 0], abs=1e-6)
    assert abs(local[2]) <= model.geom_size[handle, 1]
    outward = data.site_xmat[site].reshape(3, 3)[:, 2]
    radial = data.site_xpos[site] - data.geom_xpos[handle]
    axis = data.geom_xmat[handle].reshape(3, 3)[:, 2]
    radial -= axis * np.dot(radial, axis)
    assert np.dot(outward, radial / np.linalg.norm(radial)) > .999999
    arm = model.geom('ship_holdback_arm').id
    assert mujoco.mj_geomDistance(model, data, handle, arm, .02, None) < 0
    base = model.geom('ship_holdback_base').id
    for sx in (-1, 1):
        for sy in (-1, 1):
            anchor = model.geom(f'ship_holdback_anchor_{sx}_{sy}').id
            assert data.geom_xpos[anchor, 2] - model.geom_size[anchor, 1] < -.05
            assert mujoco.mj_geomDistance(model, data, anchor, base, .1, None) < 0
    assert max((-c.dist for c in data.contact), default=0) < .001
    source = json.loads((folder / 'model.json').read_text())
    structures = [model.geom(g['name']).id for body in source['bodies'] for g in body['geoms']
                  if g['semantic'] in ('wall', 'frame') and body['name'] != hb['station_body']
                  and model.body_weldid[model.geom_bodyid[model.geom(g['name']).id]] == 0]
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == station:
            for other in structures:
                # Static/static filtering must not hide a holder embedded in
                # a wall or jamb. Its four anchors intentionally enter floor.
                assert mujoco.mj_geomDistance(model, data, geom, other, .05, None) >= -1e-6
    before = model.qpos0.copy()
    stop = first_ship_holdback_stop_angle(model, meta)
    assert 1.7 < stop['angle_rad'] < hb['full_open_angle_rad'] - .01
    assert 0 <= stop['gap_m'] < 1e-5
    assert np.array_equal(model.qpos0, before)


@pytest.mark.parametrize('spec', SPECS, ids=lambda s: s['id'])
def test_removing_extension_spring_removes_actual_closing_torque(exports, spec):
    model, meta, _ = load(exports, spec)
    hook = model.joint(meta['ship_holdback']['hook_joint']).id
    dof = int(model.jnt_dofadr[hook]); tendon = model.tendon(meta['ship_holdback']['spring']).id
    source = mujoco.MjData(model); mujoco.mj_forward(model, source)
    mass = model.body_mass.copy()
    model.tendon_stiffness[tendon] = 0.
    removed = mujoco.MjData(model); mujoco.mj_forward(model, removed)
    # Gravity also closes this hook; this isolates the real spring's useful
    # contribution rather than falsely claiming gravity cannot retain it.
    contribution = float(source.qfrc_passive[dof] - removed.qfrc_passive[dof])
    assert contribution < -.05
    assert np.array_equal(model.body_mass, mass)
    assert np.array_equal(source.qfrc_bias, removed.qfrc_bias)


@pytest.mark.parametrize('tier', ['full', 'simple', 'minimal'])
@pytest.mark.parametrize('spec', SPECS, ids=lambda s: s['id'])
def test_two_actual_hand_cycles_with_hands_free_loaded_hold(exports, spec, tier):
    model, meta, folder = load(exports, spec, tier)
    result = run_ship_holdback_qa(model, meta)
    (folder / f'holdback-native-{tier}.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['ok'], result['failures']
    assert len(result['cycles']) == 2
    assert result['max_penetration_m'] < .001
    assert result['peak_hand_force_N'] <= 120
    assert not result['native_warning_messages'] and not any(result['warning_counters'])
    holds = [row for row in result['trace'] if row['phase'] == 'hold']
    assert holds and all(not row['site_forces'] for row in holds)
    for cycle in result['cycles']:
        assert cycle['jaw_load_observed_s'] > .1
        assert cycle['shoulder_load_observed_s'] > .1
        assert cycle['release_started_after_unloading']
        assert cycle['release_peak_hook_force_N'] < 30
        assert cycle['opening_stop_load_observed_s'] > .1


def test_half_step_keeps_the_same_contact_and_retention_gates(exports):
    spec = next(s for s in SPECS if s['id'] == 'db0744_ship_watertight')
    model, meta, folder = load(exports, spec)
    model.opt.timestep *= .5
    result = run_ship_holdback_qa(model, meta)
    (folder / 'holdback-native-halfstep.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['ok'], result['failures']
    baseline = run_ship_holdback_qa(*load(exports, spec)[:2])
    assert abs(result['cycles'][0]['held_rad'] - baseline['cycles'][0]['held_rad']) < .001
    assert abs(result['peak_hand_force_N'] - baseline['peak_hand_force_N']) < 2.


def test_removing_jaw_contact_after_real_capture_loses_retention(exports, monkeypatch):
    import doorbench.ship_holdback_qa as gate_module
    gate_module._CACHE.clear()
    spec = next(s for s in SPECS if s['id'] == 'db0744_ship_watertight')
    model, meta, folder = load(exports, spec)
    native_step = mujoco.mj_step; jaw = model.geom(meta['ship_holdback']['load_face_geom']).id
    leaf = int(model.jnt_qposadr[model.joint(meta['ship_holdback']['leaf_joint']).id])
    removal = {}
    def step(m, d, *args, **kwargs):
        # Source wheel release lasts 6 s, opening 10 s, then one second of
        # actual loaded capture. Change contact flags only, never geometry
        # bounds, inertia, positions, joint constraints or hand forces.
        if m is model and not removal and d.time >= 17.:
            removal.update(time_s=float(d.time), captured_leaf_rad=float(d.qpos[leaf]))
            assert abs(d.qfrc_applied[model.jnt_dofadr[model.joint('leaf_hinge').id]] + 80) < 1e-9
            m.geom_contype[jaw] = 0; m.geom_conaffinity[jaw] = 0
        return native_step(m, d, *args, **kwargs)
    monkeypatch.setattr(mujoco, 'mj_step', step)
    result = run_ship_holdback_qa(model, meta)
    (folder / 'holdback-removed-jaw.json').write_text(json.dumps({**result, 'removal': removal}, indent=2) + '\n')
    assert removal['captured_leaf_rad'] > 1.8
    assert not result['ok']
    hold = [row for row in result['trace'] if row['phase'] == 'hold' and row['time_s'] > 17.5]
    assert hold and min(row['leaf_q'] for row in hold) < removal['captured_leaf_rad'] - .2
    assert 'holdback_did_not_retain_open_leaf' in result['failures']

"""Physical mounting and native release/recapture regressions for gate hardware."""
import copy
import json
import math

import mujoco
import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.gate_hardware_qa import run_gate_hardware_qa, probe_magnetic_latch
from doorbench.geometry.gate_hardware import (
    magnetic_potential_force, compile_magnetic_latches, apply_magnetic_latches,
)
from doorbench.spec import generate_all


@pytest.fixture(scope="module")
def magnetic_fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("magnetic-gates")
    rows = []
    for spec in generate_all():
        if spec["operator"]["model"] != "gate_latch_magnetic":
            continue
        summary = export_door(spec, str(root / "doors"), str(root / "hardware"),
                              formats=("mjcf", "json"))
        path = root / "doors" / spec["id"]
        metadata = json.loads((path / "model.json").read_text())["meta"]
        rows.append((spec, metadata, summary["files"]["mjcf"]))
    assert {s["id"] for s, _, _ in rows} == {
        "db0014_gate_swing", "db0075_gate_swing", "db0287_gate_swing",
        "db0540_gate_swing", "db0854_gate_swing", "db0877_gate_swing"}
    return rows


def test_all_six_have_connected_mounts_and_native_latch_cycles(magnetic_fixtures):
    for spec, metadata, paths in magnetic_fixtures:
        for tier in ("full", "simple"):
            model = mujoco.MjModel.from_xml_path(paths[tier])
            report = run_gate_hardware_qa(model, spec, metadata)
            assert report["ok"], (spec["id"], tier, report)
            assert len(report["attachments"]) >= 24
            assert report["pull_bodies"] == ["leaf", "leaf"]
            assert min(report["released_keeper_gaps_m"]) > .010
            assert report["native_behavior"]["max_contact_penetration_m"] < .001


def test_floating_release_knob_and_disconnected_mount_fail(magnetic_fixtures):
    spec, metadata, paths = magnetic_fixtures[0]
    for name, offset in [("leaf_pin_knob", [0, 0, .035]),
                         ("leaf_cup_mount", [0, .080, 0]),
                         ("leaf_pin_post_plate_0", [.080, 0, 0])]:
        model = mujoco.MjModel.from_xml_path(paths["full"])
        model.geom_pos[model.geom(name).id] += offset
        report = run_gate_hardware_qa(model, spec, metadata, dynamic=False)
        assert not report["ok"], name
        assert any("detached" in f for f in report["failures"] if isinstance(f, dict))


def test_missing_mechanism_metadata_does_not_pass_old_generic_pin(magnetic_fixtures):
    spec, _, paths = magnetic_fixtures[0]
    model = mujoco.MjModel.from_xml_path(paths["full"])
    assert not run_gate_hardware_qa(model, spec, {})["ok"]


def test_release_and_leaf_pull_sites_lie_on_their_actual_grips(magnetic_fixtures):
    for spec, metadata, paths in magnetic_fixtures:
        model = mujoco.MjModel.from_xml_path(paths["full"])
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        hardware = metadata["gate_hardware"][0]
        for site_name, geom_name in [(hardware["release_site"], hardware["knob_geom"]),
                                    (hardware["pull_sites"][0], "leaf_gate_pull_n"),
                                    (hardware["pull_sites"][1], "leaf_gate_pull_p")]:
            sid, gid = model.site(site_name).id, model.geom(geom_name).id
            local = data.geom_xmat[gid].reshape(3, 3).T @ (data.site_xpos[sid] - data.geom_xpos[gid])
            # MJCF serializes coordinates to six decimals.
            assert abs(np.linalg.norm(local[:2]) - model.geom_size[gid, 0]) < 2e-6
            assert abs(local[2]) < model.geom_size[gid, 1]
            normal = data.site_xmat[sid].reshape(3, 3)[:, 2]
            assert abs(np.dot(normal, data.site_xpos[sid] - data.geom_xpos[gid])
                       - model.geom_size[gid, 0]) < 2e-6
        assert abs(data.site_xpos[model.site(hardware["release_site"]).id, 2] - 1.5) < 1e-6


def test_absent_magnet_returns_pin_up_and_does_not_hold_closed(magnetic_fixtures):
    _, metadata, paths = magnetic_fixtures[0]
    model = mujoco.MjModel.from_xml_path(paths["full"])
    without = copy.deepcopy(metadata)
    without["magnetic_latches"] = []
    report = probe_magnetic_latch(model, without)
    assert not report["ok"]
    assert report["states"]["settle"]["pin_m"] > .029
    assert report["states"]["hold"]["door_rad"] > .5
    assert not report["checks"]["magnet_recaptures_pin"]


def test_native_probe_restores_callers_callback(magnetic_fixtures):
    _, metadata, paths = magnetic_fixtures[0]
    model = mujoco.MjModel.from_xml_path(paths["full"])
    def existing(m, d):
        pass
    mujoco.set_mjcb_passive(existing)
    try:
        assert probe_magnetic_latch(model, metadata)["ok"]
        assert mujoco.get_mjcb_passive() is existing
    finally:
        mujoco.set_mjcb_passive(None)


def test_magnet_is_a_smooth_conservative_force_with_bounded_axial_peak():
    axes, peak = np.array([.018, .018, .050]), 15.
    rng = np.random.default_rng(41)
    for d in rng.uniform(-.3, .3, (20, 3))*axes + [0, 0, .02]:
        _, force = magnetic_potential_force(d, axes, peak)
        eps = 1e-7
        gradient = []
        for e in np.eye(3)*eps:
            gradient.append((magnetic_potential_force(d+e, axes, peak)[0]
                            -magnetic_potential_force(d-e, axes, peak)[0])/(2*eps))
        np.testing.assert_allclose(force, -np.array(gradient), atol=1e-7)
    _, f = magnetic_potential_force([0, 0, axes[2]/math.sqrt(5)], axes, peak)
    assert f[2] == pytest.approx(-peak)
    assert magnetic_potential_force([0, 0, .050], axes, peak)[0] == 0
    assert np.linalg.norm(magnetic_potential_force([0, 0, .050-1e-8], axes, peak)[1]) < 1e-8
    assert np.linalg.norm(magnetic_potential_force([.02, 0, .01], axes, peak)[1]) == 0


def test_passive_binding_requires_real_sites_and_leaves_state_untouched(magnetic_fixtures):
    _, metadata, paths = magnetic_fixtures[0]
    model = mujoco.MjModel.from_xml_path(paths["full"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos, qvel, applied = data.qpos.copy(), data.qvel.copy(), data.qfrc_applied.copy()
    passive = data.qfrc_passive.copy()
    rules = compile_magnetic_latches(model, metadata)
    apply_magnetic_latches(model, data, rules)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    np.testing.assert_array_equal(data.qfrc_applied, applied)
    assert np.linalg.norm(data.qfrc_passive-passive) > 1
    bad = copy.deepcopy(metadata)
    bad["magnetic_latches"][0]["pin_site"] = "missing_pin_pole"
    with pytest.raises(ValueError, match="explicitly bound"):
        compile_magnetic_latches(model, bad)
    bad["magnetic_latches"][0]["pin_site"] = "leaf_gate_pull_grip_n"
    with pytest.raises(ValueError, match="release joint body"):
        compile_magnetic_latches(model, bad)


@pytest.mark.parametrize("axes,force", [([0,.1,.1],15),([.1,.1,float('nan')],15),([.1,.1,.1],-1)])
def test_invalid_magnet_parameters_rejected(axes, force):
    with pytest.raises(ValueError):
        magnetic_potential_force([0,0,.01], axes, force)


@pytest.fixture(scope="module")
def baby_fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("baby-lift-gates")
    rows = []
    for spec in generate_all():
        if spec["family"] == "baby_gate":
            summary = export_door(spec, str(root/"doors"), str(root/"hardware"),
                                  formats=("mjcf", "json"))
            metadata = json.loads((root/"doors"/spec["id"]/"model.json").read_text())["meta"]
            rows.append((spec, metadata, summary["files"]["mjcf"]))
    assert len(rows) == 10
    return rows


def test_all_baby_latches_are_connected_and_use_actual_20mm_release(baby_fixtures):
    for spec, metadata, paths in baby_fixtures:
        assert not metadata.get("magnetic_latches")
        for tier in ("full", "simple"):
            model = mujoco.MjModel.from_xml_path(paths[tier])
            report = run_gate_hardware_qa(model, spec, metadata)
            assert report["ok"], (spec["id"], tier, report)
            assert report["native_behavior"]["checks"]["pin_returns_down_when_released"]
            assert min(report["released_keeper_gaps_m"]) > .007
            np.testing.assert_allclose(model.joint("leaf_pin_slide").range, [0,.020], atol=1e-9)
            data = mujoco.MjData(model)
            mujoco.mj_kinematics(model, data)
            assert data.site_xpos[model.site("leaf_grip_pin").id,2] == pytest.approx(spec["operator"]["height"])


def test_released_baby_hardware_clears_fixed_housing_over_both_authored_swings(baby_fixtures):
    for spec, metadata, paths in baby_fixtures:
        model = mujoco.MjModel.from_xml_path(paths["full"])
        data = mujoco.MjData(model)
        data.qpos[model.joint("leaf_pin_slide").qposadr] = .020
        hardware = [i for i in range(model.ngeom) if model.geom(i).name.startswith(("leaf_cup_", "leaf_gate_pull_"))]
        fixed = [i for i in range(model.ngeom) if model.geom(i).name.startswith("leaf_pin_housing_")]
        hinge = model.joint("leaf_hinge")
        for q in np.linspace(*hinge.range, 41):
            data.qpos[hinge.qposadr] = q
            mujoco.mj_kinematics(model, data)
            for a in hardware:
                for b in fixed:
                    gap = mujoco.mj_geomDistance(model, data, a, b, .02, None)
                    assert gap >= .001, (spec["id"],q,model.geom(a).name,model.geom(b).name,gap)


def test_baby_old_floating_knob_and_wrong_travel_do_not_pass(baby_fixtures):
    spec, metadata, paths = baby_fixtures[0]
    model = mujoco.MjModel.from_xml_path(paths["full"])
    model.geom_pos[model.geom("leaf_pin_knob").id,2] += .050
    assert not run_gate_hardware_qa(model, spec, metadata, dynamic=False)["ok"]
    model = mujoco.MjModel.from_xml_path(paths["full"])
    model.jnt_range[model.joint("leaf_pin_slide").id,1] = .050
    assert not run_gate_hardware_qa(model, spec, metadata, dynamic=False)["ok"]


@pytest.fixture(scope='module')
def fork_fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp('fork-gates')
    rows = []
    for spec in generate_all():
        if spec['operator']['model'] != 'gate_latch_fork':
            continue
        summary = export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        path = root/'doors'/spec['id']
        metadata = json.loads((path/'model.json').read_text())['meta']
        rows.append((spec,metadata,summary['files']['mjcf']))
    assert len(rows) == 12
    return rows


def test_all_twelve_forks_clear_actual_stile_carrier_and_post_and_operate(fork_fixtures):
    for spec,metadata,paths in fork_fixtures:
        for tier in ('full','simple'):
            m = mujoco.MjModel.from_xml_path(paths[tier])
            report = run_gate_hardware_qa(m,spec,metadata)
            assert report['ok'], (spec['id'],tier,report)
            assert report['sweeps']['operator_min_gap_m'] >= .0005
            assert report['sweeps']['released_post_min_gap_m'] >= .001
            assert report['native_behavior']['max_contact_penetration_m'] < .001
            assert metadata['gate_hardware'][0]['self_latching'] is False


def test_fork_detached_clamp_floating_grip_and_short_lift_are_rejected(fork_fixtures):
    spec,metadata,paths = fork_fixtures[0]
    for name,shift in [('leaf_fork_mount',[0,.1,0]),('leaf_fork_handle',[0,0,.05])]:
        m = mujoco.MjModel.from_xml_path(paths['full'])
        m.geom_pos[m.geom(name).id] += shift
        report = run_gate_hardware_qa(m,spec,metadata,dynamic=False)
        assert not report['ok'], (name,report)
        assert any('detached' in x for x in report['failures'] if isinstance(x,dict))
    m = mujoco.MjModel.from_xml_path(paths['full'])
    m.jnt_range[m.joint('leaf_fork_hinge').id,1] = 1.2
    assert not run_gate_hardware_qa(m,spec,metadata,dynamic=False)['ok']


def test_fork_false_inboard_pivot_cannot_hide_in_parent_contact_filter(fork_fixtures):
    spec,metadata,paths = fork_fixtures[0]
    m = mujoco.MjModel.from_xml_path(paths['full'])
    # Restore the old inboard layout's failure deliberately. The static native
    # contact filter excludes parent-child pairs; our geometric sweep does not.
    m.body_pos[m.body('leaf_fork').id,0] -= metadata['u']*.05
    report = run_gate_hardware_qa(m,spec,metadata,dynamic=False)
    assert not report['ok']
    assert report['sweeps']['operator_min_gap_m'] < 0


def test_fork_release_site_is_on_round_handle_and_has_upward_moment(fork_fixtures):
    for spec,metadata,paths in fork_fixtures:
        m = mujoco.MjModel.from_xml_path(paths['full'])
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m,d)
        sid=m.site('leaf_fork_grip').id
        gid=m.geom('leaf_fork_handle').id
        local=d.geom_xmat[gid].reshape(3,3).T@(d.site_xpos[sid]-d.geom_xpos[gid])
        assert np.isclose(np.linalg.norm(local[:2]),m.geom_size[gid,0],atol=2e-6)
        assert abs(local[2]) < m.geom_size[gid,1]
        jid=m.joint('leaf_fork_hinge').id
        moment=np.dot(np.cross(d.site_xpos[sid]-d.xanchor[jid],[0,0,22.2]),d.xaxis[jid])
        assert moment > .5


@pytest.fixture(scope='module')
def suffolk_fixtures(tmp_path_factory):
    root=tmp_path_factory.mktemp('suffolk-gates');rows=[]
    for spec in generate_all():
        if spec['operator']['model']!='thumb_latch_suffolk':
            continue
        summary=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        path=root/'doors'/spec['id']
        metadata=json.loads((path/'model.json').read_text())['meta']
        rows.append((spec,metadata,summary['files']['mjcf']))
    assert len(rows)==12
    assert sum(s['family']=='gate_swing' for s,_,_ in rows)==5
    return rows


def test_all_twelve_suffolk_mechanisms_transmit_native_contact(suffolk_fixtures):
    for spec,metadata,paths in suffolk_fixtures:
        for tier in ('full','simple'):
            model=mujoco.MjModel.from_xml_path(paths[tier])
            report=run_gate_hardware_qa(model,spec,metadata)
            assert report['ok'],(spec['id'],tier,report)
            assert report['tang_slot_min_gap_m']>.005
            assert report['thumb_bearing_min_gap_m']>.0019
            assert report['native_behavior']['tang_transmission_contact_samples']>10
            assert model.neq==0
            expected='leaf_thumb_hinge' if metadata['v']>0 else 'leaf_latch_bar_hinge'
            assert metadata['operator_joint']==expected


def test_disabling_tang_contact_prevents_thumb_bar_transmission(suffolk_fixtures):
    from doorbench.gate_hardware_qa import probe_suffolk_latch
    spec,metadata,paths=suffolk_fixtures[0]
    m=mujoco.MjModel.from_xml_path(paths['full'])
    g=m.geom('leaf_thumb_lifter').id
    m.geom_contype[g]=m.geom_conaffinity[g]=0
    r=probe_suffolk_latch(m,metadata)
    assert not r['ok']
    assert not r['checks']['native_tang_transmits_load']
    assert not r['checks']['thumb_press_lifts_bar']
    assert r['states']['thumb_press']['thumb_rad']>.29
    assert r['states']['thumb_press']['bar_rad']<.02


def test_suffolk_blocked_slot_and_equality_shortcut_are_rejected(suffolk_fixtures):
    import xml.etree.ElementTree as ET
    from pathlib import Path
    spec,metadata,paths=suffolk_fixtures[0]
    m=mujoco.MjModel.from_xml_path(paths['full'])
    m.geom_pos[m.geom('leaf_thumb_lifter').id,0]+=.008
    assert not run_gate_hardware_qa(m,spec,metadata,dynamic=False)['ok']
    tree=ET.parse(paths['full']);root=tree.getroot()
    eq=ET.SubElement(root,'equality')
    ET.SubElement(eq,'joint',joint1='leaf_latch_bar_hinge',joint2='leaf_thumb_hinge',polycoef='0 1 0 0 0')
    path=Path(paths['full']).with_name('equality-negative.xml');tree.write(path)
    m=mujoco.MjModel.from_xml_path(str(path))
    r=run_gate_hardware_qa(m,spec,metadata,dynamic=False)
    assert not r['ok']
    assert any('equality' in x for x in r['failures'] if isinstance(x,str))


def test_suffolk_real_contact_surfaces_generate_correct_release_moments(suffolk_fixtures):
    for spec,meta,paths in suffolk_fixtures:
        m=mujoco.MjModel.from_xml_path(paths['full']);d=mujoco.MjData(m);mujoco.mj_kinematics(m,d)
        for joint,site,force in [('leaf_thumb_hinge','leaf_thumb_push',[0,0,-22.2]),
                                 ('leaf_latch_bar_hinge','leaf_latch_bar_grip',[0,0,22.2])]:
            jid=m.joint(joint).id;sid=m.site(site).id
            moment=np.dot(np.cross(d.site_xpos[sid]-d.xanchor[jid],force),d.xaxis[jid])
            assert moment>.5
        assert meta['gate_hardware'][0]['bar_face']==meta['v']
        assert meta['gate_hardware'][0]['thumb_face']==-meta['v']


def test_cut_slot_prorates_explicit_mass_instead_of_duplicating_it():
    from doorbench.geometry.suffolk_latch import _cut_slot
    from doorbench.ir import Geom
    pieces=[Geom('slab','box',(1.,.05,1.),mass_override=10.,semantic='leaf')]
    _cut_slot(pieces,'slot',-.1,.1,-.2,.2)
    assert len(pieces)==4
    assert np.isclose(sum(g.mass() for g in pieces),9.8)


def test_contact_preview_preserves_caller_model_and_rejects_missing_transmission(suffolk_fixtures):
    from doorbench.gate_hardware_qa import SuffolkContactPreview
    _,meta,paths=suffolk_fixtures[0]
    m=mujoco.MjModel.from_xml_path(paths['full'])
    original=m.jnt_range.copy();q=m.qpos0.copy();adr=int(m.joint('leaf_thumb_hinge').qposadr[0])
    preview=SuffolkContactPreview(m,meta)
    for value in (0.,.1,.2,.3):
        q[adr]=value;before=q.copy();report=preview.resolve(q,'leaf_thumb_hinge')
        assert report['ok'],report
        np.testing.assert_array_equal(q,before)
    np.testing.assert_array_equal(m.jnt_range,original)
    report['qpos'][adr]=999
    assert preview.resolve(q,'leaf_thumb_hinge')['qpos'][adr]<.31
    g=m.geom('leaf_thumb_lifter').id;m.geom_contype[g]=m.geom_conaffinity[g]=0
    assert not SuffolkContactPreview(m,meta).resolve(q,'leaf_thumb_hinge')['ok']


def test_fork_first_contact_arrests_scan_before_incidental_handle_collision(fork_fixtures):
    from doorbench.gate_hardware_qa import first_fork_contact_angle
    for _,meta,paths in fork_fixtures:
        m=mujoco.MjModel.from_xml_path(paths['full']);original=m.jnt_range.copy()
        report=first_fork_contact_angle(m,meta)
        assert report['ok'],report
        assert 0<report['contact_angle_rad']<.008
        np.testing.assert_array_equal(m.jnt_range,original)
    for n in meta['gate_hardware'][0]['tine_geoms']:
        m.geom_pos[m.geom(n).id,2]+=5.
    assert not first_fork_contact_angle(m,meta)['ok']


def test_thumb_bearings_cannot_overlap_pad_despite_native_parent_filter(suffolk_fixtures):
    spec,metadata,paths=suffolk_fixtures[0]
    model=mujoco.MjModel.from_xml_path(paths['full'])
    for side in (-1,1):
        for axis in (1,2):
            for sign in (-1,1):
                model.geom_pos[model.geom(f'leaf_thumb_bearing_{side}_{axis}_{sign}').id,0]-=side*.006
    report=run_gate_hardware_qa(model,spec,metadata,dynamic=False)
    assert not report['ok']
    assert any('thumb_bearing_interference' in f for f in report['failures'] if isinstance(f,dict))

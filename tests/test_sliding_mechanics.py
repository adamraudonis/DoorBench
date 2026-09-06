"""Mechanical regression: real cavities, accessible press, force transfer and lanes."""
import copy
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.sliding_mechanics_qa import run_sliding_mechanics_qa
from doorbench.sliding_track_qa import run_sliding_track_qa


@pytest.fixture(scope="module")
def sliding(tmp_path_factory):
    mujoco=pytest.importorskip("mujoco")
    root=tmp_path_factory.mktemp("mechanical_sliding")
    result={}
    for spec in generate_all():
        if spec["family"]!="sliding_bypass" and spec["kinematics"].get("track")!="top_hung_pocket":
            continue
        export_door(spec,str(root/"doors"),str(root/"hardware"),formats=("json","mjcf"))
        path=root/"doors"/spec["id"]
        native=mujoco.MjModel.from_xml_path(str(path/"door.xml"))
        meta=json.loads((path/"model.json").read_text())["meta"]
        result[spec["id"]]=(native,meta,path)
    assert len(result)==57
    return result


def test_all_35_bypass_and_22_pocket_geometry(sliding):
    for door,(native,meta,_) in sliding.items():
        assert (r:=run_sliding_mechanics_qa(native,meta))["ok"],(door,r)
        assert (r:=run_sliding_track_qa(native,meta))["ok"],(door,r)
        for control,support in zip(meta["sliding_leaf_controls"],meta["sliding_track_supports"]):
            assert control["nominal_range"]==support["nominal_range"]
            if meta["family"]=="sliding_bypass":
                np.testing.assert_allclose(native.jnt_range[native.joint(control["joint"]).id],support["nominal_range"],atol=1e-9)


def test_all_pockets_handoff_before_cup_occlusion_and_push_real_edge(sliding):
    import mujoco
    for door,(m,meta,_) in sliding.items():
        if 'pocket_edge_pull' not in meta: continue
        p=meta['pocket_edge_pull'];d=mujoco.MjData(m)
        slide=m.joint(p['leaf_joint']).id;address=int(m.jnt_qposadr[slide])
        site=m.site(p['final_push_site']).id;direction=np.asarray(p['final_push_direction'])
        assert m.site_bodyid[site] == m.body(p['leaf_body']).id
        assert p['final_push_switch_q']+.020 == pytest.approx(p['face_cup_occlusion_q'])
        assert 0 < p['final_push_switch_q'] < p['recessed_leaf_q']
        for q in np.linspace(p['final_push_switch_q'],p['recessed_leaf_q'],17):
            d.qpos[address]=q;mujoco.mj_forward(m,d)
            hit=np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(m,d,d.site_xpos[site]-.08*direction,direction,None,True,-1,hit)
            assert distance == pytest.approx(.08,abs=1e-6),(door,q,distance)
            assert int(hit[0])==m.geom(p['final_push_geom']).id,(door,q,m.geom(int(hit[0])).name)
        # The threshold follows actual rim geometry and opening-side polarity.
        cup_geoms={name for cup in meta['sliding_recessed_pulls'] if cup['body']==p['leaf_body'] for name in cup['side_geoms']}
        d.qpos[address]=p['face_cup_occlusion_q'];mujoco.mj_forward(m,d)
        leading=max(direction[0]*d.geom_xpos[m.geom(name).id,0]+m.geom_size[m.geom(name).id,0] for name in cup_geoms)
        assert leading == pytest.approx(direction[0]*p['pocket_mouth_x'],abs=1e-6)


def test_native_edge_push_finishes_recession_without_loading_hidden_face_cup(sliding):
    import mujoco
    m,meta,_=sliding['db0018_sliding_single'];p=meta['pocket_edge_pull'];d=mujoco.MjData(m)
    j=m.joint(p['leaf_joint']).id;address=int(m.jnt_qposadr[j]);dof=int(m.jnt_dofadr[j])
    d.qpos[address]=p['final_push_switch_q'];mujoco.mj_forward(m,d)
    sid=m.site(p['final_push_site']).id
    for _ in range(round(3/m.opt.timestep)):
        d.qfrc_applied[:]=0
        force=np.asarray(p['final_push_direction'])*(m.dof_frictionloss[dof]+12.)
        mujoco.mj_applyFT(m,d,force,np.zeros(3),d.site_xpos[sid],int(m.site_bodyid[sid]),d.qfrc_applied)
        mujoco.mj_step(m,d)
    assert d.qpos[address]>p['recessed_leaf_q']-.002
    assert abs(d.qpos[m.jnt_qposadr[m.joint(p['joint']).id]])<.01


def test_pocket_handoff_rejects_late_fabricated_threshold(sliding):
    m,meta,_=sliding['db0018_sliding_single'];broken=copy.deepcopy(meta)
    p=broken['pocket_edge_pull'];p['final_push_switch_q']+=.06;p['face_cup_occlusion_q']+=.06
    result=run_sliding_mechanics_qa(m,broken)
    assert not result['ok']
    assert any(f['check']=='pocket_cup_handoff_not_at_actual_rim' for f in result['failures'])


def test_pocket_handoff_rejects_push_site_in_empty_air(sliding):
    m,meta,_=sliding['db0018_sliding_single'];site=m.site(meta['pocket_edge_pull']['final_push_site']).id
    old=m.site_pos[site].copy()
    try:
        m.site_pos[site,1]+=.04
        result=run_sliding_mechanics_qa(m,meta)
        assert not result['ok']
        assert any(f['check']=='final_edge_push_occluded_or_off_surface' for f in result['failures'])
    finally:m.site_pos[site]=old


def test_each_bypass_leaf_moves_without_driving_others(sliding):
    import mujoco
    for door,(m,meta,_) in sliding.items():
        if meta["family"]!="sliding_bypass":continue
        controls=meta["sliding_leaf_controls"]
        for active in controls:
            d=mujoco.MjData(m)
            addresses={c["joint"]:int(m.jnt_qposadr[m.joint(c["joint"]).id]) for c in controls}
            dof=int(m.jnt_dofadr[m.joint(active["joint"]).id])
            force=float(m.dof_frictionloss[dof]+30)
            for _ in range(3000):
                d.qfrc_applied[:]=0;d.qfrc_applied[dof]=force;mujoco.mj_step(m,d)
            assert abs(d.qpos[addresses[active["joint"]]]-active["nominal_range"][1])<.002,(door,active["joint"],d.qpos)
            for name,address in addresses.items():
                if name!=active["joint"]:assert abs(d.qpos[address])<1e-5,(door,active["joint"],name,d.qpos)


@pytest.mark.parametrize("door",["db0018_sliding_single","db0041_sliding_single","db0274_sliding_single"])
def test_press_then_pull_native_force_transfer_and_spring_return(sliding,door):
    import mujoco
    m,meta,_=sliding[door];p=meta["pocket_edge_pull"];d=mujoco.MjData(m)
    qa=int(m.jnt_qposadr[m.joint(p["joint"]).id]);qs=int(m.jnt_qposadr[m.joint(p["leaf_joint"]).id])
    body=m.body(p["body"]).id;direction=np.asarray(p["press_direction"])
    d.qpos[qs]=p["recessed_leaf_q"];mujoco.mj_forward(m,d)
    def apply(site,force,steps):
        for _ in range(steps):
            d.qfrc_applied[:]=0
            mujoco.mj_applyFT(m,d,force,np.zeros(3),d.site_xpos[m.site(site).id],body,d.qfrc_applied)
            mujoco.mj_step(m,d)
    apply(p["press_site"],direction*12,500)
    assert .70<d.qpos[qa]<1.0
    assert abs(d.qpos[qs]-p["recessed_leaf_q"])<.001
    force=max(25,float(m.dof_frictionloss[m.jnt_dofadr[m.joint(p["leaf_joint"]).id]])+20)
    apply(p["grip_site"],-direction*force,400)
    assert p["recessed_leaf_q"]-d.qpos[qs]>.14
    assert d.qpos[qa]>.70
    apply(p["grip_site"],np.zeros(3),1200)
    assert d.qpos[qa]<.05


def test_contacting_fingertip_deploys_recessed_pull(sliding):
    import mujoco
    original,meta,path=sliding["db0018_sliding_single"];p=meta["pocket_edge_pull"]
    xml=ET.parse(path/"door.xml");world=xml.getroot().find("worldbody")
    finger=ET.SubElement(world,"body",{"name":"qa_fingertip","mocap":"true","pos":"0 0 5"})
    ET.SubElement(finger,"geom",{"name":"qa_fingertip_geom","type":"sphere","size":"0.006","friction":"0.6 0.005 0.0001"})
    probe=path/"contact_probe.xml";xml.write(probe)
    m=mujoco.MjModel.from_xml_path(str(probe));d=mujoco.MjData(m)
    qs=int(m.jnt_qposadr[m.joint(p["leaf_joint"]).id]);qa=int(m.jnt_qposadr[m.joint(p["joint"]).id])
    d.qpos[qs]=p["recessed_leaf_q"];mujoco.mj_forward(m,d)
    point=d.site_xpos[m.site(p["press_site"]).id].copy();direction=np.asarray(p["press_direction"])
    contacts=0
    for travel in np.linspace(-.014,.011,350):
        d.mocap_pos[0]=point+direction*travel;mujoco.mj_step(m,d)
        contacts+=sum(m.geom(c.geom1).name=="qa_fingertip_geom" or m.geom(c.geom2).name=="qa_fingertip_geom" for c in d.contact)
    assert contacts>0
    assert d.qpos[qa]>.70,(d.qpos,contacts)
    assert abs(d.qpos[qs]-p["recessed_leaf_q"])<.002


def test_pocket_travel_shortfall_is_not_hidden(sliding):
    m,meta,_=sliding["db0018_sliding_single"]
    broken=copy.deepcopy(meta);broken["pocket_edge_pull"]["recessed_leaf_q"]-=.05
    report=run_sliding_mechanics_qa(m,broken)
    assert not report["ok"]
    assert any(f["check"]=="edge_not_at_pocket_mouth" for f in report["failures"])


def test_filled_mortise_fails_even_with_parent_child_filter(sliding):
    m,meta,_=sliding["db0018_sliding_single"]
    geom=m.geom("leaf_edge_pull_case_back").id;old=m.geom_size[geom].copy()
    try:
        m.geom_size[geom,0]=.040
        r=run_sliding_mechanics_qa(m,meta)
        assert not r["ok"]
        assert any(f["check"] in ("edge_pull_sweep_collision","recessed_press_occluded") for f in r["failures"])
    finally:m.geom_size[geom]=old


def test_removed_rocker_spring_rejected(sliding):
    m,meta,_=sliding["db0018_sliding_single"];j=m.joint(meta["pocket_edge_pull"]["joint"]).id;old=m.jnt_stiffness[j]
    try:
        m.jnt_stiffness[j]=0
        assert any(f["check"]=="edge_pull_missing_return_spring" for f in run_sliding_mechanics_qa(m,meta)["failures"])
    finally:m.jnt_stiffness[j]=old


def test_right_running_rail_moved_is_not_waived(sliding):
    m,meta,_=sliding["db0008_sliding_bypass"];g=m.geom(meta["sliding_track_supports"][0]["channel_running_rails"][1]).id;old=m.geom_pos[g].copy()
    try:
        m.geom_pos[g,2]-=.012
        r=run_sliding_track_qa(m,meta)
        assert not r["ok"] and any(f["check"]=="wheel_contact" for f in r["failures"])
    finally:m.geom_pos[g]=old


@pytest.mark.parametrize("door",["db0018_sliding_single","db0008_sliding_bypass"])
def test_end_stop_bears_load_without_joint_limit(sliding,door):
    import mujoco
    m,meta,_=sliding[door];support=meta["sliding_track_supports"][0]
    j=m.joint(support["joint"]).id;qa=int(m.jnt_qposadr[j]);dof=int(m.jnt_dofadr[j]);old=m.jnt_limited[j]
    d=mujoco.MjData(m);d.qpos[qa]=support["nominal_range"][1]-.01
    try:
        m.jnt_limited[j]=False
        force=float(m.dof_frictionloss[dof]+40)
        for _ in range(400):
            d.qfrc_applied[dof]=force;mujoco.mj_step(m,d)
        assert support["nominal_range"][1]-.001<d.qpos[qa]<support["nominal_range"][1]+.002
        stop_ids={m.geom(n).id for n in support["end_stops"]}
        carriage_ids={m.geom(n).id for n in support["carriage_collision_geoms"]}
        assert any((c.geom1 in stop_ids and c.geom2 in carriage_ids) or (c.geom2 in stop_ids and c.geom1 in carriage_ids) for c in d.contact)
    finally:m.jnt_limited[j]=old


def test_endpoint_braking_still_rejects_obstruction_before_full_travel(sliding):
    """A real static block 70 mm into travel remains a jam with endpoint braking."""
    import mujoco
    from doorbench.qa import JAM_FORCE_N, jam_sweep

    original,meta,path=sliding["db0008_sliding_bypass"]
    support=meta["sliding_track_supports"][0]
    joint=original.joint(support["joint"]).id
    push=float(original.dof_frictionloss[original.jnt_dofadr[joint]]+70)
    endpoint=support["nominal_range"][1]
    clear=jam_sweep(original,mujoco.MjData(original),joint,push,.05,end_position=endpoint)
    assert clear["peak_force_N"]<JAM_FORCE_N,clear
    assert clear["moved"]>.2,clear

    xml=ET.parse(path/"door.xml")
    world=xml.getroot().find("worldbody")
    body=original.body(support["body"]).id
    obstruction=original.body_pos[body]+np.array([support["leaf_width_m"]/2+.08,0,1.2])
    ET.SubElement(world,"geom",{
        "name":"qa_midtravel_obstruction","type":"box","size":"0.01 0.012 0.2",
        "pos":" ".join(map(str,obstruction)),"friction":"0.6 0.005 0.0001",
    })
    probe=path/"obstruction_probe.xml";xml.write(probe)
    blocked=mujoco.MjModel.from_xml_path(str(probe))
    result=jam_sweep(blocked,mujoco.MjData(blocked),blocked.joint(support["joint"]).id,
                     push,.05,end_position=endpoint)
    assert result["peak_force_N"]>JAM_FORCE_N,result
    assert "qa_midtravel_obstruction" in result["peak_pair"],result
    assert result["moved"]<.2<endpoint,result

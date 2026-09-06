"""Gate latch checks on compiled native geometry and explicit force rules.

This is a mounting/release/latching regression gate, not a product certification,
strength calculation, pool-safety check, or cross-engine dynamics certificate.
"""
from __future__ import annotations

import numpy as np


def probe_magnetic_latch(model, metadata):
    """Continuous hold/lift/open/release/reclose/rehold test on a fresh MjData.

    Native contact solves load transfer. Only the declared magnetic interaction
    supplements MJCF; no latch weld, joint lock, or forced latch coordinates are
    used. The closing phase applies bounded leaf torque, not a pose reset.
    """
    import mujoco
    from .geometry.gate_hardware import compile_magnetic_latches, apply_magnetic_latches

    rules = compile_magnetic_latches(model, metadata)
    hardware = metadata["gate_hardware"][0]
    magnetic = hardware["kind"] == "magnetic_top_pull"
    hinge = model.joint(metadata["primary_joint"])
    pin = model.joint(hardware["operator_joint"])
    ha, hd = int(hinge.qposadr[0]), int(hinge.dofadr[0])
    pa, pd = int(pin.qposadr[0]), int(pin.dofadr[0])
    data = mujoco.MjData(model)
    states = {}
    max_contact_penetration = 0.
    worst_contact = None
    previous = mujoco.get_mjcb_passive()

    def callback(m, d):
        if m is model:
            apply_magnetic_latches(m, d, rules)

    def stage(name, seconds, torque, pull):
        nonlocal max_contact_penetration, worst_contact
        for _ in range(round(seconds / model.opt.timestep)):
            data.qfrc_applied[:] = 0
            data.qfrc_applied[hd] = torque(data) if callable(torque) else torque
            data.qfrc_applied[pd] = pull
            mujoco.mj_step(model, data)
            if data.ncon:
                contact = min(data.contact, key=lambda c: c.dist)
                if -float(contact.dist) > max_contact_penetration:
                    max_contact_penetration = -float(contact.dist)
                    worst_contact = {"stage": name, "time_s": float(data.time),
                        "pair": [model.geom(contact.geom1).name, model.geom(contact.geom2).name]}
        mujoco.mj_forward(model, data)
        states[name] = {"door_rad": float(data.qpos[ha]), "pin_m": float(data.qpos[pa]),
                        "time_s": float(data.time)}

    mujoco.set_mjcb_passive(callback)
    try:
        stage("settle", .5, 0., 0.)
        stage("hold", 1., 30., 0.)
        stage("lift", .4, 0., 22.2)
        stage("open", 1.2, 30., 22.2)
        open_position = float(data.qpos[ha])
        # Hold the gate itself while letting go of its independent release.
        # A fast self-closer must not move the keeper back under the pin before
        # the test has observed the pin's absent-keeper return behavior.
        stage("release", .7, lambda d: np.clip(
            40*(open_position-d.qpos[ha])-12*d.qvel[hd], -30, 30), 0.)
        # Approach the closed stop at a bounded 0.35 rad/s, slowing further in
        # the last 0.12 rad. A high-gain angle step would test an imposed slam.
        if magnetic:
            closing = lambda d: np.clip(40*(np.clip(-3*d.qpos[ha], -.35, .35)-d.qvel[hd]), -20, 20)
        else:
            # The spring latch has a real seating force. Continue a bounded
            # manual closing push over its ramp, then brake once the pin drops.
            # An angle-PD target at exactly zero would stall before the ramp.
            seated = False
            def closing(d):
                nonlocal seated
                seated = seated or (abs(d.qpos[ha]) < .003 and d.qpos[pa] < .002)
                return np.clip(-40*d.qvel[hd] if seated else 40*(-.30-d.qvel[hd]), -20, 20)
        stage("reclose", 6., closing, 0.)
        stage("rehold", 1., 30., 0.)
    finally:
        mujoco.set_mjcb_passive(previous)
    travel = hardware["release_travel_m"]
    checks = {
        "initially_engaged": abs(states["settle"]["pin_m"]) < .002,
        "withstands_30Nm_without_release": abs(states["hold"]["door_rad"]) < .006,
        "22_2N_releases": states["lift"]["pin_m"] > travel-.001,
        "released_gate_opens": states["open"]["door_rad"] > .5,
        ("pin_returns_up_when_striker_absent" if magnetic else "pin_returns_down_when_released"):
            states["release"]["pin_m"] > travel-.001 if magnetic else abs(states["release"]["pin_m"]) < .002,
        "gate_closes": abs(states["reclose"]["door_rad"]) < .006,
        ("magnet_recaptures_pin" if magnetic else "spring_pin_recaptures"):
            abs(states["reclose"]["pin_m"]) < .002,
        "recaptured_pin_holds": abs(states["rehold"]["door_rad"]) < .006,
        "finite_state": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "no_native_warnings": not bool(np.any(data.warning.number)),
        "contact_penetration_below_1mm": max_contact_penetration < .001,
    }
    return {"ok": all(checks.values()), "checks": checks, "states": states,
            "max_contact_penetration_m": max_contact_penetration,
            "worst_contact": worst_contact,
            "scope": "Native MJCF contacts plus declared conservative magnetic potential when applicable; bounded applied forces"}


def run_gate_hardware_qa(model, spec, metadata, *, dynamic=True):
    """Check explicitly bound mounting, connected rod, release gap and behavior."""
    import mujoco
    op = spec.get("operator", {}).get("model")
    if op == "thumb_latch_suffolk":
        return run_suffolk_hardware_qa(model,spec,metadata,dynamic=dynamic)
    if op == "gate_latch_fork":
        return run_fork_hardware_qa(model, spec, metadata, dynamic=dynamic)
    if op not in ("gate_latch_magnetic", "baby_gate_latch"):
        return {"ok": True, "applicable": False, "failures": []}
    rows = metadata.get("gate_hardware", [])
    kind = "magnetic_top_pull" if op == "gate_latch_magnetic" else "spring_lift_pin"
    travel = .030 if op == "gate_latch_magnetic" else .020
    if len(rows) != 1 or rows[0].get("kind") != kind:
        return {"ok": False, "applicable": True, "failures": ["Missing lift-latch mechanism evidence"]}
    row = rows[0]
    data = mujoco.MjData(model)
    # No dynamics callback is needed to measure static authored geometry.
    mujoco.mj_kinematics(model, data)
    failures, attachment_distances = [], []
    for first, second in row["attachments"]:
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in (first, second)]
        if min(ids) < 0:
            failures.append({"missing_attachment_geometry": [first, second]})
            continue
        distance = float(mujoco.mj_geomDistance(model, data, *ids, 1., None))
        attachment_distances.append({"pair": [first, second], "gap_m": distance})
        if distance > .0005:
            failures.append({"detached": [first, second], "gap_m": distance})
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, row["operator_joint"])
    release = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, row["release_site"])
    if min(jid, release) < 0:
        failures.append("Missing actual release joint/site")
    else:
        height = float(data.site_xpos[release, 2])
        if abs(height - spec["operator"]["height"]) > .0001:
            failures.append({"release_height_m": height, "specified_m": spec["operator"]["height"]})
        if abs(model.jnt_range[jid, 1] - travel) > 1e-6:
            failures.append(f"Release travel differs from the {travel*1000:g} mm mechanism")
    pull_bodies = []
    for name in row["pull_sites"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            failures.append({"missing_leaf_pull": name})
        else:
            body = model.body(int(model.site_bodyid[sid])).name
            pull_bodies.append(body)
            if body != "leaf":
                failures.append({"leaf_pull_on_wrong_body": [name, body]})
    release_distances = []
    if jid >= 0:
        data.qpos[model.jnt_qposadr[jid]] = model.jnt_range[jid, 1]
        mujoco.mj_kinematics(model, data)
        pin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, row["pin_geom"])
        for name in row["keeper_geoms"]:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if min(pin, gid) < 0:
                failures.append({"missing_load_bearing_geometry": name})
                continue
            gap = float(mujoco.mj_geomDistance(model, data, pin, gid, 1., None))
            release_distances.append(gap)
            if gap < .001:
                failures.append({"pin_not_clear_when_released": name, "gap_m": gap})
    motion = probe_magnetic_latch(model, metadata) if dynamic and not failures else None
    if motion is not None and not motion["ok"]:
        failures.append({"native_behavior": [k for k, v in motion["checks"].items() if not v]})
    return {"ok": not failures, "applicable": True, "failures": failures,
            "attachments": attachment_distances, "released_keeper_gaps_m": release_distances,
            "pull_bodies": pull_bodies, "native_behavior": motion,
            "scope": "Connected mounting and release geometry, plus optional native magnetic latch cycle"}


def run_fork_hardware_qa(model, spec, metadata, *, dynamic=True):
    """Exact mounting and 81-sample release/door sweeps, bypassing body filters."""
    import mujoco
    rows = metadata.get("gate_hardware", [])
    if len(rows) != 1 or rows[0].get("kind") != "gravity_fork":
        return {"ok": False, "applicable": True, "failures": ["Missing fork mechanism evidence"]}
    row = rows[0]
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model,d)
    failures, attachments = [], []
    def gap(a,b):
        return float(mujoco.mj_geomDistance(model,d,model.geom(a).id,model.geom(b).id,1.,None))
    for a,b in row["attachments"]:
        distance=gap(a,b)
        attachments.append({"pair":[a,b],"gap_m":distance})
        if distance>.0005:
            failures.append({"detached":[a,b],"gap_m":distance})
    joint=model.joint(row["operator_joint"])
    hinge=model.joint(metadata["primary_joint"])
    ja,ha=int(joint.qposadr[0]),int(hinge.qposadr[0])
    expected=.001 if row["locked"] else row["release_travel_rad"]
    if abs(model.jnt_range[joint.id,1]-expected)>1e-6:
        failures.append("Actual fork joint range differs from declared mechanism")
    height=float(d.site_xpos[model.site(row["release_site"]).id,2])
    if abs(height-spec["operator"]["height"])>.0001:
        failures.append({"release_height_m":height,"specified_m":spec["operator"]["height"]})
    support=row["support_geom"]
    sweep={"operator_min_gap_m":1.,"released_post_min_gap_m":1.}
    # Even the locked example is checked against its unlocked design envelope.
    for q in np.linspace(0.,row["release_travel_rad"],81):
        d.qpos[ja]=q
        mujoco.mj_kinematics(model,d)
        for a in row["moving_geoms"]:
            for b in row["fixed_mount_geoms"]+[support]:
                distance=gap(a,b)
                if distance<sweep["operator_min_gap_m"]:
                    sweep.update(operator_min_gap_m=distance,operator_worst_pair=[a,b],operator_q_rad=float(q))
    if sweep["operator_min_gap_m"]<.0005:
        failures.append({"fork_release_obstructed":sweep})
    post_geoms=[model.geom(i).name for i in range(model.ngeom)
                if model.geom(i).name == row["post_geom"] or model.geom(i).name.startswith(row["post_geom"]+"_")]
    # A socket/pocket can split the authored post into several real solids.
    # Include those actual pieces, including the cap, rather than an absent name.
    if not post_geoms:
        failures.append("Missing actual post geometry")
    d.qpos[ja]=row["release_travel_rad"]
    for q in np.linspace(*model.jnt_range[hinge.id],81):
        d.qpos[ha]=q
        mujoco.mj_kinematics(model,d)
        for a in row["moving_geoms"]+row["fixed_mount_geoms"]:
            distance=min((gap(a,b) for b in post_geoms),default=-1.)
            if distance<sweep["released_post_min_gap_m"]:
                sweep.update(released_post_min_gap_m=distance,released_post_worst_geom=a,door_q_rad=float(q))
    if sweep["released_post_min_gap_m"]<.001:
        failures.append({"released_fork_hits_post":sweep})
    native=probe_fork_latch(model,metadata) if dynamic and not failures else None
    if native and not native["ok"]:
        failures.append({"native_behavior":[k for k,v in native["checks"].items() if not v]})
    return {"ok":not failures,"applicable":True,"failures":failures,"attachments":attachments,
            "sweeps":sweep,"native_behavior":native,
            "scope":"Exact native primitive distances including parent-child pairs; optional native fork cycle"}


def probe_fork_latch(model,metadata):
    """Force-driven lift-to-open AND lift-to-close; never writes qpos."""
    import mujoco
    row=metadata["gate_hardware"][0]
    d=mujoco.MjData(model)
    joint=model.joint(row["operator_joint"]);hinge=model.joint(metadata["primary_joint"])
    ja,jd=int(joint.qposadr[0]),int(joint.dofadr[0])
    ha,hd=int(hinge.qposadr[0]),int(hinge.dofadr[0])
    states={};penetration=0.
    max_door_torque=40.*abs(metadata["leaf_edge_x_local"])
    hinge_friction=float(model.dof_frictionloss[hd])
    def stage(name,seconds,door,operator):
        nonlocal penetration
        for _ in range(round(seconds/model.opt.timestep)):
            d.qfrc_applied[:]=0
            d.qfrc_applied[hd]=door(d) if callable(door) else door
            d.qfrc_applied[jd]=operator(d) if callable(operator) else operator
            mujoco.mj_step(model,d)
            if d.ncon: penetration=max(penetration,-min(float(c.dist) for c in d.contact))
        states[name]={"door_rad":float(d.qpos[ha]),"fork_rad":float(d.qpos[ja])}
    stage("settle",.5,0,0)
    stage("hold",1,30,0)
    # 1.5 Nm at a real ~50–100 mm grip is a modest one-hand lifting force.
    lifting=lambda d:np.clip(8*(row["release_travel_rad"]-d.qpos[ja])-.12*d.qvel[jd],-1.5,1.5)
    stage("lift",1,0,lifting)
    target=min(.9,float(model.jnt_range[hinge.id,1])*.75)
    opening=lambda d:np.clip(70*(np.clip(3*(target-d.qpos[ha]),-.4,.4)-d.qvel[hd])
                            +hinge_friction, -max_door_torque,max_door_torque)
    stage("open",5,opening,lifting)
    if not row["locked"]:
        opened=float(d.qpos[ha])
        holding=lambda d:np.clip(80*(opened-d.qpos[ha])-40*d.qvel[hd],
                                 -max_door_torque,max_door_torque)
        stage("drop_open",2,holding,0)
        stage("relift",1,holding,lifting)
        closing=lambda d:np.clip(70*(np.clip(-3*d.qpos[ha],-.35,.35)-d.qvel[hd])
                                 -hinge_friction-1,-max_door_torque,max_door_torque)
        stage("reclose",8,closing,lifting)
        stage("drop_closed",2,closing,0)
        stage("rehold",1,30,0)
    checks={"holds_without_lift":abs(states["hold"]["door_rad"])<.008,
            "finite":bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
            "no_native_warnings":not bool(np.any(d.warning.number)),
            "contact_penetration_below_1mm":penetration<.001}
    if row["locked"]:
        checks.update(padlock_limits_fork=states["lift"]["fork_rad"]<.005,
                      padlocked_gate_holds=abs(states["open"]["door_rad"])<.008)
    else:
        checks.update(lift_clears_post=states["lift"]["fork_rad"]>1.44,
            released_gate_opens=states["open"]["door_rad"]>.5,
            fork_returns_by_gravity=abs(states["drop_open"]["fork_rad"])<.02,
            raised_fork_allows_closing=abs(states["reclose"]["door_rad"])<.005,
            dropped_fork_recaptures=abs(states["drop_closed"]["fork_rad"])<.02,
            recaptured_fork_holds=abs(states["rehold"]["door_rad"])<.008)
    return {"ok":all(checks.values()),"checks":checks,"states":states,
            "max_contact_penetration_m":penetration,
            "max_applied_leaf_torque_Nm":max_door_torque,
            "equivalent_leaf_pull_force_limit_N":40.,
            "scope":"Native contact, gravity and ideal joints/limits; bounded applied torque, no coordinate resets"}


def probe_suffolk_latch(model,metadata):
    """Both release methods, native tang/bar transmission and ramp recapture."""
    import mujoco
    row=metadata['gate_hardware'][0]
    d=mujoco.MjData(model)
    t=model.joint(row['thumb_joint']);b=model.joint(row['bar_joint']);h=model.joint(metadata['primary_joint'])
    ta,td=int(t.qposadr[0]),int(t.dofadr[0]);ba,bd=int(b.qposadr[0]),int(b.dofadr[0]);ha,hd=int(h.qposadr[0]),int(h.dofadr[0])
    limit=40*abs(metadata['leaf_edge_x_local']);friction=float(model.dof_frictionloss[hd])
    states={};penetration=0.;worst=None;transmission_frames=0
    def stage(name,seconds,door,thumb=0.,bar=0.):
        nonlocal penetration,worst,transmission_frames
        for _ in range(round(seconds/model.opt.timestep)):
            d.qfrc_applied[:]=0;d.qfrc_applied[hd]=door(d) if callable(door) else door
            d.qfrc_applied[td]=thumb;d.qfrc_applied[bd]=bar
            mujoco.mj_step(model,d)
            for c in d.contact:
                names={model.geom(c.geom1).name,model.geom(c.geom2).name}
                if names=={row['tang_geom'],row['bar_geom']} and name=='thumb_press':transmission_frames+=1
                if -c.dist>penetration:
                    penetration=-float(c.dist);worst={'stage':name,'pair':sorted(names),'time_s':float(d.time)}
        mujoco.mj_forward(model,d)
        states[name]={'door_rad':float(d.qpos[ha]),'thumb_rad':float(d.qpos[ta]),'bar_rad':float(d.qpos[ba])}
    stage('settle',.5,0.)
    stage('hold',1.,30.)
    # Remove the preceding hold load against the catch before operating it.
    # A modest seating push overcomes hinge stiction and unloads the bar.
    stage('thumb_press',1.,-friction-1.,.5)
    opening=lambda d:np.clip(70*(np.clip(3*(.8-d.qpos[ha]),-.4,.4)-d.qvel[hd])+friction,-limit,limit)
    stage('open',4.,opening,.5)
    if row.get('independent_blocking_lock',False):
        stage('release_blocked',1.,0.)
        stage('direct_bar_lift',1.,-friction-1.,0.,.4)
        checks={
            'native_tang_transmits_load':transmission_frames>10,
            'thumb_press_lifts_bar':states['thumb_press']['thumb_rad']>.29 and states['thumb_press']['bar_rad']>.08,
            'independent_lock_still_blocks_leaf':abs(states['open']['door_rad'])<.03,
            'direct_bar_release_works':states['direct_bar_lift']['bar_rad']>.20,
            'finite':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
            'no_native_warnings':not bool(np.any(d.warning.number)),
            'contact_penetration_below_1mm':penetration<.001}
        return {'ok':all(checks.values()),'checks':checks,'states':states,
                'tang_transmission_contact_samples':transmission_frames,'max_contact_penetration_m':penetration,
                'worst_contact':worst,'scope':'Suffolk release works while independent native lock remains blocking; no traversal/closing-cycle claim'}
    opened=float(d.qpos[ha])
    holding=lambda d:np.clip(80*(opened-d.qpos[ha])-40*d.qvel[hd],-limit,limit)
    stage('release_open',1.5,holding)
    closing=lambda d:np.clip(70*(np.clip(-3*d.qpos[ha],-.30,.30)-d.qvel[hd])-friction-1,-limit,limit)
    stage('reclose',7.,closing)
    stage('rehold',1.,30.)
    stage('direct_bar_lift',1.,-friction-1.,0.,.4)
    checks={
        'native_tang_transmits_load':transmission_frames>10,
        'holds_before_release':abs(states['hold']['door_rad'])<.008,
        'thumb_press_lifts_bar':states['thumb_press']['thumb_rad']>.29 and states['thumb_press']['bar_rad']>.08,
        'released_gate_opens':states['open']['door_rad']>.5,
        'gravity_bar_returns':states['release_open']['bar_rad']<.02,
        'thumb_returns':abs(states['release_open']['thumb_rad'])<.01,
        'gate_recloses':abs(states['reclose']['door_rad'])<.008,
        'bar_recaptures_keeper':states['reclose']['bar_rad']<.02,
        'recaptured_bar_holds':abs(states['rehold']['door_rad'])<.008,
        'direct_bar_release_works':states['direct_bar_lift']['bar_rad']>.20,
        'finite':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
        'no_native_warnings':not bool(np.any(d.warning.number)),
        'contact_penetration_below_1mm':penetration<.001}
    return {'ok':all(checks.values()),'checks':checks,'states':states,
            'tang_transmission_contact_samples':transmission_frames,'max_contact_penetration_m':penetration,
            'worst_contact':worst,'scope':'Native tang/bar contact, keeper ramps, gravity and ideal bearings; no equality or coordinate resets'}


def run_suffolk_hardware_qa(model,spec,metadata,*,dynamic=True):
    import mujoco
    rows=metadata.get('gate_hardware',[])
    if len(rows)!=1 or rows[0].get('kind')!='contact_suffolk':
        return {'ok':False,'applicable':True,'failures':['Missing contact-driven Suffolk mechanism evidence']}
    row=rows[0];d=mujoco.MjData(model);mujoco.mj_kinematics(model,d)
    failures=[];attachments=[]
    def names(group):
        result=[]
        for n in group:
            if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n)>=0:result.append(n)
            elif n in ('post_latch','jamb_strike'):result.extend(model.geom(i).name for i in range(model.ngeom) if model.geom(i).name.startswith(n+'_'))
        return result
    def gap(a,b):return float(mujoco.mj_geomDistance(model,d,model.geom(a).id,model.geom(b).id,1.,None))
    for edge in row['attachments']:
        a,b=names(edge['first']),names(edge['second'])
        distance=min((gap(x,y) for x in a for y in b),default=1.)
        attachments.append({'label':edge['label'],'gap_m':distance})
        if distance>.0005:failures.append({'detached':edge,'gap_m':distance})
    tid=model.joint(row['thumb_joint']).id;bid=model.joint(row['bar_joint']).id
    if model.neq and any(int(model.eq_obj1id[i]) in (tid,bid) or int(model.eq_obj2id[i]) in (tid,bid)
                        for i in range(model.neq) if model.eq_type[i]==mujoco.mjtEq.mjEQ_JOINT):
        failures.append('Joint equality substitutes for physical thumb/bar contact')
    qadr=model.jnt_qposadr[tid]
    # All leaf solids and backing plates must have a real through-slot. The
    # tang/bar contact is deliberately omitted from this obstruction query.
    native_leaf=model.body('leaf').id
    solids=[model.geom(i).name for i in range(model.ngeom) if model.geom_bodyid[i]==native_leaf
            and (model.geom(i).name.startswith('leaf_slab') or model.geom(i).name.startswith('leaf_picket')
                 or model.geom(i).name in row['plate_geoms'])]
    least=(1.,None,None)
    bearings=[model.geom(i).name for i in range(model.ngeom)
              if model.geom(i).name.startswith('leaf_thumb_bearing_')]
    moving=[model.geom(i).name for i in range(model.ngeom)
            if model.geom_bodyid[i]==model.body('leaf_thumb').id]
    bearing_least=(1.,None,None,None)
    for q in np.linspace(0,.30,81):
        d.qpos[qadr]=q;mujoco.mj_kinematics(model,d)
        for n in solids:
            distance=gap(row['tang_geom'],n)
            if distance<least[0]:least=(distance,n,float(q))
        for a in moving:
            for b in bearings:
                distance=gap(a,b)
                if distance<bearing_least[0]:bearing_least=(distance,a,b,float(q))
    if least[0]<.0005:failures.append({'obstructed_tang_slot':least})
    if bearing_least[0]<.0005:failures.append({'thumb_bearing_interference':bearing_least})
    native=probe_suffolk_latch(model,metadata) if dynamic and not failures else None
    if native and not native['ok']:failures.append({'native_behavior':[k for k,v in native['checks'].items() if not v]})
    return {'ok':not failures,'applicable':True,'failures':failures,'attachments':attachments,
            'tang_slot_min_gap_m':least[0],'thumb_bearing_min_gap_m':bearing_least[0],'native_behavior':native,
            'scope':'Explicit actual supports, through-slot clearance and contact-driven native operation'}


class SuffolkContactPreview:
    """Private native pose conditioning for geometry inspection ONLY.

    Narrow joint limits prescribe the requested active joint and leaf pose in
    a private model. Passive contact response is integrated and checked. This
    is not a benchmark controller or a dynamics-success certificate. Increasing
    targets with the same held leaf reuse the preceding settled state.
    """
    def __init__(self, model, metadata):
        import copy
        import mujoco
        row=next((r for r in metadata.get('gate_hardware',[]) if r.get('kind')=='contact_suffolk'),None)
        if row is None:raise ValueError('Suffolk contact metadata is required')
        self.model=copy.copy(model)
        self.data=mujoco.MjData(self.model)
        self.row=row
        self.primary=self.model.joint(metadata['primary_joint']).id
        self.allowed={row['thumb_joint'],row['bar_joint']}
        self.original_ranges=self.model.jnt_range.copy()
        self.original_limited=self.model.jnt_limited.copy()
        self.cache={};self.last=None

    def resolve(self, qpos, driven_joint):
        import mujoco
        m,d=self.model,self.data
        if driven_joint not in self.allowed:
            raise ValueError('Only the authored Suffolk active joints may be conditioned')
        q=np.asarray(qpos,dtype=float)
        if q.shape!=(m.nq,) or not np.isfinite(q).all():raise ValueError('Invalid inspection qpos')
        jid=m.joint(driven_joint).id;adr=int(m.jnt_qposadr[jid]);ha=int(m.jnt_qposadr[self.primary])
        target=float(q[adr]);held=float(q[ha]);key=(driven_joint,tuple(float(x) for x in q))
        if key in self.cache:
            cached=self.cache[key]
            return {**cached,'qpos':list(cached['qpos'])}
        if target<self.original_ranges[jid,0]-1e-6 or target>self.original_ranges[jid,1]+1e-6:
            return {'ok':False,'failures':['Requested active joint lies outside its authored limits']}
        if held<self.original_ranges[self.primary,0]-1e-6 or held>self.original_ranges[self.primary,1]+1e-6:
            return {'ok':False,'failures':['Requested leaf lies outside its authored limits']}
        passive_addrs={int(m.jnt_qposadr[m.joint(n).id]) for n in self.allowed}
        context=tuple(float(q[i]) for i in range(m.nq) if i not in passive_addrs)
        continued=self.last is not None and self.last[:2]==(driven_joint,context) and target>=self.last[2]
        if not continued:
            mujoco.mj_resetData(m,d)
            d.qpos[:]=q
            # Begin at the unpressed authored configuration, not a deeply
            # interpenetrating thumb-max/bar-zero configuration.
            d.qpos[m.jnt_qposadr[m.joint(self.row['thumb_joint']).id]]=0.
            d.qpos[m.jnt_qposadr[m.joint(self.row['bar_joint']).id]]=0.
        m.jnt_range[:]=self.original_ranges;m.jnt_limited[:]=self.original_limited
        m.jnt_limited[[jid,self.primary]]=1
        m.jnt_range[self.primary]=[held-1e-7,held+1e-7]
        initial=float(d.qpos[adr])
        increments=max(1,int(np.ceil(abs(target-initial)/.005)))
        for value in np.linspace(initial,target,increments+1)[1:]:
            m.jnt_range[jid]=[value-1e-7,value+1e-7]
            for _ in range(max(1,round(.025/m.opt.timestep))):mujoco.mj_step(m,d)
        m.jnt_range[jid]=[target-1e-7,target+1e-7]
        for _ in range(round(1.5/m.opt.timestep)):mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        residual=max(abs(float(d.qpos[adr])-target),abs(float(d.qpos[ha])-held))
        penetration=max((max(0.,-float(c.dist)) for c in d.contact),default=0.)
        speed=float(np.max(np.abs(d.qvel)))
        failures=[]
        if residual>.001:failures.append('Conditioned active/leaf pose not reached within 1 mrad')
        if penetration>.001:failures.append('Settled native contacts penetrate by more than 1 mm')
        if speed>.002:failures.append('Passive mechanism has not settled below 0.002 rad/s')
        if driven_joint==self.row['thumb_joint'] and target>.02:
            expected={m.geom(self.row['tang_geom']).id,m.geom(self.row['bar_geom']).id}
            if not any({c.geom1,c.geom2}==expected for c in d.contact):
                failures.append('Pressed thumb has no native load-transmitting contact with the passive bar')
        if any(abs(float(d.qpos[i])-float(q[i]))>.001 for i in range(m.nq)
               if i not in passive_addrs and i!=ha):failures.append('An unrelated joint changed during inspection conditioning')
        if np.any(d.warning.number) or not np.isfinite(d.qpos).all():failures.append('Native warning or nonfinite state')
        report={'ok':not failures,'failures':failures,'qpos':d.qpos.tolist(),
                'active_residual_rad':residual,'max_contact_penetration_m':penetration,
                'max_speed':speed,'scope':'Private kinematically conditioned inspection pose; passive response solved by native contact'}
        self.cache[key]=report
        self.last=(driven_joint,context,target) if not failures else None
        return {**report,'qpos':list(report['qpos'])}


def first_fork_contact_angle(model,metadata,*,direction=1.,max_angle=None):
    """First actual tine/post contact arrests a dropped-fork leaf scan.

    This does not waive any overlap: initial overlap is a failure, and a
    separate fully released sweep remains required. No native model is changed.
    """
    import mujoco
    row=next((r for r in metadata.get('gate_hardware',[]) if r.get('kind')=='gravity_fork'),None)
    if row is None:raise ValueError('Fork metadata is required')
    d=mujoco.MjData(model);h=model.joint(metadata['primary_joint']);ha=int(h.qposadr[0])
    posts=[model.geom(i).id for i in range(model.ngeom) if model.geom(i).name==row['post_geom'] or model.geom(i).name.startswith(row['post_geom']+'_')]
    tines=[model.geom(n).id for n in row['tine_geoms']]
    if not posts or not tines:return {'ok':False,'failure':'Missing intended load-contact geometry'}
    sign=1. if direction>=0 else -1.
    end=float(model.jnt_range[h.id,1] if sign>0 else model.jnt_range[h.id,0]) if max_angle is None else float(max_angle)
    if sign*end<0:raise ValueError('Scan angle and direction disagree')
    def clearance(angle):
        d.qpos[ha]=angle;mujoco.mj_kinematics(model,d)
        return min(float(mujoco.mj_geomDistance(model,d,a,b,1.,None)) for a in tines for b in posts)
    initial=clearance(0.)
    if initial<-.00001:return {'ok':False,'failure':'Dropped fork already intersects intended post','initial_gap_m':initial}
    previous=0.
    for angle in np.linspace(0.,end,129)[1:]:
        if clearance(float(angle))<=0:
            lo,hi=previous,float(angle)
            for _ in range(28):
                middle=(lo+hi)/2
                if clearance(middle)>0:lo=middle
                else:hi=middle
            return {'ok':True,'contact_angle_rad':lo,'initial_gap_m':initial,
                    'scope':'Exact first intended tine/post geometric contact; released sweep still mandatory'}
        previous=float(angle)
    return {'ok':False,'failure':'No intended fork/post arrest within authored scan','initial_gap_m':initial}

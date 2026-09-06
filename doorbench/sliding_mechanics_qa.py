"""Native-pose checks for usable sliding pulls, independent lanes and pocket access.

No model ranges or geometry are mutated. Locked doors are inspected at their
explicit nominal unlocked stroke, separately from dynamic access-control tests.
"""
from __future__ import annotations

import numpy as np


def _point_box_distance(model, data, geom, point):
    local = data.geom_xmat[geom].reshape(3,3).T @ (point-data.geom_xpos[geom])
    q = np.abs(local)-model.geom_size[geom]
    return float(np.linalg.norm(np.maximum(q,0))+min(float(max(q)),0))


def run_sliding_mechanics_qa(model, metadata, samples=25, tolerance=0.001):
    import mujoco

    if samples < 2 or not 0 < tolerance <= 0.003:
        raise ValueError("Use at least two samples and a tolerance in (0, 3 mm]")
    failures, measurements = [], {}
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    def fail(check, **details):
        failures.append({"check":check, **details})
    controls = metadata.get("sliding_leaf_controls", [])
    expected = metadata.get("family") in ("sliding_single","sliding_bypass","automatic_sliding","elevator","gate_sliding")
    if expected and not controls:
        fail("missing_leaf_controls")
    for control in controls:
        try:
            body = model.body(control["body"]).id
            joint = model.joint(control["joint"]).id
            if model.jnt_bodyid[joint] != body:
                fail("leaf_joint_body",body=control["body"])
            for site in control["grip_sites"]:
                if model.site_bodyid[model.site(site).id] != body:
                    fail("leaf_grip_attachment",site=site)
            if control.get("manual_contact_required") and not (control["grip_sites"] or control.get("operator_grip_sites")):
                fail("manual_leaf_missing_contact",body=control["body"])
            for name in control.get("operator_grip_sites",[]):
                owner=int(model.site_bodyid[model.site(name).id])
                if model.body_parentid[owner]!=body:
                    fail("leaf_operator_grip_attachment",site=name)
        except KeyError as error:
            fail("missing_control_geometry",reason=str(error))
    for support in metadata.get("sliding_track_supports",[]):
        if not support.get("suspension_model"):
            continue
        try:
            header=model.geom(support["header_geom"]).id
            roof=model.geom(support["rail"]+"_roof").id
            if not support.get("header_mounts"):
                fail("missing_header_mounts",body=support["body"])
            for name in support.get("header_mounts",[]):
                mount=model.geom(name).id
                # Check native geometry, including lateral overlap of outer
                # lanes with the lintel, not merely semantic attachment names.
                for other in (header,roof):
                    gap=mujoco.mj_geomDistance(model,data,mount,other,.10,None)
                    if gap>tolerance:
                        fail("floating_track_mount",body=support["body"],mount=name,other=model.geom(other).name,gap_m=float(gap))
                if model.geom_bodyid[mount]!=0:
                    fail("moving_track_mount",mount=name)
        except KeyError as error:
            fail("missing_suspension_geometry",reason=str(error))
    if metadata.get("family") == "sliding_bypass":
        if metadata.get("interaction_mode") != "independent_bypass":
            fail("bypass_interaction_mode")
        leaf_ids = {model.joint(c["joint"]).id for c in controls}
        for i in range(model.neq):
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and int(model.eq_obj1id[i]) in leaf_ids and int(model.eq_obj2id[i]) in leaf_ids:
                fail("bypass_unwanted_coupling",equality=model.equality(i).name)
        if any(not c["grip_sites"] for c in controls):
            fail("bypass_missing_pull")
    # A cup's empty interior must remain empty in the actual compiled geometry,
    # including invisible slab collision proxies. Its back is a real collider.
    cups = metadata.get("sliding_recessed_pulls", [])
    smallest_cup_clearance = float("inf")
    for cup in cups:
        try:
            body=model.body(cup["body"]).id
            p=data.xpos[body]+data.xmat[body].reshape(3,3)@np.asarray(cup["center"])
            half=np.asarray(cup["interior_half_size"])
            boxes=[g for g in range(model.ngeom) if model.geom_bodyid[g]==body and model.geom_contype[g] and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX]
            # Points a millimetre inside each side, plus centre: a decorative
            # plate on a solid slab fails, while the genuine open cup passes.
            offsets=[np.zeros(3)]
            offsets += [np.eye(3)[axis]*sign*(half[axis]-0.001) for axis in (0,2) for sign in (-1,1)]
            gap=min(_point_box_distance(model,data,g,p+data.xmat[body].reshape(3,3)@offset) for offset in offsets for g in boxes)
            smallest_cup_clearance=min(smallest_cup_clearance,gap)
            if gap < -1e-6:
                fail("filled_pull_cavity",body=cup["body"],grip=cup["grip_site"],gap_m=gap)
            if not cup.get("through_pull"):
                back=model.geom(cup["back_geom"]).id
                if model.geom_bodyid[back]!=body or not model.geom_contype[back]:
                    fail("pull_back_not_attached",grip=cup["grip_site"])
            # Approach the opening, not the metal side contact point. A second
            # cup back on a thin panel would otherwise block the first face.
            origin=p+np.array([0,cup["face"]*.10,0])
            ray=np.array([0,-cup["face"],0.0]);hit=np.array([-1],dtype=np.int32)
            dist=mujoco.mj_ray(model,data,origin,ray,None,True,-1,hit)
            if 0 <= dist < .10-1e-5:
                fail("pull_cavity_approach_blocked",grip=cup["grip_site"],hit=model.geom(int(hit[0])).name,distance_m=float(dist))
        except (KeyError,ValueError) as error:
            fail("missing_pull_geometry",reason=str(error))
    measurements["recessed_pulls_checked"]=len(cups)
    measurements["min_cup_interior_clearance_m"]=None if not cups else smallest_cup_clearance
    pull=metadata.get("pocket_edge_pull")
    if pull:
        try:
            leaf=model.body(pull["leaf_body"]).id
            rocker=model.body(pull["body"]).id
            joint=model.joint(pull["joint"]).id
            slide=model.joint(pull["leaf_joint"]).id
            qa,qs=int(model.jnt_qposadr[joint]),int(model.jnt_qposadr[slide])
            press=model.site(pull["press_site"]).id
            grip=model.site(pull["grip_site"]).id
            if model.body_parentid[rocker]!=leaf or model.jnt_bodyid[joint]!=rocker:
                fail("edge_pull_disconnected")
            if model.site_bodyid[press]!=rocker or model.site_bodyid[grip]!=rocker:
                fail("edge_pull_sites_disconnected")
            if model.jnt_stiffness[joint]<=0:
                fail("edge_pull_missing_return_spring")
            if not np.allclose(model.jnt_range[joint],pull["deploy_range"],atol=1e-9):
                fail("edge_pull_range")
            if abs(float(model.jnt_range[slide,1])-pull["recessed_leaf_q"])>1e-8:
                fail("pocket_native_travel_mismatch",native_max=float(model.jnt_range[slide,1]),required_max=pull["recessed_leaf_q"])
            data.qpos[qs]=pull["recessed_leaf_q"];data.qpos[qa]=0
            mujoco.mj_forward(model,data)
            edge=data.xpos[leaf,0]+pull["edge_local_x"]
            mouth_error=abs(float(edge-pull["pocket_mouth_x"]))
            measurements["recessed_edge_error_m"]=mouth_error
            if mouth_error>tolerance:
                fail("edge_not_at_pocket_mouth",error_m=mouth_error)
            push=model.site(pull['final_push_site']).id
            push_direction=np.asarray(pull['final_push_direction'],float)
            if model.site_bodyid[push]!=leaf or not np.allclose(push_direction,model.jnt_axis[slide]):
                fail('final_edge_push_attachment_or_direction')
            handoff=float(pull['final_push_switch_q'])
            if not 0<handoff<pull['recessed_leaf_q'] or abs(handoff+pull['final_push_handoff_margin_m']-pull['face_cup_occlusion_q'])>1e-8:
                fail('pocket_cup_handoff_threshold')
            rim_names={name for cup in metadata.get('sliding_recessed_pulls',[]) if cup['body']==pull['leaf_body'] for name in cup['side_geoms']}
            data.qpos[qs]=pull['face_cup_occlusion_q'];mujoco.mj_forward(model,data)
            leading=max(push_direction[0]*data.geom_xpos[model.geom(name).id,0]+
                        float(np.abs(data.geom_xmat[model.geom(name).id].reshape(3,3)[0])@model.geom_size[model.geom(name).id])
                        for name in rim_names)
            if abs(leading-push_direction[0]*pull['pocket_mouth_x'])>1e-6:
                fail('pocket_cup_handoff_not_at_actual_rim')
            for position in np.linspace(handoff,pull['recessed_leaf_q'],samples):
                data.qpos[qs]=position;mujoco.mj_forward(model,data)
                hit=np.array([-1],dtype=np.int32)
                distance=mujoco.mj_ray(model,data,data.site_xpos[push]-.08*push_direction,push_direction,None,True,-1,hit)
                if int(hit[0])!=model.geom(pull['final_push_geom']).id or abs(distance-.08)>1e-5:
                    fail('final_edge_push_occluded_or_off_surface',q=float(position),hit=model.geom(int(hit[0])).name if hit[0]>=0 else None)
            data.qpos[qs]=pull['recessed_leaf_q'];mujoco.mj_forward(model,data)
            direction=np.asarray(pull["press_direction"],float)
            origin=data.site_xpos[press]-direction*0.080
            hit=np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(model,data,origin,direction,None,True,-1,hit)
            if int(hit[0])!=model.geom(pull["press_geom"]).id or not 0.075<=distance<=0.085:
                fail("recessed_press_occluded",hit=model.geom(int(hit[0])).name if hit[0]>=0 else None,distance_m=float(distance))
            measurements["press_access_ray_m"]=float(distance)
            moving=[g for g in range(model.ngeom) if model.geom_bodyid[g]==rocker and model.geom_contype[g]]
            fixed=[g for g in range(model.ngeom) if model.geom_bodyid[g]!=rocker and model.geom_contype[g]]
            min_gap=float("inf")
            # Include parent-body geoms explicitly: MuJoCo's parent-child
            # filter alone would hide a rocker rotating through an uncut slab.
            for angle in np.linspace(*pull["deploy_range"],samples):
                data.qpos[qa]=angle;mujoco.mj_forward(model,data)
                for a in moving:
                    for b in fixed:
                        gap=mujoco.mj_geomDistance(model,data,a,b,0.003,None)
                        min_gap=min(min_gap,gap)
                        if gap < -0.00005:
                            fail("edge_pull_sweep_collision",q=float(angle),moving=model.geom(a).name,obstacle=model.geom(b).name,gap_m=float(gap))
            measurements["edge_pull_sweep_min_gap_m"]=float(min_gap)
            data.qpos[qa]=pull["minimum_grasp_q"]
            mujoco.mj_forward(model,data)
            projection=float(np.dot(data.site_xpos[grip]-np.array([edge,0,data.site_xpos[grip,2]]),-direction))
            measurements["grip_projection_at_minimum_q_m"]=projection
            if projection<0.024:
                fail("edge_pull_ungraspable_projection",projection_m=projection)
        except (KeyError,ValueError) as error:
            fail("missing_edge_pull_geometry",reason=str(error))
    return {"ok":not failures,"n_failures":len(failures),"failures":failures[:30],"measurements":measurements,
            "scope":"Compiled native geometry and sampled kinematics; dynamics are exercised by separate force/press regression tests."}

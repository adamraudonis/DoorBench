"""Original convex sliding pull hardware; dimensions and limits are documented in
docs/review/mechanical-foundations/sliding.md. No manufacturer CAD is embedded.

The edge pull is a single spring-return rocker: pressing its upper pad rotates
the lower paddle out of a real mortise. It is not a push-push latch.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..ir import ALL_TIERS, Body, Joint, QUAT_ID, Site, quat_to_mat
from . import common as C


def cut_box_recess(body, lower, upper, suffix):
    """Subtract an axis-aligned mortise from authored leaf/glass boxes.

    Split both visible and collision geometry; keep original volume density and
    distribute an explicit mass over the surviving pieces. Parent-child contact
    filtering must never stand in for the missing mortise.
    """
    low, high = np.asarray(lower, float), np.asarray(upper, float)
    if np.any(high <= low):
        raise ValueError("Recess must have positive dimensions")
    output, removed = [], []
    for geom in body.geoms:
        if geom.semantic not in ("leaf", "glass"):
            output.append(geom)
            continue
        if geom.type != "box" or not np.allclose(geom.quat, QUAT_ID):
            if geom.type == "box":
                ext=np.abs(quat_to_mat(geom.quat))@np.asarray(geom.size)
                if np.any(np.asarray(geom.pos)+ext <= low+1e-9) or np.any(np.asarray(geom.pos)-ext >= high-1e-9):
                    output.append(geom)
                    continue
            raise ValueError(f"Cannot mortise non-axis-aligned leaf geometry {geom.name}")
        a, b = np.asarray(geom.pos) - geom.size, np.asarray(geom.pos) + geom.size
        lo, hi = np.maximum(a, low), np.minimum(b, high)
        if np.any(hi-lo <= 1e-9):
            output.append(geom)
            continue
        removed.append(geom.name)
        segments = []
        core_a, core_b = a.copy(), b.copy()
        for axis in range(3):
            if lo[axis] > core_a[axis] + 1e-9:
                end = core_b.copy(); end[axis] = lo[axis]
                segments.append((core_a.copy(), end))
            if hi[axis] < core_b[axis] - 1e-9:
                start = core_a.copy(); start[axis] = hi[axis]
                segments.append((start, core_b.copy()))
            core_a[axis], core_b[axis] = lo[axis], hi[axis]
        original_volume = float(np.prod(b - a))
        for i, (aa, bb) in enumerate(segments):
            mass = None if geom.mass_override is None else geom.mass_override * float(np.prod(bb-aa)) / original_volume
            output.append(replace(geom, name=f"{geom.name}_{suffix}_{i}", pos=tuple((aa+bb)/2),
                                  size=tuple((bb-aa)/2), mass_override=mass))
    body.geoms = output
    return removed


def add_recessed_pull(model, body, op, direction, x, z, thickness, face, name):
    """A 12 mm deep open cup, with collision side walls and a real finger space.

    A thin mirrored panel uses a through-cutout and a rear cup, not an impossible
    blind recess in 6 mm glass. The 50 mm bypass lane gap clears that cup.
    """
    shape = op.style_params.get("shape", "flush")
    width, height = op.style_params.get("size", (0.050, 0.100))
    if shape == "cup":
        width = height = op.style_params.get("diameter", 0.055)
    wall, depth = 0.002, 0.012
    front = face * thickness / 2
    back = front - face * depth
    lower = (x-width/2, min(front, back)-wall, z-height/2)
    upper = (x+width/2, max(front, back)+wall, z+height/2)
    cut_box_recess(body, lower, upper, f"{name}_mortise_{'p' if face > 0 else 'n'}")
    mat = C.mat_from_material(model, op.material, f"mat_op_{op.material}")
    prefix = f"{name}_{'p' if face > 0 else 'n'}"
    center = (front+back)/2
    # Back-to-back cups cannot occupy a 10 mm glass panel. Use an open lined
    # finger aperture shared by the two faces, with no fictitious internal back.
    paired = next((p for p in model.meta.get("sliding_recessed_pulls", [])
                   if p["body"] == body.name and p["face"] == -face
                   and abs(p["center"][0]-x)<1e-9 and abs(p["center"][2]-z)<1e-9
                   and thickness < 2*(depth+wall)), None)
    if paired:
        names = set(paired["side_geoms"]+paired["end_geoms"]+[paired["back_geom"]])
        body.geoms = [g for g in body.geoms if g.name not in names]
        center, depth = 0.0, thickness
        paired.update({"center":[x,0.0,z],"depth_m":depth,"back_geom":None,"through_pull":True,
                       "interior_half_size":[width/2-wall,depth/2,height/2-wall]})
        for site in body.sites:
            if site.name in (paired["grip_site"],paired["close_grip_site"]):
                site.pos=(site.pos[0],0.0,site.pos[2])
    for sx in (-1, 1):
        body.geoms.append(C.box(f"{prefix}_side_{sx}", (x+sx*(width-wall)/2, center, z),
                                (wall/2, depth/2, height/2), mat, 2700, True, True, ALL_TIERS, "operator", "Recessed pull side wall"))
    for sz in (-1, 1):
        body.geoms.append(C.box(f"{prefix}_end_{sz}", (x, center, z+sz*(height-wall)/2),
                                (width/2-wall, depth/2, wall/2), mat, 2700, True, True, ALL_TIERS, "operator", "Recessed pull end wall"))
    if not paired:
        body.geoms.append(C.box(f"{prefix}_back", (x, back-face*wall/2, z),
                                (width/2, wall/2, height/2), mat, 2700, True, True, ALL_TIERS, "operator", "Recessed pull back"))
    else:
        paired["side_geoms"]=[f"{prefix}_side_{s}" for s in (-1,1)]
        paired["end_geoms"]=[f"{prefix}_end_{s}" for s in (-1,1)]
    # Two explicit contact points on the inner cup walls. The usual grip name is
    # the wall against which a finger pushes to open this leaf.
    grip = f"{name}_grip_{'p' if face > 0 else 'n'}"
    close_grip = f"{name}_close_grip_{'p' if face > 0 else 'n'}"
    for site, sign in ((grip, direction), (close_grip, -direction)):
        body.sites.append(Site(site, (x+sign*(width/2-wall), center, z), QUAT_ID, 0.004, "grip"))
    model.meta.setdefault("sliding_recessed_pulls", []).append({
        "body": body.name, "face": face, "grip_site": grip, "close_grip_site": close_grip,
        "center": [x, center, z], "interior_half_size": [width/2-wall, depth/2, height/2-wall],
        "depth_m": depth, "thin_panel_through_cutout": depth+wall > thickness,
        "back_geom":None if paired else f"{prefix}_back", "through_pull":bool(paired),
        "side_geoms": [f"{prefix}_side_{s}" for s in (-1, 1)],"end_geoms":[f"{prefix}_end_{s}" for s in (-1,1)],
    })
    return grip


def add_pocket_edge_pull(model, body, spec, direction, edge_x, height, nominal_travel):
    """Mortised 98 x 19 mm rocker; own simplified mechanics within an OEM envelope."""
    name = f"{body.name}_edge_pull"
    width, overall_height, depth = 0.019, 0.098, 0.028
    # Pivot near the top: a short upper press arm deploys a longer lower paddle.
    pivot_z = height + 0.014
    zlo, zhi = height-overall_height/2, height+overall_height/2
    inside_x = edge_x + direction*depth
    cut_box_recess(body, (min(edge_x, inside_x)-0.0001, -width/2, zlo),
                   (max(edge_x, inside_x)+0.0001, width/2, zhi), name+"_mortise")
    mat = C.mat_from_material(model, "stainless", "mat_pocket_edge_pull")
    wall = 0.0015
    for sy in (-1, 1):
        body.geoms.append(C.box(f"{name}_case_side_{sy}", ((edge_x+inside_x)/2, sy*(width-wall)/2, height),
                                (depth/2, wall/2, overall_height/2), mat, 7900, True, True, ALL_TIERS, "operator", "Edge pull mortise case"))
    body.geoms.append(C.box(f"{name}_case_back", (inside_x-direction*wall/2, 0, height),
                            (wall/2, width/2, overall_height/2), mat, 7900, True, True, ALL_TIERS, "operator", "Edge pull mortise back"))
    for zz, suffix in ((zlo+wall/2, "bottom"), (zhi-wall/2, "top")):
        body.geoms.append(C.box(f"{name}_case_{suffix}", ((edge_x+inside_x)/2, 0, zz),
                                (depth/2, width/2-wall, wall/2), mat, 7900, True, True, ALL_TIERS, "operator", "Edge pull case end"))
    # Flush faceplate rails have an open centre. In glass this case is carried by
    # an original metal patch fitting over a prepared edge notch, never screws
    # into the glass itself; pad/clamp geometry is below.
    for sy in (-1, 1):
        body.geoms.append(C.box(f"{name}_face_rail_{sy}", (edge_x+direction*wall/2, sy*(width-wall)/2, height),
                                (wall/2, wall/2, overall_height/2), mat, 7900, True, True, ALL_TIERS, "operator", "Flush edge faceplate"))
    for zz, suffix in ((zhi-0.004, "top"), (zlo+0.004, "bottom")):
        body.geoms.append(C.cyl(f"{name}_fixing_{suffix}", (edge_x+direction*0.001, 0, zz),
                                0.0023, 0.001, mat, (1,0,0), 7900, False, True, ALL_TIERS, "operator", "Case fixing screw"))
    if spec["leaf"]["panel_style"] == "glass_frameless":
        t = spec["leaf"]["thickness"]
        pad_mat = C.mat_from_material(model, "rubber", "mat_pocket_glass_pad")
        inside_cover = max(t/2+0.001, width/2+0.001)
        for face in (-1,1):
            # Pads bear on intact glass above/below the notch, leaving the
            # complete rocker cavity empty. The cover clears the wider case.
            for side in (-1,1):
                body.geoms.append(C.box(f"{name}_glass_pad_{face}_{side}", (edge_x+direction*0.023,face*(t/2+inside_cover)/2,height+side*0.056),
                                        (0.023,(inside_cover-t/2)/2,0.005),pad_mat,1100,True,True,ALL_TIERS,"operator","Glass patch gasket on intact glass"))
            body.geoms.append(C.box(f"{name}_glass_patch_{face}", (edge_x+direction*0.023,face*(inside_cover+0.001),height),
                                    (0.023,0.001,0.063),mat,7900,True,True,ALL_TIERS,"operator","Clamped edge-pull patch fitting"))
    rocker = Body(name, body.name, (edge_x+direction*0.002,0,pivot_z), QUAT_ID, None, [], [], ALL_TIERS, "operator", "Press to deploy pocket edge pull")
    rocker.joint = Joint(name+"_hinge", "hinge", (0,direction,0), (0,0,0), (0,0.95),
                         damping=0.045, frictionloss=0.008, stiffness=0.08, springref=-0.08,
                         armature=0.002, role="operator", label="Edge pull (+ = press upper pad to deploy lower grip)")
    rocker.geoms.append(C.box(name+"_press_pad", (0,0,0.018), (0.0015,0.0065,0.009), mat,7900,True,True,ALL_TIERS,"operator","Upper press pad"))
    # A true square bore around the 4 mm axle, assembled from convex parts.
    # The minimum radial gap stays 0.25 mm at every rocker angle.
    for sign in (-1,1):
        rocker.geoms.append(C.box(name+f"_hub_z_{sign}",(0,0,sign*0.0035),(0.00475,0.0065,0.00125),mat,7900,True,True,ALL_TIERS,"operator","Rocker bored hub"))
        rocker.geoms.append(C.box(name+f"_hub_x_{sign}",(sign*0.0035,0,0),(0.00125,0.0065,0.00225),mat,7900,True,True,ALL_TIERS,"operator","Rocker bored hub"))
    rocker.geoms.append(C.box(name+"_web_upper",(0,0,0.00675),(0.0015,0.0065,0.00225),mat,7900,True,True,ALL_TIERS,"operator","Upper rocker web"))
    rocker.geoms.append(C.box(name+"_web_lower",(0,0,-0.010375),(0.0015,0.0065,0.005625),mat,7900,True,True,ALL_TIERS,"operator","Lower rocker web"))
    rocker.geoms.append(C.box(name+"_paddle", (0,0,-0.034), (0.0015,0.0065,0.018), mat,7900,True,True,ALL_TIERS,"operator","Deployable lower finger paddle"))
    rocker.sites.append(Site(name+"_press", (-direction*0.0015,0,0.018), QUAT_ID,0.004,"push"))
    rocker.sites.append(Site(name+"_grip", (-direction*0.0015,0,-0.043), QUAT_ID,0.004,"grip"))
    body.geoms.append(C.cyl(name+"_axle", (rocker.pos[0],0,pivot_z),0.002,0.0095,mat,(0,1,0),7900,False,True,ALL_TIERS,"operator","Leaf-mounted rocker axle"))
    model.add_body(rocker)
    # Finish recessing the leaf by pushing its exposed leading edge. A face cup
    # disappears behind the pocket skin before the leaf reaches its end stop.
    # Use intact authored material above the rocker, never its moving paddle.
    push_geom = None
    for push_z in (height+.10, height+.20, height-.20):
        point = np.array([edge_x, 0., push_z])
        for geom in body.geoms:
            if not geom.collision or geom.semantic not in ('leaf','glass') or geom.type != 'box':
                continue
            if not np.allclose(geom.quat, QUAT_ID):
                continue
            low=np.asarray(geom.pos)-geom.size; high=np.asarray(geom.pos)+geom.size
            if np.all(point >= low-1e-8) and np.all(point <= high+1e-8):
                outer=float(geom.pos[0])-direction*float(geom.size[0])
                if abs(outer-edge_x)<1e-8:
                    push_geom=geom.name
                    break
        if push_geom: break
    if push_geom is None:
        raise ValueError('Pocket leaf has no intact leading-edge material for final push')
    body.sites.append(Site(name+'_final_push',tuple(point),QUAT_ID,.004,'push'))
    mouth=direction*spec['opening']['width']/2
    cups=[cup for cup in model.meta.get('sliding_recessed_pulls',[]) if cup['body']==body.name]
    if not cups:
        raise ValueError('Pocket final push requires an authored face-cup handoff')
    rim_names={g for cup in cups for g in cup['side_geoms']}
    leading_rim=max(direction*g.pos[0]+g.size[0] for g in body.geoms if g.name in rim_names)
    occlusion_q=direction*(mouth-body.pos[0])-leading_rim
    handoff_q=max(0.,min(nominal_travel,occlusion_q-.020))
    model.meta["pocket_edge_pull"] = {
        "body":name,"joint":rocker.joint.name,"leaf_body":body.name,"leaf_joint":body.joint.name,
        "press_site":name+"_press","grip_site":name+"_grip","press_direction":[direction,0,0],
        "extract_direction":[-direction,0,0],"deploy_range":[0,0.95],"minimum_grasp_q":0.70,
        "recessed_leaf_q":nominal_travel,"face_grip_after_extract_m":0.14,"spring_return":True,
        "edge_local_x":edge_x,"pocket_mouth_x":mouth,
        "final_push_site":name+'_final_push',"final_push_geom":push_geom,
        "final_push_direction":[direction,0,0],"final_push_switch_q":handoff_q,
        "face_cup_occlusion_q":occlusion_q,"final_push_handoff_margin_m":.020,
        "press_geom":name+"_press_pad","grip_geom":name+"_paddle",
        "envelope_m":{"height":overall_height,"width":width,"mortise_depth":depth},
        "glass_patch":spec["leaf"]["panel_style"]=="glass_frameless",
        "scope":"Original rocker with prescribed return spring; no push-push latch or strength certification.",
    }
    return rocker

"""Connected gate mechanisms and their explicitly approximate passive magnet.

The top-pull assembly follows D&D ML3TP's installation topology: a post-mounted
column guides a continuous release rod; a gate-mounted striker captures its
lower pin. The pin retracts when the striker is absent. It is not a ramp latch.
Dimensions that are not in the public drawing are engineering approximations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..ir import Body, Geom, Joint, Site, ALL_TIERS, QUAT_ID, quat_z_to
from . import common as C


def _box(geoms, name, pos, size, mat, semantic="latch", label=""):
    geoms.append(C.box(name, pos, size, mat, 1400, True, True, ALL_TIERS,
                       semantic, label or name))


def _screw(geoms, name, point, face, mat):
    # The screw penetrates its mounting substrate by 12 mm. Both head and shank
    # are authored; neither is an independent moving body.
    geoms.append(C.cyl(name + "_shank", (point[0], point[1] - face * .006, point[2]),
                       .002, .009, mat, (0, face, 0), 7850, False, True,
                       ALL_TIERS, "latch", "Mounting screw"))
    geoms.append(C.cyl(name + "_head", (point[0], point[1] + face * .004, point[2]),
                       .004, .002, mat, (0, face, 0), 7850, False, True,
                       ALL_TIERS, "latch", "Mounting screw head"))


def add_fork_latch(model, world, leaf_body, spec, *, u, v, hx, x_edge,
                   leaf_bottom, leaf_height, leaf_name="leaf"):
    """Stile-clamped flat fork with an outboard pivot and actual post capture.

    This is the common pressed-steel fork/clamp topology, dimensioned to the
    authored rectangular post. It must be lifted to both open and close; it is
    not a self-latching mechanism. Joint bearings and a padlock's immobilization
    are ideal joints/limits; load contact between fork and post is native.
    """
    if spec["operator"]["model"] != "gate_latch_fork":
        raise ValueError("Fork assembly requires gate_latch_fork")
    ps = C.frame_jamb_thickness(spec)
    sx = u * spec["opening"]["width"] / 2
    gap = abs(sx - (hx + x_edge))
    if not .050 <= gap <= .055:
        raise ValueError(f"Fork clamp needs the authored 51 mm operating gap, got {gap:g}")
    stile = next((g for g in leaf_body.geoms if g.name == f"{leaf_name}_stile_l"), None)
    if stile is None:
        stile = next(g for g in leaf_body.geoms if g.name == f"{leaf_name}_slab")
    t = 2 * stile.size[1]
    name = leaf_name
    mat = C.mat_from_material(model, "steel_galvanized", "mat_gate_fork")
    leaf_geom_start = len(leaf_body.geoms)
    pivot = x_edge + u * .0255
    xm = x_edge - u * .0225
    hz = float(spec["operator"]["height"]) - .008
    yarm = -(t / 2 + .020)
    post_from_pivot = gap - .0255
    root = post_from_pivot - .010
    tip = post_from_pivot + ps + .008
    inner_y = ps / 2 + .002
    arm_y = min(yarm, -inner_y - .004)
    # The moving eye, arm and crossbar are flat. An elevated bridge folds back
    # through the leaf even when a pivot is nominally mounted on its stile.
    fork = Body(f"{name}_fork", leaf_body.name, (pivot, 0., hz), QUAT_ID,
                tiers=ALL_TIERS, semantic="latch", label="Lift-to-open fork latch")
    locked = spec.get("lock", {}).get("model") == "padlock" and spec["lock"].get("engaged", False)
    fork.joint = Joint(f"{name}_fork_hinge", "hinge", (0, -u, 0), (0, 0, 0),
                       (0., .001 if locked else 1.55), damping=.002, frictionloss=.0005, armature=.00001,
                       role="operator", label="Lift fork clear of post to open AND close")
    for sign in (-1, 1):
        _box(fork.geoms, f"{name}_fork_eye_x_{sign}", (u*sign*.009, arm_y, 0),
             (.003, .006, .012), mat, label="Open pivot eye side")
        _box(fork.geoms, f"{name}_fork_eye_z_{sign}", (0, arm_y, sign*.009),
             (.006, .006, .003), mat, label="Open pivot eye side")
    _box(fork.geoms, f"{name}_fork_arm", (u*(root+.009)/2, arm_y, 0),
         ((root-.009)/2, .004, .004), mat, label="Flat fork arm")
    _box(fork.geoms, f"{name}_fork_bridge", (u*root, (arm_y+inner_y+.006)/2, 0),
         (.004, (inner_y+.006-arm_y)/2, .004), mat, label="Flat fork crossbar")
    tines = []
    for face in (-1, 1):
        g = f"{name}_fork_tine_{'p' if face > 0 else 'n'}"
        _box(fork.geoms, g, (u*(root+tip)/2, face*(inner_y+.003), 0),
             ((tip-root)/2, .003, .009), mat, label="Post-capturing fork prong")
        tines.append(g)
    # A side grip stays in the fork plane throughout the lift; its 16 mm round
    # section is a real force-applying surface, not a site above empty space.
    grip_x = (root+tip)/2
    grip_y = -inner_y-.017
    fork.geoms.append(C.cyl(f"{name}_fork_handle", (u*grip_x, grip_y, 0),
        .008, .022, mat, (u, 0, 0), 7850, True, True, ALL_TIERS, "operator", "Fork lift grip"))
    for k, dx in enumerate((-.018, .018)):
        _box(fork.geoms, f"{name}_fork_handle_arm_{k}",
             (u*(grip_x+dx), -inner_y-.010, 0), (.003, .007, .004), mat,
             label="Fork grip connection")
    fork.sites.append(Site(f"{name}_fork_grip", (u*grip_x, grip_y, .008),
                            QUAT_ID, .008, "grip"))
    model.add_body(fork)
    attachments = []
    # Front plate meets the actual continuous stile; a carrier runs beneath
    # the rotating eye to two clevis cheeks. No clamp is fastened to wire mesh.
    plate = f"{name}_fork_mount"
    _box(leaf_body.geoms, plate, (xm, -(t/2+.003), hz-.012),
         (.020, .003, .037), mat, label="Fork clamp mounting plate")
    ylo, yhi = arm_y-.013, -(t/2+.003)
    _box(leaf_body.geoms, f"{name}_fork_carrier", ((xm+pivot)/2, (ylo+yhi)/2, hz-.026),
         (abs(pivot-xm)/2+.006, (yhi-ylo)/2, .004), mat, label="Clamped pivot carrier")
    for side in (-1, 1):
        cheek = f"{name}_fork_cheek_{side}"
        _box(leaf_body.geoms, cheek, (pivot-u*.004, arm_y+side*.010, hz-.012),
             (.010, .003, .018), mat, label="Fork pivot clevis cheek")
        attachments.extend([[cheek, f"{name}_fork_carrier"]])
    leaf_body.geoms.append(C.cyl(f"{name}_fork_pin", (pivot, arm_y, hz),
        .004, .016, mat, (0, 1, 0), 7850, True, True, ALL_TIERS, "latch", "Fork pivot pin"))
    for dz in (-.032, .010):
        _screw(leaf_body.geoms, f"{name}_fork_screw_{int(dz*1000)}",
               (xm, -(t/2+.006), hz+dz), -1, mat)
    attachments.extend([[plate, stile.name], [plate, f"{name}_fork_carrier"]])
    attachments.extend([[f"{name}_fork_pin", f"{name}_fork_cheek_{side}"] for side in (-1, 1)])
    attachments.extend([[f"{name}_fork_eye_z_{side}",f"{name}_fork_eye_x_1"] for side in (-1,1)])
    attachments.extend([[f"{name}_fork_eye_x_-1",f"{name}_fork_eye_z_1"],
                        [f"{name}_fork_eye_x_1",f"{name}_fork_arm"],
                        [f"{name}_fork_arm",f"{name}_fork_bridge"]])
    attachments.extend([[f"{name}_fork_bridge",g] for g in tines])
    for k in (0,1):
        attachments.extend([[f"{name}_fork_tine_n",f"{name}_fork_handle_arm_{k}"],
                            [f"{name}_fork_handle_arm_{k}",f"{name}_fork_handle"]])
    # Exact native geometry sweeps test the moving components against all of
    # these fixed parts. The pin sits in the eye's open six-millimetre bore.
    fixed = [plate, f"{name}_fork_carrier", f"{name}_fork_pin"] + [f"{name}_fork_cheek_{s}" for s in (-1, 1)]
    pulls = []
    for face in (-1, 1):
        tag = 'p' if face > 0 else 'n'
        g = f"{name}_gate_pull_{tag}"
        z = hz-.16
        y = face*(t/2+.050)
        leaf_body.geoms.append(C.cyl(g, (xm,y,z), .009,.055,mat,(0,0,1),7850,
                                     True,True,ALL_TIERS,"operator","Fixed gate pull"))
        for k,dz in enumerate((-.055,.055)):
            leaf_body.geoms.append(C.cyl(f"{g}_arm_{k}", (xm,face*(t/2+.025),z+dz),
                .007,.025,mat,(0,1,0),7850,True,True,ALL_TIERS,"operator","Pull mounting arm"))
            attachments.extend([[f"{g}_arm_{k}",stile.name],[f"{g}_arm_{k}",g]])
        site = f"{name}_gate_pull_grip_{tag}"
        leaf_body.sites.append(Site(site,(xm,y+face*.009,z),tuple(quat_z_to((0,face,0))),.008,"grip"))
        pulls.append(site)
    # Every load-bearing fork part is steel, including its carrier and cheeks.
    for geom in leaf_body.geoms[leaf_geom_start:] + fork.geoms:
        geom.density = 7850.
    if locked:
        pm = C.mat_from_material(model,"brass","mat_padlock")
        # Two pierced lugs share the shackle. The fixed lug's neck passes beside
        # the hanging lock body, rather than occupying that body's volume.
        for geoms,prefix,x,y,z in ((leaf_body.geoms,f"{name}_fork_lug",pivot+u*root,arm_y-.024,hz),
                                  (fork.geoms,f"{name}_fork_lock_eye",u*root,arm_y-.012,0.)):
            for sign in (-1,1):
                _box(geoms,f"{prefix}_x_{sign}",(x+u*sign*.008,y,z),(.003,.002,.011),mat,"lock","Pierced padlock lug")
                _box(geoms,f"{prefix}_z_{sign}",(x,y,z+sign*.008),(.005,.002,.003),mat,"lock","Pierced padlock lug")
        _box(leaf_body.geoms,f"{name}_fork_lug_neck",(pivot+u*(root-.018),arm_y-.024,hz-.016),
             (.007,.003,.022),mat,"lock","Locking lug side support")
        _box(leaf_body.geoms,f"{name}_fork_lug_base",(pivot+u*(root-.018),arm_y-.018,hz-.026),
             (.007,.006,.004),mat,"lock","Locking lug carrier connection")
        _box(fork.geoms,f"{name}_fork_lock_eye_arm",(u*root,arm_y-.006,0),
             (.006,.006,.003),mat,"lock","Moving locking eye connection")
        C.add_padlock(fork.geoms,f"{name}_fork_padlock",(u*root,arm_y-.018,0),
                      (0,1,0),(0,0,-1),pm,ALL_TIERS,"lock","Padlocked fork")
        model.meta.setdefault("notes",[]).append("Fork padlock immobilization uses an ideal joint limit; no key or shackle deformation simulation.")
    model.meta.setdefault("gate_hardware",[]).append({
        "schema":"doorbench.gate-hardware.v1", "kind":"gravity_fork",
        "operator_joint":fork.joint.name,"release_site":f"{name}_fork_grip",
        "pull_sites":pulls,"attachments":attachments,"fixed_mount_geoms":fixed,
        "moving_geoms":[g.name for g in fork.geoms if g.semantic != "lock"],
        "post_geom":"post_latch","support_geom":stile.name,"tine_geoms":tines,
        "gate_gap_m":gap,"release_travel_rad":1.55,"grip_height_m":hz+.008,
        "locked":locked,"self_latching":False,"magnetic_callback_required":False})
    return {"operator_joint":fork.joint.name,"grip_height":hz+.008}


def add_magnetic_latch(model, world, leaf_body, spec, *, u, v, hx, x_edge,
                       leaf_bottom, leaf_height, leaf_name="leaf"):
    """Add one post-mounted top-pull latch; return joint and actual grip height.

    Call before the surrounding post is added. Post size and position use the
    same authored dimensions as ``common.add_frame``. The actual latch-side
    stile determines the moving mounting face; sheet-infill thickness is never
    used as a substitute for the structural frame depth.
    """
    if spec["operator"]["model"] != "gate_latch_magnetic":
        raise ValueError("The magnetic assembly requires gate_latch_magnetic")
    ps = C.frame_jamb_thickness(spec)
    sx = u * spec["opening"]["width"] / 2
    gap = abs(sx - (hx + x_edge))
    if not .013 - 1e-6 <= gap <= .038 + 1e-6:
        raise ValueError(f"Magnetic latch needs a 13–38 mm gate gap; got {gap:g} m")
    # Slatted gates have a continuous latch stile. Solid panels use the slab.
    stile = next((g for g in leaf_body.geoms if g.name == f"{leaf_name}_stile_l"), None)
    t = 2 * stile.size[1] if stile is not None and stile.type == "box" else spec["leaf"]["thickness"]
    if t < .015:
        raise ValueError("Magnetic striker requires a structural mounting member, not thin infill")
    grip_z = float(spec["operator"]["height"])
    top = min(grip_z - .440, leaf_bottom + leaf_height - .08)
    if top < leaf_bottom + .20 or grip_z - top < .15:
        raise ValueError("Insufficient gate/column height for the top-pull assembly")
    travel, engagement, radius = .030, .018, .006
    tip_z = top - engagement
    # Hardware is on the opening-side face so the striker arm moves away from
    # the thick fixed post as the gate opens. The pin axis itself is on the post.
    xpin, ypin = sx + u * .018, v * (ps / 2 + .024)
    mat = C.mat_from_material(model, "black_matte_metal", "mat_gate_latch")
    steel = C.mat_from_material(model, "stainless", "mat_gate_latch_steel")
    magmat = C.mat_from_material(model, "steel", "mat_gate_magnet")
    name = leaf_name
    pin = Body(f"{name}_pin", None, (xpin, ypin, tip_z), QUAT_ID,
               tiers=ALL_TIERS, semantic="latch", label="Top-pull rod and latch pin")
    pin.joint = Joint(f"{name}_pin_slide", "slide", (0, 0, 1), (0, 0, 0),
                      (0., travel), damping=1., frictionloss=.10, stiffness=60.,
                      springref=travel + 4. / 60., armature=.15, role="operator",
                      label="Lift release (+ = pin withdrawn); upward return spring")
    pin.geoms.append(Geom(f"{name}_pin_geom", "capsule", (radius, .008),
                          (0, 0, .014), material=steel, density=7850.,
                          semantic="latch", part_label="Magnetic latch pin"))
    # Shaft overlaps both the pin and knob: no disconnected decorative grip.
    shaft_lo, shaft_hi = .018, grip_z - tip_z
    pin.geoms.append(C.cyl(f"{name}_pin_rod", (0, 0, (shaft_lo + shaft_hi) / 2),
                           .0035, (shaft_hi - shaft_lo) / 2, steel, (0, 0, 1),
                           7850, True, True, ALL_TIERS, "latch", "Continuous release rod"))
    pin.geoms.append(C.cyl(f"{name}_pin_knob", (0, 0, grip_z - tip_z), .017, .018,
                           mat, (0, 0, 1), 1200, True, True, ALL_TIERS,
                           "operator", "Top-pull release knob"))
    pin.sites.append(Site(f"{name}_grip_pin", (0, v * .017, grip_z - tip_z),
                           tuple(quat_z_to((0, v, 0))), .008, "grip"))
    for face,tag in ((-1,'n'),(1,'p')):
        pin.sites.append(Site(f'{name}_grip_pin_{tag}',(0,face*.017,grip_z-tip_z),
            tuple(quat_z_to((0,face,0))),.008,'grip'))
    pin.sites.append(Site(f"{name}_pin_pole", (0, 0, 0), QUAT_ID, .002, "sensor"))
    model.add_body(pin)

    # A real hollow guide channel; no collision exclusion conceals a solid
    # housing through which the pin would otherwise pass.
    housing_lo, housing_hi = top + .003, grip_z - .020
    zc, hz = (housing_lo + housing_hi) / 2, (housing_hi - housing_lo) / 2
    for axis in (0, 1):
        for sign in (-1, 1):
            pos = [xpin, ypin, zc]
            pos[axis] += sign * .014
            half = [.018, .018, hz]
            half[axis] = .004
            _box(world.geoms, f"{name}_pin_housing_{axis}_{sign}", tuple(pos), tuple(half), mat,
                 label="Hollow rod guide housing")
    # Two screwed support brackets stay below the actual post top. The upper
    # unbracketed portion is a continuous cantilever, never a floating block.
    post_top = spec["opening"].get("ground_clearance", .05) + spec["leaf"]["height"] + .05
    lower_z, upper_z = top + .045, min(grip_z - .13, post_top - .04)
    if upper_z - lower_z < .08:
        raise ValueError("Post is too short for two separated latch mounting brackets")
    attachment_pairs = []
    for k, z in enumerate((lower_z, upper_z)):
        plate_name = f"{name}_pin_post_plate_{k}"
        _box(world.geoms, plate_name, (sx + u * .031, v * (ps / 2 + .003), z),
             (.031, .003, .022), mat, label="Post mounting bracket")
        _box(world.geoms, f"{name}_pin_post_spacer_{k}", (xpin, v * (ps / 2 + .008), z),
             (.018, .002, .018), mat, label="Bracket to guide connection")
        for dz in (-.012, .012):
            _screw(world.geoms, f"{name}_pin_post_screw_{k}_{int(dz*1000)}",
                   (sx + u * .048, v * (ps / 2 + .006), z + dz), v, steel)
        attachment_pairs.extend([[plate_name, "post_latch"],
                                 [plate_name, f"{name}_pin_post_spacer_{k}"],
                                 [f"{name}_pin_post_spacer_{k}", f"{name}_pin_housing_1_{-int(v)}"]])

    # Four-sided striker pocket, closed underneath by the magnet. It has no
    # approach ramp: the up-returning pin clears it until magnetic alignment.
    xp = xpin - hx
    bottom = tip_z - .010
    cup_z = (top + bottom) / 2
    for axis in (0, 1):
        for sign in (-1, 1):
            pos = [xp, ypin, cup_z]
            pos[axis] += sign * .011
            half = [.014, .014, (top - bottom) / 2]
            half[axis] = .003
            _box(leaf_body.geoms, f"{name}_cup_wall_{axis}_{sign}", tuple(pos), tuple(half), mat,
                 label="Striker pocket wall")
    _box(leaf_body.geoms, f"{name}_cup_magnet", (xp, ypin, bottom + .002),
         (.008, .008, .002), magmat, label="Striker permanent magnet")
    _box(leaf_body.geoms, f"{name}_cup_floor", (xp, ypin, bottom - .002),
         (.014, .014, .002), mat, label="Striker base")
    leaf_body.sites.append(Site(f"{name}_striker_pole", (xp, ypin, tip_z - .008),
                                QUAT_ID, .002, "sensor"))
    xm = x_edge - u * .0225
    mount_name = f"{name}_cup_mount"
    _box(leaf_body.geoms, mount_name, (xm, v * (t / 2 + .003), cup_z),
         (.020, .003, .026), mat, label="Striker bracket on gate stile")
    y0, y1 = sorted((v * (t / 2 + .006), ypin))
    _box(leaf_body.geoms, f"{name}_cup_standoff", (xm, (y0 + y1) / 2, bottom - .002),
         (.020, (y1 - y0) / 2, .005), mat, label="Striker depth bracket")
    _box(leaf_body.geoms, f"{name}_cup_arm", ((xm + xp) / 2, ypin, bottom - .002),
         (abs(xp-xm) / 2, .014, .005), mat, label="Striker arm across gate gap")
    for dz in (-.016, .016):
        _screw(leaf_body.geoms, f"{name}_cup_screw_{int(dz*1000)}",
               (xm, v * (t / 2 + .006), cup_z + dz), v, steel)
    support = stile.name if stile is not None else next((g.name for g in leaf_body.geoms
               if g.semantic == "leaf" and g.type == "box"), "")
    attachment_pairs.extend([[mount_name, support], [mount_name, f"{name}_cup_standoff"],
                             [f"{name}_cup_standoff", f"{name}_cup_arm"],
                             [f"{name}_cup_arm", f"{name}_cup_floor"],
                             [f"{name}_pin_geom", f"{name}_pin_rod"],
                             [f"{name}_pin_rod", f"{name}_pin_knob"]])
    pull_sites = []
    pull_z = max(leaf_bottom + .15, top - .16)
    for face in (-1, 1):
        tag = "p" if face > 0 else "n"
        ygrip = face * (t / 2 + .050)
        pull_name = f"{name}_gate_pull_{tag}"
        leaf_body.geoms.append(C.cyl(pull_name, (xm, ygrip, pull_z), .009, .065,
                                    steel, (0, 0, 1), 7850, True, True, ALL_TIERS,
                                    "operator", "Fixed gate pull"))
        for k, dz in enumerate((-.065, .065)):
            _box(leaf_body.geoms, f"{pull_name}_plate_{k}",
                 (xm, face * (t / 2 + .002), pull_z + dz), (.018, .002, .018),
                 steel, "operator", "Pull mounting plate")
            leaf_body.geoms.append(C.cyl(f"{pull_name}_arm_{k}",
                (xm, face * (t / 2 + .026), pull_z + dz), .007, .024, steel,
                (0, face, 0), 7850, True, True, ALL_TIERS, "operator", "Pull mounting arm"))
            _screw(leaf_body.geoms, f"{pull_name}_screw_{k}",
                   (xm, face * (t / 2 + .004), pull_z + dz), face, steel)
            attachment_pairs.extend([[f"{pull_name}_plate_{k}", support],
                                     [f"{pull_name}_plate_{k}", f"{pull_name}_arm_{k}"],
                                     [f"{pull_name}_arm_{k}", pull_name]])
        site = f"{name}_gate_pull_grip_{tag}"
        leaf_body.sites.append(Site(site, (xm, ygrip + face*.009, pull_z),
                               tuple(quat_z_to((0, face, 0))), .008, "grip"))
        pull_sites.append(site)
    rule = {"schema": "doorbench.magnetic-latch.v1", "joint": pin.joint.name,
            "pin_site": f"{name}_pin_pole", "striker_site": f"{name}_striker_pole",
            "capture_axes_m": [.018, .018, .050], "peak_axial_force_N": 15.,
            "force_model": "compact conservative ellipsoidal potential; approximate, not measured"}
    model.meta.setdefault("magnetic_latches", []).append(rule)
    model.meta.setdefault("gate_hardware", []).append({
        "schema": "doorbench.gate-hardware.v1", "kind": "magnetic_top_pull",
        "operator_joint": pin.joint.name, "release_site": f"{name}_grip_pin",
        "release_face_sites": {tag:f'{name}_grip_pin_{tag}' for tag in ('n','p')},
        "pull_sites": pull_sites, "pin_geom": f"{name}_pin_geom",
        "pin_rod": f"{name}_pin_rod", "knob_geom": f"{name}_pin_knob",
        "keeper_geoms": [f"{name}_cup_wall_{a}_{s}" for a in (0, 1) for s in (-1, 1)],
        "attachments": attachment_pairs, "gate_gap_m": gap,
        "release_travel_m": travel, "engagement_m": engagement,
        "guide_bottom_gap_m": .003, "grip_height_m": grip_z,
        "magnetic_callback_required": True})
    model.meta.setdefault("notes", []).append(
        "Top-pull magnetic latch requires the magnetic_latches passive-force rule; "
        "plain MJCF/URDF/USD loads retain geometry and up-return spring but do not simulate the magnet.")
    return {"operator_joint": pin.joint.name, "grip_height": grip_z}


def magnetic_potential_force(displacement, axes, peak_axial_force):
    """Potential [J], force on pin [N] for pin-minus-striker displacement.

    U=-E(1-s)^3 inside s=sum((d/a)^2)<1, zero outside. Its gradient
    continuously vanishes at the support boundary. E calibrates the maximum
    axial attraction to the supplied force; lateral forces follow that same
    potential, rather than creating/removing energy with an alignment switch.
    """
    d, a = np.asarray(displacement, float), np.asarray(axes, float)
    if d.shape != (3,) or a.shape != (3,) or not np.isfinite(d).all() or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("Invalid magnetic displacement/axes")
    if not math.isfinite(peak_axial_force) or peak_axial_force <= 0:
        raise ValueError("Invalid magnetic force")
    s = float(np.dot(d/a, d/a))
    if s >= 1:
        return 0., np.zeros(3)
    energy = peak_axial_force * a[2] * 25 * math.sqrt(5) / 96
    return -energy*(1-s)**3, -6*energy*(1-s)**2*d/a**2


def add_baby_gate_latch(model, world, leaf_body, spec, *, u, v, hx, x_edge,
                        leaf_bottom, leaf_height, leaf_name="leaf"):
    """A nonmagnetic 20 mm lift-and-swing pin with a connected rod and keeper.

    This is an idealized one-action lift latch, not a replica or certification of
    a branded two-action child-resistant lock. Two approach ramps permit the
    already-authored double-acting baby gates to re-latch from either side.
    """
    from ..ir import quat_from_axis_angle
    if spec["operator"]["model"] != "baby_gate_latch":
        raise ValueError("The baby-gate assembly requires baby_gate_latch")
    name, face = leaf_name, -v
    ps = C.frame_jamb_thickness(spec)
    sx = u * spec["opening"]["width"] / 2
    stile = next((g for g in leaf_body.geoms if g.name == f"{name}_stile_l"), None)
    t = 2*stile.size[1] if stile is not None and stile.type == "box" else spec["leaf"]["thickness"]
    support = stile.name if stile is not None else next((g.name for g in leaf_body.geoms
               if g.semantic == "leaf" and g.type == "box"), "")
    grip_z, travel, engagement = float(spec["operator"]["height"]), .020, .012
    top, tip_z = grip_z-.055, grip_z-.055-engagement
    xp, ypin = x_edge-u*.040, face*(t/2+.040)
    xpin = hx+xp
    mat = C.mat_from_material(model, "pvc", "mat_baby_latch")
    steel = C.mat_from_material(model, "stainless", "mat_gate_latch_steel")
    pin = Body(f"{name}_pin", None, (xpin, ypin, tip_z), QUAT_ID,
               tiers=ALL_TIERS, semantic="latch", label="Baby gate lift pin and rod")
    pin.joint = Joint(f"{name}_pin_slide", "slide", (0,0,1), (0,0,0), (0.,travel),
        damping=1., frictionloss=.15, stiffness=300., springref=-8/300,
        armature=.15, role="operator", label="Lift latch (+ = withdrawn, 20 mm travel)")
    pin.geoms.append(Geom(f"{name}_pin_geom", "capsule", (.005,.010), (0,0,.015),
                          material=steel, density=7850., friction=(.15,.005,.0001),
                          semantic="latch", part_label="Lift pin"))
    high = grip_z-tip_z
    pin.geoms.append(C.cyl(f"{name}_pin_rod", (0,0,(.020+high)/2), .0035, (high-.020)/2,
        steel, (0,0,1), 7850, True, True, ALL_TIERS, "latch", "Connected release rod"))
    pin.geoms.append(C.cyl(f"{name}_pin_knob", (0,0,high), .012,.012,mat,(0,0,1),1200,
        True,True,ALL_TIERS,"operator","Lift knob"))
    pin.sites.append(Site(f"{name}_grip_pin",(0,face*.012,high),tuple(quat_z_to((0,face,0))),.008,"grip"))
    for sign,tag in ((-1,'n'),(1,'p')):
        pin.sites.append(Site(f'{name}_grip_pin_{tag}',(0,sign*.012,high),
            tuple(quat_z_to((0,sign,0))),.008,'grip'))
    model.add_body(pin)
    lo, hi = top+.003, grip_z-.015
    for axis in (0,1):
        for sign in (-1,1):
            pos=[xpin,ypin,(lo+hi)/2];pos[axis]+=sign*.012
            half=[.016,.016,(hi-lo)/2];half[axis]=.004
            _box(world.geoms,f"{name}_pin_housing_{axis}_{sign}",tuple(pos),tuple(half),mat,
                 label="Hollow lift-pin guide")
    # The fixed guide reaches a screwed bracket on the actual post face.
    px = sx+u*ps/2
    pz = top+.026
    _box(world.geoms,f"{name}_pin_post_plate",(px,face*(ps/2+.003),pz),
         (ps/2,.003,.020),mat,label="Post-mounted latch plate")
    xa,xb=sorted((xpin+u*.016,px))
    _box(world.geoms,f"{name}_pin_bridge",((xa+xb)/2,ypin,pz),
         ((xb-xa)/2,.016,.015),mat,label="Connected housing bracket")
    ya,yb=sorted((face*(ps/2+.006),ypin))
    _box(world.geoms,f"{name}_pin_post_spacer",(px,(ya+yb)/2,pz),
         (ps/2,(yb-ya)/2,.015),mat,label="Post bracket depth spacer")
    for k,dz in enumerate((-.012,.012)):
        _screw(world.geoms,f"{name}_pin_post_screw_{k}",(px,face*(ps/2+.006),pz+dz),face,steel)
    bottom=tip_z-.004
    cup_z=(top+bottom)/2
    for axis in (0,1):
        for sign in (-1,1):
            pos=[xp,ypin,cup_z];pos[axis]+=sign*.010
            half=[.013,.013,(top-bottom)/2];half[axis]=.003
            _box(leaf_body.geoms,f"{name}_cup_wall_{axis}_{sign}",tuple(pos),tuple(half),mat,
                 label="Lift latch keeper wall")
    _box(leaf_body.geoms,f"{name}_cup_floor",(xp,ypin,bottom-.002),(.013,.013,.002),mat,
         label="Keeper floor")
    xm=x_edge-u*.0225
    _box(leaf_body.geoms,f"{name}_cup_mount",(xm,face*(t/2+.003),cup_z),(.020,.003,.019),mat,
         label="Keeper mount on the gate stile")
    ya,yb=sorted((face*(t/2+.006),ypin))
    _box(leaf_body.geoms,f"{name}_cup_standoff",(xm,(ya+yb)/2,bottom-.004),
         (.020,(yb-ya)/2,.004),mat,label="Keeper support bracket")
    _box(leaf_body.geoms,f"{name}_cup_arm",((xm+xp)/2,ypin,bottom-.004),
         (abs(xm-xp)/2+.003,.013,.004),mat,label="Keeper bracket to pocket")
    for k,dz in enumerate((-.011,.011)):
        _screw(leaf_body.geoms,f"{name}_cup_screw_{k}",(xm,face*(t/2+.006),cup_z+dz),face,steel)
    # Surface of each ramp is flush with the keeper wall's top. The pin's round
    # tip rides upward through at most 12 mm before dropping into the pocket.
    for side in (-1,1):
        phi=math.atan2(-side*.018,.045)
        normal=np.array([0,side*.018,.045])/math.hypot(.018,.045)
        middle=np.array([xp,ypin+side*.0355,top-.009])-.002*normal
        leaf_body.geoms.append(C.box(f"{name}_cup_ramp_{side}",tuple(middle),
            (.013,math.hypot(.045,.018)/2,.002),mat,1400,True,True,ALL_TIERS,"latch",
            "Self-latching approach ramp",quat=quat_from_axis_angle((1,0,0),phi),
            friction=(.12,.005,.0001)))
    edges=[[f"{name}_pin_geom",f"{name}_pin_rod"],[f"{name}_pin_rod",f"{name}_pin_knob"],
        [f"{name}_pin_post_plate","post_latch"],[f"{name}_pin_post_plate",f"{name}_pin_post_spacer"],
        [f"{name}_pin_post_spacer",f"{name}_pin_bridge"],
        [f"{name}_pin_bridge",f"{name}_pin_housing_0_{int(u)}"],
        [f"{name}_cup_mount",support],[f"{name}_cup_mount",f"{name}_cup_standoff"],
        [f"{name}_cup_standoff",f"{name}_cup_arm"],[f"{name}_cup_arm",f"{name}_cup_floor"]]
    for side in (-1,1):
        edges.append([f"{name}_cup_ramp_{side}",f"{name}_cup_wall_1_{side}"])
    pull_sites=[]
    z=top-.10
    for side in (-1,1):
        tag="p" if side>0 else "n"
        nm=f"{name}_gate_pull_{tag}"
        y=side*(t/2+.042)
        leaf_body.geoms.append(C.cyl(nm,(xm,y,z),.008,.045,steel,(0,0,1),7850,
            True,True,ALL_TIERS,"operator","Fixed baby-gate pull"))
        for k,dz in enumerate((-.045,.045)):
            _box(leaf_body.geoms,f"{nm}_plate_{k}",(xm,side*(t/2+.002),z+dz),
                 (.018,.002,.015),steel,"operator","Pull mounting plate")
            leaf_body.geoms.append(C.cyl(f"{nm}_arm_{k}",(xm,side*(t/2+.022),z+dz),.006,.020,
                steel,(0,side,0),7850,True,True,ALL_TIERS,"operator","Pull mounting arm"))
            _screw(leaf_body.geoms,f"{nm}_screw_{k}",(xm,side*(t/2+.004),z+dz),side,steel)
            edges.extend([[f"{nm}_plate_{k}",support],[f"{nm}_plate_{k}",f"{nm}_arm_{k}"],
                          [f"{nm}_arm_{k}",nm]])
        site=f"{name}_gate_pull_grip_{tag}"
        leaf_body.sites.append(Site(site,(xm,y+side*.008,z),tuple(quat_z_to((0,side,0))),.008,"grip"))
        pull_sites.append(site)
    model.meta.setdefault("gate_hardware",[]).append({
        "schema":"doorbench.gate-hardware.v1","kind":"spring_lift_pin",
        "operator_joint":pin.joint.name,"release_site":f"{name}_grip_pin",
        "release_face_sites":{tag:f'{name}_grip_pin_{tag}' for tag in ('n','p')},
        "pull_sites":pull_sites,"pin_geom":f"{name}_pin_geom","pin_rod":f"{name}_pin_rod",
        "knob_geom":f"{name}_pin_knob",
        "keeper_geoms":[f"{name}_cup_wall_{a}_{s}" for a in (0,1) for s in (-1,1)],
        "attachments":edges,"gate_gap_m":abs(sx-(hx+x_edge)),"release_travel_m":travel,
        "engagement_m":engagement,"guide_bottom_gap_m":.003,"grip_height_m":grip_z,
        "magnetic_callback_required":False,
        "scope":"One-action lift pin; child resistance and branded-product parity not certified"})
    return {"operator_joint":pin.joint.name,"grip_height":grip_z}


@dataclass(frozen=True)
class MagneticLatchRule:
    pin_site: int
    striker_site: int
    pin_body: int
    striker_body: int
    axes: tuple
    peak_force: float


def compile_magnetic_latches(model, metadata):
    """Resolve explicit sites; absent mechanism in a reduced tier is skipped."""
    import mujoco
    rules = []
    for raw in metadata.get("magnetic_latches", []):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, raw["joint"])
        if jid < 0:
            continue
        sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, raw[k])
                 for k in ("pin_site", "striker_site")]
        if min(sites) < 0:
            raise ValueError("Magnetic latch is missing an explicitly bound pole site")
        axes, peak = tuple(raw["capture_axes_m"]), float(raw["peak_axial_force_N"])
        magnetic_potential_force([0, 0, 0], axes, peak)
        if int(model.jnt_bodyid[jid]) != int(model.site_bodyid[sites[0]]):
            raise ValueError("Magnetic pole does not belong to the release joint body")
        rules.append(MagneticLatchRule(sites[0], sites[1], int(model.site_bodyid[sites[0]]),
                                      int(model.site_bodyid[sites[1]]), axes, peak))
    return tuple(rules)


def apply_magnetic_latches(model, data, rules):
    """Add equal/opposite pole forces to qfrc_passive, using current native FK."""
    import mujoco
    for rule in rules:
        p, s = data.site_xpos[rule.pin_site], data.site_xpos[rule.striker_site]
        # The anisotropic capture envelope follows the fixed guide frame.
        rotation = data.site_xmat[rule.pin_site].reshape(3, 3)
        _, local_force = magnetic_potential_force(rotation.T @ (p-s), rule.axes, rule.peak_force)
        force = rotation @ local_force
        mujoco.mj_applyFT(model, data, force, np.zeros(3), p, rule.pin_body, data.qfrc_passive)
        mujoco.mj_applyFT(model, data, -force, np.zeros(3), s, rule.striker_body, data.qfrc_passive)

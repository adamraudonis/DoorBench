"""Closer / operator mechanisms as real articulated linkages (geometry + mechanism calibration).

Every self-closing or power-operating device in ``hardware.CLOSERS`` is built here as bodies, joints and a loop
closure so that the closing torque is TRANSMITTED through the mechanism instead of being painted onto the door joint:

* surface closers (regular arm on the pull face / parallel arm on the push face), concealed overhead closers and
  automatic swing operators: pinion hinge (spring + hydraulics) -> main arm -> elbow hinge -> forearm -> ``connect``
  to the shoe (two-bar loop).  Arm lengths / shoe positions come from the installation templates in
  ``closers.ARM_TEMPLATES``; the shoe position and the rest angle are chosen by a deterministic kinematic search so
  that the pinion ratio dphi/dtheta stays monotonic and inside [0.45, 2.2] over the whole door range and the arms
  never cross the wall face, the head or the leaf.
* telescoping devices (pneumatic screen closer, gate spring, hydraulic gate closer, gas strut): cylinder body hinged
  at a bracket, rod body on a slide joint (spring + damper along the axis), rod tip ``connect``-ed to the leaf bracket.
* floor springs and spring hinges act at the pivot / hinge line (the spindle IS the door pivot): the door joint
  carries the spring and the valve law directly, with the concealed body geometry drawn.

For every mechanism the builder (1) tabulates the loop kinematics, (2) derives the pinion / rod spring and damping
so that the door-level EN 1154 design curve is realised, (3) calibrates the reduced (simple / minimal / URDF / RL)
door-joint model to the full mechanism and stores the fit errors, (4) writes the valve laws in joint coordinates
into ``phys["closer"]`` and (5) registers the loop in ``model.linkages`` (consumed by the web viewer, the clearance
gate and the QA).  Call-site: ``common.add_closer`` (hinged families) and ``add_gas_strut`` (hatches).
"""
from __future__ import annotations

import math

import numpy as np

from ..ir import Body, Geom, Joint, Site, Equality, Model, ALL_TIERS, FULL_ONLY, FULL_SIMPLE, QUAT_ID, quat_from_axis_angle, quat_z_to, quat_mul
from .. import hardware as H
from .. import closers as CK
from . import meshes as MESH
from .common import box, cyl, mesh_geom, mat_from_material, mat_rgba, q_face, body_world_pos, GAP

REDUCED = frozenset({"simple", "minimal"})
ARM_HALF_W, ARM_HALF_T = 0.009, 0.005      # arm cross-section (half extents): 18 x 10 mm forged steel
SHAFT_R = 0.008


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _leaf_extent(leaf_body: Body):
    """(z_bottom, z_top, x_far) of the leaf slab geoms in the leaf frame."""
    zs0, zs1, xs = [], [], []
    for g in leaf_body.geoms:
        if g.semantic in ("leaf", "glass") and g.type == "box":
            zs0.append(g.pos[2] - g.size[2])
            zs1.append(g.pos[2] + g.size[2])
            xs.append(g.pos[0] - g.size[0])
            xs.append(g.pos[0] + g.size[0])
    if not zs0:
        return 0.0, 2.0, 0.9
    return min(zs0), max(zs1), (max(xs) if max(xs) > -min(xs) else min(xs))


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _leaf_frame_pt(pt_w, hinge, rot_sign, theta):
    """World (2D) point -> leaf frame when the leaf is rotated by rot_sign*theta about `hinge`."""
    return hinge + _rot_z(-rot_sign * theta) @ (np.asarray(pt_w) - hinge)


def _interp(table_x, table_y, x):
    return float(np.interp(x, table_x, table_y))


# ---------------------------------------------------------------------------
# two-bar placement search
# ---------------------------------------------------------------------------
def _search_two_bar(pinion_on_leaf: bool, pinion, hinge, rot_sign, L1, fore_range, shoe_candidates, angs, theta_max, clear_fn, ratio_band=(0.45, 2.2)):
    """Pick (shoe, rest angle, L2, elbow_sign) giving the most uniform monotonic pinion ratio over [0, theta_max].
    clear_fn(theta, pts_world) -> bool must be True for every sampled configuration (arms clear of everything).
    Returns the best candidate dict or None."""
    thetas = np.linspace(0.0, theta_max, max(12, int(math.degrees(theta_max) / 5) + 1))
    best = None
    for shoe in shoe_candidates:
        shoe = np.asarray(shoe, float)
        for ang in angs:
            elbow0 = pinion + L1 * np.array([math.cos(ang), math.sin(ang)])
            L2 = float(np.linalg.norm(elbow0 - shoe))
            if not (fore_range[0] - 0.02 <= L2 <= fore_range[1] + 0.02):
                continue
            sign = None
            for sd in (1, -1):
                e, _ = CK.twobar_elbow(pinion, shoe, L1, L2, sd)
                if np.linalg.norm(e - elbow0) < 1e-6:
                    sign = sd
                    break
            if sign is None:
                continue
            q, dq, ds, elbows = CK.twobar_sweep(pinion, shoe, hinge, L1, L2, sign, thetas, rot_sign, pinion_on_leaf)
            if (L1 + L2) - ds.max() < 0.015 or ds.min() - abs(L1 - L2) < 0.012:
                continue
            s_ = 1.0 if q[-1] > 0 else -1.0
            r = dq * s_
            if r.min() < ratio_band[0] or r.max() > ratio_band[1]:
                continue
            ok = True
            for k, th in enumerate(thetas):
                c, s2 = math.cos(rot_sign * th), math.sin(rot_sign * th)
                R = np.array([[c, -s2], [s2, c]])
                if pinion_on_leaf:
                    p_w = hinge + R @ (pinion - hinge)
                    s_w = shoe
                else:
                    p_w = pinion
                    s_w = hinge + R @ (shoe - hinge)
                e_w = elbows[k]
                pts = [p_w + (e_w - p_w) * f for f in (0.25, 0.5, 0.75, 1.0)] + [e_w + (s_w - e_w) * f for f in (0.25, 0.5, 0.75, 0.92)]
                if not clear_fn(th, pts):
                    ok = False
                    break
            if not ok:
                continue
            pen = 0.0
            if L2 < fore_range[0] or L2 > fore_range[1]:
                pen += 0.3
            score = r.max() / r.min() + pen + 0.15 * abs(math.degrees(q[min(len(q) - 1, int(round(math.radians(90) / (thetas[1] - thetas[0]))))]) * s_ - 100.0) / 100.0
            if best is None or score < best["score"]:
                best = {"score": score, "shoe": shoe, "ang": ang, "L2": L2, "sign": sign, "q_sign": s_, "ratio": (float(r.min()), float(r.max()))}
    return best


# ---------------------------------------------------------------------------
# calibration: pinion spring / damping from the door-level design, reduced model fit, valve laws
# ---------------------------------------------------------------------------
def _calibrate_arm(pc: dict, spec: dict, phys: dict, thetas, q, dq, max_open):
    """Given the loop kinematics q(theta) (pinion angle, q>0 = opening) derive the pinion spring / damping realising
    the door-level curve, fit the reduced model and return (pinion_params, reduced_params, tables)."""
    tau0, k = float(pc["spring_preload_Nm"]), float(pc["spring_stiffness_Nm_per_rad"])
    m = float(phys["mass"]["total_kg"])
    W = float(spec["leaf"]["width"])
    fric = float(phys.get("hinge", {}).get("coulomb_torque_Nm", 0.0)) + 0.5 * float(phys.get("hinge", {}).get("stick_torque_Nm", 0.0))
    b_air = float(phys.get("hinge", {}).get("air_damping_Nms_per_rad", 0.0))
    st = pc["settings"]
    lim = min(math.radians(90.0), max_open)
    sel = thetas <= lim + 1e-9
    target = tau0 + k * thetas
    # pinion preload from the closing moment at 0 deg; pinion stiffness by least squares to the EN 1154 line
    tp0 = tau0 / max(dq[0], 0.2)
    A = (dq * q)[sel]
    rhs = (target - tp0 * dq)[sel]
    kp = float(np.dot(A, rhs) / max(np.dot(A, A), 1e-9)) if np.dot(A, A) > 1e-9 else 0.0
    kp = max(kp, 2.0)
    tau_full = (tp0 + kp * q) * dq            # door torque realised by the mechanism
    # pinion damping: check valve natively; sweep valve sized so the full mechanism meets the sweep-time target
    ratio2 = dq ** 2
    mean_r2 = float(np.mean(ratio2[sel]))
    bp_check = float(pc["damping_opening"]) / max(mean_r2, 1e-6)
    tau_fn = lambda th: _interp(thetas, tau_full, th)
    dq_fn = lambda th: _interp(thetas, dq, th)
    if st["sweep_time_s"] > 0:
        lo, hi = 0.0, 6000.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            _, t12, _ = CK.closing_time(m, W, 0, 0, 0, fric, None, 0.0, b_air, theta0=min(math.pi / 2, max_open), tau_fn=tau_fn, b_fn=lambda th, mid=mid: mid * dq_fn(th) ** 2)
            if t12 < st["sweep_time_s"]:
                lo = mid
            else:
                hi = mid
        bp_sweep = 0.5 * (lo + hi)
    else:
        bp_sweep = 0.0
    bp_latch = bp_sweep * st["latch_speed_factor"]
    bp_backcheck = bp_sweep * st["backcheck_factor"]
    q_of = lambda deg: _interp(thetas, q, math.radians(deg))
    q_latch = q_of(st["latch_angle_deg"])
    q_bc = q_of(st["backcheck_angle_deg"]) if st["backcheck_angle_deg"] > 0 else None
    q_delay = q_of(st["delayed_angle_deg"]) if st["delayed_action_s"] > 0 else None
    bp_delay = float(pc.get("delayed_action_damping", 0.0)) / max(dq_fn(math.radians(st["delayed_angle_deg"])) ** 2, 1e-6) if q_delay is not None else 0.0
    q_hold = q_of(st["hold_open_deg"]) if st["hold_open_deg"] > 0 and max_open >= math.radians(st["hold_open_deg"]) - 1e-6 else None
    hold_tau_door = (2.5 if st.get("hold_open_kind") == "electromagnetic" else 1.6) * tau_fn(math.radians(st["hold_open_deg"])) if q_hold is not None else 0.0
    hold_tau_p = hold_tau_door / max(dq_fn(math.radians(st["hold_open_deg"])), 0.2) if q_hold is not None else 0.0
    # closing dynamics of the full mechanism (door-level ODE with the exact torque / damping curves)
    b_full = lambda th: (bp_latch if th < math.radians(st["latch_angle_deg"]) else bp_sweep) * dq_fn(th) ** 2
    t_full, t12_full, w_full = CK.closing_time(m, W, 0, 0, 0, fric, None, 0.0, b_air, theta0=min(math.pi / 2, max_open), tau_fn=tau_fn, b_fn=b_full)
    # reduced model: linear spring fitted to the realised door torque, damping fitted to the closing time
    tr0, kr, e_max, e_rms = CK.fit_linear(thetas[sel], tau_full[sel])
    tr0 = max(tr0, 0.5)
    kr = max(kr, 0.5)
    if bp_sweep > 0:
        lo, hi = 0.0, 6000.0
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            t_r, _, _ = CK.closing_time(m, W, tr0, kr, mid, fric, mid * st["latch_speed_factor"], math.radians(st["latch_angle_deg"]), b_air, theta0=min(math.pi / 2, max_open))
            if t_r > t_full:
                hi = mid
            else:
                lo = mid
        br_sweep = 0.5 * (lo + hi)
    else:
        br_sweep = 0.0
    t_red, t12_red, w_red = CK.closing_time(m, W, tr0, kr, br_sweep, fric, br_sweep * st["latch_speed_factor"], math.radians(st["latch_angle_deg"]), b_air, theta0=min(math.pi / 2, max_open))
    b_curve = bp_sweep * ratio2
    sel12 = (thetas >= math.radians(12)) & sel
    damp_dev = float(np.max(np.abs(b_curve[sel12] / max(br_sweep, 1e-9) - 1.0))) if bp_sweep > 0 and sel12.any() else 0.0
    br_check = bp_check * mean_r2
    br_bc = bp_backcheck * (dq_fn(math.radians(st["backcheck_angle_deg"])) ** 2) if q_bc is not None else 0.0
    br_delay = float(pc.get("delayed_action_damping", 0.0))
    pinion = {"spring_preload_Nm": float(tp0), "spring_stiffness_Nm_per_rad": float(kp), "damping_check_Nms_per_rad": float(bp_check), "damping_sweep_Nms_per_rad": float(bp_sweep),
              "damping_latch_Nms_per_rad": float(bp_latch), "damping_backcheck_Nms_per_rad": float(bp_backcheck), "damping_delay_Nms_per_rad": float(bp_delay),
              "q_latch_rad": float(q_latch), "q_backcheck_rad": None if q_bc is None else float(q_bc), "q_delay_rad": None if q_delay is None else float(q_delay),
              "q_hold_rad": None if q_hold is None else float(q_hold), "hold_torque_Nm": float(hold_tau_p), "q_at_max_open_rad": float(q[-1])}
    reduced = {"spring_preload_Nm": float(tr0), "spring_stiffness_Nm_per_rad": float(kr), "damping_closing": float(br_sweep), "damping_latch": float(br_sweep * st["latch_speed_factor"]),
               "damping_opening": float(br_check), "backcheck_damping": float(br_bc), "backcheck_angle_rad": None if q_bc is None else math.radians(st["backcheck_angle_deg"]),
               "latch_angle_rad": math.radians(st["latch_angle_deg"]), "delayed_action_damping": float(br_delay), "hold_open_rad": None if q_hold is None else math.radians(st["hold_open_deg"]),
               "hold_torque_Nm": float(hold_tau_door),
               "fit": {"torque_max_rel_err": float(e_max), "torque_rms_rel_err": float(e_rms), "damping_curve_max_rel_dev": damp_dev,
                       "closing_time_full_s": float(t_full), "closing_time_reduced_s": float(t_red), "sweep_time_full_s": float(t12_full), "sweep_time_reduced_s": float(t12_red),
                       "final_speed_full_rad_s": float(w_full), "final_speed_reduced_rad_s": float(w_red),
                       "note": "reduced = linear spring least-squares fitted to the mechanism's door torque over 0-90 deg; sweep damping fitted so the reduced closing time equals the full mechanism's"}}
    table = [[round(math.degrees(th), 1), round(float(qq), 4), round(float(dd), 4), round(float(tt), 2), round(float(bb), 2)] for th, qq, dd, tt, bb in zip(thetas, q, dq, tau_full, b_curve)]
    return pinion, reduced, {"columns": ["door_deg", "pinion_rad", "dpinion_ddoor", "door_torque_Nm", "door_sweep_damping_Nms_per_rad"], "rows": table}


def _apply_door_joint_reduced(leaf_body: Body, reduced: dict, phys: dict, both_ways=False):
    """Door joint: base attributes = calibrated reduced model (simple / minimal / URDF / RL); full tier carries no
    closer spring or damping (the mechanism does)."""
    j = leaf_body.joint
    if j is None:
        return
    b_air = float(phys.get("hinge", {}).get("air_damping_Nms_per_rad", 0.0))
    k = reduced["spring_stiffness_Nm_per_rad"]
    j.stiffness = k
    j.springref = (-reduced["spring_preload_Nm"] / k) if k > 1e-9 else 0.0
    if both_ways:
        j.springref = 0.0
    j.damping = b_air + reduced["damping_opening"]
    j.damping_closing = reduced["damping_closing"]
    j.damping_opening = reduced["damping_opening"]
    j.backcheck_angle = reduced.get("backcheck_angle_rad")
    j.backcheck_damping = reduced.get("backcheck_damping") or 0.0
    j.overrides["full"] = {"stiffness": 0.0, "springref": 0.0, "damping": b_air + 0.02, "damping_closing": None, "damping_opening": None, "backcheck_angle": None, "backcheck_damping": None,
                           "notes": (j.notes + " " if j.notes else "") + "full tier: closer torque transmitted through the mechanism (see physics.closer.mechanism_params)"}


def _door_law(joint_name: str, tiers, reduced: dict, b_air: float) -> dict:
    return CK.law_from_windows(joint_name, tiers, b_air + reduced["damping_opening"], reduced["damping_closing"], reduced["damping_latch"], reduced["damping_opening"],
                               reduced["backcheck_damping"], reduced["latch_angle_rad"], reduced.get("backcheck_angle_rad"),
                               (reduced.get("delayed_angle_rad") if reduced.get("delayed_action_damping") else None), reduced.get("delayed_action_damping", 0.0),
                               reduced.get("hold_open_rad"), reduced.get("hold_torque_Nm", 0.0), math.radians(3.0), reduced.get("hold_open_kind", "mechanical"), "door_rad")


# ---------------------------------------------------------------------------
# arm closers (surface regular / parallel, concealed overhead, swing operators)
# ---------------------------------------------------------------------------
def _add_arm_closer(model: Model, world: Body, leaf_body: Body, spec: dict, phys: dict, cl: H.CloserModel, u: float, v: float, x_hinge_axis: float, Hh: float, t: float, Wo: float, jamb_t: float, pfx: str):
    pc = phys["closer"]
    mech = pc["mechanism"]
    op = spec["opening"]
    Ho = float(op["height"])
    wt = float(op["wall_thickness"])
    depth = wt if op["frame"]["kind"] != "aluminum_storefront" else max(0.114, wt)
    yw = float(model.meta.get("wall_y", 0.0))
    casing = 0.016 if op["frame"].get("casing") else 0.0
    stop_d = 0.032 if (op["frame"].get("stop_depth", 0) > 0 and spec["leaf"].get("panel_style") != "glass_frameless") else 0.0
    kin = spec["kinematics"]
    max_open = math.radians(kin.get("max_open_deg") or 90)
    both_ways = bool(kin.get("both_ways"))
    W = float(spec["leaf"]["width"])
    z_bot, z_top, _ = _leaf_extent(leaf_body)
    jt_ = leaf_body.joint
    hinge = np.array([float(jt_.pos[0]), float(jt_.pos[1])])          # hinge axis in the leaf frame (2D)
    rot_sign = u * v                                                   # leaf rotates by u*v*theta about +z
    tpl = CK.ARM_TEMPLATES.get(cl.id, CK.ARM_TEMPLATES["norton_1600"])
    ov = spec.get("closer", {})
    L1 = float(ov.get("arm_main_m", tpl["arm_main_m"]))
    fore_range = tuple(ov.get("arm_fore_range_m", tpl["fore_range_m"]))
    x_p_off = float(ov.get("pinion_offset_m", tpl["pinion_offset_m"]))
    l, w, h = cl.body_size                                             # length along the door, height, projection
    m_body = mat_from_material(model, "aluminum_dark", "mat_closer")
    m_arm = mat_from_material(model, "steel_painted", "mat_closer_arm")
    y_face_pull = yw + v * (depth / 2 + casing)                        # frame / casing face on the pull side (world y)
    y_face_push = yw - v * (depth / 2 + casing)                        # ... on the push side
    y_soffit_far = yw - v * depth / 2                                  # far edge of the head soffit on the push side
    pinion_on_leaf = mech in ("rack_pinion_regular_arm", "rack_pinion_parallel_arm")
    name_pin, name_elb = pfx + "closer_pinion", pfx + "closer_elbow"
    linkage_axis = [0.0, 0.0, 1.0]
    if mech == "rack_pinion_regular_arm":
        # ---- body on the pull face near the top rail, arm plane just above the door top in front of the head face
        face = v
        z_arm = Ho + 0.036
        zc = z_top - 0.026 - w / 2
        y_pin = face * max(t / 2 + 0.55 * h, abs(y_face_pull) + 0.014 + SHAFT_R)
        x_p = hinge[0] + u * x_p_off
        pinion = np.array([x_p, y_pin])
        y_shoe = y_face_pull + face * float(ov.get("shoe_standoff_m", 0.045))
        x_hi = min(0.65, Wo - 0.06)
        shoe_candidates = [np.array([hinge[0] + u * xs, y_shoe]) for xs in np.arange(0.25, x_hi + 1e-9, 0.025)]
        angs = [math.radians(a) for a in (55, 62.5, 70, 77.5, 85, 92.5, 100)]
        ang_dir = 1.0 if face > 0 else -1.0
        angs = [math.atan2(ang_dir * math.sin(a), u * math.cos(a)) for a in angs]      # measured from +x toward the pull side, latch-ward
        y_min_clear = abs(y_face_pull) + 0.022

        def clear_fn(th, pts):
            return all(face * p[1] >= y_min_clear for p in pts)
    elif mech == "rack_pinion_parallel_arm":
        # ---- body on the push face, arm plane under the head soffit (below the stop), shoe on the soffit
        face = -v
        z_arm = Ho - 0.026
        zc = z_arm - 0.014 - w / 2
        y_pin = face * (t / 2 + 0.55 * h)
        x_p = hinge[0] + u * x_p_off
        pinion = np.array([x_p, y_pin])
        y_shoe_mag = min(t / 2 + stop_d + 0.038, abs(y_soffit_far) - 0.012)
        y_shoe_mag = max(y_shoe_mag, t / 2 + 0.03)
        y_shoe = face * y_shoe_mag
        x_hi = min(0.45, Wo - 0.08)
        shoe_candidates = [np.array([hinge[0] + u * xs, y_shoe]) for xs in np.arange(0.05, x_hi + 1e-9, 0.025)]
        angs = [math.radians(a) for a in range(0, 360, 12)]
        band = t / 2 + 0.022

        def clear_fn(th, pts):
            # arms stay on the push side of the (rotated) leaf plane band and off the hinge jamb face
            for p in pts:
                pl = _leaf_frame_pt(p, hinge, rot_sign, th)
                if -0.03 <= u * (pl[0] - hinge[0]) <= W + 0.03 and face * pl[1] < band:
                    return False
                if u * (p[0] - hinge[0]) < -0.01 and abs(p[1]) < abs(y_face_push) + 0.0:
                    return False
            return True
    else:
        # ---- pinion on the frame: concealed closer under the head soffit / operator header on the push-side wall face
        face = -v
        z_arm = Ho - 0.026
        if mech == "rack_pinion_frame_arm":
            y_pin_mag = min(t / 2 + stop_d + 0.045, abs(y_soffit_far) - 0.014)
            y_pin_mag = max(y_pin_mag, t / 2 + 0.03)
        else:
            y_pin_mag = abs(y_face_push) + w / 2 + 0.006                     # operator header centre plane (in front of the wall face)
        y_pin = face * y_pin_mag
        x_p = hinge[0] + u * x_p_off
        pinion = np.array([x_p, y_pin])
        y_shoe = face * (t / 2 + float(ov.get("shoe_standoff_m", 0.045)))     # foot on the leaf's push face
        x_hi = min(0.60, W - 0.15)
        shoe_candidates = [np.array([hinge[0] + u * xs, y_shoe]) for xs in np.arange(0.10, x_hi + 1e-9, 0.025)]
        angs = [math.radians(a) for a in range(0, 360, 12)]
        band = t / 2 + 0.022

        def clear_fn(th, pts):
            for p in pts:
                pl = _leaf_frame_pt(p, hinge, rot_sign, th)
                if -0.03 <= u * (pl[0] - hinge[0]) <= W + 0.03 and face * pl[1] < band:
                    return False
                if face * p[1] < abs(y_face_push) + 0.012 and u * (p[0] - hinge[0]) < -0.02:
                    return False
            return True
    sweep_max = min(max_open + math.radians(8), math.radians(178))
    best = _search_two_bar(pinion_on_leaf, pinion, hinge, rot_sign, L1, fore_range, shoe_candidates, angs, sweep_max, clear_fn)
    if best is None:
        best = _search_two_bar(pinion_on_leaf, pinion, hinge, rot_sign, L1, (fore_range[0] - 0.04, fore_range[1] + 0.08), shoe_candidates, angs, sweep_max, clear_fn, ratio_band=(0.3, 3.0))
    if best is None:
        model.meta.setdefault("notes", []).append(f"{pfx}closer: no arm placement satisfies the clearance / ratio constraints; closer modelled on the door joint")
        pc["mechanism_note"] = "no feasible arm placement (narrow door / thin wall): closer torque left on the door joint"
        return None
    shoe, ang, L2, sign = best["shoe"], best["ang"], best["L2"], best["sign"]
    thetas = np.linspace(0.0, max_open, max(19, int(math.degrees(max_open) / 2.5) + 1))
    q, dq, ds, elbows = CK.twobar_sweep(pinion, shoe, hinge, L1, L2, sign, thetas, rot_sign, pinion_on_leaf)
    qs = best["q_sign"]
    q, dq = q * qs, dq * qs
    pin_axis_sign = qs                                                   # pinion joint axis (0,0,+-1): q > 0 = opening
    elbow0 = pinion + L1 * np.array([math.cos(ang), math.sin(ang)])
    th1 = math.atan2(elbow0[1] - pinion[1], elbow0[0] - pinion[0])
    th2 = math.atan2(shoe[1] - elbow0[1], shoe[0] - elbow0[0]) - th1
    # ---- geometry -------------------------------------------------------------------------------------------
    parent_name = leaf_body.name if pinion_on_leaf else world.name
    if pinion_on_leaf:
        key, mesh = MESH.closer_body_mesh(l=l, w=w, h=h)
        leaf_body.geoms.append(mesh_geom(pfx + "closer_body", key, mesh, (x_p, face * t / 2, zc), q_face(face, u), m_body, 2000, False, FULL_SIMPLE, "closer", cl.name))
        leaf_body.geoms.append(box(pfx + "closer_body_col", (x_p, face * (t / 2 + h / 2), zc), (l / 2, h / 2, w / 2), m_body, 2000, True, False, FULL_SIMPLE, "closer", "Closer body"))
        z_shaft0 = zc + w / 2
        leaf_body.geoms.append(cyl(pfx + "closer_pinion_shaft", (x_p, y_pin, (z_shaft0 + z_arm) / 2), SHAFT_R, max((z_arm - z_shaft0) / 2, 0.006), m_body, (0, 0, 1), 2700, False, True, FULL_ONLY, "closer", "Pinion shaft"))
        arm_pos = (x_p, y_pin, z_arm)
    else:
        xw = x_hinge_axis + x_p                                          # world x of the pinion (leaf frame origin = jamb face x_hinge_axis)
        if mech == "rack_pinion_frame_arm":
            # concealed body inside the head member (visual), spindle down to the arm plane
            world.geoms.append(box(pfx + "closer_concealed_body", (xw + u * 0.05, y_pin, Ho + jamb_t / 2), (min(0.22, Wo / 2 - 0.05), 0.022, max(jamb_t / 2 - 0.002, 0.008)), m_body, 2000, False, True, FULL_ONLY, "closer", "Concealed closer body (in the head)"))
            world.geoms.append(box(pfx + "closer_concealed_slot", (xw + u * 0.05, y_pin, Ho + 0.001), (min(0.22, Wo / 2 - 0.05), 0.024, 0.0015), m_body, 2000, False, True, FULL_ONLY, "closer", "Concealed closer cover plate"))
            z_shaft0 = Ho
        else:
            # operator header on the wall face above the opening (push side)
            y_hdr = y_pin
            z_hdr = Ho + jamb_t + 0.015 + h / 2
            if not any(g.name == "auto_operator_header" for g in world.geoms):
                world.geoms.append(box("auto_operator_header", (0.0, y_hdr, z_hdr), (min(l / 2, Wo / 2 + jamb_t), w / 2, h / 2), m_body, 1500, True, True, FULL_SIMPLE, "closer", "Automatic operator header"))
            z_shaft0 = z_hdr - h / 2
        world.geoms.append(cyl(pfx + "closer_pinion_shaft", (xw, y_pin, (z_shaft0 + z_arm) / 2), SHAFT_R, max((z_shaft0 - z_arm) / 2, 0.006), m_body, (0, 0, 1), 2700, False, True, FULL_ONLY, "closer", "Pinion spindle"))
        arm_pos = (xw, y_pin, z_arm)
    arm1 = Body(pfx + "closer_arm_main", parent_name, arm_pos, tuple(quat_from_axis_angle([0, 0, 1], th1)), None, [], [], FULL_ONLY, "closer", "Closer main arm")
    arm1.joint = Joint(name_pin, "hinge", (0, 0, pin_axis_sign), (0, 0, 0), None, damping=0.01, role="mechanism", label="Closer pinion (spring + hydraulics; 0 = door closed, + = opening)", robot_interactive=False)
    arm1.geoms.append(box(pfx + "closer_arm_main_geom", (L1 / 2, 0, 0), (L1 / 2, ARM_HALF_W, ARM_HALF_T), m_arm, 2700, False, True, FULL_ONLY, "closer", "Main arm"))
    arm1.geoms.append(cyl(pfx + "closer_pinion_boss", (0, 0, 0), 0.016, ARM_HALF_T + 0.002, m_arm, (0, 0, 1), 2700, False, True, FULL_ONLY, "closer", "Pinion boss"))
    model.add_body(arm1)
    arm2 = Body(pfx + "closer_arm_fore", arm1.name, (L1, 0, 0), tuple(quat_from_axis_angle([0, 0, 1], th2)), None, [], [], FULL_ONLY, "closer", "Closer forearm")
    arm2.joint = Joint(name_elb, "hinge", (0, 0, 1), (0, 0, 0), None, damping=0.005, role="mechanism", label="Closer elbow", robot_interactive=False)
    fore_vis = L2 - 0.012
    arm2.geoms.append(box(pfx + "closer_arm_fore_geom", (fore_vis / 2, 0, -0.011), (fore_vis / 2, ARM_HALF_W - 0.001, ARM_HALF_T - 0.001), m_arm, 2700, False, True, FULL_ONLY, "closer", "Forearm (adjustable)"))
    arm2.geoms.append(cyl(pfx + "closer_elbow_pin", (0, 0, -0.006), 0.007, 0.011, m_arm, (0, 0, 1), 2700, False, True, FULL_ONLY, "closer", "Elbow pin"))
    shoe_above = mech == "rack_pinion_parallel_arm"          # soffit shoe sits above the arm plane; feet on faces sit below it
    if shoe_above:
        arm2.geoms.append(cyl(pfx + "closer_shoe_pin", (L2, 0, 0.0015), 0.006, 0.0125, m_arm, (0, 0, 1), 2700, False, True, FULL_ONLY, "closer", "Shoe pin"))
    else:
        arm2.geoms.append(cyl(pfx + "closer_shoe_pin", (L2, 0, -0.016), 0.006, 0.014, m_arm, (0, 0, 1), 2700, False, True, FULL_ONLY, "closer", "Shoe pin"))
    model.add_body(arm2)
    # shoe / foot
    if pinion_on_leaf:
        sx_w = x_hinge_axis + shoe[0]
        if mech == "rack_pinion_regular_arm":
            # L-foot on the frame face above the opening: plate on the face + horizontal leg out to the pivot
            leg = abs(shoe[1] - y_face_pull)
            world.geoms.append(box(pfx + "closer_shoe_plate", (sx_w, y_face_pull + face * 0.002, z_arm + 0.005), (0.022, 0.002, 0.030), m_arm, 2700, False, True, FULL_ONLY, "closer", "Shoe plate (frame face)"))
            world.geoms.append(box(pfx + "closer_shoe", (sx_w, y_face_pull + face * leg / 2, z_arm - 0.027), (0.016, leg / 2, 0.005), m_arm, 2700, False, True, FULL_ONLY, "closer", "Shoe (forearm pivot foot)"))
            anchor_body = "world"
        else:
            # soffit shoe: block under the head on the push side
            world.geoms.append(box(pfx + "closer_shoe", (sx_w, shoe[1], Ho - 0.007), (0.024, 0.018, 0.007), m_arm, 2700, False, True, FULL_ONLY, "closer", "Parallel-arm shoe (soffit)"))
            anchor_body = "world"
        anchor_pos = [sx_w, float(shoe[1]), z_arm]
        model.equalities.append(Equality("connect", pfx + "closer_arm_connect", arm2.name, "world", (0, 0, 0, 0, 0), (L2, 0, 0), FULL_ONLY, "Closer forearm pinned to the frame shoe", solref=(0.002, 1.0), solimp=(0.99, 0.999, 0.001, 0.5, 2.0)))
    else:
        # foot on the leaf's push face at the top rail
        leaf_body.geoms.append(box(pfx + "closer_shoe_plate", (shoe[0], face * (t / 2 + 0.002), z_arm - 0.012), (0.024, 0.002, 0.03), m_arm, 2700, False, True, FULL_ONLY, "closer", "Arm foot plate (leaf)"))
        leaf_body.geoms.append(box(pfx + "closer_shoe", (shoe[0], face * (t / 2 + (abs(shoe[1]) - t / 2) / 2), z_arm - 0.027), (0.016, (abs(shoe[1]) - t / 2) / 2, 0.005), m_arm, 2700, False, True, FULL_ONLY, "closer", "Arm foot (forearm pivot)"))
        anchor_body = leaf_body.name
        anchor_pos = [float(shoe[0]), float(shoe[1]), z_arm]
        model.equalities.append(Equality("connect", pfx + "closer_arm_connect", arm2.name, leaf_body.name, (0, 0, 0, 0, 0), (L2, 0, 0), FULL_ONLY, "Closer forearm pinned to the leaf foot", solref=(0.002, 1.0), solimp=(0.99, 0.999, 0.001, 0.5, 2.0)))
    model.contact_excludes += [(arm1.name, leaf_body.name), (arm2.name, leaf_body.name), (arm1.name, arm2.name)]
    model.meta.setdefault("clearance_allow", []).extend([[pfx + "closer_shoe_pin", pfx + "closer_shoe*", "shoe pin in the shoe"], [pfx + "closer_elbow_pin", pfx + "closer_arm_*", "elbow pin through the arms"],
                                                         [pfx + "closer_pinion_boss", pfx + "closer_pinion_shaft", "arm boss on the pinion shaft"], [pfx + "closer_pinion_boss", pfx + "closer_arm_fore*", "forearm over the boss"],
                                                         [pfx + "closer_shoe_pin", pfx + "closer_arm_fore_geom", "pin through the forearm tip"], [pfx + "closer_elbow_pin", pfx + "closer_shoe_pin", "pins"],
                                                         [pfx + "closer_pinion_shaft", pfx + "closer_body*", "shaft in the body"], [pfx + "closer_pinion_shaft", pfx + "closer_concealed*", "spindle in the concealed body"],
                                                         [pfx + "closer_pinion_shaft", "auto_operator_header", "spindle in the header"], [pfx + "closer_hold_magnet", pfx + "closer_arm_main_geom", "hold-open unit on the arm"],
                                                         [pfx + "closer_shoe_pin", pfx + "closer_hold_magnet", "pin"], [pfx + "closer_shoe", pfx + "closer_shoe_plate", "foot on its plate"]])
    if mech in ("rack_pinion_regular_arm", "rack_pinion_parallel_arm") and cl.kind == "electromagnetic_hold":
        arm1.geoms.append(box(pfx + "closer_hold_magnet", (L1 * 0.45, 0, ARM_HALF_T + 0.012), (0.045, 0.014, 0.012), m_body, 2000, False, True, FULL_ONLY, "closer", "Electromagnetic hold-open unit"))
    # ---- calibration ---------------------------------------------------------------------------------------
    pinion_p, reduced, table = _calibrate_arm(pc, spec, phys, thetas, q, dq, max_open)
    kp = pinion_p["spring_stiffness_Nm_per_rad"]
    arm1.joint.stiffness = kp
    arm1.joint.springref = -pinion_p["spring_preload_Nm"] / kp
    arm1.joint.damping = pinion_p["damping_check_Nms_per_rad"]
    arm1.joint.damping_closing = pinion_p["damping_sweep_Nms_per_rad"]
    arm1.joint.damping_opening = pinion_p["damping_check_Nms_per_rad"]
    arm1.joint.backcheck_angle = pinion_p["q_backcheck_rad"]
    arm1.joint.backcheck_damping = pinion_p["damping_backcheck_Nms_per_rad"]
    arm1.joint.notes = "pinion torque = preload + k*q; door torque = pinion torque x dq/dtheta (physics.closer.mechanism_params.ratio_table)"
    st = pc["settings"]
    _apply_door_joint_reduced(leaf_body, reduced, phys, both_ways)
    b_air = float(phys.get("hinge", {}).get("air_damping_Nms_per_rad", 0.0))
    reduced["delayed_angle_rad"] = math.radians(st["delayed_angle_deg"]) if st["delayed_action_s"] > 0 else None
    reduced["hold_open_kind"] = st.get("hold_open_kind", "none")
    laws = pc.setdefault("laws", [])
    laws.append(CK.law_from_windows(name_pin, ("full",), pinion_p["damping_check_Nms_per_rad"], pinion_p["damping_sweep_Nms_per_rad"], pinion_p["damping_latch_Nms_per_rad"],
                                    pinion_p["damping_check_Nms_per_rad"], pinion_p["damping_backcheck_Nms_per_rad"], pinion_p["q_latch_rad"], pinion_p["q_backcheck_rad"],
                                    pinion_p["q_delay_rad"], pinion_p["damping_delay_Nms_per_rad"], pinion_p["q_hold_rad"], pinion_p["hold_torque_Nm"],
                                    math.radians(3.0) * max(dq[-1], 0.5), st.get("hold_open_kind", "none"), "pinion_rad", float(q[-1])))
    laws.append(_door_law(leaf_body.joint.name, REDUCED, reduced, b_air))
    geometry = {"arm": mech, "pinion_on": "leaf" if pinion_on_leaf else "frame", "arm_main_m": L1, "arm_fore_m": L2, "pinion_offset_m": x_p_off, "pinion_standoff_m": float(abs(y_pin)),
                "shoe_offset_m": float(u * (shoe[0] - hinge[0])), "shoe_standoff_m": float(abs(shoe[1])), "arm_height_m": float(z_arm), "rest_main_arm_angle_deg": float(math.degrees(th1)),
                "elbow_sign": int(sign), "pinion_ratio_range": [float(dq.min()), float(dq.max())], "pinion_travel_deg": float(math.degrees(q[-1])),
                "source": "LCN 4040XP / Norton 1600 / Dorma TS 83 templates (main arm, pinion offset); shoe placed by kinematic search (docs/PHYSICS.md)"}
    mp = pc.setdefault("mechanism_params", {})
    mp.update({"joint_pinion": name_pin, "joint_elbow": name_elb, "geometry": geometry, "pinion": pinion_p, "ratio_table": table})
    if pfx:
        pc.setdefault("mechanism_leaves", {})[leaf_body.name] = {"joint_pinion": name_pin, "joint_elbow": name_elb, "geometry": geometry, "pinion": pinion_p}
    pc["reduced"] = reduced
    model.linkages.append({"name": pfx + "closer", "type": "two_bar",
                           "pinion": {"body": arm1.name, "joint": name_pin, "parent": parent_name if pinion_on_leaf else "world"},
                           "elbow": {"body": arm2.name, "joint": name_elb},
                           "anchor": {"body": anchor_body, "pos": [float(x) for x in anchor_pos]},
                           "equality": pfx + "closer_arm_connect", "axis": [0.0, 0.0, float(pin_axis_sign)], "L1": float(L1), "L2": float(L2), "elbow_sign": int(sign),
                           "arm_dir0": [1.0, 0.0, 0.0], "fore_dir0": [1.0, 0.0, 0.0],
                           "note": "elbow = P + a*ex + elbow_sign*h*ey with ex = unit(anchor - P) projected on the plane normal to `axis`, ey = axis x ex, a = (L1^2 - L2^2 + d^2)/(2d), h = sqrt(L1^2 - a^2); P = pinion joint anchor in world"})
    if cl.kind in ("auto_operator_low_energy", "auto_operator_full"):
        mot = pc.get("motor", {})
        ratio_mean = float(np.mean(dq))
        tau_p = float(mot.get("max_torque_Nm", 60.0)) / max(ratio_mean, 0.3)
        q_max = float(q[-1])
        model.meta.setdefault("actuators", []).append({"name": pfx + "swing_operator", "joint": name_pin, "kind": "position", "kp": float(tau_p / 0.35), "kv": float(0.25 * tau_p / 0.35),
                                                      "forcerange": (-tau_p, tau_p), "ctrlrange": (0.0, q_max), "tiers": ["full"], "door_joint": leaf_body.joint.name,
                                                      "note": "position servo on the operator pinion; ctrl = pinion angle (physics.closer.mechanism_params.ratio_table maps door angle -> pinion angle); ctrl 0 with the servo unpowered = spring close"})
        model.meta.setdefault("actuators", []).append({"name": pfx + "swing_operator_reduced", "joint": leaf_body.joint.name, "kind": "position", "kp": float(mot.get("max_torque_Nm", 60.0) / 0.35), "kv": float(0.25 * mot.get("max_torque_Nm", 60.0) / 0.35),
                                                      "forcerange": (-float(mot.get("max_torque_Nm", 60.0)), float(mot.get("max_torque_Nm", 60.0))), "ctrlrange": (0.0, max_open), "tiers": ["simple", "minimal"],
                                                      "door_joint": leaf_body.joint.name, "note": "door-level equivalent for the reduced tiers / RL USD"})
        pc["motor"]["pinion_max_torque_Nm"] = tau_p
        pc["motor"]["pinion_ctrl_max_rad"] = q_max
    return arm1


# ---------------------------------------------------------------------------
# telescoping devices: pneumatic screen closer, gate spring, hydraulic gate closer (vertical-axis leaves)
# ---------------------------------------------------------------------------
def _add_telescoping(model: Model, world: Body, leaf_body: Body, spec: dict, phys: dict, cl: H.CloserModel, u: float, v: float, x_hinge_axis: float, Hh: float, t: float, Wo: float, jamb_t: float, pfx: str):
    pc = phys["closer"]
    mech = pc["mechanism"]
    tpl = CK.STRUT_TEMPLATES.get(cl.id, CK.STRUT_TEMPLATES["gate_hydraulic"])
    ov = spec.get("closer", {})
    op = spec["opening"]
    is_gate = op["frame"]["kind"] in ("gate_posts", "pressure_frame")
    yw = float(model.meta.get("wall_y", 0.0))
    depth = float(op["wall_thickness"]) if op["frame"]["kind"] != "aluminum_storefront" else max(0.114, float(op["wall_thickness"]))
    kin = spec["kinematics"]
    max_open = math.radians(kin.get("max_open_deg") or 90)
    both_ways = bool(kin.get("both_ways"))
    W = float(spec["leaf"]["width"])
    z_bot, z_top, _ = _leaf_extent(leaf_body)
    jt_ = leaf_body.joint
    hinge = np.array([float(jt_.pos[0]), float(jt_.pos[1])])
    rot_sign = u * v
    face = -v                                                        # the side the leaf swings away from (device extends on opening)
    if both_ways:
        face = 1.0
    stroke = float(ov.get("stroke_m", tpl["stroke_m"]))
    length = float(ov.get("length_m", tpl["length_m"]))
    r_tube, r_rod = tpl["r_tube"], tpl["r_rod"]
    z_s = float(ov.get("mount_height_m", z_top - 0.09))
    # jamb / post bracket pivot (world) and the leaf bracket pivot (leaf frame): searched so the stroke covers the range
    if is_gate:
        ps = jamb_t
        y_b_mag = ps / 2 + 0.018
        x_post_c = x_hinge_axis - u * ps / 2                                    # post centre (world x)
        post_candidates = [x_post_c + u * dx for dx in (-0.0, 0.02)]
    else:
        y_b_mag = abs(yw - v * depth / 2) + 0.02
        post_candidates = [x_hinge_axis - u * jamb_t / 2]
    y_leaf_mag = t / 2 + 0.028
    best = None
    for xb in post_candidates:
        for reach in np.arange(0.20, min(0.60, W - 0.08) + 1e-9, 0.02):
            A_l = np.array([hinge[0] + u * reach, face * y_leaf_mag])
            B_w = np.array([xb - x_hinge_axis, face * y_b_mag])              # in the leaf-closed frame (= world shifted by x_hinge_axis)
            thetas = np.linspace(0.0, max_open, 25)
            ds = []
            for th in thetas:
                A_w = hinge + _rot_z(rot_sign * th) @ (A_l - hinge)
                ds.append(float(np.linalg.norm(A_w - B_w)))
            ds = np.array(ds)
            ext = ds.max() - ds[0]
            if ext > stroke - 0.01 or ds[0] < 0.6 * length:
                continue
            # moment arm of the force line about the hinge at closed (closing power) - prefer larger
            dvec = (A_l - B_w) / max(ds[0], 1e-9)
            arm = abs(float(np.cross(A_l - hinge, dvec)))
            score = -arm + 0.4 * ext
            if best is None or score < best["score"]:
                best = {"score": score, "xb": xb, "reach": reach, "A_l": A_l, "B_w": B_w, "ds": ds, "thetas": thetas, "arm": arm}
    if best is None:
        model.meta.setdefault("notes", []).append(f"{pfx}closer: no strut placement covers the door range within the stroke; closer left on the door joint")
        pc["mechanism_note"] = "no feasible strut placement; closer torque left on the door joint"
        return None
    A_l, B_w, ds, thetas = best["A_l"], best["B_w"], best["ds"], best["thetas"]
    d0 = float(ds[0])
    mat = mat_from_material(model, "aluminum" if cl.kind == "pneumatic" else ("steel_galvanized" if cl.id == "gate_spring" else "black_matte_metal"), "mat_strut")
    # brackets
    B_world = np.array([best["xb"], face * y_b_mag, z_s])
    if is_gate:
        world.geoms.append(box(pfx + "closer_post_bracket", (best["xb"], face * (jamb_t / 2 + 0.006), z_s), (0.02, 0.006, 0.02), mat, 7800, False, True, FULL_ONLY, "closer", "Post bracket"))
        world.geoms.append(box(pfx + "closer_post_bracket_arm", (best["xb"], face * (jamb_t / 2 + 0.012 + (y_b_mag - jamb_t / 2 - 0.012) / 2), z_s - 0.012), (0.012, (y_b_mag - jamb_t / 2 - 0.012) / 2 + 0.004, 0.004), mat, 7800, False, True, FULL_ONLY, "closer", "Bracket ear"))
    else:
        yf = yw - v * depth / 2
        world.geoms.append(box(pfx + "closer_jamb_bracket", (best["xb"], yf + face * 0.006, z_s), (0.02, 0.006, 0.02), mat, 7800, False, True, FULL_ONLY, "closer", "Jamb bracket"))
        y_piv = face * y_b_mag
        world.geoms.append(box(pfx + "closer_jamb_bracket_arm", (best["xb"], (yf + face * 0.012 + y_piv) / 2, z_s - 0.012), (0.012, abs(y_piv - yf - face * 0.012) / 2 + 0.004, 0.004), mat, 7800, False, True, FULL_ONLY, "closer", "Bracket ear"))
    leaf_body.geoms.append(box(pfx + "closer_leaf_bracket", (A_l[0], face * (t / 2 + 0.004), z_s), (0.02, 0.004, 0.02), mat, 7800, False, True, FULL_ONLY, "closer", "Door bracket"))
    leaf_body.geoms.append(box(pfx + "closer_leaf_bracket_arm", (A_l[0], face * (t / 2 + 0.008 + (y_leaf_mag - t / 2 - 0.008) / 2), z_s - 0.012), (0.012, (y_leaf_mag - t / 2 - 0.008) / 2 + 0.004, 0.004), mat, 7800, False, True, FULL_ONLY, "closer", "Bracket ear"))
    # bodies: cylinder hinged at the bracket, rod sliding along its axis
    dir0 = (A_l - B_w)
    th0 = math.atan2(dir0[1], dir0[0])
    cyl_b = Body(pfx + "closer_cyl", world.name, tuple(B_world), tuple(quat_from_axis_angle([0, 0, 1], th0)), None, [], [], FULL_ONLY, "closer", cl.name + " (cylinder)")
    cyl_b.joint = Joint(pfx + "closer_base_hinge", "hinge", (0, 0, 1), (0, 0, 0), None, damping=0.005, role="mechanism", label="Closer body pivot", robot_interactive=False)
    tube_len = min(length, d0 - 0.035)
    cyl_b.geoms.append(cyl(pfx + "closer_tube", (0.02 + tube_len / 2, 0, 0), r_tube, tube_len / 2, mat, (1, 0, 0), 2000, False, True, FULL_ONLY, "closer", "Closer tube" if cl.id != "gate_spring" else "Coil spring"))
    cyl_b.geoms.append(cyl(pfx + "closer_base_eye", (0, 0, 0), 0.011, 0.006, mat, (0, 0, 1), 2700, False, True, FULL_ONLY, "closer", "Pivot eye"))
    model.add_body(cyl_b)
    rod = Body(pfx + "closer_rod", cyl_b.name, (0, 0, 0), QUAT_ID, None, [], [], FULL_ONLY, "closer", cl.name + " (rod)")
    rod.joint = Joint(pfx + "closer_rod_slide", "slide", (1, 0, 0), (0, 0, 0), (-0.005, stroke + 0.02), damping=0.0, role="mechanism", label="Closer rod extension (0 = door closed, + = extending)", robot_interactive=False)
    rod_len = stroke + 0.10
    rod.geoms.append(cyl(pfx + "closer_rod_geom", (d0 - 0.008 - rod_len / 2, 0, 0), r_rod, rod_len / 2, mat, (1, 0, 0), 7800, False, True, FULL_ONLY, "closer", "Piston rod"))
    rod.geoms.append(cyl(pfx + "closer_rod_eye", (d0, 0, 0), 0.010, 0.006, mat, (0, 0, 1), 7800, False, True, FULL_ONLY, "closer", "Rod eye"))
    model.add_body(rod)
    model.equalities.append(Equality("connect", pfx + "closer_rod_connect", rod.name, leaf_body.name, (0, 0, 0, 0, 0), (d0, 0, 0), FULL_ONLY, "Closer rod pinned to the door bracket", solref=(0.002, 1.0), solimp=(0.99, 0.999, 0.001, 0.5, 2.0)))
    model.contact_excludes += [(cyl_b.name, rod.name), (rod.name, leaf_body.name), (cyl_b.name, leaf_body.name)]
    model.meta.setdefault("clearance_allow", []).extend([[pfx + "closer_rod_geom", pfx + "closer_tube", "rod slides inside the tube"], [pfx + "closer_rod_geom", pfx + "closer_rod_eye", "rod eye"]])
    # ---- physics: force along the rod, door torque = F * ds/dtheta ------------------------------------------------
    strut = pc["strut"]
    F0, kN = float(strut["spring_force_closed_N"]), float(strut["spring_rate_N_per_m"])
    c_close, c_open = float(strut["damping_close_Ns_per_m"]), float(strut["damping_open_Ns_per_m"])
    s_ext = ds - d0
    dsdth = np.gradient(s_ext, thetas)
    F = F0 + kN * s_ext                                       # tension pulling the rod in (closing)
    tau_full = F * dsdth
    rod.joint.stiffness = kN
    rod.joint.springref = -F0 / kN
    rod.joint.damping = c_open
    rod.joint.damping_closing = c_close
    rod.joint.damping_opening = c_open
    m = float(phys["mass"]["total_kg"])
    fric = float(phys.get("hinge", {}).get("coulomb_torque_Nm", 0.0)) + 0.5 * float(phys.get("hinge", {}).get("stick_torque_Nm", 0.0))
    b_air = float(phys.get("hinge", {}).get("air_damping_Nms_per_rad", 0.0))
    st = pc["settings"]
    tau_fn = lambda th: _interp(thetas, tau_full, th)
    ds_fn = lambda th: _interp(thetas, dsdth, th)
    s_latch = _interp(thetas, s_ext, math.radians(st["latch_angle_deg"])) if st["latch_angle_deg"] > 0 else 0.0
    c_latch = c_close * st["latch_speed_factor"]
    b_full = lambda th: (c_latch if th < math.radians(st["latch_angle_deg"]) else c_close) * ds_fn(th) ** 2
    t_full, t12_full, w_full = CK.closing_time(m, W, 0, 0, 0, fric, None, 0.0, b_air, theta0=min(math.pi / 2, max_open), tau_fn=tau_fn, b_fn=b_full)
    sel = thetas <= min(math.radians(90), max_open) + 1e-9
    tr0, kr, e_max, e_rms = CK.fit_linear(thetas[sel], tau_full[sel])
    tr0, kr = max(tr0, 0.3), max(kr, 0.3)
    lo, hi = 0.0, 3000.0
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        t_r, _, _ = CK.closing_time(m, W, tr0, kr, mid, fric, mid * st["latch_speed_factor"], math.radians(st["latch_angle_deg"]), b_air, theta0=min(math.pi / 2, max_open))
        if t_r > t_full:
            hi = mid
        else:
            lo = mid
    br = 0.5 * (lo + hi)
    t_red, t12_red, w_red = CK.closing_time(m, W, tr0, kr, br, fric, br * st["latch_speed_factor"], math.radians(st["latch_angle_deg"]), b_air, theta0=min(math.pi / 2, max_open))
    br_open = c_open * float(np.mean(dsdth[sel] ** 2))
    s_hold = _interp(thetas, s_ext, math.radians(st["hold_open_deg"])) if st["hold_open_deg"] > 0 and max_open >= math.radians(st["hold_open_deg"]) - 1e-6 else None
    F_hold = 1.4 * float(F0 + kN * s_hold) if s_hold is not None else 0.0
    reduced = {"spring_preload_Nm": float(tr0), "spring_stiffness_Nm_per_rad": float(kr), "damping_closing": float(br), "damping_latch": float(br * st["latch_speed_factor"]), "damping_opening": float(br_open),
               "backcheck_damping": 0.0, "backcheck_angle_rad": None, "latch_angle_rad": math.radians(st["latch_angle_deg"]), "delayed_action_damping": 0.0, "delayed_angle_rad": None,
               "hold_open_rad": None if s_hold is None else math.radians(st["hold_open_deg"]), "hold_torque_Nm": float(F_hold * ds_fn(math.radians(st["hold_open_deg"]))) if s_hold is not None else 0.0,
               "hold_open_kind": st.get("hold_open_kind", "none"),
               "fit": {"torque_max_rel_err": float(e_max), "torque_rms_rel_err": float(e_rms), "closing_time_full_s": float(t_full), "closing_time_reduced_s": float(t_red),
                       "sweep_time_full_s": float(t12_full), "sweep_time_reduced_s": float(t12_red), "final_speed_full_rad_s": float(w_full), "final_speed_reduced_rad_s": float(w_red),
                       "damping_curve_max_rel_dev": float(np.max(np.abs((c_close * dsdth[sel] ** 2) / max(br, 1e-9) - 1.0))) if br > 0 else 0.0,
                       "note": "reduced = linear spring least-squares fitted to F(s)*ds/dtheta over 0-90 deg; damping fitted to the full closing time"}}
    _apply_door_joint_reduced(leaf_body, reduced, phys, both_ways)
    laws = pc.setdefault("laws", [])
    laws.append(CK.law_from_windows(rod.joint.name, ("full",), c_open, c_close, c_latch, c_open, 0.0, s_latch, None, None, 0.0, s_hold, F_hold, 0.004, st.get("hold_open_kind", "none"), "rod_m", float(s_ext.max())))
    laws.append(_door_law(leaf_body.joint.name, REDUCED, reduced, b_air))
    pc["reduced"] = reduced
    geometry = {"mechanism": mech, "length_m": float(tube_len + 0.02), "stroke_m": stroke, "closed_pivot_distance_m": d0, "extension_at_max_open_m": float(s_ext.max()), "mount_height_m": z_s,
                "leaf_reach_m": float(best["reach"]), "bracket_standoff_m": float(y_b_mag), "moment_arm_closed_m": float(best["arm"]), "side": "push" if face == -v else "pull",
                "source": "Wright V150 (screen), National V19 gate spring, Lockey TB / Kant-Slam hydraulic gate closer; brackets placed so the stroke covers the door range"}
    table = [[round(math.degrees(th), 1), round(float(s), 4), round(float(dd), 4), round(float(tt), 2)] for th, s, dd, tt in zip(thetas, s_ext, dsdth, tau_full)]
    pc.setdefault("mechanism_params", {}).update({"joint_base": cyl_b.joint.name, "joint_slide": rod.joint.name, "geometry": geometry, "rod": {"spring_force_closed_N": F0, "spring_rate_N_per_m": kN, "damping_close_Ns_per_m": c_close, "damping_open_Ns_per_m": c_open, "s_latch_m": float(s_latch), "s_hold_m": s_hold, "hold_force_N": F_hold},
                                                 "ratio_table": {"columns": ["door_deg", "rod_extension_m", "dext_ddoor_m_per_rad", "door_torque_Nm"], "rows": table}})
    model.linkages.append({"name": pfx + "closer", "type": "telescoping",
                           "base": {"body": cyl_b.name, "joint": cyl_b.joint.name, "parent": "world", "pos": [float(x) for x in B_world]},
                           "slide": {"body": rod.name, "joint": rod.joint.name, "axis_local": [1.0, 0.0, 0.0], "offset": float(d0)},
                           "anchor": {"body": leaf_body.name, "pos": [float(A_l[0]), float(A_l[1]), float(z_s)]}, "equality": pfx + "closer_rod_connect", "axis": [0.0, 0.0, 1.0]})
    return cyl_b


# ---------------------------------------------------------------------------
# pivot / hinge devices: floor spring, spring hinges (torque at the door joint, real housing geometry)
# ---------------------------------------------------------------------------
def _add_pivot_device(model: Model, world: Body, leaf_body: Body, spec: dict, phys: dict, cl: H.CloserModel, u: float, v: float, x_hinge_axis: float, Hh: float, t: float, pfx: str):
    pc = phys["closer"]
    jt_ = leaf_body.joint
    z_bot, z_top, _ = _leaf_extent(leaf_body)
    st = pc.get("settings", {})
    b_air = float(phys.get("hinge", {}).get("air_damping_Nms_per_rad", 0.0))
    if cl.kind == "floor_spring":
        m_ = mat_from_material(model, "stainless", "mat_closer")
        xw = x_hinge_axis + float(jt_.pos[0])
        yp = float(jt_.pos[1])
        l, w, h = cl.body_size
        world.geoms.append(box(pfx + "floor_spring_box", (xw + u * (l / 2 - 0.04), yp, -h / 2 - 0.004), (l / 2, w / 2, h / 2), m_, 7900, False, True, FULL_ONLY, "closer", "Floor spring body (in the floor box)"))
        world.geoms.append(box(pfx + "floor_spring_cover", (xw + u * (l / 2 - 0.04), yp, 0.002), (l / 2 + 0.02, w / 2 + 0.02, 0.002), m_, 7900, False, True, FULL_SIMPLE, "closer", "Floor spring cover plate"))
        leaf_body.geoms.append(cyl(pfx + "floor_spring_spindle", (float(jt_.pos[0]), yp, max(z_bot / 2, 0.004)), 0.011, max(z_bot / 2, 0.004), m_, (0, 0, 1), 7900, False, True, FULL_ONLY, "closer", "Floor spring spindle"))
        leaf_body.geoms.append(box(pfx + "floor_spring_shoe", (float(jt_.pos[0]) + u * 0.03, yp, z_bot + 0.02), (0.05, min(t / 2 + 0.002, 0.03), 0.02), m_, 7900, False, True, FULL_ONLY, "closer", "Bottom arm / pivot shoe"))
        hold = st.get("hold_open_deg", 0.0)
        max_open = math.radians(spec["kinematics"].get("max_open_deg") or 90)
        tau0, k = float(pc["spring_preload_Nm"]), float(pc["spring_stiffness_Nm_per_rad"])
        hold_ok = hold > 0 and max_open >= math.radians(hold) - 1e-6
        reduced = {"spring_preload_Nm": tau0, "spring_stiffness_Nm_per_rad": k, "damping_closing": float(pc["damping_closing"]), "damping_latch": float(pc["damping_latch"]),
                   "damping_opening": float(pc["damping_opening"]), "backcheck_damping": float(pc["backcheck_damping"]), "backcheck_angle_rad": pc.get("backcheck_angle_rad"),
                   "latch_angle_rad": float(pc["latch_angle_rad"]), "delayed_action_damping": 0.0, "delayed_angle_rad": None,
                   "hold_open_rad": math.radians(hold) if hold_ok else None, "hold_torque_Nm": 1.6 * (tau0 + k * math.radians(hold)) if hold_ok else 0.0, "hold_open_kind": "mechanical" if hold_ok else "none",
                   "fit": {"note": "floor spring: the spindle is the door pivot, the door joint carries the mechanism in every tier (no reduction)"}}
        pc["reduced"] = reduced
        pc.setdefault("mechanism_params", {}).update({"joint": jt_.name, "geometry": {"body_size_m": list(cl.body_size), "spindle_at_hinge_axis": True, "source": "Dorma BTS 80 floor spring, 90 deg hold-open option"}})
        pc.setdefault("laws", []).append(_door_law(jt_.name, ALL_TIERS, reduced, b_air))
        jt_.damping_closing, jt_.damping_opening = reduced["damping_closing"], reduced["damping_opening"]
        jt_.backcheck_angle, jt_.backcheck_damping = reduced["backcheck_angle_rad"], reduced["backcheck_damping"]
        jt_.damping = b_air + reduced["damping_opening"]
    elif cl.kind == "spring_hinge":
        m_ = mat_from_material(model, "steel_galvanized", "mat_spring_hinge")
        n = int(spec["hinge"]["count"])
        hg = H.HINGES[spec["hinge"]["model"]]
        hh = hg.size[0] if hg.size and hg.size[0] > 0 else 0.10
        if n == 2:
            zs = [z_bot + 0.25, z_bot + Hh - 0.18]
        elif n == 3:
            zs = [z_bot + 0.25, z_bot + Hh / 2, z_bot + Hh - 0.18]
        else:
            zs = [z_bot + 0.25 + i * (Hh - 0.43) / max(n - 1, 1) for i in range(n)]
        for k_, zz in enumerate(zs):
            # spring barrel on the hinge line + tension adjusting collar (hex + pin) at its top
            leaf_body.geoms.append(cyl(pfx + f"spring_hinge_barrel_{k_}", (float(jt_.pos[0]), float(jt_.pos[1]), zz), 0.0115, hh / 2 + 0.004, m_, (0, 0, 1), 7850, False, True, FULL_SIMPLE, "hinge", f"Spring hinge {k_ + 1} barrel"))
            leaf_body.geoms.append(cyl(pfx + f"spring_hinge_collar_{k_}", (float(jt_.pos[0]), float(jt_.pos[1]), zz + hh / 2 + 0.010), 0.0135, 0.006, m_, (0, 0, 1), 7850, False, True, FULL_ONLY, "hinge", "Tension adjusting collar"))
        pc.setdefault("mechanism_params", {}).update({"joint": jt_.name, "geometry": {"barrels": n, "barrel_height_m": hh, "source": "Bommer 4310 / 3029 spring hinges; tension set with the adjusting collar pin"}})
        pc["reduced"] = {"spring_preload_Nm": float(pc["spring_preload_Nm"]), "spring_stiffness_Nm_per_rad": float(pc["spring_stiffness_Nm_per_rad"]), "damping_closing": float(pc["damping_closing"]), "damping_opening": float(pc["damping_opening"]),
                         "fit": {"note": "spring hinges act at the door joint in every tier (no reduction)"}}


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------
def add_closer(model: Model, world: Body, leaf_body: Body, spec: dict, phys: dict, u: float, v: float, x_hinge_axis: float, Hh: float, t: float, Wo: float, jamb_t: float, tier_full_arms=True):
    """Build the door's closer / operator mechanism (see the module docstring).  Called per leaf (pairs: twice)."""
    cl = H.CLOSERS[spec["closer"]["model"]]
    if cl.kind == "none" or leaf_body.joint is None or leaf_body.joint.type != "hinge":
        return
    pc = phys["closer"]
    pfx = "" if leaf_body.name == "leaf" else leaf_body.name + "_"
    if pfx:
        # pairs share one physics block: keep per-leaf laws (joint names differ) but do not duplicate the door-level design
        pass
    if cl.kind in CK.ARM_CLOSER_KINDS:
        _add_arm_closer(model, world, leaf_body, spec, phys, cl, u, v, x_hinge_axis, Hh, t, Wo, jamb_t, pfx)
    elif cl.kind in ("pneumatic", "gate"):
        _add_telescoping(model, world, leaf_body, spec, phys, cl, u, v, x_hinge_axis, Hh, t, Wo, jamb_t, pfx)
    elif cl.kind in ("floor_spring", "spring_hinge"):
        _add_pivot_device(model, world, leaf_body, spec, phys, cl, u, v, x_hinge_axis, Hh, t, pfx)


def add_gas_strut(model: Model, world: Body, leaf_body: Body, spec: dict, phys: dict, W: float, Ho: float, t: float, zf: float, ceiling: bool):
    """Hatch lift-assist gas strut: cylinder hinged at a curb / pit bracket, rod pinned to the leaf; the strut lies in
    a vertical plane (normal = x) beside the leaf's side edge.  Leaf frame: origin at the hinge line (far edge), the
    slab spans y in [-Ho, 0]; hinge axis (-1, 0, 0), q > 0 lifts the near edge."""
    pc = phys["closer"]
    cl = H.CLOSERS[spec["closer"]["model"]]
    if cl.kind != "gas_strut" or leaf_body.joint is None:
        return
    tpl = CK.STRUT_TEMPLATES["gas_strut"]
    ov = spec.get("closer", {})
    stroke = float(ov.get("stroke_m", tpl["stroke_m"]))
    max_open = math.radians(spec["kinematics"].get("max_open_deg", 90))
    x_s = W / 2 - 0.07
    zsign = 1.0 if ceiling else -1.0          # the strut lives above a ceiling hatch (attic side), below a floor hatch (pit side)
    mat = mat_from_material(model, "stainless", "mat_strut")
    # search the bracket (base, in the leaf-closed frame = world offset by the hinge) and the tip (leaf frame)
    best = None
    for tip_y in np.arange(0.14, min(0.42, Ho - 0.12), 0.02):
        for base_y in np.arange(0.10, 0.55, 0.03):
            for base_z in np.arange(0.16, 0.42, 0.03):
                A_l = np.array([-tip_y, zsign * (t / 2 + 0.012)])                 # (y, z) in the leaf frame
                B_l = np.array([-base_y, zsign * base_z])
                thetas = np.linspace(0.0, max_open, 25)
                ds = []
                for th in thetas:
                    c, s = math.cos(th), math.sin(th)                               # rotation about -x by th: (y, z) -> (y c + z s, -y s + z c)
                    A_w = np.array([A_l[0] * c + A_l[1] * s, -A_l[0] * s + A_l[1] * c])
                    ds.append(float(np.linalg.norm(A_w - B_l)))
                ds = np.array(ds)
                ext = ds.max() - ds.min()
                d0 = ds[0]
                if ext > stroke - 0.01 or d0 > 0.60 or d0 < 0.22 or ds.argmax() < len(ds) - 3:
                    continue
                dvec = (A_l - B_l) / max(d0, 1e-9)
                arm = abs(float(np.cross(A_l, dvec)))
                score = -arm
                if best is None or score < best["score"]:
                    best = {"score": score, "A_l": A_l, "B_l": B_l, "ds": ds, "thetas": thetas, "arm": arm, "tip_y": tip_y, "base_y": base_y, "base_z": base_z}
    if best is None:
        pc["mechanism_note"] = "no feasible strut placement; lift assist left on the hatch joint"
        return
    A_l, B_l, ds, thetas = best["A_l"], best["B_l"], best["ds"], best["thetas"]
    d0 = float(ds[0])
    hinge_w = np.asarray(leaf_body.pos, float)                                     # leaf frame origin (world)
    B_w = hinge_w + np.array([x_s, B_l[0], B_l[1]])
    # mount: pit wall + bracket below a floor hatch, post on the curb above a ceiling hatch
    if ceiling:
        world.geoms.append(box("strut_post", (B_w[0], hinge_w[1] - 0.05, (zf + 0.04 + B_w[2]) / 2), (0.02, 0.02, (B_w[2] - zf - 0.04) / 2 + 0.01), mat, 7800, False, True, FULL_ONLY, "closer", "Strut mounting post (curb)"))
        world.geoms.append(box("strut_post_arm", (B_w[0], (hinge_w[1] - 0.05 + B_w[1]) / 2, B_w[2]), (0.012, abs(B_w[1] - hinge_w[1] + 0.05) / 2 + 0.01, 0.008), mat, 7800, False, True, FULL_ONLY, "closer", "Strut bracket arm"))
    else:
        world.geoms.append(box("pit_wall_r", (W / 2 + 0.06, 0.0, -0.75), (0.02, Ho / 2 + 0.06, 0.75), mat_rgba(model, "mat_pit_wall", (0.35, 0.35, 0.35, 1), 0.9), 2400, True, True, FULL_ONLY, "floor", "Pit wall"))
        world.geoms.append(box("strut_bracket", (W / 2 + 0.03 + 0.005, B_w[1], B_w[2]), (0.025, 0.02, 0.02), mat, 7800, False, True, FULL_ONLY, "closer", "Strut bracket (pit wall)"))
    leaf_body.geoms.append(box("strut_leaf_bracket", (x_s, A_l[0], zsign * (t / 2 + 0.004)), (0.02, 0.02, 0.004), mat, 7800, False, True, FULL_ONLY, "closer", "Strut bracket (leaf)"))
    r_tube, r_rod = tpl["r_tube"], tpl["r_rod"]
    dir0 = np.array([0.0, A_l[0] - B_l[0], A_l[1] - B_l[1]])
    # cylinder body: hinge axis x, local +x must point along the plane normal... use local z as the rod axis? keep local x = rod axis:
    # rotate local x onto dir0 with the hinge axis (1,0,0) mapped to the world x: choose the quaternion from z-to-direction composition
    from ..ir import mat_to_quat
    ex = dir0 / np.linalg.norm(dir0)
    ez = np.array([1.0, 0.0, 0.0])                                 # hinge axis (world x) = local z
    ey = np.cross(ez, ex)
    R = np.column_stack([ex, ey, ez])
    cyl_b = Body("strut_cyl", world.name, tuple(B_w), tuple(mat_to_quat(R)), None, [], [], FULL_ONLY, "closer", "Gas strut cylinder")
    cyl_b.joint = Joint("strut_base_hinge", "hinge", (0, 0, 1), (0, 0, 0), None, damping=0.005, role="mechanism", label="Strut body pivot", robot_interactive=False)
    tube_len = d0 - 0.06
    cyl_b.geoms.append(cyl("strut_tube", (0.02 + tube_len / 2, 0, 0), r_tube, tube_len / 2, mat, (1, 0, 0), 2500, False, True, FULL_ONLY, "closer", "Gas strut cylinder"))
    cyl_b.geoms.append(cyl("strut_base_eye", (0, 0, 0), 0.010, 0.006, mat, (0, 0, 1), 7800, False, True, FULL_ONLY, "closer", "Pivot eye"))
    model.add_body(cyl_b)
    rod = Body("strut_rod", cyl_b.name, (0, 0, 0), QUAT_ID, None, [], [], FULL_ONLY, "closer", "Gas strut rod")
    rod.joint = Joint("strut_rod_slide", "slide", (1, 0, 0), (0, 0, 0), (-0.005, stroke + 0.02), damping=0.0, role="mechanism", label="Strut extension (0 = hatch closed, + = extending)", robot_interactive=False)
    rod_len = stroke + 0.08
    rod.geoms.append(cyl("strut_rod_geom", (d0 - 0.006 - rod_len / 2, 0, 0), r_rod, rod_len / 2, mat, (1, 0, 0), 7800, False, True, FULL_ONLY, "closer", "Piston rod"))
    rod.geoms.append(cyl("strut_rod_eye", (d0, 0, 0), 0.009, 0.006, mat, (0, 0, 1), 7800, False, True, FULL_ONLY, "closer", "Rod eye"))
    model.add_body(rod)
    model.equalities.append(Equality("connect", "strut_rod_connect", rod.name, leaf_body.name, (0, 0, 0, 0, 0), (d0, 0, 0), FULL_ONLY, "Gas strut rod pinned to the hatch bracket", solref=(0.002, 1.0), solimp=(0.99, 0.999, 0.001, 0.5, 2.0)))
    model.contact_excludes += [(cyl_b.name, rod.name), (rod.name, leaf_body.name), (cyl_b.name, leaf_body.name)]
    model.meta.setdefault("clearance_allow", []).extend([["strut_rod_geom", "strut_tube", "rod slides inside the tube"], ["strut_rod_geom", "strut_rod_eye", "rod eye"]])
    # ---- physics: gas spring pushes the rod OUT (assists lifting); force decreases with extension (progression)
    strut = pc["strut"]
    F1, F2 = float(strut["force_extended_N"]), float(strut["force_compressed_N"])
    kN = (F2 - F1) / max(stroke, 0.05)                                 # N/m: force falls as the rod extends
    s_ext = ds - d0
    dsdth = np.gradient(s_ext, thetas)
    Fs = F2 - kN * s_ext                                               # push (positive = extending)
    tau_full = -Fs * dsdth                                             # negative = assists opening (door convention: tau resists opening when positive)
    rod.joint.stiffness = kN
    rod.joint.springref = F2 / kN                                      # torque = -k (q - ref) = F2 - k q  (pushing out)
    c = float(strut["damping_Ns_per_m"])
    rod.joint.damping = c
    rod.joint.damping_closing = c
    rod.joint.damping_opening = c
    sel = thetas <= min(math.radians(90), max_open) + 1e-9
    tr0, kr, e_max, e_rms = CK.fit_linear(thetas[sel], tau_full[sel])
    b_r = c * float(np.mean(dsdth[sel] ** 2))
    reduced = {"spring_preload_Nm": float(tr0), "spring_stiffness_Nm_per_rad": float(kr), "damping_closing": float(b_r), "damping_latch": float(b_r), "damping_opening": float(b_r),
               "backcheck_damping": 0.0, "backcheck_angle_rad": None, "latch_angle_rad": 0.0, "delayed_action_damping": 0.0, "delayed_angle_rad": None, "hold_open_rad": None, "hold_torque_Nm": 0.0, "hold_open_kind": "none",
               "fit": {"torque_max_rel_err": float(e_max), "torque_rms_rel_err": float(e_rms), "note": "reduced = linear fit of -F(s)*ds/dtheta (negative torque = lift assist)"}}
    pc["reduced"] = reduced
    pc["spring_preload_Nm"], pc["spring_stiffness_Nm_per_rad"] = float(tr0), float(kr)
    j = leaf_body.joint
    j.stiffness = abs(kr) if abs(kr) > 1e-9 else 0.0
    j.springref = (-tr0 / kr) if abs(kr) > 1e-9 else 0.0
    j.damping = float(phys.get("hinge", {}).get("air_damping_Nms_per_rad", 0.1)) + b_r + 0.5
    j.overrides["full"] = {"stiffness": 0.0, "springref": 0.0, "damping": float(phys.get("hinge", {}).get("air_damping_Nms_per_rad", 0.1)) + 0.5,
                           "notes": (j.notes + " " if j.notes else "") + "full tier: gas strut force transmitted through the strut linkage"}
    end_law = CK.law_from_windows(rod.joint.name, ("full",), c, c, c, c, 0.0, 0.0, None, None, 0.0, None, 0.0, 0.0, "none", "rod_m", float(s_ext.max()))
    end_law["end_damping"] = float(strut["end_damping_Ns_per_m"])
    end_law["end_zone_m"] = float(strut["end_zone_m"])
    pc.setdefault("laws", []).append(end_law)
    geometry = {"mechanism": "telescoping_gas_strut", "length_m": float(tube_len + 0.02), "stroke_m": stroke, "closed_pivot_distance_m": d0, "extension_at_max_open_m": float(s_ext.max()),
                "tip_from_hinge_m": float(best["tip_y"]), "bracket_from_hinge_m": float(best["base_y"]), "bracket_below_leaf_m": float(best["base_z"]), "moment_arm_closed_m": float(best["arm"]),
                "struts_modelled": 1, "source": "gas spring 150-400 N (Stabilus Lift-O-Mat class), one strut modelled carrying the total force"}
    table = [[round(math.degrees(th), 1), round(float(s), 4), round(float(dd), 4), round(float(tt), 2)] for th, s, dd, tt in zip(thetas, s_ext, dsdth, tau_full)]
    pc.setdefault("mechanism_params", {}).update({"joint_base": cyl_b.joint.name, "joint_slide": rod.joint.name, "geometry": geometry, "rod": {"force_compressed_N": F2, "force_extended_N": F1, "rate_N_per_m": kN, "damping_Ns_per_m": c},
                                                 "ratio_table": {"columns": ["door_deg", "rod_extension_m", "dext_ddoor_m_per_rad", "door_torque_Nm"], "rows": table}})
    model.linkages.append({"name": "gas_strut", "type": "telescoping",
                           "base": {"body": cyl_b.name, "joint": cyl_b.joint.name, "parent": "world", "pos": [float(x) for x in B_w]},
                           "slide": {"body": rod.name, "joint": rod.joint.name, "axis_local": [1.0, 0.0, 0.0], "offset": float(d0)},
                           "anchor": {"body": leaf_body.name, "pos": [float(x_s), float(A_l[0]), float(A_l[1])]}, "equality": "strut_rod_connect", "axis": [1.0, 0.0, 0.0]})


def add_sliding_operator(model: Model, world: Body, leaf_body: Body, spec: dict, phys: dict, Wo: float, Ho: float, jamb_t: float, y_leaf: float, travel: float, s_open: float, name: str):
    """Automatic sliding door drive: header unit (motor + gearbox) at one end of the track header, toothed belt run
    with pulleys, belt clamp on the leaf's carriage.  The drive itself is the position actuator on the leaf slide
    joint (force-limited to the operator's rated force); this adds the geometry and the physics description."""
    pc = phys["closer"]
    if pc.get("mechanism") != "sliding_operator_belt":
        return
    mat = mat_from_material(model, "aluminum_dark", "mat_operator")
    belt = mat_from_material(model, "rubber", "mat_belt")
    z_h = Ho + jamb_t + 0.04
    x_motor = -s_open * (Wo / 2 + 0.25)
    world.geoms.append(box(f"{name}_drive_unit", (x_motor, y_leaf, z_h + 0.03), (0.12, 0.05, 0.05), mat, 1500, False, True, FULL_ONLY, "closer", "Operator drive unit (motor + gearbox)"))
    for sx in (-1, 1):
        world.geoms.append(cyl(f"{name}_pulley_{'r' if sx > 0 else 'l'}", (sx * (Wo / 2 + travel / 2 + 0.05), y_leaf, z_h - 0.02), 0.03, 0.006, mat, (0, 1, 0), 2700, False, True, FULL_ONLY, "closer", "Belt pulley"))
    world.geoms.append(box(f"{name}_belt", (0.0, y_leaf, z_h - 0.02), (Wo / 2 + travel / 2 + 0.05, 0.0015, 0.006), belt, 1200, False, True, FULL_ONLY, "closer", "Toothed belt"))
    clamp_x = 0.0
    leaf_body.geoms.append(box(f"{name}_belt_clamp", (clamp_x, y_leaf - float(leaf_body.pos[1]) if abs(y_leaf) > 1e-9 else 0.0, z_h - 0.02 - float(leaf_body.pos[2])), (0.03, 0.006, 0.012), mat, 2700, False, True, FULL_ONLY, "closer", "Belt clamp (carriage)"))
    model.meta.setdefault("clearance_allow", []).extend([[f"{name}_belt_clamp", f"{name}_belt", "clamp grips the belt"], [f"{name}_belt_clamp", "track_header*", "clamp inside the header"], [f"{name}_belt", "track_header*", "belt inside the header"]])
    pc.setdefault("mechanism_params", {}).update({"joint": leaf_body.joint.name if leaf_body.joint else None, "geometry": {"drive_unit_at_x_m": x_motor, "belt_run_m": float(Wo + travel + 0.1), "source": "ANSI/BHMA A156.10 sliding operator: belt drive, breakout / manual push when unpowered"}})

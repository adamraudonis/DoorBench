"""Self-closing / power-operating mechanisms: physical model, tunable parameters, valve law, loop solving, QA.

This module has no geometry in it (that lives in ``doorbench/geometry/closers.py``); it is imported by the physics
derivation, the build (calibration), the clearance gate and the hardware review (loop solving), ``DoorEnv`` and the QA
(the hydraulic valve law).  Everything the simulator cannot express natively is described here once and applied
identically everywhere.

Mechanisms (``spec.json -> physics.closer.mechanism``; full documentation in docs/PHYSICS.md):

  rack_pinion_regular_arm    surface closer, pull-side mount: body on the leaf, torsion spring + hydraulics act at the
                             PINION on the body; main arm on the pinion, forearm to the shoe on the frame face; the
                             door torque emerges through the two-bar linkage's varying mechanical advantage
  rack_pinion_parallel_arm   surface closer, push-side mount: same, shoe on the frame soffit
  rack_pinion_frame_arm      concealed overhead closer: body in the head, pinion on the frame, arms to a shoe on the leaf
  swing_operator_arm         automatic swing operator: header on the frame, motor + closing spring on the pinion, arm
                             linkage to the leaf
  floor_spring               torque at the bottom pivot (the spindle IS the door pivot)
  spring_hinge               torque at the hinges
  telescoping_*              gas strut / pneumatic screen closer / gate spring / hydraulic gate closer: cylinder hinged
                             at a bracket, rod on a slide joint (spring + damper along the axis), rod tip pinned to
                             the leaf bracket
  sliding_operator_belt      automatic sliding door: belt / carriage drive on the leaf slide joint

The hydraulics (sweep / latch / backcheck / delayed-action valves, hold-open detents) are angle- and
direction-dependent damping, which MuJoCo joints cannot express natively.  The MJCF carries the check-valve (opening)
damping natively on the mechanism joint; ``passive_rules`` / ``passive_torque`` apply the rest of the law
(DoorEnv installs it as the passive-force callback, the QA adds it to ``qfrc_applied``).  The law is stored in
``physics.closer.laws`` in the coordinates of the joint it acts on (pinion angle for arm closers, rod extension for
telescoping closers, door angle for pivot / hinge closers and for the calibrated reduced tiers).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import hardware as H

# ---------------------------------------------------------------------------
# closing-time windows (s) from 90 deg to 12 deg (sweep) and to 0 (total) per mechanism family.
# ADA 404.2.8.1 / ICC A117.1: >= 5 s from 90 to 12 deg for accessible doors; EN 1154 5.2.5: closing time adjustable
# between 3 s and 20 s (size 1-7, 90 deg).  Screen-door and gate closers are not covered: 2-10 s typical.
# ---------------------------------------------------------------------------
CLOSING_WINDOWS = {
    "ada": (5.0, 20.0), "en1154": (3.0, 20.0), "residential": (2.5, 12.0), "gate": (1.5, 12.0), "spring": (0.6, 10.0),
}
MECHANISM_OF_KIND = {
    "none": "none", "surface_overhead": "rack_pinion_regular_arm", "electromagnetic_hold": "rack_pinion_regular_arm",
    "concealed_overhead": "rack_pinion_frame_arm", "auto_operator_low_energy": "swing_operator_arm", "auto_operator_full": "swing_operator_arm",
    "floor_spring": "floor_spring", "spring_hinge": "spring_hinge", "pneumatic": "telescoping_pneumatic", "gate": "telescoping_gate",
    "gas_strut": "telescoping_gas_strut",
}
ARM_CLOSER_KINDS = ("surface_overhead", "electromagnetic_hold", "concealed_overhead", "auto_operator_low_energy", "auto_operator_full")
TELESCOPING_KINDS = ("pneumatic", "gate", "gas_strut")
PIVOT_KINDS = ("floor_spring", "spring_hinge")
# template dimensions (m): main arm, pinion distance from the hinge axis, body projection is the catalogue body_size[2]
ARM_TEMPLATES = {
    "lcn_4040": {"arm_main_m": 0.305, "pinion_offset_m": 0.235, "fore_range_m": (0.22, 0.36)},          # LCN 4040XP: 12 in main arm, 9-1/4 in template
    "lcn_4040_delayed": {"arm_main_m": 0.305, "pinion_offset_m": 0.235, "fore_range_m": (0.22, 0.36)},
    "magnetic_hold": {"arm_main_m": 0.305, "pinion_offset_m": 0.235, "fore_range_m": (0.22, 0.36)},      # LCN 4040SE
    "norton_1600": {"arm_main_m": 0.28, "pinion_offset_m": 0.20, "fore_range_m": (0.20, 0.34)},          # Norton 1600 / Dorma TS 83 class
    "residential_light": {"arm_main_m": 0.24, "pinion_offset_m": 0.16, "fore_range_m": (0.18, 0.30)},
    "concealed_overhead": {"arm_main_m": 0.20, "pinion_offset_m": 0.12, "fore_range_m": (0.15, 0.30)},   # Dorma RTS 88 side arm
    "auto_low_energy": {"arm_main_m": 0.22, "pinion_offset_m": 0.12, "fore_range_m": (0.15, 0.32)},      # LCN Senior Swing / Norton 6000 push-side arm
    "auto_full_energy": {"arm_main_m": 0.24, "pinion_offset_m": 0.12, "fore_range_m": (0.15, 0.34)},
}
# telescoping closers (m / N): body length, stroke, rod radius, tube radius, mounting reach along the leaf
STRUT_TEMPLATES = {
    "pneumatic_screen": {"length_m": 0.35, "stroke_m": 0.20, "r_tube": 0.016, "r_rod": 0.006, "leaf_reach_m": 0.34, "post_reach_m": 0.05},     # Wright V150 / V920 (heavy duty)
    "gate_spring": {"length_m": 0.30, "stroke_m": 0.30, "r_tube": 0.017, "r_rod": 0.007, "leaf_reach_m": 0.36, "post_reach_m": 0.06},         # coil spring, 15 lbf class
    "gate_hydraulic": {"length_m": 0.36, "stroke_m": 0.15, "r_tube": 0.019, "r_rod": 0.007, "leaf_reach_m": 0.30, "post_reach_m": 0.05},      # Lockey TB / Kant-Slam
    "gas_strut": {"length_m": 0.55, "stroke_m": 0.25, "r_tube": 0.012, "r_rod": 0.006, "leaf_reach_m": 0.0, "post_reach_m": 0.0},           # 500-600 mm gas spring, 250 mm stroke
}


def mechanism_for(spec: dict, cl: H.CloserModel) -> str:
    """Mechanism string for a spec (spec["closer"]["arm"] / ["mechanism"] override the default)."""
    ov = spec.get("closer", {}).get("mechanism")
    if ov:
        return ov
    mech = MECHANISM_OF_KIND.get(cl.kind, "none")
    if cl.kind in ("surface_overhead", "electromagnetic_hold"):
        arm = spec.get("closer", {}).get("arm") or default_arm(spec)
        mech = "rack_pinion_parallel_arm" if arm == "parallel" else "rack_pinion_regular_arm"
    elif cl.kind == "gate":
        mech = "telescoping_gate_hydraulic" if cl.id == "gate_hydraulic" else "telescoping_gate_spring"
    return mech


def default_arm(spec: dict) -> str:
    """Surface closers go on the secure / interior face: pull side -> regular arm, push side -> parallel arm.
    The robot side is `outside` when spec.robot.robot_outside; the door opens toward +y when the robot pushes."""
    v = 1.0 if spec.get("robot", {}).get("is_push", True) else -1.0
    robot_outside = bool(spec.get("robot", {}).get("robot_outside", False))
    inside_y = 1.0 if robot_outside else -1.0          # the inside is the far (+y) side when the robot is outside
    return "regular" if inside_y == v else "parallel"   # inside face == the face the door swings toward (pull side)


# ---------------------------------------------------------------------------
# door-level design (EN 1154 sizing + valve settings) - called from physics.closer_params
# ---------------------------------------------------------------------------
def closer_design(spec: dict, mass_kg: float, friction_Nm: float, W: float, air_damping: float = 0.0) -> dict:
    """Door-level closer parameters: spring curve from the EN 1154 size, valve settings and the door-level damping
    they imply.  The mechanism block (arm geometry, pinion spring, calibrated reduced model, valve law in joint
    coordinates) is added by the geometry builder (geometry/closers.py) which knows the door's dimensions."""
    cs_ = spec["closer"]
    cl = H.CLOSERS[cs_["model"]]
    ctx = spec.get("context", "")
    fam = spec.get("family", "")
    need = 1.3 * friction_Nm
    out = {"model": cl.id, "kind": cl.kind, "mechanism": mechanism_for(spec, cl), "spring_stiffness_Nm_per_rad": 0.0, "spring_preload_Nm": 0.0,
           "damping_closing": 0.0, "damping_opening": 0.0, "en_size": None, "hold_open_rad": cl.hold_open,
           "backcheck_angle_rad": cl.backcheck_angle, "backcheck_damping": cl.backcheck_damping, "latch_boost": 1.0}
    if cl.kind == "none":
        if fam in ("automatic_sliding", "elevator"):
            act = spec["kinematics"].get("actuator", {})
            out.update({"mechanism": "sliding_operator_belt", "drive": {"kind": act.get("kind", "belt_drive"), "max_force_N": float(act.get("max_force_N", 150)),
                                                                      "open_speed_m_s": float(act.get("open_speed_m_s", 0.4)), "close_speed_m_s": float(act.get("close_speed_m_s", 0.25)),
                                                                      "hold_open_s": float(act.get("hold_open_s", 3.0)), "powered": bool(act.get("powered", True)),
                                                                      "breakout": "manual push-through (leaf free when unpowered)", "source": "ANSI/BHMA A156.10 sliding operators"}})
        return out
    settings = dict(cs_.get("settings") or {})
    adj = float(cs_.get("spring_adjust", 1.15))
    accessible = ctx in ("commercial_office", "institutional", "hospital", "fire_egress", "storefront_glass", "school", "hotel", "industrial_utility") or fam in ("automatic_swing", "cold_storage")
    if cl.kind in ARM_CLOSER_KINDS or cl.kind == "floor_spring":
        size = cl.en_size or cs_.get("en_size") or H.closer_size_for(mass_kg, W)
        size = int(max(1, min(7, size)))
        while size < 7 and H.EN1154_SIZES[size].closing_moment_min * adj < need:
            size += 1   # installer picks the next size up when the door is stiff
        cs = H.EN1154_SIZES[size]
        tau0 = max(cs.closing_moment_min * adj, need)                    # closing moment at 0-4 deg
        tau90 = min(cs.opening_moment_max * 0.85, tau0 * 2.8)            # opening moment at 88-92 deg
        k = max((tau90 - tau0) / (math.pi / 2), 0.5)
        if tau0 > cs.closing_moment_min * adj + 1e-9:
            out["note"] = f"spring tension raised to {tau0:.1f} N*m to overcome {friction_Nm:.1f} N*m hinge/seal friction"
        window = "ada" if accessible else "en1154"
        st = {"sweep_time_s": float(settings.get("sweep_time_s", 6.0 if accessible else 4.5)),        # 90 -> 12 deg
              "latch_angle_deg": float(settings.get("latch_angle_deg", 12.0)),
              "latch_speed_factor": float(settings.get("latch_speed_factor", 0.55)),                   # latch valve: damping x factor in the latch zone
              "backcheck_angle_deg": float(settings.get("backcheck_angle_deg", math.degrees(cl.backcheck_angle) if cl.backcheck_angle else 0.0)),
              "backcheck_factor": float(settings.get("backcheck_factor", 4.0 if cl.backcheck_angle else 0.0)),  # backcheck valve: x sweep damping beyond the angle
              "delayed_action_s": float(settings.get("delayed_action_s", 12.0 if cl.delayed_action else 0.0)),  # 90 -> 70 deg dwell
              "delayed_angle_deg": float(settings.get("delayed_angle_deg", 70.0)),
              "hold_open_deg": float(settings.get("hold_open_deg", math.degrees(cl.hold_open) if cl.hold_open else 0.0)),
              "hold_open_kind": settings.get("hold_open_kind", "electromagnetic" if cl.kind == "electromagnetic_hold" else ("mechanical" if cl.hold_open else "none")),
              "closing_time_window_s": list(CLOSING_WINDOWS[window])}
        out.update({"en_size": size, "spring_adjust": adj, "spring_stiffness_Nm_per_rad": k, "spring_preload_Nm": tau0, "opening_moment_90_Nm": tau90,
                    "settings": st, "damping_opening": cl.opening_damping,
                    "formula": "door torque tau(theta) = tau0 + k*theta with tau0 = spring_adjust x EN 1154 closing moment(size), tau(90) = 0.85 x EN 1154 max opening moment; the full tier realises it through the mechanism (see mechanism_params)",
                    "source": "EN 1154:1996 Table 1; LCN 4040XP / Norton 1600 / Dorma TS 83 installation templates; ADA 404.2.8.1 closing time"})
        if cl.kind in ("auto_operator_low_energy", "auto_operator_full"):
            act = spec["kinematics"].get("actuator", {})
            out["motor"] = {"max_torque_Nm": float(act.get("max_torque_Nm", 60.0)), "open_time_s": float(act.get("open_time_s", 4.0)), "hold_open_s": float(act.get("hold_open_s", 4.0)),
                            "powered": bool(act.get("powered", True)), "push_and_go": bool(act.get("push_and_go", False)), "sensor": act.get("sensor"),
                            "power_loss": "closing spring returns the door (closer function of the operator)",
                            "source": "ANSI/BHMA A156.19 (low energy: <= 15 lbf stall, >= 3 s to open 80 deg) / A156.10 (full energy)"}
            if cl.kind == "auto_operator_low_energy":
                out["motor"]["open_time_s"] = max(out["motor"]["open_time_s"], 3.0)
    elif cl.kind == "spring_hinge":
        n = spec["hinge"]["count"]
        k_each = float(cs_.get("spring_hinge_k", 2.2))
        need = max(need, latch_torque_need(spec))          # tension is set until the door latches
        st = {"sweep_time_s": 0.0, "latch_angle_deg": 0.0, "latch_speed_factor": 1.0, "backcheck_angle_deg": 0.0, "backcheck_factor": 0.0, "delayed_action_s": 0.0, "delayed_angle_deg": 0.0,
              "hold_open_deg": 0.0, "hold_open_kind": "none", "closing_time_window_s": list(CLOSING_WINDOWS["spring"]), "tension_turns": int(cs_.get("tension_turns", 2))}
        out.update({"spring_stiffness_Nm_per_rad": k_each * n, "spring_preload_Nm": max(0.9 * n, need), "settings": st, "damping_closing": cl.closing_damping, "damping_opening": cl.opening_damping,
                    "formula": "n_hinges x (0.9 N*m + k_each * theta); k_each = spring_hinge_k (tension pin turns)", "source": "Bommer 4310 adjustable spring hinge (2-3 per door)"})
    elif cl.kind == "pneumatic":
        need = max(need, latch_torque_need(spec))
        F0 = float(cs_.get("spring_force_N", 45.0 * adj))
        st = {"sweep_time_s": float(settings.get("sweep_time_s", 3.0)), "latch_angle_deg": float(settings.get("latch_angle_deg", 15.0)), "latch_speed_factor": float(settings.get("latch_speed_factor", 0.4)),
              "backcheck_angle_deg": 0.0, "backcheck_factor": 0.0, "delayed_action_s": 0.0, "delayed_angle_deg": 0.0,
              "hold_open_deg": float(settings.get("hold_open_deg", 90.0)), "hold_open_kind": "washer", "closing_time_window_s": list(CLOSING_WINDOWS["residential"])}
        out.update({"spring_preload_Nm": max(3.0 * adj, need), "spring_stiffness_Nm_per_rad": 3.0, "settings": st, "damping_opening": cl.opening_damping,
                    "strut": {"spring_force_closed_N": F0, "spring_rate_N_per_m": float(cs_.get("spring_rate_N_per_m", 250.0)), "damping_close_Ns_per_m": float(cs_.get("damping_close_Ns_per_m", 900.0)),
                              "damping_open_Ns_per_m": float(cs_.get("damping_open_Ns_per_m", 60.0)), "hold_open_washer": True},
                    "formula": "rod force F = F0 + k*s (tension spring in the tube), air-cushion damping c*ds/dt through the adjustable latch valve", "source": "Wright Products V150 pneumatic screen-door closer"})
    elif cl.kind == "gate":
        hydraulic = cl.id == "gate_hydraulic"
        need = max(need, latch_torque_need(spec))
        F0 = float(cs_.get("spring_force_N", (70.0 if hydraulic else 40.0) * adj))
        st = {"sweep_time_s": float(settings.get("sweep_time_s", 3.0 if hydraulic else 0.0)), "latch_angle_deg": float(settings.get("latch_angle_deg", 12.0 if hydraulic else 0.0)),
              "latch_speed_factor": float(settings.get("latch_speed_factor", 0.5 if hydraulic else 1.0)), "backcheck_angle_deg": 0.0, "backcheck_factor": 0.0, "delayed_action_s": 0.0, "delayed_angle_deg": 0.0,
              "hold_open_deg": 0.0, "hold_open_kind": "none", "closing_time_window_s": list(CLOSING_WINDOWS["gate" if hydraulic else "spring"])}
        out.update({"spring_preload_Nm": max(4.0 * adj, need), "spring_stiffness_Nm_per_rad": 5.0, "settings": st, "damping_opening": cl.opening_damping,
                    "strut": {"spring_force_closed_N": F0, "spring_rate_N_per_m": float(cs_.get("spring_rate_N_per_m", 320.0 if hydraulic else 260.0)),
                              "damping_close_Ns_per_m": float(cs_.get("damping_close_Ns_per_m", 1400.0 if hydraulic else 8.0)), "damping_open_Ns_per_m": float(cs_.get("damping_open_Ns_per_m", 120.0 if hydraulic else 8.0))},
                    "formula": "rod force F = F0 + k*s (coil / gas spring), hydraulic damping c*ds/dt (hydraulic closer only)",
                    "source": "Lockey TB / D&D Kant-Slam hydraulic gate closer (pool code: self-closing + self-latching)" if hydraulic else "National Hardware V19 gate spring 6-15 lbf"})
    elif cl.kind == "gas_strut":
        F = float(cs_.get("gas_force_N", 250.0))
        prog = float(cs_.get("gas_progression", 1.35))
        st = {"sweep_time_s": 0.0, "latch_angle_deg": 0.0, "latch_speed_factor": 1.0, "backcheck_angle_deg": 0.0, "backcheck_factor": 0.0, "delayed_action_s": 0.0, "delayed_angle_deg": 0.0,
              "hold_open_deg": 0.0, "hold_open_kind": "none", "closing_time_window_s": [0.0, 0.0]}
        out.update({"spring_stiffness_Nm_per_rad": -F * 0.25 * 0.3, "spring_preload_Nm": -F * 0.25, "settings": st, "damping_closing": 30.0, "damping_opening": 30.0,
                    "strut": {"force_extended_N": F, "progression": prog, "force_compressed_N": F * prog, "damping_Ns_per_m": float(cs_.get("gas_damping_Ns_per_m", 400.0)),
                              "end_damping_Ns_per_m": float(cs_.get("gas_end_damping_Ns_per_m", 2500.0)), "end_zone_m": 0.03},
                    "formula": "lift assist: F(s) = F1 + (F2 - F1)*(1 - s/stroke), F2 = progression x F1; viscous damping + end-of-stroke cushion", "source": "Gas spring 150-400 N, progression 1.2-1.6 (Stabilus Lift-O-Mat class)"})
    # door-level damping targets shared by all kinds (reduced-model defaults; the builder refines them from the mechanism)
    if cl.kind != "spring_hinge" and cl.kind != "gas_strut":
        b_sweep = sweep_damping_for_time(mass_kg, W, out["spring_preload_Nm"], out["spring_stiffness_Nm_per_rad"], friction_Nm, out["settings"]["sweep_time_s"], air_damping) if out["settings"]["sweep_time_s"] > 0 else 0.0
        out["damping_closing"] = b_sweep
        out["damping_latch"] = b_sweep * out["settings"]["latch_speed_factor"]
        out["backcheck_damping"] = b_sweep * out["settings"]["backcheck_factor"]
        out["backcheck_angle_rad"] = math.radians(out["settings"]["backcheck_angle_deg"]) if out["settings"]["backcheck_angle_deg"] > 0 else None
        out["latch_angle_rad"] = math.radians(out["settings"]["latch_angle_deg"])
        out["hold_open_rad"] = math.radians(out["settings"]["hold_open_deg"]) if out["settings"]["hold_open_deg"] > 0 else None
        out["delayed_action_damping"] = delayed_damping_for_time(mass_kg, W, out["spring_preload_Nm"], out["spring_stiffness_Nm_per_rad"], friction_Nm, out["settings"]["delayed_action_s"], out["settings"]["delayed_angle_deg"]) if out["settings"]["delayed_action_s"] > 0 else 0.0
    else:
        out["damping_latch"] = out["damping_closing"]
        out["backcheck_damping"] = 0.0
        out["backcheck_angle_rad"] = None
        out["latch_angle_rad"] = 0.0
        out["hold_open_rad"] = None
        out["delayed_action_damping"] = 0.0
    out["closing_time_est_s"] = closing_time(mass_kg, W, out["spring_preload_Nm"], out["spring_stiffness_Nm_per_rad"], out["damping_closing"], friction_Nm, out.get("damping_latch", out["damping_closing"]), out.get("latch_angle_rad", 0.0), air_damping)[0]
    return out


def latch_torque_need(spec: dict) -> float:
    """Door torque needed to push the spring latch over its strike lip while closing: bolt spring force (preload +
    rate x half throw) times the lip slope (~0.7) plus friction, at the latch edge lever arm; 1.3 safety."""
    lt = H.LATCHES[spec["latch"]["model"]]
    if lt.throw <= 0 or lt.kind not in ("tubular_latch", "deadlatch", "mortise_latch", "rim_latch", "hook", "vertical_rods"):
        return 0.0
    F_bolt = lt.spring_preload + lt.spring_rate * lt.throw * 0.5
    arm = max(0.3, spec["leaf"]["width"] - (lt.backset or 0.06))
    return 1.3 * F_bolt * (0.7 + 0.3) * arm


# ---------------------------------------------------------------------------
# door-level closing dynamics (reduced model) used for valve sizing and for the calibration checks
# ---------------------------------------------------------------------------
def closing_time(m, W, tau0, k, b_sweep, friction=0.0, b_latch=None, latch_angle=0.0, b_air=0.0, theta0=math.pi / 2, tau_fn=None, b_fn=None):
    """(t_total, t_sweep_to_12deg, omega_at_closed) of a leaf released at theta0 under the closer law.
    tau_fn(theta) / b_fn(theta) override the linear spring / windowed damping (used to compare full vs reduced)."""
    I = m * W * W / 3
    b_latch = b_sweep if b_latch is None else b_latch
    dt, th, w, t = 0.001, float(theta0), 0.0, 0.0
    t12 = None
    while th > 0.002 and t < 60:
        tau_s = tau_fn(th) if tau_fn else (tau0 + k * th)
        b = (b_fn(th) if b_fn else (b_latch if th < latch_angle else b_sweep)) + b_air
        tau = -tau_s - b * w
        if abs(w) < 1e-6:
            tau = tau - math.copysign(min(friction, abs(tau)), tau)
        else:
            tau = tau - math.copysign(friction, w)
        w += tau / I * dt
        th += w * dt
        t += dt
        if t12 is None and th <= math.radians(12):
            t12 = t
    return (t if th <= 0.002 else 60.0), (t12 if t12 is not None else 60.0), abs(w)


def sweep_damping_for_time(m, W, tau0, k, friction, t_target, b_air=0.0):
    """Sweep-valve damping (N*m*s/rad) so that 90 -> 12 deg takes t_target seconds (bisection on the reduced model)."""
    lo, hi = 0.0, 4000.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        _, t12, _ = closing_time(m, W, tau0, k, mid, friction, mid, 0.0, b_air)
        if t12 < t_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def delayed_damping_for_time(m, W, tau0, k, friction, t_delay, angle_deg):
    """Delayed-action valve damping so that the leaf takes t_delay seconds from 90 deg to angle_deg."""
    I = m * W * W / 3
    lo, hi = 0.0, 40000.0
    a = math.radians(angle_deg)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        th, w, t = math.pi / 2, 0.0, 0.0
        while th > a and t < 120:
            tau = -(tau0 + k * th) - mid * w - (math.copysign(friction, w) if abs(w) > 1e-6 else 0.0)
            w += tau / I * 0.002
            th += w * 0.002
            t += 0.002
        if t < t_delay:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# planar two-bar loop kinematics (shared by the builder, the loop solver and the viewer's formula)
# ---------------------------------------------------------------------------
def twobar_elbow(p, s, L1, L2, sign):
    """Elbow position for pinion p and shoe s (2D numpy), L1 = pinion->elbow, L2 = elbow->shoe.
    elbow = p + a*ex + sign*h*ey with ex = unit(s - p), ey = rot90(ex) (= axis x ex in 3D).  Returns (elbow, d)."""
    d = float(np.linalg.norm(s - p))
    dc = min(max(d, abs(L1 - L2) + 1e-9), L1 + L2 - 1e-9)
    ex = (s - p) / max(d, 1e-12)
    ey = np.array([-ex[1], ex[0]])
    a = (L1 * L1 - L2 * L2 + dc * dc) / (2 * dc)
    h = math.sqrt(max(L1 * L1 - a * a, 0.0))
    return p + a * ex + sign * h * ey, d


def twobar_sweep(pinion_local, shoe_local, hinge, L1, L2, sign, thetas, rot_sign, pinion_on_leaf=True):
    """Pinion angle q(theta) (rad, in the pinion's parent frame, relative to theta=0) and its derivative for a door
    rotated by rot_sign*theta about `hinge` (2D points in the leaf-closed frame).  Also returns d(theta) and the
    elbow trajectory.  pinion_on_leaf=False: the pinion is on the frame and the shoe rides on the leaf."""
    phis, ds, elbows = [], [], []
    for th in thetas:
        c, s_ = math.cos(rot_sign * th), math.sin(rot_sign * th)
        R = np.array([[c, -s_], [s_, c]])
        if pinion_on_leaf:
            p = hinge + R @ (pinion_local - hinge)
            s = shoe_local
        else:
            p = pinion_local
            s = hinge + R @ (shoe_local - hinge)
        e, d = twobar_elbow(p, s, L1, L2, sign)
        phi = math.atan2(e[1] - p[1], e[0] - p[0])
        if pinion_on_leaf:
            phi -= rot_sign * th
        phis.append(phi)
        ds.append(d)
        elbows.append(e)
    phis = np.unwrap(np.array(phis))
    q = phis - phis[0]
    dq = np.gradient(q, thetas) if len(thetas) > 1 else np.zeros_like(q)
    return q, dq, np.array(ds), np.array(elbows)


def fit_linear(theta, tau, weights=None, minimax=True):
    """Line tau0 + k*theta fitted to a torque curve: least squares, then (minimax=True) iteratively re-weighted
    toward the Chebyshev (minimum maximum relative error) line.  Returns (tau0, k, max_rel_err, rms_rel_err)."""
    theta = np.asarray(theta, float)
    tau = np.asarray(tau, float)
    scale = np.maximum(np.abs(tau), 0.05 * max(float(np.max(np.abs(tau))), 1e-6))
    w = (np.ones_like(theta) if weights is None else np.asarray(weights, float)) / scale
    A0 = np.column_stack([np.ones_like(theta), theta])
    c, *_ = np.linalg.lstsq(A0 * w[:, None], tau * w, rcond=None)
    if minimax:
        for _ in range(60):
            res = np.abs((c[0] + c[1] * theta - tau) / scale)
            w2 = w * ((res / max(res.max(), 1e-12)) ** 2 + 1e-3)
            c_new, *_ = np.linalg.lstsq(A0 * w2[:, None], tau * w2, rcond=None)
            if np.max(np.abs(c_new - c)) < 1e-7 * max(1.0, float(np.abs(c).max())):
                c = c_new
                break
            c = 0.5 * c + 0.5 * c_new
    fit = c[0] + c[1] * theta
    rel = np.abs(fit - tau) / scale
    return float(c[0]), float(c[1]), float(rel.max()), float(math.sqrt(float(np.mean(rel ** 2))))


# ---------------------------------------------------------------------------
# valve law (direction / angle dependent damping, detents, delayed action, hold-open)
# ---------------------------------------------------------------------------
def law_from_windows(joint: str, tiers, native_damping: float, b_sweep: float, b_latch: float, b_check: float, b_backcheck: float,
                     q_latch: float, q_backcheck: Optional[float], q_delay: Optional[float], b_delay: float,
                     q_hold: Optional[float], hold_torque: float, hold_width: float, hold_kind: str, unit: str, q_max: Optional[float] = None) -> dict:
    """One entry of physics.closer.laws: everything in the coordinates of `joint` (unit = 'pinion_rad', 'door_rad' or 'rod_m')."""
    return {"joint": joint, "tiers": sorted(tiers), "unit": unit, "damping_native": float(native_damping), "damping_sweep": float(b_sweep), "damping_latch": float(b_latch),
            "damping_check": float(b_check), "damping_backcheck": float(b_backcheck), "q_latch": float(q_latch), "q_backcheck": None if q_backcheck is None else float(q_backcheck),
            "q_delay": None if q_delay is None else float(q_delay), "damping_delay": float(b_delay), "q_hold": None if q_hold is None else float(q_hold),
            "hold_torque": float(hold_torque), "hold_width": float(hold_width), "hold_kind": hold_kind, "q_max": None if q_max is None else float(q_max),
            "note": "closing (v<0): sweep valve, latch valve below q_latch, delayed-action valve above q_delay; opening (v>0): check valve, backcheck valve above q_backcheck; "
                    "hold-open: detent well of +-hold_width around q_hold with holding torque hold_torque (electromagnetic: released by the environment).  "
                    "The simulator carries damping_native on the joint; the extra torque -(b_target - damping_native)*v is added by DoorEnv / QA."}


def law_torque(law: dict, q: float, v: float, released: bool = False) -> float:
    """Extra generalized force on the law's joint for state (q, v) (native damping already applied by the simulator)."""
    b0 = law["damping_native"]
    if v < 0:
        b = law["damping_sweep"]
        if q < law["q_latch"]:
            b = law["damping_latch"]
        if law.get("q_delay") is not None and q > law["q_delay"] and law["damping_delay"] > 0:
            b = law["damping_delay"]
    else:
        b = law["damping_check"]
        if law.get("q_backcheck") is not None and q > law["q_backcheck"]:
            b = law["damping_check"] + law["damping_backcheck"]
    tau = -(b - b0) * v
    qh = law.get("q_hold")
    if qh is not None and not released and law["hold_torque"] > 0:
        w = law["hold_width"]
        dq = q - qh
        if abs(dq) < w:
            # detent well: linear restoring torque saturating at hold_torque at the well edge, plus a little damping
            tau += -law["hold_torque"] * (dq / w) - 0.05 * law["hold_torque"] / max(w, 1e-6) * v * 0.1
    return tau


def passive_rules(m, phys_closer: dict | None, mujoco=None) -> list:
    """Resolve the laws of a physics.closer block against a compiled MuJoCo model: [(law, dof, qposadr)] for the laws
    whose joint exists (full-tier laws act on the mechanism joints, reduced-tier laws on the door joint; a tier's model
    contains exactly one of the two for each leaf)."""
    import mujoco as mj
    present = []
    for law in (phys_closer or {}).get("laws", []) or []:
        jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, law["joint"])
        if jid >= 0:
            present.append((law, int(m.jnt_dofadr[jid]), int(m.jnt_qposadr[jid])))
    # a full-tier model contains the mechanism joints: use the full-tier laws (and any all-tier law); a reduced model
    # only contains the door joint: use the reduced laws.  Never both (that would double the damping).
    full = [r for r in present if "full" in r[0]["tiers"]]
    if full:
        return full
    return [r for r in present if "full" not in r[0]["tiers"]]


def passive_torque(rules: list, m, d, out=None, released: bool = False):
    """Sum the law torques into `out` (defaults to a fresh array of size nv) and return it."""
    if out is None:
        out = np.zeros(m.nv)
    for law, dof, qadr in rules:
        out[dof] += law_torque(law, float(d.qpos[qadr]), float(d.qvel[dof]), released)
    return out


# ---------------------------------------------------------------------------
# loop solving on a compiled MuJoCo model (clearance gate, renders, QA): set the loop joints for the current leaves
# ---------------------------------------------------------------------------
def _signed_angle(a, b, axis):
    a = a - axis * np.dot(a, axis)
    b = b - axis * np.dot(b, axis)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    a, b = a / na, b / nb
    return math.atan2(float(np.dot(np.cross(a, b), axis)), float(np.dot(a, b)))


def solve_linkages(m, d, linkages: list, mujoco=None) -> dict:
    """Solve every closed loop described by model.json["linkages"] for the current qpos of the other joints and write
    the loop joints (pinion + elbow / hinge + slide) into d.qpos.  Returns {equality name: violation (m)} after solving."""
    import mujoco as mj
    if not linkages:
        return {}
    jid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)
    bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)

    def world_point(body, pos):
        b = bid(body) if body not in ("world", None) else 0
        if b <= 0:
            return np.asarray(pos, float)
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(pos, float)

    out = {}
    for lk in linkages:
        if lk.get("type") == "two_bar":
            jp, je = jid(lk["pinion"]["joint"]), jid(lk["elbow"]["joint"])
            if jp < 0 or je < 0:
                continue
            d.qpos[m.jnt_qposadr[jp]] = 0.0
            d.qpos[m.jnt_qposadr[je]] = 0.0
            mj.mj_kinematics(m, d)
            jaxis = d.xaxis[jp].copy()                         # pinion joint axis (may be -plane normal so that q > 0 opens)
            axis = np.asarray(lk.get("axis", [0, 0, 1]), float)  # plane normal used by the elbow formula (elbow_sign refers to it)
            axis = axis / np.linalg.norm(axis)
            P = d.xanchor[jp].copy()
            S = world_point(lk["anchor"]["body"], lk["anchor"]["pos"])
            bp, be = bid(lk["pinion"]["body"]), bid(lk["elbow"]["body"])
            arm0 = d.xmat[bp].reshape(3, 3) @ np.asarray(lk.get("arm_dir0", [1, 0, 0]), float)
            fore0 = d.xmat[be].reshape(3, 3) @ np.asarray(lk.get("fore_dir0", [1, 0, 0]), float)
            # planar solve in the plane normal to the axis
            v = S - P
            v = v - axis * np.dot(v, axis)
            dd = float(np.linalg.norm(v))
            L1, L2 = float(lk["L1"]), float(lk["L2"])
            dc = min(max(dd, abs(L1 - L2) + 1e-9), L1 + L2 - 1e-9)
            ex = v / max(dd, 1e-12)
            ey = np.cross(axis, ex)
            a = (L1 * L1 - L2 * L2 + dc * dc) / (2 * dc)
            h = math.sqrt(max(L1 * L1 - a * a, 0.0))
            E = P + a * ex + float(lk.get("elbow_sign", 1)) * h * ey
            q1 = _signed_angle(arm0, E - P, jaxis)
            d.qpos[m.jnt_qposadr[jp]] = q1
            mj.mj_kinematics(m, d)
            fore_now = d.xmat[be].reshape(3, 3) @ np.asarray(lk.get("fore_dir0", [1, 0, 0]), float)
            axis_e = d.xaxis[je].copy()
            q2 = _signed_angle(fore_now, S - E, axis_e)
            d.qpos[m.jnt_qposadr[je]] = q2
            mj.mj_kinematics(m, d)
            tip = d.xpos[be] + d.xmat[be].reshape(3, 3) @ (np.asarray(lk.get("fore_dir0", [1, 0, 0]), float) * L2)
            out[lk["equality"]] = float(np.linalg.norm(tip - S))
        elif lk.get("type") == "telescoping":
            jh, js = jid(lk["base"]["joint"]), jid(lk["slide"]["joint"])
            if jh < 0 or js < 0:
                continue
            d.qpos[m.jnt_qposadr[jh]] = 0.0
            d.qpos[m.jnt_qposadr[js]] = 0.0
            mj.mj_kinematics(m, d)
            axis = d.xaxis[jh].copy()
            B = d.xanchor[jh].copy()
            A = world_point(lk["anchor"]["body"], lk["anchor"]["pos"])
            bs = bid(lk["slide"]["body"])
            dir0 = d.xmat[bs].reshape(3, 3) @ np.asarray(lk["slide"]["axis_local"], float)
            v = A - B
            v = v - axis * np.dot(v, axis)
            q1 = _signed_angle(dir0, v, axis)
            d.qpos[m.jnt_qposadr[jh]] = q1
            d.qpos[m.jnt_qposadr[js]] = float(np.linalg.norm(v)) - float(lk["slide"]["offset"])
            mj.mj_kinematics(m, d)
            # the rod tip sits `offset` along the rod axis from the rod body origin (the body itself moved q along the axis)
            tip = d.xpos[bs] + d.xmat[bs].reshape(3, 3) @ (np.asarray(lk["slide"]["axis_local"], float) * float(lk["slide"]["offset"]))
            out[lk["equality"]] = float(np.linalg.norm(tip - A))
    return out


def loop_joints(linkages: list) -> set:
    """Names of the joints that are driven by loop closure (skipped by independent mechanism sweeps)."""
    out = set()
    for lk in linkages or []:
        if lk.get("type") == "two_bar":
            out.add(lk["pinion"]["joint"])
            out.add(lk["elbow"]["joint"])
        elif lk.get("type") == "telescoping":
            out.add(lk["base"]["joint"])
            out.add(lk["slide"]["joint"])
    return out


def equality_violation(m, d, name: str, mujoco=None) -> float:
    """Norm of the residual of a named equality constraint in the current state (after mj_forward)."""
    import mujoco as mj
    eid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_EQUALITY, name)
    if eid < 0:
        return 0.0
    worst = 0.0
    for i in range(d.nefc):
        if int(d.efc_type[i]) == int(mj.mjtConstraint.mjCNSTR_EQUALITY) and int(d.efc_id[i]) == eid:
            worst = max(worst, abs(float(d.efc_pos[i])))
    return worst


# ---------------------------------------------------------------------------
# QA for self-closing / power-operated doors (called from qa.run_qa)
# ---------------------------------------------------------------------------
def run_closer_qa(m, spec: dict, phys: dict, model_json: dict, meta: dict, primary_joint: int, latch_joint: int) -> tuple:
    """Checks (only the ones applicable to the door are returned):
      closer_linkage      loop-closure residual < 1 mm through the whole sweep (kinematic + dynamic), loop joints move
      closer_returns      (legacy) leaf returns from 60 % of its range
      closer_closes       leaf released at max opening reaches < 3 deg within the mechanism's closing-time window;
                          released at 15 deg it closes too (latch action)
      closer_latches      spring latch re-extends after the closer closes the door from 15 deg
      closer_no_slam      angular speed when reaching 2 deg below the slam threshold
      closer_backcheck    a leaf flung open at 3 rad/s reaches the backcheck zone slower than without backcheck
      closer_hold_open    a leaf placed at the hold-open angle stays there for 5 s
      closer_delayed      90 -> 70 deg takes >= 80 % of the delayed-action setting
      operator_opens      automatic operator drives the leaf to >= 70 deg within 1.5 x open_time
      operator_closes     with the operator unpowered the spring closes the leaf
    Returns (checks, metrics)."""
    import mujoco as mj
    checks, metrics = {}, {}
    pc = phys.get("closer") or {}
    kind = pc.get("kind", "none")
    fam = spec["family"]
    lk = H.LOCKS[spec["lock"]["model"]]
    lt = H.LATCHES[spec["latch"]["model"]]
    pj = primary_joint
    if pj < 0 or int(m.jnt_type[pj]) != int(mj.mjtJoint.mjJNT_HINGE):
        return checks, metrics
    linkages = model_json.get("linkages", []) or []
    d = mj.MjData(m)
    rules = passive_rules(m, pc)
    qa_, dof = m.jnt_qposadr[pj], m.jnt_dofadr[pj]
    # welds (maglocks / delayed egress) hold the leaf shut: the environment releases them before the door can move,
    # so the mechanism tests run with them released
    weld_ids = [e for e in range(m.neq) if int(m.eq_type[e]) == int(mj.mjtEq.mjEQ_WELD)]
    # surface vertical-rod exit devices hold the top rod retracted while the door is open (rod retention) and let it
    # drop into the head strike in the last degrees: emulated here (the bolt would otherwise stab the head edge)
    retention = [j for j in range(m.njnt) if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or "").endswith("top_latch_slide")]
    lo, hi = (m.jnt_range[pj] if m.jnt_limited[pj] else (0.0, math.radians(90)))
    max_open = float(hi)
    both_ways = bool(spec["kinematics"].get("both_ways"))

    def reset(angle):
        mj.mj_resetData(m, d)
        for e in weld_ids:
            d.eq_active[e] = 0
        d.qpos[qa_] = angle
        if latch_joint >= 0:
            d.qpos[m.jnt_qposadr[latch_joint]] = 0.0
        for j in retention:
            d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j][1] if angle > math.radians(5.0) else 0.0
        solve_linkages(m, d, linkages)
        mj.mj_forward(m, d)

    def step(n, hold=None, released=False, ctrl=None):
        for _ in range(n):
            d.qfrc_applied[:] = 0
            if rules:
                passive_torque(rules, m, d, d.qfrc_applied, released)
            if ctrl is not None and m.nu:
                d.ctrl[:] = ctrl
            if hold is not None:
                d.qpos[qa_] = hold
                d.qvel[dof] = 0.0
            if retention and float(d.qpos[qa_]) > math.radians(5.0):
                for j in retention:
                    d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j][1]
                    d.qvel[m.jnt_dofadr[j]] = 0.0
            mj.mj_step(m, d)

    # ---- linkage integrity (kinematic sweep + dynamic trajectory)
    leaf_name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.jnt_bodyid[pj])
    my_links = [lk for lk in linkages if leaf_name in (lk.get("pinion", {}).get("parent"), lk.get("anchor", {}).get("body"))] or linkages
    if linkages:
        worst_k, moved = 0.0, {}
        for k in range(25):
            reset(lo + (hi - lo) * k / 24)
            for lkg in my_links:
                for key in (("pinion", "elbow") if lkg["type"] == "two_bar" else ("base", "slide")):
                    jn = lkg[key]["joint"]
                    j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, jn)
                    if j >= 0:
                        q = float(d.qpos[m.jnt_qposadr[j]])
                        moved[jn] = (min(moved.get(jn, (q, q))[0], q), max(moved.get(jn, (q, q))[1], q))
                worst_k = max(worst_k, equality_violation(m, d, lkg["equality"]))
        worst_d = 0.0
        reset(max_open)
        for _ in range(int(min(pc.get("closing_time_est_s", 5.0) + 3.0, 25.0) / m.opt.timestep)):
            step(1)
            for lkg in linkages:
                worst_d = max(worst_d, equality_violation(m, d, lkg["equality"]))
        travel = {k: v[1] - v[0] for k, v in moved.items()}
        metrics.update({"closer_loop_violation_kinematic_m": worst_k, "closer_loop_violation_dynamic_m": worst_d, "closer_loop_joint_travel": travel})
        checks["closer_linkage"] = bool(worst_k < 1e-3 and worst_d < 1e-3 and all(t > 0.05 for t in travel.values()))
    # ---- self-closing behaviour
    self_closing = kind not in ("none", "gas_strut") and pc.get("spring_preload_Nm", 0) > 0 and not both_ways and not (spec["lock"]["engaged"] and lk.kind in ("chain", "swing_bar_guard", "padlock"))
    if lt.id == "fork_gravity":
        metrics["closer_note"] = "fork latch: gate closes only with the fork lifted; closer return not applicable"
        self_closing = False
    if spec["lock"]["engaged"] and lk.kind in ("mag_lock", "delayed_egress"):
        self_closing = False
    if self_closing and kind not in ("auto_operator_low_energy", "auto_operator_full"):
        st = pc.get("settings", {})
        win = st.get("closing_time_window_s", [0.5, 30.0])
        hold_deg = float(st.get("hold_open_deg", 0.0) or 0.0)
        # legacy check: return from 60 % of the range
        reset(min(math.radians(60.0), 0.8 * max_open))
        step(int(12.0 / m.opt.timestep), released=True)
        metrics["closer_final_angle"] = float(d.qpos[qa_])
        checks["closer_returns"] = bool(abs(d.qpos[qa_]) < math.radians(6.0))
        # full close from max opening (below the hold-open angle if there is one): time, final speed, latch
        start = max_open if not hold_deg or max_open < math.radians(hold_deg) - 0.05 else math.radians(hold_deg) - math.radians(6)
        start = min(start, math.radians(110))
        reset(start)
        t_close, w_end, t = None, None, 0.0
        t_max = min(max(win[1], pc.get("closing_time_est_s", 5.0) * 2.5) + 5.0, 70.0)
        n = int(t_max / m.opt.timestep)
        for i in range(n):
            step(1, released=True)
            q = float(d.qpos[qa_])
            if q < math.radians(3.0):
                t_close = d.time
                w_end = abs(float(d.qvel[dof]))
                break
        metrics.update({"closer_close_time_s": t_close, "closer_close_start_rad": start, "closer_final_speed_rad_s": w_end})
        checks["closer_closes"] = bool(t_close is not None and t_close <= win[1] + 1.0)
        if kind in ARM_CLOSER_KINDS or kind == "floor_spring":
            # the closing window (90 -> 12 deg) is a design requirement for hydraulic closers
            metrics["closer_window_s"] = win
            checks["closer_closes"] = bool(checks["closer_closes"] and t_close is not None and t_close >= 0.6 * win[0])
        slam = float(phys.get("damage", {}).get("slam_velocity_rad_s", 4.0))
        thr = 1.5 if (kind in ARM_CLOSER_KINDS or kind == "floor_spring" or pc.get("mechanism") in ("telescoping_gate_hydraulic",)) else slam
        checks["closer_no_slam"] = bool(w_end is not None and w_end < thr)
        # latch action from 15 deg: closes and the spring latch re-extends
        reset(math.radians(15.0))
        step(int(8.0 / m.opt.timestep), released=True)
        metrics["closer_latch_final_angle"] = float(d.qpos[qa_])
        latched_ok = abs(float(d.qpos[qa_])) < math.radians(3.0)
        if latch_joint >= 0 and lt.kind in ("tubular_latch", "deadlatch", "mortise_latch", "rim_latch", "hook", "gravity_bar") and lt.throw > 0:
            bq = float(d.qpos[m.jnt_qposadr[latch_joint]])
            metrics["closer_latch_bolt_m"] = bq
            latched_ok = latched_ok and bq < 0.006
        checks["closer_latches"] = bool(latched_ok)
        # backcheck: fling open at 3 rad/s from 20 deg; compare the peak speed beyond the backcheck angle with the law switched off
        if st.get("backcheck_factor", 0) > 0 and pc.get("backcheck_angle_rad"):
            bc = float(pc["backcheck_angle_rad"])
            peaks = []
            for use_law in (True, False):
                reset(max(math.radians(20.0), bc - math.radians(25.0)))
                d.qvel[dof] = 3.0
                mj.mj_forward(m, d)
                impact = 0.0
                for _ in range(int(2.0 / m.opt.timestep)):
                    d.qfrc_applied[:] = 0
                    if use_law and rules:
                        passive_torque(rules, m, d, d.qfrc_applied, False)
                    mj.mj_step(m, d)
                    if float(d.qpos[qa_]) >= hi - 0.03:          # reached the stop: impact speed
                        impact = float(d.qvel[dof])
                        break
                    if float(d.qvel[dof]) <= 0:
                        break
                peaks.append(max(impact, 0.0))
            metrics["closer_backcheck_impact_speed_rad_s"] = peaks
            if peaks[1] > 0.3:      # without backcheck the leaf slams the stop: with it the impact must be cut
                checks["closer_backcheck"] = bool(peaks[0] < 0.6 * peaks[1] or peaks[0] < 0.5)
            else:
                metrics["closer_backcheck_note"] = "leaf too heavy / stiff to reach the stop at 3 rad/s; backcheck not testable"
                checks["closer_backcheck"] = bool(peaks[0] <= peaks[1] + 1e-6)
        # hold-open: park the leaf at the hold angle, it must stay
        if hold_deg > 0 and max_open >= math.radians(hold_deg) - 1e-6:
            reset(math.radians(hold_deg))
            step(int(5.0 / m.opt.timestep), released=False)
            metrics["closer_hold_open_final_deg"] = math.degrees(float(d.qpos[qa_]))
            checks["closer_hold_open"] = bool(abs(float(d.qpos[qa_]) - math.radians(hold_deg)) < math.radians(4.0))
            # ... and release it (electromagnetic hold-open: fire alarm) -> it closes
            step(int(15.0 / m.opt.timestep), released=True)
            metrics["closer_hold_released_final_deg"] = math.degrees(float(d.qpos[qa_]))
            checks["closer_hold_open"] = bool(checks["closer_hold_open"] and abs(float(d.qpos[qa_])) < math.radians(6.0))
        # delayed action: 90 -> 70 deg takes at least 80 % of the setting
        if st.get("delayed_action_s", 0) > 0 and max_open >= math.radians(89):
            reset(math.radians(90.0))
            t70 = None
            for _ in range(int((st["delayed_action_s"] * 2 + 5) / m.opt.timestep)):
                step(1, released=True)
                if float(d.qpos[qa_]) < math.radians(st.get("delayed_angle_deg", 70.0)):
                    t70 = d.time
                    break
            metrics["closer_delay_time_s"] = t70
            checks["closer_delayed"] = bool(t70 is not None and t70 >= 0.8 * st["delayed_action_s"])
    # ---- automatic swing operators
    if kind in ("auto_operator_low_energy", "auto_operator_full") and m.nu:
        mot = pc.get("motor", {})
        acts = [a for a in meta.get("actuators", []) if mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a["name"]) >= 0]
        if acts:
            a = acts[0]
            aid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a["name"])
            target = float(a.get("ctrlrange", (0, 1.4))[1])
            reset(0.0)
            ctrl = np.zeros(m.nu)
            ctrl[aid] = target
            t_open = float(mot.get("open_time_s", 4.0))
            reached = None
            for _ in range(int(1.5 * t_open / m.opt.timestep) + 500):
                step(1, ctrl=ctrl)
                if float(d.qpos[qa_]) >= math.radians(70.0) or float(d.qpos[qa_]) >= 0.95 * max_open:
                    reached = d.time
                    break
            metrics["operator_open_time_s"] = reached
            checks["operator_opens"] = bool(reached is not None and reached <= 1.5 * t_open + 1.0)
            # power loss: the servo is unpowered (gains zeroed, not driven to 0) - the closing spring must return the leaf
            gain, bias = m.actuator_gainprm[aid].copy(), m.actuator_biasprm[aid].copy()
            m.actuator_gainprm[aid, :] = 0.0
            m.actuator_biasprm[aid, :] = 0.0
            try:
                step(int(15.0 / m.opt.timestep), ctrl=np.zeros(m.nu), released=True)
            finally:
                m.actuator_gainprm[aid] = gain
                m.actuator_biasprm[aid] = bias
            metrics["operator_power_loss_final_angle"] = float(d.qpos[qa_])
            checks["operator_closes"] = bool(abs(float(d.qpos[qa_])) < math.radians(6.0))
    return checks, metrics

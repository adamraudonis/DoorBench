"""Acceptance for static contact-transfer planning, never a physics score."""

import math


def validate_grasp_reference(reference):
    """Reject acquisition containers whose reset can precede the actual grasp."""
    if reference.get("acquisition"):
        raise ValueError("Provide a grasp-only reference: acquisition initial_joints can be open-hand approach states")
    if len(reference.get("initial_root", ())) != 7 or not reference.get("initial_joints"):
        raise ValueError("A grasp reference must declare a root pose and named joints")
    values = [*reference["initial_root"], *reference["initial_joints"].values()]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Grasp reference contains nonfinite state")


def screen_checks(rows, maximum_arm_step):
    """Require all phases, geometrical contact, and bounded finite static loads.

    A successful result is a sampled FK/static planning screen. It does not
    establish continuous collision clearance between samples or physical
    balance, contact force, execution, grasp retention, or door opening.
    """
    phases = {row["phase"] for row in rows}
    required = {"left_reach", "operator", "both_push", "right_release", "left_push"}
    checks = {"complete_static_sequence": bool(rows) and required <= phases}
    checks["positions"] = bool(rows) and all(
        0 <= row["left_position_error_m"] < .002 and
        (not row["right_pose_constrained"] or 0 <= row["right_position_error_m"] < .002)
        for row in rows)
    checks["left_palm_normal"] = bool(rows) and all(0 <= row["left_normal_error"] < .02 for row in rows)
    checks["collisions"] = bool(rows) and not any(row["collisions"] for row in rows)
    checks["arm_strength"] = bool(rows) and all(0 <= row["arm_static_motor_utilization"] < .8 for row in rows)
    checks["loopback"] = bool(rows) and all(math.isfinite(row["max_loopback_violation_rad"]) and
        row["max_loopback_violation_rad"] < 1e-9 for row in rows)
    checks["left_panel_proximity"] = bool(rows) and all(
        abs(row["left_hand_slab_distance_m"]) < .002 for row in rows if row["phase"] != "left_reach")
    checks["finite_support_plan"] = bool(rows) and all(
        row["stance_status"] in ("solved", "solved inaccurate") and
        row["planned_root_acceleration"] is not None and
        len(row["planned_root_acceleration"]) == 6 and
        all(math.isfinite(value) and abs(value) < .1 for value in row["planned_root_acceleration"])
        for row in rows)
    checks["continuous_arm_waypoints"] = math.isfinite(maximum_arm_step) and 0 <= maximum_arm_step < .2
    checks["target_aperture"] = any(row["phase"] == "left_push" and row["leaf_rad"] >= 1.2 for row in rows)
    release = [row for row in rows if row["phase"] == "right_release"]
    checks["right_release_clearance"] = bool(release) and release[-1]["right_hand_lever_distance_m"] >= .02
    return checks

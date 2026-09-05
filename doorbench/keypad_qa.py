"""QA gate ``keypad_code_works``: on every door with a keypad, a programmatic finger presses the buttons.

Five deterministic sub-checks (a door passes only if all of the applicable ones pass):

  buttons        every button travels its full stroke under a fingertip force and springs back to its stop
  code_opens     the spec's code, pressed button by button, releases the lock and the door then opens
  wrong_holds    a wrong code releases nothing: the same drive leaves the door shut (and the bolt thrown)
  timeout        electronic: a partial entry left standing for `code_timeout_s` is cleared, so finishing it
                 afterwards does not open the lock - while the full code entered fresh still does
                 mechanical: turning the lever on a partial combination clears the chamber, same consequence
  lockout        electronic: `max_attempts` wrong codes freeze the keypad for `lockout_s` - the right code is
                 refused while it is frozen and accepted once it expires

Everything runs on the compiled MJCF through the same ``doorbench.keypad`` state machine the benchmark
environment uses, so the gate tests the code path a robot has to walk, not a shortcut.
"""
from __future__ import annotations

import math

from .keypad import keypad_for


def _q(m, d, j):
    return float(d.qpos[m.jnt_qposadr[j]]) if j >= 0 else 0.0


def _wrong_code(kp) -> str:
    code = kp.cfg.get("code") or ""
    labels = [b["label"] for b in kp.buttons]
    if kp.lock.code_kind == "set":
        spare = next((l for l in labels if l not in code), None)
        if spare is None:
            return code[::-1]
        return code[:-1] + spare
    other = next(l for l in labels if l != code[0])
    return other + code[1:]


def run_keypad_qa(m, spec: dict, meta: dict, phys: dict, push: float, oj: int, pj: int) -> dict:
    """Returns {"ok", "checks", "metrics"}; ok is None when the door has no keypad."""
    import mujoco

    d = mujoco.MjData(m)
    kp = keypad_for(mujoco, m, meta, spec)
    if kp is None or not kp.present:
        return {"ok": None, "note": "no keypad on this door"}
    dt = float(m.opt.timestep)
    is_hinge = pj >= 0 and int(m.jnt_type[pj]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    target = math.radians(min(20.0, 0.5 * (spec["kinematics"].get("max_open_deg") or 90))) if is_hinge else 0.05
    thr_shut = math.radians(2.0) if is_hinge else 0.015
    code = kp.cfg["code"]
    wrong = _wrong_code(kp)
    checks, metrics = {}, {"code": code, "wrong_code": wrong, "release": kp.release_mode, "code_kind": kp.lock.code_kind}

    def restart():
        mujoco.mj_resetData(m, d)
        kp.reset(d)
        mujoco.mj_forward(m, d)

    def stepper(extra=None):
        """One simulation step: keypad release forces, the caller's efforts, then detection."""
        def go():
            kp.apply(d)
            if extra:
                extra()
            mujoco.mj_step(m, d)
            d.qfrc_applied[:] = 0
            kp.step(d)
        return go

    def drive(seconds: float, lever: float = 0.0, operator: float = 0.0, push_leaf: bool = True):
        """Work the released hardware: the outside trim (clutch) / the operator, and lean on the leaf."""
        n = int(seconds / dt)
        rel = kp.clutch if kp.release_mode == "clutch" else oj
        for _ in range(n):
            if lever and rel >= 0:
                d.qfrc_applied[m.jnt_dofadr[rel]] += lever
            if operator and oj >= 0:
                d.qfrc_applied[m.jnt_dofadr[oj]] += operator
            if push_leaf and pj >= 0 and (not is_hinge or _q(m, d, pj) < math.radians(50)):
                d.qfrc_applied[m.jnt_dofadr[pj]] += push
            stepper()()
        return _q(m, d, pj)

    # ---- 1. the buttons are real: full stroke under a fingertip, spring back to the stop
    restart()
    depths, returns = [], []
    for b in kp.buttons:
        peak = 0.0
        for _ in range(int(0.08 / dt)):
            kp.hold(d, b["label"])
            stepper()()
            peak = max(peak, _q(m, d, b["jid"]))
        for _ in range(int(0.10 / dt)):
            stepper()()
        depths.append(peak)
        returns.append(_q(m, d, b["jid"]))
    metrics["button_press_depth_m"] = round(min(depths), 6)
    metrics["button_return_m"] = round(max(returns), 6)
    metrics["button_travel_m"] = kp.travel
    checks["buttons"] = bool(min(depths) >= 0.9 * kp.travel and max(returns) <= 0.15 * kp.travel)

    # ---- 2. the right code releases the lock and the door opens
    restart()
    kp.press_sequence(d, code, step=stepper())
    metrics["unlocked_on_code"] = bool(kp.unlocked)
    metrics["unlock_time_s"] = round(float(d.time), 3)
    if kp.release_mode == "motor_bolt":
        for _ in range(int(2.0 / dt)):
            stepper()()
        metrics["bolt_after_motor_m"] = round(_q(m, d, kp.bolt), 5)
    opened = drive(3.0, lever=4.0 if kp.release_mode != "motor_bolt" else 0.0, operator=4.0 if kp.release_mode == "motor_bolt" else 0.0)
    metrics["opened_after_code_rad"] = round(opened, 4)
    checks["code_opens"] = bool(kp.unlocked and opened > target)

    # ---- 3. a wrong code releases nothing
    restart()
    kp.press_sequence(d, wrong, step=stepper())
    metrics["unlocked_on_wrong"] = bool(kp.unlocked)
    metrics["wrong_attempts"] = kp.lock.wrong_attempts
    if kp.release_mode == "motor_bolt":
        for _ in range(int(2.0 / dt)):
            stepper()()
        metrics["bolt_after_wrong_m"] = round(_q(m, d, kp.bolt), 5)
    shut = drive(3.0, lever=4.0 if kp.release_mode != "motor_bolt" else 0.0, operator=4.0 if kp.release_mode == "motor_bolt" else 0.0)
    metrics["opened_after_wrong_rad"] = round(shut, 4)
    bolt_ok = kp.release_mode != "motor_bolt" or _q(m, d, kp.bolt) < 0.2 * max(kp.bolt_throw, 1e-6)
    checks["wrong_holds"] = bool(not kp.unlocked and shut < thr_shut and bolt_ok)

    # ---- 4. a partial entry does not survive (electronic: the inactivity timeout; mechanical: the lever clears
    #         the chamber), and the full code entered fresh afterwards still works
    restart()
    if kp.lock.code_kind == "sequence":
        half = max(1, len(code) // 2)
        kp.press_sequence(d, code[:half], step=stepper())
        kp.skip_time(d, float(kp.cfg["code_timeout_s"]) + 0.5)
        kp.press_sequence(d, code[half:], step=stepper())
    else:
        kp.press_sequence(d, code[:1], step=stepper())       # one button of the set, then turn the lever: cleared
        kp.press_sequence(d, code[1:], step=stepper())        # the rest + the lever: an incomplete combination
    metrics["unlocked_after_timeout"] = bool(kp.unlocked)
    timeout_blocked = not kp.unlocked
    metrics["wrong_attempts_after_timeout"] = kp.lock.wrong_attempts
    if kp.cfg.get("code_timeout_s"):
        kp.skip_time(d, float(kp.cfg["code_timeout_s"]) + 0.5)   # let the half-entry clear again before retrying
    kp.press_sequence(d, code, step=stepper())
    metrics["unlocked_after_retry"] = bool(kp.unlocked)
    checks["timeout"] = bool(timeout_blocked and kp.unlocked)

    # ---- 5. lockout after repeated wrong codes (electronic keypads only)
    if kp.cfg.get("lockout_s") and kp.cfg.get("max_attempts"):
        restart()
        for _ in range(int(kp.cfg["max_attempts"])):
            kp.press_sequence(d, wrong, step=stepper())
        metrics["locked_out"] = bool(kp.lock.locked_out)
        kp.press_sequence(d, code, step=stepper())
        refused = not kp.unlocked
        metrics["right_code_refused_during_lockout"] = bool(refused)
        kp.skip_time(d, float(kp.cfg["lockout_s"]) + 0.5)
        kp.press_sequence(d, code, step=stepper())
        metrics["unlocked_after_lockout"] = bool(kp.unlocked)
        checks["lockout"] = bool(kp.lock.wrong_attempts >= kp.cfg["max_attempts"] and refused and kp.unlocked)

    # a keypad whose lock is not thrown has nothing to release: the code is still read (checks 1, 4) but the door
    # opens with or without it, so the release-dependent sub-checks do not apply
    if kp.release_mode == "none":
        for k in ("code_opens", "wrong_holds"):
            metrics[f"{k}_note"] = "lock not engaged: nothing to release"
            checks[k] = True

    metrics["events"] = kp.lock.events[-12:]
    return {"ok": all(checks.values()), "checks": checks, "metrics": metrics}

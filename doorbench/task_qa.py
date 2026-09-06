"""``task_achievable``: prove in MuJoCo that every benchmark task on a door can actually be performed.

Why this gate exists
--------------------
Every other QA gate asks whether the door is built right.  None of them asked whether the *task* printed
in ``spec.json["benchmark"]`` is possible.  24 doors shipped with one that was not: a lock the environment
(or the robot's own hardware) is meant to release had been modelled by clamping the primary joint's range
to 2 mm, +-2.9 deg or 3 mm.  A joint range is a static property of MJCF / URDF / USD - nothing at run time
can widen it - so those doors could not open after the release, "closed", "mid travel" and "fully open"
were the same picture, and the only QA they had was ``hold`` / ``locked_holds``, which passed *because*
the door could not move.

What it proves
--------------
Two rules, both measured on the compiled model.

**1. reach** - for every scenario whose success needs the door to move (everything except
``locked_recognize``), the primary joint must be able to sit at the scenario's own pass threshold
(``thresholds.clear_rad`` / ``clear_m`` - the opening the robot has to walk through) *after the declared
release path has been taken*:

  * every releasable holding constraint (``meta["breakable_welds"]`` with ``release`` != "none") is
    deactivated - that is exactly what ``benchmark/env.py`` does on a badge / REX / interlock release, or
    when the modelled part that withdraws the lock has been driven past its release fraction;
  * every latch / lock / mechanism joint is put in its released position, as the clearance gate does;
  * the primary joint is then placed at the threshold and the model is forwarded, and the pose must be
    admissible: the joint's own range must contain it, no equality may still pin the leaf to the world,
    and nothing may be driven into anything (the same 12 mm penetration bar the ``settle`` gate uses -
    this is what catches a leaf whose travel takes it through its own jamb).

**2. travel** - a leaf whose lock CAN be released must keep its whole declared travel in the model:
the primary joint's post-release range has to cover ``kinematics.travel_m`` / ``max_open_deg`` /
``ratchet_deg``.  A leaf nothing can release (a padlock with no key on this side) may be held by a
clamped range - the door really is shut - but then rule 1 forbids it from carrying a task that moves.

**3. force** (doors held by an environment-released lock only) - those doors get no ``actuate_opens`` or
``free_opens`` evidence, because QA has nothing to actuate: their lock opens on a credential.  So for them
the release is applied and the QA push is put on the primary joint for 6 s, and the joint has to pass the
benchmark's own ``opened`` threshold (``thresholds.open_rad`` / ``open_m``).  That is the leg that would
have failed loudly on all 24 doors before this change.
"""
from __future__ import annotations

import json
import math
import os

# scenarios whose success criteria need the leaf to move; `locked_recognize` needs the opposite and is
# covered by the `hold` / `locked_holds` gates instead.
MOVING_SCENARIOS = ("open_and_traverse", "open_then_close", "close_only", "unlock_and_traverse",
                    "hold_open_for_human", "wait_for_human", "knock_and_wait")
ENV_RELEASE_LOCK_KINDS = ("mag_lock", "delayed_egress", "card_reader", "electric_strike", "interlock")
MECH_ROLES = ("operator", "latch", "lock", "mechanism")
PEN_TOL = 0.012          # m; same bar as the `settle` gate - a pose nothing is driven into
TRAVEL_TOL = 0.95        # a releasable leaf keeps at least this fraction of its declared travel
PUSH_S = 12.0            # s of QA push / drive in the force leg (a 400 kg sectional door is slow)


def _eq_index(m, mujoco) -> dict:
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) or f"eq{i}": i for i in range(m.neq)}


def _subtree(m, body_id: int) -> set:
    """`body_id` and every body below it in the kinematic tree."""
    out = {int(body_id)}
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p in out:
                out.add(b)
                break
            p = int(m.body_parentid[p])
    return out


def declared_travel(spec: dict, model_meta: dict) -> tuple:
    """(value, unit) the primary joint must be able to cover, from the spec's own declaration."""
    kin = spec["kinematics"]
    k = kin["type"]
    if k in ("hinge_vertical", "hinge_horizontal"):
        return float(math.radians(kin.get("max_open_deg") or 90)), "rad"
    if k == "rotor":
        return float(math.radians(float(model_meta.get("ratchet_deg") or kin.get("ratchet_deg") or 90))), "rad"
    if k in ("slide_horizontal", "slide_vertical"):
        return float(kin.get("travel_m") or 0.0), "m"
    return 0.0, ""


def lock_release_path(spec: dict, model_meta: dict) -> str:
    """"env" / "robot" / "none" - who, if anyone, can free this leaf.  The welds are ground truth; the
    spec's own lock fields decide it for a door that has none."""
    rel = {w.get("release", "env") for w in (model_meta.get("breakable_welds") or []) if w.get("holds_primary")}
    if "env" in rel:
        return "env"
    if "robot" in rel:
        return "robot"
    if rel == {"none"}:
        return "none"
    from . import hardware as H
    lk = H.LOCKS[spec["lock"]["model"]]
    engaged = bool(spec["lock"].get("engaged")) and lk.kind not in ("none", "child_lock_cover", "jam_stuck")
    if not engaged:
        return "robot"                                  # nothing holds it: the robot just opens the door
    if lk.kind in ENV_RELEASE_LOCK_KINDS:
        return "env"
    return "robot" if bool(spec["lock"].get("robot_side_release")) else "none"


def released_state(m, d, mujoco, model_meta: dict, joints: dict, release: bool = True):
    """Put the model in its post-release state: holding constraints dropped, mechanism joints withdrawn."""
    mujoco.mj_resetData(m, d)
    dropped = []
    if release:
        idx = _eq_index(m, mujoco)
        for w in (model_meta.get("breakable_welds") or []):
            if w.get("release", "env") == "none":
                continue
            eid = idx.get(w["name"], -1)
            if eid >= 0:
                d.eq_active[eid] = 0
                dropped.append(w["name"])
        for name, role in joints.items():
            if role not in MECH_ROLES:
                continue
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0 or not m.jnt_limited[j]:
                continue
            lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            if hi - lo < 0.006:          # a part with no travel is not a release, it is fixed hardware
                continue
            d.qpos[m.jnt_qposadr[j]] = hi
    return dropped


def _still_pinned(m, d, mujoco, leaf: set) -> list:
    """Active WELD equalities that still tie the leaf's subtree to static structure.

    Welds only, and only against a body welded to the world: those are the six-DoF constraints that hold a leaf
    shut.  A ``connect`` is a three-DoF ball joint, and every one in this dataset is a linkage loop - a closer's
    forearm pinned to its frame shoe, rooted on the leaf and reaching the world - which the arm's own joints
    satisfy at any door angle.  Whether those loops close is `linkage_feasibility`'s gate, not this one."""
    out = []
    for i in range(m.neq):
        if not d.eq_active[i]:
            continue
        if int(m.eq_type[i]) != int(mujoco.mjtEq.mjEQ_WELD):
            continue
        a, b = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
        for x, y in ((a, b), (b, a)):
            if x in leaf and y not in leaf and int(m.body_weldid[y]) == 0:
                out.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) or f"eq{i}")
                break
    return out


def _worst_penetration(m, d, mujoco, leaf: set) -> tuple:
    """Deepest contact between the leaf's own subtree and static structure at this pose.

    Scoped that way on purpose: the pose is set by writing the primary joint, which does NOT re-solve the
    closed loops a closer arm or a fold linkage lives in (the clearance gate has a numerical solver for
    that).  The question here is only whether the LEAF can be where the task needs it."""
    worst = (0.0, None, None)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        pair = (b1 in leaf and int(m.body_weldid[b2]) == 0) or (b2 in leaf and int(m.body_weldid[b1]) == 0)
        if pair and c.dist < worst[0]:
            worst = (float(c.dist), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                     mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
    return worst


def _scenarios(spec: dict, door_dir: str) -> list:
    """The scenarios as they SHIPPED (spec.json), so the gate tests what a benchmark user reads."""
    if spec.get("benchmark"):
        return list(spec["benchmark"].get("scenarios") or [])
    try:
        with open(os.path.join(door_dir, "spec.json")) as f:
            return list((json.load(f).get("benchmark") or {}).get("scenarios") or [])
    except Exception:
        return []


def run_task_achievable(spec: dict, door_dir: str, model_meta: dict, m, d, phys: dict, joints: dict) -> dict:
    """``checks["task_achievable"]`` + its metrics.  `joints` is {joint name: IR role} from model.json."""
    import mujoco
    out = {"ok": True, "scenarios": [], "failures": []}
    pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, model_meta.get("primary_joint") or "")
    scen = _scenarios(spec, door_dir)
    release = lock_release_path(spec, model_meta)
    out["release_path"] = release
    if pj < 0:
        out["note"] = "no primary joint (strip curtain / multi-leaf): nothing to prove"
        return out
    is_hinge = int(m.jnt_type[pj]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    qadr, dof = int(m.jnt_qposadr[pj]), int(m.jnt_dofadr[pj])
    leaf_bodies = _subtree(m, int(m.jnt_bodyid[pj]))

    def fail(rule, detail, **kw):
        out["ok"] = False
        out["failures"].append({"rule": rule, "detail": detail, **kw})

    # ---- rule 2: a leaf whose lock can be released keeps its whole declared travel
    want, unit = declared_travel(spec, model_meta)
    lo, hi = (float(m.jnt_range[pj][0]), float(m.jnt_range[pj][1])) if m.jnt_limited[pj] else (-math.inf, math.inf)
    span = hi - lo
    out["primary_joint"] = model_meta.get("primary_joint")
    out["primary_range"] = None if not m.jnt_limited[pj] else [lo, hi]
    out["declared_travel"] = [round(want, 5), unit]
    if release != "none" and want > 0 and math.isfinite(span) and span < TRAVEL_TOL * want:
        fail("travel", f"the lock is releasable ({release}) but the primary joint keeps only {span:.4g} {unit} of "
                       f"its declared {want:.4g} {unit}: a release cannot widen a joint range",
             span=span, declared=want, unit=unit)

    # ---- rule 1: every moving scenario's pass threshold is reachable after the declared release
    moving = [s for s in scen if s["name"] in MOVING_SCENARIOS]
    for s in moving:
        thr = s.get("thresholds") or {}
        target = float(thr.get("clear_rad") if is_hinge else thr.get("clear_m") or 0.0) if is_hinge else float(thr.get("clear_m") or 0.0)
        if is_hinge and not target:
            target = float(thr.get("clear_rad") or 0.0)
        rec = {"name": s["name"], "target": round(target, 5), "unit": "rad" if is_hinge else "m"}
        if target <= 0:
            rec["note"] = "no pass threshold declared"
            out["scenarios"].append(rec)
            continue
        released_state(m, d, mujoco, model_meta, joints, release=(release != "none"))
        if m.jnt_limited[pj] and not (lo - 1e-6 <= target <= hi + 1e-6):
            fail("reach", f"{s['name']}: the pass threshold {target:.4g} is outside the primary joint's range "
                          f"[{lo:.4g}, {hi:.4g}] even after the {release} release", scenario=s["name"], target=target)
            rec["ok"] = False
            out["scenarios"].append(rec)
            continue
        d.qpos[qadr] = target
        mujoco.mj_forward(m, d)
        pinned = _still_pinned(m, d, mujoco, leaf_bodies)
        pen = _worst_penetration(m, d, mujoco, leaf_bodies)
        rec.update({"still_pinned": pinned, "penetration_m": round(pen[0], 5), "penetration_pair": [pen[1], pen[2]]})
        if pinned:
            fail("reach", f"{s['name']}: {', '.join(pinned)} still pins the leaf to the world at the pass threshold",
                 scenario=s["name"], equalities=pinned)
        if pen[0] < -PEN_TOL:
            fail("reach", f"{s['name']}: at the pass threshold {pen[1]} is {abs(pen[0]) * 1000:.0f} mm inside {pen[2]}",
                 scenario=s["name"], penetration_m=pen[0], pair=[pen[1], pen[2]])
        rec["ok"] = not pinned and pen[0] >= -PEN_TOL
        out["scenarios"].append(rec)

    # ---- rule 3: a leaf only the environment can release gets a force-level proof, because no other gate
    #      actuates it (QA skips `actuate_opens` on env-release locks: there is nothing for it to work)
    env_welds = [w for w in (model_meta.get("breakable_welds") or []) if w.get("release", "env") == "env"]
    if moving and env_welds:
        from .qa import qa_push
        open_thr = float((moving[0].get("thresholds") or {}).get("open_rad" if is_hinge else "open_m") or 0.0)
        if open_thr > 0:
            # qa_push resets the data (and with it every equality), so size the push BEFORE dropping the welds
            push = float(qa_push(m, d, pj, phys["mass"]["total_kg"], spec["leaf"]["width"])["push"])
            # a powered leaf (elevator landing doors, automatic sliders) is opened by its own operator once the
            # interlock clears, not by a shove: drive the servo the model ships, exactly as DoorEnv does
            drive = [a for a in range(m.nu) if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) and int(m.actuator_trnid[a][0]) == pj]
            released_state(m, d, mujoco, model_meta, joints, release=True)
            mujoco.mj_forward(m, d)
            n = int(PUSH_S / float(m.opt.timestep))
            for _ in range(n):
                d.qfrc_applied[:] = 0
                if drive:
                    for a in drive:
                        d.ctrl[a] = float(m.actuator_ctrlrange[a][1]) if bool(m.actuator_ctrllimited[a]) else 1.0
                else:
                    d.qfrc_applied[dof] = push
                mujoco.mj_step(m, d)
                if abs(float(d.qpos[qadr])) >= open_thr:
                    break
            moved = abs(float(d.qpos[qadr]))
            how = "the door's own drive" if drive else "the QA push"
            out["force_leg"] = {"driven_by": "actuator" if drive else "push", "push": round(push, 2),
                                "moved": round(moved, 5), "threshold": round(open_thr, 5),
                                "welds": [w["name"] for w in env_welds]}
            if moved < open_thr:
                fail("force", f"after the environment release {how} moved the leaf {moved:.4g} of the "
                              f"{open_thr:.4g} the benchmark's `opened` event needs", moved=moved, threshold=open_thr)
    return out


# ---------------------------------------------------------------------------
# pair_swing: a paired door swings the way its configuration says
# ---------------------------------------------------------------------------
def run_pair_swing(spec: dict, model_meta: dict, m, d) -> dict:
    """A pair's two leaves swing the way the configuration names: a **double-egress** pair one leaf each way
    (that is what the configuration is for - it is the pair you can push through from either side), every other
    pair (french, storefront, commercial panic, barn, saloon) both leaves the same way.

    Measured from where the leaf's own edge site ENDS UP, never from the hinge axis sign.  The two leaves of a
    pair are mirror images - hinges on opposite jambs, leaf x direction u = +1 and -1 - so a shared axis sign is
    *opposite* physical swing and an opposed one is the *same* swing.  Reading the sign instead of the motion is
    how a review concluded all ten double-egress pairs swung the same way when in fact every one of them is
    correct; this gate measures the thing the claim was about.
    """
    import mujoco
    out = {"ok": True, "checked": False}
    if not model_meta.get("pair") or not model_meta.get("secondary_joint"):
        return out
    double_egress = bool(spec["kinematics"].get("double_egress"))
    dy = {}
    for leaf, jn in (("leaf_a", model_meta.get("primary_joint")), ("leaf_b", model_meta.get("secondary_joint"))):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn or "")
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{leaf}_edge_mid")
        if j < 0 or sid < 0 or int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            return out
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        y0 = float(d.site_xpos[sid][1])
        lo, hi = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) if m.jnt_limited[j] else (-1.4, 1.4)
        d.qpos[m.jnt_qposadr[j]] = hi if abs(hi) >= abs(lo) else lo
        mujoco.mj_forward(m, d)
        dy[leaf] = float(d.site_xpos[sid][1]) - y0
    a, b = dy["leaf_a"], dy["leaf_b"]
    out.update({"dy_leaf_a": round(a, 4), "dy_leaf_b": round(b, 4), "double_egress": double_egress})
    if min(abs(a), abs(b)) < 0.02:
        out["note"] = "one leaf is inactive (flush bolts): only an active pair has a swing direction to check"
        return out
    out["checked"] = True
    same_side = a * b > 0
    out["ok"] = (not same_side) if double_egress else same_side
    if not out["ok"]:
        out["detail"] = (f"double egress: both leaves swing to {'+y' if a > 0 else '-y'}"
                         if double_egress else
                         f"a {spec.get('context')} pair: the leaves swing to opposite sides ({a:+.2f} / {b:+.2f})")
    return out

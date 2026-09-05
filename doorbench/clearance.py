"""Deterministic kinematic clearance gate.

The force-driven QA (qa.py) only ever sees *collision* geometry and only visits the configurations that the test
forces happen to reach.  This gate is geometric and exhaustive instead: every joint of a door is swept through its
full range with ALL geometry made collidable (visual-only parts included, because that is what a viewer shows) and
with MuJoCo's parent-child contact filter disabled, and any interpenetration deeper than a small tolerance is a
failure.  Sweeps:

* ``initial``            - the shipped configuration
* ``open:<leaf joint>``  - each leaf joint through its range with every releasable mechanism in its released state
                           (bolts retracted, hooks lifted, handles turned) - the door must open cleanly
* ``latched:<leaf joint>`` - same sweep with mechanisms at rest; pairs that are *supposed* to block (latch/lock
                           against strike/frame) are ignored, everything else must still clear
* ``mech:<joint>``       - each mechanism joint through its range with the leaf closed (coupled joints follow)
* ``coupling:<joint>``   - every joint equality: the driven joint must be able to follow its driver over the driver's
                           whole range without leaving its own limits (a driven hinge parked on a limit that the
                           coupling pushes against locks the mechanism - the accordion folds of 2026-09)

Hinge knuckles/leaves are allowed a larger overlap (they are mortised into leaf and jamb by design).
"""
from __future__ import annotations

import fnmatch
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

TOL = 0.002           # m; general interpenetration tolerance
TOL_HINGE = 0.012     # m; hinge hardware overlaps the members it is mortised into
LEAF_ROLES = ("primary", "secondary")
MECH_ROLES = ("operator", "latch", "lock", "mechanism")
BLOCKING = ("latch", "lock")          # semantics that are expected to block a latched leaf against the frame
# pairs that interpenetrate BY DESIGN (a part sliding inside its own housing); models may add more via meta["clearance_allow"]
DEFAULT_ALLOW = [("*pushbutton_geom", "*pushbutton_housing"), ("*_pin_geom", "*_pin_housing"), ("*_pin_geom", "*_pin_bracket"), ("*ring_geom", "*ring_recess"),
                 ("*_bolt_geom", "*_bolt_housing"), ("*_hasp", "*_staple"),
                 # spindles / thumbturn and cylinder stubs pass through the lock body inside the leaf; the rim bolt lives in its case
                 ("*deadbolt_box", "*thumbturn_mesh"), ("*deadbolt_box", "*cylinder_face"), ("*_spindle", "*_bolt_capsule"), ("*_knob_*", "*_bolt_capsule"),
                 ("*_device_case", "*_bolt_capsule"),
                 # a surface hasp plate lies flat across the door/frame joint (frame face modelled as a solid member)
                 ("*_hasp", "jamb_*"), ("*_hasp", "post_*"), ("*_hasp", "stop_*"), ("*_hasp", "seal_*"),
                 ("*thumbturn_mesh", "*_escutcheon_*"), ("*cylinder_face", "*_escutcheon_*")]
FRAME_LIKE = ("frame", "latch", "lock", "wall", "track")
HARDWARE = ("operator", "latch", "lock", "mechanism", "closer", "track", "hinge")


def gate_model(xml_path: str):
    import mujoco
    spec = mujoco.MjSpec.from_file(xml_path)
    for g in spec.geoms:
        g.contype = 1
        g.conaffinity = 1
        g.margin = 0.0
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    return spec.compile()


COUPLING_TOL = 1e-3   # rad / m; a driven joint may leave its range by no more than this over the driver's travel


def coupling_range_failures(m, tol: float = COUPLING_TOL, n_samples: int = 49) -> List[dict]:
    """Joint equalities whose driven joint cannot follow its driver over the driver's whole range.

    MuJoCo's joint equality is q_a = qpos0_a + poly(q_b - qpos0_b).  If the image of the driver's range leaves the
    driven joint's own limited range, the joint limit and the equality fight: the pair is locked (or the driver is
    capped short of its range) - a mechanism that looks fine in every kinematic pose and never moves under a push.
    Unlimited drivers are skipped (their image is unbounded by construction)."""
    import mujoco
    out = []
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    for e in range(m.neq):
        if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT) or not m.eq_active0[e]:
            continue
        a, b = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if b < 0 or not m.jnt_limited[a] or not m.jnt_limited[b]:
            continue
        lo_b, hi_b = (float(x) for x in m.jnt_range[b])
        lo_a, hi_a = (float(x) for x in m.jnt_range[a])
        qa0, qb0 = float(m.qpos0[m.jnt_qposadr[a]]), float(m.qpos0[m.jnt_qposadr[b]])
        c = [float(x) for x in m.eq_data[e][:5]]
        xs = np.linspace(lo_b, hi_b, n_samples)
        ys = qa0 + sum(c[k] * (xs - qb0) ** k for k in range(5))
        over = np.maximum(lo_a - ys, ys - hi_a)
        i = int(np.argmax(over))
        if over[i] > tol:
            out.append({"equality": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e), "driven": jname(a), "driver": jname(b),
                        "driver_q": float(xs[i]), "driven_q": float(ys[i]), "driven_range": [lo_a, hi_a], "overshoot": float(over[i])})
    return out


def _joint_info(model_json: dict) -> Dict[str, dict]:
    out = {}
    for b in model_json["bodies"]:
        j = b.get("joint")
        if j:
            out[j["name"]] = dict(j, body=b["name"])
    return out


def _semantics(model_json: dict) -> Dict[str, str]:
    out = {}
    for b in model_json["bodies"]:
        for g in b["geoms"]:
            out[g["name"]] = g.get("semantic", "")
    return out


class Clearance:
    def __init__(self, door_dir: str, tier: str = "full"):
        import mujoco
        self.mj = mujoco
        self.dir = door_dir
        xml = os.path.join(door_dir, {"full": "door.xml", "simple": "door_simple.xml", "minimal": "door_minimal.xml"}[tier])
        self.m = gate_model(xml)
        self.d = mujoco.MjData(self.m)
        with open(os.path.join(door_dir, "model.json")) as f:
            mj_ = json.load(f)
        self.meta = mj_["meta"]
        self.allow = list(DEFAULT_ALLOW) + [tuple(a[:2]) for a in self.meta.get("clearance_allow", [])]
        self.locked_shut = False
        try:
            with open(os.path.join(door_dir, "spec.json")) as f:
                sp_ = json.load(f)
            self.locked_shut = bool(sp_["lock"].get("engaged")) and not sp_["lock"].get("robot_side_release", True)
        except Exception:
            pass
        self.joints = _joint_info(mj_)
        self.sem = _semantics(mj_)
        m = self.m
        self.jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
        self.gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
        self.bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        # fixed tendons (one-sided couplings): list of (range_lo, [(qadr, coef)])
        self.tendons = []
        for t in range(m.ntendon):
            if m.tendon_limited[t]:
                terms = []
                for w in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
                    if int(m.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_JOINT):
                        terms.append((int(m.jnt_qposadr[m.wrap_objid[w]]), float(m.wrap_prm[w])))
                if terms:
                    self.tendons.append((float(m.tendon_range[t][0]), terms))

    # ---- kinematic helpers -------------------------------------------------------------------------------
    def resolve(self, q: np.ndarray) -> np.ndarray:
        """Apply joint-polynomial equalities and one-sided tendon couplings to make q consistent."""
        m, mujoco = self.m, self.mj
        for _ in range(2):
            for e in range(m.neq):
                if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT):
                    continue
                j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
                c = m.eq_data[e][:5]
                if j2 < 0:
                    q[m.jnt_qposadr[j1]] = c[0]
                    continue
                x = q[m.jnt_qposadr[j2]]
                q[m.jnt_qposadr[j1]] = c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4
            for lo, terms in self.tendons:
                length = sum(coef * q[adr] for adr, coef in terms)
                if length < lo - 1e-9:
                    # driven joint is the one with the positive unit coefficient (the bolt); push it to satisfy
                    for adr, coef in terms:
                        if coef > 0:
                            q[adr] += (lo - length) / coef
                            break
        return q

    def _locked(self, jname: str) -> bool:
        j = self.jid[jname]
        lo, hi = self.m.jnt_range[j]
        return bool(self.m.jnt_limited[j]) and (hi - lo) < 0.006

    def released_qpos(self) -> np.ndarray:
        q = self.m.qpos0.copy()
        for name, info in self.joints.items():
            if name not in self.jid or info.get("role") not in MECH_ROLES:
                continue
            j = self.jid[name]
            if self._locked(name) or not self.m.jnt_limited[j]:
                continue
            q[self.m.jnt_qposadr[j]] = self.m.jnt_range[j][1]
        return self.resolve(q)

    def contacts(self, q: np.ndarray, tol_fn) -> List[Tuple[str, str, float]]:
        m, d, mujoco = self.m, self.d, self.mj
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = self.gname[c.geom1], self.gname[c.geom2]
            tol = tol_fn(g1, g2)
            if c.dist < -tol:
                out.append((g1, g2, float(c.dist)))
        return out

    def _ancestor(self, b_desc: int, b_anc: int) -> bool:
        m = self.m
        b = b_desc
        for _ in range(12):
            b = int(m.body_parentid[b])
            if b == b_anc:
                return True
            if b == 0:
                return False
        return False

    def tol_for(self, g1: str, g2: str, ignore_blocking: bool = False) -> float:
        m = self.m
        s1, s2 = self.sem.get(g1, ""), self.sem.get(g2, "")
        for pa, pb in self.allow:
            if (fnmatch.fnmatch(g1, pa) and fnmatch.fnmatch(g2, pb)) or (fnmatch.fnmatch(g1, pb) and fnmatch.fnmatch(g2, pa)):
                return 1e9
        if s1 == "hinge" or s2 == "hinge":
            return TOL_HINGE
        b1, b2 = int(m.geom_bodyid[m.geom(g1).id]), int(m.geom_bodyid[m.geom(g2).id])
        # mortised hardware lives inside the leaf it is mounted on (bolt in its mortise, spindle through the door)
        for sh, sl, bh, bl in ((s1, s2, b1, b2), (s2, s1, b2, b1)):
            if sh in HARDWARE and sl in ("leaf", "glass") and self._ancestor(bh, bl):
                return 1e9
        if ignore_blocking and b1 != b2 and (s1 in BLOCKING or s2 in BLOCKING):
            return 1e9
        return TOL

    # ---- the gate ---------------------------------------------------------------------------------------
    def run(self, n_steps: int = 24) -> dict:
        m = self.m
        failures: Dict[str, dict] = {}

        def record(config: str, jname: str, qv: float, cons):
            for g1, g2, dist in cons:
                key = tuple(sorted((g1, g2)))
                depth = -dist
                prev = failures.get(key)
                if prev is None or depth > prev["depth"]:
                    failures[key] = {"geoms": list(key), "depth": round(depth, 4), "config": config, "joint": jname, "q": round(float(qv), 4),
                                     "bodies": [self.bname[m.geom_bodyid[self.m.geom(g1).id]], self.bname[m.geom_bodyid[self.m.geom(g2).id]]]}

        base = m.qpos0.copy()
        record("initial", "", 0.0, self.contacts(self.resolve(base.copy()), lambda a, b: self.tol_for(a, b)))
        released = self.released_qpos()
        leaf_joints = [n for n, j in self.joints.items() if j.get("role") in LEAF_ROLES and n in self.jid]
        mech_joints = [n for n, j in self.joints.items() if j.get("role") in MECH_ROLES and n in self.jid]
        for jn in leaf_joints:
            j = self.jid[jn]
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-math.pi, math.pi))
            if hi - lo < 1e-6:
                continue
            for k in range(n_steps + 1):
                qv = lo + (hi - lo) * k / n_steps
                q = released.copy()
                q[m.jnt_qposadr[j]] = qv
                record(f"open:{jn}", jn, qv, self.contacts(self.resolve(q), lambda a, b: self.tol_for(a, b, ignore_blocking=self.locked_shut)))
                q = base.copy()
                q[m.jnt_qposadr[j]] = qv
                record(f"latched:{jn}", jn, qv, self.contacts(self.resolve(q), lambda a, b: self.tol_for(a, b, ignore_blocking=True)))
        for jn in mech_joints:
            j = self.jid[jn]
            if not m.jnt_limited[j]:
                continue
            lo, hi = m.jnt_range[j]
            if hi - lo < 1e-6:
                continue
            for k in range(1, 13):
                qv = lo + (hi - lo) * k / 12
                q = base.copy()
                q[m.jnt_qposadr[j]] = qv
                record(f"mech:{jn}", jn, qv, self.contacts(self.resolve(q), lambda a, b: self.tol_for(a, b)))
        # couplings: a driven joint parked on a limit that its equality pushes against is a locked mechanism
        for c in coupling_range_failures(m):
            failures[("coupling", c["driven"])] = {"geoms": [c["driven"], c["driver"]], "depth": round(c["overshoot"], 4), "config": f"coupling:{c['driven']}",
                                                  "joint": c["driver"], "q": round(c["driver_q"], 4), "bodies": [self.bname[m.jnt_bodyid[self.jid[c["driven"]]]], self.bname[m.jnt_bodyid[self.jid[c["driver"]]]]],
                                                  "coupling": c}
        fails = sorted(failures.values(), key=lambda f: -f["depth"])
        return {"ok": len(fails) == 0, "n_failures": len(fails), "failures": fails[:40], "leaf_joints": leaf_joints, "mech_joints": mech_joints}


def run_clearance(door_dir: str, tier: str = "full", n_steps: int = 24) -> dict:
    try:
        return Clearance(door_dir, tier).run(n_steps)
    except Exception as e:  # a gate that cannot run is a failure, not a pass
        return {"ok": False, "n_failures": 1, "failures": [{"geoms": [], "depth": 0.0, "config": "error", "joint": "", "q": 0.0, "bodies": [], "error": f"{type(e).__name__}: {e}"}]}

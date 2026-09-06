"""MuJoCo reference runner for the parity protocol (CPU).  Same drive semantics as ``doorbench.qa.run_qa``:
plain ``mujoco.mj_step`` on door.xml (dt 0.002, implicitfast, no DoorEnv passive callback), generalized forces via
``d.qfrc_applied``, kinematic pin of the primary joint during ``release``.  Every phase records a 30 Hz curve of all
joints in DoorBench coordinates and is judged by ``protocol.phase_metrics`` / ``protocol.phase_status``.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from . import protocol as P


def _warn_counts(d):
    import mujoco
    return np.array([d.warning[i].number for i in range(mujoco.mjtWarning.mjNWARNING)], dtype=int)


def _warn_names(delta):
    import mujoco
    return [mujoco.mjtWarning(i).name for i in range(len(delta)) if delta[i] > 0]


class MujocoDoor:
    """One door in MuJoCo with the protocol's phase machinery."""

    def __init__(self, door_dir: str, xml: str = "door.xml", dt: float | None = None):
        import mujoco
        self.mj = mujoco
        self.door_dir = door_dir
        with open(os.path.join(door_dir, "spec.json")) as f:
            self.spec = json.load(f)
        with open(os.path.join(door_dir, "model.json")) as f:
            self.model_json = json.load(f)
        qa_path = os.path.join(door_dir, "qa.json")
        self.qa = json.load(open(qa_path)) if os.path.isfile(qa_path) else None
        self.m = mujoco.MjModel.from_xml_path(os.path.join(door_dir, xml))
        if dt:
            self.m.opt.timestep = dt
        self.dt = float(self.m.opt.timestep)
        self.d = mujoco.MjData(self.m)
        self.names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}" for j in range(self.m.njnt)]
        self.jid = {n: j for j, n in enumerate(self.names)}
        self.qadr = {n: int(self.m.jnt_qposadr[j]) for n, j in self.jid.items()}
        self.dadr = {n: int(self.m.jnt_dofadr[j]) for n, j in self.jid.items()}
        rl_meta = P.read_rl_meta(door_dir)
        self.inputs = P.door_inputs(self.spec, self.model_json, forces=self.measure_forces(), qa=self.qa, rl_meta=rl_meta)
        self.pj = self.inputs["primary_joint"]
        self.sample_every = 1.0 / P.SAMPLE_HZ

    # ------------------------------------------------------------------
    def measure_forces(self) -> dict:
        """qa.py's adaptive-push terms at qpos0: gravity bias, Coulomb friction and spring preload of the primary DOF."""
        mujoco, m, d = self.mj, self.m, self.d
        pj = self.model_json["meta"]["primary_joint"]
        j = self.jid[pj]
        dof = int(m.jnt_dofadr[j])
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        bias = abs(float(d.qfrc_bias[dof] - d.qfrc_passive[dof]))
        fl = float(m.dof_frictionloss[dof])
        preload = abs(float(m.jnt_stiffness[j] * m.qpos_spring[m.jnt_qposadr[j]])) if m.jnt_stiffness[j] > 0 else 0.0
        return {"bias": bias, "frictionloss": fl, "preload": preload, "source": "mujoco"}

    def pose0(self) -> dict:
        """World frames at qpos0 (for the Isaac frame check): body origins, joint anchors and axes."""
        mujoco, m, d = self.mj, self.m, self.d
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        bodies = {}
        for b in range(1, m.nbody):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"b{b}"
            bodies[n] = {"pos": [round(float(x), 6) for x in d.xpos[b]], "com": [round(float(x), 6) for x in d.xipos[b]], "mass": round(float(m.body_mass[b]), 6),
                         "moving": bool(m.body_dofnum[b] > 0 or m.body_parentid[b] != 0)}
        joints = {n: {"anchor": [round(float(x), 6) for x in d.xanchor[j]], "axis": [round(float(x), 6) for x in d.xaxis[j]]} for n, j in self.jid.items()}
        return {"bodies": bodies, "joints": joints}

    # ------------------------------------------------------------------
    def _reset(self, overrides: dict | None = None):
        """mj_resetData (+ joint overrides).  No mj_forward here: qa.py steps straight after the reset (an extra forward
        pass changes the solver warm start and, on chaotic mechanisms such as a free hatch ring, the trajectory);
        the settle phase and the closer phase call mj_forward themselves, exactly as qa.py does."""
        mujoco, m, d = self.mj, self.m, self.d
        mujoco.mj_resetData(m, d)
        if overrides:
            for n, v in overrides.items():
                if n in self.qadr:
                    d.qpos[self.qadr[n]] = v
            mujoco.mj_forward(m, d)

    def _qmap(self) -> dict:
        d = self.d
        return {n: float(d.qpos[a]) for n, a in self.qadr.items()}

    def _apply(self, eff: dict):
        d = self.d
        d.qfrc_applied[:] = 0.0
        for n, f in eff.items():
            a = self.dadr.get(n)
            if a is not None:
                d.qfrc_applied[a] = f

    def run_phase(self, phase: str, duration: float, pins: dict | None = None, early_exit=None) -> dict:
        """Step ``duration`` seconds applying ``protocol.phase_efforts``; returns the curve dict."""
        mujoco, m, d = self.mj, self.m, self.d
        inputs = self.inputs
        n_steps = int(round(duration / self.dt))
        w0 = _warn_counts(d)
        t = 0.0
        ts, qs, vs = [], [], []
        qmin, qmax = d.qpos.copy(), d.qpos.copy()
        vmax = np.zeros(m.nv)
        pen0 = None
        if phase == "settle":
            mujoco.mj_forward(m, d)
            pen0 = min([float(d.contact[i].dist) for i in range(d.ncon)], default=0.0)
        last_idx = -1
        sample_idx = math.floor(t * P.SAMPLE_HZ + 1e-6)
        ts.append(0.0); qs.append(d.qpos.copy()); vs.append(d.qvel.copy()); last_idx = sample_idx
        exited = False
        for k in range(n_steps):
            t = k * self.dt
            q = self._qmap()
            qd = {n: float(d.qvel[a]) for n, a in self.dadr.items()}
            eff = P.phase_efforts(inputs, phase, t, q, kind="mjcf", qd=qd)
            self._apply(eff)
            if pins:
                for n, v in pins.items():
                    d.qpos[self.qadr[n]] = v
                    d.qvel[self.dadr[n]] = 0.0
            mujoco.mj_step(m, d)
            t = (k + 1) * self.dt
            np.minimum(qmin, d.qpos, out=qmin); np.maximum(qmax, d.qpos, out=qmax); np.maximum(vmax, np.abs(d.qvel), out=vmax)
            sample_idx = math.floor(t * P.SAMPLE_HZ + 1e-6)
            if sample_idx > last_idx:
                ts.append(round(t, 6)); qs.append(d.qpos.copy()); vs.append(d.qvel.copy()); last_idx = sample_idx
            if early_exit is not None and early_exit(t, float(d.qpos[self.qadr[self.pj]])):
                exited = True
                break
        if exited and ts[-1] != round(t, 6):
            ts.append(round(t, 6)); qs.append(d.qpos.copy()); vs.append(d.qvel.copy())
        Q = np.array(qs); V = np.array(vs)
        finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(Q).all())
        curve = {"t": ts, "q": {n: [float(x) for x in Q[:, a]] for n, a in self.qadr.items()},
                 "v": {self.pj: [float(x) for x in V[:, self.dadr[self.pj]]]},
                 "minmax": {n: [float(qmin[a]), float(qmax[a])] for n, a in self.qadr.items()},
                 "vmax": {n: float(vmax[a]) for n, a in self.dadr.items()}, "finite": finite, "warnings": _warn_names(_warn_counts(d) - w0), "early_exit": exited}
        if pen0 is not None:
            curve["pen0_m"] = pen0
        return curve

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """The whole protocol; returns the door record (with full 30 Hz curves)."""
        t0 = time.time()
        inputs = self.inputs
        sched = inputs["schedule"]["mjcf"]
        th = inputs["thresholds"]
        phases = {}
        ctx = {}
        free_hold = sched["hold"] != "hold"
        for phase in P.PHASES:
            expected = sched[phase]
            if expected.startswith("na:"):
                phases[phase] = {"expected": expected, "status": "na", "metrics": {}, "curve": None}
                continue
            if phase == "relatch" and (ctx.get("opened") is None or ctx["opened"] <= th["relatch_min_open"]):
                m = {"finite": True, "opened_before": ctx.get("opened"), "limit_violations": [], "warnings": []}
                phases[phase] = {"expected": expected, "status": P.phase_status(inputs, phase, expected, m), "metrics": m, "curve": None}
                continue
            if P.PHASE_RESETS[phase]:
                self._reset(P.phase_initial_state(inputs, phase))
            pins, early = None, None
            if phase == "release":
                ctx["q_hold"] = float(self.d.qpos[self.qadr[self.pj]])
                pins = {self.pj: ctx["q_hold"]}
            if phase == "hold" and free_hold:
                thr_free = th["thr_free"]
                early = lambda t, q: t >= P.DURATIONS["hold"] - 1e-9 and q > thr_free
            dur = P.phase_duration(inputs, phase, "mjcf")
            curve = self.run_phase(phase, dur, pins=pins, early_exit=early)
            metrics = P.phase_metrics(inputs, phase, curve, ctx)
            status = P.phase_status(inputs, phase, expected, metrics)
            if phase == "operate":
                ctx["opened"] = metrics.get("opened")
            phases[phase] = {"expected": expected, "status": status, "metrics": metrics, "curve": curve, "informational": expected.endswith("_info")}
        rec = {"door_id": inputs["door_id"], "sim": "mujoco", "kind": "mjcf", "engine": {"mujoco": self.mj.__version__}, "dt": self.dt, "protocol_version": P.PROTOCOL_VERSION,
               "metrics_version": P.METRICS_VERSION, "inputs_hash": inputs["inputs_hash"], "inputs": inputs, "pose0": self.pose0(), "phases": phases, "emulations_used": ["servo_native"] if inputs["flags"]["automatic"] else [],
               "limits": {"violations": [dict(v, phase=p) for p, r in phases.items() for v in (r["metrics"].get("limit_violations") or [])]},
               "sanity": {"finite": all(r["metrics"].get("finite", True) for r in phases.values() if r["metrics"]), "velocity_cap_hit": any(r["metrics"].get("velocity_cap_hit") for r in phases.values() if r["metrics"]),
                          "warnings": sorted({w for r in phases.values() for w in (r["metrics"].get("warnings") or [])})},
               "qa_reproduction": self.qa_reproduction(phases), "wall_time_s": round(time.time() - t0, 3)}
        # informational phases (free-swing push, roller / magnetic catches) do not count against the door
        rec["ok"] = all(r["status"] in ("pass", "skip", "na") or r.get("informational") for r in phases.values()) and rec["sanity"]["finite"]
        return rec

    def qa_reproduction(self, phases: dict) -> dict:
        """Does the protocol reproduce the qa.json metrics of this door (same schedule -> same numbers)?"""
        ref = (self.qa or {}).get("metrics", {})
        pairs = {"qa_push": self.inputs["forces"]["push"], "hold_displacement": phases.get("hold", {}).get("metrics", {}).get("hold_displacement"),
                 "actuate_displacement": phases.get("operate", {}).get("metrics", {}).get("opened"), "operator_travel_reached": phases.get("operate", {}).get("metrics", {}).get("operator_travel_reached"),
                 "bolt_after_release_m": phases.get("release", {}).get("metrics", {}).get("bolt_after_release_m"), "relatch_closed_angle": phases.get("relatch", {}).get("metrics", {}).get("relatch_closed_angle"),
                 "relatch_repush_angle": phases.get("relatch", {}).get("metrics", {}).get("relatch_repush_angle"), "closer_final_angle": phases.get("closer", {}).get("metrics", {}).get("closer_final_angle"),
                 "locked_displacement": phases.get("locked", {}).get("metrics", {}).get("locked_displacement")}
        out = {"available": bool(ref), "mismatches": [], "compared": 0}
        for k, mine in pairs.items():
            if k not in ref or ref[k] is None or mine is None:
                continue
            out["compared"] += 1
            ok = abs(float(ref[k]) - float(mine)) <= max(1e-3, 1e-3 * abs(float(ref[k])))
            if not ok:
                out["mismatches"].append({"metric": k, "qa": ref[k], "protocol": mine})
        out["ok"] = not out["mismatches"]
        return out


# ---------------------------------------------------------------------------
def compact_record(rec: dict, keep_joints: int = 3, hz: int = 5) -> dict:
    """The aggregate-file version of a record: no full curves, a downsampled primary / operator / bolt curve per phase."""
    inputs = rec["inputs"]
    keep = [j for j in (inputs["primary_joint"], inputs["operator_joint"], inputs["latch_bolt_joint"]) if j]
    step = max(1, P.SAMPLE_HZ // hz)
    out = {k: v for k, v in rec.items() if k not in ("phases", "inputs", "pose0")}
    out["metrics_version"] = rec.get("metrics_version", P.METRICS_VERSION)
    out["pose0"] = {"bodies": {n: b["pos"] for n, b in rec.get("pose0", {}).get("bodies", {}).items() if b.get("moving")}, "joints": rec.get("pose0", {}).get("joints", {})}
    out["inputs"] = {k: inputs[k] for k in ("door_id", "family", "is_hinge", "primary_joint", "operator_joint", "latch_bolt_joint", "secondary_joint", "flags", "forces", "thresholds", "schedule", "inputs_hash", "coupling", "rl", "reference_qa", "push_base", "joints", "thumbturn_joint", "aux_joints", "dog_joints", "unlimited_joints", "latch_joints", "max_open_deg", "travel_m", "unit", "mass_kg", "kinematics_type", "leaf_width_m", "push_lever_m", "task", "protocol_version")}
    out["phases"] = {}
    for p, r in rec["phases"].items():
        row = {"expected": r["expected"], "status": r["status"], "metrics": r["metrics"], "informational": bool(r.get("informational"))}
        c = r.get("curve")
        if c:
            row["curve"] = {"t": [round(x, 4) for x in c["t"][::step]], "q": {j: [round(x, 5) for x in c["q"][j][::step]] for j in keep if j in c["q"]}, "hz": hz}
        out["phases"][p] = row
    return out


def run_door(door_dir: str, dt: float | None = None, cache_dir: str | None = None, force: bool = False) -> dict:
    """Run (or load from cache) one door; returns the full record.  The cache key is protocol version + inputs hash."""
    door_id = os.path.basename(door_dir.rstrip("/"))
    cache_path = os.path.join(cache_dir, f"{door_id}.json") if cache_dir else None
    door = MujocoDoor(door_dir, dt=dt)
    if cache_path and not force and os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                rec = json.load(f)
            if rec.get("protocol_version") == P.PROTOCOL_VERSION and rec.get("metrics_version") == P.METRICS_VERSION \
                    and rec.get("inputs_hash") == door.inputs["inputs_hash"] and abs(rec.get("dt", 0) - door.dt) < 1e-12:
                rec["cached"] = True
                return rec
        except Exception:
            pass
    rec = door.run()
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, cache_path)
    return rec

"""Result-file schema, tolerances, per-door comparison and discrepancy classes of the Isaac parity gate.

Both runners (``doorbench/parity/protocol.py``: MuJoCo on the CPU, Isaac Sim / PhysX on the GPU) write one JSON per
simulator and USD kind::

    results/parity/mujoco.json          {door_id: {phases: {name: {pass: bool, metrics: {...}}}, curves: {joint: [[t, q], ...]},
    results/parity/isaac_full.json                  metrics: {...}, errors: [...]}}
    results/parity/isaac_rl.json

Everything here reads that shape *defensively*: a file may wrap the doors in ``{"doors": {...}}`` or ``{"meta": ..,
"results": ..}``, phases may be a list, a phase may carry ``status`` instead of ``pass``, curves may be nested per
phase, names may be the long protocol labels ("P4 operate_open") or short ones ("operate").  Missing keys never raise.

The comparison follows the protocol's verdict logic: per phase both simulators must reach the same pass / fail status
(a status disagreement is grade C), and when they agree the metrics must be within tolerance (else grade B); a door
that could not be loaded / compared is grade X.  Each disagreement is tagged with a discrepancy class whose "likely
root cause" comes from the analysis of the first 40-door GPU probe (docs/ISAAC_PARITY.md).
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Iterable

# ---------------------------------------------------------------------------------------------------------------------
# phases

PHASES = ["structure", "pose0", "settle", "hold", "operate_open", "release", "relatch", "closer_return", "locked_holds", "limits", "sanity"]

PHASE_LABELS = {
    "structure": "P0 structure (joints / limits / gains / mass match model.json)",
    "pose0": "P1 pose0 (body, COM and joint-anchor frames at q0)",
    "settle": "P2 settle (1 s free, spring targets kept)",
    "hold": "P3 hold / free_opens (adaptive QA push on the door joint only)",
    "operate_open": "P4 operate + open (operator, thumbturn / aux / dogs, then push)",
    "release": "P5 release (latch bolt re-extends)",
    "relatch": "P6 relatch (close and re-push)",
    "closer_return": "P7 closer return from 60 deg",
    "locked_holds": "P8 locked holds (operator worked + push)",
    "limits": "P9 limits (every joint inside its range)",
    "sanity": "P10 sanity (finite, no explosion)",
}

_PHASE_ALIASES = {
    "structure": "structure", "struct": "structure", "load": "structure", "p0": "structure",
    "pose0": "pose0", "pose": "pose0", "frames": "pose0", "p1": "pose0",
    "settle": "settle", "p2": "settle",
    "hold": "hold", "free_opens": "hold", "hold_free": "hold", "hold_or_free": "hold", "push": "hold", "p3": "hold",
    "free_opens_fs": "free_swing", "free_swing": "free_swing",
    "operate_open": "operate_open", "operate": "operate_open", "actuate": "operate_open", "actuate_opens": "operate_open", "open": "operate_open", "p4": "operate_open",
    "release": "release", "latch_returns": "release", "latch_return": "release", "p5": "release",
    "relatch": "relatch", "p6": "relatch",
    "closer_return": "closer_return", "closer": "closer_return", "closer_returns": "closer_return", "p7": "closer_return",
    "locked_holds": "locked_holds", "locked": "locked_holds", "p8": "locked_holds",
    "limits": "limits", "p9": "limits",
    "sanity": "sanity", "p10": "sanity",
}

# qa.json check that the reference (MuJoCo) result of a phase must reproduce; a MuJoCo failure on a phase whose qa.json
# check passed is a protocol / nondeterminism problem (REFERENCE_QA_FAILURE), not a PhysX bug.
PHASE_QA_CHECKS = {"settle": ("settle",), "hold": ("hold", "free_opens"), "operate_open": ("actuate_opens",), "release": ("latch_returns",),
                   "relatch": ("relatch",), "closer_return": ("closer_returns",), "locked_holds": ("locked_holds",)}


def canonical_phase(name: str) -> str:
    """'P4 operate_open' / 'operate' / 'actuate_opens' -> 'operate_open'; unknown names are kept (lower-case)."""
    s = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"^p\d+_+", "", s) if not re.fullmatch(r"p\d+", s) else s
    return _PHASE_ALIASES.get(s, s)


# ---------------------------------------------------------------------------------------------------------------------
# loading

KINDS = ("full", "rl")
_KIND_ALIASES = {"full": "full", "usd_full": "full", "door": "full", "rl": "rl", "usd_rl": "rl", "canonical": "rl"}


def kind_from_filename(path: str) -> tuple[str | None, str | None, str]:
    """results/parity/isaac_full_dt240.json -> ("isaac", "full", "dt240"); mujoco.json -> ("mujoco", None, "")."""
    base = os.path.splitext(os.path.basename(path))[0].lower()
    toks = base.split("_")
    sim = toks[0]
    if sim not in ("isaac", "physx", "mujoco", "mj"):
        return None, None, base
    sim = "mujoco" if sim in ("mujoco", "mj") else "isaac"
    rest = toks[1:]
    kind = None
    if rest and rest[0] in _KIND_ALIASES:
        kind = _KIND_ALIASES[rest.pop(0)]
    elif len(rest) >= 2 and "_".join(rest[:2]) in _KIND_ALIASES:
        kind = _KIND_ALIASES["_".join(rest[:2])]
        rest = rest[2:]
    return sim, kind, "_".join(rest)


def _unwrap(doc: Any) -> tuple[dict, dict]:
    """Return (doors, meta) from any of the tolerated top-level shapes."""
    if isinstance(doc, list):
        doors = {}
        for rec in doc:
            if isinstance(rec, dict):
                did = rec.get("door_id") or rec.get("id")
                if did:
                    doors[str(did)] = rec
        return doors, {}
    if not isinstance(doc, dict):
        return {}, {}
    for key in ("doors", "results", "records"):
        if isinstance(doc.get(key), (dict, list)):
            inner, _ = _unwrap(doc[key])
            meta = {k: v for k, v in doc.items() if k != key}
            return inner, meta
    # plain {door_id: record}: records are dicts and keys look like door ids (or at least not meta keys)
    meta_keys = {"meta", "engine", "generated", "commit", "dt", "version", "protocol", "inputs_hash", "sim", "kind"}
    doors = {k: v for k, v in doc.items() if isinstance(v, dict) and k not in meta_keys and ("phases" in v or "errors" in v or "metrics" in v or k.startswith("db"))}
    meta = {k: v for k, v in doc.items() if k not in doors}
    return doors, meta


def load_results(path: str) -> tuple[dict[str, dict], dict]:
    """Load one runner file -> ({door_id: normalised record}, meta).  Missing / unreadable file -> ({}, {})."""
    if not path or not os.path.exists(path):
        return {}, {}
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}, {}
    doors, meta = _unwrap(doc)
    return {did: normalize_record(rec) for did, rec in doors.items()}, meta


def _pass_of(ph: Any) -> bool | None:
    """pass: bool | status: pass/fail/skip/na/error -> True / False / None (not run)."""
    if isinstance(ph, bool):
        return ph
    if not isinstance(ph, dict):
        return None
    if isinstance(ph.get("pass"), bool):
        return ph["pass"]
    if isinstance(ph.get("ok"), bool):
        return ph["ok"]
    st = str(ph.get("status", "")).lower()
    if st in ("pass", "passed", "ok", "true"):
        return True
    if st in ("fail", "failed", "false", "error"):
        return False
    return None


def _status_of(ph: Any) -> str:
    p = _pass_of(ph)
    if p is True:
        return "pass"
    if p is False:
        return "fail"
    if isinstance(ph, dict):
        st = str(ph.get("status", "")).lower()
        if st in ("skip", "skipped", "na", "n/a", "not_applicable"):
            return "skip"
    return "skip"


def normalize_record(rec: Any) -> dict:
    """One door's record -> {phases: {canonical: {pass, status, metrics, expected, reason, raw_name}}, curves, metrics, errors}."""
    if not isinstance(rec, dict):
        return {"phases": {}, "curves": {}, "metrics": {}, "errors": [f"malformed record: {type(rec).__name__}"]}
    phases_in = rec.get("phases") or {}
    items: list[tuple[str, Any]] = []
    if isinstance(phases_in, dict):
        items = list(phases_in.items())
    elif isinstance(phases_in, list):
        for ph in phases_in:
            if isinstance(ph, dict) and (ph.get("name") or ph.get("phase")):
                items.append((str(ph.get("name") or ph.get("phase")), ph))
    phases: dict[str, dict] = {}
    for raw, ph in items:
        canon = canonical_phase(raw)
        key = canon if canon not in phases else str(raw).lower()
        d = ph if isinstance(ph, dict) else {}
        phases[key] = {
            "pass": _pass_of(ph), "status": _status_of(ph), "raw_name": str(raw),
            "metrics": dict(d.get("metrics") or {}) if isinstance(d.get("metrics"), dict) else {},
            "expected": d.get("expected"), "reason": d.get("reason"),
        }
        # scalar metrics written next to `pass` (runner shorthand) count as metrics too
        for k, v in d.items():
            if k not in ("pass", "ok", "status", "metrics", "expected", "reason", "name", "phase", "curves", "t_start_s", "duration_s") and isinstance(v, (int, float, bool)) and k not in phases[key]["metrics"]:
                phases[key]["metrics"][k] = v
    errors = rec.get("errors") or []
    if isinstance(errors, str):
        errors = [errors]
    return {
        "phases": phases,
        "curves": rec.get("curves") if isinstance(rec.get("curves"), dict) else {},
        "metrics": dict(rec.get("metrics") or {}) if isinstance(rec.get("metrics"), dict) else {},
        "errors": [str(e) for e in errors],
        "engine": rec.get("engine"), "dt": rec.get("dt"),
    }


def has_load_error(rec: dict | None) -> bool:
    """spawn / load / structure failures make the door not comparable (grade X)."""
    if not rec:
        return False
    for e in rec.get("errors", []):
        el = e.lower()
        if any(t in el for t in ("spawn", "load", "usd", "stage", "batch exception", "articulation", "malformed", "structure")):
            return True
    st = rec["phases"].get("structure")
    return bool(st and st["pass"] is False)


# ---------------------------------------------------------------------------------------------------------------------
# curves

_CURVE_HINTS = {
    "primary": ("primary", "q_primary", "door", "door_hinge", "door_slide", "leaf_hinge", "leaf_slide", "leaf_a_hinge", "hatch_hinge", "gate_hinge", "curtain_slide"),
    "operator": ("operator", "q_operator", "op", "operator_hinge", "operator_slide", "leaf_handle_hinge", "leaf_lever_hinge", "handle"),
    "latch": ("bolt", "latch", "q_bolt", "q_latch", "latch_slide", "leaf_latch_bolt_slide", "latch_bolt"),
    "secondary": ("secondary", "q_secondary", "leaf2", "leaf2_hinge", "leaf2_slide", "leaf_b_hinge"),
}


def _as_series(v: Any) -> list[list[float]] | None:
    """[[t, q], ...] or {"t": [...], "q": [...]} -> [[t, q], ...]; anything else -> None."""
    if isinstance(v, dict) and isinstance(v.get("t"), list):
        q = v.get("q") or v.get("y") or v.get("values")
        if isinstance(q, list) and len(q) == len(v["t"]):
            return [[float(a), float(b)] for a, b in zip(v["t"], q) if _finite(a) and _finite(b)]
        return None
    if isinstance(v, list) and v and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in v[:3]):
        out = []
        for p in v:
            try:
                t, q = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            if _finite(t) and _finite(q):
                out.append([t, q])
        return out
    return None


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def flatten_curves(curves: dict, phase: str | None = None) -> dict[str, list[list[float]]]:
    """{joint: series} or {phase: {joint: series}} -> {joint: series} (the requested phase first, then the others)."""
    if not isinstance(curves, dict):
        return {}
    flat: dict[str, list[list[float]]] = {}
    nested = [(k, v) for k, v in curves.items() if isinstance(v, dict) and _as_series(v) is None]
    if nested:
        order = sorted(nested, key=lambda kv: 0 if phase and canonical_phase(kv[0]) == canonical_phase(phase) else 1)
        for ph, sub in order:
            for j, s in sub.items():
                ser = _as_series(s)
                if ser and j not in flat:
                    flat[j] = ser
        return flat
    for j, s in curves.items():
        ser = _as_series(s)
        if ser:
            flat[j] = ser
    return flat


def pick_curve(curves: dict, role: str, extra_names: Iterable[str] = (), phase: str | None = None) -> list[list[float]] | None:
    flat = flatten_curves(curves, phase)
    if not flat:
        return None
    names = list(extra_names) + list(_CURVE_HINTS.get(role, ()))
    lower = {k.lower(): k for k in flat}
    for n in names:
        if n and n.lower() in lower:
            return flat[lower[n.lower()]]
    if role == "primary":   # the joint with the largest excursion is the door
        best = max(flat.items(), key=lambda kv: (max(q for _, q in kv[1]) - min(q for _, q in kv[1])) if kv[1] else -1)
        return best[1]
    return None


def curve_rmse(a: list[list[float]] | None, b: list[list[float]] | None) -> float | None:
    """RMSE of b resampled (linear) onto a's time grid, over the overlapping time span."""
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    tb = [p[0] for p in b]
    qb = [p[1] for p in b]
    lo, hi = max(a[0][0], tb[0]), min(a[-1][0], tb[-1])
    acc, n, j = 0.0, 0, 0
    for t, q in a:
        if t < lo or t > hi:
            continue
        while j + 1 < len(tb) - 1 and tb[j + 1] < t:
            j += 1
        t0, t1 = tb[j], tb[j + 1]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        qi = qb[j] * (1 - w) + qb[j + 1] * w
        acc += (q - qi) ** 2
        n += 1
    return math.sqrt(acc / n) if n else None


# ---------------------------------------------------------------------------------------------------------------------
# tolerances (docs/ISAAC_PARITY.md "Tolerances"); a delta passes when within EITHER the absolute or the relative bound

TOLERANCES: dict[str, dict[str, float]] = {
    # settle
    "settle_drift": {"abs_hinge": 0.02, "abs_slide": 0.005, "rel": 0.0},
    "settle_drift_primary": {"abs_hinge": 0.02, "abs_slide": 0.005, "rel": 0.0},
    "settle_drift_operator": {"abs_hinge": 0.02, "abs_slide": 0.005, "rel": 0.0},
    "settle_drift_latch": {"abs_hinge": 0.002, "abs_slide": 0.002, "rel": 0.0},
    "pen0_m": {"abs_hinge": 0.003, "abs_slide": 0.003, "rel": 0.0},
    # hold / free_opens
    "hold_displacement": {"abs_hinge": 0.01, "abs_slide": 0.003, "rel": 0.0},
    "t_free": {"abs_hinge": 0.25, "abs_slide": 0.25, "rel": 0.30},
    "q_at_1s": {"abs_hinge": 0.1, "abs_slide": 0.05, "rel": 0.20},
    # operate
    "opened": {"abs_hinge": 0.1, "abs_slide": 0.05, "rel": 0.20},
    "actuate_displacement": {"abs_hinge": 0.1, "abs_slide": 0.05, "rel": 0.20},
    "t_open": {"abs_hinge": 0.3, "abs_slide": 0.3, "rel": 0.30},
    "t_open_bench": {"abs_hinge": 0.3, "abs_slide": 0.3, "rel": 0.30},
    "t_unlatch": {"abs_hinge": 0.2, "abs_slide": 0.2, "rel": 0.0},
    "operator_travel_reached": {"abs_hinge": 0.05, "abs_slide": 0.005, "rel": 0.10},
    "bolt_retract_max_frac": {"abs_hinge": 0.15, "abs_slide": 0.15, "rel": 0.0},
    "curve_rmse_primary": {"abs_hinge": 0.15, "abs_slide": 0.05, "rel": 0.0},
    # release
    "bolt_after_release_m": {"abs_hinge": 0.002, "abs_slide": 0.002, "rel": 0.0},
    "t_bolt_return": {"abs_hinge": 0.2, "abs_slide": 0.2, "rel": 0.0},
    "operator_after_release_frac": {"abs_hinge": 0.1, "abs_slide": 0.1, "rel": 0.0},
    # relatch
    "relatch_closed_angle": {"abs_hinge": math.radians(1.0), "abs_slide": 0.005, "rel": 0.0},
    "relatch_repush_angle": {"abs_hinge": math.radians(1.0), "abs_slide": 0.005, "rel": 0.0},
    "t_close": {"abs_hinge": 0.5, "abs_slide": 0.5, "rel": 0.30},
    "arrival_speed": {"abs_hinge": 0.2, "abs_slide": 0.1, "rel": 0.30},
    # closer
    "closer_final_angle": {"abs_hinge": math.radians(2.0), "abs_slide": 0.01, "rel": 0.0},
    "closer_t_close": {"abs_hinge": 0.5, "abs_slide": 0.5, "rel": 0.30},
    "peak_closing_speed": {"abs_hinge": 0.2, "abs_slide": 0.1, "rel": 0.30},
    "curve_rmse_closer": {"abs_hinge": 0.1, "abs_slide": 0.05, "rel": 0.0},
    # locked
    "locked_displacement": {"abs_hinge": 0.01, "abs_slide": 0.003, "rel": 0.0},
}
DEFAULT_TOLERANCE = {"abs_hinge": 0.05, "abs_slide": 0.02, "rel": 0.20}
# metrics that are informational only (never decide agreement)
NON_GATING_METRICS = {"t_start_s", "duration_s", "wall_time_s", "n_steps", "steps", "dt", "push", "qa_push", "effort", "eff", "warnings", "pen0_pair"}


def tolerance_for(metric: str, is_hinge: bool) -> tuple[float, float]:
    t = TOLERANCES.get(metric, DEFAULT_TOLERANCE)
    return t["abs_hinge"] if is_hinge else t["abs_slide"], t["rel"]


def metric_delta(metric: str, mj: Any, px: Any, is_hinge: bool, tol_key: str | None = None) -> dict | None:
    """[mj, px, delta, tol, ok] for numeric metrics; None when either side is missing / non-numeric.  `tol_key` looks the
    tolerance up under another name (a free-opening door's hold_displacement is judged like q_at_1s, not like a latched one)."""
    if metric in NON_GATING_METRICS:
        return None
    if isinstance(mj, bool) or isinstance(px, bool):
        if isinstance(mj, bool) and isinstance(px, bool):
            return {"mj": mj, "px": px, "delta": 0 if mj == px else 1, "tol": 0, "ok": mj == px}
        return None
    if not (_finite(mj) and _finite(px)):
        return None
    a, b = float(mj), float(px)
    abs_tol, rel_tol = tolerance_for(tol_key or metric, is_hinge)
    d = abs(a - b)
    ok = d <= abs_tol or (rel_tol > 0 and d <= rel_tol * max(abs(a), abs(b)))
    return {"mj": a, "px": b, "delta": a - b, "tol": abs_tol, "ok": bool(ok)}


# ---------------------------------------------------------------------------------------------------------------------
# discrepancy classes (codes -> meaning, likely root cause from the 40-door probe analysis, fix direction)

CLASS_INFO: dict[str, dict[str, str]] = {
    "OK": {"label": "parity", "meaning": "every applicable phase agrees within tolerance", "root_cause": "-", "fix": "-"},
    "QUANT": {"label": "quantitative", "meaning": "both simulators reach the same pass / fail verdicts but at least one metric is outside tolerance",
              "root_cause": "soft MuJoCo contacts / limits (solref 0.005) vs rigid PhysX limits, constraint-based vs effort-based Coulomb friction, implicitfast vs implicit drives at high damping; solver dt",
              "fix": "rerun at Isaac dt 1/240 (32/8 iterations) and MuJoCo dt 0.001; if the delta shrinks below tolerance tag SOLVER_SENSITIVITY, else triage by phase"},
    "EXPORT_COUPLING": {"label": "operator -> latch coupling missing", "meaning": "the operator turns but the bolt does not retract (or does not return) in PhysX, so the door stays latched",
                        "root_cause": "H1: the MJCF one-sided tendon (bolt_q >= scale * operator_q) is exported only as doorbench:couplings JSON / doorbench:latch_coupling; the runner must emulate it as a kinematic clamp each step (the soft target offset under-retracts by 40-60 %)",
                        "fix": "shared clamp function (write_joint_state_to_sim(max(latch_q, scale * op_q))) in the parity runner and DoorMechanismAction; read scale from doorbench:rl.latch_coupling / doorbench:latch_coupling_scale"},
    "EXPORT_WELD": {"label": "holding constraint missing in USD", "meaning": "MuJoCo holds the leaf (weld / lock equality) but PhysX has nothing holding it, so the door opens under the push",
                    "root_cause": "H5 (FIXED in the export): weld-type lock equalities (mag_lock / delayed_egress / electric_bolt / interlock leaf -> world) used to be exported only as doorbench:couplings JSON. Both USD kinds now carry a breakable UsdPhysics.FixedJoint base -> leaf with physics:excludeFromArticulation, breakForce == breakTorque == holding_force_N and physics:jointEnabled (doorbench:env_release). A remaining occurrence means the joint is absent (stale assets) or PhysX did not parse the loop joint",
                    "fix": "regenerate the dataset; if PhysX rejects an excludeFromArticulation joint between two articulation links, fall back to --emulate-weld and report it"},
    "EXPORT_FRAME": {"label": "export frame / COM / axis", "meaning": "body origins, COMs or joint anchors / axes differ between the MJCF and the USD at q0",
                     "root_cause": "usd.py frame composition (zero_offset, quaternion order, axis sign) or canonical-link welding in write_usd_rl",
                     "fix": "compare pose0 per body against model.json; fix the offending transform in doorbench/export/usd.py"},
    "PHYSICS_PARAM": {"label": "physics parameter mapping", "meaning": "the same mechanism behaves differently because a joint parameter maps differently (spring target / preload, friction, damping, mass)",
                      "root_cause": "H2 / D1 / D2: position targets written as zeros erase every USD drive target (springref preload) -> levers sag, closer preload lost, counterbalance lost; H3: joint Coulomb friction double-authored (legacy coefficient + per-axis efforts) or ignored",
                      "fix": "runner writes set_joint_position_target = spring target every step; read back stiffness / damping / friction / armature / mass against model.json before dynamics; author physxJoint:jointFriction = 0 for Isaac Sim >= 5.0"},
    "PHYSICS_PARAM_PRELOAD": {"label": "spring preload lost", "meaning": "settle drift or a false opening that matches a spring whose target was zeroed (operator sag q = tau_g / k, closer preload gone)",
                              "root_cause": "H2 / D1: Isaac Lab forwards zero position targets to the PhysX drives; the USD targetPosition (springref) is lost while the stiffness stays",
                              "fix": "restore doorbench:target_si / rl joints[*].target each step (as DoorMechanismAction does); report drift per joint"},
    "PHYSICS_PARAM_FRICTION": {"label": "friction / push magnitude", "meaning": "a free-swinging door opens in one simulator but not the other (timing or threshold), pointing at Coulomb friction or gravity bias mapping",
                               "root_cause": "H3: physxJointAxis friction efforts vs MuJoCo frictionloss (stick / slip differs), legacy jointFriction coefficient adding load-dependent friction, or an effort below the adaptive QA push",
                               "fix": "measure breakaway effort on one door in both sims; use the per-door qa_push; zero the legacy coefficient"},
    "PHYSICS_PARAM_DAMPING": {"label": "closer spring / damping mapping", "meaning": "the closer returns the door at a different speed or not at all",
                              "root_cause": "H2: closer target (springref, about -2 rad) erased or drive gains converted differently; explicit asymmetric damping at 120 Hz",
                              "fix": "keep spring targets; implicit damping via write_joint_damping_to_sim; dt <= 1/240"},
    "CONTACT_GEOMETRY": {"label": "geometry / contact behaves differently under PhysX", "meaning": "the bolt retracted (or there is no bolt) yet the leaf did not move, or a latch that holds in MuJoCo does not engage in PhysX (convex hulls, strike lip, panel clearance)",
                         "root_cause": "H7: PhysX contacts with Env prims at 0-5 mm clearance (contactOffset 5 mm), convex decomposition of hooks / strikes. The global self-collision part is FIXED in the export: physxArticulation:enabledSelfCollisions is True and every pair MuJoCo suppresses (same weld body, weld parent/child, contact_excludes) is authored as PhysxFilteredPairsAPI, so a latch holding one moving link against another (swing pairs, lift pins, drop bolts) now touches in PhysX too",
                         "fix": "enable contact reporting; rerun with Env collision disabled, then without the hardware part, to bisect frame contact vs articulation; check the authored filtered pairs against validate_usd_static.py"},
    "RL_CANON": {"label": "canonical RL export (welded / omitted parts, slot logic)", "meaning": "door.usda agrees with MuJoCo but door_rl.usda does not: a welded lock / operator / panel or an empty operator slot changes the behaviour",
                 "root_cause": "H4: panic doors with robot outside and no far-side trim get operator_joint None (exit device welded, latch never retracts); engaged locks with no canonical slot welded engaged; extra leaves omitted. Parts the operator retracts (revolute hooks, cremone shoot bolts, wheel-driven dogs) are welded RELEASED since the export fix, and every decision is recorded in doorbench:rl (welded / released_parts / released_holding / welded_engaged)",
                 "fix": "the RL expectation is derived from that ground truth in protocol.expected_outcomes (hold -> na when the only holding part is welded released, operate -> stays_closed when a lock part is welded engaged); a remaining RL_CANON is a documented structural limit of the 8-link articulation"},
    "VALIDATOR_PROTOCOL": {"label": "protocol / runner", "meaning": "the effort, schedule or expectation applied by a runner does not match the protocol (fixed 60 N*m instead of the adaptive push, 8 N on a slide operator, wrong lock expectation)",
                           "root_cause": "H3 / H4 / H6: fixed efforts below the door's static load (rollup 942 N, hatch 687 N*m, slider friction 95-157 N, drop bolt 12.6 N) or the expectation 'must open' for doors with no robot-side release",
                           "fix": "consume protocol_inputs/<door>.json (qa_push, operator efforts by joint type, door_flags expectations) in both runners"},
    "REFERENCE_QA_FAILURE": {"label": "reference does not reproduce qa.json", "meaning": "MuJoCo itself fails a phase that qa.json marks as passed",
                             "root_cause": "protocol drift from qa.py (schedule, efforts, reset) or nondeterminism", "fix": "align the MuJoCo runner with qa.py before comparing to PhysX"},
    "SOLVER_SENSITIVITY": {"label": "solver sensitivity", "meaning": "the discrepancy disappears at a finer time step / more solver iterations",
                           "root_cause": "integration / contact solver differences, not a dataset bug", "fix": "record; keep the finer settings for the gate if cheap"},
    "LIMITS": {"label": "joint left its range", "meaning": "a joint exceeded its authored range (plus tolerance) in one simulator",
               "root_cause": "hard PhysX limits on a heavy gravity-loaded joint parked at the limit, mimic units (rad vs deg gearing) driving a prismatic to its stop, or soft MuJoCo limits", "fix": "per-joint min / max per phase; check mimic gearing units on rot <-> trans couplings"},
    "SANITY": {"label": "numerical health", "meaning": "non-finite state, velocity cap hit (explosion) or a body flew away in one simulator",
               "root_cause": "stiff latch / lock drives at 120 Hz, initial penetration, mass / inertia rejected by PhysX", "fix": "dt <= 1/240; compare initial penetration; apply the MJCF inertia triangle guard in usd.py"},
    "LOAD_ERROR": {"label": "not comparable (load / spawn / structure)", "meaning": "the USD failed to spawn, or its joints / limits / gains do not match model.json, so no dynamic phase can be compared",
                   "root_cause": "export or Isaac Lab API problem", "fix": "see the runner's error text"},
    "UNTESTED": {"label": "untested", "meaning": "no Isaac result for this door and kind", "root_cause": "-", "fix": "run the Isaac runner on the GPU pod"},
}

ENV_RELEASE_LOCK_KINDS = ("mag_lock", "delayed_egress", "card_reader", "electric_strike", "interlock")
SLIDE_FAMILIES = ("sliding_single", "sliding_bypass", "gate_sliding", "rollup", "garage_sectional", "automatic_sliding", "elevator", "accordion", "stall_sliding")


# ---------------------------------------------------------------------------------------------------------------------
# door context (spec / manifest / qa.json), all optional

def door_context(assets_dir: str | None, door_id: str, manifest_entry: dict | None = None) -> dict:
    """What the classifier needs to know about a door: family, kinematics, hardware kinds, QA flags, qa.json checks."""
    ctx: dict[str, Any] = {"door_id": door_id, "family": None, "kin_type": None, "is_hinge": None, "latch_kind": None, "lock_kind": None,
                           "closer_kind": None, "operator_kind": None, "lock_engaged": None, "robot_side_release": None, "flags": {}, "qa_checks": {}, "qa_metrics": {},
                           "latch_model": None, "lock_model": None, "closer_model": None, "operator_model": None, "primary_joint": None, "operator_joint": None}
    me = manifest_entry or {}
    ctx["family"] = me.get("family") or (door_id.split("_", 1)[1] if "_" in door_id else None)
    ctx["latch_model"], ctx["lock_model"], ctx["closer_model"], ctx["operator_model"] = me.get("latch"), me.get("lock"), me.get("closer"), me.get("operator")
    ctx["lock_engaged"], ctx["robot_side_release"] = me.get("lock_engaged"), me.get("robot_side_release")
    spec = None
    if assets_dir:
        ddir = os.path.join(assets_dir, "doors", door_id)
        spec = _read_json(os.path.join(ddir, "spec.json"))
        qa = _read_json(os.path.join(ddir, "qa.json"))
        if isinstance(qa, dict):
            ctx["qa_checks"] = dict(qa.get("checks") or {})
            ctx["qa_metrics"] = {k: v for k, v in (qa.get("metrics") or {}).items() if isinstance(v, (int, float, bool))}
        model = _read_json(os.path.join(ddir, "model.json"))
        if isinstance(model, dict):
            meta = model.get("meta") or {}
            ctx["primary_joint"], ctx["operator_joint"] = meta.get("primary_joint"), meta.get("operator_joint")
    if isinstance(spec, dict):
        ctx["family"] = spec.get("family", ctx["family"])
        kin = spec.get("kinematics") or {}
        ctx["kin_type"] = kin.get("type")
        for k in ("latch", "lock", "closer", "operator"):
            ctx[f"{k}_model"] = (spec.get(k) or {}).get("model", ctx[f"{k}_model"])
        ctx["lock_engaged"] = (spec.get("lock") or {}).get("engaged", ctx["lock_engaged"])
        ctx["robot_side_release"] = (spec.get("lock") or {}).get("robot_side_release", ctx["robot_side_release"])
        try:
            from doorbench.qa import door_flags
            ctx["flags"] = door_flags(spec)
        except Exception:
            ctx["flags"] = {}
    # hardware kinds from the catalogue (model ids may be absent / unknown in synthetic tests)
    try:
        from doorbench import hardware as H
        ctx["latch_kind"] = H.LATCHES[ctx["latch_model"]].kind if ctx["latch_model"] in H.LATCHES else ctx["latch_model"]
        ctx["lock_kind"] = H.LOCKS[ctx["lock_model"]].kind if ctx["lock_model"] in H.LOCKS else ctx["lock_model"]
        ctx["closer_kind"] = H.CLOSERS[ctx["closer_model"]].kind if ctx["closer_model"] in H.CLOSERS else ctx["closer_model"]
        ctx["operator_kind"] = H.OPERATORS[ctx["operator_model"]].kind if ctx["operator_model"] in H.OPERATORS else ctx["operator_model"]
    except Exception:
        ctx["latch_kind"], ctx["lock_kind"], ctx["closer_kind"], ctx["operator_kind"] = ctx["latch_model"], ctx["lock_model"], ctx["closer_model"], ctx["operator_model"]
    if not ctx["flags"]:
        ctx["flags"] = {"lock_kind": ctx["lock_kind"], "latch_kind": ctx["latch_kind"],
                        "env_release_only": bool(ctx["lock_engaged"]) and ctx["lock_kind"] in ENV_RELEASE_LOCK_KINDS,
                        "spring_latch": ctx["latch_kind"] in ("tubular_latch", "deadlatch", "mortise_latch", "rim_latch", "vertical_rods", "hook", "gravity_bar", "dogs", "multi_bolt", "electric_bolt")}
    if ctx["kin_type"]:
        ctx["is_hinge"] = ctx["kin_type"].startswith("hinge") or ctx["kin_type"] == "rotor"
    elif ctx["family"]:
        ctx["is_hinge"] = ctx["family"] not in SLIDE_FAMILIES
    return ctx


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_hinge_of(ctx: dict, *recs: dict | None) -> bool:
    for rec in recs:
        if rec:
            m = rec.get("metrics") or {}
            if isinstance(m.get("is_hinge"), bool):
                return m["is_hinge"]
            jt = str(m.get("primary_type") or m.get("joint_type") or "").lower()
            if jt in ("hinge", "revolute", "rotor"):
                return True
            if jt in ("slide", "prismatic"):
                return False
    if ctx.get("is_hinge") is not None:
        return bool(ctx["is_hinge"])
    return True


# ---------------------------------------------------------------------------------------------------------------------
# comparison

def compare_phase(name: str, mj_ph: dict | None, px_ph: dict | None, is_hinge: bool) -> dict:
    """One phase of one door: statuses, agreement, metric deltas."""
    mj_st = mj_ph["status"] if mj_ph else "missing"
    px_st = px_ph["status"] if px_ph else "missing"
    out: dict[str, Any] = {"mujoco": mj_st, "physx": px_st, "expected": (mj_ph or {}).get("expected") or (px_ph or {}).get("expected"),
                           "metric_deltas": {}, "agree": None, "within_tol": None, "reason": None}
    if mj_st in ("skip", "missing") or px_st in ("skip", "missing"):
        out["agree"] = None     # not applicable / not run on one side
        out["reason"] = (px_ph or {}).get("reason") or (mj_ph or {}).get("reason")
        return out
    out["agree"] = mj_st == px_st
    mm, pm = mj_ph["metrics"], px_ph["metrics"]
    # a free-opening door's push phase compares displacements of tenths of a rad / m, not the latch slop
    hd_mj, hd_px = mm.get("hold_displacement"), pm.get("hold_displacement")
    free = name == "hold" and ("free" in str(out["expected"] or "").lower() or (_finite(hd_mj) and _finite(hd_px) and min(abs(float(hd_mj)), abs(float(hd_px))) > 0.1))
    for k in sorted(set(mm) & set(pm)):
        d = metric_delta(k, mm.get(k), pm.get(k), is_hinge, tol_key="q_at_1s" if free and k == "hold_displacement" else None)
        if d is not None:
            out["metric_deltas"][k] = d
    out["within_tol"] = all(d["ok"] for d in out["metric_deltas"].values()) if out["metric_deltas"] else True
    return out


def _phase_curve_rmse(mj: dict, px: dict, ctx: dict, phase: str, role: str) -> float | None:
    hints = [ctx.get("primary_joint") or ""] if role == "primary" else [ctx.get("operator_joint") or ""] if role == "operator" else []
    a = pick_curve(mj.get("curves") or {}, role, hints, phase)
    b = pick_curve(px.get("curves") or {}, role, hints, phase)
    return curve_rmse(a, b)


def grade_of(phases: dict[str, dict], load_error: bool) -> str:
    """A: all agree within tolerance; B: statuses agree, a metric is off; C: a status disagreement / limits / sanity fail; X: not comparable."""
    if load_error:
        return "X"
    applicable = [p for p in phases.values() if p["agree"] is not None]
    if not applicable:
        return "X"
    if any(p["agree"] is False for p in applicable):
        return "C"
    for name in ("limits", "sanity"):
        p = phases.get(name)
        if p and (p["mujoco"] == "fail" or p["physx"] == "fail"):
            return "C"
    if any(p["within_tol"] is False for p in applicable):
        return "B"
    return "A"


def classify(phases: dict[str, dict], mj: dict, px: dict, ctx: dict, kind: str, load_error: bool) -> list[dict]:
    """Discrepancy classes for one door / kind: [{code, phase, detail}] ordered by severity."""
    out: list[dict] = []
    flags = ctx.get("flags") or {}
    lock_kind, latch_kind = ctx.get("lock_kind"), ctx.get("latch_kind")
    engaged = bool(ctx.get("lock_engaged"))
    spring_latch = bool(flags.get("spring_latch"))

    def add(code: str, phase: str | None, detail: str):
        if not any(c["code"] == code and c["phase"] == phase for c in out):
            out.append({"code": code, "phase": phase, "detail": detail})

    if load_error:
        errs = "; ".join(px.get("errors") or [])[:200] or "structure check failed"
        add("LOAD_ERROR", "structure", errs)
        return out
    for name, p in phases.items():
        if p["agree"] is not False:
            continue
        mj_fail, px_fail = p["mujoco"] == "fail", p["physx"] == "fail"
        mm = (mj.get("phases", {}).get(name) or {}).get("metrics", {})
        pm = (px.get("phases", {}).get(name) or {}).get("metrics", {})
        if name == "pose0":
            add("EXPORT_FRAME", name, "body / COM / joint anchor frames differ at q0")
        elif name == "settle":
            add("PHYSICS_PARAM_PRELOAD" if px_fail else "PHYSICS_PARAM", name,
                f"drift mujoco={_fmt(mm.get('settle_drift') or mm.get('settle_drift_primary'))} physx={_fmt(pm.get('settle_drift') or pm.get('settle_drift_primary'))}")
        elif name == "hold":
            expected = str(p.get("expected") or ("hold" if flags.get("has_holding", True) else "free_opens")).lower()
            if "free" in expected:
                add("PHYSICS_PARAM_FRICTION", name, f"free push: mujoco {'opened' if not mj_fail else 'stuck'}, physx {'opened' if not px_fail else 'stuck'} (hold_displacement {_fmt(mm.get('hold_displacement'))} vs {_fmt(pm.get('hold_displacement'))})")
            elif px_fail and not mj_fail:
                if lock_kind in ENV_RELEASE_LOCK_KINDS and engaged:
                    add("EXPORT_WELD", name, f"{lock_kind} engaged: MuJoCo weld holds ({_fmt(mm.get('hold_displacement'))}), PhysX opened {_fmt(pm.get('hold_displacement'))}")
                elif engaged and lock_kind not in (None, "none"):
                    add("EXPORT_WELD", name, f"engaged {lock_kind} holds in MuJoCo, not in PhysX (lock constraint not exported)")
                else:
                    add("CONTACT_GEOMETRY", name, f"latch ({latch_kind}) holds in MuJoCo ({_fmt(mm.get('hold_displacement'))}), PhysX opened {_fmt(pm.get('hold_displacement'))}: bolt / strike contact not engaging")
            else:
                add("CONTACT_GEOMETRY", name, f"PhysX holds ({_fmt(pm.get('hold_displacement'))}) but MuJoCo opened {_fmt(mm.get('hold_displacement'))}")
        elif name == "operate_open":
            bolt_px = pm.get("bolt_retract_max_frac")
            op_px = pm.get("operator_travel_reached")
            op_mj = mm.get("operator_travel_reached")
            if px_fail and not mj_fail:
                if spring_latch and (bolt_px is None or (_finite(bolt_px) and float(bolt_px) < 0.5)):
                    add("EXPORT_COUPLING", name, f"operator moved (travel {_fmt(op_px)}) but bolt retracted {_fmt(bolt_px)} of its throw; MuJoCo opened {_fmt(mm.get('opened') or mm.get('actuate_displacement'))}")
                elif _finite(op_px) and _finite(op_mj) and float(op_mj) > 0 and float(op_px) < 0.5 * float(op_mj):
                    add("VALIDATOR_PROTOCOL", name, f"operator travel {_fmt(op_px)} vs {_fmt(op_mj)} in MuJoCo: effort too low for this operator type")
                elif latch_kind in (None, "none") and not engaged and not spring_latch:
                    add("PHYSICS_PARAM_FRICTION", name, f"nothing holds this door, yet PhysX opened only {_fmt(pm.get('opened') or pm.get('actuate_displacement'))} vs {_fmt(mm.get('opened') or mm.get('actuate_displacement'))}: push below the gravity / friction load, or friction mapped differently")
                else:
                    add("CONTACT_GEOMETRY", name, f"latch released (bolt {_fmt(bolt_px)}) but the leaf did not open in PhysX ({_fmt(pm.get('opened') or pm.get('actuate_displacement'))} vs {_fmt(mm.get('opened') or mm.get('actuate_displacement'))})")
            else:
                if engaged and (lock_kind in ENV_RELEASE_LOCK_KINDS or not ctx.get("robot_side_release", True)):
                    add("EXPORT_WELD", name, f"locked door ({lock_kind}) opened in PhysX ({_fmt(pm.get('opened') or pm.get('actuate_displacement'))}) but not in MuJoCo")
                elif kind == "rl":
                    add("RL_CANON", name, "door_rl.usda opened where MuJoCo did not (welded / released part in the canonical export)")
                else:
                    add("CONTACT_GEOMETRY", name, f"PhysX opened {_fmt(pm.get('opened') or pm.get('actuate_displacement'))} while MuJoCo stayed at {_fmt(mm.get('opened') or mm.get('actuate_displacement'))}")
        elif name == "release":
            add("EXPORT_COUPLING", name, f"bolt after release {_fmt(pm.get('bolt_after_release_m'))} m (PhysX) vs {_fmt(mm.get('bolt_after_release_m'))} m: latch clamp / spring target")
        elif name == "relatch":
            add("CONTACT_GEOMETRY", name, f"relatch closed {_fmt(pm.get('relatch_closed_angle'))} / repush {_fmt(pm.get('relatch_repush_angle'))} (PhysX) vs {_fmt(mm.get('relatch_closed_angle'))} / {_fmt(mm.get('relatch_repush_angle'))}: strike lip contact")
        elif name == "closer_return":
            add("PHYSICS_PARAM_DAMPING", name, f"closer final angle {_fmt(pm.get('closer_final_angle'))} (PhysX) vs {_fmt(mm.get('closer_final_angle'))} (MuJoCo)")
        elif name == "locked_holds":
            if px_fail and not mj_fail:
                add("EXPORT_WELD", name, f"locked door moved {_fmt(pm.get('locked_displacement'))} in PhysX vs {_fmt(mm.get('locked_displacement'))}")
            else:
                add("RL_CANON" if kind == "rl" else "CONTACT_GEOMETRY", name, f"locked displacement {_fmt(pm.get('locked_displacement'))} vs {_fmt(mm.get('locked_displacement'))}")
        elif name == "limits":
            add("LIMITS", name, "a joint left its range in " + ("PhysX" if px_fail else "MuJoCo"))
        elif name == "sanity":
            add("SANITY", name, "numerical health failed in " + ("PhysX" if px_fail else "MuJoCo"))
        elif name == "structure":
            add("LOAD_ERROR", name, "structure check disagrees")
        else:
            add("VALIDATOR_PROTOCOL", name, f"phase {name}: mujoco {p['mujoco']}, physx {p['physx']}")
    for name in ("limits", "sanity"):
        p = phases.get(name)
        if p and p["agree"] is True and p["mujoco"] == "fail":
            add(name.upper(), name, f"{name} failed in both simulators")
    # reference must reproduce qa.json
    for name, checks in PHASE_QA_CHECKS.items():
        p = phases.get(name)
        if p and p["mujoco"] == "fail" and any(ctx.get("qa_checks", {}).get(c) is True for c in checks):
            add("REFERENCE_QA_FAILURE", name, f"MuJoCo failed {name} although qa.json passed {[c for c in checks if ctx['qa_checks'].get(c)]}")
    if not out:
        quant = [n for n, p in phases.items() if p["within_tol"] is False]
        for n in quant:
            worst = max(phases[n]["metric_deltas"].items(), key=lambda kv: 0 if kv[1]["ok"] else abs(kv[1]["delta"]) / max(kv[1]["tol"], 1e-9))
            add("QUANT", n, f"{worst[0]}: mujoco {_fmt(worst[1]['mj'])} vs physx {_fmt(worst[1]['px'])} (tol {_fmt(worst[1]['tol'])})")
    return out


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x).lower()
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return str(v)
    return f"{v:.4g}"


def compare_kind(mj: dict | None, px: dict | None, ctx: dict, kind: str, variants: dict[str, dict] | None = None) -> dict:
    """Compare one door's MuJoCo record with one Isaac kind -> {status, grade, ok, phases, classes, metrics, curve_rmse}."""
    if px is None:
        return {"status": "untested", "grade": None, "ok": None, "phases": {}, "classes": [{"code": "UNTESTED", "phase": None, "detail": "no Isaac result"}], "metrics": {}}
    if mj is None:
        return {"status": "no_reference", "grade": "X", "ok": None, "phases": {}, "classes": [{"code": "LOAD_ERROR", "phase": None, "detail": "no MuJoCo reference result"}], "metrics": {}}
    is_hinge = is_hinge_of(ctx, mj, px)
    load_error = has_load_error(px)
    names = [p for p in PHASES if p in mj["phases"] or p in px["phases"]] + sorted((set(mj["phases"]) | set(px["phases"])) - set(PHASES))
    phases = {n: compare_phase(n, mj["phases"].get(n), px["phases"].get(n), is_hinge) for n in names}
    # curve RMSE on the operate phase (primary joint) and the closer phase, gated like metrics
    rm = _phase_curve_rmse(mj, px, ctx, "operate_open", "primary")
    if rm is not None and "operate_open" in phases and phases["operate_open"]["agree"] is not None:
        d = metric_delta("curve_rmse_primary", 0.0, rm, is_hinge)
        phases["operate_open"]["metric_deltas"]["curve_rmse_primary"] = d
        phases["operate_open"]["within_tol"] = phases["operate_open"]["within_tol"] and d["ok"]
    grade = grade_of(phases, load_error)
    classes = classify(phases, mj, px, ctx, kind, load_error)
    # sensitivity rerun (isaac_<kind>_<variant>.json): the discrepancy is solver-related if the finer run agrees
    if variants and grade in ("B", "C"):
        for vname, vrec in variants.items():
            vph = {n: compare_phase(n, mj["phases"].get(n), vrec["phases"].get(n), is_hinge) for n in names if n in vrec["phases"] or n in mj["phases"]}
            vg = grade_of(vph, has_load_error(vrec))
            if (grade == "C" and vg in ("A", "B")) or (grade == "B" and vg == "A"):
                classes.append({"code": "SOLVER_SENSITIVITY", "phase": None, "detail": f"agrees in the {vname} rerun"})
                break
    metrics = {}
    for n, p in phases.items():
        for k, d in p["metric_deltas"].items():
            metrics[f"{n}.{k}"] = [d["mj"], d["px"], d["ok"]]
    ok = grade in ("A", "B")
    return {"status": "compared" if grade != "X" else "not_comparable", "grade": grade, "ok": ok, "phases": phases, "classes": classes, "metrics": metrics,
            "curve_rmse_primary": rm, "errors": list(px.get("errors") or [])[:5], "is_hinge": is_hinge}


def door_verdict(door_id: str, mj: dict | None, px_by_kind: dict[str, dict | None], ctx: dict, variants: dict[str, dict[str, dict]] | None = None) -> dict:
    """Join one door across kinds -> the record published in qa.json["isaac_parity"] and results/parity/summary.json."""
    kinds = {}
    for kind in KINDS:
        kinds[kind] = compare_kind(mj, px_by_kind.get(kind), ctx, kind, (variants or {}).get(kind))
    tested = [k for k, v in kinds.items() if v["status"] != "untested"]
    # a phase that agrees in the full USD but not in the canonical one points at the RL welding / slot logic
    if kinds["full"]["status"] == "compared" and kinds["rl"]["status"] == "compared":
        for n, p in kinds["rl"]["phases"].items():
            pf = kinds["full"]["phases"].get(n)
            if p["agree"] is False and pf and pf["agree"] is True and not any(c["code"] == "RL_CANON" and c["phase"] == n for c in kinds["rl"]["classes"]):
                kinds["rl"]["classes"].append({"code": "RL_CANON", "phase": n, "detail": f"{n} agrees in door.usda but not in door_rl.usda"})
    grades = [kinds[k]["grade"] for k in tested if kinds[k]["grade"]]
    order = {"A": 0, "B": 1, "C": 2, "X": 3}
    grade = max(grades, key=lambda g: order.get(g, 3)) if grades else None
    classes: list[str] = []
    for k in KINDS:
        for c in kinds[k]["classes"]:
            if c["code"] not in classes and c["code"] != "UNTESTED":
                classes.append(c["code"])
    if not tested:
        status, ok = "untested", None
    elif mj is None:
        status, ok = "no_reference", None
    else:
        status = "compared"
        ok = grade in ("A", "B")
    primary = next((c for c in classes if c not in ("QUANT", "SOLVER_SENSITIVITY")), classes[0] if classes else "OK")
    return {
        "door_id": door_id, "status": status, "ok": ok, "grade": grade, "family": ctx.get("family"), "kinematics": ctx.get("kin_type"),
        "hardware": {"latch": ctx.get("latch_kind"), "lock": ctx.get("lock_kind"), "closer": ctx.get("closer_kind"), "operator": ctx.get("operator_kind"), "lock_engaged": ctx.get("lock_engaged")},
        "kinds": {k: {"status": v["status"], "grade": v["grade"], "ok": v["ok"],
                      "phases": {n: ("agree" if p["agree"] is True and p["within_tol"] is not False else "quant" if p["agree"] is True else "disagree" if p["agree"] is False else "na") for n, p in v["phases"].items()},
                      "classes": [c["code"] for c in v["classes"] if c["code"] != "UNTESTED"], "details": [f"{c['phase'] or '-'}: {c['detail']}" for c in v["classes"] if c["code"] != "UNTESTED"],
                      "metrics": v["metrics"], "errors": v.get("errors", [])} for k, v in kinds.items()},
        "classes": classes, "primary_class": primary if status == "compared" else ("UNTESTED" if status == "untested" else "LOAD_ERROR"),
        "likely_root_cause": CLASS_INFO.get(primary, {}).get("root_cause", "-") if status == "compared" and primary != "OK" else "-",
        "_kinds_full": kinds,
    }


def manifest_status(verdict: dict | None) -> str:
    """qa.json / manifest badge value: ok (grade A or B) | fail (C or not comparable) | untested."""
    if not verdict or verdict.get("status") == "untested":
        return "untested"
    if verdict.get("status") == "no_reference":
        return "untested"
    return "ok" if verdict.get("ok") else "fail"


def collect_result_files(results_dir: str) -> dict:
    """{'mujoco': path|None, 'mujoco_variants': {name: path}, 'isaac': {kind: path}, 'isaac_variants': {kind: {variant: path}}}."""
    out: dict[str, Any] = {"mujoco": None, "mujoco_variants": {}, "isaac": {}, "isaac_variants": {}}
    if not os.path.isdir(results_dir):
        return out
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json") or fn in ("summary.json",):
            continue
        path = os.path.join(results_dir, fn)
        sim, kind, variant = kind_from_filename(path)
        if sim == "mujoco":
            if variant:
                out["mujoco_variants"][variant] = path
            else:
                out["mujoco"] = path
        elif sim == "isaac" and kind:
            if variant:
                out["isaac_variants"].setdefault(kind, {})[variant] = path
            else:
                out["isaac"][kind] = path
    return out

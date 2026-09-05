"""Kinematic feasibility of active point-connection mechanisms.

Leaf joints are inputs. Every free mechanism joint on either side of a connect
constraint may move, including a sliding frame shoe. Equality-driven joints
(such as a rising-hinge carrier) remain prescribed; solving them away would
hide an impossible loop. This proves geometric feasibility, not force fidelity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


RESIDUAL_TOL_M = 0.001  # maximum endpoint separation, shared with the viewer


def check_linkage_model(model, model_json: dict, n_steps: int = 24) -> dict:
    import mujoco

    data = mujoco.MjData(model)
    joints = {b["joint"]["name"]: b["joint"] for b in model_json["bodies"] if b.get("joint")}
    coupled = {e["a"] for e in model_json.get("equalities", []) if e["kind"] == "joint" and e.get("active", True)}
    coupled.update(t["sites"][0][0] for t in model_json.get("tendons", []) if t.get("sites"))
    model_ids = {model.joint(j).name: j for j in range(model.njnt)}
    inputs = [model_ids[n] for n, j in joints.items() if n in model_ids and j.get("role") in ("primary", "secondary")]
    equations = [i for i in range(model.neq) if model.eq_active0[i] and int(model.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT)]
    loops = [i for i in range(model.neq) if model.eq_active0[i] and int(model.eq_type[i]) == int(mujoco.mjtEq.mjEQ_CONNECT)]

    def resolve(q):
        # MuJoCo equalities are relative to qpos0/ref, unlike an absolute mimic.
        for _ in range(max(2, len(equations))):
            for e in equations:
                a, b = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
                aa = model.jnt_qposadr[a]
                x = q[model.jnt_qposadr[b]] - model.qpos0[model.jnt_qposadr[b]] if b >= 0 else 0.0
                q[aa] = model.qpos0[aa] + sum(model.eq_data[e][k] * x ** k for k in range(5))
        return q

    def chain(body):
        path = []
        while body > 0:
            path.append(body)
            body = int(model.body_parentid[body])
        return path

    descriptions = []
    all_owned = set()
    failures = []
    for e in loops:
        a, b = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        ca, cb = chain(a), chain(b)
        common = next((body for body in ca if body in cb), None)
        bodies = set(ca[:ca.index(common)] if common is not None else ca)
        bodies.update(cb[:cb.index(common)] if common is not None else cb)
        candidates = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in bodies
                      and model.joint(j).name in joints and model.joint(j).name not in coupled]
        owned = [j for j in candidates if joints[model.joint(j).name].get("role") == "mechanism"]
        if not owned:
            owned = [j for j in candidates if not joints[model.joint(j).name].get("robot_interactive", True)
                     and joints[model.joint(j).name].get("role") not in ("primary", "secondary", "operator")]
        owned = [j for j in owned if not model.jnt_limited[j] or np.ptp(model.jnt_range[j]) > 1e-9]
        descriptions.append((e, a, b, owned))
        all_owned.update(owned)

    # Solve all endpoints together so a shared mechanism cannot satisfy one loop
    # while silently breaking another. Keep warm starts separate for each sweep.
    owned = sorted(all_owned)
    addresses = [int(model.jnt_qposadr[j]) for j in owned]
    lower = np.array([model.jnt_range[j][0] if model.jnt_limited[j] else -np.inf for j in owned])
    upper = np.array([model.jnt_range[j][1] if model.jnt_limited[j] else np.inf for j in owned])

    def residual(x):
        data.qpos[addresses] = x
        # A free arm may itself drive another arm. Reapply its dependents on
        # every optimizer evaluation, not only before starting the solve.
        data.qpos[:] = resolve(data.qpos.copy())
        mujoco.mj_kinematics(model, data)
        offsets = []
        for e, a, b, _ in descriptions:
            tip = data.xpos[a] + data.xmat[a].reshape(3, 3) @ model.eq_data[e][:3]
            anchor = data.xpos[b] + data.xmat[b].reshape(3, 3) @ model.eq_data[e][3:6]
            offsets.extend(tip - anchor)
        return np.asarray(offsets)

    configurations = [[("rest", [])]]
    for driver in inputs:
        lo, hi = model.jnt_range[driver] if model.jnt_limited[driver] else (-np.pi, np.pi)
        configurations.append([(model.joint(driver).name, [(driver, float(q))])
                               for q in np.linspace(lo, hi, max(1, n_steps) + 1)])
    worst = 0.0
    samples = 0
    # Rest plus independent leaf sweeps. Each item is one prescribed input.
    for sweep in configurations:
        guess = model.qpos0[addresses].copy()
        for driver, values in sweep:
            data.qpos[:] = model.qpos0
            for j, q in values:
                data.qpos[model.jnt_qposadr[j]] = q
            data.qpos[:] = resolve(data.qpos.copy())
            if owned and descriptions:
                guess = np.clip(guess, lower, upper)
                fit = least_squares(residual, guess, bounds=(lower, upper),
                                    ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100)
                guess = fit.x
            distances = np.linalg.norm(residual(guess).reshape(-1, 3), axis=1) if descriptions else []
            samples += 1
            for (e, _, _, loop_owned), distance in zip(descriptions, distances):
                worst = max(worst, float(distance))
                if not np.isfinite(distance) or distance >= RESIDUAL_TOL_M:
                    failures.append({"equality": model.equality(e).name, "driver": driver,
                                     "q": values[0][1] if values else 0.0, "residual_m": float(distance),
                                     "mechanism_joints": [model.joint(j).name for j in loop_owned]})
    return {"ok": not failures, "n_loops": len(loops), "n_samples": samples,
            "max_residual_m": worst, "failures": failures[:40], "n_failures": len(failures)}


def run_linkage_qa(door_dir: str, n_steps: int = 24) -> dict:
    """Report a load/solver failure as failed QA, never as a skipped success."""
    try:
        import mujoco
        directory = Path(door_dir)
        model_json = json.loads((directory / "model.json").read_text())
        if not any(e["kind"] == "connect" and e.get("active", True) for e in model_json.get("equalities", [])):
            return {"ok": True, "n_loops": 0, "n_samples": 0, "max_residual_m": 0.0, "n_failures": 0, "failures": []}
        model = mujoco.MjModel.from_xml_path(str(directory / "door.xml"))
        return check_linkage_model(model, model_json, n_steps)
    except Exception as exc:
        return {"ok": False, "n_loops": 0, "n_samples": 0, "n_failures": 1,
                "failures": [{"error": f"{type(exc).__name__}: {exc}"}], "max_residual_m": None}

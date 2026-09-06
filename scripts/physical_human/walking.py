"""Contact-constrained whole-body walking for the one-door native prototype.

An offline linear inverted-pendulum plan keeps the centre of pressure in the
planned foot support polygon. A bounded inverse-dynamics QP tracks that centre
of mass and the feet through joint motors. The floating root is never actuated.
This is an engineered controller, not calibrated human biomechanics.
"""

from __future__ import annotations

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from doorbench.reference.gait import plan_walk


def yaw_matrix(yaw):
    return Rotation.from_rotvec([0, 0, yaw]).as_matrix()


def rotation_error(goal, actual):
    return Rotation.from_matrix(goal @ actual.T).as_rotvec()


def centroidal_plan(time, feet, rotations, contacts, initial_com, initial_velocity):
    """Optimize planar COM position, velocity and acceleration with bounded ZMP."""
    n = len(time)
    dt = time[1] - time[0]
    # [px, py, vx, vy, ax, ay] at each sample.
    size = 6 * n
    reference = feet[:, :, :2].mean(axis=1)
    weights = np.tile([8.0, 8.0, 0.02, 0.02, 0.005, 0.005], n)
    linear = np.zeros(size)
    for k in range(n):
        linear[6 * k : 6 * k + 2] = -8 * reference[k]
    objective = sparse.diags(weights, format="csc")
    derivative_rows, derivative_cols, derivative_values = [], [], []
    for k in range(n - 1):
        for axis in range(2):
            row = 2 * k + axis
            derivative_rows.extend([row, row])
            derivative_cols.extend([6 * k + 4 + axis, 6 * (k + 1) + 4 + axis])
            derivative_values.extend([-1 / dt, 1 / dt])
    jerk = sparse.coo_matrix(
        (derivative_values, (derivative_rows, derivative_cols)),
        shape=(2 * (n - 1), size),
    ).tocsc()
    objective += 0.004 * jerk.T @ jerk
    rr, cc, vv, lower, upper = [], [], [], [], []

    def constraint(indices, values, lo, hi):
        row = len(lower)
        rr.extend([row] * len(indices))
        cc.extend(indices)
        vv.extend(values)
        lower.append(lo)
        upper.append(hi)

    for k in range(n - 1):
        for axis in range(2):
            constraint(
                [6 * (k + 1) + axis, 6 * k + axis, 6 * k + 2 + axis, 6 * k + 4 + axis],
                [1, -1, -dt, -0.5 * dt * dt],
                0,
                0,
            )
            constraint(
                [6 * (k + 1) + 2 + axis, 6 * k + 2 + axis, 6 * k + 4 + axis],
                [1, -1, -dt],
                0,
                0,
            )
    for axis in range(2):
        constraint([axis], [1], initial_com[axis], initial_com[axis])
        constraint([2 + axis], [1], initial_velocity[axis], initial_velocity[axis])
        constraint([6 * (n - 1) + axis], [1], reference[-1, axis], reference[-1, axis])
        constraint([6 * (n - 1) + 2 + axis], [1], 0, 0)
        constraint([6 * (n - 1) + 4 + axis], [1], 0, 0)
    height = initial_com[2]
    for k in range(n):
        corners = []
        for side in range(2):
            if not contacts[k, side]:
                continue
            for x in [-0.043, 0.043]:
                for y in [-0.115, 0.115]:
                    corners.append(
                        (feet[k, side] + rotations[k, side] @ np.array([x, y, 0]))[:2]
                    )
        hull = ConvexHull(corners)
        for a, b, c in hull.equations:
            constraint(
                [6 * k, 6 * k + 1, 6 * k + 4, 6 * k + 5],
                [a, b, -height / 9.81 * a, -height / 9.81 * b],
                -np.inf,
                -c,
            )
        for axis in range(2):
            constraint([6 * k + 4 + axis], [1], -3, 3)
    matrix = sparse.coo_matrix((vv, (rr, cc)), shape=(len(lower), size)).tocsc()
    solver = osqp.OSQP()
    solver.setup(
        P=objective,
        q=linear,
        A=matrix,
        l=np.array(lower),
        u=np.array(upper),
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iter=15000,
        polishing=True,
    )
    result = solver.solve()
    if result.info.status_val not in (1, 2):
        raise RuntimeError("COM plan: " + result.info.status)
    state = result.x.reshape(n, 6)
    return (
        np.c_[state[:, :2], np.full(n, height)],
        np.c_[state[:, 2:4], np.zeros(n)],
        np.c_[state[:, 4:], np.zeros(n)],
    )


def make_plan(
    model, data, waypoints, headings=None, step_duration=0.65, step_length=0.40
):
    ankles = [model.body("actor_ankle_" + s).id for s in "lr"]
    root = model.body("actor_pelvis").id
    start = data.xpos[ankles, :2].mean(axis=0)
    yaw = Rotation.from_matrix(data.xmat[root].reshape(3, 3)).as_euler("xyz")[2]
    width = np.linalg.norm(data.xpos[ankles[0], :2] - data.xpos[ankles[1], :2])
    gait = plan_walk(
        start,
        yaw,
        waypoints,
        waypoint_yaws=headings,
        blend_turns=True,
        fps=50,
        step_length=step_length,
        step_duration=step_duration,
        stance_width=width,
    )
    rotations = (
        Rotation.from_quat(gait["foot_quat"][:, :, [1, 2, 3, 0]].reshape(-1, 4))
        .as_matrix()
        .reshape(-1, 2, 3, 3)
    )
    soles = gait["foot_pos"] + np.einsum(
        "nsij,j->nsi", rotations, np.array([0, 0.04, -0.055])
    )
    soles, foot_adjustments, clearance_before, clearance_after = avoid_foot_obstacles(
        model, data, gait["time"], soles, rotations, gait["foot_contact"]
    )
    # Allow the planner to finish moving before asking it to settle at rest.
    extra = 50
    gait["time"] = np.arange(len(soles) + extra) / 50
    soles = np.concatenate([soles, np.repeat(soles[-1:], extra, axis=0)])
    rotations = np.concatenate([rotations, np.repeat(rotations[-1:], extra, axis=0)])
    contact = np.concatenate([gait["foot_contact"], np.ones((extra, 2), bool)])
    yaw = np.r_[gait["pelvis_yaw"], np.repeat(gait["pelvis_yaw"][-1], extra)]
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, jac, root)
    com, velocity, acceleration = centroidal_plan(
        gait["time"], soles, rotations, contact, data.subtree_com[root], jac @ data.qvel
    )
    foot_velocity = np.gradient(soles, 0.02, axis=0)
    foot_acceleration = np.gradient(foot_velocity, 0.02, axis=0)
    return {
        "foot_adjustments": foot_adjustments,
        "planned_foot_clearance_before_m": clearance_before,
        "planned_foot_clearance_after_m": clearance_after,
        "time": gait["time"],
        "feet": soles,
        "rotations": rotations,
        "contacts": contact,
        "yaw": yaw,
        "com": com,
        "com_velocity": velocity,
        "com_acceleration": acceleration,
        "foot_velocity": foot_velocity,
        "foot_acceleration": foot_acceleration,
    }


class WholeBodyWalk:
    """Return motor torques, respecting floating dynamics, support and torque bounds."""

    def __init__(self, model, data, plan, posture):
        self.model = model
        self.plan = plan
        self.posture = posture.copy()
        self.root = model.body("actor_pelvis").id
        self.soles = [model.site("actor_site_sole_" + s).id for s in "lr"]
        self.feet = [model.body("actor_ankle_" + s).id for s in "lr"]
        self.motors = np.array(
            [
                i
                for i in range(model.nu)
                if not model.joint(model.actuator_trnid[i, 0]).name.startswith("hand_")
            ]
        )
        self.qids = model.jnt_qposadr[model.actuator_trnid[self.motors, 0]]
        self.vids = model.jnt_dofadr[model.actuator_trnid[self.motors, 0]]
        start = model.joint("actor_root").dofadr[0]
        self.v = np.r_[np.arange(start, start + 6), self.vids]
        self.n = len(self.v)
        self.size = self.n + 12
        self.M = np.zeros((model.nv, model.nv))
        self.last_com_jac = None
        self.solver = None
        self.previous = None
        self.failures = []
        self.max_residual = 0
        self.dt = 0.005
        self.mass = model.body_subtreemass[self.root]
        self.last_target = None

    def target(self, t):
        k = int(np.clip(t / 0.02, 0, len(self.plan["time"]) - 2))
        u = np.clip((t - self.plan["time"][k]) / 0.02, 0, 1)
        out = {
            n: self.plan[n][k] * (1 - u) + self.plan[n][k + 1] * u
            for n in [
                "feet",
                "com",
                "com_velocity",
                "com_acceleration",
                "foot_velocity",
                "foot_acceleration",
                "yaw",
            ]
        }
        out["rotations"] = self.plan["rotations"][k]
        out["contacts"] = self.plan["contacts"][k]
        return out

    def solve(self, data, t):
        m = self.model
        goal = self.target(t)
        nv = m.nv
        n = self.n
        size = self.size
        mujoco.mj_fullM(m, data, self.M)
        mass = self.M[np.ix_(self.v, self.v)]
        bias = (data.qfrc_bias - data.qfrc_passive)[self.v]
        jacobians = []
        derivatives = []
        foot_tasks = []
        for side, (site, body) in enumerate(zip(self.soles, self.feet)):
            jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
            dp, dr = jp.copy(), jr.copy()
            mujoco.mj_jacSite(m, data, jp, jr, site)
            mujoco.mj_jacDot(m, data, dp, dr, data.site_xpos[site], body)
            jac = np.r_[jp, jr][:, self.v]
            derivative = np.r_[dp, dr] @ data.qvel
            current = data.site_xmat[site].reshape(3, 3)
            velocity = np.r_[jp, jr] @ data.qvel
            acc = (
                np.r_[
                    goal["foot_acceleration"][side]
                    + 350 * (goal["feet"][side] - data.site_xpos[site])
                    + 35 * (goal["foot_velocity"][side] - velocity[:3]),
                    100 * rotation_error(goal["rotations"][side], current)
                    - 20 * velocity[3:],
                ]
                - derivative
            )
            jacobians.append(jac)
            derivatives.append(derivative)
            foot_tasks.append(acc)
        contact_jac = np.concatenate(jacobians)
        # Quadratic task errors, each in physical acceleration units.
        H = np.eye(size) * 1e-5
        linear = np.zeros(size)

        def task(matrix, target, weight):
            nonlocal H, linear
            weighted = matrix * weight
            H += weighted.T @ weighted
            linear -= weighted.T @ (target * weight)

        com_jac = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(m, data, com_jac, self.root)
        jdotv = (
            np.zeros(3)
            if self.last_com_jac is None
            else (com_jac - self.last_com_jac) @ data.qvel / self.dt
        )
        self.last_com_jac = com_jac.copy()
        acc = (
            goal["com_acceleration"]
            + 90 * (goal["com"] - data.subtree_com[self.root])
            + 18 * (goal["com_velocity"] - com_jac @ data.qvel)
            - jdotv
        )
        matrix = np.zeros((3, size))
        matrix[:, :n] = com_jac[:, self.v]
        task(matrix, acc, 12)
        jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
        dp, dr = jp.copy(), jr.copy()
        mujoco.mj_jacBody(m, data, jp, jr, self.root)
        mujoco.mj_jacDot(m, data, dp, dr, data.xpos[self.root], self.root)
        acc = (
            90
            * rotation_error(
                yaw_matrix(goal["yaw"]), data.xmat[self.root].reshape(3, 3)
            )
            - 18 * jr @ data.qvel
            - dr @ data.qvel
        )
        matrix = np.zeros((3, size))
        matrix[:, :n] = jr[:, self.v]
        task(matrix, acc, 10)
        # Penalize achieved torso lean relative to gravity, independently of
        # pelvis orientation. This prevents balance corrections arching the back.
        chest = m.body("actor_chest").id
        jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
        dp, dr = jp.copy(), jr.copy()
        mujoco.mj_jacBody(m, data, jp, jr, chest)
        mujoco.mj_jacDot(m, data, dp, dr, data.xpos[chest], chest)
        chest_yaw = goal["yaw"] + self.posture[m.joint("actor_spine_yaw").qposadr[0]]
        acc = (
            90 * rotation_error(yaw_matrix(chest_yaw), data.xmat[chest].reshape(3, 3))
            - 18 * jr @ data.qvel
            - dr @ data.qvel
        )
        matrix = np.zeros((3, size))
        matrix[:, :n] = jr[:, self.v]
        task(matrix[:2], acc[:2], 12)
        task(matrix[2:], acc[2:], 3)
        posture_acc = (
            50 * (self.posture[self.qids] - data.qpos[self.qids])
            - 12 * data.qvel[self.vids]
        )
        matrix = np.zeros((len(self.motors), size))
        matrix[:, 6:n] = np.eye(len(self.motors))
        task(matrix, posture_acc, 0.5)
        for side in range(2):
            matrix = np.zeros((6, size))
            matrix[:, :n] = jacobians[side]
            task(matrix, foot_tasks[side], 40 if goal["contacts"][side] else 18)
        # Force regularization chooses a moderate distribution in double support.
        H[n:, n:] += np.eye(12) * 1e-5
        rows = []
        lower = []
        upper = []

        def constraint(matrix, lo, hi):
            rows.extend(np.atleast_2d(matrix))
            lower.extend(np.broadcast_to(lo, (len(np.atleast_2d(matrix)),)))
            upper.extend(np.broadcast_to(hi, (len(np.atleast_2d(matrix)),)))

        # Floating-body Newton/Euler equations; only joint motors appear below.
        dynamics = np.c_[mass, -contact_jac.T]
        constraint(dynamics[:6], -bias[:6], -bias[:6])
        torque = dynamics[6:]
        constraint(
            torque,
            m.actuator_forcerange[self.motors, 0] - bias[6:],
            m.actuator_forcerange[self.motors, 1] - bias[6:],
        )
        for side in range(2):
            wrench = np.zeros((6, size))
            wrench[:, n + 6 * side : n + 6 * side + 6] = np.eye(6)
            rot = goal["rotations"][side].T
            local = np.zeros((6, 6))
            local[:3, :3] = rot
            local[3:, 3:] = rot
            wrench = local @ wrench
            if not goal["contacts"][side]:
                constraint(wrench, 0, 0)
                continue
            constraint(wrench[2], 0, 1200)
            for axis, limit in [(0, 0.7), (1, 0.7), (3, 0.115), (4, 0.043), (5, 0.02)]:
                constraint(wrench[axis] - limit * wrench[2], -np.inf, 0)
                constraint(-wrench[axis] - limit * wrench[2], -np.inf, 0)
        A = np.array(rows)
        lower = np.array(lower)
        upper = np.array(upper)
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(H * 0.001),
            q=linear * 0.001,
            A=sparse.csc_matrix(A),
            l=lower,
            u=upper,
            verbose=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=10000,
            polishing=True,
        )
        if self.previous is not None:
            solver.warm_start(x=self.previous)
        result = solver.solve()
        if result.info.status_val not in (1, 2):
            self.failures.append({"time": t, "status": result.info.status})
            raise RuntimeError(f"Whole body control at {t:.3f}s: " + result.info.status)
        self.previous = result.x
        self.max_residual = max(self.max_residual, result.info.prim_res)
        torques = torque @ result.x + bias[6:]
        self.last_target = goal
        return self.motors, torques


def avoid_foot_obstacles(model, data, time, feet, rotations, contacts, margin=0.018):
    """Bend colliding swing paths using native box/environment distance queries.

    Footprints are corrected before execution, then joined with C2 swing paths.
    A stance pose stays fixed throughout its own contact phase.
    This plans geometry in separate scratch buffers; it never moves simulation
    state. The full native rollout independently checks achieved contacts.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    mujoco.mj_forward(model, scratch)
    world = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g]
        and model.geom(g).name != "floor"
        and not (model.body(model.geom_bodyid[g]).name or "").startswith(
            ("actor_", "hand_")
        )
    ]
    foot_geoms = [model.geom("actor_geom_foot_" + side).id for side in "lr"]
    corrected = feet.copy()
    adjustments = []
    before = np.inf
    after = np.inf

    def clearance(side, k, position):
        g = foot_geoms[side]
        rotation = rotations[k, side]
        scratch.geom_xpos[g] = position + rotation @ np.array([0, 0, 0.0275])
        scratch.geom_xmat[g] = rotation.ravel()
        return min(
            mujoco.mj_geomDistance(model, scratch, g, w, 0.30, None) for w in world
        )

    offsets = sorted(
        (
            np.array([x, y])
            for x in np.arange(-0.06, 0.0601, 0.005)
            for y in np.arange(-0.06, 0.0601, 0.005)
        ),
        key=lambda xy: float(xy @ xy),
    )
    # First repair any planted footprint. Propagate each correction through
    # the preceding/following swing with zero velocity and acceleration at
    # contact transitions; a stance target is never moved while planted.
    baseline_feet = feet.copy()
    for side in range(2):
        planted = contacts[:, side].astype(bool)
        starts = np.flatnonzero(planted & ~np.r_[False, planted[:-1]])
        offset_track = np.zeros((len(time), 2))
        for start in starts:
            finish = start
            while finish + 1 < len(time) and planted[finish + 1]:
                finish += 1
            gap = clearance(side, start, feet[start, side])
            if gap < margin:
                if start == 0:
                    raise RuntimeError("The initial planted foot lacks clearance")
                for offset in offsets:
                    position = feet[start, side].copy()
                    position[:2] += offset
                    fixed_gap = clearance(side, start, position)
                    if fixed_gap >= margin:
                        offset_track[start : finish + 1] = offset
                        adjustments.append(
                            [
                                0,
                                side,
                                float(time[start]),
                                float(time[finish]),
                                *offset.tolist(),
                                gap,
                                fixed_gap,
                            ]
                        )
                        break
                else:
                    raise RuntimeError(
                        "No clear planted footprint within 6 cm per axis"
                    )
        swing = ~planted
        swing_starts = np.flatnonzero(swing & ~np.r_[False, swing[:-1]])
        for start in swing_starts:
            finish = start
            while finish < len(time) - 1 and swing[finish]:
                finish += 1
            origin = max(0, start - 1)
            indices = np.arange(origin, finish + 1)
            amount = (time[indices] - time[origin]) / (time[finish] - time[origin])
            ease = amount**3 * (10 - 15 * amount + 6 * amount**2)
            offset_track[indices] = offset_track[origin] + ease[:, None] * (
                offset_track[finish] - offset_track[origin]
            )
        baseline_feet[:, side, :2] += offset_track
    corrected = baseline_feet.copy()
    for side in range(2):
        swing = ~contacts[:, side].astype(bool)
        starts = np.flatnonzero(swing & ~np.r_[False, swing[:-1]])
        for start in starts:
            finish = start
            while finish < len(time) - 1 and swing[finish]:
                finish += 1
            origin = max(0, start - 1)
            indices = np.arange(origin, finish + 1)
            amount = (time[indices] - time[origin]) / (time[finish] - time[origin])
            bump = 64 * amount**3 * (1 - amount) ** 3
            gap = min(clearance(side, k, baseline_feet[k, side]) for k in indices)
            before = min(
                before, min(clearance(side, k, feet[k, side]) for k in indices)
            )
            if gap >= margin:
                after = min(after, gap)
                continue
            if (
                min(
                    clearance(side, k, baseline_feet[k, side]) for k in [origin, finish]
                )
                < margin - 1e-10
            ):
                raise RuntimeError("A planted foot needs a different route")
            for offset in offsets:
                trial = baseline_feet[indices, side].copy()
                trial[:, :2] += bump[:, None] * offset
                distances = []
                for k, position in zip(indices, trial):
                    distances.append(clearance(side, k, position))
                    if distances[-1] < margin:
                        break
                if len(distances) == len(indices) and min(distances) >= margin:
                    corrected[indices, side] = trial
                    adjustments.append(
                        [
                            1,
                            side,
                            float(time[origin]),
                            float(time[finish]),
                            *offset.tolist(),
                            gap,
                            min(distances),
                        ]
                    )
                    after = min(after, min(distances))
                    break
            else:
                raise RuntimeError(
                    "No swing-foot clearance solution within 6 cm per axis"
                )
    return (
        corrected,
        np.asarray(adjustments).reshape(-1, 8),
        float(before),
        float(after),
    )

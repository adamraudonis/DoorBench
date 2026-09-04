"""Deterministic attachment gate: nothing floats, everything is mounted on something, every mechanism moves.

The clearance gate (clearance.py) catches parts that pass *through* each other.  This gate catches the opposite
family of defects, the ones a viewer shows just as clearly: parts that hang in the air, hardware that is not
mounted on anything, closer arms that leave their bracket when the door swings, bolts that never retract, meshes
rendered upside-down, duplicated or degenerate geometry.  Like the clearance gate it is geometric and exhaustive:
the full-tier MJCF is compiled with EVERY geom collidable (visual-only parts included), MuJoCo's parent-child
contact filter disabled and a dense constraint Jacobian, and distances are measured with ``mj_geomDistance``.
Closed kinematic loops (closer arms pinned to the frame by a ``connect`` equality) are solved for every swept
configuration with a Newton iteration on MuJoCo's own equality residuals (``LoopSolver``), so the checks see the
configuration the physics engine would settle into, not the frozen tree pose.

Rules (each finding carries ``rule``, the offending body / geoms, the measured distance, the configuration and an
explanation; ``domain`` is ``"closer"`` for closer / power-operator / gas-strut parts and ``"door"`` otherwise):

  1. ``intra_body``      the geoms of one body form ONE connected cluster with gaps < TOL_INTRA (two islands =
                         a floating piece, e.g. chain-as-beads, a duplicated thumb piece)
  2. ``detached``        every non-world body is within TOL_ATTACH of its parent body's geoms, of the body it is
                         pinned to by a connect / weld equality, or (world children) of a static geom: hinges touch
                         leaf and jamb, handles and locks touch the leaf, the closer body sits on the leaf ...
  3. ``static_floating`` every static geom belongs to a cluster (gaps < TOL_ATTACH) that contains the floor, a
                         wall or a ceiling: keepers, strikes, shoes, brackets and tracks are mounted on something
  4. ``loop_open`` / ``detached_in_motion``  through the sweep of every leaf and every independent mechanism joint
                         (loops solved at every step): connect / weld partners stay within TOL_LOOP (the forearm
                         tip stays at the shoe) and every body that was attached at rest stays attached (rule 2):
                         a leaf hangs on its hinges / hangers at every angle, a bolt stays in its guides
  5. ``no_actuation``    every joint with a usable range moves: leaf and operator joints are driven directly, coupled
                         joints (joint equalities, one-sided tendons) must follow their driver, loop joints (closer
                         pinion / elbow) must change when the door swings, a spring latch bolt must be coupled to the
                         door's operator.  Reported per joint (``joints`` in the result) so it complements the
                         force-driven QA instead of duplicating it.  Joints whose driver is locked (or whose coupling
                         cannot produce a visible motion over the driver's range) are reported as ``locked_by_driver``
  6. ``degenerate``      zero-size geoms, bodies with mass but no geoms, geoms without a material, exact-duplicate
                         geoms, meshes with an implausible bounding box, collision proxies far from their visual
                         part, meshes with a known "up" direction rendered upside-down / sideways (asymmetric
                         handleset meshes used to be flipped by ``q_face`` on right-hinged doors), keypads / readers
                         on the wrong face
  7. ``no_keeper``       an engaged bolt / hook / bar (label "0 = engaged / thrown / hooked ...") must sit within
                         TOL_KEEPER of a keeper, strike, pocket or socket on another body when the door is closed

Allow-lists only with a documented justification: ``model.meta["attachment_allow"]`` is a list of
``[rule, name_pattern, justification]`` (fnmatch on the body or geom name; ``rule`` may be ``"*"``).
"""
from __future__ import annotations

import fnmatch
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- tolerances --------------------------------------------------------------------------------------------------
TOL_INTRA = 0.002          # m; rule 1: max gap between the geoms of one body (parts screwed to each other touch)
TOL_ATTACH = 0.003         # m; rules 2/3/4: max gap between a body and what it is mounted on (a mounting plate is
                           #    modelled 1-3 mm proud of the face it sits on)
TOL_LOOP = 0.001           # m; rule 4: max separation of connect / weld partners after the loop is solved
TOL_MOVE_HINGE = math.radians(0.5)   # rule 5: a hinge that changes by less than this over a full sweep did not move
TOL_MOVE_SLIDE = 0.0005    # m; rule 5: same for slide joints
LOCKED_RANGE = 0.006       # rad or m; joints with a smaller range are locked by design (same as clearance.py / viewer)
MIN_HALF_EXTENT = 0.0002   # m; rule 6: a 0.4 mm slab is the thinnest legitimate geom (strike / keeper plates)
MESH_EXTENT_MIN = 0.004    # m; rule 6: hardware meshes are at least 4 mm ...
MESH_EXTENT_MAX = 2.5      # m; ... and at most 2.5 m (ladder pulls, closer arms) in every axis
PROXY_MARGIN = 0.02        # m; rule 6: a collision-only proxy must lie inside its visual geoms' bounding box grown by this
TOL_KEEPER = 0.008         # m; rule 7: an engaged bolt is captured if a keeper is within this (fork prongs: 6 mm)
N_STEPS = 12               # sweep resolution for leaf joints (mechanism joints use N_STEPS // 2)
EPS = 1e-6                 # comparisons: "within 3 mm" means dist <= TOL + EPS (a rod authored 3.000 mm from its guide passes)

LEAF_ROLES = ("primary", "secondary")
MECH_ROLES = ("operator", "latch", "lock", "mechanism", "decor")
GROUND_SEMANTICS = ("floor", "wall")                       # static clusters must contain one of these
KEEPER_SEMANTICS = ("frame", "latch", "lock", "track", "leaf")
# closer / power-operator / gas-strut parts (owned by the closer mechanism model): findings on them are tagged
# domain="closer" so the QA can report them separately from the rest of the door
CLOSER_RE = re.compile(r"closer|strut|auto_operator|operator_arm|pneumatic|floor_spring|spring_hinge", re.I)
# library meshes whose local +y is "up" when mounted on a vertical face (mesh frame of q_face: z away from the face,
# x across the door, y up).  Symmetric meshes (levers, knobs, pulls) have no up direction and are not checked.
MESH_UP = {"knocker": (0.0, 1.0, 0.0), "coat_hook": (0.0, 1.0, 0.0), "handleset": (0.0, 1.0, 0.0), "card_reader": (0.0, 1.0, 0.0),
           "keypad_body": (0.0, 1.0, 0.0), "thumb_latch": (0.0, 1.0, 0.0), "house_numbers": (0.0, 1.0, 0.0), "exit_sign": (0.0, 1.0, 0.0),
           "magnalatch": (0.0, 1.0, 0.0)}
ENGAGED_RE = re.compile(r"0 = [^,;)]*?(engaged|thrown|extended|dropped|hooked|dogged|latched|joined|closed over|in keeper|in track|in floor socket|in the head|in socket)", re.I)


# ---- model ---------------------------------------------------------------------------------------------------------
def gate_model(xml_path: str):
    """Full-tier model with every geom collidable, no parent filter, dense Jacobian (for the loop solver)."""
    import mujoco
    spec = mujoco.MjSpec.from_file(xml_path)
    for g in spec.geoms:
        g.contype = 1
        g.conaffinity = 1
        g.margin = 0.0
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    return spec.compile()


class LoopSolver:
    """Solve closed kinematic loops (connect / weld equalities) for a given tree pose.

    ``free_dofs`` are the DOFs allowed to change (the loop's own joints: closer pinion and elbow ...).  ``solve``
    runs a projected Newton iteration on MuJoCo's equality residuals (``efc_pos`` / ``efc_J``) with contacts
    disabled, clamps limited joints to their range and returns the closed pose plus the residual per equality.
    Reusable by renderers and other gates: ``LoopSolver.from_model_json(m, model_json)``.
    """

    def __init__(self, m, free_dofs: List[int], eq_ids: Optional[List[int]] = None):
        import mujoco
        self.mj = mujoco
        self.m = m
        self.free = list(free_dofs)
        self.eq_ids = list(range(m.neq)) if eq_ids is None else list(eq_ids)
        self.eq_ids = [e for e in self.eq_ids if int(m.eq_type[e]) in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD))]
        self.d_kin = mujoco.MjData(m)
        m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE     # efc_J must be dense (nefc x nv) for the Newton step
        self.qposadr = [int(a) for a in m.jnt_qposadr]
        self.dof2jnt = [int(j) for j in m.dof_jntid]

    def _forward_kin(self, q: np.ndarray):
        """mj_forward with contacts disabled (only the equality rows of efc are needed)."""
        mujoco, m, d = self.mj, self.m, self.d_kin
        flags = int(m.opt.disableflags)
        m.opt.disableflags = flags | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        try:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
        finally:
            m.opt.disableflags = flags

    @staticmethod
    def loop_joints(m, model_json: dict) -> List[str]:
        """Joints on the kinematic path between the partners of every connect / weld equality, minus the leaf joints
        (primary / secondary: driven, never solved) and joints slaved by a joint equality."""
        import mujoco
        roles = {b["joint"]["name"]: b["joint"]["role"] for b in model_json["bodies"] if b.get("joint")}
        slaves = {e["a"] for e in model_json.get("equalities", []) if e["kind"] == "joint"}
        out: List[str] = []
        for e in range(m.neq):
            if int(m.eq_type[e]) not in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
                continue
            b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            chain = []
            for b in (b1, b2):
                path = []
                while b > 0:
                    path.append(b)
                    b = int(m.body_parentid[b])
                chain.append(path)
            common = set(chain[0]) & set(chain[1])
            for path in chain:
                for b in path:
                    if b in common:
                        break
                    for j in range(int(m.body_jntadr[b]), int(m.body_jntadr[b]) + int(m.body_jntnum[b])):
                        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                        if name and roles.get(name) not in LEAF_ROLES and name not in slaves and name not in out:
                            out.append(name)
        return out

    @classmethod
    def from_model_json(cls, m, model_json: dict) -> "LoopSolver":
        import mujoco
        names = cls.loop_joints(m, model_json)
        dofs = []
        for n in names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j >= 0:
                dofs.append(int(m.jnt_dofadr[j]))
        return cls(m, dofs)

    def residuals(self, q: np.ndarray) -> Dict[str, float]:
        """Norm of the position residual of every connect / weld equality at pose q (no solving)."""
        mujoco, m, d = self.mj, self.m, self.d_kin
        self._forward_kin(q)
        out: Dict[str, float] = {}
        for e in self.eq_ids:
            rows = [i for i in range(d.nefc) if int(d.efc_type[i]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY) and int(d.efc_id[i]) == e]
            r = d.efc_pos[rows][:3] if rows else np.zeros(3)      # weld: first 3 rows are the position part
            out[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or f"eq{e}"] = float(np.linalg.norm(r))
        return out

    def solve(self, q: np.ndarray, max_iter: int = 12, tol: float = 1e-7) -> np.ndarray:
        """Projected Newton on the loop residuals over the free DOFs.  Returns the closed pose (copy)."""
        mujoco, m, d = self.mj, self.m, self.d_kin
        q = np.array(q, dtype=float, copy=True)
        if not self.free or not self.eq_ids:
            return q
        for _ in range(max_iter):
            self._forward_kin(q)
            rows = [i for i in range(d.nefc) if int(d.efc_type[i]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY) and int(d.efc_id[i]) in self.eq_ids]
            if not rows:
                return q
            r = d.efc_pos[rows]
            if float(np.linalg.norm(r)) < tol:
                return q
            J = np.asarray(d.efc_J).reshape(d.nefc, m.nv)[rows][:, self.free]
            dq = -np.linalg.lstsq(J, r, rcond=None)[0]
            step = 1.0 if float(np.max(np.abs(dq))) < 0.5 else 0.5 / float(np.max(np.abs(dq)))
            for a, v in zip(self.free, dq):
                jn = self.dof2jnt[a]
                qa = self.qposadr[jn]
                q[qa] += step * float(v)
                if m.jnt_limited[jn]:
                    q[qa] = float(min(max(q[qa], m.jnt_range[jn][0]), m.jnt_range[jn][1]))
        return q


# ---- helpers -------------------------------------------------------------------------------------------------------
def _union_find(n: int, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def domain_of(*names: str) -> str:
    """'closer' for closer / power-operator / gas-strut parts, else 'door'."""
    return "closer" if any(n and CLOSER_RE.search(n) for n in names) else "door"


class Attachment:
    def __init__(self, door_dir: str, tier: str = "full"):
        import mujoco
        self.mj = mujoco
        self.dir = door_dir
        xml = os.path.join(door_dir, {"full": "door.xml", "simple": "door_simple.xml", "minimal": "door_minimal.xml"}[tier])
        self.m = gate_model(xml)
        self.d = mujoco.MjData(self.m)
        with open(os.path.join(door_dir, "model.json")) as f:
            self.model_json = json.load(f)
        self.meta = self.model_json["meta"]
        self.spec = None
        try:
            with open(os.path.join(door_dir, "spec.json")) as f:
                self.spec = json.load(f)
        except Exception:
            pass
        self.allow = [tuple(a[:3]) if len(a) >= 3 else ("*", a[0], "") for a in self.meta.get("attachment_allow", [])]
        m = self.m
        self.gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}" for g in range(m.ngeom)]
        self.bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}" for b in range(m.nbody)]
        self.jname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}" for j in range(m.njnt)]
        self.jid = {n: j for j, n in enumerate(self.jname)}
        self.gid = {n: g for g, n in enumerate(self.gname)}
        self.sem: Dict[str, str] = {}
        self.gvis: Dict[str, bool] = {}
        self.gcol: Dict[str, bool] = {}
        self.gmat: Dict[str, str] = {}
        self.gquat: Dict[str, tuple] = {}
        self.gmesh: Dict[str, str] = {}
        self.body_json: Dict[str, dict] = {}
        for b in self.model_json["bodies"]:
            self.body_json[b["name"]] = b
            for g in b["geoms"]:
                self.sem[g["name"]] = g.get("semantic", "")
                self.gvis[g["name"]] = bool(g.get("visual", True))
                self.gcol[g["name"]] = bool(g.get("collision", True))
                self.gmat[g["name"]] = g.get("material", "")
                self.gquat[g["name"]] = tuple(g.get("quat", (1, 0, 0, 0)))
                if g.get("mesh_name"):
                    self.gmesh[g["name"]] = g["mesh_name"]
        self.joints: Dict[str, dict] = {}
        for b in self.model_json["bodies"]:
            j = b.get("joint")
            if j:
                self.joints[j["name"]] = dict(j, body=b["name"])
        self.eq_json = self.model_json.get("equalities", [])
        self.eq_slaves = {e["a"]: e["b"] for e in self.eq_json if e["kind"] == "joint"}
        self.eq_coeff = {e["a"]: float(e["polycoeff"][1]) if len(e.get("polycoeff", [])) > 1 else 1.0 for e in self.eq_json if e["kind"] == "joint"}
        # one-sided fixed tendons: driven joint (positive unit coefficient) -> driver joints
        self.tendon_drivers: Dict[str, List[str]] = {}
        self.tendons = []
        for t in range(m.ntendon):
            if not m.tendon_limited[t]:
                continue
            terms = []
            for w in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
                if int(m.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_JOINT):
                    terms.append((int(m.wrap_objid[w]), float(m.wrap_prm[w])))
            if terms:
                self.tendons.append((float(m.tendon_range[t][0]), terms))
                driven = [self.jname[j] for j, c in terms if c > 0]
                drivers = [self.jname[j] for j, c in terms if c < 0]
                for dn in driven:
                    self.tendon_drivers.setdefault(dn, []).extend(drivers)
        self.breakable = {w["name"] for w in self.meta.get("breakable_welds", [])}
        self.loop_joint_names = LoopSolver.loop_joints(m, self.model_json)
        self.solver = LoopSolver(m, [int(m.jnt_dofadr[self.jid[n]]) for n in self.loop_joint_names if n in self.jid],
                                 [e for e in range(m.neq) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or "") not in self.breakable])
        # geoms per body, static (world) geoms
        self.body_geoms: Dict[int, List[int]] = {b: [] for b in range(m.nbody)}
        for g in range(m.ngeom):
            self.body_geoms[int(m.geom_bodyid[g])].append(g)
        self.static_geoms = list(self.body_geoms[0])
        self.findings: List[dict] = []
        self._by_key: Dict[tuple, dict] = {}
        self.detached_at_rest: set = set()
        self.joint_report: Dict[str, dict] = {}

    # ---- allow-list / recording --------------------------------------------------------------------------------
    def allowed(self, rule: str, *names: str) -> Optional[str]:
        for r, pat, why in self.allow:
            if r not in ("*", rule):
                continue
            if any(fnmatch.fnmatch(n, pat) for n in names if n):
                return why or pat
        return None

    def record(self, rule: str, body: str, geoms: List[str], dist: float, why: str, config: str = "initial", severity: str = "fail", **extra):
        """One finding per (rule, kind, body, first geom, joint): the worst configuration is kept and ``n_configs``
        counts how many swept configurations showed it (a body that is detached through the whole sweep is ONE
        finding, not one per sweep step)."""
        if self.allowed(rule, body, *geoms):
            return
        key = (rule, extra.get("kind", ""), body, geoms[0] if geoms else "", extra.get("joint", ""))
        prev = self._by_key.get(key)
        if prev is not None:
            prev["n_configs"] += 1
            if dist > prev["dist"]:
                prev.update({"geoms": list(geoms), "dist": round(float(dist), 5), "config": config, "why": why})
            return
        f = {"rule": rule, "severity": severity, "domain": domain_of(body, *geoms, extra.get("joint", "")), "body": body, "geoms": list(geoms),
             "dist": round(float(dist), 5), "config": config, "why": why, "n_configs": 1, **extra}
        self._by_key[key] = f
        self.findings.append(f)

    # ---- geometry helpers -------------------------------------------------------------------------------------
    def forward(self, q: np.ndarray):
        self.d.qpos[:] = q
        self.mj.mj_forward(self.m, self.d)

    def gdist(self, g1: int, g2: int, distmax: float) -> float:
        """Signed distance between two geoms in the current pose; returns ``distmax`` when they are farther apart
        than that (bounding-sphere broadphase first)."""
        m, d = self.m, self.d
        if float(np.linalg.norm(d.geom_xpos[g1] - d.geom_xpos[g2])) > float(m.geom_rbound[g1] + m.geom_rbound[g2]) + distmax:
            return distmax
        return float(self.mj.mj_geomDistance(m, d, g1, g2, distmax, None))

    def min_dist(self, ga: List[int], gb: List[int], distmax: float) -> Tuple[float, Optional[Tuple[str, str]]]:
        best, pair = distmax, None
        for a in ga:
            for b in gb:
                if a == b:
                    continue
                v = self.gdist(a, b, distmax)
                if v < best:
                    best, pair = v, (self.gname[a], self.gname[b])
                    if best <= -distmax:
                        return best, pair
        return best, pair

    # ---- kinematics: joint couplings (same semantics as clearance.py) ------------------------------------------
    def resolve(self, q: np.ndarray) -> np.ndarray:
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
                length = sum(coef * q[m.jnt_qposadr[j]] for j, coef in terms)
                if length < lo - 1e-9:
                    for j, coef in terms:
                        if coef > 0:
                            q[m.jnt_qposadr[j]] += (lo - length) / coef
                            break
        return q

    def _locked(self, j: int) -> bool:
        lo, hi = self.m.jnt_range[j]
        return bool(self.m.jnt_limited[j]) and (hi - lo) < LOCKED_RANGE

    def _range(self, j: int) -> float:
        return float(self.m.jnt_range[j][1] - self.m.jnt_range[j][0]) if self.m.jnt_limited[j] else 2 * math.pi

    def pose(self, q: np.ndarray) -> np.ndarray:
        """Couplings + loop closure."""
        return self.solver.solve(self.resolve(q))

    # ---- rule 1: intra-body connectivity ----------------------------------------------------------------------
    def rule_intra_body(self):
        m = self.m
        for b in range(1, m.nbody):
            gs = self.body_geoms[b]
            if len(gs) < 2:
                continue
            edges = []
            for i in range(len(gs)):
                for k in range(i + 1, len(gs)):
                    if self.gdist(gs[i], gs[k], TOL_INTRA + 2 * EPS) <= TOL_INTRA + EPS:
                        edges.append((i, k))
            comp = _union_find(len(gs), edges)
            clusters: Dict[int, List[int]] = {}
            for i, c in enumerate(comp):
                clusters.setdefault(c, []).append(gs[i])
            if len(clusters) <= 1:
                continue
            main = max(clusters.values(), key=len)
            for c in clusters.values():
                if c is main:
                    continue
                gap, _ = self.min_dist(c, main, 0.5)
                self.record("intra_body", self.bname[b], [self.gname[g] for g in c], gap,
                            f"{len(c)} geom(s) of body '{self.bname[b]}' form an island {gap * 1000:.1f} mm away from the rest of the body (nothing connects them): {', '.join(self.gname[g] for g in c[:4])}")

    # ---- rule 2 / 4: body attachment ---------------------------------------------------------------------------
    def _partners(self, b: int) -> Tuple[List[int], List[str]]:
        """Geoms a body may be attached to: its parent's geoms (world children: every static geom) plus the geoms of
        the bodies it is pinned to by a connect / weld equality."""
        m, mujoco = self.m, self.mj
        parent = int(m.body_parentid[b])
        gs = list(self.body_geoms[parent])
        names = [self.bname[parent]]
        for e in range(m.neq):
            if int(m.eq_type[e]) not in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
                continue
            b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            other = b2 if b1 == b else (b1 if b2 == b else -1)
            if other >= 0:
                gs += self.body_geoms[other]
                names.append(self.bname[other])
        return gs, names

    def check_attachment(self, config: str, rule: str):
        m = self.m
        for b in range(1, m.nbody):
            gs = self.body_geoms[b]
            if not gs:
                continue
            if rule == "detached_in_motion" and b in self.detached_at_rest:
                continue        # already reported at rest; the motion adds nothing
            partners, pnames = self._partners(b)
            if not partners:
                continue
            dist, pair = self.min_dist(gs, partners, 0.25)
            if dist > TOL_ATTACH + EPS:
                if rule == "detached":
                    self.detached_at_rest.add(b)
                self.record(rule, self.bname[b], [pair[0]] if pair else [self.gname[gs[0]]], dist,
                            f"body '{self.bname[b]}' is {dist * 1000:.1f} mm from anything it could be mounted on ({', '.join(pnames)})"
                            + (f" (closest: {pair[1]})" if pair else "") + ("" if rule == "detached" else f" at {config}"), config=config)

    # ---- rule 3: static geometry ----------------------------------------------------------------------------------
    def rule_static(self):
        gs = self.static_geoms
        if not gs:
            return
        edges = []
        for i in range(len(gs)):
            for k in range(i + 1, len(gs)):
                if self.gdist(gs[i], gs[k], TOL_ATTACH + 2 * EPS) <= TOL_ATTACH + EPS:
                    edges.append((i, k))
        comp = _union_find(len(gs), edges)
        clusters: Dict[int, List[int]] = {}
        for i, c in enumerate(comp):
            clusters.setdefault(c, []).append(gs[i])
        grounded = [c for c in clusters.values() if any(self.sem.get(self.gname[g], "") in GROUND_SEMANTICS for g in c)]
        ground_geoms = [g for c in grounded for g in c]
        for c in clusters.values():
            if any(c is gc for gc in grounded):
                continue
            gap, pair = self.min_dist(c, ground_geoms, 0.5) if ground_geoms else (0.5, None)
            self.record("static_floating", "world", [self.gname[g] for g in c], gap,
                        f"{len(c)} static geom(s) hang in the air, {gap * 1000:.1f} mm from the nearest grounded geometry" + (f" ({pair[1]})" if pair else "") + f": {', '.join(self.gname[g] for g in c[:4])}")

    # ---- rule 6: degenerate content ------------------------------------------------------------------------------
    def rule_degenerate(self):
        m, mujoco = self.m, self.mj
        mats = set(self.model_json.get("materials", {}).keys())
        seen: Dict[tuple, str] = {}
        for b in range(1, m.nbody):
            if not self.body_geoms[b] and m.body_mass[b] > 0.011 and m.body_dofnum[b] > 0:
                self.record("degenerate", self.bname[b], [], 0.0, f"body '{self.bname[b]}' has mass {m.body_mass[b]:.3f} kg but no geoms", kind="massive_empty_body")
        for g in range(m.ngeom):
            name = self.gname[g]
            gtype = int(m.geom_type[g])
            size = np.asarray(m.geom_size[g], float)
            body = self.bname[int(m.geom_bodyid[g])]
            if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                small = float(size.min())
            elif gtype in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                small = float(min(size[0], size[1] + (size[0] if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0)))
            elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                small = float(size[0])
            else:
                small = 1.0
            if small < MIN_HALF_EXTENT:
                self.record("degenerate", body, [name], small, f"geom '{name}' has a half-extent of {small * 1000:.2f} mm (zero-size)", kind="zero_size")
            if name in self.gmat and self.gmat[name] not in mats:
                self.record("degenerate", body, [name], 0.0, f"geom '{name}' references material '{self.gmat[name]}' which the model does not define", kind="no_material")
            key = (int(m.geom_bodyid[g]), gtype, tuple(np.round(m.geom_pos[g], 6)), tuple(np.round(m.geom_quat[g], 6)), tuple(np.round(size, 6)), int(m.geom_dataid[g]))
            if key in seen:
                self.record("degenerate", body, [name, seen[key]], 0.0, f"geom '{name}' duplicates '{seen[key]}' exactly (same body, pose and size)", kind="duplicate")
            else:
                seen[key] = name
            if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(m.geom_dataid[g])
                verts = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                ext = verts.max(axis=0) - verts.min(axis=0)
                if float(ext.max()) > MESH_EXTENT_MAX or float(ext.max()) < MESH_EXTENT_MIN or float(ext.min()) < 1e-6:
                    self.record("degenerate", body, [name], float(ext.max()), f"mesh '{name}' has an implausible bounding box {np.round(ext, 4).tolist()} m", kind="mesh_extent")
                self._check_mesh_up(g, name, body)
        self._check_proxies()
        self._check_faces()

    def _check_mesh_up(self, g: int, name: str, body: str):
        """Meshes with a known local up (+y in the OBJ frame) must map to world +z when mounted on a vertical face.
        The orientation is taken from the IR quaternion (model.json) and the body's world rotation: MuJoCo re-aligns
        mesh vertices to their inertial frame, so ``geom_xmat`` is not the OBJ frame."""
        mesh_name = self.gmesh.get(name, "")
        prefix = mesh_name.rsplit("_", 1)[0] if mesh_name else ""
        up = MESH_UP.get(prefix)
        if up is None:
            return
        b = int(self.m.geom_bodyid[g])
        R = np.asarray(self.d.xmat[b]).reshape(3, 3) @ _quat_to_mat(self.gquat.get(name, (1, 0, 0, 0)))
        if abs(float(R[2, 2])) > 0.5:          # local z (face normal) points up/down: hatch / floor hardware, no "up"
            return
        wz = float((R @ np.asarray(up, float))[2])
        if wz < 0.5:
            self.record("degenerate", body, [name], wz, f"mesh '{name}' ({prefix}) is mounted with its up direction pointing {'down' if wz < -0.5 else 'sideways'} (mirrored / flipped by the face quaternion)", kind="flipped_mesh")

    def _check_proxies(self):
        """A collision-only proxy (group 3) must lie within the world box of its body's visual geoms (grown by
        PROXY_MARGIN): a mesh rendered elsewhere than the thing the robot touches is misplaced / mis-scaled."""
        m = self.m
        for b in range(1, m.nbody):
            gs = self.body_geoms[b]
            vis = [g for g in gs if self.gvis.get(self.gname[g], True)]
            prox = [g for g in gs if not self.gvis.get(self.gname[g], True) and self.gcol.get(self.gname[g], True)]
            if not vis or not prox:
                continue
            lo = np.min([self.d.geom_xpos[g] - m.geom_rbound[g] for g in vis], axis=0)
            hi = np.max([self.d.geom_xpos[g] + m.geom_rbound[g] for g in vis], axis=0)
            for g in prox:
                c = self.d.geom_xpos[g]
                out = float(max(np.max(lo - c), np.max(c - hi)))
                if out > PROXY_MARGIN:
                    self.record("degenerate", self.bname[b], [self.gname[g]], out, f"collision proxy '{self.gname[g]}' lies {out * 1000:.0f} mm outside the visual geometry of body '{self.bname[b]}' (visual part misplaced, mirrored or mis-scaled)", kind="proxy_mismatch")

    def _check_faces(self):
        """Keypads / card readers belong on the outside face (the credential side)."""
        if not self.spec:
            return
        outside = -1.0 if self.spec.get("robot", {}).get("robot_outside") else 1.0
        u = float(self.meta.get("u", 1.0) or 1.0)
        for g in range(self.m.ngeom):
            name = self.gname[g]
            if not any(k in name for k in ("keypad_body", "_reader")):
                continue
            b = int(self.m.geom_bodyid[g])
            leaf = b
            while leaf > 0 and self.body_json.get(self.bname[leaf], {}).get("semantic") != "leaf":
                leaf = int(self.m.body_parentid[leaf])
            if leaf <= 0:
                continue
            R = np.asarray(self.d.xmat[leaf]).reshape(3, 3)
            rel = R.T @ (self.d.geom_xpos[g] - self.d.xpos[leaf])
            if float(rel[1]) * outside < 0 and abs(float(rel[1])) > 0.01:
                self.record("degenerate", self.bname[b], [name], float(rel[1]), f"'{name}' sits on the {'inside' if outside > 0 else 'outside'} face (y = {rel[1]:+.3f}) but the credential reader belongs on the outside face", kind="wrong_face", u=u)

    # ---- rule 7: engaged bolts need a keeper ---------------------------------------------------------------------
    def rule_keepers(self, base: np.ndarray):
        m = self.m
        slaves_of: Dict[str, List[str]] = {}
        for a, master in self.eq_slaves.items():
            slaves_of.setdefault(master, []).append(a)
        for jn, info in self.joints.items():
            if info.get("role") not in ("latch", "lock") or jn not in self.jid or not ENGAGED_RE.search(info.get("label", "") or ""):
                continue
            j = self.jid[jn]
            bodies = [int(m.jnt_bodyid[j])] + [int(m.jnt_bodyid[self.jid[s]]) for s in slaves_of.get(jn, []) if s in self.jid]
            bolt = [g for b in bodies for g in self.body_geoms[b] if self.sem.get(self.gname[g], "") in ("latch", "lock")]
            if not bolt:
                continue
            anc = set()
            for b0 in bodies:
                a = b0
                while a > 0:
                    anc.add(a)
                    a = int(m.body_parentid[a])
            cands = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) not in anc and self.sem.get(self.gname[g], "") in KEEPER_SEMANTICS]
            if not cands:
                continue
            # engaged bolt with every leaf closed (0 = closed for every leaf joint), couplings resolved
            q = base.copy()
            for ln, lj in self.joints.items():
                if lj.get("role") in LEAF_ROLES and ln in self.jid:
                    q[m.jnt_qposadr[self.jid[ln]]] = 0.0
            q[m.jnt_qposadr[j]] = 0.0
            q = self.resolve(q)
            q[m.jnt_qposadr[j]] = 0.0
            self.forward(q)
            dist, pair = self.min_dist(bolt, cands, 0.2)
            if dist > TOL_KEEPER + EPS:
                self.record("no_keeper", self.bname[int(m.jnt_bodyid[j])], [pair[0]] if pair else [self.gname[bolt[0]]], dist,
                            f"engaged '{jn}' ({info.get('label', '')}): its bolt is {dist * 1000:.0f} mm from the nearest keeper / strike / pocket" + (f" ({pair[1]})" if pair else "") + " - nothing captures it", config=f"engaged:{jn}", joint=jn)

    # ---- the gate -------------------------------------------------------------------------------------------------
    def run(self, n_steps: int = N_STEPS) -> dict:
        m, mujoco = self.m, self.mj
        base = self.pose(m.qpos0.copy())
        self.forward(base)
        self.rule_intra_body()
        self.check_attachment("initial", "detached")
        self.rule_static()
        self.rule_degenerate()
        self.rule_keepers(base)
        # ---- sweeps: leaf joints and every independent mechanism joint; loops solved at every step ----------------
        qmin = base.copy()
        qmax = base.copy()
        loop_names = set(self.loop_joint_names)
        leaf_joints = [n for n, j in self.joints.items() if j.get("role") in LEAF_ROLES and n in self.jid]
        indep = [n for n, j in self.joints.items() if j.get("role") in MECH_ROLES and n in self.jid and n not in loop_names and n not in self.eq_slaves and n not in self.tendon_drivers]
        for jn in leaf_joints + indep:
            j = self.jid[jn]
            if self._locked(j):
                continue
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-math.pi, math.pi))
            if hi - lo < 1e-6:
                continue
            steps = n_steps if jn in leaf_joints else max(2, n_steps // 2)
            q = base.copy()
            for k in range(0, steps + 1):
                q = q.copy()
                q[m.jnt_qposadr[j]] = lo + (hi - lo) * k / steps
                q = self.pose(q)
                qmin, qmax = np.minimum(qmin, q), np.maximum(qmax, q)
                config = f"sweep:{jn}@{q[m.jnt_qposadr[j]]:.3f}"
                for en, r in self.solver.residuals(q).items():
                    if r > TOL_LOOP:
                        self.record("loop_open", en, [], r, f"equality '{en}' partners are {r * 1000:.1f} mm apart with {jn} at {q[m.jnt_qposadr[j]]:.3f} (the linkage cannot follow the motion)", config=config)
                self.forward(q)
                self.check_attachment(config, "detached_in_motion")
        self.forward(base)
        # ---- rule 5: every joint moves ------------------------------------------------------------------------------
        op_joint = self.meta.get("operator_joint")
        leaves_locked = all(self._locked(self.jid[n]) for n in leaf_joints) if leaf_joints else True
        for jn, info in self.joints.items():
            if jn not in self.jid:
                continue
            j = self.jid[jn]
            adr = int(m.jnt_qposadr[j])
            moved = float(qmax[adr] - qmin[adr])
            tol = TOL_MOVE_HINGE if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE) else TOL_MOVE_SLIDE
            role = info.get("role")
            rng = None if not m.jnt_limited[j] else [float(m.jnt_range[j][0]), float(m.jnt_range[j][1])]
            rep = {"role": role, "range": rng, "moved": round(moved, 5), "status": "direct", "driver": None, "ok": True}
            if jn in self.eq_slaves:
                master = self.eq_slaves[jn]
                rep.update(status="coupled", driver=master)
                if master not in self.jid or self._locked(self.jid[master]) or abs(self.eq_coeff.get(jn, 1.0)) * self._range(self.jid[master]) < tol:
                    rep["status"] = "locked_by_driver"      # driver locked, or the coupling cannot produce a visible motion
                else:
                    rep["ok"] = moved >= tol
            elif self._locked(j):
                rep["status"] = "locked"
            elif jn in loop_names:
                rep.update(status="loop", driver="+".join(leaf_joints))
                if leaves_locked:
                    rep["status"] = "locked_by_driver"
                else:
                    rep["ok"] = moved >= tol
            elif jn in self.tendon_drivers:
                drivers = self.tendon_drivers[jn]
                rep.update(status="coupled", driver="+".join(drivers))
                if all(d in self.jid and self._locked(self.jid[d]) for d in drivers):
                    rep["status"] = "locked_by_driver"
                else:
                    rep["ok"] = moved >= tol
            elif role in LEAF_ROLES or role == "operator":
                rep["ok"] = moved >= tol
            else:
                # swept directly, but nothing on the door drives it: a spring latch bolt with an operator on the door
                # must be coupled to that operator
                rep["status"] = "passive"
                if jn.endswith("latch_bolt_slide") and op_joint and op_joint in self.jid and not self._locked(self.jid[op_joint]):
                    rep["ok"] = False
                    rep["driver_missing"] = op_joint
            if not rep["ok"]:
                why = {"loop": f"loop joint '{jn}' never changes ({moved:.4f}) while the door swings: the linkage does not articulate",
                       "coupled": f"'{jn}' is coupled to '{rep['driver']}' but never moves ({moved:.4f}) when the driver sweeps its range",
                       "passive": f"spring latch '{jn}' is not coupled to the door's operator '{op_joint}': the bolt never retracts when the operator turns",
                       "direct": f"'{jn}' cannot move ({moved:.4f})"}.get(rep["status"], f"'{jn}' does not actuate")
                self.record("no_actuation", info.get("body", ""), [], moved, why, config="sweep", joint=jn)
            self.joint_report[jn] = rep
        return self.result(leaf_joints + indep)

    def result(self, swept: List[str]) -> dict:
        fails = [f for f in self.findings if f.get("severity", "fail") == "fail"]
        counts: Dict[str, int] = {}
        for f in fails:
            counts[f["rule"]] = counts.get(f["rule"], 0) + 1
        door = [f for f in fails if f.get("domain") == "door"]
        closer = [f for f in fails if f.get("domain") == "closer"]
        return {"ok": len(fails) == 0, "ok_door": len(door) == 0, "n_findings": len(fails), "n_closer_findings": len(closer),
                "findings": sorted(fails, key=lambda f: (f.get("domain") != "door", -f["dist"]))[:60], "counts": counts,
                "joints": self.joint_report, "loop_joints": self.loop_joint_names, "swept": swept}


def run_attachment(door_dir: str, tier: str = "full", n_steps: int = N_STEPS) -> dict:
    try:
        return Attachment(door_dir, tier).run(n_steps)
    except Exception as e:  # a gate that cannot run is a failure, not a pass
        return {"ok": False, "ok_door": False, "n_findings": 1, "n_closer_findings": 0, "counts": {"error": 1}, "joints": {},
                "findings": [{"rule": "error", "severity": "fail", "domain": "door", "body": "", "geoms": [], "dist": 0.0, "config": "error", "why": f"{type(e).__name__}: {e}", "n_configs": 1}]}


def pose_with_loops(m, model_json: dict, q: np.ndarray) -> np.ndarray:
    """Convenience for renderers: close the model's kinematic loops at pose q (closer arms follow the door)."""
    return LoopSolver.from_model_json(m, model_json).solve(np.asarray(q, float))

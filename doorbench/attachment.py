"""Deterministic ATTACHMENT gate: nothing floats.

The clearance gates (doorbench/clearance.py) prove that parts do not interpenetrate and that moving parts keep a
running gap from static structure.  Neither of them can see the opposite defect: a part that touches NOTHING.  A
rubber door stop hanging 0.85 m off the wall, a closer bracket 54 mm in front of the soffit, a chain modelled as six
beads floating apart, a keeper screwed to thin air - each of those renders as an object suspended in space, which is
the single most obvious "this is wrong" a human reviewer can spot on the site.  This module is the gate for it.

It reuses ``clearance.Clearance`` verbatim for the model: the FULL tier compiled with every geom collidable (visual
trim included - a viewer draws it, so it must be attached too), MuJoCo's parent-child contact filter disabled, and
the joint / equality / tendon sweep machinery.  Distances are ``mj_geomDistance`` (exact signed separation, no
contact-generation blind spot), with a bounding-sphere prefilter.

Rules (each finding carries ``rule``, the named tolerance it broke, and the measured gap):

  intra_body_split    The geoms of ONE moving body must form a single cluster with gaps <= TOL_INTRA (2 mm).  Two
                      islands mean a piece of that body floats beside it (the six-bead chain, a thumbturn duplicated
                      above its cylinder).  Static bodies are covered by ``static_detached`` instead, which is the
                      same test generalised across every world-welded body.
  body_detached       At rest, every non-world body must be within TOL_BODY (3 mm) of what carries it: the geoms of
                      its nearest ancestor body that has any, the body it is pinned to by a connect/weld equality,
                      or - for a body hung directly off the world - the static geometry it is mounted to.  A hinged
                      leaf is attached through its hinge knuckles, a slider through its hangers/rollers.
  static_detached     Every static geom must be connected, through a chain of static geoms with gaps <= TOL_STATIC
                      (3 mm), to the structural root (the component holding the floor).  Jamb plates, keepers,
                      strikes, stops, bumpers, shoes, brackets and tracks all have to land on the frame, wall, floor
                      or ceiling.  Nothing static may hang in the air.  This is the db0024 wall-bumper case.
  detached_in_motion  The same attachment, re-measured through the travel: every body's own joint is swept through
                      its range (a body's distance to its parent can only change when its OWN joint moves) and the
                      worst sample must still satisfy TOL_BODY.  A hinge whose knuckles part company at 90 deg, a
                      bolt that leaves its housing, a roller that runs off the end of its rail.
  equality_anchor     Every connect/weld equality must be authored closed: the two anchor points must coincide
                      within TOL_EQ (1 mm) in the shipped pose.  (Whether the loop can FOLLOW the leaf through its
                      travel is the business of ``checks["linkage_feasibility"]`` / doorbench/linkage_qa.py, which
                      solves the loop's own mechanism joints with least-squares; this is the cheap authored-pose
                      half of the same requirement and needs no solve.)
  degenerate_geom     A geom with a zero / near-zero dimension (half-extent below MIN_HALF_EXTENT = 0.2 mm, i.e. a
                      part thinner than 0.4 mm) or a non-finite one: it renders as an invisible sliver and collides
                      as a degenerate shape.
  body_without_geoms  A body that carries mass but has no geometry at all in the tier - mass hanging in the void.
  duplicate_geom      Two geoms of the same body with the same type, the same size and the same pose (within
                      DUP_EPS): one of them is a copy-paste that renders twice and doubles the part's mass.
  mesh_bbox           A mesh geom whose real bounding box contradicts the size the IR declares for it, or whose box
                      is degenerate / absurd (below MIN_HALF_EXTENT or beyond MESH_MAX_HALF_EXTENT).  Mesh parts
                      carry the neutral scale marker ``(1, 1, 1)`` in the IR; anything else must match the mesh.

Allow-list.  A door may declare ``model.meta["attachment_allow"]`` entries, each of which MUST carry a written
justification, in either form::

    ["<rule>", "<geom-or-body glob>", "reason"]     # only that rule, only those parts
    ["<geom-or-body glob>", "reason"]               # every rule

Legitimate cases are the parts that really do hang: a strip curtain hangs from its header, a wreath hangs on its
hook, a pet flap swings in a hole.  Anything else is a defect and must be fixed in the generator.
"""
from __future__ import annotations

import fnmatch
import json
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# tolerances (every one of them is quoted by the finding that uses it)
# ---------------------------------------------------------------------------
TOL_INTRA = 0.002        # m; the geoms of one body form a single cluster with gaps no larger than this.  2 mm is a
#                          fabrication joint (a welded/screwed assembly is metal-on-metal); above it a viewer sees
#                          daylight between the two halves of one part.
TOL_BODY = 0.003         # m; a body must touch what carries it within this.  3 mm is the running clearance a leaf
#                          keeps at its jambs, so it is the largest gap that can still read as "in contact" here
#                          without colliding with the running-clearance gate's requirement in the other direction.
TOL_STATIC = 0.003       # m; same, for static geometry against the static structure it is mounted to.
TOL_EQ = 0.001           # m; a connect/weld equality is authored closed to a millimetre (shared with
#                          linkage_qa.RESIDUAL_TOL_M so the two gates cannot disagree).
MIN_HALF_EXTENT = 2e-4   # m; the smallest half-extent a real part may have (0.4 mm thick).  The thinnest genuine
#                          parts in the dataset are 1.2 mm faceplates and 2 mm signs, six times this.
MESH_MAX_HALF_EXTENT = 3.0   # m; no single mesh part is bigger than a 6 m box (the largest leaves are boxes).
DUP_EPS = 1e-6           # m / rad; two geoms closer than this in size AND pose are the same geom twice.
SEARCH = 1.5             # m; how far a finding looks to report "the nearest thing is X, that far away".

# geoms whose *semantic* means they are not structure at all and cannot be expected to carry anything; they still
# have to BE attached, they just never count as the thing something else is attached to.
_NEVER_SUPPORT = ()


def _load_allow(meta: dict) -> List[Tuple[str, str]]:
    """``meta["attachment_allow"]`` -> [(rule glob, name glob)] (the trailing reason is documentation)."""
    out = []
    for e in meta.get("attachment_allow", []) or []:
        if not e:
            continue
        if len(e) >= 3:
            out.append((str(e[0]), str(e[1])))
        else:
            out.append(("*", str(e[0])))
    return out


class Attachment:
    """The attachment gate for one generated door directory.

    Composes ``clearance.Clearance`` so both gates see exactly the same compiled model (full tier, every geom
    collidable, parent filter off) and the same joint/equality bookkeeping."""

    def __init__(self, door_dir: str, tier: str = "full", clearance=None):
        from .clearance import Clearance
        self.c = clearance if clearance is not None else Clearance(door_dir, tier)
        self.mj = self.c.mj
        self.m, self.d = self.c.m, self.c.d
        self.dir = door_dir
        with open(os.path.join(door_dir, "model.json")) as f:
            self.ir = json.load(f)
        self.meta = self.ir["meta"]
        self.allow = _load_allow(self.meta)
        m = self.m
        self.gname = self.c.gname
        self.bname = self.c.bname
        self.gbody = np.asarray(m.geom_bodyid)
        self.static = self.c.static                     # per geom: its body is welded to the world
        self.rbound = self.c.rbound                     # planes already widened to 1e6
        self.ir_body = {b["name"]: b for b in self.ir["bodies"]}
        self.mj.mj_forward(m, self.d)

    # ---- distance helpers -------------------------------------------------------------------------------
    def _pairs_within(self, ids_a: Sequence[int], ids_b: Sequence[int], cutoff: float, symmetric: bool = False):
        """Yield (i, j, signed distance) for every pair whose bounding spheres are within ``cutoff``."""
        m, d, mujoco = self.m, self.d, self.mj
        pos = np.asarray(d.geom_xpos)
        ids_b = np.asarray(list(ids_b), dtype=int)
        if ids_b.size == 0:
            return
        for i in ids_a:
            sep = np.linalg.norm(pos[ids_b] - pos[i], axis=1) - self.rbound[ids_b] - self.rbound[i]
            for j in ids_b[sep < cutoff]:
                j = int(j)
                if j == i or (symmetric and j < i):
                    continue
                dist = float(mujoco.mj_geomDistance(m, d, int(i), j, cutoff, None))
                if dist < cutoff:
                    yield int(i), j, dist

    def _min_dist(self, ids_a: Sequence[int], ids_b: Sequence[int], cutoff: float = SEARCH) -> Tuple[float, int, int]:
        best = (cutoff, -1, -1)
        for i, j, dist in self._pairs_within(ids_a, ids_b, cutoff):
            if dist < best[0]:
                best = (dist, i, j)
        return best

    def _clusters(self, ids: Sequence[int], tol: float) -> List[List[int]]:
        """Connected components of ``ids`` under "signed distance <= tol"."""
        ids = list(ids)
        idx = {g: k for k, g in enumerate(ids)}
        parent = list(range(len(ids)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, j, dist in self._pairs_within(ids, ids, tol, symmetric=True):
            if dist <= tol:
                ra, rb = find(idx[i]), find(idx[j])
                if ra != rb:
                    parent[ra] = rb
        groups: Dict[int, List[int]] = {}
        for k, g in enumerate(ids):
            groups.setdefault(find(k), []).append(g)
        return sorted(groups.values(), key=lambda v: -len(v))

    # ---- bookkeeping ------------------------------------------------------------------------------------
    def geoms_of(self, bid: int) -> List[int]:
        return [int(g) for g in np.flatnonzero(self.gbody == bid)]

    def _eq_body_partners(self) -> Dict[int, List[int]]:
        """body id -> bodies it is pinned to by an active connect / weld equality (both directions)."""
        m, mujoco = self.m, self.mj
        out: Dict[int, List[int]] = {}
        for e in range(m.neq):
            if int(m.eq_type[e]) not in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)) or not m.eq_active0[e]:
                continue
            a, b = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            out.setdefault(a, []).append(b)
            out.setdefault(b, []).append(a)
        return out

    def attachment_targets(self, bid: int, partners: Dict[int, List[int]]) -> Tuple[List[int], str]:
        """The geoms body ``bid`` has to be touching, and a word for what they are."""
        m = self.m
        static_ids = [int(g) for g in np.flatnonzero(self.static)]
        # nearest ancestor that actually has geometry
        p = int(m.body_parentid[bid])
        while p > 0 and not self.geoms_of(p):
            p = int(m.body_parentid[p])
        targets: List[int] = []
        what = []
        if p > 0:
            targets += self.geoms_of(p)
            what.append(f"parent {self.bname[p]}")
        for q in partners.get(bid, []):
            if q == 0:
                targets += static_ids
                what.append("world (equality)")
            elif q != bid:
                targets += self.geoms_of(q)
                what.append(f"equality partner {self.bname[q]}")
        if p == 0:
            targets += static_ids
            what.append("static structure")
        return sorted(set(targets)), " / ".join(what) or "nothing"

    # ---- allow-list -------------------------------------------------------------------------------------
    def allowed(self, rule: str, names: Iterable[str]) -> bool:
        names = [n for n in names if n]
        for r_pat, n_pat in self.allow:
            if not fnmatch.fnmatch(rule, r_pat):
                continue
            if any(fnmatch.fnmatch(n, n_pat) for n in names):
                return True
        return False

    def _finding(self, out: list, rule: str, tol_name: str, tol: float, gap, names: List[str], detail: str, **kw):
        if self.allowed(rule, names):
            return
        rec = {"rule": rule, "tolerance": tol_name, "tolerance_m": tol, "gap": None if gap is None else round(float(gap), 5),
               "names": names, "detail": detail}
        rec.update(kw)
        out.append(rec)

    # ---- rule 1 / 3: connectivity at rest ---------------------------------------------------------------
    def check_intra_body(self, out: list):
        """The geoms of one MOVING body form a single cluster (static bodies -> check_static)."""
        m = self.m
        for bid in range(1, m.nbody):
            if int(m.body_weldid[bid]) == 0:
                continue
            ids = self.geoms_of(bid)
            if len(ids) < 2:
                continue
            groups = self._clusters(ids, TOL_INTRA)
            if len(groups) < 2:
                continue
            main = groups[0]
            for island in groups[1:]:
                dist, i, j = self._min_dist(island, [g for g in ids if g not in island])
                self._finding(out, "intra_body_split", "TOL_INTRA", TOL_INTRA, dist,
                              [self.gname[g] for g in island],
                              f"{len(island)} geom(s) of body {self.bname[bid]} form a separate island; nearest other geom of the "
                              f"same body is {self.gname[j] if j >= 0 else '(none within %.2f m)' % SEARCH} at {dist * 1000:.1f} mm",
                              body=self.bname[bid], nearest=self.gname[j] if j >= 0 else None)

    def check_static(self, out: list):
        """Every static geom is connected to the structural root through static geoms."""
        ids = [int(g) for g in np.flatnonzero(self.static)]
        if len(ids) < 2:
            return
        groups = self._clusters(ids, TOL_STATIC)
        if len(groups) < 2:
            return
        # the root component is the one holding the floor (else the largest)
        floor = [g for g in ids if self.gname[g] in ("floor", "ground") or int(self.m.geom_type[g]) == int(self.mj.mjtGeom.mjGEOM_PLANE)]
        root = next((grp for grp in groups if any(f in grp for f in floor)), groups[0])
        for island in groups:
            if island is root:
                continue
            dist, i, j = self._min_dist(island, [g for g in ids if g not in island])
            self._finding(out, "static_detached", "TOL_STATIC", TOL_STATIC, dist,
                          [self.gname[g] for g in island],
                          f"{len(island)} static geom(s) touch no other static geometry; nearest is "
                          f"{self.gname[j] if j >= 0 else '(none within %.2f m)' % SEARCH} at {dist * 1000:.1f} mm",
                          body=self.bname[int(self.gbody[island[0]])], nearest=self.gname[j] if j >= 0 else None)

    # ---- rule 2: every body attached at rest -------------------------------------------------------------
    def check_bodies(self, out: list, q=None, config: str = "rest") -> Dict[int, float]:
        """Distance from each moving body to what carries it; records a finding beyond TOL_BODY."""
        m = self.m
        partners = self._eq_body_partners()
        if q is not None:
            self.d.qpos[:] = q
            self.mj.mj_forward(m, self.d)
        worst: Dict[int, float] = {}
        for bid in range(1, m.nbody):
            if int(m.body_weldid[bid]) == 0:
                continue
            ids = self.geoms_of(bid)
            if not ids:
                continue
            targets, what = self.attachment_targets(bid, partners)
            targets = [t for t in targets if t not in ids]
            if not targets:
                continue
            dist, i, j = self._min_dist(ids, targets)
            worst[bid] = dist
            if dist > TOL_BODY:
                self._finding(out, "body_detached" if config == "rest" else "detached_in_motion", "TOL_BODY", TOL_BODY, dist,
                              [self.bname[bid]] + [self.gname[g] for g in ids],
                              f"body {self.bname[bid]} is {dist * 1000:.1f} mm from {what}"
                              + (f" (closest pair {self.gname[i]} / {self.gname[j]})" if j >= 0 else f" (nothing within {SEARCH} m)")
                              + (f" at {config}" if config != "rest" else ""),
                              body=self.bname[bid], config=config,
                              nearest=[self.gname[i], self.gname[j]] if j >= 0 else None)
        return worst

    # ---- rule 4: still attached through the travel -------------------------------------------------------
    def check_motion(self, out: list, n_steps: int = 8):
        """Re-check body attachment through the travel.

        A body's distance to its parent's geoms can only change when its OWN joint moves (its frame is expressed in
        the parent frame, which the joint is what moves), so every body is evaluated over a sweep of its own joint -
        no combinatorial explosion, and no dependence on where the rest of the door happens to be.  Bodies hung
        directly off the world are evaluated the same way (their attachment target - the static structure - does not
        move either).  Joint equalities and one-sided tendons are resolved with ``Clearance.resolve`` so a driven
        joint follows its driver."""
        m = self.m
        partners = self._eq_body_partners()
        base = m.qpos0.copy()
        for j in range(m.njnt):
            bid = int(m.jnt_bodyid[j])
            if int(m.body_weldid[bid]) == 0:
                continue
            ids = self.geoms_of(bid)
            if not ids:
                continue
            targets, what = self.attachment_targets(bid, partners)
            targets = [t for t in targets if t not in ids]
            if not targets:
                continue
            if m.jnt_limited[j]:
                lo, hi = (float(x) for x in m.jnt_range[j])
            elif int(m.jnt_type[j]) == int(self.mj.mjtJoint.mjJNT_HINGE):
                lo, hi = -math.pi, math.pi
            else:
                continue
            if hi - lo < 1e-6:
                continue
            worst = (-1.0, 0.0, -1, -1)
            for k in range(n_steps + 1):
                qv = lo + (hi - lo) * k / n_steps
                q = base.copy()
                q[m.jnt_qposadr[j]] = qv
                self.d.qpos[:] = self.c.resolve(q)
                self.mj.mj_forward(m, self.d)
                dist, ia, ja = self._min_dist(ids, targets)
                if dist > worst[0]:
                    worst = (dist, qv, ia, ja)
            dist, qv, ia, ja = worst
            if dist > TOL_BODY:
                jn = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_JOINT, j)
                self._finding(out, "detached_in_motion", "TOL_BODY", TOL_BODY, dist,
                              [self.bname[bid]] + [self.gname[g] for g in ids],
                              f"body {self.bname[bid]} leaves {what}: {dist * 1000:.1f} mm at {jn}={qv:.3f}"
                              + (f" (closest pair {self.gname[ia]} / {self.gname[ja]})" if ja >= 0 else ""),
                              body=self.bname[bid], config=f"{jn}={qv:.3f}", joint=jn, q=round(float(qv), 4),
                              nearest=[self.gname[ia], self.gname[ja]] if ja >= 0 else None)
        # restore
        self.d.qpos[:] = self.c.resolve(base.copy())
        self.mj.mj_forward(m, self.d)

    def check_equalities(self, out: list):
        """connect / weld equalities are authored closed (the two anchors coincide in the shipped pose)."""
        m, mujoco, d = self.m, self.mj, self.d
        d.qpos[:] = m.qpos0
        mujoco.mj_forward(m, d)
        for e in range(m.neq):
            et = int(m.eq_type[e])
            if et not in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)) or not m.eq_active0[e]:
                continue
            a, b = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            pa = np.asarray(d.xpos[a]) + np.asarray(d.xmat[a]).reshape(3, 3) @ np.asarray(m.eq_data[e][:3])
            pb = np.asarray(d.xpos[b]) + np.asarray(d.xmat[b]).reshape(3, 3) @ np.asarray(m.eq_data[e][3:6])
            sep = float(np.linalg.norm(pa - pb))
            if sep > TOL_EQ:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or f"eq{e}"
                self._finding(out, "equality_anchor", "TOL_EQ", TOL_EQ, sep, [name, self.bname[a], self.bname[b]],
                              f"equality {name} pins {self.bname[a]} to {self.bname[b]} but its anchors are "
                              f"{sep * 1000:.1f} mm apart in the shipped pose", body=self.bname[a], equality=name)

    # ---- rule 5: degenerate content ----------------------------------------------------------------------
    def check_degenerate(self, out: list):
        m, mujoco = self.m, self.mj
        # zero / near-zero geoms and duplicates, from the IR (which carries the authored sizes)
        for b in self.ir["bodies"]:
            seen: Dict[tuple, str] = {}
            for g in b["geoms"]:
                size = [float(s) for s in g["size"]]
                n = g["name"]
                if g["type"] != "mesh":
                    used = {"box": 3, "cylinder": 2, "capsule": 2, "sphere": 1, "ellipsoid": 3}.get(g["type"], len(size))
                    dims = size[:used]
                    if not all(math.isfinite(s) for s in dims) or (dims and min(dims) < MIN_HALF_EXTENT):
                        self._finding(out, "degenerate_geom", "MIN_HALF_EXTENT", MIN_HALF_EXTENT, min(dims) if dims else 0.0,
                                      [n], f"{g['type']} {n} on {b['name']} has half-extent {min(dims) * 1000:.3f} mm "
                                           f"(size {dims})", body=b["name"])
                key = (g["type"], tuple(round(s, 6) for s in size), tuple(round(p, 6) for p in g["pos"]), tuple(round(q, 6) for q in g["quat"]),
                       g.get("mesh_name"))
                if key in seen:
                    self._finding(out, "duplicate_geom", "DUP_EPS", DUP_EPS, 0.0, [n, seen[key]],
                                  f"{n} is an exact duplicate of {seen[key]} on body {b['name']} (same type, size and pose)",
                                  body=b["name"])
                else:
                    seen[key] = n
            if not b["geoms"] and float(b.get("mass", 0.0)) > 1e-6:
                self._finding(out, "body_without_geoms", "-", 0.0, None, [b["name"]],
                              f"body {b['name']} carries {float(b['mass']):.3f} kg but has no geometry", body=b["name"])
        # mesh bounding boxes, from the compiled model (the authoritative geometry)
        for gi in range(m.ngeom):
            if int(m.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mid = int(m.geom_dataid[gi])
            if mid < 0:
                continue
            adr, num = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            v = np.asarray(m.mesh_vert[adr:adr + num]).reshape(-1, 3)
            if not num or not np.isfinite(v).all():
                self._finding(out, "mesh_bbox", "-", 0.0, None, [self.gname[gi]],
                              f"mesh {self.gname[gi]} has no usable vertices", body=self.bname[int(self.gbody[gi])])
                continue
            half = (v.max(axis=0) - v.min(axis=0)) / 2.0
            declared = [float(s) for s in m.geom_size[gi]]
            bad = None
            if float(half.min()) < MIN_HALF_EXTENT:
                bad = f"degenerate bounding box {2 * half} m"
            elif float(half.max()) > MESH_MAX_HALF_EXTENT:
                bad = f"absurd bounding box {2 * half} m"
            else:
                ir_size = self._ir_geom_size(self.gname[gi])
                if ir_size is not None and not all(abs(s - 1.0) < 1e-9 for s in ir_size):
                    ref = np.asarray(ir_size[:3], dtype=float)
                    if np.any(np.abs(ref - half) > 0.5 * np.maximum(ref, half)):
                        bad = f"declared size {list(ref)} but real half-extents {list(np.round(half, 4))}"
            if bad:
                self._finding(out, "mesh_bbox", "MESH_MAX_HALF_EXTENT", MESH_MAX_HALF_EXTENT, None, [self.gname[gi]],
                              f"mesh {self.gname[gi]}: {bad}", body=self.bname[int(self.gbody[gi])])

    def _ir_geom_size(self, name: str):
        for b in self.ir["bodies"]:
            for g in b["geoms"]:
                if g["name"] == name:
                    return [float(s) for s in g["size"]]
        return None

    # ---- the gate ----------------------------------------------------------------------------------------
    def run(self, n_steps: int = 8) -> dict:
        findings: List[dict] = []
        self.check_static(findings)
        self.check_intra_body(findings)
        self.check_bodies(findings)
        self.check_equalities(findings)
        self.check_motion(findings, n_steps)
        self.check_degenerate(findings)
        by_rule: Dict[str, int] = {}
        for f in findings:
            by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
        findings.sort(key=lambda f: -(f["gap"] or 0.0))
        return {"ok": not findings, "n_findings": len(findings), "by_rule": by_rule, "findings": findings[:40]}


def run_attachment(door_dir: str, tier: str = "full", n_steps: int = 8, clearance=None) -> dict:
    """The attachment gate for one door.  A gate that cannot run is a failure, not a pass."""
    try:
        return Attachment(door_dir, tier, clearance=clearance).run(n_steps)
    except Exception as e:
        return {"ok": False, "n_findings": 1, "by_rule": {"error": 1},
                "findings": [{"rule": "error", "tolerance": "-", "tolerance_m": 0.0, "gap": None, "names": [],
                              "detail": f"{type(e).__name__}: {e}"}]}

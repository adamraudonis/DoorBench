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
                      leaf is attached through its hinge knuckles, a slider through its hangers/rollers; a body
                      carried by a RUNNING fit (it slides, or its nearest neighbour is a track/guide/roller) is
                      attached at TOL_BODY_GUIDED instead, because the running-clearance gate requires it to keep a
                      gap there and the two gates would otherwise contradict each other.
  running_gear_lands  a body that carries running gear (rollers, carriers, hangers, glides - semantic "track")
                    hangs on THAT gear: at least one of those geoms reaches static structure within TOL_BODY_GUIDED.
                    body_detached alone is satisfied by any neighbour that happens to be close.
  static_detached     Every static geom must be connected, through a chain of static geoms with gaps <= TOL_STATIC
                      (3 mm), to the structural root (the component holding the floor).  Jamb plates, keepers,
                      strikes, stops, bumpers, shoes, brackets and tracks all have to land on the frame, wall, floor
                      or ceiling.  Nothing static may hang in the air.  This is the db0024 wall-bumper case.
  detached_in_motion  The same attachment, re-measured through the travel: every body's own joint is swept through
                      its range (a body's distance to its parent can only change when its OWN joint moves) and the
                      worst sample must still satisfy TOL_BODY.  A hinge whose knuckles part company at 90 deg, a
                      bolt that leaves its housing, a roller that runs off the end of its rail.
  equality_anchor     Every connect/weld equality must be authored closed: MuJoCo's own residual for it must be
                      below TOL_EQ (1 mm) in the shipped pose.  (Whether the loop can FOLLOW the leaf through its
                      travel is the business of ``checks["linkage_feasibility"]`` / doorbench/linkage_qa.py, which
                      solves the loop's own mechanism joints with least-squares; this is the cheap authored-pose
                      half of the same requirement and needs no solve.)
  stop_not_struck     Every opening stop the generator builds declares itself in ``meta["stops"]``; at the declared
                      leaf joint's limit the leaf must actually be ON it (within STOP_STRIKE).  A stop that is
                      bolted to the wall but that the leaf never touches is decoration, not hardware.
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
TOL_BODY_GUIDED = 0.012  # m; a body carried by a RUNNING fit - a curtain in its guides, a hanger in its rail, a
#                          roller in its track - is captured by that fit, not resting on it, and the running
#                          clearance gate REQUIRES it to keep a gap there (3 mm structural, 6 mm over a floor,
#                          10 mm on a rotor).  The two gates would contradict each other at TOL_BODY, so a running
#                          fit is attached at 12 mm: just above the largest gap running clearance ever demands, and
#                          far below anything a viewer would call floating.  What counts as a running fit is the
#                          semantics/names in GUIDE_SEM / GUIDE_NAME below.
GUIDE_SEM = ("track",)
GUIDE_NAME = ("*track*", "*guide*", "*roller*", "*hanger*", "*glide*", "*carriage*", "*rail*", "*shoe*", "*trolley*")
TOL_EQ = 0.001           # m; a connect/weld equality is authored closed to a millimetre (shared with
#                          linkage_qa.RESIDUAL_TOL_M so the two gates cannot disagree).
MIN_HALF_EXTENT = 2e-4   # m; the smallest half-extent a real part may have (0.4 mm thick).  The thinnest genuine
#                          parts in the dataset are 1.2 mm faceplates and 2 mm signs, six times this.
MESH_MAX_HALF_EXTENT = 3.0   # m; no single mesh part is bigger than a 6 m box (the largest leaves are boxes).
STOP_STRIKE = 0.003      # m; a stop the leaf never reaches is decoration.  Every opening stop the generator builds
#                          declares itself in meta["stops"]; at the leaf joint's limit the leaf must be ON it.
DUP_EPS = 1e-6           # m / rad; two geoms closer than this in size AND pose are the same geom twice.
SEARCH = 1.5             # m; how far a finding looks to report "the nearest thing is X, that far away".

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
        self.half = self._local_half()
        self._rigid_geoms = {}
        self.mj.mj_forward(m, self.d)

    # ---- distance helpers -------------------------------------------------------------------------------
    def _local_half(self) -> np.ndarray:
        """Half-extents of every geom in its own frame (for the world-AABB overlap test)."""
        m, mujoco = self.m, self.mj
        out = np.zeros((m.ngeom, 3))
        for g in range(m.ngeom):
            t = int(m.geom_type[g])
            s = np.asarray(m.geom_size[g], dtype=float)
            if t == int(mujoco.mjtGeom.mjGEOM_BOX):
                out[g] = s
            elif t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                out[g] = s[0]
            elif t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                out[g] = (s[0], s[0], s[1] + s[0])
            elif t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                out[g] = (s[0], s[0], s[1])
            elif t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                out[g] = s
            elif t == int(mujoco.mjtGeom.mjGEOM_MESH) and int(m.geom_dataid[g]) >= 0:
                mid = int(m.geom_dataid[g])
                v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]).reshape(-1, 3)
                out[g] = np.abs(v).max(axis=0) if len(v) else 0.0
            else:
                out[g] = float(self.rbound[g]) if self.rbound[g] < 1e5 else 1e6
        return out

    def _aabb_overlap(self, i: int, j: int) -> bool:
        """Do the two geoms' world AABBs overlap?

        ``mj_geomDistance`` does NOT return a negative number for every intersecting primitive pair: a capsule that
        passes THROUGH a box (a latch pin in its housing) reads +4 mm, which would make the gate call a part that is
        buried in its housing "detached".  An AABB overlap is conservative in the direction that matters here - a
        part whose bounding box overlaps another's is not a part hanging in the air - so it clamps the query."""
        d = self.d
        ci, cj = np.asarray(d.geom_xpos[i]), np.asarray(d.geom_xpos[j])
        hi = np.abs(np.asarray(d.geom_xmat[i]).reshape(3, 3)) @ self.half[i]
        hj = np.abs(np.asarray(d.geom_xmat[j]).reshape(3, 3)) @ self.half[j]
        return bool(np.all(np.abs(ci - cj) < hi + hj))

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
                if dist > 0 and self._aabb_overlap(int(i), j):
                    dist = 0.0
                if dist < cutoff:
                    yield int(i), j, dist

    def _min_dist(self, ids_a: Sequence[int], ids_b: Sequence[int], cutoff: float = SEARCH) -> Tuple[float, int, int]:
        best = (cutoff, -1, -1)
        for i, j, dist in self._pairs_within(ids_a, ids_b, cutoff):
            if dist < best[0]:
                best = (dist, i, j)
        return best

    def is_guide(self, gid: int) -> bool:
        """Is this geom a running fit (a track, guide, rail, roller, hanger) rather than a fixing?"""
        n = self.gname[gid] or ""
        return self.c.sem.get(n, "") in GUIDE_SEM or any(fnmatch.fnmatch(n, p) for p in GUIDE_NAME)

    def attach_dist(self, ids: Sequence[int], targets: Sequence[int], guided: bool = False) -> Tuple[float, float, int, int]:
        """(effective gap, tolerance that applies, geom a, geom b) for a body against what carries it.

        The closest target decides.  The guided tolerance applies when THIS body is a running fit - it slides, or the
        geom in question is its own running gear (a roller, a hanger, a glide).  Merely being near a rail does not
        earn it: a PVC strip hangs from a clamp bolted to its rail and has to touch that clamp, and while the rail's
        own semantic bought the 12 mm the strips hung 6 mm under it on nothing at all (80 doors).  ``guided`` is
        passed by the caller for a body on a slide joint."""
        best = (SEARCH, TOL_BODY_GUIDED if guided else TOL_BODY, -1, -1)
        for i, j, dist in self._pairs_within(ids, targets, SEARCH):
            tol = TOL_BODY_GUIDED if guided or self.is_guide(i) else TOL_BODY
            if dist - tol < best[0] - best[1]:
                best = (dist, tol, i, j)
        return best

    def is_guided_body(self, bid: int) -> bool:
        """A body carried by a running fit: it (or an ancestor it rides on) slides."""
        m = self.m
        for j in range(m.njnt):
            if int(m.jnt_bodyid[j]) == bid and int(m.jnt_type[j]) == int(self.mj.mjtJoint.mjJNT_SLIDE):
                return True
        return False

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

    def assembly_geoms(self, bid: int) -> List[int]:
        """Own stock plus rigid descendants, excluding every moving joint.

        Explicit material/BOM bodies do not sever a real hinge leaf, bearing
        housing or wheel mount from the member to which it is bolted.
        """
        if bid not in self._rigid_geoms:
            bodies={bid}
            for child in range(bid + 1, self.m.nbody):
                if int(self.m.body_parentid[child]) in bodies and not self.m.body_jntnum[child]:
                    bodies.add(child)
            self._rigid_geoms[bid]=[int(g) for g in np.flatnonzero(np.isin(self.gbody,list(bodies)))]
        return self._rigid_geoms[bid]

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
        while p > 0 and not self.assembly_geoms(p):
            p = int(m.body_parentid[p])
        targets: List[int] = []
        what = []
        if p > 0:
            targets += self.assembly_geoms(p)
            what.append(f"parent {self.bname[p]}")
        for q in partners.get(bid, []):
            if q == 0:
                targets += static_ids
                what.append("world (equality)")
            elif q != bid:
                targets += self.assembly_geoms(q)
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
            ids = self.assembly_geoms(bid)
            if len(ids) < 2:
                continue
            groups = self._clusters(ids, TOL_INTRA)
            if len(groups) < 2:
                continue
            # A fixed child may collect separately fastened bearing housings
            # on the same solid parent. Each island must reach that actual
            # stock. A shared body label alone is never an attachment, and a
            # jointed carriage still needs its own continuous structure.
            if not m.body_jntnum[bid]:
                parent = int(m.body_parentid[bid])
                while parent > 0 and not self.geoms_of(parent) and not m.body_jntnum[parent]:
                    parent = int(m.body_parentid[parent])
                support = [g for g in self.assembly_geoms(parent) if g not in ids] if parent > 0 else []
                # Use actual surface distance here, not the legacy AABB
                # intersection fallback: a bore's empty centre is not stock.
                def seated(island):
                    return any(self.mj.mj_geomDistance(m, self.d, i, j, SEARCH, None) <= TOL_INTRA
                               for i in island for j in support)
                if support and all(seated(island) for island in groups):
                    continue
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
            ids = self.assembly_geoms(bid)
            if not ids:
                continue
            targets, what = self.attachment_targets(bid, partners)
            targets = [t for t in targets if t not in ids]
            if not targets:
                continue
            dist, tol, i, j = self.attach_dist(ids, targets, guided=self.is_guided_body(bid))
            worst[bid] = dist
            if dist > tol:
                self._finding(out, "body_detached" if config == "rest" else "detached_in_motion",
                              "TOL_BODY" if tol == TOL_BODY else "TOL_BODY_GUIDED", tol, dist,
                              [self.bname[bid]] + [self.gname[g] for g in ids],
                              f"body {self.bname[bid]} is {dist * 1000:.1f} mm from {what}"
                              + (f" (closest pair {self.gname[i]} / {self.gname[j]})" if j >= 0 else f" (nothing within {SEARCH} m)")
                              + (f" at {config}" if config != "rest" else ""),
                              body=self.bname[bid], config=config,
                              nearest=[self.gname[i], self.gname[j]] if j >= 0 else None)
        return worst

    # ---- rule 3b: running gear lands on the structure it rides on ----------------------------------------
    def check_running_gear(self, out: list):
        """A moving body that CARRIES running gear must hang on it.

        ``body_detached`` asks only that a body be within reach of *something*; a leaf whose roller carriers end
        50 mm short of the header still passed it because an unrelated sidelite stile happened to be 7.5 mm away
        (28 wheels over 14 automatic sliding doors were built that way).  Running gear - anything the model calls
        semantic ``track``: carriers, wheels, hangers, glides, guides - exists to reach the rail, so at least one
        geom of a body's gear has to be within the running-fit tolerance of static structure.  Bodies with no
        running gear at all are not this rule's business (they are covered by ``body_detached``)."""
        m = self.m
        statics = [int(g) for g in np.flatnonzero(self.static)]
        if not statics:
            return
        for bid in range(1, m.nbody):
            if int(m.body_weldid[bid]) == 0:
                continue
            gear = [g for g in self.assembly_geoms(bid) if self.c.sem.get(self.gname[g], "") == "track"]
            if not gear:
                continue
            dist, _tol, i, j = self.attach_dist(gear, statics, guided=True)
            if dist > TOL_BODY_GUIDED and not self.allowed("running_gear_lands", [self.bname[bid]] + [self.gname[g] for g in gear]):
                self._finding(out, "running_gear_lands", "TOL_BODY_GUIDED", TOL_BODY_GUIDED, dist,
                              [self.bname[bid]] + [self.gname[g] for g in gear],
                              f"the running gear of {self.bname[bid]} ({len(gear)} geom(s)) never reaches static structure: "
                              f"the closest is {self.gname[i] if i >= 0 else '?'} at {dist * 1000:.1f} mm from "
                              f"{self.gname[j] if j >= 0 else 'nothing in range'}",
                              body=self.bname[bid],
                              nearest=[self.gname[i], self.gname[j]] if j >= 0 else None)

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
            ids = self.assembly_geoms(bid)
            if not ids:
                continue
            targets, what = self.attachment_targets(bid, partners)
            targets = [t for t in targets if t not in ids]
            if not targets:
                continue
            guided_ = self.is_guided_body(bid)
            if m.jnt_limited[j]:
                lo, hi = (float(x) for x in m.jnt_range[j])
            elif int(m.jnt_type[j]) == int(self.mj.mjtJoint.mjJNT_HINGE):
                lo, hi = -math.pi, math.pi
            else:
                continue
            jname = m.joint(j).name
            lo, hi = self.c._operating_range(jname, lo, hi)
            if hi - lo < 1e-6:
                continue
            worst = (-1.0, TOL_BODY, 0.0, -1, -1)
            for k in range(n_steps + 1):
                qv = lo + (hi - lo) * k / n_steps
                q = base.copy()
                q[m.jnt_qposadr[j]] = qv
                self.d.qpos[:] = self.c.resolve(q, driven_joint=jname)
                self.mj.mj_forward(m, self.d)
                dist, tol, ia, ja = self.attach_dist(ids, targets, guided=guided_)
                if dist - tol > worst[0] - worst[1]:
                    worst = (dist, tol, qv, ia, ja)
            dist, tol, qv, ia, ja = worst
            if dist > tol:
                jn = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_JOINT, j)
                self._finding(out, "detached_in_motion", "TOL_BODY" if tol == TOL_BODY else "TOL_BODY_GUIDED", tol, dist,
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
            # MuJoCo's own residual for this equality (the first three rows of its constraint block are the
            # position violation for both connect and weld); reading eq_data directly would need the per-type
            # anchor layout and gets welds wrong.
            rows = [k for k in range(d.nefc) if int(d.efc_type[k]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                    and int(d.efc_id[k]) == e]
            sep = float(np.linalg.norm(np.asarray(d.efc_pos)[rows[:3]])) if rows else 0.0
            if sep > TOL_EQ:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or f"eq{e}"
                self._finding(out, "equality_anchor", "TOL_EQ", TOL_EQ, sep, [name, self.bname[a], self.bname[b]],
                              f"equality {name} pins {self.bname[a]} to {self.bname[b]} but its anchors are "
                              f"{sep * 1000:.1f} mm apart in the shipped pose", body=self.bname[a], equality=name)

    # ---- opening stops: mounted AND struck ---------------------------------------------------------------
    def check_stops(self, out: list):
        """Every stop the generator declares in ``meta["stops"]`` must be reached by the leaf at its limit.

        ``static_detached`` already proves the stop is bolted to something; this proves it is not decoration: at the
        declared leaf joint's maximum the leaf has to be within ``STOP_STRIKE`` of the rubber tip.  (The shipped
        db0024 bumper was 14 mm clear of the leaf at 90 deg *and* 0.85 m from the nearest wall.)"""
        m, mujoco = self.m, self.mj
        moving = [int(g) for g in np.flatnonzero(~self.static)]
        for st in self.meta.get("stops", []) or []:
            gname, jn = st.get("geom", ""), st.get("joint", "")
            gi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gi < 0:
                self._finding(out, "stop_not_struck", "-", 0.0, None, [gname],
                              f"meta['stops'] declares {gname} but the model has no such geom")
                continue
            q = m.qpos0.copy()
            j = self.c.jid.get(jn, None)
            if j is not None:
                # the declared strike angle, which may be BEYOND the joint's shipped limit (an engaged door chain
                # shortens the travel; the stop is still correctly installed for the door's opening angle)
                q[m.jnt_qposadr[j]] = float(st["q"]) if st.get("q") is not None else float(m.jnt_range[j][1])
            self.d.qpos[:] = self.c.resolve(q)
            mujoco.mj_forward(m, self.d)
            dist, i, jj = self._min_dist([gi], moving)
            if dist > STOP_STRIKE:
                self._finding(out, "stop_not_struck", "STOP_STRIKE", STOP_STRIKE, dist, [gname],
                              f"stop {gname} ({st.get('mount')} mount) is {dist * 1000:.1f} mm from the nearest moving part "
                              f"at {jn} = limit: the leaf never reaches its stop", body=self.bname[int(self.gbody[gi])])
        self.d.qpos[:] = self.c.resolve(m.qpos0.copy())
        mujoco.mj_forward(m, self.d)

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
        self.check_running_gear(findings)
        self.check_equalities(findings)
        self.check_motion(findings, n_steps)
        self.check_stops(findings)
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

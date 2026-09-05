"""Visual common-sense review of every door (task board G8).

The deterministic gates (force QA, kinematic clearance, attachment) check what they were written to check.  This module
photographs a door the way a person would look at it - closed and fully open, from the robot side, the far side and from
above, plus a hardware close-up and a mechanism close-up - lays the frames out on one labelled review sheet together
with the facts the spec promises (kinematics, travel, hinge count, hardware names, track / stop type), and sends the
sheet to a vision model with a deficiency rubric.  The verdict is strict JSON::

    {"door_id": ..., "ok": bool, "findings": [{"category", "severity", "part", "description", "where"}], ...}

Two reviewers use the same sheets and the same rubric: the Claude API (`review_door`, `run_batch`) and a Claude Code
agent looking at the sheets itself (`reviewer: "claude-code-agent"`).  `write_report` compiles the verdicts on disk into
docs/VISION_REVIEW.md.  The CLI is scripts/vision_review.py.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import random
import textwrap
import time
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# rubric
# ---------------------------------------------------------------------------------------------------------------------
CATEGORIES = {
    "floating_part": "a part that is not attached to anything (a stop, bumper, bracket, arm, plate, wheel, bolt or keeper hanging in the air, not touching the leaf, frame, wall, floor or track it should be mounted on)",
    "interpenetration": "a part passing through another part (leaf through the frame or wall, arm through the door, bolt through a jamb where there is no pocket, panels overlapping)",
    "track_too_short": "a track / rail / guide / channel that does not extend far enough for the travel the spec states, so the leaf or its hangers run past its end when the door is fully open",
    "guide_departure": "a roller, hanger, wheel, bolt, pin or slide that leaves its track, channel, guide loop or keeper in some position of the door",
    "missing_hardware": "hardware the spec implies that is not visible (fewer hinges than the hinge count, fewer dogs / bolts than stated, a closer or operator / latch / lock named in the spec with no geometry, a sliding door with no track or hangers, a keeper / strike missing for a bolt or hook)",
    "duplicate_part": "a part that appears twice where one belongs, or an extra part with no function (two thumb pieces, a second knob, a stray plate)",
    "wrong_scale": "a part whose size is implausible for the product (a 30 cm knob, a hinge as tall as the door, a tiny closer, a wheel larger than the track section)",
    "wrong_face": "hardware on the wrong face or edge of the door (a keypad or card reader on the inside, a closer body on the pull side with a parallel arm, a surface bolt facing the hinge jamb, a pull handle on the wall side of a slider)",
    "mechanism_cannot_work": "a mechanism that cannot function as drawn (a closer arm not connected to the leaf or the frame, a latch bolt with no keeper, a bolt that cannot reach its strike, a hook with nothing to hook onto, a handle that would hit the frame, a fork with no post)",
    "implausible_proportions": "the door as a whole looks wrong (leaf much smaller / larger than the opening, a gap where there should be a jamb, a head or sill missing, a leaf floating above the floor by a large margin)",
    "other_obviously_wrong": "anything else a person would immediately call obviously wrong on a real door",
}
SEVERITIES = ("blocker", "major", "minor")
SEVERITY_HELP = {
    "blocker": "the door would not work or a photo of it would be laughed at (leaf or hangers off the rail, any part hanging in mid-air with nothing behind it - a stop, bumper, bracket, arm - a mechanism not connected, a missing track / hinges / operator)",
    "major": "clearly wrong but the door still works and nothing floats (a keeper or strike missing for a bolt or hook, hardware on the wrong face, a part clearly the wrong size, an arm that could not reach)",
    "minor": "cosmetic (a bracket a little too big, a plate protruding slightly, a small gap)",
}
REVIEWER_API = "claude-api"
REVIEWER_AGENT = "claude-code-agent"

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "door_id": {"type": "string"},
        "ok": {"type": "boolean", "description": "true when there are no blocker or major findings"},
        "summary": {"type": "string", "description": "one sentence on what the sheet shows and the overall impression"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "part": {"type": "string", "description": "the part concerned, using the spec's names where possible (e.g. 'hanger wheel', 'wall bumper stop', 'closer arm')"},
                    "description": {"type": "string", "description": "what is wrong and why a person would notice, in one or two sentences"},
                    "where": {"type": "string", "description": "which sheet panel(s) show it, e.g. 'open / front-iso', 'open / mechanism close-up'"},
                },
                "required": ["category", "severity", "part", "description", "where"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["door_id", "ok", "summary", "findings"],
    "additionalProperties": False,
}

# Anthropic first-party prices, USD per 1M tokens (input, output); batch API is 50 % of these.
# Source: claude-api skill model table (cached 2026-06-24).  Extend via --price if a model is missing.
MODEL_PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
}
DEFAULT_MODEL = "claude-opus-5"        # the model the claude-api skill prescribes; --model claude-sonnet-5 is the cheap pass
DEFAULT_EFFORT = "high"
# output budget used for the cost estimate: adaptive thinking + the JSON verdict (measured usage replaces it after a run)
EST_OUTPUT_TOKENS = 1500
MAX_TOKENS = 4096


# ---------------------------------------------------------------------------------------------------------------------
# door selection
# ---------------------------------------------------------------------------------------------------------------------
def select_doors(manifest: dict, ids: Sequence[str] | None = None, families: Sequence[str] | None = None, limit: int | None = None,
                 sample: int | None = None, per_family: int | None = None, seed: int = 20260904) -> list:
    """Manifest rows to review.  `ids` are always included; `families` filters; `per_family` picks K seeded doors of
    every family (representative sample); `sample` picks N seeded doors of the remainder; `limit` truncates."""
    rows = [d for d in manifest["doors"] if not d.get("error")]
    by_id = {d["id"]: d for d in rows}
    fams = set(families or [])
    pool = [d for d in rows if not fams or d["family"] in fams]
    chosen: dict = {}
    for i in ids or []:
        if i in by_id:
            chosen[i] = by_id[i]
    rng = random.Random(seed)
    if per_family:
        by_fam: dict = {}
        for d in pool:
            by_fam.setdefault(d["family"], []).append(d)
        for fam in sorted(by_fam):
            cand = [d for d in by_fam[fam] if d["id"] not in chosen]
            for d in rng.sample(cand, min(per_family, len(cand))):
                chosen[d["id"]] = d
    if sample:
        cand = [d for d in pool if d["id"] not in chosen]
        for d in rng.sample(cand, min(sample, len(cand))):
            chosen[d["id"]] = d
    if not ids and not per_family and not sample:
        for d in pool:
            chosen[d["id"]] = d
    out = sorted(chosen.values(), key=lambda d: d["index"])
    if limit:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# spec facts (what SHOULD be there)
# ---------------------------------------------------------------------------------------------------------------------
def door_facts(spec: dict, model_json: dict) -> dict:
    kin = spec["kinematics"]
    leaf = spec["leaf"]
    op = spec["opening"]
    joints = [b["joint"] for b in model_json["bodies"] if b.get("joint")]
    leaf_joints = [j for j in joints if j.get("role") in ("primary", "secondary")]

    def fmt_range(j):
        r = j.get("range")
        if r is None:
            return "unlimited"
        if j["type"] == "hinge":
            return f"{math.degrees(r[0]):.0f}..{math.degrees(r[1]):.0f} deg"
        return f"{r[0] * 1000:.0f}..{r[1] * 1000:.0f} mm"

    facts = {
        "door_id": spec["id"], "family": spec["family"], "context": spec.get("context"), "use_case": spec.get("use_case"),
        "kinematics": kin["type"],
        "travel": (f"{kin['travel_m']:.2f} m" if kin.get("travel_m") else (f"{kin['max_open_deg']} deg" if kin.get("max_open_deg") else "unlimited rotation")),
        "leaf": f"{leaf['count']} x {leaf['width']:.3f} x {leaf['height']:.3f} x {leaf['thickness']:.3f} m ({leaf['slab']}, {leaf['panel_style']})",
        "opening": f"{op['width']:.3f} x {op['height']:.3f} m, wall {op['wall_thickness']:.3f} m, frame {op['frame']['kind']}, threshold {op.get('threshold')}",
        "hinge": f"{spec['hinge']['model']} x {spec['hinge']['count']} ({spec['hinge']['side']} side, {spec['hinge']['swing']})",
        "operator": f"{spec['operator']['model']} at {spec['operator'].get('height')} m, sides={spec['operator'].get('sides')}",
        "latch": spec["latch"]["model"],
        "lock": f"{spec['lock']['model']} (engaged={spec['lock']['engaged']})",
        "closer": spec["closer"]["model"],
        "track": kin.get("track"), "roller": kin.get("roller"), "stop": kin.get("stop"),
        "extra_kinematics": {k: v for k, v in kin.items() if k not in ("type", "travel_m", "max_open_deg", "track", "roller", "stop")},
        "seal": spec.get("seal"), "condition": spec.get("condition"), "extras": spec.get("extras", []),
        "n_bodies": len(model_json["bodies"]), "n_joints": len(joints),
        "leaf_joints": [f"{j['name']} ({j['type']}, {fmt_range(j)})" for j in leaf_joints],
        "mechanism_joints": [f"{j['name']} ({j['role']}, {j['type']}, {fmt_range(j)})" for j in joints if j.get("role") not in ("primary", "secondary")],
        "part_labels": sorted({g.get("part_label") or g.get("semantic", "") for b in model_json["bodies"] for g in b["geoms"] if g.get("semantic") not in ("floor", "wall")}),
    }
    return facts


def facts_lines(f: dict) -> list:
    lines = [
        f"{f['door_id']}  |  family {f['family']}  |  {f['context']}: {f['use_case']}",
        f"kinematics {f['kinematics']}, travel {f['travel']}; stop {f['stop']}; track {f['track']}; roller {f['roller']}"
        + (f"; {json.dumps(f['extra_kinematics'])}" if f["extra_kinematics"] else ""),
        f"leaf {f['leaf']}; opening {f['opening']}",
        f"hinge {f['hinge']}; operator {f['operator']}; latch {f['latch']}; lock {f['lock']}; closer {f['closer']}",
        f"seal {f['seal']}; condition {f['condition']}; extras {', '.join(f['extras']) or 'none'}; {f['n_bodies']} bodies, {f['n_joints']} joints; leaf joints: {'; '.join(f['leaf_joints'])}",
        "mechanism joints: " + ("; ".join(f["mechanism_joints"]) or "none"),
    ]
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# sheet rendering
# ---------------------------------------------------------------------------------------------------------------------
HORIZONTAL_FAMILIES = ("hatch_floor", "hatch_ceiling")        # leaves that lie flat: cameras look down / up instead of across
MID_STATE_KINEMATICS = ("slide_horizontal", "slide_vertical", "rotor")
MID_STATE_FAMILIES = ("bifold", "accordion", "garage_tiltup", "revolving", "turnstile_tripod", "turnstile_fullheight")
VIEW_COLUMNS = ("front-iso (robot side, -y)", "back-iso (far side, +y)", "top (plan view)", "close-up")
MECH_NAME_HINTS = ("bumper", "stop", "holder", "hold_open", "prop", "hanger", "roller", "wheel", "track", "rail", "guide", "closer", "arm", "shoe", "strut", "dog", "bolt", "keeper", "strike", "hinge", "pivot")


class SheetRenderer:
    """Renders the review sheet of one door with MuJoCo's offscreen renderer.

    Layout (cells `cell` px, supersampled `supersample`x for antialiasing):

        header: door id + spec facts (what should be there)
        row "closed":  front-iso | back-iso | top | hardware close-up (operator / latch / lock, robot side)
        row "open":    same three views with every leaf joint at its open limit | mechanism close-up (closer, hangers +
                       track, hinges, stops) in the open state
        row "mid":     sliders / vertical lift / folding / rotors only: leaf at 50 % travel | mechanism close-up (mid)
    """

    def __init__(self, door_dir: str, cell=(400, 300), supersample: int = 2):
        import mujoco
        from doorbench.clearance import Clearance
        self.mujoco = mujoco
        self.dir = door_dir
        self.cell = tuple(int(c) for c in cell)
        self.ss = int(supersample)
        rw, rh = self.cell[0] * self.ss, self.cell[1] * self.ss
        spec = mujoco.MjSpec.from_file(os.path.join(door_dir, "door.xml"))
        spec.visual.global_.offwidth = max(rw, 640)
        spec.visual.global_.offheight = max(rh, 480)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.gate = Clearance(door_dir, "full")       # resolve(): joint couplings / tendons, same qpos layout
        with open(os.path.join(door_dir, "model.json")) as f:
            self.mj = json.load(f)
        with open(os.path.join(door_dir, "spec.json")) as f:
            self.spec = json.load(f)
        self.meta = self.mj["meta"]
        self.facts = door_facts(self.spec, self.mj)
        self.sem = {g["name"]: g.get("semantic", "") for b in self.mj["bodies"] for g in b["geoms"]}
        self.static_body = {b["name"]: bool(b.get("static")) for b in self.mj["bodies"]}
        self.joint_role = {b["joint"]["name"]: b["joint"].get("role", "") for b in self.mj["bodies"] if b.get("joint")}
        self.sites = {s["name"]: s for b in self.mj["bodies"] for s in b["sites"]}
        m = self.m
        self.gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.gsem = [self.sem.get(n, "") for n in self.gname]
        self.gbody = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "" for g in range(m.ngeom)]
        self.r = mujoco.Renderer(m, height=rh, width=rw)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        for g in (0, 1, 2):          # everything the viewer shows; group 3 = collision-only proxies stays hidden
            self.opt.geomgroup[g] = 1
        self.fovy = float(m.vis.global_.fovy)
        self.horizontal = self.spec["family"] in HORIZONTAL_FAMILIES or bool(self.meta.get("horizontal"))
        self.wall_y = float(self.meta.get("wall_y", 0.0) or 0.0)

    def close(self):
        self.r.close()

    # ---- configurations ---------------------------------------------------------------------------------------------
    def leaf_joint_ids(self) -> list:
        return [j for j in range(self.m.njnt) if self.joint_role.get(self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_JOINT, j) or "", "") in ("primary", "secondary")]

    def q_state(self, frac: float) -> np.ndarray:
        """Leaf joints at `frac` of the way from their rest value to their open limit, mechanisms at rest, couplings
        resolved (bifold followers, closer arms, joined leaves follow)."""
        m = self.m
        q = m.qpos0.copy()
        kin = self.spec["kinematics"]
        self.forced_open = False
        for j in self.leaf_joint_ids():
            adr = m.jnt_qposadr[j]
            q0 = float(m.qpos0[adr])
            is_hinge = int(m.jnt_type[j]) == int(self.mujoco.mjtJoint.mjJNT_HINGE)
            if m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]
                if hi - lo < 1e-6:
                    continue
                if hi - lo < (0.05 if is_hinge else 0.006):
                    # locked shut (engaged interlock / maglock / padlock: the builder clamps the range to ~0).  The
                    # review wants to see the door open anyway - drive it through the spec's travel in the range's sign
                    sgn = 1.0 if hi - q0 >= q0 - lo else -1.0
                    travel = math.radians(kin.get("max_open_deg") or 90.0) if is_hinge else float(kin.get("travel_m") or 0.9)
                    target = q0 + sgn * travel
                    self.forced_open = True
                else:
                    target = hi if abs(hi - q0) >= abs(lo - q0) else lo       # the limit farther from rest is "open"
            else:
                target = q0 + (1.2 if is_hinge else 1.0)
            q[adr] = q0 + frac * (target - q0)
        return self.gate.resolve(q)

    def has_mid_state(self) -> bool:
        kin = self.spec["kinematics"]["type"]
        return kin in MID_STATE_KINEMATICS or self.spec["family"] in MID_STATE_FAMILIES

    def states(self) -> list:
        """Rows of the sheet: closed, open, then mid-travel (sliders / lifts / folding / rotors) or a low-angle open row
        (hinged doors: floor and wall stops, thresholds, the gap under the leaf, the hinge edge)."""
        st = [("closed", 0.0), ("open", 1.0)]
        st.append(("mid", 0.5) if self.has_mid_state() else ("open-low", 1.0))
        return st

    # ---- framing ----------------------------------------------------------------------------------------------------
    def _set(self, q):
        self.d.qpos[:] = q
        self.mujoco.mj_forward(self.m, self.d)

    def geom_ids(self, semantics=None, names_any=None, moving=None, exclude_sem=("floor", "wall")) -> list:
        out = []
        for g in range(self.m.ngeom):
            if self.m.geom_group[g] == 3:
                continue
            s = self.gsem[g]
            if s in exclude_sem:
                continue
            if semantics is not None and s not in semantics:
                continue
            if names_any is not None and not any(h in self.gname[g] for h in names_any):
                continue
            if moving is not None and self.static_body.get(self.gbody[g], True) == moving:
                continue
            out.append(g)
        return out

    def bbox(self, ids) -> tuple:
        """World-frame AABB of the geoms (tight: the geoms' own AABBs rotated, not bounding spheres)."""
        m, d = self.m, self.d
        if not ids:
            c = np.array([0.0, self.wall_y, 1.0])
            return c - 1.0, c + 1.0
        los, his = [], []
        for g in ids:
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + R @ m.geom_aabb[g][:3]
            h = np.abs(R) @ m.geom_aabb[g][3:]
            los.append(c - h)
            his.append(c + h)
        return np.min(los, axis=0), np.max(his, axis=0)

    def camera(self, lookat, radius: float, az: float, el: float):
        cam = self.mujoco.MjvCamera()
        cam.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        aspect = self.cell[0] / self.cell[1]
        half = math.radians(self.fovy / 2)
        # fit the bounding sphere into the smaller of the vertical / horizontal half-angles
        fit = min(half, math.atan(math.tan(half) * aspect))
        cam.distance = max(0.3, 1.08 * radius / math.sin(fit))
        cam.azimuth, cam.elevation = float(az), float(el)
        return cam

    def render_view(self, q, lookat, radius, az, el):
        from PIL import Image
        self._set(q)
        cam = self.camera(lookat, radius, az, el)
        self.r.update_scene(self.d, camera=cam, scene_option=self.opt)
        img = Image.fromarray(self.r.render())
        if self.ss != 1:
            img = img.resize(self.cell, Image.LANCZOS)
        return img

    def scene_frame(self, q):
        """(lookat, radius_3d, radius_plan) of the door + frame + hardware (walls and floor excluded) in state q."""
        self._set(q)
        lo, hi = self.bbox(self.geom_ids())
        c = (lo + hi) / 2
        r3 = float(np.linalg.norm(hi - lo) / 2) + 0.15
        rp = float(np.linalg.norm((hi - lo)[:2]) / 2) + 0.25
        return c, r3, rp

    def hardware_target(self, q):
        """(centre, radius, side) of the operator / latch / lock, like scripts/hardware_review.py; side = -1 when the
        grip is on the robot side of the wall, +1 on the far side (the close-up looks from that side)."""
        self._set(q)
        m, d = self.m, self.d
        ids = self.geom_ids(semantics=("operator", "latch", "lock"))
        pts = []
        for name, s in self.sites.items():
            if s.get("role") in ("grip", "push"):
                sid = self.mujoco.mj_name2id(m, self.mujoco.mjtObj.mjOBJ_SITE, name)
                if sid >= 0:
                    pts.append(d.site_xpos[sid].copy())
        if pts and ids:
            c = np.mean(pts, axis=0)
            r = max(0.10, max(np.linalg.norm(d.geom_xpos[g] - c) + m.geom_rbound[g] for g in ids))
            return c, min(r, 0.45), self.face_side(c)
        if not ids:
            t = self.meta.get("handle_cam_target")
            c = np.array(t, float) if t else np.array([0.0, self.wall_y, 1.0])
            return c, 0.25, -1.0
        lo, hi = self.bbox(ids)
        c = (lo + hi) / 2
        return c, min(max(float(np.linalg.norm(hi - lo) / 2), 0.10), 0.45), self.face_side(c)

    def face_side(self, p) -> float:
        """-1 if the point is on the robot side (-y) of the leaf it is nearest to, +1 on the far side.  Leaves hang off
        the wall plane (barn doors, roll-up curtains, pocket sliders), so compare with the leaf, not with wall_y."""
        d = self.d
        leaves = self.geom_ids(semantics=("leaf",))
        if not leaves:
            return -1.0 if p[1] <= self.wall_y else 1.0
        g = min(leaves, key=lambda g: (d.geom_xpos[g][0] - p[0]) ** 2 + (d.geom_xpos[g][2] - p[2]) ** 2 + 4.0 * (d.geom_xpos[g][1] - p[1]) ** 2)
        return -1.0 if p[1] <= d.geom_xpos[g][1] else 1.0

    def mechanism_target(self, q, prefer: str = "closer"):
        """(centre, radius, what) of the mechanism to inspect in state q.  prefer="closer": closer linkage, then the
        moving track hardware (hangers / rollers) with the track around it, then stops, mechanisms, hinges.
        prefer="stops": stops / bumpers / hold-opens first, then the top hinge / pivot, then the closer."""
        self._set(q)
        m, d = self.m, self.d
        self._view_side = None      # set by the branch that knows better than "the side of the wall the centre is on"
        self._view_az = None        # an explicit azimuth (hinge close-ups look along the wall)
        stops = self.geom_ids(names_any=("bumper", "floor_stop", "door_stop", "hold_open", "holder", "prop_arm", "holdback", "wall_stop", "dome", "overhead_stop", "stop_arm", "kick_down"))
        hinges = self.geom_ids(semantics=("hinge",))
        closer = self.geom_ids(semantics=("closer",))

        def hinge_view():
            top = max(hinges, key=lambda g: d.geom_xpos[g][2])
            near = [g for g in hinges if abs(d.geom_xpos[g][2] - d.geom_xpos[top][2]) < 0.3]
            lo, hi = self.bbox(near)
            # look along the wall from beyond the hinge jamb, nudged toward the swing side: the knuckles between the
            # jamb and the open leaf's hinge edge are in view whatever the opening angle (a wide-open leaf hides them
            # from the swing side, the wall hides them from the other side)
            v = float(self.meta.get("v", 0.0) or 0.0)
            hx = float(self.meta.get("hinge_x", (lo[0] + hi[0]) / 2) or 0.0)
            self._view_az = (180.0 + 30.0 * v) if hx >= 0 else (-30.0 * v)
            return (lo + hi) / 2, min(max(float(np.linalg.norm(hi - lo) / 2), 0.15), 0.5), "top hinge / pivot"

        if prefer == "stops":
            if stops:
                lo, hi = self.bbox(stops)
                return (lo + hi) / 2, min(max(float(np.linalg.norm(hi - lo) / 2), 0.15), 0.6), "stop / hold-open"
            if hinges:
                return hinge_view()
        if closer:
            lo, hi = self.bbox(closer)
            return (lo + hi) / 2, min(max(float(np.linalg.norm(hi - lo) / 2), 0.15), 0.6), "closer"
        moving_track = self.geom_ids(semantics=("track",), moving=True)
        if moving_track:
            lo, hi = self.bbox(moving_track)
            c = (lo + hi) / 2
            r = float(np.linalg.norm(hi - lo) / 2)
            # include the nearby track so its end is in the frame when a hanger runs past it
            near = [g for g in self.geom_ids(semantics=("track",), moving=False) if abs(d.geom_xpos[g][2] - c[2]) < 0.4]
            if near:
                lo2, hi2 = self.bbox(near)
                # clamp the static track extent to +-0.9 m around the hangers (a 5 m rail would zoom out too far)
                lo2 = np.maximum(lo2, c - 0.9)
                hi2 = np.minimum(hi2, c + 0.9)
                lo, hi = np.minimum(lo, lo2), np.maximum(hi, hi2)
                c = (lo + hi) / 2
                r = float(np.linalg.norm(hi - lo) / 2)
            return c, min(max(r, 0.25), 1.0), "track hardware"
        mech = self.geom_ids(semantics=("mechanism",))
        if stops:
            lo, hi = self.bbox(stops)
            return (lo + hi) / 2, min(max(float(np.linalg.norm(hi - lo) / 2), 0.15), 0.6), "stop / hold-open"
        if mech:
            lo, hi = self.bbox(mech)
            return (lo + hi) / 2, min(max(float(np.linalg.norm(hi - lo) / 2), 0.15), 0.6), "mechanism"
        if hinges:
            return hinge_view()
        tracks = self.geom_ids(semantics=("track",))
        if tracks:
            lo, hi = self.bbox(tracks)
            c = (lo + hi) / 2
            return c, min(max(float(np.linalg.norm(hi - lo) / 2), 0.25), 1.0), "track"
        c, r3, _ = self.scene_frame(q)
        return c, r3, "whole door"

    # ---- views ------------------------------------------------------------------------------------------------------
    def view_angles(self):
        """(front-iso, back-iso, top) as (azimuth, elevation).  MuJoCo azimuth 90 looks along +y (camera on the robot
        side -y); the iso views are 35 deg off-axis toward the hinge / latch edge so both faces of an open leaf show."""
        u = float(self.meta.get("u", 1.0) or 1.0)
        if self.horizontal:
            above, below = (90.0 - 30.0, -50.0), (-90.0 + 30.0, 35.0)
            if self.spec["family"] == "hatch_ceiling":
                return below, above, (90.0, -8.0)          # the robot stands under a ceiling hatch
            return above, below, (90.0, -8.0)
        return (90.0 - u * 35.0, -22.0), (-90.0 + u * 35.0, -22.0), (90.0, -80.0)

    def render_sheet(self, out_path: str, quality: int = 78) -> dict:
        from PIL import Image, ImageDraw, ImageFont
        cw, ch = self.cell
        states = self.states()
        front, back, top = self.view_angles()
        header_lines = facts_lines(self.facts)
        font = ImageFont.load_default(size=12)
        font_b = ImageFont.load_default(size=14)
        wrap = max(60, int(4 * cw / 6.2))
        wrapped = []
        for ln in header_lines:
            wrapped.extend(textwrap.wrap(ln, wrap) or [""])
        head_h = 22 + 15 * len(wrapped)
        W, H = 4 * cw, head_h + ch * len(states)
        sheet = Image.new("RGB", (W, H), (24, 24, 28))
        dr = ImageDraw.Draw(sheet)
        dr.text((8, 4), f"DoorBench review sheet - {self.facts['door_id']} - {self.facts['family']}", fill=(255, 220, 120), font=font_b)
        for i, ln in enumerate(wrapped):
            dr.text((8, 22 + 15 * i), ln, fill=(230, 230, 230), font=font)
        panels = []
        u = float(self.meta.get("u", 1.0) or 1.0)
        for row, (sname, frac) in enumerate(states):
            q = self.q_state(frac)
            if self.forced_open and frac > 0:
                sname = sname + " (forced: lock engaged)"
            c, r3, rp = self.scene_frame(q)
            if sname.startswith("open-low"):
                # hinged doors, third row: low camera (floor stops, thresholds, the gap under the leaf), then a view
                # along the wall from the hinge side (hinges, closer, leaf-to-jamb gap), then the stops close-up
                low_c = np.array([c[0], c[1], min(c[2], 0.9)])
                cells = [
                    ("open / front-iso, low (robot side, -y)", self.render_view(q, low_c, r3, 90.0 + u * 30.0, -7.0)),
                    ("open / back-iso, low (far side, +y)", self.render_view(q, low_c, r3, -90.0 - u * 30.0, -7.0)),
                    ("open / hinge-side view (along the wall)", self.render_view(q, c, r3, 180.0 if u > 0 else 0.0, -12.0)),
                ]
                mc, mr, what = self.mechanism_target(q, prefer="stops")
                side = self._view_side or (-1.0 if mc[1] <= self.wall_y + 1e-6 else 1.0)
                az = 90.0 - u * 35.0 if side < 0 else -90.0 + u * 35.0
                if self._view_az is not None:
                    az = self._view_az
                cells.append((f"open / close-up ({what})", self.render_view(q, mc, mr, az, -30.0 if self._view_az is None else -18.0)))
            else:
                cells = [
                    (f"{sname} / {VIEW_COLUMNS[0]}", self.render_view(q, c, r3, *front)),
                    (f"{sname} / {VIEW_COLUMNS[1]}", self.render_view(q, c, r3, *back)),
                    (f"{sname} / {VIEW_COLUMNS[2]}", self.render_view(q, c, rp if not self.horizontal else r3, *top)),
                ]
                if sname.startswith("closed"):
                    hc, hr, side = self.hardware_target(q)
                    if self.horizontal:
                        az, el = 90.0, -55.0
                    else:
                        az, el = (90.0 - u * 32.0, -16.0) if side < 0 else (-90.0 + u * 32.0, -16.0)
                    cells.append((f"closed / hardware close-up (operator, latch, lock; {'robot' if side < 0 else 'far'} side)", self.render_view(q, hc, hr, az, el)))
                else:
                    mc, mr, what = self.mechanism_target(q)
                    # look from the side of the wall the mechanism is on (hinges: from the swing side); nearly head-on
                    # so rails read as lines and a wheel past the rail end is unmistakable; the mid row looks from
                    # higher up and the other angle
                    side = self._view_side or (-1.0 if mc[1] <= self.wall_y + 1e-6 else 1.0)
                    if sname.startswith("open"):
                        az, el = (90.0 - 18.0 * u, -12.0) if side < 0 else (-90.0 + 18.0 * u, -12.0)
                    else:
                        az, el = (90.0 + 30.0 * u, -38.0) if side < 0 else (-90.0 - 30.0 * u, -38.0)
                    if self.horizontal:
                        az, el = (60.0, -40.0) if sname.startswith("open") else (120.0, 30.0)
                    if self._view_az is not None:
                        az, el = self._view_az, -18.0
                    cells.append((f"{sname} / mechanism close-up ({what})", self.render_view(q, mc, mr, az, el)))
            for col, (label, img) in enumerate(cells):
                x, y = col * cw, head_h + row * ch
                sheet.paste(img, (x, y))
                tw = dr.textlength(label, font=font)
                dr.rectangle([x, y, x + min(cw, tw + 10), y + 16], fill=(0, 0, 0))
                dr.text((x + 4, y + 1), label, fill=(255, 255, 255), font=font)
                dr.rectangle([x, y, x + cw - 1, y + ch - 1], outline=(70, 70, 80))
                panels.append({"row": sname, "col": col, "label": label})
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        sheet.save(out_path, quality=quality, optimize=True)
        return {"path": out_path, "width": W, "height": H, "panels": panels, "states": [s for s, _ in states], "facts": self.facts}


def render_sheet(door_dir: str, out_path: str, cell=(400, 300), supersample: int = 2) -> dict:
    r = SheetRenderer(door_dir, cell=cell, supersample=supersample)
    try:
        return r.render_sheet(out_path)
    finally:
        r.close()


# ---------------------------------------------------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------------------------------------------------
def system_prompt() -> str:
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    sev = "\n".join(f"- {k}: {v}" for k, v in SEVERITY_HELP.items())
    return f"""You are a meticulous door-hardware inspector reviewing procedurally generated 3D door models for a robotics simulation dataset (DoorBench). Every door is supposed to be a complete, working mechanism that a real installer would recognise: leaf, frame, hinges or track, operator (handle), latch, lock, closer, stops, all mounted on something and all sized like the real product.

You receive one review sheet per door: a grid of renders from a physics simulator (flat shading, no textures, plain grey wall and floor). The header states what the spec promises (family, kinematics, travel, hinge count, operator / latch / lock / closer models, track, roller and stop type, joint list). Panel labels give the door state and view:
- row "closed": the shipped configuration; row "open": every leaf joint at its full travel; row "mid" (sliders, vertical lift, folding, rotors only): 50 % travel.
- columns: front-iso from the robot side (-y), back-iso from the far side (+y), top (plan view looking down; side view for hatches), and a close-up: the hardware (operator / latch / lock) in the closed row, the mechanism (closer linkage, hangers + track, hinges, stops / hold-opens) in the open and mid rows.

Your job is common sense, not physics: look for what a person would call obviously wrong on a real door. Use the header facts to know what SHOULD be visible and check that it is. Finding categories:
{cats}

Severity:
{sev}

Rules:
- Compare the open row against the closed row: a hanger, roller, bolt or arm that is on its guide when closed but past the end of the rail / track / keeper when open is a blocker (track_too_short or guide_departure). A track must extend at least the stated travel beyond the closed position of the hangers.
- Count hardware against the spec (hinges, dogs, bolts, panels, wings). Report missing_hardware only when the count is clearly lower than the spec and the missing item would be in view.
- A part touching nothing in any view (in mid-air beside the door, off the wall, above the floor) is floating_part; a stop or bumper that is not on a wall, floor, frame or leaf counts even when the door still works.
- Do NOT report: rendering artefacts (aliasing, shadow acne, z-fighting stripes, dark glass, the camera clipping a big part at the panel edge), the plain grey environment, missing textures, parts hidden inside the leaf by design (mortise bolts, spindles, concealed closers), collision-only proxies (not drawn), or the far-side environment being open space.
- Be specific: name the part, say what is wrong, and say which panel shows it. Prefer few, certain findings over many speculative ones; if the door looks like the product, return ok=true with an empty findings list.
- Output only the JSON verdict matching the schema (door_id, ok, summary, findings[category, severity, part, description, where]). ok is true when there are no blocker or major findings."""


def user_prompt(facts: dict, sheet_info: dict | None = None) -> str:
    lines = facts_lines(facts)
    panels = ""
    if sheet_info and sheet_info.get("panels"):
        panels = "\nPanels on this sheet: " + "; ".join(p["label"] for p in sheet_info["panels"])
    return "Review sheet for one door. Spec facts (what should be there):\n" + "\n".join(lines) + f"\nExpected parts by label: {', '.join(facts.get('part_labels', []))}" + panels + "\n\nInspect every panel and return the JSON verdict."


def image_block(path: str) -> dict:
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("ascii")
    media = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def build_request(sheet_path: str, facts: dict, sheet_info: dict | None = None, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT,
                  max_tokens: int = MAX_TOKENS, with_image: bool = True) -> dict:
    """Keyword arguments for client.messages.create (Claude API, vision + structured JSON output).  The system prompt is
    identical for every door and marked for prompt caching; the door-specific text follows the image."""
    content = []
    if with_image:
        content.append(image_block(sheet_path))
    else:
        content.append({"type": "text", "text": f"[image: {os.path.basename(sheet_path)}]"})
    content.append({"type": "text", "text": user_prompt(facts, sheet_info)})
    req = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": content}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
    }
    return req


# ---------------------------------------------------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------------------------------------------------
class VerdictError(ValueError):
    pass


def normalise_verdict(obj: dict, door_id: str, reviewer: str, model: str | None = None, extra: dict | None = None) -> dict:
    """Validate the model's JSON against the rubric and make `ok` consistent with the findings."""
    if not isinstance(obj, dict):
        raise VerdictError(f"verdict is not an object: {type(obj).__name__}")
    findings_in = obj.get("findings")
    if not isinstance(findings_in, list):
        raise VerdictError("findings missing or not a list")
    findings = []
    for i, f in enumerate(findings_in):
        if not isinstance(f, dict):
            raise VerdictError(f"finding {i} is not an object")
        cat = str(f.get("category", "")).strip()
        sev = str(f.get("severity", "")).strip().lower()
        if cat not in CATEGORIES:
            raise VerdictError(f"finding {i}: unknown category {cat!r}")
        if sev not in SEVERITIES:
            raise VerdictError(f"finding {i}: unknown severity {sev!r}")
        for k in ("part", "description", "where"):
            if not isinstance(f.get(k), str) or not f[k].strip():
                raise VerdictError(f"finding {i}: {k} missing")
        nf = {"category": cat, "severity": sev, "part": f["part"].strip(), "description": f["description"].strip(), "where": f["where"].strip()}
        if isinstance(f.get("triage"), dict):
            nf["triage"] = f["triage"]
        findings.append(nf)
    ok = not any(f["severity"] in ("blocker", "major") for f in findings)
    out = {"door_id": door_id, "ok": ok, "summary": str(obj.get("summary", "")).strip(), "findings": findings, "reviewer": reviewer}
    if obj.get("door_id") and obj["door_id"] != door_id:
        out["reported_door_id"] = obj["door_id"]
    if model:
        out["model"] = model
    if extra:
        out.update(extra)
    return out


def parse_verdict_text(text: str) -> dict:
    """The structured-output path guarantees JSON; be lenient with a fenced block anyway."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            return json.loads(t[a:b + 1])
        raise VerdictError(f"not JSON: {e}") from e


def message_text(message) -> str:
    parts = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def usage_dict(message) -> dict:
    u = getattr(message, "usage", None)
    if u is None:
        return {}
    out = {}
    for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        v = getattr(u, k, None)
        if v is not None:
            out[k] = int(v)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------------------------------------------------
def image_tokens(width: int, height: int) -> int:
    """Claude vision pricing: about one token per 750 pixels (images above the model's long-edge limit are downscaled
    first; the review sheets stay under 2576 px so nothing is lost on current models)."""
    return int(math.ceil(width * height / 750.0))


def text_tokens(text: str) -> int:
    """Conservative offline estimate (about 3.4 characters per token for English prose with identifiers).  A live run
    replaces it with the measured usage from the responses."""
    return int(math.ceil(len(text) / 3.4))


def estimate_cost(sheets: Iterable[dict], model: str = DEFAULT_MODEL, output_tokens: int = EST_OUTPUT_TOKENS, batch: bool = False,
                  prices: dict | None = None) -> dict:
    """Pre-run estimate from the sheet image sizes.  `sheets` are dicts with width/height/facts (render_sheet output)."""
    prices = prices or MODEL_PRICES
    if model not in prices:
        raise KeyError(f"no price for model {model!r}; known: {sorted(prices)} (pass --price in,out)")
    p_in, p_out = prices[model]
    sys_tok = text_tokens(system_prompt())
    n = 0
    img_tok = 0
    user_tok = 0
    for s in sheets:
        n += 1
        img_tok += image_tokens(s["width"], s["height"])
        user_tok += text_tokens(user_prompt(s["facts"], s))
    # the system prompt is cached after the first request (cache read = 10 % of input price on the standard tiers)
    sys_first = sys_tok
    sys_cached = sys_tok * max(0, n - 1)
    in_tokens = img_tok + user_tok + sys_first + sys_cached
    cost_in = (img_tok + user_tok + sys_first) * p_in / 1e6 + sys_cached * 0.1 * p_in / 1e6
    cost_out = n * output_tokens * p_out / 1e6
    total = cost_in + cost_out
    if batch:
        total *= 0.5
    return {"model": model, "n_doors": n, "image_tokens": img_tok, "text_tokens": user_tok + sys_first + sys_cached, "input_tokens": in_tokens,
            "output_tokens": n * output_tokens, "usd": round(total, 4), "usd_per_door": round(total / n, 5) if n else 0.0, "batch": batch}


def cost_from_usage(usages: Iterable[dict], model: str, batch: bool = False, prices: dict | None = None) -> float:
    prices = prices or MODEL_PRICES
    p_in, p_out = prices.get(model, (0.0, 0.0))
    usd = 0.0
    for u in usages:
        usd += u.get("input_tokens", 0) * p_in / 1e6 + u.get("output_tokens", 0) * p_out / 1e6
        usd += u.get("cache_creation_input_tokens", 0) * 1.25 * p_in / 1e6 + u.get("cache_read_input_tokens", 0) * 0.1 * p_in / 1e6
    return usd * (0.5 if batch else 1.0)


# ---------------------------------------------------------------------------------------------------------------------
# Claude API transport
# ---------------------------------------------------------------------------------------------------------------------
def make_client(api_key: str | None = None, max_retries: int = 4, timeout: float = 300.0):
    """anthropic.Anthropic(); the SDK retries 429 / 5xx / connection errors itself (`max_retries`)."""
    import anthropic
    kw = {"max_retries": max_retries, "timeout": timeout}
    if api_key:
        kw["api_key"] = api_key
    return anthropic.Anthropic(**kw)


def review_door(client, sheet_path: str, facts: dict, sheet_info: dict | None = None, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT,
                attempts: int = 3, sleep=time.sleep) -> tuple:
    """One synchronous Messages API call (vision + json_schema output) -> (verdict, usage).  Retries a malformed verdict
    (asks again with the validation error), a truncated one (more max_tokens) and transport errors the SDK gave up on;
    a `refusal` stop reason is not retried."""
    try:
        import anthropic
        retryable = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)
    except Exception:  # pragma: no cover - SDK not installed: only mocked clients work
        retryable = ()
    req = build_request(sheet_path, facts, sheet_info, model=model, effort=effort)
    last = None
    for attempt in range(attempts):
        try:
            msg = client.messages.create(**req)
        except retryable as e:
            last = e
            sleep(min(30.0, 2.0 * 2 ** attempt))
            continue
        usage = usage_dict(msg)
        stop = getattr(msg, "stop_reason", None)
        if stop == "refusal":
            det = getattr(msg, "stop_details", None)
            raise VerdictError(f"model refused: {getattr(det, 'category', None)} {getattr(det, 'explanation', '')}")
        if stop == "max_tokens":
            req = dict(req, max_tokens=int(req["max_tokens"] * 2))
            last = VerdictError("truncated (max_tokens)")
            continue
        try:
            verdict = normalise_verdict(parse_verdict_text(message_text(msg)), facts["door_id"], REVIEWER_API, model,
                                        extra={"effort": effort, "usage": usage, "stop_reason": stop, "request_id": getattr(msg, "_request_id", None)})
            return verdict, usage
        except (VerdictError, json.JSONDecodeError) as e:
            last = e
            # ask again, quoting the validation error
            req = dict(req)
            req["messages"] = req["messages"] + [{"role": "assistant", "content": message_text(msg) or "{}"},
                                                 {"role": "user", "content": f"That verdict was rejected: {e}. Return the corrected JSON verdict only."}]
    raise VerdictError(f"no valid verdict for {facts['door_id']} after {attempts} attempts: {last}")


def run_batch(client, items: Sequence[dict], model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, poll_s: float = 30.0, sleep=time.sleep,
              log=print) -> dict:
    """Message Batches API (50 % price, up to 24 h): items = [{door_id, sheet, facts, sheet_info}] -> {door_id: (verdict | None, error)}."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    requests = [Request(custom_id=it["door_id"], params=MessageCreateParamsNonStreaming(**build_request(it["sheet"], it["facts"], it.get("sheet_info"), model=model, effort=effort)))
                for it in items]
    batch = client.messages.batches.create(requests=requests)
    log(f"batch {batch.id}: {len(requests)} requests submitted")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        log(f"batch {batch.id}: {b.processing_status}, {b.request_counts.processing} processing")
        sleep(poll_s)
    facts_by = {it["door_id"]: it["facts"] for it in items}
    out = {}
    for res in client.messages.batches.results(batch.id):
        did = res.custom_id
        if res.result.type == "succeeded":
            msg = res.result.message
            try:
                v = normalise_verdict(parse_verdict_text(message_text(msg)), did, REVIEWER_API, model,
                                      extra={"effort": effort, "usage": usage_dict(msg), "stop_reason": getattr(msg, "stop_reason", None), "batch_id": batch.id})
                out[did] = (v, None)
            except (VerdictError, json.JSONDecodeError) as e:
                out[did] = (None, f"invalid verdict: {e}")
        else:
            err = getattr(res.result, "error", None)
            out[did] = (None, f"{res.result.type}: {getattr(err, 'type', '') if err else ''} {getattr(err, 'message', '') if err else ''}".strip())
    for did in facts_by:
        out.setdefault(did, (None, "no result returned"))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------
def load_verdicts(verdict_dir: str) -> list:
    out = []
    for fn in sorted(os.listdir(verdict_dir)) if os.path.isdir(verdict_dir) else []:
        if fn.endswith(".json") and not fn.startswith("_"):
            with open(os.path.join(verdict_dir, fn)) as f:
                try:
                    v = json.load(f)
                except json.JSONDecodeError:
                    continue
            if isinstance(v, dict) and "findings" in v:
                out.append(v)
    return out


def gate_status(door_dir: str) -> dict:
    """What the deterministic gates said about this door (qa.json): signed_off, clearance, attachment (if the gate exists)."""
    p = os.path.join(door_dir, "qa.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        qa = json.load(f)
    ch = qa.get("checks", {})
    return {"signed_off": qa.get("signed_off"), "clearance": ch.get("clearance"), "attachment": ch.get("attachment"),
            "failed": sorted(k for k, v in ch.items() if v is False)}


TRIAGE_CLASSES = {"geometry_bug": "geometry bug to fix", "render_artefact": "rendering artefact", "false_positive": "false positive", "fixed": "geometry bug, fixed"}

HOW_TO_RUN = """## How to run it

```bash
# 1. render the sheets and write the prompts, no API call (inspect docs/review/vision/<door>.jpg + <door>.prompt.json)
PYTHONPATH=$PWD python scripts/vision_review.py --dry-run --per-family 4 --doors db0079_sliding_single,db0024_swing_single

# 2. live review with the Claude API (ANTHROPIC_API_KEY in the environment); resumable: doors with a verdict on disk are skipped
export ANTHROPIC_API_KEY=sk-ant-...
PYTHONPATH=$PWD python scripts/vision_review.py --max-cost-usd 60                 # all 1000 doors, claude-opus-5
PYTHONPATH=$PWD python scripts/vision_review.py --model claude-sonnet-5 --batch    # cheaper: Sonnet 5 through the Message Batches API (50 %)
PYTHONPATH=$PWD python scripts/vision_review.py --families sliding_single --force  # re-review one family after a geometry change

# 3. rebuild docs/VISION_REVIEW.md from the verdicts on disk (no rendering, no API)
PYTHONPATH=$PWD python scripts/vision_review.py --from-verdicts
```

The pre-run cost estimate is printed (and enforced by `--max-cost-usd`) before the first request: it counts the sheet
pixels (about one token per 750 px), the prompt text, one cached system prompt, and a 1500-token output budget per door
(adaptive thinking + the JSON verdict); measured usage is written into every verdict and summed at the end.
"""


def write_report(verdicts: list, out_md: str, assets: str, sheets_dir: str, manifest: dict | None = None, cost_estimates: list | None = None,
                 intro: str = "", handoff: str = "", triage_notes: str = "") -> str:
    """docs/VISION_REVIEW.md: how to run, counts by severity / category / family, every blocker & major finding with its
    sheet grouped by family, the gate comparison, triage and handoff sections."""
    rel = lambda p: os.path.relpath(p, os.path.dirname(os.path.abspath(out_md)))  # noqa: E731
    fam_of = {}
    if manifest:
        fam_of = {d["id"]: d["family"] for d in manifest["doors"]}
    for v in verdicts:
        v.setdefault("family", fam_of.get(v["door_id"], v["door_id"].split("_", 1)[1] if "_" in v["door_id"] else "?"))
    n = len(verdicts)
    n_ok = sum(1 for v in verdicts if v["ok"])
    all_f = [(v, f) for v in verdicts for f in v["findings"]]
    by_sev = {s: sum(1 for _, f in all_f if f["severity"] == s) for s in SEVERITIES}
    fams = sorted({v["family"] for v in verdicts})
    reviewers = sorted({v.get("reviewer", "?") for v in verdicts})
    models = sorted({v.get("model", "") for v in verdicts if v.get("model")})
    L = []
    L.append("# Visual common-sense review (task G8)\n")
    L.append(intro.rstrip() + "\n" if intro else "")
    L.append(HOW_TO_RUN)
    if cost_estimates:
        L.append("### Expected cost\n")
        L.append("| model | doors | input tokens | output tokens (budget) | USD | USD / door | batch API |")
        L.append("|---|---|---|---|---|---|---|")
        for c in cost_estimates:
            L.append(f"| {c['model']} | {c['n_doors']} | {c['input_tokens']:,} | {c['output_tokens']:,} | {c['usd']:.2f} | {c['usd_per_door']:.4f} | {'yes' if c['batch'] else 'no'} |")
        L.append("")
    L.append("## Summary\n")
    L.append(f"{n} doors reviewed ({', '.join(fams) if len(fams) < 8 else f'{len(fams)} families'}); reviewer(s): {', '.join(reviewers)}"
             + (f"; model(s): {', '.join(models)}" if models else "") + ".\n")
    L.append(f"**{n_ok} / {n} ok**, {n - n_ok} with blocker or major findings; findings: **{by_sev['blocker']} blocker**, {by_sev['major']} major, {by_sev['minor']} minor.\n")
    usages = [v.get("usage") for v in verdicts if v.get("usage")]
    if usages and models:
        spent = cost_from_usage(usages, models[0], batch=any(v.get("batch_id") for v in verdicts))
        L.append(f"Measured API usage: {sum(u.get('input_tokens', 0) + u.get('cache_read_input_tokens', 0) + u.get('cache_creation_input_tokens', 0) for u in usages):,} input, "
                 f"{sum(u.get('output_tokens', 0) for u in usages):,} output tokens, about ${spent:.2f}.\n")
    # category x family table
    L.append("### Findings by category and family\n")
    cats_present = [c for c in CATEGORIES if any(f["category"] == c for _, f in all_f)]
    if cats_present:
        L.append("| category | " + " | ".join(fams) + " | total |")
        L.append("|---|" + "---|" * (len(fams) + 1))
        for c in cats_present:
            row = []
            for fam in fams:
                k = sum(1 for v, f in all_f if f["category"] == c and v["family"] == fam)
                row.append(str(k) if k else "")
            L.append(f"| {c} | " + " | ".join(row) + f" | {sum(1 for _, f in all_f if f['category'] == c)} |")
        L.append("| **doors reviewed** | " + " | ".join(str(sum(1 for v in verdicts if v['family'] == fam)) for fam in fams) + f" | {n} |")
        L.append("| **doors ok** | " + " | ".join(str(sum(1 for v in verdicts if v['family'] == fam and v['ok'])) for fam in fams) + f" | {n_ok} |")
    else:
        L.append("No findings.")
    L.append("")
    # triage summary
    triaged = [(v, f) for v, f in all_f if isinstance(f.get("triage"), dict)]
    if triaged:
        L.append("### Triage\n")
        L.append("| class | blocker | major | minor |")
        L.append("|---|---|---|---|")
        for cls, name in TRIAGE_CLASSES.items():
            row = [sum(1 for _, f in triaged if f["triage"].get("class") == cls and f["severity"] == s) for s in SEVERITIES]
            if any(row):
                L.append(f"| {name} | " + " | ".join(str(x) for x in row) + " |")
        L.append("")
        if triage_notes:
            L.append(triage_notes.rstrip() + "\n")
    # gates comparison
    L.append("## What the deterministic gates would not have caught\n")
    rows = []
    for v in verdicts:
        if v["ok"]:
            continue
        g = gate_status(os.path.join(assets, "doors", v["door_id"]))
        rows.append((v, g))
    if rows:
        L.append("Doors with blocker / major findings and what `qa.json` says about them (signed_off = force QA + mass + clearance + formats; "
                 "attachment = the G7 gate when present in this build):\n")
        L.append("| door | family | worst | signed_off | clearance | attachment | failed checks | vision findings |")
        L.append("|---|---|---|---|---|---|---|---|")
        for v, g in rows:
            worst = "blocker" if any(f["severity"] == "blocker" for f in v["findings"]) else "major"
            summary = "; ".join(f"{f['category']} ({f['severity']}): {f['part']}" for f in v["findings"] if f["severity"] != "minor")
            att = g.get("attachment")
            L.append(f"| {v['door_id']} | {v['family']} | {worst} | {g.get('signed_off')} | {g.get('clearance')} | {'n/a' if att is None else att} | {', '.join(g.get('failed', [])) or '-'} | {summary} |")
        n_signed = sum(1 for _, g in rows if g.get("signed_off"))
        n_clear = sum(1 for _, g in rows if g.get("clearance"))
        L.append(f"\n{n_signed} / {len(rows)} of these doors are signed off and {n_clear} / {len(rows)} pass the clearance gate: none of the visual findings "
                 "below is an interpenetration or a force-QA failure. They are things that only a look at the picture (or the attachment gate, "
                 "for floating parts) can catch: parts hanging in the air, rails ending before the travel does, hardware counts, wrong faces.\n")
    else:
        L.append("No blocker / major findings.\n")
    # findings by family
    L.append("## Blocker and major findings by family\n")
    for fam in fams:
        vs = [v for v in verdicts if v["family"] == fam and not v["ok"]]
        if not vs:
            continue
        L.append(f"### {fam} ({len(vs)} of {sum(1 for v in verdicts if v['family'] == fam)} reviewed doors)\n")
        for v in sorted(vs, key=lambda v: (0 if any(f['severity'] == 'blocker' for f in v['findings']) else 1, v["door_id"])):
            sheet = os.path.join(sheets_dir, f"{v['door_id']}.jpg")
            L.append(f"#### {v['door_id']}\n")
            if v.get("summary"):
                L.append(f"_{v['summary']}_\n")
            if os.path.exists(sheet):
                L.append(f"![{v['door_id']}]({rel(sheet)})\n")
            for f in sorted(v["findings"], key=lambda f: SEVERITIES.index(f["severity"])):
                tri = f.get("triage") or {}
                tri_s = ""
                if tri:
                    tri_s = f" - **{TRIAGE_CLASSES.get(tri.get('class'), tri.get('class'))}**" + (f" ({tri['owner']})" if tri.get("owner") else "") + (f": {tri['note']}" if tri.get("note") else "")
                L.append(f"- **{f['severity']}** `{f['category']}` {f['part']}: {f['description']} _[{f['where']}]_{tri_s}")
            L.append("")
    # minors
    minors = [(v, f) for v, f in all_f if f["severity"] == "minor"]
    if minors:
        L.append("## Minor findings\n")
        L.append("| door | category | part | description | where | triage |")
        L.append("|---|---|---|---|---|---|")
        for v, f in minors:
            tri = f.get("triage") or {}
            L.append(f"| {v['door_id']} | {f['category']} | {f['part']} | {f['description']} | {f['where']} | {TRIAGE_CLASSES.get(tri.get('class'), tri.get('class', ''))} {tri.get('note', '')} |")
        L.append("")
    if handoff:
        L.append("## Handoff\n")
        L.append(handoff.rstrip() + "\n")
    # all doors table
    L.append("## All reviewed doors\n")
    L.append("| door | family | ok | blocker | major | minor | sheet |")
    L.append("|---|---|---|---|---|---|---|")
    for v in sorted(verdicts, key=lambda v: v["door_id"]):
        c = {s: sum(1 for f in v["findings"] if f["severity"] == s) for s in SEVERITIES}
        sheet = os.path.join(sheets_dir, f"{v['door_id']}.jpg")
        link = f"[sheet]({rel(sheet)})" if os.path.exists(sheet) else ""
        L.append(f"| {v['door_id']} | {v['family']} | {'yes' if v['ok'] else 'no'} | {c['blocker'] or ''} | {c['major'] or ''} | {c['minor'] or ''} | {link} |")
    L.append("")
    text = "\n".join(L)
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(text)
    return text

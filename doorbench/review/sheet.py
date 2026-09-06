"""Render one labelled review sheet per door with the MuJoCo offscreen renderer.

A sheet is a single JPEG, ~1200 px wide, laid out as a caption band over a 3 x 4 grid of panels:

    row 1   CLOSED        front-iso        back-iso        hinge/track-side
    row 2   MID-TRAVEL    front-iso        back-iso        hinge/track-side
    row 3   FULLY OPEN    front-iso        back-iso        hinge/track-side
    row 4   close-ups     hardware (near)  hardware (far)  mechanism (open)

Two properties make the sheet reviewable rather than merely pretty:

* **One camera per column, shared by all three poses.**  The framing is computed once from the union
  of the door's bounding boxes over the three poses, so the three rows are the same shot of the same
  scene at three points in the travel.  A rail that is long enough when the door is shut and too
  short when it is open is then a difference between two panels, not a difference between two
  arbitrary crops.
* **The caption states what the spec says should be there** - family, kinematics and travel, hinge
  count, operator / latch / lock / closer model names, leaf size.  A reviewer can only judge
  *completeness* (hardware the spec implies but that is not visible) against a statement of intent.

Poses are kinematic, not dynamic: joint equalities and one-sided tendon couplings are resolved
exactly as the clearance gate resolves them, and closed loops (closer arms, boltwork cranks) are
solved numerically over their own mechanism joints, with the residual recorded on the sheet so an
unsolved linkage is never silently presented as a valid pose.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------------------------
POSES = ("closed", "mid", "open")
VIEWS = ("front", "back", "edge")
VIEW_TITLE = {
    "front": "front-iso (robot side, -y)",
    "back": "back-iso (far side, +y)",
    "edge": "hinge-side",          # replaced per family by sheet_view_titles()
}
POSE_TITLE = {"closed": "CLOSED", "mid": "MID-TRAVEL (50 %)", "open": "FULLY OPEN"}

# geoms that make up the *mechanism* (the thing that carries the motion) rather than the hardware
# the robot touches.  Semantics first, then names, because rollers and hangers are modelled as
# 'track' on some families and as part of the leaf body on others.
MECH_SEM = ("track", "hinge", "closer", "mechanism")
MECH_NAME_HINTS = ("roller", "hanger", "wheel", "carrier", "trolley", "guide", "rail", "track",
                   "pivot", "bearing", "arm", "shoe", "slide", "stop", "spring", "drum", "cable",
                   "crank", "rod", "gear", "torque", "sprocket", "chain", "counterbalance")
HW_SEM = ("operator", "latch", "lock")
MECH_MARGIN_M = 0.15          # m of static surroundings kept around the moving mechanism close-up
# which moving mechanism the close-up frames when several are present.  The closer arm comes first
# because a closer loop floating in mid-air (db0012, 2026-09-04) is the defect this panel exists for.
MECH_PRIORITY = ("closer", "track", "hinge", "mechanism")

# a primary joint whose entire range is under these cannot open its door; the pose shown is an
# inspection pose, not a claim that the door opens
PINNED_HINGE_RAD = math.radians(6.0)
PINNED_SLIDE_M = 0.05

# alpha the review render gives clear glazing so an open glass door does not look like a shut one
GLASS_ALPHA = 0.55

SLIDE_FAMILIES = ("sliding_single", "sliding_bypass", "garage_sectional", "rollup", "gate_sliding",
                  "automatic_sliding", "elevator")
HORIZONTAL_FAMILIES = ("hatch_floor", "hatch_ceiling")
# families built without a wall: nothing to read a leaf's standoff against in a near-edge-on view
OUTDOOR_FAMILIES = ("gate_swing", "gate_sliding", "baby_gate")


def sheet_view_titles(family: str, horizontal: bool) -> Dict[str, str]:
    t = dict(VIEW_TITLE)
    if horizontal:
        t["front"] = "above (robot side)"
        t["back"] = "below / far side"
        t["edge"] = "hinge-side elevation"
    elif family in SLIDE_FAMILIES:
        t["edge"] = "track-side (leading edge)"
    return t


# ---------------------------------------------------------------------------------------------
# caption: what the spec SAYS should be there
# ---------------------------------------------------------------------------------------------
def _fmt(x, unit: str = "", nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}{unit}".rstrip("0").rstrip(".") + ("" if unit else "") if nd else f"{x}{unit}"
    return str(x)


def caption_lines(spec: dict, meta: dict) -> List[str]:
    """The spec, stated as prose, so a reviewer can judge whether the picture is complete.

    Deliberately says what SHOULD be there rather than describing the render: 'hinges: butt_3 x 3'
    is a claim the picture either supports or contradicts.
    """
    leaf, op, kin, hin = spec["leaf"], spec["operator"], spec["kinematics"], spec["hinge"]
    latch, lock, closer, opening = spec["latch"], spec["lock"], spec["closer"], spec["opening"]
    mass = spec.get("physics", {}).get("mass", {}).get("total_kg")

    travel = kin.get("travel_m")
    max_deg = kin.get("max_open_deg")
    if kin["type"].startswith("slide") or kin["type"] in ("roll", "lift"):
        motion = f"{kin['type']}, travel {travel:.3f} m" if travel else kin["type"]
    else:
        motion = f"{kin['type']}, max {max_deg:.0f} deg" if max_deg else kin["type"]
    if kin.get("opens_toward"):
        motion += f", opens toward {kin['opens_toward']}"

    n_leaf = leaf.get("count", 1)
    lines = [
        f"{spec['id']}   family={spec['family']}   context={spec.get('context', '-')}   "
        f"use={spec.get('use_case', '-')}   condition={spec.get('condition', '-')}   task={spec.get('task', '-')}",
        f"KINEMATICS  {motion}   |  swing={hin.get('swing', '-')}   hinge_side={hin.get('side', '-')}"
        + (f"   |  track={kin['track']}" if kin.get("track") else "")
        + (f"   roller={kin['roller']}" if kin.get("roller") else "")
        + (f"   stop={kin['stop']}" if kin.get("stop") else ""),
        f"HINGES      {hin.get('model', 'none')} x {hin.get('count', 0)}"
        + (f" (axis tilt {hin['axis_tilt_deg']:.1f} deg)" if hin.get("axis_tilt_deg") else "")
        + f"   |  OPERATOR {op.get('model', 'none')} @ h={op.get('height', 0):.2f} m, sides={op.get('sides', '-')}",
        f"LATCH       {latch.get('model', 'none')}   |  LOCK {lock.get('model', 'none')}"
        f" (engaged={bool(lock.get('engaged'))}, robot-side release={bool(lock.get('robot_side_release'))})"
        f"   |  CLOSER {closer.get('model', 'none')}"
        + (f" EN{closer['en_size']}" if closer.get("en_size") else ""),
        f"LEAF        {leaf['width']:.3f} x {leaf['height']:.3f} x {leaf['thickness']:.3f} m"
        + (f" x{n_leaf} leaves" if n_leaf and n_leaf != 1 else "")
        + f", {leaf.get('slab', '-')} / {leaf.get('panel_style', '-')}"
        + (f", {mass:.1f} kg" if mass else "")
        + f"   |  OPENING {opening['width']:.3f} x {opening['height']:.3f} m, wall {opening['wall_thickness']:.3f} m,"
        f" frame {opening.get('frame', {}).get('kind', '-')}",
        f"SEAL        {spec.get('seal', 'none')}   |  THRESHOLD {opening.get('threshold', 'none')}"
        f"   |  EXTRAS {', '.join(spec.get('extras') or []) or 'none'}"
        + (f"   |  n_latches={len(meta.get('operator_joints') or [])} ({meta.get('operator_coupling', '-')})"
           if len(meta.get("operator_joints") or []) > 1 else ""),
    ]
    return lines


def spec_facts(spec: dict, meta: dict) -> dict:
    """The same statement of intent as machine-readable fields (goes into the prompt as JSON)."""
    leaf, kin, hin = spec["leaf"], spec["kinematics"], spec["hinge"]
    return {
        "door_id": spec["id"],
        "family": spec["family"],
        "context": spec.get("context"),
        "condition": spec.get("condition"),
        "kinematics": kin.get("type"),
        "travel_m": kin.get("travel_m"),
        "max_open_deg": kin.get("max_open_deg"),
        "opens_toward": kin.get("opens_toward"),
        "track": kin.get("track"),
        "roller": kin.get("roller"),
        "stop": kin.get("stop"),
        "hinge_model": hin.get("model"),
        "hinge_count": hin.get("count"),
        "hinge_side": hin.get("side"),
        "operator": spec["operator"].get("model"),
        "operator_height_m": spec["operator"].get("height"),
        "operator_sides": spec["operator"].get("sides"),
        "latch": spec["latch"].get("model"),
        "lock": spec["lock"].get("model"),
        "lock_engaged": bool(spec["lock"].get("engaged")),
        "closer": spec["closer"].get("model"),
        "leaf_w_h_t_m": [leaf["width"], leaf["height"], leaf["thickness"]],
        "leaf_count": leaf.get("count", 1),
        "slab": leaf.get("slab"),
        "panel_style": leaf.get("panel_style"),
        "mass_kg": spec.get("physics", {}).get("mass", {}).get("total_kg"),
        "opening_w_h_m": [spec["opening"]["width"], spec["opening"]["height"]],
        "seal": spec.get("seal"),
        "threshold": spec["opening"].get("threshold"),
        "extras": spec.get("extras") or [],
        "n_operator_joints": len(meta.get("operator_joints") or []),
        "operator_coupling": meta.get("operator_coupling"),
    }


# ---------------------------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------------------------
class SheetRenderer:
    """One door, loaded once, rendered into the twelve panels of a review sheet."""

    def __init__(self, door_dir: str, panel: Tuple[int, int] = (400, 300), supersample: float = 1.2):
        import mujoco
        from doorbench.clearance import Clearance

        self.mujoco = mujoco
        self.dir = door_dir
        self.panel = panel
        self.rw = int(round(panel[0] * supersample))
        self.rh = int(round(panel[1] * supersample))

        self.spec = json.load(open(os.path.join(door_dir, "spec.json")))
        self.model_json = json.load(open(os.path.join(door_dir, "model.json")))
        self.meta = self.model_json["meta"]

        ms = mujoco.MjSpec.from_file(os.path.join(door_dir, "door.xml"))
        ms.visual.global_.offwidth = max(self.rw, 640)
        ms.visual.global_.offheight = max(self.rh, 480)
        self.m = ms.compile()
        self.d = mujoco.MjData(self.m)
        self.gate = Clearance(door_dir, "full")

        self.roles = {b["joint"]["name"]: b["joint"].get("role")
                      for b in self.model_json["bodies"] if b.get("joint")}
        self.sem = {g["name"]: g.get("semantic", "")
                    for b in self.model_json["bodies"] for g in b["geoms"]}
        self.gname = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(self.m.ngeom)]

        self.u = float(self.meta.get("u") or 1.0)
        self.horizontal = bool(self.meta.get("horizontal")) or self.spec["family"] in HORIZONTAL_FAMILIES
        self.family = self.spec["family"]

        # visible geoms, minus the ground plane and the wall (they dominate the bbox and hide the door)
        self.visible = [g for g in range(self.m.ngeom)
                        if self.m.geom_group[g] != 3 and self.sem.get(self.gname[g]) not in ("floor", "wall")]
        self.hw_ids = [g for g in self.visible if self.sem.get(self.gname[g]) in HW_SEM]
        self.mech_ids = [g for g in self.visible
                         if self.sem.get(self.gname[g]) in MECH_SEM
                         or any(h in (self.gname[g] or "") for h in MECH_NAME_HINTS)]

        # closed loops (closer arms, boltwork): solved over their own mechanism joints
        self.connects = [i for i in range(self.m.neq)
                         if self.m.eq_active0[i] and int(self.m.eq_type[i]) == int(mujoco.mjtEq.mjEQ_CONNECT)]
        eqs = [i for i in range(self.m.neq)
               if self.m.eq_active0[i] and int(self.m.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT)]
        self.controlled = {self.m.joint(int(self.m.eq_obj1id[e])).name for e in eqs}
        self.loop_joints = self._loop_joints()

        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[:3] = 1
        self.renderer = mujoco.Renderer(self.m, height=self.rh, width=self.rw)
        # Inspection lighting, not presentation lighting.  Three changes, each of which fixed a case
        # where the render made a correct door look wrong (or a wrong one look fine):
        #  * reflectance off - a reflective steel garage section mirrors the skybox, and five opaque
        #    sections then read as an empty opening above the bottom panel;
        #  * ambient up - 28 doors are painted black at 4 % reflectance and render as featureless
        #    silhouettes with no visible split line, panel detail or hardware;
        #  * diffuse down - MuJoCo's default headlight blows out white hardware and hides its seams.
        self.m.mat_reflectance[:] = 0.0
        self.m.light_diffuse[:] = np.minimum(self.m.light_diffuse, 0.55)
        self.m.vis.headlight.ambient[:] = 0.40
        self.m.vis.headlight.diffuse[:] = 0.42
        # Clear glazing at its shipped alpha is invisible: a patio slider open by 0.84 m looks exactly
        # like a shut one, and a reviewer cannot tell whether the doorway is glazed or empty.  Tint any
        # very transparent material up to GLASS_ALPHA - enough to read as "there is glass here", still
        # transparent enough to see the hardware and the frame behind it.
        a = self.m.mat_rgba[:, 3]
        a[(a > 0.02) & (a < GLASS_ALPHA)] = GLASS_ALPHA

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass

    # -- poses ---------------------------------------------------------------------------------
    def _loop_joints(self) -> List[int]:
        m = self.m
        chains = set()
        for eq in self.connects:
            for body in (int(m.eq_obj1id[eq]), int(m.eq_obj2id[eq])):
                while body > 0:
                    for j in range(int(m.body_jntadr[body]), int(m.body_jntadr[body]) + int(m.body_jntnum[body])):
                        name = self.mujoco.mj_id2name(m, self.mujoco.mjtObj.mjOBJ_JOINT, j)
                        if self.roles.get(name) not in ("primary", "secondary") and name not in self.controlled:
                            chains.add(j)
                    body = int(m.body_parentid[body])
        return sorted(chains)

    def _connect_residual(self) -> np.ndarray:
        m, d = self.m, self.d
        rs = []
        for e in self.connects:
            a, b = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            pa = d.xpos[a] + d.xmat[a].reshape(3, 3) @ m.eq_data[e, :3]
            pb = d.xpos[b] + d.xmat[b].reshape(3, 3) @ m.eq_data[e, 3:6]
            rs.extend(pa - pb)
        return np.array(rs)

    def _leaf_joints(self) -> List[int]:
        m, mujoco = self.m, self.mujoco
        out = []
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if self.roles.get(name) in ("primary", "secondary") and name not in self.controlled:
                out.append(j)
        return out

    def _leaf_x_width(self, q: np.ndarray) -> float:
        """Total width of x the leaves occupy - how much of the doorway they still block."""
        self.d.qpos[:] = q
        self.mujoco.mj_forward(self.m, self.d)
        iv = []
        for g in self.visible:
            if self.sem.get(self.gname[g]) != "leaf":
                continue
            R = self.d.geom_xmat[g].reshape(3, 3)
            c = self.d.geom_xpos[g] + R @ self.m.geom_aabb[g, :3]
            h = np.abs(R) @ self.m.geom_aabb[g, 3:]
            iv.append((float(c[0] - h[0]), float(c[0] + h[0])))
        iv.sort()
        tot, cur = 0.0, None
        for a, b in iv:
            if cur is None or a > cur[1]:
                if cur:
                    tot += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        return tot + (cur[1] - cur[0] if cur else 0.0)

    def open_drive(self) -> List[int]:
        """Which leaf joints the 'fully open' pose should drive.

        Driving every leaf to its limit is right for a swing pair, a bifold or a bi-parting slider.
        It is WRONG for a bypass closet: its two leaves run on opposite tracks, so driving both just
        swaps them and the doorway is as blocked at 'fully open' as it was shut - a sheet that showed
        that would be a picture of a door that never opens.  Decide it by measurement rather than by
        family: take whichever of {all leaves, primary only} leaves less of the doorway covered.
        """
        leaves = self._leaf_joints()
        if len(leaves) < 2:
            return leaves
        pj = self.meta.get("primary_joint")
        prim = [j for j in leaves
                if self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_JOINT, j) == pj]
        if not prim:
            return leaves
        w_all = self._leaf_x_width(self._raw_pose(1.0, leaves)[0])
        w_one = self._leaf_x_width(self._raw_pose(1.0, prim)[0])
        return prim if w_one < w_all - 0.02 else leaves

    def pose(self, frac: float, drive: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, float, bool]:
        """qpos at ``frac`` of the leaf travel, loops solved.  Returns (q, loop residual m, forced)."""
        q, forced = self._raw_pose(frac, drive if drive is not None else self._leaf_joints())
        return self._solve_loops(q) + (forced,)

    def _raw_pose(self, frac: float, drive: Sequence[int]) -> Tuple[np.ndarray, bool]:
        m, d, mujoco = self.m, self.d, self.mujoco
        q = self.gate.released_qpos() if frac else self.gate.resolve(m.qpos0.copy())
        forced = False
        for j in drive:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            adr = int(m.jnt_qposadr[j])
            q0 = m.qpos0[adr]
            if m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]
                # A joint whose whole range is under 6 deg / 50 mm cannot open the door at all - and a
                # MuJoCo range is static, so no amount of unlocking widens it.  The old test was a flat
                # 6 mm, which passed a mag-locked turnstile rotor limited to +-2.9 deg and drew it as
                # "fully open".  Show the inspection pose, and say on the sheet that it is one.
                pinned = (hi - lo) < (PINNED_HINGE_RAD if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                                      else PINNED_SLIDE_M)
                if pinned and frac:
                    # a joint pinned shut by an engaged lock: show the inspection pose anyway, labelled
                    kin = self.spec["kinematics"]
                    target = (math.radians(kin.get("max_open_deg") or 90)
                              if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                              else float(kin.get("travel_m") or 0.9))
                    forced = True
                else:
                    target = hi if abs(hi - q0) >= abs(lo - q0) else lo
            else:
                target = q0 + 1.2
            q[adr] = q0 + frac * (target - q0)
        return self.gate.resolve(q), forced

    def _solve_loops(self, q: np.ndarray) -> Tuple[np.ndarray, float]:
        m, d, mujoco = self.m, self.d, self.mujoco
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        if self.loop_joints and self.connects:
            from scipy.optimize import least_squares
            adrs = [int(m.jnt_qposadr[j]) for j in self.loop_joints]
            base = q.copy()

            def fun(values):
                qq = base.copy()
                qq[adrs] = values
                qq = self.gate.resolve(qq)
                d.qpos[:] = qq
                mujoco.mj_forward(m, d)
                return self._connect_residual()

            lo = [float(m.jnt_range[j, 0]) if m.jnt_limited[j] else -np.inf for j in self.loop_joints]
            hi = [float(m.jnt_range[j, 1]) if m.jnt_limited[j] else np.inf for j in self.loop_joints]
            x0 = np.clip(base[adrs], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            sol = least_squares(fun, x0, bounds=(lo, hi), ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=120)
            q = base.copy()
            q[adrs] = sol.x
            q = self.gate.resolve(q)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r = self._connect_residual()
        res = float(np.max(np.linalg.norm(r.reshape(-1, 3), axis=1))) if len(r) else 0.0
        return q, res

    # -- framing -------------------------------------------------------------------------------
    def _bbox(self, ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        m, d = self.m, self.d
        pts = []
        for g in ids:
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
            h = np.abs(R) @ m.geom_aabb[g, 3:]
            pts.extend([c - h, c + h])
        if not pts:
            return np.array([0.0, 0.0, 1.0]), np.array([0.6, 0.6, 1.0])
        return np.min(pts, axis=0), np.max(pts, axis=0)

    def _fit_distance(self, lo: np.ndarray, hi: np.ndarray, az: float, el: float, pad: float) -> float:
        """Distance that just fits the box in the panel, fitted in BOTH axes.

        A bounding-sphere fit (``|hi-lo|/2 / sin(fovy/2)``) throws away most of the frame on the long
        thin boxes this sheet cares about - a hinge stile is 0.1 x 0.1 x 1.9 m, and a sphere fit
        renders it as a whole-door shot.  Projecting the eight corners onto the camera's own right/up
        axes and fitting each independently is what makes a close-up close.
        """
        a, e = math.radians(az), math.radians(el)
        fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(right)
        right = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else right / n
        up = np.cross(right, fwd)
        c = (lo + hi) / 2
        h = (hi - lo) / 2
        # half-extent of an axis-aligned box along an arbitrary direction is |h . |axis||
        hw = float(np.dot(np.abs(right), h))
        hh = float(np.dot(np.abs(up), h))
        hd = float(np.dot(np.abs(fwd), h))
        tan_v = math.tan(math.radians(float(self.m.vis.global_.fovy)) / 2)
        aspect = self.rw / self.rh
        d = max(hh / max(tan_v, 1e-6), hw / max(tan_v * aspect, 1e-6))
        return max(0.28, pad * d + 0.35 * hd)   # a little of the box depth so the near face is not clipped

    def _cam(self, view: str, lo: np.ndarray, hi: np.ndarray, pad: float = 1.16):
        mujoco = self.mujoco
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (lo + hi) / 2
        u = self.u
        if self.horizontal:
            az_el = {"front": (90.0, -52.0), "back": (90.0, 38.0), "edge": (0.0 if u > 0 else 180.0, -18.0)}
        else:
            # near-elevation from the hinge (or, for sliders, the leading-edge) side: the leaf is close
            # to edge-on, so its standoff from the jamb, the floor guides, the closer arm and anything
            # hanging in the reveal are all in profile.  Outdoor families have no wall to read the
            # standoff against, and a fully edge-on gate collapses into an unreadable sliver, so they
            # get a wider angle.
            skew = 35.0 if self.family in OUTDOOR_FAMILIES else 20.0
            az_el = {"front": (90.0 - u * 30.0, -16.0),
                     "back": (-90.0 + u * 30.0, -16.0),
                     "edge": (skew if u > 0 else 180.0 - skew, -10.0)}
        cam.azimuth, cam.elevation = az_el[view]
        cam.distance = self._fit_distance(lo, hi, cam.azimuth, cam.elevation, pad)
        return cam

    def _render(self, cam) -> "np.ndarray":
        self.renderer.update_scene(self.d, camera=cam, scene_option=self.opt)
        return self.renderer.render()

    # -- the sheet -----------------------------------------------------------------------------
    def panels(self) -> Tuple[List[dict], dict]:
        """Render all twelve panels.  Returns (panel dicts with an 'image' array, provenance/pose info)."""
        from PIL import Image

        poses: Dict[str, Tuple[np.ndarray, float, bool]] = {}
        boxes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        drive = self.open_drive()
        for name, frac in zip(POSES, (0.0, 0.5, 1.0)):
            poses[name] = self.pose(frac, drive)
            self.d.qpos[:] = poses[name][0]
            self.mujoco.mj_forward(self.m, self.d)
            boxes[name] = self._bbox(self.visible)
        lo = np.min([b[0] for b in boxes.values()], axis=0)
        hi = np.max([b[1] for b in boxes.values()], axis=0)
        cams = {v: self._cam(v, lo, hi) for v in VIEWS}

        titles = sheet_view_titles(self.family, self.horizontal)
        out: List[dict] = []
        for pose_name in POSES:
            q, res, forced = poses[pose_name]
            self.d.qpos[:] = q
            self.mujoco.mj_forward(self.m, self.d)
            for view in VIEWS:
                label = f"{POSE_TITLE[pose_name]} - {titles[view]}"
                if forced:
                    label += "  [INSPECTION POSE: the leaf joint is pinned and cannot open]"
                out.append({"key": f"{pose_name}_{view}", "pose": pose_name, "view": view,
                            "label": label, "image": self._render(cams[view]),
                            "loop_residual_m": res, "forced_pose": forced})

        # close-ups ------------------------------------------------------------------------------
        q, res, _ = poses["closed"]
        self.d.qpos[:] = q
        self.mujoco.mj_forward(self.m, self.d)
        ids = self.hw_ids or self.visible
        hlo, hhi = self._bbox(ids)
        if not self.hw_ids and self.meta.get("handle_cam_target"):
            c = np.array(self.meta["handle_cam_target"], float)
            hlo, hhi = c - 0.18, c + 0.18
        span = float(np.linalg.norm(hhi - hlo))
        if span < 0.16:                                    # a lone knob: give it some context
            c = (hlo + hhi) / 2
            hlo, hhi = c - 0.11, c + 0.11
        hw_names = sorted({self.gname[g] for g in self.hw_ids})
        for view, tag in (("front", "near face"), ("back", "far face")):
            cam = self._cam(view, hlo, hhi, pad=1.30)
            out.append({"key": f"hardware_{view}", "pose": "closed", "view": view,
                        "label": f"HARDWARE CLOSE-UP - {tag} (door closed, mechanism at rest)",
                        "image": self._render(cam), "loop_residual_m": res, "forced_pose": False})

        q, res, forced = poses["open"]
        self.d.qpos[:] = q
        self.mujoco.mj_forward(self.m, self.d)
        # frame the MOVING mechanism parts (rollers, hangers, hinge leaves, closer arm) plus a fixed
        # margin of static surroundings: that margin is what turns "a wheel" into "a wheel and the end
        # of the rail it is supposed to still be on".
        moving = [g for g in self.mech_ids if int(self.m.body_weldid[self.m.geom_bodyid[g]]) != 0]
        focus, focus_kind = moving, "moving mechanism"
        for sem in MECH_PRIORITY:                       # closer arm first: it is the loop that historically floated
            sel = [g for g in moving if self.sem.get(self.gname[g]) == sem]
            if sel:
                focus, focus_kind = sel, sem
                break
        mlo, mhi = self._bbox(focus or self.mech_ids or self.visible)
        mlo, mhi = mlo - MECH_MARGIN_M, mhi + MECH_MARGIN_M
        cam = self._cam("front" if self.family in SLIDE_FAMILIES or self.horizontal else "edge",
                        mlo, mhi, pad=1.06)
        out.append({"key": "mechanism_open", "pose": "open", "view": "mechanism",
                    "label": f"MECHANISM CLOSE-UP ({focus_kind}) - door FULLY OPEN",
                    "image": self._render(cam), "loop_residual_m": res, "forced_pose": forced})

        info = {
            "loop_residual_m": max(p[1] for p in poses.values()),
            "forced_pose": any(p[2] for p in poses.values()),
            "n_connect_equalities": len(self.connects),
            "hardware_geoms": hw_names,
            "mechanism_geoms": sorted({self.gname[g] for g in self.mech_ids}),
            "mechanism_focus": focus_kind,
            "scene_bbox_m": [list(map(float, lo)), list(map(float, hi))],
        }
        return out, info


# ---------------------------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------------------------
def _label_bar(draw, x: int, y: int, w: int, text: str, font, h: int = 15,
               bg=(18, 20, 26), fg=(235, 235, 235)):
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=bg)
    draw.text((x + 4, y + 2), text, fill=fg, font=font)


def compose(panels: List[dict], caption: List[str], width: int = 1200, cols: int = 3,
            footer: str = "") -> "object":
    """Tile the panels under the caption band.  Returns a PIL image ~``width`` px across."""
    from PIL import Image, ImageDraw, ImageFont

    cw = width // cols
    ch = int(round(cw * panels[0]["image"].shape[0] / panels[0]["image"].shape[1]))
    rows = (len(panels) + cols - 1) // cols
    font = ImageFont.load_default(size=12)
    small = ImageFont.load_default(size=11)
    head_h = 16 * len(caption) + 10
    foot_h = 18 if footer else 0
    sheet = Image.new("RGB", (cw * cols, head_h + rows * ch + foot_h), (12, 14, 18))
    draw = ImageDraw.Draw(sheet)
    for i, line in enumerate(caption):
        draw.text((6, 5 + 16 * i), line, fill=(255, 214, 138) if i == 0 else (198, 209, 224), font=font)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        x, y = c * cw, head_h + r * ch
        im = Image.fromarray(p["image"]).resize((cw, ch), Image.Resampling.LANCZOS)
        sheet.paste(im, (x, y))
        _label_bar(draw, x, y, cw, f"[{i + 1}] {p['label']}", small)
        draw.rectangle([x, y, x + cw - 1, y + ch - 1], outline=(46, 52, 62))
    if footer:
        draw.text((6, head_h + rows * ch + 3), footer, fill=(150, 160, 176), font=small)
    return sheet


def source_hashes(door_dir: str) -> dict:
    out = {}
    for kind, fn in (("model", "model.json"), ("spec", "spec.json"), ("xml", "door.xml")):
        with open(os.path.join(door_dir, fn), "rb") as f:
            out[f"source_{kind}_sha256"] = hashlib.sha256(f.read()).hexdigest()
    return out


def render_sheet(door_dir: str, out_path: str, width: int = 1200, quality: int = 78,
                 panel: Tuple[int, int] = (400, 300)) -> dict:
    """Render one door's review sheet to ``out_path``.  Returns the sheet's metadata record."""
    r = SheetRenderer(door_dir, panel=panel)
    try:
        panels, info = r.panels()
        cap = caption_lines(r.spec, r.meta)
        foot = (f"panels 1-9: three poses x three viewpoints, one camera per column (identical framing "
                f"across the three rows).  loop residual {info['loop_residual_m'] * 1000:.2f} mm"
                + ("   |  INSPECTION POSE: this door's leaf joint is pinned (range under 6 deg / 50 mm) and cannot open;"
                   " the open panels are prescribed, not achievable" if info["forced_pose"] else ""))
        img = compose(panels, cap, width=width, footer=foot)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        img.save(out_path, quality=quality, optimize=True)
        rec = {
            "door_id": r.spec["id"],
            "family": r.spec["family"],
            "sheet": os.path.basename(out_path),
            "sheet_size_px": list(img.size),
            "sheet_bytes": os.path.getsize(out_path),
            "panels": [{"index": i + 1, "key": p["key"], "label": p["label"]} for i, p in enumerate(panels)],
            "caption": cap,
            "spec_facts": spec_facts(r.spec, r.meta),
            **{k: v for k, v in info.items() if k != "scene_bbox_m"},
            **source_hashes(door_dir),
        }
        return rec
    finally:
        r.close()

#!/usr/bin/env python
"""Close-up renders of every distinct operator / latch / lock model for a realism review (task G3).

For every distinct hardware model in the manifest one representative door is picked (swing_single preferred; for
locks an *engaged* instance is preferred so the lock geometry is present) and rendered with the MuJoCo offscreen
renderer from a free camera aimed at the hardware:

  <kind>_<model>_a.jpg   closed door, robot (-y) face, camera ~0.5 m away, oblique from the hinge side
  <kind>_<model>_b.jpg   closed door, far (+y) face
  <kind>_<model>_c.jpg   mechanism ACTUATED: operator at full travel, bolts retracted / hooks lifted (couplings resolved
                         exactly like the clearance gate does), door still closed
  <kind>_<model>_d.jpg   latches & locks only: door opened ~40 % with the mechanism at rest, looking at the latch edge
                         (bolt extended, strike / keeper visible)

plus one contact sheet per kind (sheet_<kind>.jpg).  Group 3 (collision-only proxies) is hidden, everything the
viewer shows is rendered.

Usage:
  python scripts/hardware_review.py [--assets assets] [--out docs/review] [--kinds operators,latches,locks]
                                    [--models m1,m2] [--per-family] [--size 800x600] [--pages-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

KINDS = {"operators": "operator", "latches": "latch", "locks": "lock"}
SKIP = {"operators": {"none", "elevator_none"}, "latches": {"none"}, "locks": {"none"}}
TARGET_SEMANTIC = {"operators": ("operator",), "latches": ("latch",), "locks": ("lock",)}


def pick_doors(man: dict, kind: str, per_family: bool) -> dict:
    """model -> [door rows].  One per model (or one per (model, family) with --per-family)."""
    key = KINDS[kind]
    by = {}
    for d in man["doors"]:
        if d.get("error"):
            continue
        by.setdefault(d[key], []).append(d)
    out = {}
    for model, rows in by.items():
        if model in SKIP[kind]:
            continue

        def score(r):
            s = 0
            if r["family"] == "swing_single":
                s += 4
            if kind == "locks" and r.get("lock_engaged"):
                s += 8
            if kind == "locks" and r.get("robot_side_release"):
                s += 1
            if r.get("signed_off"):
                s += 2
            return -s, r["index"]

        if per_family:
            fams = {}
            for r in sorted(rows, key=score):
                fams.setdefault(r["family"], r)
            out[model] = list(fams.values())
        else:
            out[model] = [sorted(rows, key=score)[0]]
    return out


class DoorRenderer:
    def __init__(self, door_dir: str, size=(800, 600)):
        import mujoco
        from doorbench.clearance import Clearance
        self.mujoco = mujoco
        self.dir = door_dir
        spec = mujoco.MjSpec.from_file(os.path.join(door_dir, "door.xml"))
        spec.visual.global_.offwidth = max(size[0], 640)
        spec.visual.global_.offheight = max(size[1], 480)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.gate = Clearance(door_dir, "full")      # same qpos layout; gives resolve() / released_qpos()
        with open(os.path.join(door_dir, "model.json")) as f:
            mj = json.load(f)
        self.meta = mj["meta"]
        self.sem = {g["name"]: g.get("semantic", "") for b in mj["bodies"] for g in b["geoms"]}
        self.sites = {s["name"]: s for b in mj["bodies"] for s in b["sites"]}
        self.r = mujoco.Renderer(self.m, height=size[1], width=size[0])
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        for g in (0, 1, 2):
            self.opt.geomgroup[g] = 1
        self.u = float(self.meta.get("u", 1.0) or 1.0)
        self.horizontal = bool(self.meta.get("horizontal"))

    def close(self):
        self.r.close()

    # ---- configurations --------------------------------------------------------------------------------
    def q_rest(self):
        return self.gate.resolve(self.m.qpos0.copy())

    def q_actuated(self):
        return self.gate.released_qpos()

    def q_open(self, frac: float, actuated: bool):
        q = self.q_actuated() if actuated else self.m.qpos0.copy()
        pj = self.meta.get("primary_joint")
        if pj and pj in self.gate.jid:
            j = self.gate.jid[pj]
            if self.m.jnt_limited[j]:
                lo, hi = self.m.jnt_range[j]
                q[self.m.jnt_qposadr[j]] = lo + frac * (hi - lo)
            else:
                q[self.m.jnt_qposadr[j]] = frac * 1.2
        return self.gate.resolve(q)

    # ---- target -------------------------------------------------------------------------------------------
    def target(self, kind: str):
        """(centre, radius) in world coordinates of the hardware to inspect for the current self.d state."""
        m, d, mujoco = self.m, self.d, self.mujoco
        sems = TARGET_SEMANTIC[kind]
        ids = [g for g in range(m.ngeom) if self.sem.get(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), "") in sems]
        # operators: prefer the grip / push sites (the thing the robot touches), fall back to operator geoms
        if kind == "operators":
            pts = []
            for name, s in self.sites.items():
                if s.get("role") in ("grip", "push"):
                    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
                    if sid >= 0:
                        pts.append(d.site_xpos[sid].copy())
            if pts and ids:
                c = np.mean(pts, axis=0)
                r = max(0.08, max(np.linalg.norm(d.geom_xpos[g] - c) + m.geom_rbound[g] for g in ids))
                # very long hardware (ladder pulls, touch bars): frame the whole thing but cap the distance
                return c, min(r, 0.45)
        if not ids:
            t = self.meta.get("handle_cam_target")
            if t:
                return np.array(t, float), 0.12
            return np.array([0.0, 0.0, 1.0]), 0.3
        lo = np.min([d.geom_xpos[g] - m.geom_rbound[g] for g in ids], axis=0)
        hi = np.max([d.geom_xpos[g] + m.geom_rbound[g] for g in ids], axis=0)
        c = (lo + hi) / 2
        r = float(np.linalg.norm(hi - lo) / 2)
        return c, min(max(r, 0.08), 0.45)

    # ---- render -------------------------------------------------------------------------------------------
    def _leaf_axes(self):
        """World direction of the primary leaf's local +x (hinge -> latch edge) and the swing sign v."""
        m, d, mujoco = self.m, self.d, self.mujoco
        pj = self.meta.get("primary_joint")
        u, v = self.u, float(self.meta.get("v", 1.0) or 1.0)
        ex = np.array([u, 0.0, 0.0])
        if pj and pj in self.gate.jid:
            b = int(m.jnt_bodyid[self.gate.jid[pj]])
            R = d.xmat[b].reshape(3, 3)
            ex = R @ np.array([u, 0.0, 0.0])
        return ex, v

    def render(self, q, kind: str, view: str, path: str):
        m, d, mujoco = self.m, self.d, self.mujoco
        from PIL import Image
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        c, r = self.target(kind)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = c
        cam.distance = min(1.3, max(0.45, 2.4 * r))
        u = self.u
        if self.horizontal:
            # hatches: faces are up / down
            cam.azimuth, cam.elevation = {"a": (90.0, -55.0), "b": (90.0, 40.0), "c": (90.0, -55.0), "d": (60.0, -35.0)}[view]
        elif view == "d":
            # open door: look at the latch edge along the (rotated) leaf's edge normal, from the side the leaf swung
            # away from, so both the bolt on the leaf edge and the strike in the frame are in view
            ex, v = self._leaf_axes()
            dir_ = 0.75 * ex + np.array([0.0, -v * 0.8, 0.0])
            dir_ = dir_ / max(np.linalg.norm(dir_), 1e-9)
            cam.azimuth = math.degrees(math.atan2(-dir_[1], -dir_[0]))
            cam.elevation = -14.0
            cam.distance = min(1.0, max(0.5, 2.4 * r))
        else:
            cam.azimuth, cam.elevation = {"a": (90.0 - u * 32.0, -16.0), "b": (-90.0 + u * 32.0, -16.0), "c": (90.0 - u * 32.0, -16.0)}[view]
        self.r.update_scene(d, camera=cam, scene_option=self.opt)
        Image.fromarray(self.r.render()).save(path, quality=80)


def label(img, text):
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    w = img.size[0]
    dr.rectangle([0, 0, w, 16], fill=(0, 0, 0))
    dr.text((4, 2), text, fill=(255, 255, 255))
    return img


def contact_sheet(rows, path, cell=(320, 240), per_row=2):
    """rows: list of (title, [image paths]).  Tiles per_row models side by side."""
    from PIL import Image
    if not rows:
        return
    n_views = max(len(r[1]) for r in rows)
    cols = per_row * n_views
    n_rows = math.ceil(len(rows) / per_row)
    sheet = Image.new("RGB", (cols * cell[0], n_rows * cell[1]), (30, 30, 30))
    for i, (title, paths) in enumerate(rows):
        rr, cc = divmod(i, per_row)
        for k, p in enumerate(paths):
            try:
                im = Image.open(p).convert("RGB").resize(cell)
            except Exception:
                continue
            label(im, f"{title} [{'abcd'[k]}]")
            sheet.paste(im, (cc * n_views * cell[0] + k * cell[0], rr * cell[1]))
    sheet.save(path, quality=75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--out", default="docs/review")
    ap.add_argument("--kinds", default="operators,latches,locks")
    ap.add_argument("--models", default="")
    ap.add_argument("--per-family", action="store_true", help="one representative per (model, family) (inspection mode; files get a _<family> suffix)")
    ap.add_argument("--size", default="800x600")
    ap.add_argument("--open-frac", type=float, default=0.4)
    ap.add_argument("--pages-dir", default="", help="also write paged contact sheets (3 models per page, larger cells) here")
    a = ap.parse_args()
    size = tuple(int(x) for x in a.size.split("x"))
    os.makedirs(a.out, exist_ok=True)
    if a.pages_dir:
        os.makedirs(a.pages_dir, exist_ok=True)
    man = json.load(open(os.path.join(a.assets, "manifest.json")))
    want = set(a.models.split(",")) if a.models else None
    index = {}
    for kind in a.kinds.split(","):
        rows = []
        for model, doors in sorted(pick_doors(man, kind, a.per_family).items()):
            if want and model not in want:
                continue
            for row in doors:
                door_dir = os.path.join(a.assets, "doors", row["id"])
                stem = f"{KINDS[kind]}_{model}" + (f"_{row['family']}" if a.per_family else "")
                try:
                    dr = DoorRenderer(door_dir, size)
                except Exception as e:
                    print(f"  {stem}: cannot load {row['id']}: {e}", flush=True)
                    continue
                views = [("a", dr.q_rest()), ("b", dr.q_rest()), ("c", dr.q_actuated())]
                if kind in ("latches", "locks"):
                    views.append(("d", dr.q_open(a.open_frac, actuated=False)))
                paths = []
                for v, q in views:
                    p = os.path.join(a.out, f"{stem}_{v}.jpg")
                    try:
                        dr.render(q, kind, v, p)
                        paths.append(p)
                    except Exception as e:
                        print(f"  {stem}_{v}: render error {e}", flush=True)
                dr.close()
                title = f"{model} ({row['id']})"
                rows.append((title, paths))
                index[f"{kind}:{model}" + (f":{row['family']}" if a.per_family else "")] = {"door": row["id"], "family": row["family"], "images": [os.path.basename(p) for p in paths],
                                                                                              "operator": row["operator"], "latch": row["latch"], "lock": row["lock"], "lock_engaged": row.get("lock_engaged"), "robot_side_release": row.get("robot_side_release")}
                print(f"  {stem}: {row['id']} ({row['family']}) -> {len(paths)} views", flush=True)
        contact_sheet(rows, os.path.join(a.out, f"sheet_{kind}.jpg"))
        if a.pages_dir:
            for p in range(0, len(rows), 3):
                contact_sheet(rows[p:p + 3], os.path.join(a.pages_dir, f"{kind}_p{p // 3:02d}.jpg"), cell=(533, 400), per_row=1)
        print(f"{kind}: {len(rows)} renders", flush=True)
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()

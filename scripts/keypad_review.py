#!/usr/bin/env python
"""Render a code being entered on a keypad door: a frame sequence, right code vs wrong code.

Everything in the picture is the simulation: each button is a body on a slide joint with a return spring, a
fingertip force presses it, ``doorbench.keypad`` reads the travel (debounced) and releases the lock only when the
code is right - the outside lever's clutch (Schlage FE595 / Kaba Simplex) or the deadbolt motor (Schlage BE365).
The bottom row repeats the same drive with one digit changed: nothing is released and the door stays shut.

Usage:
  python scripts/keypad_review.py [--assets assets] [--out docs/media]
                                  [--doors db0526_swing_single,db0166_swing_single] [--size 380x300]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _q(m, d, j):
    return float(d.qpos[m.jnt_qposadr[j]]) if j >= 0 else 0.0


class Shot:
    """One rendered frame plus its caption."""

    def __init__(self, img, caption):
        self.img, self.caption = img, caption


class KeypadFilm:
    def __init__(self, door_dir: str, size=(380, 300)):
        import mujoco

        from doorbench.keypad import keypad_for

        self.mj = mujoco
        self.dir = door_dir
        spec = mujoco.MjSpec.from_file(os.path.join(door_dir, "door.xml"))
        spec.visual.global_.offwidth = max(size[0], 640)
        spec.visual.global_.offheight = max(size[1], 480)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        with open(os.path.join(door_dir, "model.json")) as f:
            mj = json.load(f)
        with open(os.path.join(door_dir, "spec.json")) as f:
            self.spec = json.load(f)
        self.meta = mj["meta"]
        self.kp = keypad_for(mujoco, self.m, self.meta, self.spec)
        if self.kp is None:
            raise SystemExit(f"{door_dir}: no keypad")
        self.r = mujoco.Renderer(self.m, height=size[1], width=size[0])
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        for g in (0, 1, 2):
            self.opt.geomgroup[g] = 1
        self.u = float(self.meta.get("u", 1.0) or 1.0)
        self.face = float(self.kp.cfg["face"])
        self.pj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, self.meta["primary_joint"])
        self.oj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, self.meta["operator_joint"] or "")
        self.sites = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, b["site"]) for b in self.kp.cfg["buttons"]]
        for b, sid in zip(self.kp.cfg["buttons"], self.sites):
            self.kp.by_label[b["label"]]["site_id"] = int(sid)

    def close(self):
        self.r.close()

    # -- rendering ---------------------------------------------------------
    def keypad_center(self):
        pts = np.array([self.d.site_xpos[s] for s in self.sites if s >= 0])
        return pts.mean(axis=0)

    def shoot(self, caption: str, view="key", at=None, oblique=25.0):
        """view: `key` (a button, 12 cm away, oblique so the stroke shows), `hardware` (keypad + lever), `door`."""
        from PIL import Image

        mujoco = self.mj
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if view == "key":
            cam.lookat[:] = at if at is not None else self.keypad_center()
            cam.distance, cam.elevation = 0.22, -14.0
            off = oblique
        elif view == "hardware":
            c = self.keypad_center()
            cam.lookat[:] = [c[0], c[1], c[2] - 0.10]
            cam.distance, cam.elevation = 0.55, -12.0
            off = 25.0
        else:
            cam.lookat[:] = [float(self.meta.get("cam_target_x", 0.0)), 0.0, float(self.meta.get("cam_target_z", 1.05))]
            cam.distance, cam.elevation = 3.6, -34.0     # from above and off to the side: the swing is unmistakable
            off = 55.0
        cam.azimuth = (90.0 - self.u * off) if self.face < 0 else (-90.0 + self.u * off)
        self.r.update_scene(self.d, camera=cam, scene_option=self.opt)
        return Shot(Image.fromarray(self.r.render()), caption)

    # -- driving -----------------------------------------------------------
    def restart(self):
        self.mj.mj_resetData(self.m, self.d)
        self.kp.reset(self.d)
        self.mj.mj_forward(self.m, self.d)

    def _step(self, extra=None):
        self.kp.apply(self.d)
        if extra:
            extra()
        self.mj.mj_step(self.m, self.d)
        self.d.qfrc_applied[:] = 0
        self.kp.step(self.d)

    def press(self, label: str, hold_s=0.09, gap_s=0.07):
        """Press one button; returns the frame at the bottom of the stroke."""
        dt = float(self.m.opt.timestep)
        shot = None
        n = max(1, int(hold_s / dt))
        for k in range(n):
            self.kp.hold(self.d, label)
            self._step()
            if k == n - 1:
                b = self.kp.by_label[label]
                depth = _q(self.m, self.d, b["jid"]) * 1000
                sid = self.kp.by_label[label].get("site_id", -1)
                at = self.d.site_xpos[sid] if sid >= 0 else None
                shot = self.shoot(f"press {label}  ({depth:.1f} mm of {self.kp.travel * 1000:.1f} travel)", view="key", at=at)
        for _ in range(max(1, int(gap_s / dt))):
            self._step()
        return shot

    def turn_and_push(self, seconds: float, lever=4.0, push=None):
        dt = float(self.m.opt.timestep)
        rel = self.kp.clutch if self.kp.release_mode == "clutch" else self.oj
        push = push if push is not None else 60.0

        def extra():
            if rel >= 0:
                self.d.qfrc_applied[self.m.jnt_dofadr[rel]] += lever
            if self.pj >= 0 and _q(self.m, self.d, self.pj) < math.radians(50):
                self.d.qfrc_applied[self.m.jnt_dofadr[self.pj]] += push

        for _ in range(int(seconds / dt)):
            self._step(extra)

    def angle_deg(self):
        return math.degrees(_q(self.m, self.d, self.pj))


def strip(rows, path, title: str, cell=(380, 300)):
    """Contact strip: one row per run, each frame captioned; `title` in a bar at the top."""
    from PIL import Image, ImageDraw

    pad, head, cap = 6, 26, 18
    cols = max(len(r) for r in rows)
    w = pad + cols * (cell[0] + pad)
    h = head + len(rows) * (cell[1] + cap + pad) + pad
    sheet = Image.new("RGB", (w, h), (24, 26, 24))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 7), title, fill=(240, 240, 235))
    for i, row in enumerate(rows):
        y = head + i * (cell[1] + cap + pad)
        for j, shot in enumerate(row):
            x = pad + j * (cell[0] + pad)
            sheet.paste(shot.img.resize(cell), (x, y))
            dr.text((x + 3, y + cell[1] + 3), shot.caption, fill=(225, 225, 220))
    sheet.save(path, quality=86)
    return path


def wrong_code(kp) -> str:
    code = kp.cfg["code"]
    labels = [b["label"] for b in kp.buttons]
    if kp.lock.code_kind == "set":
        spare = next((l for l in labels if l not in code), None)
        return code[:-1] + spare if spare else code[::-1]
    other = next(l for l in labels if l != code[0])
    return other + code[1:]


def film(door_dir: str, out_dir: str, size) -> str:
    f = KeypadFilm(door_dir, size)
    kp = f.kp
    code, bad = kp.cfg["code"], wrong_code(f.kp)
    mech = kp.lock.code_kind == "set"
    rows = []
    for label, seq in (("right code", code), ("wrong code", bad)):
        f.restart()
        row = [f.shoot(f"{label}: {seq}   (lock {'thrown' if kp.cfg['engaged'] else 'not thrown'})", view="hardware")]
        for c in seq:
            row.append(f.press(c))
        f.turn_and_push(1.2)
        state = "unlocked" if kp.unlocked else f"still locked ({kp.lock.wrong_attempts} wrong)"
        row.append(f.shoot(f"turn the lever: {state}", view="hardware"))
        f.turn_and_push(2.5)
        row.append(f.shoot(f"door {f.angle_deg():.0f}°", view="door"))
        rows.append(row)
    name = os.path.basename(door_dir)
    title = (f"{name} · {kp.cfg['lock_model']} · code {code} "
             f"({'buttons in any order, then the lever' if mech else 'digits in order'}) · "
             f"release: {kp.release_mode} · every button is a body on a {kp.travel * 1000:.1f} mm slide joint "
             f"({kp.cfg['preload_force_N']}-{kp.cfg['press_force_N']} N return spring)")
    path = os.path.join(out_dir, f"keypad_code_entry_{name.split('_')[0]}.jpg")
    strip(rows, path, title, cell=size)
    f.close()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--out", default="docs/media")
    ap.add_argument("--doors", default="db0526_swing_single,db0166_swing_single")
    ap.add_argument("--size", default="380x300")
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split("x"))
    os.makedirs(a.out, exist_ok=True)
    for did in a.doors.split(","):
        p = film(os.path.join(a.assets, "doors", did), a.out, (w, h))
        print("wrote", p)


if __name__ == "__main__":
    main()

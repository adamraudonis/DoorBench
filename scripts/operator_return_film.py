#!/usr/bin/env python
"""Frame sequence of an operator being released: evidence for the `operator_returns` gate.

Drives one operator joint to full travel with the same kinematic hand the QA gate uses, lets go, and renders the
handle every few milliseconds from the door's handle-detail camera.  Writes the frames as one contact sheet (and,
with --frames, the individual JPEGs).

Usage:
  PYTHONPATH=$PWD python scripts/operator_return_film.py --door db0048_swing_single \
      --out docs/media/operator_return_lever.jpg [--joint leaf_handle_hinge] [--open-deg 30]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from doorbench import qa as QA   # noqa: E402  (drive schedule constants shared with the gate)


def main() -> int:
    import mujoco
    from PIL import Image, ImageDraw

    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--door", default="db0048_swing_single")
    ap.add_argument("--joint", default=None, help="operator joint (default: meta.operator_joint)")
    ap.add_argument("--out", default="docs/media/operator_return_lever.jpg")
    ap.add_argument("--open-deg", type=float, default=0.0, help="hold the leaf open at this angle (latch clear of the strike)")
    ap.add_argument("--size", default="420x330")
    ap.add_argument("--every-ms", type=float, default=40.0)
    ap.add_argument("--duration-ms", type=float, default=440.0)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--frames", action="store_true", help="also write the individual frames next to --out")
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split("x"))

    door_dir = os.path.join(a.assets, "doors", a.door)
    with open(os.path.join(door_dir, "model.json")) as f:
        meta = json.load(f)["meta"]
    with open(os.path.join(door_dir, "spec.json")) as f:
        phys = json.load(f)["physics"]
    jname = a.joint or meta["operator_joint"]
    rec = (phys.get("operator", {}).get("joints") or {}).get(jname, {})

    spec = mujoco.MjSpec.from_file(os.path.join(door_dir, "door.xml"))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = max(w, 640), max(h, 480)
    m = spec.compile()
    d = mujoco.MjData(m)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["primary_joint"])
    adr, dof = m.jnt_qposadr[j], m.jnt_dofadr[j]
    lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
    dt = float(m.opt.timestep)
    hinge = int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)

    renderer = mujoco.Renderer(m, height=h, width=w)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in (0, 1, 2):
        opt.geomgroup[g] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    tgt = meta.get("handle_cam_target") or [0.0, 0.0, 1.0]
    cam.lookat[:] = tgt
    cam.distance = 0.55
    cam.azimuth = 90.0 - float(meta.get("u", 1.0)) * 38.0
    cam.elevation = -14.0

    # --- drive: the same kinematic hand the gate uses, then release
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    door_q = math.radians(a.open_deg) if a.open_deg else None
    target = lo + 0.95 * (hi - lo)
    n_ramp = max(1, int(QA.OPERATOR_DRIVE_RAMP_S / dt))
    for k in range(n_ramp + int(QA.OPERATOR_DRIVE_HOLD_S / dt)):
        f = min(1.0, (k + 1) / n_ramp)
        d.qpos[adr] = lo + f * (target - lo)
        d.qvel[dof] = (target - lo) / QA.OPERATOR_DRIVE_RAMP_S if f < 1.0 else 0.0
        d.qfrc_applied[:] = 0
        mujoco.mj_step(m, d)
    if door_q is not None:
        q_start = float(d.qpos[m.jnt_qposadr[pj]])
        n_open = max(1, int(QA.DOOR_OPEN_RAMP_S / dt))
        for k in range(n_open + int(QA.DOOR_OPEN_SETTLE_S / dt)):
            f = min(1.0, (k + 1) / n_open)
            d.qpos[adr], d.qvel[dof] = target, 0.0
            d.qfrc_applied[:] = 0
            QA._door_servo(m, d, pj, q_start + f * (door_q - q_start), 200.0)
            mujoco.mj_step(m, d)

    def shot(t_ms: float):
        renderer.update_scene(d, camera=cam, scene_option=opt)
        img = Image.fromarray(renderer.render())
        q = float(d.qpos[adr]) - lo
        txt = f"t = {t_ms:>5.0f} ms   " + (f"{math.degrees(q):5.1f} deg" if hinge else f"{q * 1000:5.2f} mm")
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, img.size[0], 18], fill=(0, 0, 0))
        dr.text((5, 4), txt + ("   RELEASED" if t_ms > 0 else "   held at full travel"), fill=(255, 255, 255))
        return img

    frames = [shot(0.0)]
    step_n = max(1, int(round(a.every_ms / 1000.0 / dt)))
    n_total = int(round(a.duration_ms / 1000.0 / dt))
    for k in range(n_total):
        d.qfrc_applied[:] = 0
        if door_q is not None:
            QA._door_servo(m, d, pj, door_q, 200.0)
        mujoco.mj_step(m, d)
        if (k + 1) % step_n == 0:
            frames.append(shot((k + 1) * dt * 1000.0))
    renderer.close()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = min(a.cols, len(frames))
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h + 22), (16, 18, 22))
    for i, img in enumerate(frames):
        sheet.paste(img, ((i % cols) * w, (i // cols) * h + 22))
    cap = (f"{a.door} {jname} ({rec.get('operator_model', '')}, {rec.get('return_kind', '')}): released from full travel"
           + (f", leaf held open {a.open_deg:.0f} deg" if a.open_deg else ", door closed")
           + f" - preload {rec.get('spring_preload', 0):.2f}, rate {rec.get('spring_rate', 0):.2f},"
           + f" damping {rec.get('damping', 0):.3f} (zeta {rec.get('damping_ratio', 0)}),"
           + f" expected return {rec.get('expected_return_time_s')} s")
    ImageDraw.Draw(sheet).text((6, 6), cap, fill=(220, 226, 235))
    sheet.save(a.out, quality=80)
    print(f"{a.out}  {len(frames)} frames  {os.path.getsize(a.out) / 1024:.0f} kB")
    if a.frames:
        base = os.path.splitext(a.out)[0]
        for i, img in enumerate(frames):
            img.save(f"{base}_{i:02d}.jpg", quality=80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

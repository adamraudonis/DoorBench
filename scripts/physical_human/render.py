"""Render only recorded native simulation states; never change the motion."""

import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    m = mujoco.MjModel.from_xml_path(str(a.directory / "scene.xml"))
    d = mujoco.MjData(m)
    clip = np.load(a.directory / "trajectory.npz")
    report = json.loads((a.directory / "report.json").read_text())
    r = mujoco.Renderer(m, height=800, width=1200)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.55, -0.30, 1.0]
    cam.distance = 3.5
    cam.azimuth = 120
    cam.elevation = -15
    selected = [
        round(f * (len(clip["qpos"]) - 1)) for f in [0, 0.2, 0.36, 0.55, 0.75, 0.98]
    ]
    frames = []
    for k, q in enumerate(clip["qpos"]):
        if not a.video and k not in selected:
            continue
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        im = r.render().copy()
        draw = ImageDraw.Draw(img := Image.fromarray(im))
        row = report["rows"][k]
        draw.text(
            (25, 22),
            f"{row['t']:.2f}s  {row['phase']} | Door {row['door_deg']:.1f} deg | Lever {row['lever_deg']:.1f} deg | Touch {row['touch_n']:.1f} N",
            fill="white",
            font=(
                ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
                if Path("/System/Library/Fonts/Helvetica.ttc").exists()
                else ImageFont.load_default(size=22)
            ),
        )
        if k in selected:
            img.save(a.directory / f"frame-{k:03}.png")
        if a.video:
            frames.append(np.asarray(img))
    if a.video:
        iio.mimwrite(
            a.directory / "overview.mp4",
            frames,
            fps=50,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
    r.close()


if __name__ == "__main__":
    main()

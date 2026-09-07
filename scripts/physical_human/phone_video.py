"""Export a portrait MP4 from saved native states: real time, then half speed.

No new simulation or trajectory editing. Requires MuJoCo, Pillow, imageio and
imageio-ffmpeg. H.264/yuv420p and faststart support phone/browser playback.
"""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.directory
    report = json.loads((source / "report.json").read_text())
    if (
        hashlib.sha256((source / "scene.xml").read_bytes()).hexdigest()
        != report["scene_sha256"]
    ):
        raise ValueError("Scene does not match the recorded run")
    trace = np.load(source / "trajectory.npz")
    model = mujoco.MjModel.from_xml_path(str(source / "scene.xml"))
    data = mujoco.MjData(model)
    original_rgba = model.geom_rgba.copy()
    hidden = [
        i
        for i in range(model.ngeom)
        if (model.body(model.geom_bodyid[i]).name or "").startswith(("actor_", "hand_"))
        and not (model.geom(i).name or "").startswith("hand_l_")
    ]
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    whole = mujoco.Renderer(model, height=480, width=720)
    hand = mujoco.Renderer(model, height=400, width=720)
    wide = mujoco.MjvCamera()
    wide.lookat[:] = [0.55, -0.3, 1.05]
    wide.distance = 3.15
    wide.azimuth = 115
    wide.elevation = -12
    detail = mujoco.MjvCamera()
    detail.distance = 0.38
    detail.azimuth = 65
    detail.elevation = 12
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.out,
        fps=25,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    frames = 0
    try:
        for speed in [1.0, 0.5]:
            duration = float(trace["time"][-1] - trace["time"][0])
            for frame in range(round(duration / speed * 25) + 1):
                time = trace["time"][0] + frame / 25 * speed
                k = int(np.argmin(abs(trace["time"] - time)))
                data.qpos[:] = trace["qpos"][k]
                mujoco.mj_forward(model, data)
                row = report["rows"][k]
                model.geom_rgba[:] = original_rgba
                whole.update_scene(data, camera=wide, scene_option=options)
                top = whole.render().copy()
                model.geom_rgba[hidden, 3] = 0
                detail.lookat[:] = data.geom_xpos[model.geom("lever_grip").id]
                hand.update_scene(data, camera=detail, scene_option=options)
                for contact in report["contacts"][k]:
                    if hand.scene.ngeom >= hand.scene.maxgeom:
                        break
                    geom = hand.scene.geoms[hand.scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.0035] * 3),
                        np.array(contact[:3]),
                        np.eye(3).ravel(),
                        np.array([1.0, 0.65, 0.1, 1.0]),
                    )
                    hand.scene.ngeom += 1
                lower = hand.render().copy()
                canvas = Image.new("RGB", (720, 1280), "#142328")
                canvas.paste(Image.fromarray(top), (0, 100))
                canvas.paste(Image.fromarray(lower), (0, 630))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (28, 19),
                    "DoorBench / Physical human",
                    font=font(29),
                    fill="#f0f0e6",
                )
                draw.text(
                    (28, 59),
                    f"{'REAL TIME' if speed == 1 else 'HALF SPEED'}  ·  {row['phase'].upper()}  ·  {row['t']:.2f} s",
                    font=font(20),
                    fill="#9fc5bf",
                )
                draw.text(
                    (28, 591),
                    "HAND CLOSE-UP  /  gold dots = contact",
                    font=font(22),
                    fill="#e6c68b",
                )
                metrics = [
                    ("DOOR", f"{max(0, row['door_deg']):.1f}°"),
                    ("LATCH", f"{max(0, row['latch_mm']):.1f} mm"),
                    ("HAND CONTACT", f"{row['touch_n']:.1f} N"),
                ]
                for j, (label, value) in enumerate(metrics):
                    x = 28 + j * 236
                    draw.text((x, 1056), label, font=font(16), fill="#9eb2ad")
                    draw.text((x, 1084), value, font=font(33), fill="#f3ead9")
                draw.line((28, 1140, 692, 1140), fill="#38504e", width=1)
                draw.text(
                    (28, 1160),
                    "Recorded MuJoCo physics · No door motor or hand weld",
                    font=font(19),
                    fill="#c4d1c9",
                )
                draw.text(
                    (28, 1194),
                    "Standing opening prototype. Not validated human ground truth.",
                    font=font(18),
                    fill="#90a8a1",
                )
                draw.text(
                    (28, 1224),
                    "Same native motion shown twice. No walking in this demo.",
                    font=font(18),
                    fill="#90a8a1",
                )
                writer.append_data(np.asarray(canvas))
                if frame == round(3.7 / speed * 25) and speed == 1:
                    canvas.save(args.out.with_suffix(".jpg"))
                frames += 1
    finally:
        writer.close()
        whole.close()
        hand.close()
    receipt = {
        "video": args.out.name,
        "frames": frames,
        "fps": 25,
        "duration_s": frames / 25,
        "speeds": [1, 0.5],
        "resolution": [720, 1280],
        "scene_sha256": report["scene_sha256"],
        "trajectory_sha256": hashlib.sha256(
            (source / "trajectory.npz").read_bytes()
        ).hexdigest(),
        "video_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

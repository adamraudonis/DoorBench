"""Export enlarged hand views of the exact native rollout as a portrait MP4.

The first pass shows the thumb side at real time; the second shows the finger
side at half speed. A body inset and an axial projection give spatial context.
No new simulation or trajectory editing. H.264/yuv420p with faststart.
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


def cross_section(canvas, model, data, row):
    """Projection along the handle axis; true landmark positions, not a diagrammed pose."""
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((600, 1080, 1080, 1520), fill="#1c3036")
    draw.text((630, 1096), "ALONG THE HANDLE", font=font(21), fill="#c5d4cf")
    draw.text((630, 1129), "Near side       Far side", font=font(19), fill="#95b2b2")
    if row["phase"] in ("settle", "reach", "place around lever"):
        draw.text((710, 1300), "Approaching", font=font(25), fill="#95b2b2")
        return
    rotation = data.xmat[model.body("lever").id].reshape(3, 3)
    origin = data.xpos[model.body("lever").id]
    points = np.array(
        [data.site_xpos[model.site(f"hand_keypoint_l_{i:02}").id] for i in range(21)]
    )
    local = (points - origin) @ rotation
    # Keep the complete five-digit projection inside the panel throughout the
    # grasp. Skipping a whole chain when one MCP leaves the panel hid fingers.
    scale = 2600
    yz = (local[:, 1:] - [-0.066, 0]) * scale
    xy = np.c_[840 + yz[:, 0], 1340 - yz[:, 1]]
    draw.line((840, 1180, 840, 1500), fill="#3a5155", width=2)
    radius = 0.012 * scale
    draw.ellipse(
        (840 - radius, 1340 - radius, 840 + radius, 1340 + radius),
        fill="#c79742",
        outline="#e9c675",
        width=3,
    )
    for ids, color in [([2, 3, 4], "#4bd3e9")] + [
        (list(range(5 + 4 * n, 9 + 4 * n)), "#ebe8d1") for n in range(4)
    ]:
        line = [tuple(p) for p in xy[ids]]
        if not all(600 < p[0] < 1078 and 1160 < p[1] < 1518 for p in line):
            raise ValueError("A hand chain leaves the diagnostic projection panel")
        draw.line(line, fill=color, width=5)
        for x, y in line:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    draw.text((630, 1485), "Native skeleton projection", font=font(19), fill="#95b2b2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.directory
    report = json.loads((source / "report.json").read_text())
    scene_hash = hashlib.sha256((source / "scene.xml").read_bytes()).hexdigest()
    trajectory_hash = hashlib.sha256(
        (source / "trajectory.npz").read_bytes()
    ).hexdigest()
    if scene_hash != report["scene_sha256"]:
        raise ValueError("Scene does not match the recorded run")
    if trajectory_hash != report["trajectory_sha256"]:
        raise ValueError("Trajectory does not match the recorded run")
    if (
        not report.get("quality_passed")
        or not report.get("grasp", {}).get("passed")
        or report["grasp"].get("minimum_loaded_fingers") != 4
    ):
        raise ValueError(
            "Refusing to present a run without passing four-finger/opposing-thumb checks"
        )
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
    options.geomgroup[4] = 0
    whole = mujoco.Renderer(model, height=440, width=600)
    hand = mujoco.Renderer(model, height=900, width=1080)
    wide = mujoco.MjvCamera()
    wide.lookat[:] = [0.52, -0.37, 1.02]
    wide.distance = 3.2
    wide.azimuth = 135
    wide.elevation = -12
    detail = mujoco.MjvCamera()
    detail.distance = 0.24
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
        for speed, azimuth, elevation, label in [
            (1.0, 140, -12, "THUMB SIDE"),
            (0.5, 310, -18, "FOUR-FINGER SIDE"),
        ]:
            duration = float(trace["time"][-1] - trace["time"][0])
            for frame in range(round(duration / speed * 25) + 1):
                time = trace["time"][0] + frame / 25 * speed
                k = int(np.argmin(abs(trace["time"] - time)))
                data.qpos[:] = trace["qpos"][k]
                mujoco.mj_forward(model, data)
                row = report["rows"][k]
                model.geom_rgba[:] = original_rgba
                whole.update_scene(data, camera=wide, scene_option=options)
                body_image = whole.render().copy()
                model.geom_rgba[hidden, 3] = 0
                model.geom_rgba[model.geom("door_leaf").id, 3] = 0.08
                detail.lookat[:] = data.geom_xpos[model.geom("lever_grip").id] + [
                    0,
                    0,
                    0.015,
                ]
                detail.azimuth = azimuth - row["door_deg"]
                detail.elevation = elevation
                hand.update_scene(data, camera=detail, scene_option=options)
                hand.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                hand_image = hand.render().copy()
                canvas = Image.new("RGB", (1080, 1920), "#142328")
                canvas.paste(Image.fromarray(hand_image), (0, 112))
                canvas.paste(Image.fromarray(body_image), (0, 1080))
                cross_section(canvas, model, data, row)
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (36, 20),
                    "DoorBench / Opposing grasp",
                    font=font(39),
                    fill="#f0f0e6",
                )
                draw.text(
                    (36, 72),
                    f"{label} · {'REAL TIME' if speed == 1 else 'HALF SPEED'} · {row['t']:.2f} s",
                    font=font(25),
                    fill="#9fc5bf",
                )
                draw.text(
                    (36, 1030),
                    "CYAN = THUMB     IVORY = FOUR FINGERS     GOLD = HANDLE",
                    font=font(24),
                    fill="#dce6da",
                )
                grasp = row["grasp"]
                metrics = [
                    ("THUMB CONTACT", f"{grasp['thumb_normal_force_n']:.1f} N"),
                    ("FINGERS OPPOSING", f"{grasp['opposed_loaded_fingers']} / 4"),
                    ("DOOR OPENING", f"{max(0, row['door_deg']):.1f}°"),
                ]
                for j, (name, value) in enumerate(metrics):
                    x = 36 + j * 352
                    draw.text((x, 1562), name, font=font(22), fill="#9eb2ad")
                    draw.text(
                        (x, 1598),
                        value,
                        font=font(48),
                        fill="#4bd3e9" if j == 0 else "#f3ead9",
                    )
                draw.line((36, 1670, 1044, 1670), fill="#38504e", width=2)
                draw.text(
                    (36, 1698), row["phase"].upper(), font=font(31), fill="#e6c68b"
                )
                working = row["phase"] in ("press lever", "pull", "hold open")
                status = (
                    "Four fingers together; thumb underneath on the opposite side"
                    if working
                    else "The open hand approaches before the fingers close"
                )
                draw.text((36, 1743), status, font=font(27), fill="#c4d1c9")
                draw.text(
                    (36, 1810),
                    "Recorded MuJoCo physics · No door motor or hand weld",
                    font=font(25),
                    fill="#90a8a1",
                )
                draw.text(
                    (36, 1851),
                    "Synthetic opening + hold reference · Not human motion capture",
                    font=font(24),
                    fill="#90a8a1",
                )
                writer.append_data(np.asarray(canvas))
                if speed == 1 and frame == round(3.7 * 25):
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
        "resolution": [1080, 1920],
        "scene_sha256": scene_hash,
        "trajectory_sha256": trajectory_hash,
        "video_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "grasp_acceptance": report["grasp"],
    }
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

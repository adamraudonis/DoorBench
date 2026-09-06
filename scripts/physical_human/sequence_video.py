"""Render the recorded full native sequence with enlarged hand/foot inspection."""

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
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def ease(value):
    u = float(np.clip(value, 0, 1))
    return u**3 * (10 - 15 * u + 6 * u * u)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.directory
    report = json.loads((source / "report.json").read_text())
    if not report.get("quality_passed") or not report.get("traversal", {}).get(
        "passed"
    ):
        raise ValueError("Refusing to present an unverified full traversal")
    for name, key in [
        ("scene.xml", "scene_sha256"),
        ("trajectory.npz", "trajectory_sha256"),
    ]:
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != report[key]:
            raise ValueError("Run hash mismatch: " + name)
    model = mujoco.MjModel.from_xml_path(str(source / "scene.xml"))
    data = mujoco.MjData(model)
    trace = np.load(source / "trajectory.npz")
    rgba = model.geom_rgba.copy()
    wings = [model.geom(name).id for name in ["wall_left", "wall_right", "wall_above"]]
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.geomgroup[4] = 0
    body_render = mujoco.Renderer(model, height=1000, width=1080)
    detail_render = mujoco.Renderer(model, height=600, width=1080)
    wide = mujoco.MjvCamera()
    wide.lookat[:] = [0.58, -0.03, 1.00]
    wide.distance = 3.55
    wide.elevation = -12
    detail = mujoco.MjvCamera()
    hand_id = model.site("palm_l").id
    grip_id = model.geom("lever_grip").id
    root = model.body("actor_pelvis").id
    chest = model.body("actor_chest").id
    hand_hidden = [
        g
        for g in range(model.ngeom)
        if (model.body(model.geom_bodyid[g]).name or "").startswith(("actor_", "hand_"))
        and not (model.geom(g).name or "").startswith("hand_l_")
    ]
    feet_hidden = [
        g
        for g in range(model.ngeom)
        if (model.body(model.geom_bodyid[g]).name or "").startswith(("actor_", "hand_"))
        and not any(
            n in (model.body(model.geom_bodyid[g]).name or "")
            for n in ["actor_hip_", "actor_knee_", "actor_ankle_"]
        )
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fps = 25
    writer = imageio.get_writer(
        args.out,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    count = round(float(trace["time"][-1] - trace["time"][0]) * fps) + 1
    selected = []
    try:
        for frame in range(count):
            t = float(trace["time"][0] + frame / fps)
            k = int(np.argmin(abs(trace["time"] - t)))
            data.qpos[:] = trace["qpos"][k]
            mujoco.mj_forward(model, data)
            row = report["rows"][k]
            walking = row["phase"] in ("walk through", "settle beyond door")
            model.geom_rgba[:] = rgba
            # Inspection cutaway: omit adjacent wall wings from the image only.
            # Their collisions remain present in the recorded simulation.
            model.geom_rgba[wings, 3] = 0
            wide.azimuth = 135 + 90 * ease((data.xpos[root, 1] + 0.55) / 0.95)
            body_render.update_scene(data, camera=wide, scene_option=options)
            full = body_render.render().copy()
            if walking:
                model.geom_rgba[feet_hidden, 3] = 0
                model.geom_rgba[model.geom("door_leaf").id, 3] = 0.16
                detail.lookat[:] = data.xpos[
                    [model.body("actor_ankle_l").id, model.body("actor_ankle_r").id]
                ].mean(axis=0)
                detail.lookat[2] = 0.05
                detail.distance = 1.8
                detail.azimuth = 90
                detail.elevation = -80
                detail_title = "FOOT CLEARANCE / overhead cutaway"
            else:
                model.geom_rgba[hand_hidden, 3] = 0
                model.geom_rgba[model.geom("door_leaf").id, 3] = 0.08
                grip = data.geom_xpos[grip_id]
                hand = data.site_xpos[hand_id]
                blend = ease((row["t"] - 7.4) / 1.0)
                detail.lookat[:] = (1 - blend) * (grip + [0, 0, 0.015]) + blend * hand
                detail.distance = 0.28 if row["t"] < 7.4 else 0.34
                detail.azimuth = 140 - row["door_deg"]
                detail.elevation = -14
                detail_title = "HAND CLOSE-UP / cyan thumb, ivory fingers"
            detail_render.update_scene(data, camera=detail, scene_option=options)
            detail_render.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            close = detail_render.render().copy()
            canvas = Image.new("RGB", (1080, 1920), "#142328")
            canvas.paste(Image.fromarray(full), (0, 100))
            canvas.paste(Image.fromarray(close), (0, 1150))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (32, 20),
                "DoorBench / Open, release & walk through",
                font=font(37),
                fill="#f0f2e9",
            )
            draw.text(
                (32, 68),
                f"{row['phase'].upper()}  /  {row['t']:.2f} s  /  REAL TIME",
                font=font(23),
                fill="#a7d8cd",
            )
            draw.text((32, 1112), detail_title, font=font(25), fill="#e2eee6")
            tilt = float(np.rad2deg(np.arccos(np.clip(data.xmat[chest, 8], -1, 1))))
            items = [
                ("CHEST TILT", f"{tilt:.1f}°"),
                ("DOOR", f"{max(0, row['door_deg']):.1f}°"),
            ]
            if row["phase"] in ("press lever", "pull", "hold open"):
                items.append(
                    ("OPPOSED FINGERS", f"{row['grasp']['opposed_loaded_fingers']} / 4")
                )
            elif row["t"] >= 8.4:
                items.append(("HAND", "Released"))
            elif row["t"] >= 6.3:
                items.append(("HAND", "Releasing"))
            else:
                items.append(("HAND", "Preparing"))
            for i, (label, value) in enumerate(items):
                x = 32 + 350 * i
                draw.text((x, 1768), label, font=font(21), fill="#96bbb3")
                draw.text((x, 1800), value, font=font(35), fill="#f0f2e9")
            draw.text(
                (32, 1862),
                "Recorded native physics · joint motors only",
                font=font(24),
                fill="#c0d2cd",
            )
            draw.text(
                (32, 1895),
                "Adjacent walls hidden for inspection; their collisions were active.",
                font=font(17),
                fill="#96bbb3",
            )
            writer.append_data(np.asarray(canvas))
            if frame in [75, 125, 175, 200, 250, 350, 450, 500, 550, count - 1]:
                path = args.out.parent / f"{args.out.stem}-frame-{frame:04}.png"
                canvas.save(path)
                selected.append(str(path.resolve()))
    finally:
        writer.close()
        body_render.close()
        detail_render.close()
    receipt = {
        "video": str(args.out.resolve()),
        "video_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "scene_sha256": report["scene_sha256"],
        "trajectory_sha256": report["trajectory_sha256"],
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frames": count,
        "fps": fps,
        "playback_speed": 1.0,
        "native_samples_only": True,
        "inspection_cutaways": "Adjacent wall wings hidden; hand-detail leaf translucent; walking-detail upper body hidden and leaf translucent. Physics unchanged.",
        "selected_frames": selected,
    }
    args.out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

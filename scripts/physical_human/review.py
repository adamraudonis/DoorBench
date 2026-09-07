"""Render enlarged native hand evidence, including automatically chosen worst frames.

Diagnostic renders are allowed for rejected candidates. This script never marks
visual quality as passed: the images and continuous videos must be inspected.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_review(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    trace = np.load(directory / "trajectory.npz")
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    rows = report["rows"]
    work = [
        i
        for i, r in enumerate(rows)
        if r["phase"] in ("press lever", "pull", "hold open")
    ]
    chosen = {
        "open-hand approach": int(np.argmin(abs(trace["time"] - 1.5))),
        "grasp closes": int(np.argmin(abs(trace["time"] - 2.1))),
        "lever press": int(np.argmin(abs(trace["time"] - 2.75))),
        "middle of pull": int(np.argmin(abs(trace["time"] - 4.3))),
        "final hold": len(rows) - 1,
        "minimum finger clearance": min(
            work, key=lambda i: rows[i]["grasp"]["minimum_finger_side_clearance_mm"]
        ),
        "minimum thumb clearance": min(
            work, key=lambda i: rows[i]["grasp"]["thumb_side_clearance_mm"]
        ),
        "fewest opposing contacts": min(
            work, key=lambda i: rows[i]["grasp"]["opposed_loaded_fingers"]
        ),
        "largest wrist error": max(work, key=lambda i: rows[i]["wrist_error_m"]),
        "fastest arm motion": max(
            work, key=lambda i: rows[i].get("arm_angular_speed_rad_s", 0)
        ),
    }
    output = directory / "visual-review"
    output.mkdir(exist_ok=True)
    rgba = model.geom_rgba.copy()
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.geomgroup[4] = 0
    renderer = mujoco.Renderer(model, height=1000, width=1600)
    camera = mujoco.MjvCamera()
    camera.distance = 0.24
    font = (
        ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 25)
        if Path("/System/Library/Fonts/Helvetica.ttc").exists()
        else ImageFont.load_default(size=25)
    )
    entries = []
    try:
        for label, k in chosen.items():
            data.qpos[:] = trace["qpos"][k]
            mujoco.mj_forward(model, data)
            row = rows[k]
            model.geom_rgba[:] = rgba
            for g in range(model.ngeom):
                body = model.body(model.geom_bodyid[g]).name or ""
                if body.startswith(("actor_", "hand_")) and not (
                    model.geom(g).name or ""
                ).startswith("hand_l_"):
                    model.geom_rgba[g, 3] = 0
            model.geom_rgba[model.geom("door_leaf").id, 3] = 0.08
            camera.lookat[:] = data.geom_xpos[model.geom("lever_grip").id] + [
                0,
                0,
                0.015,
            ]
            for view, azimuth, elevation in [
                ("thumb", 140, -12),
                ("fingers", 310, -18),
                ("along-grip", 180, 0),
            ]:
                camera.azimuth = azimuth - row["door_deg"]
                camera.elevation = elevation
                renderer.update_scene(data, camera=camera, scene_option=options)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                picture = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 934, 1600, 1000), fill="#142328")
                draw.text(
                    (20, 943),
                    f"{label} / {view} / {row['t']:.3f} s / {row['phase']}",
                    fill="white",
                    font=font,
                )
                name = f"{label.replace(' ', '-')}-{view}.png"
                picture.save(output / name)
                entries.append(
                    {
                        "file": name,
                        "frame": k,
                        "time_s": row["t"],
                        "selection": label,
                        "view": view,
                        "grasp": row["grasp"],
                    }
                )
    finally:
        renderer.close()
    manifest = {
        "schema": "doorbench.hand-visual-review.v1",
        "visual_approval": "pending inspection",
        "scene_sha256": hashlib.sha256(
            (directory / "scene.xml").read_bytes()
        ).hexdigest(),
        "trajectory_sha256": hashlib.sha256(
            (directory / "trajectory.npz").read_bytes()
        ).hexdigest(),
        "resolution": [1600, 1000],
        "native_state_edited": False,
        "images": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render_review(parser.parse_args().directory)

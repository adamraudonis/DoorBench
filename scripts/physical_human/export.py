"""Package the actual native trajectory for a local, orbitable evidence viewer."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--web", type=Path, required=True)
    p.add_argument("--three", type=Path, required=True)
    a = p.parse_args()
    out = a.web
    out.mkdir(parents=True, exist_ok=True)
    for n in ["index.html", "app.js", "style.css"]:
        shutil.copy(Path(__file__).parent / "web" / n, out / n)
    (out / "lib").mkdir(exist_ok=True)
    for n in ["three.module.js", "three.core.js"]:
        shutil.copy(a.three / "build" / n, out / "lib" / n)
    shutil.copy(
        a.three / "examples/jsm/controls/OrbitControls.js", out / "lib/OrbitControls.js"
    )
    m = mujoco.MjModel.from_xml_path(str(a.directory / "scene.xml"))
    d = mujoco.MjData(m)
    z = np.load(a.directory / "trajectory.npz")
    report = json.loads((a.directory / "report.json").read_text())
    geoms = []
    for i in range(m.ngeom):
        geoms.append(
            {
                "name": m.geom(i).name or f"joint_{i}",
                "body": m.body(m.geom_bodyid[i]).name,
                "type": int(m.geom_type[i]),
                "group": int(m.geom_group[i]),
                "size": m.geom_size[i].tolist(),
                "rgba": m.geom_rgba[i].tolist(),
            }
        )
    angle_names = [
        "hand_l_cmc_flexion",
        "hand_l_cmc_abduction",
        "hand_l_mp_flexion",
        "hand_l_ip_flexion",
        "actor_wrist_l_flexion",
        "actor_wrist_l_deviation",
    ]
    angle_ids = [m.joint(n).qposadr[0] for n in angle_names]
    frames = []
    for q in z["qpos"]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        frames.append(
            np.round(
                np.c_[
                    d.geom_xpos,
                    Rotation.from_matrix(d.geom_xmat.reshape(-1, 3, 3)).as_quat(),
                ],
                6,
            ).tolist()
        )
    (out / "replay.json").write_text(
        json.dumps(
            {
                "geoms": geoms,
                "angle_names": angle_names,
                "angles_deg": np.rad2deg(z["qpos"][:, angle_ids]).round(3).tolist(),
                "frames": frames,
                "time": z["time"].tolist(),
                "report": report,
            },
            separators=(",", ":"),
        )
    )
    shutil.copy(
        Path(__file__).parent / "anatomy/LICENSE-MyoSim.txt", out / "LICENSE-MyoSim.txt"
    )
    shutil.copy(Path(__file__).parent / "anatomy/README.md", out / "hand-provenance.md")
    for n in ["scene.xml", "trajectory.npz", "report.json"]:
        shutil.copy(a.directory / n, out / n)
    checks = {}
    for n in ["no-touch", "blocked"]:
        path = a.directory.parent / n / "report.json"
        if path.exists():
            checks[n] = {
                k: v
                for k, v in json.loads(path.read_text()).items()
                if k not in ["rows", "contacts"]
            }
    for case in checks.values():
        if (
            case["source_sha256"] != report["source_sha256"]
            or case["rig_sha256"] != report["rig_sha256"]
            or case.get("hand_source") != report.get("hand_source")
        ):
            raise ValueError(
                "Causal check comes from a different controller/rig revision"
            )
    if (a.directory / "overview.mp4").exists():
        shutil.copy(a.directory / "overview.mp4", out / "overview.mp4")
    checks["baseline"] = {
        k: v for k, v in report.items() if k not in ["rows", "contacts"]
    }
    checks["hashes"] = {
        n: hashlib.sha256((out / n).read_bytes()).hexdigest()
        for n in ["scene.xml", "trajectory.npz", "report.json"]
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    print(out.resolve())


if __name__ == "__main__":
    main()

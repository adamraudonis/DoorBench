#!/usr/bin/env python3
"""Fetch the pinned Unitree demo checkpoint, deployment config, and license."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
REVISION = "276801e46c5d433564f24658bac64f254b7d2d4b"
FILES = {
    "deploy/pre_train/g1/motion.pt": "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d",
    "deploy/deploy_mujoco/configs/g1.yaml": "73044e7d355c61915695c16d6e09eb3efef46eec1e3d708fd3eb9157dfe3bbbb",
    "LICENSE": "aef6394ba1597725a68308167324e675f562e6606027404deb1b9da254c2b9c1",
}


def main():
    for name, expected in FILES.items():
        path = ROOT / "robot_demo/third_party/unitree_g1_policy" / name
        if path.exists():
            data = path.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/{REVISION}/{name}"
            with urlopen(url, timeout=120) as response:
                data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {path}; existing files are not overwritten"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temp = path.with_suffix(path.suffix + ".partial")
            temp.write_bytes(data)
            temp.replace(path)
        print(f"{actual}  {name}")


if __name__ == "__main__":
    main()

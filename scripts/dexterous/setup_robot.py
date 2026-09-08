#!/usr/bin/env python3
"""Fetch the pinned upstream robot into ignored output; compile and audit it."""
import argparse
import json
import subprocess
from pathlib import Path
from doorbench.dexterous.model import UPSTREAM_URL, UPSTREAM_REVISION, prepare_robot

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, default=Path("out/dexterous/upstream/humanoid-bench"))
    p.add_argument("--output", type=Path)
    p.add_argument("--mechanics-profile", choices=("upstream-v1", "shadow-loopback-v2"), default="upstream-v1")
    a = p.parse_args()
    if a.output is None:
        filename = "h1-shadow.xml" if a.mechanics_profile == "upstream-v1" else "h1-shadow-loopback-v2.xml"
        a.output = Path("out/dexterous/robot") / filename
    if not a.upstream.exists():
        a.upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM_URL, str(a.upstream)], check=True)
        subprocess.run(["git", "-C", str(a.upstream), "checkout", "--detach", UPSTREAM_REVISION], check=True)
    path, audit = prepare_robot(a.upstream, a.output, mechanics_profile=a.mechanics_profile)
    print(json.dumps({"robot": str(path), **{k:v for k,v in audit.items() if k not in {"limits", "nominal_joint_positions", "actuator_names"}}}, indent=2))

if __name__ == "__main__":
    main()

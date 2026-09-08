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
    p.add_argument("--output", type=Path, default=Path("out/dexterous/robot/h1-shadow.xml"))
    a = p.parse_args()
    if not a.upstream.exists():
        a.upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM_URL, str(a.upstream)], check=True)
        subprocess.run(["git", "-C", str(a.upstream), "checkout", "--detach", UPSTREAM_REVISION], check=True)
    path, audit = prepare_robot(a.upstream, a.output)
    print(json.dumps({"robot": str(path), **{k:v for k,v in audit.items() if k not in {"limits", "nominal_joint_positions", "actuator_names"}}}, indent=2))

if __name__ == "__main__":
    main()

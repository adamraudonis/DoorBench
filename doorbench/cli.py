"""DoorBench command line.

  doorbench list [--family swing_single] [--task ...]
  doorbench show <door_id>                  print spec + physics summary
  doorbench build <door_id> --out DIR       export one door (all formats)
  doorbench qa <door_id>                    run the sign-off checks on an exported door
  doorbench view <door_id>                  open in the MuJoCo viewer
  doorbench stats                           dataset statistics from assets/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="doorbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--family"); p.add_argument("--task"); p.add_argument("--assets", default="assets")
    p = sub.add_parser("show"); p.add_argument("door_id"); p.add_argument("--assets", default="assets")
    p = sub.add_parser("build"); p.add_argument("door_id"); p.add_argument("--out", default="out/build"); p.add_argument("--formats", default="mjcf,urdf,usd,json")
    p = sub.add_parser("qa"); p.add_argument("door_id"); p.add_argument("--assets", default="assets")
    p = sub.add_parser("view"); p.add_argument("door_id"); p.add_argument("--assets", default="assets"); p.add_argument("--tier", default="full")
    p = sub.add_parser("stats"); p.add_argument("--assets", default="assets")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        man = json.load(open(os.path.join(a.assets, "manifest.json")))
        for d in man["doors"]:
            if a.family and d["family"] != a.family:
                continue
            if a.task and d.get("task") != a.task:
                continue
            print(f"{d['id']:28s} {d['family']:20s} {d['mass_kg']:7.1f} kg  {d['operator']:24s} lock={d['lock']:18s} {'locked' if d['lock_engaged'] else '      '} {d.get('task','')}")
    elif a.cmd == "show":
        spec = json.load(open(os.path.join(a.assets, "doors", a.door_id, "spec.json")))
        phys = spec.pop("physics", {})
        print(json.dumps({k: v for k, v in spec.items() if k not in ("tags",)}, indent=1))
        print("--- physics summary ---")
        print(json.dumps({k: phys[k] for k in ("mass", "hinge", "closer", "latch", "lock", "compliance") if k in phys}, indent=1))
    elif a.cmd == "build":
        from .spec import generate_all
        from .build import export_door
        specs = {s["id"]: s for s in generate_all()}
        s = specs[a.door_id]
        out = export_door(s, os.path.join(a.out, "doors"), os.path.join(a.out, "hardware"), formats=tuple(a.formats.split(",")))
        print(json.dumps(out, indent=1, default=str))
    elif a.cmd == "qa":
        from .qa import run_qa
        from .spec import generate_all
        d = os.path.join(a.assets, "doors", a.door_id)
        spec = json.load(open(os.path.join(d, "spec.json")))
        meta = json.load(open(os.path.join(d, "model.json")))["meta"]
        files = {"mjcf": {t: os.path.join(d, f) for t, f in (("full", "door.xml"), ("simple", "door_simple.xml"), ("minimal", "door_minimal.xml")) if os.path.exists(os.path.join(d, f))},
                 "urdf": {"full": os.path.join(d, "door.urdf")}, "usd": os.path.join(d, "door.usda")}
        qa = run_qa(spec, d, meta, files, spec["physics"])
        print(json.dumps(qa, indent=1))
    elif a.cmd == "view":
        import subprocess
        f = {"full": "scene.xml", "simple": "door_simple.xml", "minimal": "door_minimal.xml"}[a.tier]
        subprocess.call([sys.executable, "-m", "mujoco.viewer", "--mjcf", os.path.join(a.assets, "doors", a.door_id, f)])
    elif a.cmd == "stats":
        import collections
        man = json.load(open(os.path.join(a.assets, "manifest.json")))
        ds = [d for d in man["doors"] if not d.get("error")]
        print(f"{len(ds)} doors, {sum(1 for d in ds if d['signed_off'])} signed off")
        for key in ("family", "operator", "lock", "closer", "condition", "task", "difficulty"):
            c = collections.Counter(d[key] for d in ds)
            print(f"\n{key}:")
            for k, n in c.most_common():
                print(f"  {str(k):28s} {n}")


if __name__ == "__main__":
    main()

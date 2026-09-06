"""Before/after renders for the hardware the specs declared and the models did not draw.

Two of the classes the vision review found are best judged by eye, so they get a picture each:

  * a hatch captioned ``stop=prop_arm`` standing open with nothing holding it, beside the same hatch
    with the folding prop arm and its curb socket;
  * a door whose spec says the operator is on both faces, photographed from the FAR side - a blank
    slab before, the other half of the set after.

Usage (the "before" tree is a checkout of the commit before the fix, generated into a scratch dir)::

    python scripts/render_spec_realized_media.py --before /tmp/before/assets/doors \\
        --after assets/doors --out docs/media --pair db0389_hatch_ceiling:mechanism_open \\
        --pair db0168_ship_watertight:hardware_back
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render(door_dir: str, key: str, out_path: str, size=(760, 570)) -> str:
    from PIL import Image
    from doorbench.review.sheet import SheetRenderer

    r = SheetRenderer(door_dir, panel=size, supersample=1.4)
    try:
        panels, _ = r.panels()
    finally:
        r.close()
    panel = next(p for p in panels if p["key"] == key)
    Image.fromarray(panel["image"]).save(out_path, quality=88)
    return panel["label"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="doors/ of a dataset built before the fix")
    ap.add_argument("--after", default="assets/doors")
    ap.add_argument("--out", default="docs/media")
    ap.add_argument("--pair", action="append", required=True, metavar="DOOR_ID:PANEL_KEY")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for spec in a.pair:
        door_id, key = spec.split(":", 1)
        for tag, root in (("before", a.before), ("after", a.after)):
            d = os.path.join(root, door_id)
            if not os.path.isdir(d):
                print(f"skip {tag} {door_id}: {d} missing")
                continue
            out = os.path.join(a.out, f"spec_realized_{door_id}_{tag}.jpg")
            label = render(d, key, out)
            print(f"{out}  [{label}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

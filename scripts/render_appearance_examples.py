#!/usr/bin/env python3
"""Render reproducible Blender examples, including three looks for one door."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doorbench.appearance.pipeline import prepare_job, run_jobs
from doorbench.appearance.textures import load_texture_library


# Variant zero is reserved for the complete catalogue's default recipe.
EXAMPLES = [
    ('db0079_sliding_single', 1, 'wall_white_plaster', 'floor_limestone', 'wood_walnut', 'daylight'),
    ('db0079_sliding_single', 2, 'wall_red_brick', 'floor_concrete', 'wood_oak', 'overcast'),
    ('db0079_sliding_single', 3, 'wall_sage_paint', 'floor_oak', 'paint_charcoal', 'warm_interior'),
    ('db0044_pivot', 1, 'wall_white_plaster', 'floor_oak', None, 'daylight'),
    ('db0322_swing_single', 1, 'wall_subway_tile', 'floor_porcelain', None, 'warehouse'),
    ('db0013_swing_single', 1, 'wall_limewash', 'floor_oak', 'paint_porcelain', 'daylight'),
    ('db0017_hatch_ceiling', 1, 'wall_concrete', 'floor_concrete', None, 'warehouse'),
    ('db0241_hatch_floor', 1, 'wall_white_plaster', 'floor_slate', 'wood_oak', 'daylight'),
    ('db0168_ship_watertight', 1, 'wall_concrete', 'floor_dark_concrete', None, 'warehouse'),
    ('db0150_sliding_single', 1, 'wall_limewash', 'floor_oak', None, 'overcast'),
    ('db0033_gate_sliding', 1, 'wall_red_brick', 'floor_concrete', None, 'daylight'),
    ('db0008_sliding_bypass', 1, 'wall_white_plaster', 'floor_oak', None, 'daylight'),
    ('db0037_strip_curtain', 1, 'wall_subway_tile', 'floor_porcelain', None, 'warehouse'),
    ('db0066_revolving', 1, 'wall_white_plaster', 'floor_limestone', None, 'daylight'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', default='assets')
    ap.add_argument('--out', default='out/appearance')
    ap.add_argument('--textures', default='out/appearance-textures/manifest.json')
    ap.add_argument('--blender')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    library = load_texture_library(a.textures)
    jobs = [prepare_job(a.assets, door, a.out, variant=variant, wall=wall, floor=floor,
                        door_finish=finish, lighting=lighting, quality='photo', width=960, height=960,
                        texture_library=library, save_blend=door == 'db0079_sliding_single')
            for door, variant, wall, floor, finish, lighting in EXAMPLES]
    run_jobs(jobs, a.out, blender=a.blender, resume=a.resume)


if __name__ == '__main__':
    main()

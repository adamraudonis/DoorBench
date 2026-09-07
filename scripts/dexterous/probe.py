#!/usr/bin/env python3
"""Execute and record the native integration/balance probe. No opening claim."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.reaching import ReachingController

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--robot", type=Path, default=Path("out/dexterous/robot/h1-shadow.xml"))
    p.add_argument("--door", type=Path, default=Path("out/dexterous/assets/doors/db0055_swing_single"))
    p.add_argument("--output", type=Path, default=Path("out/dexterous/probe"))
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--reach-offset", type=float, nargs=3, default=[0, 0, 0], help="Both hand target offsets in robot-start coordinates")
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads(a.robot.with_suffix('.audit.json').read_text())
    env = DexterousDoorEnv(a.door, a.robot, audit)
    obs = env.reset(randomize=False, images=not a.no_images)
    controller = ReachingController(env, a.upstream)
    initial_targets = controller.targets.copy()
    start = time.monotonic(); rows = []; poses = []
    try:
        while env.d.time < a.seconds:
            phase = min(1., env.d.time / 2.)
            blend = phase * phase * (3 - 2 * phase)
            targets = initial_targets + blend * (controller.to_local.T @ np.array(a.reach_offset))
            action = controller.action(targets)
            obs = env.step(action, images=False)
            row = env.diagnostics(); rows.append(row); poses.append(env.d.qpos.copy())
            if len(rows) % 50 == 0:
                print(json.dumps(row), flush=True)
            if not row['finite'] or row['root_height_m'] < .45:
                break
        if not a.no_images:
            obs = env.observe(images=True)
            for key in ('rgb_left', 'rgb_right'):
                Image.fromarray(obs[key]).save(a.output / (key + '.png'))
            with mujoco.Renderer(env.m, height=720, width=960) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = env.d.xpos[env.pelvis] + [0, .45, .1]
                camera.distance = 3.8; camera.azimuth = 145; camera.elevation = -15
                options = mujoco.MjvOption(); options.sitegroup[:] = 0
                renderer.update_scene(env.d, camera=camera, scene_option=options)
                Image.fromarray(renderer.render()).save(a.output / 'whole-body.png')
        report = {'stage':'integration/balance probe', 'door_opening_claim':False,
                  'seconds_requested':a.seconds, 'elapsed_wall_s':time.monotonic()-start,
                  'max_torso_tilt_deg': max(r['torso_tilt_deg'] for r in rows),
                  'min_root_height_m':min(r['root_height_m'] for r in rows),
                  'final':rows[-1], 'observations':{k:list(v.shape) for k,v in obs.items()}}
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        np.savez_compressed(a.output/'trajectory.npz',qpos=np.array(poses))
        print(json.dumps(report,indent=2))
    finally:
        env.close()

if __name__ == '__main__':
    main()

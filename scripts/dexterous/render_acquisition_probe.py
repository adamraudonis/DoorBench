#!/usr/bin/env python3
"""Render recorded native acquisition states, with explicit outcome/time labels."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from doorbench.dexterous.environment import DexterousDoorEnv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','trial'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--view',choices=('hand','body'),default='hand')
    args=p.parse_args()
    report=json.loads((args.trial/'report.json').read_text())
    trace=json.loads((args.trial/'trace.json').read_text())
    trajectory=np.load(args.trial/'trajectory.npz')
    times=np.asarray([row['sim_time_s'] for row in trace])
    if len(times)!=len(trajectory['qpos']) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Recorded states need a matching monotonic physics clock')
    sim=DexterousDoorEnv(args.door,args.robot,json.loads(args.robot.with_suffix('.audit.json').read_text()))
    sim.reset(randomize=False,images=False)
    if trajectory['qpos'].shape[1]!=sim.m.nq:
        raise ValueError('Recording and regenerated model dimensions differ')
    # Diagnostic colors change rendering only: highlight the thumb and operator.
    for g in range(sim.m.ngeom):
        name=sim.m.body(sim.m.geom_bodyid[g]).name
        if name.startswith('robot/rh_'):
            sim.m.geom_matid[g]=-1
            sim.m.geom_rgba[g]=[.08,.5,.8,1.] if name.startswith('robot/rh_th') else [.35,.4,.45,1.]
        if sim.m.geom(g).name.startswith('leaf_handle'):
            sim.m.geom_matid[g]=-1;sim.m.geom_rgba[g]=[.65,.38,.08,1.]
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[.26,-.10,.93] if args.view=='hand' else [.05,-.15,.9]
    camera.distance=.48 if args.view=='hand' else 3.1
    camera.azimuth=150;camera.elevation=-25 if args.view=='hand' else -12
    options=mujoco.MjvOption();options.sitegroup[:]=0
    frame_times=np.arange(times[0],times[-1]+1e-8,.04)
    indices=np.minimum(np.searchsorted(times,frame_times),len(times)-1)
    out=args.trial/f'{args.view}.mp4'
    try:
        with mujoco.Renderer(sim.m,height=720,width=960) as renderer:
            with imageio.get_writer(out,fps=25,codec='libx264',quality=8) as writer:
                for frame,i in enumerate(indices):
                    sim.d.qpos[:]=trajectory['qpos'][i]
                    mujoco.mj_kinematics(sim.m,sim.d)
                    renderer.update_scene(sim.d,camera=camera,scene_option=options)
                    sim.hide_sensor_overlays(renderer.scene)
                    image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                    draw.rectangle((0,0,960,35),fill='black')
                    outcome='PROBE CHECKS PASSED' if report['passed'] else 'FAILED ACQUISITION'
                    draw.text((12,12),f'RECORDED MUJOCO PHYSICS | {outcome} | t={times[i]:.2f}s | thumb: blue',fill='white')
                    writer.append_data(np.asarray(image))
                    if frame in (0,len(indices)//2,len(indices)-1):
                        image.save(args.trial/f'{args.view}-{frame:04d}.png')
        print(out)
    finally:
        sim.close()


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Render saved native physics states; this script never fabricates motion."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import imageio.v2 as imageio
from doorbench.dexterous.reach_training import ReachTeacherEnv


def main():
    p=argparse.ArgumentParser()
    for name in ('upstream','robot','door','trajectory','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--distance',type=float,default=.25)
    a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    env=ReachTeacherEnv(a.door,a.robot,a.upstream,distance=a.distance)
    env.reset(seed=a.seed);sim=env.sim
    trajectory=np.load(a.trajectory);poses=trajectory['qpos']
    if poses.shape[1]!=sim.m.nq:raise ValueError('Trajectory and model differ')
    options=mujoco.MjvOption();options.sitegroup[:]=0
    camera=mujoco.MjvCamera();camera.distance=3.1;camera.azimuth=145;camera.elevation=-12
    camera.lookat[:]=sim.d.xpos[sim.pelvis]+[0,.3,-.1]
    try:
        with mujoco.Renderer(sim.m,height=720,width=960) as renderer:
            with imageio.get_writer(out,fps=25,codec='libx264',quality=8) as writer:
                for i,pose in enumerate(poses):
                    if i%2:continue
                    sim.d.qpos[:]=pose
                    if 'qvel' in trajectory:sim.d.qvel[:]=trajectory['qvel'][i]
                    if 'ctrl' in trajectory:sim.d.ctrl[:]=trajectory['ctrl'][i]
                    mujoco.mj_forward(sim.m,sim.d)
                    renderer.update_scene(sim.d,camera=camera,scene_option=options)
                    sim.hide_sensor_overlays(renderer.scene)
                    frame=renderer.render();writer.append_data(frame)
                    if i in (0,(len(poses)//4)*2,(len(poses)//2)*2-2):
                        imageio.imwrite(out.with_name(out.stem+f'-{i:04d}.png'),frame)
    finally:env.close()

if __name__=='__main__':main()

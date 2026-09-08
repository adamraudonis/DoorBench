#!/usr/bin/env python3
"""Render timestamp-matched native approach states in their original door scene."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import mujoco,numpy as np
from PIL import Image,ImageDraw
from doorbench.dexterous.locomotion_approach import make_door_approach


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);args=p.parse_args()
    report=json.loads((args.trial/'report.json').read_text());config=report['arguments'];trace=json.loads((args.trial/'trace.json').read_text());states=np.load(args.trial/'trajectory.npz');poses=states['qpos'];times=np.array([r['time_s'] for r in trace])
    sim,goal,yaw=make_door_approach(config['door'],config['robot'],args.trial/'reference.json',distance=config['distance'],lateral=config['lateral'],yaw_offset=config['yaw_offset'],seed=config['seed'],joint_noise=config['joint_noise']);m,d=sim.m,sim.d
    if poses.shape!=(len(times),m.nq) or np.any(np.diff(times)<=0):raise ValueError('Need matched timestamps and native model states')
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=720;m.vis.headlight.ambient[:]=[.15,.15,.15];m.vis.headlight.diffuse[:]=[.45,.45,.45];m.light_diffuse[:]*=.6
    for g in range(m.ngeom):
        name=m.body(m.geom_bodyid[g]).name
        if name.startswith('robot/'):
            m.geom_matid[g]=-1;m.geom_rgba[g]=[.86,.62,.23,1] if name.startswith(('robot/lh_','robot/rh_')) else [.25,.40,.55,1]
    camera=mujoco.MjvCamera();camera.lookat[:]=[goal[0],goal[1]-.2,1.];camera.distance=3.8;camera.azimuth=50;camera.elevation=-15
    options=mujoco.MjvOption();options.sitegroup[:]=0
    frame_times=np.arange(times[0],times[-1]+1e-8,.04);indices=np.minimum(np.searchsorted(times,frame_times),len(times)-1)
    with mujoco.Renderer(m,height=720,width=1280) as renderer:
        with imageio.get_writer(args.trial/'body.mp4',fps=25,codec='libx264',quality=8) as writer:
            for frame,index in enumerate(indices):
                d.qpos[:]=poses[index];mujoco.mj_kinematics(m,d);renderer.update_scene(d,camera=camera,scene_option=options);sim.hide_sensor_overlays(renderer.scene)
                picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture);draw.rectangle((0,0,1280,44),fill='black');row=trace[index]
                draw.text((12,8),f'RECORDED NATIVE PHYSICS | Door55 approach only | {"CHECKS PASSED" if report["passed"] else "FAILED CHECKS"} | t={times[index]:.2f} s',fill='white')
                draw.text((12,26),f'{row["stage"]} | target error: {row["position_error_m"]*1000:.1f} mm / {row["heading_error_deg"]:.2f} deg | no door commands',fill='white')
                writer.append_data(np.asarray(picture))
                if frame in (0,len(indices)//2,len(indices)-1):picture.save(args.trial/f'body-{frame:04d}.png')
    print(args.trial/'body.mp4');sim.close()
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Render recorded, timestamp-aligned initialized continuation states."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import mujoco,numpy as np
from PIL import Image,ImageDraw
from doorbench.dexterous.environment import DexterousDoorEnv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--fps',type=int,default=20);p.add_argument('--azimuth',type=float,default=30.);a=p.parse_args()
    report=json.loads((a.trial/'report.json').read_text());args=report['arguments'];rows=json.loads((a.trial/'trace.json').read_text());states=np.load(a.trial/'trajectory.npz');times=states['time_s'];poses=states['qpos']
    if len(times)!=len(rows) or len(times)!=len(poses) or np.any(np.diff(times)<=0):raise ValueError('Unaligned recording')
    robot=Path(args['robot']);s=DexterousDoorEnv(Path(args['door']),robot,json.loads(robot.with_suffix('.audit.json').read_text()));s.reset(randomize=False,images=False);m,d=s.m,s.d
    m.vis.global_.offwidth=960;m.vis.global_.offheight=720
    for g in range(m.ngeom):
        name=m.body(m.geom_bodyid[g]).name
        if name.startswith('robot/'):
            m.geom_matid[g]=-1;m.geom_rgba[g]=[.88,.61,.22,1] if name.startswith(('robot/lh_','robot/rh_')) else [.25,.4,.55,1]
    camera=mujoco.MjvCamera();camera.lookat[:]=[-.05,-.15,1.];camera.distance=3.8;camera.azimuth=a.azimuth;camera.elevation=-15
    opts=mujoco.MjvOption();opts.sitegroup[:]=0
    chosen={0,len(times)//3,2*len(times)//3,len(times)-1}
    with mujoco.Renderer(m,height=720,width=960) as renderer:
        with imageio.get_writer(a.trial/'continuation.mp4',fps=a.fps,codec='libx264',quality=7) as writer:
            for t in np.arange(times[0],times[-1]+1e-8,1/a.fps):
                i=min(np.searchsorted(times,t),len(times)-1);d.qpos[:]=poses[i];mujoco.mj_kinematics(m,d);renderer.update_scene(d,camera=camera,scene_option=opts);s.hide_sensor_overlays(renderer.scene)
                pic=Image.fromarray(renderer.render());draw=ImageDraw.Draw(pic);draw.rectangle((0,0,960,49),fill='black');draw.text((12,7),f'RECORDED NATIVE PHYSICS | INITIALIZED CONTINUATION | {"PASS" if report["passed"] else "FAILED CHECKS"}',fill='white');draw.text((12,28),f'{times[i]:.2f}s | {rows[i]["phase"]} | no root pose writes / door commands | no uninterrupted opening claim',fill='white');writer.append_data(np.asarray(pic))
                if i in chosen:pic.save(a.trial/f'body-{i:04d}.png');chosen.remove(i)
    s.close();print(a.trial/'continuation.mp4')
if __name__=='__main__':main()

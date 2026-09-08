#!/usr/bin/env python3
"""Render close-up hand evidence from measured trajectory samples, without stepping."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
from doorbench.dexterous.environment import DexterousDoorEnv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--times',type=float,nargs='+',required=True);p.add_argument('--side',choices=('lh','rh'),default='rh')
    p.add_argument('--azimuths',type=float,nargs='+',default=[30.,120.]);a=p.parse_args()
    report=json.loads((a.trial/'report.json').read_text());robot=Path(report['arguments']['robot'])
    s=DexterousDoorEnv(Path(report['arguments']['door']),robot,json.loads(robot.with_suffix('.audit.json').read_text()));s.reset(images=False,randomize=False);m,d=s.m,s.d
    z=np.load(a.trial/'trajectory.npz');times=z['time_s'];poses=z['qpos']
    if len(times)!=len(poses) or np.any(np.diff(times)<=0):raise ValueError('Unaligned measured trajectory')
    if not np.isfinite(a.times+a.azimuths).all() or min(a.times)<times[0] or max(a.times)>times[-1]+.05:raise ValueError('Requested view outside recorded interval')
    for g in range(m.ngeom):
        name=m.body(m.geom_bodyid[g]).name
        if name.startswith('robot/'):
            m.geom_matid[g]=-1;m.geom_rgba[g]=[.88,.61,.22,1] if name.startswith(('robot/lh_','robot/rh_')) else [.25,.4,.55,1]
    options=mujoco.MjvOption();options.sitegroup[:]=0;palm=m.site('robot/'+a.side+'_palm_touch').id
    with mujoco.Renderer(m,height=720,width=960) as renderer:
        for requested in a.times:
            ix=int(np.argmin(abs(times-requested)));d.qpos[:]=poses[ix];mujoco.mj_kinematics(m,d)
            for azimuth in a.azimuths:
                camera=mujoco.MjvCamera();camera.lookat[:]=d.site_xpos[palm];camera.distance=.65;camera.azimuth=azimuth;camera.elevation=-10
                renderer.update_scene(d,camera=camera,scene_option=options);s.hide_sensor_overlays(renderer.scene)
                picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture);draw.rectangle((0,0,960,30),fill='black')
                draw.text((8,8),f'{a.trial.name} | recorded {times[ix]:.3f}s | {a.side} | azimuth {azimuth:.0f} | initialized continuation',fill='white')
                picture.save(a.trial/f'hand-{requested:.3f}-{azimuth:.0f}.png')
    s.close()
if __name__=='__main__':main()

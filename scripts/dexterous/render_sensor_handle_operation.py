#!/usr/bin/env python3
"""Replay actual uninterrupted scripted handle operation, preserving failed outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from doorbench.dexterous.environment import DexterousDoorEnv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--azimuth',type=float,default=150.);p.add_argument('--hand',action='store_true');p.add_argument('--snapshots-only',action='store_true')
    a=p.parse_args();r=json.loads((a.trial/'report.json').read_text());prov=json.loads((a.trial/'provenance.json').read_text());args=prov['parameters']
    robot=Path(args['robot']);door=Path(args['door']);door=door if door.is_dir() else door.parent
    for file,wanted in ((robot,prov['robot_xml_sha256']),(door/'door.xml',prov['door_xml_sha256'])):
        if hashlib.sha256(file.read_bytes()).hexdigest()!=wanted:raise ValueError('Replay source differs from physical trial')
    archive=np.load(a.trial/'trajectory.npz');q=archive['qpos'];v=archive['qvel'];ts=archive['time']
    if len(q)!=len(ts) or len(v)!=len(ts) or abs(ts[-1]-r['duration_s'])>1e-9 or np.any(np.diff(ts)<=0):raise ValueError('Unaligned physical state archive')
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()));sim.reset(images=False,randomize=False)
    m,d=sim.m,sim.d
    for g in range(m.ngeom):
        name=m.body(m.geom_bodyid[g]).name
        if name.startswith('robot/'):
            m.geom_matid[g]=-1;m.geom_rgba[g]=[.72,.45,.13,1] if name.startswith(('robot/rh_','robot/lh_')) else [.25,.39,.53,1]
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,-.35,.87];camera.distance=3.3;camera.azimuth=a.azimuth;camera.elevation=-15
    if a.hand:camera.distance=.55;camera.elevation=-20
    opts=mujoco.MjvOption();opts.sitegroup[:]=0
    desired_times=np.array([0,ts[-1]/2,ts[-1]]) if a.snapshots_only else np.arange(0,ts[-1]+1e-8,.04)
    indices=np.minimum(np.searchsorted(ts,desired_times),len(ts)-1)
    from contextlib import nullcontext
    tag='hand' if a.hand else 'body'
    path=a.trial/f'scripted-operation-{tag}-az{a.azimuth:g}.mp4'
    with mujoco.Renderer(m,height=720,width=960) as renderer:
        with (nullcontext(None) if a.snapshots_only else imageio.get_writer(path,fps=25,codec='libx264',quality=7)) as writer:
            for frame,i in enumerate(indices):
                d.qpos[:]=q[i];d.qvel[:]=v[i];d.time=ts[i];mujoco.mj_kinematics(m,d)
                if a.hand:camera.lookat[:]=d.site_xpos[m.site('robot/rh_palm_touch').id]+np.array([0,0,-.025])
                renderer.update_scene(d,camera=camera,scene_option=opts);sim.hide_sensor_overlays(renderer.scene)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image);draw.rectangle((0,0,960,50),fill='black')
                draw.text((12,7),'RECORDED NATIVE PHYSICS | SENSOR BALANCE + SCRIPTED HANDLE OPERATION | '+('PASS' if r['passed'] else 'FAILED'),fill='white')
                draw.text((12,29),f't={ts[i]:.3f}s | Scripted high level; sensor-only balance + scripted acquisition/press; no learned/vision/opening claim',fill='white')
                if writer:writer.append_data(np.asarray(image))
                if frame in (0,len(indices)//2,len(indices)-1):image.save(a.trial/f'scripted-operation-{tag}-az{a.azimuth:g}-{frame:04d}.png')
    sim.close();print(path)
if __name__=='__main__':main()

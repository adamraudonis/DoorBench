#!/usr/bin/env python3
"""Render actual native approach/opening states, including failed trials.

No physics is stepped. Frames select recorded states; captions retain their
actual timestamps. Visual colors are diagnostic, not altered collision meshes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from doorbench.dexterous.environment import DexterousDoorEnv


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--view',choices=('body','left-hand','right-hand'),default='body')
    p.add_argument('--azimuth',type=float,default=150.)
    p.add_argument('--cutaway-walls',action='store_true',help='Hide wall visuals only for inspection through the doorway')
    p.add_argument('--from-time',type=float,default=0.);p.add_argument('--snapshots-only',action='store_true')
    a=p.parse_args();run=a.run.resolve()
    report=json.loads((run/'report.json').read_text())
    config=json.loads((run/'manifest.json').read_text())['configuration']
    robot=Path(config['robot']);door=Path(config['door'])
    for actual,stored in [(robot,run/'robot-input.xml'),(door/'door.xml',run/'door-input.xml')]:
        if digest(actual)!=digest(stored):raise ValueError('Physical replay input changed')
    trace=json.loads((run/'trace.json').read_text())
    with np.load(run/'trajectory.npz',allow_pickle=False) as z:
        poses=z['qpos'].copy();times=np.array([r['sim_time_s'] for r in trace])
        phases=[r['teacher']['phase'] for r in trace]
        terminal=float(z['terminal_time_s'])
        if terminal>times[-1]:
            poses=np.vstack([poses,z['terminal_qpos']]);times=np.r_[times,terminal];phases.append('terminal recorded state')
    if (len(poses)!=len(times) or not np.isfinite(poses).all() or np.any(np.diff(times)<=0)
            or not times[0]-.01<=a.from_time<times[-1] or not np.isfinite(a.azimuth)):
        raise ValueError('Require finite matched states, increasing timestamps and a valid replay interval')
    a.output.mkdir(parents=True,exist_ok=False)
    sim=DexterousDoorEnv(door,robot,json.loads(robot.with_suffix('.audit.json').read_text()))
    m,d=sim.m,sim.d
    if poses.shape[1]!=m.nq:raise ValueError('Replay model dimensions differ')
    m.vis.global_.offwidth=960;m.vis.global_.offheight=720
    for g in range(m.ngeom):
        name=m.body(m.geom_bodyid[g]).name
        if a.cutaway_walls and m.geom(g).name in ('wall_left','wall_right','wall_header'):
            m.geom_rgba[g,3]=0.
        if name.startswith('robot/'):
            m.geom_matid[g]=-1;m.geom_rgba[g]=[.8,.53,.17,1] if name.startswith(('robot/lh_','robot/rh_')) else [.25,.4,.55,1]
    camera=mujoco.MjvCamera();camera.azimuth=a.azimuth;camera.elevation=-15
    camera.lookat[:]=[-.05,.15,1.];camera.distance=4.5 if a.view=='body' and a.cutaway_walls else 3.5 if a.view=='body' else .6
    palm=None if a.view=='body' else m.site('robot/'+('lh' if a.view=='left-hand' else 'rh')+'_palm_touch').id
    option=mujoco.MjvOption();option.sitegroup[:]=0
    start=max(times[0],a.from_time)
    desired=np.linspace(start,times[-1],3) if a.snapshots_only else np.arange(start,times[-1]+1e-8,.04)
    indices=np.minimum(np.searchsorted(times,desired),len(times)-1)
    from contextlib import nullcontext
    with mujoco.Renderer(m,height=720,width=960) as renderer:
        with (nullcontext(None) if a.snapshots_only else imageio.get_writer(a.output/'replay.mp4',fps=25,codec='libx264',quality=7)) as writer:
            for frame,i in enumerate(indices):
                d.qpos[:]=poses[i];mujoco.mj_kinematics(m,d)
                if palm is not None:camera.lookat[:]=d.site_xpos[palm]+np.array([0,0,-.02])
                renderer.update_scene(d,camera=camera,scene_option=option);sim.hide_sensor_overlays(renderer.scene)
                picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture);draw.rectangle((0,0,960,49),fill='black')
                draw.text((12,7),'RECORDED NATIVE PHYSICS | '+('RUN CHECKS PASSED' if report['passed'] else 'FAILED RUN CHECKS'),fill='white')
                if a.cutaway_walls:draw.text((680,7),'WALL VISUALS HIDDEN',fill='white')
                draw.text((12,28),f'{times[i]:.3f}s | {phases[i]} | privileged native teacher; no Isaac or sensor-policy claim',fill='white')
                if writer is not None:writer.append_data(np.asarray(picture))
                if frame in (0,len(indices)//2,len(indices)-1):picture.save(a.output/f'frame-{frame:04d}.png')
    sim.close()
    receipt=dict(scope=__doc__,run=str(run),passed_run=report['passed'],view=a.view,azimuth=a.azimuth,cutaway_walls=a.cutaway_walls,
        replay_frames=len(indices),actual_selected_times_s=times[indices].tolist(),physics_steps=0,
        source_sha256=digest(__file__),input_sha256={n:digest(run/n) for n in ('manifest.json','report.json','trace.json','trajectory.npz','robot-input.xml','door-input.xml')})
    (a.output/'replay-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(a.output)


if __name__=='__main__':main()

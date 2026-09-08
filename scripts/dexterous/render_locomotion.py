#!/usr/bin/env python3
"""Render recorded native H1 walking states, with outcome and time labels."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from probe_locomotion import make_plant


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--robot',type=Path,required=True);p.add_argument('--trial',type=Path,required=True);args=p.parse_args()
    report=json.loads((args.trial/'report.json').read_text());trace=json.loads((args.trial/'trace.json').read_text());trajectory=np.load(args.trial/'trajectory.npz')
    times=np.array([r['time_s'] for r in trace]);poses=trajectory['qpos']
    if len(times)!=len(poses) or np.any(np.diff(times)<=0):raise ValueError('Need matched monotonic physics timestamps and states')
    sim=make_plant(args.robot);m,d=sim.m,sim.d
    if poses.shape[1]!=m.nq:raise ValueError('Robot and recorded state dimensions differ')
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
    m.vis.headlight.ambient[:]=[.45,.45,.45];m.vis.headlight.diffuse[:]=[.8,.8,.8]
    # Diagnostic surface colors affect only this replay renderer.
    for g in range(m.ngeom):
        if m.geom_bodyid[g]:
            name=m.body(m.geom_bodyid[g]).name;m.geom_matid[g]=-1
            m.geom_rgba[g]=[.85,.62,.24,1.] if name.startswith(('lh_','rh_')) else [.60,.68,.76,1.]
    camera=mujoco.MjvCamera();camera.lookat[:]=[float(np.median(poses[:,0])),float(np.median(poses[:,1])),.9]
    camera.distance=max(4.,float(np.ptp(poses[:,0]))+3.);camera.azimuth=-110;camera.elevation=-15
    options=mujoco.MjvOption();options.sitegroup[:]=0
    tactile={i for i in range(m.nsensor) if m.sensor(i).name.endswith('_touch')}
    frames=np.arange(times[0],times[-1]+1e-8,.04);indices=np.minimum(np.searchsorted(times,frames),len(times)-1)
    with mujoco.Renderer(m,height=720,width=1280) as renderer:
        with imageio.get_writer(args.trial/'body.mp4',fps=25,codec='libx264',quality=8) as writer:
            for f,i in enumerate(indices):
                d.qpos[:]=poses[i];mujoco.mj_kinematics(m,d);renderer.update_scene(d,camera=camera,scene_option=options)
                for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if geom.objtype==mujoco.mjtObj.mjOBJ_UNKNOWN and geom.category==mujoco.mjtCatBit.mjCAT_DECOR and geom.objid in tactile:geom.rgba[3]=0.
                # Visual-only 1 m ground marks provide a fixed distance reference.
                for x in range(-1,8):
                    geom=renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_BOX,np.array([.015,1.5,.0005]),np.array([x,0.,.001]),np.eye(3).ravel(),np.array([.65,.68,.70,1.]))
                    renderer.scene.ngeom+=1
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,1280,38),fill='black')
                outcome='CHECKS PASSED' if report['passed'] else 'FAILED CHECKS'
                draw.text((12,12),f'RECORDED MUJOCO PHYSICS | H1 + dual Shadow hands | {outcome} | t={times[i]:.2f} s | floor marks: 1 m',fill='white')
                writer.append_data(np.asarray(frame))
                if f in (0,len(indices)//2,len(indices)-1):frame.save(args.trial/f'body-{f:04d}.png')
    print(args.trial/'body.mp4')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Render recorded native acquisition states, with explicit outcome/time labels."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from doorbench.dexterous.environment import DexterousDoorEnv


def verify_recorded_xml(trial,robot,door):
    """Version labels alone cannot authorize a cross-machine reconstruction."""
    manifest=trial/'manifest.json'
    if manifest.exists():
        inputs=json.loads(manifest.read_text())['inputs']
        expected=((robot,inputs['robot']['sha256']),(door/'door.xml',inputs['door']['door.xml']))
    else:
        records=(trial/'robot-input.xml',trial/'door-input.xml')
        if not all(p.exists() for p in records):raise ValueError('Recorded robot and door XML identities are required')
        expected=tuple((actual,hashlib.sha256(record.read_bytes()).hexdigest()) for record,actual in zip(records,(robot,door/'door.xml')))
    for path,digest in expected:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Render on the original host with the exact recorded input XML: '+str(path))
    return {str(path):digest for path,digest in expected}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','trial'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--view',choices=('hand','left-hand','body'),default='hand')
    p.add_argument('--at',type=float,action='append',help='Render these actual recorded times only, without advancing physics')
    p.add_argument('--azimuth',type=float,default=150.)
    p.add_argument('--elevation',type=float)
    p.add_argument('--distance',type=float)
    p.add_argument('--snapshots-only',action='store_true',help='Render beginning, middle and end from recorded physics without encoding a video')
    args=p.parse_args()
    verified_inputs=verify_recorded_xml(args.trial,args.robot,args.door)
    report=json.loads((args.trial/'report.json').read_text())
    trace=json.loads((args.trial/'trace.json').read_text())
    trajectory=np.load(args.trial/'trajectory.npz')
    times=np.asarray([row['sim_time_s'] for row in trace])
    if len(times)!=len(trajectory['qpos']) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Recorded states need a matching monotonic physics clock')
    for recorded,actual in ((args.trial/'robot-input.xml',args.robot),(args.trial/'door-input.xml',args.door/'door.xml')):
        if recorded.exists() and hashlib.sha256(recorded.read_bytes()).digest()!=hashlib.sha256(actual.read_bytes()).digest():
            raise ValueError('Rendering must use the exact recorded robot and door inputs: '+str(actual))
    sim=DexterousDoorEnv(args.door,args.robot,json.loads(args.robot.with_suffix('.audit.json').read_text()))
    sim.reset(randomize=False,images=False)
    if trajectory['qpos'].shape[1]!=sim.m.nq:
        raise ValueError('Recording and regenerated model dimensions differ')
    # Diagnostic colors change rendering only: highlight the thumb and operator.
    for g in range(sim.m.ngeom):
        name=sim.m.body(sim.m.geom_bodyid[g]).name
        if name.startswith(('robot/rh_','robot/lh_')):
            sim.m.geom_matid[g]=-1
            sim.m.geom_rgba[g]=[.08,.5,.8,1.] if name.startswith(('robot/rh_th','robot/lh_th')) else [.35,.4,.45,1.]
        if sim.m.geom(g).name.startswith('leaf_handle'):
            sim.m.geom_matid[g]=-1;sim.m.geom_rgba[g]=[.65,.38,.08,1.]
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[.26,-.10,.93] if args.view=='hand' else [.05,-.15,.9]
    camera.distance=args.distance if args.distance is not None else (.48 if args.view!='body' else 3.1)
    camera.azimuth=args.azimuth;camera.elevation=args.elevation if args.elevation is not None else (-25 if args.view!='body' else -12)
    options=mujoco.MjvOption();options.sitegroup[:]=0
    if args.at and any(not np.isfinite(t) or t<times[0] or t>times[-1] for t in args.at):raise ValueError('Snapshot times must be inside the recorded episode')
    frame_times=np.asarray(args.at) if args.at else np.arange(times[0],times[-1]+1e-8,.04)
    indices=np.minimum(np.searchsorted(times,frame_times),len(times)-1)
    out=args.trial/f'{args.view}.mp4'
    outputs=[]
    from contextlib import nullcontext
    try:
        with mujoco.Renderer(sim.m,height=720,width=960) as renderer:
            with (nullcontext(None) if args.snapshots_only or args.at else imageio.get_writer(out,fps=25,codec='libx264',quality=8)) as writer:
                for frame,i in enumerate(indices):
                    if args.snapshots_only and not args.at and frame not in (0,len(indices)//2,len(indices)-1):continue
                    sim.d.qpos[:]=trajectory['qpos'][i]
                    mujoco.mj_kinematics(sim.m,sim.d)
                    if args.view=='left-hand':camera.lookat[:]=sim.d.site_xpos[sim.m.site('robot/lh_palm_touch').id]
                    renderer.update_scene(sim.d,camera=camera,scene_option=options)
                    sim.hide_sensor_overlays(renderer.scene)
                    image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                    draw.rectangle((0,0,960,35),fill='black')
                    outcome='PROBE CHECKS PASSED' if report['passed'] else 'FAILED TASK CHECKS'
                    draw.text((12,12),f'RECORDED MUJOCO PHYSICS | {outcome} | t={times[i]:.2f}s | thumb: blue',fill='white')
                    if writer:writer.append_data(np.asarray(image))
                    if args.at or frame in (0,len(indices)//2,len(indices)-1):
                        suffix=f'-t{times[i]:.3f}-az{args.azimuth:g}-el{camera.elevation:g}' if args.at else f'-az{args.azimuth:g}-el{camera.elevation:g}' if args.snapshots_only else ''
                        snapshot=args.trial/f'{args.view}{suffix}-{frame:04d}.png'
                        image.save(snapshot);outputs.append(str(snapshot))
        if not args.snapshots_only and not args.at:outputs.insert(0,str(out))
        receipt=dict(artifacts=outputs,source='recorded physical states; no physics advancement',verified_xml_sha256=verified_inputs,
            trajectory_sha256=hashlib.sha256((args.trial/'trajectory.npz').read_bytes()).hexdigest())
        (args.trial/f'{args.view}-render-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))
    finally:
        sim.close()


if __name__=='__main__':
    main()

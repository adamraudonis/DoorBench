#!/usr/bin/env python3
"""Inspect declared robot cameras on recorded native states; no new physics claim."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from doorbench.dexterous.environment import DexterousDoorEnv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('robot','door','trial','layout','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Use a new camera-review output directory')
    layout=json.loads(a.layout.read_text())
    if layout['robot_xml_sha256']!=hashlib.sha256(a.robot.read_bytes()).hexdigest():raise ValueError('Robot calibration mismatch')
    s=DexterousDoorEnv(a.door,a.robot,json.loads(a.robot.with_suffix('.audit.json').read_text()))
    s.reset(randomize=False,images=False)
    trajectory=np.load(a.trial/'trajectory.npz')['qpos']
    if trajectory.shape[1]!=s.m.nq:raise ValueError('Recorded model dimensions disagree')
    cameras={}
    for camera in layout['cameras']:
        index=s.m.camera('robot/'+camera['name']).id
        body=s.m.body(int(s.m.cam_bodyid[index])).name.removeprefix('robot/')
        if body!=camera['body_name']:raise ValueError('Camera mounted on a different body')
        s.m.cam_pos[index]=camera['position_body_m'];s.m.cam_quat[index]=camera['quaternion_wxyz_body']
        s.m.cam_fovy[index]=camera['fovy_degrees'];cameras[camera['name']]=index
    a.output.mkdir(parents=True)
    (a.output/'layout.json').write_bytes(a.layout.read_bytes())
    (a.output/'scope.json').write_text(json.dumps(dict(scope=__doc__,trial=str(a.trial),camera_motion='Fixed body-local calibration only'))+'\n')
    options=mujoco.MjvOption();options.sitegroup[:]=0
    try:
        with mujoco.Renderer(s.m,height=512,width=512) as renderer:
            for frame in sorted({0,len(trajectory)//2,len(trajectory)-1}):
                s.d.qpos[:]=trajectory[frame];mujoco.mj_kinematics(s.m,s.d);mujoco.mj_camlight(s.m,s.d)
                for name,index in cameras.items():
                    renderer.update_scene(s.d,camera=index,scene_option=options)
                    s.hide_sensor_overlays(renderer.scene)
                    Image.fromarray(renderer.render()).save(a.output/f'{name}-{frame:04d}.png')
    finally:s.close()
    print(a.output)


if __name__=='__main__':main()

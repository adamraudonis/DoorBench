#!/usr/bin/env python3
"""Inspect recorded Isaac joint states using matching native geometry, without stepping.

This is a diagnostic geometry replay, not an Isaac render or a new physics run.
Contact forces and qualification remain those of the archived original trial.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from doorbench.dexterous.environment import DexterousDoorEnv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('trial', 'robot', 'door', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--at', type=float, action='append', required=True)
    p.add_argument('--preparation-launch', type=Path, help='Qualified source-design binding for relocated sensor-acquisition geometry')
    p.add_argument('--azimuth', type=float, action='append', default=None)
    p.add_argument('--distance', type=float, default=.4)
    p.add_argument('--elevation', type=float, default=-20.)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a new output directory')
    config = json.loads((a.trial/'configuration.json').read_text())
    hashes = json.loads((a.trial/'provenance.json').read_text())['files']
    source_admission = None
    if config['args'].get('native_robot') and config['args'].get('native_door'):
        for name, path in (('native_robot', a.robot), ('native_door', a.door/'door.xml')):
            if sha(path) != hashes[config['args'][name]]:
                raise ValueError('Geometry replay requires exact original source hash: ' + name)
    else:
        if a.preparation_launch is None:
            raise ValueError('Sensor acquisition replay requires its qualified preparation source binding')
        from doorbench.dexterous.robot_design_identity import verify_robot_design_identity
        launch = json.loads(a.preparation_launch.read_text())
        if launch['native_preparation_passed'] is not True:
            raise ValueError('Require passed original preparation')
        for name in ('robot_usd','door_usd'):
            source = config['args'][name]
            if launch['input_sha256'][source] != hashes[source]:
                raise ValueError('Preparation and actual USD source identities differ')
        robot_source = config['args']['sensor_balance_robot']
        door_source = str(Path(config['args']['door_usd']).with_suffix('.xml'))
        source_admission = dict(preparation_sha256=sha(a.preparation_launch),
            robot=verify_robot_design_identity(a.robot,launch['source_design'][robot_source]),
            door=verify_robot_design_identity(a.door/'door.xml',launch['source_design'][door_source]))
    report = json.loads((a.trial/'report.json').read_text())
    trajectory = np.load(a.trial/'acquisition-physics.npz')
    measurements = None
    if (a.trial/'full-opening-steps.json.gz').exists():
        with gzip.open(a.trial/'full-opening-steps.json.gz', 'rt') as stream:
            measurements = json.load(stream)
    else:
        report = json.loads((a.trial/'balance-report.json').read_text())
        if not all(abs(t-report['duration_s']) < 1e-8 for t in a.at):
            raise ValueError('Balance archive provides its endpoint FK reconstruction only at the terminal state')
    times = trajectory['time_s']
    if (not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or
            not len(times) or not np.isfinite(a.at).all() or
            min(a.at) < times[0] or max(a.at) > times[-1]):
        raise ValueError('Snapshot times must lie inside the monotonic recorded episode')
    if (trajectory['joints'].shape != (len(times), len(config['robot_joint_names'])) or
            trajectory['door'].shape != (len(times), len(config['door_joint_names'])) or
            trajectory['root'].shape != (len(times), 13)):
        raise ValueError('Recorded model dimensions disagree')
    sim = DexterousDoorEnv(a.door, a.robot, json.loads(a.robot.with_suffix('.audit.json').read_text()))
    m, d = sim.m, sim.d
    qa = [m.jnt_qposadr[m.joint('robot/'+n).id] for n in config['robot_joint_names']]
    da = [m.jnt_qposadr[m.joint(n).id] for n in config['door_joint_names']]
    if len(set(qa)) != 69 or len(set(da)) != len(da):
        raise ValueError('Every original robot and mechanism coordinate must map once')
    # These colors affect the replay only. Contact/anatomy checks are never recolored.
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if body.startswith('robot/rh_'):
            m.geom_matid[g] = -1
            m.geom_rgba[g] = [.08,.5,.8,1.] if body.startswith('robot/rh_th') else [.6,.65,.7,1.]
        elif m.geom(g).name.startswith('leaf_handle'):
            m.geom_matid[g] = -1
            m.geom_rgba[g] = [.8,.5,.12,1.]
    a.output.mkdir(parents=True)
    receipt = dict(scope=__doc__, passed_original_task=report['passed'], physics_steps=0,
        robot_sha256=sha(a.robot), door_sha256=sha(a.door/'door.xml'),
        trajectory_sha256=sha(a.trial/'acquisition-physics.npz'), source_admission=source_admission, frames=[])
    camera = mujoco.MjvCamera()
    camera.distance = a.distance
    camera.elevation = a.elevation
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    try:
        with mujoco.Renderer(m, height=720, width=960) as renderer:
            for requested in a.at:
                i = int(np.argmin(np.abs(times-requested)))
                root, joints, door = (trajectory[k][i] for k in ('root', 'joints', 'door'))
                if not np.isfinite(np.r_[root, joints, door]).all():
                    raise ValueError('Cannot render nonfinite state')
                d.qpos[sim.root_qadr:sim.root_qadr+7] = root[:7]
                d.qpos[qa] = joints
                d.qpos[da] = door
                # Only forward kinematics; no contact solve or integration.
                mujoco.mj_kinematics(m, d)
                palm = d.site_xpos[m.site('robot/rh_palm_touch').id]
                rotation_error = None
                if measurements is not None:
                    observed = measurements[i]
                    if abs(observed['time_s']-times[i]) > 1e-8:
                        raise ValueError('Replay and original geometry clocks differ')
                    measured_pose = np.asarray(observed['geometry']['right_palm_pose'])
                    measured_rotation = np.empty(9)
                    mujoco.mju_quat2Mat(measured_rotation, measured_pose[3:])
                    rotation_error = float(np.max(np.abs(d.site_xmat[m.site('robot/rh_palm_touch').id]-measured_rotation)))
                else:
                    if i != len(times)-1 or abs(times[i]-report['duration_s']) > 1e-8:
                        raise ValueError('Require the exact terminal balance state')
                    measured_pose = np.asarray(report['actual_palm_endpoint_world_m'])
                patch_mapping = []
                if measurements is None:
                    # Endpoint palm in this report is FK-derived, not a measured
                    # PhysX body pose. Independently bind loaded digit geometry
                    # using local/world patch pairs from actual RigidBodyView
                    # transforms retained by the original contact auditor.
                    for patch in report['final_pad_grasp']['contacts']:
                        body = m.body('robot/'+patch['body'].rsplit('/',1)[-1]).id
                        point = d.xpos[body]+d.xmat[body].reshape(3,3)@np.asarray(patch['body_position_m'])
                        residual = float(np.linalg.norm(point-np.asarray(patch['position'])))
                        if residual > 2e-6:
                            raise ValueError('Replayed digit differs from the actual PhysX contact-body frame')
                        patch_mapping.append(dict(digit=patch['digit'],position_error_m=residual))
                error = float(np.linalg.norm(palm-measured_pose[:3]))
                if error > 1e-5 or (rotation_error is not None and rotation_error > 1e-4):
                    raise ValueError('Reconstructed hand disagrees with archived Isaac geometry')
                handle = d.xpos[m.body('leaf_handle').id]
                camera.lookat[:] = .6*palm + .4*handle
                for azimuth in a.azimuth or [90.,150.,230.]:
                    camera.azimuth = azimuth
                    renderer.update_scene(d, camera=camera, scene_option=options)
                    sim.hide_sensor_overlays(renderer.scene)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0,0,960,58), fill='black')
                    draw.text((12,10), 'ACTUAL ISAAC STATE / MUJOCO GEOMETRY REPLAY / NO PHYSICS ADVANCED', fill='white')
                    status = 'ORIGINAL CHECKS PASSED' if report['passed'] else 'FAILED ORIGINAL TASK'
                    draw.text((12,33), f'{status} | t={times[i]:.3f}s | thumb: blue | view={azimuth:g} deg', fill='white')
                    path = a.output/f'hand-t{times[i]:.3f}-az{azimuth:g}.png'
                    frame.save(path)
                    receipt['frames'].append(dict(path=path.name, sha256=sha(path),
                        requested_time_s=requested, recorded_time_s=float(times[i]), index=i,
                        archived_palm_position_error_m=error, archived_palm_rotation_matrix_error=rotation_error,
                        palm_reference_kind='measured body pose' if measurements is not None else 'archived FK-derived endpoint; not an independent body-pose measurement',
                        actual_physx_contact_frame_mapping=patch_mapping,
                        azimuth_deg=azimuth, elevation_deg=a.elevation, distance_m=a.distance))
    finally:
        sim.close()
    (a.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), frames=len(receipt['frames']), physics_steps=0)))


if __name__ == '__main__':
    main()

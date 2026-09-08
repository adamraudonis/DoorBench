#!/usr/bin/env python3
"""Live PhysX sensor fixture; not a robot policy or door-opening benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), *[Path(__file__).resolve().parents[2]/'doorbench/dexterous'/name
        for name in ('sensor_contract.py', 'isaac_sensors.py')]]
    (out/'manifest.json').write_text(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
        scope=__doc__, device=args.device, python=sys.version,
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}), indent=2)+'\n')
    launcher = AppLauncher(args)
    app = launcher.app
    failed = False
    try:
        import numpy as np
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, Imu, ImuCfg
        from doorbench.dexterous.sensor_contract import ActorObservationBuilder, AngularTaxelGrid, SENSOR_KEYS, SensorEffects
        from doorbench.dexterous.isaac_sensors import (PhysXTaxelAdapter, TactileMount,
            all_scene_contact_paths, enable_tactile_reporting, enqueue_camera)
        dt = .002
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, render_interval=20))
        sim.carb_settings.set_bool('/physics/disableContactProcessing', False)
        if getattr(sim, '_app_control_on_stop_handle', None) is not None:
            sim._app_control_on_stop_handle.unsubscribe()
            sim._app_control_on_stop_handle = None
        ground = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.6))
        ground.func('/World/physical_surface', ground)
        light = sim_utils.DomeLightCfg(intensity=2000.)
        light.func('/World/light', light)
        def cube(path, pos, size, mass, color):
            return RigidObject(RigidObjectCfg(prim_path=path,
                spawn=sim_utils.CuboidCfg(size=size, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.6),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos)))
        static_probe = cube('/World/probe_a', (-.3, 0., .3), (.1, .1, .1), 1., (.8, .1, .1))
        dynamic_base = cube('/World/base_b', (.3, 0., .05), (.3, .3, .1), 10., (.15, .2, .7))
        dynamic_probe = cube('/World/probe_b', (.3, 0., .35), (.1, .1, .1), 1., (.1, .8, .1))
        mounts = [TactileMount(path, (0, 0, 0), (0, 0, 0, 1), AngularTaxelGrid())
                  for path in ('/World/probe_a', '/World/probe_b')]
        enable_tactile_reporting(sim.stage, mounts)
        imu = Imu(ImuCfg(prim_path='/World/probe_a', update_period=0., debug_vis=False,
                        gravity_bias=(0., 0., 9.81)))
        camera = Camera(CameraCfg(prim_path='/World/policy_camera', update_period=0., height=128, width=128,
            data_types=['rgb'], spawn=sim_utils.PinholeCameraCfg(focal_length=24., focus_distance=2.,
                                                              horizontal_aperture=30., clipping_range=(.01, 10.))))
        sim.reset()
        camera.set_world_poses_from_view(eyes=torch.tensor([[1., -1.4, 1.0]], device=args.device),
                                        targets=torch.tensor([[0., 0., .1]], device=args.device))
        sim.carb_settings.set_bool('/physics/disableContactProcessing', False)
        adapter = PhysXTaxelAdapter(sim.physics_sim_view, sim.stage, mounts, capacity=8192)
        # Fixture has no joints/actions. The one reserved slot is explicitly a
        # zero fixture channel, never a robot capability or training observation.
        packet_builder = ActorObservationBuilder(joint_count=1, action_count=1,
            tactile_dimension=adapter.dimension, image_shape=(128, 128, 3),
            effects={'rgb_left':SensorEffects(255, delay_s=.02, max_age_s=.12),
                     'rgb_right':SensorEffects(255, delay_s=.02, max_age_s=.12)})
        rows, frames, summaries = [], [], []
        clock_origin = float(sim.current_time)
        for step in range(1000):
            if step == 600:
                static_probe.write_root_velocity_to_sim(torch.tensor([[.25, 0., 0., 0., 0., 0.]], device=args.device))
            sim.step(render=False)
            for body in (static_probe, dynamic_probe, dynamic_base):
                body.update(dt)
            imu.update(dt)
            elapsed = (step+1)*dt
            tactile = adapter.read(physics_dt=dt)
            packet_builder.push('tactile', tactile, capture_s=elapsed)
            packet_builder.push('imu_gyro', imu.data.ang_vel_b[0].cpu().numpy(), capture_s=elapsed)
            packet_builder.push('imu_accelerometer', imu.data.lin_acc_b[0].cpu().numpy(), capture_s=elapsed)
            for key in ('joint_position', 'joint_velocity'):
                packet_builder.push(key, [0.], capture_s=elapsed)
            if step % 20 == 0:
                sim.render()
                camera.update(dt*20)
                pixels = camera.data.output['rgb'][0].cpu().numpy()[..., :3].copy()
                frames.append(pixels)
                for key in ('rgb_left', 'rgb_right'):
                    enqueue_camera(packet_builder, key, camera.data, capture_s=elapsed, available_s=elapsed)
            packet = packet_builder.observe(now_s=elapsed, previous_action=[0.])
            normal_matrix = adapter.contacts.get_contact_force_matrix(dt).cpu().numpy().sum(axis=1)
            net = adapter.contacts.get_net_contact_forces(dt).cpu().numpy()
            gyro = imu.data.ang_vel_b[0].cpu().numpy()
            accel = imu.data.lin_acc_b[0].cpu().numpy()
            rows.append(dict(time_s=elapsed, sim_time_s=float(sim.current_time)-clock_origin,
                tactile=tactile.tolist(), net_force_N=net.tolist(), summed_pair_force_N=normal_matrix.tolist(),
                imu_gyro=gyro.tolist(), imu_accelerometer=accel.tolist(),
                probe_position=static_probe.data.root_pos_w[0].cpu().tolist(),
                sensor_time_s=packet['sensor_time_s'].tolist(), sensor_valid=packet['sensor_valid'].tolist()))
            if step % 100 == 0:
                progress=dict(phase='sliding' if step >= 600 else 'falling_then_settling', completed_steps=step+1, total_steps=1000)
                (out/'progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress), flush=True)
        (out/'trace.json').write_text(json.dumps(rows)+'\n')
        np.savez_compressed(out/'rgb.npz', frames=np.stack(frames), acquisition_times_s=np.arange(0, 1000, 20)*dt+dt)
        from PIL import Image
        for index in (0, 10, 25, 40, 49):
            Image.fromarray(frames[index]).save(out/f'rgb-{index:03}.png')
        steady = rows[450:550]
        steady_forces = np.array([row['tactile'] for row in steady]).reshape(-1, 2, 3, 8).sum(axis=-1)
        accel_steady = np.median([row['imu_accelerometer'] for row in steady], axis=0)
        freefall_accel = np.array([row['imu_accelerometer'] for row in rows[10:70]])
        shear = np.array([row['tactile'] for row in rows[600:620]]).reshape(-1, 2, 3, 8).sum(axis=-1)
        clock_error = max(abs(row['sim_time_s']-row['time_s']) for row in rows)
        camera_index = SENSOR_KEYS.index('rgb_left')
        camera_ages = [row['time_s']-row['sensor_time_s'][camera_index] for row in rows if row['sensor_valid'][camera_index]]
        checks = dict(
            finite_taxels=bool(np.isfinite([row['tactile'] for row in rows]).all()),
            static_support_sign_and_weight=bool(abs(np.median(steady_forces[:,0,0])-9.81) < 1.0),
            dynamic_support_sign_and_weight=bool(abs(np.median(steady_forces[:,1,0])-9.81) < 1.0),
            shear_opposes_sliding=bool(np.min(shear[:,0,1]) < -.1),
            stationary_imu=bool(np.linalg.norm(accel_steady-[0,0,9.81]) < .5),
            freefall_imu=bool(np.median(np.linalg.norm(freefall_accel, axis=1)) < .5),
            physics_clock=bool(clock_error < 1e-5),
            rgb_has_motion=bool(np.abs(frames[0].astype(float)-frames[25]).mean() > .1),
            rgb_delivery_delay=bool(camera_ages and min(camera_ages) >= .02-1e-8 and max(camera_ages) < .065),
            rgb_unavailable_before_delay=bool(not any(row['sensor_valid'][camera_index] for row in rows[:10])),
        )
        report=dict(scope=__doc__, completed_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
            passed=all(checks.values()), device=args.device, steps=len(rows), physics_dt=dt,
            steady_local_force_zxy_N=np.median(steady_forces, axis=0).tolist(),
            steady_imu_specific_force=accel_steady.tolist(), min_shear_local_x_N=float(np.min(shear[:,0,1])),
            max_clock_error_s=clock_error, camera_age_range_s=[min(camera_ages),max(camera_ages)] if camera_ages else None,
            participants=list(all_scene_contact_paths(sim.stage)),
            limitation='Standalone two-pad fixture only. Both RGB channels duplicate one fixture camera. H1 mounts, robot cameras, rotated fixtures, self-contact and batching remain unverified.')
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        failed = not report['passed']
    except BaseException:
        failed = True
        (out/'error.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        app.close()
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

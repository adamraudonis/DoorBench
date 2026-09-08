"""Opt-in physical sensor recording alongside the unchanged privileged teacher.

The recorder does not command motors, camera targets or task mechanisms. Sensor
mounts are fixed to robot bodies using the exported native calibration. Recording
sensor packets does not make the teacher a sensor-only policy.
"""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
import numpy as np

from .sensor_contract import ActorObservationBuilder, SENSOR_KEYS
from .isaac_sensors import (PhysXTaxelAdapter, mounts_from_layout, enable_tactile_reporting,
                           enqueue_robot_sensors, enqueue_camera)


def _atomic_npz(path, **arrays):
    path=Path(path);temporary=path.with_name(path.name+'.writing')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary,path)


def _atomic_json(path, value):
    path=Path(path);temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temporary,path)


class IsaacSensorRecorder:
    def __init__(self, stage, layout_path, output, *, robot_root='/World/H1', control_source='privileged_teacher'):
        from pxr import Usd, UsdPhysics
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg, Imu, ImuCfg
        if control_source not in ('privileged_teacher','sensor_actor'):raise ValueError('Explicit control source required')
        self.control_source=control_source
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.layout = json.loads(Path(layout_path).read_text())
        by_name = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                name = prim.GetName()
                if name in by_name:
                    raise ValueError('Ambiguous imported robot body name')
                by_name[name] = str(prim.GetPath())
        self.mounts = mounts_from_layout(self.layout, by_name)
        enable_tactile_reporting(stage, self.mounts)
        imu = self.layout['imu']
        if imu is None:
            raise ValueError('Export a physical native IMU site')
        self.imu = Imu(ImuCfg(prim_path=by_name[imu['body_name']], update_period=0., debug_vis=False,
            gravity_bias=(0., 0., 9.81), offset=ImuCfg.OffsetCfg(pos=tuple(imu['position_body_m']),
                                                              rot=tuple(imu['quaternion_wxyz_body']))))
        self.cameras = {}
        for calibration in self.layout['cameras']:
            name = calibration['name']
            if name not in ('left_eye_camera', 'right_eye_camera'):
                raise ValueError('Unexpected policy camera')
            key = 'rgb_left' if name == 'left_eye_camera' else 'rgb_right'
            focal_length = 24.
            aperture = 2*focal_length*np.tan(np.deg2rad(calibration['fovy_degrees'])/2)
            self.cameras[key] = Camera(CameraCfg(prim_path=by_name[calibration['body_name']]+'/Sensor_'+name,
                update_period=0., height=128, width=128, data_types=['rgb'],
                offset=CameraCfg.OffsetCfg(pos=tuple(calibration['position_body_m']),
                    rot=tuple(calibration['quaternion_wxyz_body']), convention='opengl'),
                spawn=sim_utils.PinholeCameraCfg(focal_length=focal_length, horizontal_aperture=float(aperture),
                                               vertical_aperture=float(aperture), clipping_range=(.01, 100.))))
        if set(self.cameras) != {'rgb_left', 'rgb_right'}:
            raise ValueError('Both physical eye cameras are required for this fixture')
        self.builder = ActorObservationBuilder(joint_count=len(self.layout['joint_order']),
            action_count=len(self.layout['action_order']), tactile_dimension=self.layout['tactile_dimension'])
        self.samples = {key:[] for key in (*SENSOR_KEYS, 'previous_action', 'sensor_time_s', 'sensor_valid') if not key.startswith('rgb_')}
        self.frames = {key:[] for key in self.cameras}
        self.frame_times = []
        self.times = []
        self.adapter = None
        self.max_contact_samples = 0
        (self.output/'layout.json').write_text(json.dumps(self.layout, indent=2)+'\n')
        (self.output/'scope.json').write_text(json.dumps(dict(
            scope='Physical robot sensors for closed-loop sensor-only actor' if self.control_source=='sensor_actor' else 'Physical sensor recording beside privileged teacher; NOT sensor-only control',
            previous_action='Actual bounded motor force normalized by the native motor force range',
            calibration='Fixed native robot eye cameras and IMU; no task-targeted camera routing',
            body_paths_by_name=by_name), indent=2)+'\n')

    def initialize(self, physics_view, robot_joint_names):
        self.adapter = PhysXTaxelAdapter(physics_view, self._stage(), self.mounts, capacity=65536)
        self.joint_indices = [robot_joint_names.index(name) for name in self.layout['joint_order']]
        self.builder.reset(seed=0)
        self.imu.reset()

    @staticmethod
    def _stage():
        import omni.usd
        return omni.usd.get_context().get_stage()

    def update(self, *, robot_data, dt, time_s, previous_action, rendered):
        self.imu.update(dt)
        tactile = self.adapter.read(physics_dt=dt)
        enqueue_robot_sensors(self.builder, robot_data, self.imu.data, tactile,
                              joint_indices=self.joint_indices, capture_s=time_s)
        if rendered:
            for key, camera in self.cameras.items():
                camera.update(dt*20)
                enqueue_camera(self.builder, key, camera.data, capture_s=time_s, available_s=time_s)
                self.frames[key].append(camera.data.output['rgb'][0].cpu().numpy()[..., :3].copy())
            self.frame_times.append(time_s)
        packet = self.builder.observe(now_s=time_s, previous_action=previous_action)
        for key in self.samples:
            self.samples[key].append(packet[key])
        self.times.append(time_s)
        if len(self.times) % 250 == 0:
            (self.output/'progress.json').write_text(json.dumps(dict(samples=len(self.times),
                time_s=time_s, frames=len(self.frame_times), finite=True))+'\n')

    def record_initial_decision(self, packet, motor_forces):
        """Preserve the actor's causal cold start before the first physics step."""
        path=self.output/'actor-initial-decision.npz'
        if path.exists():raise ValueError('Initial actor decision already recorded')
        _atomic_npz(path, **packet, motor_forces=np.asarray(motor_forces), time_s=np.asarray(0.))

    def finish(self, *, complete=True):
        from PIL import Image
        arrays = {key:np.asarray(values) for key, values in self.samples.items()}
        arrays['time_s'] = np.asarray(self.times)
        _atomic_npz(self.output/'actor-sensors.npz', **arrays)
        _atomic_npz(self.output/'actor-rgb.npz', time_s=np.asarray(self.frame_times),
                            **{key:np.asarray(value) for key,value in self.frames.items()})
        for key, frames in self.frames.items():
            for index in sorted({0, len(frames)//4, len(frames)//2, len(frames)-1}):
                if 0 <= index < len(frames):
                    Image.fromarray(frames[index]).save(self.output/f'{key}-{index:04}.png')
        offsets = np.cumsum([0]+[row['dimension'] for row in self.layout['sensors']])
        peaks = {}
        for i, row in enumerate(self.layout['sensors']):
            values = arrays['tactile'][:, offsets[i]:offsets[i+1]]
            peaks[row['name']] = float(np.max(np.abs(values))) if values.size else 0.
        camera_age = []
        for key in self.cameras:
            index = SENSOR_KEYS.index(key)
            valid = arrays['sensor_valid'][:, index]
            camera_age.extend((arrays['time_s'][valid]-arrays['sensor_time_s'][valid,index]).tolist())
        report = dict(scope='Sensor-interface capture; see separate actor/teacher task report',control_source=self.control_source, capture_complete=bool(complete), samples=len(self.times),
            frames_per_eye=len(self.frame_times), tactile_dimension=self.layout['tactile_dimension'],
            finite=bool(all(np.isfinite(x).all() for x in arrays.values())),
            all_sensor_streams_present=bool(arrays['sensor_valid'].all()),
            camera_age_range_s=[min(camera_age),max(camera_age)] if camera_age else None,
            tactile_peak_abs_force_by_mount_N=peaks,
            rgb_mean_pixel_change={key:float(np.abs(frames[-1].astype(float)-frames[0]).mean()) if frames else None for key,frames in self.frames.items()},
            limitations=['Sensor stream checks alone do not establish task success', 'Robot camera usefulness requires visual inspection',
                         'Exact camera extrinsics and cross-engine mount parity require independent audit'])
        report['numeric_file_sha256']={name:hashlib.sha256((self.output/name).read_bytes()).hexdigest() for name in ('actor-sensors.npz','actor-rgb.npz')}
        _atomic_json(self.output/'report.json',report)
        return report

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
from .pose_gyro import OwnImuPoseGyroscope, PROFILE as POSE_GYRO_PROFILE, DEFAULT_PROFILE as DEFAULT_GYRO_PROFILE
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


def bind_gyro_profile(layout, profile):
    """Bind changed sensor semantics without rewriting historical default layouts."""
    if profile not in (DEFAULT_GYRO_PROFILE, POSE_GYRO_PROFILE):
        raise ValueError('Unknown explicit gyroscope profile')
    result=json.loads(json.dumps(layout))
    if not isinstance(result.get('imu'),dict):
        raise ValueError('A calibrated physical robot IMU is required')
    declared=result['imu'].get('gyro_profile')
    if declared is not None and declared != profile:
        raise ValueError('Requested gyroscope differs from the declared sensor layout')
    # Absent means the historical backend profile. Preserve those checkpoint
    # fingerprints; every new alternative is explicitly part of its layout.
    if profile != DEFAULT_GYRO_PROFILE:
        result['imu']['gyro_profile']=profile
    return result


def synchronize_and_reset_pose_gyro(physics_view,producer):
    """Refresh derived GPU link poses after reset writes, without a physics step.

    PhysX107.3 requires this after set_dof_positions before reading link poses.
    Otherwise the first apparent interval can include stale pre-reset geometry.
    This setup operation never changes the requested root/joint coordinates.
    """
    physics_view.update_articulations_kinematic()
    producer.reset_episode(now_s=0.)
    return dict(api='SimulationView.update_articulations_kinematic',
        phase='after physical reset state writes, before first recorded interval',
        integration_steps_requested=0,root_or_joint_state_writes_requested=0)


class IsaacSensorRecorder:
    def __init__(self, stage, layout_path, output, *, robot_root='/World/H1', control_source='privileged_teacher', gyro_profile=DEFAULT_GYRO_PROFILE):
        from pxr import Usd, UsdPhysics
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg, Imu, ImuCfg
        if control_source not in ('privileged_teacher','sensor_actor'):raise ValueError('Explicit control source required')
        self.control_source=control_source
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        layout_bytes=Path(layout_path).read_bytes()
        self.layout = bind_gyro_profile(json.loads(layout_bytes),gyro_profile)
        self.gyro_profile=gyro_profile
        self.pose_gyro=None
        self.gyro_reset_synchronization=None
        self.robot_root=robot_root
        self.input_layout_sha256=hashlib.sha256(layout_bytes).hexdigest()
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
        self.robot_body_paths=tuple(by_name.values())
        self.imu_body_path=by_name[imu['body_name']]
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
            gyro_profile=self.gyro_profile,
            input_layout_sha256=self.input_layout_sha256,
            effective_layout_sha256=hashlib.sha256((self.output/'layout.json').read_bytes()).hexdigest(),
            body_paths_by_name=by_name), indent=2)+'\n')

    def initialize(self, physics_view, robot_joint_names):
        self.adapter = PhysXTaxelAdapter(physics_view, self._stage(), self.mounts, capacity=65536)
        self.joint_indices = [robot_joint_names.index(name) for name in self.layout['joint_order']]
        self.builder.reset(seed=0)
        self.imu.reset()
        if self.gyro_profile==POSE_GYRO_PROFILE:
            self.pose_gyro=OwnImuPoseGyroscope(self.imu._view,
                expected_body_path=self.imu_body_path,robot_body_paths=self.robot_body_paths,
                robot_root_path=self.robot_root,
                imu_quaternion_wxyz_body=self.layout['imu']['quaternion_wxyz_body'])
            self.gyro_reset_synchronization=synchronize_and_reset_pose_gyro(physics_view,self.pose_gyro)
        self._write_gyro_receipt()

    def _write_gyro_receipt(self):
        receipt=self.pose_gyro.receipt() if self.pose_gyro is not None else dict(
            schema='doorbench.imu-gyro-producer.v1',profile=self.gyro_profile,
            source='Isaac Lab calibrated own-body angular velocity',
            actor_fields=['imu_gyro'],actor_receives_orientation=False,
            accelerometer='Unchanged Isaac Lab backend velocity-derived specific force')
        if self.pose_gyro is not None:
            receipt['reset_synchronization']=getattr(self,'gyro_reset_synchronization',None)
            path=self.output/'gyro-producer-evidence.npz'
            _atomic_npz(path,**self.pose_gyro.evidence())
            receipt['evaluator_evidence_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            receipt['recorded_actor_packets']=len(self.times)
            receipt['interval_count_matches_packets']=self.pose_gyro.samples==len(self.times)
        _atomic_json(self.output/'gyro-producer.json',receipt)
        return receipt

    @staticmethod
    def _stage():
        import omni.usd
        return omni.usd.get_context().get_stage()

    def update(self, *, robot_data, dt, time_s, previous_action, rendered):
        self.imu.update(dt)
        if self.pose_gyro is not None and dt != self.pose_gyro.dt:
            raise ValueError('Recorded delta-angle gyroscope requires its declared 2ms interval')
        gyro=None if self.pose_gyro is None else self.pose_gyro.observe(now_s=time_s)
        tactile = self.adapter.read(physics_dt=dt)
        enqueue_robot_sensors(self.builder, robot_data, self.imu.data, tactile,
                              joint_indices=self.joint_indices, capture_s=time_s,gyro_override=gyro)
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
            self._write_gyro_receipt()
            (self.output/'progress.json').write_text(json.dumps(dict(samples=len(self.times),
                time_s=time_s, frames=len(self.frame_times), finite=True))+'\n')

    def record_initial_decision(self, packet, motor_forces):
        """Preserve the actor's causal cold start before the first physics step."""
        path=self.output/'actor-initial-decision.npz'
        if path.exists():raise ValueError('Initial actor decision already recorded')
        _atomic_npz(path, **packet, motor_forces=np.asarray(motor_forces), time_s=np.asarray(0.))

    def finish(self, *, complete=True):
        from PIL import Image
        producer_receipt=self._write_gyro_receipt()
        complete=bool(complete and self.times and producer_receipt.get('failed_reason') is None
                      and producer_receipt.get('interval_count_matches_packets',True))
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
            values = arrays['tactile'][:, offsets[i]:offsets[i+1]] if self.times else np.empty((0,row['dimension']))
            peaks[row['name']] = float(np.max(np.abs(values))) if values.size else 0.
        camera_age = []
        for key in self.cameras:
            index = SENSOR_KEYS.index(key)
            if self.times:
                valid = arrays['sensor_valid'][:, index]
                camera_age.extend((arrays['time_s'][valid]-arrays['sensor_time_s'][valid,index]).tolist())
        report = dict(scope='Sensor-interface capture; see separate actor/teacher task report',control_source=self.control_source, capture_complete=bool(complete), samples=len(self.times),
            gyro_profile=self.gyro_profile,gyro_producer=producer_receipt,
            input_layout_sha256=getattr(self,'input_layout_sha256',None),
            effective_layout_sha256=hashlib.sha256((self.output/'layout.json').read_bytes()).hexdigest() if (self.output/'layout.json').is_file() else None,
            frames_per_eye=len(self.frame_times), tactile_dimension=self.layout['tactile_dimension'],
            finite=bool(all(np.isfinite(x).all() for x in arrays.values())),
            all_sensor_streams_present=bool(self.times and arrays['sensor_valid'].all()),
            camera_age_range_s=[min(camera_age),max(camera_age)] if camera_age else None,
            tactile_peak_abs_force_by_mount_N=peaks,
            rgb_mean_pixel_change={key:float(np.abs(frames[-1].astype(float)-frames[0]).mean()) if frames else None for key,frames in self.frames.items()},
            limitations=['Sensor stream checks alone do not establish task success', 'Robot camera usefulness requires visual inspection',
                         'Exact camera extrinsics and cross-engine mount parity require independent audit'])
        report['numeric_file_sha256']={name:hashlib.sha256((self.output/name).read_bytes()).hexdigest() for name in ('actor-sensors.npz','actor-rgb.npz')}
        _atomic_json(self.output/'report.json',report)
        return report

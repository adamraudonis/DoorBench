"""Opt-in own-IMU delta-angle producer; never an actor pose observation.

The legacy velocity-based sensor remains the default elsewhere. This explicit
profile measures the orientation change the renderer/physics pose actually
executes, including numerical position corrections. It is an ideal interval
sensor, not a claim that backend momentum velocity or accelerometer is repaired.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

PROFILE='pose-delta-angle-v1'
DEFAULT_PROFILE='backend-angular-velocity-v1'


def interval_local_gyro(previous,current,dt):
    """Finite SO(3) delta angle / dt; both endpoints are producer-only data."""
    before=np.asarray(previous,dtype=float);after=np.asarray(current,dtype=float)
    if (before.shape!=(3,3) or after.shape!=(3,3) or not np.isfinite([dt]).all() or dt<=0
            or not np.isfinite(before).all() or not np.isfinite(after).all()):
        raise ValueError('Finite proper IMU rotations and positive interval required')
    for r in (before,after):
        if np.max(abs(r.T@r-np.eye(3)))>1e-7 or abs(np.linalg.det(r)-1)>1e-7:raise ValueError('IMU frame must be a proper rotation')
    delta=Rotation.from_matrix(before.T@after).as_rotvec()
    if np.linalg.norm(delta)>=np.pi-1e-6:raise ValueError('Ambiguous sampled rotation')
    return delta/dt


class OwnImuPoseGyroscope:
    """Producer bound to one calibrated rigid body already owned by the robot.

    ``body_view`` is the recorder's existing one-body tensor view, never passed
    to an actor. Only ``observe``'s returned three rate numbers enter the actor
    packet. ``receipt`` is evaluator/provenance metadata and contains no poses.
    This class reads; it has no body/physics writer or simulation-step API.
    """
    def __init__(self,body_view,*,expected_body_path,robot_body_paths,
                 imu_quaternion_wxyz_body,physics_dt_s=.002,robot_root_path='/World/H1'):
        self.view=body_view;self.expected_path=str(expected_body_path);self.dt=float(physics_dt_s)
        paths=tuple(map(str,robot_body_paths));root=str(robot_root_path).rstrip('/')
        if (not root.startswith('/') or root in ('','/','/World')
                or not self.expected_path.startswith(root+'/') or any(not p.startswith(root+'/') for p in paths)
                or len(paths)!=len(set(paths))
                or self.expected_path not in paths or self.dt!=.002):
            raise ValueError('Declared owned robot body and exact 2ms producer interval required')
        q=np.asarray(imu_quaternion_wxyz_body,dtype=float)
        if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-7:raise ValueError('Calibrated IMU mounting quaternion required')
        self.mount=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        self.failed_reason=None;self.previous=None;self.previous_time=None;self.samples=0;self.maximum_delta_angle_rad=0.
        self._validate_binding()

    def _validate_binding(self):
        if tuple(map(str,self.view.prim_paths))!=(self.expected_path,):
            raise ValueError('Gyroscope tensor view must contain only the declared own robot IMU body')

    def _read_rotation(self):
        self._validate_binding()
        # PhysX transforms are XYZ + quaternion XYZW. Copy before another getter
        # can reuse backend storage. Translation is deliberately unused.
        raw=self.view.get_transforms()
        if hasattr(raw,'detach'):raw=raw.detach().cpu().numpy()
        pose=np.array(raw,dtype=float,copy=True)
        if pose.shape!=(1,7) or not np.isfinite(pose).all():raise ValueError('One finite own-body transform required')
        q=pose[0,3:]
        if abs(np.linalg.norm(q)-1)>1e-5:raise ValueError('Backend body quaternion is not unit length')
        return Rotation.from_quat(q).as_matrix()@self.mount

    def reset_episode(self,*,now_s=0.):
        self.failed_reason=None;self.previous=None;self.previous_time=None;self.samples=0;self.maximum_delta_angle_rad=0.
        try:
            if not np.isfinite(now_s) or float(now_s)!=0.:raise ValueError('Producer reset must coincide with episode t0')
            self.previous=self._read_rotation();self.previous_time=0.
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc);raise

    def observe(self,*,now_s):
        if self.failed_reason is not None:raise RuntimeError('Rejected gyro producer requires episode reset: '+self.failed_reason)
        try:
            now=float(now_s)
            if self.previous is None or not np.isfinite(now) or abs(now-self.previous_time-self.dt)>1e-8:
                raise ValueError('Gyro requires reset then consecutive completed 2ms intervals')
            current=self._read_rotation();rate=interval_local_gyro(self.previous,current,self.dt)
            delta=float(np.linalg.norm(rate)*self.dt)
            self.previous=current.copy();self.previous_time=now;self.samples+=1
            self.maximum_delta_angle_rad=max(self.maximum_delta_angle_rad,delta)
            return rate.astype(np.float32)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc);raise

    def receipt(self):
        return dict(schema='doorbench.imu-gyro-producer.v1',profile=PROFILE,
            source_body_path=self.expected_path,source='Consecutive own rigid-body tensor orientations composed with fixed IMU mounting rotation',
            actor_fields=['imu_gyro'],actor_receives_orientation=False,physics_writes=0,
            output_frame='IMU local axes; finite net rotation axis has identical start/end-frame coordinates',
            quantity='SO(3) interval delta angle divided by elapsed seconds; not instantaneous backend angular velocity',
            physics_dt_s=self.dt,interval_s=self.dt,availability='At interval end',effective_midpoint_delay_s=self.dt/2,
            output_dtype='float32',added_noise_std_radps=0.,ideal_sensor=True,
            accelerometer='Unchanged backend velocity-derived specific force; mixed numerical semantics explicitly retained',
            samples=self.samples,last_interval_end_s=self.previous_time,
            maximum_delta_angle_rad=self.maximum_delta_angle_rad,
            initialized=self.previous is not None,failed_reason=self.failed_reason)

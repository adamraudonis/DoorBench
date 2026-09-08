"""Finite sensor observations; no simulator object is ever passed to the actor.

World coordinates are permitted inside sensor emulation only. The public packet
contains fixed numeric arrays, images and acquisition-time/validity metadata.
This is a new interface, not a drop-in replacement for the old tactile policy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

INTERFACE_VERSION = "doorbench.sensors.v2"
SENSOR_KEYS = ("joint_position", "joint_velocity", "imu_gyro", "imu_accelerometer",
               "tactile", "rgb_left", "rgb_right")
ACTOR_KEYS = frozenset((*SENSOR_KEYS, "previous_action", "sensor_time_s", "sensor_valid"))


@dataclass(frozen=True)
class ActorDimensions:
    joints: int = 69
    actions: int = 61
    tactile: int = 1344
    image_size: int = 128
    hidden: int = 192

    def __post_init__(self):
        if any(type(v) is not int or v<=0 for v in vars(self).values()):
            raise ValueError('Actor dimensions must be positive integers')

    @property
    def shapes(self):
        return dict(joint_position=(self.joints,),joint_velocity=(self.joints,),
                    imu_gyro=(3,),imu_accelerometer=(3,),tactile=(self.tactile,),
                    rgb_left=(self.image_size,self.image_size,3),
                    rgb_right=(self.image_size,self.image_size,3))

    @property
    def proprio_dimension(self):
        return 2*self.joints+6+self.actions+2*len(SENSOR_KEYS)


def finite_array(value, shape=None, *, dtype=np.float64):
    array = np.array(value, dtype=dtype, copy=True)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite numeric array with shape {shape}, got {array.shape}")
    return array


def rotation_xyzw(quaternion):
    """Active local-to-world rotation. PhysX tensors use xyzw, Lab uses wxyz."""
    q = finite_array(quaternion, (4,))
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
        raise ValueError("Sensor pose quaternion must be normalized")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@dataclass(frozen=True)
class AngularTaxelGrid:
    """Sensor-local angular force bins compatible with native touch_grid gamma=0.

    fov values are HALF extents as implemented by MuJoCo, despite the field's
    name: (180, 90) spans the sphere. Output is channel-first [z,x,y], then y,x.
    No range, separation, counterpart identity or individual point is output.
    """
    width: int = 4
    height: int = 2
    fov_degrees: tuple[float, float] = (180., 90.)

    def __post_init__(self):
        if (type(self.width) is not int or type(self.height) is not int or
                self.width < 1 or self.height < 1 or len(self.fov_degrees) != 2 or
                not 0 < self.fov_degrees[0] <= 180 or not 0 < self.fov_degrees[1] <= 90):
            raise ValueError("Invalid angular tactile grid")

    @property
    def dimension(self):
        return 3 * self.width * self.height

    def bin_forces(self, points_world, forces_world, sensor_position, sensor_rotation):
        points = finite_array(points_world)
        forces = finite_array(forces_world, points.shape)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Contact samples must have shape (N,3)")
        origin = finite_array(sensor_position, (3,))
        rotation = finite_array(sensor_rotation, (3, 3))
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Sensor frame must be a proper rotation")
        local_points = (points - origin) @ rotation
        local_forces = forces @ rotation
        azimuth = np.arctan2(local_points[:, 0], -local_points[:, 2])
        elevation = np.arctan2(local_points[:, 1], np.hypot(local_points[:, 0], local_points[:, 2]))
        xedges = np.linspace(-self.fov_degrees[0], self.fov_degrees[0], self.width+1) * np.pi/180
        yedges = np.linspace(-self.fov_degrees[1], self.fov_degrees[1], self.height+1) * np.pi/180
        # Match the native lower-bound convention: bins are (lower, upper].
        x = np.searchsorted(xedges, azimuth, side="left") - 1
        y = np.searchsorted(yedges, elevation, side="left") - 1
        keep = ((x >= 0) & (x < self.width) & (y >= 0) & (y < self.height))
        result = np.zeros((3, self.height, self.width))
        for channel, axis in enumerate((2, 0, 1)):
            np.add.at(result[channel], (y[keep], x[keep]), local_forces[keep, axis])
        return result.reshape(-1).astype(np.float32)


@dataclass(frozen=True)
class SensorEffects:
    clip: float
    noise_std: float = 0.
    dropout_probability: float = 0.
    delay_s: float = 0.
    max_age_s: float = .25

    def __post_init__(self):
        vals = (self.clip, self.noise_std, self.dropout_probability, self.delay_s, self.max_age_s)
        if (not all(math.isfinite(v) for v in vals) or self.clip <= 0 or self.noise_std < 0 or
                not 0 <= self.dropout_probability <= 1 or self.delay_s < 0 or self.max_age_s < self.delay_s):
            raise ValueError("Invalid sensor effects")


class _Stream:
    def __init__(self, shape, dtype, effects, rng):
        self.shape, self.dtype, self.effects, self.rng = shape, dtype, effects, rng
        self.pending = deque()
        self.latest = None
        self.last_capture = -math.inf
        self.last_delivery = -math.inf

    def push(self, value, capture_s, available_s):
        if (not math.isfinite(capture_s) or not math.isfinite(available_s) or
                capture_s < 0 or capture_s <= self.last_capture or available_s < capture_s or
                available_s < self.last_delivery):
            raise ValueError("Sensor timestamps must be ordered and available no earlier than capture")
        array = finite_array(value, self.shape)
        if self.dtype == np.uint8 and (np.any(array < 0) or np.any(array > 255)):
            raise ValueError("RGB must contain unannotated uint8-range pixels")
        array += self.rng.normal(0., self.effects.noise_std, array.shape)
        upper = min(255., self.effects.clip) if self.dtype == np.uint8 else self.effects.clip
        array = np.clip(array, 0 if self.dtype == np.uint8 else -self.effects.clip, upper)
        valid = self.rng.random() >= self.effects.dropout_probability
        if not valid:
            array.fill(0)
        self.pending.append((available_s + self.effects.delay_s, capture_s, array.astype(self.dtype), valid))
        if len(self.pending) > 4096:
            raise RuntimeError("Sensor queue overflow; drain at control cadence")
        self.last_capture, self.last_delivery = capture_s, available_s

    def read(self, now_s):
        while self.pending and self.pending[0][0] <= now_s + 1e-12:
            self.latest = self.pending.popleft()
        if self.latest is None:
            return np.zeros(self.shape, self.dtype), -1., False
        _, capture_s, value, valid = self.latest
        if now_s - capture_s > self.effects.max_age_s:
            return np.zeros(self.shape, self.dtype), capture_s, False
        return value.copy(), capture_s, valid


class ActorObservationBuilder:
    """The sole actor packet boundary; evaluator state is not an input.

    Sensor availability time includes rendering/copy completion in simulation
    time. RGB capture time is the state actually rendered, not its delivery time.
    Reset must be called at every episode boundary, including simulator resets.
    """
    def __init__(self, *, joint_count, action_count, tactile_dimension, image_shape=(128, 128, 3), effects=None, seed=0):
        if min(joint_count, action_count, tactile_dimension) <= 0 or len(image_shape) != 3 or image_shape[-1] != 3:
            raise ValueError("Invalid actor dimensions")
        self.action_count = action_count
        self.shapes = {"joint_position": (joint_count,), "joint_velocity": (joint_count,),
                       "imu_gyro": (3,), "imu_accelerometer": (3,), "tactile": (tactile_dimension,),
                       "rgb_left": tuple(image_shape), "rgb_right": tuple(image_shape)}
        defaults = {"joint_position": SensorEffects(20.), "joint_velocity": SensorEffects(100.),
                    "imu_gyro": SensorEffects(100.), "imu_accelerometer": SensorEffects(200.),
                    "tactile": SensorEffects(100.), "rgb_left": SensorEffects(255.), "rgb_right": SensorEffects(255.)}
        if effects:
            if set(effects) - set(SENSOR_KEYS):
                raise ValueError("Unknown sensor effects key")
            defaults.update(effects)
        self.effects = defaults
        self.reset(seed=seed)

    def reset(self, *, seed=0):
        sequences = np.random.SeedSequence(seed).spawn(len(SENSOR_KEYS))
        self.streams = {key: _Stream(self.shapes[key], np.uint8 if key.startswith("rgb_") else np.float32,
                                    self.effects[key], np.random.default_rng(sequence))
                        for key, sequence in zip(SENSOR_KEYS, sequences)}
        self.last_read = -math.inf

    def push(self, key, value, *, capture_s, available_s=None):
        if key not in SENSOR_KEYS:
            raise ValueError("Only declared physical sensors can enter the actor packet")
        self.streams[key].push(value, capture_s, capture_s if available_s is None else available_s)

    def observe(self, *, now_s, previous_action):
        if not math.isfinite(now_s) or now_s < 0 or now_s < self.last_read:
            raise ValueError("Observation clock must be nonnegative and monotonic; reset each episode")
        packet = {}
        times, valid = [], []
        for key in SENSOR_KEYS:
            packet[key], timestamp, good = self.streams[key].read(now_s)
            times.append(timestamp)
            valid.append(good)
        packet["previous_action"] = np.clip(finite_array(previous_action, (self.action_count,)), -1, 1).astype(np.float32)
        packet["sensor_time_s"] = np.array(times, dtype=np.float64)
        packet["sensor_valid"] = np.array(valid, dtype=np.bool_)
        validate_actor_packet(packet, self.shapes, self.action_count)
        self.last_read = now_s
        return packet


def validate_actor_packet(packet, shapes, action_count):
    if type(packet) is not dict or set(packet) != ACTOR_KEYS:
        raise ValueError("Actor packet has missing or forbidden fields")
    expected = {**shapes, "previous_action": (action_count,), "sensor_time_s": (len(SENSOR_KEYS),), "sensor_valid": (len(SENSOR_KEYS),)}
    for key, shape in expected.items():
        value = packet[key]
        if type(value) is not np.ndarray or value.shape != shape or value.dtype.kind not in "fbu" or not np.isfinite(value).all():
            raise ValueError(f"Invalid actor array: {key}")

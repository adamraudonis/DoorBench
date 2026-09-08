"""Episode-local inference boundary for a frozen H1/Shadow sensor actor.

Only static calibration/motor metadata is accepted at construction. At runtime
the controller accepts numeric sensor packets and a local acquisition clock.
It has no simulator, teacher, geometry, object routing, or kinematics dependency.
Loading a checkpoint is not evidence of closed-loop task success.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from .sensor_actor import ActorDimensions, SensorActor, native_motor_forces
from .sensor_contract import INTERFACE_VERSION, validate_actor_packet


def _finite_positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite positive scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value


def _ordered_names(value, count, name):
    if (type(value) is not list or len(value) != count or
            any(type(v) is not str or not v for v in value) or len(set(value)) != count):
        raise ValueError(f"{name} must contain {count} unique ordered names")
    return tuple(value)


def _layout_json(value):
    if type(value) is not dict:
        raise ValueError("Sensor layout must be a plain calibration dictionary")
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sensor layout must contain finite JSON calibration metadata") from exc


class SensorPolicyController:
    """Validate a frozen checkpoint and infer original 61 native motor forces.

    ``motor_contract`` and ``sensor_layout`` must be read from the actual runtime
    import and sensor exporter, not copied from the checkpoint to bypass checks.
    The caller remains responsible for checking that the imported physical plant
    realizes that contract (including passive tendons and motor transmission).

    Call ``reset_episode()`` at every plant reset. Then call ``force(packet,
    now_s)`` exactly once per physics tick. The absolute clock is used only to
    check causality and obtain relative sensor ages; it is not an actor feature.
    Failed validation does not advance the clock or recurrent state. There is no
    automatic teacher fallback or force application to any simulator.
    """

    def __init__(self, checkpoint, *, motor_contract, sensor_layout,
                 physics_dt_s, image_shape=(128, 128, 3), device="cpu"):
        self.physics_dt_s = _finite_positive(physics_dt_s, "Physics timestep")
        if type(motor_contract) is not dict:
            raise ValueError("An actual native motor contract is required")
        if motor_contract.get("hand_mechanics_profile") != "shadow-loopback-v2":
            raise ValueError("Sensor policy requires the explicit shadow-loopback-v2 plant")
        model_hash = motor_contract.get("source_xml_sha256")
        if type(model_hash) is not str or not re.fullmatch(r"[0-9a-f]{64}", model_hash):
            raise ValueError("Actual motor contract must identify the source XML SHA-256")
        self.joint_order = _ordered_names(motor_contract.get("joint_names"), 69, "Motor joint order")
        actuators = motor_contract.get("actuators")
        if type(actuators) is not list or any(type(v) is not dict for v in actuators):
            raise ValueError("Actual actuator contract is required")
        self.action_order = _ordered_names([v.get("name") for v in actuators], 61, "Motor action order")
        try:
            caps = np.asarray([v["force_range"] for v in actuators], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Every actual motor requires its original force range") from exc
        # This also checks shape, finiteness and increasing asymmetric ranges.
        native_motor_forces(np.zeros(61), caps)
        self._force_ranges = caps.copy()
        self._force_ranges.setflags(write=False)

        layout_json = _layout_json(sensor_layout)
        if sensor_layout.get("interface_version") != INTERFACE_VERSION:
            raise ValueError("Actual sensor interface version differs")
        if sensor_layout.get("robot_xml_sha256") != model_hash:
            raise ValueError("Actual sensor and motor model hashes differ")
        if _ordered_names(sensor_layout.get("joint_order"), 69, "Sensor joint order") != self.joint_order:
            raise ValueError("Actual sensor and motor joint ordering differs")
        if _ordered_names(sensor_layout.get("action_order"), 61, "Sensor action order") != self.action_order:
            raise ValueError("Actual sensor and motor action ordering differs")

        # Hash exactly the bytes that weights_only loads; never deserialize a
        # custom model object or fall back to unrestricted pickle on failure.
        checkpoint_bytes = Path(checkpoint).read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        if type(payload) is not dict or payload.get("schema") != "doorbench.sensor-actor.v1":
            raise ValueError("Unsupported sensor actor checkpoint schema")
        if _layout_json(payload.get("sensor_layout")) != layout_json:
            raise ValueError("Checkpoint sensor layout/calibration differs from the actual runtime")
        checkpoint_dt = _finite_positive(payload.get("physics_dt_s"), "Checkpoint timestep")
        if not math.isclose(checkpoint_dt, self.physics_dt_s, rel_tol=0., abs_tol=1e-12):
            raise ValueError("Checkpoint physics timestep differs from the actual runtime")
        dimensions = payload.get("dimensions")
        if type(dimensions) is not dict or set(dimensions) != {"joints", "actions", "tactile", "image_size", "hidden"}:
            raise ValueError("Checkpoint dimensions must be complete and explicit")
        self.dimensions = ActorDimensions(**dimensions)
        if (self.dimensions.joints != 69 or self.dimensions.actions != 61 or
                type(sensor_layout.get("tactile_dimension")) is not int or
                self.dimensions.tactile != sensor_layout["tactile_dimension"] or
                tuple(image_shape) != (self.dimensions.image_size, self.dimensions.image_size, 3)):
            raise ValueError("Checkpoint dimensions differ from the actual sensor/motor dimensions")
        state = payload.get("model_state")
        if (not isinstance(state, dict) or not state or
                any(type(k) is not str or type(v) is not torch.Tensor or not torch.isfinite(v).all()
                    for k, v in state.items())):
            raise ValueError("Checkpoint model state must contain only finite named weight tensors")
        self._actor = SensorActor(self.dimensions)
        self._actor.load_state_dict(state, strict=True)
        self._actor.to(device).eval().requires_grad_(False)
        self.robot_xml_sha256 = model_hash
        self.sensor_layout_sha256 = hashlib.sha256(layout_json.encode()).hexdigest()
        self._active_episode = False
        self._hidden = None
        self._last_time_s = None
        self._previous_action = np.zeros(61, np.float32)

    @property
    def previous_action(self):
        """Copy of the last normalized motor command (zero before first tick)."""
        return self._previous_action.copy()

    @property
    def force_ranges(self):
        """Copy of original motor ranges in ``action_order``."""
        return self._force_ranges.copy()

    def reset_episode(self):
        self._hidden = None
        self._last_time_s = None
        self._previous_action = np.zeros(61, np.float32)
        self._active_episode = True

    def force(self, packet, now_s):
        """Return motor forces; the only dynamic inputs are sensors and clock."""
        if not self._active_episode:
            raise ValueError("reset_episode() is required before actor inference")
        if isinstance(now_s, (bool, np.bool_)) or not isinstance(now_s, (int, float, np.integer, np.floating)):
            raise ValueError("Local clock must be a finite nonnegative scalar")
        now_s = float(now_s)
        if not math.isfinite(now_s) or now_s < 0:
            raise ValueError("Local clock must be a finite nonnegative scalar")
        if self._last_time_s is not None and not math.isclose(
                now_s - self._last_time_s, self.physics_dt_s, rel_tol=0., abs_tol=1e-8):
            raise ValueError("Actor clock must advance by exactly one physics timestep")
        validate_actor_packet(packet, self.dimensions.shapes, 61)
        if packet["sensor_valid"].dtype != np.bool_:
            raise ValueError("Sensor validity must contain boolean flags")
        times = packet["sensor_time_s"]
        if (np.any((times < 0) & (times != -1)) or
                np.any(packet["sensor_valid"] & (times < 0))):
            raise ValueError("Sensor timestamps must be acquired nonnegative times or invalid -1")
        if np.any(np.abs(packet["previous_action"]) > 1. + 1e-6):
            raise ValueError("Previous motor action must be normalized to [-1,1]")
        # Own the bytes passed to inference; caller mutation cannot alter the
        # recurrent state or retained previous command afterward.
        numeric_packet = {key: value.copy() for key, value in packet.items()}
        action, hidden = self._actor.act(numeric_packet, now_s, self._hidden)
        if hidden is None or not torch.isfinite(hidden).all():
            raise ValueError("Nonfinite recurrent actor state")
        forces = native_motor_forces(action, self._force_ranges)
        self._hidden = hidden.detach()
        self._previous_action = action.astype(np.float32, copy=True)
        self._last_time_s = now_s
        return forces

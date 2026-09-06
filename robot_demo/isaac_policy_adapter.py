"""Researcher policy plug-in contract; this module does not import Isaac Sim."""

from __future__ import annotations
import numpy as np
from robot_demo.g1_policy import G1Policy, LEG_JOINTS


def unitree_factory(context):
    """Return a fresh recurrent policy for one episode, mapping joints by name."""
    policy = G1Policy(context["config_path"], context["checkpoint"])
    names = context["joint_names"]
    ids = np.array([names.index(n) for n in LEG_JOINTS])
    default = np.asarray(context["default_joint_positions"], dtype=np.float32)

    def act(obs):
        policy.sim_steps = round(obs["time_s"] / policy.dt)
        targets = default.copy()
        targets[ids] = policy.act(
            np.asarray(obs["base_quaternion_wxyz"]),
            np.asarray(obs["base_angular_velocity_body"]),
            np.asarray(obs["joint_positions"])[ids],
            np.asarray(obs["joint_velocities"])[ids],
            obs["command_velocity"],
        )
        return {"joint_positions": targets}

    return act


def validate_action(action, joint_count):
    """One finite command for every named robot joint; never door actuation."""
    if (
        not isinstance(action, dict)
        or len(action) != 1
        or not set(action) <= {"joint_positions", "joint_efforts"}
    ):
        raise ValueError(
            "Policy must return exactly one of joint_positions or joint_efforts"
        )
    mode = next(iter(action))
    values = np.asarray(action[mode], dtype=np.float32)
    if values.shape != (joint_count,) or not np.isfinite(values).all():
        raise ValueError(f"{mode} must be a finite vector of length {joint_count}")
    return mode, values

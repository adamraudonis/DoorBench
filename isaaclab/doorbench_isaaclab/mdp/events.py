"""Door-task event terms (reset randomisation).  NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU)."""
from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from .door_state import get_door_state


def reset_door(env, env_ids: torch.Tensor, asset_name: str = "door"):
    """Door joints to the spec's initial state (closed / open for traverse_open & close tasks), zero velocity, and
    reset the benchmark labels of those envs."""
    st = get_door_state(env, door_name=asset_name)
    door: Articulation = st.door
    q = st.door_reset_joint_pos(env_ids)
    v = torch.zeros_like(q)
    door.write_joint_state_to_sim(q, v, env_ids=env_ids)
    door.set_joint_position_target(st.spring_target[env_ids], env_ids=env_ids)
    door.set_joint_velocity_target(v, env_ids=env_ids)
    st.reset(env_ids)


def reset_hand_at_approach(env, env_ids: torch.Tensor, asset_name: str = "hand", xy_noise: float = 0.15, z_range=(0.8, 1.3), yaw_noise: float = 0.3):
    """Gantry hand: joints so that the palm starts near the approach point, at handle height, finger toward the door."""
    st = get_door_state(env)
    hand: Articulation = env.scene[asset_name]
    n = len(env_ids)
    dev = env.device
    q = torch.zeros(n, hand.num_joints, device=dev)
    names = hand.joint_names
    ix, iy, iz, iyaw = names.index("hand_x"), names.index("hand_y"), names.index("hand_z"), names.index("hand_yaw")
    # the hand base sits at the env origin; the approach point is (0, -1.5, 0) in the env frame
    app = st.approach[env_ids]
    grip_h = torch.tensor([float(st.metas[int(k)].get("handle_height") or 1.0) for k in env_ids], device=dev)
    q[:, ix] = app[:, 0] + sample_uniform(-xy_noise, xy_noise, (n,), dev)
    q[:, iy] = app[:, 1] + 0.5 + sample_uniform(-xy_noise, xy_noise, (n,), dev)
    q[:, iz] = grip_h.clamp(z_range[0], z_range[1]) + sample_uniform(-0.1, 0.1, (n,), dev)
    q[:, iyaw] = sample_uniform(-yaw_noise, yaw_noise, (n,), dev)
    hand.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    hand.set_joint_position_target(q, env_ids=env_ids)
    hand.set_joint_velocity_target(torch.zeros_like(q), env_ids=env_ids)


def reset_robot_at_approach(env, env_ids: torch.Tensor, asset_name: str = "robot", xy_noise: float = 0.2, yaw_noise: float = 0.2, height: float = 0.74, back_off: float = 0.0):
    """Humanoid root at the door's approach point facing +y (toward the door), small pose noise; joints at defaults."""
    st = get_door_state(env)
    robot: Articulation = env.scene[asset_name]
    n = len(env_ids)
    dev = env.device
    app = st.approach[env_ids] + st.env_origins[env_ids]
    pos = app.clone()
    pos[:, 0] += sample_uniform(-xy_noise, xy_noise, (n,), dev)
    pos[:, 1] += sample_uniform(-xy_noise, xy_noise, (n,), dev) - back_off
    pos[:, 2] = height
    yaw = math.pi / 2 + sample_uniform(-yaw_noise, yaw_noise, (n,), dev)   # robot x-axis -> +y
    quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    root = torch.cat([pos, quat, torch.zeros(n, 6, device=dev)], dim=-1)
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    q = robot.data.default_joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

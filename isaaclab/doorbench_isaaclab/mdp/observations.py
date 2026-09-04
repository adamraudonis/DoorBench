"""Door-task observation terms (all call ``DoorState.update()`` first; cheap after the first call per step).

NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU).
"""
from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from .door_state import get_door_state


def door_state_obs(env) -> torch.Tensor:
    """(N, 10): door joint / clearance, door velocity, operator fraction, latch fraction, flags
    [is_hinge, has_operator, has_latch, is_push, lock_engaged, closer]."""
    st = get_door_state(env)
    st.update()
    return torch.stack([
        st.door_q / st.clear_thr, st.door_dq * 0.2, st.op_frac, st.latch_frac,
        st.is_hinge.float(), st.has_op.float(), st.has_latch.float(), st.is_push.float(), st.lock_engaged.float(), st.has_closer.float(),
    ], dim=-1)


def door_task_obs(env) -> torch.Tensor:
    """(N, 9): one-hot benchmark task (open_and_traverse, open_only, traverse_open, close, unlock_open_traverse,
    locked_recognize, push_through, hold_and_pass, peek)."""
    st = get_door_state(env)
    return torch.nn.functional.one_hot(st.task_id, 9).float()


def grip_rel_tip(env) -> torch.Tensor:
    """(N, 3): handle grip point relative to the closest agent tip (world axes = env axes, no yaw for the hand)."""
    st = get_door_state(env)
    st.update()
    return (st.grip_w - st.tip_w).clamp(-3.0, 3.0)


def grip_rel_base(env, asset_name: str = "robot") -> torch.Tensor:
    """(N, 3): handle grip point relative to the robot base, in the base's yaw frame."""
    st = get_door_state(env)
    st.update()
    asset = env.scene[asset_name]
    rel = st.grip_w - asset.data.root_link_pos_w
    return quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), rel).clamp(-4.0, 4.0)


def goal_rel_base(env, asset_name: str = "robot") -> torch.Tensor:
    """(N, 3): goal point (beyond the door) relative to the robot base in its yaw frame + distance."""
    st = get_door_state(env)
    st.update()
    asset = env.scene[asset_name]
    goal_w = st.env_origins + st.goal
    rel = goal_w - asset.data.root_link_pos_w
    rel_b = quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), rel)
    return rel_b.clamp(-5.0, 5.0)


def goal_rel_tip(env) -> torch.Tensor:
    """(N, 3): goal point relative to the agent tip (hand agent, env axes)."""
    st = get_door_state(env)
    st.update()
    return (st.env_origins + st.goal - st.tip_w).clamp(-5.0, 5.0)


def tip_pos_local(env) -> torch.Tensor:
    """(N, 3): agent tip position in the env frame."""
    st = get_door_state(env)
    st.update()
    return st.tip_w - st.env_origins


def door_events_obs(env) -> torch.Tensor:
    """(N, 5): sticky event flags the policy may use (touched operator, actuated, unlatched, opened, passed)."""
    st = get_door_state(env)
    st.update()
    L = st.flags
    return torch.stack([L["touched_operator"], L["operator_actuated"], L["latch_released"], L["door_opened"], L["robot_passed_through"]], dim=-1).float()

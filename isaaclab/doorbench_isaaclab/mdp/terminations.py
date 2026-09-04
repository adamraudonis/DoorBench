"""Door-task termination terms.  NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU)."""
from __future__ import annotations

import torch

from .door_state import get_door_state, TASK_IDS


def task_success(env) -> torch.Tensor:
    """Episode ends (success) when the benchmark predicate is satisfied and nothing else is required."""
    st = get_door_state(env)
    st.update()
    s = st.success()
    # tasks that need the robot to *keep* doing something until the time limit are not terminated early
    hold = (st.task_id == TASK_IDS["peek"]) | (st.task_id == TASK_IDS["locked_recognize"])
    return s & ~hold


def door_damaged(env) -> torch.Tensor:
    st = get_door_state(env)
    st.update()
    return st.flags["door_damaged"]


def hand_far_away(env, max_distance: float = 4.0) -> torch.Tensor:
    """Hand agent wandered too far from the door (env frame)."""
    st = get_door_state(env)
    st.update()
    return (st.tip_w - st.env_origins).norm(dim=-1) > max_distance

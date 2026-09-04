"""Door-task reward terms.  Event bonuses fire once per episode on the step the benchmark label flips, so the logged
``Episode_Reward/<term>`` mean divided by the term weight is the per-episode event rate (e.g. success rate for
``traverse``).

Default weights (door_task_env_cfg.py) follow the benchmark reward events:
  touch handle +1, unlatch +2, open past the open threshold +3, traverse the pass plane +10,
  close behind (close task) +5, damage -10, slam -5, time penalty -0.01 / step.

NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU).
"""
from __future__ import annotations

import torch

from .door_state import get_door_state, TASK_IDS


def _st(env):
    st = get_door_state(env)
    st.update()
    return st


def event_bonus(env, event: str, tasks: list[str] | None = None) -> torch.Tensor:
    """1.0 on the step the benchmark label `event` becomes true (optionally only for some tasks)."""
    st = _st(env)
    r = st.new[event].float()
    if tasks:
        mask = torch.zeros_like(r, dtype=torch.bool)
        for t in tasks:
            mask |= st.task_id == TASK_IDS[t]
        r = r * mask.float()
    return r


def touch_handle(env):
    return event_bonus(env, "touched_operator") + event_bonus(env, "touched_door") * (~_st(env).has_op).float()


def unlatch(env):
    st = _st(env)
    return st.new["latch_released"].float() * st.has_latch.float()


def door_open(env):
    return event_bonus(env, "door_opened")


def door_clear(env):
    return event_bonus(env, "door_open_clear")


def traverse(env):
    return event_bonus(env, "robot_passed_through")


def closed_behind(env):
    return event_bonus(env, "door_closed_after", tasks=["close"])


def damage(env):
    return event_bonus(env, "door_damaged")


def slam(env):
    return event_bonus(env, "door_slammed")


def time_penalty(env):
    return torch.ones(env.num_envs, device=env.device)


def reach_handle(env, std: float = 0.3) -> torch.Tensor:
    """Dense shaping: exp(-d^2 / std^2) with d the tip-to-grip distance (until the door is open)."""
    st = _st(env)
    r = torch.exp(-(st.tip_dist ** 2) / (std ** 2))
    return r * (~st.flags["door_opened"]).float()


def door_progress(env) -> torch.Tensor:
    """Dense shaping: opening fraction of the clearance threshold (peek / locked tasks excluded)."""
    st = _st(env)
    frac = (st.door_q.abs() / st.clear_thr).clamp(0.0, 1.0)
    want_open = (st.task_id != TASK_IDS["locked_recognize"]) & (st.task_id != TASK_IDS["close"]) & ~st.lock_engaged
    return frac * want_open.float()


def door_closing_progress(env) -> torch.Tensor:
    """Dense shaping for the close task: 1 - opening fraction after the agent passed."""
    st = _st(env)
    frac = (st.door_q.abs() / st.clear_thr).clamp(0.0, 1.0)
    return (1.0 - frac) * ((st.task_id == TASK_IDS["close"]) & st.flags["robot_passed_through"]).float()


def forward_progress(env, asset_name: str = "robot") -> torch.Tensor:
    """Dense shaping: base velocity toward the goal (m/s, clipped) until the pass plane is crossed."""
    st = _st(env)
    asset = env.scene[asset_name]
    to_goal = st.env_origins + st.goal - asset.data.root_link_pos_w
    dir_xy = torch.nn.functional.normalize(to_goal[:, :2], dim=-1)
    v = (asset.data.root_com_lin_vel_w[:, :2] * dir_xy).sum(-1).clamp(-1.0, 1.0)
    return v * (~st.flags["robot_passed_through"]).float()


def tip_progress(env) -> torch.Tensor:
    """Dense shaping for the hand agent: tip velocity toward the goal once the door is open."""
    st = _st(env)
    goal_w = st.env_origins + st.goal
    d = (goal_w - st.tip_w)[:, :2].norm(dim=-1)
    prev = getattr(st, "_prev_goal_d", None)
    if prev is None or prev.shape != d.shape:
        st._prev_goal_d = d.clone()
        return torch.zeros_like(d)
    r = (prev - d).clamp(-0.1, 0.1) * 10.0
    st._prev_goal_d = d.clone()
    return r * st.flags["door_opened"].float() * (~st.flags["robot_passed_through"]).float()


def operator_overload(env) -> torch.Tensor:
    """Penalty proxy for hardware misuse: agent contact force on the operator above the operator yield torque / 0.1 m."""
    st = _st(env)
    return (st.operator_force > st.op_yield / 0.1).float()

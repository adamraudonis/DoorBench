"""MDP terms for the DoorBench Isaac Lab tasks (plus everything from isaaclab.envs.mdp)."""
from isaaclab.envs.mdp import *  # noqa: F401, F403

try:  # biped locomotion helpers (feet_slide, feet_air_time_positive_biped) live in the isaaclab_tasks locomotion mdp
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_air_time_positive_biped, feet_slide  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from .actions import DoorMechanismAction, DoorMechanismActionCfg  # noqa: F401
from .door_state import DoorState, get_door_state, EVENTS, TASK_IDS  # noqa: F401
from .events import release_env_lock, reset_door, reset_hand_at_approach, reset_robot_at_approach  # noqa: F401
from .observations import (  # noqa: F401
    door_state_obs, door_task_obs, door_events_obs, grip_rel_tip, grip_rel_base, goal_rel_base, goal_rel_tip, tip_pos_local,
)
from .rewards import (  # noqa: F401
    event_bonus, touch_handle, unlatch, door_open, door_clear, traverse, closed_behind, damage, slam, time_penalty,
    reach_handle, door_progress, door_closing_progress, forward_progress, tip_progress, operator_overload,
)
from .terminations import task_success, door_damaged, hand_far_away  # noqa: F401

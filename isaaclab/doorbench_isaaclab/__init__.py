"""DoorBench x Isaac Lab: train policies to open 1000 procedurally generated doors.

Importing this package registers the gymnasium tasks:

  DoorBench-Open-Hand-v0 / DoorBench-Open-Hand-Play-v0   6-DoF gantry hand agent
  DoorBench-Open-G1-v0   / DoorBench-Open-G1-Play-v0     Unitree G1 humanoid

Door subset: environment variable ``DOORBENCH_DOORS`` (default ``easy-100``) or the ``--doors`` flag of the scripts in
``scripts/isaaclab/``.  Dataset location: ``DOORBENCH_ASSETS`` or ``<repo>/assets``.
"""
from __future__ import annotations

__version__ = "0.1.0"

try:
    import gymnasium as gym

    _ENTRY = "isaaclab.envs:ManagerBasedRLEnv"
    gym.register(
        id="DoorBench-Open-Hand-v0",
        entry_point=_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.door_task_env_cfg:DoorHandEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:DoorHandPPORunnerCfg",
        },
    )
    gym.register(
        id="DoorBench-Open-Hand-Play-v0",
        entry_point=_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.door_task_env_cfg:DoorHandEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:DoorHandPPORunnerCfg",
        },
    )
    gym.register(
        id="DoorBench-Open-G1-v0",
        entry_point=_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.door_task_env_cfg:DoorG1EnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:DoorG1PPORunnerCfg",
        },
    )
    gym.register(
        id="DoorBench-Open-G1-Play-v0",
        entry_point=_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.door_task_env_cfg:DoorG1EnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:DoorG1PPORunnerCfg",
        },
    )
except ImportError:  # gymnasium not installed (e.g. the pure-python helpers are used outside Isaac Lab)
    pass

TASKS = ("DoorBench-Open-Hand-v0", "DoorBench-Open-Hand-Play-v0", "DoorBench-Open-G1-v0", "DoorBench-Open-G1-Play-v0")

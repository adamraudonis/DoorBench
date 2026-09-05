"""Isaac Lab asset configurations for DoorBench doors and the agents.

``door_cfg(door_id)``      one door (canonical ``door_rl.usda`` by default, or the full ``door.usda``)
``multi_door_cfg(ids)``    a different door per environment through ``MultiUsdFileCfg`` (all doors share the
                           canonical 8-link / 7-joint structure, which is what PhysX articulation views require)
``HAND_CFG``               6-DoF gantry hand (scripts/isaaclab/make_hand_usd.py)
``g1_cfg()``               Unitree G1 from isaaclab_assets, spawned at the door's approach point facing the door

Door joints are passive: an ``ImplicitActuatorCfg`` with ``stiffness=None`` / ``damping=None`` makes Isaac Lab use
the USD drive gains (closer springs, latch springs, operator return springs, locked slots).  The
``DoorMechanismAction`` term re-applies the spring targets every physics step.

NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU): written against the Isaac Lab 2.3 API.
"""
from __future__ import annotations

import os
import random

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .doors import door_usd_paths, select_ids, usd_path, require_eligible_ids

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HAND_USD = os.path.join(DATA_DIR, "gantry_hand.usda")

DOOR_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=4,
    sleep_threshold=0.0,
    stabilization_threshold=0.0,
    fix_root_link=None,   # the USD already has base_fixed (FixedJoint to the world)
)
DOOR_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    max_linear_velocity=20.0,
    max_angular_velocity=100.0,
)
# passive joints: gains come from the USD drives (None = use USD value)
DOOR_ACTUATORS = {"passive": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)}


def _door_spawn_kwargs():
    return dict(
        activate_contact_sensors=True,
        rigid_props=DOOR_RIGID_PROPS,
        articulation_props=DOOR_ARTICULATION_PROPS,
    )


def door_cfg(door_id: str, prim_path: str = "{ENV_REGEX_NS}/Door", canonical: bool = True) -> ArticulationCfg:
    """ArticulationCfg for one door.  ``canonical=False`` loads the full-fidelity door.usda (its own joint names)."""
    require_eligible_ids([door_id])
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path(door_id, canonical=canonical), **_door_spawn_kwargs()),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators=DOOR_ACTUATORS,
        articulation_root_prim_path="/Articulation",
    )


def multi_door_cfg(ids: list[str] | str, seed: int = 0, random_choice: bool = False, shuffle: bool = True, prim_path: str = "{ENV_REGEX_NS}/Door") -> ArticulationCfg:
    """One ArticulationCfg whose spawner picks a different door per environment.

    ``ids``: list of door ids or a selection string for ``doors.select_ids`` ("easy-100", "all", "family:saloon", ...).
    ``random_choice=False`` (default) assigns door i to env i modulo len(ids) -> with num_envs >= len(ids) every door
    is present at least once (deterministic; ``shuffle`` randomises the order with ``seed``).  ``random_choice=True``
    uses Isaac Lab's random pick per env (Python ``random``, seeded by the env seed).
    """
    if isinstance(ids, str):
        ids = select_ids(ids, seed=seed)
    require_eligible_ids(list(ids))
    paths = door_usd_paths(list(ids), canonical=True)
    if shuffle:
        random.Random(seed).shuffle(paths)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.MultiUsdFileCfg(usd_path=paths, random_choice=random_choice, **_door_spawn_kwargs()),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators=DOOR_ACTUATORS,
        articulation_root_prim_path="/Articulation",
    )


HAND_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Hand",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HAND_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=3.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={"hand_x": 0.0, "hand_y": -1.0, "hand_z": 1.0, "hand_yaw": 0.0, "hand_pitch": 0.0, "hand_roll": 0.0},
    ),
    actuators={"gantry": ImplicitActuatorCfg(joint_names_expr=["hand_.*"], stiffness=None, damping=None)},
    soft_joint_pos_limit_factor=1.0,
)


def g1_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """Unitree G1 (Isaac Lab's minimal USD) at the approach point, facing +y (the door)."""
    from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

    cfg = G1_MINIMAL_CFG.copy()
    cfg.prim_path = prim_path
    cfg.init_state.pos = (0.0, -1.5, 0.74)
    cfg.init_state.rot = (0.7071068, 0.0, 0.0, 0.7071068)
    return cfg

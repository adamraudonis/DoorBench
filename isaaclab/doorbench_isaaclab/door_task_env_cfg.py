"""Manager-based RL environments for opening DoorBench doors in Isaac Lab.

  DoorBench-Open-Hand-v0   6-DoF gantry hand (fast to train; validates the door mechanics)
  DoorBench-Open-G1-v0     Unitree G1 humanoid (loco-manipulation)

Each environment holds a *different* door: ``DOORBENCH_DOORS`` (env var, default ``easy-100``; ``all`` for the
whole dataset, ``family:saloon``, explicit ids, ``@file``) selects the subset; the train/play/eval scripts expose
it as ``--doors``.  The scene needs ``replicate_physics=False`` because environments are not identical copies.

Rewards = the benchmark events (touch handle +1, unlatch +2, open +3, traverse +10, close-behind +5, damage -10,
slam -5, time -0.01/step) plus small dense shaping (reach the handle, opening fraction, progress to the goal).

NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU): written against the Isaac Lab 2.3 API.
"""
from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .assets import HAND_CFG, g1_cfg, multi_door_cfg

DOOR_SELECTION = os.environ.get("DOORBENCH_DOORS", "easy-100")
DOOR_SEED = int(os.environ.get("DOORBENCH_DOOR_SEED", "0"))


# ----------------------------------------------------------------------------------------------------------- scene
@configclass
class DoorSceneCfg(InteractiveSceneCfg):
    """Ground + light + one door per env (+ the agent added by the task cfg)."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.0)),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)))
    # filled in by the env cfg's __post_init__ (DOORBENCH_DOORS) or by set_doors(): a different door per env
    door: ArticulationCfg = MISSING
    # contacts between the agent and the leaf / operator (filter set per agent below)
    contact_leaf: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Door/Articulation/leaf", update_period=0.0, history_length=1)
    contact_operator: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Door/Articulation/operator", update_period=0.0, history_length=1)


# ----------------------------------------------------------------------------------------------------------- hand
@configclass
class HandSceneCfg(DoorSceneCfg):
    hand: ArticulationCfg = HAND_CFG

    def __post_init__(self):
        self.contact_leaf.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Hand/palm"]
        self.contact_operator.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Hand/palm"]


@configclass
class HandActionsCfg:
    hand = mdp.RelativeJointPositionActionCfg(
        asset_name="hand",
        joint_names=["hand_x", "hand_y", "hand_z", "hand_yaw", "hand_pitch", "hand_roll"],
        scale={"hand_x": 0.04, "hand_y": 0.04, "hand_z": 0.04, "hand_yaw": 0.15, "hand_pitch": 0.15, "hand_roll": 0.15},
        use_zero_offset=True,
    )
    door_mechanism = mdp.DoorMechanismActionCfg(asset_name="door")


@configclass
class HandObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("hand")}, noise=Unoise(n_min=-0.005, n_max=0.005))
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("hand")}, scale=0.2, noise=Unoise(n_min=-0.05, n_max=0.05))
        tip_pos = ObsTerm(func=mdp.tip_pos_local)
        grip_rel = ObsTerm(func=mdp.grip_rel_tip, noise=Unoise(n_min=-0.01, n_max=0.01))
        goal_rel = ObsTerm(func=mdp.goal_rel_tip)
        door = ObsTerm(func=mdp.door_state_obs)
        task = ObsTerm(func=mdp.door_task_obs)
        events = ObsTerm(func=mdp.door_events_obs)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class DoorRewardsCfg:
    """Benchmark reward events (shared by both agents)."""

    touch_handle = RewTerm(func=mdp.touch_handle, weight=1.0)
    unlatch = RewTerm(func=mdp.unlatch, weight=2.0)
    door_open = RewTerm(func=mdp.door_open, weight=3.0)
    door_clear = RewTerm(func=mdp.door_clear, weight=2.0)
    traverse = RewTerm(func=mdp.traverse, weight=10.0)
    closed_behind = RewTerm(func=mdp.closed_behind, weight=5.0)
    damage = RewTerm(func=mdp.damage, weight=-10.0)
    slam = RewTerm(func=mdp.slam, weight=-5.0)
    operator_overload = RewTerm(func=mdp.operator_overload, weight=-0.5)
    time_penalty = RewTerm(func=mdp.time_penalty, weight=-0.01)
    # dense shaping
    reach_handle = RewTerm(func=mdp.reach_handle, weight=0.5, params={"std": 0.3})
    door_progress = RewTerm(func=mdp.door_progress, weight=0.5)
    door_closing_progress = RewTerm(func=mdp.door_closing_progress, weight=0.5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


@configclass
class HandRewardsCfg(DoorRewardsCfg):
    tip_progress = RewTerm(func=mdp.tip_progress, weight=1.0)


@configclass
class HandTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.task_success)
    far_away = DoneTerm(func=mdp.hand_far_away, params={"max_distance": 4.5})


@configclass
class HandEventsCfg:
    reset_door = EventTerm(func=mdp.reset_door, mode="reset", params={"asset_name": "door"})
    reset_hand = EventTerm(func=mdp.reset_hand_at_approach, mode="reset", params={"asset_name": "hand"})
    door_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("hand", body_names="palm"), "static_friction_range": (0.8, 1.2), "dynamic_friction_range": (0.7, 1.0), "restitution_range": (0.0, 0.0), "num_buckets": 16},
    )


@configclass
class DoorHandEnvCfg(ManagerBasedRLEnvCfg):
    """DoorBench-Open-Hand-v0."""

    scene: HandSceneCfg = HandSceneCfg(num_envs=1024, env_spacing=6.0, replicate_physics=False)
    observations: HandObservationsCfg = HandObservationsCfg()
    actions: HandActionsCfg = HandActionsCfg()
    rewards: HandRewardsCfg = HandRewardsCfg()
    terminations: HandTerminationsCfg = HandTerminationsCfg()
    events: HandEventsCfg = HandEventsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(6.0, -8.0, 4.0), lookat=(0.0, 0.0, 1.0), resolution=(1920, 1080))

    def __post_init__(self):
        if self.scene.door is MISSING:
            self.scene.door = multi_door_cfg(DOOR_SELECTION, seed=DOOR_SEED)
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_type = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**22
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22
        self.sim.physics_material = self.scene.ground.spawn.physics_material
        self.scene.contact_leaf.update_period = self.sim.dt
        self.scene.contact_operator.update_period = self.sim.dt


@configclass
class DoorHandEnvCfg_PLAY(DoorHandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.episode_length_s = 30.0
        self.observations.policy.enable_corruption = False


# ------------------------------------------------------------------------------------------------------------- G1
@configclass
class G1SceneCfg(DoorSceneCfg):
    robot: ArticulationCfg = g1_cfg()
    contact_forces: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    def __post_init__(self):
        robot_links = ["{ENV_REGEX_NS}/Robot/.*"]
        self.contact_leaf.filter_prim_paths_expr = robot_links
        self.contact_operator.filter_prim_paths_expr = robot_links


@configclass
class G1ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    door_mechanism = mdp.DoorMechanismActionCfg(asset_name="door")


@configclass
class G1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        grip_rel = ObsTerm(func=mdp.grip_rel_base, params={"asset_name": "robot"}, noise=Unoise(n_min=-0.02, n_max=0.02))
        goal_rel = ObsTerm(func=mdp.goal_rel_base, params={"asset_name": "robot"})
        door = ObsTerm(func=mdp.door_state_obs)
        task = ObsTerm(func=mdp.door_task_obs)
        events = ObsTerm(func=mdp.door_events_obs)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1RewardsCfg(DoorRewardsCfg):
    """Benchmark events + locomotion regularisers from Isaac Lab's G1 velocity task."""

    forward_progress = RewTerm(func=mdp.forward_progress, weight=1.0, params={"asset_name": "robot"})
    alive = RewTerm(func=mdp.is_alive, weight=0.05)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])})
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])})
    joint_deviation_torso = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")})
    joint_deviation_fingers = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_five_joint", ".*_three_joint", ".*_six_joint", ".*_four_joint", ".*_zero_joint", ".*_one_joint", ".*_two_joint"])})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.1, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")})


@configclass
class G1TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.task_success)
    base_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0})
    fell = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("robot")})


@configclass
class G1EventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.8, 0.8), "dynamic_friction_range": (0.6, 0.6), "restitution_range": (0.0, 0.0), "num_buckets": 64},
    )
    reset_door = EventTerm(func=mdp.reset_door, mode="reset", params={"asset_name": "door"})
    reset_robot = EventTerm(func=mdp.reset_robot_at_approach, mode="reset", params={"asset_name": "robot", "xy_noise": 0.2, "yaw_noise": 0.2, "height": 0.74})


@configclass
class DoorG1EnvCfg(ManagerBasedRLEnvCfg):
    """DoorBench-Open-G1-v0."""

    scene: G1SceneCfg = G1SceneCfg(num_envs=1024, env_spacing=6.0, replicate_physics=False)
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventsCfg = G1EventsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(6.0, -8.0, 4.0), lookat=(0.0, 0.0, 1.0), resolution=(1920, 1080))

    def __post_init__(self):
        if self.scene.door is MISSING:
            self.scene.door = multi_door_cfg(DOOR_SELECTION, seed=DOOR_SEED)
        self.decimation = 4
        self.episode_length_s = 25.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_type = 1
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**22
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22
        self.sim.physics_material = self.scene.ground.spawn.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.contact_leaf.update_period = self.sim.dt
        self.scene.contact_operator.update_period = self.sim.dt
        # the door mechanism term must not appear in the last-action observation of the humanoid (it is 0-dim anyway)


@configclass
class DoorG1EnvCfg_PLAY(DoorG1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False


def set_doors(env_cfg: ManagerBasedRLEnvCfg, selection: str, seed: int = 0, random_choice: bool = False):
    """Replace the door set of an env cfg (used by the CLI scripts: --doors easy-100 | all | family:... | ids)."""
    env_cfg.scene.door = multi_door_cfg(selection, seed=seed, random_choice=random_choice)
    return env_cfg

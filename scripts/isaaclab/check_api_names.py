#!/usr/bin/env python
"""Offline API checklist for the Isaac Lab extension (no Isaac installation needed).

Parses every module of ``isaaclab/doorbench_isaaclab`` and ``scripts/isaaclab`` with ``ast`` and lists
  * every name imported from ``isaaclab*`` / ``isaaclab_rl`` / ``isaaclab_tasks`` / ``isaaclab_assets`` / ``rsl_rl``
  * every attribute accessed on the ``mdp`` module and on ``sim_utils``
and compares them against the reference list below, compiled from the Isaac Lab v2.3.0 sources / docs
(isaac-sim.github.io/IsaacLab, raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.0).  Exit status 1 on unknown names.

This is the closest thing to an import test we can run without a GPU box: it catches typos and renamed symbols, not
signature or semantic errors.
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

KNOWN = {
    "isaaclab.app": {"AppLauncher"},
    "isaaclab.actuators": {"ImplicitActuatorCfg", "ActuatorBaseCfg", "IdealPDActuatorCfg"},
    "isaaclab.assets": {"Articulation", "ArticulationCfg", "AssetBaseCfg", "RigidObject", "RigidObjectCfg"},
    "isaaclab.envs": {"ManagerBasedRLEnv", "ManagerBasedRLEnvCfg", "ManagerBasedEnvCfg", "ViewerCfg"},
    "isaaclab.envs.mdp": {"*"},
    "isaaclab.managers": {"ActionTerm", "ActionTermCfg", "EventTermCfg", "ObservationGroupCfg", "ObservationTermCfg", "RewardTermCfg", "SceneEntityCfg", "TerminationTermCfg", "CurriculumTermCfg", "ManagerTermBase"},
    "isaaclab.scene": {"InteractiveSceneCfg", "InteractiveScene"},
    "isaaclab.sensors": {"ContactSensorCfg", "ContactSensor"},
    "isaaclab.sim": {"build_simulation_context", "SimulationCfg", "SimulationContext", "PhysxCfg"},
    "isaaclab.utils": {"configclass"},
    "isaaclab.utils.assets": {"ISAAC_NUCLEUS_DIR", "ISAACLAB_NUCLEUS_DIR", "retrieve_file_path"},
    "isaaclab.utils.dict": {"print_dict"},
    "isaaclab.utils.io": {"dump_pickle", "dump_yaml"},
    "isaaclab.utils.math": {"quat_apply", "quat_apply_inverse", "yaw_quat", "quat_from_euler_xyz", "sample_uniform", "subtract_frame_transforms", "quat_mul", "matrix_from_quat"},
    "isaaclab.utils.noise": {"AdditiveUniformNoiseCfg"},
    "isaaclab_assets.robots.unitree": {"G1_MINIMAL_CFG", "G1_CFG"},
    "isaaclab_rl.rsl_rl": {"RslRlOnPolicyRunnerCfg", "RslRlPpoActorCriticCfg", "RslRlPpoAlgorithmCfg", "RslRlVecEnvWrapper", "export_policy_as_jit", "export_policy_as_onnx", "RslRlBaseRunnerCfg"},
    "isaaclab_tasks.utils": {"get_checkpoint_path", "parse_env_cfg", "load_cfg_from_registry"},
    "isaaclab_tasks.utils.parse_cfg": {"get_checkpoint_path", "parse_env_cfg", "load_cfg_from_registry"},
    "isaaclab_tasks.manager_based.locomotion.velocity.mdp": {"feet_slide", "feet_air_time_positive_biped", "feet_air_time"},
    "rsl_rl.runners": {"OnPolicyRunner", "DistillationRunner"},
}
# attributes used as `sim_utils.X`
SIM_UTILS = {"UsdFileCfg", "MultiUsdFileCfg", "MultiAssetSpawnerCfg", "GroundPlaneCfg", "DomeLightCfg", "RigidBodyPropertiesCfg", "ArticulationRootPropertiesCfg",
             "RigidBodyMaterialCfg", "CollisionPropertiesCfg", "MassPropertiesCfg", "JointDrivePropertiesCfg", "SimulationCfg", "PhysxCfg"}
# attributes used as `mdp.X` that come from isaaclab.envs.mdp (v2.3.0) or the locomotion mdp
MDP = {
    # observations
    "base_lin_vel", "base_ang_vel", "projected_gravity", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_pos_rel",
    "joint_pos_limit_normalized", "joint_vel", "joint_vel_rel", "last_action", "generated_commands", "height_scan", "body_pose_w", "joint_effort",
    # rewards
    "is_alive", "is_terminated", "is_terminated_term", "lin_vel_z_l2", "ang_vel_xy_l2", "flat_orientation_l2", "base_height_l2", "body_lin_acc_l2", "joint_torques_l2",
    "joint_vel_l1", "joint_vel_l2", "joint_acc_l2", "joint_deviation_l1", "joint_pos_limits", "joint_vel_limits", "applied_torque_limits", "action_rate_l2", "action_l2",
    "undesired_contacts", "contact_forces", "track_lin_vel_xy_exp", "track_ang_vel_z_exp",
    # terminations
    "time_out", "command_resample", "bad_orientation", "root_height_below_minimum", "joint_pos_out_of_limit", "joint_pos_out_of_manual_limit", "joint_vel_out_of_limit",
    "joint_vel_out_of_manual_limit", "joint_effort_out_of_limit", "illegal_contact",
    # events
    "randomize_rigid_body_material", "randomize_rigid_body_mass", "randomize_rigid_body_com", "randomize_actuator_gains", "randomize_joint_parameters",
    "apply_external_force_torque", "push_by_setting_velocity", "reset_root_state_uniform", "reset_root_state_with_random_orientation", "reset_joints_by_scale",
    "reset_joints_by_offset", "reset_scene_to_default", "randomize_physics_scene_gravity",
    # actions cfgs
    "JointPositionActionCfg", "RelativeJointPositionActionCfg", "JointEffortActionCfg", "JointVelocityActionCfg",
    # locomotion mdp
    "feet_slide", "feet_air_time_positive_biped", "feet_air_time",
}


def scan(path):
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    imports, attrs = [], {"mdp": set(), "sim_utils": set()}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and (node.module.startswith("isaaclab") or node.module.startswith("rsl_rl")):
            for a in node.names:
                imports.append((node.module, a.name))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in attrs:
            attrs[node.value.id].add(node.attr)
    return imports, attrs


def main():
    files = []
    for d in (os.path.join(ROOT, "isaaclab", "doorbench_isaaclab"), os.path.join(ROOT, "scripts", "isaaclab")):
        for dp, _, fns in os.walk(d):
            files += [os.path.join(dp, f) for f in fns if f.endswith(".py")]
    local_mdp = set()
    mdp_init = os.path.join(ROOT, "isaaclab", "doorbench_isaaclab", "mdp", "__init__.py")
    with open(mdp_init) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:  # relative imports = our own terms
            local_mdp |= {a.name for a in node.names}
    unknown, n_checked = [], 0
    for fp in sorted(files):
        imports, attrs = scan(fp)
        for mod, name in imports:
            n_checked += 1
            known = KNOWN.get(mod)
            if known is None:
                unknown.append((os.path.relpath(fp, ROOT), f"module {mod} not in the reference list"))
            elif "*" not in known and name not in known:
                unknown.append((os.path.relpath(fp, ROOT), f"{mod}.{name}"))
        for a in attrs["mdp"]:
            n_checked += 1
            if a not in MDP and a not in local_mdp:
                unknown.append((os.path.relpath(fp, ROOT), f"mdp.{a}"))
        for a in attrs["sim_utils"]:
            n_checked += 1
            if a not in SIM_UTILS:
                unknown.append((os.path.relpath(fp, ROOT), f"sim_utils.{a}"))
    print(f"checked {n_checked} Isaac Lab symbol references in {len(files)} files against the v2.3.0 reference list")
    for fp, name in unknown:
        print(f"  UNKNOWN {fp}: {name}")
    if unknown:
        sys.exit(1)
    print("all symbols known")


if __name__ == "__main__":
    main()

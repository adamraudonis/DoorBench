#!/usr/bin/env python
"""Offline API checklist for the Isaac Lab extension (no Isaac installation needed).

Parses every module of ``isaaclab/doorbench_isaaclab`` and ``scripts/isaaclab`` with ``ast`` and lists
  * every name imported from ``isaaclab*`` / ``isaaclab_rl`` / ``isaaclab_tasks`` / ``isaaclab_assets`` / ``rsl_rl``
  * every attribute accessed on the ``mdp`` module and on ``sim_utils``
and compares them against the reference list below, compiled from the Isaac Lab **v2.3.2** sources
(github.com/isaac-sim/IsaacLab tag v2.3.2, isaaclab extension 0.54.2, ``rsl-rl-lib==3.1.2``).  Exit status 1 on
unknown names.

Two modes:

  python scripts/isaaclab/check_api_names.py
      offline: names against the static reference list in this file (what tests/test_isaaclab_ext.py runs)

  python scripts/isaaclab/check_api_names.py --source ~/IsaacLab [--source ~/rsl_rl]
      against a checkout: resolves every imported name / module attribute in the given source trees (an Isaac Lab
      checkout with ``source/<pkg>/<pkg>``, and/or a plain package root such as an rsl_rl clone) and additionally
      checks every **keyword argument** of calls to Isaac Lab config classes and functions against the fields /
      parameters defined there (``ArticulationCfg(articulation_root_prim_path=...)``, ``ContactSensorCfg(...)``,
      ``build_simulation_context(...)`` ...).  ``--dump`` prints the reference tables for the modules our code uses,
      which is how the static list below was generated.

This is the closest thing to an import test we can run without a GPU box: it catches typos, renamed symbols and
removed config fields, not runtime semantics.  Known v2.3.2 differences versus the v2.3.0 list this file used to
carry: ``isaaclab.utils.io`` lost ``dump_pickle`` / ``load_pickle`` (Isaac Lab 0.47.0: "Removed pickle utilities");
``scripts/isaaclab/_common.py`` provides a local ``dump_pickle``.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUR_DIRS = (os.path.join(ROOT, "isaaclab", "doorbench_isaaclab"), os.path.join(ROOT, "scripts", "isaaclab"))
EXTERNAL_PREFIXES = ("isaaclab", "rsl_rl")

# ------------------------------------------------------------------------------------------------ v2.3.2 reference
# Generated with `--source <IsaacLab v2.3.2> --source <rsl_rl v3.1.2> --dump` (public names only) and hand-checked.
KNOWN = {
    "isaaclab.app": {"AppLauncher"},
    "isaaclab.actuators": {
        "ActuatorBase", "ActuatorBaseCfg", "ActuatorNetLSTM", "ActuatorNetLSTMCfg", "ActuatorNetMLP", "ActuatorNetMLPCfg", "DCMotor", "DCMotorCfg",
        "DelayedPDActuator", "DelayedPDActuatorCfg", "IdealPDActuator", "IdealPDActuatorCfg", "ImplicitActuator", "ImplicitActuatorCfg",
        "RemotizedPDActuator", "RemotizedPDActuatorCfg",
    },
    "isaaclab.assets": {
        "Articulation", "ArticulationCfg", "ArticulationData", "AssetBase", "AssetBaseCfg", "DeformableObject", "DeformableObjectCfg", "DeformableObjectData",
        "RigidObject", "RigidObjectCfg", "RigidObjectCollection", "RigidObjectCollectionCfg", "RigidObjectCollectionData", "RigidObjectData",
        "SurfaceGripper", "SurfaceGripperCfg",
    },
    "isaaclab.envs": {
        "DirectMARLEnv", "DirectMARLEnvCfg", "DirectRLEnv", "DirectRLEnvCfg", "ManagerBasedEnv", "ManagerBasedEnvCfg", "ManagerBasedRLEnv",
        "ManagerBasedRLEnvCfg", "ManagerBasedRLMimicEnv", "MimicEnvCfg", "SubTaskConfig", "SubTaskConstraintConfig", "SubTaskConstraintType",
        "SubTaskConstraintCoordinationScheme", "DataGenConfig", "VecEnvObs", "VecEnvStepReturn", "ViewerCfg", "mdp", "ui",
        "multi_agent_to_single_agent", "multi_agent_with_one_agent",
    },
    "isaaclab.envs.mdp": {"*"},
    "isaaclab.managers": {
        "ActionManager", "ActionTerm", "ActionTermCfg", "CommandManager", "CommandTerm", "CommandTermCfg", "CurriculumManager", "CurriculumTermCfg",
        "DatasetExportMode", "EventManager", "EventTermCfg", "ManagerBase", "ManagerTermBase", "ManagerTermBaseCfg", "ObservationGroupCfg",
        "ObservationManager", "ObservationTermCfg", "RecorderManager", "RecorderManagerBaseCfg", "RecorderTerm", "RecorderTermCfg", "RewardManager",
        "RewardTermCfg", "SceneEntityCfg", "TerminationManager", "TerminationTermCfg",
    },
    "isaaclab.scene": {"InteractiveScene", "InteractiveSceneCfg"},
    "isaaclab.sensors": {
        "Camera", "CameraCfg", "CameraData", "ContactSensor", "ContactSensorCfg", "ContactSensorData", "FrameTransformer", "FrameTransformerCfg",
        "FrameTransformerData", "Imu", "ImuCfg", "ImuData", "MultiMeshRayCaster", "MultiMeshRayCasterCamera", "MultiMeshRayCasterCameraCfg",
        "MultiMeshRayCasterCfg", "MultiMeshRayCasterData", "OffsetCfg", "RayCaster", "RayCasterCamera", "RayCasterCameraCfg", "RayCasterCfg",
        "RayCasterData", "SensorBase", "SensorBaseCfg", "TiledCamera", "TiledCameraCfg", "patterns",
    },
    "isaaclab.sim": {"*"},   # `import isaaclab.sim as sim_utils` attributes are checked through SIM_UTILS below
    "isaaclab.utils": {"configclass", "Timer", "class_to_dict", "update_class_from_dict", "print_dict", "update_dict", "dict_to_md5_hash", "convert_dict_to_backend"},
    "isaaclab.utils.assets": {"ISAAC_NUCLEUS_DIR", "ISAACLAB_NUCLEUS_DIR", "NVIDIA_NUCLEUS_DIR", "NUCLEUS_ASSET_ROOT_DIR", "check_file_path", "check_usd_path_with_timeout", "read_file", "retrieve_file_path"},
    "isaaclab.utils.dict": {"class_to_dict", "convert_dict_to_backend", "dict_to_md5_hash", "print_dict", "replace_slices_with_strings", "replace_strings_with_slices", "update_class_from_dict", "update_dict"},
    # v2.3.2: NO dump_pickle / load_pickle any more (removed in isaaclab 0.47.0)
    "isaaclab.utils.io": {"dump_yaml", "load_yaml", "load_torchscript_model"},
    "isaaclab.utils.math": {
        "apply_delta_pose", "axis_angle_from_quat", "combine_frame_transforms", "compute_pose_error", "convert_camera_frame_orientation_convention",
        "convert_quat", "copysign", "create_rotation_matrix_from_view", "default_orientation", "euler_xyz_from_quat", "generate_random_rotation",
        "generate_random_transformation_matrix", "generate_random_translation", "interpolate_poses", "interpolate_rotations", "is_identity_pose",
        "make_pose", "matrix_from_euler", "matrix_from_quat", "normalize", "orthogonalize_perspective_depth", "pose_in_A_to_pose_in_B", "pose_inv",
        "project_points", "quat_apply", "quat_apply_inverse", "quat_apply_yaw", "quat_box_minus", "quat_box_plus", "quat_conjugate",
        "quat_error_magnitude", "quat_from_angle_axis", "quat_from_euler_xyz", "quat_from_matrix", "quat_inv", "quat_mul", "quat_rotate",
        "quat_rotate_inverse", "quat_slerp", "quat_unique", "random_orientation", "random_yaw_orientation", "rigid_body_twist_transform",
        "sample_cylinder", "sample_gaussian", "sample_log_uniform", "sample_triangle", "sample_uniform", "saturate", "scale_transform",
        "skew_symmetric_matrix", "subtract_frame_transforms", "transform_points", "transform_poses_from_frame_A_to_frame_B", "unmake_pose",
        "unproject_depth", "unscale_transform", "wrap_to_pi", "yaw_quat",
    },
    "isaaclab.utils.noise": {
        "AdditiveGaussianNoiseCfg", "AdditiveUniformNoiseCfg", "ConstantBiasNoiseCfg", "ConstantNoiseCfg", "GaussianNoiseCfg", "NoiseCfg", "NoiseModel",
        "NoiseModelCfg", "NoiseModelWithAdditiveBias", "NoiseModelWithAdditiveBiasCfg", "UniformNoiseCfg", "constant_noise", "gaussian_noise", "uniform_noise",
    },
    "isaaclab_assets.robots.unitree": {"UNITREE_A1_CFG", "UNITREE_GO1_CFG", "UNITREE_GO2_CFG", "GO1_ACTUATOR_CFG", "H1_CFG", "H1_MINIMAL_CFG", "G1_CFG", "G1_MINIMAL_CFG", "G1_29DOF_CFG", "G1_INSPIRE_FTP_CFG"},
    "isaaclab_rl.rsl_rl": {
        "RslRlBaseRunnerCfg", "RslRlDistillationAlgorithmCfg", "RslRlDistillationRunnerCfg", "RslRlDistillationStudentTeacherCfg",
        "RslRlDistillationStudentTeacherRecurrentCfg", "RslRlOnPolicyRunnerCfg", "RslRlPpoActorCriticCfg", "RslRlPpoActorCriticRecurrentCfg",
        "RslRlPpoAlgorithmCfg", "RslRlRndCfg", "RslRlSymmetryCfg", "RslRlVecEnvWrapper", "export_policy_as_jit", "export_policy_as_onnx",
    },
    "isaaclab_tasks.utils": {"get_checkpoint_path", "import_packages", "load_cfg_from_registry", "parse_env_cfg"},
    "isaaclab_tasks.utils.parse_cfg": {"get_checkpoint_path", "load_cfg_from_registry", "parse_env_cfg"},
    "isaaclab_tasks.utils.hydra": {"hydra_task_config", "register_task_to_hydra"},
    "isaaclab_tasks.manager_based.locomotion.velocity.mdp": {
        "*",  # re-exports isaaclab.envs.mdp, plus:
        "feet_air_time", "feet_air_time_positive_biped", "feet_slide", "stand_still_joint_deviation_l1", "terrain_levels_vel", "terrain_out_of_bounds",
        "track_ang_vel_z_world_exp", "track_lin_vel_xy_yaw_frame_exp",
    },
    "rsl_rl.runners": {"DistillationRunner", "OnPolicyRunner"},
    "rsl_rl.env": {"VecEnv"},
}
# attributes used as `sim_utils.X`: the 210 public names of `isaaclab.sim` in v2.3.2 (cfg / spawner / schema / converter classes,
# the sim utils and the submodules), generated from the source tree (`--source ... --dump`), stdlib / pxr re-imports dropped
SIM_UTILS = {
    'ArticulationRootPropertiesCfg', 'AssetConverterBase', 'AssetConverterBaseCfg', 'BoundingCubePropertiesCfg', 'BoundingSpherePropertiesCfg',
    'CapsuleCfg', 'Cloner', 'CollisionPropertiesCfg', 'ConeCfg', 'ConvexDecompositionPropertiesCfg', 'ConvexHullPropertiesCfg', 'CuboidCfg',
    'CylinderCfg', 'CylinderLightCfg', 'DeformableBodyMaterialCfg', 'DeformableBodyPropertiesCfg', 'DeformableObjectSpawnerCfg', 'DiskLightCfg',
    'DistantLightCfg', 'DomeLightCfg', 'FisheyeCameraCfg', 'FixedTendonPropertiesCfg', 'GlassMdlCfg', 'GroundPlaneCfg', 'JointDrivePropertiesCfg',
    'LightCfg', 'MESH_APPROXIMATION_TOKENS', 'MassPropertiesCfg', 'MdlFileCfg', 'MeshCapsuleCfg', 'MeshCfg', 'MeshCollisionPropertiesCfg',
    'MeshConeCfg', 'MeshConverter', 'MeshConverterCfg', 'MeshCuboidCfg', 'MeshCylinderCfg', 'MeshSphereCfg', 'MjcfConverter', 'MjcfConverterCfg',
    'MjcfFileCfg', 'MultiAssetSpawnerCfg', 'MultiUsdFileCfg', 'PHYSX_MESH_COLLISION_CFGS', 'PhysicsMaterialCfg', 'PhysxCfg', 'PinholeCameraCfg',
    'PreviewSurfaceCfg', 'RenderCfg', 'RigidBodyMaterialCfg', 'RigidBodyPropertiesCfg', 'RigidObjectSpawnerCfg', 'SDFMeshPropertiesCfg', 'ShapeCfg',
    'SimulationCfg', 'SimulationContext', 'SpatialTendonPropertiesCfg', 'SpawnerCfg', 'SphereCfg', 'SphereLightCfg', 'TriangleMeshPropertiesCfg',
    'TriangleMeshSimplificationPropertiesCfg', 'USD_MESH_COLLISION_CFGS', 'UrdfConverter', 'UrdfConverterCfg', 'UrdfFileCfg', 'UsdFileCfg',
    'UsdFileWithCompliantContactCfg', 'VisualMaterialCfg', 'XformPrimView', 'activate_contact_sensors', 'add_labels', 'add_reference_to_stage',
    'add_usd_reference', 'apply_nested', 'asset_converter_base', 'asset_converter_base_cfg', 'attach_stage_to_usd_context', 'bind_physics_material',
    'bind_visual_material', 'build_simulation_context', 'change_prim_property', 'check_missing_labels', 'clear_stage', 'clone', 'close_stage',
    'configclass', 'convert_world_pose_to_local', 'converters', 'count_total_labels', 'create_new_stage', 'create_new_stage_in_memory', 'create_prim',
    'define_articulation_root_properties', 'define_collision_properties', 'define_deformable_body_properties', 'define_mass_properties',
    'define_mesh_collision_properties', 'define_prim', 'define_rigid_body_properties', 'delete_prim', 'export_prim_to_file',
    'find_first_matching_prim', 'find_global_fixed_joint_prim', 'find_matching_prim_paths', 'find_matching_prims', 'from_files', 'from_files_cfg',
    'get_all_matching_child_prims', 'get_current_stage', 'get_current_stage_id', 'get_first_matching_ancestor_prim', 'get_first_matching_child_prim',
    'get_isaac_sim_version', 'get_labels', 'get_next_free_path', 'get_next_free_prim_path', 'get_prim_at_path', 'get_prim_path', 'get_prim_type_name',
    'get_stage_up_axis', 'get_usd_references', 'is_current_stage_in_memory', 'is_prim_path_valid', 'is_stage_loading', 'legacy', 'lights',
    'lights_cfg', 'make_uninstanceable', 'materials', 'mesh_converter', 'mesh_converter_cfg', 'meshes', 'meshes_cfg', 'mjcf_converter',
    'mjcf_converter_cfg', 'modify_articulation_root_properties', 'modify_collision_properties', 'modify_deformable_body_properties',
    'modify_fixed_tendon_properties', 'modify_joint_drive_properties', 'modify_mass_properties', 'modify_mesh_collision_properties',
    'modify_rigid_body_properties', 'modify_spatial_tendon_properties', 'move_prim', 'open_stage', 'physics_materials', 'physics_materials_cfg',
    'prims', 'queries', 'remove_labels', 'resolve_prim_pose', 'resolve_prim_scale', 'safe_set_attribute_on_usd_prim',
    'safe_set_attribute_on_usd_schema', 'save_stage', 'schemas', 'select_usd_variants', 'semantics', 'sensors', 'sensors_cfg', 'set_prim_visibility',
    'shapes', 'shapes_cfg', 'sim_stage', 'simulation_cfg', 'simulation_context', 'spawn_camera', 'spawn_capsule', 'spawn_cone', 'spawn_cuboid',
    'spawn_cylinder', 'spawn_deformable_body_material', 'spawn_from_mdl_file', 'spawn_from_mjcf', 'spawn_from_urdf', 'spawn_from_usd',
    'spawn_from_usd_with_compliant_contact_material', 'spawn_ground_plane', 'spawn_light', 'spawn_mesh_capsule', 'spawn_mesh_cone',
    'spawn_mesh_cuboid', 'spawn_mesh_cylinder', 'spawn_mesh_sphere', 'spawn_multi_asset', 'spawn_multi_usd_file', 'spawn_preview_surface',
    'spawn_rigid_body_material', 'spawn_sphere', 'spawner_cfg', 'spawners', 'stage', 'standardize_xform_ops', 'to_camel_case', 'transforms',
    'traverse_stage', 'update_stage', 'urdf_converter', 'urdf_converter_cfg', 'use_stage', 'utils', 'validate_standard_xform_ops', 'views',
    'visual_materials', 'visual_materials_cfg', 'wrappers', 'wrappers_cfg', 'xform_prim_view',
}
# attributes used as `mdp.X` that come from isaaclab.envs.mdp (v2.3.2) or the locomotion mdp
MDP = {
    # observations
    "base_pos_z", "base_lin_vel", "base_ang_vel", "projected_gravity", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "body_pose_w",
    "body_projected_gravity_b", "joint_pos", "joint_pos_rel", "joint_pos_limit_normalized", "joint_vel", "joint_vel_rel", "joint_effort", "height_scan",
    "body_incoming_wrench", "imu_orientation", "imu_projected_gravity", "imu_ang_vel", "imu_lin_acc", "image", "image_features", "last_action",
    "generated_commands", "current_time_s", "remaining_time_s",
    # rewards
    "is_alive", "is_terminated", "is_terminated_term", "lin_vel_z_l2", "ang_vel_xy_l2", "flat_orientation_l2", "base_height_l2", "body_lin_acc_l2",
    "joint_torques_l2", "joint_vel_l1", "joint_vel_l2", "joint_acc_l2", "joint_deviation_l1", "joint_pos_limits", "joint_vel_limits", "applied_torque_limits",
    "action_rate_l2", "action_l2", "undesired_contacts", "desired_contacts", "contact_forces", "track_lin_vel_xy_exp", "track_ang_vel_z_exp",
    # terminations
    "time_out", "command_resample", "bad_orientation", "root_height_below_minimum", "joint_pos_out_of_limit", "joint_pos_out_of_manual_limit",
    "joint_vel_out_of_limit", "joint_vel_out_of_manual_limit", "joint_effort_out_of_limit", "illegal_contact",
    # events
    "randomize_rigid_body_scale", "randomize_rigid_body_material", "randomize_rigid_body_mass", "randomize_rigid_body_com",
    "randomize_rigid_body_collider_offsets", "randomize_physics_scene_gravity", "randomize_actuator_gains", "randomize_joint_parameters",
    "randomize_fixed_tendon_parameters", "apply_external_force_torque", "push_by_setting_velocity", "reset_root_state_uniform",
    "reset_root_state_with_random_orientation", "reset_root_state_from_terrain", "reset_joints_by_scale", "reset_joints_by_offset",
    "reset_nodal_state_uniform", "reset_scene_to_default", "randomize_visual_texture_material", "randomize_visual_color",
    # curriculums
    "modify_reward_weight", "modify_env_param", "modify_term_cfg",
    # actions cfgs
    "JointActionCfg", "JointPositionActionCfg", "RelativeJointPositionActionCfg", "JointVelocityActionCfg", "JointEffortActionCfg",
    "JointPositionToLimitsActionCfg", "EMAJointPositionToLimitsActionCfg", "BinaryJointActionCfg", "BinaryJointPositionActionCfg",
    "BinaryJointVelocityActionCfg", "AbsBinaryJointPositionActionCfg", "NonHolonomicActionCfg", "DifferentialInverseKinematicsActionCfg",
    "OperationalSpaceControllerActionCfg", "SurfaceGripperBinaryActionCfg",
    # commands cfgs
    "NullCommandCfg", "UniformVelocityCommandCfg", "NormalVelocityCommandCfg", "UniformPoseCommandCfg", "UniformPose2dCommandCfg", "TerrainBasedPose2dCommandCfg",
    # recorders
    "ActionStateRecorderManagerCfg", "InitialStateRecorder", "InitialStateRecorderCfg", "PostStepStatesRecorder", "PostStepStatesRecorderCfg",
    "PreStepActionsRecorder", "PreStepActionsRecorderCfg", "PreStepFlatPolicyObservationsRecorder", "PreStepFlatPolicyObservationsRecorderCfg",
    # locomotion mdp (isaaclab_tasks.manager_based.locomotion.velocity.mdp)
    "feet_slide", "feet_air_time_positive_biped", "feet_air_time", "stand_still_joint_deviation_l1", "terrain_levels_vel", "terrain_out_of_bounds",
    "track_ang_vel_z_world_exp", "track_lin_vel_xy_yaw_frame_exp",
}
LOCO_MDP = "isaaclab_tasks.manager_based.locomotion.velocity.mdp"


# ---------------------------------------------------------------------------------------------------- our files
def our_files() -> list[str]:
    files = []
    for d in OUR_DIRS:
        for dp, _, fns in os.walk(d):
            files += [os.path.join(dp, f) for f in fns if f.endswith(".py")]
    return sorted(files)


def our_module_name(path: str) -> str:
    """Dotted module name of one of our files (doorbench_isaaclab.mdp.actions / train)."""
    for base in (os.path.join(ROOT, "isaaclab"), os.path.join(ROOT, "scripts", "isaaclab")):
        if path.startswith(base + os.sep):
            rel = os.path.relpath(path, base)[:-3].replace(os.sep, ".")
            return rel[: -len(".__init__")] if rel.endswith(".__init__") else rel
    return os.path.basename(path)[:-3]


def _flat_body(nodes):
    """Top-level statements including the bodies of if / try blocks (TYPE_CHECKING guards, optional imports)."""
    for n in nodes:
        if isinstance(n, ast.If):
            yield from _flat_body(n.body)
            yield from _flat_body(n.orelse)
        elif isinstance(n, ast.Try):
            yield from _flat_body(n.body)
            for h in n.handlers:
                yield from _flat_body(h.body)
            yield from _flat_body(n.orelse)
            yield from _flat_body(n.finalbody)
        else:
            yield n


def scan(path):
    """imports [(module, name)], attrs {"mdp": {...}, "sim_utils": {...}}, calls [(func_node, [kw names], lineno)],
    aliases {local name: ("module", module) | ("symbol", module, name)}"""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    imports, attrs, calls = [], {"mdp": set(), "sim_utils": set()}, []
    aliases: dict[str, tuple] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.startswith(EXTERNAL_PREFIXES):
            for a in node.names:
                imports.append((node.module, a.name))
                if a.name != "*":
                    aliases[a.asname or a.name] = ("symbol", node.module, a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(EXTERNAL_PREFIXES):
                    aliases[a.asname or a.name.split(".")[0]] = ("module", a.name)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in attrs:
            attrs[node.value.id].add(node.attr)
        elif isinstance(node, ast.Call) and node.keywords:
            kws = [k.arg for k in node.keywords if k.arg is not None]  # None = **splat
            if kws:
                calls.append((node.func, kws, node.lineno))
    for node in _flat_body(tree.body):  # classes defined in this file (for `HandSceneCfg(num_envs=...)`)
        if isinstance(node, ast.ClassDef):
            aliases.setdefault(node.name, ("local", node.name))
    return imports, attrs, calls, aliases


def local_mdp_names() -> set[str]:
    mdp_init = os.path.join(ROOT, "isaaclab", "doorbench_isaaclab", "mdp", "__init__.py")
    with open(mdp_init) as f:
        tree = ast.parse(f.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:  # relative imports = our own terms
            names |= {a.asname or a.name for a in node.names}
    return names


# ------------------------------------------------------------------------------------------------ source resolver
class SourceTree:
    """Resolve modules, exported names, class fields and function parameters in one or more source checkouts (ast only)."""

    def __init__(self, roots: list[str]):
        self.roots: list[str] = []
        for r in roots:
            r = os.path.abspath(r)
            src = os.path.join(r, "source")           # Isaac Lab layout: source/<pkg>/<pkg>/...
            if os.path.isdir(src):
                for d in sorted(os.listdir(src)):
                    if os.path.isdir(os.path.join(src, d, d)):
                        self.roots.append(os.path.join(src, d))
            self.roots.append(r)                      # plain layout: <root>/<pkg>/...
        self._trees: dict[str, ast.Module] = {}
        self._exports: dict[str, tuple[set[str], set[str]] | None] = {}

    # -- files
    def module_path(self, mod: str) -> str | None:
        rel = mod.replace(".", os.sep)
        for r in self.roots:
            for cand in (os.path.join(r, rel, "__init__.py"), os.path.join(r, rel + ".py")):
                if os.path.isfile(cand):
                    return cand
        return None

    def tree(self, path: str) -> ast.Module:
        if path not in self._trees:
            with open(path, encoding="utf-8") as f:
                self._trees[path] = ast.parse(f.read(), path)
        return self._trees[path]

    @staticmethod
    def _relative(mod: str, path: str, level: int, name: str | None) -> str:
        if level == 0:
            return name or ""
        parts = mod.split(".")
        base = parts if os.path.basename(path) == "__init__.py" else parts[:-1]
        if level > 1:
            base = base[: len(base) - (level - 1)]
        return ".".join(base + ([name] if name else []))

    # -- names
    def exports(self, mod: str, _seen: set | None = None):
        """(all attribute names, names a star-import re-exports) of module `mod`, or None if it is not in the tree."""
        if mod in self._exports:
            return self._exports[mod]
        path = self.module_path(mod)
        if path is None:
            self._exports[mod] = None
            return None
        _seen = _seen if _seen is not None else set()
        if mod in _seen:
            return set(), set()
        _seen.add(mod)
        names, all_ = set(), None
        for node in _flat_body(self.tree(path).body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    for n in (t.elts if isinstance(t, ast.Tuple) else [t]):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
                            if n.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                                all_ = {c.value for c in node.value.elts if isinstance(c, ast.Constant)}
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                target = self._relative(mod, path, node.level, node.module)
                for a in node.names:
                    if a.name == "*":
                        sub = self.exports(target, _seen)
                        if sub:
                            names |= sub[1]
                    else:
                        names.add(a.asname or a.name)
                        if node.level and not node.module:  # `from . import x` -> submodule
                            names.add(a.asname or a.name)
        if os.path.basename(path) == "__init__.py":  # submodules are attributes once imported by the package itself
            pkg_dir = os.path.dirname(path)
            for fn in os.listdir(pkg_dir):
                if fn.endswith(".py") and fn != "__init__.py":
                    names.add(fn[:-3])
                elif os.path.isfile(os.path.join(pkg_dir, fn, "__init__.py")):
                    names.add(fn)
        star = set(all_) if all_ is not None else {n for n in names if not n.startswith("_")}
        self._exports[mod] = (names, star)
        return self._exports[mod]

    def resolve(self, mod: str, name: str, _seen: set | None = None):
        """(defining module, ast node) of a class / function / assignment named `name` reachable from `mod`, else None."""
        _seen = _seen if _seen is not None else set()
        if (mod, name) in _seen:
            return None
        _seen.add((mod, name))
        path = self.module_path(mod)
        if path is None:
            return None
        body = list(_flat_body(self.tree(path).body))
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return mod, node
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                if isinstance(node.value, ast.Name):  # alias: AdditiveUniformNoiseCfg = UniformNoiseCfg
                    r = self.resolve(mod, node.value.id, _seen)
                    return r if r else (mod, node)
                return mod, node
        for node in body:
            if isinstance(node, ast.ImportFrom):
                target = self._relative(mod, path, node.level, node.module)
                for a in node.names:
                    if a.name == "*":
                        ex = self.exports(target)
                        if ex and name in ex[1]:
                            r = self.resolve(target, name, _seen)
                            if r:
                                return r
                    elif (a.asname or a.name) == name:
                        if node.level and not node.module and self.module_path(f"{target}.{a.name}"):
                            return f"{target}.{a.name}", None  # a submodule
                        return self.resolve(target, a.name, _seen)
        return None

    def resolve_attr(self, mod: str, node: ast.AST, attr: str):
        """Attribute `attr` of a resolved thing: nested class of a class, or symbol of a module."""
        if node is None:  # module
            return self.resolve(mod, attr)
        if isinstance(node, ast.ClassDef):
            for n in node.body:
                if isinstance(n, ast.ClassDef) and n.name == attr:
                    return mod, n
        return None

    def class_fields(self, mod: str, cls: ast.ClassDef, _seen: set | None = None) -> set[str] | None:
        """Field / attribute names of a (config) class including its bases; None when a base cannot be resolved."""
        _seen = _seen if _seen is not None else set()
        if (mod, cls.name) in _seen:
            return set()
        _seen.add((mod, cls.name))
        fields: set[str] = set()
        for n in cls.body:
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                fields.add(n.target.id)
            elif isinstance(n, ast.Assign):
                fields |= {t.id for t in n.targets if isinstance(t, ast.Name)}
            elif isinstance(n, ast.ClassDef):
                fields.add(n.name)
            elif isinstance(n, ast.FunctionDef) and n.name == "__init__":
                for a in n.args.args[1:] + n.args.kwonlyargs:  # plain classes: __init__ parameters are the "fields"
                    fields.add(a.arg)
                if n.args.kwarg is not None:
                    return None  # accepts anything
        for b in cls.bases:
            if isinstance(b, ast.Name):
                if b.id == "object":
                    continue
                r = self.resolve(mod, b.id)
            elif isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name):
                outer = self.resolve(mod, b.value.id)
                r = self.resolve_attr(*outer, b.attr) if outer else None
            else:
                r = None
            if not r or not isinstance(r[1], ast.ClassDef):
                return None
            sub = self.class_fields(r[0], r[1], _seen)
            if sub is None:
                return None
            fields |= sub
        return fields

    @staticmethod
    def func_params(fn: ast.FunctionDef) -> set[str] | None:
        if fn.args.kwarg is not None:
            return None
        return {a.arg for a in fn.args.args + fn.args.kwonlyargs}


def check_calls(tree: SourceTree, path: str, calls, aliases, local_mdp: set[str]):
    """Keyword arguments of calls to Isaac Lab / rsl_rl / our own config classes and functions vs their definitions."""
    problems, n = [], 0
    mod_self = our_module_name(path)
    ex_mdp = tree.exports("isaaclab.envs.mdp")
    ex_loco = tree.exports(LOCO_MDP)
    for func, kws, lineno in calls:
        target = None
        if isinstance(func, ast.Name):
            al = aliases.get(func.id)
            if al is None:
                continue
            if al[0] == "symbol":
                target = tree.resolve(al[1], al[2])
            elif al[0] == "local":
                target = tree.resolve(mod_self, al[1])
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            base, attr = func.value.id, func.attr
            if base == "mdp" and attr not in local_mdp:
                if ex_mdp and attr in ex_mdp[0]:
                    target = tree.resolve("isaaclab.envs.mdp", attr)
                elif ex_loco and attr in ex_loco[0]:
                    target = tree.resolve(LOCO_MDP, attr)
            else:
                al = aliases.get(base)
                if al is None:
                    continue
                if al[0] == "module":
                    target = tree.resolve(al[1], attr)
                elif al[0] == "symbol":
                    outer = tree.resolve(al[1], al[2])
                    target = tree.resolve_attr(*outer, attr) if outer else None
                elif al[0] == "local":
                    outer = tree.resolve(mod_self, al[1])
                    target = tree.resolve_attr(*outer, attr) if outer else None
        if not target or target[1] is None:
            continue
        mod, node = target
        if isinstance(node, ast.ClassDef):
            allowed = tree.class_fields(mod, node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            allowed = tree.func_params(node)
        else:
            continue
        if allowed is None:
            continue
        n += 1
        bad = [k for k in kws if k not in allowed]
        if bad:
            problems.append(f"{os.path.relpath(path, ROOT)}:{lineno}: {mod}.{node.name}(...) has no {'field' if isinstance(node, ast.ClassDef) else 'parameter'} {bad}")
    return problems, n


# ---------------------------------------------------------------------------------------------------------- main
def run_offline(files, local_mdp):
    unknown, n = [], 0
    for fp in files:
        imports, attrs, _, _ = scan(fp)
        for mod, name in imports:
            n += 1
            known = KNOWN.get(mod)
            if known is None:
                unknown.append((os.path.relpath(fp, ROOT), f"module {mod} not in the reference list"))
            elif "*" not in known and name not in known:
                unknown.append((os.path.relpath(fp, ROOT), f"{mod}.{name}"))
        for a in attrs["mdp"]:
            n += 1
            if a not in MDP and a not in local_mdp:
                unknown.append((os.path.relpath(fp, ROOT), f"mdp.{a}"))
        for a in attrs["sim_utils"]:
            n += 1
            if a not in SIM_UTILS:
                unknown.append((os.path.relpath(fp, ROOT), f"sim_utils.{a}"))
    return unknown, n


def run_against_source(files, local_mdp, roots):
    tree = SourceTree(roots + [os.path.join(ROOT, "isaaclab"), os.path.join(ROOT, "scripts", "isaaclab")])
    for probe in ("isaaclab", "isaaclab_rl", "isaaclab_tasks"):
        if tree.module_path(probe) is None:
            print(f"  WARNING: package {probe} not found under {roots}")
    unknown, n, n_calls = [], 0, 0
    ex_mdp = tree.exports("isaaclab.envs.mdp")
    ex_loco = tree.exports(LOCO_MDP)
    ex_sim = tree.exports("isaaclab.sim")
    for fp in files:
        imports, attrs, calls, aliases = scan(fp)
        rel = os.path.relpath(fp, ROOT)
        for mod, name in imports:
            n += 1
            ex = tree.exports(mod)
            if ex is None:
                if any(tree.module_path(p) for p in (mod.split(".")[0],)):
                    unknown.append((rel, f"module {mod} not found in the source tree"))
                continue  # package not provided (e.g. rsl_rl without --source): skip silently
            if name != "*" and name not in ex[0] and tree.module_path(f"{mod}.{name}") is None:
                unknown.append((rel, f"{mod}.{name}"))
        for a in attrs["mdp"]:
            n += 1
            if a in local_mdp:
                continue
            if not ((ex_mdp and a in ex_mdp[0]) or (ex_loco and a in ex_loco[0])):
                unknown.append((rel, f"mdp.{a}"))
        for a in attrs["sim_utils"]:
            n += 1
            if ex_sim and a not in ex_sim[0]:
                unknown.append((rel, f"sim_utils.{a}"))
        probs, k = check_calls(tree, fp, calls, aliases, local_mdp)
        n_calls += k
        unknown += [(rel, p) for p in probs]
    return unknown, n, n_calls, tree


def dump_reference(tree: SourceTree):
    def show(title, names):
        print(f"  {title}: {len(names)}")
        print("    " + ", ".join(sorted(names)))

    print("# public names per module (v-source)")
    for mod in sorted(KNOWN):
        ex = tree.exports(mod)
        if ex is None:
            print(f"  {mod}: NOT FOUND")
            continue
        show(mod, {x for x in ex[0] if not x.startswith("_")})
    for mod, title in (("isaaclab.sim", "SIM_UTILS"), ("isaaclab.envs.mdp", "MDP (isaaclab.envs.mdp)"), (LOCO_MDP, "MDP (locomotion)")):
        ex = tree.exports(mod)
        if ex:
            show(title, {x for x in ex[0] if not x.startswith("_")})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", action="append", default=[], help="Isaac Lab checkout (source/<pkg>/<pkg>) or a package root such as an rsl_rl clone; repeatable")
    ap.add_argument("--dump", action="store_true", help="with --source: print the public names of every module in the reference list")
    args = ap.parse_args()
    files = our_files()
    local_mdp = local_mdp_names()

    unknown, n = run_offline(files, local_mdp)
    print(f"checked {n} Isaac Lab symbol references in {len(files)} files against the v2.3.2 reference list")
    for fp, name in unknown:
        print(f"  UNKNOWN {fp}: {name}")
    if args.source:
        unknown_src, n_src, n_calls, tree = run_against_source(files, local_mdp, args.source)
        print(f"checked {n_src} symbol references and the keyword arguments of {n_calls} calls against {args.source}")
        for fp, name in unknown_src:
            print(f"  UNKNOWN(source) {fp}: {name}")
        if args.dump:
            dump_reference(tree)
        unknown += unknown_src
    if unknown:
        sys.exit(1)
    print("all symbols known")


if __name__ == "__main__":
    main()

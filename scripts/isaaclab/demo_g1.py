#!/usr/bin/env python
"""Run a researcher policy on a physical G1 in Isaac Sim; save every trial, including failures."""

from __future__ import annotations
import argparse, hashlib, importlib, json, math, re, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--assets", default="out/isaac-g1-demo/assets")
p.add_argument("--door", default="db0123_saloon")
p.add_argument("--duration", type=float, default=16.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="out/isaac-g1-demo/results")
p.add_argument("--policy", default="robot_demo.isaac_policy_adapter:unitree_factory")
p.add_argument(
    "--checkpoint",
    default=str(
        ROOT / "robot_demo/third_party/unitree_g1_policy/deploy/pre_train/g1/motion.pt"
    ),
)
p.add_argument(
    "--policy-config",
    default=str(
        ROOT
        / "robot_demo/third_party/unitree_g1_policy/deploy/deploy_mujoco/configs/g1.yaml"
    ),
)
p.add_argument("--video", action="store_true")
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = args.enable_cameras or args.video
app = AppLauncher(args).app
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import build_simulation_context
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG
from pxr import Usd
from robot_demo.g1_policy import LEG_JOINTS
from robot_demo.isaac_policy_adapter import validate_action

sys.path.insert(0, str(ROOT / "scripts/isaaclab"))
from _common import simulator_engine

sys.path.insert(0, str(ROOT / "isaaclab"))
from doorbench_isaaclab.assets import (
    DOOR_ACTUATORS,
    DOOR_ARTICULATION_PROPS,
    DOOR_RIGID_PROPS,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    import yaml

    assets = Path(args.assets).resolve()
    folder = assets / "doors" / args.door
    cases = json.loads((assets / "demo-suite.json").read_text())["cases"]
    case = next((c for c in cases if c["id"] == args.door), None)
    if case is None:
        raise ValueError("Door must be explicitly declared in demo-suite.json")
    spec = json.loads((folder / "spec.json").read_text())
    stage = Usd.Stage.Open(str(folder / "door_rl.usda"))
    rl = json.loads(stage.GetDefaultPrim().GetAttribute("doorbench:rl").Get())
    del stage
    cfg = yaml.safe_load(Path(args.policy_config).read_text())
    dt = float(cfg["simulation_dt"])
    decimation = int(cfg["control_decimation"])
    if dt <= 0 or decimation < 1 or args.duration <= 0:
        raise ValueError("Positive simulation duration and policy cadence required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.door}-seed{args.seed}"
    started = time.monotonic()
    trace = []
    writer = None
    result = {
        "schema_version": 1,
        "door_id": args.door,
        "seed": args.seed,
        "task": case["task"],
        "description": case["description"],
        "policy": args.policy,
        "checkpoint_sha256": digest(args.checkpoint),
        "policy_config_sha256": digest(args.policy_config),
        "source_sha256": {
            n: digest(folder / n) for n in ("spec.json", "model.json", "door_rl.usda")
        },
        "engine": simulator_engine(),
        "runner_source_sha256": {
            name: digest(ROOT / name)
            for name in (
                "scripts/isaaclab/demo_g1.py",
                "robot_demo/g1_policy.py",
                "robot_demo/isaac_policy_adapter.py",
            )
        },
        "profile": "canonical_7_joint_usd",
        "scope": "Researcher integration demo: upright traversal, not a full DoorBench safety/damage benchmark or manipulation policy",
        "initial_open_fraction": case["initial_open_fraction"],
        "physics_dt_s": dt,
        "policy_dt_s": dt * decimation,
        "door_state_writes_during_episode": 0,
        "policy_direct_door_actuation": False,
        "success": False,
    }
    import warp

    result["engine"].update(
        numpy=np.__version__, warp=warp.__version__, torch_cuda=torch.version.cuda
    )
    if args.device.startswith("cuda"):
        result["hardware"] = {
            "gpu": torch.cuda.get_device_name(args.device),
            "gpu_memory_bytes": torch.cuda.get_device_properties(
                args.device
            ).total_memory,
        }
    with build_simulation_context(
        device=args.device,
        dt=dt,
        gravity_enabled=True,
        # Canonical door_rl.usda deliberately omits the floor.
        add_ground_plane=True,
        auto_add_lighting=True,
    ) as sim:
        if getattr(sim, "_app_control_on_stop_handle", None) is not None:
            sim._app_control_on_stop_handle.unsubscribe()
            sim._app_control_on_stop_handle = None
        door_props = DOOR_ARTICULATION_PROPS.copy()
        door_props.enabled_self_collisions = bool(rl.get("self_collisions", True))
        door = Articulation(
            ArticulationCfg(
                prim_path="/World/Door",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(folder / "door_rl.usda"),
                    activate_contact_sensors=False,
                    rigid_props=DOOR_RIGID_PROPS,
                    articulation_props=door_props,
                ),
                actuators=DOOR_ACTUATORS,
                articulation_root_prim_path="/Articulation",
            )
        )
        robot_cfg = G1_MINIMAL_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        robot_cfg.init_state.pos = (0.0, -1.5, 0.79)
        robot_cfg.init_state.rot = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
        # Isaac rejects overlapping regex and exact-name initial-state assignments.
        robot_cfg.init_state.joint_pos = {
            pattern: value
            for pattern, value in robot_cfg.init_state.joint_pos.items()
            if not any(re.fullmatch(pattern, name) for name in LEG_JOINTS)
        }
        for name, q in zip(LEG_JOINTS, cfg["default_angles"]):
            robot_cfg.init_state.joint_pos[name] = float(q)
        # Explicit PD uses the checkpoint deployment gains; PhysX receives only robot efforts.
        for actuator in robot_cfg.actuators.values():
            actuator.stiffness = 0.0
            actuator.damping = 0.0
        robot = Articulation(robot_cfg)
        camera = None
        if args.video:
            from isaaclab.sensors import Camera, CameraCfg

            camera = Camera(
                CameraCfg(
                    prim_path="/World/DemoCamera",
                    height=720,
                    width=960,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.1, 100.0),
                    ),
                )
            )
        sim.reset()
        robot.update(dt)
        door.update(dt)
        device = robot.device
        names = list(robot.joint_names)
        leg_ids = [names.index(n) for n in LEG_JOINTS]
        q0 = robot.data.default_joint_pos.clone()
        q0[0, leg_ids] = torch.tensor(cfg["default_angles"], device=device)
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        robot.reset()
        door_names = list(door.joint_names)
        pj = door_names.index(rl["door_joint"])
        sj = (
            door_names.index(rl["secondary_slot_joint"])
            if rl.get("secondary_slot_joint")
            else None
        )
        dq0 = door.data.default_joint_pos.clone()
        dq0[0, pj] = (
            float(case["initial_open_fraction"])
            * rl["joints"][rl["door_joint"]]["range"][1]
        )
        door.write_joint_state_to_sim(dq0, torch.zeros_like(dq0))
        door.reset()
        targets = torch.zeros_like(dq0)
        friction = torch.zeros_like(dq0)
        for name, row in rl["joints"].items():
            i = door_names.index(name)
            targets[0, i] = row.get("drive", {}).get("target", row.get("target", 0.0))
            friction[0, i] = row.get("friction", 0.0)
        door.write_joint_friction_coefficient_to_sim(
            friction,
            joint_dynamic_friction_coeff=friction.clone(),
            joint_viscous_friction_coeff=torch.zeros_like(friction),
        )
        result["door_friction_effort_readback"] = (
            door.data.joint_friction_coeff[0].cpu().numpy().tolist()
        )
        result["door_joint_names"] = door_names
        result["door_drive_stiffness"] = (
            door.data.joint_stiffness[0].cpu().numpy().tolist()
        )
        result["door_drive_damping"] = door.data.joint_damping[0].cpu().numpy().tolist()
        result["door_effort_limits"] = (
            door.data.joint_effort_limits[0].cpu().numpy().tolist()
        )
        kp = torch.full_like(q0, 40.0)
        kd = torch.full_like(q0, 3.0)
        kp[0, leg_ids] = torch.tensor(cfg["kps"], device=device, dtype=kp.dtype)
        kd[0, leg_ids] = torch.tensor(cfg["kds"], device=device, dtype=kd.dtype)
        # Preserve the asset's published effort caps even for custom torque policies.
        limits = robot.data.joint_effort_limits.clone()
        context = {
            "joint_names": names,
            "default_joint_positions": q0[0].cpu().numpy().tolist(),
            "checkpoint": args.checkpoint,
            "config_path": args.policy_config,
            "seed": args.seed,
            "policy_dt_s": dt * decimation,
            "physics_dt_s": dt,
            "device": str(device),
        }
        module, factory = args.policy.split(":", 1)
        policy = getattr(importlib.import_module(module), factory)(context)
        result["joint_names"] = names
        result["robot_asset"] = robot_cfg.spawn.usd_path
        result["robot_usd_layers_sha256"] = {
            layer.identifier: hashlib.sha256(
                layer.ExportToString().encode()
            ).hexdigest()
            for layer in sim.stage.GetUsedLayers()
            if "/Robots/Unitree/G1/" in layer.identifier
        }
        result["ground_plane"] = {
            "provided_by": "Isaac Lab GroundPlaneCfg",
            "height_m": 0.0,
        }

        result["robot_effort_limits_Nm"] = limits[0].cpu().numpy().tolist()
        action_mode = "joint_positions"
        command = q0.clone()
        max_open = 0.0
        fell = False
        passed = False
        success_since = None
        peak_effort = 0.0
        auto = spec["kinematics"].get("actuator", {})
        auto_target = 0.0
        last_seen = -1e9
        if camera:
            camera.set_world_poses_from_view(
                torch.tensor([[3.7, -4.8, 2.8]], device=device),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            )
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(out / (stem + ".mp4")), fps=25, codec="libx264", quality=7
            )
        next_frame = 0.0
        for step in range(round(args.duration / dt)):
            t = step * dt
            pos = robot.data.root_pos_w[0].cpu().numpy()
            quat = robot.data.root_quat_w[0].cpu().numpy()
            yaw = math.atan2(
                2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
            )
            heading = math.atan2(1.5 - pos[1], -pos[0])
            error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
            velocity = [
                0.5 if t >= 1.0 and pos[1] < 1.5 else 0.0,
                0.0,
                float(np.clip(1.5 * error, -0.7, 0.7)) if t >= 1.0 else 0.0,
            ]
            if step % decimation == 0:
                obs = {
                    "time_s": t,
                    "base_position_world": pos.tolist(),
                    "base_quaternion_wxyz": quat.tolist(),
                    "base_angular_velocity_body": robot.data.root_ang_vel_b[0]
                    .cpu()
                    .numpy()
                    .tolist(),
                    "base_linear_velocity_body": robot.data.root_lin_vel_b[0]
                    .cpu()
                    .numpy()
                    .tolist(),
                    "joint_positions": robot.data.joint_pos[0].cpu().numpy().tolist(),
                    "joint_velocities": robot.data.joint_vel[0].cpu().numpy().tolist(),
                    "door_joint_positions": dict(
                        zip(door_names, door.data.joint_pos[0].cpu().numpy().tolist())
                    ),
                    "command_velocity": velocity,
                    "goal_position_world": [0.0, 1.5, 0.0],
                }
                action_mode, values = validate_action(policy(obs), len(names))
                command = torch.tensor(values, device=device).unsqueeze(0)
                trace.append(
                    {**obs, "action_mode": action_mode, "robot_action": values.tolist()}
                )
            efforts = (
                (kp * (command - robot.data.joint_pos) - kd * robot.data.joint_vel)
                if action_mode == "joint_positions"
                else command
            )
            efforts = torch.maximum(torch.minimum(efforts, limits), -limits)
            peak_effort = max(peak_effort, float(efforts.abs().max()))
            robot.set_joint_effort_target(efforts)
            if spec["family"] == "automatic_sliding":
                radius = float(auto.get("sensor_range_m", 1.8))
                hold = float(auto.get("hold_open_s", 2.0))
                if np.linalg.norm(pos[:2]) < radius:
                    last_seen = t
                desired = (
                    rl["joints"][rl["door_joint"]]["range"][1]
                    if t - last_seen < hold
                    else 0.0
                )
                speed = float(
                    auto.get("open_speed_m_s", 0.3)
                    if desired > auto_target
                    else auto.get("close_speed_m_s", 0.25)
                )
                auto_target += float(
                    np.clip(desired - auto_target, -speed * dt, speed * dt)
                )
                targets[0, pj] = auto_target
                if sj is not None:
                    targets[0, sj] = auto_target
            door.set_joint_position_target(targets)
            door.set_joint_velocity_target(torch.zeros_like(targets))
            robot.write_data_to_sim()
            door.write_data_to_sim()
            sim.step(render=bool(camera and t >= next_frame))
            robot.update(dt)
            door.update(dt)
            if (
                not torch.isfinite(robot.data.root_state_w).all()
                or not torch.isfinite(door.data.joint_pos).all()
            ):
                result["failure_reason"] = "nonfinite_native_state"
                break
            pos = robot.data.root_pos_w[0].cpu().numpy()
            quat = robot.data.root_quat_w[0].cpu().numpy()
            tilt_z = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
            fell = bool(pos[2] < 0.45 or tilt_z < 0.5)
            max_open = max(max_open, abs(float(door.data.joint_pos[0, pj])))
            within = abs(float(pos[0])) < max(0.0, spec["opening"]["width"] / 2 - 0.20)
            passed = passed or bool(pos[1] > 0.3 and within and not fell)
            at_goal = bool(pos[1] >= 1.2 and within and not fell)
            success_since = (
                t
                if at_goal and success_since is None
                else success_since
                if at_goal
                else None
            )
            if camera and t >= next_frame:
                camera.update(dt, force_recompute=True)
                writer.append_data(camera.data.output["rgb"][0, :, :, :3].cpu().numpy())
                next_frame += 1 / 25
            if step % max(1, round(1.0 / dt)) == 0:
                print(
                    "G1_PROGRESS",
                    json.dumps(
                        {
                            "time_s": round(t, 3),
                            "position": pos.tolist(),
                            "primary_open": float(door.data.joint_pos[0, pj]),
                        }
                    ),
                    flush=True,
                )
            if fell:
                result["failure_reason"] = "robot_fell"
                break
            if success_since is not None and t - success_since >= 0.5:
                result["success"] = True
                break
        if writer:
            writer.close()
        result.update(
            robot_fell=fell,
            passed_plane=passed,
            final_position_world=pos.tolist(),
            elapsed_sim_s=(step + 1) * dt,
            max_primary_open_rad_or_m=max_open,
            peak_robot_effort_Nm=peak_effort,
            elapsed_wall_s=time.monotonic() - started,
            automatic_sensor_control=spec["family"] == "automatic_sliding",
            failure_reason=result.get("failure_reason")
            or (None if result["success"] else "goal_not_reached_before_timeout"),
        )
        result["trace"] = stem + ".trace.json"
        (out / result["trace"]).write_text(json.dumps(trace) + "\n")
        result["trace_sha256"] = digest(out / result["trace"])
        (out / (stem + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        print(
            "DOORBENCH_G1_RESULT "
            + json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("joint_names", "robot_effort_limits_Nm")
                }
            ),
            flush=True,
        )
    return result


if __name__ == "__main__":
    try:
        run()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        app.close()
        raise
    else:
        app.close()

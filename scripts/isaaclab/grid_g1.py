#!/usr/bin/env python3
"""Simultaneous physically independent G1 trials, with an optional 4x4 hero camera.
Uses the tested demo's checkpoint, gains, actions and upright-traversal criterion.
A uniform closed-start diagnostic is not assigned DoorBench task success.
"""

from demo_g1 import *
from datetime import datetime, timezone
import yaml
from robot_demo.isaac_policy_adapter import unitree_factory


class InstanceData:
    def __init__(self, parent, index):
        self.parent, self.index = parent, index

    def __getattr__(self, name):
        value = getattr(self.parent.data, name)
        return (
            value[self.index : self.index + 1]
            if isinstance(value, torch.Tensor)
            else value
        )


class Instance:
    """Initialization-only view; episode stepping uses the full batch directly."""

    def __init__(self, parent, index):
        self.parent, self.index = parent, index
        self.data = InstanceData(parent, index)
        self.device = parent.device
        self.joint_names = parent.joint_names

    def update(self, dt):
        pass

    def reset(self):
        self.parent.reset(torch.tensor([self.index], device=self.device))

    def write_joint_state_to_sim(self, *args, **kwargs):
        self.parent.write_joint_state_to_sim(
            *args, env_ids=torch.tensor([self.index], device=self.device), **kwargs
        )

    def write_joint_friction_coefficient_to_sim(self, *args, **kwargs):
        self.parent.write_joint_friction_coefficient_to_sim(
            *args, env_ids=torch.tensor([self.index], device=self.device), **kwargs
        )


def main(spacing=12.0, presentation=False):
    ids = args.batch_doors or [args.door]
    assets = Path(args.assets).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = {
        r["id"]: r
        for r in json.loads((assets / "demo-suite.json").read_text())["cases"]
    }
    cfg = yaml.safe_load(Path(args.policy_config).read_text())
    dt = float(cfg["simulation_dt"])
    decimation = int(cfg["control_decimation"])
    start = datetime.now(timezone.utc).isoformat()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rows = []
    side = math.ceil(math.sqrt(len(ids)))
    with build_simulation_context(
        device=args.device,
        dt=dt,
        gravity_enabled=True,
        add_ground_plane=True,
        auto_add_lighting=True,
        add_lighting=bool(args.video),
    ) as sim:
        if getattr(sim, "_app_control_on_stop_handle", None) is not None:
            sim._app_control_on_stop_handle.unsubscribe()
            sim._app_control_on_stop_handle = None
        if presentation:
            from hero_style import apply_hero_floor

            apply_hero_floor(sim.stage)
        for k, id in enumerate(ids):
            folder = assets / "doors" / id
            spec = json.loads((folder / "spec.json").read_text())
            stage = Usd.Stage.Open(str(folder / "door_rl.usda"))
            rl = json.loads(stage.GetDefaultPrim().GetAttribute("doorbench:rl").Get())
            del stage
            offset = (
                (k % side - (side - 1) / 2) * spacing,
                (k // side - (side - 1) / 2) * spacing,
            )
            prop = DOOR_ARTICULATION_PROPS.copy()
            prop.enabled_self_collisions = bool(rl.get("self_collisions", True))
            dc = ArticulationCfg(
                prim_path=f"/World/Door{k}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(folder / "door_rl.usda"),
                    rigid_props=DOOR_RIGID_PROPS,
                    articulation_props=prop,
                ),
                actuators=DOOR_ACTUATORS,
                articulation_root_prim_path="/Articulation",
            )
            dc.init_state.pos = (offset[0], offset[1], 0.0)
            dc.spawn.func(
                dc.prim_path,
                dc.spawn,
                translation=dc.init_state.pos,
                orientation=dc.init_state.rot,
            )
            rc = G1_MINIMAL_CFG.copy()
            rc.prim_path = f"/World/Robot{k}"
            rc.init_state.pos = (offset[0], offset[1] - 1.5, 0.79)
            rc.init_state.rot = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
            rc.init_state.joint_pos = {
                p: v
                for p, v in rc.init_state.joint_pos.items()
                if not any(re.fullmatch(p, n) for n in LEG_JOINTS)
            }
            rc.init_state.joint_pos.update(
                dict(zip(LEG_JOINTS, map(float, cfg["default_angles"])))
            )
            for actuator in rc.actuators.values():
                actuator.stiffness = 0.0
                actuator.damping = 0.0
            rc.spawn.func(
                rc.prim_path,
                rc.spawn,
                translation=rc.init_state.pos,
                orientation=rc.init_state.rot,
            )
            rows.append(
                dict(
                    id=id,
                    folder=folder,
                    spec=spec,
                    rl=rl,
                    offset=np.array([*offset, 0.0]),
                    ordinal=k,
                    max_open=0.0,
                    success_since=None,
                    done=False,
                    auto_target=0.0,
                    last_seen=-1e9,
                    trace=[],
                )
            )
        dc.prim_path = "/World/Door.*"
        dc.spawn = None
        rc.prim_path = "/World/Robot.*"
        rc.spawn = None
        door_all = Articulation(dc)
        robot_all = Articulation(rc)
        camera = None
        writer = None
        if args.video:
            from isaaclab.sensors import Camera, CameraCfg

            camera = Camera(
                CameraCfg(
                    prim_path="/World/HeroCamera",
                    height=1080,
                    width=1920,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=28.0 if presentation else 24.0,
                        horizontal_aperture=36.0,
                        clipping_range=(0.1, 200.0),
                    ),
                )
            )
        sim.reset()
        robot_all.update(dt)
        door_all.update(dt)
        for row in rows:
            k = row["ordinal"]
            ri = next(
                i
                for i, path in enumerate(robot_all.root_physx_view.prim_paths)
                if path == f"/World/Robot{k}" or path.startswith(f"/World/Robot{k}/")
            )
            di = next(
                i
                for i, path in enumerate(door_all.root_physx_view.prim_paths)
                if path == f"/World/Door{k}" or path.startswith(f"/World/Door{k}/")
            )
            row.update(
                ri=ri, di=di, robot=Instance(robot_all, ri), door=Instance(door_all, di)
            )
        for row in rows:
            robot, door, rl = row["robot"], row["door"], row["rl"]
            robot.update(dt)
            door.update(dt)
            device = robot.device
            names = list(robot.joint_names)
            leg_ids = [names.index(n) for n in LEG_JOINTS]
            q = robot.data.default_joint_pos.clone()
            q[0, leg_ids] = torch.tensor(
                cfg["default_angles"], device=device, dtype=q.dtype
            )
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.reset()
            dn = list(door.joint_names)
            pj = dn.index(rl["door_joint"])
            dq = door.data.default_joint_pos.clone()
            dq[0, pj] = (
                float(cases[row["id"]]["initial_open_fraction"])
                * rl["joints"][rl["door_joint"]]["range"][1]
            )
            door.write_joint_state_to_sim(dq, torch.zeros_like(dq))
            door.reset()
            targets = torch.zeros_like(dq)
            friction = torch.zeros_like(dq)
            for n, r in rl["joints"].items():
                i = dn.index(n)
                targets[0, i] = r.get("drive", {}).get("target", r.get("target", 0.0))
                friction[0, i] = r.get("friction", 0.0)
            door.write_joint_friction_coefficient_to_sim(
                friction,
                joint_dynamic_friction_coeff=friction.clone(),
                joint_viscous_friction_coeff=torch.zeros_like(friction),
            )
            kp = torch.full_like(q, 40.0)
            kd = torch.full_like(q, 3.0)
            kp[0, leg_ids] = torch.tensor(cfg["kps"], device=device, dtype=q.dtype)
            kd[0, leg_ids] = torch.tensor(cfg["kds"], device=device, dtype=q.dtype)
            policy = unitree_factory(
                dict(
                    joint_names=names,
                    default_joint_positions=q[0].cpu().numpy().tolist(),
                    checkpoint=args.checkpoint,
                    config_path=args.policy_config,
                )
            )
            row.update(
                names=names,
                door_names=dn,
                pj=pj,
                sj=dn.index(rl["secondary_slot_joint"])
                if rl.get("secondary_slot_joint")
                else None,
                kp=kp,
                kd=kd,
                limits=robot.data.joint_effort_limits.clone(),
                policy=policy,
                command=q,
                targets=targets,
            )
        kp_all = torch.zeros_like(robot_all.data.joint_pos)
        kd_all = kp_all.clone()
        commands = kp_all.clone()
        limits_all = robot_all.data.joint_effort_limits.clone()
        targets_all = torch.zeros_like(door_all.data.joint_pos)
        for row in rows:
            ri, di = row["ri"], row["di"]
            kp_all[ri] = row["kp"][0]
            kd_all[ri] = row["kd"][0]
            commands[ri] = row["command"][0]
            targets_all[di] = row["targets"][0]
            row["targets"] = targets_all[di : di + 1]
        commands_np = commands.cpu().numpy().copy()
        initial_positions = robot_all.data.root_pos_w.cpu().numpy()
        for row in rows:
            expected = row["offset"] + np.array([0.0, -1.5, 0.79])
            if np.linalg.norm(initial_positions[row["ri"]] - expected) > 0.01:
                raise RuntimeError(
                    f"Incorrect robot-to-door origin mapping: {row['id']}"
                )
        if camera:
            extent = (side - 1) * spacing + 7.5 if presentation else side * spacing
            camera.set_world_poses_from_view(
                torch.tensor(
                    [[extent * 0.55, -extent * 0.85, extent * 0.80]], device=args.device
                ),
                torch.tensor([[0.0, 0.0, 0.7]], device=args.device),
            )
            import imageio.v2 as imageio

            writer = imageio.get_writer(
                str(out / "g1-grid.mp4"),
                fps=25,
                codec="libx264",
                quality=8,
                macro_block_size=8,
            )

        def save(row, t, success, reason, pos):
            row["done"] = True
            trace = out / (row["id"] + ".trace.json")
            trace.write_text(json.dumps(row["trace"]) + "\n")
            result = dict(
                door_id=row["id"],
                seed=args.seed,
                success=success,
                failure_reason=reason,
                elapsed_sim_s=t,
                final_position_local=pos.tolist(),
                max_primary_open_rad_or_m=row["max_open"],
                started_at_utc=start,
                completed_at_utc=datetime.now(timezone.utc).isoformat(),
                scope="Uniform closed-start upright traversal in canonical USD; not assigned core task success",
                initial_open_fraction=cases[row["id"]]["initial_open_fraction"],
                source_sha256={
                    n: digest(row["folder"] / n)
                    for n in ("spec.json", "model.json", "door_rl.usda")
                },
                checkpoint_sha256=digest(args.checkpoint),
                runner_sha256=digest(__file__),
                supporting_source_sha256={
                    name: digest(ROOT / name)
                    for name in (
                        "scripts/isaaclab/demo_g1.py",
                        "robot_demo/g1_policy.py",
                        "robot_demo/isaac_policy_adapter.py",
                        "isaaclab/doorbench_isaaclab/assets.py",
                    )
                },
                engine=simulator_engine(),
                gpu=torch.cuda.get_device_name(),
                trace_sha256=digest(trace),
                policy_direct_door_actuation=False,
                grid_spacing_m=spacing,
                presentation=bool(presentation),
                presentation_source_sha256={
                    name: digest(ROOT / "scripts/isaaclab" / name)
                    for name in ("hero_g1.py", "hero_style.py")
                }
                if presentation
                else {},
                policy_config_sha256=digest(args.policy_config),
                door_state_writes_during_episode=0,
            )
            (out / (row["id"] + ".json")).write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(
                "GRID_RESULT",
                json.dumps(
                    {k: result[k] for k in ("door_id", "success", "failure_reason")}
                ),
                flush=True,
            )

        next_frame = 0.0
        hero_saved = False
        still_times = iter((5.0, 7.0, 9.0, 11.0))
        next_still = next(still_times, None)
        for step in range(round(args.duration / dt)):
            t = step * dt
            pos_batch = robot_all.data.root_pos_w.cpu().numpy()
            quat_batch = robot_all.data.root_quat_w.cpu().numpy()
            if step % decimation == 0:
                omega_batch = robot_all.data.root_ang_vel_b.cpu().numpy()
                joint_batch = robot_all.data.joint_pos.cpu().numpy()
                velocity_batch = robot_all.data.joint_vel.cpu().numpy()
                door_batch = door_all.data.joint_pos.cpu().numpy()
            for row in rows:
                ri, di = row["ri"], row["di"]
                pos = pos_batch[ri] - row["offset"]
                quat = quat_batch[ri]
                if step % decimation == 0:
                    yaw = math.atan2(
                        2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                        1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
                    )
                    heading = math.atan2(1.5 - pos[1], -pos[0])
                    err = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
                    v = [
                        0.5 if t >= 1 and pos[1] < 1.5 and not row["done"] else 0.0,
                        0.0,
                        float(np.clip(1.5 * err, -0.7, 0.7))
                        if t >= 1 and not row["done"]
                        else 0.0,
                    ]
                    obs = dict(
                        time_s=t,
                        base_quaternion_wxyz=quat.tolist(),
                        base_angular_velocity_body=omega_batch[ri].tolist(),
                        joint_positions=joint_batch[ri].tolist(),
                        joint_velocities=velocity_batch[ri].tolist(),
                        command_velocity=v,
                    )
                    _, values = validate_action(row["policy"](obs), len(row["names"]))
                    commands_np[ri] = values
                    if not row["done"]:
                        row["trace"].append(
                            dict(
                                time_s=t,
                                position=pos.tolist(),
                                quat=quat.tolist(),
                                door_q=door_batch[di].tolist(),
                            )
                        )
                if row["spec"]["family"] == "automatic_sliding":
                    auto = row["spec"]["kinematics"].get("actuator", {})
                    radius = float(auto.get("sensor_range_m", 1.8))
                    hold = float(auto.get("hold_open_s", 2.0))
                    if np.linalg.norm(pos[:2]) < radius:
                        row["last_seen"] = t
                    desired = (
                        row["rl"]["joints"][row["rl"]["door_joint"]]["range"][1]
                        if t - row["last_seen"] < hold
                        else 0.0
                    )
                    speed = float(
                        auto.get("open_speed_m_s", 0.3)
                        if desired > row["auto_target"]
                        else auto.get("close_speed_m_s", 0.25)
                    )
                    row["auto_target"] += float(
                        np.clip(desired - row["auto_target"], -speed * dt, speed * dt)
                    )
                    row["targets"][0, row["pj"]] = row["auto_target"]
                    if row["sj"] is not None:
                        row["targets"][0, row["sj"]] = row["auto_target"]
            if step % decimation == 0:
                commands = torch.as_tensor(commands_np, device=args.device)
            effort = (
                kp_all * (commands - robot_all.data.joint_pos)
                - kd_all * robot_all.data.joint_vel
            )
            robot_all.set_joint_effort_target(
                torch.maximum(torch.minimum(effort, limits_all), -limits_all)
            )
            door_all.set_joint_position_target(targets_all)
            door_all.set_joint_velocity_target(torch.zeros_like(targets_all))
            robot_all.write_data_to_sim()
            door_all.write_data_to_sim()
            render = bool(camera and t >= next_frame)
            sim.step(render=render)
            robot_all.update(dt)
            door_all.update(dt)
            pos_batch = robot_all.data.root_pos_w.cpu().numpy()
            quat_batch = robot_all.data.root_quat_w.cpu().numpy()
            door_batch = door_all.data.joint_pos.cpu().numpy()
            for row in rows:
                if row["done"]:
                    continue
                pos = pos_batch[row["ri"]] - row["offset"]
                quat = quat_batch[row["ri"]]
                q = door_batch[row["di"]]
                if not (
                    np.isfinite(pos).all()
                    and np.isfinite(quat).all()
                    and np.isfinite(q).all()
                ):
                    save(row, t, False, "nonfinite_native_state", pos)
                    continue
                row["max_open"] = max(row["max_open"], abs(float(q[row["pj"]])))
                fell = pos[2] < 0.45 or 1 - 2 * (quat[1] ** 2 + quat[2] ** 2) < 0.5
                goal = (
                    pos[1] >= 1.2
                    and abs(pos[0])
                    < max(0.0, row["spec"]["opening"]["width"] / 2 - 0.2)
                    and not fell
                )
                row["success_since"] = (
                    (t if row["success_since"] is None else row["success_since"])
                    if goal
                    else None
                )
                if fell:
                    save(row, t, False, "robot_fell", pos)
                elif (
                    row["success_since"] is not None and t - row["success_since"] >= 0.5
                ):
                    save(row, t, True, None, pos)
            if render:
                camera.update(dt, force_recompute=True)
                rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
                writer.append_data(rgb)
                if presentation and next_still is not None and t >= next_still:
                    imageio.imwrite(out / f"g1-grid-{int(next_still):02d}.png", rgb)
                    next_still = next(still_times, None)
                next_frame += 1 / 25
                if t >= 7.0 and not hero_saved:
                    imageio.imwrite(out / "g1-grid.png", rgb)
                    hero_saved = True
            if step % round(1 / dt) == 0:
                print(
                    "GRID_PROGRESS",
                    round(t, 2),
                    sum(r["done"] for r in rows),
                    flush=True,
                )
            if all(r["done"] for r in rows) and not camera:
                break
        for row in rows:
            if not row["done"]:
                save(
                    row,
                    args.duration,
                    False,
                    "goal_not_reached_before_timeout",
                    row["robot"].data.root_pos_w[0].cpu().numpy() - row["offset"],
                )
        if writer:
            writer.close()
    print("GRID_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        app.close()
        raise
    else:
        app.close()

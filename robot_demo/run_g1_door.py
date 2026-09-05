#!/usr/bin/env python
"""Unitree G1 humanoid walking through a DoorBench door in plain MuJoCo (CPU), driven by the pretrained
unitree_rl_gym sim2sim locomotion policy, recorded headlessly to mp4 + gif.

Off-the-shelf pieces (all BSD-3, see LICENSES.md; nothing is modified on disk):
  * robot model   : MuJoCo Menagerie `unitree_g1/g1.xml` (29 dof, 12-dof legs actuated by the policy, waist + arms
                    held at the `stand` keyframe by the model's own position servos)
                    fallback `--robot rlgym`: unitree_rl_gym `g1_12dof.xml` (the exact model the policy ships with)
  * policy        : unitree_rl_gym `deploy/pre_train/g1/motion.pt` (TorchScript LSTM + MLP actor, 47-d obs, 12 actions,
                    50 Hz, PD gains / scales from `deploy/deploy_mujoco/configs/g1.yaml`)
  * door          : any `assets/doors/<id>/door.xml`, merged with `mujoco.MjSpec.attach` through DoorBench's
                    `DoorEnv` (closer / ratchet / lock logic + `LabelTracker` labels come for free)

Observation / action construction is a straight port of `deploy_mujoco.py`:
  obs = [omega_base * 0.25, gravity_in_base, cmd * (2, 2, 0.25), (q - q_default), dq * 0.05, prev_action, sin/cos gait phase]
  target_q = action * 0.25 + q_default ;  tau = kp * (target_q - q) - kd * dq   (500 Hz PD, 50 Hz policy)
On top of that this script adds only (i) a P-controller on the yaw-rate command so the robot keeps heading for the
goal point through the opening and (ii) a presence-sensor controller for automatic doors.

Examples
  python robot_demo/run_g1_door.py --door db0119_swing_single --door-open-frac 1.0          # open doorway
  python robot_demo/run_g1_door.py --door db0990_automatic_sliding                          # sensor-opened slider
  python robot_demo/run_g1_door.py --door db0123_saloon                                     # push through a saloon pair
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

from doorbench.benchmark.env import DoorEnv  # noqa: E402

TP = os.path.join(HERE, "third_party")
MENAGERIE_G1 = os.path.join(TP, "mujoco_menagerie", "unitree_g1", "g1.xml")
RLGYM = os.path.join(TP, "unitree_rl_gym")
RLGYM_G1 = os.path.join(RLGYM, "resources", "robots", "g1_description", "g1_12dof.xml")
POLICY = os.path.join(RLGYM, "deploy", "pre_train", "g1", "motion.pt")
CFG = os.path.join(RLGYM, "deploy", "deploy_mujoco", "configs", "g1.yaml")

PREFIX = "robot/"
BASE_BODY = PREFIX + "pelvis"
FREE_JOINT = PREFIX + "floating_base_joint"
# order == unitree_rl_gym g1.yaml (kps / kds / default_angles) == g1_12dof.xml actuator order
LEG_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]


# ----------------------------------------------------------------------------------------------------------------------
class G1DoorEnv(DoorEnv):
    """DoorEnv whose robot is attached at the approach point, facing +y, with policy-ready actuators."""

    def __init__(self, door_dir, robot_xml, tier="full", start_pos=None, yaw_deg=90.0):
        self._start_pos = start_pos
        self._yaw_deg = yaw_deg
        self.hold_pose = {}  # non-leg joint name -> angle (Menagerie only)
        super().__init__(door_dir, tier=tier, robot_xml=robot_xml, robot_body_prefix=PREFIX, robot_base_body=BASE_BODY)

    def _build(self, with_human: bool = False):
        """DoorEnv._build with the robot attached at the approach point facing +y (yaw 90 deg), its stand keyframe
        turned into a hold pose for the waist / arms, the leg actuators switched to torque mode for the policy's PD
        loop, and (human suite) DoorEnv's kinematic person."""
        spec = mujoco.MjSpec.from_file(self.xml_path)
        robot = mujoco.MjSpec.from_file(self.robot_xml)
        # stand keyframe -> hold pose for waist / arms (the policy only drives the 12 leg joints)
        self.hold_pose = {}
        if robot.keys:
            key = robot.keys[0]
            names = [j.name for j in robot.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
            for i, n in enumerate(names):
                if n not in LEG_JOINTS:
                    self.hold_pose[PREFIX + n] = float(key.qpos[7 + i])
            for k in list(robot.keys):
                robot.delete(k)
        for light in list(robot.worldbody.lights):  # the door scene has its own lights
            robot.delete(light)
        for a in robot.actuators:
            if a.name in LEG_JOINTS:
                a.set_to_motor()  # ctrl = joint torque (clamped by the joint's actuatorfrcrange)
        pos = self._start_pos
        if pos is None:
            ap = [s for s in spec.sites if s.name == "approach_point"]
            pos = list(ap[0].pos) if ap else [0.0, -1.5, 0.0]
        yaw = math.radians(self._yaw_deg)
        frame = spec.worldbody.add_frame(pos=pos, quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        spec.attach(robot, prefix=PREFIX, frame=frame)
        if with_human:
            hb = self.benchmark.get("human", {"radius_m": 0.22, "height_m": 1.75})
            r, h = float(hb["radius_m"]), float(hb["height_m"])
            body = spec.worldbody.add_body(name="human", mocap=True, pos=[0.0, -50.0, h / 2])
            body.add_geom(name="human_capsule", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, max(0.05, h / 2 - r), 0.0], rgba=[0.95, 0.55, 0.3, 0.9], group=0)
        m = spec.compile()
        if self.timestep:
            m.opt.timestep = self.timestep
        return m, mujoco.MjData(m)


# ----------------------------------------------------------------------------------------------------------------------
class G1Policy:
    """unitree_rl_gym pretrained G1 locomotion policy + the PD loop from deploy_mujoco.py."""

    def __init__(self, cfg_path=CFG, policy_path=POLICY):
        import yaml
        self.cfg = cfg = yaml.safe_load(open(cfg_path))
        self.policy_path = policy_path
        self.kps = np.array(cfg["kps"], np.float32)
        self.kds = np.array(cfg["kds"], np.float32)
        self.default = np.array(cfg["default_angles"], np.float32)
        self.na, self.nobs = cfg["num_actions"], cfg["num_obs"]
        self.decimation, self.dt = cfg["control_decimation"], cfg["simulation_dt"]
        self.cmd_scale = np.array(cfg["cmd_scale"], np.float32)
        self.reset()

    def reset(self):
        import torch
        torch.set_num_threads(1)
        self.policy = torch.jit.load(self.policy_path)  # reload -> fresh LSTM hidden state
        self.action = np.zeros(self.na, np.float32)
        self.target = self.default.copy()
        self.obs = np.zeros(self.nobs, np.float32)
        self.sim_steps = 0
        self.inference_s = 0.0

    def torque(self, q, dq):
        return (self.target - q) * self.kps - dq * self.kds

    @staticmethod
    def gravity_in_base(quat):
        qw, qx, qy, qz = quat
        return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)], np.float32)

    def act(self, quat, omega_local, q, dq, cmd):
        """Call every `decimation` sim steps. Returns the new target joint positions."""
        import torch
        phase = (self.sim_steps * self.dt) % 0.8 / 0.8
        o, na = self.obs, self.na
        o[:3] = omega_local * self.cfg["ang_vel_scale"]
        o[3:6] = self.gravity_in_base(quat)
        o[6:9] = np.asarray(cmd, np.float32) * self.cmd_scale
        o[9:9 + na] = (q - self.default) * self.cfg["dof_pos_scale"]
        o[9 + na:9 + 2 * na] = dq * self.cfg["dof_vel_scale"]
        o[9 + 2 * na:9 + 3 * na] = self.action
        o[9 + 3 * na:9 + 3 * na + 2] = (math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase))
        t0 = time.perf_counter()
        with torch.no_grad():
            self.action = self.policy(torch.from_numpy(o).unsqueeze(0)).numpy().squeeze().astype(np.float32)
        self.inference_s += time.perf_counter() - t0
        self.target = self.action * self.cfg["action_scale"] + self.default
        return self.target


# ----------------------------------------------------------------------------------------------------------------------
class AutoDoorController:
    """Presence / motion sensor for automatic doors: open while the robot is within `sensor_range_m` of the door
    plane, hold `hold_open_s` after it leaves, ramp the position-servo target at the spec's open / close speed."""

    def __init__(self, env, act_ids):
        kin = env.spec["kinematics"]
        act = kin.get("actuator", {})
        self.range = float(act.get("sensor_range_m", 1.5))
        self.hold = float(act.get("hold_open_s", 2.0))
        self.v_open = float(act.get("open_speed_m_s", 0.3))
        self.v_close = float(act.get("close_speed_m_s", 0.25))
        self.travel = float(env.m.jnt_range[env.pj][1])
        sid = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_SITE, "door_plane_center")
        self.center = env.m.site_pos[sid][:2].copy() if sid >= 0 else np.zeros(2)
        self.act_ids = act_ids
        self.target = 0.0
        self.last_seen = -1e9
        self.dt = env.m.opt.timestep

    def step(self, d, robot_xy, t):
        if np.linalg.norm(np.asarray(robot_xy) - self.center) < self.range:
            self.last_seen = t
        want = self.travel if (t - self.last_seen) < self.hold else 0.0
        self.target += float(np.clip(want - self.target, -self.v_close * self.dt, self.v_open * self.dt))
        d.ctrl[self.act_ids] = self.target


# ----------------------------------------------------------------------------------------------------------------------
class Recorder:
    """Side-by-side video: door `iso` camera | free camera following the pelvis."""

    def __init__(self, m, fps=30, size=(640, 480), iso_cam="iso", follow=(60.0, -18.0, 3.2)):
        self.m, self.fps = m, fps
        self.w, self.h = size
        self.r = mujoco.Renderer(m, height=self.h, width=self.w)
        self.iso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, iso_cam)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.elevation, self.cam.distance = follow
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # collision + visual groups, hide group-3 collision meshes
        self.frames = []
        self.next_t = 0.0
        self.lookat = None
        self.render_s = 0.0

    def maybe(self, d, base_pos, force=False):
        if d.time + 1e-9 < self.next_t and not force:
            return
        self.next_t += 1.0 / self.fps
        t0 = time.perf_counter()
        target = np.asarray(base_pos) + np.array([0, 0, 0.05])
        self.lookat = target if self.lookat is None else 0.85 * self.lookat + 0.15 * target
        self.cam.lookat[:] = self.lookat
        self.r.update_scene(d, camera=self.iso, scene_option=self.opt)
        a = self.r.render().copy()
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        b = self.r.render().copy()
        self.frames.append(np.hstack([a, b]))
        self.render_s += time.perf_counter() - t0

    def save(self, mp4_path, gif_path=None, gif_width=640, gif_fps=12):
        import imageio
        os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
        with imageio.get_writer(mp4_path, fps=self.fps, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=16) as w:
            for f in self.frames:
                w.append_data(f)
        if gif_path:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            vf = f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4"
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", mp4_path, "-vf", vf, gif_path], check=True)
        self.r.close()


# ----------------------------------------------------------------------------------------------------------------------
def quat_yaw(q):
    qw, qx, qy, qz = q
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def run(args):
    door_dir = os.path.join(ROOT, "assets", "doors", args.door)
    robot_xml = MENAGERIE_G1 if args.robot == "menagerie" else RLGYM_G1
    t_build = time.perf_counter()
    env = G1DoorEnv(door_dir, robot_xml, tier=args.tier, yaw_deg=args.yaw)
    m, d = env.m, env.d
    build_s = time.perf_counter() - t_build

    def jid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    def aid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    fj = jid(FREE_JOINT)
    qa, va = m.jnt_qposadr[fj], m.jnt_dofadr[fj]
    leg_q = np.array([m.jnt_qposadr[jid(PREFIX + n)] for n in LEG_JOINTS])
    leg_v = np.array([m.jnt_dofadr[jid(PREFIX + n)] for n in LEG_JOINTS])
    leg_a = np.array([aid(PREFIX + n) for n in LEG_JOINTS])
    hold_a = {aid(n): v for n, v in env.hold_pose.items() if aid(n) >= 0}
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    door_acts = [aid(a["name"]) for a in env.meta.get("actuators", []) if aid(a["name"]) >= 0]
    robot_geoms = set(np.nonzero([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]).startswith(PREFIX) for g in range(m.ngeom)])[0].tolist())
    door_geoms = {g for g in range(m.ngeom) if g not in robot_geoms and m.geom_bodyid[g] != 0}
    frame_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") != "floor"}
    goal_y = float(m.site_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "goal_point")][1])

    print(f"[{args.door}] robot={args.robot} nbody={m.nbody} njnt={m.njnt} nu={m.nu} ngeom={m.ngeom} dt={m.opt.timestep} build={build_s:.2f}s")
    print(f"  robot mass {sum(m.body_mass[b] for b in range(m.nbody) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).startswith(PREFIX)):.1f} kg; "
          f"door primary joint {env.meta.get('primary_joint')} range {m.jnt_range[env.pj] if env.pj >= 0 else None}; door actuators {door_acts}; hold joints {len(hold_a)}")

    # ---- reset door + robot
    task = args.task or env.spec.get("task", "open_and_traverse")
    env.reset(task=task)
    if args.door_open_frac is not None and env.pj >= 0:
        lo, hi = m.jnt_range[env.pj]
        d.qpos[m.jnt_qposadr[env.pj]] = lo + args.door_open_frac * (hi - lo)
        sj = env.meta.get("secondary_joint")
        if sj and jid(sj) >= 0:
            lo2, hi2 = m.jnt_range[jid(sj)]
            d.qpos[m.jnt_qposadr[jid(sj)]] = lo2 + args.door_open_frac * (hi2 - lo2)
        bj = jid("leaf_latch_bolt_slide")
        if bj >= 0:
            d.qpos[m.jnt_qposadr[bj]] = 0.0
    pol = G1Policy()
    d.qpos[leg_q] = pol.default
    for a, v in hold_a.items():
        d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]] = v
        d.ctrl[a] = v
    d.qpos[qa + 2] += args.drop  # small drop so the feet settle onto the floor box
    mujoco.mj_forward(m, d)
    auto = AutoDoorController(env, door_acts) if door_acts and env.spec["family"].startswith("automatic") else None

    rec = Recorder(m, fps=args.fps, iso_cam=args.iso_cam) if not args.no_video else None
    log = []
    max_door_f = 0.0
    max_frame_f = 0.0
    f6 = np.zeros(6)
    t_walk0, t_stop = args.settle, None
    passed = False
    fell = False
    cmd = np.zeros(3, np.float32)
    t0 = time.perf_counter()
    sim_s = 0.0
    n_steps = int(round(args.duration / m.opt.timestep))
    for i in range(n_steps):
        ts = time.perf_counter()
        q, dq = d.qpos[leg_q], d.qvel[leg_v]
        d.ctrl[leg_a] = pol.torque(q, dq)
        if auto is not None:
            auto.step(d, d.xpos[base][:2], d.time)
        env.step()  # mj_step + lock logic + labels
        pol.sim_steps += 1
        if pol.sim_steps % pol.decimation == 0:
            pos = d.xpos[base]
            quat = d.qpos[qa + 3:qa + 7]
            yaw = quat_yaw(quat)
            if d.time < t_walk0 or t_stop is not None:
                cmd[:] = 0.0
            else:
                # steer towards a look-ahead point on the door centre line (x = 0), then keep going to the goal
                look = np.array([0.0, pos[1] + args.lookahead])
                desired = math.atan2(look[1] - pos[1], look[0] - pos[0])
                err = wrap(desired - yaw)
                cmd[:] = (args.vx, 0.0, float(np.clip(args.k_yaw * err, -args.max_yaw_rate, args.max_yaw_rate)))
            pol.act(quat, d.qvel[va + 3:va + 6], d.qpos[leg_q], d.qvel[leg_v], cmd)
            door_q = float(d.qpos[m.jnt_qposadr[env.pj]]) if env.pj >= 0 else 0.0
            log.append({"t": round(float(d.time), 3), "x": round(float(pos[0]), 3), "y": round(float(pos[1]), 3), "z": round(float(pos[2]), 3),
                        "yaw_deg": round(math.degrees(yaw), 1), "door_q": round(door_q, 4), "cmd": [round(float(c), 3) for c in cmd]})
            if pos[1] > goal_y + args.overshoot and t_stop is None:
                t_stop = d.time
            if pos[2] < 0.45 and not fell:
                fell = True
                t_stop = t_stop or d.time
        # robot <-> door / frame contact forces
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            r1, r2 = g1 in robot_geoms, g2 in robot_geoms
            if r1 == r2:
                continue
            other = g2 if r1 else g1
            if other in door_geoms or other in frame_geoms:
                mujoco.mj_contactForce(m, d, c, f6)
                fn = abs(float(f6[0]))
                if other in door_geoms:
                    max_door_f = max(max_door_f, fn)
                else:
                    max_frame_f = max(max_frame_f, fn)
        sim_s += time.perf_counter() - ts
        if rec is not None:
            rec.maybe(d, d.xpos[base])
        if t_stop is not None and d.time - t_stop > args.tail:
            break
    wall = time.perf_counter() - t0
    labels = env.labels().to_dict()
    passed = bool(labels.get("robot_passed_through"))
    pos = d.xpos[base]
    grav = G1Policy.gravity_in_base(d.qpos[qa + 3:qa + 7])
    upright = bool(pos[2] > 0.6 and grav[2] < -0.9)
    result = {
        "door": args.door, "task": task, "robot_model": os.path.relpath(robot_xml, ROOT), "policy": os.path.relpath(POLICY, ROOT),
        "sim_time_s": round(float(d.time), 3), "wall_time_s": round(wall, 2), "physics_policy_wall_s": round(sim_s, 2),
        "rtf_physics_policy": round(float(d.time) / max(sim_s, 1e-9), 1), "rtf_incl_render": round(float(d.time) / max(wall, 1e-9), 1),
        "policy_inference_ms": round(1e3 * pol.inference_s / max(1, pol.sim_steps // pol.decimation), 3),
        "render_wall_s": round(rec.render_s, 2) if rec else None, "frames": len(rec.frames) if rec else 0,
        "final_base_pos": [round(float(v), 3) for v in pos], "goal_y": goal_y, "reached_goal": bool(pos[1] > goal_y),
        "passed_through": passed, "upright_at_end": upright, "fell": fell or bool(labels.get("robot_fell")),
        "max_robot_door_contact_N": round(max_door_f, 1), "max_robot_frame_contact_N": round(max_frame_f, 1),
        "max_door_joint": round(float(labels.get("max_door_angle", 0.0)), 4),
        "labels": labels, "cmd_vx": args.vx, "trajectory": log,
    }
    print(json.dumps({k: v for k, v in result.items() if k not in ("trajectory", "labels")}, indent=1))
    print("labels:", {k: v for k, v in labels.items() if k not in ("damage_events", "notes")})
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"g1_door_{args.door.split('_')[0]}"
    with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=1)
    if rec is not None:
        mp4 = os.path.join(args.media_dir, f"{tag}.mp4")
        gif = os.path.join(args.media_dir, f"{tag}.gif") if not args.no_gif else None
        rec.save(mp4, gif, gif_width=args.gif_width, gif_fps=args.gif_fps)
        print("wrote", mp4, os.path.getsize(mp4) // 1024, "kB", (gif, os.path.getsize(gif) // 1024, "kB") if gif else "")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--door", default="db0119_swing_single")
    p.add_argument("--robot", choices=["menagerie", "rlgym"], default="menagerie")
    p.add_argument("--tier", default="full")
    p.add_argument("--task", default=None, help="DoorEnv task preset (default: the door's spec.task)")
    p.add_argument("--door-open-frac", type=float, default=None, help="start the primary joint at this fraction of its range")
    p.add_argument("--duration", type=float, default=14.0, help="max sim seconds")
    p.add_argument("--settle", type=float, default=1.0, help="stand still this long before walking")
    p.add_argument("--tail", type=float, default=1.0, help="keep simulating this long after reaching the goal / falling")
    p.add_argument("--vx", type=float, default=0.5, help="forward velocity command (m/s)")
    p.add_argument("--k-yaw", type=float, default=1.5)
    p.add_argument("--max-yaw-rate", type=float, default=0.6)
    p.add_argument("--lookahead", type=float, default=1.2)
    p.add_argument("--overshoot", type=float, default=0.3, help="stop this far past goal_point")
    p.add_argument("--yaw", type=float, default=90.0, help="robot yaw at start (deg); 90 = facing +y")
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--iso-cam", default="iso")
    p.add_argument("--gif-width", type=int, default=640)
    p.add_argument("--gif-fps", type=int, default=12)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--media-dir", default=os.path.join(ROOT, "docs", "media"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

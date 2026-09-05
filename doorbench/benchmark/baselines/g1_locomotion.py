"""Unitree G1 + pretrained unitree_rl_gym locomotion policy as a benchmark baseline (robot embodiment).

The robot (MuJoCo Menagerie `unitree_g1/g1.xml`, BSD-3) is attached to the door scene at the approach point by
`robot_demo/run_g1_door.py`'s `G1DoorEnv` (DoorEnv places it at the scenario's seeded start pose); the legs are
driven by the pretrained sim2sim policy (`motion.pt`, BSD-3) through the 500 Hz PD loop of `deploy_mujoco.py`, the
waist / arms are parked, and a P-controller on the yaw-rate command steers the robot along the door centre line
toward `goal_point` after a 1 s settle.

This is a locomotion-only controller: it can walk through open doorways, automatic doors and free-swinging doors
(saloon, strip curtains, turnstiles it can push), and it will walk into everything else.  It cannot reach for a
lever, knob, bar, keypad or bolt, never declares a door locked and never knocks.  Run `bash robot_demo/setup.sh` once to fetch the model and the policy.
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import sys

import numpy as np

from ..policy import Policy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUN_G1 = os.path.join(ROOT, "robot_demo", "run_g1_door.py")
_MOD = None


def _robot_demo():
    """Import robot_demo/run_g1_door.py as a module (it is a script, not a package)."""
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location("doorbench_robot_demo_run_g1_door", RUN_G1)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _MOD = mod
    return _MOD


def _quat_yaw(q):
    qw, qx, qy, qz = q
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class G1LocomotionPolicy(Policy):
    name = "g1_locomotion"
    description = "Unitree G1 (MuJoCo Menagerie) driven by the pretrained unitree_rl_gym sim2sim locomotion policy: walks toward goal_point with a yaw P-controller after a 1 s settle; arms parked, no manipulation."
    embodiment = "robot"
    control_dt = 0.002       # the 500 Hz PD loop; the network runs every 10th step
    requires_tier = None

    vx = 0.5
    k_yaw = 1.5
    max_yaw_rate = 0.6
    lookahead = 1.2
    settle_s = 1.0

    @classmethod
    def check(cls):
        rd = _robot_demo()
        missing = [p for p in (rd.MENAGERIE_G1, rd.POLICY, rd.CFG) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"G1 baseline needs the third-party robot + policy: run `bash robot_demo/setup.sh` (missing: {missing[0]})")

    @classmethod
    def info(cls) -> dict:
        out = {"robot": "mujoco_menagerie/unitree_g1/g1.xml", "policy": "unitree_rl_gym/deploy/pre_train/g1/motion.pt"}
        try:
            with open(os.path.join(ROOT, "robot_demo", "setup.sh")) as f:
                txt = f.read()
            for key in ("MENAGERIE_SHA", "RLGYM_SHA"):
                mt = re.search(rf"{key}=([0-9a-f]+)", txt)
                if mt:
                    out[key.lower()] = mt.group(1)
        except Exception:
            pass
        try:
            import torch
            out["torch"] = torch.__version__
        except Exception:
            pass
        return out

    @classmethod
    def make_env(cls, door_dir: str, tier: str, seed: int):
        rd = _robot_demo()
        cls.check()
        return rd.G1DoorEnv(door_dir, rd.MENAGERIE_G1, tier=tier)

    def __init__(self):
        self._pol = None

    def reset(self, info: dict, env=None) -> None:
        rd = _robot_demo()
        mujoco = env.mj
        self.env = env
        m, d = env.m, env.d
        if self._pol is None:
            self._pol = rd.G1Policy()
        else:
            self._pol.reset()
        pol = self._pol

        def jid(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

        def aid(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        fj = jid(rd.FREE_JOINT)
        self.qa, self.va = int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])
        self.leg_q = np.array([m.jnt_qposadr[jid(rd.PREFIX + n)] for n in rd.LEG_JOINTS])
        self.leg_v = np.array([m.jnt_dofadr[jid(rd.PREFIX + n)] for n in rd.LEG_JOINTS])
        self.leg_a = np.array([aid(rd.PREFIX + n) for n in rd.LEG_JOINTS])
        hold_a = {aid(n): v for n, v in env.hold_pose.items() if aid(n) >= 0}
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, rd.BASE_BODY)
        # initial pose (legs at the policy's default, waist / arms at the stand keyframe), seeded x jitter
        d.qpos[self.leg_q] = pol.default
        self.ctrl = np.zeros(m.nu)
        for a, v in hold_a.items():
            d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]] = v
            self.ctrl[a] = v
        # the environment has already placed the robot at the scenario's seeded start pose (x, y, yaw)
        mujoco.mj_forward(m, d)
        self.goal_y = float(info["goal_point"][1])
        self.cmd = np.zeros(3, np.float32)
        self.stop = False

    def act(self, obs: dict) -> dict:
        d, m = self.env.d, self.env.m
        pol = self._pol
        q, dq = d.qpos[self.leg_q], d.qvel[self.leg_v]
        self.ctrl[self.leg_a] = pol.torque(q, dq)
        pol.sim_steps += 1
        if pol.sim_steps % pol.decimation == 0:
            pos = d.xpos[self.base]
            quat = d.qpos[self.qa + 3:self.qa + 7]
            yaw = _quat_yaw(quat)
            if d.time < self.settle_s or self.stop:
                self.cmd[:] = 0.0
            else:
                look = np.array([0.0, pos[1] + self.lookahead])
                desired = math.atan2(look[1] - pos[1], look[0] - pos[0])
                err = _wrap(desired - yaw)
                self.cmd[:] = (self.vx, 0.0, float(np.clip(self.k_yaw * err, -self.max_yaw_rate, self.max_yaw_rate)))
            if pos[1] > self.goal_y + 0.3:
                self.stop = True
            pol.act(quat, d.qvel[self.va + 3:self.va + 6], d.qpos[self.leg_q], d.qvel[self.leg_v], self.cmd)
        return {"ctrl": self.ctrl}

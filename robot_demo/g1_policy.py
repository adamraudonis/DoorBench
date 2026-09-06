"""Shared Unitree checkpoint adapter for MuJoCo and Isaac Sim demos.

The 47-value observation, 12 leg joint order and PD gains preserve the pinned
unitree_rl_gym deployment contract. No door state enters this locomotion policy.
"""

from __future__ import annotations
import math, time
from pathlib import Path
import numpy as np

RLGYM = Path(__file__).resolve().parent / "third_party" / "unitree_rl_gym"
POLICY = str(RLGYM / "deploy/pre_train/g1/motion.pt")
CFG = str(RLGYM / "deploy/deploy_mujoco/configs/g1.yaml")
LEG_JOINTS = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "ankle_pitch",
        "ankle_roll",
    )
]


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
        self.policy = torch.jit.load(
            self.policy_path
        )  # reload -> fresh LSTM hidden state
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
        return np.array(
            [
                2 * (-qz * qx + qw * qy),
                -2 * (qz * qy + qw * qx),
                1 - 2 * (qw * qw + qz * qz),
            ],
            np.float32,
        )

    def act(self, quat, omega_local, q, dq, cmd):
        """Call every `decimation` sim steps. Returns the new target joint positions."""
        import torch

        phase = (self.sim_steps * self.dt) % 0.8 / 0.8
        o, na = self.obs, self.na
        o[:3] = omega_local * self.cfg["ang_vel_scale"]
        o[3:6] = self.gravity_in_base(quat)
        o[6:9] = np.asarray(cmd, np.float32) * self.cmd_scale
        o[9 : 9 + na] = (q - self.default) * self.cfg["dof_pos_scale"]
        o[9 + na : 9 + 2 * na] = dq * self.cfg["dof_vel_scale"]
        o[9 + 2 * na : 9 + 3 * na] = self.action
        o[9 + 3 * na : 9 + 3 * na + 2] = (
            math.sin(2 * math.pi * phase),
            math.cos(2 * math.pi * phase),
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            self.action = (
                self.policy(torch.from_numpy(o).unsqueeze(0))
                .numpy()
                .squeeze()
                .astype(np.float32)
            )
        self.inference_s += time.perf_counter() - t0
        self.target = self.action * self.cfg["action_scale"] + self.default
        return self.target

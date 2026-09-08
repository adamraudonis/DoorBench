"""Adapter for the upstream frozen two-hand reaching checkpoint.

Uses robot state and commanded reach targets. Exact door targets, when used by a
teacher, must be labelled privileged. This is not a trained door-opening policy.
Checkpoint architecture and normalization follow HumanoidBench (MIT); see its
LICENSE and humanoid_bench/mjx/flax_to_torch.py at the pinned revision.
"""
from pathlib import Path
import numpy as np
import torch
import mujoco


class ReachingController:
    def __init__(self, env, upstream):
        self.env = env
        torch.set_num_threads(1)
        folder = Path(upstream) / "data/reach_two_hands"
        self.weights = torch.load(folder / "torch_model.pt", map_location="cpu", weights_only=True)
        self.mean = np.load(folder / "mean.npy")[0]
        self.var = np.load(folder / "var.npy")[0]
        if self.mean.shape != (61,) or self.weights["dense1.weight"].shape != (256, 61):
            raise ValueError("Unexpected reaching checkpoint")
        m, d = env.m, env.d
        self.body_joints = [j for j in env.joints if not any(x in m.joint(j).name for x in ("/lh_", "/rh_", "wrist"))]
        self.body_actuators = [i for i in env.actuators if not any(x in m.actuator(i).name for x in ("/lh_", "/rh_", "wrist"))]
        if len(self.body_joints) != 19 or len(self.body_actuators) != 19:
            raise ValueError("Reaching controller requires the unchanged 19-actuator H1 body")
        q = d.qpos[env.root_qadr + 3:env.root_qadr + 7]
        self.yaw = 2 * np.arctan2(q[3], q[0])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.to_local = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        self.inverse_yaw = np.array([np.cos(self.yaw / 2), 0, 0, -np.sin(self.yaw / 2)])
        self.targets = np.array([d.site("robot/left_hand").xpos.copy(), d.site("robot/right_hand").xpos.copy()])

    def observation(self, targets=None):
        env = self.env
        m, d = env.m, env.d
        if targets is not None:
            self.targets = np.asarray(targets).copy()
        root = d.qpos[env.root_qadr:env.root_qadr + 7]
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, self.inverse_yaw, root[3:])
        robot_q = np.r_[root[2], quat, d.qpos[m.jnt_qposadr[self.body_joints]]]
        root_vel = d.qvel[env.root_vadr:env.root_vadr + 6].copy()
        root_vel[:3] = self.to_local @ root_vel[:3]
        robot_v = np.r_[root_vel, d.qvel[m.jnt_dofadr[self.body_joints]]]
        offset = np.array([root[0], root[1], 0])
        hands = np.array([d.site("robot/left_hand").xpos, d.site("robot/right_hand").xpos])
        local_hands = (hands - offset) @ self.to_local.T
        local_targets = (self.targets - offset) @ self.to_local.T
        obs = np.r_[robot_q, robot_v, local_hands.ravel(), local_targets.ravel()]
        return (obs - self.mean) / np.sqrt(self.var + 1e-8)

    def action(self, targets=None):
        return self.body_action(self.predict_body(targets))

    def predict_body(self, targets=None):
        obs = self.observation(targets)
        x = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            for layer in ("dense1", "dense2", "dense3"):
                x = torch.nn.functional.linear(x, self.weights[layer + ".weight"], self.weights[layer + ".bias"])
                if layer != "dense3":
                    x = x.tanh()
        return x.numpy()

    def body_action(self, body_action):
        env = self.env; m, d = env.m, env.d
        control = d.ctrl[env.actuators].copy()
        local = {a: k for k, a in enumerate(env.actuators)}
        lo = m.actuator_ctrlrange[self.body_actuators, 0]
        hi = m.actuator_ctrlrange[self.body_actuators, 1]
        body_ctrl = lo + (np.clip(body_action, -1, 1) + 1) * .5 * (hi - lo)
        for a, val in zip(self.body_actuators, body_ctrl):
            control[local[a]] = val
        for side in ("left", "right"):
            control[local[m.actuator(f"robot/{side}_wrist_yaw").id]] = 1.57
        return env.normalize(control)

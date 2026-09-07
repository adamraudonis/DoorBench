"""Privileged whole-body reaching curriculum on the full 61-actuator model.

The first stage learns 19 body actuators with hands held at their current open
pose. It is a reusable body skill, not a dexterous door-opening score.
"""
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment import DexterousDoorEnv
from .reaching import ReachingController


class ReachTeacherEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, door_dir, robot_xml, upstream, *, distance=.25, horizon=300):
        robot_xml = Path(robot_xml)
        audit = json.loads(robot_xml.with_suffix('.audit.json').read_text())
        self.sim = DexterousDoorEnv(door_dir, robot_xml, audit)
        self.upstream = upstream
        self.distance, self.horizon = distance, horizon
        self.action_space = spaces.Box(-1., 1., shape=(19,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(61,), dtype=np.float32)
        self.controller = None
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(seed=int(self.np_random.integers(0, 2**30)), randomize=True, images=False)
        self.controller = ReachingController(self.sim, self.upstream)
        # Starts remain in the original door start region; no fixed base or foot anchors.
        self.initial_targets = self.controller.targets.copy()
        offset = np.array([self.np_random.uniform(0., self.distance),
                           self.np_random.uniform(-.08,.08), self.np_random.uniform(-.08,.2)])
        self.targets = self.initial_targets + self.controller.to_local.T @ offset
        self.controller.targets = self.targets.copy()
        self.steps = 0; self.previous_action = np.zeros(19); self.success_steps = 0
        self.initial_root = self.sim.d.qpos[self.sim.root_qadr:self.sim.root_qadr+3].copy()
        return self.controller.observation().astype(np.float32), {}

    def step(self, action):
        sim = self.sim
        sim.step(self.controller.body_action(action), images=False)
        self.steps += 1
        hands = np.array([sim.d.site('robot/left_hand').xpos, sim.d.site('robot/right_hand').xpos])
        errors = np.linalg.norm(hands-self.targets,axis=1)
        diag = sim.diagnostics()
        tilt = np.deg2rad(diag['torso_tilt_deg'])
        height = diag['root_height_m']
        root_vel = sim.d.qvel[sim.root_vadr:sim.root_vadr+3]
        reaching = np.exp(-6*errors).mean()
        upright = np.exp(-8*tilt**2)
        standing = np.exp(-20*(height-.95)**2)
        smoothness = np.mean((np.asarray(action)-self.previous_action)**2)
        reward = float(2*reaching + upright + standing - .02*smoothness - .005*np.sum(root_vel**2))
        fallen = height < .5 or tilt > .9 or not diag['finite']
        if fallen:
            reward -= 10.
        good = bool(max(errors) < .08 and tilt < np.deg2rad(12) and height > .8)
        self.success_steps = self.success_steps + 1 if good else 0
        success = self.success_steps >= 25
        self.previous_action = np.asarray(action).copy()
        info = {**diag, 'max_reach_error_m':float(max(errors)), 'is_success':success,
                'fell':fallen, 'scope':'privileged body-reaching skill; no door-opening claim'}
        return self.controller.observation().astype(np.float32), reward, fallen or success, self.steps >= self.horizon, info

    def close(self):
        self.sim.close()

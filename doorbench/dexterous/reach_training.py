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

    def __init__(self, door_dir, robot_xml, upstream, *, distance=.25, horizon=300, standing_weight=1., continue_after_success=False):
        robot_xml = Path(robot_xml)
        audit = json.loads(robot_xml.with_suffix('.audit.json').read_text())
        self.sim = DexterousDoorEnv(door_dir, robot_xml, audit)
        self.upstream = upstream
        self.distance, self.horizon = distance, horizon
        self.standing_weight=standing_weight
        self.continue_after_success=continue_after_success
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
        self.steps = 0; self.previous_action = np.zeros(19); self.success_steps = 0; self.reached_success=False
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
        reward = float(2*reaching + upright + self.standing_weight*standing - .02*smoothness - .005*np.sum(root_vel**2))
        fallen = height < .5 or tilt > .9 or not diag['finite'] or diag['numerical_warnings']>0
        if fallen:
            reward -= 10.
        good = bool(max(errors) < .08 and tilt < np.deg2rad(12) and height > .8)
        self.success_steps = self.success_steps + 1 if good else 0
        self.reached_success = self.reached_success or self.success_steps >= 25
        success = self.reached_success and not fallen
        self.previous_action = np.asarray(action).copy()
        info = {**diag, 'max_reach_error_m':float(max(errors)), 'is_success':success,
                'fell':fallen, 'scope':'privileged body-reaching skill; no door-opening claim'}
        return self.controller.observation().astype(np.float32), reward, fallen or (success and not self.continue_after_success), self.steps >= self.horizon, info

    def close(self):
        self.sim.close()

    def configuration_audit(self):
        m=self.sim.m
        return {'backend':'mujoco-native','robot_adapter':'h1-shadow-v1',
            'timestep_s':float(m.opt.timestep),'frame_skip':self.sim.frame_skip,
            'arena_memory_bytes':int(m.narena),
            'integrator':int(m.opt.integrator),'solver':int(m.opt.solver),'cone':int(m.opt.cone),
            'iterations':int(m.opt.iterations),'tolerance':float(m.opt.tolerance),
            'gravity':m.opt.gravity.tolist(),'joint_count':m.njnt,'actuator_count':m.nu,
            'robot_actuators':[m.actuator(i).name for i in self.sim.actuators],
            'learned_body_actuators':[m.actuator(i).name for i in self.sim.actuators
                if not any(s in m.actuator(i).name for s in ('/lh_','/rh_','wrist'))],
            'control_ranges':m.actuator_ctrlrange[self.sim.actuators].tolist(),
            'force_ranges':m.actuator_forcerange[self.sim.actuators].tolist(),
            'tactile_values':len(self.sim.tactile_indices),'horizon_steps':self.horizon,
            'target_distance_m':self.distance,'standing_reward_weight':self.standing_weight,
            'continue_after_success':self.continue_after_success,'sensor_policy':False}

"""Random baseline: uniformly random generalized forces (within the hand's limits) on every reachable door joint,
resampled every 50 ms, and a random-walking base.  The floor of the leaderboard."""
from __future__ import annotations

import numpy as np

from ..policy import Policy


class RandomPolicy(Policy):
    name = "random"
    description = "Uniform random torques within the hand limits on every reachable door joint (resampled at 20 Hz) and a random-walk base velocity."
    control_dt = 0.05

    def reset(self, door_info: dict, env=None) -> None:
        self.rng = np.random.default_rng([int(door_info.get("seed", 0)), int(door_info["spec"].get("index", 0)), 7])
        self.joints = [(j, lim) for j, lim in door_info["torque_limits"].items() if lim > 0]

    def act(self, obs: dict) -> dict:
        r = self.rng
        return {"torques": {j: float(r.uniform(-lim, lim)) for j, lim in self.joints},
                "base_velocity": r.uniform(-1.0, 1.0, size=2).tolist()}

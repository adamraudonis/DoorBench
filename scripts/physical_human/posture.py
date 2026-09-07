"""Measured upright reward and acceptance bounds for the achieved torso pose."""

import numpy as np


class UprightAudit:
    def __init__(self, model):
        self.model = model
        self.chest = model.body("actor_chest").id
        self.root = model.body("actor_pelvis").id
        self.maximum = 0.0
        self.maximum_backward = 0.0
        self.square_sum = 0.0
        self.backward_square_sum = 0.0
        self.samples = 0
        self.phases = {}

    def observe(self, data, phase):
        up = data.xmat[self.chest].reshape(3, 3)[:, 2]
        forward = data.xmat[self.root].reshape(3, 3)[:, 1].copy()
        forward[2] = 0
        forward /= max(np.linalg.norm(forward), 1e-9)
        tilt = float(np.arccos(np.clip(up[2], -1, 1)))
        backward = max(0.0, float(np.arctan2(-up @ forward, up[2])))
        self.maximum = max(self.maximum, tilt)
        self.maximum_backward = max(self.maximum_backward, backward)
        self.square_sum += tilt * tilt
        self.backward_square_sum += backward * backward
        self.samples += 1
        row = self.phases.setdefault(
            phase, {"max_tilt_deg": 0.0, "max_backward_lean_deg": 0.0}
        )
        row["max_tilt_deg"] = max(row["max_tilt_deg"], float(np.rad2deg(tilt)))
        row["max_backward_lean_deg"] = max(
            row["max_backward_lean_deg"], float(np.rad2deg(backward))
        )

    def result(self):
        mean = self.square_sum / max(1, self.samples)
        backward = self.backward_square_sum / max(1, self.samples)
        return {
            "samples": self.samples,
            "max_torso_tilt_deg": float(np.rad2deg(self.maximum)),
            "max_backward_lean_deg": float(np.rad2deg(self.maximum_backward)),
            "rms_torso_tilt_deg": float(np.rad2deg(np.sqrt(mean))),
            "mean_squared_tilt_rad2": mean,
            "mean_squared_backward_lean_rad2": backward,
            "upright_reward": float(-25 * mean - 50 * backward),
            "reward_formula": "-25 mean(torso tilt^2) - 50 mean(backward lean^2); radians",
            "limits": {"max_torso_tilt_deg": 8.0, "max_backward_lean_deg": 5.0},
            "phases": self.phases,
            "passed": bool(
                self.maximum < np.deg2rad(8) and self.maximum_backward < np.deg2rad(5)
            ),
        }

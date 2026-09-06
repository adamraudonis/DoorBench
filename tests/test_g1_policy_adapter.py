"""Policy contract tests run without Isaac or the optional downloaded checkpoint."""

import numpy as np
import pytest
from robot_demo import isaac_policy_adapter as adapter
from robot_demo.g1_policy import LEG_JOINTS, G1Policy


def test_joint_mapping_uses_names_and_preserves_upper_body(monkeypatch):
    instances = []

    class FakePolicy:
        dt = 0.002

        def __init__(self, *args):
            instances.append(self)

        def act(self, quat, omega, q, dq, cmd):
            np.testing.assert_array_equal(q, np.arange(12))
            np.testing.assert_array_equal(dq, np.arange(12) * 2)
            assert self.sim_steps == 10
            return q + 100

    monkeypatch.setattr(adapter, "G1Policy", FakePolicy)
    names = ["torso_joint"] + list(reversed(LEG_JOINTS))
    context = dict(
        joint_names=names,
        default_joint_positions=[0.7] + [0] * 12,
        config_path="unused",
        checkpoint="unused",
    )
    act = adapter.unitree_factory(context)
    action = act(
        dict(
            time_s=0.02,
            base_quaternion_wxyz=[1, 0, 0, 0],
            base_angular_velocity_body=[0, 0, 0],
            joint_positions=[0.7] + list(reversed(range(12))),
            joint_velocities=[0] + list(reversed(np.arange(12) * 2)),
            command_velocity=[0.5, 0, 0],
        )
    )
    np.testing.assert_allclose(
        action["joint_positions"], [0.7] + list(reversed(np.arange(12) + 100))
    )
    adapter.unitree_factory(context)
    assert len(instances) == 2  # Recurrent state never leaks between episodes.


@pytest.mark.parametrize(
    "action",
    [
        {},
        {"joint_positions": [0, 1], "joint_efforts": [0, 1]},
        {"door_position": [0, 1]},
        {"joint_positions": [0]},
        {"joint_efforts": [0, float("nan")]},
        {"joint_positions": [[0, 1]]},
    ],
)
def test_invalid_actions_rejected(action):
    with pytest.raises(ValueError):
        adapter.validate_action(action, 2)


def test_finite_robot_torque_command():
    mode, values = adapter.validate_action({"joint_efforts": [2, -3]}, 2)
    assert mode == "joint_efforts"
    np.testing.assert_array_equal(values, [2, -3])


def test_gravity_is_body_frame_and_yaw_invariant():
    np.testing.assert_allclose(G1Policy.gravity_in_base([1, 0, 0, 0]), [0, 0, -1])
    np.testing.assert_allclose(
        G1Policy.gravity_in_base([2**-0.5, 0, 0, 2**-0.5]), [0, 0, -1], atol=1e-6
    )
    np.testing.assert_allclose(
        G1Policy.gravity_in_base([2**-0.5, 2**-0.5, 0, 0]), [0, -1, 0], atol=1e-6
    )


def test_suite_counts_masked_simulator_failure(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/isaaclab/run_g1_demo_suite.py"
    spec = importlib.util.spec_from_file_location("g1_suite", path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "demo-suite.json").write_text(
        json.dumps({"cases": [{"id": "first"}, {"id": "second"}]})
    )
    out = tmp_path / "results"
    calls = []

    def run(command, **kwargs):
        from types import SimpleNamespace

        door = command[command.index("--door") + 1]
        calls.append(door)
        if door == "first":
            kwargs["stdout"].write(
                "Traceback (most recent call last):\nNative failure\n"
            )
        else:
            (out / f"{door}-seed0.json").write_text(
                json.dumps(
                    {
                        "door_id": door,
                        "seed": 0,
                        "success": False,
                        "failure_reason": "robot_fell",
                    }
                )
            )
            kwargs["stdout"].write("DOORBENCH_G1_RESULT {}\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", run)
    monkeypatch.setattr(
        sys, "argv", ["suite", "--assets", str(assets), "--out", str(out)]
    )
    assert suite.main() == 1
    summary = json.loads((out / "summary.json").read_text())
    assert calls == ["first", "second"]
    assert summary["episodes"] == 2 and summary["simulator_errors"] == 1
    assert summary["successful_episodes"] == 0
    # Prevent old successful result files from being mistaken for a new run.
    with pytest.raises(FileExistsError):
        suite.main()

"""The DoorBench policy interface: what the benchmark runner calls and what it hands to a policy.

A policy is any Python class with two methods::

    class MyPolicy(Policy):
        name = "my_policy"                      # shown on the leaderboard
        control_dt = 0.01                       # seconds between act() calls (torques are zero-order held)

        def reset(self, door_info, env=None):   # once per episode
            ...
        def act(self, obs) -> dict:             # every control_dt seconds
            return {"torques": {"leaf_handle_hinge": 3.0, "leaf_hinge": 40.0}, "base_velocity": [0.0, 1.0]}

Scenarios and suites
--------------------
Each episode evaluates one of the door's scenarios (`door_info["scenario"]`, `doorbench.benchmark.scenarios`):
the `core` suite (`open_and_traverse`, `open_then_close`, `close_only`, `unlock_and_traverse`, `locked_recognize`)
involves nobody but the robot and is what `doorbench benchmark run` evaluates by default; the opt-in `human` suite
(`hold_open_for_human`, `wait_for_human`, `knock_and_wait`) adds a simulated person whose position the policy sees
in `obs["human_xy"]` (`door_info["human"]` carries its radius, speed and nominal path).  Success is the scenario's
own criterion (`door_info["scenario_spec"]["success"]`, evaluated by `DoorEnv.success`).

Reference embodiment ("hand + base")
------------------------------------
The reference runner (`doorbench.benchmark.runner`) evaluates policies with DoorEnv's programmatic hand and a
synthetic robot base, the same embodiment as `scripts/demo_mujoco.py`:

* the hand applies generalized forces on named door joints (`DoorEnv.apply_joint_torque`): pressing a lever, pushing
  a touch bar, turning a thumbturn or handwheel, pressing keypad keys, pushing / pulling / sliding / lifting a leaf.
  Every torque is clamped to `door_info["torque_limits"][joint]` (N*m for hinges, N for slides); the leaf limit is
  the calibrated "strong push" of the sign-off QA (`qa.json: metrics.qa_push`).  Lock parts that sit on the far side
  of the door (`spec.lock.robot_side_release == false`) have a limit of 0 - the robot cannot reach them.
* the base is a point at z = 0.5 m that starts at the scenario's seeded start pose (`door_info["base"]["start"]`,
  drawn from the start zone) and moves with the commanded planar velocity (<= `base.max_speed` m/s).  It can only
  cross the wall plane (|y| < `base.radius`) while the opening is clear (`LabelTracker.door_open_clear`: hinge
  >= 60 deg, slide >= 0.55 m or 95 % of travel, overhead >= 1.9 m) and |x| is inside the opening.  It does not
  collide with the leaf otherwise, but it does count as a collision with the simulated person of the human suite
  when it comes within `robot.body_radius_m + human.radius_m` (0.52 m) of them.

Observation (dict, all floats in SI, positive joint values = opening / actuating)
--------------------------------------------------------------------------------
    t                 sim time since reset (s)
    door_q, door_dq   primary joint position / velocity (rad or m)
    secondary_q/dq    second leaf / panel (or None)
    joints            {name: {"q": float, "dq": float}} for every robot-interactive joint (roles primary, secondary,
                      operator, latch, lock)
    sites             {name: [x, y, z]} world positions of grip / push sites and approach_point, goal_point,
                      door_plane_center
    base              {"pos": [x, y, z]}  the robot base
    flags             {"touched", "operator_actuated", "latch_released", "lock_released", "door_opened",
                       "door_open_clear", "passed_through", "closed_after", "slammed", "damaged"}  the LabelTracker
                      state so far
    locked            True while an engaged lock has not been released
    fired             names of the scenario reward events fired so far (e.g. ["touch_handle", "unlatch", "opened"])
    return            episode return so far (reward table of the scenario)
    success           whether the scenario's success criterion holds right now
    human_xy          [x, y] of the simulated person (human suite) or None

Action (dict; every key optional)
---------------------------------
    torques           {joint_name: float}   generalized force on door joints (clamped, zero-order held)
    base_velocity     [vx, vy] m/s
    badge             True to present a credential (card readers / turnstiles with a robot-side release)
    knock             True to knock on the closed leaf (knock_and_wait; robots with geoms knock by contact)
    declare_locked    True to declare the door locked (locked_recognize): fires `recognized_locked`, ends the episode
    done              True to end the episode early
    ctrl              array for the model's actuators (robot embodiments only, see below)

`door_info` (dict, given once at reset)
---------------------------------------
    id, family, scenario, suite, tier, seed, difficulty, time_budget_s, control_dt
    scenario_spec     the scenario block of spec.json["benchmark"] (initial_state, start zone, handle_targets,
                      pass_plane, goal, human, thresholds, rewards, success, time_budget_s, expected_transit_s)
    start             {"xy", "z", "yaw", "seed"}  the sampled start pose;  human  the person (or None)
    spec              the door's spec.json (every physical parameter, lock state, closer, damage thresholds, ...)
    meta              model.json["meta"] (primary_joint, operator_joint, secondary_joint, pair, actuators, ...)
    joints            {name: {"role", "type": "hinge"|"slide", "range": [lo, hi] | None, "label"}}
    operator_joints, lock_joints, latch_joints, primary_joint, secondary_joint
    lock              {"model", "kind", "engaged", "robot_side_release", "code"}  (`code` only when the robot is
                      allowed to know it, i.e. robot_side_release)
    torque_limits     {joint_name: float}
    base              {"max_speed": float, "radius": float, "start": [x, y, z], "yaw": float}
    approach_point, goal_point, pass_plane, handle_targets, opening_width, leaf_width, kinematics

Robot embodiments
-----------------
A policy that brings its own robot (see `doorbench.benchmark.baselines.g1_locomotion`) sets `embodiment = "robot"`
and implements `make_env(door_dir, tier, seed) -> DoorEnv` (the robot MJCF attached, `robot_base_body` set).  The
runner then calls `reset(door_info, env=env)` with the live environment (the robot already placed at the start
pose), passes `action["ctrl"]` to `env.step` and reads the base position from the robot's base body; torques /
base_velocity are ignored.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys

# name -> "module:Class" of the baselines shipped with DoorBench
BASELINES = {
    "random": "doorbench.benchmark.baselines.random_policy:RandomPolicy",
    "scripted_hand": "doorbench.benchmark.baselines.scripted_hand:ScriptedHandPolicy",
    "g1_locomotion": "doorbench.benchmark.baselines.g1_locomotion:G1LocomotionPolicy",
}


class Policy:
    """Base class (subclassing is optional: any object with reset/act and the class attributes below works)."""

    name: str = "policy"
    description: str = ""
    embodiment: str = "hand_base"      # "hand_base" (reference: DoorEnv hand + synthetic base) | "robot"
    control_dt: float = 0.01           # seconds between act() calls
    requires_tier: str | None = None   # a robot policy may need e.g. "full" (mesh collision)

    def reset(self, door_info: dict, env=None) -> None:
        pass

    def act(self, obs: dict) -> dict:
        return {}

    def close(self) -> None:
        pass

    @classmethod
    def make_env(cls, door_dir: str, tier: str, seed: int):
        """Robot embodiments build their own DoorEnv (robot attached); the reference runner builds a plain one."""
        raise NotImplementedError


def resolve_policy_spec(spec: str) -> str:
    """'random' -> 'doorbench.benchmark.baselines.random_policy:RandomPolicy'; module:Class / file.py:Class pass through."""
    return BASELINES.get(spec, spec)


def load_policy_class(spec: str):
    """Import a policy class from 'module.path:ClassName' or '/path/to/file.py:ClassName' (or a baseline name)."""
    spec = resolve_policy_spec(spec)
    if ":" not in spec:
        raise ValueError(f"policy spec must be 'module:Class' or 'file.py:Class' (or one of {sorted(BASELINES)}), got {spec!r}")
    mod_spec, cls_name = spec.rsplit(":", 1)
    if mod_spec.endswith(".py") or os.sep in mod_spec:
        path = os.path.abspath(mod_spec)
        name = "doorbench_user_policy_" + os.path.splitext(os.path.basename(path))[0]
        if name in sys.modules:
            mod = sys.modules[name]
        else:
            mspec = importlib.util.spec_from_file_location(name, path)
            if mspec is None or mspec.loader is None:
                raise ImportError(f"cannot import policy file {path}")
            mod = importlib.util.module_from_spec(mspec)
            sys.modules[name] = mod
            mspec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_spec)
    try:
        return getattr(mod, cls_name)
    except AttributeError as e:
        raise ImportError(f"{mod_spec} has no class {cls_name}") from e


def policy_meta(cls) -> dict:
    return {"name": getattr(cls, "name", cls.__name__), "class": f"{cls.__module__}:{cls.__name__}", "description": getattr(cls, "description", ""),
            "embodiment": getattr(cls, "embodiment", "hand_base"), "control_dt": float(getattr(cls, "control_dt", 0.01))}

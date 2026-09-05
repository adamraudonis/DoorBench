"""Benchmark scenarios: spec emission, seeded start sampling, and DoorEnv running every scenario type on one door."""
import json
import math
import os

import pytest

from doorbench.spec import generate_all
from doorbench.build import export_door
from doorbench.benchmark.scenarios import SCENARIO_TYPES, assign_scenarios, build_benchmark, make_scenario, sample_start, human_pose


@pytest.fixture(scope="module")
def specs():
    return generate_all()


@pytest.fixture(scope="module")
def door(tmp_path_factory, specs):
    """An unlocked lever + spring-latch swing door without a closer, exported (MJCF + JSON) to a temp dir."""
    pytest.importorskip("mujoco")
    tmp = tmp_path_factory.mktemp("bench")
    s = next(x for x in specs if x["family"] == "swing_single" and x["operator"]["model"].startswith("lever") and not x["lock"]["engaged"] and x["closer"]["model"] == "none" and x["latch"]["model"] != "none")
    export_door(s, str(tmp / "doors"), str(tmp / "hardware"), formats=("mjcf", "json"))
    return str(tmp / "doors" / s["id"])


@pytest.fixture(scope="module")
def locked_door(tmp_path_factory, specs):
    """A locked door that the robot can release from its side (thumbturn deadbolt)."""
    pytest.importorskip("mujoco")
    tmp = tmp_path_factory.mktemp("bench_locked")
    s = next(x for x in specs if x["family"] == "swing_single" and x["lock"]["engaged"] and x["lock"]["robot_side_release"] and x["lock"]["model"] in ("deadbolt_single", "thumbturn_only", "privacy_button"))
    export_door(s, str(tmp / "doors"), str(tmp / "hardware"), formats=("mjcf", "json"))
    return str(tmp / "doors" / s["id"])


def test_assignment_rule(specs):
    counts = {}
    for s in specs:
        names = assign_scenarios(s)
        assert names == assign_scenarios(s)                      # seeded -> deterministic
        assert len(names) == len(set(names))
        if s["family"] == "pet_door":
            assert names == []
            continue
        lock = s["lock"]
        if lock["engaged"] and lock["robot_side_release"]:
            assert names[0] == "unlock_and_traverse"
        elif lock["engaged"]:
            assert names[0] == "locked_recognize"
        else:
            assert names[0] == "open_and_traverse"
        for n in names:
            assert n in SCENARIO_TYPES
            counts[n] = counts.get(n, 0) + 1
    assert counts["open_and_traverse"] + counts["unlock_and_traverse"] + counts["locked_recognize"] == 985
    assert counts["hold_open_for_human"] + counts["wait_for_human"] > 30
    assert counts["open_then_close"] > 100


def test_spec_json_has_benchmark(door):
    spec = json.load(open(os.path.join(door, "spec.json")))
    b = spec["benchmark"]
    assert b["primary_scenario"] == "open_and_traverse"
    sc = b["scenarios"][0]
    for key in ("start", "approach_point", "handle_targets", "pass_plane", "goal", "rewards", "success", "time_budget_s", "expected_transit_s"):
        assert key in sc
    assert sc["handle_targets"], "lever door must expose grip sites"
    assert sc["rewards"]["traversed"] == 10 and sc["rewards"]["damage"] == -10
    assert sc["time_budget_s"] > sc["expected_transit_s"] > 0
    terms = sc["expected_transit_terms"]
    assert abs(sum(terms[k] for k in ("approach_s", "operate_s", "open_s", "pass_s", "scenario_extra_s")) - terms["total_s"]) < 0.05


def test_sample_start_is_seeded_and_in_zone(door):
    spec = json.load(open(os.path.join(door, "spec.json")))
    sc = spec["benchmark"]["scenarios"][0]
    a, b = sample_start(sc, 7), sample_start(sc, 7)
    assert a == b
    assert sample_start(sc, 8) != a
    for seed in range(50):
        p = sample_start(sc, seed)
        c = sc["start"]["center"]
        assert math.hypot(p["xy"][0] - c[0], p["xy"][1] - c[1]) <= sc["start"]["radius"] + 1e-9
        assert sc["start"]["yaw_range"][0] - 1e-9 <= p["yaw"] <= sc["start"]["yaw_range"][1] + 1e-9


def _press_operator(env, frac=0.9, tau_max=2.0, kp=30.0):
    """A hand on the lever: a saturating position servo toward `frac` of the travel (a constant torque would slam the
    lever into its stop and trip the operator-overload damage label).

    `kp` has to be stiff enough that the steady-state error against the lever's return spring is a few percent of the
    travel: a Grade 1 lever needs ~1.6 N*m at 90 % of its throw, so at kp = 6 the hand stalled 15 deg short and the
    latch bolt came out only 73 % - right on the edge of clearing its strike.  The saturation (2 N*m on a 115 mm
    lever = 17 N at the grip) is what keeps this a human-sized hand, not the gain."""
    if env.oj < 0:
        return
    m, d = env.m, env.d
    q, dq = d.qpos[m.jnt_qposadr[env.oj]], d.qvel[m.jnt_dofadr[env.oj]]
    tau = max(-tau_max, min(tau_max, kp * (frac * m.jnt_range[env.oj][1] - q) - 0.2 * dq))
    env.apply_joint_torque(env.meta["operator_joint"], tau)


def _walk_through(env, n_steps, torque_door=25.0, start=None):
    """Programmatic 'robot': turns the handle, pushes the door, and its base walks from the start pose to the goal."""
    sc = env.scenario()
    x0, y0 = (start or env.start_pose)["xy"]
    gx, gy = sc["goal"]["center"][:2] if sc.get("goal") else (x0, y0)
    obs = done = None
    for i in range(n_steps):
        f = i / max(1, n_steps - 1)
        _press_operator(env)
        env.apply_joint_torque(env.meta["primary_joint"], torque_door)
        base = [x0 + (gx - x0) * f, y0 + (gy - y0) * f, 0.9]
        obs, done = env.step(robot_base_pos=base)
        if done:
            break
    return obs, done


def test_every_scenario_type_runs(door):
    from doorbench.benchmark import DoorEnv
    env = DoorEnv(door, tier="full")
    for name in SCENARIO_TYPES:
        obs = env.reset(scenario=name, seed=1)
        assert obs["scenario"] == name and obs["start"] is not None
        sc = env.scenario()
        assert sc["name"] == name
        if sc.get("human"):
            assert "human_xy" in obs
        for _ in range(60):
            _, done = env.step()
        assert env.tracker.L.steps == 60
        assert isinstance(env.reward(), float)
        assert isinstance(env.success, bool)
        L = env.labels()
        assert L.task == name


def test_open_and_traverse_rewards(door):
    from doorbench.benchmark import DoorEnv
    env = DoorEnv(door, tier="full")
    env.reset(scenario="open_and_traverse", seed=2)
    _walk_through(env, 1500)
    fired = set(env._fired)
    assert {"touch_handle", "unlatch", "opened", "traversed"} <= fired, fired
    assert env.success
    L = env.labels()
    assert L.success and L.episode_return > 10
    assert all(e["event"] in env.scenario()["rewards"] for e in L.reward_events)


def test_close_only_starts_open_and_rewards_closing(door):
    from doorbench.benchmark import DoorEnv
    env = DoorEnv(door, tier="full")
    obs = env.reset(scenario="close_only", seed=3)
    assert abs(obs["door_q"]) > 0.5
    m, pj = env.m, env.pj
    for _ in range(4000):                       # pull it shut with a soft servo (no slam)
        q, dq = env.d.qpos[m.jnt_qposadr[pj]], env.d.qvel[m.jnt_dofadr[pj]]
        env.apply_joint_torque(env.meta["primary_joint"], max(-20.0, min(20.0, 30.0 * (0.0 - q) - 8.0 * dq)))
        _, done = env.step(robot_base_pos=[0.0, -1.5, 0.9])
        if "latched" in env._fired:
            break
    assert {"closed", "latched"} <= set(env._fired), env._fired
    assert not env.tracker.L.door_slammed
    assert env.success


def test_hold_open_for_human_moves_human_and_waits_at_closed_door(door):
    from doorbench.benchmark import DoorEnv
    env = DoorEnv(door, tier="full")
    obs = env.reset(scenario="hold_open_for_human", seed=4)
    h = env.scenario()["human"]
    x0, y0 = obs["human_xy"]
    assert abs(x0 - h["path"][0][1]) < 1e-6 and abs(y0 - h["path"][0][2]) < 1e-6
    # nobody opens the door: the human walks up and waits in front of the closed leaf
    n = int((h["path"][-1][0] + 2.0) / env.m.opt.timestep)
    for _ in range(n):
        obs, done = env.step(robot_base_pos=[1.5, -3.0, 0.9])
    hx, hy = obs["human_xy"]
    assert hy < env.scenario()["pass_plane"]["center"][1] - 0.3, "human must not walk through a closed door"
    assert "collision_with_human" not in env._fired
    assert not env.success
    # now open the door and hold it: the human passes and held_for_human fires
    env.reset(scenario="hold_open_for_human", seed=4)
    for _ in range(int((h["path"][-1][0] + 4.0) / env.m.opt.timestep)):
        _press_operator(env)
        env.apply_joint_torque(env.meta["primary_joint"], 30.0)
        obs, done = env.step(robot_base_pos=[-1.2, -2.0, 0.9])
        if "held_for_human" in env._fired:
            break
    assert "held_for_human" in env._fired, (env._fired, obs["human_xy"], obs["door_q"])


def test_wait_for_human_env_opens_door_for_human(door):
    from doorbench.benchmark import DoorEnv
    env = DoorEnv(door, tier="full")
    obs = env.reset(scenario="wait_for_human", seed=5)
    h = env.scenario()["human"]
    max_q = 0.0
    for _ in range(int((h["path"][-1][0] + 1.0) / env.m.opt.timestep)):
        obs, done = env.step(robot_base_pos=[1.0, -3.0, 0.9])     # robot stands aside
        max_q = max(max_q, abs(obs["door_q"]))
    assert max_q > env.scenario()["thresholds"]["clear_rad"] - 0.05, "the person must have opened the door"
    assert "yielded_to_human" in env._fired, env._fired
    assert "collision_with_human" not in env._fired


def test_wait_for_human_collision_is_penalised(door):
    from doorbench.benchmark import DoorEnv
    env = DoorEnv(door, tier="full")
    obs = env.reset(scenario="wait_for_human", seed=6)
    h = env.scenario()["human"]
    for _ in range(int((h["path"][-1][0]) / env.m.opt.timestep)):
        hx, hy = obs["human_xy"]
        obs, done = env.step(robot_base_pos=[hx, hy, 0.9])       # robot walks into the person
    assert "collision_with_human" in env._fired
    assert env.episode_return < 0


def test_unlock_and_traverse_on_locked_door(locked_door):
    from doorbench.benchmark import DoorEnv
    env = DoorEnv(locked_door, tier="full")
    spec = json.load(open(os.path.join(locked_door, "spec.json")))
    assert spec["benchmark"]["primary_scenario"] == "unlock_and_traverse"
    env.reset(scenario="unlock_and_traverse", seed=1)
    assert "unlock" in env.scenario()["rewards"]
    for _ in range(200):
        env.step()
    assert not env.success


def test_locked_recognize_declare(door):
    from doorbench.benchmark import DoorEnv
    env = DoorEnv(door, tier="full")
    env.reset(scenario="locked_recognize", seed=1)
    for _ in range(50):
        _, done = env.step()
    env.declare_locked()
    assert "recognized_locked" in env._fired
    _, done = env.step()
    assert done and env.success


def test_knock_and_wait(door):
    from doorbench.benchmark import DoorEnv
    env = DoorEnv(door, tier="full")
    env.reset(scenario="knock_and_wait", seed=1)
    env.knock()
    assert "knocked" in env._fired
    for _ in range(int(3.2 / env.m.opt.timestep)):
        env.step(robot_base_pos=[0.0, -1.5, 0.9])
    _walk_through(env, 1500)
    assert {"knocked", "waited", "opened", "traversed"} <= set(env._fired), env._fired
    assert env.success


def test_human_pose_interpolates():
    h = {"path": [[1.0, 0.0, -3.0], [3.0, 0.0, -1.0], [4.0, 1.0, -1.0]]}
    assert human_pose(h, 0.0) == (0.0, -3.0)
    assert human_pose(h, 2.0) == (0.0, -2.0)
    assert human_pose(h, 3.5) == (0.5, -1.0)
    assert human_pose(h, 9.0) == (1.0, -1.0)


def test_make_scenario_any_type_any_door(specs):
    from doorbench.build import build_model
    from doorbench import physics as P
    import json as _json
    from doorbench.build import _json_default
    for s in specs[::97]:
        phys = P.derive(s)
        model = _json.loads(_json.dumps(build_model(s, phys).to_dict("full"), default=_json_default))
        b = build_benchmark(s, phys, model)
        if s["family"] == "pet_door":
            assert b["scenarios"] == []
            continue
        assert b["scenarios"] and b["scenarios"][0]["name"] == b["primary_scenario"]
        for n in SCENARIO_TYPES:
            sc = make_scenario(n, s, phys, model)
            assert sc["time_budget_s"] >= 3 * sc["expected_transit_s"]
            if n in ("hold_open_for_human", "wait_for_human"):
                assert sc["human"] and len(sc["human"]["path"]) >= 3


def test_suites_segregate_human_interaction(specs):
    """The core suite (default) never involves a person; the primary scenario of every door is a core scenario."""
    from doorbench.benchmark.scenarios import CORE_SCENARIOS, HUMAN_SCENARIOS, SCENARIO_SUITE, SCENARIO_TYPES, assign_scenarios, scenarios_in_suite
    assert set(CORE_SCENARIOS) | set(HUMAN_SCENARIOS) == set(SCENARIO_TYPES)
    assert not set(CORE_SCENARIOS) & set(HUMAN_SCENARIOS)
    for spec in specs:
        names = assign_scenarios(spec)
        if spec["family"] == "pet_door":
            assert names == []
            continue
        assert SCENARIO_SUITE[names[0]] == "core"
        assert scenarios_in_suite(names, "core") + scenarios_in_suite(names, "human") == sorted(names, key=lambda n: SCENARIO_SUITE[n] == "human")
        assert scenarios_in_suite(names, "all") == list(names)


def test_dataset_suites_segregated():
    import glob
    import json
    from pathlib import Path
    from doorbench.benchmark.scenarios import SCENARIO_SUITE
    files = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "assets" / "doors" / "*" / "spec.json")))
    assert files, "dataset not generated"
    for f in files:
        spec = json.load(open(f))
        b = spec.get("benchmark")
        if spec["family"] == "pet_door":
            # Older downloaded metadata may retain historical scenarios; runtime guards own exclusion.
            if b.get("benchmark_eligibility"):
                assert b["scenarios"] == [] and b["primary_scenario"] is None
            continue
        assert b, f
        assert SCENARIO_SUITE[b["primary_scenario"]] == "core", f
        assert set(b["suites"]) == {"core", "human"} and b["suites"]["core"][0] == b["primary_scenario"], f
        for s in b["scenarios"]:
            assert s["suite"] == SCENARIO_SUITE[s["name"]], f
            assert s["requires_human"] == (s.get("human") is not None), f
            if s["suite"] == "core":
                assert s.get("human") is None, f

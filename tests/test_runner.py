"""Benchmark runner tests (task board R1-R3): policy loading, door selection, scenarios, the synthetic base gate,
a 5-door run of the random and scripted baselines (serial and with a worker pool), determinism, and validation of
the written JSON against results/schema.json + the submission rules.

Run:  python -m pytest -q tests/test_runner.py        (~20 s; skipped when assets/ has not been generated)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

mujoco = pytest.importorskip("mujoco")

from doorbench.benchmark import runner as R  # noqa: E402
from doorbench.benchmark.policy import BASELINES, Policy, load_policy_class, policy_meta  # noqa: E402
from doorbench.benchmark.scenarios import SCENARIOS, Scenario, parse_scenarios, success_of  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.environ.get("DOORBENCH_ASSETS", os.path.join(ROOT, "assets"))
MANIFEST = os.path.join(ASSETS, "manifest.json")
pytestmark = pytest.mark.skipif(not os.path.exists(MANIFEST), reason=f"generated dataset not found at {ASSETS}")

# one lever swing door, a patio slider with a hook lock, a garage sectional, a panic pair, a revolving door
FIVE = "db0002_swing_single,db0345_sliding_single,db0148_garage_sectional,db0019_swing_double,db0066_revolving"


def _load_script(name):
    p = os.path.join(ROOT, "scripts", name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _no_global_passive_callback():
    mujoco.set_mjcb_passive(None)
    yield
    mujoco.set_mjcb_passive(None)


# ----------------------------------------------------------------------------------------------- units
def test_baselines_load():
    for name in ("random", "scripted_hand"):
        cls = load_policy_class(name)
        assert issubclass(cls, Policy)
        meta = policy_meta(cls)
        assert meta["name"] == name and meta["embodiment"] == "hand_base" and meta["control_dt"] > 0
    assert set(BASELINES) >= {"random", "scripted_hand", "g1_locomotion"}
    cls = load_policy_class("doorbench.benchmark.baselines.random_policy:RandomPolicy")
    assert cls.name == "random"
    with pytest.raises((ValueError, ImportError)):
        load_policy_class("no_such_module_xyz:Nope")


def test_policy_from_file(tmp_path):
    p = tmp_path / "my_policy.py"
    p.write_text("from doorbench.benchmark.policy import Policy\n\nclass MyPolicy(Policy):\n    name = 'mine'\n    def act(self, obs):\n        return {'base_velocity': [0.0, 1.0]}\n")
    cls = load_policy_class(f"{p}:MyPolicy")
    assert cls.name == "mine" and cls().act({}) == {"base_velocity": [0.0, 1.0]}


def test_scenarios():
    assert parse_scenarios("default")[0].name == "default"
    assert [s.name for s in parse_scenarios("all")] == list(SCENARIOS)
    assert len(parse_scenarios("default,traverse,default")) == 2
    with pytest.raises(KeyError):
        parse_scenarios("bogus")
    d = SCENARIOS["default"]
    assert d.task_for({"task": "peek"}) == "peek" and SCENARIOS["traverse"].task_for({"task": "peek"}) == "open_and_traverse"
    assert d.budget_for({"lock": {"model": "delayed_egress", "engaged": True}}) >= 40.0
    assert d.budget_for({"benchmark": {"time_budget_s": 7.5}}) == 7.5


def test_success_predicates():
    base = {k: False for k in ("touched_door", "door_opened", "door_open_clear", "robot_passed_through", "door_damaged", "door_slammed", "hardware_misuse", "robot_fell")}
    ok = dict(base, touched_door=True, door_opened=True, door_open_clear=True, robot_passed_through=True)
    d = SCENARIOS["default"]
    assert success_of("open_and_traverse", ok, d, 1.2, True)
    assert not success_of("open_and_traverse", dict(ok, door_damaged=True), d, 1.2, True)
    assert not success_of("open_and_traverse", dict(ok, robot_passed_through=False), d, 1.2, True)
    assert not success_of("open_and_traverse", dict(ok, door_opened=False), d, 1.2, True), "a closed door cannot have been traversed"
    assert not success_of("open_and_traverse", ok, d, 1.2, True, goal_reached=False), "crossing the plane by a few cm is not a traversal"
    assert success_of("unlock_open_traverse", ok, d, 1.2, True), "unlock is evidenced by the door opening"
    assert not success_of("traverse_open", ok, d, 1.2, True) and success_of("traverse_open", dict(ok, touched_door=False), d, 1.2, True)
    assert success_of("open_only", dict(base, touched_door=True, door_opened=True, door_open_clear=True), d, 1.2, True)
    assert not success_of("open_only", dict(base, touched_door=True, door_open_clear=True), d, 0.002, False), "'clear' at 95 % of a 2 mm locked range is not open"
    assert success_of("peek", dict(base, touched_door=True, door_opened=True), d, 0.4, True)
    assert not success_of("peek", ok, d, 1.2, True)
    assert success_of("close", dict(base, touched_door=True), d, 0.01, True) and not success_of("close", dict(base, touched_door=True), d, 0.5, True)
    assert not success_of("close", base, d, 0.01, True), "a door that closed on its own was not closed by the policy"
    assert success_of("locked_recognize", dict(base, touched_door=True), d, 0.0, True)
    assert not success_of("locked_recognize", dict(base, touched_door=True, door_opened=True), d, 0.3, True)
    tc = SCENARIOS["traverse_close"]
    assert success_of("open_and_traverse", ok, tc, 0.0, True) and not success_of("open_and_traverse", ok, tc, 1.0, True)


def test_synthetic_base_gate():
    b = R.SyntheticBase([0.0, -1.5, 0.0], half_opening=0.45)
    for _ in range(2000):
        b.step([0.0, 1.5], 0.002, clear=False)
    assert b.pos[1] == pytest.approx(-R.BASE_RADIUS), "blocked at the wall band while the opening is not clear"
    for _ in range(2000):
        b.step([0.0, 1.5], 0.002, clear=True)
    assert b.pos[1] > 1.0, "walks through once clear"
    narrow = R.SyntheticBase([0.0, -1.5, 0.0], half_opening=0.15)
    for _ in range(3000):
        narrow.step([0.0, 1.5], 0.002, clear=True)
    assert narrow.pos[1] < 0, "a pet-door sized opening cannot be passed"
    fast = R.SyntheticBase([0.0, -1.5, 0.0], half_opening=0.45)
    fast.step([0.0, 100.0], 0.1, clear=True)
    assert fast.pos[1] == pytest.approx(-1.5 + R.BASE_MAX_SPEED * 0.1)


def test_select_doors():
    man = R.load_manifest(ASSETS)
    assert len(R.select_doors(man, "all")) == len([d for d in man["doors"] if not d.get("error")])
    assert [d["id"] for d in R.select_doors(man, FIVE)] == FIVE.split(",")
    assert all(d["family"] == "vault" for d in R.select_doors(man, "family:vault"))
    assert len(R.select_doors(man, "first:7")) == 7
    assert len(R.select_doors(man, "sample:12:3")) == 12 and R.select_doors(man, "sample:12:3") == R.select_doors(man, "sample:12:3")
    assert all(d["lock_engaged"] for d in R.select_doors(man, "lock:locked"))
    with pytest.raises(KeyError):
        R.select_doors(man, "db9999_nope")


# ----------------------------------------------------------------------------------------------- runs
def _check_result(res, n_doors, n_seeds, tmp_path, name):
    vr = _load_script("validate_result.py")
    with open(os.path.join(ROOT, "results", "schema.json")) as f:
        schema = json.load(f)
    man = R.load_manifest(ASSETS)
    out = tmp_path / f"{name}.json"
    R.write_result(res, str(out))
    errs = vr.validate_file(str(out), schema, man, submission=False)
    assert not errs, errs
    with open(out) as f:
        doc = json.load(f)
    assert doc["schema_version"] == R.SCHEMA_VERSION
    assert doc["run"]["n_doors"] == n_doors and doc["run"]["seeds"] == list(range(n_seeds))
    assert len(doc["episodes"]) == n_doors * n_seeds
    a = doc["aggregate"]
    for k in ("n_doors", "n_episodes", "n_success", "success_rate", "doors_solved", "doors_solved_any", "damage_rate", "by_family", "by_task", "by_difficulty", "by_scenario", "by_lock_state"):
        assert k in a, k
    assert a["n_episodes"] == n_doors * n_seeds and 0 <= a["success_rate"] <= 1
    for e in doc["episodes"]:
        assert e["outcome"] in R.OUTCOMES and e["outcome"] != "error", e.get("error")
        assert e["sim_time"] > 0 and e["steps"] > 0 and e["wall_s"] >= 0
        assert all(isinstance(ev[0], str) and isinstance(ev[1], (int, float)) for ev in e["events"])
        assert e["success"] == (e["outcome"] == "success")
        if e["success"]:
            assert not e["damage"]
    # the submission rules reject a 5-door run
    errs = vr.validate_file(str(out), schema, man, submission=True)
    assert any("all" in x and "doors" in x for x in errs), errs
    return doc


def test_scripted_hand_5_doors(tmp_path):
    res = R.run_benchmark("scripted_hand", doors=FIVE, seeds=1, scenarios="default", workers=1, tier="full", assets=ASSETS, progress=lambda *_: None)
    doc = _check_result(res, 5, 1, tmp_path, "scripted")
    a = doc["aggregate"]
    assert a["n_success"] >= 4, {e["door_id"]: (e["outcome"], e["events"]) for e in doc["episodes"]}
    lever = next(e for e in doc["episodes"] if e["door_id"] == "db0002_swing_single")
    names = [ev[0] for ev in lever["events"]]
    assert lever["success"] and lever["time_to_pass"] is not None
    assert names.index("operator_actuated") < names.index("door_opened") < names.index("robot_passed_through")
    assert a["mean_wall_s"] < 2.0, "one episode must stay well under 2 s of wall time"


def test_random_5_doors_pool(tmp_path):
    res = R.run_benchmark("random", doors=FIVE, seeds=2, scenarios="default", workers=2, tier="full", assets=ASSETS, progress=lambda *_: None)
    doc = _check_result(res, 5, 2, tmp_path, "random")
    assert {e["seed"] for e in doc["episodes"]} == {0, 1}
    assert any(e["randomized"] for e in doc["episodes"]) and any(not e["randomized"] for e in doc["episodes"])


def test_episode_is_deterministic():
    man = R.load_manifest(ASSETS)
    door = next(d for d in man["doors"] if d["id"] == "db0002_swing_single")
    R._init_worker("scripted_hand")
    job = R.Job(door=door, door_dir=os.path.join(ASSETS, "doors", door["id"]), scenario="default", seed=1, tier="full", policy_spec="scripted_hand")
    a, b = R.run_episode(job), R.run_episode(job)
    for k in ("success", "outcome", "sim_time", "steps", "events", "time_to_pass", "door_q_end", "energy_J"):
        assert a[k] == b[k], k


def test_traverse_scenario_locked_door_fails():
    """A locked door without a robot-side release stays shut under the traverse scenario (no exploit through the lock parts)."""
    man = R.load_manifest(ASSETS)
    locked = next(d for d in man["doors"] if d["lock_engaged"] and not d["robot_side_release"] and d["family"] == "swing_single" and d["lock"] in ("deadbolt_single", "keyed_cylinder"))
    R._init_worker("scripted_hand")
    job = R.Job(door=locked, door_dir=os.path.join(ASSETS, "doors", locked["id"]), scenario="traverse", seed=0, tier="full", policy_spec="scripted_hand")
    ep = R.run_episode(job)
    assert ep["outcome"] != "error", ep.get("error")
    assert not ep["success"] and not ep["labels"]["door_opened"], ep


def test_cli_dry_run_and_lists(capsys):
    from doorbench.cli import main
    main(["benchmark", "list-scenarios"])
    out = capsys.readouterr().out
    assert "default" in out and "traverse_close" in out
    main(["benchmark", "run", "--policy", "random", "--doors", FIVE, "--seeds", "2", "--assets", ASSETS, "--dry-run"])
    out = capsys.readouterr().out
    assert "5 doors" in out and "db0066_revolving" in out


def test_build_results_index(tmp_path, monkeypatch):
    bi = _load_script("build_results_index.py")
    res = R.run_benchmark("scripted_hand", doors="db0002_swing_single,db0345_sliding_single", seeds=1, scenarios="default", workers=1, tier="full", assets=ASSETS, progress=lambda *_: None)
    rdir = tmp_path / "results"
    rdir.mkdir()
    with open(os.path.join(ROOT, "results", "schema.json")) as f:
        (rdir / "schema.json").write_text(f.read())
    R.write_result(res, str(rdir / "scripted_hand.json"))
    monkeypatch.setattr(bi, "RESULTS", str(rdir))
    monkeypatch.setattr(bi, "ROOT", str(tmp_path))      # never touch the repo's README.md / manifest from a test
    assert bi.build() == 0
    idx = json.loads((rdir / "index.json").read_text())
    assert idx["results"][0]["policy"] == "scripted_hand" and set(idx["results"][0]["doors"]) == {"db0002_swing_single", "db0345_sliding_single"}
    assert "| policy |" in (rdir / "README.md").read_text()
    assert bi.build(check=True) == 0

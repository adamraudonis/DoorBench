"""Benchmark runner tests (task board R1-R3): policy loading, door / scenario selection and suites, the synthetic base
gate, small runs of the random / scripted baselines in the core suite (serial and with a worker pool) and of the
scripted hand in the human suite, determinism, and validation of the written JSON against results/schema.json +
the submission rules (including the rejection of mixed core / human tables).

Run:  python -m pytest -q tests/test_runner.py        (~40 s; skipped when assets/ has not been generated)
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
from doorbench.benchmark.scenarios import CORE_SCENARIOS, HUMAN_SCENARIOS, SCENARIO_SUITE, SCENARIO_TYPES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.environ.get("DOORBENCH_ASSETS", os.path.join(ROOT, "assets"))
MANIFEST = os.path.join(ASSETS, "manifest.json")
pytestmark = pytest.mark.skipif(not os.path.exists(MANIFEST), reason=f"generated dataset not found at {ASSETS}")

# one knob swing door (open_and_traverse + open_then_close + knock_and_wait), a patio slider with a hook lock
# (unlock_and_traverse + open_then_close + close_only), a garage sectional, a panic pair, a revolving door
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


def test_suites_and_scenario_selection():
    """The core suite is the default and never contains a human scenario; the validator's inline table matches the package."""
    vr = _load_script("validate_result.py")
    assert vr.SUITE_OF == SCENARIO_SUITE and set(vr.CORE_SCENARIOS) == set(CORE_SCENARIOS) and set(vr.HUMAN_SCENARIOS) == set(HUMAN_SCENARIOS)
    man = R.load_manifest(ASSETS)
    doors = R.select_doors(man, "all")
    assert all(R.door_scenarios(d, "core") for d in doors), "every door has a core scenario (the primary one)"
    assert all(SCENARIO_SUITE[s] == "core" for d in doors for s in R.door_scenarios(d, "core"))
    assert all(SCENARIO_SUITE[s] == "human" for d in doors for s in R.door_scenarios(d, "human"))
    n_human = sum(1 for d in doors if R.door_scenarios(d, "human"))
    assert 0 < n_human < 200
    assert R.parse_scenarios(None, "core") is None and R.parse_scenarios("all", "human") is None
    assert R.parse_scenarios("open_then_close,close_only", "core") == ["open_then_close", "close_only"]
    assert R.parse_scenarios("primary", "core") == ["primary"]
    with pytest.raises(ValueError):
        R.parse_scenarios("knock_and_wait", "core")        # a human scenario is not part of the core suite
    assert R.parse_scenarios("knock_and_wait", "all") == ["knock_and_wait"]
    with pytest.raises(KeyError):
        R.parse_scenarios("bogus", "core")
    d = next(x for x in doors if x["id"] == "db0002_swing_single")
    assert R.scenarios_for(d, "core", None) == ["open_and_traverse", "open_then_close"]
    assert R.scenarios_for(d, "human", None) == ["knock_and_wait"]
    assert R.scenarios_for(d, "all", None) == ["open_and_traverse", "open_then_close", "knock_and_wait"]
    assert R.scenarios_for(d, "core", ["primary"]) == ["open_and_traverse"]
    assert R.scenarios_for(d, "core", ["close_only"]) == []
    jobs = R.make_jobs([d], ASSETS, "core", None, [0, 1], "full", "random")
    assert [(j.scenario, j.seed, j.suite) for j in jobs] == [("open_and_traverse", 0, "core"), ("open_and_traverse", 1, "core"), ("open_then_close", 0, "core"), ("open_then_close", 1, "core")]


def test_synthetic_base_gate():
    b = R.SyntheticBase([0.0, -1.5], half_opening=0.45)
    for _ in range(2000):
        b.step([0.0, 1.5], 0.002, clear=False)
    assert b.pos[1] == pytest.approx(-R.BASE_RADIUS), "blocked at the wall band while the opening is not clear"
    for _ in range(2000):
        b.step([0.0, 1.5], 0.002, clear=True)
    assert b.pos[1] > 1.0, "walks through once clear"
    narrow = R.SyntheticBase([0.0, -1.5], half_opening=0.15)
    for _ in range(3000):
        narrow.step([0.0, 1.5], 0.002, clear=True)
    assert narrow.pos[1] < 0, "a pet-door sized opening cannot be passed"
    fast = R.SyntheticBase([0.0, -1.5], half_opening=0.45)
    fast.step([0.0, 100.0], 0.1, clear=True)
    assert fast.pos[1] == pytest.approx(-1.5 + R.BASE_MAX_SPEED * 0.1)


def test_select_doors():
    man = R.load_manifest(ASSETS)
    assert len(R.select_doors(man, "all")) == len([d for d in man["doors"] if not d.get("error")])
    assert [d["id"] for d in R.select_doors(man, FIVE)] == FIVE.split(",")
    assert all(d["family"] == "vault" for d in R.select_doors(man, "family:vault"))
    assert len(R.select_doors(man, "first:7")) == 7
    assert len(R.select_doors(man, "every:4")) == 250 and R.select_doors(man, "every:4")[1]["index"] == R.select_doors(man, "every:4")[0]["index"] + 4
    assert len(R.select_doors(man, "sample:12:3")) == 12 and R.select_doors(man, "sample:12:3") == R.select_doors(man, "sample:12:3")
    assert all(d["lock_engaged"] for d in R.select_doors(man, "lock:locked"))
    assert all("knock_and_wait" in R.door_scenarios(d, "all") for d in R.select_doors(man, "scenario:knock_and_wait"))
    with pytest.raises(KeyError):
        R.select_doors(man, "db9999_nope")


# ----------------------------------------------------------------------------------------------- runs
def _validate(res, tmp_path, name, submission=False):
    vr = _load_script("validate_result.py")
    with open(os.path.join(ROOT, "results", "schema.json")) as f:
        schema = json.load(f)
    man = R.load_manifest(ASSETS)
    out = tmp_path / f"{name}.json"
    R.write_result(res, str(out))
    errs = vr.validate_file(str(out), schema, man, submission=submission)
    with open(out) as f:
        doc = json.load(f)
    return doc, errs


def _check_result(res, n_doors, n_episodes, tmp_path, name, suite="core"):
    doc, errs = _validate(res, tmp_path, name)
    assert not errs, errs
    assert doc["schema_version"] == R.SCHEMA_VERSION and doc["run"]["suite"] == suite
    assert doc["run"]["n_doors"] == n_doors and len(doc["episodes"]) == n_episodes
    assert list(doc["aggregate"]) == [suite], "one table per suite, nothing mixed"
    a = doc["aggregate"][suite]
    assert a["suite"] == suite
    for k in ("n_doors", "n_episodes", "n_success", "success_rate", "doors_solved", "doors_solved_any", "damage_rate", "by_family", "by_scenario", "by_difficulty", "by_lock_state"):
        assert k in a, k
    assert a["n_episodes"] == n_episodes and 0 <= a["success_rate"] <= 1
    assert all(SCENARIO_SUITE[s] == suite for s in a["scenarios"]) and all(SCENARIO_SUITE[s] == suite for s in a["by_scenario"])
    for e in doc["episodes"]:
        assert e["suite"] == suite == SCENARIO_SUITE[e["scenario"]]
        assert e["outcome"] in R.OUTCOMES and e["outcome"] != "error", e.get("error")
        assert e["sim_time"] > 0 and e["steps"] > 0 and e["wall_s"] >= 0
        assert all(isinstance(ev[0], str) and isinstance(ev[1], (int, float)) for ev in e["events"])
        assert e["success"] == (e["outcome"] == "success")
        assert set(e["criteria"]) and all(e["criteria"].values()) == e["success"], (e["door_id"], e["scenario"], e["criteria"])
        if e["success"]:
            assert not e["damage"]
    # the submission rules reject a partial run
    _, errs = _validate(res, tmp_path, name + "_sub", submission=True)
    assert any("must cover all" in x for x in errs), errs
    return doc


def test_scripted_hand_5_doors_core(tmp_path):
    res = R.run_benchmark("scripted_hand", doors=FIVE, seeds=1, suite="core", workers=1, tier="full", assets=ASSETS, progress=lambda *_: None)
    doc = _check_result(res, 5, 9, tmp_path, "scripted")
    a = doc["aggregate"]["core"]
    assert a["n_success"] >= 7, {(e["door_id"], e["scenario"]): (e["outcome"], e["criteria"]) for e in doc["episodes"]}
    lever = next(e for e in doc["episodes"] if e["door_id"] == "db0002_swing_single" and e["scenario"] == "open_and_traverse")
    names = [ev[0] for ev in lever["events"]]
    assert lever["success"] and lever["time_to_pass"] is not None and lever["time_to_goal"] is not None
    assert names.index("operator_actuated") < names.index("door_opened") < names.index("robot_passed_through") < names.index("goal_reached")
    assert [r[0] for r in lever["reward_events"]][:4] == ["touch_handle", "unlatch", "opened", "traversed"] and lever["episode_return"] > 10
    otc = next(e for e in doc["episodes"] if e["door_id"] == "db0002_swing_single" and e["scenario"] == "open_then_close")
    assert otc["success"] and otc["criteria"]["closed_behind"] and otc["criteria"]["!slam"]
    close = next(e for e in doc["episodes"] if e["scenario"] == "close_only")
    assert close["success"] and close["door_q_end"] is not None and abs(close["door_q_end"]) < 0.03
    unlock = next(e for e in doc["episodes"] if e["scenario"] == "unlock_and_traverse")
    assert unlock["success"] and unlock["labels"]["lock_released"]
    assert a["mean_wall_s"] < 2.0, "one episode must stay well under 2 s of wall time"


def test_random_5_doors_pool(tmp_path):
    res = R.run_benchmark("random", doors=FIVE, seeds=2, suite="core", scenarios="primary", workers=2, tier="full", assets=ASSETS, progress=lambda *_: None)
    doc = _check_result(res, 5, 10, tmp_path, "random")
    assert {e["seed"] for e in doc["episodes"]} == {0, 1}
    assert any(e["randomized"] for e in doc["episodes"]) and any(not e["randomized"] for e in doc["episodes"])
    assert doc["run"]["scenario_filter"] == "primary"


def test_scripted_hand_human_suite(tmp_path):
    """The human suite is opt-in: it runs only the doors that list a human scenario, in its own table."""
    man = R.load_manifest(ASSETS)
    pick = {}
    for d in man["doors"]:
        for s in R.door_scenarios(d, "human"):
            pick.setdefault(s, d["id"])
    assert set(pick) == set(HUMAN_SCENARIOS)
    ids = ",".join(pick[s] for s in HUMAN_SCENARIOS)
    res = R.run_benchmark("scripted_hand", doors=ids, seeds=1, suite="human", workers=1, tier="full", assets=ASSETS, progress=lambda *_: None)
    doc = _check_result(res, 3, 3, tmp_path, "hand_human", suite="human")
    by = {e["scenario"]: e for e in doc["episodes"]}
    assert by["knock_and_wait"]["success"] and [ev[0] for ev in by["knock_and_wait"]["events"]][0] == "knock"
    assert by["hold_open_for_human"]["criteria"]["held_for_human"] and not by["hold_open_for_human"]["human_collision"]
    assert by["wait_for_human"]["criteria"]["yielded_to_human"] and not by["wait_for_human"]["human_collision"]
    assert "human_collision_rate" in doc["aggregate"]["human"]
    # the same doors in the core suite never see a person
    res2 = R.run_benchmark("scripted_hand", doors=ids, seeds=1, suite="core", scenarios="primary", workers=1, tier="full", assets=ASSETS, progress=lambda *_: None)
    assert all(e["human_collision"] is None and e["suite"] == "core" for e in res2["episodes"])


def test_validator_rejects_mixed_tables(tmp_path):
    """A table that mixes core and human episodes, or a mislabelled suite, fails validation."""
    man = R.load_manifest(ASSETS)
    d = next(x for x in man["doors"] if x["id"] == "db0002_swing_single")
    res = R.run_benchmark("scripted_hand", doors=d["id"], seeds=1, suite="all", workers=1, tier="full", assets=ASSETS, progress=lambda *_: None)
    doc, errs = _validate(res, tmp_path, "both")
    assert not errs and set(doc["aggregate"]) == {"core", "human"} and doc["run"]["suite"] == "all"
    bad = json.loads(json.dumps(doc))
    bad["episodes"][0]["suite"] = "human"                              # a core scenario labelled human
    _, errs = _validate(bad, tmp_path, "bad1")
    assert any("belongs to the core suite" in e for e in errs), errs
    bad = json.loads(json.dumps(doc))
    bad["aggregate"]["core"]["by_scenario"]["knock_and_wait"] = bad["aggregate"]["human"]["by_scenario"]["knock_and_wait"]
    _, errs = _validate(bad, tmp_path, "bad2")
    assert any("mixed table" in e for e in errs), errs
    bad = json.loads(json.dumps(doc))
    bad["aggregate"]["all"] = bad["aggregate"].pop("human")           # no 'all' table
    _, errs = _validate(bad, tmp_path, "bad3")
    assert errs
    bad = json.loads(json.dumps(doc))
    bad["aggregate"]["human"]["suite"] = "core"
    _, errs = _validate(bad, tmp_path, "bad4")
    assert any("mislabelled" in e for e in errs), errs


def test_episode_is_deterministic():
    man = R.load_manifest(ASSETS)
    door = next(d for d in man["doors"] if d["id"] == "db0002_swing_single")
    R._init_worker("scripted_hand")
    job = R.Job(door=door, door_dir=os.path.join(ASSETS, "doors", door["id"]), scenario="open_and_traverse", seed=1, tier="full", policy_spec="scripted_hand")
    a, b = R.run_episode(job), R.run_episode(job)
    for k in ("success", "outcome", "sim_time", "steps", "events", "time_to_pass", "door_q_end", "energy_J", "episode_return", "start"):
        assert a[k] == b[k], k
    job2 = R.Job(door=door, door_dir=os.path.join(ASSETS, "doors", door["id"]), scenario="open_and_traverse", seed=2, tier="full", policy_spec="scripted_hand")
    assert R.run_episode(job2)["start"] != a["start"], "the start pose is drawn from the start zone per seed"


def test_locked_recognize_needs_no_release():
    """A locked door without a robot-side release: the hand tries gently, declares it locked, and the door stays shut."""
    man = R.load_manifest(ASSETS)
    locked = next(d for d in man["doors"] if d["lock_engaged"] and not d["robot_side_release"] and d["family"] == "swing_single" and d["lock"] in ("deadbolt_single", "keyed_cylinder") and "locked_recognize" in R.door_scenarios(d, "core"))
    R._init_worker("scripted_hand")
    job = R.Job(door=locked, door_dir=os.path.join(ASSETS, "doors", locked["id"]), scenario="locked_recognize", seed=0, tier="full", policy_spec="scripted_hand")
    ep = R.run_episode(job)
    assert ep["outcome"] != "error", ep.get("error")
    assert not ep["labels"]["door_opened"], ep
    assert any(ev[0] == "declare_locked" for ev in ep["events"]) and "recognized_locked" in [r[0] for r in ep["reward_events"]]
    assert ep["success"] and ep["sim_time"] < ep["time_budget_s"], "declaring ends the episode early"


def test_cli_dry_run_and_lists(capsys):
    from doorbench.cli import main
    main(["benchmark", "list-scenarios"])
    out = capsys.readouterr().out
    assert "open_and_traverse" in out and "suite=core" in out and "knock_and_wait" in out and "suite=human" in out
    main(["benchmark", "run", "--policy", "random", "--doors", FIVE, "--seeds", "2", "--assets", ASSETS, "--dry-run"])
    out = capsys.readouterr().out
    assert "suite core: 5 doors" in out and "db0066_revolving" in out and "knock_and_wait" not in out
    main(["benchmark", "run", "--policy", "random", "--doors", FIVE, "--seeds", "1", "--suite", "human", "--assets", ASSETS, "--dry-run"])
    out = capsys.readouterr().out
    assert "suite human: 1 doors (5 selected)" in out and "knock_and_wait" in out


def test_build_results_index(tmp_path, monkeypatch):
    bi = _load_script("build_results_index.py")
    res = R.run_benchmark("scripted_hand", doors="db0002_swing_single,db0345_sliding_single", seeds=1, suite="all", workers=1, tier="full", assets=ASSETS, progress=lambda *_: None)
    rdir = tmp_path / "results"
    rdir.mkdir()
    with open(os.path.join(ROOT, "results", "schema.json")) as f:
        (rdir / "schema.json").write_text(f.read())
    R.write_result(res, str(rdir / "scripted_hand.json"))
    monkeypatch.setattr(bi, "RESULTS", str(rdir))
    monkeypatch.setattr(bi, "ROOT", str(tmp_path))      # never touch the repo's README.md / manifest from a test
    assert bi.build() == 0
    idx = json.loads((rdir / "index.json").read_text())
    r = idx["results"][0]
    assert r["policy"] == "scripted_hand" and set(r["suites"]) == {"core", "human"} and not r["leaderboard"]
    assert set(r["suites"]["core"]["doors"]) == {"db0002_swing_single", "db0345_sliding_single"} and set(r["suites"]["human"]["doors"]) == {"db0002_swing_single"}
    md = (rdir / "README.md").read_text()
    assert "### Core suite" in md and "### Human suite" in md
    assert bi.build(check=True) == 0

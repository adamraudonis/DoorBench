"""Keypad codes are real: the buttons are pressable hardware and the lock only opens for the right code.

Background: the keypad on a code-lock door was a decoration.  `spec.lock.code` existed, the buttons existed as
bodies, but nothing enforced them - an engaged `keypad_code_4` door opened by turning the lever, and `DoorEnv`
released keypad locks from an API flag.  These tests pin the state machine (order, timeout, lockout, the
mechanical set + lever), the hardware (stroke and spring of every button), the physical release (clutch / bolt
motor) and the scenario block that hands a policy the code.

Run:  pytest -q tests/test_keypad_codes.py      (~60 s)
"""
from __future__ import annotations

import json
import math
import os

import pytest

from doorbench import hardware as H
from doorbench.build import build_model, export_door
from doorbench.keypad import CodeLock, keypad_for
from doorbench.keypad_qa import run_keypad_qa
from doorbench.physics import derive
from doorbench.spec import generate_all, keypad_code
from doorbench.benchmark.scenarios import make_scenario

# one door per lock model / release path (electronic clutch, motorised deadbolt, mechanical Simplex, unlocked)
DOORS = ("db0526_swing_single", "db0086_swing_single", "db0166_swing_single", "db0165_swing_single")


@pytest.fixture(scope="module")
def specs():
    return {s["id"]: s for s in generate_all()}


@pytest.fixture(scope="module")
def built(tmp_path_factory, specs):
    """Export the sample doors once and load the full-tier MJCF."""
    import mujoco

    root = str(tmp_path_factory.mktemp("keypad"))
    out = {}
    for did in DOORS:
        spec = specs[did]
        summary = export_door(spec, os.path.join(root, "doors"), os.path.join(root, "hardware"), formats=("mjcf", "json"))
        m = mujoco.MjModel.from_xml_path(summary["files"]["mjcf"]["full"])
        model_json = json.load(open(os.path.join(root, "doors", did, "model.json")))
        out[did] = (spec, m, model_json)
    return out


# ---------------------------------------------------------------------------
# the state machine (no simulator)
# ---------------------------------------------------------------------------
def _seq(code="4821", **kw):
    cfg = {"code": code, "code_kind": "sequence", "code_timeout_s": 5.0, "lockout_s": 30.0, "max_attempts": 3}
    cfg.update(kw)
    return CodeLock(cfg)


def test_sequence_right_code_unlocks():
    lk = _seq()
    for i, c in enumerate("4821"):
        lk.press(c, 0.1 * i)
    assert lk.unlocked and lk.code_entered and lk.wrong_attempts == 0


def test_sequence_wrong_code_does_not_unlock():
    lk = _seq()
    for i, c in enumerate("4812"):
        lk.press(c, 0.1 * i)
    assert not lk.unlocked and lk.wrong_attempts == 1


def test_sequence_order_matters():
    lk = _seq()
    for i, c in enumerate("1248"):      # the right digits, the wrong order
        lk.press(c, 0.1 * i)
    assert not lk.unlocked


def test_sequence_timeout_clears_a_partial_entry():
    lk = _seq()
    lk.press("4", 0.0)
    lk.press("8", 0.2)
    lk.tick(6.0)                         # 5 s of inactivity
    assert lk.entered == []
    lk.press("2", 6.1)
    lk.press("1", 6.2)
    assert not lk.unlocked and lk.wrong_attempts == 0     # not a wrong attempt, just gone
    lk.tick(12.0)                        # ... and the half entry goes the same way (5 s after the last press)
    assert lk.entered == []
    for i, c in enumerate("4821"):       # entered fresh, it still works
        lk.press(c, 12.0 + 0.1 * i)
    assert lk.unlocked


def test_sequence_lockout_after_three_wrong_codes():
    lk = _seq()
    for a in range(3):
        for i, c in enumerate("4822"):
            lk.press(c, a + 0.1 * i)
    assert lk.locked_out and lk.wrong_attempts == 3
    for i, c in enumerate("4821"):       # the right code is ignored while the keypad is frozen
        lk.press(c, 3.0 + 0.1 * i)
    assert not lk.unlocked
    lk.tick(3.0 + 30.0)                  # lockout expires
    assert not lk.locked_out
    for i, c in enumerate("4821"):
        lk.press(c, 34.0 + 0.1 * i)
    assert lk.unlocked


def test_mechanical_set_needs_the_lever_and_ignores_order():
    lk = CodeLock({"code": "234", "code_kind": "set"})
    for i, c in enumerate("432"):        # any order
        lk.press(c, 0.1 * i)
    assert not lk.unlocked               # ... but nothing happens until the lever is turned
    lk.lever(1.0)
    assert lk.unlocked and lk.code_entered


def test_mechanical_wrong_set_is_cleared_by_the_lever():
    lk = CodeLock({"code": "234", "code_kind": "set"})
    lk.press("2", 0.0)
    lk.press("5", 0.1)                   # a button that is not in the combination
    lk.lever(0.5)
    assert not lk.unlocked and lk.wrong_attempts == 1 and lk.entered == []
    lk.press("3", 1.0)                   # the rest of the combination alone is not enough: "2" was cleared
    lk.press("4", 1.1)
    lk.lever(1.5)
    assert not lk.unlocked
    for c in "234":
        lk.press(c, 2.0)
    lk.lever(2.5)
    assert lk.unlocked


def test_mechanical_has_no_timeout_or_lockout():
    lk = CodeLock({"code": "234", "code_kind": "set"})
    lk.press("2", 0.0)
    lk.tick(600.0)                       # a mechanical chamber holds the buttons for ever
    assert lk.entered == ["2"]


# ---------------------------------------------------------------------------
# the spec's codes
# ---------------------------------------------------------------------------
def test_every_keypad_lock_has_a_usable_code(specs):
    n = 0
    for s in specs.values():
        lk = H.LOCKS[s["lock"]["model"]]
        if lk.kind != "keypad_code":
            continue
        n += 1
        code = s["lock"].get("code")
        assert code, f"{s['id']}: keypad lock without a code"
        kp = H.KEYPADS[lk.id]
        assert set(code) <= set(kp.labels), f"{s['id']}: code {code} uses buttons this keypad does not have"
        if kp.code_kind == "set":
            # a mechanical combination chamber holds each button at most once
            assert len(set(code)) == len(code), f"{s['id']}: Simplex combination {code} repeats a button"
        else:
            assert len(code) == (6 if "6" in lk.id else 4)
    assert n >= 20, "the dataset should still have keypad doors"


def test_mechanical_codes_are_distinct_buttons():
    import random

    lk = H.LOCKS["keypad_mechanical"]
    for seed in range(200):
        code = keypad_code(lk, random.Random(seed))
        assert 3 <= len(code) <= 4 and len(set(code)) == len(code) and set(code) <= set("12345")


# ---------------------------------------------------------------------------
# the hardware
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("did", DOORS)
def test_buttons_are_real_bodies(built, did):
    spec, m, model_json = built[did]
    kp = model_json["meta"]["keypad"]
    kmodel = H.KEYPADS[kp["keypad_model"]]
    assert len(kp["buttons"]) == len(kmodel.labels)
    bodies = {b["name"]: b for b in model_json["bodies"]}
    for b in kp["buttons"]:
        body = bodies[b["body"]]
        j = body["joint"]
        assert j["type"] == "slide" and j["name"] == b["joint"]
        assert j["range"][1] == pytest.approx(kmodel.travel)
        # the spring: preload at rest, press_force bottomed out
        assert abs(j["stiffness"] * j["springref"]) == pytest.approx(kmodel.preload_force, rel=1e-3)
        assert j["stiffness"] * (kmodel.travel - j["springref"]) == pytest.approx(kmodel.press_force, rel=1e-3)


@pytest.mark.parametrize("did", DOORS)
def test_keypad_gate_passes(built, did):
    """The QA gate itself: right code opens, wrong code holds, timeout / lockout behave."""
    import mujoco

    from doorbench.qa import qa_push

    spec, m, model_json = built[did]
    meta = model_json["meta"]
    phys = derive(spec)
    pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["primary_joint"])
    oj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["operator_joint"])
    push = qa_push(m, mujoco.MjData(m), pj, phys["mass"]["total_kg"], spec["leaf"]["width"])["push"]
    res = run_keypad_qa(m, spec, meta, phys, push, oj, pj)
    assert res["ok"], (did, res["checks"], {k: v for k, v in res["metrics"].items() if k != "events"})
    if spec["lock"]["engaged"]:
        # the gate is meaningful: the right code opens the door far, the wrong one does not move it
        assert res["metrics"]["opened_after_code_rad"] > math.radians(20)
        assert res["metrics"]["opened_after_wrong_rad"] < math.radians(2)


@pytest.mark.parametrize("did", ("db0526_swing_single", "db0166_swing_single"))
def test_locked_outside_lever_retracts_nothing(built, did):
    """The declutched outside lever jiggles in the lock's free play and does not touch the latch."""
    import mujoco

    spec, m, model_json = built[did]
    kp = model_json["meta"]["keypad"]
    assert kp["release"] == "physical_catch"
    j = m.joint(kp["clutch_joint"]).id
    row = model_json["meta"]["rotary_locksets"][0]
    original = m.jnt_range.copy()
    assert m.jnt_range[j][1] >= row['operator_travel_rad']
    d = mujoco.MjData(m)
    mujoco.mj_forward(m,d)
    keypad = keypad_for(mujoco,m,model_json['meta'],spec)
    bolt=m.joint(row['latch_joint']).id
    for _ in range(round(.5/m.opt.timestep)):
        d.qfrc_applied[:] = 0
        keypad.turn(d)
        mujoco.mj_step(m,d)
    import numpy as np
    assert np.array_equal(m.jnt_range,original)
    assert float(d.qpos[m.jnt_qposadr[j]]) < .06
    assert float(d.qpos[m.jnt_qposadr[bolt]]) < .002


# ---------------------------------------------------------------------------
# environment + scenario
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("did", ("db0526_swing_single", "db0086_swing_single", "db0166_swing_single"))
def test_env_enter_code(built, did, tmp_path_factory, specs):
    """DoorEnv: the wrong code is refused (and counted), the right one releases the lock and the door then opens."""
    import mujoco

    from doorbench.benchmark.env import DoorEnv

    root = str(tmp_path_factory.mktemp("env"))
    export_door(specs[did], os.path.join(root, "doors"), os.path.join(root, "hardware"), formats=("mjcf", "json"))
    env = DoorEnv(os.path.join(root, "doors", did), tier="full")
    env.reset(seed=0)
    assert env.keypad is not None
    code = env.keypad.cfg["code"]
    wrong = ("1" if code[0] != "1" else "2") + code[1:]
    assert env.enter_code(wrong) is False
    assert env.keypad.lock.wrong_attempts == 1
    assert env.enter_code() is True
    L = env.labels()
    assert L.credential_accepted and L.code_entered and L.wrong_code_attempts == 1
    rel = env.keypad.clutch if env.keypad.release_mode in ("clutch","physical_catch") else env.oj
    for _ in range(round(2./env.m.opt.timestep)):env.step()
    assert env.labels().lock_released
    import numpy as np
    from doorbench.qa import qa_push
    push=qa_push(env.m,mujoco.MjData(env.m),env.pj,env.spec['physics']['mass']['dynamics_mass_kg'],specs[did]['leaf']['width'],env.meta)['push']
    row=env.meta['rotary_locksets'][0]
    site=env.m.site(row['input_sites'][row['outside_joint']][0]).id
    for _ in range(round(3./env.m.opt.timestep)):
        if rel >= 0:
            env.keypad.turn(env.d)
        tangent=np.cross(env.d.xaxis[env.pj],env.d.site_xpos[site]-env.d.xanchor[env.pj])
        radius=float(np.linalg.norm(tangent))
        force=tangent/radius*min(120.,push/radius)
        mujoco.mj_applyFT(env.m,env.d,force,np.zeros(3),env.d.site_xpos[site],int(env.m.site_bodyid[site]),env.d.qfrc_applied)
        env.step()
    assert float(env.d.qpos[env.m.jnt_qposadr[env.pj]]) > math.radians(20)
    env.close()


def test_scenario_carries_the_code(built):
    spec, m, model_json = built["db0526_swing_single"]
    sc = make_scenario("unlock_and_traverse", spec, derive(spec), model_json)
    lock = sc["lock"]
    assert lock["code"] == spec["lock"]["code"]
    assert lock["code_kind"] == "sequence" and lock["release"] == "physical_catch"
    assert len(lock["buttons"]) == 10 and all(b["pos"] and b["joint"] for b in lock["buttons"])
    assert lock["code_timeout_s"] == 5.0 and lock["lockout_s"] == 30.0 and lock["max_attempts"] == 3


def test_scenario_of_a_mechanical_lock_says_so(built):
    spec, m, model_json = built["db0166_swing_single"]
    sc = make_scenario("unlock_and_traverse", spec, derive(spec), model_json)
    assert sc["lock"]["code_kind"] == "set"
    assert sc["lock"]["code_timeout_s"] is None and sc["lock"]["lockout_s"] is None
    assert "any order" in sc["lock"]["note"]


def test_gate_catches_a_keypad_that_releases_nothing(tmp_path_factory, specs):
    """The gate has teeth: with the clutch left engaged (the old behaviour - the keypad was a decoration and the
    lever opened the door on its own) `keypad_code_works` must fail."""
    import mujoco
    from doorbench.qa import qa_push
    did = "db0526_swing_single"
    root = str(tmp_path_factory.mktemp("broken"))
    spec = specs[did]
    summary = export_door(spec, os.path.join(root, "doors"), os.path.join(root, "hardware"), formats=("mjcf", "json"))
    path = summary["files"]["mjcf"]["full"]
    model_json = json.load(open(os.path.join(root, "doors", did, "model.json")))
    row=model_json['meta']['rotary_locksets'][0]
    source=mujoco.MjSpec.from_file(path)
    pin=source.geom(row['catch_geom'])
    pin.contype=pin.conaffinity=0
    # Recompile the source-level negative; keep pin mass and every joint range.
    m=source.compile()
    model_json = json.load(open(os.path.join(root, "doors", did, "model.json")))
    meta = model_json["meta"]
    phys = derive(spec)
    pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["primary_joint"])
    oj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, meta["operator_joint"])
    push = qa_push(m, mujoco.MjData(m), pj, phys["mass"]["total_kg"], spec["leaf"]["width"])["push"]
    res = run_keypad_qa(m, spec, meta, phys, push, oj, pj)
    assert res["ok"] is False
    assert res["checks"]["wrong_holds"] is False
    assert res["metrics"]["opened_after_wrong_rad"] > math.radians(10)

"""Boundary tests use random synthetic weights, not a trained robot policy."""
import copy
from dataclasses import asdict
import inspect
import pickle

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from doorbench.dexterous.sensor_actor import ActorDimensions, SensorActor
from doorbench.dexterous.sensor_contract import SENSOR_KEYS
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController
from doorbench.dexterous.motor_contract_identity import motor_contract_fingerprint, SENSOR_ACTOR_CHECKPOINT_SCHEMA


class UnsupportedPayload:
    pass


def test_target_checkpoint_uses_encoder_feedback_and_retains_force_history(fixture):
    from doorbench.dexterous.motor_target_control import MOTOR_TARGET_SCHEMA
    f=copy.deepcopy(fixture)
    for i,a in enumerate(f['motor_contract']['actuators']):
        a.update(terms={f'joint_{i}':1.},kp=10.,bias=[0.,-10.,-2.],control_range=[-1.,1.])
    f['weights']['schema']=MOTOR_TARGET_SCHEMA
    f['weights']['motor_contract_sha256']=motor_contract_fingerprint(f['motor_contract'])
    for k,v in f['weights']['model_state'].items():
        if k.startswith('action.'):v.zero_()
    torch.save(f['weights'],f['checkpoint']);c=controller(f);c.reset_episode()
    obs=packet(f['dimensions']);obs['joint_position'][0]=.02
    force=c.force(obs,0.)
    assert force[0]==pytest.approx(-.2)
    assert c.previous_action[0]==pytest.approx(2*(-.2+1)/3-1)
    assert c.action_semantics=='original_motor_target_v1'
    obs['sensor_time_s'][:]=.002;obs['sensor_valid'][SENSOR_KEYS.index('joint_position')]=False
    with pytest.raises(ValueError,match='current valid joint encoders'):c.force(obs,.002)


@pytest.fixture
def fixture(tmp_path):
    torch.manual_seed(18)
    torch.set_num_threads(1)
    d = ActorDimensions(tactile=24, image_size=32, hidden=16)
    names = [f"joint_{i}" for i in range(69)]
    actions = [f"motor_{i}" for i in range(61)]
    motors = dict(source_xml_sha256="a" * 64, hand_mechanics_profile="shadow-loopback-v2",
                  joint_names=names, actuators=[dict(name=n, force_range=[-i - 1., i + 2.])
                                                for i, n in enumerate(actions)])
    layout = dict(interface_version="doorbench.sensors.v2", robot_xml_sha256="a" * 64,
                  joint_order=names.copy(), action_order=actions.copy(), tactile_dimension=24,
                  cameras={"left_eye_camera": {"fovy_degrees": 70}},
                  channel_order=["z", "x", "y"])
    weights = dict(schema=SENSOR_ACTOR_CHECKPOINT_SCHEMA, dimensions=asdict(d),
                   motor_contract_sha256=motor_contract_fingerprint(motors),
                   model_state=SensorActor(d).state_dict(), sensor_layout=copy.deepcopy(layout),
                   physics_dt_s=.002, training_episodes=[], validation_episodes=[])
    checkpoint = tmp_path / "untrained-synthetic-actor.pt"
    torch.save(weights, checkpoint)
    return dict(checkpoint=checkpoint, dimensions=d, weights=weights, motor_contract=motors,
                sensor_layout=layout, physics_dt_s=.002, image_shape=(32, 32, 3))


def controller(f, **overrides):
    args = {k: v for k, v in f.items() if k not in ("weights", "dimensions")}
    args.update(overrides)
    return SensorPolicyController(**args)


def packet(d, now=0.):
    p = {key: np.zeros(shape, dtype=np.uint8 if key.startswith("rgb_") else np.float32)
         for key, shape in d.shapes.items()}
    p.update(previous_action=np.zeros(d.actions, np.float32),
             sensor_time_s=np.full(len(SENSOR_KEYS), now, np.float64),
             sensor_valid=np.ones(len(SENSOR_KEYS), bool))
    return p


def test_weights_only_load_is_explicit_and_unsupported_pickle_cannot_load(fixture, monkeypatch):
    original = torch.load
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    c = controller(fixture)
    assert calls == [dict(map_location="cpu", weights_only=True)]
    assert len(c.checkpoint_sha256) == 64
    weights = copy.deepcopy(fixture["weights"])
    weights["untrusted_object"] = UnsupportedPayload()
    torch.save(weights, fixture["checkpoint"])
    with pytest.raises(pickle.UnpicklingError):
        controller(fixture)
    assert all(row["weights_only"] is True for row in calls)


@pytest.mark.parametrize("change", [
    "model_hash", "sensor_model_hash", "joint_order", "action_order", "sensor_joint_order",
    "sensor_action_order", "calibration", "layout_version", "profile", "dt", "image_shape",
    "force_caps", "force_caps_nan", "duplicate_motor", "missing_motor", "missing_joint",
])
def test_actual_runtime_contract_mismatches_are_rejected(fixture, change):
    f = copy.deepcopy(fixture)
    m, layout = f["motor_contract"], f["sensor_layout"]
    if change == "model_hash": m["source_xml_sha256"] = "b" * 64
    elif change == "sensor_model_hash": layout["robot_xml_sha256"] = "b" * 64
    elif change == "joint_order": m["joint_names"][0:2] = m["joint_names"][1::-1]
    elif change == "action_order": m["actuators"][0:2] = m["actuators"][1::-1]
    elif change == "sensor_joint_order": layout["joint_order"][0:2] = layout["joint_order"][1::-1]
    elif change == "sensor_action_order": layout["action_order"][0:2] = layout["action_order"][1::-1]
    elif change == "calibration": layout["cameras"]["left_eye_camera"]["fovy_degrees"] = 80
    elif change == "layout_version": layout["interface_version"] = "doorbench.sensors.v1"
    elif change == "profile": m["hand_mechanics_profile"] = "upstream-v1"
    elif change == "dt": f["physics_dt_s"] = .004
    elif change == "image_shape": f["image_shape"] = (128, 128, 3)
    elif change == "force_caps": m["actuators"][0]["force_range"] = [2., -1.]
    elif change == "force_caps_nan": m["actuators"][0]["force_range"] = [-1., float("nan")]
    elif change == "duplicate_motor": m["actuators"][0]["name"] = m["actuators"][1]["name"]
    elif change == "missing_motor": m["actuators"].pop()
    elif change == "missing_joint": m["joint_names"].pop()
    with pytest.raises(ValueError):
        controller(f)


@pytest.mark.parametrize("change", ["schema", "dimensions", "nan_weight", "missing_weight", "extra_weight"])
def test_checkpoint_weights_are_complete_finite_and_strict(fixture, change):
    weights = fixture["weights"]
    if change == "schema": weights["schema"] = "other"
    elif change == "dimensions": weights["dimensions"]["actions"] = 60
    elif change == "nan_weight": weights["model_state"]["action.0.bias"][0] = float("nan")
    elif change == "missing_weight": del weights["model_state"]["action.0.bias"]
    elif change == "extra_weight": weights["model_state"]["teacher.extra"] = torch.zeros(3)
    torch.save(weights, fixture["checkpoint"])
    with pytest.raises((ValueError, RuntimeError)):
        controller(fixture)


def test_reset_is_required_and_clears_hidden_state_and_clock(fixture):
    c = controller(fixture)
    p = packet(fixture["dimensions"])
    with pytest.raises(ValueError, match="reset_episode"):
        c.force(p, 0.)
    c.reset_episode()
    a = c.force(p, 0.)
    p = packet(fixture["dimensions"], .002)
    p["previous_action"] = c.previous_action
    b = c.force(p, .002)
    assert not np.allclose(a, b)
    c.reset_episode()
    assert not c.previous_action.any()
    np.testing.assert_array_equal(c.force(packet(fixture["dimensions"]), 0.), a)
    # A changed absolute time origin does not encode episode phase.
    c.reset_episode()
    np.testing.assert_array_equal(c.force(packet(fixture["dimensions"], 123.), 123.), a)


@pytest.mark.parametrize("bad_clock", [0., .001, .004, -.002, float("nan"), True, np.array([.002])])
def test_clock_requires_one_physics_tick_without_implicit_reset(fixture, bad_clock):
    c = controller(fixture); c.reset_episode()
    c.force(packet(fixture["dimensions"]), 0.)
    with pytest.raises(ValueError):
        c.force(packet(fixture["dimensions"], .002), bad_clock)


@pytest.mark.parametrize("change", ["root", "teacher", "object_array", "not_array", "future",
                                    "validity", "valid_negative_time", "negative_time", "previous_action", "rgb"])
def test_invalid_packet_cannot_change_recurrent_state(fixture, change):
    c = controller(fixture); c.reset_episode()
    baseline = controller(fixture); baseline.reset_episode()
    good = packet(fixture["dimensions"])
    p = copy.deepcopy(good)
    if change == "root": p["root_pose"] = np.zeros(7)
    elif change == "teacher": p["teacher_phase"] = np.zeros(1)
    elif change == "object_array": p["tactile"] = p["tactile"].astype(object)
    elif change == "not_array": p["tactile"] = p["tactile"].tolist()
    elif change == "future": p["sensor_time_s"][0] = .001
    elif change == "validity": p["sensor_valid"] = np.ones(len(SENSOR_KEYS), np.float32)
    elif change == "valid_negative_time": p["sensor_time_s"][0] = -1
    elif change == "negative_time": p["sensor_time_s"][0] = -2; p["sensor_valid"][0] = False
    elif change == "previous_action": p["previous_action"][0] = 1.1
    elif change == "rgb": p["rgb_left"] = p["rgb_left"].astype(np.float32)
    with pytest.raises(ValueError):
        c.force(p, 0.)
    np.testing.assert_array_equal(c.force(good, 0.), baseline.force(good, 0.))


def test_only_numeric_packet_and_clock_runtime_api_and_original_caps(fixture):
    assert tuple(inspect.signature(SensorPolicyController.force).parameters) == ("self", "packet", "now_s")
    # Force saturation proves conversion uses each original asymmetric range.
    state = fixture["weights"]["model_state"]
    state["action.0.weight"].zero_()
    state["action.0.bias"][:] = torch.tensor([100. if i % 2 else -100. for i in range(61)])
    torch.save(fixture["weights"], fixture["checkpoint"])
    c = controller(fixture); c.reset_episode()
    ranges = c.force_ranges
    expected = np.array([ranges[i, i % 2] for i in range(61)])
    ranges[:] = 1e9
    fixture["motor_contract"]["actuators"][0]["force_range"][:] = [-1e6, 1e6]
    forces = c.force(packet(fixture["dimensions"]), 0.)
    np.testing.assert_array_equal(forces, expected)
    assert forces.shape == (61,)
    copied = c.previous_action
    copied[:] = 0
    assert np.all(np.abs(c.previous_action) == 1.)


def test_missing_sensor_flags_zero_payload_without_teacher_fallback(fixture):
    c = controller(fixture); c.reset_episode()
    p = packet(fixture["dimensions"])
    p["sensor_valid"][:] = False
    p["sensor_time_s"][:] = -1.
    a = c.force(p, 0.)
    c.reset_episode()
    for key in SENSOR_KEYS:
        p[key][:] = 100
    np.testing.assert_array_equal(c.force(p, 0.), a)


@pytest.mark.parametrize('change',['valid_force_caps','transmission','passive_damping','passive_tendon_limit'])
def test_valid_but_changed_motor_mechanics_cannot_reuse_xml_identity(fixture,change):
    f=copy.deepcopy(fixture);m=f['motor_contract']
    if change=='valid_force_caps':m['actuators'][0]['force_range']=[-2.,3.]
    elif change=='transmission':m['actuators'][0]['terms']={'joint_0':2.}
    elif change=='passive_damping':m['passive']={'joint_0':{'damping':.1}}
    elif change=='passive_tendon_limit':m['passive_tendons']=[{'name':'rh_FF_loopback','range_rad':[-2.5708,.1]}]
    assert m['source_xml_sha256']==fixture['motor_contract']['source_xml_sha256']
    with pytest.raises(ValueError,match='motor mechanics'):
        controller(f)


def test_canonical_motor_identity_ignores_dict_order_and_missing_identity_fails(fixture):
    motors=fixture['motor_contract']
    reordered=dict(reversed(list(motors.items())))
    assert motor_contract_fingerprint(reordered)==motor_contract_fingerprint(motors)
    controller(fixture,motor_contract=reordered)
    del fixture['weights']['motor_contract_sha256']
    torch.save(fixture['weights'],fixture['checkpoint'])
    with pytest.raises(ValueError,match='motor mechanics'):
        controller(fixture)

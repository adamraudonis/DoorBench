import builtins
import copy
import json
from pathlib import Path

import mujoco
import pytest

from doorbench.dexterous import sensor_reset_preflight as preflight


@pytest.fixture
def fixture(tmp_path):
    # Small collision plant with the real 69/61/eight-loopback contract shape.
    # It tests rejection logic; it is not evidence that an H1 reset is feasible.
    bodies, tendons = [], []
    for i, (side, digit) in enumerate((s, d) for s in ('lh', 'rh') for d in ('FF', 'MF', 'RF', 'LF')):
        bodies.append(f'''<body name="{side}_{digit.lower()}middle" pos="{2+i} 0 0">
          <joint name="{side}_{digit}J2" range="0 1.5"/><geom type="sphere" size=".01"/>
          <body name="{side}_{digit.lower()}distal" pos="0 0 .1">
            <joint name="{side}_{digit}J1" range="0 1.5"/><geom type="sphere" size=".01"/>
          </body></body>''')
        tendons.append(f'''<fixed name="{side}_{digit}_loopback" limited="true" range="-3 0">
           <joint joint="{side}_{digit}J1" coef="1"/><joint joint="{side}_{digit}J2" coef="-1"/>
           </fixed>''')
    for i in range(53):
        bodies.append(f'<body name="link{i}" pos="{20+i} 0 0"><joint name="joint{i}" range="-1 1"/><geom type="sphere" size=".01"/></body>')
    names = [f'{s}_{d}J{k}' for s in ('lh', 'rh') for d in ('FF', 'MF', 'RF', 'LF') for k in (2, 1)] + [f'joint{i}' for i in range(53)]
    robot = tmp_path/'robot.xml'
    robot.write_text('''<mujoco><compiler angle="radian"/><default><joint damping=".1"/>
      <geom friction="1 .005 .0001"/></default><worldbody><body name="pelvis">
      <freejoint name="free_base"/><inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>'''+
      ''.join(bodies)+'</body></worldbody><tendon>'+''.join(tendons)+'</tendon><actuator>'+''.join(
      f'<position name="motor{i}" joint="{name}" kp="1" forcerange="-10 10" ctrlrange="-1 1"/>'
      for i, name in enumerate(names[:61]))+'</actuator></mujoco>')
    door = tmp_path/'door.xml'
    door.write_text('''<mujoco><compiler angle="radian"/><worldbody><geom name="floor" type="plane" size="100 100 .1"/>
      <body name="leaf"><joint name="leaf_hinge" axis="0 0 1"/><geom name="slab" type="box" size=".1 .1 .2"/>
        <body name="leaf_handle" pos=".2 .1 .2"><joint name="leaf_handle_hinge"/><geom type="sphere" size=".01"/></body>
        <body name="bolt" pos=".2 0 0"><joint name="leaf_latch_bolt_slide" type="slide"/><geom type="sphere" size=".01"/></body>
      </body></worldbody></mujoco>''')
    model = mujoco.MjModel.from_xml_path(str(robot))
    motors = tmp_path/'motors.json'
    motors.write_text(json.dumps(preflight._native_contract(model, robot)))
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps(dict(initial_root=[0, 0, 2, 1, 0, 0, 0], initial_joints={'ignored':99},
        acquisition=dict(joint_names=names, path_qpos=[[0.]*69, [.1]*69]), controls='not consumed')))
    robot_usd, door_usd = tmp_path/'robot.usda', tmp_path/'door.usda'
    robot_usd.write_text('#usda 1.0\n'); door_usd.write_text('#usda 1.0\n')
    return dict(reference=reference, motors=motors, native_robot=robot, native_door=door,
                robot_usd=robot_usd, door_usd=door_usd)


def test_receipt_proves_clear_static_reset_without_runtime_simulator_import(fixture, monkeypatch):
    receipt = preflight.build_sensor_reset_preflight(**fixture)
    assert receipt['passed'] and receipt['native_simulation_steps'] == 0
    assert receipt['right_hand_contacts'] == []
    original_import = builtins.__import__
    def no_physics(name, *args, **kwargs):
        if name.startswith(('mujoco', 'numpy', 'isaac', 'pxr')):
            raise AssertionError('Runtime reset validator imported physics: '+name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_physics)
    reset = preflight.validate_sensor_reset_preflight(receipt, **fixture)
    assert reset['joint_position'] == [0.]*69
    assert reset['source'] == 'acquisition.path_qpos[0]'


@pytest.mark.parametrize('changed', preflight.INPUTS)
def test_each_actual_input_is_bound_even_when_it_still_parses(fixture, changed):
    receipt = preflight.build_sensor_reset_preflight(**fixture)
    with fixture[changed].open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='input differs: '+changed):
        preflight.validate_sensor_reset_preflight(receipt, **fixture)


@pytest.mark.parametrize('change', ['passed', 'source', 'threshold', 'missing_check', 'reset', 'motor_identity', 'contacts'])
def test_receipt_cannot_change_protocol_or_reset_after_screen(fixture, change):
    receipt = preflight.build_sensor_reset_preflight(**fixture)
    if change == 'passed': receipt['passed'] = False
    elif change == 'source': receipt['sources']['doorbench/dexterous/sensor_reset_preflight.py'] = '0'*64
    elif change == 'threshold': receipt['thresholds'] = dict(receipt['thresholds'], self_penetration_m=1.)
    elif change == 'missing_check': del receipt['checks']['zero_right_hand_contacts']
    elif change == 'reset': receipt['reset']['root_xyz_wxyz'][0] += .1
    elif change == 'motor_identity': receipt['motor_contract_sha256'] = '0'*64
    elif change == 'contacts': receipt['right_hand_contacts'] = [dict(separation_m=-.001)]
    with pytest.raises(ValueError):
        preflight.validate_sensor_reset_preflight(receipt, **fixture)


@pytest.mark.parametrize('changed', ['motor_cap', 'damping', 'loopback', 'model_hash', 'legacy_profile'])
def test_forged_motor_contract_cannot_reuse_native_xml(fixture, changed):
    motors = json.loads(fixture['motors'].read_text())
    if changed == 'motor_cap': motors['actuators'][0]['force_range'][1] += 1
    elif changed == 'damping': motors['passive'][motors['joint_names'][0]]['damping'] += .1
    elif changed == 'loopback': motors['passive_tendons'][0]['range_rad'][1] = .1
    elif changed == 'model_hash': motors['source_xml_sha256'] = '0'*64
    elif changed == 'legacy_profile': motors['hand_mechanics_profile'] = 'upstream-v1'
    fixture['motors'].write_text(json.dumps(motors))
    with pytest.raises(ValueError):
        preflight.build_sensor_reset_preflight(**fixture)


def test_real_collision_and_limit_failures_are_retained_in_failed_receipt(fixture):
    reference = json.loads(fixture['reference'].read_text())
    # Put the rh_FF middle at the origin, inside the actual slab.
    reference['initial_root'][:3] = [-6., 0., 0.]
    fixture['reference'].write_text(json.dumps(reference))
    collided = preflight.build_sensor_reset_preflight(**fixture)
    assert not collided['passed'] and not collided['checks']['zero_right_hand_contacts']
    assert not collided['checks']['nonfoot_environment_collision']
    assert collided['right_hand_contacts']
    reference['initial_root'] = [0, 0, 2, 1, 0, 0, 0]
    reference['acquisition']['path_qpos'][0][1] = 2.
    fixture['reference'].write_text(json.dumps(reference))
    illegal = preflight.build_sensor_reset_preflight(**fixture)
    assert not illegal['checks']['joint_reset_limits']
    assert not illegal['checks']['passive_loopback_limits']
    with pytest.raises(ValueError):
        preflight.validate_sensor_reset_preflight(illegal, **fixture)


@pytest.mark.parametrize('change', ['nonfinite', 'quaternion', 'duplicate', 'incomplete'])
def test_reset_numbers_are_complete_finite_and_canonical(fixture, change):
    reference = json.loads(fixture['reference'].read_text())
    if change == 'nonfinite': reference['acquisition']['path_qpos'][0][0] = float('nan')
    elif change == 'quaternion': reference['initial_root'][3] = 2.
    elif change == 'duplicate': reference['acquisition']['joint_names'][0] = reference['acquisition']['joint_names'][1]
    elif change == 'incomplete': reference['acquisition']['path_qpos'][0].pop()
    with pytest.raises(ValueError):
        preflight.frozen_acquisition_reset(reference, fixture['motors'])


def test_changed_included_xml_cannot_hide_behind_same_main_xml(fixture):
    included = fixture['native_door'].parent/'extra.xml'
    included.write_text('<mujocoinclude><worldbody><geom name="remote" type="sphere" pos="100 100 100" size=".1"/></worldbody></mujocoinclude>')
    door = fixture['native_door']
    door.write_text(door.read_text().replace('</mujoco>', '<include file="extra.xml"/></mujoco>'))
    receipt = preflight.build_sensor_reset_preflight(**fixture)
    assert receipt['passed']
    included.write_text(included.read_text().replace('100 100 100', '0 0 2'))
    with pytest.raises(ValueError, match='Referenced native asset changed'):
        preflight.validate_sensor_reset_preflight(receipt, **fixture)

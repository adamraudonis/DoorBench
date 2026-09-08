"""Frozen contact-free actor reset proof, separate from actor observations.

The builder compiles an unstepped native scene on CPU. The runtime validator
only reads JSON/bytes; it imports no simulator, kinematics, or teacher module.
This proves native reset geometry, not USD import parity or policy success.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'doorbench.sensor-reset-preflight.v1'
INPUTS = ('reference', 'motors', 'native_robot', 'native_door', 'robot_usd', 'door_usd')
CHECKS = ('native_mechanics_match', 'complete_robot_reset', 'unit_root_quaternion',
          'closed_door_reset', 'zero_right_hand_contacts', 'joint_reset_limits',
          'passive_loopback_limits', 'self_collision', 'nonfoot_environment_collision',
          'foot_floor_penetration', 'finite', 'unstepped')
THRESHOLDS = dict(joint_reset_tolerance_rad=1e-5, loopback_tolerance_rad=1e-5,
                  self_penetration_m=.003, nonfoot_penetration_m=0., foot_penetration_m=.003)
SOURCES = ('doorbench/dexterous/sensor_reset_preflight.py',
           'scripts/dexterous/preflight_sensor_reset.py',
           'doorbench/dexterous/motor_contract_identity.py',
           'doorbench/dexterous/robot_design_identity.py',
           'doorbench/dexterous/isaac_tendons.py',
           'doorbench/dexterous/isaac_materials.py')


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _load(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: _hash(root/name) for name in SOURCES}


def _native_dependencies(path):
    """Freeze includes and every resolved native file asset, not only XML text."""
    import mujoco
    from .robot_design_identity import _expanded_xml, _parse, _FILE_ATTRIBUTES, robot_design_identity
    path = Path(path).resolve(strict=True)
    # The design contract rejects unknown file providers and opaque plugins.
    design = robot_design_identity(path)
    root = _expanded_xml(path)
    spec = mujoco.MjSpec.from_file(str(path))
    directories = dict(mesh=spec.meshdir, texture=spec.texturedir, hfield=spec.meshdir, skin=spec.meshdir)
    compiler = {}
    for node in root.findall('compiler'):
        compiler.update(node.attrib)
    dependencies = {path}
    def includes(current):
        for node in _parse(current).iter('include'):
            included = (path.parent/node.attrib['file']).resolve(strict=True)
            if included not in dependencies:
                dependencies.add(included)
                includes(included)
    includes(path)
    for node in root.iter():
        for attribute in set(node.attrib) & _FILE_ATTRIBUTES:
            if not node.attrib[attribute]:
                continue
            asset = Path(node.attrib[attribute].replace('\\', '/'))
            if compiler.get('strippath', 'false') == 'true':
                asset = Path(asset.name)
            if not asset.is_absolute():
                asset = path.parent/directories[node.tag]/asset
            dependencies.add(asset.resolve(strict=True))
    return dict(design_sha256=design['sha256'], files={str(p): _hash(p) for p in sorted(dependencies)})


def frozen_acquisition_reset(reference, motors):
    """Get exactly the first path joint positions and root; never saved controls."""
    reference, motors = _load(reference), _load(motors)
    if motors.get('hand_mechanics_profile') != 'shadow-loopback-v2':
        raise ValueError('Actor reset requires explicit shadow-loopback-v2 mechanics')
    names = motors.get('joint_names', [])
    if len(names) != 69 or len(set(names)) != 69 or any(type(n) is not str for n in names):
        raise ValueError('Expected all 69 uniquely named robot joints')
    path = reference.get('acquisition', {})
    order, rows = path.get('joint_names', []), path.get('path_qpos', [])
    if len(order) != 69 or set(order) != set(names) or len(rows) < 2 or len(rows[0]) != 69:
        raise ValueError('Frozen acquisition first path configuration must name all 69 joints')
    root = reference.get('initial_root', [])
    values = [*root, *rows[0]]
    if len(root) != 7 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Reset must contain finite numeric root and joint positions')
    if abs(math.sqrt(sum(v*v for v in root[3:]))-1.) > 1e-6:
        raise ValueError('Reset root quaternion must be unit length')
    joints = dict(zip(order, rows[0]))
    return dict(source='acquisition.path_qpos[0]', root_xyz_wxyz=root,
                joint_order=names, joint_position=[joints[n] for n in names],
                root_velocity_world=[0.]*6, joint_velocity=[0.]*69)


def _native_contract(model, robot_path):
    """Independently realize the importer motor contract from compiled mechanics."""
    import mujoco
    from .isaac_tendons import native_passive_tendons
    from .isaac_materials import native_contact_contract
    scalar = [j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1 or model.joint(free[0]).name != 'free_base' or model.nu != 61 or len(scalar) != 69:
        raise ValueError('Expected the unchanged free-base 69-joint, 61-motor H1/Shadow plant')
    result = dict(source_xml_sha256=_hash(robot_path), hand_mechanics_profile='shadow-loopback-v2',
                  mass_kg=float(model.body_mass.sum()), free_root=True,
                  joint_names=[model.joint(j).name for j in scalar], actuators=[], passive={},
                  passive_tendons=native_passive_tendons(model), contact_material=native_contact_contract(model))
    if len(result['passive_tendons']) != 8:
        raise ValueError('All eight documented passive loopbacks must be present')
    for i in range(model.nu):
        tid = int(model.actuator_trnid[i, 0])
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            terms = {model.joint(tid).name: float(model.actuator_gear[i, 0])}
        elif model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
            terms = {}
            for k in range(model.tendon_adr[tid], model.tendon_adr[tid]+model.tendon_num[tid]):
                if model.wrap_type[k] != mujoco.mjtWrap.mjWRAP_JOINT:
                    raise ValueError('Unsupported non-fixed actuator tendon')
                name = model.joint(int(model.wrap_objid[k])).name
                terms[name] = terms.get(name, 0.)+float(model.actuator_gear[i, 0]*model.wrap_prm[k])
        else:
            raise ValueError('Unsupported motor transmission')
        result['actuators'].append(dict(name=model.actuator(i).name, terms=terms,
            kp=float(model.actuator_gainprm[i, 0]), bias=model.actuator_biasprm[i, :3].tolist(),
            force_range=model.actuator_forcerange[i].tolist(), control_range=model.actuator_ctrlrange[i].tolist()))
    for j in scalar:
        v = model.jnt_dofadr[j]
        result['passive'][model.joint(j).name] = dict(damping=float(model.dof_damping[v]),
            armature=float(model.dof_armature[v]), friction=float(model.dof_frictionloss[v]),
            stiffness=float(model.jnt_stiffness[j]), springref=float(model.qpos_spring[model.jnt_qposadr[j]]))
    return result


def _matching(expected, actual, name='mechanics'):
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or set(expected) != set(actual):
            raise ValueError('Native motor contract structure differs at '+name)
        for key in actual:
            _matching(expected[key], actual[key], name+'.'+key)
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(expected) != len(actual):
            raise ValueError('Native motor contract array differs at '+name)
        for i, value in enumerate(actual):
            _matching(expected[i], value, name+f'[{i}]')
    elif type(actual) is float:
        if type(expected) not in (float, int) or not math.isclose(expected, actual, rel_tol=0., abs_tol=1e-12):
            raise ValueError('Native motor mechanics differ at '+name)
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError('Native motor mechanics differ at '+name)


def _screen(scene, reset):
    import mujoco
    import numpy as np
    m, d = scene, mujoco.MjData(scene)
    root = m.jnt_qposadr[m.joint('robot/free_base').id]
    d.qpos[root:root+7] = reset['root_xyz_wxyz']
    ids = np.array([m.joint('robot/'+n).id for n in reset['joint_order']])
    d.qpos[m.jnt_qposadr[ids]] = reset['joint_position']
    door_reset = {}
    for j in range(m.njnt):
        if m.joint(j).name.startswith('robot/'):
            continue
        if int(m.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError('Door reset needs an explicit scalar-joint contract')
        door_reset[m.joint(j).name] = 0.
        d.qpos[m.jnt_qposadr[j]] = 0.
    if not {'leaf_hinge', 'leaf_handle_hinge', 'leaf_latch_bolt_slide'} <= set(door_reset):
        raise ValueError('This acquisition curriculum requires the declared Door55 mechanism')
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    limited = ids[m.jnt_limited[ids].astype(bool)]
    violation = max(0., float(np.maximum(m.jnt_range[limited, 0]-d.qpos[m.jnt_qposadr[limited]],
        d.qpos[m.jnt_qposadr[limited]]-m.jnt_range[limited, 1]).max()))
    loopback = 0.
    for side in ('lh', 'rh'):
        for digit in ('FF', 'MF', 'RF', 'LF'):
            q1 = d.qpos[m.jnt_qposadr[m.joint(f'robot/{side}_{digit}J1').id]]
            q2 = d.qpos[m.jnt_qposadr[m.joint(f'robot/{side}_{digit}J2').id]]
            loopback = max(loopback, float(q1-q2))
    contacts, right_hand = [], []
    self_depth = nonfoot_depth = foot_depth = 0.
    for c in d.contact[:d.ncon]:
        bodies = [m.body(m.geom_bodyid[g]).name for g in c.geom]
        geoms = [m.geom(g).name for g in c.geom]
        robot = [name.startswith('robot/') for name in bodies]
        if not any(robot):
            continue
        row = dict(bodies=bodies, geoms=geoms, separation_m=float(c.dist))
        contacts.append(row)
        if any(name.startswith('robot/rh_') for name in bodies):
            right_hand.append(row)  # Includes self and positive-gap native contact entries.
        depth = max(0., -float(c.dist))
        if all(robot):
            self_depth = max(self_depth, depth)
        else:
            rb = bodies[robot.index(True)]
            floor_pair = rb in ('robot/left_ankle_link', 'robot/right_ankle_link') and 'floor' in geoms
            if floor_pair:
                foot_depth = max(foot_depth, depth)
            else:
                nonfoot_depth = max(nonfoot_depth, depth)
    checks = dict(native_mechanics_match=True, complete_robot_reset=True, unit_root_quaternion=True,
        closed_door_reset=True, zero_right_hand_contacts=not right_hand,
        joint_reset_limits=violation <= THRESHOLDS['joint_reset_tolerance_rad'],
        passive_loopback_limits=loopback <= THRESHOLDS['loopback_tolerance_rad'],
        self_collision=self_depth < THRESHOLDS['self_penetration_m'],
        nonfoot_environment_collision=nonfoot_depth <= THRESHOLDS['nonfoot_penetration_m'],
        foot_floor_penetration=foot_depth < THRESHOLDS['foot_penetration_m'],
        finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.xpos).all()), unstepped=float(d.time)==0.)
    return dict(checks=checks, passed=all(checks.values()), native_simulation_steps=0,
        door_reset=door_reset, right_hand_contacts=right_hand, all_robot_contacts=contacts,
        maximum_joint_violation_rad=violation, maximum_loopback_violation_rad=loopback,
        maximum_self_penetration_m=self_depth, maximum_nonfoot_penetration_m=nonfoot_depth,
        maximum_foot_penetration_m=foot_depth)


def build_sensor_reset_preflight(*, reference, motors, native_robot, native_door, robot_usd, door_usd):
    """CPU-only static check. Failed geometry produces a retained failed receipt."""
    import mujoco
    from .motor_contract_identity import motor_contract_fingerprint
    paths = {k: Path(v).resolve(strict=True) for k, v in locals().copy().items() if k in INPUTS}
    files = {name: dict(path=str(path), sha256=_hash(path)) for name, path in paths.items()}
    dependencies = {name: _native_dependencies(paths[name]) for name in ('native_robot', 'native_door')}
    contract = _load(paths['motors'])
    reset = frozen_acquisition_reset(paths['reference'], contract)
    native = mujoco.MjSpec.from_file(str(paths['native_robot'])).compile()
    actual = _native_contract(native, paths['native_robot'])
    _matching({key: contract.get(key) for key in actual}, actual)
    spec = mujoco.MjSpec.from_file(str(paths['native_door']))
    spec.memory = 128*1024*1024
    frame = spec.worldbody.add_frame(pos=[0., -1.5, 0.])
    spec.attach(mujoco.MjSpec.from_file(str(paths['native_robot'])), prefix='robot/', frame=frame)
    evidence = _screen(spec.compile(), reset)
    receipt = dict(schema=SCHEMA, scope='Native static contact-free acquisition reset; not USD parity, dynamics, or policy success',
        files=files, native_dependencies=dependencies, sources=_source_hashes(), motor_contract_sha256=motor_contract_fingerprint(contract),
        reset=reset, thresholds=THRESHOLDS, native_mujoco_version=mujoco.__version__, **evidence)
    # Reject files changed while native compilation was taking place.
    if any(_hash(paths[name]) != row['sha256'] for name, row in files.items()):
        raise ValueError('Preflight input changed during geometry audit')
    if any(_hash(path) != sha for row in dependencies.values() for path, sha in row['files'].items()):
        raise ValueError('Referenced native asset changed during geometry audit')
    return receipt


def validate_sensor_reset_preflight(receipt, *, reference, motors, robot_usd, door_usd,
                                    native_robot=None, native_door=None):
    """Pure byte/JSON validation before reset; never imports a simulator or FK."""
    from .motor_contract_identity import motor_contract_fingerprint
    receipt = _load(receipt)
    if not isinstance(receipt, dict) or receipt.get('schema') != SCHEMA or receipt.get('passed') is not True:
        raise ValueError('A passed explicit sensor reset preflight is required')
    checks = receipt.get('checks', {})
    if set(checks) != set(CHECKS) or any(value is not True for value in checks.values()):
        raise ValueError('Reset preflight is missing required geometry/mechanics checks')
    if receipt.get('native_simulation_steps') != 0 or receipt.get('thresholds') != THRESHOLDS:
        raise ValueError('Reset preflight changed its static protocol')
    if receipt.get('sources') != _source_hashes():
        raise ValueError('Reset preflight source changed; regenerate its geometry evidence')
    files = receipt.get('files', {})
    if set(files) != set(INPUTS):
        raise ValueError('Reset preflight must bind all six runtime inputs')
    actual = dict(reference=reference, motors=motors, robot_usd=robot_usd, door_usd=door_usd,
                  native_robot=native_robot or files['native_robot']['path'],
                  native_door=native_door or files['native_door']['path'])
    for name, path in actual.items():
        if _hash(path) != files[name]['sha256']:
            raise ValueError('Reset preflight input differs: '+name)
    dependencies = receipt.get('native_dependencies', {})
    if set(dependencies) != {'native_robot', 'native_door'}:
        raise ValueError('Reset proof must bind both native asset dependency sets')
    for name, row in dependencies.items():
        if not row.get('files') or row['files'].get(files[name]['path']) != files[name]['sha256']:
            raise ValueError('Native dependency proof is incomplete')
        for path, sha in row['files'].items():
            if _hash(path) != sha:
                raise ValueError('Referenced native asset changed: '+path)
    contract = _load(motors)
    if contract.get('source_xml_sha256') != files['native_robot']['sha256']:
        raise ValueError('Actual motor contract names a different native robot')
    if motor_contract_fingerprint(contract) != receipt.get('motor_contract_sha256'):
        raise ValueError('Reset preflight motor mechanics differ')
    reset = frozen_acquisition_reset(reference, contract)
    if _json(reset) != _json(receipt.get('reset')):
        raise ValueError('Runtime reset numbers differ from the screened configuration')
    if receipt.get('right_hand_contacts') != []:
        raise ValueError('Preflight contains right-hand contacts')
    return reset

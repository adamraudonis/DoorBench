"""Complete attained-state identity for new destination-engine plans.

This numeric boundary neither reads nor writes a simulator. Passing it proves
that a plan starts at its bound state, not that imported geometry, forces or the
future trajectory are valid. Keep those independent gates and legacy guards.
"""
import hashlib
import json
import re
import numpy as np
from scipy.spatial.transform import Rotation
from .motor_contract_identity import motor_contract_fingerprint

SCHEMA='doorbench.destination-state.v1'
ROOT_CONVENTION='actor-origin-world-v1'
STATE_TOLERANCE=1e-5
EPOCH_TOLERANCE=1e-8


def _number(value):
    if isinstance(value,(bool,np.bool_)) or not np.isscalar(value):
        raise ValueError('Finite scalar measurement required')
    result=float(value)
    if not np.isfinite(result):raise ValueError('Finite scalar measurement required')
    return result


def _hash(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('Original source SHA-256 required')
    return value


def _mapping(values,names):
    if type(values) is not dict or set(values)!=set(names):
        raise ValueError('Complete named measured coordinates required')
    return {name:_number(values[name]) for name in names}


def _digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def freeze_destination_state(*,motor_contract,door_source_sha256,time_s,measured_time_s,
                             root_state_world,joint_position,joint_velocity,
                             door_joint_order,door_position,door_velocity,root_state_convention):
    """Copy all 69 positions/velocities, root13 and door state at one epoch.

    The caller must explicitly convert native root angular rates to world axes,
    or use Isaac's root_link_state_w. Its legacy root_state_w mixes actor pose
    with COM velocity and cannot be declared as this root convention.
    """
    now=_number(time_s);measured=_number(measured_time_s)
    if now<0 or abs(now-measured)>EPOCH_TOLERANCE:
        raise ValueError('Require the coherent current attained-state epoch')
    if root_state_convention!=ROOT_CONVENTION:
        raise ValueError('Require actor-origin pose and world actor-origin velocities')
    if type(motor_contract) is not dict:raise ValueError('Original complete motor contract required')
    names=motor_contract.get('joint_names')
    if (type(names) is not list or len(names)!=69 or any(type(n) is not str or not n for n in names)
            or len(set(names))!=69):
        raise ValueError('Require all 69 unique original robot joint names')
    root=np.array(root_state_world,dtype=float,copy=True)
    if root.shape!=(13,) or not np.isfinite(root).all() or abs(np.linalg.norm(root[3:7])-1)>1e-6:
        raise ValueError('Require finite complete root13 and unit WXYZ orientation')
    if (type(door_joint_order) is not list or not door_joint_order
            or any(type(n) is not str or not n for n in door_joint_order)
            or len(set(door_joint_order))!=len(door_joint_order)):
        raise ValueError('Require the complete declared door coordinate inventory')
    door_names=door_joint_order.copy()
    payload=dict(schema=SCHEMA,scope='Attained-state identity only; imported FK, collision, contact and physical gates remain separate',
        robot_source_sha256=_hash(motor_contract.get('source_xml_sha256')),
        motor_contract_sha256=motor_contract_fingerprint(motor_contract),
        door_source_sha256=_hash(door_source_sha256),time_s=now,measured_time_s=measured,
        root_state_convention=ROOT_CONVENTION,root_state_world=root.tolist(),joint_order=names.copy(),door_joint_order=door_names,
        joint_position=_mapping(joint_position,names),joint_velocity=_mapping(joint_velocity,names),
        door_position=_mapping(door_position,door_names),door_velocity=_mapping(door_velocity,door_names))
    return dict(payload,sha256=_digest(payload))


def admit_destination_state(binding,**current):
    """Reject stale, partial, changed-identity or changed-state plan admission."""
    if type(binding) is not dict:raise ValueError('Require an immutable numeric state binding')
    payload={k:v for k,v in binding.items() if k!='sha256'}
    if binding.get('schema')!=SCHEMA or _digest(payload)!=binding.get('sha256'):
        raise ValueError('Attained-state binding checksum or schema differs')
    actual=freeze_destination_state(**current)
    for field in ('schema','scope','robot_source_sha256','motor_contract_sha256','door_source_sha256',
                  'root_state_convention','joint_order','door_joint_order'):
        if binding[field]!=actual[field]:raise ValueError('Destination identity differs: '+field)
    if abs(binding['time_s']-actual['time_s'])>EPOCH_TOLERANCE:
        raise ValueError('Destination plan has a stale attained-state epoch')
    if set(binding['door_position'])!=set(actual['door_position']):
        raise ValueError('Destination door coordinate inventory differs')
    before=np.asarray(binding['root_state_world']);after=np.asarray(actual['root_state_world'])
    qa=before[[4,5,6,3]];qb=after[[4,5,6,3]]
    errors=dict(root_position_m=float(np.max(abs(after[:3]-before[:3]))),
        root_orientation_rad=float((Rotation.from_quat(qb)*Rotation.from_quat(qa).inv()).magnitude()),
        root_linear_velocity_mps=float(np.max(abs(after[7:10]-before[7:10]))),
        root_angular_velocity_radps=float(np.max(abs(after[10:13]-before[10:13]))))
    for field in ('joint_position','joint_velocity','door_position','door_velocity'):
        errors[field]=max(abs(actual[field][name]-binding[field][name]) for name in binding[field])
    if any(value>STATE_TOLERANCE for value in errors.values()):
        raise ValueError('Destination plan requires its exact attained state: '+json.dumps(errors,sort_keys=True))
    return dict(passed=True,state_sha256=binding['sha256'],actual_state_sha256=actual['sha256'],
        measured_time_s=actual['measured_time_s'],complete_robot_joint_count=69,
        tolerance=STATE_TOLERANCE,maximum_errors=errors,
        scope='State admission only; no geometry, contact, imported-physics or task qualification')

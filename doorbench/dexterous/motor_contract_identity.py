"""Portable static identity for the motor contract used to normalize actions."""
import hashlib
import json

SENSOR_ACTOR_CHECKPOINT_SCHEMA = 'doorbench.sensor-actor.v2'


def motor_contract_fingerprint(contract):
    """Hash finite canonical JSON, independently of file location/formatting.

    The full supplied static contract is included: valid but changed motor caps,
    tendon coefficients, damping or passive limits cannot reuse a checkpoint by
    retaining its source XML label. Caller must separately validate the actual
    imported physical plant against this contract.
    """
    if type(contract) is not dict:
        raise ValueError('Motor contract must be a plain dictionary')
    try:
        encoded=json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError) as exc:
        raise ValueError('Motor contract must contain finite JSON metadata') from exc
    return hashlib.sha256(encoded.encode()).hexdigest()

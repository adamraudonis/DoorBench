The sensor actor must start from the frozen first acquisition configuration,
with a clear right hand. An empty PhysX contact buffer before its first step is
insufficient evidence: an already grasping reset can also have an empty buffer.

Run the native CPU preflight before launching the actor in Isaac:

```bash
PYTHONPATH=. /workspace/asset-venv/bin/python scripts/dexterous/preflight_sensor_reset.py \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --motors /path/from/ready/motor-contract.json \
  --native-robot /path/from/ready/h1-shadow-loopback-v2.xml \
  --native-door /path/from/ready/door.xml \
  --robot-usd /path/from/ready/robot.usd \
  --door-usd /path/from/ready/door.usda \
  --output /fresh/run/reset-preflight.json
```

The tool combines the unchanged native door and H1/Shadow models, writes the
frozen root and all 69 joint positions into planning data, and computes geometry
without stepping physics. It checks the original 61 motor transmissions, force
and control caps, passive mechanics, mass and all eight hand loopbacks against
the actual import contract. Every right-hand contact is rejected, including
self contacts and native positive-gap entries. Joint and loopback reset errors
must stay within 10 microradians. Non-foot penetration is rejected; the existing
3 mm self/sole penetration limits remain explicit.

The receipt binds all six input files, exact reset numbers, code hashes and
native referenced mesh/texture/include bytes. A changed reference, contract,
asset, limit, source file or failed check requires a new preflight. A failed
geometry screen is saved with its contacts and metrics; the CLI returns failure
and will not overwrite previous evidence.

The runner can validate it before writing the reset:

```python
from doorbench.dexterous.sensor_reset_preflight import validate_sensor_reset_preflight

reset = validate_sensor_reset_preflight(
    receipt_path, reference=reference_path, motors=motor_contract_path,
    robot_usd=actual_robot_usd, door_usd=actual_door_usd,
)
```

This validator reads only bytes and JSON. It imports no simulator or kinematics
and passes no scene, reset receipt or geometry to the actor. The actor still
requires `--reset-from-acquisition-path`; root and velocities are written only
once at reset. Native paths default to those frozen in the receipt and remain
required on the host for rehashing. Relocating assets requires regenerating the
receipt on the destination host.

This is a native static reset proof. It does not certify USD import parity,
external USD dependencies, physical grasp acquisition, balance or policy
success. Use the ready receipt and live every-step PhysX checks for those
separate obligations. The small synthetic unit model tests rejection behavior;
the actual H1/Door55 geometry check is recorded separately under `out/`.

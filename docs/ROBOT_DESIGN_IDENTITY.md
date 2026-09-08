# Moving a robot model between machines

Use `robot_design_identity.py` to identify the authored robot and its referenced
asset bytes. Keep `robot_identity.py` for the separate strict compiled-model
receipt. A matching design does **not** certify matching compiled collision
geometry or a qualified imported simulator.

```python
from doorbench.dexterous.robot_design_identity import (
    robot_design_identity, verify_robot_design_identity,
)

expected = robot_design_identity(source_robot_xml)
# Save this full receipt with the experiment configuration.
actual = verify_robot_design_identity(destination_robot_xml, expected)
```

Both functions read files and parse MJCF; they do not compile or step a model.
The identity includes the pinned MuJoCo parser version, an ordered canonical XML
tree digest, and byte hashes of every resolved referenced mesh, texture, hfield
and skin file. Texture cube faces are each included. File formats and implicit
asset names remain part of the binding.

Recursive includes are expanded relative to the main model. Default classes,
inheritance references, element order and exact numeric strings remain in the
tree. Attribute order, indentation, XML comments and resolved file locations
are excluded. This follows MuJoCo's documented include and asset path rules.
[MuJoCo XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html)

The implementation hashes the original attributes rather than serialized or
compiled numerical values. A tiny authored change therefore cannot disappear
through a compiler's floating-point formatting. Equivalent numeric spellings
may conservatively produce different identities. Unsupported file providers,
attached model assets, opaque plugins and repeated/cyclic includes fail closed.
Only the built-in `mujoco.sensor.touch_grid` plugin is currently allowed.

## Required target-machine procedure

1. Verify the full source-design receipt against the destination XML and files.
2. Produce a fresh strict compiled-model receipt on the target runtime. Retain
   any difference from the source machine; do not increase rounding tolerances
   or erase compiler arrays to manufacture a match.
3. If compiled identity differs, regenerate the collision/kinematic screen on
   that actual target model. Bind its receipt to the exact destination compiled
   identity, target configuration and trajectory bytes. Matching authored design
   is a prerequisite for this rescreen, not a replacement for it.
4. Run the existing post-import mechanical checks and physical rollout gates on
   the actual simulator. Neither identity contract replaces those checks.

The `verify_robot_design_identity` function intentionally cannot authorize an
old collision-screen receipt on a different compiled model. That decision belongs
to the controller loader's explicit destination-screen contract.

## September 8, 2026 cross-OS result

The Mac and owned Linux pod both compiled the same XML bytes using MuJoCo 3.12.0.
The strict compiled identities differ; the new source-design identities match.

| Evidence | Mac | Linux | Result |
| --- | --- | --- | --- |
| Source XML SHA-256 | `3148cbbefa04…` | `3148cbbefa04…` | Exact match |
| Source-design SHA-256 | `55f824908b05…` | `55f824908b05…` | Exact receipt match |
| Strict compiled SHA-256 | `18191db3fc74…` | `98c0fdc7e6c4…` | **Mismatch retained** |
| Referenced assets | 731 references, 538 unique byte hashes | Same | Exact match |

Of 327 compiled physical-array receipts, 18 differ. All 29 solver-option
receipts, named bindings, plugin declarations and state sizes match. The largest
source-frame corresponding mesh-vertex difference is 1.45×10⁻¹¹ m; compiled
inverse-weight and actuator-derived values differ by at most 8.42×10⁻¹¹.
Convex-hull and BVH indexing also differs, so indexing corresponding topology
arrays does not measure physical shape error. These observations do **not**
constitute a collision-hull equivalence certificate.

No Isaac process, live plant or GPU run was changed during this CPU-only check.
The source-design contract passed 12 focused tests for relocation, nested
includes, defaults, exact small numeric changes, changed asset bytes, directory
resolution, all six texture faces and malformed evidence.

```bash
PYTHONPATH=. python -m pytest -q tests/test_robot_design_identity.py
```

The compact committed result is
[`results/dexterous/2026-09-08/robot-identity-cross-os.json`](../results/dexterous/2026-09-08/robot-identity-cross-os.json).
Full receipts, raw array differences and the downloaded comparison arrays are
under `/tmp/doorbench-sensor-inference/out/identity-cross-os-001/`. Generated model
assets and raw compiled arrays are not committed.

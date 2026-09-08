# Replan left contact at the attained stance

`plan_attained_left_contact` is an opt-in geometric helper for a future
continuous native or Isaac controller. Existing controllers and defaults are
unchanged. It accepts detached actual measurements, fits the left arm and wrist,
and returns a target only if the independent dense screen passes. It cannot
step or write an active plant and does not qualify a physical opening.

```python
from doorbench.dexterous.runtime_left_planner import (
    plan_attained_left_contact, RuntimeLeftPlanFailure,
)

measured = {
    "pose_time_s": actual_global_time,
    "root": actual_root_xyz_wxyz,       # seven numbers, actor origin
    "joints": all_named_robot_angles,  # all 69 H1/Shadow scalar joints
    "door_positions": all_named_door_coordinates,
    "door_body_poses": {
        "leaf": actual_leaf_xyz_wxyz,
        "leaf_handle": actual_handle_body_xyz_wxyz,
    },
}
try:
    targets, receipt = plan_attained_left_contact(
        robot_xml, door_xml, original_target_json, measured,
        at_time_s=actual_global_time,
        clearance_profile="strict-v1", subdivisions=20,
    )
except RuntimeLeftPlanFailure as error:
    save_failure_receipt(error.receipt)
    raise
```

The caller should invoke it immediately before beginning left approach from the
current measured state. Use body origins, not a hand touch site or an inertial
COM frame. Preserve the actual global pose timestamp; a local acquisition clock
needs an explicit offset, not silently relabeled state. A stale timestamp,
incomplete joint map or inconsistent leaf/handle frame raises a failure with a
retained receipt. The body-frame check uses the existing 3 mm position and 0.02
rotation-matrix tolerances; it does not change collision or contact thresholds.

The torso, root, legs, opposite arm, fingers and complete door coordinates remain
frozen in the unstepped calculator. Seven arm/wrist joints fit the panel-local
contact plane and palm normal, allowing at most 15 cm of tangent adjustment under
the original bounded-fit contract. The initial arm matches the attained state
exactly. Named target rows and both independent interpolation screens are
returned; no nominal source pose is installed in the active robot.

## Explicit clearance option

`intermediate-clearance-3mm-v1` permits the existing panel-normal sine envelope
of at most 3 mm only if an endpoint-valid strict path fails its dense screen.
A passing strict path is left alone. A failed endpoint is rejected. Both strict
and clearance attempts are retained, and the clearance candidate must pass the
same dense joint, loopback, collision and contact-start gates. Its endpoints are
unchanged. This adjusts a planned reach path, not a permissible penetration.

The default 20 subdivisions screen 1,201 poses on each of the nominal joint and
Cartesian/IK interpolation routes for a 61-row target. The receipt binds exact
input files, target content, complete measured state, compiled/source robot
identities and planner/auditor source hashes. Reuse it only for that measured
planning event. Actual motion, balance, force limits, moving-door effects and
contact epochs still require the live controller's independent checks.

## Measured-state CPU checks

Thirty-six focused tests pass, including real MuJoCo scenes with every stepping
entry point disabled, stale/wrong frames, missing joints, rejected endpoints and
failed screens. No active plant is passed to the helper.

The actual native walking002 pre-left state at 45.504 s was also tested against
the source004 near-hand target. Replanning passed both 1,201-pose screens under
the strict profile. Endpoint plane error was 0.327 micrometres and normal error
0.000388 degrees; the minimum optimized arm margin was 0.0311 rad. Planning took
4.35 seconds on CPU. Its original failed source episode is not promoted.

That fixture uses the archived actual handle-body origin; its leaf-body pose is
reconstructed from the actual joint state and is explicitly labeled as such.
It is a static test on an attained state, not new independent live-frame evidence.
The full inputs and receipt are retained at:

`/tmp/doorbench-continuous/out/continuous/runtime-left-plan-001/`

A separate archived Isaac002 pre-left state reproduced a strict path failure.
The explicit 3 mm option then passed both 601-pose screens, retaining the failed
strict attempt, in 4.77 CPU seconds. Its door poses were reconstructed from the
archived coordinates solely for this static fixture. Those inputs and reports
are at `out/continuous/runtime-left-plan-002/` in the same worktree.

A standalone CPU command is available as
`scripts/dexterous/plan_runtime_left_contact.py`; provide `--robot`, `--door`,
`--targets`, `--measured`, `--at-time` and a new `--output` directory. It writes
`target-config.json` only on success and always retains planning failure receipts.

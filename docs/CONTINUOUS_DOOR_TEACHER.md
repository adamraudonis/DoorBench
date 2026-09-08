# Continuous Door55 controller contract

`doorbench.dexterous.continuous_door_teacher.ContinuousDoorTeacher` composes the
privileged walking/opening controller with the measured-state post-opening
controller. It returns 61 motor forces under the original H1/Shadow caps. It has
no active simulator reference, stepping method, pose writer or root support.
Its component models are unstepped calculators.

This is an integration candidate. Numeric composition tests and separately
qualified components do not establish a successful continuous physical episode.
In particular, the older `walking-opening-native-002` prefix failed its right-pad
gate; its successful initialized post-opening continuation cannot change that
result. An uninterrupted new native or Isaac run and an independent audit are
still required.

## Construction

```python
controller = ContinuousDoorTeacher(
    robot_xml, motors, reference, preparation, body_reset, h1_checkpoint,
    joint_geometry,
    door_xml=door_xml,
    left_targets=left_target_config,
    release_screen=right_release_screen,
    runtime_screen=destination_collision_receipt,  # optional
    opening_options=declared_opening_options,
    prepare_seconds=8.,
    maximum_seconds=180.,
)
```

The first arguments match `WalkingOpeningTeacher`. Reference, preparation,
body-reset and motor contracts are parsed dictionaries; paths retain their
component-specific contracts. Use the matching corrected `shadow-loopback-v2`
robot, 69 joint names, 61 ordered motors, eight passive loopbacks and official
H1 checkpoint. The source/model checks in both components remain active.

The post-opening configuration is fixed to `sequential-v2`, five seconds per
stow phase, 0.07 rad inward shoulder roll and enabled passage. This is the
configuration qualified on the revised door by initialized `native-012`; see
[the measured continuation report](POST_OPENING_REVISED_DOOR.md). Its fresh
geometric plan is built from the attained opening state, so a different endpoint
can still fail before continuation.

## One call per actual 2 ms step

```python
motor_forces, info = controller.force(
    t, root, joints, velocities, foot_loads, handle_pose, leaf_pose,
    angles, hand_forces,
    evidence=opening_evidence,
    right_palm_pose=right_palm_touch_site_pose,
    pose_time_s=t,
    contact_interval_s=[max(0., t - .002), t],
    body_poses=body_poses,
    door_velocities=door_velocities,
    continuation_evidence=continuation_evidence,
    applied_motor_forces=preceding_actual_motor_forces,
    release_normal_world=outward_left_panel_normal,
)
```

- `root`: 13 numbers, world xyz, unit wxyz, world linear/angular velocity.
- `joints` and `velocities`: exact 69 native joint-name mappings.
- `foot_loads`: actual left/right support loads in newtons.
- `handle_pose` and `leaf_pose`: actual BODY world xyz/wxyz at `t`.
- `right_palm_pose`: actual palm **touch SITE** world xyz/wxyz. This differs from
  the palm body origin and must retain the site offset.
- `angles`: exactly `operator`, `leaf`, `latch`, in radians/radians/metres.
- `door_velocities`: exactly `leaf_handle_hinge`, `leaf_hinge`,
  `leaf_latch_bolt_slide`, in radians/s or metres/s.
- `body_poses`: at least `left_ankle_link`, `right_ankle_link`, `lh_palm`,
  `rh_palm`, `leaf`, `leaf_handle`, all actual BODY origins at `t`.
- `hand_forces`: actual world force ON each hand collision body, including
  normal and friction forces, following the existing component contract.
- `applied_motor_forces`: the preceding interval's actual 61 motor forces in
  `controller.motor_names` order. Delivery must match the last returned command
  within 1e-5 Nm. At reset use the actual reset delivery, normally zeros.
- `release_normal_world`: actual outward left-panel unit normal. It can be
  supplied each call; it is consumed only on the first qualified crossing.

All contacts and loads describe the actual preceding physical interval. The
integrated poses describe its end. Use `NativeTransitionRecorder` before any
force-recomputing `mj_forward`, or the actual PhysX step buffers and measured
post-step articulation state. Geometric contact guesses do not replace loads.
See [native transition timing](NATIVE_TRANSITION_AUDIT.md).

`opening_evidence` retains exactly these seven fields on every call:

| Field | Meaning |
| --- | --- |
| `grasp_qualified` | Actual five-digit opposition/grasp qualification |
| `physics_qualified` | Every-step original-plant physical audit passed |
| `right_pad_patches_valid` | Every actual right-pad contact is valid |
| `hand_contact_count` | Actual total hand contact count |
| `left_panel_load_N` | Actual left-hand normal load on the panel |
| `left_palm_load_N` | Actual left-palm normal load on the panel |
| `right_lever_clearance_m` | Current signed right-hand/lever shape clearance |

`continuation_evidence` retains exactly four fields:

| Field | Meaning |
| --- | --- |
| `physics_qualified` | Same every-step physical qualification |
| `left_hand_contacts` | All actual left-hand touches, including self contact |
| `left_hand_load_N` | Actual normal load across all left-hand touches |
| `right_environment_contacts` | Actual right-hand contacts outside the robot |

Booleans must be booleans and counts nonnegative integers. The caller owns the
independent physical audit, including all warning categories, limits, loopbacks,
collision depths, original caps and absence of external root/foot supports.
These qualification booleans must come from that audit. This privileged teacher
interface must never become the numeric input contract of a sensor-only actor.

## Handoff and acceptance

Create one instance per physical episode. Start at time zero, at least 0.5 m
from the approach goal, with a closed resting door and contact-free hands.
Call once per 2 ms. Skipped, repeated or mixed-clock measurements fail closed.

The wrapper independently records actual grasp/support history and verifies the
opening event order: qualified grasp, latch release, left approach, supported
right release, right clearance, panel continuation. Contact holds require at
least 0.5 continuous seconds. Both ankles must have actually lifted at least
15 mm during approach, and preparation must have been screened and contact-free.

At the first actual target-aperture crossing (at least 1.2 rad), the complete
opening prefix must pass its physics and right-pad gates, final palm support
must have held for 0.5 seconds, and the right hand must be clear. The wrapper
freezes that opening receipt, then initializes post-opening from the same root,
joint positions, velocities, body poses and global contact interval. The new
controller is seeded with the preceding **delivered** motor forces. The opening
force computed at the boundary is discarded, never mislabeled as delivered.
The active plant continues without any reset. Subsequent calls route only to
post-opening.

Final completion requires actual whole-body passage (`minimum_body_y_m > .2`),
the post controller's completed walking stop, horizontal root speed below
0.03 m/s, both feet supporting at least 10 N, both hands clear and the left hand
unloaded, continuously for one second with less than 3 cm XY excursion.
Intentional left-hand release does not invalidate the frozen opening receipt.
Any later physics/right-pad failure still fails the overall episode.

## Runner-facing reports

| Property | Meaning |
| --- | --- |
| `.walking`, `.opening`, `.post` | Component diagnostics; do not mutate/reset them |
| `.done` / `.completed` | Current independently qualified quiet traversal flag |
| `.blocked_reason` / `.failure` | Sticky failure reason / timestamp receipt |
| `.opening_audit` | Detached qualified crossing receipt, or `None` |
| `.handoff` | Exact attained state, prior motor forces and post initialization |
| `.handoffs` | Detached walking/opening/continuation chronology |
| `.info` | Current component phase, clocks, delivery error and scope |

`.opening_audit` freezes the crossing checks, measured evidence, the full final
0.5-second qualification window, and the canonical SHA-256 of
`.opening_audit['state']` (also `.handoff['state']` after successful continuation
initialization). `handoff_state_sha256(state)` recomputes that identity using
sorted compact JSON and finite detached numeric values. Joint/door velocities,
contact interval, body and touch-site poses, delivered forces and evidence are
included. The receipt remains available if continuation initialization fails.
In that case `.handoff` is absent and the overall episode is failed.

The runner must freeze its independent opening audit at this same crossing,
then audit continuation through the end. It must not demand left-palm pressure
at the final traversal endpoint. Archive both reports and the full transition
stream with explicit contact epochs; never reduce the final result to a
successful endpoint pose. Catch `ContinuousDoorFailure`, stop the active run
and retain the failure. Retrying requires a new explicitly reset episode.

## Verification

```bash
python -m pytest tests/test_continuous_door_teacher.py \
  tests/test_post_opening_teacher.py tests/test_walking_opening_teacher.py -q
```

The tests use numeric component fakes to isolate routing, clocks, state/force
continuity, original caps, sticky failure, independent history checks and final
quiet qualification. They include a post-opening failure after a qualified
opening and verify that no whole-task success is claimed. They are not physical
robot trials. The separately run actual-model constructor smoke checks source,
motor and component compatibility without stepping any plant.

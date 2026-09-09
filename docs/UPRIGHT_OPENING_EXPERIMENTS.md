# Upright opening: experiment and portability guide

These are development experiments on `db0055_swing_single`, using H1 with two
Shadow hands and the corrected `shadow-loopback-v2` mechanics. They use privileged
simulator state. They are not a learned vision/tactile policy or a catalogue score.
See the [current execution ledger](DEXTEROUS_NEXT_STEPS.md) for live status.

## What is established

| Experiment | Evidence | Limit |
|---|---|---|
| Native panel006 | 33 runtime and 12 independent checks; continuous 120 s from closed door, upright grasp, lever operation, intentional release, supported 42.5° opening | Fixed feet during opening; no traversal |
| Native panel007 | Supported segment handoff and contact accounting reproduced | Failed to reach the wider target; stalled at 43.8° |
| Native panel008 | Contact accounting reproduced; recorded hinge-torque analysis explains the stall | Failed to reach the wider target; stopped at 63.0° |
| Isaac pressure014 | Independent audit and off-pod archive verified | Failed grasp; recorded thumb contact on an excluded surface; stopped at 27.4 s |
| Isaac lead015 | Its entire 20 s physical prefix exactly matches pressure014 | The lead limit never intervened; the identical failure is not a correction |

[Qualified native opening](evidence/native-standing-panel-006.json) ·
[Wider-opening force diagnosis](evidence/native-standing-panel-008.json) ·
[Isaac contact failure](evidence/isaac-standing-pressure-014.json) ·
[Controlled comparison](evidence/isaac-standing-lead-015.json).

## Run a fresh partial-opening comparison

First use the [one-click environment preparation](ISAAC_ONE_CLICK.md). Require
its actual `ready: true` receipt and corrected v2 mechanics. Use a frozen source
bundle and its source manifest, not a checkout another process can edit. The
standing coordinator verifies every source hash before launching physics.

Arm and verify the owned pod's local and remote teardown guards using the
[RunPod procedure](RUNPOD.md). The coordinator deadline must precede the guard
and leave an evidence-export reserve. A deadline argument alone does not arm a
teardown guard. Leave another worker's node untouched.

Run the coordinator from the prepared source directory. Replace the capitalized
paths and deadline with the values from that preparation and guard receipt:

```sh
python3 scripts/isaac/run_standing_operation.py \
  --source FROZEN_SOURCE \
  --ready READY_JSON \
  --reference SOURCE_REFERENCE_JSON \
  --output NEW_RUN_DIRECTORY \
  --work WORKSPACE \
  --deadline-unix VERIFIED_COORDINATOR_DEADLINE \
  --actual-material-pads \
  --material-pad-profile measured-pressure-v1 \
  --operation-leaf-target-rad .082
```

The coordinator rescreens geometry on the destination, runs a 36 s native
prerequisite, independently audits contacts, then admits the Isaac trial. A
native failure blocks Isaac. Final task success requires the actual Isaac result
and its independent audit. Register the run and attach the evidence collector
before launch, as described in the [Run Center guide](GPU_RUN_CENTER.md).
Recorded operation runs now save both `live-isaac.mp4` and
`live-isaac-hand.mp4`; neither video substitutes for the physical audit.

Declare one new experimental change before running it:

- `--operation-leaf-lead-limit-rad .012` limits the motor reference's lead over
  measured leaf travel. This particular setting did not change the failed
  pressure014 trajectory. Do not describe a configured limit as an intervention
  without evidence that it bound the reference.
- `--operation-operator-follow-after-leaf-rad .02` blends the hand's operator
  reference toward the measured handle rotation over one second, after actual
  leaf clearance. This allows the handle spring to return instead of continuing
  to demand full depression. Follow016 passed its native prerequisite; its Isaac
  outcome was pending when this guide was written.

The declared partial-opening acceptance remains 0.075–0.10 rad. These switches
are restricted to standalone operation experiments; they do not silently alter
an existing full-sequence controller.

## Qualify a wider panel continuation

1. Start from an archived actual pose, with the robot/door bytes verified.
2. Generate a fresh unstepped whole-body path with
   `scripts/dexterous/screen_whole_body_panel.py`. Numerical warm starts must match
   the exact source state and aperture grid. They confer no qualification.
3. Run `scripts/dexterous/audit_whole_body_panel_screen.py` on at least 2,001
   samples. Check hand and foot frames, joint limits, upright posture, clearance,
   and derivative bounds. Sampled endpoint fits alone are insufficient.
4. Bind the resulting target-plan SHA-256 into the withdrawal configuration.
   Each next segment requires a measured half-second supported target hold and
   validates the actual attained state before starting.
5. Execute a fresh physical trial and run the independent grasp/release and
   external-support audits. Keep failed reports and original targets unchanged.

The panel-force controller only requests left-palm load through original motors.
Its target cap is 6 N. The optional `stiction_assist: true` on a subsequent panel
segment increases the integral only when requested progress is positive and
measured motion is stalled. It retains terminal braking and support. A force
increase cannot make an infeasible geometric path acceptable.

## Move to another cluster or robot

Preserve the source manifest, prepared robot/door and motor contracts, controller
configuration, planner inputs, commands, dependency versions, seed/reset state,
and full interval recordings. Absolute XML paths can affect source identity;
rescreen against destination assets instead of editing stored hashes to pass an
audit. The renderer rejects mismatched native XML. For original Isaac evidence,
`render_isaac_contact_snapshot.py` can render measured hand-body transforms and
contact points on the original node (`MUJOCO_GL=egl` for headless rendering).
It labels the result as a source-mesh diagnostic, not an Isaac camera image.

The current H1 controller is **not a generic robot adapter**. A different robot
requires a new actuator/transmission mapping, joint limits and passive-coupling
checks, hand-surface calibration, frame conventions, balance/locomotion skills,
and fresh geometric and physical qualification. Native teacher poses use WXYZ
quaternions; archived PhysX contact-body transforms use XYZW. Preserve that
explicit distinction when building the next adapter.

A larger GPU does not establish larger-batch correctness. First repeat one
qualified episode on the new node, then profile state-only, RGB, and RGB+tactile
batches with matching physics and sensor clocks. Report environment throughput,
peak memory, and success checks together. Complete traversal, repeatability,
sensor-only learning and unseen-door coverage remain separate milestones in the
[approved plan](DEXTEROUS_NEXT_STEPS.md).


## September9 continuation: measured torque and released hand

Panel008's final measured contact moment was0.4557117Nm, below the declared
0.457638Nm hinge friction. The prospective `bounded-7N-v1` continuation load
profile raises the palm target cap from6 to7N and the integral cap from3 to4.5N.
It must explicitly enable terminal stiction assistance. The initial qualified
segment retains the original6N profile. The same motor caps, contact checks,
geometry, terminal braking and69° acceptance remain in effect. This changes
the controller command range, not the success criteria or a physics parameter.

Released-hand planner options can explicitly follow the root, retreat up to12cm,
and permit up to0.35rad orientation deviation after release. These define a
new free-hand motion; they do not alter grasp qualification. Screen021 using
these options failed foot-pose checks and is not eligible for physical replay.
Default world-fixed hand behavior remains unchanged.

Follow016's actual hand camera at24s was inspected at close range. Independent
raw-contact reconstruction under the existing volar profile still rejects
lateral little-finger middle-link contact, and at17.368s thumb loading is absent.
The historical trial remains failed; no prospective profile change is justified
by these observations. A graceful failed-prefix export was requested.
